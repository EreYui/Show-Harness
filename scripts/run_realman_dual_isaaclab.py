#!/usr/bin/env python3
"""Run the official RMC-AIDA-L / dual RM65-B-V model in Isaac Lab."""
from __future__ import annotations

import argparse
import os
import sys
import traceback
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from core.config import deep_merge, load_secrets_env, load_yaml
from core.prompting.prompt_loader import load_prompt_dir
from core.record.episode_logger import EpisodeLogger
from core.sim.launch import (
    build_config,
    enable_original_zero_shot_reasoning,
    make_vlm_client,
)
from core.v0_types import V0Config
from core.vlm.dual_mvtoken_roles import DualMvTokenController
from core.vlm.dual_roles import DualDecision
from interpreters.robolab_atomic_controller import RobolabAtomicController
from plugins.config import PluginsConfig


class ScriptedDualAgent:
    """State-machine policy used to validate physics, cameras, closure and video."""

    scheme = "scripted"
    last_prompt = ""
    last_media = None

    def decide(self, **kwargs) -> DualDecision:
        from core.sim.realman_dual_isaaclab_task import scripted_token_for_phase

        tokens = {
            side: scripted_token_for_phase(str(kwargs[f"{side}_phase"]))
            for side in ("left", "right")
        }
        return DualDecision(
            tokens=tokens,
            reasoning="deterministic measured-state acceptance policy",
            raw_text=f"{tokens['left']} {tokens['right']}",
            payload={"scheme": "scripted"},
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Static-base RMC-AIDA-L / dual RM65-B-V in Isaac Lab / Isaac Sim"
    )
    parser.add_argument(
        "--robot-config",
        default=str(ROOT / "configs/robot_realman_dual_isaaclab.yaml"),
    )
    parser.add_argument(
        "--rmc-aidal-root",
        "--realman-root",
        dest="rmc_aidal_root",
        default=os.environ.get("RMC_AIDAL_ROOT"),
        help="Official RealManRobot/Ecosystem_Cases checkout on branch RMC-AIDA-L",
    )
    parser.add_argument("--prompts-dir", default=str(ROOT / "prompts"))
    parser.add_argument("--version", default="v3")
    parser.add_argument("--device", default=None)
    parser.add_argument("--gui", action="store_true")
    parser.add_argument("--episodes", type=int, default=None)
    parser.add_argument("--max-steps", type=int, default=None)
    parser.add_argument("--loop-period-s", type=float, default=None)
    parser.add_argument("--log-dir", default=None)
    parser.add_argument("--vlm-backend", default=os.environ.get("VLM_BACKEND"))
    parser.add_argument(
        "--vlm-url", default=os.environ.get("VLM_URL") or os.environ.get("VLLM_BASE_URL")
    )
    parser.add_argument("--model", default=os.environ.get("VLLM_MODEL"))
    parser.add_argument(
        "--policy-stack",
        choices=("adapted", "original_zero_shot"),
        default="adapted",
        help=(
            "adapted uses the RealMan task adapter; original_zero_shot uses the "
            "repository's generic dual subgoal planner and controller_dual.txt"
        ),
    )
    parser.add_argument(
        "--task",
        choices=(
            "parallel_forward",
            "diagonal_inward",
            "diagonal_outward",
            "split_depth",
        ),
        default=None,
        help="Dual-arm target layout preset",
    )
    parser.add_argument(
        "--scripted",
        action="store_true",
        help="Run the full closed loop without an API call; still records the video",
    )
    parser.add_argument("--no-rollout", action="store_true")
    parser.add_argument("--dump-views", action="store_true")
    parser.add_argument("--probe-axes", action="store_true")
    parser.add_argument("--probe-gripper", action="store_true")
    parser.add_argument("--prompt-log-every", type=int, default=10)
    parser.add_argument("--debug", action="store_true")
    parser.set_defaults(task_suite_name=None, task_id=None, episode_index=None)
    return parser.parse_args()


def make_controller(cfg: dict) -> RobolabAtomicController:
    return RobolabAtomicController(
        move_vectors=cfg["move_vectors"],
        step_m=float(cfg["step_m"]),
        ik_scale=1.0,
        sim_steps_per_decision=int(cfg["sim_steps_per_decision"]),
        max_delta_m=float(cfg.get("max_delta_m", 0.02)),
        motion_frame=str(cfg.get("motion_frame", "base")),
        tool_axis=cfg.get("tool_axis", [0.0, 0.0, 1.0]),
    )


def make_runner(env, controllers, agent, logger, cfg, args):
    from core.sim.dual_mvtoken_isaaclab_runner import DualIsaaclabMvTokenRunner

    return DualIsaaclabMvTokenRunner(
        env=env,
        controllers=controllers,
        agent=agent,
        logger=logger,
        task=str(cfg["task_description"]),
        config=V0Config.from_dict(cfg.get("v0", {})),
        max_steps=int(cfg["max_steps"]),
        num_steps_wait=int(cfg["num_steps_wait"]),
        loop_period_s=float(cfg["loop_period_s"]),
        sim_steps_per_decision=int(cfg["sim_steps_per_decision"]),
        settle_steps_per_decision=int(cfg.get("settle_steps_per_decision", 0)),
        gripper_hold_steps=int(cfg.get("gripper_hold_steps", 40)),
        debug=bool(args.debug),
        prompt_log_every=int(args.prompt_log_every),
        agentview_rotation_degrees=int(cfg.get("agentview_rotation_degrees", 0)),
        wrist_rotation_degrees=int(cfg.get("wrist_rotation_degrees", 0)),
        agentview_flip=str(
            cfg.get(
                "original_zero_shot_agentview_flip",
                cfg.get("agentview_flip", "none"),
            )
            if args.policy_stack == "original_zero_shot"
            else cfg.get("agentview_flip", "none")
        ),
        wrist_flip=str(
            cfg.get(
                "original_zero_shot_wrist_flip",
                cfg.get("wrist_flip", "none"),
            )
            if args.policy_stack == "original_zero_shot"
            else cfg.get("wrist_flip", "none")
        ),
        agentview_crop_aspect=cfg.get("agentview_crop_aspect"),
        wrist_crop_aspect=cfg.get("wrist_crop_aspect"),
        agentview_square_size=cfg.get("agentview_square_size"),
        wrist_square_size=cfg.get("wrist_square_size"),
        wrist_translations=(
            cfg.get("original_zero_shot_wrist_translations_px", {})
            if args.policy_stack == "original_zero_shot"
            else {}
        ),
        wrist_focus_crop_px=(
            cfg.get("original_zero_shot_wrist_focus_crop_px")
            if args.policy_stack == "original_zero_shot"
            else None
        ),
    )


def main() -> int:
    args = parse_args()
    load_secrets_env()
    raw_cfg = load_yaml(args.robot_config)
    task_name = str(args.task or raw_cfg.get("realman_task", "parallel_forward"))
    task_presets = raw_cfg.get("realman_tasks", {})
    if task_name not in task_presets:
        raise ValueError(
            f"Unknown RealMan task {task_name!r}; choose one of {sorted(task_presets)}"
        )
    raw_cfg = deep_merge(raw_cfg, task_presets[task_name])
    raw_cfg["realman_task"] = task_name
    if args.device:
        raw_cfg["device"] = args.device
    if args.episodes is not None:
        raw_cfg["episodes"] = args.episodes
    if args.gui:
        raw_cfg["headless"] = False
    cfg = build_config(args, raw_cfg)
    if args.policy_stack == "original_zero_shot":
        enable_original_zero_shot_reasoning(cfg)
    if not str(cfg["device"]).startswith("cuda:"):
        raise ValueError("Isaac Lab RTX cameras require a CUDA GPU device")

    from core.sim.realman_assets import prepare_rmc_aidal_urdf, resolve_rmc_aidal_root

    rmc_aidal_root = resolve_rmc_aidal_root(args.rmc_aidal_root)
    robot_urdf = prepare_rmc_aidal_urdf(rmc_aidal_root)
    print(f"RMC-AIDA-L source: {rmc_aidal_root}")
    print(f"Prepared full-body simulation URDF: {robot_urdf}")

    client = None
    original_prompts = None
    if args.policy_stack == "original_zero_shot":
        original_prompts = load_prompt_dir(Path(args.prompts_dir))
        prompt_path = Path(args.prompts_dir) / "controller_dual.txt"
        prompt_template = prompt_path.read_text(encoding="utf-8").strip()
        from plugins.coords import CoordsPlugin
        from plugins.ego import EgoPlugin
        from plugins.wrist_frame import WristFramePlugin

        prompt_template = CoordsPlugin(
            PluginsConfig.from_config(cfg).enabled("coords", default=False),
            positive_y_token=str(cfg.get("coords_positive_y_token", "MV_RIGHT")),
            positive_y_image_edge=str(
                cfg.get("coords_positive_y_image_edge", "right")
            ),
        ).apply(prompt_template)
        prompt_template = WristFramePlugin(
            str(cfg.get("motion_frame", "base")).lower() == "wrist"
        ).apply(prompt_template)
        prompt_template = EgoPlugin(bool(cfg.get("is_ego", False))).apply(
            prompt_template
        )
    else:
        prompt_path = (
            Path(args.prompts_dir) / args.version / "realman_deepseek_dual_pick_place.txt"
        )
        prompt_template = prompt_path.read_text(encoding="utf-8").strip()
    if not args.scripted and not args.no_rollout:
        client = make_vlm_client(args, cfg)
        vlm_cfg = cfg["vlm"]
        if str(vlm_cfg.get("provider", "vllm")).lower() == "vllm":
            client.health_check(
                wait_s=float(vlm_cfg.get("startup_wait_s", 0)),
                poll_s=float(vlm_cfg.get("startup_poll_s", 5)),
            )
        elif str(vlm_cfg.get("provider", "")).lower() == "deepseek":
            client.verify_model_available()

    from core.sim.realman_dual_isaaclab_task import launch_isaac

    app = launch_isaac(headless=bool(cfg.get("headless", True)), device=str(cfg["device"]))
    print("Isaac Sim launched; creating full RMC-AIDA-L scene.", flush=True)
    env = None
    try:
        from core.record.images import save_png
        from core.sim.realman_dual_isaaclab_task import RealmanDualIsaacLabEnv

        env = RealmanDualIsaacLabEnv(cfg, robot_urdf)
        print("RMC-AIDA-L dual-arm scene ready.", flush=True)
        controllers = {side: make_controller(cfg) for side in ("left", "right")}
        video_fps = V0Config.from_dict(cfg.get("v0", {})).video_fps
        variant = (
            f"RealMan-Dual-IsaacLab-Scripted-{task_name}"
            if args.scripted
            else (
                f"RealMan-Dual-IsaacLab-OriginalZeroShot-{task_name}"
                if args.policy_stack == "original_zero_shot"
                else f"RealMan-Dual-IsaacLab-{task_name}"
            )
        )
        logger = EpisodeLogger(
            ROOT / str(cfg["log_dir"]),
            0,
            variant=variant,
            video_fps=video_fps,
        )

        def build_agent(ep_logger):
            if args.scripted or args.no_rollout:
                return ScriptedDualAgent()
            if args.policy_stack == "original_zero_shot":
                from core.sim.original_zero_shot import OriginalZeroShotDualAgent
                from plugins.mem_text import MemTextPlugin
                from plugins.proprioception import ProprioceptionPlugin
                from plugins.recovery import RecoveryPlugin
                from plugins.variable_step import VariableStepPlugin
                from plugins.view_select import ViewSelectPlugin

                v0 = V0Config.from_dict(cfg.get("v0", {}))
                flags = PluginsConfig.from_config(cfg)
                fine_step = float(cfg.get("zero_shot_fine_step_m", cfg["step_m"]))
                coarse_step = float(cfg.get("zero_shot_coarse_step_m", cfg["step_m"]))
                high_above = float(cfg.get("high_above_table_m", 0.08))
                table_heights = {
                    side: float(cfg.get("min_tcp_z_m", 0.703))
                    for side in ("left", "right")
                }
                return OriginalZeroShotDualAgent(
                    client=client,
                    common_context=original_prompts["common_context"],
                    controller_prompt=prompt_template,
                    plan_dir=ep_logger.run_dir,
                    cot_mode=bool(cfg["vlm"].get("reasoning_cot", False)),
                    max_subgoal_steps=v0.max_subgoal_steps,
                    max_replans=v0.max_replans,
                    proprio_plugin=ProprioceptionPlugin(
                        enabled=flags.enabled("proprioception", default=False),
                        high_above_table_m=high_above,
                        fine_step_m=fine_step,
                        coarse_step_m=(
                            coarse_step
                            if flags.enabled("variable_step", default=False)
                            else None
                        ),
                    ),
                    mem_text_plugin=MemTextPlugin(
                        enabled=flags.enabled("mem_text", default=False),
                        max_recent=5,
                    ),
                    view_select_plugin=ViewSelectPlugin(
                        enabled=flags.enabled("view_select", default=False)
                    ),
                    variable_step_plugins={
                        side: VariableStepPlugin(
                            enabled=flags.enabled("variable_step", default=False),
                            coarse_step_m=coarse_step,
                            high_above_table_m=high_above,
                        )
                        for side in ("left", "right")
                    },
                    recovery_plugins={
                        side: RecoveryPlugin(
                            enabled=flags.enabled("recovery", default=False),
                            empty_width_m=float(cfg.get("empty_width_m", 0.008)),
                            open_width_m=float(cfg.get("open_width_m", 0.045)),
                            empty_realign_token=(
                                cfg.get("empty_grasp_realign_tokens", {}).get(side)
                            ),
                        )
                        for side in ("left", "right")
                    },
                    table_heights=table_heights,
                    transport_heights={
                        side: float(cfg.get("transport_tcp_z_m", 0.80))
                        for side in ("left", "right")
                    },
                    include_goal_error=bool(
                        flags.enabled("coords", default=False)
                        and cfg.get("coords_include_goal_error", False)
                    ),
                    enforce_goal_error=bool(cfg.get("coords_enforce_goal_error", False)),
                    goal_error_tolerance_m=float(cfg.get("align_tolerance_m", 0.015)),
                    placement_tolerance_m=float(
                        cfg.get("target_success_tolerance_m", 0.035)
                    ),
                    positive_y_token=str(cfg.get("coords_positive_y_token", "MV_RIGHT")),
                    fine_step_m=fine_step,
                    empty_width_m=float(cfg.get("empty_width_m", 0.008)),
                )
            return DualMvTokenController(
                client=client,
                prompt_template=prompt_template,
                scheme="once",
            )

        agent = build_agent(logger)
        metadata = {
            "task": cfg["task_description"],
            "task_name": task_name,
            "robot": "RealMan RMC-AIDA-L dual RM65-B-V (static mobile base)",
            "physical_platform": cfg.get("physical_platform", {}),
            "source_description": str(rmc_aidal_root),
            "generated_urdf": str(robot_urdf),
            "robot_config": cfg,
            "prompt_file": str(prompt_path),
            "planner_prompt_file": (
                str(ROOT / "plugins/subgoal/subgoal_planner_dual.txt")
                if args.policy_stack == "original_zero_shot" else None
            ),
            "policy_stack": args.policy_stack,
            "control_loop": f"realman_dual_isaaclab_{args.policy_stack}",
            "scripted": bool(args.scripted),
        }
        logger.write_metadata(metadata)
        runner = make_runner(env, controllers, agent, logger, cfg, args)

        calibration: dict = {}
        if args.dump_views or args.probe_axes or args.probe_gripper or args.no_rollout:
            obs, _, _ = env.reset(settle_steps=int(cfg["num_steps_wait"]))
            front, left, right = runner._images(obs)
            views = logger.run_dir / "views"
            if args.dump_views or args.no_rollout:
                save_png(views / "agentview_raw.png", obs["agentview_camera"])
                save_png(views / "overview_raw.png", obs["overview_camera"])
                save_png(views / "wrist_left_raw.png", obs["wrist_left_camera"])
                save_png(views / "wrist_right_raw.png", obs["wrist_right_camera"])
                save_png(views / "agentview_sent.png", front)
                save_png(views / "wrist_left_sent.png", left)
                save_png(views / "wrist_right_sent.png", right)
                print(f"View dump: {views}")
            if args.probe_axes:
                axis_results = {}
                for side in ("left", "right"):
                    axis_results[side] = {}
                    for token in cfg["move_vectors"]:
                        env.reset(settle_steps=20)
                        for controller in controllers.values():
                            controller.open_gripper()
                            controller.set_orientation_reference(None)
                        before = env.tcp(side)
                        tokens = {"left": "STILL", "right": "STILL"}
                        tokens[side] = token
                        runner._execute(tokens)
                        axis_results[side][token] = [
                            round(float(x), 6) for x in env.tcp(side) - before
                        ]
                calibration["move_axis_tcp_delta"] = axis_results
                print(f"Axis probe: {axis_results}")
            if args.probe_gripper:
                env.reset(settle_steps=20)
                opened = {side: env.gripper_width(side) for side in ("left", "right")}
                runner._execute({"left": "GRASP", "right": "GRASP"})
                closed = {side: env.gripper_width(side) for side in ("left", "right")}
                runner._execute({"left": "RELEASE", "right": "RELEASE"})
                reopened = {side: env.gripper_width(side) for side in ("left", "right")}
                calibration["gripper_widths_m"] = {
                    side: {
                        "open_before": opened[side],
                        "closed": closed[side],
                        "open_after": reopened[side],
                    }
                    for side in ("left", "right")
                }
                print(f"Gripper probe: {calibration['gripper_widths_m']}")
            if calibration:
                logger.write_calibration(calibration)

        if args.no_rollout:
            video = logger.close(success=False, fps=video_fps)
            logger.write_summary(
                {
                    "success": False,
                    "end_reason": "diagnostics_only",
                    "video_path": str(video),
                    "run_dir": str(logger.run_dir),
                }
            )
            print(f"Diagnostics complete: {logger.run_dir}")
            return 0

        successes = 0
        total = max(1, int(cfg.get("episodes", 1)))
        for episode in range(total):
            if episode == 0:
                ep_logger = logger
                ep_runner = runner
            else:
                ep_logger = EpisodeLogger(
                    ROOT / str(cfg["log_dir"]),
                    episode,
                    variant=variant,
                    video_fps=video_fps,
                )
                ep_logger.write_metadata({**metadata, "episode_index": episode})
                ep_runner = make_runner(
                    env, controllers, build_agent(ep_logger), ep_logger, cfg, args
                )
            result = ep_runner.run()
            successes += int(result.success)
            print(
                f"episode {episode}: success={result.success}, steps={result.steps}, "
                f"reason={result.end_reason}\nvideo: {result.video_path}\nrun: {result.run_dir}"
            )
        print(f"success rate: {successes}/{total}")
        return 0 if successes == total else 2
    except Exception:
        traceback.print_exc()
        return 1
    finally:
        if env is not None:
            env.close()
            del env
        app.close(wait_for_replicator=False)


if __name__ == "__main__":
    raise SystemExit(main())
