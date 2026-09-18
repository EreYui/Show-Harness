#!/usr/bin/env python3
"""Run Show-Harness's atomic-token policy in a single-Piper Isaac Lab scene.

Use ``--no-rollout --dump-views --probe-axes --probe-gripper --probe-pick`` first. This starts Isaac Sim and
tests the robot, cameras and action mapping without a VLM server. The robot USD
comes from AgileX's piper_isaac_sim repository (see docs/simulators.md).
"""
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
from core.vlm.mvtoken_roles import MvTokenController
from plugins.auto_release import AutoReleasePlugin
from plugins.config import PluginsConfig


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Single-arm Piper in Isaac Lab / Isaac Sim")
    parser.add_argument("--robot-config", default=str(ROOT / "configs/robot_piper_isaaclab.yaml"))
    parser.add_argument("--piper-usd", default=os.environ.get("PIPER_USD"))
    parser.add_argument("--prompts-dir", default=str(ROOT / "prompts"))
    parser.add_argument("--version", default="v3")
    parser.add_argument("--device", default=None)
    parser.add_argument("--gui", action="store_true")
    parser.add_argument("--episodes", type=int, default=None)
    parser.add_argument("--max-steps", type=int, default=None)
    parser.add_argument("--loop-period-s", type=float, default=None)
    parser.add_argument("--log-dir", default=None)
    parser.add_argument("--vlm-backend", default=os.environ.get("VLM_BACKEND"))
    parser.add_argument("--vlm-url", default=os.environ.get("VLM_URL") or os.environ.get("VLLM_BASE_URL"))
    parser.add_argument("--model", default=os.environ.get("VLLM_MODEL"))
    parser.add_argument(
        "--policy-stack",
        choices=("adapted", "original_zero_shot"),
        default="adapted",
        help=(
            "adapted uses the flat Piper/MVTOKEN prompt; original_zero_shot uses "
            "the repository's generic subgoal planner plus prompts/controller.txt"
        ),
    )
    parser.add_argument(
        "--prompt-profile",
        choices=("auto", "generic", "deepseek"),
        default="auto",
        help=(
            "auto selects the DeepSeek adapter only for the DeepSeek backend; "
            "generic forces the original shared MVTOKEN prompt for A/B tests"
        ),
    )
    parser.add_argument(
        "--task",
        choices=("pick_lift", "pick_place_left", "pick_place_forward"),
        default=None,
        help="Piper task variant; pick-place variants add a visible green target pad",
    )
    parser.add_argument("--dump-views", action="store_true")
    parser.add_argument("--probe-axes", action="store_true")
    parser.add_argument("--probe-gripper", action="store_true")
    parser.add_argument("--probe-pick", action="store_true", help="Script a physical cube grasp and lift")
    parser.add_argument("--probe-place", action="store_true", help="Script the configured pick-place task")
    parser.add_argument("--no-rollout", action="store_true", help="Start scene and run diagnostics without VLM")
    parser.add_argument("--prompt-log-every", type=int, default=20)
    parser.add_argument("--debug", action="store_true")
    # Shared build_config accepts ManiSkill selectors, which are not used here.
    parser.set_defaults(task_suite_name=None, task_id=None, episode_index=None)
    return parser.parse_args()


def _prompt_path(
    prompts_dir: Path,
    version: str,
    backend: str,
    task_name: str = "pick_lift",
    profile: str = "auto",
) -> Path:
    base = prompts_dir / version
    selected = str(profile).lower()
    if selected not in {"auto", "generic", "deepseek"}:
        raise ValueError(f"Unknown Piper prompt profile: {profile!r}")
    if selected == "deepseek" or (selected == "auto" and backend == "deepseek"):
        path = base / (
            "piper_deepseek_pick_place.txt"
            if task_name.startswith("pick_place") else "piper_deepseek.txt"
        )
        if not path.is_file():
            raise FileNotFoundError(f"DeepSeek Piper prompt not found: {path}")
        return path
    for name in ("mvtoken_generator_lite.txt", "piper_mvtoken_lite.txt"):
        path = base / name
        if path.is_file():
            return path
    raise FileNotFoundError(f"No Piper MVTOKEN prompt found under {base}")


def _dump_views(runner, obs: dict, run_dir: Path) -> None:
    from core.record.images import save_png

    out = run_dir / "views"
    save_png(out / "agentview_raw.png", obs[runner.agentview_camera])
    save_png(out / "wrist_raw.png", obs[runner.wrist_camera])
    agentview, wrist = runner._images(obs)
    save_png(out / "agentview_sent.png", agentview)
    if wrist is not None:
        save_png(out / "wrist_sent.png", wrist)
    print(f"Piper view dump: {out}")
    print("Check that the cube is visible and that wrist left/right matches the front view.")


def main() -> int:
    args = parse_args()
    load_secrets_env()
    from core.sim.piper_isaaclab_task import resolve_piper_usd

    piper_usd = resolve_piper_usd(args.piper_usd)
    raw_cfg = load_yaml(args.robot_config)
    task_name = str(args.task or raw_cfg.get("piper_task", "pick_lift"))
    task_presets = raw_cfg.get("piper_tasks", {})
    if task_name not in task_presets:
        raise ValueError(f"Unknown Piper task {task_name!r}; choose one of {sorted(task_presets)}")
    raw_cfg = deep_merge(raw_cfg, task_presets[task_name])
    raw_cfg["piper_task"] = task_name
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
        raise ValueError("Isaac Lab's RTX cameras require a CUDA GPU device")

    client = None
    prompt_path = None
    prompt_template = None
    original_prompts = None
    if not args.no_rollout:
        if args.policy_stack == "original_zero_shot":
            original_prompts = load_prompt_dir(Path(args.prompts_dir))
            prompt_path = Path(args.prompts_dir) / "controller.txt"
            prompt_template = original_prompts["controller_prompt"]
            from plugins.coords import CoordsPlugin
            from plugins.ego import EgoPlugin
            from plugins.wrist_frame import WristFramePlugin

            plugin_flags = PluginsConfig.from_config(cfg)
            prompt_template = CoordsPlugin(
                plugin_flags.enabled("coords", default=False),
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
            prompt_path = _prompt_path(
                Path(args.prompts_dir),
                args.version,
                str(cfg["vlm_backend"]),
                task_name,
                profile=args.prompt_profile,
            )
            prompt_template = prompt_path.read_text(encoding="utf-8").strip()
        client = make_vlm_client(args, cfg)
        vlm_cfg = cfg["vlm"]
        if str(vlm_cfg.get("provider", "vllm")).lower() == "vllm":
            client.health_check(
                wait_s=float(vlm_cfg.get("startup_wait_s", 0)),
                poll_s=float(vlm_cfg.get("startup_poll_s", 5)),
            )
        elif str(vlm_cfg.get("provider", "")).lower() == "deepseek":
            client.verify_model_available()

    # Must launch Kit before importing the Piper scene or any Isaac Lab assets.
    from core.sim.piper_isaaclab_task import launch_isaac

    app = launch_isaac(headless=bool(cfg.get("headless", True)), device=str(cfg["device"]))
    print("Isaac Sim launched; creating Piper scene.", flush=True)
    env = None
    try:
        from core.sim.mvtoken_robolab_runner import MvTokenRobolabRunner
        from core.sim.piper_isaaclab_task import (
            PiperIsaacBackend, PiperIsaacLabEnv, probe_gripper, probe_move_axes, probe_pick,
            probe_place,
        )
        from interpreters.robolab_atomic_controller import RobolabAtomicController

        env = PiperIsaacLabEnv(cfg, piper_usd)
        print("Piper scene ready.", flush=True)
        controller = RobolabAtomicController(
            move_vectors=cfg["move_vectors"],
            step_m=float(cfg["step_m"]),
            ik_scale=env.ik_scale,
            sim_steps_per_decision=int(cfg["sim_steps_per_decision"]),
            max_delta_m=float(cfg.get("max_delta_m", 0.02)),
            motion_frame=str(cfg.get("motion_frame", "base")),
            tool_axis=cfg.get("tool_axis", [0.0, 0.0, 1.0]),
        )
        plugins_cfg = PluginsConfig.from_config(cfg)
        auto_release = AutoReleasePlugin(
            enabled=plugins_cfg.enabled("auto_release", default=False),
            empty_width_m=float(cfg.get("empty_width_m", 0.005)),
        )
        configured_video_fps = V0Config.from_dict(cfg.get("v0", {})).video_fps
        variant = (
            "Piper-IsaacLab-OriginalZeroShot"
            if args.policy_stack == "original_zero_shot"
            else "Piper-IsaacLab"
        )
        logger = EpisodeLogger(
            ROOT / cfg["log_dir"], 0, variant=variant,
            video_fps=configured_video_fps,
        )
        task_description = str(cfg.get("task_description", "Pick up the red cube from the table."))
        metadata = {
            "task": task_name,
            "task_description": task_description,
            "robot_usd": str(piper_usd),
            "robot_config": cfg,
            "prompt_file": str(prompt_path) if prompt_path else None,
            "prompt_version": args.version,
            "prompt_profile": args.prompt_profile,
            "policy_stack": args.policy_stack,
            "planner_prompt_file": (
                str(ROOT / "plugins/subgoal/subgoal_planner.txt")
                if args.policy_stack == "original_zero_shot" else None
            ),
            "control_loop": f"piper_isaaclab_{args.policy_stack}",
        }
        logger.write_metadata(metadata)

        def make_runner(ep_logger):
            if client is None:
                agent = None
            elif args.policy_stack == "original_zero_shot":
                from core.sim.original_zero_shot import OriginalZeroShotSingleAgent
                from plugins.mem_text import MemTextPlugin
                from plugins.proprioception import ProprioceptionPlugin
                from plugins.recovery import RecoveryPlugin
                from plugins.variable_step import VariableStepPlugin

                fine_step = float(cfg.get("zero_shot_fine_step_m", cfg["step_m"]))
                coarse_step = float(cfg.get("zero_shot_coarse_step_m", cfg["step_m"]))
                high_above = float(cfg.get("high_above_table_m", 0.08))
                variable_step = VariableStepPlugin(
                    enabled=plugins_cfg.enabled("variable_step", default=False),
                    coarse_step_m=coarse_step,
                    high_above_table_m=high_above,
                )

                agent = OriginalZeroShotSingleAgent(
                    client=client,
                    common_context=original_prompts["common_context"],
                    controller_prompt=prompt_template,
                    plan_dir=ep_logger.run_dir,
                    cot_mode=bool(cfg["vlm"].get("reasoning_cot", False)),
                    gripper_color=str(cfg.get("gripper_color", "black")),
                    max_subgoal_steps=V0Config.from_dict(cfg.get("v0", {})).max_subgoal_steps,
                    max_replans=V0Config.from_dict(cfg.get("v0", {})).max_replans,
                    planner_max_tokens=int(cfg.get("planner_max_tokens", 4096)),
                    proprio_plugin=ProprioceptionPlugin(
                        enabled=plugins_cfg.enabled("proprioception", default=False),
                        high_above_table_m=high_above,
                        fine_step_m=fine_step,
                        coarse_step_m=(coarse_step if variable_step.enabled else None),
                    ),
                    mem_text_plugin=MemTextPlugin(
                        enabled=plugins_cfg.enabled("mem_text", default=False),
                        max_recent=5,
                    ),
                    variable_step_plugin=variable_step,
                    recovery_plugin=RecoveryPlugin(
                        enabled=plugins_cfg.enabled("recovery", default=False),
                        empty_width_m=float(cfg.get("empty_width_m", 0.005)),
                        open_width_m=float(cfg.get("open_width_m", 0.08)),
                    ),
                    table_height_m=float(cfg.get("min_tcp_z_m", 0.025)),
                    transport_height_m=float(cfg.get("transport_tcp_z_m", 0.12)),
                    include_goal_error=bool(
                        plugins_cfg.enabled("coords", default=False)
                        and cfg.get("coords_include_goal_error", False)
                    ),
                    enforce_goal_error=bool(cfg.get("coords_enforce_goal_error", False)),
                    goal_error_tolerance_m=float(cfg.get("align_tolerance_m", 0.018)),
                    placement_tolerance_m=float(
                        cfg.get("target_success_tolerance_m", 0.04)
                    ),
                    positive_y_token=str(cfg.get("coords_positive_y_token", "MV_RIGHT")),
                    fine_step_m=fine_step,
                )
            else:
                agent = MvTokenController(client=client, prompt_template=prompt_template)
            return MvTokenRobolabRunner(
                env=env,
                backend=PiperIsaacBackend(),
                task_description=task_description,
                controller=controller,
                agent=agent,
                logger=ep_logger,
                config=V0Config.from_dict(cfg.get("v0", {})),
                max_steps=int(cfg["max_steps"]),
                num_steps_wait=int(cfg["num_steps_wait"]),
                loop_period_s=float(cfg["loop_period_s"]),
                sim_steps_per_decision=int(cfg["sim_steps_per_decision"]),
                settle_steps_per_decision=int(cfg.get("settle_steps_per_decision", 0)),
                agentview_camera=str(cfg["agentview_camera"]),
                wrist_camera=str(cfg["wrist_camera"]),
                agentview_rotation_degrees=int(cfg["agentview_rotation_degrees"]),
                wrist_rotation_degrees=int(cfg["wrist_rotation_degrees"]),
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
                use_wrist_image=bool(cfg["use_wrist_image"]),
                auto_release=auto_release,
                debug=args.debug,
                prompt_log_every=args.prompt_log_every,
                agentview_square_size=cfg.get("agentview_square_size"),
                agentview_crop_aspect=cfg.get("agentview_crop_aspect"),
                wrist_crop_aspect=cfg.get("wrist_crop_aspect"),
                wrist_square_size=cfg.get("wrist_square_size"),
                gripper_hold_steps=int(cfg.get("gripper_hold_steps", 0)),
                close_env=False,
                control_mode=f"piper_isaaclab_{args.policy_stack}",
            )

        if args.dump_views or args.probe_axes or args.probe_gripper or args.probe_pick or args.probe_place:
            obs, _, _ = env.reset(settle_steps=int(cfg["num_steps_wait"]))
            calibration = {}
            if args.dump_views:
                _dump_views(make_runner(logger), obs, logger.run_dir)
            if args.probe_axes:
                axes = probe_move_axes(env, controller, int(cfg["sim_steps_per_decision"]))
                calibration.update({"move_axis_tcp_delta": axes, "step_m": cfg["step_m"]})
                for token, delta in axes.items():
                    magnitude = sum(x * x for x in delta) ** 0.5
                    print(f"{token:8s}: {delta}, |d|={magnitude:.4f} m")
            if args.probe_gripper:
                widths = probe_gripper(env, controller)
                calibration["gripper_widths_m"] = widths
                print(f"Piper gripper widths: {widths}")
            if args.probe_pick:
                pick = probe_pick(env)
                calibration["scripted_pick"] = pick
                print(f"Piper scripted pick: {pick}")
                from core.record.images import save_png

                picked = env._observe()
                save_png(logger.run_dir / "views" / "agentview_picked.png", picked["agentview_camera"])
                save_png(logger.run_dir / "views" / "wrist_picked.png", picked["wrist_camera"])
            if args.probe_place:
                place = probe_place(env)
                calibration["scripted_place"] = place
                print(f"Piper scripted place: {place}")
            if calibration:
                logger.write_calibration(calibration)

        if args.no_rollout:
            print("Piper Isaac Lab scene started; policy rollout skipped.")
            print(f"Run directory: {logger.run_dir}")
            return 0

        successes = 0
        for episode in range(max(1, int(cfg.get("episodes", 1)))):
            ep_logger = (
                logger if episode == 0 else EpisodeLogger(
                    ROOT / cfg["log_dir"], episode, variant=variant,
                    video_fps=configured_video_fps,
                )
            )
            if episode > 0:
                ep_logger.write_metadata({**metadata, "episode_index": episode})
            result = make_runner(ep_logger).run()
            successes += int(result.success)
            print(f"episode {episode}: success={result.success}, steps={result.steps}, reason={result.end_reason}")
            print(f"run directory: {result.run_dir}")
        print(f"success rate: {successes}/{max(1, int(cfg.get('episodes', 1)))}")
        return 0 if successes else 2
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
