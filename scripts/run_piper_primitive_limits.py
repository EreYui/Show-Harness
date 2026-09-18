#!/usr/bin/env python3
"""Measure task-level limits of fixed-axis primitives in the Piper Isaac Lab scene.

Both policies are state-feedback oracles.  They share the same reset, robot, relative
IK controller, 1 cm maximum decision length, physics rate and decision budget.  The
only independent variable is the action representation:

* ``atomic``: one fixed-length X/Y/Z MVTOKEN translation per decision;
* ``direct``: an arbitrary XYZ displacement, norm-clipped to the same length.

This removes vision and language-model errors.  A task success gap therefore measures
the action interface rather than prompt or model quality.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
from collections import defaultdict
from datetime import datetime
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from core.config import load_yaml
from core.record.episode_logger import EpisodeLogger
from core.sim.primitive_task_benchmark import (
    atomic_delta,
    continuous_delta,
    line_constraint_errors,
    summarize_task,
)


TASKS = {
    "precision_diagonal_reach": {
        "offsets": ((0.075, 0.075, 0.0), (0.095, -0.055, 0.0), (0.055, 0.095, 0.0)),
        "endpoint_tolerance_m": 0.002,
        "max_decisions": 36,
        "description": "Insert the TCP into a 2 mm diagonal goal region.",
    },
    "narrow_diagonal_wipe": {
        "offsets": ((0.080, 0.080, 0.0), (0.080, -0.080, 0.0), (0.060, 0.100, 0.0)),
        "endpoint_tolerance_m": 0.006,
        "corridor_tolerance_m": 0.004,
        "required_constraint_pass_rate": 0.95,
        "max_decisions": 36,
        "description": (
            "Wipe from start to goal while at least 95% of physics samples stay "
            "inside an 8 mm-wide diagonal strip and within 4 mm of the surface height."
        ),
    },
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--robot-config", default=str(ROOT / "configs/robot_piper_isaaclab.yaml"))
    parser.add_argument("--piper-usd", default=os.environ.get("PIPER_USD"))
    parser.add_argument("--device", default=None)
    parser.add_argument("--gui", action="store_true")
    parser.add_argument("--output-dir", default=str(ROOT / "rollouts/piper_primitive_limits"))
    parser.add_argument("--step-m", type=float, default=0.010)
    parser.add_argument("--trials", type=int, default=3, choices=(1, 2, 3))
    parser.add_argument("--sim-steps-per-decision", type=int, default=None)
    parser.add_argument(
        "--tasks", nargs="+", choices=tuple(TASKS), default=tuple(TASKS)
    )
    parser.add_argument(
        "--controllers", nargs="+", choices=("atomic", "direct"), default=("atomic", "direct")
    )
    return parser.parse_args()


def _direct_action(delta_m: np.ndarray, sim_steps: int, ik_scale: float) -> np.ndarray:
    action = np.zeros(7, dtype=np.float32)
    action[:3] = np.asarray(delta_m, dtype=float) / (float(sim_steps) * float(ik_scale))
    action[6] = 0.0
    return action


def _execute_decision(env, action: np.ndarray, sim_steps: int, trace: list[np.ndarray]):
    obs = None
    truncated = False
    for _ in range(sim_steps):
        obs, _, truncated, _ = env.step(action)
        trace.append(env.tcp())
        if truncated:
            break
    return obs, truncated


def _move_to_setup(env, target: np.ndarray, step_m: float, sim_steps: int):
    """Closed-loop setup motion; excluded from the scored task trace and budget."""
    obs = None
    for _ in range(80):
        error = target - env.tcp()
        if float(np.linalg.norm(error)) <= 0.003:
            break
        delta, _ = continuous_delta(error, step_m)
        obs, truncated = _execute_decision(
            env, _direct_action(delta, sim_steps, env.ik_scale), sim_steps, []
        )
        if truncated:
            raise RuntimeError("Piper episode truncated during benchmark setup")
    return obs


def _task_map(
    trace: list[np.ndarray], start: np.ndarray, target: np.ndarray, *, corridor_m: float | None
) -> np.ndarray:
    """Top-down task plot embedded in every rollout video."""
    size = 512
    margin = 52
    image = Image.new("RGB", (size, size), (248, 248, 246))
    draw = ImageDraw.Draw(image)
    points = np.asarray(trace if trace else [start], dtype=float)
    all_xy = np.vstack((points[:, :2], start[:2], target[:2]))
    centre = (start[:2] + target[:2]) / 2.0
    span = max(0.14, float(np.ptp(all_xy[:, 0])), float(np.ptp(all_xy[:, 1]))) * 1.35

    def px(xy):
        x = margin + (float(xy[0] - centre[0]) / span + 0.5) * (size - 2 * margin)
        y = size - margin - (float(xy[1] - centre[1]) / span + 0.5) * (size - 2 * margin)
        return int(round(x)), int(round(y))

    start_px, target_px = px(start[:2]), px(target[:2])
    if corridor_m is not None:
        width_px = max(2, int(round(2.0 * corridor_m / span * (size - 2 * margin))))
        draw.line((start_px, target_px), fill=(190, 224, 202), width=width_px)
    draw.line((start_px, target_px), fill=(90, 105, 120), width=2)
    if len(points) > 1:
        draw.line([px(p[:2]) for p in points], fill=(37, 99, 235), width=5, joint="curve")
    r = 8
    draw.ellipse((start_px[0] - r, start_px[1] - r, start_px[0] + r, start_px[1] + r), fill=(35, 160, 90))
    draw.ellipse((target_px[0] - r, target_px[1] - r, target_px[0] + r, target_px[1] + r), fill=(220, 50, 50))
    draw.text((18, 16), "TCP path (blue)  start (green)  goal (red)", fill=(30, 34, 39))
    return np.asarray(image)


def _run_trial(
    *, env, cfg: dict, task_name: str, controller_name: str, trial: int,
    benchmark_root: Path, step_m: float, sim_steps: int,
) -> dict:
    spec = TASKS[task_name]
    obs, _, _ = env.reset(settle_steps=int(cfg.get("num_steps_wait", 8)))
    if task_name == "narrow_diagonal_wipe":
        home = env.tcp()
        setup = np.asarray(
            (min(0.40, home[0] + 0.07), -0.13, env.min_tcp_z_m + 0.012), dtype=float
        )
        obs = _move_to_setup(env, setup, step_m, sim_steps) or obs

    start = env.tcp()
    offset = np.asarray(spec["offsets"][trial], dtype=float)
    target = start + offset
    trace = [start.copy()]
    variant = f"{task_name}-{controller_name}"
    logger = EpisodeLogger(benchmark_root / "runs", trial, variant=variant, video_fps=5.0)
    logger.write_metadata(
        {
            "benchmark": "piper_primitive_task_limits",
            "task": task_name,
            "task_description": spec["description"],
            "controller": controller_name,
            "oracle_state_feedback": True,
            "step_m": step_m,
            "sim_steps_per_decision": sim_steps,
            "start_m": start,
            "target_m": target,
            "target_offset_m": offset,
            "success_spec": spec,
        }
    )
    decisions = 0
    truncated = False
    final_result = None
    try:
        for decision_idx in range(int(spec["max_decisions"])):
            error = target - env.tcp()
            if float(np.linalg.norm(error)) <= float(spec["endpoint_tolerance_m"]):
                break
            if controller_name == "atomic":
                delta, action_name = atomic_delta(error, step_m)
            else:
                delta, action_name = continuous_delta(error, step_m)
            action = _direct_action(delta, sim_steps, env.ik_scale)
            obs, truncated = _execute_decision(env, action, sim_steps, trace)
            decisions += 1

            lateral_error = 0.0
            pass_rate = 1.0
            corridor = spec.get("corridor_tolerance_m")
            if corridor is not None:
                planar = np.asarray(trace).copy()
                planar[:, 2] = 0.0
                start_planar, target_planar = start.copy(), target.copy()
                start_planar[2] = target_planar[2] = 0.0
                lateral = line_constraint_errors(planar, start_planar, target_planar)
                height = np.abs(np.asarray(trace)[:, 2] - start[2])
                constraint = np.maximum(lateral, height)
                lateral_error = float(lateral[-1])
                pass_rate = float(np.mean(constraint <= float(corridor)))
            current_error = float(np.linalg.norm(target - env.tcp()))
            record = {
                "i": decision_idx,
                "stage": task_name,
                "act": action_name,
                "grip": "OPEN",
                "done": current_error <= float(spec["endpoint_tolerance_m"]),
                "eef": env.tcp(),
                "goal_error_m": current_error,
                "constraint_error_m": lateral_error,
                "constraint_pass_rate": pass_rate,
                "vlm": {
                    "c": {
                        "why": (
                            f"{controller_name} oracle; goal error {current_error*1000:.1f} mm; "
                            f"constraint pass {pass_rate*100:.1f}%"
                        )
                    }
                },
            }
            logger.log_step(
                decision_idx,
                agentview=np.asarray(obs[cfg["agentview_camera"]]),
                wrist=np.asarray(obs[cfg["wrist_camera"]]),
                overview=_task_map(trace, start, target, corridor_m=corridor),
                record=record,
            )
            if truncated:
                break

        constraint_errors = None
        if "corridor_tolerance_m" in spec:
            positions = np.asarray(trace)
            planar = positions.copy()
            planar[:, 2] = 0.0
            start_planar, target_planar = start.copy(), target.copy()
            start_planar[2] = target_planar[2] = 0.0
            lateral = line_constraint_errors(planar, start_planar, target_planar)
            height = np.abs(positions[:, 2] - start[2])
            constraint_errors = np.maximum(lateral, height)
        result = summarize_task(
            task=task_name,
            controller=controller_name,
            trial=trial,
            trace=trace,
            target=target,
            target_offset=offset,
            decisions=decisions,
            endpoint_tolerance_m=float(spec["endpoint_tolerance_m"]),
            constraint_errors=constraint_errors,
            constraint_tolerance_m=spec.get("corridor_tolerance_m"),
            required_constraint_pass_rate=float(spec.get("required_constraint_pass_rate", 1.0)),
        )
        final_result = result.to_dict()
        final_result.update(
            {
                "run_dir": str(logger.run_dir),
                "truncated": truncated,
                "start_m": start.tolist(),
                "target_m": target.tolist(),
            }
        )
        logger.write_summary(final_result)
        video_path = logger.close(success=result.success, fps=5.0)
        final_result["video_path"] = str(video_path)
        print(
            f"[{task_name}][{controller_name}][{trial}] success={result.success} "
            f"endpoint={result.endpoint_reached} error={result.final_error_m*1000:.1f}mm "
            f"constraint={result.constraint_pass_rate*100:.1f}% decisions={decisions}",
            flush=True,
        )
        return final_result
    except Exception:
        if final_result is None:
            logger.close(success=False, fps=5.0)
        raise


def _aggregate(results: list[dict]) -> dict:
    grouped: dict[tuple[str, str], list[dict]] = defaultdict(list)
    for result in results:
        grouped[(result["task"], result["controller"])].append(result)
    rows = {}
    for (task, controller), values in sorted(grouped.items()):
        rows[f"{task}/{controller}"] = {
            "trials": len(values),
            "success_rate": sum(bool(x["success"]) for x in values) / len(values),
            "endpoint_rate": sum(bool(x["endpoint_reached"]) for x in values) / len(values),
            "mean_final_error_m": sum(float(x["final_error_m"]) for x in values) / len(values),
            "mean_constraint_pass_rate": sum(float(x["constraint_pass_rate"]) for x in values) / len(values),
            "mean_path_length_m": sum(float(x["path_length_m"]) for x in values) / len(values),
            "mean_decisions": sum(int(x["decisions"]) for x in values) / len(values),
        }
    gaps = {}
    for task in sorted({x["task"] for x in results}):
        atomic = rows.get(f"{task}/atomic")
        direct = rows.get(f"{task}/direct")
        if atomic and direct:
            gaps[task] = {
                "direct_minus_atomic_success_rate": direct["success_rate"] - atomic["success_rate"],
                "direct_minus_atomic_endpoint_rate": direct["endpoint_rate"] - atomic["endpoint_rate"],
                "direct_minus_atomic_constraint_pass_rate": (
                    direct["mean_constraint_pass_rate"] - atomic["mean_constraint_pass_rate"]
                ),
            }
    return {"groups": rows, "gaps": gaps}


def main() -> int:
    args = parse_args()
    if args.step_m <= 0.0 or not math.isfinite(args.step_m):
        raise ValueError("--step-m must be a finite positive number")
    cfg = load_yaml(args.robot_config)
    if args.device:
        cfg["device"] = args.device
    if args.gui:
        cfg["headless"] = False
    sim_steps = int(args.sim_steps_per_decision or cfg["sim_steps_per_decision"])
    if sim_steps <= 0:
        raise ValueError("--sim-steps-per-decision must be positive")
    benchmark_root = Path(args.output_dir) / datetime.now().strftime("%Y%m%d_%H%M%S")
    benchmark_root.mkdir(parents=True, exist_ok=True)

    from core.sim.piper_isaaclab_task import launch_isaac, resolve_piper_usd

    piper_usd = resolve_piper_usd(args.piper_usd)
    app = launch_isaac(headless=bool(cfg.get("headless", True)), device=str(cfg["device"]))
    env = None
    results = []
    try:
        from core.sim.piper_isaaclab_task import PiperIsaacLabEnv

        env = PiperIsaacLabEnv(cfg, piper_usd)
        env.max_physics_steps = max(env.max_physics_steps, 8000)
        for task_name in args.tasks:
            for trial in range(args.trials):
                for controller_name in args.controllers:
                    results.append(
                        _run_trial(
                            env=env,
                            cfg=cfg,
                            task_name=task_name,
                            controller_name=controller_name,
                            trial=trial,
                            benchmark_root=benchmark_root,
                            step_m=float(args.step_m),
                            sim_steps=sim_steps,
                        )
                    )
        # Isaac Sim's app.close() may terminate the interpreter, so persist the
        # benchmark aggregate while Kit is still alive.
        summary = {
            "benchmark": "piper_primitive_task_limits",
            "method": (
                "Paired state-feedback oracles with identical resets, physics and a 1 cm decision "
                "bound; only fixed-axis/fixed-length versus arbitrary XYZ actions differ."
            ),
            "task_specs": {name: TASKS[name] for name in args.tasks},
            "step_m": args.step_m,
            "sim_steps_per_decision": sim_steps,
            "results": results,
            **_aggregate(results),
        }
        (benchmark_root / "results.json").write_text(
            json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8"
        )
        print(json.dumps(summary["groups"], indent=2), flush=True)
        print(f"Benchmark results: {benchmark_root / 'results.json'}", flush=True)
    finally:
        if env is not None:
            env.close()
        app.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
