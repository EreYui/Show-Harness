"""Shared three-view, two-arm MVTOKEN loop for the standalone RealMan scene."""
from __future__ import annotations

import time
from typing import Any, Optional

import numpy as np

from core.action_units import MOVE_ATOMS
from core.record.episode_logger import EpisodeLogger
from core.record.images import prepare_view
from core.v0_types import EpisodeResult, V0Config
from core.vlm.dual_mvtoken_roles import SIDES
from interpreters.robolab_atomic_controller import RobolabAtomicController

GRASP_TOKEN = "GRASP"
RELEASE_TOKEN = "RELEASE"
DONE_TOKEN = "DONE"
STILL_TOKEN = "STILL"
FALLBACK_TOKEN = "STILL"
RECENT_MOVES_MAX = 5
HISTORY_TOKENS = frozenset((*MOVE_ATOMS, STILL_TOKEN))


def translate_view(image: np.ndarray, dx_px: int = 0, dy_px: int = 0) -> np.ndarray:
    """Translate a calibrated camera image without wraparound.

    The official RealMan wrist cameras sit outside their grippers, so the fingertips
    appear at opposite image edges.  A per-arm translation recenters that fixed camera
    offset while preserving all relative object/fingertip geometry.
    """
    source = np.asarray(image)
    dx, dy = int(dx_px), int(dy_px)
    if dx == 0 and dy == 0:
        return source
    height, width = source.shape[:2]
    fill = np.median(source.reshape(-1, source.shape[-1]), axis=0).astype(source.dtype)
    out = np.empty_like(source)
    out[...] = fill
    src_x0, src_x1 = max(0, -dx), min(width, width - dx)
    src_y0, src_y1 = max(0, -dy), min(height, height - dy)
    if src_x1 <= src_x0 or src_y1 <= src_y0:
        return out
    dst_x0, dst_y0 = src_x0 + dx, src_y0 + dy
    out[
        dst_y0 : dst_y0 + (src_y1 - src_y0),
        dst_x0 : dst_x0 + (src_x1 - src_x0),
    ] = source[src_y0:src_y1, src_x0:src_x1]
    return out


def center_focus_view(image: np.ndarray, crop_size_px: int, output_size_px: int) -> np.ndarray:
    """Crop a square camera focus region and resize it through the shared view path."""
    source = np.asarray(image)
    size = min(max(1, int(crop_size_px)), source.shape[0], source.shape[1])
    y0 = (source.shape[0] - size) // 2
    x0 = (source.shape[1] - size) // 2
    focused = source[y0 : y0 + size, x0 : x0 + size]
    return prepare_view(focused, square_size=int(output_size_px))


class DualIsaaclabMvTokenRunner:
    """Execute one atomic token per arm in a single synchronized physics step loop."""

    def __init__(
        self,
        *,
        env: Any,
        controllers: dict[str, RobolabAtomicController],
        agent: Any,
        logger: EpisodeLogger,
        task: str,
        config: V0Config,
        max_steps: int,
        num_steps_wait: int,
        loop_period_s: float,
        sim_steps_per_decision: int,
        settle_steps_per_decision: int,
        gripper_hold_steps: int,
        debug: bool,
        prompt_log_every: int = 20,
        agentview_rotation_degrees: int = 0,
        wrist_rotation_degrees: int = 0,
        agentview_flip: str = "none",
        wrist_flip: str = "none",
        agentview_crop_aspect: Optional[float] = None,
        wrist_crop_aspect: Optional[float] = None,
        agentview_square_size: Optional[int] = 256,
        wrist_square_size: Optional[int] = 256,
        wrist_translations: dict[str, list[int]] | None = None,
        wrist_focus_crop_px: int | None = None,
    ) -> None:
        self.env = env
        self.controllers = controllers
        self.agent = agent
        self.logger = logger
        self.task = str(task)
        self.config = config
        self.max_steps = int(max_steps)
        self.num_steps_wait = int(num_steps_wait)
        self.loop_period_s = float(loop_period_s)
        self.sim_steps_per_decision = max(1, int(sim_steps_per_decision))
        self.settle_steps_per_decision = max(0, int(settle_steps_per_decision))
        self.gripper_hold_steps = max(1, int(gripper_hold_steps))
        self.debug = bool(debug)
        self.prompt_log_every = int(prompt_log_every)
        self.agentview_rotation_degrees = int(agentview_rotation_degrees)
        self.wrist_rotation_degrees = int(wrist_rotation_degrees)
        self.agentview_flip = str(agentview_flip or "none")
        self.wrist_flip = str(wrist_flip or "none")
        self.agentview_crop_aspect = agentview_crop_aspect
        self.wrist_crop_aspect = wrist_crop_aspect
        self.agentview_square_size = agentview_square_size
        self.wrist_square_size = wrist_square_size
        self.wrist_translations = dict(wrist_translations or {})
        self.wrist_focus_crop_px = (
            int(wrist_focus_crop_px) if wrist_focus_crop_px else None
        )

    def run(self) -> EpisodeResult:
        reset_history = getattr(self.agent, "reset", None)
        if callable(reset_history):
            reset_history()
        for side in SIDES:
            self.controllers[side].set_orientation_reference(None)
            self.controllers[side].open_gripper()
        obs, _, truncated = self.env.reset(settle_steps=self.num_steps_wait)
        for side in SIDES:
            self.controllers[side].set_orientation_reference(self.env.ee_quat(side))

        recent: dict[str, list[str]] = {side: [] for side in SIDES}
        last_descent: dict[str, dict[str, float]] = {side: {} for side in SIDES}
        success = self.env.success()
        end_reason = "max_steps_exceeded"
        steps = 0
        video_path: Any = self.logger.run_dir / "rollout_failure.mp4"
        try:
            for step_idx in range(self.max_steps):
                steps = step_idx + 1
                started = time.monotonic()
                agentview, wrist_left, wrist_right = self._images(obs)
                overview = np.asarray(obs["overview_camera"])
                before = self.env.policy_state()
                response = None
                try:
                    response = self.agent.decide(
                        task=self.task,
                        recent_left=self._recent_text(recent["left"]),
                        recent_right=self._recent_text(recent["right"]),
                        agentview_image=agentview,
                        wrist_left_image=wrist_left,
                        wrist_right_image=wrist_right,
                        debug=self.debug,
                        **self._prompt_state(before, last_descent),
                    )
                    tokens = {side: response.tokens[side] for side in SIDES}
                except (RuntimeError, KeyError, ValueError) as exc:
                    print(
                        f"[realman-dual-sim] step {step_idx}: policy parse failed ({exc}); "
                        "holding both arms"
                    )
                    tokens = {side: FALLBACK_TOKEN for side in SIDES}

                if response is not None and self.prompt_log_every > 0:
                    if step_idx % self.prompt_log_every == 0:
                        prompt = getattr(self.agent, "last_prompt", "")
                        if prompt:
                            self.logger.save_controller_prompt(
                                step_idx,
                                prompt,
                                media=getattr(self.agent, "last_media", None),
                            )

                both_done = all(tokens[side] == DONE_TOKEN for side in SIDES)
                if both_done and success:
                    end_reason = "done_after_success"
                    break
                if both_done:
                    print(
                        f"[realman-dual-sim] step {step_idx}: rejected early DONE; "
                        "both cubes must be placed and both arms returned home"
                    )
                    tokens = {side: STILL_TOKEN for side in SIDES}
                else:
                    for side in SIDES:
                        if tokens[side] == DONE_TOKEN:
                            tokens[side] = STILL_TOKEN

                step_sizes: dict[str, float | None] = {side: None for side in SIDES}
                step_selector = getattr(self.agent, "movement_step_m", None)
                if callable(step_selector) and response is not None:
                    for side in SIDES:
                        if tokens[side] in MOVE_ATOMS:
                            step_sizes[side] = float(
                                step_selector(
                                    side,
                                    tokens[side],
                                    response,
                                    float(before[side]["tcp_position_m"][2]),
                                )
                            )
                motion_frames: dict[str, str | None] = {side: None for side in SIDES}
                frame_selector = getattr(self.agent, "movement_frame", None)
                if callable(frame_selector) and response is not None:
                    for side in SIDES:
                        motion_frames[side] = frame_selector(side, response)
                obs, terminated, truncated = self._execute(
                    tokens,
                    step_sizes=step_sizes,
                    motion_frames=motion_frames,
                )
                success = bool(terminated) or self.env.success()
                after = self.env.policy_state()
                for side in SIDES:
                    last_descent[side] = (
                        {
                            "descend_moved_m": max(
                                0.0,
                                float(before[side]["tcp_position_m"][2])
                                - float(after[side]["tcp_position_m"][2]),
                            ),
                            "descend_commanded_m": float(
                                step_sizes[side] or self.controllers[side].step_m
                            ),
                        }
                        if tokens[side] == "MV_DOWN"
                        else {}
                    )
                self._print_step(step_idx, tokens, before, after, response)
                for side in SIDES:
                    if tokens[side] in HISTORY_TOKENS:
                        recent[side].insert(0, tokens[side])
                        del recent[side][RECENT_MOVES_MAX:]
                zero_shot = (getattr(response, "payload", None) or {}).get("zero_shot")
                if isinstance(zero_shot, dict):
                    for side in zero_shot.get("reset_history", []):
                        if side in recent:
                            recent[side].clear()

                self.logger.log_step(
                    step_idx=step_idx,
                    agentview=agentview,
                    wrist=[wrist_left, wrist_right],
                    record=self._record(
                        step_idx,
                        tokens,
                        before,
                        after,
                        response,
                        success,
                        step_sizes,
                    ),
                    overview=overview,
                )
                if success:
                    end_reason = "success"
                    break
                if truncated:
                    end_reason = "env_truncated"
                    break
                elapsed = time.monotonic() - started
                if self.loop_period_s > elapsed:
                    time.sleep(self.loop_period_s - elapsed)
        finally:
            video_path = self.logger.close(
                success=success,
                fps=min(30.0, max(0.5, float(self.config.video_fps))),
            )
            self.logger.write_summary(
                {
                    "success": success,
                    "steps": steps,
                    "max_steps": self.max_steps,
                    "end_reason": end_reason,
                    "video_path": str(video_path),
                    "run_dir": str(self.logger.run_dir),
                    "control_mode": "realman_dual_isaaclab_mvtoken",
                    "scheme": getattr(self.agent, "scheme", "scripted"),
                    "task": self.task,
                }
            )
        return EpisodeResult(
            success=success,
            steps=steps,
            end_reason=end_reason,
            video_path=str(video_path),
            run_dir=str(self.logger.run_dir),
        )

    def _execute(
        self,
        tokens: dict[str, str],
        *,
        step_sizes: dict[str, float | None] | None = None,
        motion_frames: dict[str, str | None] | None = None,
    ) -> tuple[dict, bool, bool]:
        step_sizes = step_sizes or {}
        motion_frames = motion_frames or {}
        actions: dict[str, np.ndarray] = {}
        durations: dict[str, int] = {}
        for side in SIDES:
            token = tokens[side]
            controller = self.controllers[side]
            if token in MOVE_ATOMS:
                actions[side] = controller.action_for_atomic(
                    token,
                    step_m=step_sizes.get(side),
                    ee_quat=self.env.ee_quat(side),
                    motion_frame=motion_frames.get(side),
                )
                durations[side] = self.sim_steps_per_decision
            elif token == GRASP_TOKEN:
                actions[side] = controller.close_gripper()
                durations[side] = self.gripper_hold_steps
            elif token == RELEASE_TOKEN:
                actions[side] = controller.open_gripper()
                durations[side] = self.gripper_hold_steps
            else:
                actions[side] = controller.hold_action()
                durations[side] = 1

        obs = None
        terminated = truncated = False
        for index in range(max(durations.values())):
            step_actions = {}
            for side in SIDES:
                action = (
                    actions[side]
                    if index < durations[side]
                    else self.controllers[side].hold_action()
                )
                step_actions[side] = self.controllers[side].with_orientation_hold(
                    action, self.env.ee_quat(side)
                )
            obs, terminated, truncated, _ = self.env.step(step_actions)
            if terminated or truncated:
                return obs, terminated, truncated

        if self.settle_steps_per_decision:
            for _ in range(self.settle_steps_per_decision):
                holds = {
                    side: self.controllers[side].with_orientation_hold(
                        self.controllers[side].hold_action(), self.env.ee_quat(side)
                    )
                    for side in SIDES
                }
                obs, terminated, truncated, _ = self.env.step(holds)
                if terminated or truncated:
                    break
        if obs is None:
            raise RuntimeError("dual simulator executed zero physics steps")
        return obs, terminated, truncated

    def _images(self, obs: dict[str, Any]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        agentview = prepare_view(
            obs["agentview_camera"],
            rotation_degrees=self.agentview_rotation_degrees,
            flip=self.agentview_flip,
            crop_aspect=self.agentview_crop_aspect,
            square_size=self.agentview_square_size,
        )
        wrists = []
        for side in SIDES:
            view = prepare_view(
                    obs[f"wrist_{side}_camera"],
                    rotation_degrees=self.wrist_rotation_degrees,
                    flip=self.wrist_flip,
                    crop_aspect=self.wrist_crop_aspect,
                    square_size=self.wrist_square_size,
                )
            translation = self.wrist_translations.get(side, [0, 0])
            view = translate_view(
                view,
                int(translation[0]) if len(translation) > 0 else 0,
                int(translation[1]) if len(translation) > 1 else 0,
            )
            if self.wrist_focus_crop_px:
                view = center_focus_view(
                    view,
                    self.wrist_focus_crop_px,
                    int(self.wrist_square_size or self.wrist_focus_crop_px),
                )
            wrists.append(view)
        return agentview, wrists[0], wrists[1]

    @staticmethod
    def _recent_text(moves: list[str]) -> str:
        return ", ".join(moves) if moves else "none"

    @staticmethod
    def _vec(value) -> str:
        return "[" + ", ".join(f"{float(x):.3f}" for x in value) + "]"

    def _prompt_state(
        self,
        state: dict[str, dict[str, Any]],
        last_descent: dict[str, dict[str, float]] | None = None,
    ) -> dict[str, Any]:
        fields: dict[str, Any] = {}
        for side in SIDES:
            item = state[side]
            fields[f"{side}_phase"] = item["phase"]
            fields[f"{side}_tcp"] = self._vec(item["tcp_position_m"])
            fields[f"{side}_cube"] = self._vec(item["cube_position_m"])
            fields[f"{side}_target"] = self._vec(item["target_position_m"])
            fields[f"{side}_home"] = self._vec(item["home_position_m"])
            fields[f"{side}_width"] = f"{float(item['finger_width_m']):.3f}"
            fields[f"{side}_gripper"] = (
                "closed" if item["gripper_closed"] else "open"
            )
            fields[f"{side}_tcp_position_m"] = item["tcp_position_m"]
            fields[f"{side}_cube_position_m"] = item["cube_position_m"]
            fields[f"{side}_target_position_m"] = item["target_position_m"]
            fields[f"{side}_home_position_m"] = item["home_position_m"]
            fields[f"{side}_finger_width_m"] = float(item["finger_width_m"])
            descent = (last_descent or {}).get(side, {})
            fields[f"{side}_descend_moved_m"] = descent.get("descend_moved_m")
            fields[f"{side}_descend_commanded_m"] = descent.get(
                "descend_commanded_m"
            )
        return fields

    def _record(
        self,
        step_idx: int,
        tokens: dict[str, str],
        before: dict[str, dict[str, Any]],
        after: dict[str, dict[str, Any]],
        response: Any,
        success: bool,
        step_sizes: dict[str, float | None] | None = None,
    ) -> dict[str, Any]:
        record: dict[str, Any] = {"i": int(step_idx), "stage": "dual_pick_place"}
        for side in SIDES:
            tag = side[0].upper()
            record[f"act_{tag}"] = tokens[side]
            record[f"stage_{tag}"] = before[side]["phase"]
            record[f"next_{tag}"] = after[side]["phase"]
            record[f"eef_{tag}"] = [
                round(float(x), 3) for x in after[side]["tcp_position_m"]
            ]
            record[f"w_{tag}"] = round(float(after[side]["finger_width_m"]), 5)
            if step_sizes and step_sizes.get(side) is not None:
                record[f"step_m_{tag}"] = round(float(step_sizes[side]), 5)
            record[f"grip_{tag}"] = (
                "CLOSED" if after[side]["gripper_closed"] else "OPEN"
            )
            record[side] = {
                "stage": before[side]["phase"],
                "dec": tokens[side],
                "grip": record[f"grip_{tag}"],
                "done": after[side]["phase"] == "COMPLETE",
            }
        latency = (getattr(response, "payload", None) or {}).get("latency_s")
        if latency is not None:
            record["vlm_ms"] = int(round(float(latency) * 1000.0))
        if response is not None:
            record["vlm"] = {
                "c": {
                    "reasoning": getattr(response, "reasoning", "") or "",
                    "raw": getattr(response, "raw_text", "") or "",
                }
            }
            if getattr(response, "views", None):
                record["guiding_views"] = dict(response.views)
            zero_shot = (getattr(response, "payload", None) or {}).get("zero_shot")
            if isinstance(zero_shot, dict):
                record["zero_shot"] = zero_shot
        if success:
            record["ok"] = True
            record["done"] = True
        return record

    @staticmethod
    def _print_step(
        step_idx: int,
        tokens: dict[str, str],
        before: dict[str, dict[str, Any]],
        after: dict[str, dict[str, Any]],
        response: Any,
    ) -> None:
        line = " | ".join(
            f"{side[0].upper()} {before[side]['phase']} -> {tokens[side]} -> "
            f"{after[side]['phase']}"
            for side in SIDES
        )
        latency = (getattr(response, "payload", None) or {}).get("latency_s")
        suffix = f" | VLM {float(latency) * 1000:.0f} ms" if latency is not None else ""
        views = getattr(response, "views", None) or {}
        if views:
            suffix += (
                f" | GUIDE L={views.get('left') or '-'}"
                f" R={views.get('right') or '-'}"
            )
        print(f"[realman-dual-sim] {step_idx:03d} {line}{suffix}")


__all__ = ["DualIsaaclabMvTokenRunner", "center_focus_view", "translate_view"]
