"""Adapters that run the repository's original planner/controller prompts in simulation.

The simulator entry points normally use the flat MVTOKEN policies under ``prompts/v3``.
These adapters preserve that path while making the original Show-Harness zero-shot stack
selectable: the co-located subgoal planner produces visual stages, then ``controller.txt``
or ``controller_dual.txt`` controls one stage at a time.  A stage-level ``DONE`` advances
the track; only an exhausted track exposes episode-level ``DONE`` to the simulator runner.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from core.v0_types import Subgoal
from core.vlm.dual_roles import DualControllerAgent, DualDecision, SIDES
from core.vlm.roles import ControllerAgent, DIRECTION_TOKENS
from core.vlm.vlm_client import VLMResponse
from plugins.subgoal import SubgoalPlanner, SubgoalPlannerAgent
from plugins.subgoal.dual_agent import DualSubgoalPlannerAgent
from plugins.subgoal.dual_plugin import DualSubgoalPlanner
from plugins.mem_text import MemTextPlugin
from plugins.proprioception import ProprioceptionPlugin
from plugins.recovery import RecoveryPlugin
from plugins.variable_step import VariableStepPlugin


STILL_TOKEN = "STILL"
DONE_TOKEN = "DONE"


def _previous_direction(recent: str, fallback: str = "") -> str:
    first = str(recent or "").split(",", 1)[0].strip()
    return first if first in DIRECTION_TOKENS else fallback


def _gripper_state(value: str) -> str:
    return "CLOSED" if str(value).strip().lower() in {"closed", "close"} else "OPEN"


def _subgoal_dict(item: Subgoal) -> dict[str, str]:
    return item.to_prompt_dict()


def _goal_error_token(
    eef_pos,
    goal_pos,
    motion: str,
    tolerance_m: float,
    positive_y_token: str,
) -> str | None:
    """Return the atomic token that reduces the largest measured stage error."""
    if eef_pos is None or goal_pos is None or len(eef_pos) < 3 or len(goal_pos) < 3:
        return None
    try:
        delta = [float(goal_pos[i]) - float(eef_pos[i]) for i in range(3)]
    except (TypeError, ValueError):
        return None
    stage = str(motion or "").strip().upper()
    axes = (0, 1) if stage in {"GRASP", "MOVE"} else (0, 1, 2) if stage in {
        "PLACE", "HOME", "RETURN"
    } else ()
    if not axes:
        return None
    axis = max(axes, key=lambda index: abs(delta[index]))
    tolerance = max(0.0, float(tolerance_m))
    horizontal_error = (delta[0] ** 2 + delta[1] ** 2) ** 0.5
    within_tolerance = (
        horizontal_error <= tolerance
        if stage in {"GRASP", "MOVE"}
        else abs(delta[axis]) <= tolerance
    )
    if within_tolerance:
        return "GRASP" if stage == "GRASP" else None
    positive_y = str(positive_y_token).strip().upper()
    negative_y = "MV_RIGHT" if positive_y == "MV_LEFT" else "MV_LEFT"
    positive = ("MV_FWD", positive_y, "MV_UP")
    negative = ("MV_BACK", negative_y, "MV_DOWN")
    return positive[axis] if delta[axis] > 0.0 else negative[axis]


def _placement_error_token(
    eef_pos,
    goal_pos,
    tolerance_m: float,
    positive_y_token: str,
    table_height_m: float | None,
    transport_height_m: float | None,
) -> str | None:
    """Choose a collision-safe correction during a PLACE stage."""
    if eef_pos is None or goal_pos is None or len(eef_pos) < 3 or len(goal_pos) < 3:
        return None
    try:
        delta = [float(goal_pos[i]) - float(eef_pos[i]) for i in range(3)]
        tcp_z = float(eef_pos[2])
    except (TypeError, ValueError):
        return None
    tolerance = max(0.0, float(tolerance_m))
    horizontal_axis = max((0, 1), key=lambda index: abs(delta[index]))
    if (delta[0] ** 2 + delta[1] ** 2) ** 0.5 > tolerance:
        # If lowering introduced lateral drift, regain clearance before translating.
        if transport_height_m is not None and tcp_z < float(transport_height_m) - 0.008:
            return "MV_UP"
        positive_y = str(positive_y_token).strip().upper()
        negative_y = "MV_RIGHT" if positive_y == "MV_LEFT" else "MV_LEFT"
        positive = ("MV_FWD", positive_y)
        negative = ("MV_BACK", negative_y)
        return positive[horizontal_axis] if delta[horizontal_axis] > 0.0 else negative[
            horizontal_axis
        ]
    if table_height_m is not None and tcp_z > float(table_height_m) + 0.005:
        return "MV_DOWN"
    return None


def _within_xy(eef_pos, goal_pos, tolerance_m: float) -> bool:
    """Whether a measured TCP is horizontally aligned with a stage goal."""
    if eef_pos is None or goal_pos is None or len(eef_pos) < 2 or len(goal_pos) < 2:
        return False
    try:
        delta = [float(goal_pos[index]) - float(eef_pos[index]) for index in (0, 1)]
        return (delta[0] ** 2 + delta[1] ** 2) ** 0.5 <= max(
            0.0, float(tolerance_m)
        )
    except (TypeError, ValueError):
        return False


def _object_at_target(cube_pos, target_pos, tolerance_m: float) -> bool:
    """Verify that the manipulated object itself is resting at the destination."""
    if not _within_xy(cube_pos, target_pos, tolerance_m):
        return False
    if cube_pos is None or target_pos is None or len(cube_pos) < 3 or len(target_pos) < 3:
        return False
    try:
        # Targets describe the support surface while cube positions describe their
        # centers. Five centimetres covers the configured 4 cm cubes with margin.
        return float(cube_pos[2]) <= float(target_pos[2]) + 0.05
    except (TypeError, ValueError):
        return False


def _sensor_stage_complete(
    *,
    motion: str,
    holding: bool,
    gripper_closed: bool,
    tcp_position_m,
    cube_position_m,
    target_position_m,
    home_position_m,
    table_height_m: float | None,
    transport_height_m: float | None,
    tolerance_m: float,
    object_tolerance_m: float | None = None,
) -> str | None:
    """Return a measured completion reason for one original-prompt stage.

    The original controller is still responsible for choosing atomic actions. This
    guard only converts already-achieved physical predicates into the planner's
    stage-level ``DONE`` transition, preventing a late visual answer from moving an
    aligned grasp away from its target.
    """
    stage = str(motion or "").strip().upper()
    aligned = _within_xy(tcp_position_m, target_position_m, tolerance_m)
    if stage == "GRASP" and holding:
        return "measured gripper width verifies a held object"
    if stage == "LIFT" and holding:
        try:
            if transport_height_m is not None and float(tcp_position_m[2]) >= float(
                transport_height_m
            ) - 0.008:
                return "measured TCP height verifies safe transport clearance"
        except (TypeError, ValueError, IndexError):
            pass
    if stage == "MOVE" and holding and aligned:
        return "measured TCP-to-target XY error is within alignment tolerance"
    if stage == "PLACE" and aligned and _object_at_target(
        cube_position_m,
        target_position_m,
        tolerance_m if object_tolerance_m is None else object_tolerance_m,
    ):
        try:
            if table_height_m is not None and float(tcp_position_m[2]) <= float(
                table_height_m
            ) + 0.005:
                return "measured target alignment and TCP height verify placement contact"
        except (TypeError, ValueError, IndexError):
            pass
    if stage == "RELEASE" and not gripper_closed:
        return "measured gripper state verifies release"
    if stage == "RETREAT" and not gripper_closed:
        try:
            if transport_height_m is not None and float(tcp_position_m[2]) >= float(
                transport_height_m
            ) - 0.008:
                return "measured TCP height verifies post-release clearance"
        except (TypeError, ValueError, IndexError):
            pass
    if stage in {"HOME", "RETURN"} and not gripper_closed:
        if tcp_position_m is None or home_position_m is None:
            return None
        try:
            delta = [
                abs(float(home_position_m[index]) - float(tcp_position_m[index]))
                for index in range(3)
            ]
        except (TypeError, ValueError, IndexError):
            return None
        if max(delta[:2]) <= max(0.0, float(tolerance_m)) and delta[2] <= 0.010:
            return "measured TCP pose verifies return to home"
    return None


class OriginalZeroShotSingleAgent:
    """Original single-arm subgoal planner + generic controller as a sim policy."""

    scheme = "original_zero_shot"
    last_media = None

    def __init__(
        self,
        *,
        client: Any,
        common_context: str,
        controller_prompt: str,
        plan_dir: str | Path | None = None,
        cot_mode: bool = False,
        gripper_color: str = "black",
        max_subgoal_steps: int = 45,
        max_replans: int = 1,
        planner_max_tokens: int = 4096,
        proprio_plugin: ProprioceptionPlugin | None = None,
        mem_text_plugin: MemTextPlugin | None = None,
        variable_step_plugin: VariableStepPlugin | None = None,
        recovery_plugin: RecoveryPlugin | None = None,
        table_height_m: float | None = None,
        transport_height_m: float | None = None,
        include_goal_error: bool = False,
        enforce_goal_error: bool = False,
        goal_error_tolerance_m: float = 0.015,
        placement_tolerance_m: float | None = None,
        positive_y_token: str = "MV_RIGHT",
        fine_step_m: float = 0.02,
    ) -> None:
        self.planner = SubgoalPlanner(
            SubgoalPlannerAgent(
                client=client,
                common_context=common_context,
                max_tokens=planner_max_tokens,
            )
        )
        self.controller = ControllerAgent(
            client=client,
            prompt_template=controller_prompt,
            common_context=common_context,
            cot_mode=cot_mode,
            gripper_color=gripper_color,
            proprio_plugin=proprio_plugin,
            mem_text_plugin=mem_text_plugin,
            variable_step_plugin=variable_step_plugin,
            table_height_m=table_height_m,
        )
        self.variable_step_plugin = variable_step_plugin
        self.proprio_plugin = proprio_plugin
        self.recovery_plugin = recovery_plugin
        self.table_height_m = table_height_m
        self.transport_height_m = transport_height_m
        self.include_goal_error = bool(include_goal_error)
        self.enforce_goal_error = bool(enforce_goal_error)
        self.goal_error_tolerance_m = float(goal_error_tolerance_m)
        self.placement_tolerance_m = float(
            goal_error_tolerance_m
            if placement_tolerance_m is None
            else placement_tolerance_m
        )
        self.positive_y_token = str(positive_y_token)
        self.fine_step_m = float(fine_step_m)
        self.plan_dir = Path(plan_dir) if plan_dir is not None else None
        self.max_subgoal_steps = max(1, int(max_subgoal_steps))
        self.max_replans = max(0, int(max_replans))
        self.reset()

    def reset(self) -> None:
        if self.recovery_plugin is not None:
            self.recovery_plugin.reset()
        self.subgoals: list[Subgoal] = []
        self.subgoal_index = 0
        self.subgoal_steps = 0
        self.replans = 0
        self.last_prompt = ""
        self.recovery_note = ""
        self._needs_plan = True

    def _plan(self, task: str, agentview, wrist, debug: bool) -> None:
        subgoals, raw = self.planner.plan(
            task,
            agentview,
            wrist,
            debug=debug,
            image_roles=["AgentView", "Wrist view"] if wrist is not None else ["AgentView"],
        )
        self.subgoals = subgoals
        self.subgoal_index = 0
        self.subgoal_steps = 0
        self._needs_plan = False
        self._write_plan(task, raw)
        print(
            "[original-zero-shot] planned "
            f"{len(subgoals)} stages: " + " -> ".join(stage.motion for stage in subgoals)
        )

    def _write_plan(self, task: str, raw: str) -> None:
        if self.plan_dir is None:
            return
        self.plan_dir.mkdir(parents=True, exist_ok=True)
        path = self.plan_dir / f"zero_shot_plan_{self.replans:02d}.json"
        path.write_text(
            json.dumps(
                {
                    "scheme": self.scheme,
                    "task": task,
                    "planner_prompt": self.planner.last_prompt(),
                    "raw_response": raw,
                    "subgoals": [_subgoal_dict(stage) for stage in self.subgoals],
                },
                ensure_ascii=False,
                indent=2,
            ),
            encoding="utf-8",
        )

    def _replan_or_finish(self, task: str, agentview, wrist, debug: bool) -> VLMResponse:
        if self.replans < self.max_replans:
            self.replans += 1
            self._plan(task, agentview, wrist, debug)
            return VLMResponse(
                token=STILL_TOKEN,
                raw_text=json.dumps({"decision": STILL_TOKEN, "reasoning": "replanned"}),
                payload={"zero_shot": {"event": "replan", "replan": self.replans}},
            )
        return VLMResponse(
            token=DONE_TOKEN,
            raw_text=json.dumps({"decision": DONE_TOKEN, "reasoning": "plan exhausted"}),
            payload={"zero_shot": {"event": "plan_exhausted", "replans": self.replans}},
        )

    def decide(
        self,
        *,
        task: str,
        gripper_state: str,
        recent_moves: str,
        agentview_image,
        wrist_image=None,
        debug: bool = False,
        tcp_position_m=None,
        finger_width_m=None,
        cube_position_m=None,
        target_position_m=None,
        home_position_m=None,
        descend_moved_m=None,
        descend_commanded_m=None,
        **_: Any,
    ) -> VLMResponse:
        if self._needs_plan:
            self._plan(task, agentview_image, wrist_image, debug)
        if not self.subgoals or self.subgoal_index >= len(self.subgoals):
            return self._replan_or_finish(task, agentview_image, wrist_image, debug)

        if self.recovery_plugin is not None:
            stage = self.subgoals[self.subgoal_index]
            motion = stage.motion.strip().upper()
            holding = (
                _gripper_state(gripper_state) == "CLOSED"
                and self.recovery_plugin.phase_from_width(finger_width_m) == "holding"
            )
            completion_reason = _sensor_stage_complete(
                motion=motion,
                holding=holding,
                gripper_closed=(_gripper_state(gripper_state) == "CLOSED"),
                tcp_position_m=tcp_position_m,
                cube_position_m=cube_position_m,
                target_position_m=target_position_m,
                home_position_m=home_position_m,
                table_height_m=self.table_height_m,
                transport_height_m=self.transport_height_m,
                tolerance_m=self.goal_error_tolerance_m,
                object_tolerance_m=self.placement_tolerance_m,
            )
            if completion_reason:
                completed_index = self.subgoal_index
                self.subgoal_index += 1
                self.subgoal_steps = 0
                return VLMResponse(
                    token=STILL_TOKEN,
                    raw_text=json.dumps(
                        {"decision": STILL_TOKEN, "reasoning": completion_reason}
                    ),
                    payload={
                        "zero_shot": {
                            "event": "sensor_stage_advance",
                            "stage_index": completed_index,
                            "stage_id": stage.id,
                            "stage_motion": stage.motion,
                        }
                    },
                )
            recovery = self.recovery_plugin.before_decision(
                current_index=self.subgoal_index,
                subgoals=self.subgoals,
                measured_width_m=finger_width_m,
                gripper_closed=(_gripper_state(gripper_state) == "CLOSED"),
            )
            if recovery is not None:
                if recovery.rollback_index is not None:
                    self.subgoal_index = recovery.rollback_index
                    self.subgoal_steps = 0
                self.recovery_note = recovery.prompt_note
                token = STILL_TOKEN if recovery.token == "STOP" else (recovery.token or STILL_TOKEN)
                return VLMResponse(
                    token=token,
                    raw_text=json.dumps(
                        {"decision": token, "reasoning": recovery.reason},
                        ensure_ascii=False,
                    ),
                    payload={
                        "zero_shot": {
                            "event": recovery.event,
                            "stage_index": self.subgoal_index,
                            "reset_history": recovery.reset_history,
                            "recovery": recovery.reason,
                        }
                    },
                )
        if self.subgoal_steps >= self.max_subgoal_steps:
            print(
                f"[original-zero-shot] stage {self.subgoal_index} exceeded "
                f"{self.max_subgoal_steps} decisions; replanning"
            )
            return self._replan_or_finish(task, agentview_image, wrist_image, debug)

        stage = self.subgoals[self.subgoal_index]
        proprio = {
            "eef_pos": tcp_position_m,
            "gripper_width": finger_width_m,
            "gripper_command_name": _gripper_state(gripper_state),
        }
        if self.include_goal_error:
            motion = stage.motion.strip().upper()
            goal = (
                cube_position_m
                if motion == "GRASP"
                else target_position_m
                if motion in {"MOVE", "PLACE", "RELEASE"}
                else home_position_m
                if motion in {"HOME", "RETURN"}
                else None
            )
            if goal is not None:
                proprio["goal_pos"] = goal
        if descend_moved_m is not None and descend_commanded_m is not None:
            proprio.update(
                descend_moved_m=descend_moved_m,
                descend_commanded_m=descend_commanded_m,
            )
        response = self.controller.decide(
            task=task,
            subgoal=_subgoal_dict(stage),
            recent_moves=recent_moves,
            previous_direction=_previous_direction(recent_moves),
            gripper_state=_gripper_state(gripper_state),
            agentview_image=agentview_image,
            wrist_image=wrist_image,
            proprio=proprio,
            recovery_context=(
                self.recovery_plugin.render_prompt_context(self.recovery_note)
                if self.recovery_plugin is not None else ""
            ),
            debug=debug,
        )
        self.recovery_note = ""
        self.last_prompt = self.controller.last_prompt
        proprio_descent_override = bool(
            stage.motion.strip().upper() == "GRASP"
            and self.proprio_plugin is not None
            and self.proprio_plugin.must_descend_first(
                proprio,
                self.table_height_m,
                holding=(_gripper_state(gripper_state) == "CLOSED"),
            )
            and response.token != "MV_DOWN"
        )
        controller_token = response.token
        goal_error_override = False
        expected_goal_token = None
        if self.enforce_goal_error:
            expected_goal_token = (
                _placement_error_token(
                    tcp_position_m,
                    proprio.get("goal_pos"),
                    self.goal_error_tolerance_m,
                    self.positive_y_token,
                    self.table_height_m,
                    self.transport_height_m,
                )
                if stage.motion.strip().upper() == "PLACE"
                else _goal_error_token(
                    tcp_position_m,
                    proprio.get("goal_pos"),
                    stage.motion,
                    self.goal_error_tolerance_m,
                    self.positive_y_token,
                )
            )
            if expected_goal_token and response.token != expected_goal_token:
                response = VLMResponse(
                    token=expected_goal_token,
                    raw_text=response.raw_text,
                    payload=dict(response.payload or {}),
                )
                goal_error_override = True
        # The goal guard may turn an already-correct MV_DOWN into GRASP when XY is
        # aligned. Height safety has final precedence while the open gripper is high.
        proprio_descent_override = proprio_descent_override or bool(
            stage.motion.strip().upper() == "GRASP"
            and self.proprio_plugin is not None
            and self.proprio_plugin.must_descend_first(
                proprio,
                self.table_height_m,
                holding=(_gripper_state(gripper_state) == "CLOSED"),
            )
            and response.token != "MV_DOWN"
        )
        if proprio_descent_override:
            response = VLMResponse(
                token="MV_DOWN",
                raw_text=response.raw_text,
                payload=dict(response.payload or {}),
            )
        token_safety_override = False
        if (
            (response.token == "RELEASE" and stage.motion.strip().upper() != "RELEASE")
            or (response.token == "GRASP" and stage.motion.strip().upper() != "GRASP")
        ):
            response = VLMResponse(
                token=STILL_TOKEN,
                raw_text=response.raw_text,
                payload=dict(response.payload or {}),
            )
            token_safety_override = True
        self.subgoal_steps += 1
        event = "action"
        executed_token = response.token
        if response.token == DONE_TOKEN:
            print(
                f"[original-zero-shot] completed stage {self.subgoal_index}: "
                f"{stage.motion} ({stage.id})"
            )
            self.subgoal_index += 1
            self.subgoal_steps = 0
            if self.subgoal_index < len(self.subgoals):
                executed_token = STILL_TOKEN
                event = "stage_advance"
            else:
                return self._replan_or_finish(task, agentview_image, wrist_image, debug)
        payload = dict(response.payload or {})
        payload["zero_shot"] = {
            "event": event,
            "stage_index": min(self.subgoal_index, len(self.subgoals) - 1),
            "stage_id": stage.id,
            "stage_motion": stage.motion,
            "controller_token": response.token,
            "executed_token": executed_token,
            "replan": self.replans,
            "proprio_descent_override": proprio_descent_override,
            "token_safety_override": token_safety_override,
            "goal_error_override": goal_error_override,
            "expected_goal_token": expected_goal_token,
        }
        payload["zero_shot"]["controller_token"] = controller_token
        return VLMResponse(token=executed_token, raw_text=response.raw_text, payload=payload)

    def movement_step_m(self, token: str, response: VLMResponse, eef_height_m: float) -> float:
        """Return the formal variable-step magnitude for a selected movement token."""
        if self.variable_step_plugin is None:
            return self.fine_step_m
        return self.variable_step_plugin.step_m_for(
            token,
            self.fine_step_m,
            eef_height_m=eef_height_m,
            table_height_m=self.table_height_m,
            target_in_wrist=(response.payload or {}).get("target_in_wrist"),
        )


class OriginalZeroShotDualAgent:
    """Original dual planner + ``controller_dual.txt`` as a two-arm sim policy."""

    scheme = "original_zero_shot"
    last_media = None

    def __init__(
        self,
        *,
        client: Any,
        common_context: str,
        controller_prompt: str,
        plan_dir: str | Path | None = None,
        cot_mode: bool = False,
        max_subgoal_steps: int = 45,
        max_replans: int = 1,
        proprio_plugin: ProprioceptionPlugin | None = None,
        mem_text_plugin: MemTextPlugin | None = None,
        view_select_plugin: Any = None,
        variable_step_plugins: dict[str, VariableStepPlugin] | None = None,
        recovery_plugins: dict[str, RecoveryPlugin] | None = None,
        table_heights: dict[str, float] | None = None,
        transport_heights: dict[str, float] | None = None,
        include_goal_error: bool = False,
        enforce_goal_error: bool = False,
        goal_error_tolerance_m: float = 0.015,
        placement_tolerance_m: float | None = None,
        positive_y_token: str = "MV_RIGHT",
        fine_step_m: float = 0.02,
        empty_width_m: float | None = None,
    ) -> None:
        self.planner = DualSubgoalPlanner(
            DualSubgoalPlannerAgent(client=client, common_context=common_context)
        )
        self.controller = DualControllerAgent(
            client=client,
            prompt_template=controller_prompt,
            common_context=common_context,
            cot_mode=cot_mode,
            proprio_plugin=proprio_plugin,
            mem_text_plugin=mem_text_plugin,
            view_select_plugin=view_select_plugin,
            table_heights=table_heights,
            empty_width_m=empty_width_m,
        )
        self.variable_step_plugins = dict(variable_step_plugins or {})
        self.view_select_plugin = view_select_plugin
        self.proprio_plugin = proprio_plugin
        self.recovery_plugins = dict(recovery_plugins or {})
        self.table_heights = dict(table_heights or {})
        self.transport_heights = dict(transport_heights or {})
        self.include_goal_error = bool(include_goal_error)
        self.enforce_goal_error = bool(enforce_goal_error)
        self.goal_error_tolerance_m = float(goal_error_tolerance_m)
        self.placement_tolerance_m = float(
            goal_error_tolerance_m
            if placement_tolerance_m is None
            else placement_tolerance_m
        )
        self.positive_y_token = str(positive_y_token)
        self.fine_step_m = float(fine_step_m)
        self.plan_dir = Path(plan_dir) if plan_dir is not None else None
        self.max_subgoal_steps = max(1, int(max_subgoal_steps))
        self.max_replans = max(0, int(max_replans))
        self.reset()

    def reset(self) -> None:
        for tool in self.recovery_plugins.values():
            tool.reset()
        self.tracks: dict[str, list[Subgoal]] = {side: [] for side in SIDES}
        self.indices = {side: 0 for side in SIDES}
        self.stage_steps = {side: 0 for side in SIDES}
        self.replans = 0
        self.last_prompt = ""
        self.recovery_notes = {side: "" for side in SIDES}
        self._needs_plan = True

    def _plan(self, task: str, front, wrist_left, wrist_right, debug: bool) -> None:
        tracks, raw = self.planner.plan(
            task, front, wrist_left, wrist_right, debug=debug
        )
        self.tracks = tracks
        self.indices = {side: 0 for side in SIDES}
        self.stage_steps = {side: 0 for side in SIDES}
        self._needs_plan = False
        self._write_plan(task, raw)
        print(
            "[original-zero-shot-dual] planned "
            + " | ".join(
                f"{side}: " + " -> ".join(stage.motion for stage in tracks[side])
                for side in SIDES
            )
        )

    def _write_plan(self, task: str, raw: str) -> None:
        if self.plan_dir is None:
            return
        self.plan_dir.mkdir(parents=True, exist_ok=True)
        path = self.plan_dir / f"zero_shot_plan_{self.replans:02d}.json"
        path.write_text(
            json.dumps(
                {
                    "scheme": self.scheme,
                    "task": task,
                    "raw_response": raw,
                    "tracks": {
                        side: [_subgoal_dict(stage) for stage in self.tracks[side]]
                        for side in SIDES
                    },
                },
                ensure_ascii=False,
                indent=2,
            ),
            encoding="utf-8",
        )

    def _current(self, side: str) -> Subgoal | None:
        index = self.indices[side]
        return self.tracks[side][index] if index < len(self.tracks[side]) else None

    def _all_exhausted(self) -> bool:
        return all(self._current(side) is None for side in SIDES)

    def _replan_or_finish(
        self, task: str, front, wrist_left, wrist_right, debug: bool
    ) -> DualDecision:
        if self.replans < self.max_replans:
            self.replans += 1
            self._plan(task, front, wrist_left, wrist_right, debug)
            return DualDecision(
                tokens={side: STILL_TOKEN for side in SIDES},
                reasoning="Replanned after the previous visual plan ended or stalled.",
                raw_text=json.dumps({side: STILL_TOKEN for side in SIDES}),
                payload={"zero_shot": {"event": "replan", "replan": self.replans}},
            )
        return DualDecision(
            tokens={side: DONE_TOKEN for side in SIDES},
            reasoning="Both visual stage tracks are exhausted.",
            raw_text=json.dumps({side: DONE_TOKEN for side in SIDES}),
            payload={"zero_shot": {"event": "plan_exhausted", "replans": self.replans}},
        )

    def decide(
        self,
        *,
        task: str,
        recent_left: str,
        recent_right: str,
        agentview_image,
        wrist_left_image,
        wrist_right_image,
        debug: bool = False,
        **state: Any,
    ) -> DualDecision:
        if self._needs_plan:
            self._plan(task, agentview_image, wrist_left_image, wrist_right_image, debug)
        if self._all_exhausted():
            return self._replan_or_finish(
                task, agentview_image, wrist_left_image, wrist_right_image, debug
            )
        if any(
            self._current(side) is not None
            and self.stage_steps[side] >= self.max_subgoal_steps
            for side in SIDES
        ):
            print("[original-zero-shot-dual] a stage exceeded its decision limit; replanning")
            return self._replan_or_finish(
                task, agentview_image, wrist_left_image, wrist_right_image, debug
            )

        recent = {"left": recent_left, "right": recent_right}
        current = {side: self._current(side) for side in SIDES}
        sensor_advanced: list[str] = []
        sensor_reasons: dict[str, str] = {}
        for side in SIDES:
            stage = current[side]
            tool = self.recovery_plugins.get(side)
            motion = stage.motion.strip().upper() if stage is not None else ""
            holding = (
                stage is not None
                and tool is not None
                and _gripper_state(str(state.get(f"{side}_gripper", "open")))
                == "CLOSED"
                and tool.phase_from_width(state.get(f"{side}_finger_width_m"))
                == "holding"
            )
            completion_reason = _sensor_stage_complete(
                motion=motion,
                holding=holding,
                gripper_closed=(
                    _gripper_state(str(state.get(f"{side}_gripper", "open")))
                    == "CLOSED"
                ),
                tcp_position_m=state.get(f"{side}_tcp_position_m"),
                cube_position_m=state.get(f"{side}_cube_position_m"),
                target_position_m=state.get(f"{side}_target_position_m"),
                home_position_m=state.get(f"{side}_home_position_m"),
                table_height_m=self.table_heights.get(side),
                transport_height_m=self.transport_heights.get(side),
                tolerance_m=self.goal_error_tolerance_m,
                object_tolerance_m=self.placement_tolerance_m,
            )
            if completion_reason:
                self.indices[side] += 1
                self.stage_steps[side] = 0
                current[side] = self._current(side)
                sensor_advanced.append(side)
                sensor_reasons[side] = completion_reason

        # Width-based recovery runs after measured stage completion. At placement
        # contact a gripper may read as empty because the cube is now supported by the
        # target; that is a successful PLACE predicate, not a lost grasp.
        recovery_overrides: dict[str, Any] = {}
        for side in SIDES:
            tool = self.recovery_plugins.get(side)
            stage = current[side]
            if tool is None or stage is None or side in sensor_advanced:
                continue
            decision = tool.before_decision(
                current_index=self.indices[side],
                subgoals=self.tracks[side],
                measured_width_m=state.get(f"{side}_finger_width_m"),
                gripper_closed=(
                    _gripper_state(str(state.get(f"{side}_gripper", "open"))) == "CLOSED"
                ),
            )
            if decision is not None:
                if decision.rollback_index is not None:
                    self.indices[side] = decision.rollback_index
                    self.stage_steps[side] = 0
                    current[side] = self._current(side)
                self.recovery_notes[side] = decision.prompt_note
                recovery_overrides[side] = decision

        proprios = {
            side: {
                "eef_pos": state.get(f"{side}_tcp_position_m"),
                "gripper_width": state.get(f"{side}_finger_width_m"),
                "gripper_command_name": _gripper_state(
                    str(state.get(f"{side}_gripper", "open"))
                ),
                **(
                    {
                        "descend_moved_m": state[f"{side}_descend_moved_m"],
                        "descend_commanded_m": state[f"{side}_descend_commanded_m"],
                    }
                    if state.get(f"{side}_descend_moved_m") is not None
                    and state.get(f"{side}_descend_commanded_m") is not None
                    else {}
                ),
            }
            for side in SIDES
        }
        if self.include_goal_error:
            for side in SIDES:
                stage = current[side]
                if stage is None:
                    continue
                motion = stage.motion.strip().upper()
                goal_key = (
                    f"{side}_cube_position_m"
                    if motion == "GRASP"
                    else f"{side}_target_position_m"
                    if motion in {"MOVE", "PLACE", "RELEASE"}
                    else f"{side}_home_position_m"
                    if motion in {"HOME", "RETURN"}
                    else ""
                )
                if goal_key and state.get(goal_key) is not None:
                    proprios[side]["goal_pos"] = state[goal_key]
        response = self.controller.decide(
            task=task,
            subgoals={
                side: (_subgoal_dict(current[side]) if current[side] is not None else None)
                for side in SIDES
            },
            gripper_states={
                side: _gripper_state(str(state.get(f"{side}_gripper", "open")))
                for side in SIDES
            },
            recent_moves=recent,
            previous_directions={
                side: _previous_direction(recent[side], STILL_TOKEN) for side in SIDES
            },
            recovery_contexts={
                side: (
                    self.recovery_plugins[side].render_prompt_context(
                        self.recovery_notes[side]
                    )
                    if side in self.recovery_plugins else ""
                )
                for side in SIDES
            },
            proprios=proprios,
            agentview_image=agentview_image,
            wrist_left_image=wrist_left_image,
            wrist_right_image=wrist_right_image,
            debug=debug,
        )
        self.last_prompt = self.controller.last_prompt
        executed = dict(response.tokens)
        proprio_overrides: list[str] = []
        for side in SIDES:
            stage = current[side]
            if (
                stage is not None
                and stage.motion.strip().upper() == "GRASP"
                and self.proprio_plugin is not None
                and self.proprio_plugin.must_descend_first(
                    proprios[side],
                    self.table_heights.get(side),
                    holding=(
                        _gripper_state(str(state.get(f"{side}_gripper", "open")))
                        == "CLOSED"
                    ),
                )
                and executed[side] != "MV_DOWN"
            ):
                executed[side] = "MV_DOWN"
                proprio_overrides.append(side)
        token_safety_overrides: list[str] = []
        goal_error_overrides: dict[str, str] = {}
        if self.enforce_goal_error:
            for side in SIDES:
                stage = current[side]
                if stage is None or side in recovery_overrides or side in proprio_overrides:
                    continue
                expected = (
                    _placement_error_token(
                        proprios[side].get("eef_pos"),
                        proprios[side].get("goal_pos"),
                        self.goal_error_tolerance_m,
                        self.positive_y_token,
                        self.table_heights.get(side),
                        self.transport_heights.get(side),
                    )
                    if stage.motion.strip().upper() == "PLACE"
                    else _goal_error_token(
                        proprios[side].get("eef_pos"),
                        proprios[side].get("goal_pos"),
                        stage.motion,
                        self.goal_error_tolerance_m,
                        self.positive_y_token,
                    )
                )
                if expected and executed[side] != expected:
                    executed[side] = expected
                    goal_error_overrides[side] = expected
        # As in the single-arm path, measured height wins if the XY guard selected
        # GRASP before the open gripper reached its calibrated contact height.
        for side in SIDES:
            stage = current[side]
            if (
                stage is not None
                and side not in recovery_overrides
                and side not in proprio_overrides
                and stage.motion.strip().upper() == "GRASP"
                and self.proprio_plugin is not None
                and self.proprio_plugin.must_descend_first(
                    proprios[side],
                    self.table_heights.get(side),
                    holding=(
                        _gripper_state(str(state.get(f"{side}_gripper", "open")))
                        == "CLOSED"
                    ),
                )
                and executed[side] != "MV_DOWN"
            ):
                executed[side] = "MV_DOWN"
                proprio_overrides.append(side)
        for side in SIDES:
            stage = current[side]
            if stage is None or side in recovery_overrides:
                continue
            motion = stage.motion.strip().upper()
            if (
                (executed[side] == "RELEASE" and motion != "RELEASE")
                or (executed[side] == "GRASP" and motion != "GRASP")
            ):
                executed[side] = STILL_TOKEN
                token_safety_overrides.append(side)
        recovery_info: dict[str, Any] = {}
        reset_history: list[str] = []
        for side, decision in recovery_overrides.items():
            executed[side] = (
                STILL_TOKEN if decision.token == "STOP" else (decision.token or STILL_TOKEN)
            )
            recovery_info[side] = {"event": decision.event, "reason": decision.reason}
            if decision.reset_history:
                reset_history.append(side)
        self.recovery_notes = {side: "" for side in SIDES}
        stage_info: dict[str, Any] = {}
        for side in SIDES:
            stage = current[side]
            if stage is None:
                executed[side] = STILL_TOKEN
                stage_info[side] = {"finished": True}
                continue
            if side not in recovery_overrides and side not in proprio_overrides:
                self.stage_steps[side] += 1
            stage_info[side] = {
                "index": self.indices[side],
                "id": stage.id,
                "motion": stage.motion,
                "controller_token": response.tokens[side],
            }
            if (
                side not in recovery_overrides
                and side not in proprio_overrides
                and response.tokens[side] == DONE_TOKEN
            ):
                print(
                    f"[original-zero-shot-dual] {side} completed stage "
                    f"{self.indices[side]}: {stage.motion} ({stage.id})"
                )
                self.indices[side] += 1
                self.stage_steps[side] = 0
                executed[side] = STILL_TOKEN

        payload = dict(response.payload or {})
        payload["zero_shot"] = {
            "event": "stage_control",
            "stages": stage_info,
            "executed_tokens": executed,
            "replan": self.replans,
            "recovery": recovery_info,
            "reset_history": reset_history,
            "proprio_descent_overrides": proprio_overrides,
            "sensor_stage_advances": sensor_advanced,
            "sensor_stage_reasons": sensor_reasons,
            "token_safety_overrides": token_safety_overrides,
            "goal_error_overrides": goal_error_overrides,
        }
        if self._all_exhausted():
            # Replan on the next observation if the simulator has not already accepted
            # success. This mirrors the original runner's visual replan opportunity.
            payload["zero_shot"]["event"] = "tracks_exhausted"
        return DualDecision(
            tokens=executed,
            reasoning=response.reasoning,
            raw_text=response.raw_text,
            payload=payload,
            views=response.views,
        )

    def movement_step_m(
        self, side: str, token: str, response: DualDecision, eef_height_m: float
    ) -> float:
        tool = self.variable_step_plugins.get(side)
        if tool is None:
            return self.fine_step_m
        target_in_wrist = None
        if getattr(response, "views", None):
            selected = str(response.views.get(side) or "").upper()
            target_in_wrist = (
                True if selected == "WRIST" else (False if selected == "FRONT" else None)
            )
        return tool.step_m_for(
            token,
            self.fine_step_m,
            eef_height_m=eef_height_m,
            table_height_m=self.table_heights.get(side),
            target_in_wrist=target_in_wrist,
        )

    def movement_frame(self, side: str, response: DualDecision) -> str | None:
        """Map the controller's per-arm guiding view to its calibrated motion frame."""
        zero_shot = (response.payload or {}).get("zero_shot", {})
        # Recovery nudges and goal-error corrections are defined explicitly in base
        # axes. Do not rotate them merely because the discarded model answer named a
        # wrist view.
        if (
            side in zero_shot.get("recovery", {})
            or side in zero_shot.get("goal_error_overrides", {})
        ):
            return "base"
        if self.view_select_plugin is None:
            return None
        selected = (response.views or {}).get(side)
        return self.view_select_plugin.frame_for(selected)


__all__ = ["OriginalZeroShotSingleAgent", "OriginalZeroShotDualAgent"]
