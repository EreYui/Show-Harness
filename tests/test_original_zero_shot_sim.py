from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from core.sim.original_zero_shot import (
    OriginalZeroShotDualAgent,
    OriginalZeroShotSingleAgent,
    _placement_error_token,
    _sensor_stage_complete,
)
from plugins.mem_text import MemTextPlugin
from plugins.proprioception import ProprioceptionPlugin
from plugins.recovery import RecoveryPlugin
from plugins.variable_step import VariableStepPlugin
from plugins.coords import CoordsPlugin
from plugins.ego import EgoPlugin


def _stage(stage_id: str, motion: str) -> dict[str, str]:
    return {
        "id": stage_id,
        "target": "red cube",
        "affordance": "main body",
        "motion": motion,
        "description": f"perform {motion.lower()}",
        "completion": f"{motion.lower()} is visibly complete",
    }


class _FakeClient:
    def __init__(self, plan: dict, decisions: list[dict]) -> None:
        self.plan = plan
        self.decisions = list(decisions)
        self.planner_calls = 0

    def complete_json(self, prompt, image, wrist_image=None, schema=None, **kwargs):
        properties = (schema or {}).get("properties", {})
        dual_plan = (
            {"left", "right"}.issubset(properties)
            and properties.get("left", {}).get("type") == "array"
        )
        if "subgoals" in properties or dual_plan:
            self.planner_calls += 1
            payload = self.plan
        else:
            payload = self.decisions.pop(0)
        return SimpleNamespace(payload={"json": payload, "latency_s": 0.01}, raw_text=str(payload), token="")


class OriginalZeroShotSimTest(unittest.TestCase):
    def setUp(self) -> None:
        self.image = np.zeros((8, 8, 3), dtype=np.uint8)

    def test_coords_plugin_rewrites_dual_direction_contract(self):
        source = (
            Path(__file__).resolve().parents[1] / "prompts" / "controller_dual.txt"
        ).read_text(encoding="utf-8")
        result = CoordsPlugin(
            enabled=True,
            positive_y_token="MV_LEFT",
            positive_y_image_edge="left",
        ).apply(source)
        result = EgoPlugin(enabled=True).apply(result)
        self.assertIn("ROBOT BASE AXES:", result)
        self.assertIn("MV_FWD  -> move along +x", result)
        self.assertIn("MV_LEFT  -> move along +y", result)
        self.assertIn("AFFORD to gripper's left -> MV_LEFT", result)
        self.assertIn("AFFORD near image bottom and far from fingertips -> MV_BACK", result)
        self.assertIn("AFFORD between image top and fingertips -> MV_FWD", result)
        self.assertIn("COORDINATION:", result)
        self.assertIn("GRIPPER (per arm):", result)
        self.assertIn("Prefer x/y alignment", result)
        self.assertIn("{mem_rules}", result)
        self.assertNotIn("{mem_text_rules}", result)
        self.assertNotIn("After a RELEASE, MV_UP first", result)

    def test_recovery_applies_configured_empty_grasp_realign_once(self):
        recovery = RecoveryPlugin(
            enabled=True,
            empty_width_m=0.005,
            open_width_m=0.08,
            empty_realign_token="MV_FWD",
        )
        stages = [SimpleNamespace(motion="GRASP")]
        empty = recovery.before_decision(
            current_index=0,
            subgoals=stages,
            measured_width_m=0.001,
            gripper_closed=True,
        )
        realign = recovery.before_decision(
            current_index=0,
            subgoals=stages,
            measured_width_m=0.08,
            gripper_closed=False,
        )
        resumed = recovery.before_decision(
            current_index=0,
            subgoals=stages,
            measured_width_m=0.08,
            gripper_closed=False,
        )
        self.assertEqual(empty.token, "RELEASE")
        self.assertEqual(realign.token, "MV_FWD")
        self.assertEqual(realign.event, "empty_grasp_realign")
        self.assertIsNone(resumed)

    def test_sensor_stage_completion_covers_full_pick_place_closure(self):
        common = dict(
            holding=True,
            gripper_closed=True,
            tcp_position_m=[0.76, 0.15, 0.80],
            cube_position_m=[0.76, 0.15, 0.701],
            target_position_m=[0.76, 0.15, 0.68],
            home_position_m=[0.62, 0.15, 0.80],
            table_height_m=0.703,
            transport_height_m=0.80,
            tolerance_m=0.015,
        )
        self.assertIsNotNone(_sensor_stage_complete(motion="MOVE", **common))
        self.assertIsNotNone(
            _sensor_stage_complete(
                motion="PLACE", **{**common, "tcp_position_m": [0.76, 0.15, 0.705]}
            )
        )
        open_state = {**common, "holding": False, "gripper_closed": False}
        self.assertIsNotNone(_sensor_stage_complete(motion="RELEASE", **open_state))
        self.assertIsNotNone(_sensor_stage_complete(motion="RETREAT", **open_state))
        self.assertIsNotNone(
            _sensor_stage_complete(
                motion="RETURN",
                **{**open_state, "tcp_position_m": [0.625, 0.15, 0.805]},
            )
        )
        self.assertIsNone(
            _sensor_stage_complete(
                motion="MOVE",
                **{**common, "tcp_position_m": [0.72, 0.15, 0.80]},
            )
        )
        self.assertIsNone(
            _sensor_stage_complete(
                motion="PLACE",
                **{
                    **common,
                    "tcp_position_m": [0.76, 0.15, 0.705],
                    "cube_position_m": [0.62, 0.15, 0.701],
                },
            )
        )

    def test_place_guard_lifts_before_correcting_low_horizontal_drift(self):
        common = dict(
            goal_pos=[0.76, 0.15, 0.68],
            tolerance_m=0.015,
            positive_y_token="MV_LEFT",
            table_height_m=0.703,
            transport_height_m=0.80,
        )
        self.assertEqual(
            _placement_error_token(eef_pos=[0.73, 0.15, 0.72], **common),
            "MV_UP",
        )
        self.assertEqual(
            _placement_error_token(eef_pos=[0.745, 0.144, 0.72], **common),
            "MV_UP",
        )
        self.assertEqual(
            _placement_error_token(eef_pos=[0.73, 0.15, 0.80], **common),
            "MV_FWD",
        )
        self.assertEqual(
            _placement_error_token(eef_pos=[0.76, 0.15, 0.75], **common),
            "MV_DOWN",
        )
        self.assertIsNone(
            _placement_error_token(eef_pos=[0.76, 0.15, 0.705], **common)
        )

    def test_single_sensor_advances_lift_at_safe_transport_height(self):
        client = _FakeClient(
            {
                "subgoals": [
                    _stage("grasp", "GRASP"),
                    _stage("lift", "LIFT"),
                ]
            },
            [],
        )
        recovery = RecoveryPlugin(
            enabled=True, empty_width_m=0.005, open_width_m=0.08
        )
        agent = OriginalZeroShotSingleAgent(
            client=client,
            common_context="context",
            controller_prompt=(
                "{task} {stage} {target} {affordance} {description} {completion} "
                "{gripper_state} {gripper_color} {recent_moves} {mem_text} "
                "{mem_text_rules} {variable_step} {action_chunk} {rotation} "
                "{recovery} {proprio} {gripper_proprio} {subgoal_json} {output_contract}"
            ),
            max_replans=0,
            recovery_plugin=recovery,
            transport_height_m=0.12,
        )
        common = dict(
            task="pick and lift",
            gripper_state="closed",
            recent_moves="none",
            agentview_image=self.image,
            wrist_image=self.image,
            finger_width_m=0.02,
        )
        grasp_done = agent.decide(tcp_position_m=[0.3, 0.0, 0.04], **common)
        lift_done = agent.decide(tcp_position_m=[0.3, 0.0, 0.115], **common)
        self.assertEqual(grasp_done.payload["zero_shot"]["event"], "sensor_stage_advance")
        self.assertEqual(lift_done.payload["zero_shot"]["event"], "sensor_stage_advance")
        self.assertIn("transport clearance", lift_done.raw_text)

    def test_single_done_advances_stage_without_ending_episode(self):
        client = _FakeClient(
            {"subgoals": [_stage("grasp", "GRASP"), _stage("lift", "LIFT")]},
            [
                {"decision": "DONE", "reasoning": "grasp stage complete"},
                {"decision": "MV_UP", "reasoning": "lift the cube"},
            ],
        )
        with tempfile.TemporaryDirectory() as temp_dir:
            agent = OriginalZeroShotSingleAgent(
                client=client,
                common_context="context",
                controller_prompt=(
                    "{task} {stage} {target} {affordance} {description} {completion} "
                    "{gripper_state} {gripper_color} {recent_moves} {mem_text} "
                    "{mem_text_rules} {variable_step} {action_chunk} {rotation} "
                    "{recovery} {proprio} {gripper_proprio} {subgoal_json} {output_contract}"
                ),
                plan_dir=temp_dir,
                max_replans=0,
            )
            first = agent.decide(
                task="pick and lift",
                gripper_state="closed",
                recent_moves="none",
                agentview_image=self.image,
                wrist_image=self.image,
            )
            second = agent.decide(
                task="pick and lift",
                gripper_state="closed",
                recent_moves="none",
                agentview_image=self.image,
                wrist_image=self.image,
            )
            self.assertEqual(first.token, "STILL")
            self.assertEqual(first.payload["zero_shot"]["event"], "stage_advance")
            self.assertEqual(second.token, "MV_UP")
            self.assertEqual(client.planner_calls, 1)
            self.assertTrue((Path(temp_dir) / "zero_shot_plan_00.json").is_file())

    def test_dual_done_is_per_arm_and_tracks_finish_together(self):
        client = _FakeClient(
            {"left": [_stage("left_grasp", "GRASP")], "right": [_stage("right_grasp", "GRASP")]},
            [
                {"left": "DONE", "right": "MV_UP", "reasoning": "left is ready"},
                {"left": "STILL", "right": "DONE", "reasoning": "right is ready"},
            ],
        )
        agent = OriginalZeroShotDualAgent(
            client=client,
            common_context="context",
            controller_prompt=(
                "{task} {stage_l} {target_l} {afford_l} {description_l} {completion_l} "
                "{gripper_l} {mem_l} {recovery_l} {proprio_l} {stage_r} {target_r} "
                "{afford_r} {description_r} {completion_r} {gripper_r} {mem_r} "
                "{recovery_r} {proprio_r} {mem_rules} {output_contract}"
            ),
            max_replans=0,
        )
        kwargs = dict(
            task="move both cubes",
            recent_left="none",
            recent_right="none",
            agentview_image=self.image,
            wrist_left_image=self.image,
            wrist_right_image=self.image,
            left_gripper="open",
            right_gripper="open",
        )
        first = agent.decide(**kwargs)
        second = agent.decide(**kwargs)
        final = agent.decide(**kwargs)
        self.assertEqual(first.tokens, {"left": "STILL", "right": "MV_UP"})
        self.assertEqual(second.tokens, {"left": "STILL", "right": "STILL"})
        self.assertEqual(final.tokens, {"left": "DONE", "right": "DONE"})
        self.assertEqual(client.planner_calls, 1)

    def test_single_formal_plugins_render_feedback_and_select_fine_step(self):
        client = _FakeClient(
            {"subgoals": [_stage("approach", "MOVE")]},
            [{"decision": "MV_DOWN", "reasoning": "WRIST: YES; descend to the cube"}],
        )
        variable = VariableStepPlugin(
            enabled=True, coarse_step_m=0.023, high_above_table_m=0.08
        )
        agent = OriginalZeroShotSingleAgent(
            client=client,
            common_context="context",
            controller_prompt=(
                "{task} {stage} {target} {affordance} {description} {completion} "
                "{gripper_state} {gripper_color} {recent_moves} {mem_text} "
                "{mem_text_rules} {variable_step} {action_chunk} {rotation} "
                "{recovery} {proprio} {gripper_proprio} {subgoal_json} {output_contract}"
            ),
            max_replans=0,
            proprio_plugin=ProprioceptionPlugin(
                enabled=True,
                high_above_table_m=0.08,
                fine_step_m=0.01,
                coarse_step_m=0.023,
            ),
            mem_text_plugin=MemTextPlugin(enabled=True, max_recent=5),
            variable_step_plugin=variable,
            recovery_plugin=RecoveryPlugin(
                enabled=True, empty_width_m=0.005, open_width_m=0.08
            ),
            table_height_m=0.025,
            include_goal_error=True,
            fine_step_m=0.01,
        )
        response = agent.decide(
            task="pick",
            gripper_state="open",
            recent_moves="MV_FWD",
            agentview_image=self.image,
            wrist_image=self.image,
            tcp_position_m=[0.30, 0.0, 0.07],
            finger_width_m=0.10,
            descend_moved_m=0.001,
            descend_commanded_m=0.01,
            target_position_m=[0.34, 0.02, 0.025],
        )
        self.assertEqual(response.token, "MV_DOWN")
        self.assertIn("Recent moves", agent.last_prompt)
        self.assertIn("above the table", agent.last_prompt)
        self.assertIn("Last MV_DOWN", agent.last_prompt)
        self.assertIn("base-axis error", agent.last_prompt)
        self.assertIn("dx=+4.0 cm", agent.last_prompt)
        self.assertTrue(response.payload["target_in_wrist"])
        self.assertEqual(agent.movement_step_m("MV_DOWN", response, 0.07), 0.01)

    def test_proprioception_enforces_descent_before_grasp(self):
        client = _FakeClient(
            {"subgoals": [_stage("grasp", "GRASP")]},
            [{"decision": "MV_RIGHT", "reasoning": "visual lateral guess"}],
        )
        proprio = ProprioceptionPlugin(
            enabled=True, high_above_table_m=0.005, fine_step_m=0.01
        )
        agent = OriginalZeroShotSingleAgent(
            client=client,
            common_context="context",
            controller_prompt=(
                "{task} {stage} {target} {affordance} {description} {completion} "
                "{gripper_state} {gripper_color} {recent_moves} {mem_text} "
                "{mem_text_rules} {variable_step} {action_chunk} {rotation} "
                "{recovery} {proprio} {gripper_proprio} {subgoal_json} {output_contract}"
            ),
            max_replans=0,
            proprio_plugin=proprio,
            table_height_m=0.025,
            fine_step_m=0.01,
        )
        response = agent.decide(
            task="pick",
            gripper_state="open",
            recent_moves="none",
            agentview_image=self.image,
            wrist_image=self.image,
            tcp_position_m=[0.30, 0.0, 0.05],
            finger_width_m=0.10,
        )
        self.assertEqual(response.token, "MV_DOWN")
        self.assertEqual(response.payload["zero_shot"]["controller_token"], "MV_RIGHT")
        self.assertTrue(response.payload["zero_shot"]["proprio_descent_override"])

    def test_single_sensor_advances_verified_grasp_and_blocks_early_release(self):
        client = _FakeClient(
            {
                "subgoals": [
                    _stage("grasp", "GRASP"),
                    _stage("lift", "LIFT"),
                ]
            },
            [{"decision": "RELEASE", "reasoning": "premature visual guess"}],
        )
        recovery = RecoveryPlugin(
            enabled=True, empty_width_m=0.005, open_width_m=0.08
        )
        agent = OriginalZeroShotSingleAgent(
            client=client,
            common_context="context",
            controller_prompt=(
                "{task} {stage} {target} {affordance} {description} {completion} "
                "{gripper_state} {gripper_color} {recent_moves} {mem_text} "
                "{mem_text_rules} {variable_step} {action_chunk} {rotation} "
                "{recovery} {proprio} {gripper_proprio} {subgoal_json} {output_contract}"
            ),
            max_replans=0,
            recovery_plugin=recovery,
        )
        common = dict(
            task="pick and place",
            gripper_state="closed",
            recent_moves="none",
            agentview_image=self.image,
            wrist_image=self.image,
            finger_width_m=0.02,
        )
        verified = agent.decide(**common)
        protected = agent.decide(**common)
        self.assertEqual(verified.token, "STILL")
        self.assertEqual(verified.payload["zero_shot"]["event"], "sensor_stage_advance")
        self.assertEqual(protected.token, "STILL")
        self.assertEqual(protected.payload["zero_shot"]["controller_token"], "RELEASE")
        self.assertTrue(protected.payload["zero_shot"]["token_safety_override"])

    def test_dual_sensor_advances_both_verified_grasps_before_control(self):
        plan = {
            side: [_stage(f"{side}_grasp", "GRASP"), _stage(f"{side}_lift", "LIFT")]
            for side in ("left", "right")
        }
        client = _FakeClient(
            plan,
            [{"left": "MV_UP", "right": "MV_UP", "reasoning": "lift both"}],
        )
        agent = OriginalZeroShotDualAgent(
            client=client,
            common_context="context",
            controller_prompt=(
                "{task} {stage_l} {target_l} {afford_l} {description_l} {completion_l} "
                "{gripper_l} {mem_l} {recovery_l} {proprio_l} {stage_r} {target_r} "
                "{afford_r} {description_r} {completion_r} {gripper_r} {mem_r} "
                "{recovery_r} {proprio_r} {mem_rules} {output_contract}"
            ),
            max_replans=0,
            recovery_plugins={
                side: RecoveryPlugin(
                    enabled=True, empty_width_m=0.005, open_width_m=0.08
                )
                for side in ("left", "right")
            },
        )
        response = agent.decide(
            task="move both cubes",
            recent_left="none",
            recent_right="none",
            agentview_image=self.image,
            wrist_left_image=self.image,
            wrist_right_image=self.image,
            left_gripper="closed",
            right_gripper="closed",
            left_finger_width_m=0.02,
            right_finger_width_m=0.02,
        )
        self.assertEqual(response.tokens, {"left": "MV_UP", "right": "MV_UP"})
        self.assertEqual(
            response.payload["zero_shot"]["sensor_stage_advances"],
            ["left", "right"],
        )
        self.assertEqual(agent.indices, {"left": 1, "right": 1})

    def test_dual_verified_place_wins_over_empty_width_recovery(self):
        plan = {
            side: [
                _stage(f"{side}_place", "PLACE"),
                _stage(f"{side}_release", "RELEASE"),
            ]
            for side in ("left", "right")
        }
        client = _FakeClient(
            plan,
            [{"left": "RELEASE", "right": "RELEASE", "reasoning": "open both"}],
        )
        agent = OriginalZeroShotDualAgent(
            client=client,
            common_context="context",
            controller_prompt=(
                "{task} {stage_l} {target_l} {afford_l} {description_l} {completion_l} "
                "{gripper_l} {mem_l} {recovery_l} {proprio_l} {stage_r} {target_r} "
                "{afford_r} {description_r} {completion_r} {gripper_r} {mem_r} "
                "{recovery_r} {proprio_r} {mem_rules} {output_contract}"
            ),
            max_replans=0,
            recovery_plugins={
                side: RecoveryPlugin(
                    enabled=True, empty_width_m=0.0199, open_width_m=0.045
                )
                for side in ("left", "right")
            },
            table_heights={"left": 0.703, "right": 0.703},
            goal_error_tolerance_m=0.015,
        )
        response = agent.decide(
            task="place both cubes",
            recent_left="none",
            recent_right="none",
            agentview_image=self.image,
            wrist_left_image=self.image,
            wrist_right_image=self.image,
            left_gripper="closed",
            right_gripper="closed",
            left_finger_width_m=0.018,
            right_finger_width_m=0.018,
            left_tcp_position_m=[0.76, 0.15, 0.705],
            right_tcp_position_m=[0.76, -0.15, 0.705],
            left_cube_position_m=[0.76, 0.15, 0.701],
            right_cube_position_m=[0.76, -0.15, 0.701],
            left_target_position_m=[0.76, 0.15, 0.68],
            right_target_position_m=[0.76, -0.15, 0.68],
        )
        self.assertEqual(response.tokens, {"left": "RELEASE", "right": "RELEASE"})
        self.assertEqual(response.payload["zero_shot"]["recovery"], {})
        self.assertEqual(
            response.payload["zero_shot"]["sensor_stage_advances"],
            ["left", "right"],
        )
        self.assertEqual(agent.indices, {"left": 1, "right": 1})

    def test_dual_coords_guard_corrects_reasoning_action_mismatch(self):
        plan = {
            side: [_stage(f"{side}_move", "MOVE")]
            for side in ("left", "right")
        }
        client = _FakeClient(
            plan,
            [{"left": "MV_LEFT", "right": "MV_RIGHT", "reasoning": "wrong final"}],
        )
        agent = OriginalZeroShotDualAgent(
            client=client,
            common_context="context",
            controller_prompt=(
                "{task} {stage_l} {target_l} {afford_l} {description_l} {completion_l} "
                "{gripper_l} {mem_l} {recovery_l} {proprio_l} {stage_r} {target_r} "
                "{afford_r} {description_r} {completion_r} {gripper_r} {mem_r} "
                "{recovery_r} {proprio_r} {mem_rules} {output_contract}"
            ),
            max_replans=0,
            include_goal_error=True,
            enforce_goal_error=True,
            goal_error_tolerance_m=0.01,
            positive_y_token="MV_LEFT",
        )
        response = agent.decide(
            task="move forward",
            recent_left="none",
            recent_right="none",
            agentview_image=self.image,
            wrist_left_image=self.image,
            wrist_right_image=self.image,
            left_gripper="closed",
            right_gripper="closed",
            left_tcp_position_m=[0.0, 0.0, 0.2],
            right_tcp_position_m=[0.0, 0.0, 0.2],
            left_target_position_m=[0.1, 0.0, 0.0],
            right_target_position_m=[0.1, 0.0, 0.0],
        )
        self.assertEqual(response.tokens, {"left": "MV_FWD", "right": "MV_FWD"})
        self.assertEqual(
            response.payload["zero_shot"]["goal_error_overrides"],
            {"left": "MV_FWD", "right": "MV_FWD"},
        )
        self.assertEqual(agent.movement_frame("left", response), "base")

    def test_dual_coords_guard_closes_when_grasp_xy_is_aligned(self):
        plan = {
            side: [_stage(f"{side}_grasp", "GRASP")]
            for side in ("left", "right")
        }
        client = _FakeClient(
            plan,
            [{"left": "MV_FWD", "right": "MV_BACK", "reasoning": "overshoot"}],
        )
        agent = OriginalZeroShotDualAgent(
            client=client,
            common_context="context",
            controller_prompt=(
                "{task} {stage_l} {target_l} {afford_l} {description_l} {completion_l} "
                "{gripper_l} {mem_l} {recovery_l} {proprio_l} {stage_r} {target_r} "
                "{afford_r} {description_r} {completion_r} {gripper_r} {mem_r} "
                "{recovery_r} {proprio_r} {mem_rules} {output_contract}"
            ),
            max_replans=0,
            include_goal_error=True,
            enforce_goal_error=True,
            goal_error_tolerance_m=0.015,
            positive_y_token="MV_LEFT",
        )
        response = agent.decide(
            task="grasp both",
            recent_left="none",
            recent_right="none",
            agentview_image=self.image,
            wrist_left_image=self.image,
            wrist_right_image=self.image,
            left_gripper="open",
            right_gripper="open",
            left_tcp_position_m=[0.60, 0.15, 0.70],
            right_tcp_position_m=[0.60, -0.15, 0.70],
            left_cube_position_m=[0.61, 0.15, 0.68],
            right_cube_position_m=[0.61, -0.15, 0.68],
        )
        self.assertEqual(response.tokens, {"left": "GRASP", "right": "GRASP"})


if __name__ == "__main__":
    unittest.main()
