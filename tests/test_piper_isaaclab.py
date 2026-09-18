"""CPU-only checks for the Piper asset preflight and shared Isaac Lab rollout API."""
from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from core.sim.mvtoken_robolab_runner import MvTokenRobolabRunner
from core.sim.piper_isaaclab_task import piper_policy_phase, resolve_piper_usd
from core.v0_types import V0Config
from interpreters.robolab_atomic_controller import RobolabAtomicController


MOVE_VECTORS = {
    "MV_FWD": (1, 0, 0),
    "MV_BACK": (-1, 0, 0),
    "MV_LEFT": (0, 1, 0),
    "MV_RIGHT": (0, -1, 0),
    "MV_UP": (0, 0, 1),
    "MV_DOWN": (0, 0, -1),
}


class FakeEnv:
    def __init__(self):
        self.pos = np.zeros(3)


class FakeBackend:
    hold_orientation = False
    @staticmethod
    def reset(env, hold_action, settle_steps):
        env.pos[:] = 0
        return {"front": np.zeros((32, 32, 3), np.uint8), "wrist": np.zeros((32, 32, 3), np.uint8)}, False, False

    @staticmethod
    def step(env, action):
        env.pos += action[:3]
        return FakeBackend.reset_observation(), FakeBackend.success(env), False, {}

    @staticmethod
    def reset_observation():
        return {"front": np.zeros((32, 32, 3), np.uint8), "wrist": np.zeros((32, 32, 3), np.uint8)}

    @staticmethod
    def rgb(obs, camera_name):
        return obs[camera_name]

    @staticmethod
    def tcp(env):
        return env.pos

    @staticmethod
    def ee_quat(env):
        return (1, 0, 0, 0)

    @staticmethod
    def gripper_width(env):
        return 0.1

    @staticmethod
    def success(env):
        return env.pos[0] >= 0.019


class FakeLogger:
    def __init__(self, run_dir):
        self.run_dir = run_dir
        self.records = []
        self.summary = None

    def log_step(self, **kwargs):
        self.records.append(kwargs["record"])

    def close(self, **kwargs):
        return self.run_dir / "video.mp4"

    def write_summary(self, data):
        self.summary = data


class PiperIsaacLabTests(unittest.TestCase):
    def test_atomic_controller_accepts_per_step_wrist_frame_override(self):
        controller = RobolabAtomicController(
            move_vectors=MOVE_VECTORS,
            step_m=0.02,
            ik_scale=1.0,
            sim_steps_per_decision=2,
            motion_frame="base",
            tool_axis=[1.0, 0.0, 0.0],
        )
        quarter_turn_z = (2**-0.5, 0.0, 0.0, 2**-0.5)
        base = controller.action_for_atomic(
            "MV_FWD", ee_quat=quarter_turn_z, motion_frame="base"
        )
        wrist = controller.action_for_atomic(
            "MV_FWD", ee_quat=quarter_turn_z, motion_frame="wrist"
        )
        self.assertGreater(base[0], 0.0)
        self.assertAlmostEqual(float(base[1]), 0.0, places=6)
        self.assertAlmostEqual(float(wrist[0]), 0.0, places=6)
        self.assertGreater(wrist[1], 0.0)

    def test_pick_place_phase_sequence_from_measured_state(self):
        common = dict(
            task_name="pick_place_left",
            target_position_m=(0.30, 0.18, 0.0),
            home_position_m=(0.292, 0.001, 0.144),
            min_tcp_z_m=0.025,
            transport_tcp_z_m=0.12,
            align_tolerance_m=0.018,
            target_success_tolerance_m=0.04,
            cube_rest_z_m=0.021,
        )
        cases = [
            (False, 0.10, (0.30, 0.00, 0.14), (0.30, 0.00, 0.02), "DESCEND_TO_CUBE"),
            (False, 0.10, (0.30, 0.00, 0.025), (0.30, 0.00, 0.02), "AT_GRASP_HEIGHT"),
            (True, 0.04, (0.30, 0.00, 0.06), (0.30, 0.00, 0.05), "LIFT_WITH_CUBE"),
            (True, 0.04, (0.30, 0.04, 0.12), (0.30, 0.04, 0.11), "CARRY_TO_TARGET"),
            (True, 0.04, (0.30, 0.18, 0.12), (0.30, 0.18, 0.11), "LOWER_AT_TARGET"),
            (True, 0.04, (0.30, 0.18, 0.025), (0.30, 0.18, 0.02), "RELEASE_AT_TARGET"),
            (True, 0.00, (0.30, 0.00, 0.025), (0.30, 0.00, 0.02), "EMPTY_GRASP"),
            (False, 0.10, (0.30, 0.18, 0.025), (0.30, 0.18, 0.021), "RETREAT_UP"),
            (False, 0.10, (0.30, 0.18, 0.120), (0.30, 0.18, 0.021), "RETURN_HOME_RIGHT"),
            (False, 0.10, (0.292, 0.001, 0.120), (0.30, 0.18, 0.021), "RETURN_HOME_UP"),
            (False, 0.10, (0.292, 0.001, 0.144), (0.30, 0.18, 0.021), "COMPLETE"),
        ]
        for closed, width, tcp, cube, expected in cases:
            with self.subTest(expected=expected):
                self.assertEqual(
                    piper_policy_phase(
                        gripper_closed=closed,
                        finger_width_m=width,
                        tcp_position_m=tcp,
                        cube_position_m=cube,
                        **common,
                    ),
                    expected,
                )

    def test_piper_asset_preflight_requires_companion_layers(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "piper_description_v100_realsense_camera_v2.usd"
            path.touch()
            with self.assertRaisesRegex(FileNotFoundError, "Incomplete Piper USD package"):
                resolve_piper_usd(str(path))
            (path.parent / "configuration").mkdir()
            for part in ("base", "physics", "robot", "sensor"):
                (path.parent / "configuration" / f"{path.stem}_{part}.usd").touch()
            self.assertEqual(resolve_piper_usd(str(path)), path)

    def test_shared_runner_uses_piper_backend(self):
        with tempfile.TemporaryDirectory() as tmp:
            env = FakeEnv()
            logger = FakeLogger(Path(tmp))
            decisions = []
            def decide(**kwargs):
                decisions.append(kwargs)
                return SimpleNamespace(token="MV_FWD", payload={})
            agent = SimpleNamespace(decide=decide)
            controller = RobolabAtomicController(
                move_vectors=MOVE_VECTORS, step_m=0.02, ik_scale=1.0, sim_steps_per_decision=2
            )
            runner = MvTokenRobolabRunner(
                env=env,
                backend=FakeBackend(),
                control_mode="piper_isaaclab_mvtoken",
                task_description="Move forward",
                controller=controller,
                agent=agent,
                logger=logger,
                config=V0Config.from_dict({}),
                max_steps=2,
                num_steps_wait=0,
                loop_period_s=0,
                sim_steps_per_decision=2,
                settle_steps_per_decision=0,
                agentview_camera="front",
                wrist_camera="wrist",
                agentview_rotation_degrees=0,
                wrist_rotation_degrees=0,
                agentview_flip="none",
                wrist_flip="none",
                use_wrist_image=True,
                debug=False,
            )
            result = runner.run()
            self.assertTrue(result.success)
            self.assertAlmostEqual(env.pos[0], 0.02)
            self.assertEqual(logger.records[0]["act"], "MV_FWD")
            self.assertEqual(logger.summary["control_mode"], "piper_isaaclab_mvtoken")
            self.assertEqual(decisions[0]["piper_phase"], "AT_GRASP_HEIGHT")
            self.assertEqual(decisions[0]["finger_width_m"], 0.1)


if __name__ == "__main__":
    unittest.main()
