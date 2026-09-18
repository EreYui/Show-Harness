"""CPU-only checks for the dual RM65 asset and measured-state policy contract."""
from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
import xml.etree.ElementTree as ET

import numpy as np

from core.config import deep_merge, load_yaml
from core.record.episode_logger import _make_analysis_frame
from core.sim.realman_assets import (
    RMC_AIDAL_PACKAGE,
    RMC_AIDAL_RELATIVE_PACKAGE,
    prepare_rm65_urdf,
    prepare_rmc_aidal_urdf,
    resolve_realman_root,
    resolve_rmc_aidal_root,
)
from core.sim.realman_dual_isaaclab_task import (
    realman_policy_phase,
    scripted_token_for_phase,
)
from core.sim.dual_mvtoken_isaaclab_runner import center_focus_view, translate_view
from core.vlm.dual_mvtoken_roles import DualMvTokenController


class RealmanDualIsaacLabTests(unittest.TestCase):
    def test_wrist_translation_recenters_without_wrapping_pixels(self):
        image = np.full((5, 6, 3), 10, dtype=np.uint8)
        image[1, 1] = [250, 20, 30]
        shifted = translate_view(image, dx_px=3, dy_px=2)
        np.testing.assert_array_equal(shifted[3, 4], [250, 20, 30])
        self.assertTrue(np.all(shifted[:, :3] == 10))
        self.assertFalse(np.any(shifted[0] == [250, 20, 30]))

    def test_wrist_focus_crop_excludes_distant_edge_and_keeps_center(self):
        image = np.full((12, 12, 3), 10, dtype=np.uint8)
        image[6, 6] = [250, 20, 30]
        image[0, 0] = [20, 250, 30]
        focused = center_focus_view(image, crop_size_px=8, output_size_px=8)
        self.assertEqual(focused.shape, (8, 8, 3))
        self.assertTrue(np.any(np.all(focused == [250, 20, 30], axis=-1)))
        self.assertFalse(np.any(np.all(focused == [20, 250, 30], axis=-1)))

    def test_deepseek_prompt_keeps_measured_phase_authoritative(self):
        prompt = (
            Path(__file__).resolve().parents[1]
            / "prompts"
            / "v3"
            / "realman_deepseek_dual_pick_place.txt"
        ).read_text(encoding="utf-8")
        required = {
            "LIFT_WITH_CUBE": "MV_UP",
            "CARRY_FWD": "MV_FWD",
            "CARRY_BACK": "MV_BACK",
            "CARRY_LEFT": "MV_LEFT",
            "CARRY_RIGHT": "MV_RIGHT",
            "RELEASE_AT_TARGET": "RELEASE",
            "RETREAT_UP": "MV_UP",
            "COMPLETE": "DONE",
        }
        for phase, token in required.items():
            self.assertIn(f"{phase} -> {token}", prompt)
        self.assertIn("This table is absolute", prompt)

    def test_four_panel_analysis_frame_keeps_policy_views_and_overview(self):
        image = np.zeros((48, 64, 3), dtype=np.uint8)
        record = {
            "i": 3,
            "left": {"stage": "CARRY_FWD", "dec": "MV_FWD", "grip": "CLOSED"},
            "right": {"stage": "CARRY_BACK", "dec": "MV_BACK", "grip": "CLOSED"},
        }
        frame = _make_analysis_frame(
            agentview=image,
            wrist=[image, image],
            overview=image,
            record=record,
        )
        self.assertEqual(frame.shape, (752, 2048, 3))

    def test_realman_task_presets_cover_forward_lateral_and_reverse_transport(self):
        cfg = load_yaml(
            Path(__file__).resolve().parents[1]
            / "configs"
            / "robot_realman_dual_isaaclab.yaml"
        )
        tasks = cfg["realman_tasks"]
        self.assertEqual(
            cfg["robot"], "realman_rmc_aida_l_rm65_b_v_static_base"
        )
        self.assertTrue(cfg["physical_platform"]["mobile_base"])
        self.assertEqual(cfg["physical_platform"]["model"], "RMC-AIDAL")
        self.assertEqual(cfg["physical_platform"]["lift_axis_travel_m"], 0.90)
        self.assertEqual(
            cfg["physical_platform"]["simulation_fidelity"],
            "official_full_body_static_base",
        )
        self.assertEqual(cfg["tcp_offset_link6_m"], [0.0, 0.0, 0.20])
        self.assertGreater(cfg["gripper_open_rad"], cfg["gripper_closed_rad"])
        self.assertLess(cfg["transport_tcp_z_m"], 0.82)
        self.assertEqual(
            set(tasks),
            {"parallel_forward", "diagonal_inward", "diagonal_outward", "split_depth"},
        )
        resolved = {name: deep_merge(cfg, preset) for name, preset in tasks.items()}
        self.assertGreater(
            resolved["parallel_forward"]["target_xy"]["right"][0],
            cfg["cube_xy"]["right"][0],
        )
        self.assertLess(
            abs(resolved["diagonal_inward"]["target_xy"]["left"][1]),
            abs(cfg["cube_xy"]["left"][1]),
        )
        self.assertGreater(
            abs(resolved["diagonal_outward"]["target_xy"]["right"][1]),
            abs(cfg["cube_xy"]["right"][1]),
        )
        self.assertLess(
            resolved["split_depth"]["target_xy"]["right"][0],
            cfg["cube_xy"]["right"][0],
        )

    def test_generated_urdf_resolves_meshes_and_adds_parallel_gripper(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "ros2_rm_robot"
            mesh = root / "rm_description" / "meshes" / "rm_65_arm" / "link6.STL"
            mesh.parent.mkdir(parents=True)
            mesh.write_bytes(b"solid link6\nendsolid link6\n")
            source = root / "rm_description" / "urdf" / "rm_65.urdf"
            source.parent.mkdir(parents=True)
            source.write_text(
                """<?xml version="1.0"?>
<robot name="rm65"><link name="Link6"><visual><geometry>
<mesh filename="package://rm_description/meshes/rm_65_arm/link6.STL"/>
</geometry></visual></link></robot>""",
                encoding="utf-8",
            )
            self.assertEqual(resolve_realman_root(root), root.resolve())
            output = prepare_rm65_urdf(root, Path(tmp) / "cache")
            robot = ET.parse(output).getroot()
            mesh_path = next(robot.iter("mesh")).get("filename")
            self.assertEqual(mesh_path, mesh.resolve().as_posix())
            joints = {item.get("name") for item in robot.findall("joint")}
            self.assertIn("finger_joint_left", joints)
            self.assertIn("finger_joint_right", joints)
            self.assertIn("gripper_mount", joints)

    def test_full_body_urdf_uses_official_meshes_and_locks_only_wheels(self):
        with tempfile.TemporaryDirectory() as tmp:
            checkout = Path(tmp) / "Ecosystem_Cases_RMC_AIDA_L"
            package = checkout / RMC_AIDAL_RELATIVE_PACKAGE
            meshes = package / "meshes"
            meshes.mkdir(parents=True)
            for name in (
                "base_link.STL",
                "link_5.STL",
                "link_left_6.STL",
                "link_right_6.STL",
            ):
                (meshes / name).write_bytes(b"solid model\nendsolid model\n")
            source = package / "urdf" / f"{RMC_AIDAL_PACKAGE}.urdf"
            source.parent.mkdir(parents=True)
            source.write_text(
                f'''<?xml version="1.0"?>
<robot name="{RMC_AIDAL_PACKAGE}">
  <link name="base_link"><visual><geometry><mesh filename="package://x/meshes/base_link.STL"/></geometry></visual></link>
  <link name="wheel_left"/><link name="wheel_right"/><link name="lift"/>
  <joint name="joint_1" type="continuous"><parent link="base_link"/><child link="wheel_left"/><axis xyz="0 1 0"/><limit effort="7" velocity="5"/></joint>
  <joint name="joint_2" type="continuous"><parent link="base_link"/><child link="wheel_right"/><axis xyz="0 1 0"/><limit effort="7" velocity="5"/></joint>
  <joint name="joint_5" type="prismatic"><parent link="base_link"/><child link="lift"/><axis xyz="0 0 -1"/><limit lower="-1" upper="0" effort="250" velocity="0.2"/></joint>
</robot>''',
                encoding="utf-8",
            )
            self.assertEqual(resolve_rmc_aidal_root(checkout), package.resolve())
            output = prepare_rmc_aidal_urdf(checkout, Path(tmp) / "cache")
            robot = ET.parse(output).getroot()
            joints = {joint.get("name"): joint for joint in robot.findall("joint")}
            self.assertEqual(joints["joint_1"].get("type"), "fixed")
            self.assertEqual(joints["joint_2"].get("type"), "fixed")
            self.assertEqual(joints["joint_5"].get("type"), "prismatic")
            self.assertEqual(
                next(robot.iter("mesh")).get("filename"),
                (meshes / "base_link.STL").resolve().as_posix(),
            )

    def test_full_pick_place_phase_contract(self):
        common = dict(
            target_position_m=(0.50, 0.14, 0.0),
            home_position_m=(0.36, 0.14, 0.19),
            min_tcp_z_m=0.023,
            transport_tcp_z_m=0.16,
            align_tolerance_m=0.015,
            target_success_tolerance_m=0.035,
            cube_rest_z_m=0.021,
        )
        cases = [
            (False, 0.074, (0.36, 0.14, 0.19), (0.36, 0.14, 0.021), "DESCEND_TO_CUBE", "MV_DOWN"),
            (False, 0.074, (0.36, 0.14, 0.023), (0.36, 0.14, 0.021), "AT_GRASP_HEIGHT", "GRASP"),
            (True, 0.040, (0.36, 0.14, 0.08), (0.36, 0.14, 0.04), "LIFT_WITH_CUBE", "MV_UP"),
            (True, 0.040, (0.38, 0.14, 0.16), (0.38, 0.14, 0.12), "CARRY_FWD", "MV_FWD"),
            (True, 0.040, (0.50, 0.14, 0.16), (0.50, 0.14, 0.12), "LOWER_AT_TARGET", "MV_DOWN"),
            (True, 0.040, (0.50, 0.14, 0.023), (0.50, 0.14, 0.021), "RELEASE_AT_TARGET", "RELEASE"),
            (False, 0.074, (0.50, 0.14, 0.023), (0.50, 0.14, 0.021), "RETREAT_UP", "MV_UP"),
            (False, 0.074, (0.50, 0.14, 0.16), (0.50, 0.14, 0.021), "RETURN_HOME_BACK", "MV_BACK"),
            (False, 0.074, (0.36, 0.14, 0.19), (0.50, 0.14, 0.021), "COMPLETE", "DONE"),
        ]
        for closed, width, tcp, cube, phase, token in cases:
            with self.subTest(phase=phase):
                actual = realman_policy_phase(
                    gripper_closed=closed,
                    finger_width_m=width,
                    tcp_position_m=tcp,
                    cube_position_m=cube,
                    **common,
                )
                self.assertEqual(actual, phase)
                self.assertEqual(scripted_token_for_phase(actual), token)

    def test_dual_prompt_accepts_runtime_robot_state(self):
        controller = DualMvTokenController(
            client=object(),
            prompt_template="{task}|{left_phase}|{right_phase}|{left_tcp}|{right_tcp}",
            scheme="once",
        )
        text = controller._render(
            "pick",
            "none",
            "none",
            left_phase="DESCEND_TO_CUBE",
            right_phase="CARRY_FWD",
            left_tcp="[0.1, 0.2, 0.3]",
            right_tcp="[0.4, 0.5, 0.6]",
        )
        self.assertEqual(
            text,
            "pick|DESCEND_TO_CUBE|CARRY_FWD|[0.1, 0.2, 0.3]|[0.4, 0.5, 0.6]",
        )


if __name__ == "__main__":
    unittest.main()
