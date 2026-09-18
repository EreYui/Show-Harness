"""Official RMC-AIDA-L dual RM65-B-V model for Isaac Lab 2.2 / Isaac Sim 5.

Isaac Lab imports deliberately stay inside :func:`launch_isaac` and the environment
constructor.  The module can therefore be imported by CPU-only tests and setup checks.

The complete official URDF supplies the chassis, lift, arm mounts, both arms, cameras
and grippers.  The chassis is fixed and its wheels are locked for manipulation-only
experiments; no navigation command is exposed by this environment.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping

import numpy as np

SIDES = ("left", "right")


def realman_policy_phase(
    *,
    gripper_closed: bool,
    finger_width_m: float,
    tcp_position_m,
    cube_position_m,
    target_position_m,
    home_position_m,
    min_tcp_z_m: float,
    transport_tcp_z_m: float,
    align_tolerance_m: float,
    target_success_tolerance_m: float,
    cube_rest_z_m: float,
) -> str:
    """Classify one arm's measured state for the dual policy and scripted probe."""
    tcp = np.asarray(tcp_position_m, dtype=float)
    cube = np.asarray(cube_position_m, dtype=float)
    target = np.asarray(target_position_m, dtype=float)
    home = np.asarray(home_position_m, dtype=float)
    holding = (
        gripper_closed
        and float(finger_width_m) >= 0.020
        and float(np.linalg.norm(cube - tcp)) < 0.14
    )
    if gripper_closed and not holding:
        return "EMPTY_GRASP"

    placed = (
        float(np.linalg.norm(cube[:2] - target[:2])) <= target_success_tolerance_m
        and float(cube[2]) <= cube_rest_z_m + 0.02
    )
    if not gripper_closed and placed:
        if float(tcp[2]) < transport_tcp_z_m - 0.008:
            return "RETREAT_UP"
        delta = home - tcp
        if abs(float(delta[0])) > align_tolerance_m:
            return "RETURN_HOME_FWD" if delta[0] > 0.0 else "RETURN_HOME_BACK"
        if abs(float(delta[1])) > align_tolerance_m:
            return "RETURN_HOME_LEFT" if delta[1] > 0.0 else "RETURN_HOME_RIGHT"
        if abs(float(delta[2])) > 0.010:
            return "RETURN_HOME_UP" if delta[2] > 0.0 else "RETURN_HOME_DOWN"
        return "COMPLETE"

    if not gripper_closed:
        delta_xy = cube[:2] - tcp[:2]
        if abs(float(delta_xy[0])) > align_tolerance_m:
            return "APPROACH_CUBE_FWD" if delta_xy[0] > 0.0 else "APPROACH_CUBE_BACK"
        if abs(float(delta_xy[1])) > align_tolerance_m:
            return "APPROACH_CUBE_LEFT" if delta_xy[1] > 0.0 else "APPROACH_CUBE_RIGHT"
        if float(tcp[2]) <= min_tcp_z_m + 0.005:
            return "AT_GRASP_HEIGHT"
        return "DESCEND_TO_CUBE"

    delta_xy = target[:2] - tcp[:2]
    target_aligned = float(np.linalg.norm(delta_xy)) <= align_tolerance_m
    if not target_aligned and float(tcp[2]) < transport_tcp_z_m - 0.008:
        return "LIFT_WITH_CUBE"
    if abs(float(delta_xy[0])) > align_tolerance_m:
        return "CARRY_FWD" if delta_xy[0] > 0.0 else "CARRY_BACK"
    if abs(float(delta_xy[1])) > align_tolerance_m:
        return "CARRY_LEFT" if delta_xy[1] > 0.0 else "CARRY_RIGHT"
    if float(tcp[2]) > min_tcp_z_m + 0.005:
        return "LOWER_AT_TARGET"
    return "RELEASE_AT_TARGET"


def scripted_token_for_phase(phase: str) -> str:
    """Deterministic action used by the no-API end-to-end acceptance run."""
    table = {
        "APPROACH_CUBE_FWD": "MV_FWD",
        "APPROACH_CUBE_BACK": "MV_BACK",
        "APPROACH_CUBE_LEFT": "MV_LEFT",
        "APPROACH_CUBE_RIGHT": "MV_RIGHT",
        "DESCEND_TO_CUBE": "MV_DOWN",
        "AT_GRASP_HEIGHT": "GRASP",
        "EMPTY_GRASP": "RELEASE",
        "LIFT_WITH_CUBE": "MV_UP",
        "CARRY_FWD": "MV_FWD",
        "CARRY_BACK": "MV_BACK",
        "CARRY_LEFT": "MV_LEFT",
        "CARRY_RIGHT": "MV_RIGHT",
        "LOWER_AT_TARGET": "MV_DOWN",
        "RELEASE_AT_TARGET": "RELEASE",
        "RETREAT_UP": "MV_UP",
        "RETURN_HOME_FWD": "MV_FWD",
        "RETURN_HOME_BACK": "MV_BACK",
        "RETURN_HOME_LEFT": "MV_LEFT",
        "RETURN_HOME_RIGHT": "MV_RIGHT",
        "RETURN_HOME_UP": "MV_UP",
        "RETURN_HOME_DOWN": "MV_DOWN",
        "COMPLETE": "DONE",
    }
    return table.get(str(phase), "STILL")


def launch_isaac(*, headless: bool, device: str) -> Any:
    """Launch Kit with RTX cameras before importing any Isaac Lab scene classes."""
    import argparse

    from isaaclab.app import AppLauncher

    parser = argparse.ArgumentParser(add_help=False)
    AppLauncher.add_app_launcher_args(parser)
    app_args = parser.parse_args([])
    app_args.headless = bool(headless)
    app_args.enable_cameras = True
    app_args.device = str(device)
    return AppLauncher(app_args).app


class RealmanDualIsaacLabEnv:
    """One fixed-chassis RMC-AIDA-L articulation and four rendered RGB cameras.

    Each side accepts the same seven-element relative-IK action used by RoboLab:
    ``[dx, dy, dz, drx, dry, drz, gripper]``.  Translation is expressed in the
    RMC-AIDA-L root frame. The base is intentionally stationary in this rig.
    The gripper channel is 0=open, 1=close.
    """

    def __init__(self, cfg: dict[str, Any], robot_urdf: Path) -> None:
        import torch
        import isaaclab.sim as sim_utils
        from isaaclab.actuators import ImplicitActuatorCfg
        from isaaclab.assets import ArticulationCfg, AssetBaseCfg, RigidObjectCfg
        from isaaclab.controllers import DifferentialIKController, DifferentialIKControllerCfg
        from isaaclab.managers import SceneEntityCfg
        from isaaclab.scene import InteractiveScene, InteractiveSceneCfg
        from isaaclab.sensors import CameraCfg
        from isaaclab.utils import configclass
        from isaaclab.utils.math import quat_apply, subtract_frame_transforms

        self.torch = torch
        self.quat_apply = quat_apply
        self.subtract_frame_transforms = subtract_frame_transforms
        self.cfg = cfg
        self.ik_scale = 1.0
        self.physics_steps = 0
        self.max_physics_steps = int(cfg.get("max_physics_steps", 8000))
        self.min_tcp_z_m = float(cfg.get("min_tcp_z_m", 0.023))
        self.transport_tcp_z_m = float(cfg.get("transport_tcp_z_m", 0.16))
        self.align_tolerance_m = float(cfg.get("align_tolerance_m", 0.015))
        self.target_success_tolerance_m = float(
            cfg.get("target_success_tolerance_m", 0.035)
        )
        self.home_tolerance_m = float(cfg.get("home_tolerance_m", 0.018))
        self.gripper_open_rad = float(cfg.get("gripper_open_rad", -0.01))
        self.gripper_closed_rad = float(cfg.get("gripper_closed_rad", -0.90))
        self.gripper_max_width_m = float(cfg.get("gripper_max_width_m", 0.050))
        self.gripper_closed = {side: False for side in SIDES}
        self.tcp_offset_link6_m = torch.tensor(
            cfg.get("tcp_offset_link6_m", [0.0, 0.0, 0.13]),
            dtype=torch.float32,
            device=str(cfg["device"]),
        ).reshape(1, 3)

        homes = cfg.get("home_joints", {})
        for side in SIDES:
            if len(homes.get(side, [])) != 6:
                raise ValueError(f"home_joints.{side} must contain six RM65 angles")
        platform_pos = cfg.get("platform_position", [0.0, 0.0, 0.2412])
        if len(platform_pos) != 3:
            raise ValueError("platform_position must contain XYZ")
        self.platform_position = tuple(float(x) for x in platform_pos)
        self.lift_joint_position = float(cfg.get("lift_joint_position", -0.30))

        joint_drive = sim_utils.UrdfFileCfg.JointDriveCfg(
            drive_type="force",
            target_type="position",
            gains=sim_utils.UrdfFileCfg.JointDriveCfg.PDGainsCfg(
                stiffness={
                    "joint_(left|right)_[1-6]": 0.0,
                    "Left_1_Joint2?": 0.0,
                    "joint_5": 0.0,
                    "joint_mid_[12]": 0.0,
                },
                damping={
                    "joint_(left|right)_[1-6]": 0.0,
                    "Left_1_Joint2?": 0.0,
                    "joint_5": 0.0,
                    "joint_mid_[12]": 0.0,
                },
            ),
        )

        home_joint_pos = {
            **{
                f"joint_{side}_{index + 1}": float(angle)
                for side in SIDES
                for index, angle in enumerate(homes[side])
            },
            "joint_5": self.lift_joint_position,
            "Left_1_Joint": self.gripper_open_rad,
            "Left_1_Joint2": self.gripper_open_rad,
            "joint_mid_1": 0.0,
            "joint_mid_2": 0.0,
        }
        robot_cfg = ArticulationCfg(
            prim_path="{ENV_REGEX_NS}/Robot",
            spawn=sim_utils.UrdfFileCfg(
                asset_path=str(robot_urdf),
                fix_base=True,
                merge_fixed_joints=False,
                self_collision=False,
                joint_drive=joint_drive,
                rigid_props=sim_utils.RigidBodyPropertiesCfg(
                    disable_gravity=False, max_depenetration_velocity=1.0
                ),
                articulation_props=sim_utils.ArticulationRootPropertiesCfg(
                    enabled_self_collisions=False,
                    solver_position_iteration_count=12,
                    solver_velocity_iteration_count=2,
                ),
            ),
            init_state=ArticulationCfg.InitialStateCfg(
                pos=self.platform_position,
                joint_pos=home_joint_pos,
            ),
            actuators={
                "arms": ImplicitActuatorCfg(
                    joint_names_expr=["joint_(left|right)_[1-6]"],
                    effort_limit_sim=80.0,
                    stiffness=1800.0,
                    damping=75.0,
                ),
                "grippers": ImplicitActuatorCfg(
                    joint_names_expr=["Left_1_Joint2?"],
                    effort_limit_sim=40.0,
                    stiffness=900.0,
                    damping=35.0,
                ),
                "lift": ImplicitActuatorCfg(
                    joint_names_expr=["joint_5"],
                    effort_limit_sim=400.0,
                    stiffness=6000.0,
                    damping=300.0,
                ),
                "head": ImplicitActuatorCfg(
                    joint_names_expr=["joint_mid_[12]"],
                    effort_limit_sim=20.0,
                    stiffness=400.0,
                    damping=30.0,
                ),
            },
        )

        cube_xy = cfg.get("cube_xy", {})
        target_xy = cfg.get("target_xy", {})
        for side in SIDES:
            if len(cube_xy.get(side, [])) != 2 or len(target_xy.get(side, [])) != 2:
                raise ValueError(f"cube_xy/target_xy for {side} must contain XY")
        cube_size = float(cfg.get("cube_size_m", 0.04))
        self.table_top_z = float(cfg.get("table_top_z", 0.68))
        self.cube_rest_z = self.table_top_z + cube_size / 2.0 + 0.001
        self.target_positions = {
            side: np.asarray(
                [float(target_xy[side][0]), float(target_xy[side][1]), self.table_top_z],
                dtype=float,
            )
            for side in SIDES
        }
        target_size = float(cfg.get("target_size_m", 0.085))
        agentview = cfg["agentview_pose"]
        overview = cfg["overview_pose"]
        wrist = cfg["wrist_pose"]

        def cube_cfg(side: str, color: tuple[float, float, float]) -> RigidObjectCfg:
            xy = cube_xy[side]
            return RigidObjectCfg(
                prim_path=f"{{ENV_REGEX_NS}}/{side.title()}Cube",
                spawn=sim_utils.CuboidCfg(
                    size=(cube_size, cube_size, cube_size),
                    rigid_props=sim_utils.RigidBodyPropertiesCfg(
                        max_depenetration_velocity=1.0
                    ),
                    mass_props=sim_utils.MassPropertiesCfg(mass=0.06),
                    collision_props=sim_utils.CollisionPropertiesCfg(),
                    visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=color),
                ),
                init_state=RigidObjectCfg.InitialStateCfg(
                    pos=(float(xy[0]), float(xy[1]), self.cube_rest_z)
                ),
            )

        def target_cfg(side: str, color: tuple[float, float, float]) -> AssetBaseCfg:
            xy = target_xy[side]
            return AssetBaseCfg(
                prim_path=f"{{ENV_REGEX_NS}}/{side.title()}Target",
                spawn=sim_utils.CuboidCfg(
                    size=(target_size, target_size, 0.003),
                    visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=color),
                ),
                init_state=AssetBaseCfg.InitialStateCfg(
                    pos=(float(xy[0]), float(xy[1]), self.table_top_z + 0.0015)
                ),
            )

        @configclass
        class RealmanDualSceneCfg(InteractiveSceneCfg):
            ground = AssetBaseCfg(
                prim_path="/World/Ground",
                spawn=sim_utils.GroundPlaneCfg(),
                init_state=AssetBaseCfg.InitialStateCfg(pos=(0.0, 0.0, 0.0)),
            )
            light = AssetBaseCfg(
                prim_path="/World/Light",
                spawn=sim_utils.DomeLightCfg(
                    intensity=2800.0, color=(0.82, 0.82, 0.82)
                ),
            )
            table = AssetBaseCfg(
                prim_path="{ENV_REGEX_NS}/Table",
                spawn=sim_utils.CuboidCfg(
                    size=tuple(float(x) for x in cfg.get("table_size", [0.58, 0.82, 0.05])),
                    collision_props=sim_utils.CollisionPropertiesCfg(),
                    visual_material=sim_utils.PreviewSurfaceCfg(
                        diffuse_color=(0.52, 0.39, 0.27)
                    ),
                ),
                init_state=AssetBaseCfg.InitialStateCfg(
                    pos=(
                        float(cfg.get("table_center_x", 0.73)),
                        0.0,
                        self.table_top_z - 0.025,
                    )
                ),
            )
            robot: ArticulationCfg = robot_cfg
            cube_left = cube_cfg("left", (0.90, 0.10, 0.06))
            cube_right = cube_cfg("right", (0.05, 0.28, 0.92))
            target_left = target_cfg("left", (0.10, 0.78, 0.22))
            target_right = target_cfg("right", (0.92, 0.72, 0.08))
            agentview_camera: CameraCfg = CameraCfg(
                prim_path="{ENV_REGEX_NS}/AgentviewCamera",
                update_period=0.0,
                width=int(cfg.get("camera_width", 640)),
                height=int(cfg.get("camera_height", 480)),
                data_types=["rgb"],
                spawn=sim_utils.PinholeCameraCfg(
                    focal_length=23.0, clipping_range=(0.05, 3.0)
                ),
                offset=CameraCfg.OffsetCfg(
                    pos=tuple(float(x) for x in agentview["pos"]),
                    rot=tuple(float(x) for x in agentview["rot"]),
                    convention="world",
                ),
            )
            overview_camera: CameraCfg = CameraCfg(
                prim_path="{ENV_REGEX_NS}/OverviewCamera",
                update_period=0.0,
                width=int(cfg.get("camera_width", 640)),
                height=int(cfg.get("camera_height", 480)),
                data_types=["rgb"],
                spawn=sim_utils.PinholeCameraCfg(
                    focal_length=float(cfg.get("overview_focal_length", 18.0)),
                    clipping_range=(0.05, 4.0),
                ),
                offset=CameraCfg.OffsetCfg(
                    pos=tuple(float(x) for x in overview["pos"]),
                    rot=tuple(float(x) for x in overview["rot"]),
                    convention="world",
                ),
            )
            wrist_left_camera: CameraCfg = CameraCfg(
                prim_path="{ENV_REGEX_NS}/Robot/left_camera/WristCameraLeft",
                update_period=0.0,
                width=int(cfg.get("camera_width", 640)),
                height=int(cfg.get("camera_height", 480)),
                data_types=["rgb"],
                spawn=sim_utils.PinholeCameraCfg(
                    focal_length=18.0, clipping_range=(0.025, 2.0)
                ),
                offset=CameraCfg.OffsetCfg(
                    pos=tuple(float(x) for x in wrist["pos"]),
                    rot=tuple(float(x) for x in wrist["rot"]),
                    convention="ros",
                ),
            )
            wrist_right_camera: CameraCfg = CameraCfg(
                prim_path="{ENV_REGEX_NS}/Robot/link_right_camera/WristCameraRight",
                update_period=0.0,
                width=int(cfg.get("camera_width", 640)),
                height=int(cfg.get("camera_height", 480)),
                data_types=["rgb"],
                spawn=sim_utils.PinholeCameraCfg(
                    focal_length=18.0, clipping_range=(0.025, 2.0)
                ),
                offset=CameraCfg.OffsetCfg(
                    pos=tuple(float(x) for x in wrist["pos"]),
                    rot=tuple(float(x) for x in wrist["rot"]),
                    convention="ros",
                ),
            )

        self.sim = sim_utils.SimulationContext(
            sim_utils.SimulationCfg(
                dt=float(cfg.get("physics_dt", 1.0 / 120.0)),
                device=str(cfg["device"]),
            )
        )
        self.scene = InteractiveScene(RealmanDualSceneCfg(num_envs=1, env_spacing=2.0))
        self.sim.reset()
        self.robot = self.scene["robot"]
        self.robots = {side: self.robot for side in SIDES}
        self.cubes = {side: self.scene[f"cube_{side}"] for side in SIDES}
        self.arm_entities: dict[str, Any] = {}
        self.arm_joint_ids: dict[str, Any] = {}
        self.finger_joint_ids: dict[str, Any] = {}
        self.ee_body_ids: dict[str, int] = {}
        self.ee_jacobi_ids: dict[str, int] = {}
        self.iks: dict[str, Any] = {}
        self._target_pos_b: dict[str, Any] = {}
        self._target_quat_b: dict[str, Any] = {}
        for side in SIDES:
            arm = SceneEntityCfg(
                "robot",
                joint_names=[f"joint_{side}_[1-6]"],
                body_names=[f"link_{side}_6"],
            )
            arm.resolve(self.scene)
            fingers = SceneEntityCfg(
                "robot",
                joint_names=["Left_1_Joint" if side == "left" else "Left_1_Joint2"],
            )
            fingers.resolve(self.scene)
            if len(arm.joint_ids) != 6 or len(fingers.joint_ids) != 1:
                raise RuntimeError(
                    f"{side} RMC-AIDA-L arm must expose six joints and one gripper master"
                )
            self.arm_entities[side] = arm
            self.arm_joint_ids[side] = arm.joint_ids
            self.finger_joint_ids[side] = fingers.joint_ids
            self.ee_body_ids[side] = arm.body_ids[0]
            self.ee_jacobi_ids[side] = arm.body_ids[0] - 1
            self.iks[side] = DifferentialIKController(
                DifferentialIKControllerCfg(
                    command_type="position", use_relative_mode=False, ik_method="dls"
                ),
                num_envs=1,
                device=self.sim.device,
            )
        self.home_tcp_positions: dict[str, np.ndarray] = {}
        self.reset(settle_steps=int(cfg.get("num_steps_wait", 60)))

    def _link6_pose_b(self, side: str):
        robot = self.robots[side]
        body = self.ee_body_ids[side]
        ee_pose_w = robot.data.body_pose_w[:, body]
        root_pose_w = robot.data.root_pose_w
        return self.subtract_frame_transforms(
            root_pose_w[:, :3],
            root_pose_w[:, 3:7],
            ee_pose_w[:, :3],
            ee_pose_w[:, 3:7],
        )

    def _tcp_pos_w(self, side: str):
        robot = self.robots[side]
        body = self.ee_body_ids[side]
        quat = robot.data.body_quat_w[:, body]
        offset = self.tcp_offset_link6_m.expand(quat.shape[0], -1)
        return robot.data.body_pos_w[:, body] + self.quat_apply(quat, offset)

    def _tcp_pos_b(self, side: str):
        robot = self.robots[side]
        _, link_quat_b = self._link6_pose_b(side)
        tcp_w = self._tcp_pos_w(side)
        tcp_b, _ = self.subtract_frame_transforms(
            robot.data.root_pos_w,
            robot.data.root_quat_w,
            tcp_w,
            robot.data.body_quat_w[:, self.ee_body_ids[side]],
        )
        return tcp_b

    def _observe(self) -> dict[str, np.ndarray]:
        names = (
            "agentview_camera",
            "overview_camera",
            "wrist_left_camera",
            "wrist_right_camera",
        )
        out: dict[str, np.ndarray] = {}
        for name in names:
            rgb = self.scene[name].data.output["rgb"]
            if rgb.numel() == 0:
                raise RuntimeError(
                    f"{name} produced no RGB frame; check ENABLE_CAMERAS and RTX rendering"
                )
            out[name] = rgb[0, :, :, :3].cpu().numpy().copy()
        return out

    def reset(self, *, settle_steps: int = 60) -> tuple[dict[str, np.ndarray], bool, bool]:
        root = self.robot.data.default_root_state.clone()
        root[:, :3] += self.scene.env_origins
        self.robot.write_root_pose_to_sim(root[:, :7])
        self.robot.write_root_velocity_to_sim(root[:, 7:])
        self.robot.write_joint_state_to_sim(
            self.robot.data.default_joint_pos.clone(),
            self.robot.data.default_joint_vel.clone(),
        )
        for side in SIDES:
            cube = self.cubes[side]
            cube_state = cube.data.default_root_state.clone()
            cube_state[:, :3] += self.scene.env_origins
            cube.write_root_pose_to_sim(cube_state[:, :7])
            cube.write_root_velocity_to_sim(cube_state[:, 7:])
        self.scene.reset()
        for side in SIDES:
            self.iks[side].reset()
            self.gripper_closed[side] = False
        self.physics_steps = 0
        for _ in range(max(1, int(settle_steps))):
            self.robot.set_joint_position_target(self.robot.data.default_joint_pos.clone())
            self.scene.write_data_to_sim()
            self.sim.step()
            self.scene.update(self.sim.get_physics_dt())
        for side in SIDES:
            self._target_pos_b[side] = self._tcp_pos_b(side).clone()
            self._target_quat_b[side] = self._link6_pose_b(side)[1].clone()
            self.home_tcp_positions[side] = self.tcp(side)
        return self._observe(), False, False

    def step(
        self, actions: Mapping[str, np.ndarray]
    ) -> tuple[dict[str, np.ndarray], bool, bool, dict[str, Any]]:
        for side in SIDES:
            action = np.asarray(actions[side], dtype=np.float32)
            if action.shape != (7,) or not np.isfinite(action).all():
                raise ValueError(f"{side} RealMan action must be a finite 7-element vector")
            robot = self.robots[side]
            tcp_pos_b = self._tcp_pos_b(side)
            delta = self.torch.as_tensor(
                action[:3] * self.ik_scale, device=self.sim.device
            ).reshape(1, 3)
            self._target_pos_b[side] += delta
            min_z_b = self.min_tcp_z_m - self.platform_position[2]
            self._target_pos_b[side][0, 2] = max(
                float(self._target_pos_b[side][0, 2]), min_z_b
            )
            self._target_pos_b[side] = self.torch.maximum(
                self.torch.minimum(self._target_pos_b[side], tcp_pos_b + 0.045),
                tcp_pos_b - 0.045,
            )
            link_pos_b, link_quat_b = self._link6_pose_b(side)
            self.iks[side].set_command(
                self._target_pos_b[side], ee_quat=self._target_quat_b[side]
            )
            jacobian = robot.root_physx_view.get_jacobians()[
                :, self.ee_jacobi_ids[side], :, self.arm_joint_ids[side]
            ]
            offset_w = self._tcp_pos_w(side) - robot.data.body_pos_w[
                :, self.ee_body_ids[side]
            ]
            angular = jacobian[:, 3:6].transpose(1, 2)
            tool_linear = jacobian[:, :3] + self.torch.cross(
                angular, offset_w.unsqueeze(1).expand_as(angular), dim=-1
            ).transpose(1, 2)
            tool_jacobian = self.torch.cat((tool_linear, jacobian[:, 3:6]), dim=1)
            joint_pos = robot.data.joint_pos[:, self.arm_joint_ids[side]]
            joint_des = self.iks[side].compute(
                tcp_pos_b, link_quat_b, tool_jacobian, joint_pos
            )
            joint_des = self.torch.clamp(joint_des, joint_pos - 0.10, joint_pos + 0.10)
            robot.set_joint_position_target(
                joint_des, joint_ids=self.arm_joint_ids[side]
            )
            self.gripper_closed[side] = bool(action[6] > 0.5)
            finger = (
                self.gripper_closed_rad
                if self.gripper_closed[side]
                else self.gripper_open_rad
            )
            robot.set_joint_position_target(
                self.torch.tensor([[finger]], device=self.sim.device),
                joint_ids=self.finger_joint_ids[side],
            )
        self.scene.write_data_to_sim()
        self.sim.step()
        self.scene.update(self.sim.get_physics_dt())
        self.physics_steps += 1
        success = self.success()
        truncated = self.physics_steps >= self.max_physics_steps
        return self._observe(), success, truncated, {"success": success}

    def tcp(self, side: str) -> np.ndarray:
        return self._tcp_pos_w(side)[0].cpu().numpy().copy()

    def ee_quat(self, side: str) -> np.ndarray:
        return (
            self.robots[side]
            .data.body_quat_w[0, self.ee_body_ids[side]]
            .cpu()
            .numpy()
            .copy()
        )

    def gripper_width(self, side: str) -> float:
        joint = float(
            self.robots[side].data.joint_pos[0, self.finger_joint_ids[side][0]]
        )
        span = self.gripper_open_rad - self.gripper_closed_rad
        openness = max(
            0.0,
            min(1.0, (joint - self.gripper_closed_rad) / max(1.0e-6, span)),
        )
        return 0.004 + openness * (self.gripper_max_width_m - 0.004)

    def cube_position(self, side: str) -> np.ndarray:
        return self.cubes[side].data.root_pos_w[0].cpu().numpy().copy()

    def policy_state(self) -> dict[str, dict[str, Any]]:
        state: dict[str, dict[str, Any]] = {}
        for side in SIDES:
            tcp = self.tcp(side)
            cube = self.cube_position(side)
            phase = realman_policy_phase(
                gripper_closed=self.gripper_closed[side],
                finger_width_m=self.gripper_width(side),
                tcp_position_m=tcp,
                cube_position_m=cube,
                target_position_m=self.target_positions[side],
                home_position_m=self.home_tcp_positions[side],
                min_tcp_z_m=self.min_tcp_z_m,
                transport_tcp_z_m=self.transport_tcp_z_m,
                align_tolerance_m=self.align_tolerance_m,
                target_success_tolerance_m=self.target_success_tolerance_m,
                cube_rest_z_m=self.cube_rest_z,
            )
            state[side] = {
                "phase": phase,
                "tcp_position_m": tcp,
                "cube_position_m": cube,
                "target_position_m": self.target_positions[side].copy(),
                "home_position_m": self.home_tcp_positions[side].copy(),
                "finger_width_m": self.gripper_width(side),
                "gripper_closed": self.gripper_closed[side],
            }
        return state

    def success(self) -> bool:
        # One source of truth for closure: both sides must have placed their cube,
        # opened, retreated, and reached the prompt's explicit COMPLETE phase.
        return all(
            item["phase"] == "COMPLETE" for item in self.policy_state().values()
        )

    def close(self) -> None:
        if getattr(self, "_closed", False):
            return
        self._closed = True
        self.robots.clear()
        self.cubes.clear()
        del self.robot
        del self.scene
        self.sim.clear_all_callbacks()
        self.sim.clear_instance()


__all__ = [
    "RealmanDualIsaacLabEnv",
    "SIDES",
    "launch_isaac",
    "realman_policy_phase",
    "scripted_token_for_phase",
]
