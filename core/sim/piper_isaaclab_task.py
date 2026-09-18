"""Single-arm Piper pick-and-lift scene for Isaac Lab 2.2 / Isaac Sim 5.

The Isaac Lab imports live inside ``launch_isaac`` and ``PiperIsaacLabEnv.__init__``:
Isaac Sim must be launched before the scene, assets, sensors or controllers are imported.
The robot USD is supplied separately by AgileX; it is not redistributed here.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

DEFAULT_PIPER_USD = (
    Path.home()
    / "piper_isaac_sim/piper_description/urdf/piper_description_v100_realsense_camera_v2"
    / "piper_description_v100_realsense_camera_v2.usd"
)


def piper_policy_phase(
    *,
    task_name: str,
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
    """Classify measured robot/object state for the DeepSeek Piper prompt."""
    tcp = np.asarray(tcp_position_m, dtype=float)
    cube = np.asarray(cube_position_m, dtype=float)
    target = np.asarray(target_position_m, dtype=float)
    home = np.asarray(home_position_m, dtype=float)
    holding = (
        gripper_closed
        and float(finger_width_m) >= 0.020
        and float(np.linalg.norm(cube - tcp)) < 0.12
    )
    if gripper_closed and not holding:
        return "EMPTY_GRASP"

    if not task_name.startswith("pick_place"):
        if holding:
            return "CUBE_HELD"
        xy_aligned = float(np.linalg.norm(tcp[:2] - cube[:2])) <= align_tolerance_m
        if xy_aligned and tcp[2] <= min_tcp_z_m + 0.005:
            return "AT_GRASP_HEIGHT"
        return "APPROACH"

    placed = (
        float(np.linalg.norm(cube[:2] - target[:2])) <= target_success_tolerance_m
        and cube[2] <= cube_rest_z_m + 0.02
    )
    if not gripper_closed and placed:
        if tcp[2] < transport_tcp_z_m - 0.008:
            return "RETREAT_UP"
        x_error = float(home[0] - tcp[0])
        y_error = float(home[1] - tcp[1])
        z_error = float(home[2] - tcp[2])
        if abs(x_error) > align_tolerance_m:
            return "RETURN_HOME_FWD" if x_error > 0.0 else "RETURN_HOME_BACK"
        if abs(y_error) > align_tolerance_m:
            return "RETURN_HOME_LEFT" if y_error > 0.0 else "RETURN_HOME_RIGHT"
        if abs(z_error) > 0.010:
            return "RETURN_HOME_UP" if z_error > 0.0 else "RETURN_HOME_DOWN"
        return "COMPLETE"

    if not gripper_closed:
        if float(np.linalg.norm(tcp[:2] - cube[:2])) > align_tolerance_m:
            return "APPROACH_CUBE"
        if tcp[2] <= min_tcp_z_m + 0.005:
            return "AT_GRASP_HEIGHT"
        return "DESCEND_TO_CUBE"
    target_aligned = float(np.linalg.norm(tcp[:2] - target[:2])) <= align_tolerance_m
    if not target_aligned and tcp[2] < transport_tcp_z_m - 0.008:
        return "LIFT_WITH_CUBE"
    if not target_aligned:
        return "CARRY_TO_TARGET"
    if tcp[2] > min_tcp_z_m + 0.005:
        return "LOWER_AT_TARGET"
    return "RELEASE_AT_TARGET"


def resolve_piper_usd(path: str | None) -> Path:
    asset = Path(path).expanduser() if path else DEFAULT_PIPER_USD
    asset = asset.resolve()
    if not asset.is_file():
        raise FileNotFoundError(
            f"Piper USD not found: {asset}. Download AgileX piper_isaac_sim and set "
            "--piper-usd to its piper_description_v100_realsense_camera_v2.usd file."
        )
    # The entry USD is only a variant layer; its four companion layers contain the
    # meshes, joints and physics. Catch an incomplete copy before Kit starts.
    stem = asset.stem
    required = [asset.parent / "configuration" / f"{stem}_{part}.usd" for part in ("base", "physics", "robot", "sensor")]
    missing = [str(p) for p in required if not p.is_file()]
    if missing:
        raise FileNotFoundError("Incomplete Piper USD package; missing: " + ", ".join(missing))
    return asset


def launch_isaac(*, headless: bool, device: str) -> Any:
    """Launch one Kit app, with RTX cameras enabled, before any Isaac Lab scene import."""
    import argparse

    from isaaclab.app import AppLauncher

    parser = argparse.ArgumentParser(add_help=False)
    AppLauncher.add_app_launcher_args(parser)
    app_args = parser.parse_args([])
    app_args.headless = bool(headless)
    app_args.enable_cameras = True
    app_args.device = str(device)
    return AppLauncher(app_args).app


class PiperIsaacLabEnv:
    """One Piper, one cube and two RGB cameras; actions are 7-D relative IK.

    Action layout matches ``RobolabAtomicController``: ``[dx, dy, dz, drx, dry,
    drz, gripper]``. Translation is divided by ``ik_scale`` in the controller,
    then multiplied by it here. The gripper is 0=open, 1=close. Position is held
    at a persistent target through absolute-position differential IK. Piper's
    home pose has a wrist singularity, so orientation is free and the three
    rotation slots are ignored.
    """

    def __init__(self, cfg: dict[str, Any], piper_usd: Path) -> None:
        import torch
        import isaaclab.sim as sim_utils
        from isaaclab.actuators import ImplicitActuatorCfg
        from isaaclab.assets import ArticulationCfg, AssetBaseCfg, RigidObjectCfg
        from isaaclab.controllers import DifferentialIKController, DifferentialIKControllerCfg
        from isaaclab.managers import SceneEntityCfg
        from isaaclab.scene import InteractiveScene, InteractiveSceneCfg
        from isaaclab.sensors import CameraCfg
        from isaaclab.utils import configclass
        from isaaclab.utils.math import subtract_frame_transforms

        self.torch = torch
        self.subtract_frame_transforms = subtract_frame_transforms
        self.cfg = cfg
        self.ik_scale = 1.0
        self.physics_steps = 0
        self.max_physics_steps = int(cfg.get("max_physics_steps", 2000))
        self.cube_lift_m = float(cfg.get("cube_lift_m", 0.08))
        self.min_tcp_z_m = float(cfg.get("min_tcp_z_m", 0.025))
        self.task_name = str(cfg.get("piper_task", "pick_lift"))
        self.transport_tcp_z_m = float(cfg.get("transport_tcp_z_m", 0.12))
        self.target_xy_tolerance_m = float(cfg.get("target_xy_tolerance_m", 0.018))
        self.target_success_tolerance_m = float(cfg.get("target_success_tolerance_m", 0.04))
        self.gripper_open_m = float(cfg.get("gripper_open_m", 0.05))
        self.gripper_closed = False
        home = [float(x) for x in cfg["home_joints"]]
        if len(home) != 6:
            raise ValueError("home_joints must contain six Piper joint angles in radians")

        robot_cfg = ArticulationCfg(
            prim_path="{ENV_REGEX_NS}/Robot",
            spawn=sim_utils.UsdFileCfg(
                usd_path=str(piper_usd),
                variants={"Physics": "PhysX", "Robot": "Robot", "Sensor": "None"},
                rigid_props=sim_utils.RigidBodyPropertiesCfg(disable_gravity=False),
            ),
            init_state=ArticulationCfg.InitialStateCfg(
                joint_pos={
                    **{f"joint{i + 1}": angle for i, angle in enumerate(home)},
                    "joint7": self.gripper_open_m,
                    "joint8": -self.gripper_open_m,
                }
            ),
            actuators={
                "arm": ImplicitActuatorCfg(
                    joint_names_expr=["joint[1-6]"],
                    effort_limit_sim=100.0,
                    stiffness=2000.0,
                    damping=80.0,
                ),
                "gripper": ImplicitActuatorCfg(
                    joint_names_expr=["joint[78]"],
                    effort_limit_sim=100.0,
                    stiffness=800.0,
                    damping=50.0,
                ),
            },
        )
        cube_xy = tuple(float(x) for x in cfg.get("cube_xy", [0.22, 0.0]))
        if len(cube_xy) != 2:
            raise ValueError("cube_xy must contain x and y coordinates")
        cube_size = float(cfg.get("cube_size_m", 0.04))
        target_xy = tuple(float(x) for x in cfg.get("target_xy", [0.30, 0.18]))
        if len(target_xy) != 2:
            raise ValueError("target_xy must contain x and y coordinates")
        self.target_position_m = np.asarray((target_xy[0], target_xy[1], 0.0), dtype=float)
        target_size = float(cfg.get("target_size_m", 0.09))
        self.cube_rest_z = cube_size / 2.0 + 0.001
        agentview = cfg["agentview_pose"]
        wrist = cfg["wrist_pose"]

        @configclass
        class PiperSceneCfg(InteractiveSceneCfg):
            ground = AssetBaseCfg(
                prim_path="/World/Ground",
                spawn=sim_utils.GroundPlaneCfg(),
                init_state=AssetBaseCfg.InitialStateCfg(pos=(0.0, 0.0, -0.1)),
            )
            light = AssetBaseCfg(
                prim_path="/World/Light",
                spawn=sim_utils.DomeLightCfg(intensity=2500.0, color=(0.8, 0.8, 0.8)),
            )
            table = AssetBaseCfg(
                prim_path="{ENV_REGEX_NS}/Table",
                spawn=sim_utils.CuboidCfg(
                    size=(0.5, 0.6, 0.05),
                    collision_props=sim_utils.CollisionPropertiesCfg(),
                    visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.55, 0.43, 0.31)),
                ),
                init_state=AssetBaseCfg.InitialStateCfg(pos=(0.25, 0.0, -0.025)),
            )
            robot: ArticulationCfg = robot_cfg
            cube = RigidObjectCfg(
                prim_path="{ENV_REGEX_NS}/Cube",
                spawn=sim_utils.CuboidCfg(
                    size=(cube_size, cube_size, cube_size),
                    rigid_props=sim_utils.RigidBodyPropertiesCfg(max_depenetration_velocity=1.0),
                    mass_props=sim_utils.MassPropertiesCfg(mass=0.05),
                    collision_props=sim_utils.CollisionPropertiesCfg(),
                    visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.9, 0.12, 0.08)),
                ),
                init_state=RigidObjectCfg.InitialStateCfg(pos=(cube_xy[0], cube_xy[1], cube_size / 2.0 + 0.001)),
            )
            if self.task_name.startswith("pick_place"):
                target = AssetBaseCfg(
                    prim_path="{ENV_REGEX_NS}/TargetPad",
                    spawn=sim_utils.CuboidCfg(
                        size=(target_size, target_size, 0.003),
                        visual_material=sim_utils.PreviewSurfaceCfg(
                            diffuse_color=(0.08, 0.75, 0.20)
                        ),
                    ),
                    init_state=AssetBaseCfg.InitialStateCfg(
                        pos=(target_xy[0], target_xy[1], 0.0015)
                    ),
                )
            agentview_camera: CameraCfg = CameraCfg(
                prim_path="{ENV_REGEX_NS}/AgentviewCamera",
                update_period=0.0,
                width=int(cfg.get("camera_width", 640)),
                height=int(cfg.get("camera_height", 480)),
                data_types=["rgb"],
                spawn=sim_utils.PinholeCameraCfg(focal_length=24.0, clipping_range=(0.05, 3.0)),
                offset=CameraCfg.OffsetCfg(
                    pos=tuple(float(x) for x in agentview["pos"]),
                    rot=tuple(float(x) for x in agentview["rot"]),
                    convention="world",
                ),
            )
            wrist_camera: CameraCfg = CameraCfg(
                prim_path="{ENV_REGEX_NS}/Robot/link6/d435_camera_link/WristCamera",
                update_period=0.0,
                width=int(cfg.get("camera_width", 640)),
                height=int(cfg.get("camera_height", 480)),
                data_types=["rgb"],
                spawn=sim_utils.PinholeCameraCfg(focal_length=18.0, clipping_range=(0.025, 2.0)),
                offset=CameraCfg.OffsetCfg(
                    pos=tuple(float(x) for x in wrist["pos"]),
                    rot=tuple(float(x) for x in wrist["rot"]),
                    convention="ros",
                ),
            )

        self.sim = sim_utils.SimulationContext(
            sim_utils.SimulationCfg(dt=float(cfg.get("physics_dt", 1.0 / 120.0)), device=str(cfg["device"]))
        )
        self.scene = InteractiveScene(PiperSceneCfg(num_envs=1, env_spacing=2.0))
        self.sim.reset()
        self.robot = self.scene["robot"]
        self.cube = self.scene["cube"]
        self.arm_entity = SceneEntityCfg("robot", joint_names=["joint[1-6]"], body_names=["link6"])
        self.arm_entity.resolve(self.scene)
        finger_entity = SceneEntityCfg("robot", joint_names=["joint[78]"], body_names=["link7", "link8"])
        finger_entity.resolve(self.scene)
        if len(self.arm_entity.joint_ids) != 6 or len(finger_entity.joint_ids) != 2:
            raise RuntimeError("Piper USD must expose six arm joints and two finger joints")
        self.arm_joint_ids = self.arm_entity.joint_ids
        self.finger_joint_ids = finger_entity.joint_ids
        self.ee_body_id = self.arm_entity.body_ids[0]
        self.finger_body_ids = finger_entity.body_ids
        self.ee_jacobi_idx = self.ee_body_id - 1 if self.robot.is_fixed_base else self.ee_body_id
        self.ik = DifferentialIKController(
            DifferentialIKControllerCfg(command_type="position", use_relative_mode=False, ik_method="dls"),
            num_envs=1,
            device=self.sim.device,
        )
        self._target_pos_b = None
        self.reset(settle_steps=int(cfg.get("num_steps_wait", 8)))

    def _ee_pose_b(self):
        ee_pose_w = self.robot.data.body_pose_w[:, self.ee_body_id]
        root_pose_w = self.robot.data.root_pose_w
        return self.subtract_frame_transforms(
            root_pose_w[:, :3], root_pose_w[:, 3:7], ee_pose_w[:, :3], ee_pose_w[:, 3:7]
        )

    def _tcp_pos_b(self):
        # This scene fixes Piper's base at the origin with an identity rotation.
        return (
            self.robot.data.body_pos_w[:, self.finger_body_ids].mean(dim=1)
            - self.robot.data.root_pos_w
        )

    def _observe(self) -> dict[str, np.ndarray]:
        out = {}
        for key in ("agentview_camera", "wrist_camera"):
            rgb = self.scene[key].data.output["rgb"]
            if rgb.numel() == 0:
                raise RuntimeError(f"{key} produced no RGB frame; check ENABLE_CAMERAS and RTX rendering")
            out[key] = rgb[0, :, :, :3].cpu().numpy().copy()
        return out

    def reset(self, *, settle_steps: int = 8) -> tuple[dict[str, np.ndarray], bool, bool]:
        root = self.robot.data.default_root_state.clone()
        root[:, :3] += self.scene.env_origins
        self.robot.write_root_pose_to_sim(root[:, :7])
        self.robot.write_root_velocity_to_sim(root[:, 7:])
        self.robot.write_joint_state_to_sim(
            self.robot.data.default_joint_pos.clone(), self.robot.data.default_joint_vel.clone()
        )
        cube = self.cube.data.default_root_state.clone()
        cube[:, :3] += self.scene.env_origins
        self.cube.write_root_pose_to_sim(cube[:, :7])
        self.cube.write_root_velocity_to_sim(cube[:, 7:])
        self.scene.reset()
        self.ik.reset()
        self.physics_steps = 0
        self.gripper_closed = False
        self._target_pos_b = None
        for _ in range(max(1, int(settle_steps))):
            self.robot.set_joint_position_target(self.robot.data.default_joint_pos.clone())
            self.scene.write_data_to_sim()
            self.sim.step()
            self.scene.update(self.sim.get_physics_dt())
        self._target_pos_b = self._tcp_pos_b().clone()
        self.home_tcp_position_m = self.tcp()
        return self._observe(), False, False

    def step(self, action: np.ndarray) -> tuple[dict[str, np.ndarray], bool, bool, dict]:
        a = np.asarray(action, dtype=np.float32)
        if a.shape != (7,) or not np.isfinite(a).all():
            raise ValueError("Piper Isaac Lab action must be a finite 7-element vector")
        _, ee_quat_b = self._ee_pose_b()
        tcp_pos_b = self._tcp_pos_b()
        delta = self.torch.as_tensor(a[:3] * self.ik_scale, device=self.sim.device).reshape(1, 3)
        # Retain the commanded goal across physics steps. Re-targeting the measured
        # pose every step makes an underdamped Piper accept gravity sag as its new
        # position and almost erases small atomic movements.
        self._target_pos_b += delta
        self._target_pos_b[0, 2] = max(float(self._target_pos_b[0, 2]), self.min_tcp_z_m)
        self._target_pos_b = self.torch.maximum(
            self.torch.minimum(self._target_pos_b, tcp_pos_b + 0.05), tcp_pos_b - 0.05
        )
        self.ik.set_command(self._target_pos_b, ee_quat=ee_quat_b)
        jacobian = self.robot.root_physx_view.get_jacobians()[
            :, self.ee_jacobi_idx, :, self.arm_joint_ids
        ]
        # The IK body is link6, whereas the policy moves the midpoint of the two
        # fingers. Shift its translational Jacobian to that tool point; without
        # this term, wrist rotation turns a vertical command into an X drift.
        offset_w = (
            self.robot.data.body_pos_w[:, self.finger_body_ids].mean(dim=1)
            - self.robot.data.body_pos_w[:, self.ee_body_id]
        )
        angular = jacobian[:, 3:6].transpose(1, 2)
        tool_linear = jacobian[:, :3] + self.torch.cross(
            angular, offset_w.unsqueeze(1).expand_as(angular), dim=-1
        ).transpose(1, 2)
        tool_jacobian = self.torch.cat((tool_linear, jacobian[:, 3:6]), dim=1)
        joint_pos = self.robot.data.joint_pos[:, self.arm_joint_ids]
        joint_des = self.ik.compute(tcp_pos_b, ee_quat_b, tool_jacobian, joint_pos)
        # A single 2 mm token substep must not demand a sudden IK branch change.
        joint_des = self.torch.clamp(joint_des, joint_pos - 0.12, joint_pos + 0.12)
        self.robot.set_joint_position_target(joint_des, joint_ids=self.arm_joint_ids)
        self.gripper_closed = bool(a[6] > 0.5)
        finger_target = (
            (0.0, 0.0) if self.gripper_closed else (self.gripper_open_m, -self.gripper_open_m)
        )
        self.robot.set_joint_position_target(
            self.torch.tensor([finger_target], device=self.sim.device), joint_ids=self.finger_joint_ids
        )
        self.scene.write_data_to_sim()
        self.sim.step()
        self.scene.update(self.sim.get_physics_dt())
        self.physics_steps += 1
        success = self.success()
        truncated = self.physics_steps >= self.max_physics_steps
        return self._observe(), success, truncated, {"success": success}

    def tcp(self) -> np.ndarray:
        pos = self.robot.data.body_pos_w[0, self.finger_body_ids]
        return pos.mean(dim=0).cpu().numpy().copy()

    def ee_quat(self) -> np.ndarray:
        return self.robot.data.body_quat_w[0, self.ee_body_id].cpu().numpy().copy()

    def gripper_width(self) -> float:
        joints = self.robot.data.joint_pos[0, self.finger_joint_ids]
        return abs(float(joints[0] - joints[1]))

    def cube_position(self) -> np.ndarray:
        return self.cube.data.root_pos_w[0].cpu().numpy().copy()

    def policy_state(self) -> dict[str, Any]:
        tcp = self.tcp()
        cube = self.cube_position()
        phase = piper_policy_phase(
            task_name=self.task_name,
            gripper_closed=self.gripper_closed,
            finger_width_m=self.gripper_width(),
            tcp_position_m=tcp,
            cube_position_m=cube,
            target_position_m=self.target_position_m,
            home_position_m=self.home_tcp_position_m,
            min_tcp_z_m=self.min_tcp_z_m,
            transport_tcp_z_m=self.transport_tcp_z_m,
            align_tolerance_m=self.target_xy_tolerance_m,
            target_success_tolerance_m=self.target_success_tolerance_m,
            cube_rest_z_m=self.cube_rest_z,
        )
        return {
            "phase": phase,
            "cube_position_m": cube,
            "target_position_m": self.target_position_m.copy(),
            "home_position_m": self.home_tcp_position_m.copy(),
        }

    def success(self) -> bool:
        cube_pos = self.cube_position()
        if self.task_name.startswith("pick_place"):
            on_target = (
                float(np.linalg.norm(cube_pos[:2] - self.target_position_m[:2]))
                <= self.target_success_tolerance_m
            )
            settled = float(cube_pos[2]) <= self.cube_rest_z + 0.02
            tcp = self.tcp()
            home_xy = float(np.linalg.norm(tcp[:2] - self.home_tcp_position_m[:2]))
            home_z = abs(float(tcp[2] - self.home_tcp_position_m[2]))
            return (
                on_target
                and settled
                and not self.gripper_closed
                and home_xy <= self.target_xy_tolerance_m
                and home_z <= 0.010
            )
        lifted = float(cube_pos[2]) >= self.cube_rest_z + self.cube_lift_m
        held = float(np.linalg.norm(cube_pos - self.tcp())) < 0.12
        return lifted and held and self.gripper_closed

    def close(self) -> None:
        if getattr(self, "_closed", False):
            return
        self._closed = True
        # Release RTX cameras and PhysX views before shutting down Kit.
        del self.robot
        del self.cube
        del self.scene
        self.sim.clear_all_callbacks()
        self.sim.clear_instance()


class PiperIsaacBackend:
    """Adapt the standalone Piper scene to the shared Isaac Lab rollout loop."""

    hold_orientation = False
    @staticmethod
    def reset(env: PiperIsaacLabEnv, hold_action=None, settle_steps: int = 8):
        del hold_action
        return env.reset(settle_steps=settle_steps)

    @staticmethod
    def step(env: PiperIsaacLabEnv, action):
        return env.step(action)

    @staticmethod
    def rgb(obs: dict, camera_name: str):
        return obs[camera_name]

    @staticmethod
    def tcp(env: PiperIsaacLabEnv):
        return env.tcp()

    @staticmethod
    def ee_quat(env: PiperIsaacLabEnv):
        return env.ee_quat()

    @staticmethod
    def gripper_width(env: PiperIsaacLabEnv):
        return env.gripper_width()

    @staticmethod
    def success(env: PiperIsaacLabEnv):
        return env.success()


def probe_move_axes(env: PiperIsaacLabEnv, controller, repeats: int) -> dict[str, list[float]]:
    from core.action_units import MOVE_ATOMS

    results = {}
    for token in MOVE_ATOMS:
        env.reset()
        controller.open_gripper()
        controller.set_orientation_reference(None)
        before = env.tcp()
        action = controller.action_for_atomic(token)
        for _ in range(int(repeats)):
            env.step(controller.with_orientation_hold(action, env.ee_quat()))
        results[token] = [round(float(x), 6) for x in env.tcp() - before]
    return results


def probe_gripper(env: PiperIsaacLabEnv, controller, steps: int = 30) -> dict[str, float]:
    """Measure open/closed finger widths without a VLM endpoint."""
    env.reset()
    opened_before = env.gripper_width()
    close_action = controller.close_gripper()
    for _ in range(int(steps)):
        env.step(close_action)
    closed = env.gripper_width()
    open_action = controller.open_gripper()
    for _ in range(int(steps)):
        env.step(open_action)
    opened_after = env.gripper_width()
    return {"open_before_m": opened_before, "closed_m": closed, "open_after_m": opened_after}


def probe_pick(env: PiperIsaacLabEnv) -> dict[str, float | bool]:
    """Script the configured cube pick to verify real finger/cube contact."""
    env.reset(settle_steps=int(env.cfg.get("num_steps_wait", 60)))
    cube_start_z = float(env.cube.data.root_pos_w[0, 2])
    down = np.array([0.0, 0.0, -0.001, 0.0, 0.0, 0.0, 0.0], dtype=np.float32)
    close = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0], dtype=np.float32)
    up = np.array([0.0, 0.0, 0.001, 0.0, 0.0, 0.0, 1.0], dtype=np.float32)
    for _ in range(160):
        env.step(down)
        if env.tcp()[2] <= env.min_tcp_z_m + 0.001:
            break
    for _ in range(80):
        env.step(close)
    for _ in range(140):
        env.step(up)
        if env.success():
            break
    return {
        "success": env.success(),
        "cube_lift_m": round(float(env.cube.data.root_pos_w[0, 2]) - cube_start_z, 6),
        "gripper_width_m": round(env.gripper_width(), 6),
    }


def probe_place(env: PiperIsaacLabEnv) -> dict[str, Any]:
    """Script pick/carry/place to validate the longer task without a VLM."""
    if not env.task_name.startswith("pick_place"):
        raise ValueError("--probe-place requires a pick_place task")
    env.reset(settle_steps=int(env.cfg.get("num_steps_wait", 60)))
    open_down = np.array([0.0, 0.0, -0.001, 0.0, 0.0, 0.0, 0.0], dtype=np.float32)
    close_hold = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0], dtype=np.float32)
    close_up = np.array([0.0, 0.0, 0.001, 0.0, 0.0, 0.0, 1.0], dtype=np.float32)
    open_hold = np.zeros(7, dtype=np.float32)
    for _ in range(180):
        env.step(open_down)
        if env.tcp()[2] <= env.min_tcp_z_m + 0.001:
            break
    for _ in range(80):
        env.step(close_hold)
    for _ in range(180):
        env.step(close_up)
        if env.tcp()[2] >= env.transport_tcp_z_m:
            break
    for axis in (0, 1):
        for _ in range(300):
            error = float(env.target_position_m[axis] - env.tcp()[axis])
            if abs(error) <= 0.002:
                break
            action = close_hold.copy()
            action[axis] = 0.001 if error > 0.0 else -0.001
            env.step(action)
    for _ in range(180):
        action = close_hold.copy()
        action[2] = -0.001
        env.step(action)
        if env.tcp()[2] <= env.min_tcp_z_m + 0.001:
            break
    for _ in range(80):
        env.step(open_hold)
    for _ in range(180):
        if env.tcp()[2] >= env.transport_tcp_z_m:
            break
        action = open_hold.copy()
        action[2] = 0.001
        env.step(action)
    for axis in (0, 1):
        for _ in range(300):
            error = float(env.home_tcp_position_m[axis] - env.tcp()[axis])
            if abs(error) <= 0.002:
                break
            action = open_hold.copy()
            action[axis] = 0.001 if error > 0.0 else -0.001
            env.step(action)
    for _ in range(180):
        error = float(env.home_tcp_position_m[2] - env.tcp()[2])
        if abs(error) <= 0.002:
            break
        action = open_hold.copy()
        action[2] = 0.001 if error > 0.0 else -0.001
        env.step(action)
    for _ in range(30):
        env.step(open_hold)
        if env.success():
            break
    return {
        "success": env.success(),
        "cube_position_m": [round(float(x), 6) for x in env.cube_position()],
        "target_position_m": [round(float(x), 6) for x in env.target_position_m],
        "target_error_m": round(
            float(np.linalg.norm(env.cube_position()[:2] - env.target_position_m[:2])), 6
        ),
    }
