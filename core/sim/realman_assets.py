"""Resolve RealMan's official RM65 and RMC-AIDA-L robot descriptions.

Show-Harness does not redistribute RealMan's meshes. The current RMC-AIDA-L path
uses the complete official Ecosystem_Cases model, including its four-bar grippers.
The older standalone RM65 helper remains available for existing arms-only callers;
it reads ros2_rm_robot and appends a deterministic simulation gripper.
"""
from __future__ import annotations

import hashlib
import os
from pathlib import Path
import xml.etree.ElementTree as ET


OFFICIAL_REPOSITORY = "https://github.com/RealManRobot/ros2_rm_robot.git"
OFFICIAL_BRANCH = "humble"
RMC_AIDAL_REPOSITORY = "https://github.com/RealManRobot/Ecosystem_Cases.git"
RMC_AIDAL_BRANCH = "RMC-AIDA-L"
RMC_AIDAL_PACKAGE = "Embodied lifting robot_two wheels_RM65-B-V"
RMC_AIDAL_RELATIVE_PACKAGE = (
    Path("模型文件") / "URDF" / "具身URDF" / RMC_AIDAL_PACKAGE
)


def resolve_realman_root(path: str | os.PathLike[str] | None = None) -> Path:
    """Return a checkout containing ``rm_description/urdf/rm_65.urdf``."""
    candidates: list[Path] = []
    if path:
        candidates.append(Path(path).expanduser())
    env = os.environ.get("REALMAN_ROS2_ROOT")
    if env:
        candidates.append(Path(env).expanduser())
    candidates.extend(
        [
            Path.home() / "ros2_rm_robot",
            Path.home() / "code" / "ros2_rm_robot",
        ]
    )
    checked: list[str] = []
    for candidate in candidates:
        root = candidate.resolve()
        # Also accept a direct path to the rm_description package.
        if root.name == "rm_description":
            root = root.parent
        urdf = root / "rm_description" / "urdf" / "rm_65.urdf"
        checked.append(str(urdf))
        if urdf.is_file():
            mesh = root / "rm_description" / "meshes" / "rm_65_arm" / "link6.STL"
            if not mesh.is_file():
                raise FileNotFoundError(f"RM65 description is incomplete; missing {mesh}")
            return root
    locations = "\n  - ".join(checked) if checked else "(no candidates)"
    raise FileNotFoundError(
        "Official RM65 description not found. Clone it with:\n"
        f"  git clone --depth 1 --branch {OFFICIAL_BRANCH} {OFFICIAL_REPOSITORY} "
        "$HOME/ros2_rm_robot\n"
        "or pass --realman-root /path/to/ros2_rm_robot. Checked:\n  - " + locations
    )


def _append_parallel_gripper(robot: ET.Element) -> None:
    """Append a compact, physically actuated two-finger gripper to ``Link6``."""
    fragment = ET.fromstring(
        """
<fragment>
  <link name="gripper_palm">
    <inertial>
      <origin xyz="0 0 0" rpy="0 0 0"/>
      <mass value="0.18"/>
      <inertia ixx="0.00008" ixy="0" ixz="0" iyy="0.00018" iyz="0" izz="0.00018"/>
    </inertial>
    <visual>
      <geometry><box size="0.045 0.095 0.035"/></geometry>
      <material name="realman_gripper"><color rgba="0.12 0.14 0.17 1"/></material>
    </visual>
    <collision><geometry><box size="0.045 0.095 0.035"/></geometry></collision>
  </link>
  <joint name="gripper_mount" type="fixed">
    <origin xyz="0 0 0.035" rpy="0 0 0"/>
    <parent link="Link6"/><child link="gripper_palm"/>
  </joint>
  <link name="finger_left">
    <inertial>
      <origin xyz="0 0 0.050" rpy="0 0 0"/>
      <mass value="0.035"/>
      <inertia ixx="0.000030" ixy="0" ixz="0" iyy="0.000030" iyz="0" izz="0.000004"/>
    </inertial>
    <visual>
      <origin xyz="0 0 0.050" rpy="0 0 0"/>
      <geometry><box size="0.020 0.012 0.100"/></geometry>
      <material name="realman_finger"><color rgba="0.06 0.07 0.08 1"/></material>
    </visual>
    <collision>
      <origin xyz="0 0 0.050" rpy="0 0 0"/>
      <geometry><box size="0.020 0.012 0.100"/></geometry>
    </collision>
  </link>
  <joint name="finger_joint_left" type="prismatic">
    <origin xyz="0 0.008 0.005" rpy="0 0 0"/>
    <parent link="gripper_palm"/><child link="finger_left"/>
    <axis xyz="0 1 0"/>
    <limit lower="0" upper="0.035" effort="80" velocity="0.25"/>
    <dynamics damping="2" friction="0.4"/>
  </joint>
  <link name="finger_right">
    <inertial>
      <origin xyz="0 0 0.050" rpy="0 0 0"/>
      <mass value="0.035"/>
      <inertia ixx="0.000030" ixy="0" ixz="0" iyy="0.000030" iyz="0" izz="0.000004"/>
    </inertial>
    <visual>
      <origin xyz="0 0 0.050" rpy="0 0 0"/>
      <geometry><box size="0.020 0.012 0.100"/></geometry>
      <material name="realman_finger"><color rgba="0.06 0.07 0.08 1"/></material>
    </visual>
    <collision>
      <origin xyz="0 0 0.050" rpy="0 0 0"/>
      <geometry><box size="0.020 0.012 0.100"/></geometry>
    </collision>
  </link>
  <joint name="finger_joint_right" type="prismatic">
    <origin xyz="0 -0.008 0.005" rpy="0 0 0"/>
    <parent link="gripper_palm"/><child link="finger_right"/>
    <axis xyz="0 -1 0"/>
    <limit lower="0" upper="0.035" effort="80" velocity="0.25"/>
    <dynamics damping="2" friction="0.4"/>
  </joint>
</fragment>
"""
    )
    for child in list(fragment):
        robot.append(child)


def prepare_rm65_urdf(
    root: str | os.PathLike[str] | Path,
    cache_dir: str | os.PathLike[str] | Path | None = None,
) -> Path:
    """Create and return the cached RM65 + parallel-gripper URDF."""
    checkout = resolve_realman_root(root)
    description = checkout / "rm_description"
    source = description / "urdf" / "rm_65.urdf"
    digest = hashlib.sha256(
        source.read_bytes() + str(description.resolve()).encode("utf-8")
    ).hexdigest()[:12]
    out_dir = (
        Path(cache_dir).expanduser()
        if cache_dir is not None
        else Path.home() / ".cache" / "show_harness" / "realman"
    )
    out_dir.mkdir(parents=True, exist_ok=True)
    output = out_dir / f"rm65_show_harness_{digest}.urdf"
    if output.is_file():
        return output.resolve()

    tree = ET.parse(source)
    robot = tree.getroot()
    robot.set("name", "rm65_show_harness")
    prefix = "package://rm_description/"
    for mesh in robot.iter("mesh"):
        filename = mesh.get("filename", "")
        if filename.startswith(prefix):
            resolved = (description / filename[len(prefix) :]).resolve()
            if not resolved.is_file():
                raise FileNotFoundError(f"RM65 mesh referenced by URDF is missing: {resolved}")
            mesh.set("filename", resolved.as_posix())
    _append_parallel_gripper(robot)
    ET.indent(tree, space="  ")
    tree.write(output, encoding="utf-8", xml_declaration=True)
    return output.resolve()


def resolve_rmc_aidal_root(path: str | os.PathLike[str] | None = None) -> Path:
    """Return the official package directory for the RMC-AIDA-L RM65-B-V model."""
    candidates: list[Path] = []
    if path:
        candidates.append(Path(path).expanduser())
    env = os.environ.get("RMC_AIDAL_ROOT")
    if env:
        candidates.append(Path(env).expanduser())
    candidates.extend(
        [
            Path.home() / "Ecosystem_Cases_RMC_AIDA_L",
            Path.home() / "Ecosystem_Cases",
        ]
    )
    checked: list[str] = []
    urdf_name = f"{RMC_AIDAL_PACKAGE}.urdf"
    for candidate in candidates:
        candidate = candidate.resolve()
        package = (
            candidate
            if candidate.name == RMC_AIDAL_PACKAGE
            else candidate / RMC_AIDAL_RELATIVE_PACKAGE
        )
        source = package / "urdf" / urdf_name
        checked.append(str(source))
        if source.is_file():
            required = ("base_link.STL", "link_5.STL", "link_left_6.STL", "link_right_6.STL")
            missing = [name for name in required if not (package / "meshes" / name).is_file()]
            if missing:
                raise FileNotFoundError(
                    f"RMC-AIDA-L package is incomplete under {package}; missing {missing}"
                )
            return package
    locations = "\n  - ".join(checked) if checked else "(no candidates)"
    raise FileNotFoundError(
        "Official RMC-AIDA-L RM65-B-V model not found. Install it with:\n"
        "  bash scripts/realman/setup_sim_assets.sh\n"
        "or pass --rmc-aidal-root /path/to/Ecosystem_Cases_RMC_AIDA_L. Checked:\n"
        "  - "
        + locations
    )


def prepare_rmc_aidal_urdf(
    root: str | os.PathLike[str] | Path,
    cache_dir: str | os.PathLike[str] | Path | None = None,
) -> Path:
    """Resolve mesh paths and lock the official mobile-base wheel joints.

    The base link remains fixed by Isaac Lab.  Locking both wheel joints removes
    unused wheel degrees of freedom while retaining the official chassis, lift,
    dual-arm, camera and gripper geometry.  The lift stays actuated so the scene
    can set one static working height without enabling base navigation.
    """
    package = resolve_rmc_aidal_root(root)
    source = package / "urdf" / f"{RMC_AIDAL_PACKAGE}.urdf"
    digest = hashlib.sha256(
        source.read_bytes() + str(package.resolve()).encode("utf-8")
    ).hexdigest()[:12]
    out_dir = (
        Path(cache_dir).expanduser()
        if cache_dir is not None
        else Path.home() / ".cache" / "show_harness" / "realman"
    )
    out_dir.mkdir(parents=True, exist_ok=True)
    output = out_dir / f"rmc_aida_l_rm65_b_v_static_base_{digest}.urdf"
    if output.is_file():
        return output.resolve()

    tree = ET.parse(source)
    robot = tree.getroot()
    robot.set("name", "rmc_aida_l_rm65_b_v_static_base")
    for mesh in robot.iter("mesh"):
        filename = mesh.get("filename", "")
        if not filename.startswith("package://"):
            continue
        resolved = package / "meshes" / Path(filename).name
        if not resolved.is_file():
            raise FileNotFoundError(
                f"RMC-AIDA-L mesh referenced by URDF is missing: {resolved}"
            )
        mesh.set("filename", resolved.resolve().as_posix())

    by_name = {joint.get("name"): joint for joint in robot.findall("joint")}
    for wheel_name in ("joint_1", "joint_2"):
        joint = by_name[wheel_name]
        joint.set("type", "fixed")
        for child_name in ("axis", "limit", "dynamics", "mimic"):
            child = joint.find(child_name)
            if child is not None:
                joint.remove(child)

    ET.indent(tree, space="  ")
    tree.write(output, encoding="utf-8", xml_declaration=True)
    return output.resolve()


__all__ = [
    "OFFICIAL_BRANCH",
    "OFFICIAL_REPOSITORY",
    "RMC_AIDAL_BRANCH",
    "RMC_AIDAL_PACKAGE",
    "RMC_AIDAL_REPOSITORY",
    "prepare_rm65_urdf",
    "prepare_rmc_aidal_urdf",
    "resolve_realman_root",
    "resolve_rmc_aidal_root",
]
