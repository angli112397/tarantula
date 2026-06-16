# Copyright (c) 2026 Tarantula project
# SPDX-License-Identifier: BSD-3-Clause
"""Tarantula rover articulation config for Isaac Lab (M5/M7 v4).

Reuses the USD produced by the ``UrdfConverter`` run: ``susp_*_joint``
position drive with stiffness=120/damping=8 (matches the Gazebo
``springStiffness``/``springReference``/``implicitSpringDamper`` + joint
``dynamics damping`` -- see docs/01 §7; v4 applies position targets directly
from the kinematic mapping without a SuspensionController wrapper).
``wheel_*_joint`` is velocity-driven (``target_type="velocity"``,
stiffness=``WHEEL_DRIVE_GAIN``) for direct differential-drive control.
"""

import os
import xml.etree.ElementTree as ET

import isaaclab.sim as sim_utils
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets import ArticulationCfg

TARANTULA_USD_DIR = "/tmp/tarantula_usd"
TARANTULA_USD_PATH = os.path.join(TARANTULA_USD_DIR, "tarantula_core_v2.usd")

# Generated via: xacro tarantula_core.urdf.xacro lidar:=false > URDF_PATH
URDF_PATH = "/tmp/tarantula_core.urdf"

# Velocity-drive P gain for wheel_*_joint (rad/s -> N*m). Wheel limit is
# effort=30 N*m / velocity=30 rad/s. Chosen so that a full-scale RL action
# (+-1 * action_scale_wheel_omega=3.0 rad/s, see suspension_env_cfg.py) demands
# exactly 30 N*m -- the full [-1,1] action range maps onto the actuator's full
# torque range with no saturated dead zone.
WHEEL_DRIVE_GAIN = 10.0


def _compute_wheel_bottom_z(urdf_path: str) -> float:
    """Lowest point of any wheel, in base_link's frame at default joint angles.

    Walks each ``wheel_*_joint``'s parent chain back to ``base_link``,
    summing joint-origin z offsets, then subtracts the wheel collision
    cylinder's radius. This assumes every joint origin along the
    ``base_link -> ... -> wheel_*_link`` chain has ``rpy="0 0 0"`` (true for
    the current chassis -- all leg-angle rotation is baked into link-local
    visual/collision geometry, not joint origins); raises if that assumption
    is violated so a future chassis redesign with rotated joint origins fails
    loudly here instead of silently producing a wrong spawn height.
    """
    if not os.path.exists(urdf_path):
        raise FileNotFoundError(
            f"{urdf_path} not found -- generate it first with (ROS sourced):\n"
            "  xacro src/tarantula_description/urdf/tarantula_core.urdf.xacro lidar:=false"
            f" > {urdf_path}"
        )

    root = ET.parse(urdf_path).getroot()

    joints_by_child = {}
    for joint in root.findall("joint"):
        child = joint.find("child").get("link")
        parent = joint.find("parent").get("link")
        origin = joint.find("origin")
        xyz = [float(v) for v in (origin.get("xyz", "0 0 0") if origin is not None else "0 0 0").split()]
        rpy = [float(v) for v in (origin.get("rpy", "0 0 0") if origin is not None else "0 0 0").split()]
        if any(abs(v) > 1e-9 for v in rpy):
            raise NotImplementedError(
                f"joint '{joint.get('name')}' has a non-zero origin rpy={rpy} -- "
                "_compute_wheel_bottom_z only sums joint-origin translations and "
                "assumes zero joint-origin rotation; update this helper for the "
                "new chassis geometry."
            )
        joints_by_child[child] = (parent, xyz)

    lowest_z = None
    for joint in root.findall("joint"):
        name = joint.get("name")
        if not (name.startswith("wheel_") and name.endswith("_joint")):
            continue
        wheel_link = joint.find("child").get("link")

        z = 0.0
        link = wheel_link
        while link != "base_link":
            parent, xyz = joints_by_child[link]
            z += xyz[2]
            link = parent

        cylinder = root.find(f"./link[@name='{wheel_link}']/collision/geometry/cylinder")
        bottom = z - float(cylinder.get("radius"))
        lowest_z = bottom if lowest_z is None else min(lowest_z, bottom)

    if lowest_z is None:
        raise ValueError(f"no wheel_*_joint found in {urdf_path}")
    return lowest_z


# Default spawn height above the local terrain surface: derived from the
# chassis URDF (lowest wheel point below base_link's origin) plus a small
# clearance margin, so it can't drift out of sync if the leg geometry
# changes -- see _compute_wheel_bottom_z.
SPAWN_GROUND_CLEARANCE = 0.03
SPAWN_Z_OFFSET = SPAWN_GROUND_CLEARANCE - _compute_wheel_bottom_z(URDF_PATH)


def ensure_tarantula_usd(urdf_path: str = URDF_PATH) -> str:
    """Regenerate the tarantula USD if missing (M7 v2 joint-drive config)."""
    if os.path.exists(TARANTULA_USD_PATH):
        return TARANTULA_USD_PATH

    from isaaclab.sim.converters import UrdfConverter, UrdfConverterCfg

    urdf_cfg = UrdfConverterCfg(
        asset_path=urdf_path,
        usd_dir=TARANTULA_USD_DIR,
        usd_file_name="tarantula_core_v2.usd",
        force_usd_conversion=True,
        fix_base=False,
        merge_fixed_joints=True,
        self_collision=False,
        # Keep False: native USD Cylinder (radius=0.12, height=0.07) matches
        # Gazebo collision geometry exactly. True caused near-zero in-place yaw
        # (capsule end caps ~2x wheel width).
        joint_drive=UrdfConverterCfg.JointDriveCfg(
            drive_type="force",
            target_type={"susp_.*_joint": "position", "wheel_.*_joint": "velocity"},
            gains=UrdfConverterCfg.JointDriveCfg.PDGainsCfg(
                stiffness={"susp_.*_joint": 120.0, "wheel_.*_joint": WHEEL_DRIVE_GAIN},
                damping={"susp_.*_joint": 8.0, "wheel_.*_joint": 0.0},
            ),
        ),
    )
    return UrdfConverter(urdf_cfg).usd_path


TARANTULA_CFG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(usd_path=TARANTULA_USD_PATH, activate_contact_sensors=True),
    init_state=ArticulationCfg.InitialStateCfg(pos=(0.0, 0.0, SPAWN_Z_OFFSET)),
    # Single implicit-actuator group covering every joint: stiffness/damping
    # explicitly set to None (ActuatorBaseCfg has no default -- MISSING) so
    # the USD-configured drive gains are preserved as-is, i.e. susp_*_joint
    # keeps stiffness=120/damping=8 and wheel_*_joint keeps its
    # velocity-drive gain (WHEEL_DRIVE_GAIN).
    actuators={"all": ImplicitActuatorCfg(joint_names_expr=[".*"], stiffness=None, damping=None)},
)
"""Articulation config for the tarantula rover (susp_*_joint position-drive, wheel_*_joint velocity-drive)."""
