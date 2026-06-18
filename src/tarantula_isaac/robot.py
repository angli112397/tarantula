# Copyright (c) 2026 Tarantula project
# SPDX-License-Identifier: BSD-3-Clause
"""Tarantula rover articulation config for the v3 active-suspension baseline.

The Gazebo and Isaac baselines share ``tarantula_core_v3.urdf.xacro``:
``susp_*_joint`` is a position-controlled hip trim/posture joint and
``wheel_*_joint`` is velocity-driven for direct wheel control.
"""

import os
import hashlib
import shutil
import subprocess
import xml.etree.ElementTree as ET
from pathlib import Path

import isaaclab.sim as sim_utils
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets import ArticulationCfg

TARANTULA_USD_DIR = "/tmp/tarantula_usd"
TARANTULA_USD_NAME = "tarantula_v3_active_suspension_sphere_wheels.usd"
TARANTULA_USD_PATH = os.path.join(TARANTULA_USD_DIR, TARANTULA_USD_NAME)

# Generated via: xacro tarantula_core_v3.urdf.xacro wheel_collision:=sphere lidar:=false > URDF_PATH
URDF_PATH = "/tmp/tarantula_v3.urdf"
URDF_STAMP_PATH = f"{URDF_PATH}.sha256"
USD_STAMP_PATH = os.path.join(TARANTULA_USD_DIR, f"{TARANTULA_USD_NAME}.sha256")

# Velocity-drive P gain for wheel_*_joint (rad/s -> N*m). Wheel limit is
# effort=38 N*m / velocity=30 rad/s. Stage B uses a ±6 rad/s final wheel target
# envelope around the scheduled skid-steer baseline. This gain preserves enough
# yaw authority for large differential wheel targets.
WHEEL_DRIVE_GAIN = 12.7


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _xacro_sources() -> list[Path]:
    urdf_dir = _repo_root() / "src" / "tarantula_description" / "urdf"
    return [
        urdf_dir / "tarantula_core_v2.urdf.xacro",
        urdf_dir / "tarantula_core_v3.urdf.xacro",
        urdf_dir / "tarantula_chassis_v2.xacro",
        urdf_dir / "tarantula_common.xacro",
    ]


def _fingerprint(paths: list[Path], extra: str = "") -> str:
    digest = hashlib.sha256()
    digest.update(extra.encode("utf-8"))
    for path in paths:
        digest.update(str(path).encode("utf-8"))
        digest.update(path.read_bytes())
    return digest.hexdigest()


def _read_stamp(path: str) -> str:
    try:
        return Path(path).read_text(encoding="utf-8").strip()
    except FileNotFoundError:
        return ""


def _prepend_env_paths(env: dict[str, str], key: str, paths: list[Path | str]) -> None:
    existing = env.get(key, "")
    values = [str(path) for path in paths if os.path.exists(path)]
    if existing:
        values.append(existing)
    if values:
        env[key] = os.pathsep.join(values)


def _xacro_env() -> dict[str, str]:
    env = os.environ.copy()
    repo_install = _repo_root() / "install"
    package_prefixes = [path for path in repo_install.iterdir()] if repo_install.exists() else []
    ros_prefix = Path("/opt/ros/humble")
    ros_python_paths = [
        ros_prefix / "lib" / "python3.10" / "site-packages",
        ros_prefix / "local" / "lib" / "python3.10" / "dist-packages",
    ]
    _prepend_env_paths(env, "AMENT_PREFIX_PATH", package_prefixes + [ros_prefix])
    _prepend_env_paths(env, "CMAKE_PREFIX_PATH", package_prefixes + [ros_prefix])
    _prepend_env_paths(env, "PYTHONPATH", ros_python_paths)
    return env


def _ensure_core_urdf(urdf_path: str) -> None:
    """Generate the temporary Isaac URDF from xacro when sources changed."""
    source_paths = _xacro_sources()
    source_hash = _fingerprint(source_paths, extra="v3;wheel_collision=sphere;lidar=false")
    if os.path.exists(urdf_path) and _read_stamp(URDF_STAMP_PATH) == source_hash:
        return

    xacro_path = _repo_root() / "src" / "tarantula_description" / "urdf" / "tarantula_core_v3.urdf.xacro"
    xacro_bin = shutil.which("xacro") or "/opt/ros/humble/bin/xacro"
    if not os.path.exists(xacro_bin):
        raise FileNotFoundError(
            f"{urdf_path} not found and xacro is unavailable. Source ROS 2 Humble or install xacro, then run:\n"
            f"  xacro {xacro_path} wheel_collision:=sphere lidar:=false > {urdf_path}"
        )
    if not xacro_path.exists():
        raise FileNotFoundError(f"missing Tarantula xacro source: {xacro_path}")

    os.makedirs(os.path.dirname(urdf_path), exist_ok=True)
    command = [xacro_bin, str(xacro_path), "wheel_collision:=sphere", "lidar:=false"]
    result = subprocess.run(
        command,
        check=False,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        env=_xacro_env(),
    )
    if result.returncode != 0:
        raise RuntimeError(
            "failed to generate Isaac URDF with xacro:\n"
            f"  command: {' '.join(command)}\n"
            f"  stderr: {result.stderr.strip()}\n"
            f"  stdout: {result.stdout.strip()}"
        )
    Path(urdf_path).write_text(result.stdout, encoding="utf-8")
    Path(URDF_STAMP_PATH).write_text(source_hash, encoding="utf-8")


def _compute_wheel_bottom_z(urdf_path: str) -> float:
    """Lowest point of any wheel, in base_link's frame at default joint angles.

    Walks each ``wheel_*_joint``'s parent chain back to ``base_link``,
    summing joint-origin z offsets, then subtracts the wheel collision
    primitive radius. This assumes every joint origin along the
    ``base_link -> ... -> wheel_*_link`` chain has ``rpy="0 0 0"`` (true for
    the current chassis -- all leg-angle rotation is baked into link-local
    visual/collision geometry, not joint origins); raises if that assumption
    is violated so a future chassis redesign with rotated joint origins fails
    loudly here instead of silently producing a wrong spawn height.
    """
    _ensure_core_urdf(urdf_path)

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

        geometry = root.find(f"./link[@name='{wheel_link}']/collision/geometry")
        sphere = geometry.find("sphere") if geometry is not None else None
        cylinder = geometry.find("cylinder") if geometry is not None else None
        if sphere is not None:
            radius = float(sphere.get("radius"))
        elif cylinder is not None:
            radius = float(cylinder.get("radius"))
        else:
            raise ValueError(f"{wheel_link} collision geometry must be sphere or cylinder")
        bottom = z - radius
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
    """Regenerate the tarantula USD if missing or if the URDF/control contract changed."""
    _ensure_core_urdf(urdf_path)
    usd_hash = _fingerprint(
        [Path(urdf_path)],
        extra=f"usd={TARANTULA_USD_NAME};wheel_drive_gain={WHEEL_DRIVE_GAIN}",
    )
    if os.path.exists(TARANTULA_USD_PATH) and _read_stamp(USD_STAMP_PATH) == usd_hash:
        return TARANTULA_USD_PATH

    from isaaclab.sim.converters import UrdfConverter, UrdfConverterCfg

    urdf_cfg = UrdfConverterCfg(
        asset_path=urdf_path,
        usd_dir=TARANTULA_USD_DIR,
        usd_file_name=TARANTULA_USD_NAME,
        force_usd_conversion=True,
        fix_base=False,
        merge_fixed_joints=True,
        self_collision=False,
        joint_drive=UrdfConverterCfg.JointDriveCfg(
            drive_type="force",
            target_type={"susp_.*_joint": "position", "wheel_.*_joint": "velocity"},
            gains=UrdfConverterCfg.JointDriveCfg.PDGainsCfg(
                stiffness={"susp_.*_joint": 130.0, "wheel_.*_joint": WHEEL_DRIVE_GAIN},
                damping={"susp_.*_joint": 11.0, "wheel_.*_joint": 0.0},
            ),
        ),
    )
    usd_path = UrdfConverter(urdf_cfg).usd_path
    Path(USD_STAMP_PATH).write_text(usd_hash, encoding="utf-8")
    return usd_path


TARANTULA_CFG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(usd_path=TARANTULA_USD_PATH, activate_contact_sensors=True),
    init_state=ArticulationCfg.InitialStateCfg(pos=(0.0, 0.0, SPAWN_Z_OFFSET)),
    # Single implicit-actuator group covering every joint: stiffness/damping
    # explicitly set to None (ActuatorBaseCfg has no default -- MISSING) so
    # the USD-configured drive gains are preserved as-is, i.e. susp_*_joint
    # keeps stiffness=130/damping=11 and wheel_*_joint keeps its
    # velocity-drive gain (WHEEL_DRIVE_GAIN).
    actuators={"all": ImplicitActuatorCfg(joint_names_expr=[".*"], stiffness=None, damping=None)},
)
"""Articulation config for the tarantula rover (susp_*_joint position-drive, wheel_*_joint velocity-drive)."""
