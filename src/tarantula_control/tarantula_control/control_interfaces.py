"""Shared control-interface helpers for Gazebo deployment nodes."""

from dataclasses import dataclass

from .suspension_core import LEGS
from .vehicle_geometry import VEHICLE_GEOMETRY


WHEEL_RADIUS = VEHICLE_GEOMETRY.wheel_radius       # 0.13 m (tarantula_common.xacro)
WHEEL_SEPARATION = VEHICLE_GEOMETRY.wheel_center_track  # 0.66 m (2*(pivot_y + wheel_lateral_offset))

# Calibrated skid-steer effective track multiplier. Physical track is 0.66 m.
# Combined with yaw_track_scale (0.7287) in MotionControlConfig:
#   turn_track = 0.66 * 1.6 * 0.7287 = 0.770 m
SKID_STEER_EFFECTIVE_TRACK_MULTIPLIER = 1.6
EFFECTIVE_TRACK = WHEEL_SEPARATION * SKID_STEER_EFFECTIVE_TRACK_MULTIPLIER  # 1.056 m base

DEFAULT_TRACK_SCALE = 1.0

# Wheel speed cap: vx_max / wheel_radius = 0.78 m/s → 6.0 rad/s provides ~25% headroom
# above nav2 max (0.6 m/s). URDF joint velocity limit is 30 rad/s.
MAX_ABS_WHEEL_OMEGA = 6.0

# Wheel-load reference for F/T normalization (RL observation), derived from
# the same URDF VEHICLE_GEOMETRY.total_mass everything else uses -- NOT the
# old hardcoded 23.1 kg. That number was checked against an actual settled
# Isaac Lab rollout (2026-06-19): real per-wheel contact force averaged
# ~54.5 N, matching total_mass/6*g (53.96 N) almost exactly, not 23.1's
# implied 37.77 N. The "unsprung mass reduces effective load" story this
# constant used to be justified by does not hold empirically -- weight is
# transmitted to the ground through the wheels regardless of sprung/unsprung
# classification. Must equal suspension_env_cfg.py's nominal_wheel_load --
# training and deployment normalize wheel_force_b by this same constant.
NOMINAL_WHEEL_LOAD = VEHICLE_GEOMETRY.total_mass * 9.81 / 6.0

LEFT_LEGS = ("fl", "ml", "rl")
RIGHT_LEGS = ("fr", "mr", "rr")
# Joint direction calibration from semantic wheel speed to ros2_control
# velocity command. Current URDF uses the same +Y wheel axis on both sides, so
# positive wheel velocity means forward for every wheel.
WHEEL_DIRECTION = {
    "fl": 1.0,
    "fr": 1.0,
    "ml": 1.0,
    "mr": 1.0,
    "rl": 1.0,
    "rr": 1.0,
}


def clamp(value: float, lower: float, upper: float) -> float:
    return max(lower, min(upper, value))


def clamp_abs(value: float, limit: float) -> float:
    limit = abs(limit)
    return clamp(value, -limit, limit)


def wheel_force_normalized(force_xyz: tuple[float, float, float]) -> tuple[float, float, float]:
    return tuple(clamp(component / NOMINAL_WHEEL_LOAD, -3.0, 3.0) for component in force_xyz)


def mean_wheel_forward_velocity(wheel_vel: dict[str, float]) -> float:
    return sum(wheel_vel[leg] * WHEEL_DIRECTION[leg] for leg in LEGS) / len(LEGS) * WHEEL_RADIUS


def skid_steer_wheel_speeds(
    cmd_vx: float,
    cmd_wz: float,
    *,
    track_scale: float = DEFAULT_TRACK_SCALE,
    left_scale: float = 1.0,
    right_scale: float = 1.0,
) -> list[float]:
    """Map cmd_vel to per-wheel angular velocities in LEGS order."""

    turn_track = EFFECTIVE_TRACK * float(track_scale)
    left = ((cmd_vx - 0.5 * turn_track * cmd_wz) / WHEEL_RADIUS) * float(left_scale)
    right = ((cmd_vx + 0.5 * turn_track * cmd_wz) / WHEEL_RADIUS) * float(right_scale)
    semantic = {leg: left if leg in LEFT_LEGS else right for leg in LEGS}
    return [semantic[leg] * WHEEL_DIRECTION[leg] for leg in LEGS]


@dataclass
class CmdVelLimiter:
    max_abs_vx: float
    max_abs_wz: float

    def clamp(self, vx: float, wz: float) -> tuple[float, float]:
        return clamp_abs(vx, self.max_abs_vx), clamp_abs(wz, self.max_abs_wz)
