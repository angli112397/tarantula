"""Shared control-interface helpers for Gazebo deployment nodes."""

from dataclasses import dataclass

from .suspension_core import LEGS


WHEEL_RADIUS = 0.13
WHEEL_SEPARATION = 0.64
WHEEL_SEPARATION_MULTIPLIER = 1.6
EFFECTIVE_TRACK = WHEEL_SEPARATION * WHEEL_SEPARATION_MULTIPLIER
YAW_AUTHORITY_MULTIPLIER = 4.0
MAX_ABS_WHEEL_OMEGA = 6.0
TRACK_SCALE_DELTA_LIMIT = 0.30
DRIVE_SCALE_DELTA_LIMIT = 0.20

NOMINAL_WHEEL_LOAD = 23.1 * 9.81 / 6.0

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
    track_scale: float = YAW_AUTHORITY_MULTIPLIER,
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
