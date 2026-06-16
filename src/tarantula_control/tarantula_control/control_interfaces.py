"""Shared control-interface helpers for Gazebo deployment nodes."""

from dataclasses import dataclass

from .suspension_core import LEGS, SuspensionConfig


WHEEL_RADIUS = 0.13
WHEEL_SEPARATION = 0.64
WHEEL_SEPARATION_MULTIPLIER = 1.6
EFFECTIVE_TRACK = WHEEL_SEPARATION * WHEEL_SEPARATION_MULTIPLIER

_geom = SuspensionConfig()
SUSP_ACTUATOR_KP = _geom.ff_stiffness
SUSP_ACTUATOR_KD = _geom.actuator_damping
SUSP_EFFORT_LIMIT = _geom.actuator_effort_limit
SUSP_JOINT_LIMIT = 0.6
NOMINAL_WHEEL_LOAD = 23.1 * 9.81 / 6.0

LEFT_LEGS = ("fl", "ml", "rl")
RIGHT_LEGS = ("fr", "mr", "rr")


def clamp(value: float, lower: float, upper: float) -> float:
    return max(lower, min(upper, value))


def clamp_abs(value: float, limit: float) -> float:
    limit = abs(limit)
    return clamp(value, -limit, limit)


def wheel_load_normalized(force_n: float) -> float:
    return clamp(force_n / NOMINAL_WHEEL_LOAD, 0.0, 3.0)


def mean_wheel_forward_velocity(wheel_vel: dict[str, float]) -> float:
    return sum(wheel_vel[leg] for leg in LEGS) / len(LEGS) * WHEEL_RADIUS


def skid_steer_wheel_speeds(cmd_vx: float, cmd_wz: float) -> list[float]:
    """Map cmd_vel to per-wheel angular velocities in LEGS order."""

    left = (cmd_vx - 0.5 * EFFECTIVE_TRACK * cmd_wz) / WHEEL_RADIUS
    right = (cmd_vx + 0.5 * EFFECTIVE_TRACK * cmd_wz) / WHEEL_RADIUS
    return [left if leg in LEFT_LEGS else right for leg in LEGS]


@dataclass
class CmdVelLimiter:
    max_abs_vx: float
    max_abs_wz: float

    def clamp(self, vx: float, wz: float) -> tuple[float, float]:
        return clamp_abs(vx, self.max_abs_vx), clamp_abs(wz, self.max_abs_wz)
