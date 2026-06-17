"""Shared chassis posture constants and IMU math.

Current baseline posture control is intentionally simple and deployable:

* Gazebo commands hip/arm joints through
  ``/suspension_controller/joint_trajectory``.
* Isaac holds the same joints at the neutral target during Stage A motion
  compensation training.
* Future posture RL should output bounded hip target residuals around these
  profiles, not joint torques or simulator-only contact logic.

This module has no ROS dependencies so it can be shared by Gazebo nodes, Isaac
environments, and offline acceptance scripts.
"""

from __future__ import annotations

import math
from collections.abc import Iterable, Sequence


LEGS = ("fl", "fr", "ml", "mr", "rl", "rr")
HIP_JOINTS = tuple(f"susp_{leg}_joint" for leg in LEGS)
WHEEL_JOINTS = tuple(f"wheel_{leg}_joint" for leg in LEGS)

HIP_TARGET_LIMIT = 0.45
NEUTRAL_HIP_TARGET = tuple(0.0 for _ in LEGS)

# Small deterministic profiles used for direct Gazebo acceptance and as the
# future residual-RL reference surface.
POSTURE_PROFILES = {
    "neutral": NEUTRAL_HIP_TARGET,
    "front_down": (-0.10, -0.10, 0.0, 0.0, 0.06, 0.06),
    "rear_down": (0.06, 0.06, 0.0, 0.0, -0.10, -0.10),
    "raise": (0.08, 0.08, 0.08, 0.08, 0.08, 0.08),
    "lower": (-0.08, -0.08, -0.08, -0.08, -0.08, -0.08),
    "left_trim": (0.06, -0.06, 0.04, -0.04, 0.02, -0.02),
}


def clamp(value: float, lower: float, upper: float) -> float:
    return max(lower, min(upper, value))


def validate_hip_targets(targets: Sequence[float]) -> tuple[float, ...]:
    if len(targets) != len(LEGS):
        raise ValueError(f"expected {len(LEGS)} hip targets, got {len(targets)}")
    return tuple(clamp(float(value), -HIP_TARGET_LIMIT, HIP_TARGET_LIMIT) for value in targets)


def posture_profile(name: str) -> tuple[float, ...]:
    try:
        return POSTURE_PROFILES[name]
    except KeyError as exc:
        valid = ", ".join(sorted(POSTURE_PROFILES))
        raise ValueError(f"unknown posture profile {name!r}; valid profiles: {valid}") from exc


def blend_hip_targets(
    base: Sequence[float],
    residual: Iterable[float],
    *,
    residual_limit: float,
) -> tuple[float, ...]:
    """Apply bounded residual hip targets around a baseline profile."""

    residual_values = tuple(float(value) for value in residual)
    if len(residual_values) != len(LEGS):
        raise ValueError(f"expected {len(LEGS)} residual targets, got {len(residual_values)}")
    return validate_hip_targets(
        [
            float(base_value) + clamp(delta, -abs(residual_limit), abs(residual_limit))
            for base_value, delta in zip(base, residual_values)
        ]
    )


def quat_roll_pitch(w: float, x: float, y: float, z: float) -> tuple[float, float]:
    """Return roll and pitch from a wxyz quaternion."""

    sinr_cosp = 2.0 * (w * x + y * z)
    cosr_cosp = 1.0 - 2.0 * (x * x + y * y)
    roll = math.atan2(sinr_cosp, cosr_cosp)
    sinp = clamp(2.0 * (w * y - z * x), -1.0, 1.0)
    return roll, math.asin(sinp)


def projected_gravity(w: float, x: float, y: float, z: float) -> tuple[float, float, float]:
    """Project world gravity ``[0, 0, -1]`` into the body frame.

    This is algebraically equivalent to Isaac Lab's
    ``quat_rotate_inverse(quat_w, [0, 0, -1])`` for wxyz quaternions.
    """

    gx = 2.0 * w * y - 2.0 * x * z
    gy = -2.0 * w * x - 2.0 * y * z
    gz = 1.0 - 2.0 * w * w - 2.0 * z * z
    return gx, gy, gz
