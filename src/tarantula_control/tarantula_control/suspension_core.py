"""Shared chassis posture constants and IMU math.

Current baseline posture control is intentionally simple and deployable:

* Gazebo commands hip/arm joints through
  ``/suspension_controller/joint_trajectory``.
* Isaac and Gazebo use the same joint order for hip target commands.
* Posture RL outputs bounded hip position targets, not joint torques or
  simulator-only contact logic.

This module has no ROS dependencies so it can be shared by Gazebo nodes, Isaac
environments, and offline acceptance scripts.
"""

from __future__ import annotations

import math
from collections.abc import Sequence


LEGS = ("fl", "fr", "ml", "mr", "rl", "rr")
HIP_JOINTS = tuple(f"susp_{leg}_joint" for leg in LEGS)
WHEEL_JOINTS = tuple(f"wheel_{leg}_joint" for leg in LEGS)

HIP_TARGET_LIMIT = 0.45
NEUTRAL_HIP_TARGET = tuple(0.0 for _ in LEGS)

def clamp(value: float, lower: float, upper: float) -> float:
    return max(lower, min(upper, value))


def validate_hip_targets(targets: Sequence[float]) -> tuple[float, ...]:
    if len(targets) != len(LEGS):
        raise ValueError(f"expected {len(LEGS)} hip targets, got {len(targets)}")
    return tuple(clamp(float(value), -HIP_TARGET_LIMIT, HIP_TARGET_LIMIT) for value in targets)


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
