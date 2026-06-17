"""Stop-turn-drive skid-steer baseline and optional structured RL compensation.

The deployable baseline first shapes normal ``cmd_vel`` input into simple
execution primitives, then maps those primitives to per-wheel skid-steer
velocity targets. RL is intentionally limited to bounded corrections to the
effective track scale and left/right drive scales, so it learns terrain/contact
compensation instead of relearning planar kinematics.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from enum import Enum

import numpy as np

from .control_interfaces import (
    CmdVelLimiter,
    DRIVE_SCALE_DELTA_LIMIT,
    MAX_ABS_WHEEL_OMEGA,
    RIGHT_LEGS,
    TRACK_SCALE_DELTA_LIMIT,
    YAW_AUTHORITY_MULTIPLIER,
    clamp_abs,
    clamp,
    skid_steer_wheel_speeds,
    wheel_force_normalized,
)
from .suspension_core import LEGS


STAGE_A_OBSERVATION_DIM = 47
STAGE_A_ACTION_DIM = 3
WHEEL_TARGET_DIM = 6

STAGE_A_OBSERVATION_LAYOUT = (
    ("projected_gravity_b", 3, "IMU orientation"),
    ("root_ang_vel_b", 3, "IMU angular velocity"),
    ("susp_joint_pos", 6, "joint_states hip/arm positions"),
    ("susp_joint_vel", 6, "joint_states hip/arm velocities"),
    ("wheel_joint_vel", 6, "joint_states wheel velocities"),
    ("wheel_force", 18, "wheel-end F/T force vector, normalized, LEGS order"),
    ("cmd_vx", 1, "shaped execution forward velocity"),
    ("cmd_wz", 1, "shaped execution yaw rate"),
    ("prev_action", 3, "previous RL structured compensation action"),
)


class MotionMode(str, Enum):
    STOP = "stop"
    DRIVE = "drive"
    TURN = "turn"


@dataclass(frozen=True)
class MotionControlConfig:
    max_abs_cmd_vx: float = 0.3
    max_abs_cmd_wz: float = 0.4
    max_abs_wheel_omega: float = MAX_ABS_WHEEL_OMEGA
    pure_turn_track_scale: float = 3.0
    track_scale_delta_limit: float = TRACK_SCALE_DELTA_LIMIT
    drive_scale_delta_limit: float = DRIVE_SCALE_DELTA_LIMIT
    yaw_rate_kp: float = 0.0
    yaw_rate_ki: float = 0.0
    yaw_integral_limit: float = 0.8
    max_wheel_accel: float = 12.0
    pure_turn_forward_bias: float = 0.0
    pure_turn_vx_deadband: float = 0.03
    turn_enter_wz: float = 0.08
    turn_exit_wz: float = 0.04


@dataclass(frozen=True)
class MotionCommand:
    vx: float
    wz: float


@dataclass(frozen=True)
class StructuredCompensation:
    """Bounded RL correction around calibrated skid-steer kinematics."""

    track_scale_delta: float = 0.0
    left_drive_scale_delta: float = 0.0
    right_drive_scale_delta: float = 0.0


class CommandShaper:
    """Shape application cmd_vel into deployable motion primitives.

    The shaper owns behavior policy such as stop-turn-drive. The wheel
    controller below remains a plain skid-steer velocity mapper.
    """

    def __init__(self, config: MotionControlConfig | None = None):
        self.config = config or MotionControlConfig()
        self._mode = MotionMode.STOP

    @property
    def mode(self) -> MotionMode:
        return self._mode

    def update_config(self, **kwargs) -> None:
        self.config = replace(self.config, **kwargs)

    def shape(self, command: MotionCommand) -> MotionCommand:
        abs_wz = abs(command.wz)
        if self._mode == MotionMode.TURN:
            if abs_wz <= self.config.turn_exit_wz:
                self._mode = MotionMode.DRIVE if abs(command.vx) >= 1.0e-4 else MotionMode.STOP
        elif abs_wz >= self.config.turn_enter_wz:
            self._mode = MotionMode.TURN
        elif abs(command.vx) >= 1.0e-4:
            self._mode = MotionMode.DRIVE
        else:
            self._mode = MotionMode.STOP

        if self._mode == MotionMode.TURN:
            return MotionCommand(0.0, command.wz)
        if self._mode == MotionMode.DRIVE:
            return MotionCommand(command.vx, 0.0)
        return MotionCommand(0.0, 0.0)


class SkidSteerMotionController:
    """Deployable cmd_vel to six-wheel velocity controller."""

    def __init__(self, config: MotionControlConfig | None = None):
        self.config = config or MotionControlConfig()
        self._limiter = CmdVelLimiter(
            max_abs_vx=self.config.max_abs_cmd_vx,
            max_abs_wz=self.config.max_abs_cmd_wz,
        )
        self._yaw_error_integral = 0.0
        self._last_wheel_targets = [0.0] * WHEEL_TARGET_DIM

    def limit_command(self, vx: float, wz: float) -> MotionCommand:
        vx_limited, wz_limited = self._limiter.clamp(vx, wz)
        return MotionCommand(vx_limited, wz_limited)

    def reset_feedback(self) -> None:
        self._yaw_error_integral = 0.0

    def reset_output_filter(self) -> None:
        self._last_wheel_targets = [0.0] * WHEEL_TARGET_DIM

    def update_config(self, **kwargs) -> None:
        self.config = replace(self.config, **kwargs)

    def _rate_limit_wheel_targets(self, targets: list[float], dt: float | None) -> list[float]:
        if dt is None or dt <= 0.0 or self.config.max_wheel_accel <= 0.0:
            self._last_wheel_targets = list(targets)
            return targets
        max_delta = self.config.max_wheel_accel * float(dt)
        limited = [
            prev + clamp(target - prev, -max_delta, max_delta)
            for prev, target in zip(self._last_wheel_targets, targets)
        ]
        self._last_wheel_targets = limited
        return limited

    def _effective_vx_for_turn(self, command: MotionCommand) -> float:
        if (
            self.config.pure_turn_forward_bias <= 0.0
            or abs(command.vx) > self.config.pure_turn_vx_deadband
            or abs(command.wz) < 1.0e-4
        ):
            return command.vx
        yaw_fraction = min(abs(command.wz) / max(self.config.max_abs_cmd_wz, 1.0e-6), 1.0)
        return command.vx + self.config.pure_turn_forward_bias * yaw_fraction

    def scheduled_track_scale(self, command: MotionCommand) -> float:
        if abs(command.wz) < 1.0e-4:
            return 1.0
        return self.config.pure_turn_track_scale

    def _bounded_compensation(
        self,
        compensation_action: np.ndarray | list[float] | tuple[float, ...] | StructuredCompensation | None,
    ) -> StructuredCompensation:
        if compensation_action is None:
            return StructuredCompensation()
        if isinstance(compensation_action, StructuredCompensation):
            raw = np.asarray(
                [
                    compensation_action.track_scale_delta,
                    compensation_action.left_drive_scale_delta,
                    compensation_action.right_drive_scale_delta,
                ],
                dtype=np.float32,
            )
        else:
            raw = np.asarray(compensation_action, dtype=np.float32).reshape(-1)
        if raw.shape[0] != STAGE_A_ACTION_DIM:
            raise ValueError(f"Stage A compensation action must have 3 values, got {raw.shape[0]}")
        raw = np.clip(raw, -1.0, 1.0)
        return StructuredCompensation(
            track_scale_delta=float(raw[0]) * float(self.config.track_scale_delta_limit),
            left_drive_scale_delta=float(raw[1]) * float(self.config.drive_scale_delta_limit),
            right_drive_scale_delta=float(raw[2]) * float(self.config.drive_scale_delta_limit),
        )

    def _mode_gated_compensation(
        self,
        execution_command: MotionCommand,
        compensation: StructuredCompensation,
    ) -> StructuredCompensation:
        if abs(execution_command.vx) < 1.0e-4 and abs(execution_command.wz) < 1.0e-4:
            return StructuredCompensation()
        track_delta = compensation.track_scale_delta if abs(execution_command.wz) >= 1.0e-4 else 0.0
        left_delta = compensation.left_drive_scale_delta if abs(execution_command.vx) >= 1.0e-4 else 0.0
        right_delta = compensation.right_drive_scale_delta if abs(execution_command.vx) >= 1.0e-4 else 0.0
        return StructuredCompensation(
            track_scale_delta=track_delta,
            left_drive_scale_delta=left_delta,
            right_drive_scale_delta=right_delta,
        )

    def wheel_targets(
        self,
        command: MotionCommand,
        *,
        compensation_action: np.ndarray | list[float] | tuple[float, ...] | StructuredCompensation | None = None,
        measured_wz: float | None = None,
        dt: float | None = None,
    ) -> list[float]:
        effective_vx = self._effective_vx_for_turn(command)
        compensation = self._mode_gated_compensation(
            command,
            self._bounded_compensation(compensation_action),
        )
        base_track_scale = self.scheduled_track_scale(command)
        track_scale = max(0.1, base_track_scale * (1.0 + compensation.track_scale_delta))
        left_scale = max(0.0, 1.0 + compensation.left_drive_scale_delta)
        right_scale = max(0.0, 1.0 + compensation.right_drive_scale_delta)
        wheel = [
            clamp_abs(omega, self.config.max_abs_wheel_omega)
            for omega in skid_steer_wheel_speeds(
                effective_vx,
                command.wz,
                track_scale=track_scale,
                left_scale=left_scale,
                right_scale=right_scale,
            )
        ]
        if measured_wz is None:
            return wheel
        if abs(command.wz) < 1.0e-4:
            self.reset_feedback()
            return wheel

        yaw_error = command.wz - float(measured_wz)
        if dt is not None and dt > 0.0 and self.config.yaw_rate_ki != 0.0:
            self._yaw_error_integral = clamp(
                self._yaw_error_integral + yaw_error * float(dt),
                -self.config.yaw_integral_limit,
                self.config.yaw_integral_limit,
            )
        yaw_correction = (
            self.config.yaw_rate_kp * yaw_error
            + self.config.yaw_rate_ki * self._yaw_error_integral
        )
        corrected = []
        for leg, omega in zip(LEGS, wheel):
            semantic_sign = 1.0 if leg in RIGHT_LEGS else -1.0
            corrected.append(
                clamp_abs(
                    omega + semantic_sign * yaw_correction,
                    self.config.max_abs_wheel_omega,
                )
            )
        return corrected

    def compensated_wheel_targets(
        self,
        command: MotionCommand,
        compensation_action: np.ndarray | list[float] | tuple[float, ...] | StructuredCompensation | None,
        *,
        measured_wz: float | None = None,
        dt: float | None = None,
    ) -> list[float]:
        compensated = self.wheel_targets(
            command,
            compensation_action=compensation_action,
            measured_wz=measured_wz,
            dt=dt,
        )
        return self._rate_limit_wheel_targets(compensated, dt)


def build_stage_a_observation(
    *,
    projected_gravity_b: tuple[float, float, float],
    root_ang_vel_b: tuple[float, float, float],
    susp_joint_pos: dict[str, float],
    susp_joint_vel: dict[str, float],
    wheel_joint_vel: dict[str, float],
    wheel_force: dict[str, tuple[float, float, float]],
    command: MotionCommand,
    prev_action: np.ndarray,
) -> np.ndarray:
    """Build the deployable 47D Stage A observation in the Isaac training order."""

    obs_values = (
        list(projected_gravity_b)
        + list(root_ang_vel_b)
        + [susp_joint_pos[leg] for leg in LEGS]
        + [susp_joint_vel[leg] for leg in LEGS]
        + [wheel_joint_vel[leg] for leg in LEGS]
        + [component for leg in LEGS for component in wheel_force_normalized(wheel_force[leg])]
        + [command.vx, command.wz]
        + list(np.asarray(prev_action, dtype=np.float32).reshape(-1))
    )
    obs = np.asarray(obs_values, dtype=np.float32)
    if obs.shape[0] != STAGE_A_OBSERVATION_DIM:
        raise ValueError(f"Stage A observation must be {STAGE_A_OBSERVATION_DIM}D, got {obs.shape[0]}")
    return obs
