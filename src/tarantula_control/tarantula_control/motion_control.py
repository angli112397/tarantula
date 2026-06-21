"""Skid-steer motion baseline.

Maps ``cmd_vel`` to per-wheel velocity targets via a calibrated skid-steer
differential. ``cmd_vx`` and ``cmd_wz`` are allowed to act at the same time;
Nav2-style curved motion is part of the baseline contract.

RL does not alter wheel commands. The learned posture policy is a separate
active-suspension controller running in its own node.
"""

from __future__ import annotations

from dataclasses import dataclass, replace

import numpy as np

from .control_interfaces import (
    CmdVelLimiter,
    MAX_ABS_WHEEL_OMEGA,
    RIGHT_LEGS,
    clamp_abs,
    clamp,
    skid_steer_wheel_speeds,
    wheel_force_normalized,
)
from .suspension_core import LEGS


POSTURE_OBSERVATION_DIM = 56
POSTURE_ACTION_DIM = 6
WHEEL_TARGET_DIM = 6

POSTURE_OBSERVATION_LAYOUT = (
    ("projected_gravity_b", 3, "IMU orientation"),
    ("root_ang_vel_b", 3, "IMU angular velocity"),
    ("susp_joint_pos", 6, "joint_states hip/arm positions"),
    ("susp_joint_vel", 6, "joint_states hip/arm velocities"),
    ("wheel_joint_vel", 6, "joint_states wheel velocities"),
    ("wheel_force", 18, "wheel-end F/T force vector, normalized, LEGS order"),
    # Per-leg EMA "contact uptime" (~1s window, contact_uptime_alpha above) --
    # added because the contact_support/contact_uptime reward depends on
    # this exact recursive state, and a memoryless policy/critic that never
    # sees it can't tell "I've held solid contact for a while" apart from
    # "I just barely touched down" from a single instantaneous F/T reading
    # alone. Computed identically on both sides: same threshold/alpha
    # (MotionControlConfig), same raw (unfiltered) F/T input as wheel_force
    # above -- see control_interfaces.ema_contact_uptime.
    ("contact_uptime", 6, "per-leg EMA contact persistence, LEGS order"),
    ("cmd_vx", 1, "limited command forward velocity"),
    ("cmd_wz", 1, "limited command yaw rate"),
    ("prev_action", 6, "previous hip target action"),
)


@dataclass(frozen=True)
class MotionControlConfig:
    max_abs_cmd_vx: float = 0.3
    max_abs_cmd_wz: float = 0.4
    max_abs_wheel_omega: float = MAX_ABS_WHEEL_OMEGA
    drive_scale: float = 1.1532
    yaw_track_scale: float = 0.7287
    # This Gazebo/ODE value is deliberately NOT what Isaac Lab trains with --
    # suspension_env_cfg.py's TarantulaSuspensionEnvCfg.yaw_track_scale=1.6
    # is a separate, intentionally larger constant compensating for PhysX's
    # different skid-steer curve response. Don't "fix" one to match the
    # other; see that docstring for why they're allowed to diverge.
    # drive_scale/yaw_track_scale were calibrated open-loop against the flat
    # floor under world.sdf's skid-steer baseline. On a raw heightmap-mesh
    # collision surface (e.g. generated/terrains/*/world.sdf used by the
    # Isaac Lab terrain exporter -- mesh IS the collision, no flat floor) that
    # same open-loop differential measured ~0.00 rad/s actual chassis yaw rate
    # for a sustained -0.4 rad/s command (confirmed via /imu/data angular_velocity.z
    # while holding cmd_wz constant): the wheels reach the commanded speed
    # differential but the contact can't turn the chassis at all. Nonzero
    # kp/ki close that loop with the already-measured yaw rate; verified on
    # flat floor too (steady-state error drops from ~0.03 rad/s to <0.01,
    # no oscillation) so this is a strict improvement, not a tradeoff.
    yaw_rate_kp: float = 3.0
    yaw_rate_ki: float = 1.0
    yaw_integral_limit: float = 0.8
    max_wheel_accel: float = 12.0
    # EMA coefficient for the wheel-F/T observation slice (not the reward,
    # which stays on the raw instantaneous reading -- see suspension_env.py).
    # Inspired by the planetary-rover active-suspension RL literature
    # (arXiv:2606.06790), which explicitly low-pass filters F/T sensor data
    # before it reaches the policy. alpha=0.3 at this project's 30 Hz control
    # rate gives a time constant of ~0.1s (~1.7 Hz cutoff): enough to knock
    # down per-step contact-force noise without lagging real load-transfer
    # events, which happen on a slower timescale than a single control tick.
    wheel_force_filter_alpha: float = 0.3
    # Normalized F/T magnitude above which a wheel counts as "loaded" --
    # shared by both the contact_support/contact_uptime reward computation
    # (suspension_env.py) and the deployable contact_uptime observation
    # feature below (this used to live only in Isaac's RewardsCfg, which was
    # fine while it was reward-only, but it's no longer reward-only once a
    # deployable obs feature needs the identical loaded/unloaded judgement).
    contact_force_threshold: float = 0.15
    # EMA coefficient for the per-leg "contact uptime" feature (both reward
    # and -- this is the part that was missing -- observation, see
    # POSTURE_OBSERVATION_LAYOUT below). 1/30 at 30 Hz gives a ~1s time
    # constant: long enough that a brief touch-and-go barely moves it (so the
    # policy/critic can tell "sustained contact" apart from "just blipped
    # on"), short enough to track real terrain-driven load transfer within
    # an episode.
    contact_uptime_alpha: float = 1.0 / 30.0


@dataclass(frozen=True)
class MotionCommand:
    vx: float
    wz: float


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

    def scheduled_track_scale(self, command: MotionCommand) -> float:
        if abs(command.wz) < 1.0e-4:
            return 1.0
        return self.config.yaw_track_scale

    def wheel_targets(
        self,
        command: MotionCommand,
        *,
        measured_wz: float | None = None,
        dt: float | None = None,
    ) -> list[float]:
        effective_vx = command.vx * self.config.drive_scale
        wheel = [
            clamp_abs(omega, self.config.max_abs_wheel_omega)
            for omega in skid_steer_wheel_speeds(
                effective_vx,
                command.wz,
                track_scale=self.scheduled_track_scale(command),
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

    def filtered_wheel_targets(
        self,
        command: MotionCommand,
        *,
        measured_wz: float | None = None,
        dt: float | None = None,
    ) -> list[float]:
        targets = self.wheel_targets(command, measured_wz=measured_wz, dt=dt)
        return self._rate_limit_wheel_targets(targets, dt)


def build_posture_observation(
    *,
    projected_gravity_b: tuple[float, float, float],
    root_ang_vel_b: tuple[float, float, float],
    susp_joint_pos: dict[str, float],
    susp_joint_vel: dict[str, float],
    wheel_joint_vel: dict[str, float],
    wheel_force: dict[str, tuple[float, float, float]],
    contact_uptime: dict[str, float],
    command: MotionCommand,
    prev_action: np.ndarray,
) -> np.ndarray:
    """Build the deployable active-suspension observation in Isaac/Gazebo order."""

    prev_action_values = np.asarray(prev_action, dtype=np.float32).reshape(-1)
    if prev_action_values.shape[0] != POSTURE_ACTION_DIM:
        raise ValueError(f"prev_action must be {POSTURE_ACTION_DIM}D, got {prev_action_values.shape[0]}")
    obs_values = (
        list(projected_gravity_b)
        + list(root_ang_vel_b)
        + [susp_joint_pos[leg] for leg in LEGS]
        + [susp_joint_vel[leg] for leg in LEGS]
        + [wheel_joint_vel[leg] for leg in LEGS]
        + [component for leg in LEGS for component in wheel_force_normalized(wheel_force[leg])]
        + [contact_uptime[leg] for leg in LEGS]
        + [command.vx, command.wz]
        + list(prev_action_values)
    )
    obs = np.asarray(obs_values, dtype=np.float32)
    if obs.shape[0] != POSTURE_OBSERVATION_DIM:
        raise ValueError(f"observation must be {POSTURE_OBSERVATION_DIM}D, got {obs.shape[0]}")
    return obs
