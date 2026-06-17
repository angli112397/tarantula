# Copyright (c) 2026 Tarantula project
# SPDX-License-Identifier: BSD-3-Clause
"""DirectRLEnv for the Tarantula structured-compensation Stage A task.

The policy's 3D action is a bounded correction around analytic skid-steer
targets:
  action[0] adjusts effective track scale
  action[1] adjusts left drive scale
  action[2] adjusts right drive scale

Suspension is held at a neutral target by the env. Gazebo deployment keeps hip
posture on the v2 trajectory controller and uses this structured compensation
actor for Stage A.

Wheel force signal (18D continuous, force vector normalized by nominal wheel
load) is added to the observation, matching the deployable /ft_wheel/{leg} F/T
signal used in the Gazebo deployment. Geometry contact booleans and simulator
root linear velocity are intentionally not part of the actor observation.

The raw command interface is cmd_vel-style: cmd_vx (m/s) and cmd_wz (rad/s).
The policy observes and is rewarded against the shaped execution command,
matching Gazebo's default stop-turn-drive baseline.

Action space = 3D. Obs = 47D. See suspension_env_cfg.py for full layout.
"""

from __future__ import annotations

import torch

import isaaclab.sim as sim_utils
from isaaclab.assets import Articulation
from isaaclab.envs import DirectRLEnv
from isaaclab.sensors import ContactSensor, Imu
from isaaclab.utils.math import quat_rotate_inverse, sample_uniform

from tarantula_control.control_interfaces import (
    EFFECTIVE_TRACK,
    WHEEL_DIRECTION,
    WHEEL_RADIUS,
)
from tarantula_control.suspension_core import LEGS

from .suspension_env_cfg import TarantulaSuspensionEnvCfg


REWARD_KEYS = (
    "tracking_lin_vel",
    "tracking_yaw_rate",
    "yaw_sign",
    "orientation",
    "lin_vel_z",
    "lateral_vel",
    "stuck",
    "pure_turn_drift",
    "ang_vel_xy",
    "action_rate",
    "action_magnitude",
    "action_saturation",
    "joint_limit",
    "alive",
    "termination",
    "total",
)

METRIC_KEYS = (
    "abs_vx_error",
    "abs_wz_error",
    "vx_error_sq",
    "wz_error_sq",
    "roll_abs",
    "pitch_abs",
    "action_saturation_rate",
    "wheel_target_saturation_rate",
    "cmd_vx_abs",
    "cmd_wz_abs",
)


def _quat_roll_pitch(quat_w: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Batched roll/pitch from wxyz quaternions (N, 4)."""
    w, x, y, z = quat_w.unbind(-1)
    roll = torch.atan2(2.0 * (w * x + y * z), 1.0 - 2.0 * (x * x + y * y))
    pitch = torch.asin(torch.clamp(2.0 * (w * y - z * x), -1.0, 1.0))
    return roll, pitch


class TarantulaSuspensionEnv(DirectRLEnv):
    cfg: TarantulaSuspensionEnvCfg

    def __init__(self, cfg: TarantulaSuspensionEnvCfg, render_mode: str | None = None, **kwargs):
        super().__init__(cfg, render_mode, **kwargs)

        # Friction domain randomization (PhysX tensor API).
        lo, hi = self.cfg.friction_range
        ranges = torch.tensor([[lo, hi], [lo, hi]], device="cpu")
        self._friction_buckets = sample_uniform(
            ranges[:, 0], ranges[:, 1], (self.cfg.friction_num_buckets, 2), device="cpu"
        )
        self._friction_buckets[:, 1] = torch.min(self._friction_buckets[:, 0], self._friction_buckets[:, 1])
        self._num_shapes = self._robot.root_physx_view.max_shapes
        self._randomize_friction(torch.arange(self.num_envs))

        # Mass domain randomization
        self._default_masses = self._robot.root_physx_view.get_masses().clone()
        self._base_body_idx = list(self._robot.body_names).index("base_link")

        # Push perturbation countdown
        lo, hi = self.cfg.push_interval_steps
        self._push_countdown = torch.randint(lo, hi, (self.num_envs,), device=self.device)

        # Joint ID lookup (LEGS order = fl/fr/ml/mr/rl/rr)
        self._susp_joint_ids, _ = self._robot.find_joints(
            [f"susp_{leg}_joint" for leg in LEGS], preserve_order=True
        )
        self._wheel_joint_ids, _ = self._robot.find_joints(
            [f"wheel_{leg}_joint" for leg in LEGS], preserve_order=True
        )

        # Stage A: 3D structured compensation. Suspension is held at neutral target.
        self._actions = torch.zeros(self.num_envs, 3, device=self.device)
        self._previous_actions = torch.zeros(self.num_envs, 3, device=self.device)
        self._cmd_vx = torch.zeros(self.num_envs, device=self.device)
        self._cmd_wz = torch.zeros(self.num_envs, device=self.device)
        self._exec_cmd_vx = torch.zeros(self.num_envs, device=self.device)
        self._exec_cmd_wz = torch.zeros(self.num_envs, device=self.device)
        self._turn_mode = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        self._command_time_left = torch.zeros(self.num_envs, device=self.device)
        self._episode_sums = {
            key: torch.zeros(self.num_envs, dtype=torch.float, device=self.device)
            for key in REWARD_KEYS
        }
        self._metric_sums = {
            key: torch.zeros(self.num_envs, dtype=torch.float, device=self.device)
            for key in METRIC_KEYS
        }
        self._last_wheel_target = torch.zeros(self.num_envs, 6, device=self.device)

    def _setup_scene(self):
        self._robot = Articulation(self.cfg.robot)
        self.scene.articulations["robot"] = self._robot
        self._imu = Imu(self.cfg.imu)
        self.scene.sensors["imu"] = self._imu
        # Wheel-load sensors: fl/fr/ml/mr/rl/rr alphabetical = LEGS order
        self._wheel_load_sensors = ContactSensor(self.cfg.wheel_loads)
        self.scene.sensors["wheel_loads"] = self._wheel_load_sensors

        self.cfg.terrain.num_envs = self.scene.cfg.num_envs
        self.cfg.terrain.env_spacing = self.scene.cfg.env_spacing
        self._terrain = self.cfg.terrain.class_type(self.cfg.terrain)

        self.scene.clone_environments(copy_from_source=False)
        if self.device == "cpu":
            self.scene.filter_collisions(global_prim_paths=[self.cfg.terrain.prim_path])

        light_cfg = sim_utils.DomeLightCfg(intensity=2000.0, color=(0.75, 0.75, 0.75))
        light_cfg.func("/World/Light", light_cfg)

    def _pre_physics_step(self, actions: torch.Tensor) -> None:
        self._actions = torch.clamp(actions, -1.0, 1.0)
        self._update_execution_commands()

        # Push perturbations
        self._push_countdown -= 1
        push_envs = (self._push_countdown <= 0).nonzero(as_tuple=False).flatten()
        if len(push_envs) > 0:
            plo, phi = self.cfg.push_lin_vel_range
            n = len(push_envs)
            vel_delta = torch.zeros(n, 6, device=self.device)
            vel_delta[:, 0] = (torch.rand(n, device=self.device) * 2 - 1) * phi
            vel_delta[:, 1] = (torch.rand(n, device=self.device) * 2 - 1) * phi * 0.6
            current_vel = torch.cat([
                self._robot.data.root_lin_vel_w[push_envs],
                self._robot.data.root_ang_vel_w[push_envs],
            ], dim=-1)
            self._robot.write_root_velocity_to_sim(current_vel + vel_delta, push_envs)
            slo, shi = self.cfg.push_interval_steps
            self._push_countdown[push_envs] = torch.randint(slo, shi, (n,), device=self.device)

    def _apply_action(self) -> None:
        susp = torch.full(
            (self.num_envs, 6),
            float(self.cfg.stand_susp_target),
            device=self.device,
        )
        self._robot.set_joint_position_target(susp.clamp(-0.6, 0.6), joint_ids=self._susp_joint_ids)

        drive_active = torch.abs(self._exec_cmd_vx) >= 1.0e-4
        yaw_active = torch.abs(self._exec_cmd_wz) >= 1.0e-4
        track_delta = torch.where(
            yaw_active,
            self._actions[:, 0] * float(self.cfg.track_scale_delta_limit),
            torch.zeros_like(self._actions[:, 0]),
        )
        left_delta = torch.where(
            drive_active,
            self._actions[:, 1] * float(self.cfg.drive_scale_delta_limit),
            torch.zeros_like(self._actions[:, 1]),
        )
        right_delta = torch.where(
            drive_active,
            self._actions[:, 2] * float(self.cfg.drive_scale_delta_limit),
            torch.zeros_like(self._actions[:, 2]),
        )

        base_track_scale = torch.where(
            torch.abs(self._exec_cmd_wz) >= 1.0e-4,
            torch.full_like(self._exec_cmd_wz, float(self.cfg.pure_turn_track_scale)),
            torch.ones_like(self._exec_cmd_wz),
        )
        turn_track = EFFECTIVE_TRACK * base_track_scale * (1.0 + track_delta)
        left = (self._exec_cmd_vx - 0.5 * turn_track * self._exec_cmd_wz) / WHEEL_RADIUS
        right = (self._exec_cmd_vx + 0.5 * turn_track * self._exec_cmd_wz) / WHEEL_RADIUS
        left = left * torch.clamp(1.0 + left_delta, min=0.0)
        right = right * torch.clamp(1.0 + right_delta, min=0.0)
        base_wheel = torch.stack((left, right, left, right, left, right), dim=-1)
        direction = torch.tensor([WHEEL_DIRECTION[leg] for leg in LEGS], device=self.device)
        wheel = base_wheel * direction
        wheel = torch.clamp(
            wheel,
            -float(self.cfg.max_abs_wheel_omega),
            float(self.cfg.max_abs_wheel_omega),
        )
        self._last_wheel_target = wheel.detach()
        self._robot.set_joint_velocity_target(wheel, joint_ids=self._wheel_joint_ids)

    def _get_observations(self) -> dict:
        self._advance_command_curriculum()

        susp_joint_pos = self._robot.data.joint_pos[:, self._susp_joint_ids]
        susp_joint_vel = self._robot.data.joint_vel[:, self._susp_joint_ids]
        wheel_joint_vel = self._robot.data.joint_vel[:, self._wheel_joint_ids]

        # Wheel force: net contact force vector in base frame, normalized.
        contact_forces = self._wheel_load_sensors.data.net_forces_w_history[:, 0]  # (N, 6, 3)
        base_quat = self._robot.data.root_quat_w
        wheel_force_b = quat_rotate_inverse(
            base_quat.unsqueeze(1).expand(-1, contact_forces.shape[1], -1).reshape(-1, 4),
            contact_forces.reshape(-1, 3),
        ).reshape(self.num_envs, 18)
        wheel_force_b = torch.clamp(wheel_force_b / self.cfg.nominal_wheel_load, -3.0, 3.0)

        obs = torch.cat(
            (
                self._robot.data.projected_gravity_b,   # 3
                self._robot.data.root_ang_vel_b,        # 3
                susp_joint_pos,                         # 6
                susp_joint_vel,                         # 6
                wheel_joint_vel,                        # 6
                wheel_force_b,                          # 18
                self._exec_cmd_vx.unsqueeze(-1),        # 1
                self._exec_cmd_wz.unsqueeze(-1),        # 1
                self._previous_actions,                 # 3
            ),
            dim=-1,
        )  # total: 47
        self._previous_actions = self._actions.clone()
        if self.cfg.obs_noise_std > 0.0:
            obs = obs + torch.randn_like(obs) * self.cfg.obs_noise_std
        return {"policy": obs}

    def _advance_command_curriculum(self) -> None:
        if not self.cfg.command_resampling_enabled:
            return
        dt = float(self.cfg.sim.dt) * float(self.cfg.decimation)
        self._command_time_left -= dt
        resample_envs = (self._command_time_left <= 0.0).nonzero(as_tuple=False).flatten()
        if len(resample_envs) > 0:
            self._sample_commands(resample_envs)

    def _update_execution_commands(self) -> None:
        abs_wz = torch.abs(self._cmd_wz)
        enter = abs_wz >= float(self.cfg.turn_enter_wz)
        exit_turn = abs_wz <= float(self.cfg.turn_exit_wz)
        self._turn_mode = torch.where(self._turn_mode, ~exit_turn, enter)
        drive_active = (~self._turn_mode) & (torch.abs(self._cmd_vx) >= 1.0e-4)
        self._exec_cmd_vx = torch.where(
            drive_active,
            self._cmd_vx,
            torch.zeros_like(self._cmd_vx),
        )
        self._exec_cmd_wz = torch.where(
            self._turn_mode,
            self._cmd_wz,
            torch.zeros_like(self._cmd_wz),
        )

    def _termination_terms(self) -> dict[str, torch.Tensor]:
        roll, pitch = _quat_roll_pitch(self._imu.data.quat_w)
        root_pos_w = self._robot.data.root_pos_w
        root_lin_vel_w = self._robot.data.root_lin_vel_w
        root_ang_vel_w = self._robot.data.root_ang_vel_w

        x_min, x_max, y_min, y_max = self._terrain.terrain_bounds
        out_of_bounds = (
            (root_pos_w[:, 0] < x_min + self.cfg.episode_bounds_margin)
            | (root_pos_w[:, 0] > x_max - self.cfg.episode_bounds_margin)
            | (root_pos_w[:, 1] < y_min + self.cfg.episode_bounds_margin)
            | (root_pos_w[:, 1] > y_max - self.cfg.episode_bounds_margin)
        )
        bad_height = (
            (root_pos_w[:, 2] < self.cfg.episode_min_base_height)
            | (root_pos_w[:, 2] > self.cfg.episode_max_base_height)
        )
        bad_velocity = (
            torch.linalg.norm(root_lin_vel_w, dim=-1) > self.cfg.episode_max_lin_vel
        ) | (
            torch.linalg.norm(root_ang_vel_w, dim=-1) > self.cfg.episode_max_ang_vel
        )
        bad_tilt = (roll**2 + pitch**2) > (self.cfg.episode_tilt_limit**2)
        non_finite = (
            ~torch.isfinite(root_pos_w).all(dim=-1)
            | ~torch.isfinite(root_lin_vel_w).all(dim=-1)
            | ~torch.isfinite(root_ang_vel_w).all(dim=-1)
            | ~torch.isfinite(self._actions).all(dim=-1)
        )
        return {
            "tilt": bad_tilt,
            "height": bad_height,
            "velocity": bad_velocity,
            "bounds": out_of_bounds,
            "non_finite": non_finite,
        }

    def _termination_flags(self) -> torch.Tensor:
        terms = self._termination_terms()
        return torch.stack(tuple(terms.values()), dim=0).any(dim=0)

    def _reward_terms(self) -> dict[str, torch.Tensor]:
        roll, pitch = _quat_roll_pitch(self._imu.data.quat_w)
        action_rate = torch.sum(torch.square(self._actions - self._previous_actions), dim=-1)
        action_magnitude = torch.sum(torch.square(self._actions), dim=-1)
        action_saturation_margin = torch.clamp(
            torch.abs(self._actions) - self.cfg.action_saturation_soft_limit,
            min=0.0,
        )
        action_saturation = torch.sum(torch.square(action_saturation_margin), dim=-1)

        ang_vel_b = self._robot.data.root_ang_vel_b
        lin_vel_b = self._robot.data.root_lin_vel_b

        v_x = lin_vel_b[:, 0]
        target_v_x = self._exec_cmd_vx
        tracking_lin_vel = torch.exp(
            -torch.square(v_x - target_v_x) / (self.cfg.reward_tracking_lin_vel_sigma**2)
        )

        yaw_rate_err = ang_vel_b[:, 2] - self._exec_cmd_wz
        tracking_yaw_rate = torch.exp(
            -torch.square(yaw_rate_err) / (self.cfg.reward_tracking_yaw_rate_sigma**2)
        )
        yaw_active = torch.abs(self._exec_cmd_wz) >= self.cfg.command_min_abs_wz
        drive_active = torch.abs(self._exec_cmd_vx) >= self.cfg.command_min_abs_vx
        pure_turn_active = yaw_active & ~drive_active
        yaw_dir = torch.sign(self._exec_cmd_wz)
        yaw_progress = torch.clamp(ang_vel_b[:, 2] * yaw_dir / torch.clamp(torch.abs(self._exec_cmd_wz), min=1.0e-3), 0.0, 1.0)
        yaw_sign_reward = torch.where(yaw_active, yaw_progress, torch.ones_like(yaw_progress))

        orientation_err = torch.square(roll) + torch.square(pitch)
        orientation_reward = torch.exp(
            -orientation_err / (self.cfg.reward_orientation_sigma**2)
        )

        lateral_vel_penalty = torch.square(lin_vel_b[:, 1])
        min_expected_vx = 0.45 * torch.abs(self._exec_cmd_vx)
        stuck_penalty = torch.where(
            drive_active,
            torch.square(torch.clamp(min_expected_vx - torch.abs(v_x), min=0.0)),
            torch.zeros_like(v_x),
        )
        pure_turn_drift_penalty = torch.where(
            pure_turn_active,
            torch.square(v_x) + lateral_vel_penalty,
            torch.zeros_like(v_x),
        )

        susp_joint_pos = self._robot.data.joint_pos[:, self._susp_joint_ids]
        soft_limit_margin = torch.clamp(torch.abs(susp_joint_pos) - 0.52, min=0.0)
        joint_limit_penalty = torch.sum(torch.square(soft_limit_margin), dim=-1)

        terminated = self._termination_flags()

        return {
            "tracking_lin_vel": self.cfg.reward_tracking_lin_vel_weight * tracking_lin_vel,
            "tracking_yaw_rate": self.cfg.reward_tracking_yaw_rate_weight * tracking_yaw_rate,
            "yaw_sign": self.cfg.reward_yaw_sign_weight * yaw_sign_reward,
            "orientation": self.cfg.reward_orientation_weight * orientation_reward,
            "ang_vel_xy": -self.cfg.reward_ang_vel_xy_weight * torch.sum(torch.square(ang_vel_b[:, :2]), dim=-1),
            "lin_vel_z": -self.cfg.reward_lin_vel_z_weight * torch.square(lin_vel_b[:, 2]),
            "lateral_vel": -self.cfg.reward_lateral_vel_weight * lateral_vel_penalty,
            "stuck": -self.cfg.reward_stuck_weight * stuck_penalty,
            "pure_turn_drift": -self.cfg.reward_pure_turn_drift_weight * pure_turn_drift_penalty,
            "action_rate": -self.cfg.reward_action_rate_weight * action_rate,
            "action_magnitude": -self.cfg.reward_action_magnitude_weight * action_magnitude,
            "action_saturation": -self.cfg.reward_action_saturation_weight * action_saturation,
            "joint_limit": -self.cfg.reward_joint_limit_weight * joint_limit_penalty,
            "alive": torch.full((self.num_envs,), self.cfg.reward_alive_bonus, device=self.device),
            "termination": -self.cfg.reward_termination_penalty * terminated.float(),
        }

    def _get_rewards(self) -> torch.Tensor:
        reward_terms = self._reward_terms()
        reward = torch.sum(torch.stack(tuple(reward_terms.values()), dim=0), dim=0)
        for key, value in reward_terms.items():
            self._episode_sums[key] += value
        self._accumulate_metrics()
        self._episode_sums["total"] += reward
        return reward

    def _accumulate_metrics(self) -> None:
        roll, pitch = _quat_roll_pitch(self._imu.data.quat_w)
        lin_vel_b = self._robot.data.root_lin_vel_b
        ang_vel_b = self._robot.data.root_ang_vel_b
        vx_error = lin_vel_b[:, 0] - self._exec_cmd_vx
        wz_error = ang_vel_b[:, 2] - self._exec_cmd_wz
        action_saturation_rate = (torch.abs(self._actions) >= 0.98).float().mean(dim=-1)
        wheel_limit = max(float(self.cfg.max_abs_wheel_omega), 1.0e-6)
        wheel_saturation_rate = (torch.abs(self._last_wheel_target) >= 0.98 * wheel_limit).float().mean(dim=-1)

        self._metric_sums["abs_vx_error"] += torch.abs(vx_error)
        self._metric_sums["abs_wz_error"] += torch.abs(wz_error)
        self._metric_sums["vx_error_sq"] += torch.square(vx_error)
        self._metric_sums["wz_error_sq"] += torch.square(wz_error)
        self._metric_sums["roll_abs"] += torch.abs(roll)
        self._metric_sums["pitch_abs"] += torch.abs(pitch)
        self._metric_sums["action_saturation_rate"] += action_saturation_rate
        self._metric_sums["wheel_target_saturation_rate"] += wheel_saturation_rate
        self._metric_sums["cmd_vx_abs"] += torch.abs(self._exec_cmd_vx)
        self._metric_sums["cmd_wz_abs"] += torch.abs(self._exec_cmd_wz)

    def _get_dones(self) -> tuple[torch.Tensor, torch.Tensor]:
        terminated = self._termination_flags()
        time_out = self.episode_length_buf >= self.max_episode_length - 1
        return terminated, time_out

    def _randomize_mass(self, env_ids: torch.Tensor) -> None:
        lo, hi = self.cfg.body_mass_delta_range
        n = len(env_ids)
        delta = torch.rand(n) * (hi - lo) + lo
        masses = self._robot.root_physx_view.get_masses().clone()
        env_ids_cpu = env_ids.cpu()
        masses[env_ids_cpu, self._base_body_idx] = (
            self._default_masses[env_ids_cpu, self._base_body_idx] + delta
        )
        self._robot.root_physx_view.set_masses(masses, env_ids_cpu)

    def _randomize_friction(self, env_ids: torch.Tensor) -> None:
        env_ids_cpu = env_ids.cpu()
        bucket_ids = torch.randint(0, self._friction_buckets.shape[0], (len(env_ids_cpu), self._num_shapes), device="cpu")
        samples = self._friction_buckets[bucket_ids]
        materials = self._robot.root_physx_view.get_material_properties()
        materials[env_ids_cpu, :, 0] = samples[:, :, 0]
        materials[env_ids_cpu, :, 1] = samples[:, :, 1]
        self._robot.root_physx_view.set_material_properties(materials, env_ids_cpu)

    def _reset_idx(self, env_ids: torch.Tensor | None):
        if env_ids is None:
            env_ids = self._robot._ALL_INDICES

        extras = {}
        denom = max(float(self.max_episode_length), 1.0)
        for key in self._episode_sums.keys():
            extras[f"Episode_Reward/{key}"] = torch.mean(self._episode_sums[key][env_ids]) / denom
            self._episode_sums[key][env_ids] = 0.0
        for key in self._metric_sums.keys():
            value = torch.mean(self._metric_sums[key][env_ids]) / denom
            if key.endswith("_sq"):
                metric_key = key[:-3] + "_rms"
                extras[f"Episode_Metric/{metric_key}"] = torch.sqrt(torch.clamp(value, min=0.0))
            else:
                extras[f"Episode_Metric/{key}"] = value
            self._metric_sums[key][env_ids] = 0.0

        # The first reset happens before the robot is placed on generated terrain
        # origins, so termination terms from zero-length episodes are not useful.
        logged_env_ids = env_ids[self.episode_length_buf[env_ids] > 0]
        term_terms = self._termination_terms()
        for key, value in term_terms.items():
            extras[f"Episode_Termination/{key}"] = torch.count_nonzero(value[logged_env_ids]).item()
        extras["Episode_Termination/time_out"] = torch.count_nonzero(self.reset_time_outs[logged_env_ids]).item()
        self.extras["log"] = extras

        super()._reset_idx(env_ids)

        self._randomize_friction(env_ids)
        self._randomize_mass(env_ids)
        lo, hi = self.cfg.push_interval_steps
        self._push_countdown[env_ids] = torch.randint(lo, hi, (len(env_ids),), device=self.device)

        joint_pos = self._robot.data.default_joint_pos[env_ids]
        joint_vel = self._robot.data.default_joint_vel[env_ids]
        default_root_state = self._robot.data.default_root_state[env_ids]
        default_root_state[:, :3] += self._terrain.env_origins[env_ids]
        self._robot.write_root_pose_to_sim(default_root_state[:, :7], env_ids)
        self._robot.write_root_velocity_to_sim(default_root_state[:, 7:], env_ids)
        self._robot.write_joint_state_to_sim(joint_pos, joint_vel, None, env_ids)

        self._actions[env_ids] = 0.0
        self._previous_actions[env_ids] = 0.0
        self._last_wheel_target[env_ids] = 0.0
        self._turn_mode[env_ids] = False
        self._exec_cmd_vx[env_ids] = 0.0
        self._exec_cmd_wz[env_ids] = 0.0

        self._sample_commands(env_ids)

    def _sample_abs_with_sign(self, n: int, lo: float, hi: float) -> torch.Tensor:
        mag = torch.rand(n, device=self.device) * (hi - lo) + lo
        sign = torch.where(torch.rand(n, device=self.device) < 0.5, -1.0, 1.0)
        return mag * sign

    def _sample_commands(self, env_ids: torch.Tensor) -> None:
        n = len(env_ids)
        r = torch.rand(n, device=self.device)
        total = max(
            float(
                self.cfg.command_stop_prob
                + self.cfg.command_straight_prob
                + self.cfg.command_pure_turn_prob
            ),
            1.0e-6,
        )
        stop_end = float(self.cfg.command_stop_prob) / total
        straight_end = stop_end + float(self.cfg.command_straight_prob) / total
        pure_turn_end = straight_end + float(self.cfg.command_pure_turn_prob) / total

        self._cmd_vx[env_ids] = 0.0
        self._cmd_wz[env_ids] = 0.0
        self._command_time_left[env_ids] = float(self.cfg.command_resampling_time_s)

        vx_abs_hi = max(abs(self.cfg.command_vx_range[0]), abs(self.cfg.command_vx_range[1]))
        wz_abs_hi = max(abs(self.cfg.command_wz_range[0]), abs(self.cfg.command_wz_range[1]))

        straight_mask = (r >= stop_end) & (r < straight_end)
        straight_ids = env_ids[straight_mask]
        if len(straight_ids) > 0:
            self._cmd_vx[straight_ids] = self._sample_abs_with_sign(
                len(straight_ids), self.cfg.command_min_abs_vx, vx_abs_hi
            )

        turn_mask = (r >= straight_end) & (r < pure_turn_end)
        turn_ids = env_ids[turn_mask]
        if len(turn_ids) > 0:
            self._cmd_wz[turn_ids] = self._sample_abs_with_sign(
                len(turn_ids), self.cfg.command_min_abs_wz, wz_abs_hi
            )
        self._update_execution_commands()
