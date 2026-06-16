# Copyright (c) 2026 Tarantula project
# SPDX-License-Identifier: BSD-3-Clause
"""DirectRLEnv for the tarantula active-suspension task (M7 v5).

v5 removes the kinematic mapping entirely. The policy's 12D action directly
drives joints:
  action[0:6]  -> susp_*_joint position targets (per-wheel independent, ±0.5 rad)
  action[6:12] -> wheel_*_joint velocity targets (per-wheel independent, ±3 rad/s)

Wheel contact signal (6D boolean, net force > 5 N) is added to the observation,
matching the /ft_wheel/{leg} signal used in the Gazebo deployment.

Action space = 12D. Obs = 47D. See suspension_env_cfg.py for full layout.
"""

from __future__ import annotations

import torch

import isaaclab.sim as sim_utils
from isaaclab.assets import Articulation
from isaaclab.envs import DirectRLEnv
from isaaclab.sensors import ContactSensor, Imu
from isaaclab.utils.math import sample_uniform

from tarantula_control.suspension_core import LEGS

from .suspension_env_cfg import TarantulaSuspensionEnvCfg


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

        # Friction domain randomization (PhysX tensor API, same as v4)
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

        # v5: 12D action (6 susp + 6 wheel), no kinematic mapping constants needed
        self._actions = torch.zeros(self.num_envs, 12, device=self.device)
        self._previous_actions = torch.zeros(self.num_envs, 12, device=self.device)
        self._move_cmd = torch.zeros(self.num_envs, device=self.device)
        self._heading_rate_cmd = torch.zeros(self.num_envs, device=self.device)

    def _setup_scene(self):
        self._robot = Articulation(self.cfg.robot)
        self.scene.articulations["robot"] = self._robot
        self._imu = Imu(self.cfg.imu)
        self.scene.sensors["imu"] = self._imu
        # Wheel contact sensors: fl/fr/ml/mr/rl/rr alphabetical = LEGS order
        self._wheel_contacts = ContactSensor(self.cfg.wheel_contacts)
        self.scene.sensors["wheel_contacts"] = self._wheel_contacts

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
        # v5: direct per-joint control, no kinematic mapping
        susp = self._actions[:, 0:6] * self.cfg.action_scale_susp
        self._robot.set_joint_position_target(susp.clamp(-0.6, 0.6), joint_ids=self._susp_joint_ids)

        wheel = self._actions[:, 6:12] * self.cfg.action_scale_wheel_omega
        self._robot.set_joint_velocity_target(wheel, joint_ids=self._wheel_joint_ids)

    def _get_observations(self) -> dict:
        susp_joint_pos = self._robot.data.joint_pos[:, self._susp_joint_ids]
        susp_joint_vel = self._robot.data.joint_vel[:, self._susp_joint_ids]
        wheel_joint_vel = self._robot.data.joint_vel[:, self._wheel_joint_ids]

        # Wheel contact: net force magnitude > threshold -> boolean (N, 6)
        contact_forces = self._wheel_contacts.data.net_forces_w_history[:, 0]  # (N, 6, 3)
        in_contact = (contact_forces.norm(dim=-1) > self.cfg.contact_force_threshold).float()

        obs = torch.cat(
            (
                self._robot.data.projected_gravity_b,   # 3
                self._robot.data.root_ang_vel_b,        # 3
                self._robot.data.root_lin_vel_b,        # 3
                susp_joint_pos,                         # 6
                susp_joint_vel,                         # 6
                wheel_joint_vel,                        # 6
                in_contact,                             # 6  <- new
                self._move_cmd.unsqueeze(-1),           # 1
                self._heading_rate_cmd.unsqueeze(-1),   # 1
                self._actions,                          # 12 (prev_action after this step)
            ),
            dim=-1,
        )  # total: 47
        self._previous_actions = self._actions.clone()
        if self.cfg.obs_noise_std > 0.0:
            obs = obs + torch.randn_like(obs) * self.cfg.obs_noise_std
        return {"policy": obs}

    def _get_rewards(self) -> torch.Tensor:
        roll, pitch = _quat_roll_pitch(self._imu.data.quat_w)
        action_rate = torch.sum(torch.square(self._actions - self._previous_actions), dim=-1)

        tilt_sq = torch.square(roll) + torch.square(pitch)
        attitude_reward = torch.exp(-tilt_sq / (self.cfg.reward_attitude_sigma**2))

        ang_vel_b = self._robot.data.root_ang_vel_b
        lin_vel_b = self._robot.data.root_lin_vel_b

        lo, hi = self.cfg.target_speed_range
        v_x = lin_vel_b[:, 0]
        band_err = torch.clamp(lo - v_x, min=0.0) + torch.clamp(v_x - hi, min=0.0)
        stop_err = torch.abs(v_x)
        move = self._move_cmd
        vel_err = move * band_err + (1.0 - move) * stop_err
        vel_sigma = move * self.cfg.reward_velocity_sigma_move + (1.0 - move) * self.cfg.reward_velocity_sigma_stop
        velocity_reward = torch.exp(-torch.square(vel_err / vel_sigma))

        yaw_rate_err = ang_vel_b[:, 2] - self._heading_rate_cmd
        yaw_rate_reward = torch.exp(-torch.square(yaw_rate_err / self.cfg.reward_yaw_rate_sigma))

        # Contact reward: fraction of wheels touching the ground (0-1)
        contact_forces = self._wheel_contacts.data.net_forces_w_history[:, 0]  # (N, 6, 3)
        in_contact = (contact_forces.norm(dim=-1) > self.cfg.contact_force_threshold).float()
        contact_frac = in_contact.mean(dim=-1)

        terminated = (torch.square(roll) + torch.square(pitch)) > (self.cfg.episode_tilt_limit ** 2)

        reward = (
            self.cfg.reward_attitude_weight   * attitude_reward
            + self.cfg.reward_velocity_weight * velocity_reward
            + self.cfg.reward_yaw_rate_weight * yaw_rate_reward
            + self.cfg.reward_contact_weight  * contact_frac
            - self.cfg.reward_ang_vel_xy_weight * torch.sum(torch.square(ang_vel_b[:, :2]), dim=-1)
            - self.cfg.reward_lin_vel_z_weight  * torch.square(lin_vel_b[:, 2])
            - self.cfg.reward_action_rate_weight * action_rate
            + self.cfg.reward_alive_bonus
            - self.cfg.reward_fall_penalty * terminated.float()
        )
        return reward

    def _get_dones(self) -> tuple[torch.Tensor, torch.Tensor]:
        roll, pitch = _quat_roll_pitch(self._imu.data.quat_w)
        terminated = (roll**2 + pitch**2) > (self.cfg.episode_tilt_limit**2)
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

        self._move_cmd[env_ids] = (torch.rand(len(env_ids), device=self.device) < self.cfg.move_prob).float()
        hr_lo, hr_hi = self.cfg.heading_rate_range
        self._heading_rate_cmd[env_ids] = torch.rand(len(env_ids), device=self.device) * (hr_hi - hr_lo) + hr_lo
