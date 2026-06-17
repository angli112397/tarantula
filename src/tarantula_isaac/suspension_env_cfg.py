# Copyright (c) 2026 Tarantula project
# SPDX-License-Identifier: BSD-3-Clause
"""DirectRLEnv config for the Tarantula Stage B wheel + hip residual task.

Action space = 9D by default:
  action[0] = bounded effective-track scale correction
  action[1] = bounded left-drive scale correction
  action[2] = bounded right-drive scale correction
  action[3:9] = direct bounded hip/arm position targets, LEGS order

The first three dimensions preserve the classical skid-steer residual contract.
The final six dimensions let the policy command bounded hip position targets
directly. The PPO network size is intentionally unchanged from Stage A.

Observation space = 53D:
  projected_gravity_b(3) + root_ang_vel_b(3)
  + susp_joint_pos(6) + susp_joint_vel(6) + wheel_joint_vel(6)
  + wheel_force_b(18)      <- wheel-axis F/T equivalent, normalized by nominal wheel load
  + cmd_vx(1) + cmd_wz(1) + prev_action(9)
"""

from isaaclab.envs import DirectRLEnvCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sensors import ContactSensorCfg, ImuCfg
from isaaclab.sim import SimulationCfg
from isaaclab.utils import configclass

from .robot import TARANTULA_CFG
from .shared_heightmap_terrain import SharedHeightmapTerrainImporterCfg, make_shared_heightmap_terrain_cfg


@configclass
class TarantulaSuspensionEnvCfg(DirectRLEnvCfg):
    # env
    decimation = 4
    episode_length_s = 15.0
    action_space = 9
    observation_space = 53  # see module docstring
    state_space = 0

    # action scaling: policy outputs ±1 -> bounded structured compensation
    stand_susp_target = 0.0         # rad, fallback neutral suspension target
    pure_turn_track_scale = 3.0     # effective track scale for near-zero-vx turns
    turn_enter_wz = 0.08            # rad/s, stop-turn-drive enter threshold
    turn_exit_wz = 0.04             # rad/s, stop-turn-drive exit threshold
    track_scale_delta_limit = 0.30  # fractional correction around base track scale
    drive_scale_delta_limit = 0.20  # fractional correction around left/right drive scale
    max_abs_wheel_omega = 6.0       # rad/s, final per-wheel velocity target limit
    hip_action_enabled = True
    hip_action_target_limit = 0.30   # rad, direct hip target clamp for Stage B baseline
    reward_hip_action_rate_weight = 0.015
    reward_hip_action_magnitude_weight = 0.01

    # driving: Stage B trains the shaped stop-turn-drive baseline plus hip targets.
    wheel_radius = 0.13  # m, from tarantula_common.xacro
    command_vx_range = (-0.3, 0.3)  # m/s
    command_wz_range = (-0.4, 0.4)  # rad/s
    command_stop_prob = 0.20
    command_straight_prob = 0.40
    command_pure_turn_prob = 0.40
    command_min_abs_vx = 0.12
    command_min_abs_wz = 0.15
    command_resampling_enabled = True
    command_resampling_time_s = 3.0

    # Wheel-force observation: deployable equivalent is wheel-axis 3D F/T force.
    # nominal_wheel_load is used for observation normalization only.
    gravity = 9.81
    nominal_wheel_load = 23.1 * gravity / 6.0  # body(18) + 6*(arm 0.8 + wheel 1.5)

    # domain randomization
    friction_range = (0.3, 1.5)
    friction_num_buckets = 64
    body_mass_delta_range = (-3.0, 3.0)  # kg additive to base_link
    obs_noise_std = 0.02
    push_interval_steps = (150, 300)
    push_lin_vel_range = (-0.5, 0.5)  # m/s x/y delta

    # Reward baseline: trimmed rough-terrain locomotion rewards plus
    # yaw-focused structured-compensation terms.
    reward_tracking_lin_vel_weight = 1.5
    reward_tracking_lin_vel_sigma = 0.12
    reward_tracking_yaw_rate_weight = 1.2
    reward_tracking_yaw_rate_sigma = 0.12
    reward_delta_lin_vel_weight = 0.45
    reward_delta_yaw_rate_weight = 0.55
    baseline_vx_error_floor = 0.06
    baseline_vx_error_fraction = 0.45
    baseline_wz_error_floor = 0.12
    baseline_wz_error_fraction = 0.65
    reward_yaw_sign_weight = 0.25
    reward_pure_turn_drift_weight = 0.35
    reward_orientation_weight = 0.4
    reward_orientation_sigma = 0.35
    reward_ang_vel_xy_weight = 0.03
    reward_lin_vel_z_weight = 0.5
    reward_lateral_vel_weight = 0.08
    reward_stuck_weight = 0.25
    reward_action_rate_weight = 0.03
    reward_action_magnitude_weight = 0.02
    reward_action_saturation_weight = 0.50
    action_saturation_soft_limit = 0.65
    reward_wheel_target_saturation_weight = 0.35
    wheel_target_saturation_soft_limit = 0.85
    reward_joint_limit_weight = 0.05
    reward_alive_bonus = 0.05
    reward_termination_penalty = 8.0

    # Termination baseline. The terrain importer also keeps reset origins away
    # from the heightmap edge via spawn_xy_margin, so bounds terminations measure
    # policy drift rather than edge-biased spawn placement.
    episode_tilt_limit = 0.75  # rad (~43 deg)
    episode_min_base_height = 0.05
    episode_max_base_height = 1.20
    episode_bounds_margin = 0.50
    episode_max_lin_vel = 5.0
    episode_max_ang_vel = 8.0

    # simulation
    sim: SimulationCfg = SimulationCfg(dt=1 / 120, render_interval=decimation)

    # terrain
    terrain: SharedHeightmapTerrainImporterCfg = make_shared_heightmap_terrain_cfg()

    # scene
    scene: InteractiveSceneCfg = InteractiveSceneCfg(num_envs=16, env_spacing=8.0, replicate_physics=True)

    # robot
    robot: object = TARANTULA_CFG.replace(prim_path="/World/envs/env_.*/Robot")

    # sensors
    imu: ImuCfg = ImuCfg(prim_path="/World/envs/env_.*/Robot/base_link")

    # Wheel-force sensors: simulation contact-force backend used as the Isaac-side
    # equivalent of deployable wheel-axis F/T sensors.
    wheel_loads: ContactSensorCfg = ContactSensorCfg(
        prim_path="/World/envs/env_.*/Robot/wheel_.*_link",
        update_period=0.0,
        history_length=1,
        track_air_time=False,
    )
