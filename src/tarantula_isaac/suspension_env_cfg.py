# Copyright (c) 2026 Tarantula project
# SPDX-License-Identifier: BSD-3-Clause
"""DirectRLEnv config for the Tarantula wheel-only Stage A locomotion task.

Action space = 6D:
  action[0:6] = wheel_{fl,fr,ml,mr,rl,rr}_joint velocity targets (LEGS order)
                 normalized ±1 -> ±action_scale_wheel_omega rad/s

Suspension is held at a neutral target in Isaac. Gazebo deployment uses
stand_suspension_hold for the same Stage A separation.

Observation space = 41D:
  projected_gravity_b(3) + root_ang_vel_b(3) + root_lin_vel_b(3)
  + susp_joint_pos(6) + susp_joint_vel(6) + wheel_joint_vel(6)
  + wheel_load(6)          <- wheel-axis F/T equivalent, normalized by nominal wheel load
  + cmd_vx(1) + cmd_wz(1) + prev_action(6)
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
    action_space = 6
    observation_space = 41  # see module docstring
    state_space = 0

    # action scaling: policy outputs ±1 -> physical units
    stand_susp_target = 0.0         # rad, Stage A neutral suspension target
    action_scale_wheel_omega = 3.0  # rad/s, per-wheel velocity target

    # driving: each episode samples a deployable cmd_vel-style command.
    wheel_radius = 0.13  # m, from tarantula_chassis.xacro
    command_vx_range = (-0.3, 0.3)  # m/s
    command_wz_range = (-0.4, 0.4)  # rad/s
    command_stop_prob = 0.2

    # Wheel-load observation: deployable equivalent is wheel-axis F/T magnitude.
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

    # Reward baseline: trimmed from common rough-terrain locomotion rewards
    # used in legged_gym-style policies.
    reward_tracking_lin_vel_weight = 1.5
    reward_tracking_lin_vel_sigma = 0.12
    reward_tracking_yaw_rate_weight = 0.5
    reward_tracking_yaw_rate_sigma = 0.15
    reward_orientation_weight = 0.4
    reward_orientation_sigma = 0.35
    reward_ang_vel_xy_weight = 0.03
    reward_lin_vel_z_weight = 0.5
    reward_action_rate_weight = 0.01
    reward_action_magnitude_weight = 0.002
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

    # Wheel-load sensors: simulation contact-force backend used as the Isaac-side
    # equivalent of deployable wheel-axis F/T sensors.
    wheel_loads: ContactSensorCfg = ContactSensorCfg(
        prim_path="/World/envs/env_.*/Robot/wheel_.*_link",
        update_period=0.0,
        history_length=1,
        track_air_time=False,
    )
