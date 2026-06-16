# Copyright (c) 2026 Tarantula project
# SPDX-License-Identifier: BSD-3-Clause
"""DirectRLEnv config for the tarantula active-suspension task (M7 v5).

Action space = 12D (6 susp joint angles + 6 wheel velocities, all independent).
  action[0:6]  = susp_{fl,fr,ml,mr,rl,rr}_joint position targets (LEGS order)
                 normalized ±1 -> ±action_scale_susp rad; clamped ±0.6 rad
  action[6:12] = wheel_{fl,fr,ml,mr,rl,rr}_joint velocity targets (LEGS order)
                 normalized ±1 -> ±action_scale_wheel_omega rad/s

No kinematic mapping: the policy learns the geometry from experience.

Observation space = 47D:
  projected_gravity_b(3) + root_ang_vel_b(3) + root_lin_vel_b(3)
  + susp_joint_pos(6) + susp_joint_vel(6) + wheel_joint_vel(6)
  + wheel_in_contact(6)    <- ft_wheel magnitude > contact_force_threshold
  + move_cmd(1) + heading_rate_cmd(1) + prev_action(12)
"""

import isaaclab.sim as sim_utils
from isaaclab.envs import DirectRLEnvCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sensors import ContactSensorCfg, ImuCfg
from isaaclab.sim import SimulationCfg
from isaaclab.terrains import TerrainImporterCfg
from isaaclab.utils import configclass

from .robot import TARANTULA_CFG
from .terrains import TARANTULA_TERRAIN_CFG


@configclass
class TarantulaSuspensionEnvCfg(DirectRLEnvCfg):
    # env
    decimation = 4
    episode_length_s = 15.0
    action_space = 12   # susp(6) + wheel(6), all per-joint independent
    observation_space = 47  # see module docstring
    state_space = 0

    # action scaling: policy outputs ±1 -> physical units
    action_scale_susp = 0.5         # rad, direct joint angle (±1 -> ±0.5 rad; URDF limit ±0.6)
    action_scale_wheel_omega = 3.0  # rad/s, per-wheel velocity target

    # driving: each episode samples a direction-only command pair
    wheel_radius = 0.12  # m, from tarantula_chassis.xacro
    target_speed_range = (0.1, 0.3)  # m/s when move_cmd=1
    move_prob = 0.7
    heading_rate_range = (-0.2, 0.2)  # rad/s

    # contact detection: matches /ft_wheel threshold in active_suspension.py
    contact_force_threshold = 5.0  # N, wheel net-force magnitude -> in_contact boolean

    # domain randomization
    friction_range = (0.3, 1.5)
    friction_num_buckets = 64
    body_mass_delta_range = (-3.0, 3.0)  # kg additive to base_link
    obs_noise_std = 0.02
    push_interval_steps = (150, 300)
    push_lin_vel_range = (-0.5, 0.5)  # m/s x/y delta

    # reward weights
    reward_attitude_weight = 0.1
    reward_attitude_sigma = 0.25    # rad
    reward_velocity_weight = 1.5
    reward_velocity_sigma_move = 0.1
    reward_velocity_sigma_stop = 0.05
    reward_yaw_rate_weight = 0.5
    reward_yaw_rate_sigma = 0.1
    reward_ang_vel_xy_weight = 0.01
    reward_lin_vel_z_weight = 0.5
    reward_action_rate_weight = 0.01
    reward_contact_weight = 0.3     # fraction of wheels in contact (0-1)
    reward_alive_bonus = 0.1
    reward_fall_penalty = 10.0

    episode_tilt_limit = 0.6  # rad (~34 deg)

    # simulation
    sim: SimulationCfg = SimulationCfg(dt=1 / 120, render_interval=decimation)

    # terrain
    terrain: TerrainImporterCfg = TerrainImporterCfg(
        prim_path="/World/ground",
        terrain_type="generator",
        terrain_generator=TARANTULA_TERRAIN_CFG,
        max_init_terrain_level=1,
        collision_group=-1,
        physics_material=sim_utils.RigidBodyMaterialCfg(
            friction_combine_mode="multiply",
            restitution_combine_mode="multiply",
            static_friction=1.0,
            dynamic_friction=1.0,
        ),
        debug_vis=False,
    )

    # scene
    scene: InteractiveSceneCfg = InteractiveSceneCfg(num_envs=16, env_spacing=8.0, replicate_physics=True)

    # robot
    robot: object = TARANTULA_CFG.replace(prim_path="/World/envs/env_.*/Robot")

    # sensors
    imu: ImuCfg = ImuCfg(prim_path="/World/envs/env_.*/Robot/base_link")

    # wheel contact sensors: prim regex matches fl/fr/ml/mr/rl/rr alphabetically = LEGS order
    wheel_contacts: ContactSensorCfg = ContactSensorCfg(
        prim_path="/World/envs/env_.*/Robot/wheel_.*_link",
        update_period=0.0,
        history_length=1,
        track_air_time=False,
    )
