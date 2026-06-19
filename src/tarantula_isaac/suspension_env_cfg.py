# Copyright (c) 2026 Tarantula project
# SPDX-License-Identifier: BSD-3-Clause
"""DirectRLEnv config for the Tarantula active-suspension posture task.

Action space = 6D:
  action[0:6] = direct bounded hip/arm position targets, LEGS order

The PPO policy is an active suspension controller. Wheel commands stay under
the classical skid-steer baseline and are not policy outputs.

Observation space = 50D:
  projected_gravity_b(3) + root_ang_vel_b(3)
  + susp_joint_pos(6) + susp_joint_vel(6) + wheel_joint_vel(6)
  + wheel_force_b(18)      <- wheel-axis F/T equivalent, normalized by nominal wheel load
  + cmd_vx(1) + cmd_wz(1) + prev_action(6)
"""

from isaaclab.envs import DirectRLEnvCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sensors import ContactSensorCfg, ImuCfg
from isaaclab.sim import SimulationCfg
from isaaclab.utils import configclass

from tarantula_control.control_interfaces import WHEEL_RADIUS
from tarantula_control.motion_control import MotionControlConfig
from tarantula_control.vehicle_geometry import VEHICLE_GEOMETRY

from .robot import TARANTULA_CFG
from .shared_heightmap_terrain import SharedHeightmapTerrainImporterCfg, make_shared_heightmap_terrain_cfg


MOTION_DEFAULTS = MotionControlConfig()


@configclass
class TarantulaSuspensionEnvCfg(DirectRLEnvCfg):
    # env
    decimation = 4
    episode_length_s = 15.0
    action_space = 6
    observation_space = 50  # see module docstring
    state_space = 0

    # action scaling: policy outputs ±1 -> bounded hip targets
    stand_susp_target = 0.0         # rad, fallback neutral suspension target
    drive_scale = MOTION_DEFAULTS.drive_scale
    # Isaac PhysX skid-steer curve response needs a larger effective yaw track
    # than the Gazebo/Nav2 deployment controller. The external cmd_vel contract
    # stays identical; this is backend calibration for wheel target generation.
    yaw_track_scale = 1.6
    max_abs_wheel_omega = MOTION_DEFAULTS.max_abs_wheel_omega
    hip_action_target_limit = 0.25   # rad, conservative active-suspension clamp
    reward_hip_action_rate_weight = 0.02

    # driving: commands provide long enough motion for posture evaluation.
    wheel_radius = WHEEL_RADIUS
    command_vx_range = (-MOTION_DEFAULTS.max_abs_cmd_vx, MOTION_DEFAULTS.max_abs_cmd_vx)
    command_wz_range = (-MOTION_DEFAULTS.max_abs_cmd_wz, MOTION_DEFAULTS.max_abs_cmd_wz)
    command_stop_prob = 0.20
    command_straight_prob = 0.40
    command_turn_prob = 0.25
    command_curve_prob = 0.25
    command_mission_prob = 0.40
    command_min_abs_vx = 0.12
    command_min_abs_wz = 0.15
    command_resampling_enabled = True
    command_resampling_time_s = 3.0

    # Wheel-force observation: deployable equivalent is wheel-axis 3D F/T force.
    # nominal_wheel_load is used for observation normalization only.
    gravity = 9.81
    nominal_wheel_load = 23.1 * gravity / 6.0  # body(18) + 6*(arm 0.8 + wheel 1.5)

    # Domain randomization is opt-in by curriculum/profile. The baseline task
    # must first prove stable posture control on deterministic terrain; random
    # pushes are too easy to confuse with contact explosions during GUI smoke.
    friction_range = (0.3, 1.5)
    friction_num_buckets = 64
    body_mass_delta_range = (-3.0, 3.0)  # kg additive to base_link
    obs_noise_std = 0.02
    push_interval_steps = (10_000_000, 10_000_001)
    push_lin_vel_range = (0.0, 0.0)  # m/s x/y velocity delta

    # Reward baseline: active-suspension stability. Motion tracking belongs to
    # ROS2/Nav2 and the classical wheel controller, not to RL.
    reward_orientation_weight = 1.2
    reward_orientation_sigma = 0.25
    reward_roll_pitch_rate_weight = 0.08
    reward_contact_support_weight = 0.35
    reward_wheel_load_balance_weight = 0.12
    contact_force_threshold = 0.15
    contact_min_loaded_wheels = 4
    reward_lin_vel_z_weight = 0.5
    reward_stuck_weight = 0.25
    reward_action_rate_weight = 0.03
    reward_joint_limit_weight = 0.05
    reward_alive_bonus = 0.05
    reward_termination_penalty = 8.0

    # Termination baseline. The terrain importer also keeps reset origins away
    # from the heightmap edge via spawn_xy_margin, so bounds terminations measure
    # policy drift rather than edge-biased spawn placement.
    episode_tilt_limit = 0.75  # rad (~43 deg)
    episode_min_base_height = 0.05
    episode_max_base_height = 1.20
    episode_bounds_margin = 0.5 * VEHICLE_GEOMETRY.reference_length
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
