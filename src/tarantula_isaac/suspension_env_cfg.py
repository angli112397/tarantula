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
class CommandsCfg:
    """cmd_vel-style command sampling: probabilities, ranges, mission mix."""

    vx_range = (-MOTION_DEFAULTS.max_abs_cmd_vx, MOTION_DEFAULTS.max_abs_cmd_vx)
    wz_range = (-MOTION_DEFAULTS.max_abs_cmd_wz, MOTION_DEFAULTS.max_abs_cmd_wz)
    stop_prob = 0.20
    straight_prob = 0.40
    turn_prob = 0.25
    curve_prob = 0.25
    mission_prob = 0.40
    min_abs_vx = 0.12
    min_abs_wz = 0.15
    resampling_enabled = True
    resampling_time_s = 3.0

    # Pure-pursuit checkpoint chasing: sample pursuit_checkpoint_count random
    # waypoints inside the terrain bounds box and steer toward them in
    # sequence via curvature-based pursuit (wz recomputed every step from
    # heading error to the current checkpoint — closed-loop, unlike holding
    # an open-loop vx/wz for a fixed time/distance). Off by default like
    # push DR; opt in via command profile.
    pursuit_prob = 0.0
    pursuit_checkpoint_count = 3
    pursuit_arrival_radius = 0.3  # m
    # wz = clamp(pursuit_heading_gain * heading_error, -wz_abs_hi, wz_abs_hi),
    # heading_error = atan2(ly, lx) (signed bearing to the checkpoint in the
    # robot's body frame, full -pi..pi range). Deliberately NOT the textbook
    # curvature formula (kappa = 2*sin(alpha)/L): that weakens as the
    # checkpoint gets farther away (kappa ~ 1/L for a fixed angle) and, worse,
    # weakens again as the checkpoint swings toward directly behind the robot
    # (sin(alpha) -> 0 at alpha -> +-pi, exactly when the sharpest turn is
    # needed). A plain proportional law on the full signed angle saturates at
    # max turn rate in both of those cases instead of going quiet.
    pursuit_heading_gain = 1.5  # rad/s per rad of heading error


@configclass
class DomainRandCfg:
    """Domain randomization. Opt-in by curriculum/profile (see train_v5.py):
    the baseline task must first prove stable posture control on
    deterministic terrain — random pushes are too easy to confuse with
    contact explosions during GUI smoke, so push DR defaults to off.

    friction_range/hip_stiffness_scale_range/hip_damping_scale_range are
    widened/added specifically for the Isaac Lab (PhysX) -> Gazebo (ODE/DART)
    sim-to-sim gap: the two engines solve contact and joint drives with
    different solvers, so a friction or PD-gain number tuned to "feel right"
    in one has no guaranteed equivalent behavior in the other. Published
    cross-simulator work (e.g. the Isaac Gym -> Gazebo fall-recovery transfer
    in arXiv:2412.16924) randomizes ground friction over roughly [0.05, 1.75]
    and joint Kp/Kd over ±20% specifically to cover this gap; our previous
    friction_range=(0.3, 1.5) had no low-friction coverage (exactly where
    slip-prone behavior — e.g. the slope-climbing stuck case — is most
    sensitive to solver differences), and hip stiffness/damping were never
    randomized at all (fixed at robot.py's USD-baked 130.0/11.0).
    """

    friction_range = (0.05, 1.75)
    friction_num_buckets = 64
    body_mass_delta_range = (-3.0, 3.0)  # kg additive to base_link
    obs_noise_std = 0.02
    push_interval_steps = (10_000_000, 10_000_001)
    push_lin_vel_range = (0.0, 0.0)  # m/s x/y velocity delta
    hip_stiffness_scale_range = (0.8, 1.2)  # multiplicative, applied to susp_*_joint
    hip_damping_scale_range = (0.8, 1.2)


@configclass
class RewardsCfg:
    """Active-suspension stability reward weights. Motion tracking belongs to
    ROS2/Nav2 and the classical wheel controller, not to RL.
    """

    orientation_weight = 1.2
    orientation_sigma = 0.25
    roll_pitch_rate_weight = 0.08
    # loaded_wheels/6.0 (see suspension_env.py) is a continuous 0..1 fraction,
    # not a threshold -- a prior version only paid out above
    # contact_min_loaded_wheels=4, which had zero gradient between 0 and 4
    # loaded wheels and let the policy park 1-3 wheels in the air (swinging
    # for balance) for free as long as the other >=4 stayed planted. Weight
    # bumped alongside the formula fix since the old 0.35 was tuned against
    # the old (much smaller, threshold-gated) typical term value.
    contact_support_weight = 0.45
    wheel_load_balance_weight = 0.12
    contact_force_threshold = 0.15
    lin_vel_z_weight = 0.5
    stuck_weight = 0.25
    # Was split into action_rate_weight (0.03) + hip_action_rate_weight
    # (0.02) penalizing the *exact same* quantity twice under different
    # names -- action_space is hip-only (6D), so "action rate" and "hip
    # action rate" were never different things. Consolidated into one term;
    # weight roughly doubles the old combined 0.05, per legged-locomotion RL
    # convention (ANYmal/legged_gym-style reward sets weight action-rate
    # penalties meaningfully against the tracking/orientation term, not as
    # an afterthought) -- the old combined weight was ~40x smaller than
    # orientation_weight and visibly under-suppressed flat-ground jitter.
    hip_action_rate_weight = 0.08
    # New: standard legged-locomotion reward term we were missing -- penalizes
    # the *physical* hip joint velocity (rad/s, from sim state), not just the
    # commanded action's frame-to-frame delta. action_rate only sees the
    # network's output target; if the actuator overshoots/oscillates getting
    # there (confirmed possible empirically via step-response testing), that
    # physical jitter is invisible to action_rate alone.
    joint_vel_weight = 0.01
    # New: "default pose"/nominal-posture regularization, standard in
    # ANYmal/legged_gym-style reward sets (pull joints toward a homing
    # position unless the task needs otherwise). Pulls susp_joint_pos toward
    # stand_susp_target=0 directly. This is the term that's actually missing
    # to suppress a *slow, large-amplitude* cyclic lift (e.g. front-left +
    # rear-left up, mid-right down, hold, then settle to flat, repeat, seen
    # even on flat ground) -- hip_action_rate/joint_vel are frame-to-frame
    # (rate/velocity) penalties, structurally blind to a slow deliberate
    # sweep into a large deviation: each individual step's delta and
    # velocity stay small even though the cumulative excursion is large.
    # This term penalizes the absolute deviation regardless of how slowly
    # it got there.
    joint_pos_weight = 0.08
    alive_bonus = 0.05
    termination_penalty = 8.0
    # Reward-only, deliberately NOT an observation: true per-wheel slip needs
    # ground-truth body velocity (root_lin_vel_b), which the module docstring
    # already excludes from the actor observation since it isn't available
    # at Gazebo/hardware deployment. Penalizing it in the reward still trains
    # the policy to avoid slip-inducing postures without requiring the
    # policy to *observe* a signal it won't have outside simulation.
    slip_weight = 0.15


@configclass
class TerminationsCfg:
    """Episode termination thresholds. The terrain importer also keeps reset
    origins away from the heightmap edge via spawn_xy_margin, so bounds
    terminations measure policy drift rather than edge-biased spawn placement.

    bounds_margin ends the episode (cleanly, with termination_penalty) the
    moment root_pos_w crosses into this margin -- a safe, instant stop well
    before the robot could reach the literal mesh edge. gazebo_pursuit_eval.py
    mirrors this exact value for checkpoint sampling (DEFAULT_MARGIN), but
    deliberately does NOT need an Isaac-side equivalent of its surround_copies
    terrain tiling: that fix exists because Gazebo's eval keeps running after
    crossing the nominal boundary (no termination check) and the heightmap
    mesh has a real edge a few meters out, so a slow controller can drive
    over it into open space. Here, the termination check runs every step and
    fires this margin before the robot can physically reach that edge --
    there's no failure mode to tile around.
    """

    tilt_limit = 0.75  # rad (~43 deg)
    min_base_height = 0.05
    max_base_height = 1.20
    bounds_margin = 0.5 * VEHICLE_GEOMETRY.reference_length
    max_lin_vel = 5.0
    max_ang_vel = 8.0


@configclass
class TarantulaSuspensionEnvCfg(DirectRLEnvCfg):
    # env
    decimation = 4
    # 15s, then 45s, 240s, were all short of the eval-side demo scale
    # (gazebo_pursuit_eval.py runs typically need 180-350s to complete a
    # 3-5 checkpoint chase). 300s covers most of that range while still
    # resetting often enough for per-reset terrain/domain-rand draws to stay
    # varied -- the policy is a stateless MLP with no notion of elapsed
    # episode time, so going much longer (e.g. matching a full real
    # deployment's duration) buys no representational benefit and only
    # makes resets rarer.
    episode_length_s = 300.0
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
    # Closed-loop yaw-rate correction (see motion_control.py's
    # MotionControlConfig docstring for why this exists: on Gazebo's
    # mesh-direct-collision terrain, this open-loop differential alone
    # measured ~0.00 rad/s actual chassis yaw rate for a sustained command --
    # closed-loop control was required, not optional). Sourced from the same
    # MotionControlConfig defaults the Gazebo deployment controller uses
    # (not retyped here) so training experiences the same control law/gains
    # it will be deployed under, rather than a purely open-loop approximation
    # of it -- narrows the sim-to-sim gap the existing domain randomization
    # has to cover, instead of substituting for it.
    yaw_rate_kp = MOTION_DEFAULTS.yaw_rate_kp
    yaw_rate_ki = MOTION_DEFAULTS.yaw_rate_ki
    yaw_integral_limit = MOTION_DEFAULTS.yaw_integral_limit
    max_abs_wheel_omega = MOTION_DEFAULTS.max_abs_wheel_omega
    # rad. Real URDF hip joint limit (HIP_TARGET_LIMIT, suspension_core.py) is
    # 0.45 -- 0.25 left a lot of unused mechanical range, making the active
    # suspension visibly underuse its authority on rough terrain. 0.35 keeps
    # a 0.10 rad margin below the hard physical limit.
    hip_action_target_limit = 0.35

    # driving: commands provide long enough motion for posture evaluation.
    wheel_radius = WHEEL_RADIUS
    commands: CommandsCfg = CommandsCfg()

    # Wheel-force observation: deployable equivalent is wheel-axis 3D F/T force.
    # nominal_wheel_load is used for observation normalization only -- derived
    # from VEHICLE_GEOMETRY.total_mass (URDF-sourced, not a hand-tuned guess).
    # An empirical Isaac Lab settle-state check (2026-06-19) confirmed actual
    # per-wheel contact force averages ~total_mass/6*g; see
    # control_interfaces.py's NOMINAL_WHEEL_LOAD for the full story (it used
    # to be a stale 23.1 kg constant, ~30% low). Keep these two constants
    # equal -- training and deployment must normalize wheel_force_b the same way.
    gravity = 9.81
    nominal_wheel_load = VEHICLE_GEOMETRY.total_mass * gravity / 6.0

    domain_rand: DomainRandCfg = DomainRandCfg()
    rewards: RewardsCfg = RewardsCfg()
    terminations: TerminationsCfg = TerminationsCfg()

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
