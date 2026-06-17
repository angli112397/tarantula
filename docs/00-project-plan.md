# Tarantula Baseline Plan

日期：2026-06-17

本文是当前开发的 source of truth。仓库只保留当前 baseline 相关文档。

## Current Baseline

当前 baseline 链路：

```text
tarantula_terrain.generate
  -> generated/terrains/flat_smoke/42 and gazebo_demo/42
  -> tarantula_v2.urdf.xacro
  -> Gazebo direct hip/wheel GUI acceptance
  -> classical skid-steer motion controller
  -> tarantula_isaac.SharedHeightmapTerrainImporter
  -> train_v5.py staged structured-compensation RL
  -> export_weights_v5.py actor .npz
  -> optional Gazebo structured RL deployment
```

当前目标是让 Gazebo、Isaac Lab、RL I/O、模型参数和文档保持同一个 baseline。
主运动控制必须是可解释、可单测的传统 skid-steer controller；RL 只作为可关闭的
bounded structured compensation，用于复杂地形下的 yaw/slip/slope/stuck 修正。

当前开发纪律：

- 先冻结一个可重复的 baseline model，再训练 RL；
- 先在 Gazebo GUI 和脚本中验证物理/接口，再判断策略质量；
- 排查要分层，但不能无限细碎：每轮只改变一个层级的主变量，并通过固定 gate 判断是否进入下一层；
- 失败策略不推动模型重构，除非 direct physics baseline 和 classical controller 都已经通过。

## Terrain Contract

Active presets:

- `gazebo_demo`
  - finite Gazebo inspection terrain;
  - generated heightmap mesh plus boundary walls;
  - no hand-authored internal obstacle models;
  - height limit `[-0.08, 0.14] m`.
- `rl_curriculum`
  - legged_gym-style terrain grid;
  - `metadata.env_origins` defines Isaac reset origins;
  - height limit `[-0.08, 0.18] m`.

Generated output:

```text
generated/terrains/<preset>/<seed>/
  height.npy
  height.png
  preview.png
  terrain.obj
  terrain.mtl
  terrain.sdf
  world.sdf
  metadata.json
```

Rules:

- Gazebo and Isaac must use the same `height.npy` and `metadata.json`.
- Terrain complexity belongs in the heightmap generator, not in hand-authored world objects.
- Spawn areas must be cleared by `spawn_clear_radius`.
- Isaac reset origins must be lifted to local terrain height plus spawn clearance.
- Isaac reset origins must keep a terrain-edge safety margin; `bounds`
  terminations should measure policy drift, not edge-biased spawn placement.
- RL curriculum maps use row-indexed terrain difficulty. Isaac training and
  eval can restrict reset origins with `--terrain-level-min` and
  `--terrain-level-max` while Gazebo still loads the same full heightmap for
  inspection.

Terrain difficulty curriculum:

| Batch | Terrain levels | Purpose | Exit gate |
| --- | --- | --- | --- |
| Stage 0 | `0:0` | command obedience on easiest roughness tiles | low `vx/wz` error, low action edge-rate |
| Stage 1 | `0:1` | introduce moderate roughness while preserving yaw | no fall/stuck, pure-turn and straight remain stable |
| Stage 2 | `0:2` | add hard slopes/steps/blocks progressively | mixed commands pass Isaac eval without high saturation |
| Stage 3 | `0:3` | full training distribution | Gazebo benchmark agrees with Isaac on same actor |
| Holdout | fixed `gazebo_demo/42` and unseen seeds | validation only, not main training | pass/fail baseline freeze |

## Vehicle Contract

Current model baseline:

- file: `tarantula_v2.urdf.xacro`;
- six single-DOF hip/arm joints commanded by `JointTrajectoryController`;
- one driven wheel per arm;
- wheel visual is cylindrical;
- wheel collision baseline is cylindrical;
- `sphere` is retained only as a rough-terrain contact A/B experiment;
- wheel-end F/T is the deployable wheel-load signal;
- geometry contact truth is not a policy input;
- Gazebo hip command is a bounded position trajectory profile, initialized from the current natural posture;
- wheel joints are velocity controlled in both Gazebo RL and Isaac.
- direct Gazebo GUI checks have verified stable natural posture, clean left/right in-place turning under direct six-wheel commands, and stable hip posture trajectories.

Detailed model decisions are in [docs/05-chassis-model-redesign.md](docs/05-chassis-model-redesign.md).

## ROS2 / Gazebo Contract

Launch:

```bash
ros2 launch tarantula_bringup sim.launch.py \
  gui:=true robot_model:=tarantula_v2.urdf.xacro motion_control:=true \
  start_motion_control:=false rl_compensation_enabled:=false \
  wheel_collision:=cylinder spawn_z:=0.55
```

Default world:

```text
generated/terrains/gazebo_demo/42/world.sdf
```

Direct Gazebo acceptance path:

- `scripts/gazebo_chassis_pose_diffdrive_test.py --profile turn-only`
  reproduces the verified GUI pure-turn test:
  - left turn: `[-2, +2, -2, +2, -2, +2]` for 6 seconds;
  - right turn: `[+2, -2, +2, -2, +2, -2]` for 6 seconds;
  - hip target is the initial natural posture.
- `scripts/gazebo_chassis_pose_diffdrive_test.py --profile posture-only`
  tests hip trajectory commands with zero wheel speed.
- `--profile full` combines wheel and posture checks.

Controller path:

- launch with `motion_control:=true start_motion_control:=true rl_compensation_enabled:=false`;
- `tarantula_control.motion_control.SkidSteerMotionController` is the
  baseline controller: `/cmd_vel -> six wheel velocities`;
- the baseline is calibrated skid-steer, not pure ideal differential drive:
  `arc_track_scale=1.0` is the moving-arc baseline,
  `pure_turn_track_scale=3.0` is the near-zero-vx turn baseline, and IMU
  yaw-rate feedback corrects residual yaw error;
- `motion_control_node` is only the ROS I/O wrapper around that controller;
- hip posture is owned by the v2 `JointTrajectoryController`; Stage A does not
  send hip residual commands.
- use this before RL to verify wheel signs, contact, terrain, and basic command response.
- optional RL compensation path:
  - `wheel_velocity_controller`
  - `suspension_controller`
  - `/wheel_velocity_controller/commands`
  - `/suspension_controller/joint_trajectory`
  - launch with `motion_control:=true start_motion_control:=true rl_compensation_enabled:=true`;
  - `rl_compensation_enabled:=false` runs the same motion node without policy weights;
  - Stage A final wheel command is always the scheduled motion-control baseline
    plus bounded structured compensation.
- legacy Gazebo effort-hold, wheel contact lab, and fixed-hip diagnostic paths
  have been removed. Use the v2 direct acceptance script as the baseline gate.

Sensor paths:

- `/imu/data`
- `/joint_states`
- `/ft_wheel/fl`
- `/ft_wheel/fr`
- `/ft_wheel/ml`
- `/ft_wheel/mr`
- `/ft_wheel/rl`
- `/ft_wheel/rr`
- `/scan`

Leg order is always:

```text
fl/fr/ml/mr/rl/rr
```

## Motion Control Contract

The deployable controller is layered:

```text
/cmd_vel
  -> command limiter
  -> classical skid-steer wheel target
  -> optional RL structured compensation
  -> final /wheel_velocity_controller/commands
```

The classical skid-steer controller is the baseline and source of truth for
normal planar motion. It must
work without RL and must pass Gazebo/Isaac open-loop command tracking before any
policy is judged.

The RL compensation layer is only allowed to output bounded structured
skid-steer corrections. Its intended roles are:

- yaw-rate correction when rough terrain, side slope, or wheel slip causes
  understeer/oversteer;
- longitudinal assist when pitch, velocity error, or wheel-speed mismatch
  indicates slope climbing or local stuck behavior;
- left/right traction redistribution based on wheel force and
  wheel-speed mismatch;
- smooth recovery from contact disturbances without violating the high-level
  `/cmd_vel` intent.

It must not own the basic `cmd_vx/cmd_wz -> wheel speed` mapping.

## Posture Control Contract

Traditional posture control is a bounded position-profile interface, not a
torque controller:

```text
posture profile or future hip residual
  -> six hip targets in fl/fr/ml/mr/rl/rr order
  -> /suspension_controller/joint_trajectory
```

Current baseline profiles are `neutral`, `front_down`, `rear_down`, `raise`,
`lower`, and `left_trim` in `tarantula_control.suspension_core`. Stage A does
not command hip residuals. Ring 5 may add RL hip-target residuals around these
profiles after Ring 1 motion tracking is stable.

## RL Contract

Stage A action space: 3D.

```text
action[0] = track_scale_delta, bounded by +/-0.30
action[1] = left_drive_scale_delta, bounded by +/-0.20
action[2] = right_drive_scale_delta, bounded by +/-0.20

effective_track = calibrated_track * (1 + track_scale_delta)
left_wheel      = skid_steer_left(cmd_vx, cmd_wz, effective_track) * (1 + left_drive_scale_delta)
right_wheel     = skid_steer_right(cmd_vx, cmd_wz, effective_track) * (1 + right_drive_scale_delta)
wheel_target    = clamp([left,right,left,right,left,right], -6.0, 6.0) rad/s
```

Stage A observation space: 47D.

```text
projected_gravity_b(3)
root_ang_vel_b(3)
susp_joint_pos(6)
susp_joint_vel(6)
wheel_joint_vel(6)
wheel_force_b(18)
cmd_vx(1)
cmd_wz(1)
prev_action(3)
```

Gazebo Stage A deployment uses the v2 `JointTrajectoryController` for hip
posture and `motion_control_node` for `/wheel_velocity_controller/commands`.
Set `rl_compensation_enabled:=false` to run a pure classical controller without
policy weights, or `rl_compensation_enabled:=true` to add the current 47D/3D
structured-compensation actor. Isaac uses simulator state for reward/eval only
and contact-force backend for the wheel-force equivalent. The actor observation
does not include simulator root linear velocity. Optional `truth_odom:=true` publishes
`/tarantula/truth_odom` from Gazebo truth pose for short diagnostics, but it is
not consumed by `motion_control_node` or any deployable control algorithm.
Runtime command input is `/cmd_vel` (`linear.x`, `angular.z`); launch
`cmd_vx/cmd_wz` are fallback defaults when no upstream command source is running.

Motion-compensation dimension coverage:

| Compensation need | Required observation | Current coverage | Status |
| --- | --- | --- | --- |
| yaw compensation | `cmd_wz`, measured yaw rate, wheel velocities | `cmd_wz`, IMU `root_ang_vel_b.z`, `wheel_joint_vel(6)` | covered |
| forward/slope assist | `cmd_vx`, pitch/gravity, wheel velocities, wheel force | `cmd_vx`, projected gravity, wheel velocities, wheel 3D F/T | covered without actor linear velocity |
| traction/load redistribution | per-wheel force and wheel velocity | wheel-end 3D F/T in Gazebo, contact-force equivalent in Isaac, wheel velocities | covered for simulation; real hardware needs wheel-end F/T or current/load estimator |
| stuck detection | high wheel speed/action with poor reward progress, pitch/force context | wheel velocities, command, force, pitch; truth velocity only in reward/eval; `/rl_policy/status` exposes action saturation and measured yaw rate | covered for training and Gazebo deployment diagnostics |
| side-slope correction | roll/projected gravity, yaw error, force asymmetry | projected gravity, yaw rate, wheel force | covered |
| terrain preview | local heightmap ahead of robot | generated heightmap exists in sim, not in Stage A obs | deferred to Stage C |
| absolute localization/nav quality | odom/SLAM pose | available through Gazebo/Nav stack, not policy obs | not needed for Stage A compensation |

Current sensors and actuators are sufficient for Stage A yaw/slip/slope/stuck
structured wheel compensation in simulation. The main real-world gap is not a new
actuator; it is a deployable velocity source and deployable wheel-load estimate.
Wheel-end F/T covers the paper-style load signal if the hardware can implement
it. If not, motor current plus suspension deflection/load calibration is the
fallback estimator. Terrain preview is useful but not required until Stage C.

Sensor decision by development ring:

| Ring | Sensor/interface | Decision | Reason |
| --- | --- | --- | --- |
| 0/1 | IMU | keep | chassis attitude and yaw rate are required for stability and command tracking |
| 0/1 | joint encoders from `/joint_states` | keep | hip position/velocity and wheel velocity cover proprioception |
| 0/1 | wheel-end F/T | keep as simulation baseline | directly supports wheel-load and traction reasoning; hardware may replace it with calibrated motor current/load estimation |
| 1/2 | deployable wheel odometry topic | add before Nav/RL deployment gates | needed as an engineering interface; Gazebo truth odom remains diagnostic only |
| 2/3 | `/rl_policy/status` | keep | exposes RL enable state, action values, action saturation, wheel-command magnitude, current command, and measured yaw rate |
| 3 | geometry contact booleans | do not add | not deployable and encourages policies to depend on contact truth |
| 5 | hip torque/current estimate | defer | useful for posture RL and load inference, but not needed for Stage A structured compensation |
| 6/7 | terrain preview from heightmap/stereo/lidar | defer to Stage C | papers use exteroception for harder terrain, but adding it before baseline parity will hide model/control bugs |

Literature alignment:

- Wiberg et al. use high-dimensional observations for rough-terrain vehicles
  with six wheels and active suspensions, rewarding safe, smooth, efficient
  traversal over slopes and obstacles:
  <https://arxiv.org/abs/2107.01867>.
- Bouton et al. combine proprioception and exteroception for an actively
  articulated rover: sparse terrain elevation, chassis attitude, joint states,
  and force-torque measurements:
  <https://arxiv.org/abs/2606.06790>.
- Gerdes et al. specifically evaluate wheel-assembly force/torque sensors plus
  chassis IMU on a six-wheeled rover and discuss their value for terrain and
  locomotion-performance inference:
  <https://arxiv.org/abs/2411.04700>.

Reward baseline follows the trimmed rough-terrain locomotion pattern used by legged_gym-style policies:

```text
+ track forward velocity command
+ track yaw-rate command with elevated Stage A weight
+ yaw direction/sign correctness for nonzero turn commands
+ keep base orientation near level
- penalize vertical base velocity
- penalize lateral body velocity as a slip proxy
- penalize low forward progress under nonzero forward command as a stuck proxy
- penalize roll/pitch angular velocity
- penalize action rate
- penalize action magnitude lightly
- penalize soft suspension joint-limit approach
+ small alive bonus
- termination penalty
```

Stage A command sampling is intentionally not uniform random over a continuous
box. During training, each environment periodically resamples one of four
command families so yaw does not disappear into near-zero angular commands and
one episode is not dominated by a single easy command:

```text
20% stop
25% straight forward/backward
25% pure turn left/right
30% arc turn
```

Straight and arc commands use `|cmd_vx| >= 0.12 m/s`. Pure-turn and arc
commands use `|cmd_wz| >= 0.15 rad/s`. Left/right and forward/backward signs
are sampled symmetrically. The `stage0` command profile narrows these ranges
for the first terrain row; fixed Isaac evaluation and yaw-authority sweeps
disable command resampling and set deterministic segment commands.

Wheel-load balance is not part of the Stage A reward. It remains an observation
and diagnostic metric; adding it to reward is deferred until the vehicle can
already traverse terrain reliably.

Termination baseline:

- roll/pitch tilt exceeds limit;
- base height is below/above safe range;
- root linear or angular velocity is physically implausible;
- root pose leaves terrain bounds;
- non-finite state/action values;
- timeout ends the episode without fall penalty.

Initial environment reset is excluded from termination-cause logging because the
robot has not yet been placed on generated-terrain origins. This keeps
`Episode_Termination/bounds` from reporting false positives at step 0.

Training diagnostics must expose:

- `Episode_Reward/*` for each reward term and total reward;
- `Episode_Termination/*` for tilt, height, velocity, bounds, non-finite state, and timeout;
- `Episode_Metric/*` for benchmark-style tracking and stability metrics:
  absolute/RMS `vx/wz` error, roll/pitch magnitude, action saturation,
  wheel-target saturation, and sampled command magnitudes;
- smoke tests must fail if these keys disappear.

Deterministic Isaac evaluation is mandatory before Gazebo deployment. Run
`src/tarantula_isaac/eval_policy_v5.py` on the same generated terrain with:

- `--mode open_loop` to establish the analytic wheel-speed baseline inside Isaac;
- `--mode npz --policy-npz <actor.npz>` to evaluate the exported policy;
- fixed low-speed `cmd_vx/cmd_wz` segments: stop, forward/backward
  (`+/-0.10 m/s`), low-speed left/right turn sign checks
  (`+/-0.15 rad/s`), separate turn-authority checks (`+/-0.25 rad/s`),
  left/right arc (`0.10 m/s`, `+/-0.12 rad/s`), final stop.

Traditional skid-steer sign convention:

- straight command: all six wheels rotate in the same direction;
- pure left turn: left row reverses and right row drives forward;
- pure right turn: left row drives forward and right row reverses;
- left/right row targets are shared across front/middle/rear wheels. Per-wheel
  differences belong only to structured RL or traction compensation, not the
  classical baseline.

The eval records spawn health, segment displacement, reward, time-averaged
command tracking error, final segment velocity, roll/pitch peak, action
saturation, and termination counts. A policy that does not move or that
saturates actions in Isaac is not eligible for Gazebo judgement.

## Verified Smoke Baseline

Current verified checks:

- generated `gazebo_demo/42`;
- generated `rl_curriculum/42`;
- `scripts/isaac_shared_terrain_smoke.sh` passes;
- lightweight PPO with `--max_iterations 1` reaches checkpoint export using
  rsl-rl actor/critic model config without deprecated `policy` or
  `empirical_normalization` warnings;
- exported `.npz` actor can command Gazebo through `motion_control_node`;
- Gazebo vehicle moves on generated terrain under the exported actor.
- `scripts/gazebo_cmd_tracking_benchmark.py` defines the Gazebo command-tracking
  acceptance path for `cmd_vx/cmd_wz`, including raw CSV samples, per-segment
  summaries, and classical-vs-RL comparison output.
- v2 direct hip trajectory commands establish a symmetric, stable natural
  posture on generated `gazebo_demo/42`;
- `scripts/gazebo_chassis_pose_diffdrive_test.py` verifies direct wheel
  traction: all-wheel positive/negative commands move forward/backward and
  left/right split commands produce yaw.
- current Stage A contract is 47D actor observation and 3D structured
  compensation. `yaw_authority_sweep.py` sweeps `track_action`.

## Next Required Work

Use a spiral process. Each ring has a fixed gate; do not start the next ring
until the gate passes.

| Ring | Scope | Change budget | Gate |
| --- | --- | --- | --- |
| 0 | Gazebo direct model baseline | v2 geometry, hip trajectory, wheel contact only | `turn-only` and `posture-only` GUI/script tests pass |
| 1 | Classical motion baseline | `/cmd_vel -> six wheel velocities`, no RL | Gazebo command tracking passes low-speed straight, reverse, pure turns, arcs |
| 2 | Isaac open-loop parity | same terrain and model parameters | Isaac open-loop segment metrics agree with Gazebo direction/sign/order |
| 3 | Stage A structured RL | track scale and left/right drive scale only | deterministic Isaac eval beats open-loop without high action edge-rate |
| 4 | Gazebo structured deployment | exported `.npz` only | Gazebo benchmark improves or preserves classical baseline |
| 5 | RL posture control | hip target residuals, no geometry change | posture policy improves rough-terrain stability without breaking Ring 1 |
| 6 | Outer model tuning | wheelbase, track, arm length, wheel radius, COM, contact parameters | A/B result improves gates across Gazebo and Isaac |
| 7 | Terrain curriculum widening | terrain difficulty rows and unseen seeds | policy survives wider terrain without regressions on Ring 0/1 |

Immediate rules:

1. Current baseline model is `tarantula_v2.urdf.xacro` with
   `wheel_collision:=cylinder`. Do not change it while debugging RL.
2. Treat all previous actors as chain-validation artifacts, not deployment
   baselines.
3. Keep the 3D structured compensation contract as Stage A because it preserves
   the classical control boundary.
4. Do not continue long RL runs just to compensate for a bad baseline. If Ring
   0 or Ring 1 fails, fix model/control before training.
5. Do not split every symptom into a separate investigation. Use the ring gate:
   one failing gate means inspect only the layer owned by that gate.
6. Keep `arc_track_scale=1.0` and `pure_turn_track_scale=3.0` as the calibrated
   classical expectations and let RL adjust only bounded `track_scale_delta` and
   left/right drive deltas. Reject policies with high action edge-rate even if
   reward improves.
7. Run deterministic Isaac eval in `open_loop` and `npz` modes before Gazebo RL
   deployment.
8. Use `/rl_policy/status` and `gazebo_cmd_tracking_benchmark.py compare` before
   longer deployment tests so saturation and tracking regressions are visible.

Application-facing interfaces to add after the command baseline:

1. `/rl_policy/enabled` or lifecycle transition to switch policy output on/off.
2. Emergency stop input that forces zero wheel command and neutral suspension.
3. Higher-level odometry interface for Nav2 that remains separate from Gazebo
   truth observer data.
