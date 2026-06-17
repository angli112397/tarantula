# Tarantula

六轮主动悬挂/轮腿式底盘仿真项目。当前 baseline 是：

```text
shared heightmap terrain
  -> Gazebo GUI / ROS2 integration
  -> v2 chassis baseline
  -> classical skid-steer motion control
  -> optional structured RL compensation
  -> Isaac Lab curriculum only after Gazebo/Isaac baselines agree
```

项目 source of truth：

- [docs/00-project-plan.md](docs/00-project-plan.md)
- [docs/05-chassis-model-redesign.md](docs/05-chassis-model-redesign.md)

## Repository Layout

```text
src/tarantula_description   URDF/xacro robot model and Gazebo adapters
src/tarantula_bringup       Gazebo launch, ROS2 controllers, SLAM/Nav2 config
src/tarantula_control       Motion/posture helpers and RL deployment node
src/tarantula_isaac         Isaac Lab robot/env/training/export code
src/tarantula_terrain       Shared heightmap generator and Gazebo exporters
docs/                       Current plan and chassis baseline
scripts/                    Current smoke/train helpers
generated/terrains/         Generated heightmaps, meshes, SDF, metadata
```

## Build

```bash
cd /home/ang/Documents/tarantula
source /opt/ros/humble/setup.bash
colcon build --symlink-install --parallel-workers 1 --executor sequential
source install/setup.bash
```

## Generate Baseline Terrain

```bash
PYTHONPATH=src/tarantula_terrain \
python3 -m tarantula_terrain.generate --preset gazebo_demo --seed 42 --output-root generated/terrains

PYTHONPATH=src/tarantula_terrain \
python3 -m tarantula_terrain.generate --preset rl_curriculum --seed 42 --output-root generated/terrains
```

Generated assets are written to:

```text
generated/terrains/<preset>/42/
  height.npy
  height.png
  preview.png
  terrain.obj
  terrain.mtl
  terrain.sdf
  world.sdf
  metadata.json
```

`gazebo_demo` is the Gazebo inspection baseline. `rl_curriculum` is the Isaac
curriculum baseline with `env_origins` metadata.

## Gazebo Baseline

Current model baseline:

- model: `tarantula_v2.urdf.xacro`
- hip command interface: `/suspension_controller/joint_trajectory`
- wheel command interface: `/wheel_velocity_controller/commands`
- wheel collision: `cylinder` for kinematic A/B; use `sphere` only when reproducing the current rough-terrain contact comparison.
- validated GUI behavior: stable natural hip posture, clean left/right in-place turning, and stable hip posture trajectories.

Launch the baseline GUI:

```bash
source /opt/ros/humble/setup.bash
source install/setup.bash
ros2 launch tarantula_bringup sim.launch.py \
  gui:=true \
  robot_model:=tarantula_v2.urdf.xacro \
  world:=$(pwd)/generated/terrains/flat_smoke/42/world.sdf \
  motion_control:=true \
  start_motion_control:=false \
  rl_compensation_enabled:=false \
  wheel_collision:=cylinder \
  spawn_z:=0.55
```

`sim.launch.py` defaults now point at the same v2 baseline model and spawn
height; the explicit arguments above are kept to make review runs unambiguous.

Run the direct chassis acceptance profiles while Gazebo is running:

```bash
# Reproduces the verified GUI in-place left/right turn test.
scripts/gazebo_chassis_pose_diffdrive_test.py --profile turn-only

# Tests hip posture trajectories only, with wheel commands held at zero.
scripts/gazebo_chassis_pose_diffdrive_test.py --profile posture-only

# Runs the combined acceptance suite.
scripts/gazebo_chassis_pose_diffdrive_test.py --profile full
```

The script records Gazebo truth pose as observer data only. It does not feed
truth pose into the controller. Every profile returns the hip targets to the
initial natural posture and sends zero wheel speed at the end.

Run the classical `/cmd_vel` motion-control baseline:

```bash
ros2 launch tarantula_bringup sim.launch.py \
  gui:=true \
  robot_model:=tarantula_v2.urdf.xacro \
  motion_control:=true \
  start_motion_control:=true \
  rl_compensation_enabled:=false \
  cmd_vx:=0.1 \
  cmd_wz:=0.0 \
  wheel_collision:=cylinder \
  spawn_z:=0.55
```

Publish runtime velocity commands:

```bash
ros2 topic pub /cmd_vel geometry_msgs/msg/Twist \
  "{linear: {x: 0.1}, angular: {z: 0.0}}"
```

Run the deployment path with an exported structured-compensation actor only after the
classical baseline passes:

```bash
ros2 launch tarantula_bringup sim.launch.py \
  gui:=true \
  robot_model:=tarantula_v2.urdf.xacro \
  motion_control:=true \
  start_motion_control:=true \
  rl_compensation_enabled:=true \
  truth_odom:=false \
  cmd_vx:=0.1 \
  cmd_wz:=0.0 \
  policy_weights_npz:=$(pwd)/generated/policies/cmd_vel_actor.npz \
  wheel_collision:=cylinder \
  spawn_z:=0.55
```

ROS control topics:

- `/suspension_controller/joint_trajectory`: six hip position targets in `fl/fr/ml/mr/rl/rr` order.
- `/wheel_velocity_controller/commands`: six wheel velocity targets in `fl/fr/ml/mr/rl/rr` order.
- `/cmd_vel`: application command input consumed by `motion_control_node`.
- `/rl_policy/status`: diagnostic observer output from `motion_control_node`;
  data order is `enabled, track_scale_action, left_drive_action,
  right_drive_action, action_saturation, wheel_cmd_max_abs, cmd_vx, cmd_wz,
  measured_wz, motion_mode_turn`. `cmd_vx/cmd_wz` are the shaped execution
  command values. With the default `command_strategy:=stop_turn_drive`, a
  large-yaw input executes as `vx_exec=0,wz_exec=cmd_wz`; low-yaw driving
  executes as `vx_exec=cmd_vx,wz_exec=0`.

Run the Gazebo command-tracking benchmark after launching the sim:

```bash
# Classical baseline run.
scripts/gazebo_cmd_tracking_benchmark.py \
  --label classical \
  --duration 4.0 \
  --settle 1.0 \
  --rate 5 \
  --out-dir generated/benchmarks/cmd_tracking/classical

# RL-compensated run, launched separately with rl_compensation_enabled:=true.
scripts/gazebo_cmd_tracking_benchmark.py \
  --label rl \
  --duration 4.0 \
  --settle 1.0 \
  --rate 5 \
  --out-dir generated/benchmarks/cmd_tracking/rl

# Offline A/B report.
scripts/gazebo_cmd_tracking_benchmark.py compare \
  --baseline generated/benchmarks/cmd_tracking/classical \
  --candidate generated/benchmarks/cmd_tracking/rl \
  --out generated/benchmarks/cmd_tracking/compare.json
```

The benchmark writes `samples.csv` and `summary.json`. Truth pose is used only
for evaluation. Controller inputs remain `/cmd_vel`, IMU, joint state, and
wheel F/T.

Debugging order:

1. Direct wheel/hip GUI tests with `start_motion_control:=false`.
2. Classical `motion_control_node` with `rl_compensation_enabled:=false`.
3. Isaac open-loop eval on the same terrain contract.
4. Structured RL in Isaac, then Gazebo deployment.
5. Model outer-loop changes such as wheelbase, track width, arm length, wheel radius, COM, and contact parameters.

`truth_odom:=true` may publish `/tarantula/truth_odom` for short diagnostics and
benchmarks, but it is observer data and must not be used by deployable control.

RL observation inputs in Gazebo:

- `/imu/data`
- `/joint_states`
- `/ft_wheel/{fl,fr,ml,mr,rl,rr}`
- limited `cmd_vx/cmd_wz`; launch `cmd_vx/cmd_wz` are fallback defaults before
  `/cmd_vel` arrives.

Wheel collision A/B:

```bash
# baseline
ros2 launch tarantula_bringup sim.launch.py \
  gui:=true motion_control:=true start_motion_control:=false \
  robot_model:=tarantula_v2.urdf.xacro \
  wheel_collision:=cylinder spawn_z:=0.55

# comparison
ros2 launch tarantula_bringup sim.launch.py \
  gui:=true motion_control:=true start_motion_control:=false \
  robot_model:=tarantula_v2.urdf.xacro \
  wheel_collision:=sphere spawn_z:=0.55
```

## Isaac Baseline

Smoke-check the same terrain in Isaac Lab:

```bash
scripts/isaac_shared_terrain_smoke.sh
```

Run a lightweight PPO smoke training:

```bash
NUM_ENVS=2 scripts/run_ppo_train_v5.sh \
  --max_iterations 1 \
  --terrain-dir "$(pwd)/generated/terrains/rl_curriculum/42" \
  --terrain-level-min 0 \
  --terrain-level-max 0 \
  --command-profile stage0
```

Train on staged terrain difficulty batches using the `rl_curriculum` heightmap:

```bash
source ~/isaac_venv/bin/activate
export PYTHONPATH="$(pwd)/src:$(pwd)/src/tarantula_control:${PYTHONPATH:-}"

# Stage 0: easiest row only.
python3 src/tarantula_isaac/train_v5.py \
  --num_envs 64 \
  --max_iterations 150 \
  --terrain-dir "$(pwd)/generated/terrains/rl_curriculum/42" \
  --terrain-level-min 0 \
  --terrain-level-max 0 \
  --command-profile stage0 \
  --command-resampling-time 3.0 \
  --max-abs-wheel-omega 6.0 \
  --track-scale-delta-limit 0.30 \
  --drive-scale-delta-limit 0.20 \
  --entropy-coef 0.0002 \
  --policy-init-std 0.35 \
  --action-magnitude-weight 0.02 \
  --action-rate-weight 0.03 \
  --action-saturation-weight 0.50 \
  --action-saturation-soft-limit 0.65

# Stage 1/2/3: resume from the previous checkpoint and widen max level.
python3 src/tarantula_isaac/train_v5.py \
  --resume logs/rsl_rl/tarantula_suspension/<run>/model_199.pt \
  --num_envs 64 \
  --max_iterations 200 \
  --terrain-dir "$(pwd)/generated/terrains/rl_curriculum/42" \
  --terrain-level-min 0 \
  --terrain-level-max 1 \
  --command-profile mixed \
  --command-resampling-time 3.0 \
  --max-abs-wheel-omega 6.0 \
  --track-scale-delta-limit 0.30 \
  --drive-scale-delta-limit 0.20 \
  --entropy-coef 0.001 \
  --action-saturation-weight 0.08
```

Stage A training samples only stop, straight/backward, and pure-turn execution
commands. Raw high-yaw `/cmd_vel` pairs may still be present in deterministic
checks, but the baseline shapes them into pure turn before observation, reward,
and benchmark scoring. Keep `gazebo_demo/42` and unseen `rl_curriculum` seeds as
holdout validation instead of training-only evidence.

Export the actor:

```bash
source ~/isaac_venv/bin/activate
python3 src/tarantula_isaac/export_weights_v5.py \
  --checkpoint logs/rsl_rl/tarantula_suspension/<run>/model_0.pt \
  --npz-out generated/policies/cmd_vel_actor.npz
```

Exported `.npz` actors include `max_abs_wheel_omega`,
`track_scale_delta_limit`, and `drive_scale_delta_limit`. Isaac eval and Gazebo
`motion_control_node` read that metadata so deployment uses the same wheel clamp
and residual action scale as training.

Before judging the actor in Gazebo, run the deterministic Isaac eval on the
same terrain:

```bash
source ~/isaac_venv/bin/activate
export PYTHONPATH="$(pwd)/src:$(pwd)/src/tarantula_control:${PYTHONPATH:-}"

# Analytic wheel-speed baseline inside Isaac.
python3 src/tarantula_isaac/eval_policy_v5.py \
  --mode open_loop \
  --num-envs 16 \
  --terrain-dir "$(pwd)/generated/terrains/gazebo_demo/42" \
  --out generated/benchmarks/isaac_eval/open_loop_summary.json

# Exported actor inside Isaac.
python3 src/tarantula_isaac/eval_policy_v5.py \
  --mode npz \
  --policy-npz generated/policies/cmd_vel_actor.npz \
  --num-envs 16 \
  --terrain-dir "$(pwd)/generated/terrains/gazebo_demo/42" \
  --out generated/benchmarks/isaac_eval/policy_summary.json
```

The deterministic command sequence is intentionally low-speed and covers the
deployable command surface: stop, raw high-yaw commands that shape into pure
turns, straight driving, backward driving, and pure-turn authority checks. The
metric target is the shaped execution command, not the raw `/cmd_vel` pair.

## Current Contracts

- Terrain source: generated `height.npy` + `metadata.json`.
- Gazebo default world: `generated/terrains/gazebo_demo/42/world.sdf`.
- Isaac terrain importer: `SharedHeightmapTerrainImporter`.
- Isaac reset origins are lifted to local heightmap height and kept inside a
  terrain-edge safety margin for RL.
- Motion-control baseline is split into two layers. `CommandShaper` accepts
  normal `/cmd_vel`; the default `command_strategy:=stop_turn_drive` turns
  high-yaw commands into pure rotation and low-yaw commands into straight
  drive. `command_strategy:=continuous` remains available for A/B testing.
  `SkidSteerMotionController` then maps the shaped execution command to six
  wheel velocity targets. `yaw_rate_kp` is default-off; measured-yaw feedback is
  treated as an optional calibrated experiment because it can move the pure-turn
  rotation center away from the chassis center. RL is a switchable bounded
  structured compensation layer, not the owner of basic planar kinematics.
- Stage A action space: 3D structured compensation:
  `track_scale_delta`, `left_drive_scale_delta`, `right_drive_scale_delta`.
  The action is component-gated against the shaped execution command:
  yaw-active commands may apply `track_scale_delta`, drive-active commands may
  apply left/right drive scale deltas, and STOP applies no RL compensation.
- Stage A actor `.npz` files carry the final wheel clamp; do not deploy a
  policy if metadata does not match its training run. Current actors also carry
  the trained residual action scale for track and drive compensation.
- Stage A observation space: 47D, including IMU, joint state, wheel velocity,
  wheel 3D F/T force, shaped execution `cmd_vx/cmd_wz`, and previous structured
  action.
- Motion-control baseline and RL compensation share
  `/wheel_velocity_controller/commands`; this is the required interface boundary
  before judging policy quality.
- Posture baseline is bounded hip position profiles through
  `/suspension_controller/joint_trajectory`. Current profiles live in
  `tarantula_control.suspension_core` and are used only for direct acceptance
  and future hip-target residual RL.
- Gazebo truth odom is optional diagnostic input, not a default runtime dependency.
- Leg order everywhere: `fl/fr/ml/mr/rl/rr`.
- Wheel visual: cylinder.
- Wheel collision: cylinder for the Gazebo/Isaac kinematic baseline; sphere is
  retained only as rough-terrain contact A/B.
- Wheel force observation: wheel-end 3D F/T in Gazebo, contact-force equivalent in Isaac.
- Geometry contact booleans are not policy inputs.

## Next Work

Before serious RL runs, prove the classical `stop_turn_drive` skid-steer
baseline in Gazebo and Isaac. Then train RL only as structured residual
compensation: drive-scale residuals for forward/backward traction and
track-scale residuals for yaw authority. Run deterministic Isaac eval before
the Gazebo command-tracking benchmark.

Immediate RL next step: train the 3D structured-compensation actor on stop,
straight/backward, and pure-turn execution commands. Reject candidates that
improve reward by saturating the compensation actions.

Stage 0 rejection gate: do not launch Gazebo unless deterministic Isaac eval
shows lower weighted command error than open loop, no velocity terminations on
the fixed sequence, and mean action saturation below 0.20. Failed Stage 0 runs on
2026-06-17 showed good straight-line tracking but excessive action saturation
and poor yaw authority in the old continuous-arc experiment. The deployable
baseline now handles that by shape-first stop-turn-drive, matching Nav2
rotation-shim style behavior.
