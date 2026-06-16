# Tarantula Baseline Plan

日期：2026-06-16

本文是当前开发的 source of truth。仓库只保留当前 baseline 相关文档。

## Current Baseline

当前 baseline 链路：

```text
tarantula_terrain.generate
  -> generated/terrains/gazebo_demo/42
  -> Gazebo sim.launch.py default world
  -> stand_suspension_hold Gazebo physics baseline
  -> gazebo_wheel_open_loop_benchmark.py
  -> tarantula_isaac.SharedHeightmapTerrainImporter
  -> train_v5.py lightweight PPO smoke run (Stage A wheel-only)
  -> export_weights_v5.py actor .npz
  -> rl_suspension_policy Gazebo deployment
```

当前目标是让 Gazebo、Isaac Lab、RL I/O、模型参数和文档保持同一个 baseline。

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

## Vehicle Contract

Current model baseline:

- six single-DOF suspension arms;
- one driven wheel per arm;
- wheel visual is cylindrical;
- wheel collision is spherical;
- wheel-end F/T is the deployable wheel-load signal;
- geometry contact truth is not a policy input;
- Gazebo suspension command is explicit bounded PD effort;
- Isaac suspension command is the equivalent joint position drive;
- wheel joints are velocity controlled in both Gazebo RL and Isaac.

Detailed model decisions are in [docs/05-chassis-model-redesign.md](docs/05-chassis-model-redesign.md).

## ROS2 / Gazebo Contract

Launch:

```bash
ros2 launch tarantula_bringup sim.launch.py gui:=true leveling:=false
```

Default world:

```text
generated/terrains/gazebo_demo/42/world.sdf
```

Controller paths:

- classic Nav path:
  - `diff_drive_controller`
  - `/diff_drive_controller/cmd_vel_unstamped`
  - `/diff_drive_controller/odom`
- manual per-wheel baseline path:
  - launch with `manual_wheel:=true stand_hold:=true`;
  - `cmd_vel_wheel_baseline` maps `/cmd_vel` to six wheel velocities;
  - `stand_suspension_hold` owns `/suspension_controller/commands`;
  - use this before RL to verify wheel signs, contact, terrain, and basic command response.
- RL path:
  - `wheel_velocity_controller`
  - `suspension_controller`
  - `/wheel_velocity_controller/commands`
  - `/suspension_controller/commands`
- Gazebo physics baseline path:
  - launch with `rl_policy:=true start_rl_policy:=false stand_hold:=true`;
  - `stand_suspension_hold` owns `/suspension_controller/commands`;
  - `gazebo_wheel_open_loop_benchmark.py` owns `/wheel_velocity_controller/commands`;
  - this path is the required baseline before judging wheel/terrain contact or
    RL deployment behavior.

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

## RL Contract

Stage A action space: 6D.

```text
action[0:6]  wheel velocity targets, scaled by 3.0 rad/s
```

Stage A observation space: 41D.

```text
projected_gravity_b(3)
root_ang_vel_b(3)
root_lin_vel_b(3)
susp_joint_pos(6)
susp_joint_vel(6)
wheel_joint_vel(6)
wheel_load(6)
cmd_vx(1)
cmd_wz(1)
prev_action(6)
```

Gazebo Stage A deployment runs `stand_suspension_hold` for suspension and a
wheel-only RL actor for `/wheel_velocity_controller/commands`. The actor must be
launched with `rl_policy_mode:=wheel_only` to prevent accidental use of a legacy
12D suspension+wheel actor. Isaac uses simulator state for base velocity and
contact-force backend for the wheel-load equivalent. Gazebo deployment defaults
to wheel-speed forward velocity estimation; optional `truth_odom:=true` publishes
`/tarantula/truth_odom` from Gazebo truth pose for short diagnostics, but it is
not a default runtime dependency because it currently samples Gazebo through the
CLI. Runtime command input is `/cmd_vel` (`linear.x`, `angular.z`); launch
`cmd_vx/cmd_wz` are fallback defaults when no upstream command source is running.

Reward baseline follows the trimmed rough-terrain locomotion pattern used by legged_gym-style policies:

```text
+ track forward velocity command
+ track yaw-rate command
+ keep base orientation near level
- penalize vertical base velocity
- penalize roll/pitch angular velocity
- penalize action rate
- penalize action magnitude lightly
- penalize soft suspension joint-limit approach
+ small alive bonus
- termination penalty
```

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
- smoke tests must fail if these keys disappear.

## Verified Smoke Baseline

Current verified checks:

- generated `gazebo_demo/42`;
- generated `rl_curriculum/42`;
- `scripts/isaac_shared_terrain_smoke.sh` passes;
- lightweight PPO with `--max_iterations 1` reaches checkpoint export using
  rsl-rl actor/critic model config without deprecated `policy` or
  `empirical_normalization` warnings;
- exported `.npz` actor can command Gazebo through `rl_suspension_policy`;
- Gazebo vehicle moves on generated terrain under the exported actor.
- `scripts/gazebo_cmd_tracking_benchmark.py` defines the Gazebo command-tracking
  acceptance path for `cmd_vx/cmd_wz`.
- `stand_suspension_hold` can establish a symmetric, stable stand posture on
  generated `gazebo_demo/42`;
- `scripts/gazebo_wheel_open_loop_benchmark.py` verifies basic wheel traction
  under stand hold: all-wheel positive/negative commands move forward/backward
  and left/right split commands produce yaw.

## Next Required Work

Before serious RL:

1. Run manual per-wheel baseline with `manual_wheel:=true stand_hold:=true`.
2. Train and export a 6D wheel-only Stage A actor.
3. Run the deterministic Gazebo command-tracking benchmark with
   `stand_hold:=true rl_policy_mode:=wheel_only`.
4. Record progress, roll/pitch RMS, stand-hold effort saturation, stuck/fall events.
5. Freeze `model_baseline_v1` from benchmark metrics.
6. Add `/rl_policy/status` for live deployment diagnostics.
7. Only after Stage A command obedience is stable, design Stage B suspension
   residual actions around the stand target.

Application-facing interfaces to add after the command baseline:

1. `/rl_policy/enabled` or lifecycle transition to switch policy output on/off.
2. `/rl_policy/status` with command tracking error, saturation, timeout, and
   fall/stuck flags.
3. Emergency stop input that forces zero wheel command and neutral suspension.
4. Gazebo/Isaac benchmark metrics for `cmd_vx/cmd_wz` tracking error.
