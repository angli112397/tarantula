# Tarantula

六轮主动悬挂/轮腿式底盘仿真项目。当前 baseline 是：

```text
shared heightmap terrain
  -> Gazebo GUI / ROS2 integration
  -> Isaac Lab DirectRLEnv
  -> lightweight PPO smoke training
  -> exported actor .npz
  -> Gazebo RL deployment check
```

项目 source of truth：

- [docs/00-project-plan.md](docs/00-project-plan.md)
- [docs/05-chassis-model-redesign.md](docs/05-chassis-model-redesign.md)

## Repository Layout

```text
src/tarantula_description   URDF/xacro robot model and Gazebo adapters
src/tarantula_bringup       Gazebo launch, ROS2 controllers, SLAM/Nav2 config
src/tarantula_control       Suspension helpers and RL deployment node
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

Launch the generated terrain world:

```bash
source /opt/ros/humble/setup.bash
source install/setup.bash
ros2 launch tarantula_bringup sim.launch.py gui:=true leveling:=false
```

Run the manual per-wheel baseline on the same wheel controller used by RL:

```bash
ros2 launch tarantula_bringup sim.launch.py \
  gui:=true \
  leveling:=false \
  manual_wheel:=true \
  stand_hold:=true \
  cmd_vx:=0.2 \
  cmd_wz:=0.0 \
  spawn_z:=0.45
```

Run the RL deployment path with an exported actor:

```bash
ros2 launch tarantula_bringup sim.launch.py \
  gui:=true \
  leveling:=false \
  rl_policy:=true \
  stand_hold:=true \
  rl_policy_mode:=wheel_only \
  velocity_source:=auto \
  truth_odom:=false \
  cmd_vx:=0.2 \
  cmd_wz:=0.0 \
  policy_weights_npz:=$(pwd)/generated/policies/cmd_vel_actor.npz \
  spawn_z:=0.45
```

Publish runtime velocity commands:

```bash
ros2 topic pub /cmd_vel geometry_msgs/msg/Twist \
  "{linear: {x: 0.2}, angular: {z: 0.0}}"
```

Per-wheel command topics:

- `/suspension_controller/commands`: 6 suspension efforts, `fl/fr/ml/mr/rl/rr`
- `/wheel_velocity_controller/commands`: 6 wheel velocity targets, `fl/fr/ml/mr/rl/rr`
- `manual_wheel:=true` maps `/cmd_vel` directly to per-wheel skid-steer speeds.
- Stage A wheel-only actors do not publish `/suspension_controller/commands`;
  `stand_suspension_hold` owns suspension during Gazebo deployment.

RL observation inputs in Gazebo:

- `/imu/data`
- `/joint_states`
- `/ft_wheel/{fl,fr,ml,mr,rl,rr}`
- Optional `/tarantula/truth_odom` for Gazebo body-frame linear/angular velocity
  when `truth_odom:=true`; this is a diagnostic adapter and is off by default
  because it samples Gazebo through the CLI.
- `/cmd_vel` for runtime `cmd_vx/cmd_wz`; launch `cmd_vx/cmd_wz` are fallback defaults.

Run the command-tracking benchmark after Gazebo is running:

```bash
source /opt/ros/humble/setup.bash
source install/setup.bash
scripts/gazebo_cmd_tracking_benchmark.py
```

Benchmark outputs are written under `generated/benchmarks/cmd_tracking/`.

Run the wheel open-loop physics benchmark without the RL node:

```bash
ros2 launch tarantula_bringup sim.launch.py \
  gui:=true \
  leveling:=false \
  rl_policy:=true \
  start_rl_policy:=false \
  stand_hold:=true \
  spawn_z:=0.45

scripts/gazebo_wheel_open_loop_benchmark.py
```

This publishes direct `/wheel_velocity_controller/commands` while
`stand_suspension_hold` owns `/suspension_controller/commands`. Use it before
RL runs to isolate wheel contact, terrain collision, joint-axis direction, and
left/right geometry bias. Outputs are written under
`generated/benchmarks/wheel_open_loop/`.

Current Gazebo physics baseline: `stand_hold:=true` establishes a symmetric
stable posture; all-wheel open-loop `+3/-3 rad/s` commands move the vehicle
forward/backward on `gazebo_demo/42`.

Use this order when debugging motion:

1. `manual_wheel:=true stand_hold:=true`: verifies geometry, wheel signs, contact,
   and terrain without RL.
2. `rl_policy:=true stand_hold:=true rl_policy_mode:=wheel_only`: verifies the
   exported actor on the same execution surface.
3. `truth_odom:=true` only when a short diagnostic run needs Gazebo truth body
   velocity in the RL observation.

Wheel collision A/B:

```bash
# baseline
ros2 launch tarantula_bringup sim.launch.py \
  gui:=true leveling:=false rl_policy:=true start_rl_policy:=false \
  stand_hold:=true wheel_collision:=sphere spawn_z:=0.45

# comparison
ros2 launch tarantula_bringup sim.launch.py \
  gui:=true leveling:=false rl_policy:=true start_rl_policy:=false \
  stand_hold:=true wheel_collision:=cylinder spawn_z:=0.45
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
  --terrain-dir "$(pwd)/generated/terrains/gazebo_demo/42"
```

Export the actor:

```bash
source ~/isaac_venv/bin/activate
python3 src/tarantula_isaac/export_weights_v5.py \
  --checkpoint logs/rsl_rl/tarantula_suspension/<run>/model_0.pt \
  --npz-out generated/policies/cmd_vel_actor.npz
```

## Current Contracts

- Terrain source: generated `height.npy` + `metadata.json`.
- Gazebo default world: `generated/terrains/gazebo_demo/42/world.sdf`.
- Isaac terrain importer: `SharedHeightmapTerrainImporter`.
- Isaac reset origins are lifted to local heightmap height and kept inside a
  terrain-edge safety margin for RL.
- Stage A action space: 6D per-wheel velocities.
- Stage A observation space: 41D, including IMU, joint state, wheel velocity, wheel load, `cmd_vx/cmd_wz`, previous wheel action.
- Manual baseline and RL share `/wheel_velocity_controller/commands`; this is
  the required interface boundary before judging policy quality.
- Gazebo truth odom is optional diagnostic input, not a default runtime dependency.
- Leg order everywhere: `fl/fr/ml/mr/rl/rr`.
- Wheel visual: cylinder.
- Wheel collision: sphere.
- Wheel load observation: wheel-end F/T in Gazebo, contact-force equivalent in Isaac.
- Geometry contact booleans are not policy inputs.

## Next Work

Before serious RL runs, implement a deterministic Gazebo benchmark runner and
freeze a model baseline version from its metrics.
