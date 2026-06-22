# Tarantula

六轮主动悬挂/轮腿式底盘仿真项目。当前 baseline 已切换为：

```text
shared heightmap terrain
  -> tarantula_v3 active-suspension chassis
  -> ROS2/Nav2-standard planar motion and static-map navigation
  -> official diff_drive_controller for Nav2 smoke tests
  -> optional 6D RL active-suspension posture policy
  -> Gazebo/Nav2 and posture/stability acceptance
  -> Isaac Lab posture-control curriculum
```

项目 source of truth：

- [docs/00-project-plan.md](docs/00-project-plan.md)
- [docs/05-chassis-model-redesign.md](docs/05-chassis-model-redesign.md)

## Project Contract

RL 不再参与轮速、yaw、track width、drive scale 或轨迹跟踪补偿。运动、SLAM、导航使用 ROS2/Nav2 最佳实践；RL 只作为主动悬挂 policy，发布六个髋关节目标，让底盘在可滚过复杂路况上保持更小 roll/pitch、更稳定轮载和更好的 LiDAR/SLAM 输入。

当前承诺的 traversability envelope：

- 低速可滚过粗糙地面、浅坑、缓坡、低台阶、横坡。
- Navi 负责绕开不可通行障碍并提供合理路径。
- 主动悬挂只改善支撑和传感器姿态，不承诺高台阶、深沟或步态式越障。

## Repository Layout

```text
src/tarantula_description   v3 URDF/xacro model and Gazebo adapters
src/tarantula_bringup       Gazebo launch, ROS2 controllers, SLAM/Nav2 config
src/tarantula_control       Classical motion, posture policy runtime, diagnostics
src/tarantula_isaac         Isaac Lab active-suspension env/training/export
src/tarantula_terrain       Shared heightmap generator and Gazebo exporters
docs/                       Current project plan and chassis decisions
scripts/                    Current smoke/train/commissioning helpers
generated/terrains/         Generated heightmaps, meshes, SDF, metadata
```

## Navigation Maze

Generate an aligned Nav2/Gazebo maze layer:

```bash
PYTHONPATH=src:src/tarantula_control:src/tarantula_terrain \
python3 -m tarantula_terrain.nav_maze --seed 42 --output-root generated/terrains
```

Output:

```text
generated/terrains/nav_maze/42/
  height.npy       # rl_curriculum-derived rough height layer with clear pads
  occupancy.npy    # navigation obstacle layer, same size/resolution
  map.pgm/yaml     # Nav2 occupancy map
  traversability_cost.npy
  terrain_cost_map.pgm/yaml   # Nav2 2.5D planning cost map
  terrain_speed_mask.pgm/yaml # Nav2 SpeedFilter mask for slow terrain
  world.sdf        # Gazebo Nav2 world: flat contact floor + terrain visual + walls
  world_mesh_contact.sdf # experimental world: wheels contact the terrain mesh
  metadata.json    # spawn/goals/grid/layer metadata
```

The default maze is `24m x 16m @ 0.10m`, matching the current RL curriculum
grid so height and occupancy can later be composed as aligned grayscale layers.
It uses a seeded medium-complexity large-chassis navigation baseline:
`door_width=5.2m`, `min_corridor_width=4.8m`, `obstacle_count=3`, origin spawn,
and a forced center safety pad carved out of the generated segmented-wall
layout. The goal is to keep the map visually useful for SLAM/Nav2 demos while
avoiding spawn-side clearance failures.

`height.npy` is generated from the same `rl_curriculum` heightmap source used by
Isaac Lab, then the spawn/goal pads are flattened. This is the shared baseline
for later “one grayscale height layer + one occupancy layer” experiments.
Gazebo Nav2 smoke tests use `world.sdf`, which keeps a thick flat contact floor
for stable skid-steer wheel contact and overlays the terrain mesh as visual
context. Use `world_mesh_contact.sdf` for explicit mesh-contact A/B tests
(e.g. `scripts/gazebo_pursuit_eval.py` on `rl_curriculum`) -- this contact mode
is calibrated and works (see `docs/00-project-plan.md`'s Gazebo world policy),
migrating the Nav2 demo to it is just not done yet.

Static-map Nav2 uses three maps at once: `map.yaml` is the pure occupancy map
for AMCL localization on `/map`; `terrain_cost_map.yaml` is published on
`/terrain_cost_map` for global/local costmaps; `terrain_speed_mask.yaml` is
published on `/terrain_speed_mask` for Nav2's official SpeedFilter. The terrain
layers are derived from the shared height layer using slope and local relief,
but `/map` stays a normal localization map. All maps preserve Gazebo/world x
columns and flip only image rows when exporting, matching ROS `map_server`'s
top-row image convention while keeping world `x/y` coordinates aligned with the
SDF wall boxes.

## Build

```bash
cd /home/ang/Documents/tarantula
source /opt/ros/humble/setup.bash
colcon build --symlink-install --parallel-workers 1 --executor sequential
source install/setup.bash
```

## Gazebo/Nav2 Baseline

Current model baseline:

- model: `tarantula_v3.urdf.xacro`
- Nav2 wheel controller: official `diff_drive_controller`
- custom wheel command path: `/wheel_velocity_controller/commands`
- hip command: `/suspension_controller/joint_trajectory`
- motion input: `/cmd_vel`
- posture policy status: `/posture_policy/status`
- motion status: `/motion_control/status`
- odometry: Nav2 demo uses `/diff_drive_controller/odom`; custom controller
  experiments use `/wheel/odom -> robot_localization -> /odometry/filtered`
- sensors: `/imu/data`, `/joint_states`, `/ft_wheel/*`, `/scan`
- wheel collision: `sphere` by default; `cylinder` is only a contact A/B check.

Nav2 demo baseline:

- static map navigation: `nav_static.launch.py` launches two map servers:
  `map.yaml` on `/map` for AMCL, `terrain_cost_map.yaml` on
  `/terrain_cost_map` for Nav2 global/local costmaps, and
  `terrain_speed_mask.yaml` on `/terrain_speed_mask` for SpeedFilter
- odom: official diff-drive odom is the current Nav2 smoke-test source; EKF
  remains available for the custom per-wheel controller path
- costmap footprint: v3-derived navigation footprint (`1.12m x 0.74m`), no extra footprint padding
- conservative large-chassis costmap: local inflation `0.55m`, global inflation
  `0.65m`, slow cost decay, and higher Smac cost travel multiplier
- online SLAM remains available through `nav.launch.py`, but automatic
  exploration is not part of the current baseline.

Current Nav2 status:

- static grayscale map loading, AMCL initial pose, `map->odom`, and Nav2 action
  execution are connected;
- Gazebo static-map navigation works on the generated baseline maze with the
  official `diff_drive_controller`;
- online SLAM with `slam_toolbox` publishes `/map` from `/scan_gated`.

Launch static-map navigation on the generated maze:

```bash
source /opt/ros/humble/setup.bash
source install/setup.bash
ros2 launch tarantula_bringup sim.launch.py \
  gui:=true \
  drive_controller:=diff_drive \
  bridge_lidar:=true \
  bridge_force_torque:=false \
  spawn_x:=0.0 spawn_y:=0.0 spawn_z:=0.62 spawn_yaw:=0.0

ros2 launch tarantula_bringup nav_static.launch.py \
  localization_map:=$(pwd)/generated/terrains/nav_maze/42/map.yaml \
  terrain_cost_map:=$(pwd)/generated/terrains/nav_maze/42/terrain_cost_map.yaml \
  speed_mask:=$(pwd)/generated/terrains/nav_maze/42/terrain_speed_mask.yaml \
  cmd_vel_topic:=/diff_drive_controller/cmd_vel_unstamped \
  odom_topic:=/diff_drive_controller/odom
```

`nav_static.launch.py` defaults to the generated maze spawn at the world origin:
`initial_pose_x:=0.0 initial_pose_y:=0.0 initial_pose_a:=0.0`. Override those
only when `sim.launch.py spawn_x/spawn_y/spawn_yaw` changes. AMCL can also be corrected
manually:

```bash
ros2 topic pub --once /initialpose geometry_msgs/msg/PoseWithCovarianceStamped \
  "{header: {frame_id: map}, pose: {pose: {position: {x: 0.0, y: 0.0, z: 0.0}, orientation: {w: 1.0}}}}"
```

Launch online SLAM + Nav2 manually:

```bash
source /opt/ros/humble/setup.bash
source install/setup.bash
ros2 launch tarantula_bringup nav.launch.py \
  cmd_vel_topic:=/diff_drive_controller/cmd_vel_unstamped \
  odom_topic:=/diff_drive_controller/odom
```

Launch classical motion only:

```bash
source /opt/ros/humble/setup.bash
source install/setup.bash
ros2 launch tarantula_bringup sim.launch.py \
  gui:=true \
  robot_model:=tarantula_v3.urdf.xacro \
  motion_control:=true \
  start_motion_control:=true \
  posture_policy_enabled:=false \
  bridge_lidar:=false \
  wheel_collision:=sphere \
  spawn_z:=0.55
```

Launch active-suspension policy:

```bash
ros2 launch tarantula_bringup sim.launch.py \
  gui:=true \
  robot_model:=tarantula_v3.urdf.xacro \
  motion_control:=true \
  start_motion_control:=true \
  posture_policy_enabled:=true \
  policy_weights_npz:=$(pwd)/generated/policies/posture_actor.npz \
  bridge_force_torque:=true \
  bridge_lidar:=false \
  wheel_collision:=sphere \
  spawn_z:=0.55
```

Publish a low-speed command:

```bash
ros2 topic pub /cmd_vel geometry_msgs/msg/Twist \
  "{linear: {x: 0.10}, angular: {z: 0.0}}"
```

Run posture acceptance while Gazebo is running:

```bash
scripts/gazebo_posture_eval.py \
  --label no_rl_flat \
  --duration 30 \
  --cmd-vx 0.10 \
  --out-dir generated/benchmarks/posture_eval/no_rl_flat
```

For a route-based A/B comparison (same checkpoint sequence, RL on vs off),
use `gazebo_pursuit_eval.py` against an unwalled-interior world (e.g.
`rl_curriculum`, not `nav_maze` -- pure pursuit has no obstacle avoidance)
with `bridge_ground_truth_odom:=true`, then diff the two summaries:

```bash
scripts/gazebo_pursuit_eval.py --label no_rl --seed 7 \
  --out-dir generated/benchmarks/pursuit_eval/no_rl
# relaunch sim.launch.py with posture_policy_enabled:=true policy_weights_npz:=...
scripts/gazebo_pursuit_eval.py --label rl_active --seed 7 \
  --out-dir generated/benchmarks/pursuit_eval/rl_active
scripts/gazebo_eval_compare.py \
  generated/benchmarks/pursuit_eval/no_rl/summary.json \
  generated/benchmarks/pursuit_eval/rl_active/summary.json \
  --label-a no_rl --label-b rl_active
```

## Isaac Lab RL

Current policy contract:

```text
observation: 56D
  projected_gravity_b(3)
  root_ang_vel_b(3)
  susp_joint_pos(6)
  susp_joint_vel(6)
  wheel_joint_vel(6)
  wheel_force(18)
  contact_uptime(6)
  cmd_vx(1)
  cmd_wz(1)
  previous hip action(6)

action: 6D
  direct hip targets in fl/fr/ml/mr/rl/rr order
```

Train:

```bash
source /home/ang/isaac_venv/bin/activate
PYTHONPATH=src:src/tarantula_control \
python3 src/tarantula_isaac/train_v5.py \
  --num_envs 128 \
  --max_iterations 2000 \
  --terrain-dir "$(pwd)/generated/terrains/rl_curriculum/42" \
  --command-profile stage0 \
  --pursuit-prob 0.3
```

`--pursuit-prob` opts into pure-pursuit checkpoint-chasing commands
(default 0.0/off); friction and hip stiffness/damping domain randomization
are on by default (see docs/03-isaac-lab-setup.md). Training always samples
resets across the terrain's full difficulty range (`suspension_env.py`'s
`_reset_idx` re-rolls a random tile every reset) -- there's no
`--terrain-level-min/max` on this entry point anymore; an earlier version
quietly capped every run to the easiest difficulty row, which is what this
removed.

Export:

```bash
python3 src/tarantula_isaac/export_weights_v5.py \
  --checkpoint logs/rsl_rl/tarantula_suspension/<run>/model_399.pt \
  --npz-out generated/policies/posture_actor.npz
```

Smoke test:

```bash
scripts/run_rl_env_smoke_v5.sh
```

GUI smoke for Isaac/Gazebo alignment:

```bash
source /home/ang/isaac_venv/bin/activate
PYTHONPATH=src:src/tarantula_control \
python3 src/tarantula_isaac/gui_smoke.py \
  --terrain-mode heightmap \
  --terrain-dir generated/terrains/rl_curriculum/42 \
  --terrain-level-min 0 \
  --terrain-level-max 0 \
  --settle-seconds 1.0 \
  --drive-seconds 30.0 \
  --cmd-vx 0.20 \
  --cmd-wz 0.40
```

Add `--pursuit [--pursuit-checkpoints N]` to drive a pure-pursuit checkpoint
chase instead of a fixed cmd_vel (zero hip action throughout -- pursuit
steering is wheel-only, independent of the RL policy, so this isolates
whether checkpoint-chasing itself works on the shared terrain). Logs on every
checkpoint/command-mode transition plus a 5s heartbeat.

## Acceptance

The active-suspension policy is accepted only if it improves posture without hurting basic mobility:

- lower roll/pitch RMS or lower roll/pitch threshold time than no-RL posture;
- no visible hip jitter or initial yaw bias in Gazebo GUI;
- wheel load/contact support does not degrade;
- scan gate drop ratio does not increase;
- vehicle completes enough low-speed travel distance for the comparison to be meaningful.

SLAM/Nav stays on standard ROS2 interfaces. The accepted Gazebo/Nav2 demo path
uses official `diff_drive_controller` odom; the custom per-wheel path can feed
wheel odom + IMU into `robot_localization`. `slam_toolbox` publishes `/map` and
`map->odom` during online mapping.
