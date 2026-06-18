# Tarantula Active-Suspension Baseline Plan

日期：2026-06-18

本文是当前开发的 source of truth。旧的 RL 轮速补偿、轨迹纠偏、yaw/track scale/drive scale 补偿路线已经停止。

## Current Goal

项目叙事：

```text
ROS2/Nav2 owns motion, SLAM, and navigation.
Classical skid-steer owns /cmd_vel -> wheel speed.
RL owns only six hip targets as active suspension.
```

RL 的价值不是“替代驾驶”，而是在传统导航已经给出合理路径时，主动调节 6 个髋关节，降低 roll/pitch、改善轮载支撑、稳定 LiDAR/SLAM 输入。

## Baseline Chain

```text
tarantula_terrain.generate
  -> shared Gazebo/Isaac heightmap
  -> aligned Nav2 occupancy maze layer
  -> tarantula_v3.urdf.xacro
  -> Gazebo direct chassis acceptance
  -> wheel encoder + IMU odometry through robot_localization
  -> classical /cmd_vel skid-steer motion
  -> SLAM/Nav2 standard integration smoke
  -> 6D posture eval: no-RL vs RL active suspension
  -> Isaac Lab active-suspension training
  -> Gazebo GUI acceptance
  -> terrain difficulty increase
```

## Traversability Envelope

The current project targets可滚过复杂路况, not full legged locomotion.

Accepted terrain class:

- rough heightmap, shallow pits, low bumps, low steps;
- gentle slopes and lateral slopes;
- corridors wide enough for Nav2 footprint + inflation;
- low-speed operation where wheel contact remains the primary locomotion mode.

Out of scope for this baseline:

- deep trenches, high steps, wheel-high climbing;
- terrain that requires foot placement or gait planning;
- policy rescuing an invalid Nav2 route through non-traversable terrain.

## Map Layer Contract

Navigation demo maps are generated as two aligned grayscale layers:

```text
height.npy / height.png
  terrain elevation layer for Gazebo and Isaac Lab

occupancy.npy / map.pgm
  2D obstacle layer for Nav2 and Gazebo wall generation
```

The current `nav_maze` default is `24m x 16m @ 0.10m`, matching
`rl_curriculum`. This keeps future composition simple: height and occupancy
share the same origin, resolution, shape, and world coordinates. Gazebo can
render height as terrain and occupancy as wall boxes; Nav2 consumes the
occupancy map; Isaac can later consume the same obstacle mask if needed.

The navigation maze is a large-chassis demo baseline, not a final clearance
stress test: default doors are `5.2m`, nominal corridors are `4.8m`,
`obstacle_count=3`, and a `4.8m x 4.8m` center safety pad is carved out around
the origin spawn. The physical URDF remains the source of truth for vehicle
collisions, while the Nav2 demo costmaps use a v3-derived navigation footprint
(`1.12m x 0.74m`) with zero extra padding.
`map.pgm` preserves Gazebo/world x columns and flips only image rows when
exporting, matching ROS `map_server`'s top-row image convention while keeping
world `x/y` coordinates aligned with the SDF wall boxes.

Current Nav2 tuning policy:

- prefer wider generated clearances over further shrinking the navigation footprint;
- use conservative inflation/cost tuning to bias paths toward corridor centers;
- keep online SLAM and static-map Nav2 both available;
- treat `Starting point in lethal space` after recovery as a map/costmap
  clearance failure, not an RL or vehicle-control problem.

Current status:

- static `map_server + amcl + Nav2` loads the generated `map.yaml`, uses origin
  AMCL initialization, and executes navigation actions in Gazebo;
- online `slam_toolbox + Nav2` publishes `/map` from `/scan_gated`;
- `scripts/nav2_frontier_explore.py` has completed repeated automatic frontier
  goals in the baseline maze;
- the current Gazebo/Nav2 demo baseline uses the official
  `diff_drive_controller` for `/cmd_vel -> wheels` and `/diff_drive_controller/odom`.

## Vehicle Contract

Current model baseline:

- `tarantula_v3.urdf.xacro`;
- v3 reuses the proven v2 topology and lengthens the wheel arms for more posture authority;
- six position-controlled hip joints: `susp_fl/fr/ml/mr/rl/rr_joint`;
- six velocity-controlled wheel joints: `wheel_fl/fr/ml/mr/rl/rr_joint`;
- wheel collision baseline is spherical;
- LiDAR, IMU, wheel F/T, joint states, and wheel velocities are deployable signals;
- `/wheel/odom` is inferred from wheel encoders;
- `/odometry/filtered` and `odom->base_link` are owned by robot_localization.

Leg order is always:

```text
fl/fr/ml/mr/rl/rr
```

## Control Contract

Motion:

```text
/cmd_vel
  -> official diff_drive_controller for Nav2 demo
  -> wheel joints
```

Custom chassis calibration path:

```text
/cmd_vel
  -> motion_control_node
  -> /wheel_velocity_controller/commands
  -> /wheel/odom -> robot_localization -> /odometry/filtered
```

Active suspension:

```text
IMU + joint states + wheel velocity + wheel F/T + shaped cmd + previous action
  -> 50D observation
  -> 6D RLPosturePolicy
  -> /suspension_controller/joint_trajectory
```

The policy must never publish wheel commands and must never alter `/cmd_vel`.

## RL Contract

Observation space: 50D.

```text
projected_gravity_b(3)
root_ang_vel_b(3)
susp_joint_pos(6)
susp_joint_vel(6)
wheel_joint_vel(6)
wheel_force_b(18)
exec_cmd_vx(1)
exec_cmd_wz(1)
prev_hip_action(6)
```

Action space: 6D.

```text
action[0:6] = direct hip position targets in fl/fr/ml/mr/rl/rr order
hip_target = clamp(action, -1, 1) * hip_action_target_limit
```

Reward intent:

- reward low roll/pitch;
- penalize roll/pitch angular rate;
- reward enough loaded wheels and balanced wheel load;
- penalize vertical bounce, stuck behavior, hip target rate, and hip soft-limit approach;
- alive bonus and fall/stability termination penalty.

No reward term should ask RL to improve wheel speed, yaw tracking, path tracking, or effective track scale.

## Evaluation Plan

Gazebo posture eval should compare:

```text
classical motion + neutral/profile posture
vs
classical motion + RL active suspension
```

Metrics:

- roll RMS / pitch RMS;
- max roll / max pitch;
- roll/pitch threshold time ratio;
- wheel load balance;
- loaded wheel count;
- hip target RMS/rate;
- scan gate drop ratio when LiDAR demo is enabled;
- completed distance and stuck/fall flags.

The route does not need strict trajectory tracking. It only needs enough low-speed travel distance to make posture statistics meaningful.

## Development Rings

1. v3 model smoke: xacro, spawn, stable neutral posture.
2. Classical motion smoke: straight, reverse, pure turn, low-speed Nav2 commands.
3. SLAM/Nav smoke: generated grayscale maze, standard Nav2 config, no custom motion research.
   Static-map Nav2 and online SLAM/frontier exploration are connected on the
   generated baseline maze.
4. Posture eval baseline: neutral/no-RL vs scripted posture on flat and mild terrain.
5. 6D RL smoke: Isaac reset, train short run, export actor, load in Gazebo.
6. RL posture acceptance: lower roll/pitch without visible jitter or motion degradation.
7. Model outer-loop tuning: arm length, hip limit, wheel radius, COM, contact parameters.
8. Terrain curriculum: increase roughness only after the previous ring passes.

## Hard Rules

- Do not reintroduce RL wheel residuals without explicitly changing this plan.
- Do not use Gazebo model truth odom as a controller, policy, SLAM, or Nav2 input.
- Do not judge RL only by reward; Gazebo GUI posture and motion behavior are acceptance gates.
- Keep Nav2/SLAM implementation standard and lightweight.
