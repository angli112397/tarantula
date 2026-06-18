# Chassis v3 Baseline

当前底盘 baseline 是 `tarantula_v3.urdf.xacro`。

## Design Intent

v3 不重新发明运动系统。它保留已经验证过的六轮/六髋关节拓扑，但把叙事和优化重点转到主动悬挂：

- 传统差速/滑移转向负责移动；
- Nav2 demo 默认使用官方 `diff_drive_controller` 推断 `odom->base_link`；
- 自定义 per-wheel controller 的标定/姿态评估路径可通过 wheel encoder odom + IMU 进入 `robot_localization`；
- Nav2/SLAM 负责路径和地图；
- RL 只负责六个髋关节，让车身更稳、轮载更均衡、LiDAR 姿态更适合建图。

## Geometry Changes

v3 复用 v2 chassis macro，并覆盖少量关键几何参数：

- arm length: `0.22 -> 0.28 m`;
- arm height: `0.060 -> 0.055 m`;
- wheel lateral offset: `0.065 -> 0.070 m`.

这样增加髋关节姿态控制力臂，同时避免引入新的被动弹簧、rocker、额外接触自由度或复杂关节拓扑。

## Interfaces

Unchanged deployable interfaces:

- `/cmd_vel`;
- `/wheel_velocity_controller/commands`;
- `/suspension_controller/joint_trajectory`;
- `/imu/data`;
- `/joint_states`;
- `/ft_wheel/fl`, `/ft_wheel/fr`, `/ft_wheel/ml`, `/ft_wheel/mr`, `/ft_wheel/rl`, `/ft_wheel/rr`;
- `/scan`.

Navigation odometry:

- `/diff_drive_controller/odom`: current Nav2 smoke-test odom source;
- `/wheel/odom`: custom per-wheel controller odom source;
- `/odometry/filtered`: robot_localization EKF output for the custom per-wheel path;
- no Gazebo truth odom is used as a control, SLAM, Nav2, or policy input.

New policy runtime:

- `posture_policy_node`;
- loads only `50D/6D` active-suspension actors;
- publishes only hip targets;
- never publishes wheel targets.

## Acceptance Order

1. xacro generates URDF for `tarantula_v3.urdf.xacro`.
2. Gazebo spawn is stable on flat terrain.
3. Direct wheel/hip acceptance profiles still pass.
4. Classical `/cmd_vel` straight and pure-turn behavior remains usable.
5. SLAM/Nav smoke runs on the generated Nav2/Gazebo baseline maze.
6. No-RL posture baseline is recorded.
7. RL active suspension improves roll/pitch or scan stability without visible hip jitter.

## Explicit Non-Goals

- no passive rocker or spring visual/mechanism in the current baseline;
- no RL wheel speed residual;
- no terrain contact truth as policy input;
- no gait/foot placement controller;
- no claim of high-step or deep-trench traversal.
