# Tarantula — 六轮主动悬挂底盘仿真（ROS2 Humble + Gazebo Sim / Isaac Lab）

六轮蜘蛛腿式悬挂差速底盘的全链路仿真：每腿一个摆臂悬挂关节 + 独立驱动轮。

两条控制路径（可按需切换）：
- **经典路径**（`leveling:=true`）：IMU 外环 PI → 几何映射 → 前馈力矩平移弹簧平衡点。实测：8° 斜坡调平 98.9%、崎岖路俯仰 RMS -27%。
- **RL 路径**（`rl_policy:=true`）：34D obs → PPO actor → 几何映射 → 关节 PD 直驱。Isaac Lab 训练（domain rand: 摩擦/质量/推力扰动），Gazebo 部署验证。

基于 2D LiDAR + slam_toolbox 在线建图，Nav2 自主导航穿越崎岖地形。
算法选型、架构演进与实验定界详见 `docs/01-control-architecture.md`；项目目标与 RL 主线见 `docs/02-project-goals.md`。

## 架构

```
tarantula_description   模块化底盘建模（产品是底盘，整车是配置）：
                        tarantula_chassis（底盘模块宏：prefix 可复用、
                        payload_mount 载荷位、IMU 固有，单一事实来源）
                        → tarantula_core（演示配置 = 底盘 + LiDAR 载荷，
                        lidar:=false 输出裸底盘，Isaac Lab 导入入口）
                        → tarantula（Gazebo Ignition 适配层：弹簧标签/传感器
                        /ros2_control via gz_ros2_control）
tarantula_bringup       launch / 控制器配置 / 测试 world×4 / SLAM / Nav2 / RViz 配置
tarantula_control       suspension_core（零 ROS 依赖算法核心，Isaac 直接
                        import）+ active_suspension（ROS 适配层）+
                        scan_gate（姿态门控扫描过滤，崎岖地形 2D 感知守门员）+
                        rl_suspension_policy（M7 v4 PPO actor 推理节点，
                        rl_policy:=true，34D obs，直接几何映射绕过 SuspensionController）
tarantula_isaac         IsaacLab DirectRLEnv（M7 v4：34D obs，5D action，
                        kinematic mapping 直驱关节，domain rand，docs/04）
docs/                   01 控制架构选型报告 / 02 项目目标与里程碑（v3 修订）/
                        03 Isaac Sim/Lab 环境搭建 / 04 M7 RL 训练+集成记录
scripts/                可复现实验：平地稳定 / 8°斜坡对照 / 崎岖路动态对照 /
                        自动化导航任务（nav_mission.py）
```

控制链路：

```
                        ┌─ joint_state_broadcaster ──> /joint_states
gz_ros2_control ────────┼─ diff_drive_controller  <── cmd_vel（六轮差速）
 (effort/velocity 接口) └─ suspension_controller  <── /suspension_controller/commands
                                                          ▲
/imu/data ──┬─> active_suspension（100Hz，算法见 suspension_core.py）
~/body_cmd ─┤     外环：roll/pitch 各一 PI（条件积分抗饱和、0.5° 死区、输出限幅=行程界限）
/ft/* ──────┘     映射：dz_i = x_i·u_pitch − y_i·u_roll + z_cmd
                  q_target_i = q0 + DIR_i·dz_i/(L·cosθ₀)，限幅+斜率限制
                  前馈：tau = k_spring·(q_target − q0 + dq_probe)
                        —— 平移物理弹簧平衡点，无软件快环（话题延迟下
                        软件位置 PD 会颤振，见 docs §6 v2→v3）
                  M2：z_cmd 来自 body_cmd 的 height（±0.06m，默认 0 = 等价 v3）
                  M1：dq_probe 为接触保持下探量（默认关闭，contact_keeping=false）
                  安全：落地保持期零力矩 / 增益渐入 / 包络保护（倾角>20° 回名义位）
```

`/imu/data`、`/scan`、`/ft/{fl,fr,ml,mr,rl,rr}` 与 `/clock`（use_sim_time）
均由 `sim.launch.py` 内的 `ros_gz_bridge parameter_bridge` 从 gz 话题单向桥接到
ROS（`gz.msgs.IMU/LaserScan/Wrench/Clock` → `sensor_msgs`/`geometry_msgs`），
`/imu` 在桥接节点里 remap 为 `/imu/data`。每路 `/ft/{leg}` 是该腿摆臂关节对车身的
反作用力（车身坐标，frame=parent），`force.z` 即支撑力，供 active_suspension
做 M1 接触判据（`force.z > contact_force_threshold`）。

控制核心 `suspension_core.py` 零 ROS 依赖，三个暴露面：
**参数面** `SuspensionConfig`（全部可调参数）、**观测面** `SuspensionInputs`
（姿态/关节角/轮地接触）、**动作面** `roll_ref/pitch_ref/height_cmd`
（车身位姿指令，复用几何映射，天然有界）。Isaac Lab 直接 import 此模块。

## 依赖

```bash
sudo apt install ros-humble-ros-gz ros-humble-gz-ros2-control \
  ros-humble-ros2-controllers ros-humble-joint-state-broadcaster \
  ros-humble-diff-drive-controller ros-humble-effort-controllers \
  ros-humble-controller-manager \
  ros-humble-slam-toolbox \
  ros-humble-navigation2 ros-humble-nav2-bringup
```

## 构建（本机注意事项）

```bash
cd ~/tarantula_ws
source /opt/ros/humble/setup.bash
# miniconda 的 python 会破坏 colcon 构建，必须先从 PATH 剔除
export PATH=$(echo "$PATH" | tr ':' '\n' | grep -v miniconda | paste -sd:)
# 本机内存有限，务必单 worker
colcon build --symlink-install --parallel-workers 1 --executor sequential
source install/setup.bash
```

## 运行

```bash
# 1. 仿真（gui:=false 无界面；leveling:=false 纯被动对照；spawn_x/y/z 出生点）
ros2 launch tarantula_bringup sim.launch.py

# 2. 键盘遥控（新终端，记得 source）
ros2 run teleop_twist_keyboard teleop_twist_keyboard \
  --ros-args -r cmd_vel:=/diff_drive_controller/cmd_vel_unstamped

# 3. SLAM + RViz
ros2 launch tarantula_bringup slam.launch.py
rviz2 -d src/tarantula_bringup/config/slam.rviz

# 4. SLAM + Nav2 自主导航（综合测试场，RViz 里 2D Goal Pose 下发目标）
ros2 launch tarantula_bringup sim.launch.py \
  world:=$(ros2 pkg prefix tarantula_bringup)/share/tarantula_bringup/worlds/proving_ground.world
ros2 launch tarantula_bringup nav.launch.py       # 新终端：slam_toolbox + Nav2
rviz2 -d src/tarantula_bringup/config/nav.rviz    # 新终端

# 5. 验收实验（自动化，输出逐秒姿态/RMS 统计）
scripts/flat_stability_check.sh    # G1 平地
scripts/leveling_benchmark.sh      # G2 8°斜坡 主动/被动对照
scripts/dynamic_benchmark.sh       # G3 崎岖路 RMS 对照
```

`rough_terrain.world`（基准实验场）：地形沿 +x：减速带 ×2 → 左/右单侧台阶 →
斜坡-平台-斜坡 → 碎石区，四周围墙供 SLAM 回环。

`proving_ground.world`（SLAM/Nav2/悬挂综合测试场）：外墙 20×14 + 中央房间构成
环形回廊，从西侧 1.6m 窄门进环，绕环一周回到窄门触发回环闭合；四段走廊
各承担一类测试，地形横贯走廊无法绕行——南廊减速带/碎石/左右交替单侧台阶
（侧倾冲击）、东廊斜坡-平台-斜坡（俯仰调平）、北廊立柱迷阵+箱体（局部避障）。
尺度设计：墙/柱高 0.8 > 雷达扫描面 ~0.5 > 平台 0.275，即地形对 2D 雷达不可见
（costmap 视为自由空间）——**穿越地形交给悬挂，避障交给雷达**；车身在坡上
倾斜时扫描面打地会产生幻影障碍，主动调平保持扫描面水平正是抑制点。

Nav2 集成（`nav.launch.py` + `config/nav2.yaml`）：自建最小 bringup
（controller/planner/behaviors/bt_navigator + lifecycle_manager），不走
nav2_bringup —— 需要把 cmd_vel remap 到 `/diff_drive_controller/cmd_vel_unstamped`
而不改机器人侧接口。map→odom 由 slam_toolbox 提供（无 AMCL，边建图边导航）。
关键取舍（均为实测定界，调试记录见 git log）：
- **scan_gate 姿态门控**：车身倾斜>3° 时丢弃整帧扫描——倾斜瞬间 2D 扫描面
  打地，会在 2-3m 外画出横贯走廊的幻影障碍（实测把局部 costmap 堵死、
  导航死锁）。与主动调平天然协同：调平把车身长期压在阈值内；
- **DWB ObstacleFootprint 评价器**（而非 BaseObstacle）：1.32×0.78 长方形
  车体过 1.6m 窄门，只查车体中心点代价会斜切贴墙（实测离门墩 3cm，
  全局规划起点落入致死区）；footprint 多边形同理（圆半径把窄门膨胀死）；
- **Smac 2D 规划器**（而非 NavFn）：边建图边导航时目标常在未知区深处，
  NavFn 的势场梯度提取在均匀未知区会失败（实测复现其 "This shouldn't
  happen" 错误路径）；
- **全向限转速 0.5-0.6 rad/s**：滑移转向原地转大量打滑，转速过高时 10Hz
  扫描匹配跟不上里程计角度误差（实测 1.0 rad/s 自旋恢复后 map 系转歪
  102°）；允许倒车（min_vel_x=-0.2）作为贴墙死位的逃逸自由度；
- 全局 costmap 只挂静态层+膨胀层：瞬态幻影若写入全局会永久堵路，
  局部 costmap 的 raytrace 清除可自愈。

实测（proving_ground，单 SLAM 会话连续任务）：穿 1.6m 窄门到位误差
0.36m → 连续横穿南廊全部地形（减速带×2 + 碎石场 + 左右单侧台阶，
scan_gate 期间丢弃倾斜帧）→ 平地终点到位误差 0.43m，全程定位一致。
自动化复现：`scripts/nav_mission.py "3.0,-4.5,-1.57" "14.5,-4.5,0"`。

车身位姿指令（M2，默认零=等价 v3）：
```bash
ros2 topic pub /active_suspension/body_cmd std_msgs/msg/Float64MultiArray \
  "{data: [0.0, 0.0, 0.06]}"   # [roll_ref, pitch_ref, height_m]
```

调试话题 `/active_suspension/debug`（35 个字段）：
`[roll, pitch, u_roll, u_pitch, q_target×6, q×6, tau×6, contact×6, probe_dz×6, height]`。

## 关键参数（active_suspension，均可 --ros-args -p 覆盖）

| 参数 | 默认 | 说明 |
|---|---|---|
| roll_kp / roll_ki | 0.8 / 1.2 | 侧倾外环 PI |
| pitch_kp / pitch_ki | 0.3 / 0.5 | 俯仰外环 PI（杠杆比 3.0，增益须折减） |
| att_deadband / att_out_limit | 0.009 / 0.22 | 外环死区（0.5°）/ 输出限幅 |
| ff_stiffness | 120 | 前馈刚度，必须等于 URDF springStiffness |
| target_slew_rate | 0.20 | 平衡点斜率限制 rad/s（0.06/0.3 均实测翻车，勿改） |
| target_limit | 0.45 | 平衡点限幅（关节硬限位 0.6，0.55 撞限位翻车） |
| startup_hold / gain_ramp | 5.0 / 2.0 | 落地保持期（零力矩）/ 增益渐入（0=无渐入立即满增益） |
| tilt_freeze | 0.35 | 包络保护阈值（rad） |
| height_limit | 0.06 | M2 车身高度指令限幅 m（默认指令 0，行为=v3） |
| contact_keeping | false | M1 接触保持开关（默认关，行为=v3） |
| contact_debounce | 0.10 | M1 悬空消抖时间 s |
| contact_force_threshold | 5.0 | M1 接触判据：/ft/{leg} force.z > 此值视为着地 N |
| probe_slew / probe_limit | 0.10 / 0.05 | M1 下探/撤回速率 m/s ÷ 单腿下探上限 m |

## 路线图（详见 docs/02-project-goals.md）

调平（已完成）：
- [x] G1 平地 0.05° / G2 静坡 8.16°→**0.09°**（98.9%）/ G3 pitch RMS -27% / G4 SLAM 成图

功能扩展（Gazebo）：
- [x] 控制核心重构为仿真器无关的 suspension_core + ROS 适配层
- [x] SLAM + Nav2 集成：proving_ground 综合测试场端到端验证
      （窄门/全地形横穿/自主导航，姿态门控扫描过滤）
- [ ] M1 接触保持状态机：算法已实现（默认关闭），合成信号验证通过，
      待接入 Gazebo 接触传感器调参验收（接触丢失时长 ≥40% 下降）
- [ ] M2 车身高度调节：算法已实现并在 Gazebo 验证方向正确（默认指令零），
      待调参验收（roll/pitch/z 三维车身指令，8° 坡调平精度不退化）
- [ ] M3 设计研究：TeCVP 吸收 + 2-DOF 构型 IK 工作空间论证

Isaac Lab（GPU 到货后，进行中，详见 docs/03、docs/04）：
- [x] M4 冒烟测试（core URDF 导入 + 等效关节驱动）
- [x] M7 RL 调平策略：环境/训练/导出/Gazebo 集成已完成（`rl_policy:=true`），
      Isaac 内 16-env 对照 mean tilt 0.0066→0.0057；Gazebo 部署发现"动作
      空间角点卡死"问题，排查中（docs/04）
- [ ] M5 v3 移植复现 / M6 kHz 延迟归因

交付：
- [ ] 外观目检、演示录像（调平/高度/接触保持 A/B + SLAM）、GitHub 发布
