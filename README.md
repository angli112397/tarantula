# Tarantula — 六轮主动避震底盘仿真（ROS2 Humble + Gazebo Classic 11）

六轮蜘蛛腿式悬挂差速底盘的全链路仿真：每腿一个摆臂悬挂关节 + 独立驱动轮，
**前馈式平衡点平移调平 + 天棚阻尼**（IMU 姿态外环 PI → 腿高度几何映射 →
前馈力矩平移物理弹簧平衡点；陀螺角速度天棚阻尼通道）实现车身自稳，
并基于 2D LiDAR + slam_toolbox 在线建图。

**实测（v3 定稿）**：8° 斜坡静态调平 **8.16°→0.09°（消除 98.9%）**、20s 指数收敛、
全程零自旋；平地 0.05° 零漂移；崎岖路 0.8 m/s 俯仰 RMS -27%。
算法选型、三代架构演进与实验定界详见 `docs/01-control-architecture.md`。

## 架构

```
tarantula_description   模块化底盘建模（产品是底盘，整车是配置）：
                        tarantula_chassis（底盘模块宏：prefix 可复用、
                        payload_mount 载荷位、IMU 固有，单一事实来源）
                        → tarantula_core（演示配置 = 底盘 + LiDAR 载荷，
                        lidar:=false 输出裸底盘，Isaac Lab 导入入口）
                        → tarantula（Gazebo 适配层：弹簧标签/插件/ros2_control）
tarantula_bringup       launch / 控制器配置 / 三个测试 world / SLAM / RViz 配置
tarantula_control       suspension_core（零 ROS 依赖算法核心，Isaac 直接
                        import）+ active_suspension（ROS 适配层）；
                        动作面 ~/body_cmd = [roll_ref, pitch_ref, height_m]
docs/                   01 控制架构选型报告 / 02 项目目标与里程碑（v2 修订）
scripts/                可复现实验：平地稳定 / 8°斜坡对照 / 崎岖路动态对照
```

控制链路：

```
                        ┌─ joint_state_broadcaster ──> /joint_states
gazebo_ros2_control ────┼─ diff_drive_controller  <── cmd_vel（六轮差速）
 (effort/velocity 接口) └─ suspension_controller  <── /suspension_controller/commands
                                                          ▲
/imu/data ──> active_suspension（100Hz）──────────────────┘
  外环：roll/pitch 各一 PI（条件积分抗饱和、0.5° 死区、输出限幅=行程界限）
  映射：q_target_i = q0 + DIR_i·(x_i·u_pitch − y_i·u_roll)/(L·cosθ₀)，斜率限制
  前馈：tau = k_spring·(q_target − q0 + dq_sky) —— 平移物理弹簧平衡点，
        无软件快环（话题延迟下软件位置 PD 会颤振，见 docs §6 v2→v3）
  天棚：dq_sky 由陀螺 roll/pitch 角速度映射（相位超前，耐延迟）
  安全：落地保持期零力矩 / 增益渐入 / 包络保护（倾角>20° 回名义位）
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

# 4. 验收实验（自动化，输出逐秒姿态/RMS 统计）
scripts/flat_stability_check.sh    # G1 平地
scripts/leveling_benchmark.sh      # G2 8°斜坡 主动/被动对照
scripts/dynamic_benchmark.sh       # G3 崎岖路 RMS 对照
```

地形沿 +x：减速带 ×2 → 左/右单侧台阶 → 斜坡-平台-斜坡 → 碎石区，四周围墙供
SLAM 回环。调试话题 `/active_suspension/debug`：[roll, pitch, u_roll, u_pitch,
q_target×6, q×6, tau×6]。

## 关键参数（active_suspension，均可 --ros-args -p 覆盖）

| 参数 | 默认 | 说明 |
|---|---|---|
| roll_kp / roll_ki | 0.8 / 1.2 | 侧倾外环 PI |
| pitch_kp / pitch_ki | 0.3 / 0.5 | 俯仰外环 PI（杠杆比 3.0，增益须折减） |
| att_deadband / att_out_limit | 0.009 / 0.22 | 外环死区（0.5°）/ 输出限幅 |
| ff_stiffness | 120 | 前馈刚度，必须等于 URDF springStiffness |
| sky_roll_damp / sky_pitch_damp | 0.15 / 0.12 | 天棚阻尼（>0.15 延迟致负阻尼翻车） |
| target_slew_rate | 0.20 | 平衡点斜率限制 rad/s（0.06/0.3 均实测翻车，勿改） |
| target_limit | 0.45 | 平衡点限幅（关节硬限位 0.6，0.55 撞限位翻车） |
| startup_hold / gain_ramp | 5.0 / 2.0 | 落地保持期（零力矩）/ 增益渐入 |
| tilt_freeze | 0.35 | 包络保护阈值（rad） |

## 验收状态（目标 v2，详见 docs/02-project-goals.md）

Phase 1 调平（已完成）：
- [x] G1 平地 0.05° / G2 静坡 8.16°→**0.09°**（98.9%）/ G3 pitch RMS -27% / G4 SLAM 成图

Phase 2 功能扩展（Gazebo）：
- [ ] M1 接触保持状态机（接触丢失时长 ≥40% 下降）
- [ ] M2 车身高度调节（roll/pitch/z 三维车身指令）
- [ ] M3 设计研究：TeCVP 吸收 + 2-DOF 构型 IK 工作空间论证

Phase 3 Isaac Lab（GPU 到货后）：
- [ ] M4 冒烟 / M5 v3 移植复现 / M6 kHz 延迟归因 / M7 RL 对照（冲刺项）

Phase 4 交付：
- [ ] 外观目检、演示录像（调平/高度/接触保持 A/B + SLAM）、GitHub 发布
