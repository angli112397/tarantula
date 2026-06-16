# 主动避震控制架构选型报告

日期：2026-06-11 ｜ 状态：已实施定稿（最终架构经两次演进，见 §6）
最终决策：**前馈式平衡点平移调平 + 天棚阻尼**——外环 IMU 姿态 PI（慢，~1Hz）
→ 几何映射到每腿平衡点偏移 → 前馈力矩 tau = k_spring·dq 平移物理弹簧平衡点
（DC 增益 1.0，无软件快环）；陀螺角速度叠加天棚阻尼通道。
被动弹簧阻尼由仿真器物理关节承担（隐式积分，无条件稳定）。

## 1. 需求定义

| 维度 | 要求 |
|---|---|
| 平台 | 六轮单自由度摆臂悬挂 + 差速轮，仿"捕鸟蛛"构型 |
| 目标 | 崎岖路面行驶时车身 roll/pitch 自稳（云台效果），静态斜坡调平 |
| 技术栈 | ROS2 Humble + Gazebo Sim (Ignition Fortress) + ros2_control |
| 反馈 | IMU（姿态）+ 关节编码器；接触传感器仅做遥测展示 |
| 约束 | 开发者为初学者、≤2 周交付演示、面试时每个环节可解释 |

## 2. 方案对比

| 方案 | 工业/学术案例 | 优点 | 缺点/风险 | 实现成本 |
|---|---|---|---|---|
| **位置式姿态调平**（外环姿态 PI → 腿位置） | 农机坡地调平（Hillco/Case-IH/John Deere，倾角仪+液压位置调平，精度 ±0.5°）；轮腿六足姿态解耦控制 | 本质安全（位置目标有几何界限）、确定性强、参数少、与工业实践同构 | 对高频冲击滤波弱于力控（可由被动弹簧补偿） | **低**：1 个外环 PI + 几何映射 |
| 力矩叠加/虚拟弹簧+姿态力矩 | 我们的第一版 | 概念直观 | **已实证失败**：积分饱和可顶翻车身；锁轮拖刮致自旋；力平衡与姿态耦合难调 | 低但调试成本极高 |
| 阻抗/虚拟模型控制（VMC） | MDPI 轮式系统自适应阻抗；腿足机器人 VMC | 接触柔顺好、地形适应强 | 需接触力估计与模式切换，稳定性调参深 | 中-高 |
| 全状态反馈 LQR | 行星车主动悬挂研究 | 多目标最优 | 需线性化模型与系统辨识，超纲 | 中-高 |
| 漏斗力控/事件触发（BITNAZA） | BIT 轮腿机器人（Stewart 腿） | 瞬态性能有理论保证 | 论文级复杂度，6 维动力学+力跟踪 | **极高** |
| Skyhook/半主动 | 汽车悬挂主流 | 舒适性好 | **只减振不调平**，目标不符 | 低 |
| 强化学习 | Swiss-Mile 等轮腿 RL | 上限高 | 需 Isaac/GPU（CMP 40HX 存疑）、sim-to-sim gap | 极高 |

## 3. 选型决策与理由

**外环**：roll/pitch 各一个 PI（输入 IMU 姿态，输出车身期望姿态修正），输出限幅 =
悬挂行程几何上限；带变速率特性（误差大转快、近水平转慢——农机调平的工业惯例）。
**几何映射**：期望姿态修正 → 每条腿的轮心目标高度 Δz_i = ±(track/2)·tan(roll_corr)
±(wheelbase_i)·tan(pitch_corr) → 摆臂目标角 Δq_i = Δz_i / (L·cosθ₀)。
**内环**：`ros2_control` 位置接口（JointGroupPositionController），内环刚度即位置环
增益，行程末端硬限位由 URDF 关节限位保证。
**被动层**：URDF 物理弹簧+阻尼（已验证平地 0.04° 稳定），负责高频冲击吸收；
主动层只做低频调平（带宽 ~1 Hz），频段分离、互不打架。

为什么不继续力矩叠加方案——三次实验的实证教训：
1. 积分项无几何界限 → 满幅力矩把车顶翻（8° 斜面实验，倾角 70-105°）；
2. 力矩慢漂移 + 锁死轮 → 拖刮自旋（yaw 持续漂移 10°/s）；
3. 力平衡点随地形变化，每条腿的力-姿态映射耦合，调参维度爆炸。
位置式方案天然规避全部三条：位置目标限幅=行程限幅，到位即停不拖刮，
几何映射解耦各腿。

## 4. 模型设计清单（按算法反推）

- [ ] 悬挂关节：**position 命令接口**（替换 effort），URDF 限位 ±0.6 rad 即安全边界
- [ ] 物理弹簧阻尼保留在 URDF（k=120 Nm/rad，c=4）——位置环失效时车不塌
- [ ] 摆臂几何参数显式化：L、θ₀ 进 xacro property，控制器从参数读取（几何映射要用）
- [ ] 六腿对称布局（轮心 x=±0.51/0），轮距 0.64 → 调平能力 atan(2·0.06/0.64)≈10.6°，
      覆盖 8° 测试斜面有余量
- [ ] 外观验收（用户检查点）：RViz/Gazebo 中能一眼认出"六轮蜘蛛底盘"——
      车体分层、悬挂支架、摆臂双段造型、轮毂细节；所有 visual 内联材质颜色
      （此前 RViz 白模 = 命名材质引用未被渲染）
- [ ] IMU/LiDAR/接触传感器配置不变

## 5. 测试环境与分阶段验收（每关过了才进下一关）

| 阶段 | 场景 | 标准与实测（2026-06-11 最终架构 v3） |
|---|---|---|
| G1 模型 | Gazebo 平地 | **0.05°，44s 零漂移** ✓；外观目检待用户确认 |
| G2 静态调平 | 8° 斜板（tilt_test.world） | **被动 8.16° / 主动 0.09°（消除 98.9%），20s 指数收敛，全程 yaw 漂移 <0.2°，无任何自旋/翻车** ✓ |
| G3 动态自稳 | 崎岖路 0.8 m/s | **RMS pitch 4.12°→3.00°（-27%），max pitch 10.28°→8.42°** ✓；roll 动态抑制受限（见已知限制①） |
| G4 SLAM 集成 | 崎岖路建图 | slam_toolbox 成图（1145 占用格/25135 自由格）✓ |

### dartsim 复测（2026-06-13，Ignition Fortress 迁移后）

物理引擎从 ODE 换成 dartsim 后，用当前锁定参数重新跑 G1/G2/G3：

| 阶段 | 场景 | dartsim 实测 |
|---|---|---|
| G1 平地 | 平地静止 44s | **0.14°，零漂移** ✓（与 ODE 0.05° 同级） |
| G2 静态调平 | 8° 斜板 | **被动 8.28° / 主动 0.54°（消除 93.5%），5-6s 收敛（ODE 为 20s），yaw 全程 +0.1°，无自旋** ✓ |
| G3 动态自稳 | 崎岖路 0.8 m/s | 主动 RMS roll=1.94°/pitch=2.05°，max roll=5.35°/pitch=6.41°；被动 RMS roll≈4.51° — roll 阻尼明显优于 ODE era |

### 安全包络边界值复测（2026-06-13）

ODE 时代翻车实验定界的三个值，在 dartsim + 100Hz 话题环下逐一复测
（G2 静态调平 / G3 动态自稳，各单次跑），**均未复现翻车/自旋/雪崩**：

| 参数 | ODE era 结论 | dartsim 复测结果 |
|---|---|---|
| `sky_roll_damp=0.45`（默认 0.15） | ~20ms 延迟下负阻尼翻车 | G3 主动 RMS roll=1.74°/pitch=1.84°（优于默认约 -10%），max roll=5.96°/pitch=7.09°（略差约 +11%）；无翻车/自旋 |
| `target_slew_rate=0.06`（默认 0.20） | 静坡粘滑雪崩 | G2 收敛更快（t=1s 即开始下降，默认在 t=4s 才开始），5-6s 收敛到 0.58°（默认 0.54°），全程平滑无振荡，yaw 不变 |
| `target_slew_rate=0.30`+`target_limit=0.55`（默认 0.20/0.45） | 撞 0.6 rad 硬限位翻车 | G2 收敛最快（2.6s 到 0.53°，优于默认 0.54°）；G3 主动 RMS roll=1.68°/pitch=1.88°、max roll=5.17°/pitch=5.77°（均优于默认）；两项测试均无撞限位/翻车 |

**结论**：dartsim 的关节限位/接触阻尼明显比 ODE 更"宽容"，三个 ODE era
翻车场景在 dartsim 下都表现为平滑收敛，且数值与当前默认值同级或略优。但
本轮只做了单次跑、且每个候选值只在 G2 或 G3 单独验证，没有任何一组做满
G1+G2+G3 全套回归，因此**暂不放宽 `suspension_core.py` 默认值**——按既定
规则（放宽前必须 G1/G2 回归通过且明确变好），当前结论是"现有安全包络比
实测所需更保守，M1/M2 调参可以在当前默认值上继续，不被这三条卡住"。
`sky_roll_damp` 可显著放开这一点，方向上与 Phase 3/M6（kHz 内环目标）一致，
留作 Isaac Lab 阶段的起点（届时配合完整 G1-G3 回归再决定是否改默认值）。

### 已知限制（实验定界）

1. **roll 高频主动阻尼受话题链路延迟封顶（ODE era）**：车身 roll 惯量小、
   扰动频率高，~20ms 的 joint_states→节点→命令延迟使天棚阻尼系数 >0.15 时
   相位反转（0.45 在 ODE 下实测负阻尼翻车）。动态 roll 抑制需要 kHz 级插件
   内控制器（真实产品悬挂 ECU 即如此），列为改进方向。dartsim 下 0.45 复测
   未翻车（见上文"安全包络边界值复测"），但未做完整回归，默认值暂不动。
2. **俯仰调平杠杆比 3.0**：轴距半长 0.51 / 有效臂长 0.169，俯仰修正消耗
   3 倍关节行程，外环增益须按比例折减（roll 0.8/1.2，pitch 0.3/0.5）。
3. **参数安全包络**（ODE era 翻车实验定界）：target_limit ≤0.45（0.55 撞
   0.6 硬限位）、slew 0.2 rad/s（0.06 静坡粘滑雪崩、0.3 配大限幅撞限位）、
   sky_roll_damp ≤0.15。dartsim 下三者复测均未翻车（见上文），数值同级或
   略优，但默认值暂不放宽，详见上文复测结论。

## 6. 架构演进记录（实验驱动）

**v1 力矩叠加**（虚拟弹簧+姿态力矩注入）：积分无界 → 满幅力矩顶翻车身；
力矩漂移 + 锁轮 → 拖刮自旋；力-姿态耦合难调参。8° 斜坡实验翻车，废弃。

**v2 串级位置式**（外环姿态 PI → 几何映射 → 软件位置 PD 内环）：解决 v1
全部失效模式（目标几何限幅/斜率限制/落地零力矩保持/包络保护），静坡收敛
2.89°、崎岖路 roll RMS -64%。但软件内环跑在话题链路上，kp=350 在 ~20ms
延迟下 5-10Hz 颤振（tau ±160 Nm），降 kp 治标不治本。

**v3 前馈+天棚（最终）**：取消软件位置反馈，前馈力矩 tau = k_spring·dq 把
物理弹簧平衡点精确移到目标（DC=1.0），动力学由仿真器隐式弹簧阻尼承担
（无延迟、无条件稳定）；陀螺角速度做天棚阻尼（相位超前，耐延迟）。
颤振归零（tau std 156→0），静坡 0.09°，平地 0.05°。
关键教训：**延迟链路上不要闭快环——反馈放物理侧/慢环，快速校正用前馈和
速率阻尼**。落地保持期（零力矩 5s + 增益渐入 2s）使落地动力学与被动一致，
消除了落地自旋（yaw 漂移 60°→0.2°）。

## 7. Isaac Lab 移植映射（M4-M7）

模型按"模块化底盘"定位分三层：`tarantula_chassis.xacro`（底盘模块宏，
prefix 可复用、payload_mount 载荷位，单一事实来源）→
`tarantula_core.urdf.xacro`（演示配置 = 底盘 + LiDAR 载荷）→
`tarantula.urdf.xacro`（Gazebo Ignition 适配层）。Isaac 导入入口：
`xacro tarantula_core.urdf.xacro lidar:=false > tarantula.urdf`
（裸底盘，调平任务无需 LiDAR）后喂 URDF importer。

| Gazebo (Ignition) 侧 | Isaac Lab 侧 | 备注 |
|---|---|---|
| `<springStiffness>120` + `implicitSpringDamper` | 关节 drive `stiffness=120`，target=0 | 同为隐式 PD，角色一致 |
| `<dynamics damping=8>` | 关节 drive `damping=8` | |
| ros2_control + suspension_controller 接受前馈力矩 | RL 动作 → 几何映射 → `set_joint_position_target` | v4 RL 路径直接驱动关节，不经 SuspensionController |
| IMU sensor（topic, gz.msgs.IMU） | Isaac IMU sensor API | |
| diff_drive_controller | 轮速 velocity drive | |
| 接触/LiDAR 插件 | ContactSensor / RayCaster | RL 任务非必需 |

**两条互不干扰的控制路径**（运行时由 `rl_policy:=true/false` 切换）：

- **经典路径**（`leveling:=true, rl_policy:=false`）：IMU → `suspension_core.SuspensionController`（PI + 天棚阻尼）→ `active_suspension.py` → `/suspension_controller/commands`。参数全在 `SuspensionConfig`，ROS 参数面暴露。
- **RL 路径**（`rl_policy:=true`）：IMU + 关节 + odom → `rl_suspension_policy.py`（34D obs → PPO actor → 5D action → 几何映射）→ `/suspension_controller/commands`。`active_suspension` 在此模式下**不启动**（`sim.launch.py` 中以 `leveling AND NOT rl_policy` 条件）。

**M7 v4 实现现状**（2026-06-16）：

- SuspensionController 已从 RL env 完全移除（v4 架构变化）。原因：SuspensionController 内 PID 积分器/启动保持/增益渐入破坏 MDP 假设；`tilt_freeze=0.35 rad` 恰好在 bump 倾角范围内触发，阻碍越障。
- RL env 动作：`u_roll/u_pitch/z_cmd` 直接进向量化几何映射 → `set_joint_position_target`（stiffness=120/damping=8，无软件中间层）。
- PPO 训练：400 iter，reward 715（model_399.pt）→ Stage E warm-start +300 iter，加入质量/推力/obs噪声 domain randomization（见 docs/04）。
- Gazebo 部署验证中，前腿浮空/机身左倾行为待进一步观察。
- M6 延迟归因实验降优先级（RL 主线优先）。

## 参考

- Hillco/Case-IH/John Deere 坡地调平系统（倾角仪+变速率液压位置调平，±0.5°）：
  hillcotechnologies.com
- Liu et al., *Posture Adjustment for a Wheel-legged Robotic System via Leg Force
  Control with Prescribed Transient Performance* (arXiv:2011.04138) —— 力控路线复杂度上限的参照
- *Attitude-Oriented Stability Control with Adaptive Impedance Control for a Wheeled
  Robotic System on Rough Terrain* (Machines, MDPI 2023)
- *Whole-body stability control with high contact redundancy for wheel-legged hexapod
  robot driving over rough terrain* (Mechanism and Machine Theory, 2022)
- Skyhook 综述：*Skyhook-Based Techniques for Vehicle Suspension Control* (MDPI Machines 2025)
- Rocker-bogie 被动调平原理：Wikipedia/Hackaday rocker-bogie
