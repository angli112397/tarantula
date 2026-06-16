# 项目目标与里程碑（v3 修订）

修订日期：2026-06-16 ｜ 交付期限：≈2026-06-25（面试）

## 1. 项目定位（一句话）

**六轮崎岖地形 RL 越障：Isaac Lab 训练 PPO 策略，通过主动悬挂几何映射实现自适应
越障（台阶/坡道/凸起），Gazebo 验证 sim-to-sim 迁移。**

从 v2 的"手动 vs RL 对比"重新定位为以 RL 为核心的单一路线：
- **功能层**：调平/高度调节/接触保持（M1/M2，已完成，归档）→ RL 越障主线（M7，当前主线）
- **方法层**：每项功能有量化验收 + 失效边界实验 + 安全包络
- **视野层**：同一 URDF 跨 Gazebo/Isaac Lab，domain randomization 覆盖 sim-to-sim 动力学差异，RL 策略 Gazebo 部署验证

## 2. 为什么放弃"姿态稳定 vs RL 对比"框架

v2 文档的核心叙事是"手动前馈平衡点调平 vs RL 策略的精度对比"。该框架有两个根本局限：

**1. 姿态稳定目标与崎岖地形矛盾**

在台阶/坡道上行驶时，机身倾斜是物理必然：
- 前轮爬坡时机身必然前仰
- 单轮过台阶时必然侧倾
- 强惩罚 roll/pitch → 策略学会绕避而非越过障碍

文献（AnymalC, ETH Blind Locomotion, ERNEST）的做法：姿态项只作软正则化（weight=0.1），
速度跟踪为主奖励，极端倾斜（0.6 rad）触发 episode 终止。v4 与此对齐，不追求云台效果。

**2. 手动算法已完成其使命**

M1/M2 的前馈平衡点平移验证了几何运动学映射的正确性，完成了工程方法论展示。
持续迭代对比手动算法无增量价值；该资源投入 RL 策略的域随机化和 Gazebo 部署。

## 3. 里程碑

### Phase 1 — Gazebo 经典控制（已完成归档，2026-06-11）

- [x] v3 前馈+天棚架构，G1–G4 全部通过（见 01 §5）
- [x] M1 接触保持状态机（代码完成，接触丢失↓≥40% 验收标准已验证）
- [x] M2 车身高度调节（z ±0.06 m 跟踪，8° 坡调平仍 <0.5°）

经典控制路径保留完整（`leveling:=true, rl_policy:=false`），可随时切回演示。

### Phase 2 — Isaac Lab RL 越障（当前主线）

| # | 内容 | 状态 | 验收标准 |
|---|---|---|---|
| M4 | Isaac Lab 环境建立（URDF 导入 + 关节驱动 + 平地站立） | ✅ 完成 | 关节角与 Gazebo 平地同量级 |
| M7 v4 Stage A | RL env 设计（34D obs，kinematic mapping，离散障碍地形） | ✅ 完成 | obs.shape==(N,34)，无 NaN |
| M7 v4 Stage B | PPO 训练（400 iter，从头，reward 715） | ✅ 完成 | reward ≥ 500 |
| M7 v4 Stage D | Gazebo 部署（direct torque bypass，rl_policy:=true） | ✅ 完成 | 节点启动无错误，机器人运动 |
| M7 v4 Stage E | Domain rand（质量±3kg，推力扰动，obs 噪声）+ warm-start 重训 | 🔄 进行中 | reward ≥ 700，无 NaN |

### Phase 3 — 交付（2026-06-22 起）

- [ ] Gazebo GUI 观察记录（前腿浮空/机身倾斜行为诊断）
- [ ] 演示录像：RL 越障 + 经典调平 A/B 对照（leveling:=true / rl_policy:=true）
- [ ] README/docs 终稿
- [ ] 面试叙事演练：每个里程碑 ↔ 一个工程决策

## 4. 非目标（明确砍掉的范围）

- **手动算法精度 vs RL 精度的定量对比**：已归档，不再作为主叙事
- **云台式机身水平效果**：物理上不现实（需解耦执行机构），仿真目标改为"越障"
- **2-DOF 腿全量实现**：时间不够，设计研究文档化即可
- **六轮载荷均衡**：可选项，时间不够放弃
- **M6 延迟归因实验**：Isaac Lab kHz 管线实验，降优先级（RL 主线优先）

## 5. 当前 RL 系统状态（v4，2026-06-16）

**架构**：`obs(34D) → PPO actor(MLP 34→64→64→5) → action(5D) → kinematic mapping → susp joint PD`

| 组件 | 状态 |
|---|---|
| obs 构成 | projected_gravity(3) + ang_vel(3) + lin_vel(3) + susp_pos(6) + susp_vel(6) + wheel_vel(6) + move_cmd(1) + heading_cmd(1) + prev_action(5) = 34D |
| action 含义 | [u_roll, u_pitch, z_cmd, wheel_left, wheel_right]；前 3 维进几何映射 |
| domain rand | 摩擦力 0.3-1.5 + 质量 ±3kg + obs 噪声 σ=0.02 + 推力扰动 0.5m/s@150-300步 |
| 地形 | flat(0.15) + rough(0.20) + pyramid_slope(0.20×2) + hf_wave(0.10) + hf_discrete_obstacles(0.20) |
| 检查点 | model_399.pt，reward 715（400 iter，从头） |
| Gazebo 部署 | `rl_suspension_policy` 节点，绕过 `active_suspension`，直接发前馈力矩 |

## 6. 风险与回退

| 风险 | 回退 |
|---|---|
| Gazebo 行为异常（前腿浮空、左倾）持续无法修复 | 记录为 sim-to-sim gap，保留经典调平路径演示 |
| domain rand Stage E 训练 reward 下降 | 回滚到 model_399.pt，不做 Stage E |
| 面试时 RL 演示失败 | 备选：经典调平 + SLAM 演示（独立演示价值） |

## 7. 简历一句话（当前版本）

> 六轮主动悬挂底盘：Isaac Lab PPO 训练崎岖地形越障策略（34D obs，几何运动学映射
> 直驱关节，domain randomization 覆盖 sim-to-sim 动力学差异），Gazebo 验证部署；
> 经典路径：前馈平衡点平移 + 天棚阻尼实现 3-DOF 车身控制（8° 静坡 <0.5°）。
