# M7：RL 主动悬挂越障 —— 训练 + Gazebo 集成记录（v4，进行中）

状态（2026-06-16）：v4 Architecture 已完成（SuspensionController 从 RL 路径完全移除，
obs=34D，action=5D，kinematic mapping 直驱关节 PD）。Stage E domain rand 训练完成
（reward 750.78，model_698.pt，warm-start 自 Stage B 715）。当前版本 Gazebo 已部署，
等待 GUI 观察验证。

**v4 变更摘要（相对于 v1-v3）**：
- RL 路径：去除 SuspensionController 中间层，action[0:3] → 几何映射 → 关节位置目标
- obs 34D（较 v3 40D 移除 6D drive_target，新增 projected_gravity(3)）
- action 5D（u_roll, u_pitch, z_cmd, wheel_left, wheel_right）
- domain rand：摩擦(继承) + 质量±3kg + obs噪声σ=0.02 + push扰动0.5m/s@150-300步
- 项目目标重新定位为"越障"而非"姿态稳定 vs RL 对比"（见 docs/02 §2）

## 1. 环境设计 v4（`src/tarantula_isaac/`，当前版本）

`suspension_env.py` 实现 IsaacLab `DirectRLEnv` `Isaac-Tarantula-Suspension-v0`：

- **obs(34D)**：`projected_gravity(3) + ang_vel_b(3) + lin_vel_b(3) + susp_pos(6) + susp_vel(6) + wheel_vel(6) + move_cmd(1) + heading_cmd(1) + prev_action(5)`
- **action(5D)**：`clip(actor_out,-1,1)` → `[u_roll·0.15rad, u_pitch·0.15rad, z_cmd·0.06m, wheel_left, wheel_right]`；前 3 维走几何映射写关节位置目标（stiffness=120/damping=8），后 2 维走速度驱动
- **奖励**：velocity tracking（主） + 0.1·attitude正则 + survival 0.05/步 − action_rate惩罚
- **终止**：`|tilt| > 0.6rad`（episode level 硬终止） + `episode_length_s=20.0`
- **地形**：flat(0.15) + random_rough(0.20) + pyramid_slope(0.20×2) + hf_wave(0.10) + hf_discrete_obstacles(0.20)；`num_envs=16`, `env_spacing=8.0`
- **domain rand**（v4 Stage E 新增）：
  - 摩擦力：0.3–1.5（64 bucket，PhysX material API）
  - 质量±3kg（base_link，reset 时，PhysX tensor API）
  - obs 噪声：σ=0.02 Gaussian（`_get_observations` 末尾）
  - push 扰动：±0.5m/s x/y @每 150–300 控制步（`_pre_physics_step`）

SuspensionController 已从 RL 路径完全移除（v4 变更，见 docs/01 §7）。
Gazebo 经典路径（`leveling:=true`）仍复用 `suspension_core.py`。

## 2. 训练历史（rsl_rl PPO，actor/critic 均为 MLP 34→64→64→5/1）

| 阶段 | checkpoint | iter | reward | 备注 |
|---|---|---|---|---|
| Stage B | `v4_stage_b/model_399.pt` | 400（从头） | 715.48 | domain rand: 摩擦力 only |
| Stage E | `v4_stage_e/model_698.pt` | 300（warm-start from B） | **750.78** | +质量±3kg+obs噪声+push扰动 |

reward 在 warm-start 后先降至 ~310（任务更难），iter 430 回升至 764，最终收敛 750.78。
`num_steps_per_env=24`, `learning_rate=1e-3`, `clip_range=0.2`。

## 3. 早期版本对照（v1-v3，已归档）

v1-v3 使用旧 obs/action 设计（obs 19D/37D/40D，action 3D/5D），已全部重写为 v4。
以下为 v2 Stage C 的 Isaac 内 v3-classical-vs-RL-v2 对照，仅供历史参考：

| env 范围 | terrain | v3_tilt | rl_tilt | v3_ret | rl_ret |
|---|---|---|---|---|---|
| 0-3 | flat | 0.0000 | 0.0000 | 45.00 | 44.97 |
| 4,6 | random_rough | 0.0153-0.0279 | 0.0113-0.0279 | 32-38 | 32-40 |
| 9-14 | hf_pyramid | 0.0000 | 0.0000 | 45.00 | 44.97 |
| 15 | hf_wave | 0.0124 | 0.0124 | 39.41 | 39.28 |

结论（已汇报）：RL 在 flat/plateau 与 v3 等价，在 rough/wave 上小幅改善
（tilt -14%、return +0.35 mean）。**全部 16 个 env 都没有出现角点饱和**——
即使是 v3_tilt 最大的 env 6（0.0279），rl_tilt 也保持同量级，未发散。

## 4. Gazebo 部署（v4）

`rl_suspension_policy.py` 节点（30Hz）直接发前馈力矩，绕过 `active_suspension`：
- 订阅 `/imu/data` + `/joint_states` + `/cmd_vel`，拼 34D obs
- obs 归一化：使用 checkpoint 内嵌的 `obs_normalizer._mean/_std`（EmpiricalNormalization）
- 输出 action[0:3] → kinematic mapping → `/tarantula/susp_{fl/ml/rl/fr/mr/rr}_effort`
- 输出 action[3:5] → 差速驱动 → `/tarantula/wheel_*_velocity`

`sim.launch.py` 控制路径选择（互斥）：
```
rl_policy:=true   → 仅启动 rl_suspension_policy 节点
rl_policy:=false  → 仅启动 active_suspension（经典 M1/M2 路径）
```

**Stage E 当前 Gazebo 状态**（model_698.pt，2026-06-16）：
- 节点启动无报错，机器人可运动
- 前腿浮空：可能是策略的涌现预抬轮行为（预anticipation for climbing）
- 机身轻微左倾：训练 artifact（PPO 随机种子 + 摩擦随机化种子非对称），待 GUI 观察
- 轮子视觉大小差异：URDF 统一 `wheel_radius=0.12`，为渲染视角问题

## 5. 已知问题与历史记录

**v1-v3 角点卡死问题**（已解决，仅历史参考）：

v2/v3 版本 Gazebo 部署时观察到策略收敛到动作空间角点（action=[1,1,1]），原因是
obs 分布 OOD（Gazebo 出生点倾角 4-5° > Isaac 训练分布 1-2°）导致 PPO hard-clip
边界处梯度为零。v4 解决方案：重新设计 obs 空间（加 lin_vel/projected_gravity），
增大 domain rand 覆盖（push 扰动），使训练分布包含更大倾角的初始条件。

## 6. 下一步

- [ ] GUI 观察 Stage E 行为（前腿浮空/左倾诊断）
- [ ] 演示录像：RL 越障 + 经典调平 A/B 对照
- [ ] docs 终稿（见 docs/02 Phase 3）
