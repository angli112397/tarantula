# 底盘物理模型重设计记录

日期：2026-06-16

目标：按粗糙地形车辆 RL 论文维护当前底盘物理 baseline。本文只记录当前有效模型决策。

## 论文实践映射

参考方向：

- Wiberg et al., *Control of rough terrain vehicles using deep reinforcement learning*：
  六轮主动悬挂粗糙地形车辆，强调真实质量/力矩限制、轮载、轮滑、能耗、底盘触地
  和地形课程；轮子用简化刚体接触以提升仿真稳定性。
- Bouton et al., *Learning All-Terrain Locomotion for a Planetary Rover with Actively
  Articulated Suspension*：主动悬挂行星车使用姿态、关节、力/力矩和地形高程输入，
  通过 domain randomization、传感器噪声和系统辨识做迁移。
- Margolis et al., *Rapid Locomotion via Reinforcement Learning*：动作输出关节目标，
  执行器动态、质量、COM、摩擦和 motor strength 都需要随机化/对齐。

## 本轮决策

1. 轮胎视觉仍为圆柱，碰撞默认使用球形，并支持圆柱 A/B：
   - 文件：`tarantula_chassis.xacro`
   - launch 参数：`wheel_collision:=sphere|cylinder`
   - 原因：圆柱轮在台阶边缘和侧向擦碰时接触不连续，Gazebo/Isaac 更容易产生尖峰力。
   - 取舍：球形轮会弱化轮胎侧壁和宽度效应；如需轮胎宽度，再测试 capsule / 多球近似。

2. 底盘腹部 collision 显式命名并略收腹：
   - `base_belly_collision`
   - 原因：粗糙地形任务应明确建模托底风险，而不是让视觉外壳或装饰件参与接触。

3. base COM 显式参数化：
   - `body_com_x/y/z`
   - 原因：后续 Isaac/Gazebo 都可以围绕 COM 做域随机化和载荷敏感性测试。

4. 删除几何 contact sensor 作为控制/观测来源：
   - 不再发布 `/contact/{leg}` 或 `/contact/base`；
   - 策略观测统一为轮轴 F/T 推出的连续 `wheel_load(6)`；
   - 原因：真实系统中几何接触真值难以实现，轮轴/轮毂 F/T 更接近论文中的
     force-torque measurement。

5. 删除 Gazebo-only 髋关节虚拟弹簧：
   - 不再写 `<springStiffness>` / `<implicitSpringDamper>`；
   - Gazebo RL 部署由 `rl_suspension_policy.py` 显式计算
     `tau = kp(target-q) - kd*qdot`，并按 URDF effort limit 限幅；
   - Isaac 使用同参数 joint position drive，角色是执行器模型，不是免费被动弹簧。

6. Isaac USD 缓存文件名：
   - `tarantula_core_baseline_pd_sphere_wheels.usd`

7. reward 方向同步：
   - baseline reward 只保留速度跟踪、yaw-rate 跟踪、姿态、动作平滑、
     关节软限位和终止惩罚；
   - `wheel_load(6)` 先作为观测和诊断指标，不进入 baseline reward。

8. Gazebo 物理验收增加独立站姿保持层：
   - `stand_suspension_hold` 只做 6 个悬挂关节的限幅 PD stand hold；
   - wheel open-loop benchmark 只发布轮速，不发布悬挂力矩；
   - 原因：删除 Gazebo-only 髋关节弹簧后，`active_suspension` 是姿态调平器，
     不是从趴地状态建立站姿的支撑控制器。物理接触验收必须先建立稳定站姿。

9. RL Stage A 改为 wheel-only：
   - Isaac 中悬挂固定在 neutral stand target；
   - Gazebo 中悬挂由 `stand_suspension_hold` 控制；
   - actor action 只输出 6 路轮速，先验证 `cmd_vx/cmd_wz` obedience；
   - 悬挂 RL 留到 Stage B，以 stand target residual 的形式重新设计。

## 后续验证

- Gazebo：RL low-speed flat / single bump / side step / rough terrain 行为。
- Gazebo physics baseline：`stand_hold:=true` 下跑 `gazebo_wheel_open_loop_benchmark.py`。
- Wheel collision A/B：只切换 `wheel_collision:=sphere|cylinder`，其他 launch
  参数、地形、stand-hold 参数和 benchmark 序列保持一致。
- Isaac：重新生成 USD 后检查 obs/action 无 NaN。
- 对齐项：轮速响应、轮端力分布、同一地形的 roll/pitch RMS、关节力矩饱和率。

## 暂不改动

- 2-DOF 腿：先把单自由度摆臂的执行器、轮载、轮胎接触和奖励闭环做干净。
- 地形高程图：本轮先不加，下一阶段评估 blind + wheel load 是否足够。

## Gazebo Baseline Tune

目标：让 generated heightmap baseline 中的粗糙块、横坡、低台阶和浅沟有更稳定、
可解释的车辆响应。本轮不改拓扑，只调可追踪的物理参数。

改动：

- wheel radius: `0.12 -> 0.13 m`
  - 目的：提高台阶/碎石通过裕度，降低球形 collision 在小障碍上的卡滞概率。
  - 同步：`controllers.yaml`、`rl_suspension_policy.py`、`suspension_env_cfg.py`。
- wheel width/mass: `0.07 -> 0.075 m`, `1.5 -> 1.7 kg`
  - 目的：让轮子视觉和惯量更接近越障轮，不让轮端过轻导致接触尖峰过敏。
- body COM z: `0.0 -> -0.025 m`
  - 目的：在不改变外形的情况下提高横坡和碎石上的抗翻滚裕度。
- suspension actuator: `kp 120 -> 130 Nm/rad`, `kd 8 -> 11 Nms/rad`,
  effort limit `60 -> 75 Nm`
  - 目的：增加前方组合障碍中的姿态保持和冲击阻尼；仍保留硬限位 `±0.6 rad`。
  - 同步：URDF、Gazebo RL 显式 PD、Isaac joint drive。
- wheel joint effort: `30 -> 38 Nm`
  - 目的：避免低摩擦后接台阶时轮速控制过早力矩饱和。
  - 同步：Isaac wheel velocity-drive gain `10.0 -> 12.7`。
- Gazebo tire contact: `mu1 1.2 -> 1.35`, `mu2 0.8 -> 1.05`, `kd 100 -> 140`
  - 目的：球形轮接触下补偿横向抓地不足，并增加接触阻尼。

验收重点：

- 默认 spawn 后直接前进，应能在 generated `gazebo_demo/42` 上产生稳定前进；
- 观察是否有弹飞、横向滑落、悬挂力矩长时间饱和、台阶后无法恢复；
- 若仍托底，下一轮优先调腹部 collision / arm length；
- 若仍翻滚，下一轮优先调 COM、track width 或 suspension target limit；
- 若轮速发散，下一轮优先调 wheel effort / velocity-drive gain / friction。
