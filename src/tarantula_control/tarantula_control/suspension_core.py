"""主动悬挂控制核心 —— 仿真器无关，零 ROS 依赖。

分层契约（Gazebo 与 Isaac Lab 共享同一份算法）：
  - 本文件只依赖标准库；适配层（ROS 节点 / Isaac env）负责喂数和发令。
三个暴露面：
  参数面  SuspensionConfig   —— 全部可调参数（RL 域随机化/调参的自由度）
  观测面  SuspensionInputs   —— 姿态/关节角/轮地接触（RL observation 候选）
  动作面  inputs 中的 roll_ref/pitch_ref/height_cmd（RL action 候选：
          车身位姿指令，复用几何映射，天然有界——比直接出力矩安全）

算法：
  外环：roll/pitch 各一 PID（D 项作用于姿态角速率；条件积分抗饱和、死区、输出限幅=行程界限）
  映射：dz_i = x_i·u_pitch − y_i·u_roll + z_cmd（M2 高度通道）
        q_target_i = q0 + DIR_i·dz_i/(L·cosθ₀)，限幅+斜率限制
  前馈：tau = k_spring·dq 平移物理弹簧平衡点（DC=1.0，无软件快环）
  接触保持：每腿 支撑/悬空/重着地 状态机，悬空超过消抖时间
        后以 probe_slew 缓慢下探找地，重着地后同速率撤回；
        下探量并入平衡点偏移，受 target_limit 总限幅约束
  安全：落地保持期零力矩 + 增益渐入；倾角>tilt_freeze 包络冻结
"""
import math
from dataclasses import dataclass, field, fields

# 腿序与 suspension_controller 的 joints 参数一致
LEGS = ['fl', 'fr', 'ml', 'mr', 'rl', 'rr']
# 轮心在车身系的水平坐标（与 URDF 几何一致：x=+0.51/0/-0.51, y=±0.32）
WHEEL_X = {'fl': 0.51, 'fr': 0.51, 'ml': 0.0, 'mr': 0.0, 'rl': -0.51, 'rr': -0.51}
WHEEL_Y = {'fl': 0.32, 'fr': -0.32, 'ml': 0.32, 'mr': -0.32, 'rl': 0.32, 'rr': -0.32}
# 摆臂朝向（URDF leg 宏 dir）：dz>0=轮心下压，dq = DIR·dz/L_eff
DIR = {'fl': 1.0, 'fr': 1.0, 'ml': -1.0, 'mr': -1.0, 'rl': -1.0, 'rr': -1.0}


def clamp(v, lo, hi):
    return max(lo, min(hi, v))


def quat_roll_pitch(w, x, y, z):
    """四元数 -> (roll, pitch)。纯标量入参，不依赖消息类型。"""
    sinr_cosp = 2.0 * (w * x + y * z)
    cosr_cosp = 1.0 - 2.0 * (x * x + y * y)
    roll = math.atan2(sinr_cosp, cosr_cosp)
    sinp = clamp(2.0 * (w * y - z * x), -1.0, 1.0)
    return roll, math.asin(sinp)


def projected_gravity(w, x, y, z):
    """四元数 -> 重力向量 [0,0,-1] 在车体系下的投影 (gx,gy,gz)。

    与 IsaacLab ``ArticulationData.projected_gravity_b``
    （``quat_rotate_inverse(quat_w, [0,0,-1])``）逐式等价，供 M7 v2
    Gazebo 部署构造与训练一致的观测（见 rl_suspension_policy.py）。
    """
    gx = 2.0 * w * y - 2.0 * x * z
    gy = -2.0 * w * x - 2.0 * y * z
    gz = 1.0 - 2.0 * w * w - 2.0 * z * z
    return gx, gy, gz


@dataclass
class SuspensionConfig:
    """参数面。默认值为当前 Gazebo/Isaac baseline。"""
    # 几何（与 URDF 一致，修改须同步 tarantula_chassis.xacro）
    arm_length: float = 0.22
    arm_angle: float = 0.698
    nominal_angle: float = 0.0
    # 姿态外环 PID（俯仰杠杆比 3.0，增益须折减；kd 抑制 bump 激起的被动悬挂共振放大，
    # grid search 选定 0.3/0.1（M1 验收 total contact-loss 局部最优）；
    # kd=0 时退化为 v3 纯 PI）
    roll_kp: float = 0.8
    roll_ki: float = 1.2
    roll_kd: float = 0.3
    pitch_kp: float = 0.3
    pitch_ki: float = 0.5
    pitch_kd: float = 0.1
    att_deadband: float = 0.009     # rad ≈ 0.5°
    att_out_limit: float = 0.22     # rad，行程内
    # 悬挂执行器等效参数：Gazebo/Isaac 都应通过显式受限执行器模型使用这些值，
    # 不再依赖 Gazebo-only joint spring。
    ff_stiffness: float = 130.0
    actuator_damping: float = 11.0
    actuator_effort_limit: float = 75.0
    # 平衡点安全包络（0.55/0.3 等组合实测翻车，勿动）
    target_slew_rate: float = 0.20  # rad/s
    target_limit: float = 0.45      # rad，关节硬限位 0.6
    # 落地保持/渐入/包络冻结
    startup_hold: float = 5.0
    gain_ramp: float = 2.0
    tilt_freeze: float = 0.35       # rad ≈ 20°
    # M2 车身高度通道（goals v2：±0.06 m）
    height_limit: float = 0.06      # m
    # M1 接触保持（默认关：未调参，开启前先跑 M1 验收）
    contact_keeping: bool = True
    contact_debounce: float = 0.10  # s，悬空消抖
    contact_force_threshold: float = 5.0  # N，/ft_wheel/{leg} 力幅值接触判定阈值
    # （轮轴 |F|，恒为正；与几何接触地面真值对比误判率<1%，
    #   髋关节 force.z 因悬挂振荡误判率约33%，已弃用）
    probe_slew: float = 0.10        # m/s，轮心下探/撤回速度
    probe_limit: float = 0.05       # m，单腿最大下探量


@dataclass
class SuspensionInputs:
    """观测面 + 动作面。适配层每控制步构造一份。"""
    roll: float = 0.0
    pitch: float = 0.0
    joint_pos: dict = field(default_factory=dict)   # leg -> 悬挂关节角
    contacts: dict = field(default_factory=dict)    # leg -> bool，缺省视为着地
    # 动作面：车身位姿指令（默认零 = 纯调平，与 v3 等价）
    roll_ref: float = 0.0
    pitch_ref: float = 0.0
    height_cmd: float = 0.0                          # m，+ 为升高车身


@dataclass
class SuspensionOutputs:
    torques: list = field(default_factory=list)      # 按 LEGS 序
    # 遥测（调试/录包用，不参与控制）
    u_roll: float = 0.0
    u_pitch: float = 0.0
    q_target: dict = field(default_factory=dict)
    probe_dz: dict = field(default_factory=dict)     # leg -> 当前下探量 m
    height: float = 0.0                              # 实际生效的 z 指令
    frozen: bool = False
    holding: bool = False
    # 等效关节位置目标 = q0 + gain·dq_total（leg -> rad）。
    drive_target: dict = field(default_factory=dict)


class Pid:
    """姿态外环 PID，带输出限幅与条件积分（输出饱和时停止积分，防 windup）。
    D 项作用于姿态角速率而非误差，避免死区边界误差跳变引起微分突变。"""

    def __init__(self, kp, ki, kd, out_limit):
        self.kp, self.ki, self.kd, self.out_limit = kp, ki, kd, out_limit
        self.integral = 0.0

    def update(self, err, rate, dt):
        out = self.kp * err + self.ki * self.integral + self.kd * rate
        if abs(out) < self.out_limit or err * out < 0:
            self.integral += err * dt
        out = self.kp * err + self.ki * self.integral + self.kd * rate
        return clamp(out, -self.out_limit, self.out_limit)

    def reset(self):
        self.integral = 0.0


class SuspensionController:
    """纯算法控制器：reset() 后按固定节拍调 step(inputs, dt)。"""

    def __init__(self, cfg: SuspensionConfig):
        self.cfg = cfg
        self.L_eff = cfg.arm_length * math.cos(cfg.arm_angle)
        self.q0 = cfg.nominal_angle
        self.roll_pi = Pid(cfg.roll_kp, cfg.roll_ki, cfg.roll_kd, cfg.att_out_limit)
        self.pitch_pi = Pid(cfg.pitch_kp, cfg.pitch_ki, cfg.pitch_kd, cfg.att_out_limit)
        self.reset()

    def reset(self):
        self.t = 0.0
        self.roll_pi.reset()
        self.pitch_pi.reset()
        self.q_target = {leg: self.q0 for leg in LEGS}
        self.probe = {leg: 0.0 for leg in LEGS}      # 轮心下探量 m（dz 意义）
        self.lost_t = {leg: 0.0 for leg in LEGS}     # 悬空持续时间 s
        self.prev_roll = 0.0
        self.prev_pitch = 0.0

    def step(self, x: SuspensionInputs, dt: float) -> SuspensionOutputs:
        cfg = self.cfg
        self.t += dt
        out = SuspensionOutputs()

        # 落地保持期：零力矩，落地动力学与纯被动一致（消除落地自旋）
        if self.t < cfg.startup_hold:
            self.reset_targets_only()
            self.prev_roll, self.prev_pitch = x.roll, x.pitch
            out.torques = [0.0] * len(LEGS)
            out.holding = True
            out.q_target = dict(self.q_target)
            out.probe_dz = dict(self.probe)
            out.drive_target = {leg: self.q0 for leg in LEGS}
            return out
        gain = 1.0 if cfg.gain_ramp <= 0 else min(1.0, (self.t - cfg.startup_hold) / cfg.gain_ramp)

        roll_rate = (x.roll - self.prev_roll) / dt
        pitch_rate = (x.pitch - self.prev_pitch) / dt
        self.prev_roll, self.prev_pitch = x.roll, x.pitch

        # 包络保护：姿态异常时外环清零，只回名义位，绝不挣扎
        frozen = math.sqrt(x.roll ** 2 + x.pitch ** 2) > cfg.tilt_freeze
        if frozen:
            self.roll_pi.reset()
            self.pitch_pi.reset()
            u_roll, u_pitch = 0.0, 0.0
        else:
            def db(err):
                return 0.0 if abs(err) < cfg.att_deadband else err
            u_roll = self.roll_pi.update(db(x.roll - x.roll_ref), roll_rate, dt)
            u_pitch = self.pitch_pi.update(db(x.pitch - x.pitch_ref), pitch_rate, dt)

        # M2 高度通道：限幅后并入几何映射（斜率由各腿 slew 统一约束）
        z = 0.0 if frozen else clamp(x.height_cmd, -cfg.height_limit, cfg.height_limit)

        max_step = cfg.target_slew_rate * dt
        probe_step = cfg.probe_slew * dt
        for leg in LEGS:
            # 几何映射：车身位姿指令 -> 轮心目标高度差 -> 摆臂平衡点偏移
            dz = WHEEL_X[leg] * u_pitch - WHEEL_Y[leg] * u_roll + z
            raw_target = clamp(self.q0 + DIR[leg] * dz / self.L_eff,
                               -cfg.target_limit, cfg.target_limit)
            prev = self.q_target[leg]
            self.q_target[leg] = clamp(raw_target, prev - max_step, prev + max_step)

            # M1 接触保持：悬空消抖后缓慢下探，重着地同速率撤回
            if cfg.contact_keeping and not frozen:
                if x.contacts.get(leg, True):
                    self.lost_t[leg] = 0.0
                    self.probe[leg] = max(0.0, self.probe[leg] - probe_step)
                else:
                    self.lost_t[leg] += dt
                    if self.lost_t[leg] > cfg.contact_debounce:
                        self.probe[leg] = min(cfg.probe_limit,
                                              self.probe[leg] + probe_step)
            else:
                self.probe[leg] = max(0.0, self.probe[leg] - probe_step)
            # 下探量并入平衡点偏移，总偏移仍受 target_limit 约束
            dq_total = clamp(self.q_target[leg] - self.q0
                             + DIR[leg] * self.probe[leg] / self.L_eff,
                             -cfg.target_limit, cfg.target_limit)

            # 前馈：平移物理弹簧平衡点（增益渐入）
            out.torques.append(gain * cfg.ff_stiffness * dq_total)
            out.drive_target[leg] = self.q0 + gain * dq_total

        out.u_roll, out.u_pitch = u_roll, u_pitch
        out.q_target = dict(self.q_target)
        out.probe_dz = dict(self.probe)
        out.height = z
        out.frozen = frozen
        return out

    def reset_targets_only(self):
        """保持期内归位：渐入从零力矩开始，PI 不带累积状态。"""
        self.roll_pi.reset()
        self.pitch_pi.reset()
        for leg in LEGS:
            self.q_target[leg] = self.q0
            self.probe[leg] = 0.0
            self.lost_t[leg] = 0.0


def config_fields():
    """适配层用：自动把参数面映射为 ROS 参数（名称即字段名）。"""
    return [(f.name, f.default) for f in fields(SuspensionConfig)]
