"""主动避震控制器 —— 串级位置式调平（设计见 docs/01-control-architecture.md）。

外环：IMU roll/pitch 各一个 PI，输出车身姿态修正量（rad，限幅）。
几何映射：姿态修正 -> 每条腿轮心目标高度 -> 摆臂目标角
    q_target_i = q0 + DIR_i * (x_i * u_pitch - y_i * u_roll) / (L * cos(theta0))
内环：关节位置 PD（刚度高于 URDF 物理弹簧，保证位置主导权）
    tau_i = kp_in * (q_target_i - q_i) - kd_in * q̇_i

安全特性（对应力矩叠加方案的三个实证失效模式）：
- 目标角限幅 = 行程几何界限，不存在无界积分顶翻车身；
- 目标角斜率限制，摆臂缓变，锁死轮不被拖刮自旋；
- 包络保护：倾角异常时目标回名义位，绝不挣扎。
"""
import math

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Imu, JointState
from std_msgs.msg import Float64MultiArray

# 腿序与 suspension_controller 的 joints 参数一致
LEGS = ['fl', 'fr', 'ml', 'mr', 'rl', 'rr']
# 轮心在车身系的水平坐标（与 URDF 几何一致：x=+0.51/0/-0.51, y=±0.32）
WHEEL_X = {'fl': 0.51, 'fr': 0.51, 'ml': 0.0, 'mr': 0.0, 'rl': -0.51, 'rr': -0.51}
WHEEL_Y = {'fl': 0.32, 'fr': -0.32, 'ml': 0.32, 'mr': -0.32, 'rl': 0.32, 'rr': -0.32}
# 摆臂朝向（URDF leg 宏 dir）：+q 对 dir=+1 是轮子下压，对 dir=-1 是轮子上抬
DIR = {'fl': 1.0, 'fr': 1.0, 'ml': -1.0, 'mr': -1.0, 'rl': -1.0, 'rr': -1.0}


def quat_to_roll_pitch(q):
    sinr_cosp = 2.0 * (q.w * q.x + q.y * q.z)
    cosr_cosp = 1.0 - 2.0 * (q.x * q.x + q.y * q.y)
    roll = math.atan2(sinr_cosp, cosr_cosp)
    sinp = max(-1.0, min(1.0, 2.0 * (q.w * q.y - q.z * q.x)))
    return roll, math.asin(sinp)


class Pi:
    """姿态外环 PI，带输出限幅与条件积分（输出饱和时停止积分，防 windup）。"""

    def __init__(self, kp, ki, out_limit):
        self.kp, self.ki, self.out_limit = kp, ki, out_limit
        self.integral = 0.0

    def update(self, err, dt):
        out = self.kp * err + self.ki * self.integral
        if abs(out) < self.out_limit or err * out < 0:
            self.integral += err * dt
        out = self.kp * err + self.ki * self.integral
        return max(-self.out_limit, min(self.out_limit, out))

    def reset(self):
        self.integral = 0.0


class ActiveSuspension(Node):
    def __init__(self):
        super().__init__('active_suspension')

        # 几何（与 URDF 一致）
        self.declare_parameter('arm_length', 0.22)
        self.declare_parameter('arm_angle', 0.698)
        self.declare_parameter('nominal_angle', 0.0)
        # 外环姿态 PI：输出为车身姿态修正量（rad）。
        # 俯仰几何杠杆 x/L_eff=3.02 vs 侧倾 1.89，回路增益差 1.6 倍，
        # 俯仰增益必须按比例折减（合并增益曾致俯仰轴 ±5° 极限环，前腿打摆）
        self.declare_parameter('roll_kp', 0.8)
        self.declare_parameter('roll_ki', 1.2)
        self.declare_parameter('pitch_kp', 0.3)
        self.declare_parameter('pitch_ki', 0.5)
        self.declare_parameter('att_deadband', 0.009)   # rad ≈ 0.5°，防微振
        self.declare_parameter('att_out_limit', 0.22)   # rad ≈ 12.6°，行程内
        # 内环为纯前馈：tau = ff_stiffness * dq，配合 URDF 物理弹簧把平衡点
        # 精确移到 dq（DC 增益 1.0）。不做软件位置反馈——话题链路 ~20ms 延迟下
        # 软件 PD 会颤振（kp=350 实测 ±160Nm 打摆，180 仍 ±85Nm）。
        # 动力学由仿真器隐式弹簧阻尼承担（无延迟、无条件稳定），
        # 唯一反馈是 ~1Hz 带宽的姿态外环。力矩天然有界 120*0.45=54 < 60 限幅。
        self.declare_parameter('ff_stiffness', 120.0)   # 必须等于 URDF springStiffness
        # 天棚阻尼：陀螺角速度 -> 抑制车身摆动的腿力矩。阻尼项相位超前，
        # 对话题链路延迟鲁棒（位置反馈做不到）；静态时角速度=0，不扰动调平
        # 阻尼上限受话题链路 ~20ms 延迟约束：roll 惯量小/频率高对相位最敏感，
        # 0.45 实测延迟致负阻尼翻车，0.15 为安全值（roll 动态抑制受限，
        # 见 docs 已知限制；真实产品悬挂环路跑 kHz 嵌入式控制器即为此）
        self.declare_parameter('sky_roll_damp', 0.15)   # s，roll_rate -> 等效姿态修正
        self.declare_parameter('sky_pitch_damp', 0.12)
        self.declare_parameter('sky_limit', 0.20)       # rad，阻尼项等效偏移限幅
        # 安全（实验标定 2026-06-11：limit 0.55/kp 450/slew 0.3 会撞关节硬限位翻车，
        # 0.45/350/0.2 已验证稳定收敛）
        # 0.2 经 8 度静坡 50s 实验验证稳定；更慢(0.06)会粘滑积累应力雪崩翻车，
        # 更快(0.3)配合大限幅会撞关节硬限位。静止锁轮时调平动作有拖刮偏航伪影
        # （行驶中轮子滚动可自然吸收，无此问题），属已知限制
        self.declare_parameter('target_slew_rate', 0.20)  # rad/s
        self.declare_parameter('target_limit', 0.45)      # rad，离关节硬限位 0.6 留足余量
        # 落地保持期：期间输出零力矩，落地动力学与纯被动完全一致
        # （实测被动落地 yaw 漂移仅 0.2°；若保持期用刚性 PD 抓名义位，
        # 等效刚度 470 Nm/rad 落地弹跳会产生 ~60° 自旋）。时长按仿真时钟计。
        self.declare_parameter('startup_hold', 5.0)       # s
        self.declare_parameter('gain_ramp', 2.0)          # s，保持期后 PD 增益渐入
        self.declare_parameter('tilt_freeze', 0.35)       # rad ≈ 20°
        self.declare_parameter('control_rate', 100.0)

        gp = self.get_parameter
        self.L_eff = gp('arm_length').value * math.cos(gp('arm_angle').value)
        self.q0 = gp('nominal_angle').value
        self.k_ff = gp('ff_stiffness').value
        self.c_roll = gp('sky_roll_damp').value
        self.c_pitch = gp('sky_pitch_damp').value
        self.sky_limit = gp('sky_limit').value
        self.roll_rate = 0.0
        self.pitch_rate = 0.0
        self.slew = gp('target_slew_rate').value
        self.t_limit = gp('target_limit').value
        self.tilt_freeze = gp('tilt_freeze').value
        self.hold = gp('startup_hold').value
        self.ramp_time = gp('gain_ramp').value
        self.t_first = None  # 首帧关节数据的仿真时刻

        lim = gp('att_out_limit').value
        self.deadband = gp('att_deadband').value
        self.roll_pi = Pi(gp('roll_kp').value, gp('roll_ki').value, lim)
        self.pitch_pi = Pi(gp('pitch_kp').value, gp('pitch_ki').value, lim)

        self.roll = 0.0
        self.pitch = 0.0
        self.joint_pos = {}
        self.joint_vel = {}
        self.q_target = {leg: self.q0 for leg in LEGS}  # 斜率限制后的目标角

        self.cmd_pub = self.create_publisher(
            Float64MultiArray, '/suspension_controller/commands', 10)
        # 调试：[roll, pitch, u_roll, u_pitch, q_target x6, q x6, tau x6]
        self.debug_pub = self.create_publisher(Float64MultiArray, '~/debug', 10)
        self.create_subscription(Imu, '/imu/data', self.imu_cb, 50)
        self.create_subscription(JointState, '/joint_states', self.joint_cb, 50)

        self.dt = 1.0 / gp('control_rate').value
        self._step = 0
        self.create_timer(self.dt, self.control_step)
        self.get_logger().info('Active suspension started (cascade position leveling).')

    def imu_cb(self, msg: Imu):
        self.roll, self.pitch = quat_to_roll_pitch(msg.orientation)
        self.roll_rate = msg.angular_velocity.x
        self.pitch_rate = msg.angular_velocity.y

    def joint_cb(self, msg: JointState):
        for i, name in enumerate(msg.name):
            self.joint_pos[name] = msg.position[i]
            if i < len(msg.velocity):
                self.joint_vel[name] = msg.velocity[i]

    def control_step(self):
        if f'susp_{LEGS[0]}_joint' not in self.joint_pos:
            return

        now = self.get_clock().now().nanoseconds * 1e-9
        if self.t_first is None:
            self.t_first = now
        elapsed = now - self.t_first

        # 落地保持期：零力矩，与纯被动落地完全一致
        if elapsed < self.hold:
            self.roll_pi.reset()
            self.pitch_pi.reset()
            for leg in LEGS:  # 前馈基准为名义位，保持期内归位使渐入从零力矩开始
                self.q_target[leg] = self.q0
            self.cmd_pub.publish(Float64MultiArray(data=[0.0] * len(LEGS)))
            return
        gain = min(1.0, (elapsed - self.hold) / self.ramp_time)

        # 包络保护：姿态异常时外环清零，PD 仅维持名义位
        if math.sqrt(self.roll ** 2 + self.pitch ** 2) > self.tilt_freeze:
            self.roll_pi.reset()
            self.pitch_pi.reset()
            u_roll, u_pitch = 0.0, 0.0
        else:
            def db(err):
                return 0.0 if abs(err) < self.deadband else err
            u_roll = self.roll_pi.update(db(self.roll), self.dt)
            u_pitch = self.pitch_pi.update(db(self.pitch), self.dt)

        # 天棚阻尼：角速度通道，不经过慢速调平路径（需要快路径抗动态扰动）
        u_roll_d = self.c_roll * self.roll_rate
        u_pitch_d = self.c_pitch * self.pitch_rate

        cmd = Float64MultiArray()
        max_step = self.slew * self.dt
        for leg in LEGS:
            # 几何映射：姿态修正 -> 轮心目标高度差 -> 摆臂平衡点偏移
            dz = WHEEL_X[leg] * u_pitch - WHEEL_Y[leg] * u_roll
            raw_target = self.q0 + DIR[leg] * dz / self.L_eff
            raw_target = max(-self.t_limit, min(self.t_limit, raw_target))
            # 斜率限制（仅调平通道）
            prev = self.q_target[leg]
            self.q_target[leg] = max(prev - max_step, min(prev + max_step, raw_target))
            # 阻尼通道等效偏移（限幅，无斜率限制）
            dz_d = WHEEL_X[leg] * u_pitch_d - WHEEL_Y[leg] * u_roll_d
            dq_d = DIR[leg] * dz_d / self.L_eff
            dq_d = max(-self.sky_limit, min(self.sky_limit, dq_d))
            # 前馈：调平平衡点偏移 + 天棚阻尼（增益渐入）
            tau = gain * self.k_ff * (self.q_target[leg] - self.q0 + dq_d)
            cmd.data.append(tau)
        self.cmd_pub.publish(cmd)

        self._step += 1
        if self._step % 10 == 0:
            dbg = Float64MultiArray()
            dbg.data = ([self.roll, self.pitch, u_roll, u_pitch]
                        + [self.q_target[leg] for leg in LEGS]
                        + [self.joint_pos.get(f'susp_{leg}_joint', 0.0) for leg in LEGS]
                        + list(cmd.data))
            self.debug_pub.publish(dbg)


def main():
    rclpy.init()
    node = ActiveSuspension()
    try:
        rclpy.spin(node)
    except (KeyboardInterrupt, rclpy.executors.ExternalShutdownException):
        pass
    finally:
        node.destroy_node()
        try:
            rclpy.shutdown()
        except Exception:
            pass


if __name__ == '__main__':
    main()
