"""RL wheel/suspension deployment node.

Stage A wheel-only baseline:
  - action[0:6] -> 6 个驱动轮各自速度目标
  - 不发布 /suspension_controller/commands；悬挂由 stand_suspension_hold 负责

Legacy suspension+wheel baseline：
  - action[0:6]  -> 6 个悬挂关节角度目标（直接，无几何映射）
    Gazebo 等价：显式受限 PD 执行器
      effort_i = kp * (target_i - q_i) - kd * qdot_i
  - action[6:12] -> 6 个驱动轮各自速度目标（取代 diff_drive 左/右分组）
    发布到 /wheel_velocity_controller/commands (Float64MultiArray, LEGS 顺序)
  - 移除 diff_drive_controller 订阅/发布
  - 移除几何运动学映射（dz/dq/DIR）
  - 订阅 /ft_wheel/{leg} ×6 作为 wheel_load(6D) 观测
  - 默认从 /tarantula/truth_odom 读取 Gazebo truth body velocity；
    不可用时回退到平均轮速 × 轮半径估算

观测构成：
  projected_gravity_b(3)  <- /imu/data
  root_ang_vel_b(3)        <- /tarantula/truth_odom 或 /imu/data
  root_lin_vel_b(3)        <- /tarantula/truth_odom 或平均轮速估算
  susp_joint_pos(6)        <- /joint_states (LEGS 顺序)
  susp_joint_vel(6)        <- /joint_states
  wheel_joint_vel(6)       <- /joint_states
  wheel_load(6)            <- /ft_wheel/{leg}，||F|| / nominal_wheel_load
  cmd_vx(1)                <- /cmd_vel.linear.x, fallback ROS 参数
  cmd_wz(1)                <- /cmd_vel.angular.z, fallback ROS 参数
  prev_action(6 or 12)     <- 上一步策略输出

几何常量（与 suspension_env_cfg.py 一致）：
  ACTION_SCALE_SUSP = 0.5 rad  (Isaac: position target ±0.5)
  SUSP_ACTUATOR_KP = 130.0 Nm/rad, KD = 11.0 Nms/rad, effort limit = 75 Nm
"""
import numpy as np
import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist, Wrench
from nav_msgs.msg import Odometry
from sensor_msgs.msg import Imu, JointState
from std_msgs.msg import Float64MultiArray
import time

from .control_interfaces import (
    CmdVelLimiter,
    SUSP_ACTUATOR_KD,
    SUSP_ACTUATOR_KP,
    SUSP_EFFORT_LIMIT,
    SUSP_JOINT_LIMIT,
    clamp_abs,
    mean_wheel_forward_velocity,
    wheel_load_normalized,
)
from .rl_policy import ACTION_SCALE_SUSP, ACTION_SCALE_WHEEL_OMEGA, RLSuspensionPolicy
from .suspension_core import LEGS, projected_gravity


class RLSuspensionPolicyNode(Node):
    def __init__(self):
        super().__init__('rl_suspension_policy')
        self.declare_parameter('rate', 30.0)
        self.declare_parameter('cmd_vx', 0.2)
        self.declare_parameter('cmd_wz', 0.0)
        self.declare_parameter('max_abs_cmd_vx', 0.3)
        self.declare_parameter('max_abs_cmd_wz', 0.4)
        self.declare_parameter('policy_weights_npz', '')
        self.declare_parameter('policy_mode', 'auto')
        self.declare_parameter('velocity_source', 'auto')
        self.declare_parameter('truth_odom_topic', '/tarantula/truth_odom')
        self.declare_parameter('truth_odom_timeout', 0.5)

        weights_npz = self.get_parameter('policy_weights_npz').value
        self.policy = RLSuspensionPolicy(weights_npz_path=weights_npz)
        self.policy_mode = str(self.get_parameter('policy_mode').value)
        if self.policy_mode not in ('auto', 'wheel_only', 'suspension_wheel'):
            raise ValueError(f'unsupported policy_mode={self.policy_mode!r}')
        if self.policy_mode == 'wheel_only' and self.policy.action_dim != 6:
            raise ValueError('policy_mode=wheel_only requires a 6D actor')
        if self.policy_mode == 'suspension_wheel' and self.policy.action_dim != 12:
            raise ValueError('policy_mode=suspension_wheel requires a 12D actor')
        self.cmd_limiter = CmdVelLimiter(
            max_abs_vx=float(self.get_parameter('max_abs_cmd_vx').value),
            max_abs_wz=float(self.get_parameter('max_abs_cmd_wz').value),
        )
        self.cmd_vx, self.cmd_wz = self.cmd_limiter.clamp(
            float(self.get_parameter('cmd_vx').value),
            float(self.get_parameter('cmd_wz').value),
        )
        self.velocity_source = str(self.get_parameter('velocity_source').value)
        if self.velocity_source not in ('auto', 'truth_odom', 'wheel'):
            raise ValueError(f'unsupported velocity_source={self.velocity_source!r}')
        self.truth_odom_timeout = float(self.get_parameter('truth_odom_timeout').value)

        self.proj_grav = (0.0, 0.0, -1.0)
        self.ang_vel = (0.0, 0.0, 0.0)
        self.truth_lin_vel = (0.0, 0.0, 0.0)
        self.truth_ang_vel = (0.0, 0.0, 0.0)
        self.truth_odom_seen_time = 0.0
        self.joint_pos = {leg: 0.0 for leg in LEGS}
        self.joint_vel = {leg: 0.0 for leg in LEGS}
        self.wheel_vel = {leg: 0.0 for leg in LEGS}
        self.wheel_load_force = {leg: 0.0 for leg in LEGS}  # ||F|| in N
        self.prev_action = np.zeros(self.policy.action_dim, dtype=np.float32)
        self.joint_seen = False

        # Suspension: feedforward effort = stiffness * clamped_target
        self.susp_pub = self.create_publisher(
            Float64MultiArray, '/suspension_controller/commands', 10)
        # Per-wheel velocity (replaces diff_drive_controller for RL path)
        self.wheel_pub = self.create_publisher(
            Float64MultiArray, '/wheel_velocity_controller/commands', 10)

        self.create_subscription(Imu, '/imu/data', self.imu_cb, 50)
        self.create_subscription(JointState, '/joint_states', self.joint_cb, 50)
        self.create_subscription(Twist, '/cmd_vel', self.cmd_vel_cb, 10)
        self.create_subscription(
            Odometry,
            str(self.get_parameter('truth_odom_topic').value),
            self.truth_odom_cb,
            20,
        )
        for leg in LEGS:
            self.create_subscription(
                Wrench, f'/ft_wheel/{leg}',
                lambda msg, l=leg: self.ft_wheel_cb(msg, l),
                50,
            )

        rate = self.get_parameter('rate').value
        self.create_timer(1.0 / rate, self.step)
        self.wheel_only = self.policy.action_dim == 6
        self.get_logger().info(
            f'RL policy started (action_dim={self.policy.action_dim}, obs_dim={self.policy.obs_dim}, '
            f'mode={"wheel_only" if self.wheel_only else "suspension_wheel"}, '
            f'velocity_source={self.velocity_source}, '
            f'cmd_vx={self.cmd_vx} m/s, cmd_wz={self.cmd_wz} rad/s, '
            f'policy_weights_npz={weights_npz or "<embedded>"}).')

    def imu_cb(self, msg: Imu):
        q = msg.orientation
        self.proj_grav = projected_gravity(q.w, q.x, q.y, q.z)
        self.ang_vel = (msg.angular_velocity.x, msg.angular_velocity.y, msg.angular_velocity.z)

    def joint_cb(self, msg: JointState):
        for i, name in enumerate(msg.name):
            if name.startswith('susp_') and name.endswith('_joint'):
                leg = name[5:-6]
                if leg in self.joint_pos:
                    self.joint_pos[leg] = msg.position[i]
                    if i < len(msg.velocity):
                        self.joint_vel[leg] = msg.velocity[i]
            elif name.startswith('wheel_') and name.endswith('_joint'):
                leg = name[6:-6]
                if leg in self.wheel_vel and i < len(msg.velocity):
                    self.wheel_vel[leg] = msg.velocity[i]
        self.joint_seen = True

    def ft_wheel_cb(self, msg: Wrench, leg: str):
        fx, fy, fz = msg.force.x, msg.force.y, msg.force.z
        self.wheel_load_force[leg] = (fx * fx + fy * fy + fz * fz) ** 0.5

    def cmd_vel_cb(self, msg: Twist):
        self.cmd_vx, self.cmd_wz = self.cmd_limiter.clamp(msg.linear.x, msg.angular.z)

    def truth_odom_cb(self, msg: Odometry):
        self.truth_lin_vel = (
            msg.twist.twist.linear.x,
            msg.twist.twist.linear.y,
            msg.twist.twist.linear.z,
        )
        self.truth_ang_vel = (
            msg.twist.twist.angular.x,
            msg.twist.twist.angular.y,
            msg.twist.twist.angular.z,
        )
        self.truth_odom_seen_time = time.monotonic()

    def _root_velocity_obs(self) -> tuple[tuple[float, float, float], tuple[float, float, float]]:
        wheel_lin_vel = (mean_wheel_forward_velocity(self.wheel_vel), 0.0, 0.0)
        wheel_ang_vel = self.ang_vel

        truth_available = (time.monotonic() - self.truth_odom_seen_time) <= self.truth_odom_timeout
        if self.velocity_source == 'truth_odom':
            return (self.truth_lin_vel, self.truth_ang_vel) if truth_available else ((0.0, 0.0, 0.0), self.ang_vel)
        if self.velocity_source == 'wheel':
            return wheel_lin_vel, wheel_ang_vel
        if truth_available:
            return self.truth_lin_vel, self.truth_ang_vel
        return wheel_lin_vel, wheel_ang_vel

    def step(self):
        if not self.joint_seen:
            return

        lin_vel_b, ang_vel_b = self._root_velocity_obs()

        wheel_load = [
            wheel_load_normalized(self.wheel_load_force[leg])
            for leg in LEGS
        ]

        obs_values = (
            list(self.proj_grav)
            + list(ang_vel_b)
            + list(lin_vel_b)
            + [self.joint_pos[leg] for leg in LEGS]
            + [self.joint_vel[leg] for leg in LEGS]
            + [self.wheel_vel[leg] for leg in LEGS]
            + wheel_load
            + [self.cmd_vx, self.cmd_wz]
            + list(self.prev_action)
        )
        obs = np.array(obs_values, dtype=np.float32)
        action = self.policy.act(obs)
        self.prev_action = action

        if self.wheel_only:
            wheel_actions = action
        else:
            # action[0:6] -> suspension joint target through explicit bounded PD actuator
            susp_efforts = []
            for i in range(6):
                leg = LEGS[i]
                target = float(action[i]) * ACTION_SCALE_SUSP
                target = clamp_abs(target, SUSP_JOINT_LIMIT)
                q = self.joint_pos[leg]
                qd = self.joint_vel[leg]
                effort = SUSP_ACTUATOR_KP * (target - q) - SUSP_ACTUATOR_KD * qd
                effort = clamp_abs(effort, SUSP_EFFORT_LIMIT)
                susp_efforts.append(effort)
            self.susp_pub.publish(Float64MultiArray(data=susp_efforts))
            wheel_actions = action[6:12]

        wheel_cmds = [float(wheel_actions[i]) * ACTION_SCALE_WHEEL_OMEGA for i in range(6)]
        self.wheel_pub.publish(Float64MultiArray(data=wheel_cmds))


def main():
    rclpy.init()
    node = RLSuspensionPolicyNode()
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
