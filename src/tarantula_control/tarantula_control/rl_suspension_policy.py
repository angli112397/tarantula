"""M7 v5 RL 悬挂+驱动策略节点 —— 每关节独立直驱，无运动学映射。

v5 架构变化（相比 v4）：
  - action[0:6]  -> 6 个悬挂关节角度目标（直接，无几何映射）
    Gazebo 等价：effort_i = FF_STIFFNESS * clamp(action[i] * ACTION_SCALE_SUSP, -0.6, 0.6)
  - action[6:12] -> 6 个驱动轮各自速度目标（取代 diff_drive 左/右分组）
    发布到 /wheel_velocity_controller/commands (Float64MultiArray, LEGS 顺序)
  - 移除 diff_drive_controller 订阅/发布
  - 移除几何运动学映射（dz/dq/DIR）
  - 新增订阅 /ft_wheel/{leg} ×6 作为 wheel_contact(6D) 观测
  - 线速度由平均轮速 × 轮半径估算（不再依赖 /odom）

观测构成（47 维，与 suspension_env._get_observations 一致）：
  projected_gravity_b(3)  <- /imu/data
  root_ang_vel_b(3)        <- /imu/data
  root_lin_vel_b(3)        <- 平均轮速估算
  susp_joint_pos(6)        <- /joint_states (LEGS 顺序)
  susp_joint_vel(6)        <- /joint_states
  wheel_joint_vel(6)       <- /joint_states
  wheel_in_contact(6)      <- /ft_wheel/{leg}，||F|| > 5 N
  move_cmd(1)              <- ROS 参数
  heading_rate_cmd(1)      <- ROS 参数
  prev_action(12)          <- 上一步策略输出

几何常量（与 suspension_env_cfg.py 一致）：
  ACTION_SCALE_SUSP = 0.5 rad  (Isaac: position target ±0.5)
  FF_STIFFNESS = 120.0 Nm/rad  (Gazebo: effort = stiffness * clamped_target)
"""
import numpy as np
import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Wrench
from sensor_msgs.msg import Imu, JointState
from std_msgs.msg import Float64MultiArray

from .rl_policy import ACTION_SCALE_SUSP, ACTION_SCALE_WHEEL_OMEGA, RLSuspensionPolicy
from .suspension_core import LEGS, SuspensionConfig, projected_gravity

_geom = SuspensionConfig()
_FF_STIFFNESS = _geom.ff_stiffness      # 120.0, must equal URDF springStiffness
_CONTACT_THRESHOLD = _geom.contact_force_threshold  # 5.0 N

WHEEL_RADIUS = 0.12   # m, matches diff_drive_controller wheel_radius
_SUSP_LIMIT = 0.6     # rad, hard joint limit from URDF


class RLSuspensionPolicyNode(Node):
    def __init__(self):
        super().__init__('rl_suspension_policy')
        self.declare_parameter('rate', 30.0)
        self.declare_parameter('move_cmd', 1.0)
        self.declare_parameter('heading_rate_cmd', 0.0)

        self.policy = RLSuspensionPolicy()
        self.move_cmd = self.get_parameter('move_cmd').value
        self.heading_rate_cmd = self.get_parameter('heading_rate_cmd').value

        self.proj_grav = (0.0, 0.0, -1.0)
        self.ang_vel = (0.0, 0.0, 0.0)
        self.joint_pos = {leg: 0.0 for leg in LEGS}
        self.joint_vel = {leg: 0.0 for leg in LEGS}
        self.wheel_vel = {leg: 0.0 for leg in LEGS}
        self.contact_force = {leg: 0.0 for leg in LEGS}  # ||F|| in N
        self.prev_action = np.zeros(12, dtype=np.float32)
        self.joint_seen = False

        # Suspension: feedforward effort = stiffness * clamped_target
        self.susp_pub = self.create_publisher(
            Float64MultiArray, '/suspension_controller/commands', 10)
        # Per-wheel velocity (replaces diff_drive_controller for RL path)
        self.wheel_pub = self.create_publisher(
            Float64MultiArray, '/wheel_velocity_controller/commands', 10)

        self.create_subscription(Imu, '/imu/data', self.imu_cb, 50)
        self.create_subscription(JointState, '/joint_states', self.joint_cb, 50)
        for leg in LEGS:
            self.create_subscription(
                Wrench, f'/ft_wheel/{leg}',
                lambda msg, l=leg: self.ft_wheel_cb(msg, l),
                50,
            )

        rate = self.get_parameter('rate').value
        self.create_timer(1.0 / rate, self.step)
        self.get_logger().info(
            f'RL suspension+drive policy started (M7 v5, per-joint direct control, '
            f'move_cmd={self.move_cmd}, heading_rate_cmd={self.heading_rate_cmd} rad/s).')

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
        self.contact_force[leg] = (fx * fx + fy * fy + fz * fz) ** 0.5

    def step(self):
        if not self.joint_seen:
            return

        # Estimate forward velocity from mean wheel speed
        mean_omega = sum(self.wheel_vel[leg] for leg in LEGS) / len(LEGS)
        lin_vel_x = mean_omega * WHEEL_RADIUS

        in_contact = [1.0 if self.contact_force[leg] > _CONTACT_THRESHOLD else 0.0
                      for leg in LEGS]

        obs = np.array(
            list(self.proj_grav)
            + list(self.ang_vel)
            + [lin_vel_x, 0.0, 0.0]
            + [self.joint_pos[leg] for leg in LEGS]
            + [self.joint_vel[leg] for leg in LEGS]
            + [self.wheel_vel[leg] for leg in LEGS]
            + in_contact
            + [self.move_cmd, self.heading_rate_cmd]
            + list(self.prev_action),
            dtype=np.float32,
        )  # 47D
        action = self.policy.act(obs)
        self.prev_action = action

        # action[0:6] -> suspension joint effort (feedforward stiffness * clamped angle)
        susp_efforts = []
        for i in range(6):
            target = float(action[i]) * ACTION_SCALE_SUSP
            target = max(-_SUSP_LIMIT, min(_SUSP_LIMIT, target))
            susp_efforts.append(_FF_STIFFNESS * target)
        self.susp_pub.publish(Float64MultiArray(data=susp_efforts))

        # action[6:12] -> per-wheel velocity command (rad/s, LEGS order)
        wheel_cmds = [float(action[6 + i]) * ACTION_SCALE_WHEEL_OMEGA for i in range(6)]
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
