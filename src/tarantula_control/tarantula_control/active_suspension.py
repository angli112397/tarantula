"""主动悬挂 ROS2 适配层 —— 算法在 suspension_core（仿真器无关），
本文件只做 ROS 管道：参数声明、话题订阅/发布、定时器。
Isaac Lab 不走本文件：env 直接 import suspension_core（见其模块文档）。

话题：
  入  /imu/data                         姿态/角速度（外环反馈）
  入  /joint_states                     悬挂关节角（遥测/守门）
  入  contact/{fl..rr}                  轮地接触（M1 接触保持，默认未启用）
  入  ~/body_cmd  Float64MultiArray     [roll_ref, pitch_ref, height_m]
                                        车身位姿指令（M2，动作面）
  出  /suspension_controller/commands   六腿前馈力矩
  出  ~/debug                           [roll,pitch,u_roll,u_pitch,
                                         q_target×6, q×6, tau×6,
                                         contact×6, probe_dz×6, height] = 35
"""
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Imu, JointState
from std_msgs.msg import Float64MultiArray
from gazebo_msgs.msg import ContactsState

from .suspension_core import (LEGS, SuspensionConfig, SuspensionController,
                              SuspensionInputs, config_fields, quat_roll_pitch)


class ActiveSuspension(Node):
    def __init__(self):
        super().__init__('active_suspension')

        # 参数面自动暴露：SuspensionConfig 字段名即 ROS 参数名
        for name, default in config_fields():
            self.declare_parameter(name, default)
        self.declare_parameter('control_rate', 100.0)
        cfg = SuspensionConfig(**{name: self.get_parameter(name).value
                                  for name, _ in config_fields()})
        self.ctrl = SuspensionController(cfg)

        self.inputs = SuspensionInputs()
        self.joint_seen = False

        self.cmd_pub = self.create_publisher(
            Float64MultiArray, '/suspension_controller/commands', 10)
        self.debug_pub = self.create_publisher(Float64MultiArray, '~/debug', 10)
        self.create_subscription(Imu, '/imu/data', self.imu_cb, 50)
        self.create_subscription(JointState, '/joint_states', self.joint_cb, 50)
        self.create_subscription(Float64MultiArray, '~/body_cmd', self.body_cb, 10)
        for leg in LEGS:
            self.create_subscription(
                ContactsState, f'contact/{leg}',
                lambda msg, leg=leg: self.contact_cb(leg, msg), 10)

        self.dt = 1.0 / self.get_parameter('control_rate').value
        self._step = 0
        self.create_timer(self.dt, self.control_step)
        self.get_logger().info('Active suspension started (core: v3 + M1/M2 channels).')

    def imu_cb(self, msg: Imu):
        q = msg.orientation
        self.inputs.roll, self.inputs.pitch = quat_roll_pitch(q.w, q.x, q.y, q.z)
        self.inputs.roll_rate = msg.angular_velocity.x
        self.inputs.pitch_rate = msg.angular_velocity.y

    def joint_cb(self, msg: JointState):
        for i, name in enumerate(msg.name):
            if name.startswith('susp_') and name.endswith('_joint'):
                self.inputs.joint_pos[name[5:-6]] = msg.position[i]
        self.joint_seen = True

    def contact_cb(self, leg, msg: ContactsState):
        self.inputs.contacts[leg] = len(msg.states) > 0

    def body_cb(self, msg: Float64MultiArray):
        if len(msg.data) >= 3:
            self.inputs.roll_ref = msg.data[0]
            self.inputs.pitch_ref = msg.data[1]
            self.inputs.height_cmd = msg.data[2]

    def control_step(self):
        if not self.joint_seen:
            return  # 核心时钟从首帧关节数据起算（落地保持期基准）
        out = self.ctrl.step(self.inputs, self.dt)
        self.cmd_pub.publish(Float64MultiArray(data=out.torques))

        self._step += 1
        if self._step % 10 == 0:
            dbg = ([self.inputs.roll, self.inputs.pitch, out.u_roll, out.u_pitch]
                   + [out.q_target[leg] for leg in LEGS]
                   + [self.inputs.joint_pos.get(leg, 0.0) for leg in LEGS]
                   + list(out.torques)
                   + [float(self.inputs.contacts.get(leg, True)) for leg in LEGS]
                   + [out.probe_dz[leg] for leg in LEGS]
                   + [out.height])
            self.debug_pub.publish(Float64MultiArray(data=dbg))


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
