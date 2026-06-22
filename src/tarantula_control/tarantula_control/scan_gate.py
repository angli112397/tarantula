"""姿态门控扫描过滤：车身倾斜超阈值时丢弃整帧 2D 扫描。

崎岖地形上 2D 雷达的根本缺陷：车身倾斜瞬间扫描面打地，在 2-3m 外
画出横贯走廊的幻影障碍（实测把局部 costmap 堵死导致导航死锁）。
本节点在 /scan 与 SLAM/Nav2 之间做门控——倾斜帧直接丢弃（不标记
也不清除，等回平后真实扫描自然修正）。与主动调平天然协同：调平把
车身长期压在阈值内，感知可用时间窗最大化（被动悬挂在 8 度坡上会
永久超阈值、建图饿死，这本身就是调平价值的 A/B 证据）。

订阅  /scan (LaserScan)、/imu/data (Imu)
发布  /scan_gated (LaserScan)
参数  tilt_gate (rad，默认 0.05 ≈ 3 度)
参数  output_frame (string，默认 lidar_link)：把 Gazebo scoped sensor frame
      收敛到 URDF frame。
"""
import math
from copy import deepcopy

import rclpy
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import Imu, LaserScan

from .suspension_core import quat_roll_pitch


class ScanGate(Node):

    def __init__(self):
        super().__init__('scan_gate')
        self.declare_parameter('tilt_gate', 0.05)
        self.declare_parameter('output_frame', 'lidar_link')
        self.gate = self.get_parameter('tilt_gate').value
        self.output_frame = str(self.get_parameter('output_frame').value)
        self.tilt = 0.0
        self.dropped = 0
        self.create_subscription(Imu, '/imu/data', self.imu_cb,
                                 qos_profile_sensor_data)
        self.create_subscription(LaserScan, '/scan', self.scan_cb,
                                 qos_profile_sensor_data)
        self.pub = self.create_publisher(LaserScan, '/scan_gated',
                                         qos_profile_sensor_data)

    def imu_cb(self, msg):
        q = msg.orientation
        roll, pitch = quat_roll_pitch(q.w, q.x, q.y, q.z)
        self.tilt = math.sqrt(roll * roll + pitch * pitch)

    def scan_cb(self, msg):
        if self.tilt > self.gate:
            self.dropped += 1
            if self.dropped % 50 == 1:
                self.get_logger().info(
                    f'倾斜 {math.degrees(self.tilt):.1f}° 超阈值，丢弃扫描'
                    f'（累计 {self.dropped}）')
            return
        out = deepcopy(msg)
        if self.output_frame:
            out.header.frame_id = self.output_frame
        self.pub.publish(out)


def main():
    rclpy.init()
    rclpy.spin(ScanGate())


if __name__ == '__main__':
    main()
