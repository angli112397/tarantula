"""Wheel-encoder odometry source for robot_localization.

This node estimates planar skid-steer odometry from the six wheel joint
velocities. It intentionally does not publish TF; robot_localization owns the
deployable odom->base_link transform after fusing this source with IMU data.
"""

from __future__ import annotations

import math

import rclpy
from geometry_msgs.msg import Quaternion
from nav_msgs.msg import Odometry
from rclpy.node import Node
from sensor_msgs.msg import JointState

from .control_interfaces import EFFECTIVE_TRACK, LEFT_LEGS, RIGHT_LEGS, WHEEL_DIRECTION, WHEEL_RADIUS
from .motion_control import MotionControlConfig
from .suspension_core import LEGS


def _quat_from_yaw(yaw: float) -> Quaternion:
    half = 0.5 * yaw
    return Quaternion(x=0.0, y=0.0, z=math.sin(half), w=math.cos(half))


class WheelOdometryNode(Node):
    def __init__(self):
        super().__init__("wheel_odometry_node")
        defaults = MotionControlConfig()
        self.declare_parameter("odom_topic", "/wheel/odom")
        self.declare_parameter("frame_id", "odom")
        self.declare_parameter("child_frame_id", "base_link")
        self.declare_parameter("effective_track_scale", defaults.pure_turn_track_scale)
        self.declare_parameter("publish_rate_limit", 50.0)

        self.odom_topic = str(self.get_parameter("odom_topic").value)
        self.frame_id = str(self.get_parameter("frame_id").value)
        self.child_frame_id = str(self.get_parameter("child_frame_id").value)
        self.effective_track = EFFECTIVE_TRACK * max(
            1.0e-6,
            float(self.get_parameter("effective_track_scale").value),
        )
        self.publish_period_ns = int(1.0e9 / max(1.0, float(self.get_parameter("publish_rate_limit").value)))

        self.x = 0.0
        self.y = 0.0
        self.yaw = 0.0
        self.last_stamp_ns: int | None = None
        self.last_publish_ns: int | None = None

        self.pub = self.create_publisher(Odometry, self.odom_topic, 10)
        self.create_subscription(JointState, "/joint_states", self.joint_cb, 50)
        self.get_logger().info(
            "wheel odometry started "
            f"(topic={self.odom_topic}, effective_track={self.effective_track:.4f} m, "
            f"wheel_radius={WHEEL_RADIUS:.4f} m, publish_tf=false)."
        )

    def joint_cb(self, msg: JointState):
        # Use joint state measurement timestamp (sim time) rather than callback arrival time.
        s = msg.header.stamp
        stamp_ns = s.sec * 1_000_000_000 + s.nanosec
        if stamp_ns == 0:
            stamp_ns = self.get_clock().now().nanoseconds
        if self.last_publish_ns is not None and stamp_ns - self.last_publish_ns < self.publish_period_ns:
            return

        wheel_vel = {}
        for i, name in enumerate(msg.name):
            if not (name.startswith("wheel_") and name.endswith("_joint")):
                continue
            leg = name[6:-6]
            if leg in LEGS and i < len(msg.velocity):
                wheel_vel[leg] = float(msg.velocity[i]) * WHEEL_DIRECTION[leg]
        if any(leg not in wheel_vel for leg in LEGS):
            return

        left = sum(wheel_vel[leg] for leg in LEFT_LEGS) / len(LEFT_LEGS) * WHEEL_RADIUS
        right = sum(wheel_vel[leg] for leg in RIGHT_LEGS) / len(RIGHT_LEGS) * WHEEL_RADIUS
        vx = 0.5 * (left + right)
        wz = (right - left) / self.effective_track

        if self.last_stamp_ns is not None:
            dt = max((stamp_ns - self.last_stamp_ns) * 1.0e-9, 0.0)
            if dt > 0.0:
                mid_yaw = self.yaw + 0.5 * wz * dt
                self.x += vx * math.cos(mid_yaw) * dt
                self.y += vx * math.sin(mid_yaw) * dt
                self.yaw = math.atan2(math.sin(self.yaw + wz * dt), math.cos(self.yaw + wz * dt))

        odom = Odometry()
        odom.header.stamp = msg.header.stamp
        odom.header.frame_id = self.frame_id
        odom.child_frame_id = self.child_frame_id
        odom.pose.pose.position.x = self.x
        odom.pose.pose.position.y = self.y
        odom.pose.pose.position.z = 0.0
        odom.pose.pose.orientation = _quat_from_yaw(self.yaw)
        odom.twist.twist.linear.x = vx
        odom.twist.twist.angular.z = wz

        odom.pose.covariance[0] = 0.05
        odom.pose.covariance[7] = 0.05
        odom.pose.covariance[35] = 0.10
        odom.twist.covariance[0] = 0.02
        odom.twist.covariance[35] = 0.05

        self.pub.publish(odom)
        self.last_stamp_ns = stamp_ns
        self.last_publish_ns = stamp_ns


def main():
    rclpy.init()
    node = WheelOdometryNode()
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


if __name__ == "__main__":
    main()
