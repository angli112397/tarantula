"""Gazebo-only truth odometry publisher for RL deployment checks.

This node samples `ign model -m <model> -p` and publishes a nav_msgs/Odometry
message whose twist is expressed in the robot body frame. It is intentionally a
simulation adapter, not a real-robot interface.
"""

import math
import subprocess
import time
from dataclasses import dataclass

import rclpy
from geometry_msgs.msg import Quaternion
from nav_msgs.msg import Odometry
from rclpy.node import Node


@dataclass(frozen=True)
class TruthPose:
    x: float
    y: float
    z: float
    roll: float
    pitch: float
    yaw: float


def _wrap_angle(angle: float) -> float:
    return math.atan2(math.sin(angle), math.cos(angle))


def _quat_from_rpy(roll: float, pitch: float, yaw: float) -> Quaternion:
    cr = math.cos(roll * 0.5)
    sr = math.sin(roll * 0.5)
    cp = math.cos(pitch * 0.5)
    sp = math.sin(pitch * 0.5)
    cy = math.cos(yaw * 0.5)
    sy = math.sin(yaw * 0.5)
    return Quaternion(
        x=sr * cp * cy - cr * sp * sy,
        y=cr * sp * cy + sr * cp * sy,
        z=cr * cp * sy - sr * sp * cy,
        w=cr * cp * cy + sr * sp * sy,
    )


def _world_to_body_velocity(vx: float, vy: float, vz: float, pose: TruthPose) -> tuple[float, float, float]:
    cr = math.cos(pose.roll)
    sr = math.sin(pose.roll)
    cp = math.cos(pose.pitch)
    sp = math.sin(pose.pitch)
    cy = math.cos(pose.yaw)
    sy = math.sin(pose.yaw)

    # R = Rz(yaw) * Ry(pitch) * Rx(roll). Body velocity = R^T * world velocity.
    r00 = cy * cp
    r01 = cy * sp * sr - sy * cr
    r02 = cy * sp * cr + sy * sr
    r10 = sy * cp
    r11 = sy * sp * sr + cy * cr
    r12 = sy * sp * cr - cy * sr
    r20 = -sp
    r21 = cp * sr
    r22 = cp * cr

    return (
        r00 * vx + r10 * vy + r20 * vz,
        r01 * vx + r11 * vy + r21 * vz,
        r02 * vx + r12 * vy + r22 * vz,
    )


def _read_truth_pose(model_name: str, timeout_s: float) -> TruthPose:
    result = subprocess.run(
        ["ign", "model", "-m", model_name, "-p"],
        check=False,
        capture_output=True,
        text=True,
        timeout=timeout_s,
    )
    lines = result.stdout.strip().splitlines()
    if result.returncode != 0 or len(lines) < 2:
        detail = (result.stderr or result.stdout).strip()
        raise RuntimeError(f"ign model pose failed: {detail}")
    xyz = [float(v) for v in lines[-2].strip().strip("[]").split()]
    rpy = [float(v) for v in lines[-1].strip().strip("[]").split()]
    return TruthPose(x=xyz[0], y=xyz[1], z=xyz[2], roll=rpy[0], pitch=rpy[1], yaw=rpy[2])


class GazeboTruthOdometry(Node):
    def __init__(self):
        super().__init__("gazebo_truth_odometry")
        self.declare_parameter("model_name", "tarantula")
        self.declare_parameter("frame_id", "odom")
        self.declare_parameter("child_frame_id", "base_link")
        self.declare_parameter("topic", "/tarantula/truth_odom")
        self.declare_parameter("rate", 5.0)
        self.declare_parameter("ign_timeout", 1.0)

        self.model_name = str(self.get_parameter("model_name").value)
        self.frame_id = str(self.get_parameter("frame_id").value)
        self.child_frame_id = str(self.get_parameter("child_frame_id").value)
        topic = str(self.get_parameter("topic").value)
        rate = float(self.get_parameter("rate").value)
        self.ign_timeout = float(self.get_parameter("ign_timeout").value)

        self.pub = self.create_publisher(Odometry, topic, 10)
        self.prev_pose: TruthPose | None = None
        self.prev_time: float | None = None
        self.last_error_log = 0.0

        self.create_timer(1.0 / rate, self.step)
        self.get_logger().info(
            f"Gazebo truth odometry started (model={self.model_name}, topic={topic}, rate={rate:.1f} Hz)."
        )

    def step(self):
        now_wall = time.monotonic()
        try:
            pose = _read_truth_pose(self.model_name, self.ign_timeout)
        except Exception as exc:
            if now_wall - self.last_error_log > 2.0:
                self.get_logger().warn(f"Waiting for Gazebo truth pose: {exc}")
                self.last_error_log = now_wall
            return

        odom = Odometry()
        odom.header.stamp = self.get_clock().now().to_msg()
        odom.header.frame_id = self.frame_id
        odom.child_frame_id = self.child_frame_id
        odom.pose.pose.position.x = pose.x
        odom.pose.pose.position.y = pose.y
        odom.pose.pose.position.z = pose.z
        odom.pose.pose.orientation = _quat_from_rpy(pose.roll, pose.pitch, pose.yaw)

        if self.prev_pose is not None and self.prev_time is not None:
            dt = max(now_wall - self.prev_time, 1e-6)
            vx_w = (pose.x - self.prev_pose.x) / dt
            vy_w = (pose.y - self.prev_pose.y) / dt
            vz_w = (pose.z - self.prev_pose.z) / dt
            vx_b, vy_b, vz_b = _world_to_body_velocity(vx_w, vy_w, vz_w, pose)
            odom.twist.twist.linear.x = vx_b
            odom.twist.twist.linear.y = vy_b
            odom.twist.twist.linear.z = vz_b
            odom.twist.twist.angular.x = _wrap_angle(pose.roll - self.prev_pose.roll) / dt
            odom.twist.twist.angular.y = _wrap_angle(pose.pitch - self.prev_pose.pitch) / dt
            odom.twist.twist.angular.z = _wrap_angle(pose.yaw - self.prev_pose.yaw) / dt

        self.pub.publish(odom)
        self.prev_pose = pose
        self.prev_time = now_wall


def main():
    rclpy.init()
    node = GazeboTruthOdometry()
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
