"""ROS2 wrapper for the classical skid-steer motion baseline.

This node owns only planar motion:

``/cmd_vel -> six wheel velocity targets``.

cmd_vel is passed directly to the skid-steer differential without any
stop-turn-drive shaping. RL active-suspension runs in a separate node and
only commands the six hip joints.
"""

from __future__ import annotations

import time

import rclpy
from geometry_msgs.msg import Twist
from rcl_interfaces.msg import SetParametersResult
from rclpy.node import Node
from sensor_msgs.msg import Imu, JointState
from std_msgs.msg import Float64MultiArray

from .motion_control import MotionControlConfig, SkidSteerMotionController


class MotionControlNode(Node):
    def __init__(self):
        super().__init__("motion_control_node")
        defaults = MotionControlConfig()
        self.declare_parameter("rate", 30.0)
        self.declare_parameter("cmd_vx", 0.0)
        self.declare_parameter("cmd_wz", 0.0)
        self.declare_parameter("max_abs_cmd_vx", defaults.max_abs_cmd_vx)
        self.declare_parameter("max_abs_cmd_wz", defaults.max_abs_cmd_wz)
        self.declare_parameter("max_abs_wheel_omega", defaults.max_abs_wheel_omega)
        self.declare_parameter("drive_scale", defaults.drive_scale)
        self.declare_parameter("pure_turn_track_scale", defaults.pure_turn_track_scale)
        self.declare_parameter("yaw_rate_kp", defaults.yaw_rate_kp)
        self.declare_parameter("yaw_rate_ki", defaults.yaw_rate_ki)
        self.declare_parameter("yaw_integral_limit", defaults.yaw_integral_limit)
        self.declare_parameter("max_wheel_accel", defaults.max_wheel_accel)

        control_config = MotionControlConfig(
            max_abs_cmd_vx=float(self.get_parameter("max_abs_cmd_vx").value),
            max_abs_cmd_wz=float(self.get_parameter("max_abs_cmd_wz").value),
            max_abs_wheel_omega=float(self.get_parameter("max_abs_wheel_omega").value),
            drive_scale=float(self.get_parameter("drive_scale").value),
            pure_turn_track_scale=float(self.get_parameter("pure_turn_track_scale").value),
            yaw_rate_kp=float(self.get_parameter("yaw_rate_kp").value),
            yaw_rate_ki=float(self.get_parameter("yaw_rate_ki").value),
            yaw_integral_limit=float(self.get_parameter("yaw_integral_limit").value),
            max_wheel_accel=float(self.get_parameter("max_wheel_accel").value),
        )
        self.motion_controller = SkidSteerMotionController(control_config)
        initial_command = self.motion_controller.limit_command(
            float(self.get_parameter("cmd_vx").value),
            float(self.get_parameter("cmd_wz").value),
        )
        self.cmd_vx, self.cmd_wz = initial_command.vx, initial_command.wz

        self.ang_vel = (0.0, 0.0, 0.0)
        self.joint_seen = False
        self.last_step_time = time.monotonic()

        self.wheel_pub = self.create_publisher(Float64MultiArray, "/wheel_velocity_controller/commands", 10)
        self.status_pub = self.create_publisher(Float64MultiArray, "/motion_control/status", 10)

        self.create_subscription(Imu, "/imu/data", self.imu_cb, 50)
        self.create_subscription(JointState, "/joint_states", self.joint_cb, 50)
        self.create_subscription(Twist, "/cmd_vel", self.cmd_vel_cb, 10)

        rate = float(self.get_parameter("rate").value)
        self.create_timer(1.0 / rate, self.step)
        self.add_on_set_parameters_callback(self._on_parameter_update)
        self.get_logger().info(
            "classical motion controller started "
            f"(max_abs_wheel_omega={self.motion_controller.config.max_abs_wheel_omega}, "
            f"drive_scale={self.motion_controller.config.drive_scale}, "
            f"pure_turn_track_scale={self.motion_controller.config.pure_turn_track_scale}, "
            f"yaw_rate_kp={self.motion_controller.config.yaw_rate_kp}, "
            f"yaw_rate_ki={self.motion_controller.config.yaw_rate_ki}, "
            f"max_wheel_accel={self.motion_controller.config.max_wheel_accel}, "
            f"cmd_vx={self.cmd_vx} m/s, cmd_wz={self.cmd_wz} rad/s)."
        )

    def _on_parameter_update(self, params):
        updates = {}
        for param in params:
            if param.name in {
                "max_abs_cmd_vx",
                "max_abs_cmd_wz",
                "max_abs_wheel_omega",
                "drive_scale",
                "pure_turn_track_scale",
                "yaw_rate_kp",
                "yaw_rate_ki",
                "yaw_integral_limit",
                "max_wheel_accel",
            }:
                updates[param.name] = float(param.value)
        if updates:
            self.motion_controller.update_config(**updates)
            self.motion_controller.reset_feedback()
            self.get_logger().info(
                "updated motion control parameters: "
                + ", ".join(f"{key}={value}" for key, value in sorted(updates.items()))
            )
        return SetParametersResult(successful=True)

    def imu_cb(self, msg: Imu):
        self.ang_vel = (msg.angular_velocity.x, msg.angular_velocity.y, msg.angular_velocity.z)

    def joint_cb(self, msg: JointState):
        self.joint_seen = True

    def cmd_vel_cb(self, msg: Twist):
        command = self.motion_controller.limit_command(msg.linear.x, msg.angular.z)
        self.cmd_vx, self.cmd_wz = command.vx, command.wz

    def step(self):
        if not self.joint_seen:
            return

        now = time.monotonic()
        dt = max(now - self.last_step_time, 0.0)
        self.last_step_time = now

        command = self.motion_controller.limit_command(self.cmd_vx, self.cmd_wz)
        wheel_cmds = self.motion_controller.filtered_wheel_targets(
            command,
            measured_wz=self.ang_vel[2],
            dt=dt,
        )
        self.wheel_pub.publish(Float64MultiArray(data=wheel_cmds))
        self.status_pub.publish(Float64MultiArray(data=[
            max((abs(float(v)) for v in wheel_cmds), default=0.0),
            float(command.vx),
            float(command.wz),
            float(self.ang_vel[2]),
            0.0,  # reserved
        ]))


def main():
    rclpy.init()
    node = MotionControlNode()
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
