"""Manual cmd_vel baseline on the same per-wheel execution surface as RL."""

import rclpy
from geometry_msgs.msg import Twist
from rclpy.node import Node
from std_msgs.msg import Float64MultiArray

from .control_interfaces import CmdVelLimiter, clamp_abs, skid_steer_wheel_speeds


class CmdVelWheelBaseline(Node):
    def __init__(self):
        super().__init__("cmd_vel_wheel_baseline")
        self.declare_parameter("rate", 30.0)
        self.declare_parameter("cmd_vx", 0.0)
        self.declare_parameter("cmd_wz", 0.0)
        self.declare_parameter("max_abs_cmd_vx", 0.3)
        self.declare_parameter("max_abs_cmd_wz", 0.4)
        self.declare_parameter("max_abs_wheel_omega", 3.0)
        self.declare_parameter("cmd_timeout", 0.5)

        self.limiter = CmdVelLimiter(
            max_abs_vx=float(self.get_parameter("max_abs_cmd_vx").value),
            max_abs_wz=float(self.get_parameter("max_abs_cmd_wz").value),
        )
        self.cmd_vx, self.cmd_wz = self.limiter.clamp(
            float(self.get_parameter("cmd_vx").value),
            float(self.get_parameter("cmd_wz").value),
        )
        self.max_abs_wheel_omega = abs(float(self.get_parameter("max_abs_wheel_omega").value))
        self.cmd_timeout = float(self.get_parameter("cmd_timeout").value)
        self.last_cmd_time = self.get_clock().now()
        self.external_cmd_seen = False

        self.pub = self.create_publisher(Float64MultiArray, "/wheel_velocity_controller/commands", 10)
        self.create_subscription(Twist, "/cmd_vel", self.cmd_vel_cb, 10)
        self.create_timer(1.0 / float(self.get_parameter("rate").value), self.step)
        self.get_logger().info(
            "cmd_vel wheel baseline started "
            f"(cmd_vx={self.cmd_vx:.3f} m/s, cmd_wz={self.cmd_wz:.3f} rad/s, "
            f"wheel_limit={self.max_abs_wheel_omega:.2f} rad/s)."
        )

    def cmd_vel_cb(self, msg: Twist):
        self.cmd_vx, self.cmd_wz = self.limiter.clamp(msg.linear.x, msg.angular.z)
        self.last_cmd_time = self.get_clock().now()
        self.external_cmd_seen = True

    def _command_is_fresh(self) -> bool:
        if not self.external_cmd_seen:
            return True
        age = (self.get_clock().now() - self.last_cmd_time).nanoseconds * 1e-9
        return age <= self.cmd_timeout

    def step(self):
        vx = self.cmd_vx if self._command_is_fresh() else 0.0
        wz = self.cmd_wz if self._command_is_fresh() else 0.0
        wheel_cmds = [
            clamp_abs(omega, self.max_abs_wheel_omega)
            for omega in skid_steer_wheel_speeds(vx, wz)
        ]
        self.pub.publish(Float64MultiArray(data=wheel_cmds))


def main():
    rclpy.init()
    node = CmdVelWheelBaseline()
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
