"""Neutral stand hold controller for suspension physics bring-up.

This node is intentionally simpler than ``active_suspension``: it does not
level the body and does not infer terrain. It only drives the six suspension
joints toward a deployable stand target using bounded PD efforts. Use it for
wheel/contact open-loop tests before trusting RL or terrain conclusions.
"""

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import JointState
from std_msgs.msg import Float64MultiArray

from .suspension_core import LEGS


class StandSuspensionHold(Node):
    def __init__(self):
        super().__init__("stand_suspension_hold")
        self.declare_parameter("rate", 100.0)
        self.declare_parameter("target", 0.0)
        self.declare_parameter("kp", 95.0)
        self.declare_parameter("kd", 18.0)
        self.declare_parameter("effort_limit", 45.0)
        self.declare_parameter("target_ramp_rate", 0.18)

        self.target = float(self.get_parameter("target").value)
        self.kp = float(self.get_parameter("kp").value)
        self.kd = float(self.get_parameter("kd").value)
        self.effort_limit = float(self.get_parameter("effort_limit").value)
        self.target_ramp_rate = abs(float(self.get_parameter("target_ramp_rate").value))
        rate = float(self.get_parameter("rate").value)
        self.dt = 1.0 / rate

        self.pos = {leg: 0.0 for leg in LEGS}
        self.vel = {leg: 0.0 for leg in LEGS}
        self.cmd_target = {leg: 0.0 for leg in LEGS}
        self.joint_seen = False
        self.initialized_targets = False

        self.pub = self.create_publisher(Float64MultiArray, "/suspension_controller/commands", 10)
        self.debug_pub = self.create_publisher(Float64MultiArray, "~/debug", 10)
        self.create_subscription(JointState, "/joint_states", self.joint_cb, 50)
        self.create_timer(self.dt, self.step)
        self.get_logger().info(
            "Stand suspension hold started "
            f"(target={self.target:+.3f} rad, kp={self.kp:.1f}, kd={self.kd:.1f}, "
            f"effort_limit={self.effort_limit:.1f} Nm)."
        )

    def joint_cb(self, msg: JointState):
        for i, name in enumerate(msg.name):
            if name.startswith("susp_") and name.endswith("_joint"):
                leg = name[5:-6]
                if leg in self.pos:
                    if i < len(msg.position):
                        self.pos[leg] = float(msg.position[i])
                    if i < len(msg.velocity):
                        self.vel[leg] = float(msg.velocity[i])
        self.joint_seen = True

    def _init_targets_from_state(self):
        for leg in LEGS:
            self.cmd_target[leg] = self.pos[leg]
        self.initialized_targets = True

    def _ramp_targets(self):
        max_step = self.target_ramp_rate * self.dt
        for leg in LEGS:
            err = self.target - self.cmd_target[leg]
            if abs(err) <= max_step:
                self.cmd_target[leg] = self.target
            else:
                self.cmd_target[leg] += max_step if err > 0.0 else -max_step

    def step(self):
        if not self.joint_seen:
            return
        if not self.initialized_targets:
            self._init_targets_from_state()

        self._ramp_targets()
        efforts = []
        for leg in LEGS:
            effort = self.kp * (self.cmd_target[leg] - self.pos[leg]) - self.kd * self.vel[leg]
            effort = max(-self.effort_limit, min(self.effort_limit, effort))
            efforts.append(effort)
        self.pub.publish(Float64MultiArray(data=efforts))
        self.debug_pub.publish(
            Float64MultiArray(
                data=[self.pos[leg] for leg in LEGS]
                + [self.cmd_target[leg] for leg in LEGS]
                + efforts
            )
        )


def main():
    rclpy.init()
    node = StandSuspensionHold()
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
