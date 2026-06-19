"""Active-suspension policy node.

The node loads a 50D/6D posture actor and publishes six hip position targets.
It never publishes wheel velocity commands and never changes ``/cmd_vel``.
"""

from __future__ import annotations

import numpy as np
import rclpy
from builtin_interfaces.msg import Duration
from geometry_msgs.msg import Twist, Wrench
from rclpy.node import Node
from sensor_msgs.msg import Imu, JointState
from std_msgs.msg import Float64MultiArray
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint

from .motion_control import (
    MotionControlConfig,
    POSTURE_ACTION_DIM,
    POSTURE_OBSERVATION_DIM,
    SkidSteerMotionController,
    build_posture_observation,
)
from .rl_policy import RLPosturePolicy
from .suspension_core import HIP_JOINTS, HIP_TARGET_LIMIT, LEGS, projected_gravity


class PosturePolicyNode(Node):
    def __init__(self):
        super().__init__("posture_policy_node")
        defaults = MotionControlConfig()
        self.declare_parameter("rate", 30.0)
        self.declare_parameter("policy_weights_npz", "")
        self.declare_parameter("cmd_vx", 0.0)
        self.declare_parameter("cmd_wz", 0.0)
        self.declare_parameter("max_abs_cmd_vx", defaults.max_abs_cmd_vx)
        self.declare_parameter("max_abs_cmd_wz", defaults.max_abs_cmd_wz)
        self.declare_parameter("force_observation_enabled", True)

        self.policy = RLPosturePolicy(str(self.get_parameter("policy_weights_npz").value))
        if self.policy.obs_dim != POSTURE_OBSERVATION_DIM or self.policy.action_dim != POSTURE_ACTION_DIM:
            raise ValueError(
                f"policy must be {POSTURE_OBSERVATION_DIM}D/{POSTURE_ACTION_DIM}D, "
                f"got {self.policy.obs_dim}D/{self.policy.action_dim}D"
            )
        if not bool(self.get_parameter("force_observation_enabled").value):
            raise ValueError("force_observation_enabled must remain true; wheel F/T is part of the posture policy input")

        control_config = MotionControlConfig(
            max_abs_cmd_vx=float(self.get_parameter("max_abs_cmd_vx").value),
            max_abs_cmd_wz=float(self.get_parameter("max_abs_cmd_wz").value),
        )
        self.motion_controller = SkidSteerMotionController(control_config)
        initial_command = self.motion_controller.limit_command(
            float(self.get_parameter("cmd_vx").value),
            float(self.get_parameter("cmd_wz").value),
        )
        self.cmd_vx, self.cmd_wz = initial_command.vx, initial_command.wz

        self.proj_grav = (0.0, 0.0, -1.0)
        self.ang_vel = (0.0, 0.0, 0.0)
        self.joint_pos = {leg: 0.0 for leg in LEGS}
        self.joint_vel = {leg: 0.0 for leg in LEGS}
        self.wheel_vel = {leg: 0.0 for leg in LEGS}
        self.wheel_force = {leg: (0.0, 0.0, 0.0) for leg in LEGS}
        self.prev_action = np.zeros(POSTURE_ACTION_DIM, dtype=np.float32)
        self.joint_seen = False

        self.hip_pub = self.create_publisher(JointTrajectory, "/suspension_controller/joint_trajectory", 10)
        self.status_pub = self.create_publisher(Float64MultiArray, "/posture_policy/status", 10)

        self.create_subscription(Imu, "/imu/data", self.imu_cb, 50)
        self.create_subscription(JointState, "/joint_states", self.joint_cb, 50)
        self.create_subscription(Twist, "/cmd_vel", self.cmd_vel_cb, 10)
        for leg in LEGS:
            self.create_subscription(Wrench, f"/ft_wheel/{leg}", lambda msg, l=leg: self.ft_wheel_cb(msg, l), 50)

        self.create_timer(1.0 / float(self.get_parameter("rate").value), self.step)
        self.get_logger().info(
            f"active-suspension policy started (obs_dim={self.policy.obs_dim}, "
            f"action_dim={self.policy.action_dim}, hip_limit={self.policy.hip_action_target_limit})."
        )

    def imu_cb(self, msg: Imu):
        q = msg.orientation
        self.proj_grav = projected_gravity(q.w, q.x, q.y, q.z)
        self.ang_vel = (msg.angular_velocity.x, msg.angular_velocity.y, msg.angular_velocity.z)

    def joint_cb(self, msg: JointState):
        for i, name in enumerate(msg.name):
            if name.startswith("susp_") and name.endswith("_joint"):
                leg = name[5:-6]
                if leg in self.joint_pos:
                    self.joint_pos[leg] = msg.position[i]
                    if i < len(msg.velocity):
                        self.joint_vel[leg] = msg.velocity[i]
            elif name.startswith("wheel_") and name.endswith("_joint"):
                leg = name[6:-6]
                if leg in self.wheel_vel and i < len(msg.velocity):
                    self.wheel_vel[leg] = msg.velocity[i]
        self.joint_seen = True

    def ft_wheel_cb(self, msg: Wrench, leg: str):
        self.wheel_force[leg] = (msg.force.x, msg.force.y, msg.force.z)

    def cmd_vel_cb(self, msg: Twist):
        command = self.motion_controller.limit_command(msg.linear.x, msg.angular.z)
        self.cmd_vx, self.cmd_wz = command.vx, command.wz

    def _publish_hip_targets(self, action: np.ndarray) -> None:
        limit = min(abs(float(self.policy.hip_action_target_limit)), HIP_TARGET_LIMIT)
        targets = np.clip(action, -1.0, 1.0) * limit
        point = JointTrajectoryPoint()
        point.positions = [float(value) for value in targets]
        point.time_from_start = Duration(sec=0, nanosec=100_000_000)
        msg = JointTrajectory()
        msg.joint_names = list(HIP_JOINTS)
        msg.points = [point]
        self.hip_pub.publish(msg)

    def step(self):
        if not self.joint_seen:
            return
        command = self.motion_controller.limit_command(self.cmd_vx, self.cmd_wz)
        obs = build_posture_observation(
            projected_gravity_b=self.proj_grav,
            root_ang_vel_b=self.ang_vel,
            susp_joint_pos=self.joint_pos,
            susp_joint_vel=self.joint_vel,
            wheel_joint_vel=self.wheel_vel,
            wheel_force=self.wheel_force,
            command=command,
            prev_action=self.prev_action,
        )
        action = self.policy.act(obs)
        self.prev_action = action
        self._publish_hip_targets(action)
        self.status_pub.publish(Float64MultiArray(data=[
            float(np.max(np.abs(action))),
            float(command.vx),
            float(command.wz),
            float(self.ang_vel[0]),
            float(self.ang_vel[1]),
            float(self.ang_vel[2]),
        ] + [float(value) for value in action]))


def main():
    rclpy.init()
    node = PosturePolicyNode()
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
