"""Classical motion-control node with optional RL wheel compensation.

Stage B motion-control baseline:
  - 传统 skid-steer controller 将 /cmd_vel 映射为 6 个驱动轮速度目标
  - RL 可选输出 action[0:3] 作为 structured skid-steer compensation:
    track_scale_delta, left_drive_scale_delta, right_drive_scale_delta
  - 不发布 hip/suspension 命令；v2 baseline 的 hip 姿态由
    JointTrajectoryController 或外部 posture profile 负责

观测构成：
  projected_gravity_b(3)  <- /imu/data
  root_ang_vel_b(3)        <- /imu/data
  susp_joint_pos(6)        <- /joint_states (LEGS 顺序)
  susp_joint_vel(6)        <- /joint_states
  wheel_joint_vel(6)       <- /joint_states
  wheel_force(18)          <- /ft_wheel/{leg}，Fx/Fy/Fz / nominal_wheel_load
  cmd_vx(1)                <- /cmd_vel.linear.x, fallback ROS 参数
  cmd_wz(1)                <- /cmd_vel.angular.z, fallback ROS 参数
  prev_action(9)           <- 上一步 structured compensation + hip 策略输出

控制常量：
  compensation action scale 和 wheel clamp 优先从 actor .npz metadata 读取；
  未启用 RL 时使用 control_interfaces.py 中的传统主控默认值。
"""
import numpy as np
import rclpy
from rclpy.node import Node
from rcl_interfaces.msg import SetParametersResult
from geometry_msgs.msg import Twist, Wrench
from sensor_msgs.msg import Imu, JointState
from std_msgs.msg import Float64MultiArray
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint
from builtin_interfaces.msg import Duration
import time

from .control_interfaces import (
    MAX_ABS_WHEEL_OMEGA,
)
from .motion_control import (
    CommandShaper,
    MotionControlConfig,
    SkidSteerMotionController,
    STAGE_A_ACTION_DIM,
    STAGE_A_OBSERVATION_DIM,
    STAGE_B_ACTION_DIM,
    build_stage_a_observation,
)
from .rl_policy import RLWheelCompensationPolicy
from .suspension_core import HIP_JOINTS, HIP_TARGET_LIMIT, LEGS, projected_gravity


class MotionControlNode(Node):
    def __init__(self):
        super().__init__('motion_control_node')
        self.declare_parameter('rate', 30.0)
        self.declare_parameter('cmd_vx', 0.1)
        self.declare_parameter('cmd_wz', 0.0)
        self.declare_parameter('max_abs_cmd_vx', 0.3)
        self.declare_parameter('max_abs_cmd_wz', 0.4)
        self.declare_parameter('pure_turn_track_scale', 3.0)
        self.declare_parameter('track_scale_delta_limit', 0.3)
        self.declare_parameter('drive_scale_delta_limit', 0.2)
        self.declare_parameter('yaw_rate_kp', 0.0)
        self.declare_parameter('yaw_rate_ki', 0.0)
        self.declare_parameter('yaw_integral_limit', 0.8)
        self.declare_parameter('max_wheel_accel', 12.0)
        self.declare_parameter('pure_turn_forward_bias', 0.0)
        self.declare_parameter('pure_turn_vx_deadband', 0.03)
        self.declare_parameter('turn_enter_wz', 0.08)
        self.declare_parameter('turn_exit_wz', 0.04)
        self.declare_parameter('policy_weights_npz', '')
        self.declare_parameter('rl_compensation_enabled', True)
        self.declare_parameter('force_observation_enabled', False)

        weights_npz = self.get_parameter('policy_weights_npz').value
        self.rl_compensation_enabled = bool(self.get_parameter('rl_compensation_enabled').value)
        self.force_observation_enabled = bool(self.get_parameter('force_observation_enabled').value)
        self.policy = RLWheelCompensationPolicy(weights_npz_path=weights_npz) if self.rl_compensation_enabled else None
        max_abs_wheel_omega = (
            self.policy.max_abs_wheel_omega if self.policy is not None else MAX_ABS_WHEEL_OMEGA
        )
        track_scale_delta_limit = (
            self.policy.track_scale_delta_limit
            if self.policy is not None
            else float(self.get_parameter('track_scale_delta_limit').value)
        )
        drive_scale_delta_limit = (
            self.policy.drive_scale_delta_limit
            if self.policy is not None
            else float(self.get_parameter('drive_scale_delta_limit').value)
        )
        control_config = MotionControlConfig(
            max_abs_cmd_vx=float(self.get_parameter('max_abs_cmd_vx').value),
            max_abs_cmd_wz=float(self.get_parameter('max_abs_cmd_wz').value),
            max_abs_wheel_omega=max_abs_wheel_omega,
            pure_turn_track_scale=float(self.get_parameter('pure_turn_track_scale').value),
            track_scale_delta_limit=track_scale_delta_limit,
            drive_scale_delta_limit=drive_scale_delta_limit,
            yaw_rate_kp=float(self.get_parameter('yaw_rate_kp').value),
            yaw_rate_ki=float(self.get_parameter('yaw_rate_ki').value),
            yaw_integral_limit=float(self.get_parameter('yaw_integral_limit').value),
            max_wheel_accel=float(self.get_parameter('max_wheel_accel').value),
            pure_turn_forward_bias=float(self.get_parameter('pure_turn_forward_bias').value),
            pure_turn_vx_deadband=float(self.get_parameter('pure_turn_vx_deadband').value),
            turn_enter_wz=float(self.get_parameter('turn_enter_wz').value),
            turn_exit_wz=float(self.get_parameter('turn_exit_wz').value),
        )
        self.command_shaper = CommandShaper(control_config)
        self.motion_controller = SkidSteerMotionController(control_config)
        initial_command = self.motion_controller.limit_command(
            float(self.get_parameter('cmd_vx').value),
            float(self.get_parameter('cmd_wz').value),
        )
        self.cmd_vx, self.cmd_wz = initial_command.vx, initial_command.wz

        self.proj_grav = (0.0, 0.0, -1.0)
        self.ang_vel = (0.0, 0.0, 0.0)
        self.joint_pos = {leg: 0.0 for leg in LEGS}
        self.joint_vel = {leg: 0.0 for leg in LEGS}
        self.wheel_vel = {leg: 0.0 for leg in LEGS}
        self.wheel_force = {leg: (0.0, 0.0, 0.0) for leg in LEGS}  # Fx/Fy/Fz in N
        action_dim = self.policy.action_dim if self.policy is not None else STAGE_A_ACTION_DIM
        self.prev_action = np.zeros(action_dim, dtype=np.float32)
        self.joint_seen = False
        self.last_step_time = time.monotonic()

        self.wheel_pub = self.create_publisher(
            Float64MultiArray, '/wheel_velocity_controller/commands', 10)
        self.hip_pub = self.create_publisher(
            JointTrajectory, '/suspension_controller/joint_trajectory', 10)
        self.status_pub = self.create_publisher(
            Float64MultiArray, '/rl_policy/status', 10)

        self.create_subscription(Imu, '/imu/data', self.imu_cb, 50)
        self.create_subscription(JointState, '/joint_states', self.joint_cb, 50)
        self.create_subscription(Twist, '/cmd_vel', self.cmd_vel_cb, 10)
        if self.force_observation_enabled:
            for leg in LEGS:
                self.create_subscription(
                    Wrench, f'/ft_wheel/{leg}',
                    lambda msg, l=leg: self.ft_wheel_cb(msg, l),
                    50,
                )

        rate = self.get_parameter('rate').value
        self.create_timer(1.0 / rate, self.step)
        self.add_on_set_parameters_callback(self._on_parameter_update)
        obs_dim = self.policy.obs_dim if self.policy is not None else STAGE_A_OBSERVATION_DIM
        action_dim = self.policy.action_dim if self.policy is not None else 0
        self.get_logger().info(
            f'motion controller started (rl_compensation_enabled={self.rl_compensation_enabled}, '
            f'force_observation_enabled={self.force_observation_enabled}, '
            f'action_dim={action_dim}, obs_dim={obs_dim}, '
            f'velocity_source=none_in_actor, '
            f'max_abs_wheel_omega={self.motion_controller.config.max_abs_wheel_omega}, '
            f'pure_turn_track_scale={self.motion_controller.config.pure_turn_track_scale}, '
            f'track_scale_delta_limit={self.motion_controller.config.track_scale_delta_limit}, '
            f'drive_scale_delta_limit={self.motion_controller.config.drive_scale_delta_limit}, '
            f'yaw_rate_kp={self.motion_controller.config.yaw_rate_kp}, '
            f'yaw_rate_ki={self.motion_controller.config.yaw_rate_ki}, '
            f'max_wheel_accel={self.motion_controller.config.max_wheel_accel}, '
            f'pure_turn_forward_bias={self.motion_controller.config.pure_turn_forward_bias}, '
            f'turn_enter_wz={self.motion_controller.config.turn_enter_wz}, '
            f'turn_exit_wz={self.motion_controller.config.turn_exit_wz}, '
            f'cmd_vx={self.cmd_vx} m/s, cmd_wz={self.cmd_wz} rad/s, '
            f'policy_weights_npz={weights_npz or "<none>"}).')

    def _on_parameter_update(self, params):
        updates = {}
        for param in params:
            if param.name in {
                'max_abs_cmd_vx',
                'max_abs_cmd_wz',
                'max_abs_wheel_omega',
                'pure_turn_track_scale',
                'track_scale_delta_limit',
                'drive_scale_delta_limit',
                'yaw_rate_kp',
                'yaw_rate_ki',
                'yaw_integral_limit',
                'max_wheel_accel',
                'pure_turn_forward_bias',
                'pure_turn_vx_deadband',
                'turn_enter_wz',
                'turn_exit_wz',
            }:
                updates[param.name] = float(param.value)
        if updates:
            self.command_shaper.update_config(**updates)
            self.motion_controller.update_config(**updates)
            self.motion_controller.reset_feedback()
            self.get_logger().info(
                'updated motion control parameters: '
                + ', '.join(f'{key}={value}' for key, value in sorted(updates.items()))
            )
        return SetParametersResult(successful=True)

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
        self.wheel_force[leg] = (msg.force.x, msg.force.y, msg.force.z)

    def cmd_vel_cb(self, msg: Twist):
        command = self.motion_controller.limit_command(msg.linear.x, msg.angular.z)
        self.cmd_vx, self.cmd_wz = command.vx, command.wz

    def publish_hip_targets(self, action: np.ndarray) -> None:
        if action.shape[0] < STAGE_B_ACTION_DIM:
            return
        limit = self.policy.hip_action_target_limit if self.policy is not None else HIP_TARGET_LIMIT
        targets = np.clip(action[3:9], -1.0, 1.0) * min(abs(float(limit)), HIP_TARGET_LIMIT)
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

        ang_vel_b = self.ang_vel
        now = time.monotonic()
        dt = max(now - self.last_step_time, 0.0)
        self.last_step_time = now

        command = self.motion_controller.limit_command(self.cmd_vx, self.cmd_wz)
        execution_command = self.command_shaper.shape(command)
        action = np.zeros_like(self.prev_action)
        if self.policy is not None:
            obs = build_stage_a_observation(
                projected_gravity_b=self.proj_grav,
                root_ang_vel_b=ang_vel_b,
                susp_joint_pos=self.joint_pos,
                susp_joint_vel=self.joint_vel,
                wheel_joint_vel=self.wheel_vel,
                wheel_force=self.wheel_force,
                command=execution_command,
                prev_action=self.prev_action,
            )
            action = self.policy.act(obs)
            self.prev_action = action

        compensation = action if self.policy is not None else None
        wheel_cmds = self.motion_controller.compensated_wheel_targets(
            execution_command,
            compensation,
            measured_wz=ang_vel_b[2],
            dt=dt,
        )
        self.wheel_pub.publish(Float64MultiArray(data=wheel_cmds))
        self.publish_hip_targets(action)
        action_saturation = float(np.max(np.abs(action))) if action.size else 0.0
        self.status_pub.publish(Float64MultiArray(data=[
            1.0 if self.policy is not None else 0.0,
            float(action[0]) if action.size > 0 else 0.0,
            float(action[1]) if action.size > 1 else 0.0,
            float(action[2]) if action.size > 2 else 0.0,
            action_saturation,
            max((abs(float(v)) for v in wheel_cmds), default=0.0),
            float(execution_command.vx),
            float(execution_command.wz),
            float(ang_vel_b[2]),
            1.0 if self.command_shaper.mode.value == 'turn' else 0.0,
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


if __name__ == '__main__':
    main()
