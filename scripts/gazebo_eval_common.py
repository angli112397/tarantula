"""Shared sensor subscriptions and metric helpers for Gazebo posture eval scripts.

Factored out of gazebo_posture_eval.py so gazebo_pursuit_eval.py doesn't
duplicate the IMU/joint_states/F-T subscription and CSV/summary plumbing.
"""

from __future__ import annotations

import math
import csv
from pathlib import Path

from geometry_msgs.msg import Twist, Wrench
from nav_msgs.msg import Odometry
from rclpy.node import Node
from sensor_msgs.msg import Imu, JointState

from tarantula_control.suspension_core import HIP_JOINTS, LEGS, quat_roll_pitch, quat_yaw


def force_norm(force: tuple[float, float, float]) -> float:
    return math.sqrt(sum(component * component for component in force))


def mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def rms(values: list[float]) -> float:
    return math.sqrt(mean([value * value for value in values]))


class PostureEvalNode(Node):
    """IMU + joint_states + per-wheel F/T subscriber, optionally ground-truth odom.

    Ground-truth odom (bridge_ground_truth_odom:=true on sim.launch.py) is
    opt-in via track_position=True -- it's an eval-only privileged signal
    (see tarantula_v3.urdf.xacro's OdometryPublisher plugin docstring), never
    used to drive control, only to measure where the robot actually went.
    """

    def __init__(self, node_name: str = "gazebo_posture_eval", track_position: bool = False):
        super().__init__(node_name)
        self.roll = 0.0
        self.pitch = 0.0
        self.yaw = 0.0
        self.ang_x = 0.0
        self.ang_y = 0.0
        self.hip_pos = {leg: 0.0 for leg in LEGS}
        self.wheel_force = {leg: (0.0, 0.0, 0.0) for leg in LEGS}
        self.pos_x = 0.0
        self.pos_y = 0.0
        self.seen_imu = False
        self.seen_joint = False
        self.seen_odom = not track_position
        self.cmd_pub = self.create_publisher(Twist, "/cmd_vel", 10)
        self.create_subscription(Imu, "/imu/data", self._imu_cb, 50)
        self.create_subscription(JointState, "/joint_states", self._joint_cb, 50)
        for leg in LEGS:
            self.create_subscription(Wrench, f"/ft_wheel/{leg}", lambda msg, l=leg: self._ft_cb(msg, l), 50)
        if track_position:
            self.create_subscription(Odometry, "/ground_truth_odom", self._odom_cb, 50)

    def _imu_cb(self, msg: Imu) -> None:
        q = msg.orientation
        self.roll, self.pitch = quat_roll_pitch(q.w, q.x, q.y, q.z)
        self.ang_x = msg.angular_velocity.x
        self.ang_y = msg.angular_velocity.y
        self.seen_imu = True

    def _joint_cb(self, msg: JointState) -> None:
        for i, name in enumerate(msg.name):
            if name in HIP_JOINTS:
                leg = name[5:-6]
                self.hip_pos[leg] = msg.position[i]
        self.seen_joint = True

    def _ft_cb(self, msg: Wrench, leg: str) -> None:
        self.wheel_force[leg] = (msg.force.x, msg.force.y, msg.force.z)

    def _odom_cb(self, msg: Odometry) -> None:
        self.pos_x = msg.pose.pose.position.x
        self.pos_y = msg.pose.pose.position.y
        q = msg.pose.pose.orientation
        self.yaw = quat_yaw(q.w, q.x, q.y, q.z)
        self.seen_odom = True

    def publish_cmd(self, vx: float, wz: float) -> None:
        msg = Twist()
        msg.linear.x = float(vx)
        msg.angular.z = float(wz)
        self.cmd_pub.publish(msg)

    def ready(self) -> bool:
        return self.seen_imu and self.seen_joint and self.seen_odom


def posture_sample_fields(node: PostureEvalNode) -> dict:
    """Posture/load fields shared by every sample row regardless of eval script."""
    loads = [force_norm(node.wheel_force[leg]) for leg in LEGS]
    mean_load = mean(loads)
    load_var = mean([(load - mean_load) ** 2 for load in loads])
    loaded_count = sum(1 for load in loads if load > 5.0)
    return {
        "roll": node.roll,
        "pitch": node.pitch,
        "roll_pitch_rate": math.sqrt(node.ang_x * node.ang_x + node.ang_y * node.ang_y),
        "loaded_wheels": loaded_count,
        "wheel_load_var": load_var,
        **{f"hip_{leg}": node.hip_pos[leg] for leg in LEGS},
    }


def posture_summary_fields(rows: list[dict]) -> dict:
    """Posture/load summary stats shared by every eval script."""
    roll = [float(row["roll"]) for row in rows]
    pitch = [float(row["pitch"]) for row in rows]
    rate = [float(row["roll_pitch_rate"]) for row in rows]
    load_var = [float(row["wheel_load_var"]) for row in rows]
    loaded = [float(row["loaded_wheels"]) for row in rows]
    hip_abs = [abs(float(row[f"hip_{leg}"])) for row in rows for leg in LEGS]
    return {
        "samples": len(rows),
        "roll_rms_rad": rms(roll),
        "pitch_rms_rad": rms(pitch),
        "max_abs_roll_rad": max((abs(v) for v in roll), default=0.0),
        "max_abs_pitch_rad": max((abs(v) for v in pitch), default=0.0),
        "tilt_over_0p20_ratio": mean([1.0 if abs(r) > 0.20 or abs(p) > 0.20 else 0.0 for r, p in zip(roll, pitch)]),
        "roll_pitch_rate_mean": mean(rate),
        "wheel_load_var_mean": mean(load_var),
        "loaded_wheels_mean": mean(loaded),
        "hip_abs_mean_rad": mean(hip_abs),
        "hip_abs_max_rad": max(hip_abs, default=0.0),
    }


def write_csv(path: Path, rows: list[dict], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def write_attitude_plot(path: Path, rows: list[dict], x_key: str = "t") -> None:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception:
        return
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    xs = [row[x_key] for row in rows]
    fig, ax = plt.subplots(figsize=(9, 4))
    ax.plot(xs, [row["roll"] for row in rows], label="roll")
    ax.plot(xs, [row["pitch"] for row in rows], label="pitch")
    ax.set_xlabel(x_key)
    ax.set_ylabel("rad")
    ax.grid(True, linewidth=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
