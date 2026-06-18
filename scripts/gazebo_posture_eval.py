#!/usr/bin/env python3
"""Gazebo posture evaluator for active-suspension acceptance.

The script does not score trajectory tracking. It publishes a simple low-speed
command profile and records posture/support metrics so neutral posture and RL
active suspension can be compared on the same world.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import time
from pathlib import Path

import rclpy
from geometry_msgs.msg import Twist, Wrench
from rclpy.node import Node
from sensor_msgs.msg import Imu, JointState

LEGS = ("fl", "fr", "ml", "mr", "rl", "rr")
HIP_JOINTS = tuple(f"susp_{leg}_joint" for leg in LEGS)


def clamp(value: float, lower: float, upper: float) -> float:
    return max(lower, min(upper, value))


def quat_roll_pitch(w: float, x: float, y: float, z: float) -> tuple[float, float]:
    roll = math.atan2(2.0 * (w * x + y * z), 1.0 - 2.0 * (x * x + y * y))
    pitch = math.asin(clamp(2.0 * (w * y - z * x), -1.0, 1.0))
    return roll, pitch


class PostureEvalNode(Node):
    def __init__(self):
        super().__init__("gazebo_posture_eval")
        self.roll = 0.0
        self.pitch = 0.0
        self.ang_x = 0.0
        self.ang_y = 0.0
        self.hip_pos = {leg: 0.0 for leg in LEGS}
        self.wheel_force = {leg: (0.0, 0.0, 0.0) for leg in LEGS}
        self.seen_imu = False
        self.seen_joint = False
        self.cmd_pub = self.create_publisher(Twist, "/cmd_vel", 10)
        self.create_subscription(Imu, "/imu/data", self.imu_cb, 50)
        self.create_subscription(JointState, "/joint_states", self.joint_cb, 50)
        for leg in LEGS:
            self.create_subscription(Wrench, f"/ft_wheel/{leg}", lambda msg, l=leg: self.ft_cb(msg, l), 50)

    def imu_cb(self, msg: Imu) -> None:
        q = msg.orientation
        self.roll, self.pitch = quat_roll_pitch(q.w, q.x, q.y, q.z)
        self.ang_x = msg.angular_velocity.x
        self.ang_y = msg.angular_velocity.y
        self.seen_imu = True

    def joint_cb(self, msg: JointState) -> None:
        for i, name in enumerate(msg.name):
            if name in HIP_JOINTS:
                leg = name[5:-6]
                self.hip_pos[leg] = msg.position[i]
        self.seen_joint = True

    def ft_cb(self, msg: Wrench, leg: str) -> None:
        self.wheel_force[leg] = (msg.force.x, msg.force.y, msg.force.z)

    def publish_cmd(self, vx: float, wz: float) -> None:
        msg = Twist()
        msg.linear.x = float(vx)
        msg.angular.z = float(wz)
        self.cmd_pub.publish(msg)


def force_norm(force: tuple[float, float, float]) -> float:
    return math.sqrt(sum(component * component for component in force))


def collect_sample(node: PostureEvalNode, t: float, cmd_vx: float, cmd_wz: float) -> dict:
    loads = [force_norm(node.wheel_force[leg]) for leg in LEGS]
    mean_load = sum(loads) / len(loads)
    load_var = sum((load - mean_load) ** 2 for load in loads) / len(loads)
    loaded_count = sum(1 for load in loads if load > 5.0)
    return {
        "t": t,
        "cmd_vx": cmd_vx,
        "cmd_wz": cmd_wz,
        "roll": node.roll,
        "pitch": node.pitch,
        "roll_pitch_rate": math.sqrt(node.ang_x * node.ang_x + node.ang_y * node.ang_y),
        "loaded_wheels": loaded_count,
        "wheel_load_var": load_var,
        **{f"hip_{leg}": node.hip_pos[leg] for leg in LEGS},
    }


def mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def rms(values: list[float]) -> float:
    return math.sqrt(mean([value * value for value in values]))


def summarize(rows: list[dict], label: str) -> dict:
    roll = [float(row["roll"]) for row in rows]
    pitch = [float(row["pitch"]) for row in rows]
    rate = [float(row["roll_pitch_rate"]) for row in rows]
    load_var = [float(row["wheel_load_var"]) for row in rows]
    loaded = [float(row["loaded_wheels"]) for row in rows]
    hip_abs = [abs(float(row[f"hip_{leg}"])) for row in rows for leg in LEGS]
    return {
        "label": label,
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


def write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "t", "cmd_vx", "cmd_wz", "roll", "pitch", "roll_pitch_rate",
        "loaded_wheels", "wheel_load_var",
    ] + [f"hip_{leg}" for leg in LEGS]
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def write_plot(path: Path, rows: list[dict]) -> None:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception:
        return
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    ts = [row["t"] for row in rows]
    fig, ax = plt.subplots(figsize=(9, 4))
    ax.plot(ts, [row["roll"] for row in rows], label="roll")
    ax.plot(ts, [row["pitch"] for row in rows], label="pitch")
    ax.set_xlabel("time [s]")
    ax.set_ylabel("rad")
    ax.grid(True, linewidth=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def wait_for_inputs(node: PostureEvalNode, timeout_s: float) -> None:
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        rclpy.spin_once(node, timeout_sec=0.1)
        if node.seen_imu and node.seen_joint:
            return
    raise RuntimeError("timed out waiting for /imu/data and /joint_states")


def main() -> int:
    parser = argparse.ArgumentParser(description="Evaluate Gazebo posture stability under low-speed commands.")
    parser.add_argument("--label", default="posture_eval")
    parser.add_argument("--duration", type=float, default=30.0)
    parser.add_argument("--settle", type=float, default=2.0)
    parser.add_argument("--rate", type=float, default=20.0)
    parser.add_argument("--cmd-vx", type=float, default=0.10)
    parser.add_argument("--cmd-wz", type=float, default=0.0)
    parser.add_argument("--out-dir", default="generated/benchmarks/posture_eval/latest")
    parser.add_argument("--input-timeout", type=float, default=10.0)
    args = parser.parse_args()

    rclpy.init()
    node = PostureEvalNode()
    rows: list[dict] = []
    try:
        wait_for_inputs(node, args.input_timeout)
        period = 1.0 / args.rate
        start = time.monotonic()
        end = start + args.settle + args.duration
        while time.monotonic() < end:
            rclpy.spin_once(node, timeout_sec=0.0)
            elapsed = time.monotonic() - start
            active = elapsed >= args.settle
            vx = args.cmd_vx if active else 0.0
            wz = args.cmd_wz if active else 0.0
            node.publish_cmd(vx, wz)
            if active:
                rows.append(collect_sample(node, elapsed - args.settle, vx, wz))
            time.sleep(period)
        node.publish_cmd(0.0, 0.0)

        out_dir = Path(args.out_dir)
        summary = summarize(rows, args.label)
        write_csv(out_dir / "samples.csv", rows)
        (out_dir / "summary.json").parent.mkdir(parents=True, exist_ok=True)
        (out_dir / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
        write_plot(out_dir / "attitude.png", rows)
        print(json.dumps(summary, indent=2, sort_keys=True))
    finally:
        node.publish_cmd(0.0, 0.0)
        node.destroy_node()
        rclpy.shutdown()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
