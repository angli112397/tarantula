#!/usr/bin/env python3
"""Gazebo command-tracking benchmark for Tarantula.

Prerequisite: start Gazebo with the RL policy path, for example:

  ros2 launch tarantula_bringup sim.launch.py \\
    gui:=false robot_model:=tarantula_v2.urdf.xacro \\
    motion_control:=true start_motion_control:=true \\
    rl_compensation_enabled:=true truth_odom:=false \\
    wheel_collision:=cylinder policy_weights_npz:=/path/to/cmd_vel_actor.npz \\
    spawn_z:=0.55

The benchmark publishes a fixed /cmd_vel sequence, samples Gazebo truth pose via
`ign model -m tarantula -p`, and records command tracking metrics.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import subprocess
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import rclpy
from geometry_msgs.msg import Twist
from rclpy.node import Node
from sensor_msgs.msg import JointState
from std_msgs.msg import Float64MultiArray


DEFAULT_SEQUENCE = [
    ("stop", 0.0, 0.0),
    ("forward", 0.1, 0.0),
    ("backward", -0.1, 0.0),
    ("turn_left", 0.0, 0.15),
    ("turn_right", 0.0, -0.15),
    ("turn_left_authority", 0.0, 0.25),
    ("turn_right_authority", 0.0, -0.25),
    ("arc_left", 0.1, 0.12),
    ("arc_right", 0.1, -0.12),
    ("final_stop", 0.0, 0.0),
]


@dataclass
class TruthPose:
    x: float
    y: float
    z: float
    roll: float
    pitch: float
    yaw: float


class BenchmarkNode(Node):
    def __init__(self):
        super().__init__("gazebo_cmd_tracking_benchmark")
        self.cmd_pub = self.create_publisher(Twist, "/cmd_vel", 10)
        self.last_wheel_cmd: list[float] = []
        self.last_joint_state: JointState | None = None
        self.create_subscription(
            Float64MultiArray,
            "/wheel_velocity_controller/commands",
            self._wheel_cmd_cb,
            10,
        )
        self.create_subscription(JointState, "/joint_states", self._joint_cb, 10)

    def _wheel_cmd_cb(self, msg: Float64MultiArray) -> None:
        self.last_wheel_cmd = [float(v) for v in msg.data]

    def _joint_cb(self, msg: JointState) -> None:
        self.last_joint_state = msg

    def publish_cmd(self, vx: float, wz: float) -> None:
        msg = Twist()
        msg.linear.x = float(vx)
        msg.angular.z = float(wz)
        self.cmd_pub.publish(msg)


def wrap_angle(angle: float) -> float:
    return math.atan2(math.sin(angle), math.cos(angle))


def truth_pose(model_name: str, *, attempts: int = 3) -> TruthPose:
    last_error: Exception | None = None
    lines: list[str] = []
    for attempt in range(attempts):
        try:
            result = subprocess.run(
                ["ign", "model", "-m", model_name, "-p"],
                check=False,
                capture_output=True,
                text=True,
                timeout=6.0,
            )
            lines = result.stdout.strip().splitlines()
            if result.returncode == 0 and len(lines) >= 2:
                break
            last_error = RuntimeError((result.stderr or result.stdout).strip())
        except Exception as exc:
            last_error = exc
        if attempt + 1 < attempts:
            time.sleep(0.2)
    if len(lines) < 2:
        raise RuntimeError(f"unexpected ign model output: {lines!r}; last_error={last_error}")
    xyz = [float(v) for v in lines[-2].strip().strip("[]").split()]
    rpy = [float(v) for v in lines[-1].strip().strip("[]").split()]
    return TruthPose(x=xyz[0], y=xyz[1], z=xyz[2], roll=rpy[0], pitch=rpy[1], yaw=rpy[2])


def spin_until_ready(node: BenchmarkNode, timeout_s: float) -> None:
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        rclpy.spin_once(node, timeout_sec=0.1)
        try:
            truth_pose("tarantula")
        except Exception:
            continue
        if node.last_joint_state is not None:
            return
    raise RuntimeError("Gazebo truth pose or /joint_states not available; is sim.launch.py running?")


def sample_loop(
    node: BenchmarkNode,
    *,
    model_name: str,
    segment_name: str,
    cmd_vx: float,
    cmd_wz: float,
    duration_s: float,
    rate_hz: float,
) -> list[dict]:
    samples: list[dict] = []
    period = 1.0 / rate_hz
    num_samples = max(2, int(math.ceil(duration_s * rate_hz)) + 1)
    prev_t: float | None = None
    prev_pose: TruthPose | None = None

    for _ in range(num_samples):
        loop_t = time.monotonic()
        node.publish_cmd(cmd_vx, cmd_wz)
        rclpy.spin_once(node, timeout_sec=0.0)

        try:
            pose = truth_pose(model_name, attempts=2)
        except RuntimeError as exc:
            node.get_logger().warn(f"Skipping truth sample: {exc}")
            time.sleep(period)
            continue
        vx = 0.0
        wz = 0.0
        if prev_pose is not None and prev_t is not None:
            dt = max(loop_t - prev_t, 1e-6)
            dx = pose.x - prev_pose.x
            dy = pose.y - prev_pose.y
            # body-frame forward velocity from world delta and previous heading.
            vx = (math.cos(prev_pose.yaw) * dx + math.sin(prev_pose.yaw) * dy) / dt
            wz = wrap_angle(pose.yaw - prev_pose.yaw) / dt

        wheel_cmd = node.last_wheel_cmd
        hip_pos = []
        if node.last_joint_state is not None:
            positions = dict(zip(node.last_joint_state.name, node.last_joint_state.position))
            hip_pos = [
                float(positions[name])
                for name in (
                    "susp_fl_joint",
                    "susp_fr_joint",
                    "susp_ml_joint",
                    "susp_mr_joint",
                    "susp_rl_joint",
                    "susp_rr_joint",
                )
                if name in positions
            ]
        samples.append(
            {
                "t_wall": time.time(),
                "segment": segment_name,
                "cmd_vx": cmd_vx,
                "cmd_wz": cmd_wz,
                "x": pose.x,
                "y": pose.y,
                "z": pose.z,
                "roll": pose.roll,
                "pitch": pose.pitch,
                "yaw": pose.yaw,
                "actual_vx": vx,
                "actual_wz": wz,
                "wheel_cmd_max_abs": max((abs(v) for v in wheel_cmd), default=0.0),
                "hip_pos_max_abs": max((abs(v) for v in hip_pos), default=0.0),
            }
        )
        prev_t = loop_t
        prev_pose = pose
        sleep_s = period - (time.monotonic() - loop_t)
        if sleep_s > 0:
            time.sleep(sleep_s)

    return samples


def summarize_segment(samples: list[dict]) -> dict:
    if len(samples) <= 1:
        return {}
    usable = samples[1:]
    cmd_vx = float(usable[0]["cmd_vx"])
    cmd_wz = float(usable[0]["cmd_wz"])
    start = samples[0]
    end = samples[-1]
    mean_vx = sum(float(s["actual_vx"]) for s in usable) / len(usable)
    mean_wz = sum(float(s["actual_wz"]) for s in usable) / len(usable)
    rms_vx_error = math.sqrt(sum((float(s["actual_vx"]) - cmd_vx) ** 2 for s in usable) / len(usable))
    rms_wz_error = math.sqrt(sum((float(s["actual_wz"]) - cmd_wz) ** 2 for s in usable) / len(usable))
    max_roll = max(abs(float(s["roll"])) for s in usable)
    max_pitch = max(abs(float(s["pitch"])) for s in usable)
    max_hip_pos = max(float(s["hip_pos_max_abs"]) for s in usable)
    max_wheel_cmd = max(float(s["wheel_cmd_max_abs"]) for s in usable)
    displacement = math.hypot(float(end["x"]) - float(start["x"]), float(end["y"]) - float(start["y"]))
    direction_mismatch = (
        (cmd_vx > 0.05 and mean_vx < -0.02)
        or (cmd_vx < -0.05 and mean_vx > 0.02)
        or (cmd_wz > 0.05 and mean_wz < -0.02)
        or (cmd_wz < -0.05 and mean_wz > 0.02)
    )
    stuck = abs(cmd_vx) > 0.05 and displacement < 0.03
    return {
        "segment": str(usable[0]["segment"]),
        "cmd_vx": cmd_vx,
        "cmd_wz": cmd_wz,
        "mean_vx": mean_vx,
        "mean_wz": mean_wz,
        "rms_vx_error": rms_vx_error,
        "rms_wz_error": rms_wz_error,
        "displacement_m": displacement,
        "max_abs_roll_rad": max_roll,
        "max_abs_pitch_rad": max_pitch,
        "max_abs_wheel_cmd_rad_s": max_wheel_cmd,
        "max_abs_hip_pos_rad": max_hip_pos,
        "wheel_cmd_saturated": max_wheel_cmd >= 5.95,
        "hip_near_limit": max_hip_pos >= 0.40,
        "direction_mismatch": direction_mismatch,
        "stuck": stuck,
    }


def parse_sequence(specs: list[str]) -> list[tuple[str, float, float]]:
    if not specs:
        return DEFAULT_SEQUENCE
    sequence = []
    for spec in specs:
        parts = spec.split(",")
        if len(parts) != 3:
            raise ValueError(f"bad --segment {spec!r}; expected name,vx,wz")
        sequence.append((parts[0], float(parts[1]), float(parts[2])))
    return sequence


def write_outputs(out_dir: Path, samples: list[dict], summary: dict) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    sample_path = out_dir / "samples.csv"
    summary_path = out_dir / "summary.json"
    if samples:
        with sample_path.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(samples[0].keys()))
            writer.writeheader()
            writer.writerows(samples)
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
    print(f"SAMPLES={sample_path}")
    print(f"SUMMARY={summary_path}")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="tarantula")
    parser.add_argument("--duration", type=float, default=4.0, help="Seconds per command segment.")
    parser.add_argument("--settle", type=float, default=1.0, help="Seconds of zero command before sequence.")
    parser.add_argument("--rate", type=float, default=2.0, help="Truth sample/publish rate in Hz.")
    parser.add_argument(
        "--segment",
        action="append",
        default=[],
        help="Custom segment as name,vx,wz. Repeat to replace default sequence.",
    )
    parser.add_argument(
        "--out-dir",
        default="",
        help="Output dir. Defaults to generated/benchmarks/cmd_tracking/<timestamp>.",
    )
    args = parser.parse_args()

    sequence = parse_sequence(args.segment)
    out_dir = Path(args.out_dir) if args.out_dir else Path("generated/benchmarks/cmd_tracking") / datetime.now().strftime("%Y%m%d_%H%M%S")

    rclpy.init()
    node = BenchmarkNode()
    try:
        spin_until_ready(node, timeout_s=20.0)
        sample_loop(
            node,
            model_name=args.model,
            segment_name="settle",
            cmd_vx=0.0,
            cmd_wz=0.0,
            duration_s=args.settle,
            rate_hz=args.rate,
        )

        all_samples: list[dict] = []
        segment_summaries = []
        for name, vx, wz in sequence:
            print(f"SEGMENT {name}: cmd_vx={vx:.3f}, cmd_wz={wz:.3f}", flush=True)
            samples = sample_loop(
                node,
                model_name=args.model,
                segment_name=name,
                cmd_vx=vx,
                cmd_wz=wz,
                duration_s=args.duration,
                rate_hz=args.rate,
            )
            all_samples.extend(samples)
            segment_summaries.append(summarize_segment(samples))

        node.publish_cmd(0.0, 0.0)
        summary = {
            "model": args.model,
            "duration_s": args.duration,
            "rate_hz": args.rate,
            "segments": segment_summaries,
            "failed_segments": [
                s["segment"]
                for s in segment_summaries
                if s.get("direction_mismatch") or s.get("stuck")
            ],
        }
        write_outputs(out_dir, all_samples, summary)
        print(json.dumps(summary, indent=2, sort_keys=True))
    finally:
        node.destroy_node()
        rclpy.shutdown()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
