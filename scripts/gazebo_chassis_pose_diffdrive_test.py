#!/usr/bin/env python3
"""Gazebo chassis posture and differential-drive acceptance test.

Prerequisite: start Gazebo with the v2 chassis and without motion_control owning
wheel commands, for example:

  ros2 launch tarantula_bringup sim.launch.py \
    gui:=true robot_model:=tarantula_v2.urdf.xacro \
    motion_control:=true start_motion_control:=false rl_compensation_enabled:=false \
    wheel_collision:=sphere spawn_z:=0.55

The test drives the current v2 control surface directly:
  - /suspension_controller/joint_trajectory for hip/arm angle targets
  - /wheel_velocity_controller/commands for six wheel velocities

It records Gazebo truth pose only as an observation signal. Truth pose is not
used by the controller path.
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
from rclpy.node import Node
from sensor_msgs.msg import JointState
from std_msgs.msg import Float64MultiArray
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint

from tarantula_control.suspension_core import (
    HIP_JOINTS,
    LEGS,
    WHEEL_JOINTS,
    posture_profile,
    validate_hip_targets,
)

# The default suite keeps velocities deliberately low. This is an acceptance
# test for command sign, coupling, posture stability, and contact quality.
DEFAULT_SUITE = (
    {
        "name": "settle_hold_current",
        "hip": "initial",
        "wheel": [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        "duration": 2.0,
    },
    {
        "name": "level_forward_slow",
        "hip": "current",
        "wheel": [1.2, 1.2, 1.2, 1.2, 1.2, 1.2],
        "duration": 4.0,
    },
    {
        "name": "level_backward_slow",
        "hip": "current",
        "wheel": [-1.2, -1.2, -1.2, -1.2, -1.2, -1.2],
        "duration": 3.0,
    },
    {
        "name": "level_turn_left",
        "hip": "current",
        "wheel": [-2.0, 2.0, -2.0, 2.0, -2.0, 2.0],
        "duration": 6.0,
    },
    {
        "name": "level_turn_right",
        "hip": "current",
        "wheel": [2.0, -2.0, 2.0, -2.0, 2.0, -2.0],
        "duration": 6.0,
    },
    {
        "name": "level_left_wheel_bias",
        "hip": "current",
        "wheel": [0.8, 1.4, 0.8, 1.4, 0.8, 1.4],
        "duration": 4.0,
    },
    {
        "name": "front_down_forward",
        "hip": "front_down",
        "wheel": [1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
        "duration": 4.0,
    },
    {
        "name": "rear_down_forward",
        "hip": "rear_down",
        "wheel": [1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
        "duration": 4.0,
    },
    {
        "name": "left_right_symmetric_turn_left",
        "hip": [0.06, 0.06, 0.0, 0.0, -0.06, -0.06],
        "wheel": [-1.4, 1.4, -1.4, 1.4, -1.4, 1.4],
        "duration": 4.0,
    },
    {
        "name": "final_stop_level",
        "hip": "initial",
        "wheel": [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        "duration": 2.0,
    },
)

TURN_ONLY_SUITE = (
    {
        "name": "settle_hold_initial",
        "hip": "initial",
        "wheel": [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        "duration": 2.0,
    },
    {
        "name": "turn_left_verified",
        "hip": "initial",
        "wheel": [-2.0, 2.0, -2.0, 2.0, -2.0, 2.0],
        "duration": 6.0,
    },
    {
        "name": "stop_between_turns",
        "hip": "initial",
        "wheel": [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        "duration": 1.0,
    },
    {
        "name": "turn_right_verified",
        "hip": "initial",
        "wheel": [2.0, -2.0, 2.0, -2.0, 2.0, -2.0],
        "duration": 6.0,
    },
    {
        "name": "final_stop_initial",
        "hip": "initial",
        "wheel": [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        "duration": 2.0,
    },
)

POSTURE_ONLY_SUITE = (
    {
        "name": "natural_hold",
        "hip": "initial",
        "wheel": [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        "duration": 2.0,
    },
    {
        "name": "front_down_rear_up",
        "hip": "front_down",
        "wheel": [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        "duration": 4.0,
    },
    {
        "name": "rear_down_front_up",
        "hip": "rear_down",
        "wheel": [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        "duration": 4.0,
    },
    {
        "name": "all_raise",
        "hip": "raise",
        "wheel": [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        "duration": 4.0,
    },
    {
        "name": "all_lower",
        "hip": "lower",
        "wheel": [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        "duration": 4.0,
    },
    {
        "name": "left_right_trim",
        "hip": "left_trim",
        "wheel": [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        "duration": 4.0,
    },
    {
        "name": "natural_return",
        "hip": "initial",
        "wheel": [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        "duration": 2.0,
    },
)

PROFILES = {
    "full": DEFAULT_SUITE,
    "turn-only": TURN_ONLY_SUITE,
    "posture-only": POSTURE_ONLY_SUITE,
}


@dataclass
class TruthPose:
    x: float
    y: float
    z: float
    roll: float
    pitch: float
    yaw: float


class ChassisTestNode(Node):
    def __init__(self) -> None:
        super().__init__("gazebo_chassis_pose_diffdrive_test")
        self.hip_pub = self.create_publisher(JointTrajectory, "/suspension_controller/joint_trajectory", 10)
        self.wheel_pub = self.create_publisher(Float64MultiArray, "/wheel_velocity_controller/commands", 10)
        self.last_joint_state: JointState | None = None
        self.hip_pos = {joint: 0.0 for joint in HIP_JOINTS}
        self.hip_vel = {joint: 0.0 for joint in HIP_JOINTS}
        self.wheel_vel = {joint: 0.0 for joint in WHEEL_JOINTS}
        self.create_subscription(JointState, "/joint_states", self._joint_cb, 10)

    def _joint_cb(self, msg: JointState) -> None:
        self.last_joint_state = msg
        for index, name in enumerate(msg.name):
            if name in self.hip_pos:
                if index < len(msg.position):
                    self.hip_pos[name] = float(msg.position[index])
                if index < len(msg.velocity):
                    self.hip_vel[name] = float(msg.velocity[index])
            if name in self.wheel_vel and index < len(msg.velocity):
                self.wheel_vel[name] = float(msg.velocity[index])

    def current_hip_positions(self) -> list[float]:
        return [self.hip_pos[joint] for joint in HIP_JOINTS]

    def publish_hip_target(self, positions: list[float], duration_s: float) -> None:
        point = JointTrajectoryPoint()
        point.positions = [float(v) for v in positions]
        point.velocities = [0.0] * len(HIP_JOINTS)
        whole = max(0, int(duration_s))
        point.time_from_start.sec = whole
        point.time_from_start.nanosec = int((duration_s - whole) * 1_000_000_000)

        msg = JointTrajectory()
        msg.joint_names = list(HIP_JOINTS)
        msg.points = [point]
        self.hip_pub.publish(msg)

    def publish_wheels(self, values: list[float]) -> None:
        self.wheel_pub.publish(Float64MultiArray(data=[float(v) for v in values]))


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


def spin_until_ready(node: ChassisTestNode, timeout_s: float, model_name: str) -> None:
    deadline = time.monotonic() + timeout_s
    print(
        f"Waiting for Gazebo model '{model_name}' and /joint_states for up to {timeout_s:.1f}s...",
        flush=True,
    )
    while time.monotonic() < deadline:
        rclpy.spin_once(node, timeout_sec=0.1)
        try:
            truth_pose(model_name)
        except Exception:
            continue
        if node.last_joint_state is not None:
            return
    raise RuntimeError("Gazebo truth pose or /joint_states not available; is sim.launch.py running?")


def parse_vector(value: object, current: list[float], initial: list[float]) -> list[float]:
    if value == "current":
        return list(current)
    if value == "initial":
        return list(initial)
    if isinstance(value, str):
        return list(posture_profile(value))
    if not isinstance(value, list) or len(value) != 6:
        raise ValueError(f"expected a 6-value vector, posture profile, 'current', or 'initial', got {value!r}")
    return list(validate_hip_targets(value))


def parse_suite(path: str, profile: str) -> list[dict]:
    if not path:
        return [dict(item) for item in PROFILES[profile]]
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(data, list):
        raise ValueError("suite JSON must be a list of segment objects")
    for item in data:
        if not isinstance(item, dict):
            raise ValueError("suite JSON entries must be objects")
        if "name" not in item or "hip" not in item or "wheel" not in item:
            raise ValueError("each suite segment needs name, hip, and wheel")
    return data


def sample_segment(
    node: ChassisTestNode,
    *,
    model_name: str,
    name: str,
    hip_target: list[float],
    wheel_cmd: list[float],
    duration_s: float,
    rate_hz: float,
    hip_transition_s: float,
) -> list[dict]:
    node.publish_hip_target(hip_target, hip_transition_s)
    deadline = time.monotonic() + max(0.0, hip_transition_s)
    while time.monotonic() < deadline:
        node.publish_wheels([0.0] * 6)
        rclpy.spin_once(node, timeout_sec=0.02)
        time.sleep(0.02)

    samples: list[dict] = []
    period = 1.0 / rate_hz
    num_samples = max(2, int(math.ceil(duration_s * rate_hz)) + 1)
    prev_t: float | None = None
    prev_pose: TruthPose | None = None

    for _ in range(num_samples):
        loop_t = time.monotonic()
        node.publish_wheels(wheel_cmd)
        rclpy.spin_once(node, timeout_sec=0.0)
        try:
            pose = truth_pose(model_name, attempts=2)
        except RuntimeError as exc:
            node.get_logger().warn(f"Skipping truth sample: {exc}")
            time.sleep(period)
            continue

        vx = 0.0
        vy = 0.0
        wz = 0.0
        if prev_pose is not None and prev_t is not None:
            dt = max(loop_t - prev_t, 1e-6)
            dx = pose.x - prev_pose.x
            dy = pose.y - prev_pose.y
            vx = (math.cos(prev_pose.yaw) * dx + math.sin(prev_pose.yaw) * dy) / dt
            vy = (-math.sin(prev_pose.yaw) * dx + math.cos(prev_pose.yaw) * dy) / dt
            wz = wrap_angle(pose.yaw - prev_pose.yaw) / dt

        hip_pos = node.current_hip_positions()
        hip_error = [hip_pos[i] - hip_target[i] for i in range(6)]
        row = {
            "t_wall": time.time(),
            "segment": name,
            "x": pose.x,
            "y": pose.y,
            "z": pose.z,
            "roll": pose.roll,
            "pitch": pose.pitch,
            "yaw": pose.yaw,
            "actual_vx": vx,
            "actual_vy": vy,
            "actual_wz": wz,
            "hip_error_max_abs": max(abs(v) for v in hip_error),
            "hip_vel_max_abs": max(abs(node.hip_vel[joint]) for joint in HIP_JOINTS),
            "wheel_joint_vel_mean_abs": sum(abs(node.wheel_vel[joint]) for joint in WHEEL_JOINTS) / 6.0,
        }
        for i, leg in enumerate(LEGS):
            row[f"hip_target_{leg}"] = hip_target[i]
            row[f"hip_pos_{leg}"] = hip_pos[i]
            row[f"hip_error_{leg}"] = hip_error[i]
            row[f"wheel_cmd_{leg}"] = wheel_cmd[i]
            row[f"wheel_vel_{leg}"] = node.wheel_vel[WHEEL_JOINTS[i]]
        samples.append(row)
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
    start = samples[0]
    end = samples[-1]
    mean_vx = sum(float(s["actual_vx"]) for s in usable) / len(usable)
    mean_vy = sum(float(s["actual_vy"]) for s in usable) / len(usable)
    mean_wz = sum(float(s["actual_wz"]) for s in usable) / len(usable)
    displacement = math.hypot(float(end["x"]) - float(start["x"]), float(end["y"]) - float(start["y"]))
    yaw_delta = wrap_angle(float(end["yaw"]) - float(start["yaw"]))
    wheel_cmd = [float(start[f"wheel_cmd_{leg}"]) for leg in LEGS]
    left_mean = (wheel_cmd[0] + wheel_cmd[2] + wheel_cmd[4]) / 3.0
    right_mean = (wheel_cmd[1] + wheel_cmd[3] + wheel_cmd[5]) / 3.0
    expected_vx_sign = math.copysign(1.0, left_mean + right_mean) if abs(left_mean + right_mean) > 0.2 else 0.0
    expected_wz_sign = math.copysign(1.0, right_mean - left_mean) if abs(right_mean - left_mean) > 0.2 else 0.0
    direction_mismatch = (
        (expected_vx_sign > 0 and mean_vx < -0.02)
        or (expected_vx_sign < 0 and mean_vx > 0.02)
        or (expected_wz_sign > 0 and mean_wz < -0.02)
        or (expected_wz_sign < 0 and mean_wz > 0.02)
    )
    commanded = max(abs(v) for v in wheel_cmd)
    stuck = commanded > 0.5 and abs(mean_vx) < 0.01 and abs(mean_wz) < 0.01 and displacement < 0.03
    return {
        "segment": str(start["segment"]),
        "wheel_cmd": wheel_cmd,
        "mean_vx": mean_vx,
        "mean_vy": mean_vy,
        "mean_wz": mean_wz,
        "displacement_m": displacement,
        "yaw_delta_rad": yaw_delta,
        "max_abs_roll_rad": max(abs(float(s["roll"])) for s in usable),
        "max_abs_pitch_rad": max(abs(float(s["pitch"])) for s in usable),
        "max_abs_hip_error_rad": max(float(s["hip_error_max_abs"]) for s in usable),
        "max_abs_hip_vel_rad_s": max(float(s["hip_vel_max_abs"]) for s in usable),
        "mean_abs_wheel_joint_vel_rad_s": sum(float(s["wheel_joint_vel_mean_abs"]) for s in usable) / len(usable),
        "direction_mismatch": direction_mismatch,
        "stuck": stuck,
    }


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
    parser.add_argument("--suite-json", default="", help="Optional JSON list replacing the default segment suite.")
    parser.add_argument("--profile", choices=tuple(PROFILES), default="full", help="Built-in suite to run.")
    parser.add_argument("--rate", type=float, default=10.0, help="Publish/sample rate in Hz.")
    parser.add_argument("--duration-scale", type=float, default=1.0, help="Multiply every segment duration.")
    parser.add_argument("--hip-transition", type=float, default=1.0, help="Seconds for each hip trajectory target.")
    parser.add_argument("--ready-timeout", type=float, default=20.0, help="Seconds to wait for Gazebo and ROS topics.")
    parser.add_argument(
        "--out-dir",
        default="",
        help="Output dir. Defaults to generated/benchmarks/chassis_pose_diffdrive/<timestamp>.",
    )
    args = parser.parse_args()

    suite = parse_suite(args.suite_json, args.profile)
    out_dir = (
        Path(args.out_dir)
        if args.out_dir
        else Path("generated/benchmarks/chassis_pose_diffdrive") / datetime.now().strftime("%Y%m%d_%H%M%S")
    )

    rclpy.init()
    node = ChassisTestNode()
    all_samples: list[dict] = []
    summaries: list[dict] = []
    try:
        spin_until_ready(node, timeout_s=args.ready_timeout, model_name=args.model)
        initial_hips = node.current_hip_positions()
        for item in suite:
            current_hips = node.current_hip_positions()
            hip_target = parse_vector(item["hip"], current_hips, initial_hips)
            wheel_cmd = parse_vector(item["wheel"], current_hips, initial_hips)
            duration_s = float(item.get("duration", 4.0)) * args.duration_scale
            name = str(item["name"])
            print(f"SEGMENT {name}: hip={hip_target} wheel={wheel_cmd} duration={duration_s:.2f}s", flush=True)
            samples = sample_segment(
                node,
                model_name=args.model,
                name=name,
                hip_target=hip_target,
                wheel_cmd=wheel_cmd,
                duration_s=duration_s,
                rate_hz=args.rate,
                hip_transition_s=args.hip_transition,
            )
            all_samples.extend(samples)
            summaries.append(summarize_segment(samples))

        node.publish_wheels([0.0] * 6)
        node.publish_hip_target(initial_hips, args.hip_transition)
        summary = {
            "model": args.model,
            "profile": args.profile,
            "rate_hz": args.rate,
            "duration_scale": args.duration_scale,
            "hip_transition_s": args.hip_transition,
            "segments": summaries,
            "failed_segments": [
                s["segment"]
                for s in summaries
                if s.get("direction_mismatch") or s.get("stuck")
            ],
        }
        write_outputs(out_dir, all_samples, summary)
        print(json.dumps(summary, indent=2, sort_keys=True))
    except (KeyboardInterrupt, rclpy.executors.ExternalShutdownException):
        node.publish_wheels([0.0] * 6)
    finally:
        node.destroy_node()
        try:
            rclpy.shutdown()
        except Exception:
            pass
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
