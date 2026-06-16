#!/usr/bin/env python3
"""Gazebo wheel open-loop benchmark for Tarantula.

Prerequisite: start Gazebo with independent wheel/suspension controllers but
without the RL policy node, and let stand_suspension_hold own suspension
commands, for example:

  ros2 launch tarantula_bringup sim.launch.py \\
    gui:=true leveling:=false rl_policy:=true start_rl_policy:=false \\
    stand_hold:=true spawn_z:=0.45

By default the benchmark publishes wheel commands only, samples Gazebo truth
pose via `ign model -m tarantula -p`, and records stand_suspension_hold debug
if that node is running.
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
from sensor_msgs.msg import Imu, JointState
from std_msgs.msg import Float64MultiArray


LEGS = ("fl", "fr", "ml", "mr", "rl", "rr")
SUSP_KP = 130.0
SUSP_KD = 11.0
SUSP_EFFORT_LIMIT = 75.0
DEFAULT_SEQUENCE = [
    ("settle", [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]),
    ("all_positive", [3.0, 3.0, 3.0, 3.0, 3.0, 3.0]),
    ("all_negative", [-3.0, -3.0, -3.0, -3.0, -3.0, -3.0]),
    ("left_negative_right_positive", [-3.0, 3.0, -3.0, 3.0, -3.0, 3.0]),
    ("left_positive_right_negative", [3.0, -3.0, 3.0, -3.0, 3.0, -3.0]),
    ("front_pair_positive", [3.0, 3.0, 0.0, 0.0, 0.0, 0.0]),
    ("middle_pair_positive", [0.0, 0.0, 3.0, 3.0, 0.0, 0.0]),
    ("rear_pair_positive", [0.0, 0.0, 0.0, 0.0, 3.0, 3.0]),
    ("final_stop", [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]),
]


@dataclass
class TruthPose:
    x: float
    y: float
    z: float
    roll: float
    pitch: float
    yaw: float


class OpenLoopNode(Node):
    def __init__(self):
        super().__init__("gazebo_wheel_open_loop_benchmark")
        self.wheel_pub = self.create_publisher(Float64MultiArray, "/wheel_velocity_controller/commands", 10)
        self.susp_pub = self.create_publisher(Float64MultiArray, "/suspension_controller/commands", 10)
        self.last_joint_state: JointState | None = None
        self.last_imu: Imu | None = None
        self.last_stand_debug: list[float] = []
        self.susp_pos = {leg: 0.0 for leg in LEGS}
        self.susp_vel = {leg: 0.0 for leg in LEGS}
        self.create_subscription(JointState, "/joint_states", self._joint_cb, 10)
        self.create_subscription(Imu, "/imu/data", self._imu_cb, 10)
        self.create_subscription(Float64MultiArray, "/stand_suspension_hold/debug", self._stand_debug_cb, 10)

    def _joint_cb(self, msg: JointState) -> None:
        self.last_joint_state = msg
        for i, name in enumerate(msg.name):
            if name.startswith("susp_") and name.endswith("_joint"):
                leg = name[5:-6]
                if leg in self.susp_pos:
                    if i < len(msg.position):
                        self.susp_pos[leg] = float(msg.position[i])
                    if i < len(msg.velocity):
                        self.susp_vel[leg] = float(msg.velocity[i])

    def _imu_cb(self, msg: Imu) -> None:
        self.last_imu = msg

    def _stand_debug_cb(self, msg: Float64MultiArray) -> None:
        self.last_stand_debug = [float(v) for v in msg.data]

    def stand_debug_values(self) -> dict[str, float]:
        values: dict[str, float] = {
            "stand_debug_seen": 0.0,
            "stand_susp_pos_max_abs": 0.0,
            "stand_target_max_abs": 0.0,
            "stand_effort_max_abs": 0.0,
        }
        if len(self.last_stand_debug) < 18:
            return values
        pos = self.last_stand_debug[0:6]
        target = self.last_stand_debug[6:12]
        effort = self.last_stand_debug[12:18]
        values.update(
            {
                "stand_debug_seen": 1.0,
                "stand_susp_pos_max_abs": max(abs(v) for v in pos),
                "stand_target_max_abs": max(abs(v) for v in target),
                "stand_effort_max_abs": max(abs(v) for v in effort),
            }
        )
        for i, leg in enumerate(LEGS):
            values[f"stand_pos_{leg}"] = pos[i]
            values[f"stand_target_{leg}"] = target[i]
            values[f"stand_effort_{leg}"] = effort[i]
        return values

    def suspension_efforts(
        self,
        *,
        mode: str,
        constant_effort: float,
        target: float,
        kp: float,
        kd: float,
        limit: float,
    ) -> list[float] | None:
        if mode == "external":
            return None
        if mode == "zero":
            return [0.0] * 6
        if mode == "constant":
            return [float(constant_effort)] * 6
        efforts = []
        for leg in LEGS:
            effort = kp * (target - self.susp_pos[leg]) - kd * self.susp_vel[leg]
            efforts.append(max(-limit, min(limit, effort)))
        return efforts

    def publish(
        self,
        wheel_cmd: list[float],
        *,
        susp_mode: str,
        susp_effort: float,
        susp_target: float,
        susp_kp: float,
        susp_kd: float,
        susp_limit: float,
    ) -> list[float] | None:
        efforts = self.suspension_efforts(
            mode=susp_mode,
            constant_effort=susp_effort,
            target=susp_target,
            kp=susp_kp,
            kd=susp_kd,
            limit=susp_limit,
        )
        if efforts is not None:
            self.susp_pub.publish(Float64MultiArray(data=efforts))
        self.wheel_pub.publish(Float64MultiArray(data=[float(v) for v in wheel_cmd]))
        return efforts


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


def joint_velocity_stats(msg: JointState | None) -> dict[str, float]:
    if msg is None:
        return {"mean_abs_wheel_joint_vel": 0.0, "max_abs_wheel_joint_vel": 0.0}
    values = []
    for i, name in enumerate(msg.name):
        if name.startswith("wheel_") and name.endswith("_joint") and i < len(msg.velocity):
            values.append(abs(float(msg.velocity[i])))
    if not values:
        return {"mean_abs_wheel_joint_vel": 0.0, "max_abs_wheel_joint_vel": 0.0}
    return {
        "mean_abs_wheel_joint_vel": sum(values) / len(values),
        "max_abs_wheel_joint_vel": max(values),
    }


def spin_until_ready(node: OpenLoopNode, timeout_s: float, model_name: str) -> None:
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        rclpy.spin_once(node, timeout_sec=0.1)
        try:
            truth_pose(model_name)
        except Exception:
            continue
        if node.last_joint_state is not None:
            return
    raise RuntimeError("Gazebo truth pose or /joint_states not available; is sim.launch.py running?")


def sample_segment(
    node: OpenLoopNode,
    *,
    model_name: str,
    segment_name: str,
    wheel_cmd: list[float],
    susp_mode: str,
    susp_effort: float,
    susp_target: float,
    susp_kp: float,
    susp_kd: float,
    susp_limit: float,
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
        susp_efforts = node.publish(
            wheel_cmd,
            susp_mode=susp_mode,
            susp_effort=susp_effort,
            susp_target=susp_target,
            susp_kp=susp_kp,
            susp_kd=susp_kd,
            susp_limit=susp_limit,
        )
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

        joint_stats = joint_velocity_stats(node.last_joint_state)
        stand_debug = node.stand_debug_values()
        samples.append(
            {
                "t_wall": time.time(),
                "segment": segment_name,
                "wheel_fl": wheel_cmd[0],
                "wheel_fr": wheel_cmd[1],
                "wheel_ml": wheel_cmd[2],
                "wheel_mr": wheel_cmd[3],
                "wheel_rl": wheel_cmd[4],
                "wheel_rr": wheel_cmd[5],
                "susp_mode": susp_mode,
                "susp_effort_mean": 0.0 if susp_efforts is None else sum(susp_efforts) / len(susp_efforts),
                "susp_effort_max_abs": 0.0 if susp_efforts is None else max(abs(v) for v in susp_efforts),
                "x": pose.x,
                "y": pose.y,
                "z": pose.z,
                "roll": pose.roll,
                "pitch": pose.pitch,
                "yaw": pose.yaw,
                "actual_vx": vx,
                "actual_vy": vy,
                "actual_wz": wz,
                **joint_stats,
                **stand_debug,
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
    start = samples[0]
    end = samples[-1]
    mean_vx = sum(float(s["actual_vx"]) for s in usable) / len(usable)
    mean_vy = sum(float(s["actual_vy"]) for s in usable) / len(usable)
    mean_wz = sum(float(s["actual_wz"]) for s in usable) / len(usable)
    mean_abs_wheel_joint_vel = sum(float(s["mean_abs_wheel_joint_vel"]) for s in usable) / len(usable)
    max_abs_wheel_joint_vel = max(float(s["max_abs_wheel_joint_vel"]) for s in usable)
    max_abs_roll = max(abs(float(s["roll"])) for s in usable)
    max_abs_pitch = max(abs(float(s["pitch"])) for s in usable)
    max_abs_susp_effort = max(float(s["susp_effort_max_abs"]) for s in usable)
    max_abs_stand_effort = max(float(s.get("stand_effort_max_abs", 0.0)) for s in usable)
    max_abs_stand_pos = max(float(s.get("stand_susp_pos_max_abs", 0.0)) for s in usable)
    stand_debug_seen = any(float(s.get("stand_debug_seen", 0.0)) > 0.5 for s in usable)
    mean_roll = sum(float(s["roll"]) for s in usable) / len(usable)
    mean_pitch = sum(float(s["pitch"]) for s in usable) / len(usable)
    displacement = math.hypot(float(end["x"]) - float(start["x"]), float(end["y"]) - float(start["y"]))
    commanded = max(abs(float(start[f"wheel_{leg}"])) for leg in LEGS)
    spinning_without_motion = commanded > 0.5 and mean_abs_wheel_joint_vel > 0.5 and displacement < 0.03
    return {
        "segment": str(start["segment"]),
        "wheel_cmd": [float(start[f"wheel_{leg}"]) for leg in LEGS],
        "mean_vx": mean_vx,
        "mean_vy": mean_vy,
        "mean_wz": mean_wz,
        "displacement_m": displacement,
        "mean_roll_rad": mean_roll,
        "mean_pitch_rad": mean_pitch,
        "max_abs_roll_rad": max_abs_roll,
        "max_abs_pitch_rad": max_abs_pitch,
        "max_abs_susp_effort_nm": max_abs_susp_effort,
        "susp_effort_saturated": max_abs_susp_effort >= SUSP_EFFORT_LIMIT * 0.98,
        "stand_debug_seen": stand_debug_seen,
        "max_abs_stand_effort_nm": max_abs_stand_effort,
        "max_abs_stand_susp_pos_rad": max_abs_stand_pos,
        "mean_abs_wheel_joint_vel": mean_abs_wheel_joint_vel,
        "max_abs_wheel_joint_vel": max_abs_wheel_joint_vel,
        "spinning_without_motion": spinning_without_motion,
    }


def parse_sequence(specs: list[str]) -> list[tuple[str, list[float]]]:
    if not specs:
        return DEFAULT_SEQUENCE
    sequence = []
    for spec in specs:
        parts = spec.split(",")
        if len(parts) != 7:
            raise ValueError(f"bad --segment {spec!r}; expected name,fl,fr,ml,mr,rl,rr")
        sequence.append((parts[0], [float(v) for v in parts[1:]]))
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
    parser.add_argument("--duration", type=float, default=4.0, help="Seconds per wheel command segment.")
    parser.add_argument("--rate", type=float, default=2.0, help="Truth sample/publish rate in Hz.")
    parser.add_argument(
        "--susp-mode",
        choices=("external", "hold", "zero", "constant"),
        default="external",
        help="Suspension command mode. external does not publish suspension commands.",
    )
    parser.add_argument("--susp-effort", type=float, default=0.0, help="Constant effort when --susp-mode=constant.")
    parser.add_argument("--susp-target", type=float, default=0.0, help="Suspension hold target angle in rad.")
    parser.add_argument("--susp-kp", type=float, default=SUSP_KP)
    parser.add_argument("--susp-kd", type=float, default=SUSP_KD)
    parser.add_argument("--susp-limit", type=float, default=SUSP_EFFORT_LIMIT)
    parser.add_argument(
        "--segment",
        action="append",
        default=[],
        help="Custom segment as name,fl,fr,ml,mr,rl,rr. Repeat to replace default sequence.",
    )
    parser.add_argument(
        "--out-dir",
        default="",
        help="Output dir. Defaults to generated/benchmarks/wheel_open_loop/<timestamp>.",
    )
    args = parser.parse_args()

    sequence = parse_sequence(args.segment)
    out_dir = (
        Path(args.out_dir)
        if args.out_dir
        else Path("generated/benchmarks/wheel_open_loop") / datetime.now().strftime("%Y%m%d_%H%M%S")
    )

    rclpy.init()
    node = OpenLoopNode()
    try:
        spin_until_ready(node, timeout_s=20.0, model_name=args.model)
        all_samples: list[dict] = []
        summaries = []
        for name, wheel_cmd in sequence:
            print(f"SEGMENT {name}: wheel_cmd={wheel_cmd}", flush=True)
            samples = sample_segment(
                node,
                model_name=args.model,
                segment_name=name,
                wheel_cmd=wheel_cmd,
                susp_mode=args.susp_mode,
                susp_effort=args.susp_effort,
                susp_target=args.susp_target,
                susp_kp=args.susp_kp,
                susp_kd=args.susp_kd,
                susp_limit=args.susp_limit,
                duration_s=args.duration,
                rate_hz=args.rate,
            )
            all_samples.extend(samples)
            summaries.append(summarize_segment(samples))
        node.publish(
            [0.0] * 6,
            susp_mode=args.susp_mode,
            susp_effort=args.susp_effort,
            susp_target=args.susp_target,
            susp_kp=args.susp_kp,
            susp_kd=args.susp_kd,
            susp_limit=args.susp_limit,
        )
        summary = {
            "model": args.model,
            "duration_s": args.duration,
            "rate_hz": args.rate,
            "susp_mode": args.susp_mode,
            "susp_effort": args.susp_effort,
            "susp_target": args.susp_target,
            "susp_kp": args.susp_kp,
            "susp_kd": args.susp_kd,
            "susp_limit": args.susp_limit,
            "segments": summaries,
            "failed_segments": [s["segment"] for s in summaries if s.get("spinning_without_motion")],
        }
        write_outputs(out_dir, all_samples, summary)
        print(json.dumps(summary, indent=2, sort_keys=True))
    except (KeyboardInterrupt, rclpy.executors.ExternalShutdownException):
        node.publish(
            [0.0] * 6,
            susp_mode="external",
            susp_effort=0.0,
            susp_target=0.0,
            susp_kp=args.susp_kp,
            susp_kd=args.susp_kd,
            susp_limit=args.susp_limit,
        )
    finally:
        node.destroy_node()
        try:
            rclpy.shutdown()
        except Exception:
            pass
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
