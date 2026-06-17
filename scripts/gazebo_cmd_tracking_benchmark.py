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
`ign model -m tarantula -p`, and records command tracking metrics against the
shaped execution command reported by /rl_policy/status.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import subprocess
import sys
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

try:
    import rclpy
    from geometry_msgs.msg import Twist, Wrench
    from rclpy.node import Node
    from sensor_msgs.msg import JointState
    from std_msgs.msg import Float64MultiArray
except ModuleNotFoundError:
    rclpy = None
    Twist = object
    Wrench = object
    Node = object
    JointState = object
    Float64MultiArray = object


DEFAULT_SEQUENCE = [
    ("stop", 0.0, 0.0),
    ("turn_left_from_drive_cmd", 0.1, 0.15),
    ("drive_after_left", 0.1, 0.0),
    ("turn_right_from_drive_cmd", 0.1, -0.15),
    ("drive_after_right", 0.1, 0.0),
    ("backward", -0.1, 0.0),
    ("turn_left_authority", 0.0, 0.25),
    ("turn_right_authority", 0.0, -0.25),
    ("final_stop", 0.0, 0.0),
]

LEGS = ("fl", "fr", "ml", "mr", "rl", "rr")
HIP_JOINTS = tuple(f"susp_{leg}_joint" for leg in LEGS)
WHEEL_CMD_FIELDS = tuple(f"wheel_cmd_{leg}" for leg in LEGS)
HIP_POS_FIELDS = tuple(f"hip_pos_{leg}" for leg in LEGS)
WHEEL_FORCE_FIELDS = tuple(f"wheel_force_norm_{leg}" for leg in LEGS)
RL_ACTION_FIELDS = (
    "rl_action_track_scale",
    "rl_action_left_drive",
    "rl_action_right_drive",
)

STATUS_LAYOUT = (
    "rl_enabled",
    "rl_action_track_scale",
    "rl_action_left_drive",
    "rl_action_right_drive",
    "rl_action_saturation",
    "status_wheel_cmd_max_abs",
    "status_cmd_vx",
    "status_cmd_wz",
    "measured_wz",
    "motion_mode_turn",
)

NOMINAL_WHEEL_LOAD_N = 23.1 * 9.81 / 6.0
MAX_ABS_WHEEL_CMD_RAD_S = 6.0
WHEEL_CMD_SATURATION_FRACTION = 0.98
ACTION_SATURATION_THRESHOLD = 0.95
MAX_ACCEPTABLE_ACTION_SATURATION_RATE = 0.20
HIP_SOFT_LIMIT_RAD = 0.40
STABILITY_REGRESSION_RATIO = 1.15
TRACKING_REGRESSION_RATIO = 1.05


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
        self.last_rl_status: list[float] = []
        self.last_wheel_force = {leg: 0.0 for leg in LEGS}
        self.create_subscription(
            Float64MultiArray,
            "/wheel_velocity_controller/commands",
            self._wheel_cmd_cb,
            10,
        )
        self.create_subscription(JointState, "/joint_states", self._joint_cb, 10)
        self.create_subscription(
            Float64MultiArray,
            "/rl_policy/status",
            self._rl_status_cb,
            10,
        )
        for leg in LEGS:
            self.create_subscription(
                Wrench,
                f"/ft_wheel/{leg}",
                lambda msg, l=leg: self._wheel_force_cb(msg, l),
                10,
            )

    def _wheel_cmd_cb(self, msg: Float64MultiArray) -> None:
        self.last_wheel_cmd = [float(v) for v in msg.data]

    def _joint_cb(self, msg: JointState) -> None:
        self.last_joint_state = msg

    def _rl_status_cb(self, msg: Float64MultiArray) -> None:
        self.last_rl_status = [float(v) for v in msg.data]

    def _wheel_force_cb(self, msg: Wrench, leg: str) -> None:
        force = msg.force
        force_norm_n = math.sqrt(
            force.x * force.x + force.y * force.y + force.z * force.z
        )
        self.last_wheel_force[leg] = min(force_norm_n / NOMINAL_WHEEL_LOAD_N, 3.0)

    def publish_cmd(self, vx: float, wz: float) -> None:
        msg = Twist()
        msg.linear.x = float(vx)
        msg.angular.z = float(wz)
        self.cmd_pub.publish(msg)


def wrap_angle(angle: float) -> float:
    return math.atan2(math.sin(angle), math.cos(angle))


def rms(values: list[float]) -> float:
    if not values:
        return 0.0
    return math.sqrt(sum(value * value for value in values) / len(values))


def mean(values: list[float]) -> float:
    if not values:
        return 0.0
    return sum(values) / len(values)


def sample_value(sample: dict, key: str) -> float:
    value = sample.get(key, 0.0)
    if value in ("", None):
        return 0.0
    return float(value)


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
        wheel_cmd_by_leg = {
            leg: float(wheel_cmd[idx]) if idx < len(wheel_cmd) else 0.0
            for idx, leg in enumerate(LEGS)
        }
        hip_pos_by_leg = {leg: 0.0 for leg in LEGS}
        if node.last_joint_state is not None:
            positions = dict(zip(node.last_joint_state.name, node.last_joint_state.position))
            hip_pos_by_leg = {
                leg: float(positions.get(joint_name, 0.0))
                for leg, joint_name in zip(LEGS, HIP_JOINTS)
            }
        status = {
            name: float(node.last_rl_status[idx]) if idx < len(node.last_rl_status) else 0.0
            for idx, name in enumerate(STATUS_LAYOUT)
        }
        sample = {
            "t_wall": time.time(),
            "segment": segment_name,
            "cmd_vx": cmd_vx,
            "cmd_wz": cmd_wz,
            "target_vx": status.get("status_cmd_vx", cmd_vx),
            "target_wz": status.get("status_cmd_wz", cmd_wz),
            "x": pose.x,
            "y": pose.y,
            "z": pose.z,
            "roll": pose.roll,
            "pitch": pose.pitch,
            "yaw": pose.yaw,
            "actual_vx": vx,
            "actual_wz": wz,
            "vx_error": vx - status.get("status_cmd_vx", cmd_vx),
            "wz_error": wz - status.get("status_cmd_wz", cmd_wz),
            "wheel_cmd_max_abs": max((abs(v) for v in wheel_cmd_by_leg.values()), default=0.0),
            "hip_pos_max_abs": max((abs(v) for v in hip_pos_by_leg.values()), default=0.0),
        }
        sample.update({field: wheel_cmd_by_leg[leg] for field, leg in zip(WHEEL_CMD_FIELDS, LEGS)})
        sample.update({field: hip_pos_by_leg[leg] for field, leg in zip(HIP_POS_FIELDS, LEGS)})
        sample.update({field: node.last_wheel_force[leg] for field, leg in zip(WHEEL_FORCE_FIELDS, LEGS)})
        sample.update(status)
        samples.append(sample)
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
    target_vx_values = [sample_value(s, "target_vx") for s in usable]
    target_wz_values = [sample_value(s, "target_wz") for s in usable]
    target_vx = mean(target_vx_values)
    target_wz = mean(target_wz_values)
    start = samples[0]
    end = samples[-1]
    mean_vx = sum(float(s["actual_vx"]) for s in usable) / len(usable)
    mean_wz = sum(float(s["actual_wz"]) for s in usable) / len(usable)
    vx_errors = [float(s["actual_vx"]) - sample_value(s, "target_vx") for s in usable]
    wz_errors = [float(s["actual_wz"]) - sample_value(s, "target_wz") for s in usable]
    rms_vx_error = rms(vx_errors)
    rms_wz_error = rms(wz_errors)
    max_roll = max(abs(float(s["roll"])) for s in usable)
    max_pitch = max(abs(float(s["pitch"])) for s in usable)
    roll_rms = rms([float(s["roll"]) for s in usable])
    pitch_rms = rms([float(s["pitch"]) for s in usable])
    max_hip_pos = max(float(s["hip_pos_max_abs"]) for s in usable)
    max_wheel_cmd = max(float(s["wheel_cmd_max_abs"]) for s in usable)
    wheel_saturation_threshold = MAX_ABS_WHEEL_CMD_RAD_S * WHEEL_CMD_SATURATION_FRACTION
    wheel_saturation_rate = mean([
        1.0 if sample_value(s, "wheel_cmd_max_abs") >= wheel_saturation_threshold else 0.0
        for s in usable
    ])
    action_saturation_rate = mean([
        1.0 if sample_value(s, "rl_action_saturation") >= ACTION_SATURATION_THRESHOLD else 0.0
        for s in usable
    ])
    action_rate_values = []
    for prev, current in zip(usable, usable[1:]):
        dt = max(sample_value(current, "t_wall") - sample_value(prev, "t_wall"), 1.0e-6)
        action_rate_values.extend(
            (sample_value(current, field) - sample_value(prev, field)) / dt
            for field in RL_ACTION_FIELDS
        )
    max_wheel_force = max(
        (sample_value(s, field) for s in usable for field in WHEEL_FORCE_FIELDS),
        default=0.0,
    )
    displacement = math.hypot(float(end["x"]) - float(start["x"]), float(end["y"]) - float(start["y"]))
    yaw_delta = wrap_angle(float(end["yaw"]) - float(start["yaw"]))
    direction_mismatch = (
        (target_vx > 0.05 and mean_vx < -0.02)
        or (target_vx < -0.05 and mean_vx > 0.02)
        or (target_wz > 0.05 and mean_wz < -0.02)
        or (target_wz < -0.05 and mean_wz > 0.02)
    )
    stuck = abs(target_vx) > 0.05 and displacement < 0.03
    return {
        "segment": str(usable[0]["segment"]),
        "cmd_vx": cmd_vx,
        "cmd_wz": cmd_wz,
        "target_vx": target_vx,
        "target_wz": target_wz,
        "mean_vx": mean_vx,
        "mean_wz": mean_wz,
        "rms_vx_error": rms_vx_error,
        "rms_wz_error": rms_wz_error,
        "mean_abs_vx_error": mean([abs(v) for v in vx_errors]),
        "mean_abs_wz_error": mean([abs(v) for v in wz_errors]),
        "final_vx": float(usable[-1]["actual_vx"]),
        "final_wz": float(usable[-1]["actual_wz"]),
        "displacement_m": displacement,
        "yaw_delta_rad": yaw_delta,
        "max_abs_roll_rad": max_roll,
        "max_abs_pitch_rad": max_pitch,
        "roll_rms_rad": roll_rms,
        "pitch_rms_rad": pitch_rms,
        "max_abs_wheel_cmd_rad_s": max_wheel_cmd,
        "wheel_cmd_saturation_rate": wheel_saturation_rate,
        "max_abs_hip_pos_rad": max_hip_pos,
        "max_wheel_force_norm": max_wheel_force,
        "rl_enabled": bool(round(sample_value(usable[-1], "rl_enabled"))),
        "rl_action_saturation_rate": action_saturation_rate,
        "rl_action_rate_rms": rms(action_rate_values),
        "wheel_cmd_saturated": max_wheel_cmd >= wheel_saturation_threshold,
        "hip_near_limit": max_hip_pos >= HIP_SOFT_LIMIT_RAD,
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


def weighted_tracking_score(summary: dict) -> float:
    segments = [segment for segment in summary.get("segments", []) if segment]
    if not segments:
        return float("inf")
    return mean([
        float(segment.get("rms_vx_error", 0.0)) + 0.5 * float(segment.get("rms_wz_error", 0.0))
        for segment in segments
    ])


def compare_summaries(baseline: dict, candidate: dict) -> dict:
    baseline_segments = {
        segment["segment"]: segment for segment in baseline.get("segments", []) if segment
    }
    candidate_segments = {
        segment["segment"]: segment for segment in candidate.get("segments", []) if segment
    }
    segment_names = sorted(set(baseline_segments) & set(candidate_segments))
    comparisons = []
    improved = []
    regressed = []
    hard_failures = []
    for name in segment_names:
        base = baseline_segments[name]
        cand = candidate_segments[name]
        delta_vx = float(cand.get("rms_vx_error", 0.0)) - float(base.get("rms_vx_error", 0.0))
        delta_wz = float(cand.get("rms_wz_error", 0.0)) - float(base.get("rms_wz_error", 0.0))
        delta_roll = float(cand.get("roll_rms_rad", 0.0)) - float(base.get("roll_rms_rad", 0.0))
        delta_pitch = float(cand.get("pitch_rms_rad", 0.0)) - float(base.get("pitch_rms_rad", 0.0))
        base_score = float(base.get("rms_vx_error", 0.0)) + 0.5 * float(base.get("rms_wz_error", 0.0))
        cand_score = float(cand.get("rms_vx_error", 0.0)) + 0.5 * float(cand.get("rms_wz_error", 0.0))
        if cand_score < base_score:
            improved.append(name)
        elif cand_score > base_score * TRACKING_REGRESSION_RATIO:
            regressed.append(name)
        if cand.get("direction_mismatch") or cand.get("stuck") or cand.get("hip_near_limit"):
            hard_failures.append(name)
        if (
            float(cand.get("roll_rms_rad", 0.0))
            > STABILITY_REGRESSION_RATIO * max(float(base.get("roll_rms_rad", 0.0)), 1.0e-6)
            or float(cand.get("pitch_rms_rad", 0.0))
            > STABILITY_REGRESSION_RATIO * max(float(base.get("pitch_rms_rad", 0.0)), 1.0e-6)
        ):
            hard_failures.append(f"{name}:stability_regression")
        comparisons.append({
            "segment": name,
            "baseline_score": base_score,
            "candidate_score": cand_score,
            "delta_rms_vx_error": delta_vx,
            "delta_rms_wz_error": delta_wz,
            "delta_roll_rms_rad": delta_roll,
            "delta_pitch_rms_rad": delta_pitch,
            "delta_action_saturation_rate": (
                float(cand.get("rl_action_saturation_rate", 0.0))
                - float(base.get("rl_action_saturation_rate", 0.0))
            ),
        })
    baseline_score = weighted_tracking_score(baseline)
    candidate_score = weighted_tracking_score(candidate)
    pass_candidate = (
        bool(segment_names)
        and not hard_failures
        and candidate_score <= baseline_score
        and mean([
            float(segment.get("rl_action_saturation_rate", 0.0))
            for segment in candidate_segments.values()
        ]) <= MAX_ACCEPTABLE_ACTION_SATURATION_RATE
    )
    return {
        "baseline_label": baseline.get("label", "baseline"),
        "candidate_label": candidate.get("label", "candidate"),
        "baseline_score": baseline_score,
        "candidate_score": candidate_score,
        "score_delta": candidate_score - baseline_score,
        "segments_compared": segment_names,
        "segments_improved": improved,
        "segments_regressed": regressed,
        "hard_failures": hard_failures,
        "pass": pass_candidate,
        "comparisons": comparisons,
    }


def load_summary(path: Path) -> dict:
    if path.is_dir():
        path = path / "summary.json"
    return json.loads(path.read_text(encoding="utf-8"))


def compare_main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(
        description="Compare two Gazebo command-tracking benchmark summaries."
    )
    parser.add_argument("--baseline", required=True, help="Baseline summary.json or benchmark output directory.")
    parser.add_argument("--candidate", required=True, help="Candidate summary.json or benchmark output directory.")
    parser.add_argument("--out", default="", help="Optional output JSON path.")
    args = parser.parse_args(argv)
    comparison = compare_summaries(load_summary(Path(args.baseline)), load_summary(Path(args.candidate)))
    if args.out:
        out_path = Path(args.out)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(comparison, indent=2, sort_keys=True), encoding="utf-8")
        print(f"COMPARISON={out_path}")
    print(json.dumps(comparison, indent=2, sort_keys=True))
    return 0


def run_main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--label", default="run", help="Run label, for example classical or rl.")
    parser.add_argument("--model", default="tarantula")
    parser.add_argument("--duration", type=float, default=4.0, help="Seconds per command segment.")
    parser.add_argument("--settle", type=float, default=1.0, help="Seconds of zero command before sequence.")
    parser.add_argument("--rate", type=float, default=5.0, help="Truth sample/publish rate in Hz.")
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
    args = parser.parse_args(argv)
    if rclpy is None:
        raise RuntimeError("ROS2 Python modules are not available. Run `source /opt/ros/humble/setup.bash` first.")

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
            "label": args.label,
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


def main() -> int:
    argv = sys.argv[1:]
    if argv and argv[0] == "compare":
        return compare_main(argv[1:])
    return run_main(argv)


if __name__ == "__main__":
    raise SystemExit(main())
