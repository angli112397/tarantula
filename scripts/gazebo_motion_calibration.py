#!/usr/bin/env python3
"""Gazebo commissioning calibration for the Tarantula motion baseline.

Run this against an already-launched Gazebo session with the classical motion
controller enabled. The script publishes simple /cmd_vel primitives, measures
Gazebo truth pose as observer data, and writes a compact JSON report. It is a
commissioning/eval tool; do not run it in RL reset or runtime controller init.
"""

from __future__ import annotations

import argparse
import json
import math
import subprocess
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from tarantula_control.motion_control import MotionControlConfig
from tarantula_control.suspension_core import HIP_JOINTS, NEUTRAL_HIP_TARGET
from tarantula_control.vehicle_geometry import VEHICLE_GEOMETRY


rclpy: Any = None
Twist: Any = None
RosNode: Any = object
JointState: Any = object


def import_ros() -> None:
    global rclpy, Twist, RosNode, JointState
    if rclpy is not None:
        return
    try:
        import rclpy as rclpy_module
        from geometry_msgs.msg import Twist as TwistMsg
        from rclpy.node import Node as NodeBase
        from sensor_msgs.msg import JointState as JointStateMsg
    except ImportError as exc:  # pragma: no cover - environment guard
        raise RuntimeError("ROS2 Python modules are not available. Run `source /opt/ros/humble/setup.bash` first.") from exc
    rclpy = rclpy_module
    Twist = TwistMsg
    RosNode = NodeBase
    JointState = JointStateMsg


def make_calibration_node():
    import_ros()
    from rosgraph_msgs.msg import Clock

    class CalibrationNode(RosNode):
        def __init__(self):
            super().__init__("gazebo_motion_calibration")
            self.cmd_pub = self.create_publisher(Twist, "/cmd_vel", 10)
            self.clock_s: float | None = None
            self.last_joint_state: JointState | None = None
            self.create_subscription(Clock, "/clock", self._clock_cb, 10)
            self.create_subscription(JointState, "/joint_states", self._joint_cb, 10)

        def _clock_cb(self, msg) -> None:
            self.clock_s = float(msg.clock.sec) + float(msg.clock.nanosec) * 1.0e-9

        def _joint_cb(self, msg) -> None:
            self.last_joint_state = msg

        def publish_cmd(self, vx: float, wz: float) -> None:
            msg = Twist()
            msg.linear.x = float(vx)
            msg.angular.z = float(wz)
            self.cmd_pub.publish(msg)

    return CalibrationNode()


@dataclass(frozen=True)
class TruthPose:
    x: float
    y: float
    z: float
    roll: float
    pitch: float
    yaw: float


@dataclass(frozen=True)
class SegmentResult:
    name: str
    cmd_vx: float
    cmd_wz: float
    requested_duration_s: float
    sim_duration_s: float
    forward_delta_m: float
    lateral_delta_m: float
    yaw_delta_rad: float
    measured_vx: float
    measured_wz: float
    direction_ok: bool
    start_pose: dict
    end_pose: dict


@dataclass(frozen=True)
class NeutralPoseResult:
    pose: dict
    hip_positions: dict
    max_abs_roll_rad: float
    max_abs_pitch_rad: float
    max_abs_hip_error_rad: float
    pass_neutral_pose: bool


def wrap_angle(angle: float) -> float:
    return math.atan2(math.sin(angle), math.cos(angle))


def truth_pose(model_name: str, *, attempts: int = 3) -> TruthPose:
    last_error: Exception | None = None
    lines: list[str] = []

    def parse_vector(line: str) -> list[float] | None:
        try:
            return [float(v) for v in line.strip().strip("[]").split()]
        except ValueError:
            return None

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
                xyz = parse_vector(lines[-2])
                rpy = parse_vector(lines[-1])
                if xyz is not None and rpy is not None and len(xyz) >= 3 and len(rpy) >= 3:
                    return TruthPose(x=xyz[0], y=xyz[1], z=xyz[2], roll=rpy[0], pitch=rpy[1], yaw=rpy[2])
            last_error = RuntimeError((result.stderr or result.stdout).strip())
        except Exception as exc:
            last_error = exc
        if attempt + 1 < attempts:
            time.sleep(0.2)
    raise RuntimeError(f"unexpected ign model output: {lines!r}; last_error={last_error}")


def wait_until_ready(node, model_name: str, timeout_s: float) -> None:
    deadline = time.monotonic() + timeout_s
    last_error: Exception | None = None
    while time.monotonic() < deadline:
        rclpy.spin_once(node, timeout_sec=0.1)
        if node.clock_s is None or node.last_joint_state is None:
            continue
        try:
            truth_pose(model_name)
            return
        except Exception as exc:
            last_error = exc
    raise RuntimeError(f"Gazebo /clock, /joint_states, or truth pose not ready; last_error={last_error}")


def publish_for_wall_time(node, vx: float, wz: float, seconds: float, rate_hz: float) -> None:
    count = max(1, int(float(seconds) * float(rate_hz)))
    for _ in range(count):
        node.publish_cmd(vx, wz)
        rclpy.spin_once(node, timeout_sec=0.0)
        time.sleep(1.0 / rate_hz)


def measure_segment(
    node,
    *,
    model_name: str,
    name: str,
    cmd_vx: float,
    cmd_wz: float,
    duration_s: float,
    rate_hz: float,
    settle_s: float,
    stop_hold_s: float,
) -> SegmentResult:
    publish_for_wall_time(node, 0.0, 0.0, settle_s, rate_hz)
    start_pose = truth_pose(model_name)
    start_clock = float(node.clock_s)
    end_clock = start_clock + float(duration_s)
    while node.clock_s is None or node.clock_s < end_clock:
        node.publish_cmd(cmd_vx, cmd_wz)
        rclpy.spin_once(node, timeout_sec=0.0)
        time.sleep(1.0 / rate_hz)
    measurement_end_clock = float(node.clock_s)
    end_pose = truth_pose(model_name)
    publish_for_wall_time(node, 0.0, 0.0, stop_hold_s, rate_hz)

    sim_duration = max(measurement_end_clock - start_clock, 1.0e-9)
    dx = end_pose.x - start_pose.x
    dy = end_pose.y - start_pose.y
    forward_delta = math.cos(start_pose.yaw) * dx + math.sin(start_pose.yaw) * dy
    lateral_delta = -math.sin(start_pose.yaw) * dx + math.cos(start_pose.yaw) * dy
    yaw_delta = wrap_angle(end_pose.yaw - start_pose.yaw)
    measured_vx = forward_delta / sim_duration
    measured_wz = yaw_delta / sim_duration
    vx_ok = abs(cmd_vx) < 1.0e-6 or measured_vx * cmd_vx > 0.0
    wz_ok = abs(cmd_wz) < 1.0e-6 or measured_wz * cmd_wz > 0.0
    return SegmentResult(
        name=name,
        cmd_vx=float(cmd_vx),
        cmd_wz=float(cmd_wz),
        requested_duration_s=float(duration_s),
        sim_duration_s=sim_duration,
        forward_delta_m=forward_delta,
        lateral_delta_m=lateral_delta,
        yaw_delta_rad=yaw_delta,
        measured_vx=measured_vx,
        measured_wz=measured_wz,
        direction_ok=bool(vx_ok and wz_ok),
        start_pose=asdict(start_pose),
        end_pose=asdict(end_pose),
    )


def neutral_pose_result(node, model_name: str, *, roll_limit: float, pitch_limit: float, hip_error_limit: float) -> NeutralPoseResult:
    pose = truth_pose(model_name)
    joint_state = node.last_joint_state
    positions = dict(zip(joint_state.name, joint_state.position)) if joint_state is not None else {}
    hip_positions = {
        joint: float(positions.get(joint, 0.0))
        for joint in HIP_JOINTS
    }
    hip_errors = [
        abs(float(hip_positions[joint]) - float(target))
        for joint, target in zip(HIP_JOINTS, NEUTRAL_HIP_TARGET)
    ]
    max_hip_error = max(hip_errors, default=0.0)
    max_roll = abs(pose.roll)
    max_pitch = abs(pose.pitch)
    return NeutralPoseResult(
        pose=asdict(pose),
        hip_positions=hip_positions,
        max_abs_roll_rad=max_roll,
        max_abs_pitch_rad=max_pitch,
        max_abs_hip_error_rad=max_hip_error,
        pass_neutral_pose=bool(max_roll <= roll_limit and max_pitch <= pitch_limit and max_hip_error <= hip_error_limit),
    )


def recommendation_ratio(target: float, measured: float, *, direction_ok: bool) -> tuple[float, bool]:
    if abs(target) <= 1.0e-6:
        return 1.0, False
    if not direction_ok or abs(measured) <= 1.0e-6:
        return 1.0, False
    return abs(target / measured), True


def build_report(args: argparse.Namespace) -> dict:
    import_ros()
    rclpy.init()
    node = make_calibration_node()
    defaults = MotionControlConfig()
    try:
        wait_until_ready(node, args.model, args.timeout)
        neutral = neutral_pose_result(
            node,
            args.model,
            roll_limit=args.neutral_roll_limit,
            pitch_limit=args.neutral_pitch_limit,
            hip_error_limit=args.neutral_hip_error_limit,
        )
        segments = [
            measure_segment(
                node,
                model_name=args.model,
                name="forward",
                cmd_vx=args.cmd_vx,
                cmd_wz=0.0,
                duration_s=args.drive_duration,
                rate_hz=args.rate,
                settle_s=args.settle,
                stop_hold_s=args.stop_hold,
            ),
            measure_segment(
                node,
                model_name=args.model,
                name="turn_left",
                cmd_vx=0.0,
                cmd_wz=args.cmd_wz,
                duration_s=args.turn_duration,
                rate_hz=args.rate,
                settle_s=args.settle,
                stop_hold_s=args.stop_hold,
            ),
            measure_segment(
                node,
                model_name=args.model,
                name="turn_right",
                cmd_vx=0.0,
                cmd_wz=-args.cmd_wz,
                duration_s=args.turn_duration,
                rate_hz=args.rate,
                settle_s=args.settle,
                stop_hold_s=args.stop_hold,
            ),
        ]
    finally:
        node.destroy_node()
        rclpy.shutdown()

    forward = segments[0]
    drive_ratio, drive_valid = recommendation_ratio(
        forward.cmd_vx,
        forward.measured_vx,
        direction_ok=forward.direction_ok,
    )
    yaw_ratios: list[float] = []
    yaw_valid_flags: list[bool] = []
    for segment in segments[1:]:
        ratio, valid = recommendation_ratio(
            segment.cmd_wz,
            segment.measured_wz,
            direction_ok=segment.direction_ok,
        )
        yaw_valid_flags.append(valid)
        if valid:
            yaw_ratios.append(ratio)
    yaw_ratio = sum(yaw_ratios) / len(yaw_ratios) if yaw_ratios else 1.0
    yaw_valid = bool(yaw_ratios) and all(yaw_valid_flags)
    recommended_track_scale = float(args.current_track_scale) * yaw_ratio

    return {
        "schema": "tarantula_gazebo_motion_calibration_v1",
        "model": args.model,
        "vehicle_geometry": asdict(VEHICLE_GEOMETRY),
        "current": {
            "drive_scale": float(args.current_drive_scale),
            "yaw_track_scale": float(args.current_track_scale),
            "motion_config_default_yaw_track_scale": float(defaults.yaw_track_scale),
        },
        "recommended": {
            "drive_scale": float(args.current_drive_scale) * drive_ratio,
            "yaw_track_scale": recommended_track_scale,
            "drive_scale_valid": drive_valid,
            "yaw_track_scale_valid": yaw_valid,
        },
        "acceptance": {
            "all_directions_ok": all(segment.direction_ok for segment in segments),
            "neutral_pose_ok": neutral.pass_neutral_pose,
            "max_abs_lateral_drift_m": max(abs(segment.lateral_delta_m) for segment in segments),
        },
        "neutral_pose": asdict(neutral),
        "segments": [asdict(segment) for segment in segments],
        "notes": [
            "drive_scale is an effective forward gain recommendation; keep it as a baseline/randomization center unless a deployable parameter is added.",
            "yaw_track_scale maps to MotionControlConfig.yaw_track_scale and the sim.launch.py yaw_track_scale argument.",
            "The report uses Gazebo truth pose only as observer data. Do not feed truth pose into runtime control or RL observations.",
        ],
    }


def main() -> int:
    defaults = MotionControlConfig()
    parser = argparse.ArgumentParser(description="Calibrate Gazebo straight-drive and pure-turn baseline response.")
    parser.add_argument("--model", default="tarantula")
    parser.add_argument("--cmd-vx", type=float, default=0.10)
    parser.add_argument("--cmd-wz", type=float, default=0.12)
    parser.add_argument("--drive-duration", type=float, default=4.0, help="Straight-drive measurement duration in Gazebo sim seconds.")
    parser.add_argument("--turn-duration", type=float, default=4.0, help="Pure-turn measurement duration in Gazebo sim seconds.")
    parser.add_argument("--settle", type=float, default=0.5, help="Zero-command settle time before each segment, wall seconds.")
    parser.add_argument("--stop-hold", type=float, default=0.5, help="Zero-command hold after each segment, wall seconds.")
    parser.add_argument("--rate", type=float, default=20.0)
    parser.add_argument("--timeout", type=float, default=20.0)
    parser.add_argument("--current-drive-scale", type=float, default=1.0)
    parser.add_argument("--current-track-scale", type=float, default=float(defaults.yaw_track_scale))
    parser.add_argument("--neutral-roll-limit", type=float, default=0.08)
    parser.add_argument("--neutral-pitch-limit", type=float, default=0.08)
    parser.add_argument("--neutral-hip-error-limit", type=float, default=0.08)
    parser.add_argument("--out", default="", help="Optional JSON output path.")
    args = parser.parse_args()

    report = build_report(args)
    text = json.dumps(report, indent=2, sort_keys=True)
    print(text)
    recommended = report["recommended"]
    print(
        "RECOMMENDED "
        f"drive_scale={recommended['drive_scale']:.4f} "
        f"yaw_track_scale={recommended['yaw_track_scale']:.4f}"
    )
    if args.out:
        out_path = Path(args.out)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(text + "\n", encoding="utf-8")
        print(f"OUT={out_path}")
    acceptance = report["acceptance"]
    return 0 if acceptance["all_directions_ok"] and acceptance["neutral_pose_ok"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
