#!/usr/bin/env python3
"""Gazebo posture evaluator for active-suspension acceptance.

The script does not score trajectory tracking. It publishes a simple low-speed
command profile and records posture/support metrics so neutral posture and RL
active suspension can be compared on the same world.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import rclpy

from gazebo_eval_common import (
    LEGS,
    PostureEvalNode,
    posture_sample_fields,
    posture_summary_fields,
    write_attitude_plot,
    write_csv,
)


def collect_sample(node: PostureEvalNode, t: float, cmd_vx: float, cmd_wz: float) -> dict:
    return {"t": t, "cmd_vx": cmd_vx, "cmd_wz": cmd_wz, **posture_sample_fields(node)}


def summarize(rows: list[dict], label: str) -> dict:
    return {"label": label, **posture_summary_fields(rows)}


def wait_for_inputs(node: PostureEvalNode, timeout_s: float) -> None:
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        rclpy.spin_once(node, timeout_sec=0.1)
        if node.ready():
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
    node = PostureEvalNode("gazebo_posture_eval")
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
        fieldnames = ["t", "cmd_vx", "cmd_wz", "roll", "pitch", "roll_pitch_rate",
                      "loaded_wheels", "wheel_load_var"] + [f"hip_{leg}" for leg in LEGS]
        write_csv(out_dir / "samples.csv", rows, fieldnames)
        (out_dir / "summary.json").parent.mkdir(parents=True, exist_ok=True)
        (out_dir / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
        write_attitude_plot(out_dir / "attitude.png", rows)
        print(json.dumps(summary, indent=2, sort_keys=True))
    finally:
        node.publish_cmd(0.0, 0.0)
        node.destroy_node()
        rclpy.shutdown()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
