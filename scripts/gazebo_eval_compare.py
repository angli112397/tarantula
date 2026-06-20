#!/usr/bin/env python3
"""Diff two gazebo_posture_eval.py / gazebo_pursuit_eval.py summary.json files.

Does not touch Gazebo/ROS -- run gazebo_posture_eval.py or
gazebo_pursuit_eval.py twice first (e.g. once with
posture_policy_enabled:=false, once with :=true, same --seed/--checkpoints
for gazebo_pursuit_eval.py so both runs chase the same checkpoints), then
point this at the two resulting summary.json files.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

# Lower-is-better metrics get delta = b - a (positive delta = b worse).
# Higher-is-better metrics get delta = a - b. Anything not listed (e.g.
# "label", "samples", or hip_abs_* -- active suspension is *expected* to
# move the hips more than a frozen baseline, so "more hip motion" isn't
# inherently worse) is reported but not signed/judged.
LOWER_IS_BETTER = (
    "roll_rms_rad", "pitch_rms_rad", "max_abs_roll_rad", "max_abs_pitch_rad",
    "tilt_over_0p20_ratio", "roll_pitch_rate_mean", "wheel_load_var_mean",
)
HIGHER_IS_BETTER = ("loaded_wheels_mean", "checkpoints_reached", "distance_traveled_m")


def load_summary(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def compare(a: dict, b: dict) -> list[dict]:
    rows = []
    keys = sorted(set(a.keys()) | set(b.keys()))
    for key in keys:
        if key in ("label", "samples", "checkpoint_count"):
            continue
        va, vb = a.get(key), b.get(key)
        if not isinstance(va, (int, float)) or not isinstance(vb, (int, float)):
            continue
        if key in LOWER_IS_BETTER:
            delta, better = vb - va, "a" if va <= vb else "b"
        elif key in HIGHER_IS_BETTER:
            delta, better = vb - va, "b" if vb >= va else "a"
        else:
            delta, better = vb - va, None
        rows.append({"metric": key, "a": va, "b": vb, "delta_b_minus_a": delta, "better": better})
    return rows


def print_table(rows: list[dict], label_a: str, label_b: str) -> None:
    header = f"{'metric':<24}{label_a:>14}{label_b:>14}{'delta':>14}  better"
    print(header)
    print("-" * len(header))
    for row in rows:
        better = row["better"] or "-"
        print(f"{row['metric']:<24}{row['a']:>14.5g}{row['b']:>14.5g}{row['delta_b_minus_a']:>14.5g}  {better}")


def main() -> int:
    parser = argparse.ArgumentParser(description="Compare two eval summary.json files (e.g. non-RL vs RL).")
    parser.add_argument("summary_a", help="Path to the baseline summary.json (e.g. posture_policy_enabled:=false).")
    parser.add_argument("summary_b", help="Path to the comparison summary.json (e.g. posture_policy_enabled:=true).")
    parser.add_argument("--label-a", default="a")
    parser.add_argument("--label-b", default="b")
    parser.add_argument("--out", default=None, help="Optional path to write the comparison as JSON.")
    args = parser.parse_args()

    a = load_summary(Path(args.summary_a))
    b = load_summary(Path(args.summary_b))
    for label, summary in ((args.label_a, a), (args.label_b, b)):
        if summary.get("out_of_bounds"):
            print(f"[WARN] {label} run left the safe bounds box and was aborted early -- "
                  f"its posture/motion metrics cover a shorter, incomplete run and are not "
                  f"a fair comparison against a run that finished normally.")
    rows = compare(a, b)
    print_table(rows, args.label_a, args.label_b)

    if args.out:
        out_path = Path(args.out)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(rows, indent=2, sort_keys=True), encoding="utf-8")
        print(f"\nwrote {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
