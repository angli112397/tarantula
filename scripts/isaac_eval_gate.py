#!/usr/bin/env python3
"""Gate an Isaac policy eval against the analytic open-loop baseline."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


HARD_TERMINATION_KEYS = ("tilt", "height", "velocity", "bounds", "non_finite")
TURN_AUTHORITY_SEGMENTS = ("turn_left_authority", "turn_right_authority")


def _load_json(path: str | Path) -> dict:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _segments_by_name(summary: dict) -> dict[str, dict]:
    return {str(segment["segment"]): segment for segment in summary.get("segments", [])}


def weighted_score(summary: dict) -> float:
    segments = summary.get("segments", [])
    if not segments:
        raise ValueError("summary has no segments")
    total = 0.0
    for segment in segments:
        total += float(segment.get("rms_vx_error", 0.0)) + 0.5 * float(segment.get("rms_wz_error", 0.0))
    return total / len(segments)


def mean_action_saturation(summary: dict) -> float:
    segments = summary.get("segments", [])
    if not segments:
        return 0.0
    return sum(float(segment.get("action_saturation_rate", 0.0)) for segment in segments) / len(segments)


def hard_termination_count(summary: dict) -> int:
    total = 0
    for segment in summary.get("segments", []):
        terminations = segment.get("termination_counts", {})
        total += sum(int(terminations.get(key, 0)) for key in HARD_TERMINATION_KEYS)
    spawn_terms = summary.get("spawn_health", {}).get("initial_termination_counts", {})
    total += sum(int(spawn_terms.get(key, 0)) for key in HARD_TERMINATION_KEYS)
    return total


def turn_authority_regressions(open_loop: dict, policy: dict, *, tolerance: float) -> list[str]:
    baseline = _segments_by_name(open_loop)
    candidate = _segments_by_name(policy)
    regressions: list[str] = []
    for name in TURN_AUTHORITY_SEGMENTS:
        if name not in baseline or name not in candidate:
            regressions.append(f"{name}:missing")
            continue
        base_error = float(baseline[name].get("rms_wz_error", 0.0))
        policy_error = float(candidate[name].get("rms_wz_error", 0.0))
        if policy_error > base_error * tolerance:
            regressions.append(f"{name}:wz_error {policy_error:.6f}>{base_error * tolerance:.6f}")
    return regressions


def evaluate_gate(
    open_loop: dict,
    policy: dict,
    *,
    min_score_improvement: float = 0.10,
    max_mean_saturation: float = 0.15,
    turn_tolerance: float = 1.0,
) -> dict:
    open_score = weighted_score(open_loop)
    policy_score = weighted_score(policy)
    required_score = open_score * (1.0 - min_score_improvement)
    policy_saturation = mean_action_saturation(policy)
    hard_terms = hard_termination_count(policy)
    regressions = turn_authority_regressions(open_loop, policy, tolerance=turn_tolerance)

    failures: list[str] = []
    if policy_score > required_score:
        failures.append(f"weighted_score {policy_score:.6f}>{required_score:.6f}")
    if policy_saturation > max_mean_saturation:
        failures.append(f"mean_action_saturation {policy_saturation:.6f}>{max_mean_saturation:.6f}")
    if hard_terms:
        failures.append(f"hard_terminations {hard_terms}")
    failures.extend(regressions)

    return {
        "pass": not failures,
        "failures": failures,
        "open_loop_weighted_score": open_score,
        "policy_weighted_score": policy_score,
        "required_policy_weighted_score": required_score,
        "score_improvement_fraction": (open_score - policy_score) / open_score if open_score > 0.0 else 0.0,
        "policy_mean_action_saturation": policy_saturation,
        "max_mean_action_saturation": max_mean_saturation,
        "policy_hard_terminations": hard_terms,
        "turn_authority_regressions": regressions,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Gate Isaac policy eval against open-loop.")
    parser.add_argument("--open-loop", required=True, help="Open-loop Isaac eval JSON.")
    parser.add_argument("--policy", required=True, help="Policy Isaac eval JSON.")
    parser.add_argument("--out", default="", help="Optional JSON output path.")
    parser.add_argument("--min-score-improvement", type=float, default=0.10)
    parser.add_argument("--max-mean-saturation", type=float, default=0.15)
    parser.add_argument("--turn-tolerance", type=float, default=1.0)
    args = parser.parse_args()

    result = evaluate_gate(
        _load_json(args.open_loop),
        _load_json(args.policy),
        min_score_improvement=args.min_score_improvement,
        max_mean_saturation=args.max_mean_saturation,
        turn_tolerance=args.turn_tolerance,
    )
    text = json.dumps(result, indent=2)
    if args.out:
        out = Path(args.out)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(text + "\n", encoding="utf-8")
    print(text)
    return 0 if result["pass"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
