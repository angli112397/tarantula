#!/usr/bin/env python3
"""Gazebo pure-pursuit evaluator: compare non-RL vs RL active suspension under
the same checkpoint-chasing drive, not just a fixed crawl.

Steers with the same proportional heading-error law
TarantulaSuspensionEnv._update_pursuit_commands uses in Isaac Lab training
(see suspension_env.py / suspension_env_cfg.py CommandsCfg.pursuit_heading_gain
docstring for why a plain proportional law on the full signed angle is used
instead of the textbook curvature formula), plus one deliberate addition
Isaac's trainer does not have: --rotate-to-heading-threshold, mirroring Nav2
regulated-pure-pursuit's use_rotate_to_heading (drop cmd_vx to 0 and rotate in
place above a heading-error threshold, instead of always blending forward
drive with a turn). Added here, not in Isaac, because this rover's skid-steer
yaw authority on Gazebo's terrain mesh is much weaker than in Isaac/PhysX (a
known, currently-unresolved dartsim/gz-physics5 limitation -- see
motion_control.py's MotionControlConfig docstring), so blending forward speed
into a turn here leaves too little of that authority to actually rotate;
Isaac has no such deficiency to work around. Steering reads /ground_truth_odom
(Gazebo's OdometryPublisher plugin, bridge_ground_truth_odom:=true on
sim.launch.py) rather than /odometry/filtered: this is a controlled A/B
comparison tool whose job is to isolate the suspension-policy effect, not to
validate localization, so the same deterministic checkpoint sequence and
steering decisions should not vary with EKF/AMCL noise between the non-RL and
RL runs. (A real Nav2 deployment must use /odometry/filtered -- this script
intentionally does not exercise that path; see README/docs for the Nav2 demo.)

Does not avoid obstacles. Run this against an unwalled world (e.g.
generated/terrains/rl_curriculum/<seed>/world.sdf -- the same heightmap Isaac
Lab trains on, no walls at all, bounded by export_world_sdf's surround_copies
tiling instead) so checkpoints sampled anywhere inside the bounds margin are
reachable in a straight-ish line. Do not run it against nav_maze's worlds;
pure pursuit has no path planner and will drive into the maze walls.

Skid-steer turning on a direct-mesh-collision world (no flat floor) needs
all three of: stiff contact via exporters.SurfaceProps, closed-loop
yaw_rate_kp/ki (see motion_control.py's MotionControlConfig docstring), and
isotropic wheel friction (tarantula_v3.urdf.xacro -- anisotropic mu1/mu2
needed a direction-frame attribute that never survives this project's
URDF->SDF conversion under dartsim, see that file's comment) -- if a terrain
dir predates the contact/yaw fix, regenerate it (scripts/check_terrain_freshness.py
flags this) rather than re-debugging "it won't turn" from scratch.
"""

from __future__ import annotations

import argparse
import json
import math
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
from tarantula_control.vehicle_geometry import VEHICLE_GEOMETRY

# Mirrors CommandsCfg.pursuit_heading_gain's default in suspension_env_cfg.py.
DEFAULT_HEADING_GAIN = 1.5

# Mirrors suspension_env_cfg.py's TerminationsCfg.bounds_margin -- same
# single-source-of-truth formula, so checkpoints sampled here stay inside the
# same margin-shrunk box _sample_pursuit_waypoint samples training-side
# checkpoints in, rather than two independently-guessed numbers drifting
# apart.
DEFAULT_MARGIN = 0.5 * VEHICLE_GEOMETRY.reference_length


def checkpoint_bounds(terrain_dir: Path, margin: float) -> tuple[float, float, float, float]:
    """(x_lo, x_hi, y_lo, y_hi) margin-shrunk box matching the nominal
    training-tile size_x/size_y -- where checkpoints get sampled."""
    metadata = json.loads((terrain_dir / "metadata.json").read_text(encoding="utf-8"))
    size_x, size_y = float(metadata["size_x"]), float(metadata["size_y"])
    return (-size_x / 2.0 + margin, size_x / 2.0 - margin, -size_y / 2.0 + margin, size_y / 2.0 - margin)


def safety_bounds(terrain_dir: Path, margin: float) -> tuple[float, float, float, float]:
    """(x_lo, x_hi, y_lo, y_hi) safety-stop box for the in-flight abort check.

    Wider than checkpoint_bounds() when world.sdf tiles repeat copies of the
    same heightmap around the nominal training tile (see export_world_sdf's
    surround_copies, generate.py's metadata["surround_copies"]): straying
    past the nominal size_x/size_y edge there still lands on more terrain,
    not a void, so the real fall-off-the-map edge is surround_copies tiles
    farther out. Defaults to checkpoint_bounds() (surround_copies=0) for
    terrains generated before this field existed.
    """
    metadata = json.loads((terrain_dir / "metadata.json").read_text(encoding="utf-8"))
    size_x, size_y = float(metadata["size_x"]), float(metadata["size_y"])
    surround_copies = int(metadata.get("surround_copies", 0))
    tiled_x = size_x * (2 * surround_copies + 1)
    tiled_y = size_y * (2 * surround_copies + 1)
    return (-tiled_x / 2.0 + margin, tiled_x / 2.0 - margin, -tiled_y / 2.0 + margin, tiled_y / 2.0 - margin)


def sample_checkpoints(terrain_dir: Path, count: int, margin: float, seed: int) -> list[tuple[float, float]]:
    import random

    x_lo, x_hi, y_lo, y_hi = checkpoint_bounds(terrain_dir, margin)
    rng = random.Random(seed)
    return [(rng.uniform(x_lo, x_hi), rng.uniform(y_lo, y_hi)) for _ in range(count)]


def parse_checkpoints(spec: str) -> list[tuple[float, float]]:
    points = []
    for pair in spec.split(";"):
        pair = pair.strip()
        if not pair:
            continue
        x_str, y_str = pair.split(",")
        points.append((float(x_str), float(y_str)))
    return points


def pursuit_heading_error(pos_x: float, pos_y: float, yaw: float, target: tuple[float, float]) -> tuple[float, float]:
    """Return (heading_error, distance) -- identical law to
    TarantulaSuspensionEnv._update_pursuit_commands (body-frame bearing via
    atan2, not the textbook curvature formula)."""
    dx, dy = target[0] - pos_x, target[1] - pos_y
    distance = math.hypot(dx, dy)
    cos_h, sin_h = math.cos(yaw), math.sin(yaw)
    lx = cos_h * dx + sin_h * dy
    ly = -sin_h * dx + cos_h * dy
    return math.atan2(ly, lx), distance


def collect_sample(node: PostureEvalNode, t: float, cmd_vx: float, cmd_wz: float, checkpoint_index: int,
                    distance_to_checkpoint: float, distance_traveled: float) -> dict:
    return {
        "t": t,
        "cmd_vx": cmd_vx,
        "cmd_wz": cmd_wz,
        "pos_x": node.pos_x,
        "pos_y": node.pos_y,
        "checkpoint_index": checkpoint_index,
        "distance_to_checkpoint": distance_to_checkpoint,
        "distance_traveled": distance_traveled,
        **posture_sample_fields(node),
    }


def summarize(rows: list[dict], label: str, checkpoints: list[tuple[float, float]], checkpoints_reached: int,
              out_of_bounds: bool) -> dict:
    return {
        "label": label,
        "checkpoint_count": len(checkpoints),
        "checkpoints_reached": checkpoints_reached,
        "out_of_bounds": out_of_bounds,
        "duration_s": rows[-1]["t"] if rows else 0.0,
        "distance_traveled_m": rows[-1]["distance_traveled"] if rows else 0.0,
        **posture_summary_fields(rows),
    }


def write_trajectory_plot(path: Path, rows: list[dict], checkpoints: list[tuple[float, float]]) -> None:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception:
        return
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(7, 7))
    ax.plot([row["pos_x"] for row in rows], [row["pos_y"] for row in rows], label="path", linewidth=1.2)
    if checkpoints:
        cx, cy = zip(*checkpoints)
        ax.scatter(cx, cy, c="red", marker="x", s=80, label="checkpoints")
    ax.set_xlabel("x [m]")
    ax.set_ylabel("y [m]")
    ax.set_aspect("equal", adjustable="datalim")
    ax.grid(True, linewidth=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def wait_for_inputs(node: PostureEvalNode, timeout_s: float) -> None:
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        rclpy.spin_once(node, timeout_sec=0.1)
        if node.ready():
            return
    raise RuntimeError("timed out waiting for /imu/data, /joint_states and /ground_truth_odom")


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Drive a pure-pursuit checkpoint chase in Gazebo and record posture + motion metrics."
    )
    parser.add_argument("--label", default="pursuit_eval")
    parser.add_argument("--terrain-dir", default="generated/terrains/rl_curriculum/42",
                         help="Used for checkpoint bounds via metadata.json; must match the launched world.")
    parser.add_argument("--checkpoint-count", type=int, default=5)
    parser.add_argument("--checkpoints", default=None,
                         help='Explicit "x1,y1;x2,y2;..." overrides --checkpoint-count/--seed sampling.')
    parser.add_argument("--seed", type=int, default=42, help="Use the same seed for non-RL/RL A/B runs.")
    parser.add_argument("--margin", type=float, default=DEFAULT_MARGIN,
                         help="m, keep-out from the nominal training-tile edge when sampling checkpoints. "
                              "Defaults to suspension_env_cfg.py's TerminationsCfg.bounds_margin value.")
    parser.add_argument("--bounds-margin", type=float, default=1.0,
                         help="m, keep-out from the world's actual outer edge for the in-flight safety-stop check "
                              "(wider than --margin when world.sdf tiles surround_copies of the same terrain).")
    parser.add_argument("--cmd-vx", type=float, default=0.20)
    parser.add_argument("--max-cmd-wz", type=float, default=0.4)
    parser.add_argument("--heading-gain", type=float, default=DEFAULT_HEADING_GAIN)
    parser.add_argument("--rotate-to-heading-threshold", type=float, default=0.5,
                         help="rad. Nav2 regulated-pure-pursuit practice (use_rotate_to_heading, "
                              "'recommended on for all robot types that can rotate in place'): "
                              "above this |heading_error|, drop cmd_vx to 0 and rotate in place "
                              "instead of blending forward drive with a turn. Matters more than "
                              "usual here -- this rover's skid-steer yaw authority on the terrain "
                              "mesh is weak, and asking it to also hold forward speed while turning "
                              "leaves even less of that authority to actually rotate, observed as "
                              "the robot stalling mid-turn rather than completing it. Set <=0 to "
                              "disable and always blend (the old behavior).")
    parser.add_argument("--arrival-radius", type=float, default=1.0,
                         help="Matches CommandsCfg.pursuit_arrival_radius's training default. "
                              "Kept well outside bearing-only pursuit's ill-conditioned zone near "
                              "the target (heading_error = atan2(ly, lx) gets very sensitive to "
                              "position noise as distance -> 0, and cmd_vx never slows on approach).")
    parser.add_argument("--rate", type=float, default=20.0)
    parser.add_argument("--settle", type=float, default=2.0)
    parser.add_argument("--max-duration", type=float, default=180.0, help="Safety cap in case a checkpoint is never reached.")
    parser.add_argument("--out-dir", default="generated/benchmarks/pursuit_eval/latest")
    parser.add_argument("--input-timeout", type=float, default=10.0)
    parser.add_argument("--tilt-gate-rad", type=float, default=0.05,
                         help="Mirrors scan_gate.py's tilt_gate default (rad) -- drawn as a "
                              "reference line on attitude.png. Set <=0 to omit.")
    args = parser.parse_args()

    if args.checkpoints:
        checkpoints = parse_checkpoints(args.checkpoints)
    else:
        checkpoints = sample_checkpoints(Path(args.terrain_dir), args.checkpoint_count, args.margin, args.seed)
    # Safety-stop region: a checkpoint being inside checkpoint_bounds() only
    # guarantees a valid destination, not a valid path there -- check every
    # step, not just at sample time. Uses the wider safety_bounds() box (the
    # world's real outer edge, accounting for surround_copies tiling).
    x_lo, x_hi, y_lo, y_hi = safety_bounds(Path(args.terrain_dir), args.bounds_margin)

    rclpy.init()
    node = PostureEvalNode("gazebo_pursuit_eval", track_position=True)
    # Mutable across step() calls -- a plain dict instead of nonlocal locals
    # since step() is a nested function, not a class method.
    state = {
        "rows": [],
        "checkpoint_index": 0,
        "distance_traveled": 0.0,
        "out_of_bounds": False,
        "last_pos": None,
        "start": None,
        "done": False,
    }

    def step() -> None:
        # First tick lazily captures sim-time zero -- can't read node.now_s()
        # before the node exists, and this runs inside create_timer's own
        # callback, fired by the node's (sim-time, once use_sim_time is on)
        # clock, not time.sleep()-paced wall clock. See PostureEvalNode's
        # use_sim_time comment: this is what makes the per-tick control
        # cadence (not just the overall --max-duration budget, fixed
        # separately via now_s() below) immune to real_time_factor drift
        # from system/render load -- a manual while+time.sleep loop, even
        # one gated on now_s() for its *stop* condition, still ticks at a
        # load-dependent rate, since time.sleep() itself is wall-clock.
        if state["start"] is None:
            state["start"] = node.now_s()
        elapsed = node.now_s() - state["start"]
        active = elapsed >= args.settle

        if state["last_pos"] is not None:
            state["distance_traveled"] += math.hypot(
                node.pos_x - state["last_pos"][0], node.pos_y - state["last_pos"][1]
            )
        state["last_pos"] = (node.pos_x, node.pos_y)

        if not (x_lo <= node.pos_x <= x_hi and y_lo <= node.pos_y <= y_hi):
            state["out_of_bounds"] = True
            print(f"[ABORT] left the safe bounds box at ({node.pos_x:.2f}, {node.pos_y:.2f}) "
                  f"-- x in [{x_lo:.2f},{x_hi:.2f}], y in [{y_lo:.2f},{y_hi:.2f}]")
            state["done"] = True
            return

        if elapsed >= args.settle + args.max_duration or state["checkpoint_index"] >= len(checkpoints):
            state["done"] = True
            return

        if not active:
            node.publish_cmd(0.0, 0.0)
            return

        target = checkpoints[state["checkpoint_index"]]
        heading_error, distance = pursuit_heading_error(node.pos_x, node.pos_y, node.yaw, target)
        cmd_wz = max(-args.max_cmd_wz, min(args.max_cmd_wz, args.heading_gain * heading_error))
        if args.rotate_to_heading_threshold > 0.0 and abs(heading_error) > args.rotate_to_heading_threshold:
            cmd_vx = 0.0
        else:
            cmd_vx = args.cmd_vx
        node.publish_cmd(cmd_vx, cmd_wz)
        state["rows"].append(collect_sample(node, elapsed - args.settle, cmd_vx, cmd_wz,
                                             state["checkpoint_index"], distance, state["distance_traveled"]))
        if distance < args.arrival_radius:
            state["checkpoint_index"] += 1

    try:
        wait_for_inputs(node, args.input_timeout)
        timer = node.create_timer(1.0 / args.rate, step)
        while not state["done"]:
            rclpy.spin_once(node, timeout_sec=0.05)
        timer.cancel()
        node.publish_cmd(0.0, 0.0)

        rows = state["rows"]
        checkpoint_index = state["checkpoint_index"]
        out_of_bounds = state["out_of_bounds"]
        out_dir = Path(args.out_dir)
        summary = summarize(rows, args.label, checkpoints, checkpoint_index, out_of_bounds)
        fieldnames = ["t", "cmd_vx", "cmd_wz", "pos_x", "pos_y", "checkpoint_index",
                      "distance_to_checkpoint", "distance_traveled", "roll", "pitch",
                      "roll_pitch_rate", "loaded_wheels", "wheel_load_var"] + [f"hip_{leg}" for leg in LEGS]
        write_csv(out_dir / "samples.csv", rows, fieldnames)
        out_dir.mkdir(parents=True, exist_ok=True)
        (out_dir / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
        (out_dir / "checkpoints.json").write_text(json.dumps(checkpoints, indent=2), encoding="utf-8")
        write_attitude_plot(out_dir / "attitude.png", rows,
                             tilt_gate_rad=args.tilt_gate_rad if args.tilt_gate_rad > 0.0 else None)
        write_trajectory_plot(out_dir / "trajectory.png", rows, checkpoints)
        print(json.dumps(summary, indent=2, sort_keys=True))
        if out_of_bounds:
            print("[WARN] run aborted early: left the safe bounds box (see trajectory.png)")
        elif checkpoint_index < len(checkpoints):
            print(f"[WARN] only reached {checkpoint_index}/{len(checkpoints)} checkpoints before --max-duration")
    finally:
        node.publish_cmd(0.0, 0.0)
        node.destroy_node()
        rclpy.shutdown()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
