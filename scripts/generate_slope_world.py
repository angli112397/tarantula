#!/usr/bin/env python3
"""Generate a simple smooth slope world for Gazebo traction diagnostics."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np

from tarantula_terrain.exporters import (
    export_height_assets,
    export_obj,
    export_terrain_sdf,
    export_world_sdf,
)


def smoothstep(u: np.ndarray) -> np.ndarray:
    u = np.clip(u, 0.0, 1.0)
    return u * u * (3.0 - 2.0 * u)


def generate_slope(
    *,
    size_x: float,
    size_y: float,
    resolution: float,
    slope_deg: float,
    ramp_length: float,
    flat_length: float,
) -> tuple[np.ndarray, dict]:
    nx = int(round(size_x / resolution)) + 1
    ny = int(round(size_y / resolution)) + 1
    xs = np.linspace(-size_x / 2.0, size_x / 2.0, nx, dtype=np.float32)
    ys = np.linspace(-size_y / 2.0, size_y / 2.0, ny, dtype=np.float32)
    x, _ = np.meshgrid(xs, ys)

    start = -size_x / 2.0 + flat_length
    end = start + ramp_length
    grade = math.tan(math.radians(slope_deg))
    u = (x - start) / max(ramp_length, 1.0e-6)
    ramp = smoothstep(u)
    height = ramp * grade * ramp_length
    height[x < start] = 0.0
    height[x > end] = grade * ramp_length
    height = height - float(height.min())

    metadata = {
        "preset": "single_slope",
        "slope_deg": slope_deg,
        "size_x": size_x,
        "size_y": size_y,
        "resolution": resolution,
        "ramp_length": ramp_length,
        "flat_length": flat_length,
        "height_limit": [float(height.min()), float(height.max())],
        "height_min": float(height.min()),
        "height_max": float(height.max()),
        "wall_height": 0.0,
        "wall_thickness": 0.0,
        "labels": [
            {
                "type": "single_smooth_slope",
                "center": [0.0, 0.0],
                "size": [ramp_length, size_y],
                "slope_deg": slope_deg,
            }
        ],
        "env_origins": [{"row": 0, "col": 0, "difficulty": 0.0, "terrain_type": "single_slope", "xyz": [start - 0.5, 0.0, 0.20]}],
    }
    return height.astype(np.float32), metadata


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--slope-deg", type=float, default=10.0)
    parser.add_argument("--size-x", type=float, default=12.0)
    parser.add_argument("--size-y", type=float, default=6.0)
    parser.add_argument("--resolution", type=float, default=0.05)
    parser.add_argument("--ramp-length", type=float, default=5.0)
    parser.add_argument("--flat-length", type=float, default=1.0)
    parser.add_argument("--output-dir", default="")
    args = parser.parse_args()

    out_dir = Path(args.output_dir) if args.output_dir else Path("generated/terrains/slope") / f"{args.slope_deg:g}deg"
    height, metadata = generate_slope(
        size_x=args.size_x,
        size_y=args.size_y,
        resolution=args.resolution,
        slope_deg=args.slope_deg,
        ramp_length=args.ramp_length,
        flat_length=args.flat_length,
    )
    export_height_assets(out_dir, height, metadata)
    obj = export_obj(out_dir, height, args.resolution)
    terrain = export_terrain_sdf(out_dir, obj)
    world = export_world_sdf(
        out_dir,
        terrain,
        size_x=args.size_x,
        size_y=args.size_y,
        wall_height=0.0,
        wall_thickness=0.0,
    )
    print(json.dumps({"world": str(world), "height_max": float(height.max())}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
