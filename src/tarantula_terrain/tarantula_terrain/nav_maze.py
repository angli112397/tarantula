"""Generate aligned navigation occupancy maps and Gazebo maze worlds.

The output uses the same world coordinate convention as Tarantula heightmaps:
the map is centered at (0, 0), with x spanning size_x and y spanning size_y.
This lets a future RL height layer and a navigation occupancy layer share one
grid and be combined without coordinate conversion.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass, replace
from pathlib import Path

import numpy as np

from .exporters import (
    GENERATOR_SCHEMA_VERSION,
    export_height_assets,
    export_navigation_mesh_contact_world_sdf,
    export_navigation_world_sdf,
    export_obj,
    export_terrain_sdf,
    occupancy_to_pgm_image,
    scaled_cost_to_pgm_image,
    write_map_yaml,
    write_pgm,
)
from .generator import generate_heightmap
from .terrain_cfg import RL_CURRICULUM, TerrainCfg


@dataclass(frozen=True)
class NavMazeCfg:
    # Grid geometry is delegated to a TerrainCfg rather than copied, so it
    # can't silently drift from RL_CURRICULUM (nav_maze reuses its heightmap
    # generator and must share the same grid). Wall/door fields below are
    # maze-layout concepts with no TerrainCfg equivalent, kept independent.
    terrain: TerrainCfg = RL_CURRICULUM
    wall_thickness: float = 0.22
    wall_height: float = 1.10
    door_width: float = 5.20
    min_corridor_width: float = 4.80
    obstacle_count: int = 3

    @property
    def size_x(self) -> float:
        return self.terrain.size_x

    @property
    def size_y(self) -> float:
        return self.terrain.size_y

    @property
    def resolution(self) -> float:
        return self.terrain.resolution

    @property
    def nx(self) -> int:
        return self.terrain.nx

    @property
    def ny(self) -> int:
        return self.terrain.ny


# Pad positions designed for the default 24x16 arena.
# The navigation maze is intentionally a large-chassis baseline, not a tight
# clearance benchmark. The robot starts at the world origin in a broad clear
# zone; goals sit in different quadrants so static-map Nav2 and online SLAM
# both exercise turning, corridor choice, and loop-closing-friendly geometry.
_PAD_POSITIONS: tuple[tuple[float, float], ...] = (
    (0.0, 0.0),     # spawn: origin baseline
    (7.8, 5.2),     # goal:  NE room
    (-8.2, 5.0),    # goal:  NW room
    (8.0, -5.0),    # goal:  SE room
    (-8.0, -5.2),   # goal:  SW room
)
_PAD_CLEAR_SIZE = 4.8  # m — enough room for skid-steer turns without wall contact


def _local_relief(height: np.ndarray, radius_cells: int) -> np.ndarray:
    padded = np.pad(height, radius_cells, mode="edge")
    window = 2 * radius_cells + 1
    windows = np.lib.stride_tricks.sliding_window_view(padded, (window, window))
    return windows.max(axis=(-2, -1)) - windows.min(axis=(-2, -1))


def _compute_traversability_cost(height: np.ndarray, occ: np.ndarray, cfg: NavMazeCfg) -> tuple[np.ndarray, dict]:
    """Convert the shared height layer into a 0..100 Nav2 terrain cost layer.

    This is deliberately conservative and interpretable: Nav2 remains a 2D
    planner, while the height layer contributes extra cost for steep or locally
    discontinuous ground. Walls stay lethal through the occupancy layer.
    """

    gy, gx = np.gradient(height.astype(np.float32), cfg.resolution, cfg.resolution)
    slope_deg = np.degrees(np.arctan(np.hypot(gx, gy)))
    relief = _local_relief(height.astype(np.float32), radius_cells=max(1, int(round(0.35 / cfg.resolution))))

    slope_soft, slope_hard = 3.0, 12.0
    relief_soft, relief_hard = 0.020, 0.090
    slope_cost = np.clip((slope_deg - slope_soft) / (slope_hard - slope_soft), 0.0, 1.0)
    relief_cost = np.clip((relief - relief_soft) / (relief_hard - relief_soft), 0.0, 1.0)

    # Max keeps single sharp terrain hazards visible; weighted blend avoids
    # over-penalizing gentle rolling terrain that is visually rough but drivable.
    edge_cost = np.clip((relief - 0.055) / 0.035, 0.0, 1.0)
    edge_cost = _local_relief(edge_cost.astype(np.float32), radius_cells=max(1, int(round(0.25 / cfg.resolution))))
    edge_cost = np.clip(edge_cost, 0.0, 1.0)

    risk = np.maximum(0.55 * slope_cost + 0.45 * relief_cost, np.maximum(slope_cost, relief_cost) * 0.80)
    risk = np.maximum(risk, edge_cost * 0.90)
    cost = np.rint(risk * 99.0).astype(np.uint8)
    cost[occ] = 100

    for cx, cy in _PAD_POSITIONS:
        _clear_rect(cost, cfg, cx, cy, _PAD_CLEAR_SIZE, _PAD_CLEAR_SIZE)

    stats = {
        "slope_deg_min": float(slope_deg.min()),
        "slope_deg_max": float(slope_deg.max()),
        "slope_soft_deg": slope_soft,
        "slope_hard_deg": slope_hard,
        "relief_m_min": float(relief.min()),
        "relief_m_max": float(relief.max()),
        "relief_window_m": round((2 * max(1, int(round(0.35 / cfg.resolution))) + 1) * cfg.resolution, 3),
        "relief_soft_m": relief_soft,
        "relief_hard_m": relief_hard,
        "cost_min": int(cost.min()),
        "cost_max": int(cost.max()),
        "lethal_cells": int(np.count_nonzero(cost >= 100)),
        "medium_cost_cells": int(np.count_nonzero((cost >= 35) & (cost < 100))),
        "high_cost_cells": int(np.count_nonzero((cost >= 70) & (cost < 100))),
    }
    return cost, stats


def _compute_speed_mask(traversability_cost: np.ndarray, occ: np.ndarray, cfg: NavMazeCfg) -> np.ndarray:
    """Nav2 SpeedFilter mask: 0 means no speed limit, larger values slow down.

    With the launch configuration ``base=100`` and ``multiplier=-1``, mask value
    40 means a 60% speed limit. Walls are not encoded here; they remain ordinary
    occupancy/costmap obstacles.
    """

    mask = np.clip(traversability_cost.astype(np.float32) * 0.65, 0.0, 65.0).astype(np.uint8)
    mask[occ] = 0
    for cx, cy in _PAD_POSITIONS:
        _clear_rect(mask, cfg, cx, cy, _PAD_CLEAR_SIZE, _PAD_CLEAR_SIZE)
    return mask


def _world_to_index(value: float, size: float, resolution: float, limit: int) -> int:
    return int(np.clip(round((value + size / 2.0) / resolution), 0, limit - 1))


def _mark_rect(occ: np.ndarray, cfg: NavMazeCfg, cx: float, cy: float, sx: float, sy: float) -> None:
    x0 = _world_to_index(cx - sx / 2.0, cfg.size_x, cfg.resolution, cfg.nx)
    x1 = _world_to_index(cx + sx / 2.0, cfg.size_x, cfg.resolution, cfg.nx)
    y0 = _world_to_index(cy - sy / 2.0, cfg.size_y, cfg.resolution, cfg.ny)
    y1 = _world_to_index(cy + sy / 2.0, cfg.size_y, cfg.resolution, cfg.ny)
    occ[y0 : y1 + 1, x0 : x1 + 1] = True


def _clear_rect(occ: np.ndarray, cfg: NavMazeCfg, cx: float, cy: float, sx: float, sy: float) -> None:
    x0 = _world_to_index(cx - sx / 2.0, cfg.size_x, cfg.resolution, cfg.nx)
    x1 = _world_to_index(cx + sx / 2.0, cfg.size_x, cfg.resolution, cfg.nx)
    y0 = _world_to_index(cy - sy / 2.0, cfg.size_y, cfg.resolution, cfg.ny)
    y1 = _world_to_index(cy + sy / 2.0, cfg.size_y, cfg.resolution, cfg.ny)
    occ[y0 : y1 + 1, x0 : x1 + 1] = False


def _add_wall(rects: list[dict], occ: np.ndarray, cfg: NavMazeCfg, name: str, cx: float, cy: float, sx: float, sy: float) -> None:
    rects.append({"name": name, "center": [round(cx, 3), round(cy, 3)], "size": [round(sx, 3), round(sy, 3)]})
    _mark_rect(occ, cfg, cx, cy, sx, sy)


def _add_segmented_vertical_wall(
    rects: list[dict],
    occ: np.ndarray,
    cfg: NavMazeCfg,
    name: str,
    x: float,
    y_min: float,
    y_max: float,
    doors: list[float],
) -> None:
    doors = sorted(float(v) for v in doors)
    cursor = y_min
    for door in doors:
        seg_end = max(cursor, door - cfg.door_width / 2.0)
        if seg_end - cursor >= cfg.min_corridor_width * 0.25:
            cy = 0.5 * (cursor + seg_end)
            _add_wall(rects, occ, cfg, f"{name}_{len(rects):03d}", x, cy, cfg.wall_thickness, seg_end - cursor)
        cursor = min(y_max, door + cfg.door_width / 2.0)
    if y_max - cursor >= cfg.min_corridor_width * 0.25:
        cy = 0.5 * (cursor + y_max)
        _add_wall(rects, occ, cfg, f"{name}_{len(rects):03d}", x, cy, cfg.wall_thickness, y_max - cursor)


def _add_segmented_horizontal_wall(
    rects: list[dict],
    occ: np.ndarray,
    cfg: NavMazeCfg,
    name: str,
    y: float,
    x_min: float,
    x_max: float,
    doors: list[float],
) -> None:
    doors = sorted(float(v) for v in doors)
    cursor = x_min
    for door in doors:
        seg_end = max(cursor, door - cfg.door_width / 2.0)
        if seg_end - cursor >= cfg.min_corridor_width * 0.25:
            cx = 0.5 * (cursor + seg_end)
            _add_wall(rects, occ, cfg, f"{name}_{len(rects):03d}", cx, y, seg_end - cursor, cfg.wall_thickness)
        cursor = min(x_max, door + cfg.door_width / 2.0)
    if x_max - cursor >= cfg.min_corridor_width * 0.25:
        cx = 0.5 * (cursor + x_max)
        _add_wall(rects, occ, cfg, f"{name}_{len(rects):03d}", cx, y, x_max - cursor, cfg.wall_thickness)


def _nearest_free(occ: np.ndarray, cfg: NavMazeCfg, x: float, y: float) -> list[float]:
    ix = _world_to_index(x, cfg.size_x, cfg.resolution, cfg.nx)
    iy = _world_to_index(y, cfg.size_y, cfg.resolution, cfg.ny)
    if not bool(occ[iy, ix]):
        return [round(x, 3), round(y, 3), 0.0]
    yy, xx = np.where(~occ)
    dist = (xx - ix) ** 2 + (yy - iy) ** 2
    best = int(np.argmin(dist))
    wx = float(xx[best] * cfg.resolution - cfg.size_x / 2.0)
    wy = float(yy[best] * cfg.resolution - cfg.size_y / 2.0)
    return [round(wx, 3), round(wy, 3), 0.0]


def _trim_rects_around_pads(rects: list[dict]) -> list[dict]:
    """Split wall rects so that no SDF box occupies any navigation pad area.

    _clear_rect only fixes the occupancy grid; this fixes the Gazebo geometry.
    Horizontal rects are split in x; vertical rects are split in y.
    Slivers narrower than wall_thickness are discarded.
    Boundary walls are never trimmed — pad positions can be near corners.
    """
    half_pad = _PAD_CLEAR_SIZE / 2.0
    min_keep = 0.22  # m — don't keep slivers thinner than a wall
    result = []
    for rect in rects:
        if rect["name"].startswith("boundary_"):
            result.append(rect)
            continue
        cx, cy = float(rect["center"][0]), float(rect["center"][1])
        sx, sy = float(rect["size"][0]), float(rect["size"][1])
        # Work with AABB segments: list of (x0, x1, y0, y1, name)
        segments = [(cx - sx / 2.0, cx + sx / 2.0, cy - sy / 2.0, cy + sy / 2.0, rect["name"])]
        for px, py in _PAD_POSITIONS:
            px0, px1 = px - half_pad, px + half_pad
            py0, py1 = py - half_pad, py + half_pad
            next_segs = []
            for x0, x1, y0, y1, name in segments:
                if x1 <= px0 or x0 >= px1 or y1 <= py0 or y0 >= py1:
                    next_segs.append((x0, x1, y0, y1, name))
                    continue
                # Overlapping: split along the longer axis to avoid small slivers.
                if (x1 - x0) >= (y1 - y0):  # horizontal — split in x
                    if x0 < px0:
                        next_segs.append((x0, px0, y0, y1, name + "_L"))
                    if x1 > px1:
                        next_segs.append((px1, x1, y0, y1, name + "_R"))
                else:  # vertical — split in y
                    if y0 < py0:
                        next_segs.append((x0, x1, y0, py0, name + "_B"))
                    if y1 > py1:
                        next_segs.append((x0, x1, py1, y1, name + "_T"))
            segments = next_segs
        for x0, x1, y0, y1, name in segments:
            w, h = x1 - x0, y1 - y0
            if w >= min_keep and h >= min_keep:
                result.append({
                    "name": name,
                    "center": [round((x0 + x1) / 2.0, 3), round((y0 + y1) / 2.0, 3)],
                    "size": [round(w, 3), round(h, 3)],
                })
    return result


def _build_maze_layout(cfg: NavMazeCfg, rng: np.random.Generator) -> tuple[np.ndarray, list[dict]]:
    """Lay out boundary/interior walls and obstacle blocks; return (occupancy, wall_rects)."""

    occ = np.zeros((cfg.ny, cfg.nx), dtype=bool)
    rects: list[dict] = []

    # Outer walls keep SLAM features inside lidar range and make map boundaries explicit.
    _add_wall(rects, occ, cfg, "boundary_north", 0.0, cfg.size_y / 2.0, cfg.size_x, cfg.wall_thickness)
    _add_wall(rects, occ, cfg, "boundary_south", 0.0, -cfg.size_y / 2.0, cfg.size_x, cfg.wall_thickness)
    _add_wall(rects, occ, cfg, "boundary_east", cfg.size_x / 2.0, 0.0, cfg.wall_thickness, cfg.size_y)
    _add_wall(rects, occ, cfg, "boundary_west", -cfg.size_x / 2.0, 0.0, cfg.wall_thickness, cfg.size_y)

    x_min = -cfg.size_x / 2.0 + cfg.wall_thickness
    x_max = cfg.size_x / 2.0 - cfg.wall_thickness
    y_min = -cfg.size_y / 2.0 + cfg.wall_thickness
    y_max = cfg.size_y / 2.0 - cfg.wall_thickness

    # Seeded segmented-wall layout: richer than a hand-placed demo map, but
    # still wide enough for the current skid-steer chassis. The center safety
    # pad is cleared after generation and then trimmed out of the Gazebo SDF.
    vwall_xs = (-cfg.size_x / 4.0, 0.0, cfg.size_x / 4.0)
    hwall_ys = (-cfg.size_y / 4.0, 0.0, cfg.size_y / 4.0)

    for x in vwall_xs:
        door_count = 3 if abs(x) > 1.0 else 2
        doors = list(rng.choice(np.linspace(y_min + 2.2, y_max - 2.2, 7), size=door_count, replace=False))
        doors.append(0.0)
        _add_segmented_vertical_wall(rects, occ, cfg, f"vwall_{x:g}", x, y_min, y_max, doors)

    for y in hwall_ys:
        door_count = 3 if abs(y) > 1.0 else 2
        doors = list(rng.choice(np.linspace(x_min + 2.5, x_max - 2.5, 8), size=door_count, replace=False))
        doors.append(0.0)
        _add_segmented_horizontal_wall(rects, occ, cfg, f"hwall_{y:g}", y, x_min, x_max, doors)

    # Block obstacles: avoid pad areas and any existing wall rect to prevent Gazebo
    # physics ejection (blocks spawning inside walls bounce upward at startup).
    for i in range(cfg.obstacle_count):
        for _ in range(30):
            bsx = float(rng.uniform(0.45, 0.9))
            bsy = float(rng.uniform(0.45, 0.9))
            cx = float(rng.uniform(x_min + 1.5, x_max - 1.5))
            cy = float(rng.uniform(y_min + 1.5, y_max - 1.5))
            half_pad = _PAD_CLEAR_SIZE / 2.0 + 0.5
            if any(abs(cx - px) < half_pad and abs(cy - py) < half_pad for px, py in _PAD_POSITIONS):
                continue
            if any(
                abs(cx - float(r["center"][0])) < (bsx + float(r["size"][0])) / 2.0
                and abs(cy - float(r["center"][1])) < (bsy + float(r["size"][1])) / 2.0
                for r in rects
            ):
                continue
            _add_wall(rects, occ, cfg, f"block_{i:02d}", cx, cy, bsx, bsy)
            break

    # Clear flat navigation pads at spawn and goal positions (occupancy only).
    for cx, cy in _PAD_POSITIONS:
        _clear_rect(occ, cfg, cx, cy, _PAD_CLEAR_SIZE, _PAD_CLEAR_SIZE)

    return occ, rects


def _build_cost_layers(
    height: np.ndarray, occ: np.ndarray, cfg: NavMazeCfg
) -> tuple[np.ndarray, np.ndarray, dict]:
    """Derive (traversability_cost, speed_mask, traversability_stats) from height+occupancy."""

    traversability_cost, traversability_stats = _compute_traversability_cost(height, occ, cfg)
    speed_mask = _compute_speed_mask(traversability_cost, occ, cfg)
    return traversability_cost, speed_mask, traversability_stats


def generate_nav_maze(cfg: NavMazeCfg, seed: int) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, list[dict], dict]:
    rng = np.random.default_rng(seed)
    occ, rects = _build_maze_layout(cfg, rng)

    height, height_metadata = generate_heightmap(cfg.terrain, seed)
    if height.shape != (cfg.ny, cfg.nx):
        raise ValueError(f"height layer shape {height.shape} does not match occupancy shape {(cfg.ny, cfg.nx)}")
    for cx, cy in _PAD_POSITIONS:
        _clear_rect(height, cfg, cx, cy, _PAD_CLEAR_SIZE, _PAD_CLEAR_SIZE)
    traversability_cost, speed_mask, traversability_stats = _build_cost_layers(height, occ, cfg)
    spawn_pad = _PAD_POSITIONS[0]
    metadata = {
        "generator_schema_version": GENERATOR_SCHEMA_VERSION,
        "preset": "nav_maze",
        "seed": seed,
        "size_x": cfg.size_x,
        "size_y": cfg.size_y,
        "resolution": cfg.resolution,
        "height_min": float(height.min()),
        "height_max": float(height.max()),
        "height_source": {
            "preset": height_metadata["preset"],
            "seed": seed,
            "height_limit": height_metadata["height_limit"],
            "labels": height_metadata["labels"],
        },
        "occupancy_shape": [cfg.ny, cfg.nx],
        "occupancy_convention": "False/free, True/occupied; map.pgm uses white/free and black/occupied",
        "wall_height": cfg.wall_height,
        "wall_thickness": cfg.wall_thickness,
        "door_width": cfg.door_width,
        "min_corridor_width": cfg.min_corridor_width,
        "obstacle_count": cfg.obstacle_count,
        "layering": {
            "height_layer": "height.npy",
            "occupancy_layer": "occupancy.npy",
            "traversability_cost_layer": "traversability_cost.npy",
            "speed_filter_mask_layer": "terrain_speed_mask.npy",
            "gazebo_world": "world.sdf",
            "gazebo_mesh_contact_world": "world_mesh_contact.sdf",
            "nav2_map": "map.yaml",
            "nav2_terrain_cost_map": "terrain_cost_map.yaml",
            "nav2_speed_filter_mask": "terrain_speed_mask.yaml",
        },
        "traversability": traversability_stats,
        "spawn": _nearest_free(occ, cfg, spawn_pad[0], spawn_pad[1]),
        "goals": [
            {"name": "north_east", "xyz": _nearest_free(occ, cfg, _PAD_POSITIONS[1][0], _PAD_POSITIONS[1][1])},
            {"name": "north_west", "xyz": _nearest_free(occ, cfg, _PAD_POSITIONS[2][0], _PAD_POSITIONS[2][1])},
            {"name": "south_east", "xyz": _nearest_free(occ, cfg, _PAD_POSITIONS[3][0], _PAD_POSITIONS[3][1])},
            {"name": "south_west", "xyz": _nearest_free(occ, cfg, _PAD_POSITIONS[4][0], _PAD_POSITIONS[4][1])},
        ],
        "wall_rects": rects,
    }
    return height, occ, traversability_cost, speed_mask, rects, metadata


def export_nav_maze(
    out_dir: Path,
    height: np.ndarray,
    occ: np.ndarray,
    traversability_cost: np.ndarray,
    speed_mask: np.ndarray,
    rects: list[dict],
    metadata: dict,
) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    export_height_assets(out_dir, height, metadata)
    np.save(out_dir / "occupancy.npy", occ.astype(np.bool_))
    np.save(out_dir / "traversability_cost.npy", traversability_cost.astype(np.uint8))
    np.save(out_dir / "terrain_speed_mask.npy", speed_mask.astype(np.uint8))
    write_pgm(out_dir / "map.pgm", occupancy_to_pgm_image(occ))
    write_pgm(out_dir / "terrain_cost_map.pgm", scaled_cost_to_pgm_image(traversability_cost))
    write_pgm(out_dir / "terrain_speed_mask.pgm", scaled_cost_to_pgm_image(speed_mask))
    write_map_yaml(out_dir / "map.yaml", "map.pgm", "trinary", metadata, occupied_thresh=0.65, free_thresh=0.25)
    write_map_yaml(
        out_dir / "terrain_cost_map.yaml", "terrain_cost_map.pgm", "scale", metadata,
        occupied_thresh=1.0, free_thresh=0.0,
    )
    write_map_yaml(
        out_dir / "terrain_speed_mask.yaml", "terrain_speed_mask.pgm", "scale", metadata,
        occupied_thresh=1.0, free_thresh=0.0,
    )
    obj_path = export_obj(out_dir, height, float(metadata["resolution"]))
    terrain_sdf = export_terrain_sdf(out_dir, obj_path)
    # Trim wall rects around pad areas before exporting to SDF — _clear_rect fixes the
    # occupancy grid, but without trimming the SDF still has boxes over pad positions.
    sdf_rects = _trim_rects_around_pads(rects)
    export_navigation_world_sdf(
        out_dir,
        sdf_rects,
        wall_height=float(metadata["wall_height"]),
        terrain_visual_obj=obj_path,
    )
    export_navigation_mesh_contact_world_sdf(
        out_dir,
        terrain_sdf,
        sdf_rects,
        wall_height=float(metadata["wall_height"]),
    )
    return out_dir


def generate(seed: int, output_root: Path, cfg: NavMazeCfg | None = None) -> Path:
    cfg = cfg or NavMazeCfg()
    out_dir = output_root / "nav_maze" / str(seed)
    height, occ, traversability_cost, speed_mask, rects, metadata = generate_nav_maze(cfg, seed)
    return export_nav_maze(out_dir, height, occ, traversability_cost, speed_mask, rects, metadata)


def main() -> None:
    defaults = NavMazeCfg()
    parser = argparse.ArgumentParser(description="Generate aligned Nav2 occupancy map and Gazebo maze world.")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output-root", default="generated/terrains")
    parser.add_argument("--size-x", type=float, default=defaults.size_x)
    parser.add_argument("--size-y", type=float, default=defaults.size_y)
    parser.add_argument("--resolution", type=float, default=defaults.resolution)
    parser.add_argument("--wall-height", type=float, default=defaults.wall_height)
    parser.add_argument("--wall-thickness", type=float, default=defaults.wall_thickness)
    parser.add_argument("--door-width", type=float, default=defaults.door_width)
    parser.add_argument("--min-corridor-width", type=float, default=defaults.min_corridor_width)
    parser.add_argument("--obstacle-count", type=int, default=defaults.obstacle_count)
    args = parser.parse_args()
    cfg = NavMazeCfg(
        terrain=replace(RL_CURRICULUM, size_x=args.size_x, size_y=args.size_y, resolution=args.resolution),
        wall_height=args.wall_height,
        wall_thickness=args.wall_thickness,
        door_width=args.door_width,
        min_corridor_width=args.min_corridor_width,
        obstacle_count=args.obstacle_count,
    )
    out_dir = generate(args.seed, Path(args.output_root), cfg)
    print(out_dir)


if __name__ == "__main__":
    main()
