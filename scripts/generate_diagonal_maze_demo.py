#!/usr/bin/env python3
"""Lay a recursive-backtracker grid maze over the existing rl_curriculum/42
terrain, then clear out any wall that intrudes on a buffered diagonal
corridor from corner to corner.

Reuses the already-validated rl_curriculum/42 terrain.sdf as-is (no height
regeneration) -- only adds wall geometry on top, so the diagonal pursuit
dynamics already confirmed clean in gazebo_pursuit_eval.py are unaffected.
Maze density (cell size) and the corridor's own width are independent: pure
pursuit never enters the maze passages, only the explicitly cleared
corridor, so the maze can be as dense as desired purely for visual texture
in the resulting Gazebo/RViz scene without adding any traversal risk.

Algorithm is the standard randomized depth-first "recursive backtracker"
maze generator (a uniform-ish spanning tree over a grid of cells, walls kept
on every edge not used by the tree) -- chosen over independently-scattered
random rectangles because a spanning-tree maze is fully connected and every
wall sits flush on a shared grid line, instead of floating disconnected
segments.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from tarantula_terrain.exporters import export_navigation_mesh_contact_world_sdf
from tarantula_terrain.terrain_cfg import RL_CURRICULUM

SPAWN = (-10.0, -6.0)
GOAL = (10.0, 6.0)
# Delegated to RL_CURRICULUM rather than copied, so this can't silently drift
# from the terrain this script overlays walls onto (same convention as
# nav_maze.py's NavMazeCfg).
SIZE_X, SIZE_Y = RL_CURRICULUM.size_x, RL_CURRICULUM.size_y


def _corridor_clearance(cx: float, cy: float, sx: float, sy: float, half_width: float) -> bool:
    """True if rect (cx,cy,sx,sy) intrudes within half_width of the SPAWN-GOAL segment."""
    ax, ay = SPAWN
    bx, by = GOAL
    seg_len = float(np.hypot(bx - ax, by - ay))
    ux, uy = (bx - ax) / seg_len, (by - ay) / seg_len
    samples = max(int(seg_len / 0.1), 2)
    ts = np.linspace(0.0, seg_len, samples)
    xs = ax + ux * ts
    ys = ay + uy * ts
    expanded_x0, expanded_x1 = cx - sx / 2.0 - half_width, cx + sx / 2.0 + half_width
    expanded_y0, expanded_y1 = cy - sy / 2.0 - half_width, cy + sy / 2.0 + half_width
    inside = (xs >= expanded_x0) & (xs <= expanded_x1) & (ys >= expanded_y0) & (ys <= expanded_y1)
    return bool(inside.any())


def _recursive_backtracker(nx: int, ny: int, rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray]:
    """Carve a spanning-tree maze over an nx x ny grid of cells.

    Returns (vert_open, horiz_open): vert_open[k, j] is True if the wall
    between cell(k, j) and cell(k+1, j) is removed (a passage); horiz_open[i,
    k] is True if the wall between cell(i, k) and cell(i, k+1) is removed.
    """
    visited = np.zeros((nx, ny), dtype=bool)
    vert_open = np.zeros((max(nx - 1, 0), ny), dtype=bool)
    horiz_open = np.zeros((nx, max(ny - 1, 0)), dtype=bool)

    stack = [(0, 0)]
    visited[0, 0] = True
    while stack:
        i, j = stack[-1]
        neighbors = []
        if i > 0 and not visited[i - 1, j]:
            neighbors.append(("W", i - 1, j))
        if i < nx - 1 and not visited[i + 1, j]:
            neighbors.append(("E", i + 1, j))
        if j > 0 and not visited[i, j - 1]:
            neighbors.append(("S", i, j - 1))
        if j < ny - 1 and not visited[i, j + 1]:
            neighbors.append(("N", i, j + 1))
        if not neighbors:
            stack.pop()
            continue
        direction, ni, nj = neighbors[int(rng.integers(len(neighbors)))]
        visited[ni, nj] = True
        if direction == "E":
            vert_open[i, j] = True
        elif direction == "W":
            vert_open[ni, nj] = True
        elif direction == "N":
            horiz_open[i, j] = True
        else:  # "S"
            horiz_open[i, nj] = True
        stack.append((ni, nj))
    return vert_open, horiz_open


def _subdivide(cx: float, cy: float, sx: float, sy: float, max_chunk: float) -> list[tuple[float, float, float, float]]:
    """Split a wall rect into pieces along its long axis so corridor clearing
    can trim a wall flush to the buffer instead of dropping the whole maze-
    cell-sized segment whenever any part of it touches the buffer."""
    length, thickness, vertical = (sy, sx, True) if sy >= sx else (sx, sy, False)
    n = max(int(np.ceil(length / max_chunk)), 1)
    chunk = length / n
    start = (cy if vertical else cx) - length / 2.0
    pieces = []
    for k in range(n):
        center = start + (k + 0.5) * chunk
        pieces.append((cx, center, thickness, chunk) if vertical else (center, cy, chunk, thickness))
    return pieces


def build_wall_rects(seed: int, cell_size: float, wall_thickness: float, corridor_half_width: float) -> list[dict]:
    nx = int(round(SIZE_X / cell_size))
    ny = int(round(SIZE_Y / cell_size))
    rng = np.random.default_rng(seed)
    vert_open, horiz_open = _recursive_backtracker(nx, ny, rng)

    def cell_x(i: int) -> float:
        return -SIZE_X / 2.0 + (i + 0.5) * cell_size

    def cell_y(j: int) -> float:
        return -SIZE_Y / 2.0 + (j + 0.5) * cell_size

    rects: list[dict] = []
    for k in range(nx - 1):
        x = -SIZE_X / 2.0 + (k + 1) * cell_size
        for j in range(ny):
            if vert_open[k, j]:
                continue
            rects.append({"center": [x, cell_y(j)], "size": [wall_thickness, cell_size]})
    for i in range(nx):
        for k in range(ny - 1):
            if horiz_open[i, k]:
                continue
            y = -SIZE_Y / 2.0 + (k + 1) * cell_size
            rects.append({"center": [cell_x(i), y], "size": [cell_size, wall_thickness]})

    # Boundary walls, matching nav_maze's convention.
    rects.append({"center": [0.0, SIZE_Y / 2.0], "size": [SIZE_X, wall_thickness]})
    rects.append({"center": [0.0, -SIZE_Y / 2.0], "size": [SIZE_X, wall_thickness]})
    rects.append({"center": [SIZE_X / 2.0, 0.0], "size": [wall_thickness, SIZE_Y]})
    rects.append({"center": [-SIZE_X / 2.0, 0.0], "size": [wall_thickness, SIZE_Y]})

    kept = []
    chunk = min(wall_thickness * 2.0, 0.4)
    for rect in rects:
        cx, cy = rect["center"]
        sx, sy = rect["size"]
        if not _corridor_clearance(cx, cy, sx, sy, corridor_half_width):
            # Whole segment is already clear of the buffer -- no need to split it.
            kept.append({"name": f"maze_wall_{len(kept):03d}", "center": [round(cx, 3), round(cy, 3)],
                         "size": [round(sx, 3), round(sy, 3)]})
            continue
        for px, py, psx, psy in _subdivide(cx, cy, sx, sy, chunk):
            if _corridor_clearance(px, py, psx, psy, corridor_half_width):
                continue
            kept.append({"name": f"maze_wall_{len(kept):03d}", "center": [round(px, 3), round(py, 3)],
                         "size": [round(psx, 3), round(psy, 3)]})
    return kept


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--cell-size", type=float, default=2.0, help="Maze grid cell size in meters; must evenly divide 24x16.")
    parser.add_argument("--wall-thickness", type=float, default=0.22)
    parser.add_argument("--corridor-half-width", type=float, default=1.2)
    parser.add_argument("--terrain-sdf", default="generated/terrains/rl_curriculum/42/terrain.sdf")
    parser.add_argument("--out-dir", default="generated/terrains/diagonal_maze_demo/42")
    parser.add_argument("--wall-height", type=float, default=1.10)
    args = parser.parse_args()

    rects = build_wall_rects(args.seed, args.cell_size, args.wall_thickness, args.corridor_half_width)
    print(f"placed {len(rects)} wall segments, cell_size={args.cell_size}m, corridor half-width {args.corridor_half_width}m")

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    world_path = export_navigation_mesh_contact_world_sdf(
        out_dir,
        Path(args.terrain_sdf).resolve(),
        rects,
        wall_height=args.wall_height,
    )
    print(f"wrote {world_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
