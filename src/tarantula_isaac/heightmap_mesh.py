"""Pure-Python helpers for generated Tarantula heightmaps."""

from __future__ import annotations

import numpy as np

from tarantula_control.vehicle_geometry import VEHICLE_GEOMETRY


def heightmap_to_trimesh(height: np.ndarray, resolution: float):
    """Convert meters-valued Tarantula heightmap to centered trimesh."""
    import trimesh

    height_xy = np.asarray(height, dtype=np.float32).T
    num_rows, num_cols = height_xy.shape
    y = np.linspace(0, (num_cols - 1) * resolution, num_cols)
    x = np.linspace(0, (num_rows - 1) * resolution, num_rows)
    yy, xx = np.meshgrid(y, x)

    vertices = np.zeros((num_rows * num_cols, 3), dtype=np.float32)
    vertices[:, 0] = xx.flatten() - (num_rows - 1) * resolution / 2.0
    vertices[:, 1] = yy.flatten() - (num_cols - 1) * resolution / 2.0
    vertices[:, 2] = height_xy.flatten()

    triangles = np.empty((2 * (num_rows - 1) * (num_cols - 1), 3), dtype=np.uint32)
    for i in range(num_rows - 1):
        ind0 = np.arange(0, num_cols - 1, dtype=np.uint32) + i * num_cols
        ind1 = ind0 + 1
        ind2 = ind0 + num_cols
        ind3 = ind2 + 1
        start = 2 * i * (num_cols - 1)
        stop = start + 2 * (num_cols - 1)
        triangles[start:stop:2, 0] = ind0
        triangles[start:stop:2, 1] = ind3
        triangles[start:stop:2, 2] = ind1
        triangles[start + 1 : stop : 2, 0] = ind0
        triangles[start + 1 : stop : 2, 1] = ind2
        triangles[start + 1 : stop : 2, 2] = ind3

    return trimesh.Trimesh(vertices=vertices, faces=triangles, process=False)


def origins_from_metadata(
    metadata: dict,
    num_envs: int,
    spawn_z: float,
    spawn_xy_margin: float | None = None,
    min_level: int | None = None,
    max_level: int | None = None,
) -> np.ndarray:
    """Return Isaac Lab terrain origins from generated terrain metadata."""
    origins = metadata.get("env_origins") or []
    if origins:
        source_rows = int(metadata["num_rows"])
        cols = int(metadata["num_cols"])
        lo = 0 if min_level is None else max(0, int(min_level))
        hi = source_rows - 1 if max_level is None else min(source_rows - 1, int(max_level))
        if lo > hi:
            raise ValueError(f"invalid terrain level range: min_level={lo}, max_level={hi}")
        selected_rows = hi - lo + 1
        origin_grid = np.zeros((selected_rows, cols, 3), dtype=np.float32)
        found = np.zeros((selected_rows, cols), dtype=bool)
        for origin in origins:
            row = int(origin["row"])
            if row < lo or row > hi:
                continue
            col = int(origin["col"])
            xyz = origin["xyz"]
            origin_grid[row - lo, col] = [float(xyz[0]), float(xyz[1]), float(xyz[2])]
            found[row - lo, col] = True
        if not bool(found.all()):
            missing = np.argwhere(~found)
            raise ValueError(f"metadata env_origins missing selected terrain cells: {missing.tolist()}")
        return origin_grid

    if min_level is not None or max_level is not None:
        raise ValueError("terrain level filtering requires metadata env_origins")

    size_x = float(metadata["size_x"])
    size_y = float(metadata["size_y"])
    margin = max(
        1.5 * VEHICLE_GEOMETRY.reference_length,
        float(metadata.get("spawn_clear_radius", 0.9)) + 0.75 * VEHICLE_GEOMETRY.reference_length,
    )
    if spawn_xy_margin is not None:
        margin = max(margin, float(spawn_xy_margin))
    cols = int(np.ceil(np.sqrt(num_envs)))
    rows = int(np.ceil(num_envs / cols))
    xs = np.linspace(-size_x / 2.0 + margin, size_x / 2.0 - margin, cols)
    ys = np.linspace(-size_y / 2.0 + margin, size_y / 2.0 - margin, rows)
    origin_grid = np.zeros((rows, cols, 3), dtype=np.float32)
    for row, y in enumerate(ys):
        for col, x in enumerate(xs):
            origin_grid[row, col] = [float(x), float(y), spawn_z]
    return origin_grid


def height_at_xy(height: np.ndarray, metadata: dict, x: float, y: float) -> float:
    """Sample heightmap height at world x/y using nearest-neighbor lookup."""
    resolution = float(metadata["resolution"])
    size_x = float(metadata["size_x"])
    size_y = float(metadata["size_y"])
    ix = int(round((x + size_x / 2.0) / resolution))
    iy = int(round((y + size_y / 2.0) / resolution))
    ix = int(np.clip(ix, 0, height.shape[1] - 1))
    iy = int(np.clip(iy, 0, height.shape[0] - 1))
    return float(height[iy, ix])


def lift_origins_to_heightmap(origins: np.ndarray, height: np.ndarray, metadata: dict, clearance: float) -> np.ndarray:
    """Set origin z to local terrain height plus clearance."""
    lifted = np.asarray(origins, dtype=np.float32).copy()
    flat = lifted.reshape(-1, 3)
    for origin in flat:
        origin[2] = height_at_xy(height, metadata, float(origin[0]), float(origin[1])) + clearance
    return lifted
