import math

import numpy as np

from .exporters import GENERATOR_SCHEMA_VERSION
from .terrain_cfg import TerrainCfg


def _grid(cfg: TerrainCfg):
    x = np.linspace(-cfg.size_x / 2.0, cfg.size_x / 2.0, cfg.nx, dtype=np.float32)
    y = np.linspace(-cfg.size_y / 2.0, cfg.size_y / 2.0, cfg.ny, dtype=np.float32)
    return np.meshgrid(x, y)


def _mask_rect(x, y, cx, cy, sx, sy):
    return (np.abs(x - cx) <= sx / 2.0) & (np.abs(y - cy) <= sy / 2.0)


def _add_wave(height, x, y, cx, cy, sx, sy, amp, wavelength, yaw=0.0):
    c = math.cos(yaw)
    s = math.sin(yaw)
    xr = c * (x - cx) + s * (y - cy)
    mask = _mask_rect(x, y, cx, cy, sx, sy)
    height[mask] += amp * (0.5 + 0.5 * np.sin(2.0 * math.pi * xr[mask] / wavelength))


def _add_slope(height, x, y, cx, cy, sx, sy, dz, axis="x"):
    mask = _mask_rect(x, y, cx, cy, sx, sy)
    coord = x if axis == "x" else y
    c0 = cx if axis == "x" else cy
    span = sx if axis == "x" else sy
    local = np.clip((coord - c0) / max(span, 1e-6), -0.5, 0.5)
    height[mask] += dz * (local[mask] + 0.5)


def _add_steps(height, x, y, cx, cy, sx, sy, step_h, n_steps, axis="x"):
    mask = _mask_rect(x, y, cx, cy, sx, sy)
    coord = x if axis == "x" else y
    c0 = cx if axis == "x" else cy
    span = sx if axis == "x" else sy
    u = np.clip((coord - (c0 - span / 2.0)) / max(span, 1e-6), 0.0, 0.999)
    levels = np.floor(u * n_steps) / max(n_steps - 1, 1)
    height[mask] += step_h * levels[mask]


def _add_random_blocks(height, x, y, rng, cx, cy, sx, sy, count, h_min, h_max, size_min, size_max):
    for _ in range(count):
        bx = rng.uniform(cx - sx / 2.0, cx + sx / 2.0)
        by = rng.uniform(cy - sy / 2.0, cy + sy / 2.0)
        bsx = rng.uniform(size_min, size_max)
        bsy = rng.uniform(size_min, size_max)
        bh = rng.uniform(h_min, h_max)
        mask = _mask_rect(x, y, bx, by, bsx, bsy)
        height[mask] = np.maximum(height[mask], bh)


def _add_pit_or_gap(height, x, y, cx, cy, sx, sy, depth):
    mask = _mask_rect(x, y, cx, cy, sx, sy)
    height[mask] -= depth


def _add_uniform_roughness(height, x, y, rng, cx, cy, sx, sy, amp, cell_size):
    mask = _mask_rect(x, y, cx, cy, sx, sy)
    if not np.any(mask):
        return
    ix = np.floor((x - (cx - sx / 2.0)) / max(cell_size, 1e-6)).astype(np.int32)
    iy = np.floor((y - (cy - sy / 2.0)) / max(cell_size, 1e-6)).astype(np.int32)
    table_shape = (max(int(math.ceil(sy / cell_size)) + 2, 2), max(int(math.ceil(sx / cell_size)) + 2, 2))
    table = rng.uniform(-amp, amp, table_shape).astype(np.float32)
    height[mask] += table[iy[mask].clip(0, table_shape[0] - 1), ix[mask].clip(0, table_shape[1] - 1)]


def _add_smooth_noise(height, x, y, rng, amp, waves):
    for _ in range(waves):
        yaw = rng.uniform(-math.pi, math.pi)
        wavelength = rng.uniform(1.2, 4.0)
        phase = rng.uniform(-math.pi, math.pi)
        c = math.cos(yaw)
        s = math.sin(yaw)
        height += (amp / waves) * np.sin(2.0 * math.pi * (c * x + s * y) / wavelength + phase)


def _soften_edges(height, passes=2):
    # Edge-replicate padding avoids np.roll wrap-around contamination at borders.
    smoothed = height.copy()
    for _ in range(passes):
        pad = np.pad(smoothed, 1, mode="edge")
        smoothed = (
            pad[1:-1, 1:-1]
            + pad[0:-2, 1:-1]
            + pad[2:,   1:-1]
            + pad[1:-1, 0:-2]
            + pad[1:-1, 2:]
        ) / 5.0
    return smoothed


def _taper_outer_edge(height, x, y, size_x, size_y, band):
    """Smoothstep height to 0 within `band` meters of the heightmap's outer
    rectangle.

    export_world_sdf's surround_copies repeats this exact array at adjacent
    tile offsets so an unwalled preset has ground to drive onto past the
    nominal boundary instead of a void. That only tiles seamlessly if the
    array's own outer edge is flat: _soften_edges's edge-replicate padding
    fixes a different problem (the *interior* curriculum-tile seams) and
    explicitly does not wrap/match opposite borders, so two
    independently-generated edges meeting at a repeat seam can disagree by
    several cm packed into one mesh cell (measured up to ~6cm on
    rl_curriculum/42) -- enough for a 0.13m-radius wheel to catch on, the
    same near-vertical-wall wedging failure _generate_rl_curriculum's own
    edge-smoothing already describes, just at the outer seam instead of an
    inner tile boundary.
    """
    if band <= 0.0:
        return
    dist = np.minimum(size_x / 2.0 - np.abs(x), size_y / 2.0 - np.abs(y))
    t = np.clip(dist / band, 0.0, 1.0)
    height *= t * t * (3.0 - 2.0 * t)


def _clear_platform(height, x, y, cx, cy, size):
    mask = _mask_rect(x, y, cx, cy, size, size)
    height[mask] = 0.0


def _add_global_micro_relief(height, rng, amp):
    """Per-vertex i.i.d. jitter over the whole array, unconditionally --
    unlike _add_uniform_roughness's coarser per-cell lookup table (which
    still leaves many adjacent mesh vertices at the same table value, hence
    coplanar), this has no spatial correlation between neighbors at all, so
    no two adjacent mesh triangles can land on exactly the same height.

    Exists because every _clear_platform spawn square and the
    edge_taper_band ring are literal, exactly-flat constants -- and those
    are precisely where this project has observed skid-steer wheels skate
    with ~0 yaw torque on mesh collision, while genuinely uneven mesh
    regions transmit yaw torque fine (see motion_control.py's
    MotionControlConfig docstring). Amplitude is small (a fraction of the
    smallest deliberate terrain feature amplitude in _apply_curriculum_tile)
    so it reads as "still flat" rather than a new obstacle.

    amp=0.004 was observed once to not be enough for a cold-start in-place
    turn right at a _clear_platform spawn square (zero initial velocity,
    so the wheels must break static grip from rest rather than just
    maintain an already-turning rotation), failing the same way the
    pre-fix literal-flat case did. Tried amp=0.010 instead (5-9x closer to
    terrain types that reliably transmit yaw torque elsewhere, e.g.
    random_uniform difficulty=1.0 std~0.031) -- that fixed the cold-start
    case but then broke a previously-clean pursuit-eval run (seed=42 on
    the same checkpoint, previously 11/20 checkpoints/112.3m/0% tilt) that
    had never failed at amp=0.004, so reverted back to 0.004. Net: this
    amplitude is a real, currently-unresolved tradeoff, not a solved
    constant -- raising it can fix one symptom (cold-start grip) while
    introducing a different one elsewhere (exact mechanism not yet
    understood; possibly interacts with how curriculum-tile boundary
    seams get smoothed by _soften_edges, see _generate_rl_curriculum's
    docstring on tile-seam cliffs). Don't bump it again without re-running
    a known-good seed (42) to confirm nothing else regresses.

    Called before _taper_outer_edge, not after: taper's multiplicative
    smoothstep (height *= t*t*(3-2t)) scales this jitter back to exactly 0
    right at the array's outer boundary cells too, the same cells
    export_world_sdf's surround_copies tiling butts adjacent copies
    against -- so the literal seam stays exactly flat/matching on both
    sides, only the interior of the taper band (never a tiling seam itself)
    gains texture.
    """
    height += rng.uniform(-amp, amp, height.shape).astype(np.float32)


def _generate_gazebo_demo(cfg: TerrainCfg, rng, x, y):
    height = np.zeros_like(x, dtype=np.float32)
    labels = []
    origins = []

    _add_smooth_noise(height, x, y, rng, amp=0.030, waves=12)
    _add_uniform_roughness(height, x, y, rng, 0.0, 0.0, cfg.size_x - 1.2, cfg.size_y - 1.2, 0.022, 0.42)
    labels.append({"type": "full_field_roughness", "center": [0.0, 0.0], "size": [cfg.size_x - 1.2, cfg.size_y - 1.2]})

    _add_wave(height, x, y, -5.4, 2.9, 4.8, 2.8, 0.045, 0.62, yaw=-0.18)
    _add_wave(height, x, y, 0.0, -3.0, 5.2, 2.6, 0.050, 0.55, yaw=0.35)
    _add_wave(height, x, y, 4.7, 2.6, 4.6, 2.9, 0.040, 0.70, yaw=0.12)
    labels.append({"type": "adjacent_washboard_fields", "center": [-0.2, 0.8], "size": [14.8, 5.8]})

    _add_slope(height, x, y, -3.7, -1.3, 4.8, 3.0, 0.080, axis="x")
    _add_slope(height, x, y, 2.9, 1.0, 4.6, 3.2, -0.075, axis="y")
    _add_slope(height, x, y, 6.0, -3.2, 3.6, 2.4, 0.060, axis="x")
    labels.append({"type": "interleaved_slopes", "center": [1.7, -1.2], "size": [13.5, 6.4]})

    _add_steps(height, x, y, -5.8, -3.1, 3.5, 2.2, 0.045, 5, axis="x")
    _add_steps(height, x, y, -0.9, 3.3, 3.4, 2.0, 0.040, 5, axis="y")
    labels.append({"type": "low_step_fields", "center": [-3.3, 0.1], "size": [8.6, 6.5]})

    _add_random_blocks(height, x, y, rng, 2.4, 0.0, 6.4, 3.6, 120, 0.012, 0.075, 0.10, 0.32)
    _add_random_blocks(height, x, y, rng, -4.2, 0.4, 4.6, 3.8, 86, 0.012, 0.060, 0.10, 0.28)
    labels.append({"type": "dense_embedded_rocks", "center": [-0.6, 0.2], "size": [12.8, 4.4]})

    _add_pit_or_gap(height, x, y, -6.4, 0.0, 0.42, 3.1, 0.045)
    _add_pit_or_gap(height, x, y, 0.8, -1.1, 2.2, 0.36, 0.040)
    _add_pit_or_gap(height, x, y, 5.8, 0.6, 0.38, 2.7, 0.040)
    labels.append({"type": "shallow_heightfield_trenches", "center": [0.0, -0.2], "size": [12.6, 3.4]})

    height = _soften_edges(height, passes=1)
    return height, labels, origins


def _tile_center(cfg, row, col):
    x0 = -cfg.size_x / 2.0 + cfg.tile_size_x / 2.0
    y0 = -cfg.size_y / 2.0 + cfg.tile_size_y / 2.0
    return x0 + col * cfg.tile_size_x, y0 + row * cfg.tile_size_y


def _apply_curriculum_tile(height, x, y, rng, cfg, row, col):
    cx, cy = _tile_center(cfg, row, col)
    difficulty = 0.0 if cfg.num_rows <= 1 else row / (cfg.num_rows - 1)
    terrain_type = col % 6
    sx = cfg.tile_size_x - 0.35
    sy = cfg.tile_size_y - 0.35
    label = "flat"

    # Amplitudes roughly doubled from the original curriculum: at
    # difficulty=1.0 the old formulas topped out under 0.15m -- visually
    # flat from a top-down camera on a 24x16m map despite being labeled the
    # hardest tier. Steepest tiles (slope/stairs) now reach ~0.30m, about 2x
    # WHEEL_RADIUS (0.13m): genuinely visible relief, not just a difficulty
    # number with no corresponding height difference.
    if terrain_type == 0:
        _add_uniform_roughness(height, x, y, rng, cx, cy, sx, sy, 0.015 + 0.075 * difficulty, 0.35)
        label = "random_uniform"
    elif terrain_type == 1:
        _add_wave(height, x, y, cx, cy, sx, sy, 0.030 + 0.110 * difficulty, 0.85 - 0.25 * difficulty, yaw=0.0)
        label = "wave"
    elif terrain_type == 2:
        _add_slope(height, x, y, cx, cy, sx, sy, 0.060 + 0.240 * difficulty, axis="x")
        label = "pyramid_slope_proxy"
    elif terrain_type == 3:
        _add_steps(height, x, y, cx, cy, sx, sy, 0.050 + 0.200 * difficulty, 4 + row, axis="x")
        label = "stairs"
    elif terrain_type == 4:
        _add_random_blocks(
            height,
            x,
            y,
            rng,
            cx,
            cy,
            sx,
            sy,
            18 + row * 8,
            0.020,
            0.060 + 0.180 * difficulty,
            0.14,
            0.42,
        )
        label = "discrete_obstacles"
    else:
        if difficulty < 0.5:
            _add_pit_or_gap(height, x, y, cx, cy, 0.25 + 0.25 * difficulty, sy * 0.7, 0.040 + 0.090 * difficulty)
            label = "gap"
        else:
            _add_pit_or_gap(height, x, y, cx, cy, sx * 0.55, sy * 0.55, 0.090 + 0.110 * difficulty)
            label = "pit"

    _clear_platform(height, x, y, cx, cy, cfg.platform_size)
    return {
        "type": label,
        "row": row,
        "col": col,
        "difficulty": round(float(difficulty), 3),
        "center": [round(float(cx), 3), round(float(cy), 3)],
        "size": [round(float(sx), 3), round(float(sy), 3)],
    }


def _generate_rl_curriculum(cfg: TerrainCfg, rng, x, y):
    height = np.zeros_like(x, dtype=np.float32)
    labels = []
    origins = []
    for row in range(cfg.num_rows):
        for col in range(cfg.num_cols):
            label = _apply_curriculum_tile(height, x, y, rng, cfg, row, col)
            labels.append(label)
            origins.append(
                {
                    "row": row,
                    "col": col,
                    "difficulty": label["difficulty"],
                    "terrain_type": label["type"],
                    "xyz": [label["center"][0], label["center"][1], 0.20],
                }
            )
    # Each tile's feature (_add_slope/_add_wave/etc.) only touches height
    # inside its own rectangular mask; nothing tapers it to 0 at that mask's
    # edge, so wherever a feature doesn't happen to reach 0 right at the
    # boundary, the unsmoothed array has a literal near-vertical wall there
    # (observed: 0.113m of rise collapsing to 0 within one 0.1m grid cell --
    # not an intended difficulty step, a generation artifact a 0.13m-radius
    # wheel can get physically wedged against). _generate_gazebo_demo already
    # does this; curriculum tiles have larger feature amplitudes at high
    # difficulty so use more passes -- bumped 3->8 alongside the 2025-06-20
    # amplitude increase (slope/stairs maxing near 0.30m now vs 0.15m) to
    # keep the same proportional smoothing strength at the larger scale.
    height = _soften_edges(height, passes=8)
    return height, labels, origins


def generate_heightmap(cfg: TerrainCfg, seed: int):
    rng = np.random.default_rng(seed)
    x, y = _grid(cfg)

    if cfg.preset == "rl_curriculum":
        height, labels, origins = _generate_rl_curriculum(cfg, rng, x, y)
    elif cfg.preset == "gazebo_demo":
        height, labels, origins = _generate_gazebo_demo(cfg, rng, x, y)
    else:
        raise ValueError(f"unsupported terrain preset: {cfg.preset}")

    # No global-origin flattening here anymore (removed _clear_spawn): the
    # world coordinate origin sits at a corner where up to 4 curriculum tiles
    # meet (rl_curriculum's grid is centered on it), each potentially a
    # different terrain_type/difficulty -- smoothstep-blending that junction
    # down to 0 still left a 20-30deg local slope ring no radius/feather
    # choice fully fixed (push the boundary out and it just lands on another
    # tile-boundary discontinuity instead). The actual fix is spawning on a
    # difficulty-0 tile's own _clear_platform square instead (see
    # sim.launch.py's spawn_x/spawn_y, set to a row=0 tile center) -- that's
    # a real flat platform with no neighboring-tile seam, not a blend.
    _add_global_micro_relief(height, rng, amp=0.004)
    _taper_outer_edge(height, x, y, cfg.size_x, cfg.size_y, cfg.edge_taper_band)
    height = np.clip(height, cfg.min_height, cfg.max_height).astype(np.float32)

    metadata = {
        "generator_schema_version": GENERATOR_SCHEMA_VERSION,
        "preset": cfg.preset,
        "seed": seed,
        "size_x": cfg.size_x,
        "size_y": cfg.size_y,
        "resolution": cfg.resolution,
        "spawn_clear_radius": cfg.spawn_clear_radius,
        "height_limit": [cfg.min_height, cfg.max_height],
        "wall_height": cfg.wall_height,
        "wall_thickness": cfg.wall_thickness,
        "height_min": float(height.min()),
        "height_max": float(height.max()),
        "num_rows": cfg.num_rows,
        "num_cols": cfg.num_cols,
        "tile_size": [cfg.tile_size_x, cfg.tile_size_y],
        "platform_size": cfg.platform_size,
        "labels": labels,
        "env_origins": origins,
    }
    return height, metadata
