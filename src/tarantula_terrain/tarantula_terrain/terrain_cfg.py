from dataclasses import dataclass


@dataclass(frozen=True)
class TerrainCfg:
    preset: str = "gazebo_demo"
    size_x: float = 18.0
    size_y: float = 12.0
    resolution: float = 0.08
    # No longer used to flatten anything in generator.py (that was
    # _clear_spawn, removed -- see generate_heightmap's comment). Still read
    # from metadata by heightmap_mesh.py's origins_from_metadata() as an
    # edge-margin hint for its no-env_origins fallback grid layout, which is
    # a separate, narrower use than the flattening this field used to drive.
    spawn_clear_radius: float = 0.75
    min_height: float = -0.08
    max_height: float = 0.10
    wall_height: float = 0.85
    wall_thickness: float = 0.18
    num_rows: int = 1
    num_cols: int = 1
    tile_size_x: float = 4.0
    tile_size_y: float = 4.0
    platform_size: float = 1.0
    # >0 tapers height to exactly 0 within this many meters of the array's
    # outer rectangle (see generator._taper_outer_edge). Only meaningful for
    # wall_height<=0 presets, where export_world_sdf's surround_copies
    # repeats this same array at adjacent tile offsets: without a flat (0)
    # edge on both sides of that repeat seam, two independently-generated
    # edges can disagree by several cm packed into one mesh cell. 0.0 is a
    # no-op (walled presets have a physical boundary already, no seam to
    # protect).
    edge_taper_band: float = 0.0

    @property
    def nx(self) -> int:
        return int(round(self.size_x / self.resolution)) + 1

    @property
    def ny(self) -> int:
        return int(round(self.size_y / self.resolution)) + 1


GAZEBO_DEMO = TerrainCfg(
    preset="gazebo_demo",
    max_height=0.14,
    wall_height=0.85,
    wall_thickness=0.18,
)
RL_CURRICULUM = TerrainCfg(
    preset="rl_curriculum",
    size_x=24.0,
    size_y=16.0,
    resolution=0.10,
    spawn_clear_radius=0.90,
    # Widened from (-0.08, 0.18) -- the old curriculum's own difficulty=1.0
    # amplitudes (see _apply_curriculum_tile) only reached +0.14/-0.08, under
    # 0.25m of relief across a 24x16m map: visually flat from a top-down
    # camera despite being labeled "difficulty 1.0". New formulas push the
    # steepest tiles (slope/stairs) toward 0.30m -- about 2x WHEEL_RADIUS
    # (0.13m), genuinely challenging for the active suspension rather than
    # a difficulty label with no visible difference from "easy".
    min_height=-0.25,
    max_height=0.35,
    wall_height=0.0,
    wall_thickness=0.0,
    num_rows=4,
    num_cols=6,
    tile_size_x=4.0,
    tile_size_y=4.0,
    platform_size=1.0,
    # Comfortably inside bounds_margin (~0.68m, see
    # TarantulaSuspensionEnvCfg.terminations.bounds_margin /
    # gazebo_pursuit_eval.py's DEFAULT_MARGIN) so the flattened ring sits
    # entirely in the zone that's already out-of-bounds for training/eval --
    # no legitimate operating area loses terrain complexity.
    edge_taper_band=0.6,
)


PRESETS = {
    "gazebo_demo": GAZEBO_DEMO,
    "rl_curriculum": RL_CURRICULUM,
}
