from dataclasses import dataclass


@dataclass(frozen=True)
class TerrainCfg:
    preset: str = "gazebo_demo"
    size_x: float = 18.0
    size_y: float = 12.0
    resolution: float = 0.08
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
)


PRESETS = {
    "gazebo_demo": GAZEBO_DEMO,
    "rl_curriculum": RL_CURRICULUM,
}
