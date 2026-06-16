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
    min_height=-0.08,
    max_height=0.18,
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
