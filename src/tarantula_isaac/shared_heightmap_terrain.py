"""Isaac Lab terrain importer for Tarantula generated heightmaps."""

from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np

import isaaclab.sim as sim_utils
from isaaclab.terrains import TerrainImporter, TerrainImporterCfg
from isaaclab.utils import configclass

from .heightmap_mesh import heightmap_to_trimesh, lift_origins_to_heightmap, origins_from_metadata


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_TERRAIN_DIR = Path(
    os.environ.get("TARANTULA_TERRAIN_DIR", REPO_ROOT / "generated" / "terrains" / "gazebo_demo" / "42")
)


@configclass
class SharedHeightmapTerrainImporterCfg(TerrainImporterCfg):
    """Config for importing Tarantula generated heightmaps into Isaac Lab."""

    class_type: type = None
    terrain_type: str = "heightmap"
    height_path: str = str(DEFAULT_TERRAIN_DIR / "height.npy")
    metadata_path: str = str(DEFAULT_TERRAIN_DIR / "metadata.json")
    spawn_z: float = 0.20
    spawn_xy_margin: float = 5.0
    min_level: int | None = None
    max_level: int | None = None
    plane_size: tuple[float, float] = (100.0, 100.0)


class SharedHeightmapTerrainImporter(TerrainImporter):
    """TerrainImporter that reads the same heightmap used by Gazebo."""

    cfg: SharedHeightmapTerrainImporterCfg

    def __init__(self, cfg: SharedHeightmapTerrainImporterCfg):
        cfg.validate()
        self.cfg = cfg
        self.device = sim_utils.SimulationContext.instance().device  # type: ignore[union-attr]
        self.terrain_prim_paths = []
        self.terrain_origins = None
        self.env_origins = None
        self._terrain_flat_patches = {}

        if cfg.terrain_type == "plane":
            self.import_ground_plane("terrain", size=cfg.plane_size)
            half_x = float(cfg.plane_size[0]) / 2.0
            half_y = float(cfg.plane_size[1]) / 2.0
            self.terrain_bounds = (-half_x, half_x, -half_y, half_y)
            self.height_range = (0.0, 0.0)
            origins = np.zeros((cfg.num_envs, 3), dtype=np.float32)
            origins[:, 2] = cfg.spawn_z
        elif cfg.terrain_type == "heightmap":
            height_path = Path(cfg.height_path)
            metadata_path = Path(cfg.metadata_path)
            height = np.load(height_path)
            metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
            resolution = float(metadata["resolution"])
            self.terrain_bounds = (
                -float(metadata["size_x"]) / 2.0,
                float(metadata["size_x"]) / 2.0,
                -float(metadata["size_y"]) / 2.0,
                float(metadata["size_y"]) / 2.0,
            )
            self.height_range = (float(metadata["height_min"]), float(metadata["height_max"]))

            mesh = heightmap_to_trimesh(height, resolution)
            self.import_mesh("terrain", mesh)

            origins = origins_from_metadata(
                metadata,
                cfg.num_envs,
                cfg.spawn_z,
                cfg.spawn_xy_margin,
                min_level=cfg.min_level,
                max_level=cfg.max_level,
            )
            origins = lift_origins_to_heightmap(origins, height, metadata, cfg.spawn_z)
        else:
            raise ValueError(f"unsupported terrain_type={cfg.terrain_type!r}; expected 'plane' or 'heightmap'")
        self.configure_env_origins(origins)
        self.set_debug_vis(cfg.debug_vis)


SharedHeightmapTerrainImporterCfg.class_type = SharedHeightmapTerrainImporter


def make_shared_heightmap_terrain_cfg(
    terrain_dir: str | Path = DEFAULT_TERRAIN_DIR,
    *,
    prim_path: str = "/World/ground",
    debug_vis: bool = False,
    min_level: int | None = None,
    max_level: int | None = None,
    terrain_type: str = "heightmap",
) -> SharedHeightmapTerrainImporterCfg:
    terrain_dir = Path(terrain_dir)
    cfg = SharedHeightmapTerrainImporterCfg(
        prim_path=prim_path,
        terrain_type=terrain_type,
        height_path=str(terrain_dir / "height.npy"),
        metadata_path=str(terrain_dir / "metadata.json"),
        # Only bounds Isaac Lab's torch.randint() for each env's INITIAL
        # origin assignment at terrain construction -- TarantulaSuspensionEnv
        # ._reset_idx re-rolls a fresh random (row, col) from min_level/
        # max_level's full filtered range on every reset (including the
        # very first one), so this initial assignment never survives past
        # env construction. None lets Isaac Lab default it to the actual
        # row count rather than a hand-picked number that doesn't matter.
        max_init_terrain_level=None,
        collision_group=-1,
        physics_material=sim_utils.RigidBodyMaterialCfg(
            friction_combine_mode="multiply",
            restitution_combine_mode="multiply",
            static_friction=1.0,
            dynamic_friction=1.0,
        ),
        debug_vis=debug_vis,
        min_level=min_level,
        max_level=max_level,
    )
    cfg.class_type = SharedHeightmapTerrainImporter
    return cfg
