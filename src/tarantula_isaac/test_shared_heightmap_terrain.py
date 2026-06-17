import json
import unittest
from pathlib import Path

import numpy as np

from tarantula_isaac.heightmap_mesh import heightmap_to_trimesh, height_at_xy, lift_origins_to_heightmap, origins_from_metadata


class SharedHeightmapTerrainTest(unittest.TestCase):
    def test_heightmap_to_centered_mesh(self):
        terrain_dir = Path("generated/terrains/gazebo_demo/42")
        height = np.load(terrain_dir / "height.npy")
        metadata = json.loads((terrain_dir / "metadata.json").read_text(encoding="utf-8"))

        mesh = heightmap_to_trimesh(height, float(metadata["resolution"]))

        self.assertEqual(len(mesh.vertices), height.size)
        self.assertEqual(len(mesh.faces), 2 * (height.shape[0] - 1) * (height.shape[1] - 1))
        self.assertAlmostEqual(float(mesh.vertices[:, 0].min()), -metadata["size_x"] / 2.0, places=4)
        self.assertAlmostEqual(float(mesh.vertices[:, 0].max()), metadata["size_x"] / 2.0, places=4)
        self.assertAlmostEqual(float(mesh.vertices[:, 1].min()), -metadata["size_y"] / 2.0, places=4)
        self.assertAlmostEqual(float(mesh.vertices[:, 1].max()), metadata["size_y"] / 2.0, places=4)

    def test_curriculum_origins_use_metadata(self):
        metadata = json.loads(Path("generated/terrains/rl_curriculum/42/metadata.json").read_text(encoding="utf-8"))
        origins = origins_from_metadata(metadata, num_envs=16, spawn_z=0.2)

        self.assertEqual(origins.shape, (4, 6, 3))
        self.assertAlmostEqual(float(origins[0, 0, 0]), -10.0)
        self.assertAlmostEqual(float(origins[0, 0, 1]), -6.0)
        self.assertAlmostEqual(float(origins[0, 0, 2]), 0.2)

    def test_curriculum_origins_can_filter_levels(self):
        metadata = json.loads(Path("generated/terrains/rl_curriculum/42/metadata.json").read_text(encoding="utf-8"))
        origins = origins_from_metadata(metadata, num_envs=16, spawn_z=0.2, min_level=1, max_level=2)

        self.assertEqual(origins.shape, (2, 6, 3))
        self.assertAlmostEqual(float(origins[0, 0, 0]), -10.0)
        self.assertAlmostEqual(float(origins[0, 0, 1]), -2.0)
        self.assertAlmostEqual(float(origins[1, 0, 1]), 2.0)

    def test_curriculum_origin_level_range_must_be_valid(self):
        metadata = json.loads(Path("generated/terrains/rl_curriculum/42/metadata.json").read_text(encoding="utf-8"))

        with self.assertRaises(ValueError):
            origins_from_metadata(metadata, num_envs=16, spawn_z=0.2, min_level=3, max_level=1)

    def test_demo_origins_are_spread_on_map(self):
        metadata = json.loads(Path("generated/terrains/gazebo_demo/42/metadata.json").read_text(encoding="utf-8"))
        origins = origins_from_metadata(metadata, num_envs=16, spawn_z=0.2)

        self.assertEqual(origins.shape, (4, 4, 3))
        self.assertGreater(float(origins[:, :, 0].max() - origins[:, :, 0].min()), 8.0)
        self.assertGreater(float(origins[:, :, 1].max() - origins[:, :, 1].min()), 5.0)

    def test_level_filter_requires_curriculum_origins(self):
        metadata = json.loads(Path("generated/terrains/gazebo_demo/42/metadata.json").read_text(encoding="utf-8"))

        with self.assertRaises(ValueError):
            origins_from_metadata(metadata, num_envs=16, spawn_z=0.2, min_level=0, max_level=1)

    def test_origins_are_lifted_above_local_height(self):
        terrain_dir = Path("generated/terrains/gazebo_demo/42")
        height = np.load(terrain_dir / "height.npy")
        metadata = json.loads((terrain_dir / "metadata.json").read_text(encoding="utf-8"))
        origins = origins_from_metadata(metadata, num_envs=4, spawn_z=0.2)
        lifted = lift_origins_to_heightmap(origins, height, metadata, clearance=0.2)

        for origin in lifted.reshape(-1, 3):
            local_height = height_at_xy(height, metadata, float(origin[0]), float(origin[1]))
            self.assertAlmostEqual(float(origin[2]), local_height + 0.2, places=5)


if __name__ == "__main__":
    unittest.main()
