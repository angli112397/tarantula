import json
import tempfile
import unittest
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np

from tarantula_terrain.generate import generate
from tarantula_terrain.nav_maze import NavMazeCfg, generate as generate_nav_maze


def _read_pgm_payload(path: Path) -> np.ndarray:
    raw = path.read_bytes()
    tokens = []
    i = 0
    while len(tokens) < 4:
        while raw[i : i + 1].isspace():
            i += 1
        if raw[i : i + 1] == b"#":
            while raw[i : i + 1] != b"\n":
                i += 1
            continue
        j = i
        while j < len(raw) and not raw[j : j + 1].isspace():
            j += 1
        tokens.append(raw[i:j])
        i = j
    while raw[i : i + 1].isspace():
        i += 1
    magic, width, height, max_value = tokens
    if magic != b"P5" or max_value != b"255":
        raise ValueError(f"unsupported PGM header in {path}")
    return np.frombuffer(raw[i:], dtype=np.uint8).reshape(int(height), int(width))


class GenerateTerrainTest(unittest.TestCase):
    def test_gazebo_demo_assets_are_valid(self):
        with tempfile.TemporaryDirectory() as tmp:
            out_dir = generate("gazebo_demo", 7, Path(tmp))

            self.assertTrue((out_dir / "height.npy").is_file())
            self.assertTrue((out_dir / "terrain.obj").is_file())
            ET.parse(out_dir / "terrain.sdf")
            ET.parse(out_dir / "world.sdf")

            height = np.load(out_dir / "height.npy")
            metadata = json.loads((out_dir / "metadata.json").read_text(encoding="utf-8"))
            world = (out_dir / "world.sdf").read_text(encoding="utf-8")
            self.assertEqual(metadata["preset"], "gazebo_demo")
            self.assertEqual(height.shape, (151, 226))
            self.assertGreater(float(height.max()), 0.10)
            self.assertLess(float(height.min()), 0.0)
            self.assertNotIn("central_barrier", world)
            self.assertNotIn("landmark_blue", world)

    def test_rl_curriculum_exports_tile_origins(self):
        with tempfile.TemporaryDirectory() as tmp:
            out_dir = generate("rl_curriculum", 7, Path(tmp))

            ET.parse(out_dir / "terrain.sdf")
            ET.parse(out_dir / "world.sdf")

            height = np.load(out_dir / "height.npy")
            metadata = json.loads((out_dir / "metadata.json").read_text(encoding="utf-8"))
            self.assertEqual(metadata["preset"], "rl_curriculum")
            self.assertEqual(height.shape, (161, 241))
            self.assertEqual(metadata["num_rows"], 4)
            self.assertEqual(metadata["num_cols"], 6)
            self.assertEqual(len(metadata["env_origins"]), 24)
            self.assertEqual(len(metadata["labels"]), 24)

    def test_nav_maze_aligns_with_rl_curriculum_grid(self):
        with tempfile.TemporaryDirectory() as tmp:
            out_dir = generate_nav_maze(7, Path(tmp))

            ET.parse(out_dir / "terrain.sdf")
            ET.parse(out_dir / "world.sdf")
            ET.parse(out_dir / "world_mesh_contact.sdf")

            height = np.load(out_dir / "height.npy")
            occupancy = np.load(out_dir / "occupancy.npy")
            metadata = json.loads((out_dir / "metadata.json").read_text(encoding="utf-8"))
            map_yaml = (out_dir / "map.yaml").read_text(encoding="ascii")
            terrain_map_yaml = (out_dir / "terrain_cost_map.yaml").read_text(encoding="ascii")
            speed_mask_yaml = (out_dir / "terrain_speed_mask.yaml").read_text(encoding="ascii")
            terrain_cost = np.load(out_dir / "traversability_cost.npy")
            speed_mask = np.load(out_dir / "terrain_speed_mask.npy")

            self.assertEqual(metadata["preset"], "nav_maze")
            self.assertEqual(height.shape, (161, 241))
            self.assertEqual(occupancy.shape, height.shape)
            self.assertEqual(terrain_cost.shape, height.shape)
            self.assertEqual(speed_mask.shape, height.shape)
            self.assertEqual(metadata["height_source"]["preset"], "rl_curriculum")
            self.assertGreater(float(height.max()), 0.05)
            self.assertLess(float(height.min()), 0.0)
            self.assertTrue((out_dir / "map.pgm").is_file())
            self.assertTrue((out_dir / "terrain_cost_map.pgm").is_file())
            self.assertTrue((out_dir / "terrain_speed_mask.pgm").is_file())
            self.assertTrue((out_dir / "world_mesh_contact.sdf").is_file())
            self.assertIn("resolution: 0.100000", map_yaml)
            self.assertIn("origin: [-12.000000, -8.000000, 0.0]", map_yaml)
            self.assertIn("mode: scale", terrain_map_yaml)
            self.assertIn("origin: [-12.000000, -8.000000, 0.0]", terrain_map_yaml)
            self.assertIn("mode: scale", speed_mask_yaml)
            self.assertEqual(metadata["layering"]["nav2_terrain_cost_map"], "terrain_cost_map.yaml")
            self.assertEqual(metadata["layering"]["nav2_speed_filter_mask"], "terrain_speed_mask.yaml")
            self.assertEqual(metadata["layering"]["traversability_cost_layer"], "traversability_cost.npy")
            self.assertEqual(metadata["layering"]["speed_filter_mask_layer"], "terrain_speed_mask.npy")
            self.assertGreater(metadata["traversability"]["medium_cost_cells"], 0)
            self.assertEqual(metadata["obstacle_count"], 3)
            self.assertEqual(metadata["door_width"], 5.2)
            self.assertEqual(metadata["min_corridor_width"], 4.8)
            self.assertEqual(metadata["spawn"], [0.0, 0.0, 0.0])
            self.assertGreaterEqual(len(metadata["wall_rects"]), 12)

            world_sdf = (out_dir / "world.sdf").read_text(encoding="utf-8")
            self.assertIn('model name="flat_floor"', world_sdf)
            self.assertIn('model name="terrain_visual_only"', world_sdf)
            self.assertNotIn("<include><uri>", world_sdf)
            mesh_world_sdf = (out_dir / "world_mesh_contact.sdf").read_text(encoding="utf-8")
            self.assertIn("terrain.sdf", mesh_world_sdf)

            cfg = NavMazeCfg()
            sx, sy, _ = metadata["spawn"]
            ix = int(round((sx + cfg.size_x / 2.0) / cfg.resolution))
            iy = int(round((sy + cfg.size_y / 2.0) / cfg.resolution))
            self.assertFalse(bool(occupancy[iy, ix]))
            half = int(round(4.8 / 2.0 / cfg.resolution))
            self.assertTrue(np.allclose(height[iy - half : iy + half + 1, ix - half : ix + half + 1], 0.0))
            self.assertTrue(np.all(terrain_cost[iy - half : iy + half + 1, ix - half : ix + half + 1] == 0))
            self.assertTrue(np.all(speed_mask[iy - half : iy + half + 1, ix - half : ix + half + 1] == 0))
            self.assertTrue(np.all(terrain_cost[occupancy] == 100))
            self.assertTrue(np.all(speed_mask[occupancy] == 0))
            self.assertGreater(int(speed_mask.max()), 0)
            self.assertLessEqual(int(speed_mask.max()), 65)

            pgm = _read_pgm_payload(out_dir / "map.pgm")
            expected_pgm = np.where(occupancy[::-1, :], 0, 254).astype(np.uint8)
            self.assertTrue(np.array_equal(pgm, expected_pgm))
            terrain_pgm = _read_pgm_payload(out_dir / "terrain_cost_map.pgm")
            expected_terrain_pgm = np.rint(
                254.0 - np.clip(terrain_cost[::-1, :].astype(np.float32), 0.0, 100.0) / 100.0 * 254.0
            ).astype(np.uint8)
            self.assertTrue(np.array_equal(terrain_pgm, expected_terrain_pgm))
            speed_pgm = _read_pgm_payload(out_dir / "terrain_speed_mask.pgm")
            expected_speed_pgm = np.rint(
                254.0 - np.clip(speed_mask[::-1, :].astype(np.float32), 0.0, 100.0) / 100.0 * 254.0
            ).astype(np.uint8)
            self.assertTrue(np.array_equal(speed_pgm, expected_speed_pgm))


if __name__ == "__main__":
    unittest.main()
