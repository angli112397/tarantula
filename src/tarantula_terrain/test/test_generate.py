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

            height = np.load(out_dir / "height.npy")
            occupancy = np.load(out_dir / "occupancy.npy")
            metadata = json.loads((out_dir / "metadata.json").read_text(encoding="utf-8"))
            map_yaml = (out_dir / "map.yaml").read_text(encoding="ascii")

            self.assertEqual(metadata["preset"], "nav_maze")
            self.assertEqual(height.shape, (161, 241))
            self.assertEqual(occupancy.shape, height.shape)
            self.assertTrue((out_dir / "map.pgm").is_file())
            self.assertIn("resolution: 0.100000", map_yaml)
            self.assertIn("origin: [-12.000000, -8.000000, 0.0]", map_yaml)
            self.assertEqual(metadata["obstacle_count"], 3)
            self.assertEqual(metadata["door_width"], 5.2)
            self.assertEqual(metadata["min_corridor_width"], 4.8)
            self.assertEqual(metadata["spawn"], [0.0, 0.0, 0.0])
            self.assertGreaterEqual(len(metadata["wall_rects"]), 12)

            world_sdf = (out_dir / "world.sdf").read_text(encoding="utf-8")
            self.assertIn('model name="flat_floor"', world_sdf)
            self.assertNotIn("<include><uri>", world_sdf)

            cfg = NavMazeCfg()
            sx, sy, _ = metadata["spawn"]
            ix = int(round((sx + cfg.size_x / 2.0) / cfg.resolution))
            iy = int(round((sy + cfg.size_y / 2.0) / cfg.resolution))
            self.assertFalse(bool(occupancy[iy, ix]))

            pgm = _read_pgm_payload(out_dir / "map.pgm")
            expected_pgm = np.where(occupancy[::-1, :], 0, 254).astype(np.uint8)
            self.assertTrue(np.array_equal(pgm, expected_pgm))


if __name__ == "__main__":
    unittest.main()
