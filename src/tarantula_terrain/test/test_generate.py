import json
import tempfile
import unittest
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np

from tarantula_terrain.generate import generate


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


if __name__ == "__main__":
    unittest.main()
