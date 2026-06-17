import importlib.util
import sys
import unittest
from pathlib import Path


def load_benchmark_module():
    root = Path(__file__).resolve().parents[3]
    script = root / "scripts" / "gazebo_cmd_tracking_benchmark.py"
    spec = importlib.util.spec_from_file_location("gazebo_cmd_tracking_benchmark", script)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


class GazeboCmdTrackingBenchmarkTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.benchmark = load_benchmark_module()

    def test_compare_passes_when_candidate_improves_without_saturation(self):
        baseline = {
            "label": "classical",
            "segments": [
                {
                    "segment": "forward",
                    "rms_vx_error": 0.20,
                    "rms_wz_error": 0.10,
                    "roll_rms_rad": 0.10,
                    "pitch_rms_rad": 0.10,
                    "rl_action_saturation_rate": 0.0,
                }
            ],
        }
        candidate = {
            "label": "rl",
            "segments": [
                {
                    "segment": "forward",
                    "rms_vx_error": 0.10,
                    "rms_wz_error": 0.05,
                    "roll_rms_rad": 0.10,
                    "pitch_rms_rad": 0.10,
                    "rl_action_saturation_rate": 0.0,
                }
            ],
        }

        comparison = self.benchmark.compare_summaries(baseline, candidate)

        self.assertTrue(comparison["pass"])
        self.assertEqual(comparison["segments_improved"], ["forward"])

    def test_compare_fails_on_stability_regression(self):
        baseline = {
            "segments": [
                {
                    "segment": "turn_left",
                    "rms_vx_error": 0.10,
                    "rms_wz_error": 0.10,
                    "roll_rms_rad": 0.10,
                    "pitch_rms_rad": 0.10,
                }
            ],
        }
        candidate = {
            "segments": [
                {
                    "segment": "turn_left",
                    "rms_vx_error": 0.05,
                    "rms_wz_error": 0.05,
                    "roll_rms_rad": 0.20,
                    "pitch_rms_rad": 0.10,
                    "rl_action_saturation_rate": 0.0,
                }
            ],
        }

        comparison = self.benchmark.compare_summaries(baseline, candidate)

        self.assertFalse(comparison["pass"])
        self.assertIn("turn_left:stability_regression", comparison["hard_failures"])


if __name__ == "__main__":
    unittest.main()
