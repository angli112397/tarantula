import importlib.util
import sys
import unittest
from pathlib import Path


def load_gate_module():
    root = Path(__file__).resolve().parents[3]
    script = root / "scripts" / "isaac_eval_gate.py"
    spec = importlib.util.spec_from_file_location("isaac_eval_gate", script)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def summary(*, vx_error: float, wz_error: float, saturation: float = 0.0, turn_wz_error: float | None = None):
    turn_error = wz_error if turn_wz_error is None else turn_wz_error
    segments = [
        {
            "segment": "drive_after_left",
            "rms_vx_error": vx_error,
            "rms_wz_error": wz_error,
            "action_saturation_rate": saturation,
            "termination_counts": {},
        },
        {
            "segment": "turn_left_authority",
            "rms_vx_error": vx_error,
            "rms_wz_error": turn_error,
            "action_saturation_rate": saturation,
            "termination_counts": {},
        },
        {
            "segment": "turn_right_authority",
            "rms_vx_error": vx_error,
            "rms_wz_error": turn_error,
            "action_saturation_rate": saturation,
            "termination_counts": {},
        },
    ]
    return {
        "spawn_health": {"initial_termination_counts": {}},
        "segments": segments,
    }


class IsaacEvalGateTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.gate = load_gate_module()

    def test_policy_passes_when_score_improves_with_low_saturation(self):
        open_loop = summary(vx_error=0.10, wz_error=0.10)
        policy = summary(vx_error=0.06, wz_error=0.06, saturation=0.05)

        result = self.gate.evaluate_gate(open_loop, policy)

        self.assertTrue(result["pass"])
        self.assertGreater(result["score_improvement_fraction"], 0.10)

    def test_policy_fails_when_improvement_uses_high_saturation(self):
        open_loop = summary(vx_error=0.10, wz_error=0.10)
        policy = summary(vx_error=0.06, wz_error=0.06, saturation=0.25)

        result = self.gate.evaluate_gate(open_loop, policy)

        self.assertFalse(result["pass"])
        self.assertTrue(any("mean_action_saturation" in item for item in result["failures"]))

    def test_policy_fails_when_turn_authority_regresses(self):
        open_loop = summary(vx_error=0.10, wz_error=0.10, turn_wz_error=0.10)
        policy = summary(vx_error=0.05, wz_error=0.05, saturation=0.01, turn_wz_error=0.12)

        result = self.gate.evaluate_gate(open_loop, policy)

        self.assertFalse(result["pass"])
        self.assertTrue(any("turn_left_authority" in item for item in result["failures"]))

    def test_policy_fails_on_hard_termination(self):
        open_loop = summary(vx_error=0.10, wz_error=0.10)
        policy = summary(vx_error=0.06, wz_error=0.06, saturation=0.01)
        policy["segments"][0]["termination_counts"] = {"tilt": 1}

        result = self.gate.evaluate_gate(open_loop, policy)

        self.assertFalse(result["pass"])
        self.assertIn("hard_terminations 1", result["failures"])


if __name__ == "__main__":
    unittest.main()
