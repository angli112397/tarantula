import unittest
import math

from tarantula_control.command_profiles import PROFILE_CHOICES, parse_route_specs
from tarantula_control.vehicle_geometry import VEHICLE_GEOMETRY


def _profile_bounds(profile: str) -> tuple[float, float]:
    x = 0.0
    y = 0.0
    yaw = 0.0
    xs = [x]
    ys = [y]
    for segment in parse_route_specs([], profile=profile):
        if abs(segment.wz) > 1.0e-9:
            yaw += segment.wz * segment.duration_s
        if abs(segment.vx) > 1.0e-9:
            x += math.cos(yaw) * segment.vx * segment.duration_s
            y += math.sin(yaw) * segment.vx * segment.duration_s
            xs.append(x)
            ys.append(y)
    return max(xs) - min(xs), max(ys) - min(ys)


class CommandProfilesTest(unittest.TestCase):
    def test_profile_choices_are_current_contract(self):
        self.assertEqual(PROFILE_CHOICES, ("navi", "primitive", "both"))

    def test_defaults_to_navi_profile(self):
        sequence = parse_route_specs([])

        self.assertEqual(sequence[0].name, "navi_initial_stop")
        self.assertEqual(sequence[-1].name, "navi_final_stop")
        self.assertGreater(sum(segment.duration_s for segment in sequence), 0.0)

    def test_custom_segment_accepts_duration(self):
        sequence = parse_route_specs(
            ["arc,0.1,0.2,1.5"],
            profile="primitive",
            default_duration_s=4.0,
        )

        self.assertEqual(sequence[0].name, "arc")
        self.assertAlmostEqual(sequence[0].vx, 0.1)
        self.assertAlmostEqual(sequence[0].wz, 0.2)
        self.assertAlmostEqual(sequence[0].duration_s, 1.5)

    def test_navi_profile_is_not_body_scale_micro_motion(self):
        width, _ = _profile_bounds("navi")
        self.assertGreaterEqual(width, 10.0 * VEHICLE_GEOMETRY.reference_length)

    def test_both_profile_is_navi_then_primitive(self):
        sequence = parse_route_specs([], profile="both")
        names = [segment.name for segment in sequence]

        self.assertEqual(names[0], "navi_initial_stop")
        self.assertIn("final_stop", names)
        self.assertGreater(len(sequence), len(parse_route_specs([], profile="navi")))


if __name__ == "__main__":
    unittest.main()
