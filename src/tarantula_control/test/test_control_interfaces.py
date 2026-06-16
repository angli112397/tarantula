import unittest

from tarantula_control.control_interfaces import (
    EFFECTIVE_TRACK,
    WHEEL_RADIUS,
    skid_steer_wheel_speeds,
)
from tarantula_control.suspension_core import LEGS


class ControlInterfacesTest(unittest.TestCase):
    def test_forward_cmd_sets_all_wheels_equal(self):
        speeds = skid_steer_wheel_speeds(0.26, 0.0)
        self.assertEqual(len(speeds), len(LEGS))
        self.assertTrue(all(abs(v - 2.0) < 1e-9 for v in speeds))

    def test_yaw_cmd_splits_left_right(self):
        speeds = skid_steer_wheel_speeds(0.0, 0.2)
        left = -0.5 * EFFECTIVE_TRACK * 0.2 / WHEEL_RADIUS
        right = 0.5 * EFFECTIVE_TRACK * 0.2 / WHEEL_RADIUS
        self.assertEqual(speeds, [left, right, left, right, left, right])


if __name__ == "__main__":
    unittest.main()
