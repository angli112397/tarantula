import unittest

import numpy as np

from tarantula_control.control_interfaces import (
    DEFAULT_TRACK_SCALE,
    EFFECTIVE_TRACK,
    WHEEL_RADIUS,
    skid_steer_wheel_speeds,
)
from tarantula_control.motion_control import (
    POSTURE_OBSERVATION_DIM,
    POSTURE_ACTION_DIM,
    MotionControlConfig,
    SkidSteerMotionController,
    build_posture_observation,
)
from tarantula_control.suspension_core import LEGS
from tarantula_control.suspension_core import validate_hip_targets
from tarantula_control.vehicle_geometry import VEHICLE_GEOMETRY


TEST_HIGH_TURN_TRACK_SCALE = 3.0


class ControlInterfacesTest(unittest.TestCase):
    def test_forward_cmd_sets_all_wheels_equal(self):
        speeds = skid_steer_wheel_speeds(0.26, 0.0)
        self.assertEqual(len(speeds), len(LEGS))
        self.assertTrue(all(abs(v - 2.0) < 1e-9 for v in speeds))

    def test_vehicle_geometry_uses_v3_long_arm_baseline(self):
        self.assertGreater(VEHICLE_GEOMETRY.overall_length, 1.35)

    def test_yaw_cmd_splits_left_right(self):
        speeds = skid_steer_wheel_speeds(0.0, 0.2)
        left = -0.5 * EFFECTIVE_TRACK * DEFAULT_TRACK_SCALE * 0.2 / WHEEL_RADIUS
        right = 0.5 * EFFECTIVE_TRACK * DEFAULT_TRACK_SCALE * 0.2 / WHEEL_RADIUS
        self.assertEqual(speeds, [left, right, left, right, left, right])

    def test_yaw_feedback_increases_left_right_split_when_yaw_is_low(self):
        controller = SkidSteerMotionController(MotionControlConfig(
            yaw_track_scale=TEST_HIGH_TURN_TRACK_SCALE,
            yaw_rate_kp=2.0,
            max_abs_wheel_omega=10.0,
        ))
        command = controller.limit_command(0.0, 0.25)
        open_loop = controller.wheel_targets(command, measured_wz=None)
        closed_loop = controller.wheel_targets(command, measured_wz=0.0, dt=0.02)
        self.assertLess(closed_loop[0], open_loop[0])
        self.assertGreater(closed_loop[1], open_loop[1])

    def test_yaw_feedback_resets_when_no_yaw_command(self):
        controller = SkidSteerMotionController(MotionControlConfig(
            yaw_track_scale=TEST_HIGH_TURN_TRACK_SCALE,
            yaw_rate_kp=0.0,
            yaw_rate_ki=1.0,
            max_abs_wheel_omega=10.0,
        ))
        turn = controller.limit_command(0.0, 0.25)
        controller.wheel_targets(turn, measured_wz=0.0, dt=0.5)
        stop = controller.limit_command(0.0, 0.0)
        self.assertEqual(controller.wheel_targets(stop, measured_wz=0.0, dt=0.5), [0.0] * 6)

    def test_motion_controller_slew_limits_final_wheel_targets(self):
        controller = SkidSteerMotionController(MotionControlConfig(
            yaw_track_scale=TEST_HIGH_TURN_TRACK_SCALE,
            yaw_rate_kp=0.0,
            max_wheel_accel=10.0,
            max_abs_wheel_omega=10.0,
        ))
        command = controller.limit_command(0.0, 0.25)
        wheel = controller.filtered_wheel_targets(command, measured_wz=0.0, dt=0.1)
        self.assertEqual(wheel, [-1.0, 1.0, -1.0, 1.0, -1.0, 1.0])

    def test_drive_scale_adjusts_classical_forward_gain(self):
        controller = SkidSteerMotionController(MotionControlConfig(
            drive_scale=1.25,
            max_abs_wheel_omega=10.0,
        ))
        command = controller.limit_command(0.26, 0.0)
        wheel = controller.filtered_wheel_targets(command)
        self.assertEqual(wheel, [2.5] * 6)

    def test_motion_controller_clamps_compensated_target(self):
        controller = SkidSteerMotionController(MotionControlConfig(
            max_abs_cmd_vx=1.0,
            max_abs_wheel_omega=3.0,
        ))
        command = controller.limit_command(0.39, 0.0)
        wheel = controller.filtered_wheel_targets(command)
        self.assertEqual(wheel, [3.0] * 6)

    def test_stop_mode_outputs_zero_wheels(self):
        controller = SkidSteerMotionController(MotionControlConfig(max_abs_wheel_omega=10.0))
        command = controller.limit_command(0.0, 0.0)
        wheel = controller.filtered_wheel_targets(command)
        self.assertEqual(wheel, [0.0] * 6)

    def test_curve_cmd_applies_vx_and_wz_simultaneously(self):
        """Verify that a blended vx+wz command (Nav2 curve) produces correct differential."""
        controller = SkidSteerMotionController(MotionControlConfig(
            drive_scale=1.0,
            yaw_track_scale=1.0,
            max_abs_wheel_omega=10.0,
        ))
        command = controller.limit_command(0.2, 0.3)
        wheel = controller.wheel_targets(command)
        from tarantula_control.control_interfaces import EFFECTIVE_TRACK, WHEEL_RADIUS
        expected_left = (0.2 - 0.5 * EFFECTIVE_TRACK * 0.3) / WHEEL_RADIUS
        expected_right = (0.2 + 0.5 * EFFECTIVE_TRACK * 0.3) / WHEEL_RADIUS
        self.assertAlmostEqual(wheel[0], expected_left, places=9)   # fl
        self.assertAlmostEqual(wheel[1], expected_right, places=9)  # fr

    def test_posture_observation_layout_is_56d(self):
        zeros = {leg: 0.0 for leg in LEGS}
        ones = {leg: 1.0 for leg in LEGS}
        zero_forces = {leg: (0.0, 0.0, 0.0) for leg in LEGS}
        command = SkidSteerMotionController().limit_command(0.2, 0.1)
        obs = build_posture_observation(
            projected_gravity_b=(0.0, 0.0, -1.0),
            root_ang_vel_b=(0.0, 0.0, 0.0),
            susp_joint_pos=zeros,
            susp_joint_vel=zeros,
            wheel_joint_vel=zeros,
            wheel_force=zero_forces,
            contact_uptime=ones,
            command=command,
            prev_action=np.zeros(POSTURE_ACTION_DIM, dtype=np.float32),
        )
        self.assertEqual(obs.shape, (POSTURE_OBSERVATION_DIM,))

    def test_posture_observation_rejects_old_three_dim_prev_action(self):
        zeros = {leg: 0.0 for leg in LEGS}
        ones = {leg: 1.0 for leg in LEGS}
        zero_forces = {leg: (0.0, 0.0, 0.0) for leg in LEGS}
        command = SkidSteerMotionController().limit_command(0.2, 0.1)

        with self.assertRaises(ValueError):
            build_posture_observation(
                projected_gravity_b=(0.0, 0.0, -1.0),
                root_ang_vel_b=(0.0, 0.0, 0.0),
                susp_joint_pos=zeros,
                susp_joint_vel=zeros,
                wheel_joint_vel=zeros,
                wheel_force=zero_forces,
                contact_uptime=ones,
                command=command,
                prev_action=np.zeros(3, dtype=np.float32),
            )

    def test_validate_hip_targets_clamps_to_baseline_limit(self):
        clamped = validate_hip_targets([1.0] * 6)
        self.assertEqual(clamped, tuple([0.45] * 6))


if __name__ == "__main__":
    unittest.main()
