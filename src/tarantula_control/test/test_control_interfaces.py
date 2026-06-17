import unittest

import numpy as np

from tarantula_control.control_interfaces import (
    DRIVE_SCALE_DELTA_LIMIT,
    EFFECTIVE_TRACK,
    TRACK_SCALE_DELTA_LIMIT,
    WHEEL_RADIUS,
    WHEEL_DIRECTION,
    YAW_AUTHORITY_MULTIPLIER,
    skid_steer_wheel_speeds,
)
from tarantula_control.motion_control import (
    STAGE_A_OBSERVATION_DIM,
    MotionControlConfig,
    SkidSteerMotionController,
    build_stage_a_observation,
)
from tarantula_control.suspension_core import LEGS
from tarantula_control.suspension_core import blend_hip_targets, posture_profile, validate_hip_targets


class ControlInterfacesTest(unittest.TestCase):
    def test_forward_cmd_sets_all_wheels_equal(self):
        speeds = skid_steer_wheel_speeds(0.26, 0.0)
        self.assertEqual(len(speeds), len(LEGS))
        self.assertTrue(all(abs(v - 2.0) < 1e-9 for v in speeds))

    def test_mean_wheel_forward_velocity_applies_joint_direction(self):
        speeds = {leg: 2.0 * WHEEL_DIRECTION[leg] for leg in LEGS}
        from tarantula_control.control_interfaces import mean_wheel_forward_velocity
        self.assertAlmostEqual(mean_wheel_forward_velocity(speeds), 0.26)

    def test_yaw_cmd_splits_left_right(self):
        speeds = skid_steer_wheel_speeds(0.0, 0.2)
        left = -0.5 * EFFECTIVE_TRACK * YAW_AUTHORITY_MULTIPLIER * 0.2 / WHEEL_RADIUS
        right = 0.5 * EFFECTIVE_TRACK * YAW_AUTHORITY_MULTIPLIER * 0.2 / WHEEL_RADIUS
        self.assertEqual(speeds, [left, right, left, right, left, right])

    def test_yaw_feedback_increases_left_right_split_when_yaw_is_low(self):
        controller = SkidSteerMotionController(MotionControlConfig(
            pure_turn_track_scale=3.0,
            yaw_rate_kp=2.0,
            max_abs_wheel_omega=10.0,
            pure_turn_forward_bias=0.0,
        ))
        command = controller.limit_command(0.0, 0.25)
        open_loop = controller.wheel_targets(command, measured_wz=None)
        closed_loop = controller.wheel_targets(command, measured_wz=0.0, dt=0.02)
        self.assertLess(closed_loop[0], open_loop[0])
        self.assertGreater(closed_loop[1], open_loop[1])

    def test_yaw_feedback_resets_when_no_yaw_command(self):
        controller = SkidSteerMotionController(MotionControlConfig(
            pure_turn_track_scale=3.0,
            yaw_rate_kp=0.0,
            yaw_rate_ki=1.0,
            max_abs_wheel_omega=10.0,
            pure_turn_forward_bias=0.0,
        ))
        turn = controller.limit_command(0.0, 0.25)
        controller.wheel_targets(turn, measured_wz=0.0, dt=0.5)
        stop = controller.limit_command(0.0, 0.0)
        self.assertEqual(controller.wheel_targets(stop, measured_wz=0.0, dt=0.5), [0.0] * 6)

    def test_motion_controller_slew_limits_final_wheel_targets(self):
        controller = SkidSteerMotionController(MotionControlConfig(
            pure_turn_track_scale=3.0,
            yaw_rate_kp=0.0,
            max_wheel_accel=10.0,
            max_abs_wheel_omega=10.0,
            pure_turn_forward_bias=0.0,
        ))
        command = controller.limit_command(0.0, 0.25)
        wheel = controller.compensated_wheel_targets(command, None, measured_wz=0.0, dt=0.1)
        self.assertEqual(wheel, [-1.0, 1.0, -1.0, 1.0, -1.0, 1.0])

    def test_pure_turn_forward_bias_adds_common_forward_speed(self):
        controller = SkidSteerMotionController(MotionControlConfig(
            pure_turn_track_scale=3.0,
            yaw_rate_kp=0.0,
            max_abs_cmd_wz=0.4,
            pure_turn_forward_bias=0.16,
            max_abs_wheel_omega=10.0,
        ))
        command = controller.limit_command(0.0, 0.2)
        wheel = controller.wheel_targets(command, measured_wz=None)
        no_bias = skid_steer_wheel_speeds(0.0, 0.2, track_scale=3.0)
        expected_delta = (0.16 * (0.2 / 0.4)) / WHEEL_RADIUS
        self.assertTrue(all(abs((a - b) - expected_delta) < 1e-9 for a, b in zip(wheel, no_bias)))

    def test_motion_controller_adds_bounded_structured_compensation(self):
        controller = SkidSteerMotionController(MotionControlConfig(max_abs_wheel_omega=6.0))
        command = controller.limit_command(0.26, 0.0)
        wheel = controller.compensated_wheel_targets(command, np.ones(3, dtype=np.float32))
        scale = 1.0 + DRIVE_SCALE_DELTA_LIMIT
        expected = [
            2.0 * scale,
            2.0 * scale,
            2.0 * scale,
            2.0 * scale,
            2.0 * scale,
            2.0 * scale,
        ]
        self.assertTrue(all(abs(a - b) < 1e-6 for a, b in zip(wheel, expected)))

    def test_motion_controller_clamps_compensated_target(self):
        controller = SkidSteerMotionController(MotionControlConfig(
            max_abs_cmd_vx=1.0,
            max_abs_wheel_omega=3.0,
        ))
        command = controller.limit_command(0.39, 0.0)
        wheel = controller.compensated_wheel_targets(command, np.ones(3, dtype=np.float32))
        self.assertEqual(wheel, [3.0] * 6)

    def test_track_scale_action_changes_yaw_split(self):
        controller = SkidSteerMotionController(MotionControlConfig(
            pure_turn_track_scale=3.0,
            yaw_rate_kp=0.0,
            max_abs_wheel_omega=10.0,
        ))
        command = controller.limit_command(0.0, 0.2)
        base = controller.compensated_wheel_targets(command, np.zeros(3, dtype=np.float32))
        boosted = controller.compensated_wheel_targets(command, np.array([1.0, 0.0, 0.0], dtype=np.float32))
        self.assertAlmostEqual(abs(boosted[0] / base[0]), 1.0 + TRACK_SCALE_DELTA_LIMIT)
        self.assertAlmostEqual(abs(boosted[1] / base[1]), 1.0 + TRACK_SCALE_DELTA_LIMIT)

    def test_track_scale_scheduler_uses_arc_scale_for_moving_arc(self):
        controller = SkidSteerMotionController(MotionControlConfig(
            arc_track_scale=1.0,
            pure_turn_track_scale=3.0,
            track_scale_transition_vx=0.08,
            yaw_rate_kp=0.0,
            max_abs_wheel_omega=10.0,
        ))
        arc = controller.limit_command(0.16, 0.10)
        turn = controller.limit_command(0.0, 0.10)
        self.assertAlmostEqual(controller.scheduled_track_scale(arc), 1.0)
        self.assertAlmostEqual(controller.scheduled_track_scale(turn), 3.0)
        arc_wheel = controller.wheel_targets(arc, measured_wz=None)
        self.assertGreater(arc_wheel[0], 0.0)
        self.assertGreater(arc_wheel[1], arc_wheel[0])

    def test_stage_a_observation_layout_is_47d(self):
        zeros = {leg: 0.0 for leg in LEGS}
        zero_forces = {leg: (0.0, 0.0, 0.0) for leg in LEGS}
        command = SkidSteerMotionController().limit_command(0.2, 0.1)
        obs = build_stage_a_observation(
            projected_gravity_b=(0.0, 0.0, -1.0),
            root_ang_vel_b=(0.0, 0.0, 0.0),
            susp_joint_pos=zeros,
            susp_joint_vel=zeros,
            wheel_joint_vel=zeros,
            wheel_force=zero_forces,
            command=command,
            prev_action=np.zeros(3, dtype=np.float32),
        )
        self.assertEqual(obs.shape, (STAGE_A_OBSERVATION_DIM,))

    def test_posture_profiles_are_bounded_six_leg_targets(self):
        for name in ("neutral", "front_down", "rear_down", "raise", "lower", "left_trim"):
            target = posture_profile(name)
            self.assertEqual(len(target), len(LEGS))
            self.assertTrue(all(abs(value) <= 0.45 for value in target))

    def test_posture_residual_blend_is_bounded(self):
        base = posture_profile("neutral")
        blended = blend_hip_targets(base, [1.0] * 6, residual_limit=0.1)
        self.assertEqual(blended, tuple([0.1] * 6))
        clamped = validate_hip_targets([1.0] * 6)
        self.assertEqual(clamped, tuple([0.45] * 6))


if __name__ == "__main__":
    unittest.main()
