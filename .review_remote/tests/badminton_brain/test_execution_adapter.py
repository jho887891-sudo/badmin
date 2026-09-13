# -*- coding: utf-8 -*-
"""T9 (TDD) - execution adapter tests: SafeCommand -> wheel/arm targets -> Feedback.

Spec: docs/superpowers/plans/2026-09-13-brain-modules.md (T9 row);
      docs/simulation/BADMINTON_ROBOT.md S9 (two drive modes), S10 (inverse kinematics),
      S11 (steer-angle optimisation), S24 (robot command);
      docs/architecture/ROBOT_BRAIN.md S8 (feedback path), S11/S13 (layer duties).

The adapter is a pure-numpy command sink: it must never import Isaac/Kit/Omni.
The wheel-level numbers are checked against the canonical implementation
simulation/robots/badminton_robot/morph_one/kinematics.py (tolerance 1e-12).
"""
from __future__ import annotations
import math
import re
import sys
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'src'))
sys.path.insert(0, str(ROOT / 'simulation'))

from badminton_brain.execution import sim_adapter as sim_adapter_module  # noqa: E402
from badminton_brain.execution.sim_adapter import (  # noqa: E402
    ExecutionCommand,
    SimExecutionAdapter,
)
from badminton_brain.interfaces import ExecutionModule  # noqa: E402
from badminton_brain.types import BrainBoundaryError, Layer, SafeCommand  # noqa: E402
from robots.badminton_robot.badminton_robot_cfg import DriveMode, WheelId  # noqa: E402
from robots.badminton_robot.morph_one import kinematics as canonical_kinematics  # noqa: E402
from robots.badminton_robot.morph_one.kinematics import (  # noqa: E402
    body_twist_to_wheel_targets,
    wheel_targets_to_body_twist,
)

# TEMP wheel centres in robot_base + wheel radius: same proxies as
# tests/simulation/robots/test_morph_one_kinematics.py (the real geometry is
# Param(None, REQUIRES_MEASUREMENT) in badminton_robot_cfg).  Supplied explicitly so that
# this test never depends on an invented value living inside the adapter.
WHEEL_POSITIONS = {
    WheelId.FL: (0.25, 0.20),
    WheelId.FR: (0.25, -0.20),
    WheelId.RL: (-0.25, 0.20),
    WheelId.RR: (-0.25, -0.20),
}
R_W = 0.06


def safe_command(twist, joint=None, timestamp=0.0):
    """Build a batched SafeCommand from one row or a list of rows."""
    twist = np.atleast_2d(np.asarray(twist, dtype=float))
    n = twist.shape[0]
    if joint is None:
        joint = np.zeros((n, 6))
    else:
        joint = np.atleast_2d(np.asarray(joint, dtype=float))
    return SafeCommand(base_twist=twist, joint_position_target=joint, timestamp=timestamp)


def make_adapter(mode=DriveMode.BODY_TWIST_ACTUATOR, num_envs=1):
    return SimExecutionAdapter(num_envs=num_envs, drive_mode=mode,
                               wheel_positions=WHEEL_POSITIONS, wheel_radius_m=R_W)


def canonical(twist, current_steer_rad=None):
    return body_twist_to_wheel_targets(twist, WHEEL_POSITIONS, R_W,
                                       current_steer_rad=current_steer_rad)


def court_to_body_planar(court_velocity_xy, yaw_rad):
    """Court-frame planar velocity -> robot_base velocity: R(-yaw) @ v.

    Coordinator ruling: WholeBodyTarget.base_twist / SafeCommand.base_twist are the robot_base
    (body) twist, and the court -> body rotation is done by the PLANNER (T7).  It is recomputed
    here only so the guard test can state what the execution adapter must receive; the adapter
    itself never rotates anything.
    """
    c, s = math.cos(-yaw_rad), math.sin(-yaw_rad)
    return np.array([[c, -s], [s, c]]) @ np.asarray(court_velocity_xy, dtype=float)


class LayerContractTests(unittest.TestCase):
    def test_adapter_is_the_execution_layer_module(self) -> None:
        adapter = make_adapter()
        self.assertIsInstance(adapter, ExecutionModule)
        self.assertIs(adapter.layer, Layer.EXECUTION)
        self.assertTrue(adapter.is_implemented)

    def test_no_isaac_dependency_in_source_or_runtime(self) -> None:
        source = (ROOT / 'src' / 'badminton_brain' / 'execution' / 'sim_adapter.py').read_text(
            encoding='utf-8')
        forbidden = re.findall(r'^\s*(?:import|from)\s+(isaac|omni|pxr)\b', source, re.M | re.I)
        self.assertEqual(forbidden, [], f'adapter imports Isaac-family modules: {forbidden}')
        for name in ('isaaclab', 'omni', 'pxr'):
            self.assertNotIn(name, sys.modules)

    def test_rejects_non_safe_command(self) -> None:
        adapter = make_adapter()
        with self.assertRaises(BrainBoundaryError):
            adapter.process(object())

    def test_rejects_mismatched_batch_size(self) -> None:
        adapter = make_adapter(num_envs=2)
        with self.assertRaises(BrainBoundaryError):
            adapter.process(safe_command([[0.1, 0.0, 0.0]]))


class BodyTwistModeTests(unittest.TestCase):
    def test_body_twist_mode_drives_the_twist_and_keeps_wheels_diagnostic(self) -> None:
        adapter = make_adapter(DriveMode.BODY_TWIST_ACTUATOR)
        adapter.process(safe_command([0.5, 0.0, 0.0]))
        record = adapter.get_last_command()[0]
        self.assertIsInstance(record, ExecutionCommand)
        self.assertIs(record.drive_mode, DriveMode.BODY_TWIST_ACTUATOR)
        self.assertFalse(record.wheel_level_authoritative)
        np.testing.assert_allclose(record.body_twist, [0.5, 0.0, 0.0], atol=1e-15)
        np.testing.assert_allclose(record.steer_angle_rad, np.zeros(4), atol=1e-15)
        np.testing.assert_allclose(record.wheel_speed_rad_s, 0.5 / R_W, atol=1e-12)
        np.testing.assert_allclose(record.actuator_target['body_twist'], [0.5, 0.0, 0.0])
        self.assertNotIn('wheel_speed_rad_s', record.actuator_target)

    def test_body_twist_mode_is_stateless_across_commands(self) -> None:
        adapter = make_adapter(DriveMode.BODY_TWIST_ACTUATOR)
        adapter.process(safe_command([0.0, 0.3, 0.0]))
        adapter.process(safe_command([0.0, -0.3, 0.0]))
        record = adapter.get_last_command()[0]
        np.testing.assert_allclose(record.steer_angle_rad, -math.pi / 2.0, atol=1e-12)
        self.assertTrue(np.all(record.wheel_speed_rad_s > 0.0))
        # no steering memory: a stop command must not hold a previous angle
        adapter.process(safe_command([0.0, 0.0, 0.0]))
        np.testing.assert_allclose(adapter.get_last_command()[0].steer_angle_rad, np.zeros(4),
                                   atol=1e-15)


class WheelModeTests(unittest.TestCase):
    def test_wheel_mode_drives_wheel_level_targets(self) -> None:
        adapter = make_adapter(DriveMode.STEER_DRIVE_WHEEL_MODEL)
        adapter.process(safe_command([0.5, 0.0, 0.0]))
        record = adapter.get_last_command()[0]
        self.assertIs(record.drive_mode, DriveMode.STEER_DRIVE_WHEEL_MODEL)
        self.assertTrue(record.wheel_level_authoritative)
        self.assertIn('steer_angle_rad', record.actuator_target)
        self.assertIn('wheel_speed_rad_s', record.actuator_target)
        self.assertNotIn('body_twist', record.actuator_target)

    def test_wheel_mode_applies_the_S11_minimum_steering_solution(self) -> None:
        """S11: (theta, +w) and (theta+pi, -w) are equivalent; the wheel-mode adapter must
        pick the solution that does not rotate the steer module more than pi/2."""
        wheel_adapter = make_adapter(DriveMode.STEER_DRIVE_WHEEL_MODEL)
        body_adapter = make_adapter(DriveMode.BODY_TWIST_ACTUATOR)
        wheel_adapter.process(safe_command([0.0, 0.3, 0.0]))   # steer -> +pi/2
        body_adapter.process(safe_command([0.0, 0.3, 0.0]))
        wheel_adapter.process(safe_command([0.0, -0.3, 0.0]))  # reverse requested
        body_adapter.process(safe_command([0.0, -0.3, 0.0]))

        wheel_record = wheel_adapter.get_last_command()[0]
        body_record = body_adapter.get_last_command()[0]

        # the two modes must genuinely differ (this is the mode switch under test)
        np.testing.assert_allclose(wheel_record.steer_angle_rad, math.pi / 2.0, atol=1e-12)
        np.testing.assert_allclose(body_record.steer_angle_rad, -math.pi / 2.0, atol=1e-12)
        self.assertTrue(np.all(wheel_record.wheel_speed_rad_s < 0.0))
        self.assertTrue(np.all(body_record.wheel_speed_rad_s > 0.0))
        self.assertGreater(np.max(np.abs(wheel_record.steer_angle_rad
                                         - body_record.steer_angle_rad)), 1.0)

        # ... while both realise exactly the same chassis twist
        for record in (wheel_record, body_record):
            wire = {wid: canonical_kinematics.WheelTargets(record.steer_angle_rad[i],
                                                           record.wheel_speed_rad_s[i])
                    for i, wid in enumerate(canonical_kinematics.WHEEL_ORDER)}
            recovered = wheel_targets_to_body_twist(wire, WHEEL_POSITIONS, R_W)
            np.testing.assert_allclose(recovered, record.body_twist, atol=1e-12)

    def test_wheel_mode_holds_the_steer_angle_for_a_zero_twist(self) -> None:
        adapter = make_adapter(DriveMode.STEER_DRIVE_WHEEL_MODEL)
        adapter.process(safe_command([0.0, 0.3, 0.0]))
        adapter.process(safe_command([0.0, 0.0, 0.0]))
        record = adapter.get_last_command()[0]
        np.testing.assert_allclose(record.steer_angle_rad, math.pi / 2.0, atol=1e-12)
        np.testing.assert_allclose(record.wheel_speed_rad_s, np.zeros(4), atol=1e-15)


class KinematicsConsistencyTests(unittest.TestCase):
    def test_adapter_uses_the_canonical_kinematics_module(self) -> None:
        adapter = make_adapter()
        self.assertIs(adapter.kinematics, canonical_kinematics)

    def test_wheel_targets_match_kinematics_within_1e_12(self) -> None:
        twists = [[0.5, 0.0, 0.0], [0.0, -0.4, 0.0], [0.2, 0.1, 0.7], [0.0, 0.0, 0.0]]
        adapter = make_adapter(DriveMode.BODY_TWIST_ACTUATOR, num_envs=len(twists))
        adapter.process(safe_command(twists))
        for env_id, twist in enumerate(twists):
            expected = canonical(twist)
            record = adapter.get_last_command()[env_id]
            for index, wid in enumerate(canonical_kinematics.WHEEL_ORDER):
                self.assertAlmostEqual(record.steer_angle_rad[index],
                                       expected[wid].steer_angle_rad, places=12)
                self.assertAlmostEqual(record.wheel_speed_rad_s[index],
                                       expected[wid].wheel_speed_rad_s, places=12)
                self.assertAlmostEqual(record.wheel_tangential_speed_mps[index],
                                       expected[wid].tangential_speed_mps, places=12)

    def test_wheel_mode_targets_match_kinematics_with_steering_state(self) -> None:
        adapter = make_adapter(DriveMode.STEER_DRIVE_WHEEL_MODEL)
        adapter.process(safe_command([0.0, 0.3, 0.0]))
        held = canonical_kinematics.normalise_angle(math.pi / 2.0)
        current = {wid: held for wid in canonical_kinematics.WHEEL_ORDER}
        adapter.process(safe_command([0.0, -0.3, 0.0]))
        expected = canonical([0.0, -0.3, 0.0], current_steer_rad=current)
        record = adapter.get_last_command()[0]
        for index, wid in enumerate(canonical_kinematics.WHEEL_ORDER):
            self.assertAlmostEqual(record.steer_angle_rad[index],
                                   expected[wid].steer_angle_rad, places=12)
            self.assertAlmostEqual(record.wheel_speed_rad_s[index],
                                   expected[wid].wheel_speed_rad_s, places=12)

    def test_feedback_prediction_error_is_the_wheel_level_realisation_residual(self) -> None:
        adapter = make_adapter(DriveMode.STEER_DRIVE_WHEEL_MODEL, num_envs=2)
        feedback = adapter.process(safe_command([[0.3, -0.2, 0.5], [0.0, 0.0, 0.0]]))
        self.assertEqual(feedback.prediction_error.shape, (2,))
        self.assertTrue(np.all(np.isfinite(feedback.prediction_error)))
        self.assertTrue(np.all(feedback.prediction_error <= 1e-9),
                        f'wheel targets do not realise the commanded twist: '
                        f'{feedback.prediction_error}')


class ArmTargetTests(unittest.TestCase):
    def test_six_axis_joint_target_is_passed_through_and_cached(self) -> None:
        adapter = make_adapter(num_envs=2)
        joint = [[0.1, 0.2, 0.3, 0.4, 0.5, 0.6], [-0.1, -0.2, -0.3, -0.4, -0.5, -0.6]]
        adapter.process(safe_command([[0.1, 0.0, 0.0], [0.2, 0.0, 0.0]], joint=joint, timestamp=3.5))
        records = adapter.get_last_command()
        self.assertEqual(len(records), 2)
        for env_id in range(2):
            self.assertEqual(records[env_id].joint_position_target.shape, (6,))
            np.testing.assert_allclose(records[env_id].joint_position_target, joint[env_id],
                                       atol=1e-15)
            self.assertEqual(records[env_id].body_twist.shape, (3,))
            self.assertEqual(records[env_id].steer_angle_rad.shape, (4,))
            self.assertEqual(records[env_id].wheel_speed_rad_s.shape, (4,))
            self.assertEqual(records[env_id].env_id, env_id)
            self.assertAlmostEqual(records[env_id].timestamp, 3.5, places=15)

    def test_get_last_command_can_select_envs(self) -> None:
        adapter = make_adapter(num_envs=3)
        adapter.process(safe_command([[0.1, 0.0, 0.0], [0.2, 0.0, 0.0], [0.3, 0.0, 0.0]]))
        selected = adapter.get_last_command([2, 0])
        self.assertEqual(len(selected), 2)
        self.assertEqual(selected[0].env_id, 2)
        self.assertEqual(selected[1].env_id, 0)
        np.testing.assert_allclose(selected[0].body_twist, [0.3, 0.0, 0.0], atol=1e-15)
        with self.assertRaises(ValueError):
            adapter.get_last_command([3])


class FeedbackContractTests(unittest.TestCase):
    def test_feedback_shapes_frame_and_timestamp(self) -> None:
        adapter = make_adapter(num_envs=2)
        feedback = adapter.process(safe_command([[0.2, 0.0, 0.0], [0.2, 0.0, 0.0]],
                                                timestamp=1.25))
        self.assertEqual(feedback.frame, 'court')
        self.assertAlmostEqual(feedback.timestamp, 1.25, places=15)
        self.assertEqual(feedback.prediction_error.shape, (2,))
        self.assertEqual(feedback.contact_detected.shape, (2,))
        self.assertEqual(feedback.contact_detected.dtype, np.dtype(bool))
        # no sensor is attached to this adapter: no contact may be reported out of nowhere
        self.assertFalse(bool(np.any(feedback.contact_detected)))

    def test_contact_events_are_reported_per_env_and_cleared_by_reset(self) -> None:
        adapter = make_adapter(num_envs=2)
        adapter.process(safe_command([[0.1, 0.0, 0.0], [0.1, 0.0, 0.0]]))
        adapter.report_contact([0, 1])
        feedback = adapter.process(safe_command([[0.1, 0.0, 0.0], [0.1, 0.0, 0.0]]))
        np.testing.assert_array_equal(feedback.contact_detected, [True, True])
        adapter.reset([0])
        feedback = adapter.process(safe_command([[0.1, 0.0, 0.0], [0.1, 0.0, 0.0]]))
        np.testing.assert_array_equal(feedback.contact_detected, [False, True])


class ResetIsolationTests(unittest.TestCase):
    def test_reset_clears_only_the_selected_env_command_cache(self) -> None:
        adapter = make_adapter(DriveMode.STEER_DRIVE_WHEEL_MODEL, num_envs=2)
        adapter.process(safe_command([[0.0, 0.3, 0.0], [0.0, 0.3, 0.0]]))
        kept = adapter.get_last_command()[0]

        adapter.reset([1])
        records = adapter.get_last_command()
        self.assertIs(records[0], kept)
        np.testing.assert_allclose(records[0].body_twist, [0.0, 0.3, 0.0], atol=1e-15)
        self.assertIsNone(records[1])

        # env 0 kept its steering state (S11 memory), env 1 was re-initialised
        adapter.process(safe_command([[0.0, -0.3, 0.0], [0.0, -0.3, 0.0]]))
        records = adapter.get_last_command()
        np.testing.assert_allclose(records[0].steer_angle_rad, math.pi / 2.0, atol=1e-12)
        np.testing.assert_allclose(records[1].steer_angle_rad, -math.pi / 2.0, atol=1e-12)

    def test_reset_rejects_out_of_range_env_ids(self) -> None:
        adapter = make_adapter(num_envs=2)
        with self.assertRaises(ValueError):
            adapter.reset([2])


class ModeConfigurationTests(unittest.TestCase):
    def test_drive_mode_is_configurable_and_switchable(self) -> None:
        adapter = make_adapter(DriveMode.BODY_TWIST_ACTUATOR)
        self.assertIs(adapter.drive_mode, DriveMode.BODY_TWIST_ACTUATOR)
        adapter.set_drive_mode(DriveMode.STEER_DRIVE_WHEEL_MODEL)
        self.assertIs(adapter.drive_mode, DriveMode.STEER_DRIVE_WHEEL_MODEL)
        adapter.process(safe_command([0.1, 0.0, 0.0]))
        self.assertTrue(adapter.get_last_command()[0].wheel_level_authoritative)

    def test_drive_mode_accepts_the_documented_string_value(self) -> None:
        adapter = SimExecutionAdapter(num_envs=1, drive_mode='STEER_DRIVE_WHEEL_MODEL',
                                      wheel_positions=WHEEL_POSITIONS, wheel_radius_m=R_W)
        self.assertIs(adapter.drive_mode, DriveMode.STEER_DRIVE_WHEEL_MODEL)

    def test_invalid_drive_mode_is_rejected(self) -> None:
        with self.assertRaises(ValueError):
            make_adapter('TANK_DRIVE')
        with self.assertRaises(ValueError):
            make_adapter().set_drive_mode('TANK_DRIVE')

    def test_default_mode_is_the_task_level_body_twist_mode(self) -> None:
        adapter = SimExecutionAdapter(num_envs=1, wheel_positions=WHEEL_POSITIONS,
                                      wheel_radius_m=R_W)
        self.assertIs(adapter.drive_mode, DriveMode.BODY_TWIST_ACTUATOR)

    def test_unmeasured_geometry_stays_marked_requires_measurement(self) -> None:
        adapter = make_adapter()
        statuses = {p.status.value for p in (adapter.wheel_positions_param,
                                             adapter.wheel_radius_param)}
        self.assertTrue(all(s in {'REQUIRES_MEASUREMENT', 'TEMP_PARAMETERIZED_PROXY'}
                            for s in statuses), statuses)
        for param in (adapter.wheel_positions_param, adapter.wheel_radius_param):
            self.assertTrue(param.source.strip(), 'a Param must state its source')


class FrameConventionGuardTests(unittest.TestCase):
    """Regression guard for the coordinator ruling on the base_twist frame.

    Ruling: WholeBodyTarget.base_twist / SafeCommand.base_twist are the robot_base (body) frame
    twist [vx_body, vy_body, wz]; the court -> body rotation is the PLANNER's duty (T7).  The
    execution adapter must therefore hand the numbers it receives straight to the body-frame
    four-steer IK and must never re-rotate (or require a court-frame input).
    """
    YAW = math.pi / 2.0                                  # robot at yaw = 90 deg
    COURT_INTENT = np.array([0.5, 0.0])                  # move +X, as seen in the court frame
    BODY_TWIST = np.array([0.0, -0.5, 0.0])              # what the planner must emit

    def test_module_documents_the_body_frame_twist_convention(self) -> None:
        doc = sim_adapter_module.__doc__ or ''
        for phrase in ('robot_base (body) frame', 'planner', 'yaw'):
            self.assertIn(phrase, doc,
                          'the adapter docstring must state the frozen frame convention '
                          '(coordinator ruling): input is the robot_base twist and the planner '
                          'owns the court -> body rotation')

    def test_the_ruling_maps_a_court_intent_to_the_body_twist(self) -> None:
        body_planar = court_to_body_planar(self.COURT_INTENT, self.YAW)
        np.testing.assert_allclose(body_planar, self.BODY_TWIST[:2], atol=1e-15)

    def test_adapter_forwards_the_body_twist_to_the_body_frame_ik(self) -> None:
        adapter = make_adapter()
        adapter.process(safe_command([self.BODY_TWIST]))
        record = adapter.get_last_command()[0]
        expected = canonical(self.BODY_TWIST)
        for index, wheel in enumerate(canonical_kinematics.WHEEL_ORDER):
            self.assertAlmostEqual(record.steer_angle_rad[index],
                                   expected[wheel].steer_angle_rad, places=12)
            self.assertAlmostEqual(record.wheel_speed_rad_s[index],
                                   expected[wheel].wheel_speed_rad_s, places=12)
        np.testing.assert_allclose(record.body_twist, self.BODY_TWIST, atol=1e-15)

    def test_court_velocity_fed_straight_into_the_body_ik_is_a_different_command(self) -> None:
        """Negative guard: the bug the ruling fixes is visible in the steer angles."""
        adapter = make_adapter()
        adapter.process(safe_command([self.BODY_TWIST]))
        record = adapter.get_last_command()[0]
        wrong = canonical(np.array([self.COURT_INTENT[0], self.COURT_INTENT[1], 0.0]))
        for index, wheel in enumerate(canonical_kinematics.WHEEL_ORDER):
            self.assertAlmostEqual(
                abs(record.steer_angle_rad[index] - wrong[wheel].steer_angle_rad),
                self.YAW, places=12,
                msg='every wheel is steered wrong by exactly the yaw angle')
            # wz = 0: a pure translation has the same wheel-rate MAGNITUDE in either frame,
            # so the wheel rate alone can never reveal this frame error.
            self.assertAlmostEqual(abs(record.wheel_speed_rad_s[index]),
                                   abs(wrong[wheel].wheel_speed_rad_s), places=12)

    def test_wheel_rates_reveal_the_frame_error_when_the_robot_also_turns(self) -> None:
        yaw_rate = 0.5
        body_twist = np.array([0.0, -0.5, yaw_rate])
        court_twist = np.array([self.COURT_INTENT[0], self.COURT_INTENT[1], yaw_rate])
        body_rates = np.array([canonical(body_twist)[wheel].wheel_speed_rad_s
                               for wheel in canonical_kinematics.WHEEL_ORDER])
        court_rates = np.array([canonical(court_twist)[wheel].wheel_speed_rad_s
                                for wheel in canonical_kinematics.WHEEL_ORDER])
        self.assertFalse(np.allclose(body_rates, court_rates, atol=1e-9))
        gap = float(np.max(np.abs(body_rates - court_rates)) / np.max(np.abs(body_rates)))
        self.assertGreater(gap, 0.01, 'the two frame interpretations must be measurably different')

        adapter = make_adapter()
        adapter.process(safe_command([body_twist]))
        record = adapter.get_last_command()[0]
        np.testing.assert_allclose(record.wheel_speed_rad_s, body_rates, atol=1e-12)
        np.testing.assert_allclose(record.steer_angle_rad,
                                   [canonical(body_twist)[wheel].steer_angle_rad
                                    for wheel in canonical_kinematics.WHEEL_ORDER], atol=1e-12)
        print('[frame guard] yaw=90deg, wz=%s: body wheel rates %s rad/s vs court-passed-through %s '
              'rad/s (max gap %.1f%%)' % (yaw_rate, np.round(body_rates, 3),
                                          np.round(court_rates, 3), 100.0 * gap))


if __name__ == '__main__':
    unittest.main()
