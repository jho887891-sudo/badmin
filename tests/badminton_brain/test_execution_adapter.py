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
import warnings
from dataclasses import replace
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
from badminton_brain.status import AssetStatus, Param  # noqa: E402
from badminton_brain.types import BrainBoundaryError, Layer, SafeCommand  # noqa: E402
from robots.badminton_robot.badminton_robot_cfg import (  # noqa: E402
    DriveMode,
    WheelId,
    make_default_robot_cfg,
)
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


def measured_cfg(positions=None, radius=0.061):
    """A robot cfg whose Morph One wheel geometry is really measured (S4 VERIFIED_MEASURED)."""
    cfg = make_default_robot_cfg()
    geometry = replace(
        cfg.morph_one.geometry,
        wheel_positions_robot=Param(dict(positions if positions is not None else WHEEL_POSITIONS),
                                    AssetStatus.VERIFIED_MEASURED,
                                    'wheel centres measured on the real chassis'),
        wheel_radius_m=Param(float(radius), AssetStatus.VERIFIED_MEASURED,
                             'wheel radius measured with calipers'))
    return replace(cfg, morph_one=replace(cfg.morph_one, geometry=geometry))


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

    def test_tracking_residual_is_the_wheel_level_realisation_residual(self) -> None:
        """DEC-015: the execution layer reports its own tracking residual, never a prediction."""
        adapter = make_adapter(DriveMode.STEER_DRIVE_WHEEL_MODEL, num_envs=2)
        feedback = adapter.process(safe_command([[0.3, -0.2, 0.5], [0.0, 0.0, 0.0]]))
        self.assertEqual(feedback.tracking_residual.shape, (2,))
        self.assertTrue(np.all(np.isfinite(feedback.tracking_residual)))
        self.assertTrue(np.all(feedback.tracking_residual <= 1e-9),
                        f'wheel targets do not realise the commanded twist: '
                        f'{feedback.tracking_residual}')
        # recomputed with the same canonical kinematics (no re-implementation of the adapter)
        for env_id in range(2):
            record = adapter.get_last_command()[env_id]
            wire = {wid: canonical_kinematics.WheelTargets(record.steer_angle_rad[index],
                                                           record.wheel_speed_rad_s[index],
                                                           record.wheel_tangential_speed_mps[index])
                    for index, wid in enumerate(canonical_kinematics.WHEEL_ORDER)}
            expected = float(np.linalg.norm(
                wheel_targets_to_body_twist(wire, WHEEL_POSITIONS, R_W) - record.body_twist))
            self.assertAlmostEqual(float(feedback.tracking_residual[env_id]), expected, places=12)


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
        # DEC-015: prediction residual is the estimation/prediction side's duty - this adapter
        # has no measurement and must not fabricate one
        self.assertIsNone(feedback.prediction_error)
        self.assertEqual(feedback.tracking_residual.shape, (2,))
        self.assertTrue(np.all(np.isfinite(feedback.tracking_residual)))
        self.assertEqual(feedback.contact_detected.shape, (2,))
        self.assertEqual(feedback.contact_detected.dtype, np.dtype(bool))
        # no sensor is attached to this adapter: no contact may be reported out of nowhere
        self.assertFalse(bool(np.any(feedback.contact_detected)))

    def test_execution_layer_never_fabricates_a_prediction_residual(self) -> None:
        """DEC-015: prediction_error is the shuttle prediction-vs-measurement residual
        (metres / metres per second) and is produced by the estimation/prediction side.
        The execution adapter leaves it unset in every mode, for every command."""
        for mode in (DriveMode.BODY_TWIST_ACTUATOR, DriveMode.STEER_DRIVE_WHEEL_MODEL):
            adapter = make_adapter(mode, num_envs=2)
            adapter.report_contact([0])
            for twist in ([[0.4, 0.0, 0.0], [0.0, 0.3, 0.1]], [[0.0, 0.0, 0.0], [0.1, 0.0, 0.0]]):
                feedback = adapter.process(safe_command(twist, timestamp=1.0))
                self.assertIsNone(feedback.prediction_error)
                self.assertEqual(feedback.tracking_residual.shape, (2,))
                self.assertTrue(np.all(np.isfinite(feedback.tracking_residual)))

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


class ParameterReportingTests(unittest.TestCase):
    """DEC-017 (cfg is the single source of geometry) + DEC-018 (unresolved values are visible
    and TEMP geometry raises a RuntimeWarning)."""

    def test_zero_argument_construction_reports_the_temp_geometry_as_unresolved(self) -> None:
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter('always')
            adapter = SimExecutionAdapter(num_envs=1)
        self.assertEqual(set(dict(adapter.unresolved_limits())),
                         {'wheel_positions_robot', 'wheel_radius_m'})
        self.assertIs(adapter.wheel_positions_param.status, AssetStatus.TEMP_PARAMETERIZED_PROXY)
        self.assertIs(adapter.wheel_radius_param.status, AssetStatus.TEMP_PARAMETERIZED_PROXY)
        requirements = adapter.measurement_requirements()
        self.assertEqual(set(requirements), {'wheel_positions_robot', 'wheel_radius_m'})
        for name, param in requirements.items():
            self.assertIsNone(param.value)
            self.assertIs(param.status, AssetStatus.REQUIRES_MEASUREMENT)
            self.assertIn('measure ' + name, param.source)
        # DEC-018: a zero-argument construction on TEMP geometry must not be silent
        self.assertTrue([w for w in caught if issubclass(w.category, RuntimeWarning)],
                        'TEMP geometry must raise a RuntimeWarning')

    def test_the_temp_warning_names_the_temp_fields(self) -> None:
        with self.assertWarnsRegex(RuntimeWarning, 'wheel_positions_robot'):
            SimExecutionAdapter(num_envs=1)
        with self.assertWarnsRegex(RuntimeWarning, 'wheel_radius_m'):
            SimExecutionAdapter(num_envs=1, wheel_positions=WHEEL_POSITIONS)
        with self.assertWarnsRegex(RuntimeWarning, 'wheel_radius_m'):
            make_adapter()

    def test_measured_geometry_in_the_cfg_clears_the_requirements(self) -> None:
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter('always')
            adapter = SimExecutionAdapter(num_envs=1, cfg=measured_cfg())
        self.assertEqual(adapter.unresolved_limits(), ())
        self.assertEqual(adapter.measurement_requirements(), {})
        self.assertIs(adapter.wheel_positions_param.status, AssetStatus.VERIFIED_MEASURED)
        self.assertIs(adapter.wheel_radius_param.status, AssetStatus.VERIFIED_MEASURED)
        self.assertEqual([w for w in caught if issubclass(w.category, RuntimeWarning)], [])

    def test_measured_geometry_can_be_injected_as_a_param(self) -> None:
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter('always')
            adapter = SimExecutionAdapter(
                num_envs=1,
                wheel_positions=Param(dict(WHEEL_POSITIONS), AssetStatus.VERIFIED_MEASURED,
                                      'measured on the chassis'),
                wheel_radius_m=Param(0.061, AssetStatus.VERIFIED_MEASURED, 'measured radius'))
        self.assertEqual(adapter.unresolved_limits(), ())
        self.assertEqual([w for w in caught if issubclass(w.category, RuntimeWarning)], [])

    def test_caller_supplied_raw_values_stay_labelled_temp(self) -> None:
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            adapter = make_adapter()
        for param in (adapter.wheel_positions_param, adapter.wheel_radius_param):
            self.assertIs(param.status, AssetStatus.TEMP_PARAMETERIZED_PROXY)
            self.assertTrue(param.source.strip(), 'a Param must state its source')
            self.assertIn('caller-supplied', param.source)
        self.assertEqual(set(dict(adapter.unresolved_limits())),
                         {'wheel_positions_robot', 'wheel_radius_m'})

    def test_cfg_is_the_single_source_of_the_wheel_geometry(self) -> None:
        positions = {WheelId.FL: (0.31, 0.24), WheelId.FR: (0.31, -0.24),
                     WheelId.RL: (-0.31, 0.24), WheelId.RR: (-0.31, -0.24)}
        adapter = SimExecutionAdapter(num_envs=1, cfg=measured_cfg(positions=positions,
                                                                   radius=0.08))
        self.assertAlmostEqual(adapter.wheel_radius_m, 0.08, places=15)
        for wheel, xy in positions.items():
            np.testing.assert_allclose(adapter.wheel_positions[wheel], xy, atol=1e-15)
        # the IK really runs on the cfg geometry, not on a hardcoded copy of it
        adapter.process(safe_command([0.5, 0.0, 0.0]))
        record = adapter.get_last_command()[0]
        expected = body_twist_to_wheel_targets([0.5, 0.0, 0.0], positions, 0.08)
        for index, wheel in enumerate(canonical_kinematics.WHEEL_ORDER):
            self.assertAlmostEqual(record.wheel_speed_rad_s[index],
                                   expected[wheel].wheel_speed_rad_s, places=12)
        self.assertNotAlmostEqual(float(record.wheel_speed_rad_s[0]), 0.5 / R_W, places=6)

    def test_the_temp_proxy_source_names_the_cfg_field_it_falls_back_from(self) -> None:
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            adapter = SimExecutionAdapter(num_envs=1)
        self.assertIn('wheel_positions_robot', adapter.wheel_positions_param.source)
        self.assertIn('REQUIRES_MEASUREMENT', adapter.wheel_positions_param.source)
        # the radius proxy is the cfg engineering baseline itself, not a second copy
        baseline = make_default_robot_cfg().morph_one.geometry.wheel_radius_m
        self.assertIs(adapter.wheel_radius_param.status, baseline.status)
        self.assertEqual(adapter.wheel_radius_param.value, baseline.value)
        self.assertEqual(adapter.wheel_radius_param.source, baseline.source)

    def test_explicit_none_and_omitted_geometry_follow_the_same_path(self) -> None:
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            omitted = SimExecutionAdapter(num_envs=1)
            explicit_none = SimExecutionAdapter(num_envs=1, wheel_positions=None,
                                                wheel_radius_m=None)
            omitted.process(safe_command([0.3, 0.1, 0.0]))
            explicit_none.process(safe_command([0.3, 0.1, 0.0]))
        self.assertEqual(omitted.wheel_positions_param, explicit_none.wheel_positions_param)
        self.assertEqual(omitted.wheel_radius_param, explicit_none.wheel_radius_param)
        self.assertEqual(omitted.unresolved_limits(), explicit_none.unresolved_limits())
        self.assertEqual(set(omitted.measurement_requirements()),
                         set(explicit_none.measurement_requirements()))
        np.testing.assert_allclose(omitted.get_last_command()[0].wheel_speed_rad_s,
                                   explicit_none.get_last_command()[0].wheel_speed_rad_s,
                                   atol=1e-15)
        for kwargs in ({}, {'wheel_positions': None, 'wheel_radius_m': None}):
            with self.assertWarns(RuntimeWarning):
                SimExecutionAdapter(num_envs=1, **kwargs)
        # ... and both adopt a resolved cfg in exactly the same way
        cfg = measured_cfg(radius=0.07)
        adopted = SimExecutionAdapter(num_envs=1, cfg=cfg)
        adopted_none = SimExecutionAdapter(num_envs=1, cfg=cfg, wheel_positions=None,
                                           wheel_radius_m=None)
        self.assertAlmostEqual(adopted.wheel_radius_m, 0.07, places=15)
        self.assertEqual(adopted.measurement_requirements(),
                         adopted_none.measurement_requirements())
        self.assertEqual(adopted.unresolved_limits(), ())


class ExecutionCommandHygieneTests(unittest.TestCase):
    """The cached record must not leak mutable state and must not pretend to be hashable."""

    def test_record_is_not_hashable(self) -> None:
        adapter = make_adapter()
        adapter.process(safe_command([0.1, 0.0, 0.0]))
        record = adapter.get_last_command()[0]
        with self.assertRaises(TypeError) as caught:
            hash(record)
        self.assertIn('ExecutionCommand', str(caught.exception))

    def test_record_arrays_are_read_only(self) -> None:
        adapter = make_adapter()
        adapter.process(safe_command([0.1, 0.2, 0.0]))
        record = adapter.get_last_command()[0]
        arrays = (record.body_twist, record.joint_position_target, record.steer_angle_rad,
                  record.wheel_speed_rad_s, record.wheel_tangential_speed_mps)
        for array in arrays:
            self.assertFalse(array.flags.writeable, 'a cached command must be immutable')
            with self.assertRaises(ValueError):
                array[0] = 123.0
        self.assertIs(record.actuator_target['body_twist'], record.body_twist)
        self.assertFalse(record.as_dict()['wheel_speed_rad_s'].flags.writeable)


if __name__ == '__main__':
    unittest.main()
