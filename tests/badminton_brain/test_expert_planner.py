# -*- coding: utf-8 -*-
"""T7 expert planner + PPO policy stub (plan row T7).

Spec: docs/superpowers/plans/2026-09-13-brain-modules.md (T7 row),
      docs/architecture/ROBOT_BRAIN.md S11/S12/S15/S43,
      docs/simulation/BADMINTON_ROBOT.md S10/S11 (Morph One four-steer/four-drive limits).

Acceptance asserted here:
  1. an intercept yields WholeBodyTarget(base_twist (N,3), joint_position_target (N,6));
  2. the commanded twist, mapped by the frozen Morph One kinematics, stays inside the
     wheel-speed and steer limits even when the unconstrained command would saturate;
  3. the horizon follows the intercept deadline and stays bounded;
  4. an infeasible decision / missing intercept holds position;
  5. the arm target is a bounded first-order response to the contact-point residual;
  6. the PPO slot is NOT_IMPLEMENTED: it raises when called and final mode refuses it.
"""
from __future__ import annotations
import math
import sys
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'src'))
sys.path.insert(0, str(ROOT / 'simulation'))

from robots.badminton_robot.badminton_robot_cfg import (  # noqa: E402
    WheelId, make_default_robot_cfg,
)
from robots.badminton_robot.frames.robot_frames import quat_from_rpy, quat_to_matrix  # noqa: E402
from robots.badminton_robot.morph_one.kinematics import (  # noqa: E402
    body_twist_to_wheel_targets,
)

from badminton_brain.planning import (  # noqa: E402
    TEMP_CONTACT_JACOBIAN, ExpertPlanner, PpoPolicyStub,
)
from badminton_brain.registry import ModuleRegistry  # noqa: E402
from badminton_brain.status import AssetStatus, Param  # noqa: E402
from badminton_brain.types import (  # noqa: E402
    BestIntercept, BrainBoundaryError, HitDecision, Layer, PredictedTrajectory, UnifiedState,
    WholeBodyTarget,
)
from badminton_brain.validation import validate_architecture  # noqa: E402

N = 3

# TEMP wheel centres, identical to tests/simulation/robots/test_morph_one_kinematics.py.
# MorphOneGeometry.wheel_positions_robot is REQUIRES_MEASUREMENT; these are the Phase-3
# engineering baseline used by the frozen kinematics tests.
WHEEL_POSITIONS = {
    WheelId.FL: (0.25, 0.20),
    WheelId.FR: (0.25, -0.20),
    WheelId.RL: (-0.25, 0.20),
    WheelId.RR: (-0.25, -0.20),
}
R_W = 0.06
MAX_WHEEL_SPEED = 40.0
MAX_STEER = math.pi


def make_trajectory(n=N, t=0.1):
    times = np.linspace(0.0, 0.5, 5)
    return PredictedTrajectory(times=times, position=np.zeros((n, times.shape[0], 3)),
                               velocity=np.zeros((n, times.shape[0], 3)),
                               landing_point=np.zeros((n, 3)), arrival_time=np.zeros((n,)),
                               timestamp=t)


def make_intercept(position=(0.0, 0.0, 1.0), time_s=0.4, n=N, racket_pose=None):
    pose = None if racket_pose is None else np.tile(np.asarray(racket_pose, dtype=float), (n, 1))
    return BestIntercept(position=np.tile(np.asarray(position, dtype=float), (n, 1)),
                         time_s=np.full((n,), float(time_s)), racket_pose=pose,
                         score=np.ones((n,)), timestamp=0.1)


def make_state(n=N, base_xy=(-1.6, 0.0), yaw_deg=0.0, contact=(0.0, 0.0, 1.0), joints=None,
               timestamp=0.1):
    base_pose = np.tile(np.concatenate([[base_xy[0], base_xy[1], 0.0],
                                        quat_from_rpy(0.0, 0.0, math.radians(yaw_deg))]), (n, 1))
    joint_pos = np.zeros((n, 6)) if joints is None else np.tile(np.asarray(joints, float), (n, 1))
    contact_pose = np.concatenate([np.asarray(contact, float), [1.0, 0.0, 0.0, 0.0]])
    return UnifiedState(base_pose=base_pose, base_twist=np.zeros((n, 6)), joint_pos=joint_pos,
                        joint_vel=np.zeros((n, 6)),
                        racket_contact_pose=np.tile(contact_pose, (n, 1)),
                        racket_contact_twist=np.zeros((n, 6)),
                        shuttle_position=np.tile(np.asarray(contact, float), (n, 1)),
                        shuttle_velocity=np.zeros((n, 3)), timestamp=timestamp)


def feasible(t=0.1):
    return HitDecision(feasible=True, reason='canonical incoming inside limits', timestamp=t)


def wheel_ranges(twist, planner):
    """Map a base twist through the frozen kinematics with the planner's own geometry."""
    targets = body_twist_to_wheel_targets(twist, planner.wheel_positions, planner.wheel_radius_m)
    return ([abs(targets[w].wheel_speed_rad_s) for w in WheelId],
            [abs(targets[w].steer_angle_rad) for w in WheelId])


class ExpertPlannerTargetTests(unittest.TestCase):
    """Acceptance 1: contract-shaped base and arm targets from an intercept."""

    def setUp(self):
        self.planner = ExpertPlanner()

    def test_target_shapes_follow_the_whole_body_contract(self):
        target = self.planner.process(make_state(n=N), feasible(),
                                      make_intercept(n=N, time_s=0.4), make_trajectory(n=N))
        self.assertIsInstance(target, WholeBodyTarget)
        self.assertEqual(target.base_twist.shape, (N, 3))
        self.assertEqual(target.joint_position_target.shape, (N, 6))
        self.assertTrue(np.all(np.isfinite(target.base_twist)))
        self.assertTrue(np.all(np.isfinite(target.joint_position_target)))
        self.assertEqual(target.frame, 'court')
        self.assertAlmostEqual(target.timestamp, 0.1, places=12)

    def test_planner_uses_the_frozen_morph_one_geometry(self):
        self.assertEqual(self.planner.wheel_radius_m, R_W)
        for wid, pos in WHEEL_POSITIONS.items():
            self.assertAlmostEqual(self.planner.wheel_positions[wid][0], pos[0], places=12)
            self.assertAlmostEqual(self.planner.wheel_positions[wid][1], pos[1], places=12)

    def test_batch_envs_are_independent_and_the_module_is_deterministic(self):
        state = make_state(n=N)
        intercept = make_intercept(n=N, time_s=0.4)
        first = self.planner.process(state, feasible(), intercept, make_trajectory(n=N))
        second = self.planner.process(state, feasible(), intercept, make_trajectory(n=N))
        np.testing.assert_allclose(first.base_twist, second.base_twist, atol=0.0)
        np.testing.assert_allclose(first.joint_position_target, second.joint_position_target, atol=0.0)
        for row in range(1, N):
            np.testing.assert_allclose(first.base_twist[row], first.base_twist[0], atol=1e-12)


class BaseStationTests(unittest.TestCase):
    """Acceptance 2 + 3: stationing twist inside the steer/drive limits, bounded horizon."""

    def setUp(self):
        self.planner = ExpertPlanner()
        self.standoff = self.planner.limits.standoff_m.value

    def test_approach_and_back_off_follow_the_standoff(self):
        far = self.planner.process(make_state(n=1, base_xy=(-1.6, 0.0)),
                                   feasible(), make_intercept(n=1, position=(-0.7, 0.0, 1.0), time_s=0.4),
                                   make_trajectory(n=1))
        self.assertAlmostEqual(far.base_twist[0, 0], (0.9 - self.standoff) / 0.4, places=9)
        self.assertAlmostEqual(far.base_twist[0, 1], 0.0, places=9)
        self.assertAlmostEqual(far.base_twist[0, 2], 0.0, places=9)
        self.assertAlmostEqual(float(self.planner.last_diagnostics['clip_scale'][0]), 1.0, places=12)

        close = self.planner.process(make_state(n=1, base_xy=(-1.6, 0.0)),
                                     feasible(), make_intercept(n=1, position=(-1.4, 0.0, 1.0), time_s=0.4),
                                     make_trajectory(n=1))
        self.assertAlmostEqual(close.base_twist[0, 0], (0.2 - self.standoff) / 0.4, places=9)
        self.assertLess(close.base_twist[0, 0], 0.0)

    def test_yaw_turns_the_base_forward_axis_towards_the_intercept(self):
        target = self.planner.process(make_state(n=1, base_xy=(-1.6, 0.0), yaw_deg=0.0),
                                      feasible(), make_intercept(n=1, position=(-1.0, 0.5, 1.0), time_s=0.4),
                                      make_trajectory(n=1))
        expected = math.atan2(0.5, 0.6) / 0.4
        self.assertGreater(target.base_twist[0, 2], 0.0)
        self.assertAlmostEqual(target.base_twist[0, 2], expected, places=9)

    def test_saturated_station_command_is_clipped_to_the_wheel_limit(self):
        target = self.planner.process(make_state(n=1, base_xy=(-1.6, 0.0)),
                                      feasible(), make_intercept(n=1, position=(2.0, 0.0, 1.0), time_s=0.4),
                                      make_trajectory(n=1))
        # Unconstrained demand would be (3.6 - standoff)/0.4 m/s; the wheel limit binds.
        self.assertLess(float(self.planner.last_diagnostics['clip_scale'][0]), 1.0)
        # Pure +x on a four-wheel platform: every wheel runs at v/r, so saturation is exactly
        # max_wheel_speed_rad_s * wheel_radius_m.
        self.assertAlmostEqual(target.base_twist[0, 0], MAX_WHEEL_SPEED * R_W, places=9)
        speeds, steers = wheel_ranges(target.base_twist[0], self.planner)
        self.assertLessEqual(max(speeds), MAX_WHEEL_SPEED + 1e-9)
        self.assertLessEqual(max(steers), MAX_STEER + 1e-9)

    def test_every_env_command_respects_the_wheel_limits(self):
        state = make_state(n=N)
        state.base_pose[1, 0] = 0.0
        state.base_pose[1, 1] = -0.4
        intercept = make_intercept(n=N, position=(1.5, 0.6, 1.2), time_s=0.15)
        target = self.planner.process(state, feasible(), intercept, make_trajectory(n=N))
        for row in range(N):
            speeds, steers = wheel_ranges(target.base_twist[row], self.planner)
            self.assertLessEqual(max(speeds), MAX_WHEEL_SPEED + 1e-9)
            self.assertLessEqual(max(steers), MAX_STEER + 1e-9)

    def test_horizon_follows_the_intercept_deadline_and_is_bounded(self):
        limits = self.planner.limits
        on_time = self.planner.process(make_state(n=1), feasible(),
                                       make_intercept(n=1, time_s=0.4), make_trajectory(n=1))
        self.assertAlmostEqual(on_time.horizon_s, 0.4, places=12)

        late = self.planner.process(make_state(n=1), feasible(),
                                    make_intercept(n=1, time_s=5.0), make_trajectory(n=1))
        self.assertAlmostEqual(late.horizon_s, limits.max_horizon_s.value, places=12)

        now = self.planner.process(make_state(n=1), feasible(),
                                   make_intercept(n=1, time_s=0.0), make_trajectory(n=1))
        self.assertAlmostEqual(now.horizon_s, limits.min_horizon_s.value, places=12)

        mixed = self.planner.process(
            make_state(n=N), feasible(),
            BestIntercept(position=np.zeros((N, 3)), time_s=np.array([0.5, 0.2, 0.9]),
                          racket_pose=None, score=np.ones((N,)), timestamp=0.1),
            make_trajectory(n=N))
        self.assertAlmostEqual(mixed.horizon_s, 0.2, places=12)

    def test_infeasible_or_missing_intercept_holds_position(self):
        joints = np.array([0.1, -0.2, 0.3, 0.4, 0.5, -0.6])
        for decision, intercept in ((HitDecision(feasible=False, reason='too late', timestamp=0.1),
                                     make_intercept(n=1, time_s=0.4)),
                                    (feasible(), None)):
            target = self.planner.process(make_state(n=1, joints=joints), decision, intercept,
                                          make_trajectory(n=1))
            np.testing.assert_allclose(target.base_twist, np.zeros((1, 3)), atol=0.0)
            np.testing.assert_allclose(target.joint_position_target[0], joints, atol=0.0)
            self.assertEqual(target.horizon_s, 0.0)


class ArmTargetTests(unittest.TestCase):
    """Acceptance 5: documented first-order geometric/timing approximation (not IK/NMPC)."""

    def setUp(self):
        self.planner = ExpertPlanner()
        self.joints = np.array([0.1, -0.2, 0.3, 0.0, 0.5, -0.1])

    def _delta(self, contact, desired, yaw_deg=0.0, time_s=0.4):
        state = make_state(n=1, yaw_deg=yaw_deg, contact=contact, joints=self.joints)
        racket_pose = list(desired) + [1.0, 0.0, 0.0, 0.0]
        target = self.planner.process(state, feasible(),
                                      make_intercept(n=1, position=desired, time_s=time_s,
                                                     racket_pose=racket_pose),
                                      make_trajectory(n=1))
        return target.joint_position_target[0] - self.joints, target

    def test_first_order_step_reproduces_the_contact_residual(self):
        contact = (0.0, 0.0, 1.0)
        residual = np.array([0.05, 0.0, 0.05])
        delta, _ = self._delta(contact, tuple(np.asarray(contact) + residual))
        self.assertTrue(np.all(np.abs(delta) <= self.planner.limits.max_joint_offset_rad.value + 1e-12))
        np.testing.assert_allclose(TEMP_CONTACT_JACOBIAN @ delta, residual, atol=1e-9)
        self.assertGreater(delta[1], 0.0)   # shoulder pitch lifts towards a higher contact
        self.assertGreater(delta[2], 0.0)   # elbow follows the same lift
        self.assertGreater(delta[4], 0.0)   # wrist pitch

    def test_lateral_residual_is_absorbed_by_the_base_yaw_joint(self):
        delta, _ = self._delta((0.0, 0.0, 1.0), (0.0, 0.05, 1.0))
        self.assertGreater(delta[0], 0.0)
        np.testing.assert_allclose(TEMP_CONTACT_JACOBIAN @ delta, np.array([0.0, 0.05, 0.0]), atol=1e-9)

    def test_residual_is_taken_in_the_base_frame_not_the_court_frame(self):
        # Base yawed 90 deg: a court +y offset is a base +x offset, so the pitch joints lead.
        delta, _ = self._delta((0.0, 0.0, 1.0), (0.0, 0.05, 1.0), yaw_deg=90.0)
        self.assertGreater(delta[1], 0.0)
        np.testing.assert_allclose(TEMP_CONTACT_JACOBIAN @ delta, np.array([0.05, 0.0, 0.0]), atol=1e-8)

    def test_timing_bound_limits_the_arm_step_on_a_short_horizon(self):
        limits = self.planner.limits
        delta_short, _ = self._delta((0.0, 0.0, 1.0), (0.3, 0.0, 1.3), time_s=0.05)
        delta_long, _ = self._delta((0.0, 0.0, 1.0), (0.3, 0.0, 1.3), time_s=0.4)
        rate_bound = limits.max_joint_rate_rad_s.value * 0.05
        self.assertLessEqual(float(np.max(np.abs(delta_short))), rate_bound + 1e-9)
        self.assertGreater(rate_bound, 0.0)
        self.assertLessEqual(float(np.max(np.abs(delta_long))),
                             limits.max_joint_offset_rad.value + 1e-12)
        self.assertGreater(float(np.max(np.abs(delta_long))), float(np.max(np.abs(delta_short))))


class PpoPolicyStubTests(unittest.TestCase):
    """Acceptance 6: the learned-policy slot is explicitly NOT_IMPLEMENTED."""

    def test_stub_declares_planning_layer_and_not_implemented(self):
        stub = PpoPolicyStub()
        self.assertFalse(stub.is_implemented)
        self.assertIs(stub.layer, Layer.PLANNING)
        self.assertEqual(stub.name, 'ppo_policy')

    def test_calling_the_stub_raises_with_a_reason(self):
        stub = PpoPolicyStub()
        with self.assertRaises(NotImplementedError) as ctx:
            stub.process(make_state(n=N), feasible(), make_intercept(n=N), make_trajectory(n=N))
        message = str(ctx.exception)
        self.assertIn('ppo', message.lower())
        self.assertGreater(len(message), 80)

    def test_expert_planner_registers_cleanly_but_the_stub_is_refused_in_final_mode(self):
        expert_registry = ModuleRegistry()
        expert_registry.register(ExpertPlanner())
        development = validate_architecture(expert_registry, mode='development')
        self.assertFalse(any('not implemented' in warning.lower() for warning in development.warnings))

        stub_registry = ModuleRegistry()
        stub_registry.register(PpoPolicyStub())
        self.assertIs(stub_registry.get(Layer.PLANNING), stub_registry.modules()[0])
        final = validate_architecture(stub_registry, mode='final')
        self.assertFalse(final.ok)
        self.assertTrue(any('not implemented' in error.lower() and 'planning' in error.lower()
                            for error in final.errors))


class PlannerContractGuardTests(unittest.TestCase):
    """Guards: the planner never fabricates geometry and never silently redefines the contract."""

    def test_resolved_config_values_are_adopted_with_their_own_provenance(self):
        cfg = make_default_robot_cfg()
        planner = ExpertPlanner(cfg=cfg)
        self.assertAlmostEqual(planner.wheel_radius_m, 0.06, places=12)
        self.assertAlmostEqual(planner.max_wheel_speed_rad_s, 40.0, places=12)
        self.assertAlmostEqual(planner.max_steer_angle_rad, math.pi, places=12)
        # Provenance travels with the value: the config's Param (status + source) is adopted.
        self.assertEqual(planner.limits.wheel_radius_m.source, cfg.morph_one.geometry.wheel_radius_m.source)
        self.assertEqual(planner.limits.max_wheel_speed_rad_s.source, cfg.morph_one.max_wheel_speed_rad_s.source)
        self.assertIs(planner.limits.wheel_radius_m.status, cfg.morph_one.geometry.wheel_radius_m.status)
        # Wheel centres stay the labelled TEMP fallback: the config field is REQUIRES_MEASUREMENT.
        self.assertEqual(sorted(w.value for w in planner.wheel_positions), ['FL', 'FR', 'RL', 'RR'])
        self.assertIn('TEMP', planner.wheel_geometry_source)

    def test_mismatched_batch_or_negative_deadline_is_refused(self):
        planner = ExpertPlanner()
        with self.assertRaises(BrainBoundaryError):
            planner.process(make_state(n=1), feasible(), make_intercept(n=N, time_s=0.4),
                            make_trajectory(n=1))
        negative = BestIntercept(position=np.zeros((1, 3)), time_s=np.array([-0.1]),
                                 score=np.ones((1,)), timestamp=0.1)
        with self.assertRaises(BrainBoundaryError):
            planner.process(make_state(n=1), feasible(), negative, make_trajectory(n=1))


def wrap_to_pi(angle):
    return (angle + math.pi) % (2.0 * math.pi) - math.pi


class BodyFrameTwistTests(unittest.TestCase):
    """Coordinator verdict: WholeBodyTarget.base_twist is robot_base, never court.

    Execution receives only SafeCommand (no pose), so the planner - which holds UnifiedState
    with the base yaw - must do the court -> body rotation itself before wheel clipping.
    """

    def setUp(self):
        self.planner = ExpertPlanner()
        self.standoff = self.planner.limits.standoff_m.value

    def _spec(self, base_xy, yaw_deg, hit_xy, time_s):
        """The command the spec demands, derived from geometry only (not from the planner)."""
        rotation = quat_to_matrix(quat_from_rpy(0.0, 0.0, math.radians(yaw_deg)))
        offset = np.asarray(hit_xy, dtype=float) - np.asarray(base_xy, dtype=float)
        distance = float(np.linalg.norm(offset))
        velocity_court = (distance - self.standoff) / time_s * (offset / distance)
        yaw_rate = wrap_to_pi(math.atan2(offset[1], offset[0]) - math.radians(yaw_deg)) / time_s
        return rotation, velocity_court, yaw_rate

    def test_translation_is_rotated_into_the_body_frame_at_every_yaw(self):
        base_xy, hit_xy, time_s = (-1.6, 0.0), (-0.7, 0.0), 0.9
        for yaw_deg in (0.0, 30.0, 45.0, 90.0, 135.0, 180.0):
            with self.subTest(yaw_deg=yaw_deg):
                rotation, velocity_court, yaw_rate = self._spec(base_xy, yaw_deg, hit_xy, time_s)
                target = self.planner.process(
                    make_state(n=1, base_xy=base_xy, yaw_deg=yaw_deg), feasible(),
                    make_intercept(n=1, position=(hit_xy[0], hit_xy[1], 1.0), time_s=time_s),
                    make_trajectory(n=1))
                # No clipping in this scenario, so the command must match the spec exactly.
                self.assertAlmostEqual(float(self.planner.last_diagnostics['clip_scale'][0]), 1.0,
                                       places=12)
                np.testing.assert_allclose(target.base_twist[0, :2],
                                           rotation[:2, :2].T @ velocity_court, atol=1e-9)
                # Round trip: expressing the command back in the court frame restores the demand.
                np.testing.assert_allclose(rotation[:2, :2] @ target.base_twist[0, :2],
                                           velocity_court, atol=1e-9)
                self.assertAlmostEqual(target.base_twist[0, 2], yaw_rate, places=9)

    def test_the_rotation_changes_the_steer_angles_but_not_the_wheel_speeds(self):
        # Physical consistency: the same motion expressed in two frames drives the same wheels.
        # Expressing it in the court frame rotates the wheel positions together with the twist,
        # which shifts every steer angle by +yaw and leaves every wheel speed unchanged.
        base_xy, hit_xy, time_s = (-1.6, 0.0), (-0.7, 0.0), 0.9
        yaw_deg = 90.0
        yaw = math.radians(yaw_deg)
        rotation, velocity_court, yaw_rate = self._spec(base_xy, yaw_deg, hit_xy, time_s)
        target = self.planner.process(make_state(n=1, base_xy=base_xy, yaw_deg=yaw_deg), feasible(),
                                      make_intercept(n=1, position=(hit_xy[0], hit_xy[1], 1.0),
                                                     time_s=time_s),
                                      make_trajectory(n=1))
        body_twist = target.base_twist[0]
        court_twist = np.array([velocity_court[0], velocity_court[1], yaw_rate])
        planar = rotation[:2, :2]
        court_positions = {wid: tuple(planar @ np.asarray(pos, dtype=float))
                           for wid, pos in self.planner.wheel_positions.items()}
        body_targets = body_twist_to_wheel_targets(body_twist, self.planner.wheel_positions,
                                                   self.planner.wheel_radius_m)
        court_targets = body_twist_to_wheel_targets(court_twist, court_positions,
                                                    self.planner.wheel_radius_m)
        for wid in WheelId:
            self.assertAlmostEqual(body_targets[wid].wheel_speed_rad_s,
                                   court_targets[wid].wheel_speed_rad_s, places=9)
            self.assertAlmostEqual(wrap_to_pi(court_targets[wid].steer_angle_rad
                                              - body_targets[wid].steer_angle_rad),
                                   wrap_to_pi(yaw), places=9)
        # Feeding the court-frame twist straight into the body-frame kinematics (the pre-verdict
        # behaviour) is a *different* command, not a re-labelling - that is why the fix matters.
        misframed = body_twist_to_wheel_targets(court_twist, self.planner.wheel_positions,
                                                self.planner.wheel_radius_m)
        worst = max(abs(misframed[wid].wheel_speed_rad_s - body_targets[wid].wheel_speed_rad_s)
                    for wid in WheelId)
        self.assertGreater(worst, 0.5)
        self.assertLessEqual(max(wheel_ranges(body_twist, self.planner)[1]), MAX_STEER + 1e-9)


class SteerLimitTests(unittest.TestCase):
    """A steer limit tighter than the TEMP pi default must really be enforced and reported."""

    @staticmethod
    def _tight_planner():
        cfg = make_default_robot_cfg()
        cfg.morph_one.max_steer_angle_rad = Param(
            1.0, AssetStatus.TEMP_PARAMETERIZED_PROXY,
            'test fixture: artificially tight steer limit to exercise the steer check')
        return ExpertPlanner(cfg=cfg)

    def test_a_tighter_steer_limit_is_adopted_flagged_and_still_speed_limited(self):
        planner = self._tight_planner()
        self.assertAlmostEqual(planner.max_steer_angle_rad, 1.0, places=12)
        state = make_state(n=1, base_xy=(-1.6, 0.0), yaw_deg=0.0)
        # Pure lateral intercept: the required wheel directions sit near +-pi/2, which no limit
        # below pi/2 can represent, not even by the equivalent (theta +- pi, -omega) command.
        intercept = make_intercept(n=1, position=(-1.6, 0.9, 1.0), time_s=0.9)
        target = planner.process(state, feasible(), intercept, make_trajectory(n=1))
        self.assertTrue(bool(planner.last_diagnostics['steer_limit_exceeded'][0]))
        self.assertGreater(float(planner.last_diagnostics['steer_equivalent_max_rad'][0]), 1.0)
        speeds, steers = wheel_ranges(target.base_twist[0], planner)
        self.assertLessEqual(max(speeds), MAX_WHEEL_SPEED + 1e-9)   # the drive limit still holds
        self.assertLessEqual(max(steers), MAX_STEER + 1e-9)

        relaxed = ExpertPlanner()   # the continuous-steering baseline meets the same command
        relaxed.process(state, feasible(), intercept, make_trajectory(n=1))
        self.assertFalse(bool(relaxed.last_diagnostics['steer_limit_exceeded'][0]))


class MeasurementRequirementTests(unittest.TestCase):
    """Alignment with T5/T6: the unmeasured list is a queryable API, not prose."""

    def test_measurement_requirements_lists_every_unmeasured_quantity(self):
        planner = ExpertPlanner()
        requirements = planner.measurement_requirements()
        self.assertIsInstance(requirements, dict)
        for name in ('standoff_m', 'max_joint_offset_rad', 'max_joint_rate_rad_s',
                     'max_wheel_speed_rad_s', 'max_steer_angle_rad', 'wheel_radius_m',
                     'wheel_positions_robot', 'contact_jacobian', 'piper_joint_limits'):
            self.assertIn(name, requirements)
        for name, param in requirements.items():
            self.assertIsInstance(param, Param, name)
            self.assertIsNone(param.value, name)
            self.assertIs(param.status, AssetStatus.REQUIRES_MEASUREMENT, name)
            self.assertTrue(str(param.source).strip(), name)
        temp_limits = [name for name, param in planner.limits.param_limits().items()
                       if param.status == AssetStatus.TEMP_PARAMETERIZED_PROXY]
        self.assertTrue(temp_limits)
        for name in temp_limits:
            self.assertIn(name, requirements)

    def test_supplied_geometry_drops_the_matching_requirement(self):
        measured_jacobian = np.hstack([np.diag([0.3, 0.3, 0.3]), np.zeros((3, 3))])
        planner = ExpertPlanner(wheel_positions=WHEEL_POSITIONS, contact_jacobian=measured_jacobian)
        requirements = planner.measurement_requirements()
        self.assertNotIn('wheel_positions_robot', requirements)
        self.assertNotIn('contact_jacobian', requirements)
        self.assertIn('wheel_radius_m', requirements)   # handed-in geometry is not a measurement

        target = planner.process(
            make_state(n=1, contact=(0.0, 0.0, 1.0), joints=np.zeros(6)), feasible(),
            make_intercept(n=1, position=(0.0, 0.0, 1.05), time_s=0.4,
                           racket_pose=[0.0, 0.0, 1.05, 1.0, 0.0, 0.0, 0.0]),
            make_trajectory(n=1))
        np.testing.assert_allclose(target.joint_position_target[0],
                                   [0.0, 0.0, 0.05 / 0.3, 0.0, 0.0, 0.0], atol=1e-12)

        # Handing the TEMP proxy back in is not a measurement.
        still_temp = ExpertPlanner(contact_jacobian=TEMP_CONTACT_JACOBIAN)
        self.assertIn('contact_jacobian', still_temp.measurement_requirements())


if __name__ == '__main__':
    unittest.main()
