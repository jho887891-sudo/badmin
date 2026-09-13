# -*- coding: utf-8 -*-
"""T5 hit feasibility gate - TDD tests (plan docs/superpowers/plans/2026-09-13-brain-modules.md T5).

Reference numbers are taken from the project docs, never invented here:
  * court bounds: configs/court.yaml (13.40 x 6.10, singles 5.18) and
    COORDINATE_SYSTEM.md (X in [-6.70, 6.70], Y in [-3.05, 3.05], net plane X=0).
  * canonical incoming: SIMULATION_ENVIRONMENT.md S62-S64 -> position (1.2, 0, 1.8),
    velocity (-3.0, 0, +1.5), gravity only, lands at x ~ -1.133 m.
  * robot home: configs/court.yaml frames.robot_home_xyz_m = (-1.60, 0, 0).
Every threshold used below is supplied explicitly, either by the module defaults (all
Param with status + source) or by a limits object built inside the test.
"""
from __future__ import annotations

import dataclasses
import importlib
import math
import sys
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'src'))

from badminton_brain.decision.feasibility import (  # noqa: E402
    FeasibilityLimits, HitFeasibilityGate, HitReason, base_min_travel_time_s,
)
from badminton_brain.status import AssetStatus, Param  # noqa: E402
from badminton_brain.types import HitDecision, PredictedTrajectory, UnifiedState  # noqa: E402

G = 9.81
ROBOT_HOME_XYZ = (-1.60, 0.0, 0.0)
CANONICAL_P0 = (1.2, 0.0, 1.8)
CANONICAL_V0 = (-3.0, 0.0, 1.5)
CANONICAL_LANDING_X = -1.133          # SIMULATION_ENVIRONMENT.md S64 (gravity-only reference)


def full_flight(p0, v0, dt=0.01):
    """Gravity-only reference flight (drag off, as in SIMULATION_ENVIRONMENT.md S62)."""
    p0 = np.asarray(p0, dtype=float)
    v0 = np.asarray(v0, dtype=float)
    t_land = (v0[2] + math.sqrt(v0[2] ** 2 + 2.0 * G * p0[2])) / G
    times = np.append(np.arange(0.0, t_land, dt), t_land)
    acc = np.array([0.0, 0.0, -G])
    pos = p0 + v0 * times[:, None] + 0.5 * acc * times[:, None] ** 2
    vel = v0 + acc * times[:, None]
    return times, pos, vel, float(t_land), pos[-1]


def make_trajectory(p0=CANONICAL_P0, v0=CANONICAL_V0, dt=0.01, horizon=None,
                    landing=None, arrival=None, stamp=0.0, shift=0.0):
    """Build a PredictedTrajectory from the gravity-only reference flight."""
    times, pos, vel, t_land, land = full_flight(p0, v0, dt)
    if horizon is not None:
        keep = times <= horizon + 1e-12
        times, pos, vel = times[keep], pos[keep], vel[keep]
    times = times + shift
    land = np.asarray(landing, dtype=float) if landing is not None else land
    arrival = (float(arrival) if arrival is not None else t_land) + shift
    return PredictedTrajectory(times=times, position=pos[None], velocity=vel[None],
                               landing_point=land[None], arrival_time=np.array([arrival]),
                               timestamp=stamp)


def make_state(base_xyz=ROBOT_HOME_XYZ, p=CANONICAL_P0, v=CANONICAL_V0, timestamp=0.0):
    return UnifiedState(base_pose=np.array([[base_xyz[0], base_xyz[1], base_xyz[2],
                                             0.0, 0.0, 0.0, 1.0]]),
                        base_twist=np.zeros((1, 6)), joint_pos=np.zeros((1, 6)),
                        joint_vel=np.zeros((1, 6)), racket_contact_pose=np.zeros((1, 7)),
                        racket_contact_twist=np.zeros((1, 6)),
                        shuttle_position=np.asarray(p, dtype=float)[None],
                        shuttle_velocity=np.asarray(v, dtype=float)[None],
                        timestamp=timestamp)


class AcceptTests(unittest.TestCase):
    """The canonical incoming must be playable (plan T5 acceptance)."""

    def test_canonical_landing_matches_the_documented_reference(self) -> None:
        traj = make_trajectory()
        self.assertAlmostEqual(float(traj.landing_point[0, 0]), CANONICAL_LANDING_X, places=2)
        self.assertAlmostEqual(float(traj.arrival_time[0]), 0.778, places=2)

    def test_canonical_incoming_is_accepted(self) -> None:
        gate = HitFeasibilityGate()
        decision = gate.evaluate(make_state(), make_trajectory())
        self.assertIsInstance(decision, HitDecision)
        self.assertTrue(decision.feasible, f"rejected with {decision.reason}")
        self.assertEqual(decision.reason, HitReason.FEASIBLE.value)
        self.assertEqual(decision.frame, 'court')
        self.assertAlmostEqual(decision.timestamp, 0.0, places=9)

    def test_accepting_with_the_base_already_at_the_station(self) -> None:
        # Same canonical ball, base parked near the net-side station: no base travel needed.
        gate = HitFeasibilityGate()
        decision = gate.evaluate(make_state(base_xyz=(-1.30, 0.0, 0.0)), make_trajectory())
        self.assertTrue(decision.feasible, decision.reason)


class RejectionTests(unittest.TestCase):
    """Each hard criterion is rejected with its own stable reason."""

    def setUp(self) -> None:
        self.gate = HitFeasibilityGate()

    def test_out_of_bounds_landing_behind_the_baseline(self) -> None:
        decision = self.gate.evaluate(make_state(),
                                      make_trajectory(landing=(-7.5, 0.0, 0.0)))
        self.assertFalse(decision.feasible)
        self.assertEqual(decision.reason, HitReason.OUT_OF_BOUNDS.value)

    def test_out_of_bounds_landing_beyond_the_doubles_sideline(self) -> None:
        decision = self.gate.evaluate(make_state(),
                                      make_trajectory(landing=(-1.0, 3.40, 0.0)))
        self.assertEqual(decision.reason, HitReason.OUT_OF_BOUNDS.value)

    def test_singles_boundary_mode_is_configurable(self) -> None:
        landing = (-1.0, 2.80, 0.0)      # inside doubles (3.05), outside singles (2.59)
        doubles = self.gate.evaluate(make_state(), make_trajectory(landing=landing))
        self.assertNotEqual(doubles.reason, HitReason.OUT_OF_BOUNDS.value)
        singles = HitFeasibilityGate(dataclasses.replace(FeasibilityLimits(),
                                                         boundary_mode='singles'))
        rejected = singles.evaluate(make_state(), make_trajectory(landing=landing))
        self.assertEqual(rejected.reason, HitReason.OUT_OF_BOUNDS.value)

    def test_opponent_side_landing_is_outside_responsibility(self) -> None:
        decision = self.gate.evaluate(make_state(),
                                      make_trajectory(landing=(5.0, 0.0, 0.0)))
        self.assertEqual(decision.reason, HitReason.OUTSIDE_RESPONSIBILITY.value)

    def test_overspeed_is_rejected(self) -> None:
        decision = self.gate.evaluate(make_state(),
                                      make_trajectory(p0=(1.0, 0.0, 1.5), v0=(-30.0, 0.0, 10.0)))
        self.assertFalse(decision.feasible)
        self.assertEqual(decision.reason, HitReason.SHUTTLE_TOO_FAST.value)

    def test_too_slow_is_rejected(self) -> None:
        decision = self.gate.evaluate(make_state(),
                                      make_trajectory(p0=(1.0, 0.0, 1.5), v0=(-0.2, 0.0, 0.0)))
        self.assertEqual(decision.reason, HitReason.SHUTTLE_TOO_SLOW.value)

    def test_receding_shuttle_is_the_wrong_direction(self) -> None:
        decision = self.gate.evaluate(make_state(p=(2.0, 0.0, 1.5), v=(2.0, 0.0, 1.0)),
                                      make_trajectory(p0=(2.0, 0.0, 1.5), v0=(2.0, 0.0, 1.0)))
        self.assertEqual(decision.reason, HitReason.WRONG_DIRECTION.value)

    def test_too_late_when_the_base_is_far_away(self) -> None:
        # Same canonical ball, but the base starts 3.4 m from any useful station:
        # the flight ends before the base+arm can arrive.
        decision = self.gate.evaluate(make_state(base_xyz=(-5.0, 0.0, 0.0)),
                                      make_trajectory())
        self.assertFalse(decision.feasible)
        self.assertEqual(decision.reason, HitReason.NO_TIME_MARGIN.value)

    def test_flight_that_already_ended_is_too_late(self) -> None:
        decision = self.gate.evaluate(make_state(), make_trajectory(shift=-1.0))
        self.assertEqual(decision.reason, HitReason.NO_TIME_MARGIN.value)

    def test_ball_above_the_workspace_is_unreachable(self) -> None:
        # High lob: every predicted sample stays above the racket workspace box.
        decision = self.gate.evaluate(make_state(p=(-1.0, 0.0, 3.5), v=(-0.2, 0.0, 1.0)),
                                      make_trajectory(p0=(-1.0, 0.0, 3.5), v0=(-0.2, 0.0, 1.0),
                                                      horizon=0.50))
        self.assertFalse(decision.feasible)
        self.assertEqual(decision.reason, HitReason.UNREACHABLE.value)

    def test_outside_the_workspace_box_with_tight_limits(self) -> None:
        limits = dataclasses.replace(
            FeasibilityLimits(),
            arm_reach_y_m=Param(0.30, AssetStatus.TEMP_PARAMETERIZED_PROXY,
                                'test-only tight workspace for the unreachable case'),
            station_y_range_m=Param((-0.20, 0.20), AssetStatus.TEMP_PARAMETERIZED_PROXY,
                                    'test-only narrow station box'))
        gate = HitFeasibilityGate(limits)
        decision = gate.evaluate(make_state(),
                                 make_trajectory(p0=(1.2, 1.5, 1.8), v0=(-3.0, 0.0, 1.5),
                                                 landing=(-1.0, 1.5, 0.0)))
        self.assertEqual(decision.reason, HitReason.UNREACHABLE.value)

    def test_invalid_state_is_rejected(self) -> None:
        self.assertEqual(self.gate.evaluate(None, make_trajectory()).reason,
                         HitReason.INVALID_STATE.value)
        self.assertEqual(self.gate.evaluate('not-a-state', make_trajectory()).reason,
                         HitReason.INVALID_STATE.value)

    def test_invalid_prediction_is_rejected(self) -> None:
        self.assertEqual(self.gate.evaluate(make_state(), None).reason,
                         HitReason.INVALID_PREDICTION.value)
        empty = PredictedTrajectory(times=np.zeros(0), position=np.zeros((1, 0, 3)),
                                    velocity=np.zeros((1, 0, 3)), landing_point=np.zeros((1, 3)),
                                    arrival_time=np.zeros(1), timestamp=0.0)
        self.assertEqual(self.gate.evaluate(make_state(), empty).reason,
                         HitReason.INVALID_PREDICTION.value)

    def test_non_finite_velocity_is_rejected(self) -> None:
        # types.py checks finiteness when a message is built, but messages are mutable: a consumer
        # can poison an array afterwards, so the gate re-checks every sample it reads (S17).
        trajectory = make_trajectory()
        trajectory.velocity[0, 0, 0] = np.nan
        decision = self.gate.evaluate(make_state(), trajectory)
        self.assertFalse(decision.feasible)
        self.assertEqual(decision.reason, HitReason.INVALID_PREDICTION.value)

    def test_non_finite_landing_point_is_rejected(self) -> None:
        trajectory = make_trajectory()
        trajectory.landing_point[0, 0] = np.nan
        decision = self.gate.evaluate(make_state(), trajectory)
        self.assertFalse(decision.feasible)
        self.assertEqual(decision.reason, HitReason.INVALID_PREDICTION.value)

    def test_non_finite_arrival_time_is_rejected(self) -> None:
        trajectory = make_trajectory()
        trajectory.arrival_time[0] = np.nan
        decision = self.gate.evaluate(make_state(), trajectory)
        self.assertFalse(decision.feasible)
        self.assertEqual(decision.reason, HitReason.INVALID_PREDICTION.value)

    def test_stale_state_and_prediction_are_rejected(self) -> None:
        stale_state = self.gate.evaluate(make_state(), make_trajectory(), now=0.5)
        self.assertEqual(stale_state.reason, HitReason.STALE_STATE.value)
        stale_pred = self.gate.evaluate(make_state(timestamp=0.5), make_trajectory(), now=0.5)
        self.assertEqual(stale_pred.reason, HitReason.STALE_PREDICTION.value)


class StabilityAndBatchTests(unittest.TestCase):
    def test_reason_is_a_stable_string_enum(self) -> None:
        for member in HitReason:
            self.assertIsInstance(member.value, str)
            self.assertEqual(member.value, str(member.value))
        gate = HitFeasibilityGate()
        first = gate.evaluate(make_state(), make_trajectory(landing=(-7.5, 0.0, 0.0)))
        second = gate.evaluate(make_state(), make_trajectory(landing=(-7.5, 0.0, 0.0)))
        self.assertIsInstance(first.reason, str)
        self.assertEqual(first.reason, HitReason.OUT_OF_BOUNDS.value)
        self.assertEqual(HitReason(first.reason), HitReason.OUT_OF_BOUNDS)
        self.assertEqual(second.reason, first.reason)

    def test_evaluate_batch_is_per_environment(self) -> None:
        near = make_state()
        far = make_state(base_xyz=(-5.0, 0.0, 0.0))
        state = UnifiedState(base_pose=np.vstack([near.base_pose, far.base_pose]),
                             base_twist=np.vstack([near.base_twist, far.base_twist]),
                             joint_pos=np.vstack([near.joint_pos, far.joint_pos]),
                             joint_vel=np.vstack([near.joint_vel, far.joint_vel]),
                             racket_contact_pose=np.vstack([near.racket_contact_pose,
                                                            far.racket_contact_pose]),
                             racket_contact_twist=np.vstack([near.racket_contact_twist,
                                                             far.racket_contact_twist]),
                             shuttle_position=np.tile(near.shuttle_position, (2, 1)),
                             shuttle_velocity=np.tile(near.shuttle_velocity, (2, 1)),
                             timestamp=0.0)
        single = make_trajectory()
        trajectory = PredictedTrajectory(times=single.times,
                                         position=np.tile(single.position, (2, 1, 1)),
                                         velocity=np.tile(single.velocity, (2, 1, 1)),
                                         landing_point=np.tile(single.landing_point, (2, 1)),
                                         arrival_time=np.tile(single.arrival_time, 2),
                                         timestamp=0.0)
        gate = HitFeasibilityGate()
        decisions = gate.evaluate_batch(state, trajectory)
        self.assertEqual(len(decisions), 2)
        self.assertEqual([d.reason for d in decisions],
                         [HitReason.FEASIBLE.value, HitReason.NO_TIME_MARGIN.value])
        self.assertEqual(gate.evaluate(state, trajectory, env_id=1).reason,
                         HitReason.NO_TIME_MARGIN.value)


class LimitDeclarationTests(unittest.TestCase):
    """No threshold may hide how it was obtained (plan rule 3)."""

    def test_every_limit_is_a_param_with_status_and_source(self) -> None:
        params = FeasibilityLimits().param_limits()
        self.assertGreaterEqual(len(params), 15)
        for name, param in params.items():
            self.assertIsInstance(param, Param, name)
            self.assertIsInstance(param.status, AssetStatus, name)
            self.assertTrue(str(param.source).strip(), name)

    def test_temp_limits_declare_what_still_must_be_measured(self) -> None:
        limits = FeasibilityLimits()
        temps = [n for n, p in limits.param_limits().items()
                 if p.status == AssetStatus.TEMP_PARAMETERIZED_PROXY]
        self.assertTrue(temps, 'expected TEMP proxy limits until the rig is measured')
        requirements = limits.measurement_requirements()
        for name in temps:
            self.assertIn(name, requirements)
            self.assertIsNone(requirements[name].value, name)
            self.assertEqual(requirements[name].status, AssetStatus.REQUIRES_MEASUREMENT, name)
            self.assertTrue(str(requirements[name].source).strip(), name)

    def test_court_geometry_is_traceable_to_the_court_config(self) -> None:
        limits = FeasibilityLimits()
        self.assertEqual(limits.court_half_length_m.status, AssetStatus.TRACEABLE_REFERENCE)
        self.assertAlmostEqual(limits.court_half_length_m.value, 13.40 / 2.0, places=9)
        self.assertAlmostEqual(limits.court_half_width_m(), 6.10 / 2.0, places=9)
        singles = dataclasses.replace(limits, boundary_mode='singles')
        self.assertAlmostEqual(singles.court_half_width_m(), 5.18 / 2.0, places=9)
        self.assertAlmostEqual(limits.net_x_m.value, 0.0, places=9)

    def test_an_unresolved_limit_refuses_to_decide(self) -> None:
        limits = dataclasses.replace(
            FeasibilityLimits(),
            shuttle_speed_max_mps=Param(None, AssetStatus.REQUIRES_MEASUREMENT,
                                        'measured incoming speed ceiling not available yet'))
        self.assertIn('shuttle_speed_max_mps', limits.unresolved_limits())
        with self.assertRaises(ValueError):
            HitFeasibilityGate(limits).evaluate(make_state(), make_trajectory())


class BaseTravelModelTests(unittest.TestCase):
    def test_triangular_profile_for_short_distances(self) -> None:
        # d <= v^2/a  ->  T = 2*sqrt(d/a)
        self.assertAlmostEqual(base_min_travel_time_s(0.0, 1.0, 1.5), 0.0, places=12)
        self.assertAlmostEqual(base_min_travel_time_s(0.375, 1.0, 1.5), 1.0, places=9)

    def test_trapezoidal_profile_for_long_distances(self) -> None:
        # d > v^2/a  ->  T = d/v + v/a
        self.assertAlmostEqual(base_min_travel_time_s(3.0, 1.0, 1.5), 3.0 + 1.0 / 1.5, places=9)

    def test_non_finite_or_non_positive_inputs_are_refused(self) -> None:
        # A NaN distance must never be silently read as "no travel needed" (review C1/C6).
        for args in ((math.nan, 1.0, 1.5), (math.inf, 1.0, 1.5), (1.0, math.nan, 1.5),
                     (1.0, math.inf, 1.5), (1.0, 0.0, 1.5), (1.0, 1.0, 0.0), (1.0, -1.0, 1.5)):
            with self.assertRaises(ValueError, msg='args=%r' % (args,)):
                base_min_travel_time_s(*args)


class DirectionWindowTests(unittest.TestCase):
    """S19: the approach test uses median(vx) over a short future window, not one sample."""

    def test_a_single_noisy_sample_does_not_flip_the_direction(self) -> None:
        trajectory = make_trajectory()
        # Only the FIRST sample is noisy: velocity is (N, T, 3), so the index is (env 0, sample 0).
        trajectory.velocity[0, 0] = np.array([0.5, 0.0, 1.5])
        decision = HitFeasibilityGate().evaluate(make_state(), trajectory)
        self.assertNotEqual(decision.reason, HitReason.WRONG_DIRECTION.value)
        self.assertTrue(decision.feasible, decision.reason)

    def test_a_receding_short_window_is_the_wrong_direction(self) -> None:
        trajectory = make_trajectory()
        trajectory.velocity[0, :15, 0] = 1.0                 # env 0, first 0.15 s of samples
        decision = HitFeasibilityGate().evaluate(make_state(), trajectory)
        self.assertEqual(decision.reason, HitReason.WRONG_DIRECTION.value)


class SharedTravelModelTests(unittest.TestCase):
    """C3: one base-travel-time implementation, imported by both decision modules."""

    def test_both_decision_modules_use_the_single_travel_model(self) -> None:
        travel = importlib.import_module('badminton_brain.decision.travel_model')
        feasibility = importlib.import_module('badminton_brain.decision.feasibility')
        search = importlib.import_module('badminton_brain.decision.intercept_search')
        self.assertIs(feasibility.base_min_travel_time_s, travel.min_travel_time_s)
        self.assertIs(search.min_travel_time_s, travel.min_travel_time_s)

    def test_vectorised_model_matches_the_scalar_one_and_keeps_its_input(self) -> None:
        travel = importlib.import_module('badminton_brain.decision.travel_model')
        distances = np.array([0.0, 0.1, 0.375, 0.5, 3.0, 7.0])
        original = distances.copy()
        times = travel.min_travel_times_s(distances, 1.0, 1.5)
        expected = [base_min_travel_time_s(d, 1.0, 1.5) for d in distances]
        np.testing.assert_allclose(times, expected, rtol=0.0, atol=1e-12)
        np.testing.assert_array_equal(distances, original)       # input is never rewritten
        self.assertIsNot(times, distances)
        with self.assertRaises(ValueError):                      # NaN is refused, not propagated
            travel.min_travel_times_s(np.array([0.1, np.nan]), 1.0, 1.5)
        with self.assertRaises(ValueError):
            travel.min_travel_times_s(np.array([0.1]), 1.0, 0.0)


if __name__ == '__main__':
    unittest.main()
