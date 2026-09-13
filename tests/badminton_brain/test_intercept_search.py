# -*- coding: utf-8 -*-
"""T6 - intercept search (decision layer): acceptance tests.

Spec:
  * docs/architecture/04_HIT_DECISION.md S26-S39 (search the *whole* future trajectory, cubic
    Hermite candidate interpolation), S54-S75 (workspace box + base minimum-time model +
    T_available budgeting), S99-S115 (soft score with a recorded component breakdown),
    S41-S48 (strike pose from the racket_contact convention, no magic offset)
  * docs/architecture/COORDINATE_SYSTEM.md S6.4/S6.5 (+X_racket = face normal, +Z_racket =
    grip -> head) and S2.4 (interface quaternion order x, y, z, w)
  * docs/superpowers/plans/2026-09-13-brain-modules.md row T6 (earliest feasible intercept on
    a synthetic trajectory, deterministic ordering, no invented constant)

Every expectation below is either an exact analytic value of the synthetic trajectory or a
value this test recomputes from the trajectory samples - never a number copied out of the
module under test.
"""
from __future__ import annotations

import dataclasses
import math
import sys
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'src'))

from badminton_brain.decision.intercept_search import (  # noqa: E402
    InterceptReason, InterceptSearchConfig, InterceptSearcher, ScoreTerms, SearchCandidate,
    min_travel_time_s, quaternion_from_rotation, racket_pose_axes, racket_pose_from_contact,
    racket_pose_is_feasible, rotation_from_quaternion, search_intercepts, select_best_candidate,
)
from badminton_brain.status import AssetStatus, Param  # noqa: E402
from badminton_brain.types import BrainBoundaryError, PredictedTrajectory, UnifiedState  # noqa: E402

GRAVITY_MPS2 = 9.80665          # same gravity constant as src/trajectory/shuttle_aerodynamics.py
DT_S = 0.005                    # trajectory sample step / candidate grid step
HORIZON_S = 0.60
BASE_XY = (-0.60, 0.10)         # robot base on our side of the net, under the incoming shuttle
P0 = (-0.30, 0.20, 2.20)        # shuttle starts high above the base
V0 = (-0.50, 0.00, 0.00)        # incoming (vx < 0), flat


# --------------------------------------------------------------------------- helpers
def config(**overrides) -> InterceptSearchConfig:
    """InterceptSearchConfig with explicit test values (every field is a Param)."""
    plain = {name: (value if isinstance(value, Param) else Param(
        value, AssetStatus.TEMP_PARAMETERIZED_PROXY, 'test override'))
        for name, value in overrides.items()}
    return dataclasses.replace(InterceptSearchConfig(), **plain)


def yaw_quaternion(yaw: float) -> np.ndarray:
    """(x, y, z, w) quaternion of a pure yaw rotation (COORDINATE_SYSTEM.md S2.4 order)."""
    return np.array([0.0, 0.0, math.sin(0.5 * yaw), math.cos(0.5 * yaw)])


def make_state(shuttle_position, shuttle_velocity, *, base_xy=BASE_XY, yaw=0.0,
               timestamp=0.0) -> UnifiedState:
    position = np.atleast_2d(np.asarray(shuttle_position, dtype=float))
    velocity = np.atleast_2d(np.asarray(shuttle_velocity, dtype=float))
    n = position.shape[0]
    pose = np.concatenate(([base_xy[0], base_xy[1], 0.0], yaw_quaternion(yaw)))
    return UnifiedState(
        base_pose=np.tile(pose, (n, 1)),
        base_twist=np.zeros((n, 6)),
        joint_pos=np.zeros((n, 6)),
        joint_vel=np.zeros((n, 6)),
        racket_contact_pose=np.tile(pose, (n, 1)),
        racket_contact_twist=np.zeros((n, 6)),
        shuttle_position=position,
        shuttle_velocity=velocity,
        timestamp=timestamp)


def make_trajectory(positions, velocities, times, arrival_time, *, landing_point=None,
                    timestamp=0.0) -> PredictedTrajectory:
    positions = np.asarray(positions, dtype=float)
    velocities = np.asarray(velocities, dtype=float)
    if positions.ndim == 2:
        positions = positions[None, ...]
    if velocities.ndim == 2:
        velocities = velocities[None, ...]
    assert positions.ndim == 3 and velocities.ndim == 3, "positions/velocities must be (N, T, 3)"
    times = np.asarray(times, dtype=float)
    arrival = np.atleast_1d(np.asarray(arrival_time, dtype=float))
    if landing_point is None:
        landing_point = positions[:, -1, :].copy()
    return PredictedTrajectory(times=times, position=positions, velocity=velocities,
                               landing_point=np.asarray(landing_point, dtype=float),
                               arrival_time=arrival, timestamp=timestamp)


def ballistic(initial_position, initial_velocity, times) -> tuple:
    """Exact drag-free flight: p(t) = p0 + v0 t + g t^2 / 2 with g = (0, 0, -9.80665)."""
    p0 = np.asarray(initial_position, dtype=float).reshape(3, 1)
    v0 = np.asarray(initial_velocity, dtype=float).reshape(3, 1)
    t = np.asarray(times, dtype=float).reshape(1, -1)
    gravity = np.array([0.0, 0.0, -GRAVITY_MPS2]).reshape(3, 1)
    position = p0 + v0 * t + 0.5 * gravity * t ** 2
    velocity = v0 + gravity * t
    return position.T, velocity.T


def z_window(times, z0) -> np.ndarray:
    """Analytic height of the canonical scenario (vz0 = 0)."""
    return z0 - 0.5 * GRAVITY_MPS2 * np.asarray(times, dtype=float) ** 2


def canonical(candidate_dt_s=DT_S, base_xy=BASE_XY, p0=P0, v0=V0, horizon_s=HORIZON_S,
              arrival_time=None, timestamp=0.0):
    """The canonical scenario: an incoming shuttle falls through the racket workspace box."""
    times = np.arange(int(math.floor(horizon_s / DT_S + 1e-9)) + 1, dtype=float) * DT_S
    position, velocity = ballistic(p0, v0, times)
    arrival = float(times[-1]) if arrival_time is None else float(arrival_time)
    trajectory = make_trajectory(position, velocity, times, arrival, timestamp=timestamp)
    state = make_state(position[0], velocity[0], base_xy=base_xy, timestamp=timestamp)
    return state, trajectory, config(candidate_dt_s=candidate_dt_s), times, position, velocity


def feasibility_config(cfg: InterceptSearchConfig) -> dict:
    """The declared limits, as plain floats (never hardcoded in this test)."""
    return {name: float(param.value) for name, param in cfg.param_limits().items()
            if param.value is not None}


def reason_of(result, env_position: int, time_s: float) -> InterceptReason:
    """Reason recorded for the candidate at exactly ``time_s``."""
    for candidate in result.candidates[env_position]:
        if abs(candidate.time_s - time_s) < 1e-12:
            return candidate.reason
    raise AssertionError(f"no candidate at t={time_s}")


def candidate_at(result, env_position: int, time_s: float) -> SearchCandidate:
    for candidate in result.candidates[env_position]:
        if abs(candidate.time_s - time_s) < 1e-12:
            return candidate
    raise AssertionError(f"no candidate at t={time_s}")


# --------------------------------------------------------------------------- tests
class EarliestFeasibleInterceptTests(unittest.TestCase):
    """Acceptance: the earliest feasible intercept on the predicted trajectory is found."""

    def test_earliest_feasible_candidate_matches_the_analytic_window_entry(self) -> None:
        state, trajectory, cfg, times, positions, _ = canonical()
        result = search_intercepts(state, trajectory, config=cfg)

        limits = feasibility_config(cfg)
        z_max = limits['arm_z_max_m']
        z_min = limits['arm_z_min_m']
        heights = z_window(times, P0[2])
        inside = [t for t, z in zip(times, heights) if z_min <= z <= z_max]
        # The base needs no travel in this scenario and the time gate is slack for every
        # candidate at or after the window entry, so the height window alone decides.
        expected_times = [t for t in inside if t <= trajectory.arrival_time[0] + 1e-12]
        self.assertTrue(expected_times, "synthetic scenario must contain a hit window")

        earliest = result.earliest_for_env(0)
        self.assertIsNotNone(earliest)
        self.assertAlmostEqual(earliest.time_s, expected_times[0], places=12)
        self.assertEqual(earliest.reason, InterceptReason.BEST_INTERCEPT_FOUND)
        self.assertEqual(earliest.feasible, True)

        index = int(round(expected_times[0] / DT_S))
        # Position comes from the trajectory itself: no offset is added anywhere.
        np.testing.assert_allclose(earliest.position, positions[index], atol=1e-12)

        feasible_times = [c.time_s for c in result.candidates[0] if c.feasible]
        self.assertEqual(len(feasible_times), len(expected_times))
        np.testing.assert_allclose(feasible_times, expected_times, atol=1e-12)

        self.assertTrue(result.feasible[0])
        self.assertEqual(result.reason[0], InterceptReason.BEST_INTERCEPT_FOUND)
        self.assertIsNotNone(result.best)
        self.assertEqual(result.best.position.shape, (1, 3))
        self.assertEqual(result.best.time_s.shape, (1,))
        self.assertEqual(result.best.racket_pose.shape, (1, 7))
        self.assertEqual(result.best.score.shape, (1,))
        # The selected best is a *different* candidate in general (score maximum, S99-S115);
        # it must be one of the feasible candidates of this environment.
        self.assertTrue(any(abs(c.time_s - float(result.best.time_s[0])) < 1e-12
                            for c in result.candidates[0] if c.feasible))

    def test_candidates_are_not_restricted_to_a_fixed_x_plane(self) -> None:
        """S27: the search must not collapse onto a constant-x strike plane."""
        state, trajectory, cfg, times, _, _ = canonical()
        result = search_intercepts(state, trajectory, config=cfg)
        feasible = [c for c in result.candidates[0] if c.feasible]
        xs = [c.position[0] for c in feasible]
        zs = [c.position[2] for c in feasible]
        self.assertGreater(len(set(round(x, 9) for x in xs)), 2)
        self.assertGreater(max(xs) - min(xs), 0.05)
        self.assertGreater(max(zs) - min(zs), 0.5)

    def test_selection_is_the_documented_score_maximum(self) -> None:
        state, trajectory, cfg, _, _, _ = canonical()
        result = search_intercepts(state, trajectory, config=cfg)
        best = result.best_for_env(0)
        weights = cfg.weights()
        for candidate in result.candidates[0]:
            if not candidate.feasible:
                continue
            recomputed = candidate.score_terms.total(weights)
            self.assertAlmostEqual(candidate.score, recomputed, places=12)
            self.assertLessEqual(candidate.score, best.score + 1e-12)
        self.assertAlmostEqual(best.score, best.score_terms.total(weights), places=12)

    def test_lexicographic_tie_break_prefers_the_earliest_time(self) -> None:
        terms = ScoreTerms(time=0.5, height=0.5, speed=0.5, margin=0.5, base_cost=0.0)
        early = SearchCandidate(env_id=0, time_s=0.10, position=np.array([0.0, 0.0, 1.0]),
                                velocity=np.zeros(3), feasible=True, score=0.5,
                                reason=InterceptReason.BEST_INTERCEPT_FOUND, score_terms=terms)
        late = dataclasses.replace(early, time_s=0.20)
        better_late = dataclasses.replace(late, score=0.6)
        self.assertIs(select_best_candidate([late, early]), early)
        self.assertIs(select_best_candidate([early, late]), early)
        self.assertIs(select_best_candidate([early, better_late]), better_late)
        self.assertIsNone(select_best_candidate([]))
        self.assertIsNone(select_best_candidate(
            [dataclasses.replace(early, feasible=False, reason=InterceptReason.UNREACHABLE)]))


class NoFeasibleInterceptTests(unittest.TestCase):
    def test_unreachable_shuttle_returns_none_and_an_explainable_reason(self) -> None:
        state, trajectory, cfg, _, _, _ = canonical(base_xy=(-1.00, 0.00),
                                                    p0=(-3.00, 0.50, 1.00),
                                                    v0=(0.00, 0.00, -0.50))
        result = search_intercepts(state, trajectory, config=cfg)
        self.assertFalse(bool(np.any(result.feasible)))
        self.assertIsNone(result.best)
        self.assertEqual(result.best_env_ids.shape, (0,))
        self.assertIsNone(result.best_for_env(0))
        self.assertTrue(np.all(np.isneginf(result.score)))
        self.assertEqual(result.reason[0], InterceptReason.NO_TIME_MARGIN)
        late = [c for c in result.candidates[0] if c.time_available_s > 0.0]
        self.assertTrue(late)
        self.assertTrue(all(c.reason is InterceptReason.UNREACHABLE for c in late))
        self.assertTrue(all(c.racket_pose is None for c in result.candidates[0]))

    def test_shuttle_above_the_workspace_box_is_rejected_geometrically(self) -> None:
        state, trajectory, cfg, _, _, _ = canonical(p0=(-0.30, 0.20, 3.50),
                                                    v0=(0.00, 0.00, 1.00))
        result = search_intercepts(state, trajectory, config=cfg)
        self.assertIsNone(result.best)
        self.assertEqual(result.reason[0], InterceptReason.NO_GEOMETRIC_WINDOW)
        self.assertTrue(all(c.reason is InterceptReason.NO_GEOMETRIC_WINDOW
                            for c in result.candidates[0]))

    def test_candidates_after_the_predicted_landing_are_rejected(self) -> None:
        state, trajectory, cfg, times, _, _ = canonical(v0=(-0.50, 0.00, -2.00),
                                                        arrival_time=0.50)
        result = search_intercepts(state, trajectory, config=cfg)
        feasible = [c for c in result.candidates[0] if c.feasible]
        self.assertTrue(feasible)
        self.assertLessEqual(max(c.time_s for c in feasible), 0.50 + 1e-12)
        self.assertEqual(reason_of(result, 0, 0.505), InterceptReason.OUT_OF_BOUNDS)
        self.assertEqual(reason_of(result, 0, 0.600), InterceptReason.OUT_OF_BOUNDS)
        # The shuttle has dropped below the racket box before it lands.
        self.assertEqual(reason_of(result, 0, 0.490), InterceptReason.NO_GEOMETRIC_WINDOW)

    def test_zero_incoming_velocity_has_no_racket_orientation_solution(self) -> None:
        times = np.arange(int(HORIZON_S / DT_S) + 1, dtype=float) * DT_S
        position = np.tile(np.array([[-0.30, 0.20, 1.00]]), (times.size, 1))
        velocity = np.zeros((times.size, 3))
        trajectory = make_trajectory(position, velocity, times, times[-1])
        state = make_state(position[0], velocity[0])
        result = search_intercepts(state, trajectory, config=config(candidate_dt_s=DT_S))
        self.assertIsNone(result.best)
        # t = 0.30 has enough time margin, so the orientation gate is the one that blocks.
        self.assertGreater(candidate_at(result, 0, 0.300).time_available_s, 0.0)
        self.assertEqual(reason_of(result, 0, 0.300), InterceptReason.NO_ORIENTATION_SOLUTION)
        with self.assertRaises(ValueError):
            racket_pose_from_contact(np.zeros(3), np.zeros(3))


class TimeAndBaseTravelTests(unittest.TestCase):
    def test_base_travel_time_and_the_time_margin_gate(self) -> None:
        # A slow shuttle 0.15 m outside the workspace box: the base must move d = 0.15 m first.
        dt = DT_S
        times = np.arange(int(1.20 / dt) + 1, dtype=float) * dt
        velocity = np.tile(np.array([[0.00, 0.00, -0.05]]), (times.size, 1))
        position = np.array([[-0.75, 0.00, 1.00]]) + velocity * times.reshape(-1, 1)
        trajectory = make_trajectory(position, velocity, times, times[-1])
        state = make_state(position[0], velocity[0], base_xy=(0.0, 0.0))
        cfg = config(candidate_dt_s=dt)
        result = search_intercepts(state, trajectory, config=cfg)

        limits = feasibility_config(cfg)
        excess = 0.75 - limits['arm_reach_x_m']
        travel = min_travel_time_s(excess, limits['base_v_max_mps'], limits['base_a_max_mps2'])
        self.assertAlmostEqual(travel, 2.0 * math.sqrt(excess / limits['base_a_max_mps2']),
                               places=12)
        required = max(travel, limits['arm_slew_time_s'])
        overhead = limits['decision_latency_s'] + limits['safety_time_margin_s']
        expected_first = next(t for t in times if t - overhead > required)

        earliest = result.earliest_for_env(0)
        self.assertIsNotNone(earliest)
        self.assertAlmostEqual(earliest.time_s, expected_first, places=12)
        self.assertAlmostEqual(earliest.base_travel_m, excess, places=12)
        self.assertAlmostEqual(earliest.time_required_s, required, places=12)
        self.assertAlmostEqual(earliest.time_available_s, expected_first - overhead, places=12)
        self.assertLess(earliest.workspace_margin, 1e-12)
        previous = candidate_at(result, 0, expected_first - dt)
        self.assertFalse(previous.feasible)
        self.assertIs(previous.reason, InterceptReason.UNREACHABLE)

    def test_trapezoid_profile_beyond_the_cruise_distance(self) -> None:
        v_max, a_max = 1.0, 1.5
        distance = 3.0
        expected = 2.0 * v_max / a_max + (distance - v_max ** 2 / a_max) / v_max
        self.assertAlmostEqual(min_travel_time_s(distance, v_max, a_max), expected, places=12)
        self.assertAlmostEqual(min_travel_time_s(0.0, v_max, a_max), 0.0, places=12)
        self.assertAlmostEqual(min_travel_time_s(distance, v_max, a_max),
                               min_travel_time_s(-distance, v_max, a_max), places=12)

    def test_workspace_box_follows_the_estimated_base_yaw(self) -> None:
        """The box is carried by the base, so the base yaw decides which contacts are inside."""
        dt = DT_S
        times = np.arange(int(1.20 / dt) + 1, dtype=float) * dt
        velocity = np.tile(np.array([[0.00, 0.00, -0.05]]), (times.size, 1))
        position = np.array([[0.70, 0.00, 1.00]]) + velocity * times.reshape(-1, 1)
        trajectory = make_trajectory(position, velocity, times, times[-1])
        cfg = config(candidate_dt_s=dt)
        straight = search_intercepts(
            make_state(position[0], velocity[0], base_xy=(0.0, 0.0), yaw=0.0), trajectory, config=cfg)
        turned = search_intercepts(
            make_state(position[0], velocity[0], base_xy=(0.0, 0.0), yaw=math.pi / 4.0),
            trajectory, config=cfg)
        # yaw = 0: the contact is 0.10 m beyond the +X half extent -> the base must travel 0.10 m.
        straight_earliest = straight.earliest_for_env(0)
        self.assertIsNotNone(straight_earliest)
        limits = feasibility_config(cfg)
        self.assertAlmostEqual(straight_earliest.base_travel_m,
                               0.70 - limits['arm_reach_x_m'], places=12)
        # yaw = 45 deg: the same contact point is inside the (rotated) box -> no base travel at all,
        # and therefore an earlier feasible intercept.
        turned_earliest = turned.earliest_for_env(0)
        self.assertIsNotNone(turned_earliest)
        self.assertEqual(turned_earliest.base_travel_m, 0.0)
        self.assertLess(turned_earliest.time_s, straight_earliest.time_s)

    def test_time_too_short_for_even_the_arm_slew(self) -> None:
        state, trajectory, cfg, times, _, _ = canonical(base_xy=BASE_XY, timestamp=0.0)
        late = config(candidate_dt_s=DT_S, decision_latency_s=0.90)
        result = search_intercepts(state, trajectory, config=late)
        self.assertIsNone(result.best)
        reasons = {c.reason for c in result.candidates[0]}
        # Candidates that are still above the racket box fail the height gate first.
        self.assertTrue(reasons <= {InterceptReason.NO_TIME_MARGIN,
                                    InterceptReason.NO_GEOMETRIC_WINDOW})
        self.assertIn(InterceptReason.NO_TIME_MARGIN, reasons)


class RacketPoseConventionTests(unittest.TestCase):
    def test_pose_follows_the_racket_contact_convention(self) -> None:
        state, trajectory, cfg, _, _, velocities = canonical()  # velocities = sampled (T, 3) grid
        result = search_intercepts(state, trajectory, config=cfg)
        best = result.best_for_env(0)
        pose = np.asarray(result.best.racket_pose[0], dtype=float)
        np.testing.assert_allclose(pose[:3], best.position, atol=0.0)

        quaternion = pose[3:7]
        self.assertAlmostEqual(float(np.linalg.norm(quaternion)), 1.0, places=12)
        self.assertGreaterEqual(float(quaternion[3]), 0.0, "canonical w >= 0")

        x_axis, y_axis, z_axis = racket_pose_axes(pose)
        rotation = rotation_from_quaternion(quaternion)
        np.testing.assert_allclose(rotation[:, 0], x_axis, atol=1e-12)
        np.testing.assert_allclose(rotation[:, 2], z_axis, atol=1e-12)
        np.testing.assert_allclose(rotation @ rotation.T, np.eye(3), atol=1e-12)
        self.assertAlmostEqual(float(np.linalg.det(rotation)), 1.0, places=12)
        np.testing.assert_allclose(y_axis, np.cross(z_axis, x_axis), atol=1e-12)

        # +X_racket is the face normal: it opposes the incoming velocity (flat block, S48).
        incoming = np.asarray(best.velocity, dtype=float)
        np.testing.assert_allclose(x_axis, -incoming / np.linalg.norm(incoming), atol=1e-12)
        # +Z_racket (grip -> head) is the up-projection: no arbitrary roll is invented.
        up = np.array([0.0, 0.0, 1.0])
        expected_head = up - float(np.dot(up, x_axis)) * x_axis
        np.testing.assert_allclose(z_axis, expected_head / np.linalg.norm(expected_head),
                                   atol=1e-12)
        self.assertGreaterEqual(float(z_axis[2]), 0.0)
        np.testing.assert_allclose(z_axis @ x_axis, 0.0, atol=1e-12)
        self.assertEqual(velocities.shape, trajectory.velocity[0].shape)

    def test_vertical_drop_uses_the_documented_degenerate_fallback(self) -> None:
        pose = racket_pose_from_contact(np.array([0.0, 0.0, 1.0]), np.array([0.0, 0.0, -1.0]))
        x_axis, y_axis, z_axis = racket_pose_axes(pose)
        np.testing.assert_allclose(x_axis, [0.0, 0.0, 1.0], atol=1e-12)
        np.testing.assert_allclose(z_axis, [1.0, 0.0, 0.0], atol=1e-12)
        np.testing.assert_allclose(y_axis, np.cross(z_axis, x_axis), atol=1e-12)
        self.assertTrue(np.all(np.isfinite(pose)))
        np.testing.assert_allclose(racket_pose_axes(pose)[0] @ racket_pose_axes(pose)[1], 0.0,
                                   atol=1e-12)

    def test_edge_on_racket_pose_is_rejected(self) -> None:
        velocity = np.array([-1.0, 0.0, 0.0])       # incoming along -X
        flat = racket_pose_from_contact(np.zeros(3), velocity)
        self.assertTrue(racket_pose_is_feasible(flat, velocity))

        # Same contact point, but the face normal turned 90 deg away from the incoming velocity
        # (edge-on contact): the racket cannot strike the shuttle that way.
        edge_on = flat.copy()
        edge_on[3:7] = quaternion_from_rotation(np.column_stack((
            np.array([0.0, 0.0, 1.0]), np.array([0.0, 1.0, 0.0]), np.array([-1.0, 0.0, 0.0]))))
        self.assertFalse(racket_pose_is_feasible(edge_on, velocity))
        self.assertFalse(racket_pose_is_feasible(np.zeros(7), velocity))
        self.assertFalse(racket_pose_is_feasible(flat, np.zeros(3)))


class DeterminismTests(unittest.TestCase):
    def test_identical_inputs_give_bit_identical_outputs(self) -> None:
        state, trajectory, cfg, _, _, _ = canonical()
        first = search_intercepts(state, trajectory, config=cfg)
        second = search_intercepts(state, trajectory, config=cfg)
        third = InterceptSearcher(cfg).search(state, trajectory)

        for other in (second, third):
            self.assertTrue(np.array_equal(first.best.position, other.best.position))
            self.assertTrue(np.array_equal(first.best.time_s, other.best.time_s))
            self.assertTrue(np.array_equal(first.best.racket_pose, other.best.racket_pose))
            self.assertTrue(np.array_equal(first.best.score, other.best.score))
            self.assertTrue(np.array_equal(first.score, other.score));
            self.assertEqual(first.best_env_ids.tolist(), other.best_env_ids.tolist())
            self.assertEqual([c.reason for c in first.candidates[0]],
                             [c.reason for c in other.candidates[0]])
            self.assertEqual([c.time_s for c in first.candidates[0]],
                             [c.time_s for c in other.candidates[0]])
            self.assertEqual([c.score for c in first.candidates[0]],
                             [c.score for c in other.candidates[0]])

    def test_search_does_not_mutate_its_inputs(self) -> None:
        state, trajectory, cfg, _, _, _ = canonical()
        position_before = np.array(trajectory.position, copy=True)
        velocity_before = np.array(trajectory.velocity, copy=True)
        base_before = np.array(state.base_pose, copy=True)
        times_before = np.array(trajectory.times, copy=True)
        search_intercepts(state, trajectory, config=cfg)
        self.assertTrue(np.array_equal(trajectory.position, position_before))
        self.assertTrue(np.array_equal(trajectory.velocity, velocity_before))
        self.assertTrue(np.array_equal(state.base_pose, base_before))
        self.assertTrue(np.array_equal(trajectory.times, times_before))

    def test_fresh_but_equal_input_objects_give_the_same_result(self) -> None:
        state, trajectory, cfg, _, _, _ = canonical()
        first = search_intercepts(state, trajectory, config=cfg)
        other_state, other_trajectory, _, _, _, _ = canonical()
        second = search_intercepts(other_state, other_trajectory, config=cfg)
        self.assertTrue(np.array_equal(first.best.position, second.best.position))
        self.assertTrue(np.array_equal(first.best.score, second.best.score))
        self.assertEqual(first.best_for_env(0).time_s, second.best_for_env(0).time_s)

    def test_time_base_is_resolved_for_absolute_and_relative_grids(self) -> None:
        """S6 asks for absolute times; a grid relative to its prediction instant is recognised."""
        state, trajectory, cfg, times, position, velocity = canonical(timestamp=12.5)
        # (a) specification-compliant absolute grid: the sample times carry the simulation clock.
        absolute_trajectory = make_trajectory(position, velocity, times + 12.5, 12.5 + times[-1],
                                              timestamp=12.5)
        absolute = search_intercepts(state, absolute_trajectory, config=cfg)
        # (b) grid starting at the prediction instant (what shuttle_aerodynamics.rollout returns).
        relative_trajectory = make_trajectory(position, velocity, times, times[-1], timestamp=12.5)
        relative = search_intercepts(state, relative_trajectory, config=cfg)
        self.assertAlmostEqual(absolute.now_s, 12.5, places=12)
        self.assertAlmostEqual(relative.now_s, 12.5, places=12)
        self.assertTrue(np.array_equal(absolute.feasible, relative.feasible))
        np.testing.assert_allclose(absolute.score, relative.score, atol=1e-12)
        # DEC-016: both grids describe the same instants, so the emitted absolute times agree.
        np.testing.assert_allclose(absolute.best.time_s, relative.best.time_s, atol=1e-9)
        self.assertEqual(times.shape[0], absolute.candidate_times.shape[0])
        self.assertEqual(trajectory.times.shape, absolute.candidate_times.shape)


class AbsoluteTimeContractTests(unittest.TestCase):
    """DEC-016: BestIntercept.time_s is an absolute simulation time, never a time-to-go."""

    def test_best_intercept_time_is_absolute_simulation_time(self) -> None:
        state, trajectory, cfg, times, position, velocity = canonical(timestamp=12.5)
        result = search_intercepts(state, trajectory, config=cfg)
        self.assertAlmostEqual(result.now_s, 12.5, places=12)
        limits = feasibility_config(cfg)
        t_go = next(t for t, z in zip(times, z_window(times, P0[2]))
                    if z <= limits['arm_z_max_m'])
        earliest = result.earliest_for_env(0)
        self.assertIsNotNone(earliest)
        # The producer emits the absolute instant (= now + time-to-go); a consumer that needs the
        # time-to-go subtracts state.timestamp itself.
        self.assertAlmostEqual(earliest.time_s, result.now_s + t_go, places=9)
        self.assertAlmostEqual(earliest.time_s, float(state.timestamp) + t_go, places=9)
        self.assertAlmostEqual(earliest.time_s - state.timestamp, t_go, places=9)
        # The emitted time lies on the trajectory time axis: same grid, same simulation clock.
        absolute_axis = np.asarray(times, dtype=float) + 12.5
        self.assertLessEqual(float(np.min(np.abs(absolute_axis - earliest.time_s))), 1e-9)
        self.assertGreaterEqual(earliest.time_s, float(absolute_axis[0]) - 1e-9)
        self.assertLessEqual(earliest.time_s, float(absolute_axis[-1]) + 1e-9)
        best_time = float(result.best.time_s[0])
        self.assertLessEqual(float(np.min(np.abs(absolute_axis - best_time))), 1e-9)
        self.assertGreaterEqual(best_time, result.now_s)
        # An absolute grid (S6) describes the same instants: both representations must agree.
        absolute_trajectory = make_trajectory(position, velocity, times + 12.5, 12.5 + times[-1],
                                              timestamp=12.5)
        absolute = search_intercepts(state, absolute_trajectory, config=cfg)
        np.testing.assert_allclose(absolute.best.time_s, result.best.time_s, atol=1e-9)
        self.assertAlmostEqual(float(absolute.best.time_s[0]), best_time, places=9)


class PhysicsAgreementTests(unittest.TestCase):
    def test_candidates_reproduce_the_aerodynamics_rollout_samples(self) -> None:
        from trajectory.shuttle_aerodynamics import k_from_aerodynamic_length, rollout
        k_per_m = k_from_aerodynamic_length(6.5)
        sample = rollout(np.asarray(P0), np.asarray(V0), duration_s=HORIZON_S, dt_s=DT_S,
                         k_per_m=k_per_m)
        times = np.asarray(sample['time'], dtype=float)
        position = np.asarray(sample['position'], dtype=float)
        velocity = np.asarray(sample['velocity'], dtype=float)
        trajectory = make_trajectory(position, velocity, times, times[-1])
        state = make_state(position[0], velocity[0])
        cfg = config(candidate_dt_s=DT_S)
        result = search_intercepts(state, trajectory, config=cfg)

        limits = feasibility_config(cfg)
        expected = [index for index, point in enumerate(position)
                    if limits['arm_z_min_m'] <= point[2] <= limits['arm_z_max_m']
                    and times[index] <= times[-1] + 1e-12]
        self.assertTrue(expected, "the dragged flight must cross the racket box")

        feasible = [c for c in result.candidates[0] if c.feasible]
        self.assertEqual(len(feasible), len(expected))
        for candidate, index in zip(feasible, expected):
            self.assertAlmostEqual(candidate.time_s, times[index], places=12)
            np.testing.assert_allclose(candidate.position, position[index], atol=1e-12)
            np.testing.assert_allclose(candidate.velocity, velocity[index], atol=1e-12)
            self.assertEqual(candidate.base_travel_m, 0.0)
        self.assertAlmostEqual(result.earliest_for_env(0).time_s, times[expected[0]], places=12)

    def test_hermite_interpolation_keeps_the_exact_parabola(self) -> None:
        # Coarse samples (50 ms) of an exact parabola, searched on a 5 ms grid: the cubic
        # Hermite rule of S37 uses p and v, so it must reproduce the parabola exactly.
        coarse_times = np.arange(int(HORIZON_S / 0.05 + 1e-9) + 1, dtype=float) * 0.05
        position, velocity = ballistic(P0, V0, coarse_times)
        trajectory = make_trajectory(position, velocity, coarse_times, HORIZON_S)
        state = make_state(position[0], velocity[0])
        result = search_intercepts(state, trajectory, config=config(candidate_dt_s=DT_S))
        for candidate in result.candidates[0]:
            if not candidate.feasible:
                continue
            exact, exact_velocity = ballistic(P0, V0, [candidate.time_s])
            np.testing.assert_allclose(candidate.position, exact[0], atol=1e-9)
            np.testing.assert_allclose(candidate.velocity, exact_velocity[0], atol=1e-9)
        self.assertIsNotNone(result.best)


class BatchAndInputTests(unittest.TestCase):
    def test_batch_rows_follow_the_requested_env_order(self) -> None:
        times = np.arange(int(HORIZON_S / DT_S) + 1, dtype=float) * DT_S
        good_position, good_velocity = ballistic(P0, V0, times)
        bad_position, bad_velocity = ballistic((-3.00, 0.50, 1.00), (0.0, 0.0, -0.50), times)
        position = np.stack([good_position, bad_position], axis=0)
        velocity = np.stack([good_velocity, bad_velocity], axis=0)
        trajectory = make_trajectory(position, velocity, times, [times[-1], times[-1]])
        state = make_state(position[:, 0, :], velocity[:, 0, :])
        cfg = config(candidate_dt_s=DT_S)

        result = search_intercepts(state, trajectory, config=cfg)
        self.assertEqual(result.feasible.tolist(), [True, False])
        late = [c for c in result.candidates[1] if c.time_available_s > 0.0]
        self.assertTrue(late)
        self.assertTrue(all(c.reason is InterceptReason.UNREACHABLE for c in late))
        self.assertEqual(result.reason[1], InterceptReason.NO_TIME_MARGIN)
        self.assertEqual(result.best_env_ids.tolist(), [0])
        self.assertEqual(result.best.position.shape, (1, 3))
        self.assertIsNone(result.best_for_env(1))

        reversed_result = search_intercepts(state, trajectory, config=cfg, env_ids=[1, 0])
        self.assertEqual(reversed_result.feasible.tolist(), [False, True])
        self.assertEqual(reversed_result.best_env_ids.tolist(), [0])
        self.assertIsNotNone(reversed_result.best_for_env(0))

    def test_invalid_inputs_are_rejected_loudly(self) -> None:
        state, trajectory, cfg, _, _, _ = canonical()
        other = make_state(np.zeros((2, 3)), np.zeros((2, 3)))
        with self.assertRaises(BrainBoundaryError):
            search_intercepts(other, trajectory, config=cfg)
        with self.assertRaises(BrainBoundaryError):
            search_intercepts(state, trajectory, config=cfg, env_ids=[0, 0])
        with self.assertRaises(BrainBoundaryError):
            search_intercepts(state, trajectory, config=cfg, env_ids=[1])
        with self.assertRaises(BrainBoundaryError):
            search_intercepts(state, trajectory, config=cfg, now=1e9)
        with self.assertRaises(BrainBoundaryError):
            search_intercepts(trajectory, trajectory, config=cfg)
        flat = np.asarray(trajectory.times, dtype=float).copy()
        flat[3] = flat[2]
        with self.assertRaises(BrainBoundaryError):
            search_intercepts(state, make_trajectory(trajectory.position, trajectory.velocity,
                                                     flat, trajectory.arrival_time));

    def test_single_sample_prediction_is_reported_as_invalid(self) -> None:
        times = np.array([0.0])
        trajectory = make_trajectory([P0], [V0], times, [0.0])
        state = make_state(P0, V0)
        result = search_intercepts(state, trajectory, config=config())
        self.assertFalse(bool(np.any(result.feasible)))
        self.assertIsNone(result.best)
        self.assertEqual(result.reason[0], InterceptReason.INVALID_PREDICTION)
        self.assertEqual(result.candidates[0], ())


class AuthenticityTests(unittest.TestCase):
    def test_every_limit_is_a_param_with_a_status_and_a_source(self) -> None:
        cfg = InterceptSearchConfig()
        limits = cfg.param_limits()
        self.assertTrue(limits)
        for name, param in limits.items():
            self.assertIsInstance(param, Param, msg=name)
            self.assertIsInstance(param.status, AssetStatus, msg=name)
            self.assertTrue(param.source.strip(), msg=name)
            if param.status in (AssetStatus.REQUIRES_MEASUREMENT, AssetStatus.UNKNOWN,
                               AssetStatus.REQUIRES_CALIBRATION):
                self.assertIsNone(param.value, msg=name)

    def test_unmeasured_quantities_are_declared_not_invented(self) -> None:
        cfg = InterceptSearchConfig()
        unresolved = dict(cfg.unresolved_limits())
        self.assertIn('achievable_face_normal_cone_rad', unresolved)
        self.assertIn('effective_contact_offset_m', unresolved)
        requirements = cfg.measurement_requirements()
        self.assertTrue(set(requirements) <= set(unresolved))
        for name, param in requirements.items():
            self.assertIsNone(param.value, msg=name)
            self.assertEqual(param.status, AssetStatus.REQUIRES_MEASUREMENT, msg=name)
        self.assertEqual(cfg.weights().time, float(cfg.weight_time.value))


if __name__ == '__main__':
    unittest.main(verbosity=2)
