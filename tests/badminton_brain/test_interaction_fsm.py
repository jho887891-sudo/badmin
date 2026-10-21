#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""P7-10: every FSM transition, every FAIL-CLOSED path, the HOLD latch, and the
"no command while HOLD" guarantee.

The fixtures are synthetic and deterministic.  Perception is a fixed 3-D point,
prediction is a fixed intercept, and the IK solver is either the real DLS solver
on an exactly-solvable synthetic target or a documented pass-through stub that
isolates the FSM from the arm model.  No hardware, no camera, no physics.

Run:
    python -m pytest tests/badminton_brain/test_interaction_fsm.py -q
"""
from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
for entry in (ROOT / "src", ROOT):
    if str(entry) not in sys.path:
        sys.path.insert(0, str(entry))

from src.badminton_brain.reachability import (  # noqa: E402
    Calibration, DampedLeastSquaresIk, IkResult, IkSolver, IkTarget, KinematicLimits,
)
from src.badminton_brain.reachability import target_from_intercept  # noqa: E402
from src.badminton_brain.safety.interaction_fsm import (  # noqa: E402
    ALLOWED_TRANSITIONS, STALE_FRAME_AGE_S, CommandSuppressed, FrameInput, FsmLimits,
    FsmState, HoldReason, InteractionFsm, PlanInput, PredictionInput, ResetRejected,
)
from src.badminton_brain.timing.latency_budget import MAX_SOURCE_FRAME_AGE_S  # noqa: E402

# Face centre of the reachable fixture pose (inside the workspace box, outside the
# base-column proxy), used as the intercept point throughout.
INTERCEPT = np.array([0.3081, 0.0714, 1.1375])
APPROACH = (0.0, 0.0, -1.0)
# A point + approach the PROXY arm can converge to, used for the real-solver test.
REACHABLE_INTERCEPT = np.array([0.45, 0.0, 1.0])
REACHABLE_APPROACH = (-1.0, 0.0, 0.0)
# A converged joint vector that keeps a comfortable margin from every stop.
SEED = np.array([0.3, 0.8, -1.2, 0.1, 0.4, 0.2])


class PassThroughIk(IkSolver):
    """Test stub: returns the seed unchanged.  Isolates the FSM from the arm model."""

    solver_id = "test/pass_through"

    def solve(self, target: IkTarget, seed_rad):
        seed = np.asarray(seed_rad, dtype=float)
        return IkResult(joints_rad=tuple(float(v) for v in seed), residual_position_m=0.0,
                        residual_rotation_rad=0.0, iterations=1,
                        tolerance_position_m=1e-4, tolerance_rotation_rad=1e-3,
                        solver=self.solver_id)


class FixedSolutionIk(IkSolver):
    """Test stub: always returns one fixed converged pose, whatever the target.

    Used to put the CONVERGED joint vector far from the plan seed, which is the
    only way to make the time-sufficiency check the binding constraint.
    """

    solver_id = "test/fixed_solution"
    SOLUTION = (-1.5, 2.8, -0.1, 1.5, 1.0, 2.0)

    def solve(self, target: IkTarget, seed_rad):
        return IkResult(joints_rad=self.SOLUTION, residual_position_m=0.0,
                        residual_rotation_rad=0.0, iterations=1,
                        tolerance_position_m=1e-4, tolerance_rotation_rad=1e-3,
                        solver=self.solver_id)


class ExplodingIk(IkSolver):
    solver_id = "test/exploding"

    def solve(self, target: IkTarget, seed_rad):
        raise RuntimeError("solver exploded")


def verified():
    return Calibration(court_from_base=np.eye(4), verified=True)


def frame(frame_id=1, capture=0.0, received=0.002, dim=3, position=None, sigma=0.01):
    return FrameInput(frame_id=frame_id, capture_time_s=capture, received_at_s=received,
                      detection_dim=dim,
                      position_m=(INTERCEPT if position is None else position)
                      if dim == 3 else None,
                      sigma_m=sigma if dim == 3 else None)


def prediction(now=0.002, lead=0.30, sigma=0.01, growth=0.02, point=None, pid="p1",
               approach=APPROACH):
    return PredictionInput(prediction_id=pid, frame_id=1, reference_time_s=now,
                           intercept_time_s=now + lead,
                           intercept_point_m=INTERCEPT if point is None else point,
                           approach_direction=approach, sigma_start_m=sigma,
                           sigma_growth_mps=growth)


def plan(now=0.002, joints=SEED, seed=SEED, duration=0.002):
    return PlanInput(planned_at_s=now, joints_rad=joints, seed_rad=seed, duration_s=duration)


def make_fsm(**kwargs):
    kwargs.setdefault("calibration", verified())
    kwargs.setdefault("ik_solver", PassThroughIk())
    return InteractionFsm(**kwargs)


def drive_to_execute(fsm, now=0.002):
    assert fsm.on_frame(frame(), now) is FsmState.TRACK
    assert fsm.on_prediction(prediction(now=now), now) is FsmState.PREDICT
    assert fsm.on_plan(plan(now=now), now) is FsmState.PLAN
    return fsm.on_execute_command(SEED, now)


class TransitionTests(unittest.TestCase):
    def test_policy_constants_are_pinned(self):
        # the stale threshold must be the SAME pre-registered number as P7-08
        self.assertEqual(STALE_FRAME_AGE_S, MAX_SOURCE_FRAME_AGE_S)
        self.assertEqual(STALE_FRAME_AGE_S, 0.100)

    def test_transition_table(self):
        self.assertEqual(set(ALLOWED_TRANSITIONS), set(FsmState))
        self.assertEqual(ALLOWED_TRANSITIONS[FsmState.SEARCH],
                         (FsmState.TRACK, FsmState.HOLD))
        self.assertEqual(ALLOWED_TRANSITIONS[FsmState.TRACK],
                         (FsmState.PREDICT, FsmState.HOLD))
        self.assertEqual(ALLOWED_TRANSITIONS[FsmState.PREDICT],
                         (FsmState.PLAN, FsmState.HOLD))
        self.assertEqual(ALLOWED_TRANSITIONS[FsmState.PLAN],
                         (FsmState.EXECUTE, FsmState.HOLD))
        self.assertEqual(ALLOWED_TRANSITIONS[FsmState.EXECUTE],
                         (FsmState.PREDICT, FsmState.PLAN, FsmState.HOLD))
        self.assertEqual(ALLOWED_TRANSITIONS[FsmState.HOLD], (FsmState.TRACK,))

    def test_happy_path_transitions_and_one_command(self):
        fsm = make_fsm()
        self.assertIs(fsm.state, FsmState.SEARCH)
        state = drive_to_execute(fsm)
        self.assertIs(state, FsmState.EXECUTE)
        self.assertEqual([e.to_state for e in fsm.events],
                         [FsmState.TRACK, FsmState.PREDICT, FsmState.PLAN, FsmState.EXECUTE])
        self.assertEqual(len(fsm.emitted_commands), 1)
        self.assertEqual(fsm.emitted_commands[0]["joints_rad"], [float(v) for v in SEED])
        self.assertEqual(fsm.suppressed_commands, 0)

    def test_execute_to_plan_on_cycle_complete(self):
        fsm = make_fsm()
        drive_to_execute(fsm)
        self.assertIs(fsm.on_cycle_complete(0.01), FsmState.PLAN)
        self.assertIs(fsm.state, FsmState.PLAN)

    def test_execute_replans_on_a_new_prediction(self):
        fsm = make_fsm()
        drive_to_execute(fsm)
        self.assertIs(fsm.on_prediction(prediction(now=0.01, pid="p2"), 0.01),
                      FsmState.PREDICT)

    def test_plan_before_prediction_is_fail_closed(self):
        fsm = make_fsm()
        fsm.on_frame(frame(), 0.002)
        self.assertIs(fsm.on_plan(plan(), 0.002), FsmState.HOLD)
        self.assertIs(fsm.hold_reason, HoldReason.STAGE_ORDER)

    def test_frame_while_executing_only_updates_tracking(self):
        fsm = make_fsm()
        drive_to_execute(fsm)
        self.assertIs(fsm.on_frame(frame(frame_id=2), 0.01), FsmState.EXECUTE)
        self.assertEqual(fsm.events[-1].trigger, "FRAME_UPDATE")
        self.assertEqual(len(fsm.emitted_commands), 1)

    def test_frame_during_planning_is_fail_closed(self):
        fsm = make_fsm()
        fsm.on_frame(frame(), 0.002)
        fsm.on_prediction(prediction(now=0.002), 0.002)
        fsm.on_plan(plan(now=0.002), 0.002)
        self.assertIs(fsm.state, FsmState.PLAN)
        self.assertIs(fsm.on_frame(frame(frame_id=2), 0.02), FsmState.HOLD)
        self.assertIs(fsm.hold_reason, HoldReason.STAGE_ORDER)

    def test_real_solver_integration(self):
        # A target the PROXY arm can actually reach: reached from the inside out so the
        # test pins the integration, not a lucky seed.  (The proxy arm cannot reach
        # every orientation at every point - see the UNKNOWNs in the P7 doc.)
        target = target_from_intercept(REACHABLE_INTERCEPT, REACHABLE_APPROACH)
        q_star = np.asarray(DampedLeastSquaresIk().solve(target, np.zeros(6)).joints_rad)
        fsm = InteractionFsm(calibration=verified(), ik_solver=DampedLeastSquaresIk())
        fsm.on_frame(frame(position=REACHABLE_INTERCEPT), 0.002)
        fsm.on_prediction(prediction(now=0.002, lead=0.30, point=REACHABLE_INTERCEPT,
                                     approach=REACHABLE_APPROACH), 0.002)
        state = fsm.on_plan(plan(now=0.002, seed=q_star, joints=q_star), 0.002)
        self.assertIs(state, FsmState.PLAN)
        verdict = fsm.last_verdict
        self.assertEqual(verdict.decision.value, "REACHABLE")
        self.assertIsNotNone(verdict.ik)
        self.assertLessEqual(verdict.ik.residual_position_m, 1e-4)


class FailClosedTests(unittest.TestCase):
    def _assert_hold(self, fsm, reason, state_call):
        state = state_call()
        self.assertIs(state, FsmState.HOLD, f"expected HOLD, got {state}")
        self.assertIs(fsm.hold_reason, reason)
        self.assertTrue(fsm.report()["hold"]["latched"])
        self.assertEqual(fsm.emitted_commands, [])
        return fsm

    def test_missing_calibration(self):
        fsm = make_fsm(calibration=Calibration())
        self._assert_hold(fsm, HoldReason.CALIBRATION_MISSING,
                          lambda: fsm.on_frame(frame(), 0.002))

    def test_stale_frame(self):
        fsm = make_fsm()
        stale = frame(capture=0.0, received=0.5)
        self._assert_hold(fsm, HoldReason.STALE_FRAME, lambda: fsm.on_frame(stale, 0.5))

    def test_future_frame(self):
        fsm = make_fsm()
        self._assert_hold(fsm, HoldReason.STALE_FRAME,
                          lambda: fsm.on_frame(frame(capture=1.0, received=0.002), 0.002))

    def test_covariance_above_threshold_at_the_frame(self):
        fsm = make_fsm()
        self._assert_hold(fsm, HoldReason.COVARIANCE_ABOVE_THRESHOLD,
                          lambda: fsm.on_frame(frame(sigma=0.4), 0.002))

    def test_zero_and_reversed_time_to_hit(self):
        for lead in (0.0, -0.3):
            fsm = make_fsm()
            fsm.on_frame(frame(), 0.002)
            self._assert_hold(fsm, HoldReason.TIME_TO_HIT_INVALID,
                              lambda: fsm.on_prediction(prediction(now=0.002, lead=lead),
                                                        0.002))

    def test_time_to_hit_below_minimum(self):
        fsm = make_fsm()
        fsm.on_frame(frame(), 0.002)
        self._assert_hold(fsm, HoldReason.TIME_TO_HIT_INVALID,
                          lambda: fsm.on_prediction(prediction(now=0.002, lead=0.01), 0.002))

    def test_prediction_covariance_above_threshold(self):
        fsm = make_fsm()
        fsm.on_frame(frame(), 0.002)
        self._assert_hold(fsm, HoldReason.COVARIANCE_ABOVE_THRESHOLD,
                          lambda: fsm.on_prediction(prediction(now=0.002, sigma=0.09), 0.002))

    def test_covariance_growth_blows_the_budget_during_planning(self):
        fsm = make_fsm()
        fsm.on_frame(frame(), 0.002)
        fsm.on_prediction(prediction(now=0.002, sigma=0.04, growth=0.50), 0.002)
        slow_plan = PlanInput(planned_at_s=0.002, joints_rad=SEED, seed_rad=SEED,
                              duration_s=0.10)
        self._assert_hold(fsm, HoldReason.COVARIANCE_ABOVE_THRESHOLD,
                          lambda: fsm.on_plan(slow_plan, 0.002))

    def test_unreachable_target(self):
        fsm = make_fsm()
        fsm.on_frame(frame(), 0.002)
        far = prediction(now=0.002, point=np.array([4.0, 0.0, 1.0]))
        fsm.on_prediction(far, 0.002)
        self._assert_hold(fsm, HoldReason.UNREACHABLE_TARGET,
                          lambda: fsm.on_plan(plan(now=0.002), 0.002))

    def test_unknown_reachability_is_not_a_licence_to_move(self):
        from src.badminton_brain.reachability import UnavailableIk
        fsm = make_fsm(ik_solver=UnavailableIk())
        fsm.on_frame(frame(), 0.002)
        fsm.on_prediction(prediction(now=0.002), 0.002)
        self._assert_hold(fsm, HoldReason.UNREACHABLE_TARGET,
                          lambda: fsm.on_plan(plan(now=0.002), 0.002))
        self.assertIn("UNKNOWN", fsm.hold_detail)

    def test_time_insufficient_for_the_slew(self):
        fsm = make_fsm(ik_solver=FixedSolutionIk())
        fsm.on_frame(frame(), 0.002)
        # 60 ms of flight, converged pose 2 rad away from the seed: 1.0 s of slew
        fsm.on_prediction(prediction(now=0.002, lead=0.06), 0.002)
        self._assert_hold(fsm, HoldReason.UNREACHABLE_TARGET,
                          lambda: fsm.on_plan(plan(now=0.002), 0.002))
        self.assertIn("required slew", fsm.hold_detail)

    def test_joint_limit_margin_violation_on_the_command(self):
        fsm = make_fsm()
        drive_to_execute(fsm)
        # joint2 driven to 3.10 rad: 0.0416 rad from its upper stop, inside 0.5 rad margin
        over = np.array([0.3, 3.10, -1.2, 0.1, 0.4, 0.2])
        limits = KinematicLimits(joint_margin_rad=0.0)
        fsm2 = make_fsm(kinematic_limits=limits, limits=FsmLimits(joint_margin_rad=0.5))
        drive_to_execute(fsm2)
        fsm2.on_cycle_complete(0.02)
        self.assertIs(fsm2.on_execute_command(over, 0.02), FsmState.HOLD)
        self.assertIs(fsm2.hold_reason, HoldReason.JOINT_LIMIT_MARGIN)

    def test_velocity_cap_violation(self):
        fsm = make_fsm()
        drive_to_execute(fsm)
        fast = SEED + 0.10  # 0.10 rad in one 1/60 s step = 6 rad/s > 2.0 rad/s cap
        fsm.on_cycle_complete(0.02)
        self.assertIs(fsm.on_execute_command(fast, 0.02), FsmState.HOLD)
        self.assertIs(fsm.hold_reason, HoldReason.COMMAND_OVER_CAP)
        self.assertIn("rad/s >", fsm.hold_detail)

    def test_acceleration_cap_violation(self):
        fsm = make_fsm()
        drive_to_execute(fsm)
        fsm.on_cycle_complete(0.02)
        # 0.01 rad/step = 0.6 rad/s (inside the velocity cap) but 36 rad/s^2 > 20
        self.assertIs(fsm.on_execute_command(SEED + 0.01, 0.02), FsmState.HOLD)
        self.assertIs(fsm.hold_reason, HoldReason.COMMAND_OVER_CAP)
        self.assertIn("rad/s^2 >", fsm.hold_detail)

    def test_exception_in_the_ik_stage(self):
        fsm = make_fsm(ik_solver=ExplodingIk())
        fsm.on_frame(frame(), 0.002)
        fsm.on_prediction(prediction(now=0.002), 0.002)
        self._assert_hold(fsm, HoldReason.STAGE_EXCEPTION,
                          lambda: fsm.on_plan(plan(now=0.002), 0.002))
        self.assertIn("solver exploded", fsm.hold_detail)

    def test_two_dimensional_detection_is_never_promoted(self):
        from src.badminton_brain.reachability import DetectionDimensionalityError
        with self.assertRaises(DetectionDimensionalityError):
            FrameInput(frame_id=9, capture_time_s=0.0, received_at_s=0.002,
                       detection_dim=2, position_m=INTERCEPT)
        with self.assertRaises(DetectionDimensionalityError):
            PredictionInput(prediction_id="p", frame_id=9, reference_time_s=0.0,
                            intercept_time_s=0.3, intercept_point_m=[10.0, 20.0],
                            approach_direction=APPROACH, sigma_start_m=0.01,
                            sigma_growth_mps=0.02)

    def test_two_dimensional_frame_cannot_reach_the_prediction_stage(self):
        fsm = make_fsm()
        self.assertIs(fsm.on_frame(frame(dim=2), 0.002), FsmState.TRACK)
        self.assertIs(fsm.on_prediction(prediction(now=0.002), 0.002), FsmState.HOLD)
        self.assertIs(fsm.hold_reason, HoldReason.DETECTION_NOT_3D)
        self.assertEqual(fsm.emitted_commands, [])


class HoldLatchTests(unittest.TestCase):
    def _hold(self, fsm):
        fsm.on_frame(frame(capture=0.0, received=0.5), 0.5)
        self.assertIs(fsm.state, FsmState.HOLD)
        return fsm

    def test_every_stage_is_suppressed_while_holding(self):
        fsm = self._hold(make_fsm())
        self.assertIs(fsm.on_frame(frame(), 0.5), FsmState.HOLD)
        self.assertIs(fsm.on_prediction(prediction(now=0.5), 0.5), FsmState.HOLD)
        self.assertIs(fsm.on_plan(plan(now=0.5), 0.5), FsmState.HOLD)
        self.assertIs(fsm.on_cycle_complete(0.5), FsmState.HOLD)
        self.assertGreaterEqual(fsm.suppressed_commands, 4)
        self.assertEqual(fsm.emitted_commands, [])

    def test_no_command_is_emitted_while_in_hold(self):
        fsm = self._hold(make_fsm())
        with self.assertRaises(CommandSuppressed):
            fsm.on_execute_command(SEED, 0.5)
        self.assertEqual(fsm.emitted_commands, [])
        self.assertEqual(fsm.report()["commands_emitted"], 0)

    def test_hold_does_not_recover_by_itself(self):
        fsm = self._hold(make_fsm())
        for step in range(50):  # a whole second of valid-looking traffic
            fsm.on_frame(frame(frame_id=step + 2, capture=0.5, received=0.5), 0.5)
        self.assertIs(fsm.state, FsmState.HOLD)
        self.assertIs(fsm.hold_reason, HoldReason.STALE_FRAME)
        self.assertEqual(fsm.emitted_commands, [])

    def test_reset_with_a_stale_frame_is_refused_and_latches(self):
        fsm = self._hold(make_fsm())
        with self.assertRaises(ResetRejected):
            fsm.reset(frame=frame(capture=0.0, received=0.9),
                      prediction=prediction(now=0.9), now_s=1.0, operator="engineer")
        self.assertIs(fsm.state, FsmState.HOLD)
        self.assertIs(fsm.hold_reason, HoldReason.STALE_FRAME)

    def test_reset_with_a_two_dimensional_frame_is_refused(self):
        fsm = self._hold(make_fsm())
        with self.assertRaises(ResetRejected):
            fsm.reset(frame=frame(dim=2), prediction=prediction(now=0.5), now_s=0.5,
                      operator="engineer")
        self.assertIs(fsm.state, FsmState.HOLD)

    def test_reset_requires_a_named_operator(self):
        fsm = self._hold(make_fsm())
        with self.assertRaises(ResetRejected):
            fsm.reset(frame=frame(capture=0.5, received=0.5), prediction=prediction(now=0.5),
                      now_s=0.5, operator="  ")
        self.assertIs(fsm.state, FsmState.HOLD)

    def test_reset_outside_hold_is_refused(self):
        fsm = make_fsm()
        with self.assertRaises(ResetRejected):
            fsm.reset(frame=frame(), prediction=prediction(), now_s=0.002,
                      operator="engineer")

    def test_valid_reset_returns_to_track_only(self):
        fsm = self._hold(make_fsm())
        fresh_frame = frame(frame_id=3, capture=0.5, received=0.5)
        fresh_prediction = prediction(now=0.5, pid="p-reset")
        self.assertIs(fsm.reset(frame=fresh_frame, prediction=fresh_prediction, now_s=0.5,
                                operator="engineer"), FsmState.TRACK)
        self.assertIsNone(fsm.hold_reason)
        self.assertFalse(fsm.holding)
        # a reset NEVER jumps straight back to EXECUTE
        self.assertIs(fsm.on_execute_command(SEED, 0.5), FsmState.HOLD)
        self.assertIs(fsm.hold_reason, HoldReason.STAGE_ORDER)
        self.assertEqual(fsm.report()["commands_emitted"], 0)


class PropagationTests(unittest.TestCase):
    def test_sigma_and_time_to_hit_are_carried_into_the_decision(self):
        fsm = make_fsm()
        fsm.on_frame(frame(), 0.002)
        fsm.on_prediction(prediction(now=0.002, sigma=0.010, growth=0.05), 0.002)
        fsm.on_plan(PlanInput(planned_at_s=0.002, joints_rad=SEED, seed_rad=SEED,
                              duration_s=0.10), 0.002)
        states = [(e.trigger, e.time_to_hit_s, e.sigma_m) for e in fsm.events]
        prediction_event = [s for s in states if s[0] == "PREDICTION"][0]
        plan_event = [s for s in states if s[0] == "PLAN"][0]
        self.assertAlmostEqual(prediction_event[1], 0.30, places=6)
        self.assertAlmostEqual(plan_event[1], 0.20, places=6)  # 0.10 s spent planning
        self.assertAlmostEqual(prediction_event[2], 0.010, places=6)
        self.assertAlmostEqual(plan_event[2], 0.015, places=6)  # 0.05 m/s * 0.10 s

    def test_report_contains_the_typed_reasons(self):
        fsm = make_fsm()
        fsm.on_frame(frame(capture=0.0, received=0.5), 0.5)
        report = fsm.report()
        self.assertEqual(report["hold"]["reason"], "STALE_FRAME")
        self.assertEqual(report["hold_reasons_seen"], {"STALE_FRAME": 1})
        self.assertEqual(report["evidence_class"], "OFFLINE-SYNTHETIC")
        self.assertEqual(report["state"], "HOLD")
        self.assertTrue(all("reason" in event for event in report["events"]))


if __name__ == "__main__":
    unittest.main()
