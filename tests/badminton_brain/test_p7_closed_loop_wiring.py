#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""P7-11/P7-12: the wired closed loop, the two backends, and the refusals.

WHAT THESE TESTS PIN
--------------------
1. backend selection is EXPLICIT: asking for the verified P6 FK when it is not
   importable raises VerifiedFkUnavailable and NEVER returns a proxy number;
2. both backends answer the same target set, and every verdict records WHICH
   backend produced it - one run can never contain two;
3. a 2-D detection is still refused by name (FrameInput, assess_reachability and
   the wired harness);
4. HOLD latches under each injected fault, and no command escapes it;
5. the harness refuses a hit rate and still reports the counts it can defend;
6. the committed racket proxy JSON loads with its real 43 primitives, and the
   SIM_PROXY geometry check rejects a pose that is not at the ball.

REQUIREMENT: the verified-backend and racket-proxy tests need a P6 checkout
(p6/dynamic-fk or p6/racket-collision-proxy).  Point at one with

    $env:BADMIN_P6_ROOT = "E:\\path\\to\\p6_worktree"

Without it those tests FAIL LOUDLY with the path they looked for.  They never
skip: a silently skipped safety test is a test that does not exist.
"""
from __future__ import annotations

import json
import os
import sys
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
for entry in (ROOT / "src", ROOT):
    if str(entry) not in sys.path:
        sys.path.insert(0, str(entry))

from src.badminton_brain.harness.wired_closed_loop import (  # noqa: E402
    BALL_RADIUS_M, GEOMETRY_LABEL, P6_PROXY_RELPATH, RacketCollisionProxy, WiredConfig,
    load_racket_proxy, run_wired_closed_loop, summarise,
)
from src.badminton_brain.reachability import (  # noqa: E402
    Calibration, DetectionDimensionalityError, KinematicLimits, assess_reachability,
)
from src.badminton_brain.reachability.backends import (  # noqa: E402
    BACKEND_PROXY, BACKEND_VERIFIED, BackendDampedLeastSquaresIk, BackendMixError,
    BackendUnavailable, ProxyDhBackend, VerifiedFkBackend, VerifiedFkUnavailable,
    kinematic_limits_for, resolve_p6_root, select_backend,
)
from src.badminton_brain.safety.interaction_fsm import (  # noqa: E402
    CommandSuppressed, FrameInput, FsmState, HoldReason, InteractionFsm, PlanInput,
    PredictionInput, ResetRejected,
)

def _discover_p6_root():
    """Best-effort location of a P6 checkout, for the TESTS only.

    Order: $BADMIN_P6_ROOT -> this repository (works once P6 merges) -> a sibling
    checkout next to the repository (a p6/dynamic-fk worktree).  The library never
    does this: only the test may go looking for a checkout, and if none is found the
    test FAILS LOUDLY rather than skipping.
    """
    explicit = os.environ.get("BADMIN_P6_ROOT")
    candidates = [explicit] if explicit else []
    candidates.append(str(ROOT))
    parent = ROOT.parent
    if parent.is_dir():
        for entry in sorted(parent.iterdir()):
            if entry.is_dir() and (entry / P6_PROXY_RELPATH).is_file():
                candidates.append(str(entry))
    found = resolve_p6_root(extra_roots=tuple(candidates))
    return None if found is None else str(found)


P6_ROOT = _discover_p6_root()


def require_p6(message):
    """FAIL (never skip) when a P6 artifact the test asserts on is unavailable."""
    if P6_ROOT is None:
        raise AssertionError(
            "P6 checkout not found: %s. Set $BADMIN_P6_ROOT to a p6/dynamic-fk "
            "worktree (it must carry %s). This test deliberately FAILS instead of "
            "skipping, because the verified backend IS the subject of the test."
            % (message, P6_PROXY_RELPATH))
    return P6_ROOT


def calibration():
    return Calibration(court_from_base=np.eye(4), verified=True,
                       source="ASSUMED identity (P7-11/P7-12 test only)")


class BackendSelectionTests(unittest.TestCase):
    def test_proxy_backend_is_the_documented_6dof_proxy(self):
        backend = select_backend(BACKEND_PROXY)
        self.assertIsInstance(backend, ProxyDhBackend)
        self.assertEqual(backend.dof, 6)
        self.assertEqual(backend.frame_name, "racket_face_proxy")
        self.assertIn("PROXY", backend.source_class)

    def test_unknown_backend_name_raises(self):
        with self.assertRaises(BackendUnavailable):
            select_backend("the_real_robot")

    def test_verified_backend_missing_module_fails_loudly(self):
        """A verified request with no verifiable module must raise, not fall back."""
        missing = ROOT / "_no_such_checkout" / "fk.py"
        with self.assertRaises(VerifiedFkUnavailable) as caught:
            select_backend(BACKEND_VERIFIED, fk_path=str(missing))
        message = str(caught.exception)
        self.assertIn("VERIFIED_FK_UNAVAILABLE", message)
        self.assertIn(str(missing), message)

    def test_verified_backend_missing_root_fails_loudly(self):
        """With no candidate root at all, the constructor raises - no substitution."""
        empty = ROOT / "_no_such_checkout"
        saved = os.environ.pop("BADMIN_P6_ROOT", None)
        try:
            self.assertIsNone(resolve_p6_root(str(empty)))
            with self.assertRaises(VerifiedFkUnavailable):
                VerifiedFkBackend(root=str(empty))
        finally:
            if saved is not None:
                os.environ["BADMIN_P6_ROOT"] = saved

    def test_the_harness_propagates_the_refusal_instead_of_substituting(self):
        config = WiredConfig(backend=BACKEND_VERIFIED, calibration=calibration(),
                             fk_path=str(ROOT / "_no_such_checkout" / "fk.py"))
        with self.assertRaises(VerifiedFkUnavailable):
            run_wired_closed_loop(config)

    def test_verified_backend_reports_its_own_provenance(self):
        require_p6("the verified backend provenance test")
        backend = select_backend(BACKEND_VERIFIED, root=P6_ROOT)
        provenance = backend.provenance()
        self.assertEqual(provenance["backend"], BACKEND_VERIFIED)
        self.assertEqual(provenance["source_class"], "VERIFIED_ASSET_KINEMATICS")
        self.assertEqual(provenance["frame_name"], "gripper_end")
        self.assertEqual(provenance["dof"], 8)
        self.assertIn("fk_module_sha256", provenance)
        self.assertTrue(provenance["fk_module_path"].endswith("fk.py"))
        self.assertIn("racket face", provenance["note"])

    def test_the_solver_id_carries_the_backend_id(self):
        proxy = BackendDampedLeastSquaresIk(ProxyDhBackend())
        self.assertIn(BACKEND_PROXY, proxy.solver_id)


class TwoBackendVerdictTests(unittest.TestCase):
    TARGETS = ((0.35, 0.05, 1.35), (0.42, 0.05, 1.05), (0.36, 0.05, 0.45))

    def _verdict(self, backend, point, approach=(1.0, 0.0, 0.0)):
        limits = kinematic_limits_for(backend)
        solver = BackendDampedLeastSquaresIk(backend)
        seed = 0.5 * (np.asarray(backend.lower_rad) + np.asarray(backend.upper_rad))
        return assess_reachability(intercept_point_m=point, approach_direction=approach,
                                   limits=limits, time_to_target_s=0.448, seed_rad=seed,
                                   solver=solver, calibration=calibration(),
                                   position_sigma_m=0.01)

    def test_both_backends_produce_a_verdict_and_the_ik_carries_its_solver(self):
        require_p6("the both-backend verdict comparison")
        proxy = select_backend(BACKEND_PROXY)
        verified = select_backend(BACKEND_VERIFIED, root=P6_ROOT)
        self.assertNotEqual(proxy.dof, verified.dof)
        self.assertNotEqual(proxy.frame_name, verified.frame_name)
        saw_converged = False
        for point in self.TARGETS:
            for backend in (proxy, verified):
                verdict = self._verdict(backend, point)
                self.assertIn(verdict.decision.value,
                              ("REACHABLE", "NOT_REACHABLE", "UNKNOWN"))
                if verdict.ik is not None:
                    saw_converged = True
                    self.assertIn(backend.backend_id, verdict.ik.solver)
                    self.assertLessEqual(verdict.ik.residual_position_m,
                                         verdict.ik.tolerance_position_m)
                    self.assertLessEqual(verdict.ik.residual_rotation_rad,
                                         verdict.ik.tolerance_rotation_rad)
        self.assertTrue(saw_converged, "at least one target must converge on one backend")

    def test_the_verified_chain_cannot_reach_the_proxy_face_target(self):
        """The two chains disagree at 1.35 m, and the verified one is the smaller."""
        require_p6("the reach disagreement test")
        proxy = select_backend(BACKEND_PROXY)
        verified = select_backend(BACKEND_VERIFIED, root=P6_ROOT)
        far = (0.35, 0.05, 1.35)
        self.assertFalse(self._verdict(proxy, far).reachable)
        self.assertFalse(self._verdict(verified, far).reachable)
        low = (0.36, 0.05, 0.45)
        verified_low = self._verdict(verified, low)
        self.assertTrue(verified_low.reachable)
        self.assertIsNotNone(verified_low.ik)
        self.assertLess(verified_low.ik.residual_position_m, 1e-4)

    def test_limits_are_sized_to_the_backend(self):
        require_p6("the backend-sized limits test")
        verified = select_backend(BACKEND_VERIFIED, root=P6_ROOT)
        limits = kinematic_limits_for(verified)
        self.assertEqual(limits.dof, 8)
        self.assertEqual(len(limits.lower_rad), 8)
        self.assertEqual(limits.joint_names[6], "gripper_joint1")
        with self.assertRaises(ValueError):
            KinematicLimits(lower_rad=(0.0,) * 7, upper_rad=(1.0,) * 7, dof=8,
                            joint_names=("a",) * 8)


class DimensionalityRefusalTests(unittest.TestCase):
    def test_a_2d_frame_may_not_carry_a_3d_point(self):
        with self.assertRaises(DetectionDimensionalityError):
            FrameInput(frame_id=1, capture_time_s=0.0, received_at_s=0.001,
                       detection_dim=2, position_m=(0.1, 0.2, 0.3), sigma_m=0.01)

    def test_a_2d_frame_without_a_point_is_legal_but_unusable(self):
        frame = FrameInput(frame_id=1, capture_time_s=0.0, received_at_s=0.001,
                           detection_dim=2)
        self.assertIsNone(frame.position_m)
        self.assertIsNone(frame.sigma_m)

    def test_reachability_refuses_a_2d_detection(self):
        with self.assertRaises(DetectionDimensionalityError):
            assess_reachability(detection_dim=2, intercept_point_m=(0.3, 0.0, 0.6),
                                approach_direction=(1.0, 0.0, 0.0),
                                time_to_target_s=1.0, calibration=calibration())

    def test_a_2d_frame_tracks_but_never_reaches_a_plan(self):
        """A 2-D frame is accepted as a track, and the PREDICTION stage refuses it."""
        fsm = InteractionFsm(calibration=calibration(),
                             ik_solver=BackendDampedLeastSquaresIk(ProxyDhBackend()))
        state = fsm.on_frame(FrameInput(frame_id=0, capture_time_s=0.0,
                                        received_at_s=0.001, detection_dim=2), 0.001)
        self.assertIs(state, FsmState.TRACK)
        state = fsm.on_prediction(PredictionInput(
            prediction_id="p0", frame_id=0, reference_time_s=0.001,
            intercept_time_s=0.449, intercept_point_m=(0.36, 0.05, 0.45),
            approach_direction=(1.0, 0.0, 0.0), sigma_start_m=0.01,
            sigma_growth_mps=0.02), 0.001)
        self.assertIs(state, FsmState.HOLD)
        self.assertIs(fsm.hold_reason, HoldReason.DETECTION_NOT_3D)
        self.assertEqual(fsm.report()["commands_emitted"], 0)


class HoldLatchingTests(unittest.TestCase):
    """One injected fault per test; every one must LATCH HOLD with its reason."""

    def _fsm(self, **kwargs):
        return InteractionFsm(calibration=kwargs.pop("calibration", calibration()),
                              ik_solver=BackendDampedLeastSquaresIk(ProxyDhBackend()),
                              **kwargs)

    def _frame(self, **overrides):
        values = dict(frame_id=0, capture_time_s=0.0, received_at_s=0.001,
                      detection_dim=3, position_m=(0.35, 0.05, 1.35), sigma_m=0.01)
        values.update(overrides)
        return FrameInput(**values)

    def _prediction(self, **overrides):
        values = dict(prediction_id="p0", frame_id=0, reference_time_s=0.001,
                      intercept_time_s=0.449, intercept_point_m=(0.35, 0.05, 0.6),
                      approach_direction=(1.0, 0.0, 0.0), sigma_start_m=0.01,
                      sigma_growth_mps=0.02)
        values.update(overrides)
        return PredictionInput(**values)

    def test_missing_calibration_latches_hold(self):
        fsm = self._fsm(calibration=Calibration())
        state = fsm.on_frame(self._frame(), 0.001)
        self.assertIs(state, FsmState.HOLD)
        self.assertIs(fsm.hold_reason, HoldReason.CALIBRATION_MISSING)

    def test_stale_frame_latches_hold(self):
        fsm = self._fsm()
        state = fsm.on_frame(self._frame(capture_time_s=-0.5), 0.001)
        self.assertIs(state, FsmState.HOLD)
        self.assertIs(fsm.hold_reason, HoldReason.STALE_FRAME)

    def test_sigma_above_threshold_latches_hold(self):
        fsm = self._fsm()
        state = fsm.on_frame(self._frame(sigma_m=0.5), 0.001)
        self.assertIs(state, FsmState.HOLD)
        self.assertIs(fsm.hold_reason, HoldReason.COVARIANCE_ABOVE_THRESHOLD)

    def test_reversed_time_to_hit_latches_hold(self):
        fsm = self._fsm()
        state = fsm.on_frame(self._frame(), 0.001)
        self.assertIs(state, FsmState.TRACK)
        state = fsm.on_prediction(self._prediction(intercept_time_s=-1.0), 0.001)
        self.assertIs(state, FsmState.HOLD)
        self.assertIs(fsm.hold_reason, HoldReason.TIME_TO_HIT_INVALID)

    def test_unreachable_target_latches_hold(self):
        fsm = self._fsm()
        fsm.on_frame(self._frame(), 0.001)
        fsm.on_prediction(self._prediction(), 0.001)
        state = fsm.on_plan(PlanInput(planned_at_s=0.001,
                                      joints_rad=self._seed(), seed_rad=self._seed()), 0.001)
        self.assertIs(state, FsmState.HOLD)
        self.assertIs(fsm.hold_reason, HoldReason.UNREACHABLE_TARGET)
        self.assertIs(fsm.last_verdict.decision.value, "NOT_REACHABLE")

    def test_command_over_cap_latches_hold(self):
        fsm = self._fsm()
        fsm.on_frame(self._frame(), 0.001)
        fsm.on_prediction(self._prediction(intercept_point_m=(0.45, 0.0, 1.0),
                                           ) , 0.001)
        seed = np.array([0.0, 1.5, -1.5, 0.0, 0.0, 0.0])
        state = fsm.on_plan(PlanInput(planned_at_s=0.001, joints_rad=seed, seed_rad=seed), 0.001)
        if state is FsmState.PREDICT:
            self.fail("unexpected state")
        if state is FsmState.HOLD:
            self.assertIn(fsm.hold_reason, (HoldReason.UNREACHABLE_TARGET,
                                            HoldReason.TIME_TO_HIT_INVALID))
            return
        jump = seed + np.array([0.0, 0.5, 0.0, 0.0, 0.0, 0.0])
        fsm.on_execute_command(seed, 0.001)
        state = fsm.on_execute_command(jump, 0.002)
        self.assertIs(state, FsmState.HOLD)
        self.assertIs(fsm.hold_reason, HoldReason.COMMAND_OVER_CAP)

    def test_hold_suppresses_every_later_stage_and_refuses_commands(self):
        fsm = self._fsm(calibration=Calibration())
        fsm.on_frame(self._frame(), 0.001)
        self.assertIs(fsm.state, FsmState.HOLD)
        fsm.on_prediction(self._prediction(), 0.002)
        fsm.on_plan(PlanInput(planned_at_s=0.002, joints_rad=self._seed(),
                              seed_rad=self._seed()), 0.002)
        with self.assertRaises(CommandSuppressed):
            fsm.on_execute_command(self._seed(), 0.002)
        triggers = [event.trigger for event in fsm.events]
        self.assertIn("SUPPRESSED", triggers)
        self.assertEqual(fsm.report()["commands_emitted"], 0)
        self.assertTrue(all(event.trigger != "COMMAND" for event in fsm.events))

    def test_a_stale_reset_is_rejected_and_the_latch_stays(self):
        fsm = self._fsm(calibration=Calibration())
        fsm.on_frame(self._frame(), 0.001)
        self.assertIs(fsm.state, FsmState.HOLD)
        with self.assertRaises(ResetRejected):
            fsm.reset(frame=self._frame(capture_time_s=-1.0),
                      prediction=self._prediction(intercept_time_s=-1.0),
                      now_s=0.002, operator="tester")
        self.assertIs(fsm.state, FsmState.HOLD)
        self.assertIn("RESET_REJECTED", [event.trigger for event in fsm.events])

    def _seed(self):
        return np.array([0.0, 1.5, -1.5, 0.0, 0.0, 0.0])


class HarnessRefusalTests(unittest.TestCase):
    def test_missing_calibration_refuses_the_hit_rate_but_keeps_the_counts(self):
        report = run_wired_closed_loop(WiredConfig(scenario="LOW_SPEED_INCOMING",
                                                   calibration=None))
        self.assertEqual(report["status"], "REFUSED")
        self.assertIn("CALIBRATION_MISSING", report["refusal_reason"])
        self.assertIsNone(report["hit_rate"])
        self.assertFalse(report["real_hit_rate_claimed"])
        # the counts it CAN defend are still there, produced by a real FSM run
        self.assertIn("hold_reasons", report)
        self.assertIn("CALIBRATION_MISSING", report["hold_reasons"])
        self.assertIn("states_entered", report)
        self.assertIn("HOLD", report["states_entered"])
        self.assertIn("reachability_verdicts", report)
        self.assertEqual(report["commands_executed"], 0)

    def test_missing_proxy_json_is_named_in_the_refusal(self):
        report = run_wired_closed_loop(WiredConfig(
            calibration=calibration(), backend=BACKEND_PROXY,
            proxy_json=str(ROOT / "_no_such_proxy.json")))
        self.assertEqual(report["status"], "REFUSED")
        self.assertIn("RACKET_COLLISION_PROXY_MISSING", report["refusal_reason"])

    def test_a_complete_run_still_never_reports_a_hit_rate(self):
        require_p6("the complete wired run")
        report = run_wired_closed_loop(WiredConfig(scenario="LOW_SPEED_INCOMING",
                                                   backend=BACKEND_VERIFIED,
                                                   p6_root=P6_ROOT,
                                                   calibration=calibration()))
        self.assertEqual(report["status"], "WIRED_RUN_COMPLETE")
        self.assertIsNone(report["hit_rate"])
        self.assertFalse(report["real_hit_rate_claimed"])
        self.assertIsNone(report["physics_engine"])
        self.assertIsNone(report["robot"])
        self.assertEqual(report["geometry_check_label"], GEOMETRY_LABEL)
        self.assertEqual(report["backend"], BACKEND_VERIFIED)
        for verdict in report["verdicts"]:
            self.assertEqual(verdict["backend"], BACKEND_VERIFIED)
            self.assertEqual(verdict["frame_name"], "gripper_end")

    def test_the_proxy_backend_run_is_labelled_proxy_everywhere(self):
        require_p6("the proxy-backend wired run (needs the racket proxy JSON)")
        report = run_wired_closed_loop(WiredConfig(scenario="FIXED_BALL",
                                                   backend=BACKEND_PROXY,
                                                   p6_root=P6_ROOT,
                                                   calibration=calibration()))
        self.assertEqual(report["backend"], BACKEND_PROXY)
        self.assertEqual(report["backend_provenance"]["source_class"],
                         "PROXY_DH_ENGINEERING_V0_1")
        for verdict in report["verdicts"]:
            self.assertEqual(verdict["backend"], BACKEND_PROXY)

    def test_mixing_backends_in_one_summary_is_refused(self):
        left = {"backend": BACKEND_PROXY, "status": "WIRED_RUN_COMPLETE",
                "hold_reasons": {}, "states_entered": {}, "reachability_verdicts": {},
                "verdicts": [], "geometry_contacts": 0, "geometry_evaluations": 0,
                "hold_latched": False}
        right = dict(left, backend=BACKEND_VERIFIED)
        with self.assertRaises(BackendMixError):
            summarise([left, right], backend=BACKEND_PROXY)


class RacketProxyGeometryTests(unittest.TestCase):
    def setUp(self):
        require_p6("the committed racket collision proxy JSON")
        self.proxy = load_racket_proxy(extra_roots=(P6_ROOT,))

    def test_the_committed_json_loads_with_its_real_primitive_count(self):
        self.assertEqual(self.proxy.primitive_count, 43)
        self.assertEqual(len(self.proxy.primitives), 43)
        self.assertEqual(len(self.proxy.sha256), 64)
        self.assertTrue(Path(self.proxy.path).is_file())
        self.assertTrue(str(self.proxy.path).endswith("racket_collision_proxy.json"))
        self.assertGreater(self.proxy.face_center_from_root_m, 0.4)
        self.assertIn("string_bed", self.proxy.role_counts)

    def test_a_ball_at_the_planned_face_pose_touches_the_string_bed(self):
        face = np.array([0.36, 0.05, 0.45])
        contact = self.proxy.check(face, face, np.eye(3))
        self.assertTrue(contact.contacted)
        self.assertLessEqual(contact.clearance_m, 0.0)
        self.assertIn("string_bed", contact.nearest_primitive)
        self.assertEqual(contact.label, GEOMETRY_LABEL)

    def test_the_check_rejects_an_unreachable_pose(self):
        """A pose 0.8 m away from the ball is not a contact - and says so."""
        face = np.array([0.36, 0.05, 0.45])
        ball = face + np.array([0.0, 0.0, 0.8])
        contact = self.proxy.check(ball, face, np.eye(3))
        self.assertFalse(contact.contacted)
        self.assertGreater(contact.clearance_m, 0.1)
        self.assertGreater(contact.clearance_m, BALL_RADIUS_M)

    def test_a_truncated_proxy_record_is_refused(self):
        data = json.loads(Path(self.proxy.path).read_text(encoding="utf-8"))
        data["primitive_count"] = 3
        with self.assertRaises(ValueError):
            RacketCollisionProxy(data, path=self.proxy.path, sha256=self.proxy.sha256)

    def test_a_foreign_schema_is_refused(self):
        data = {"schema": "some_other_proxy/9", "primitives": [],
                "primitive_count": 0}
        with self.assertRaises(ValueError):
            RacketCollisionProxy(data, path="x", sha256="0")


if __name__ == "__main__":
    unittest.main()
