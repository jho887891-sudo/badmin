#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""P7-09: reachability verdicts and the IK interface contract.

Every fixture is synthetic and deterministic: joint vectors are fixed constants,
the IK seed is fixed, and the targets are produced by the SAME proxy kinematics
being tested.  That makes the numbers reproducible, and it is also the limit of
what these tests prove: they validate the SOLVER and the DECISION LOGIC, never
the real arm (the DH table is an ENGINEERING_V0_1 proxy).

Run:
    python -m pytest tests/badminton_brain/test_reachability_ik.py -q
"""
from __future__ import annotations

import math
import sys
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
for entry in (ROOT / "src", ROOT):
    if str(entry) not in sys.path:
        sys.path.insert(0, str(entry))

from src.badminton_brain.reachability import (  # noqa: E402
    DOF, DampedLeastSquaresIk, DetectionDimensionalityError, IkNotConverged,
    IkNotImplemented, IkResult, IkTarget, InvalidIkTarget, KinematicLimits,
    ReachabilityDecision, ReachabilityReason, UnavailableIk, assess_reachability,
    forward_kinematics_face, require_world_3d, target_from_intercept,
)
from src.badminton_brain.reachability.ik import rotation_error_vector  # noqa: E402

# A joint vector whose face centre sits inside the workspace box, used everywhere.
Q_REACHABLE = np.array([0.3, 0.8, -1.2, 0.1, 0.4, 0.2])
SEED = np.array([0.0, 1.5, -1.5, 0.0, 0.0, 0.0])
VERIFIED = True


def calibration():
    from src.badminton_brain.reachability import Calibration
    return Calibration(court_from_base=np.eye(4), verified=VERIFIED)


def pose_target(q=Q_REACHABLE):
    T = forward_kinematics_face(np.asarray(q, dtype=float))
    return IkTarget(position_m=T[:3, 3], rotation=T[:3, :3])


class KinematicsTests(unittest.TestCase):
    def test_forward_kinematics_is_deterministic(self):
        a = forward_kinematics_face(Q_REACHABLE)
        b = forward_kinematics_face(Q_REACHABLE)
        self.assertTrue(np.array_equal(a, b))
        self.assertEqual(a.shape, (4, 4))
        self.assertTrue(np.allclose(a[:3, :3] @ a[:3, :3].T, np.eye(3), atol=1e-12))
        np.testing.assert_allclose(a[3], [0.0, 0.0, 0.0, 1.0])

    def test_zero_pose_matches_the_documented_proxy_geometry(self):
        T = forward_kinematics_face(np.zeros(DOF))
        # the chain is a proxy, but it must at least be a 1.27 m class arm
        self.assertGreater(np.linalg.norm(T[:3, 3]), 0.5)
        self.assertLess(np.linalg.norm(T[:3, 3]), 1.30)

    def test_bad_joint_vectors_are_refused(self):
        with self.assertRaises(ValueError):
            forward_kinematics_face(np.zeros(DOF - 1))
        with self.assertRaises(ValueError):
            forward_kinematics_face(np.array([0.0] * (DOF - 1) + [np.nan]))


class IkContractTests(unittest.TestCase):
    def test_converges_with_documented_tolerance_and_reports_residual(self):
        target = pose_target()
        solver = DampedLeastSquaresIk()
        result = solver.solve(target, SEED)
        self.assertIsInstance(result, IkResult)
        self.assertTrue(result.converged)
        self.assertLessEqual(result.residual_position_m, solver.tolerance_position_m)
        self.assertLessEqual(result.residual_rotation_rad, solver.tolerance_rotation_rad)
        self.assertGreaterEqual(result.iterations, 1)
        self.assertEqual(result.solver, DampedLeastSquaresIk.solver_id)
        # the reported joint vector really is at the target
        T = forward_kinematics_face(np.asarray(result.joints_rad))
        np.testing.assert_allclose(T[:3, 3], np.asarray(target.position_m), atol=1e-4)

    def test_solution_is_deterministic(self):
        target = pose_target()
        solver = DampedLeastSquaresIk()
        first = solver.solve(target, SEED)
        second = solver.solve(target, SEED)
        self.assertEqual(first.joints_rad, second.joints_rad)
        self.assertEqual(first.residual_position_m, second.residual_position_m)

    def test_unreachable_target_raises_instead_of_returning_an_approximation(self):
        solver = DampedLeastSquaresIk()
        far = IkTarget(position_m=[2.2, 0.4, 1.6], rotation=np.eye(3))
        with self.assertRaises(IkNotConverged) as caught:
            solver.solve(far, SEED)
        error = caught.exception
        self.assertGreater(error.residual_position_m, solver.tolerance_position_m)
        self.assertGreater(error.iterations, 0)
        # and the refusal carries the numbers, so a caller can log them
        self.assertEqual(error.tol_position_m, solver.tolerance_position_m)

    def test_placeholder_solver_raises_not_implemented(self):
        with self.assertRaises(IkNotImplemented):
            UnavailableIk().solve(pose_target(), SEED)

    def test_a_non_converged_result_cannot_be_constructed(self):
        with self.assertRaises(ValueError):
            IkResult(joints_rad=(0.0,) * DOF, residual_position_m=0.5,
                     residual_rotation_rad=0.5, iterations=3, tolerance_position_m=1e-4,
                     tolerance_rotation_rad=1e-3, converged=False)

    def test_rotation_error_is_small_angle_equivalent(self):
        angle = 0.05
        c, s = math.cos(angle), math.sin(angle)
        R = np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])
        err = rotation_error_vector(np.eye(3), R)
        self.assertAlmostEqual(float(np.linalg.norm(err)), math.sin(angle), places=12)

    def test_invalid_target_is_refused(self):
        with self.assertRaises(InvalidIkTarget):
            IkTarget(position_m=[0.0, 0.0], rotation=np.eye(3))
        with self.assertRaises(InvalidIkTarget):
            IkTarget(position_m=[0.0, 0.0, 1.0], rotation=np.eye(3) * 2.0)


class ReachabilityDecisionTests(unittest.TestCase):
    def test_reachable_target(self):
        verdict = assess_reachability(target=pose_target(), time_to_target_s=0.5,
                                      seed_rad=SEED, calibration=calibration())
        self.assertIs(verdict.decision, ReachabilityDecision.REACHABLE)
        self.assertIs(verdict.reason, ReachabilityReason.IK_CONVERGED)
        self.assertTrue(verdict.reachable)
        self.assertIsNotNone(verdict.ik)
        self.assertGreater(verdict.joint_margin_rad, 0.05)
        self.assertEqual(verdict.evidence_class, "OFFLINE-SYNTHETIC")

    def test_workspace_bound(self):
        target = IkTarget(position_m=[2.0, 0.0, 1.0], rotation=np.eye(3))
        verdict = assess_reachability(target=target, time_to_target_s=0.5, seed_rad=SEED,
                                      calibration=calibration())
        self.assertIs(verdict.decision, ReachabilityDecision.NOT_REACHABLE)
        self.assertIs(verdict.reason, ReachabilityReason.WORKSPACE_BOUND)

    def test_self_collision_proxy(self):
        target = IkTarget(position_m=[0.12, 0.05, 0.60], rotation=np.eye(3))
        verdict = assess_reachability(target=target, time_to_target_s=0.5, seed_rad=SEED,
                                      calibration=calibration())
        self.assertIs(verdict.decision, ReachabilityDecision.NOT_REACHABLE)
        self.assertIs(verdict.reason, ReachabilityReason.SELF_COLLISION_PROXY)
        self.assertIn("PROXY", verdict.detail)

    def test_joint_limit_margin(self):
        tight = KinematicLimits(joint_margin_rad=3.0)
        verdict = assess_reachability(target=pose_target(), time_to_target_s=0.5,
                                      seed_rad=SEED, calibration=calibration(),
                                      limits=tight)
        self.assertIs(verdict.decision, ReachabilityDecision.NOT_REACHABLE)
        self.assertIs(verdict.reason, ReachabilityReason.JOINT_LIMIT_MARGIN)

    def test_ik_not_converged_is_not_reachable(self):
        # inside the box, outside the arm's real reach: the box alone is not enough
        target = IkTarget(position_m=[1.0, 0.6, 1.8], rotation=np.eye(3))
        verdict = assess_reachability(target=target, time_to_target_s=0.5, seed_rad=SEED,
                                      calibration=calibration())
        self.assertIs(verdict.decision, ReachabilityDecision.NOT_REACHABLE)
        self.assertIs(verdict.reason, ReachabilityReason.IK_NOT_CONVERGED)
        self.assertIn("no approximate pose is emitted", verdict.detail)

    def test_zero_time_to_target(self):
        for value in (0.0, -0.2):
            verdict = assess_reachability(target=pose_target(), time_to_target_s=value,
                                          seed_rad=SEED, calibration=calibration())
            self.assertIs(verdict.decision, ReachabilityDecision.NOT_REACHABLE)
            self.assertIs(verdict.reason, ReachabilityReason.TIME_TO_TARGET)

    def test_time_insufficient_for_the_required_slew(self):
        # reachable pose, but only 30 ms of flight time: the slew cannot finish
        verdict = assess_reachability(target=pose_target(), time_to_target_s=0.03,
                                      seed_rad=SEED, calibration=calibration(),
                                      time_margin_s=0.02)
        self.assertIs(verdict.decision, ReachabilityDecision.NOT_REACHABLE)
        self.assertIs(verdict.reason, ReachabilityReason.TIME_TO_TARGET)
        self.assertIn("required slew", verdict.detail)

    def test_unknown_without_an_ik_solver(self):
        verdict = assess_reachability(target=pose_target(), time_to_target_s=0.5,
                                      seed_rad=SEED, calibration=calibration(),
                                      solver=UnavailableIk())
        self.assertIs(verdict.decision, ReachabilityDecision.UNKNOWN)
        self.assertIs(verdict.reason, ReachabilityReason.IK_NOT_IMPLEMENTED)

    def test_unknown_without_calibration(self):
        from src.badminton_brain.reachability import Calibration
        verdict = assess_reachability(target=pose_target(), time_to_target_s=0.5,
                                      seed_rad=SEED, calibration=Calibration())
        self.assertIs(verdict.decision, ReachabilityDecision.UNKNOWN)
        self.assertIs(verdict.reason, ReachabilityReason.CALIBRATION_MISSING)

    def test_covariance_threshold(self):
        verdict = assess_reachability(target=pose_target(), time_to_target_s=0.5,
                                      seed_rad=SEED, calibration=calibration(),
                                      position_sigma_m=0.20)
        self.assertIs(verdict.decision, ReachabilityDecision.NOT_REACHABLE)
        self.assertIs(verdict.reason, ReachabilityReason.COVARIANCE_ABOVE_THRESHOLD)

    def test_intercept_point_and_approach_direction(self):
        target = pose_target()
        verdict = assess_reachability(
            intercept_point_m=np.asarray(target.position_m),
            approach_direction=np.asarray(target.rotation)[:, 0],
            time_to_target_s=0.5, seed_rad=SEED, calibration=calibration())
        self.assertIn(verdict.decision,
                      {ReachabilityDecision.REACHABLE, ReachabilityDecision.NOT_REACHABLE})
        self.assertIsNotNone(verdict.target_position_m)

    def test_verdict_is_json_serialisable(self):
        verdict = assess_reachability(target=pose_target(), time_to_target_s=0.5,
                                      seed_rad=SEED, calibration=calibration())
        data = verdict.as_dict()
        self.assertEqual(data["decision"], "REACHABLE")
        self.assertEqual(data["ik"]["converged"], True)


class TwoDimensionalRefusalTests(unittest.TestCase):
    def test_require_world_3d_refuses_a_2d_point(self):
        with self.assertRaises(DetectionDimensionalityError):
            require_world_3d([120.0, 240.0], source="test")
        with self.assertRaises(DetectionDimensionalityError):
            require_world_3d(None, source="test")

    def test_reachability_refuses_a_2d_detection(self):
        with self.assertRaises(DetectionDimensionalityError):
            assess_reachability(target=pose_target(), time_to_target_s=0.5, seed_rad=SEED,
                                calibration=calibration(), detection_dim=2)

    def test_intercept_target_refuses_a_2d_point(self):
        with self.assertRaises(DetectionDimensionalityError):
            target_from_intercept([10.0, 20.0], [0.0, 0.0, -1.0])

    def test_target_from_intercept_builds_an_orthonormal_face_frame(self):
        target = target_from_intercept([0.5, 0.0, 1.0], [0.0, -1.0, 0.0])
        R = np.asarray(target.rotation)
        np.testing.assert_allclose(R @ R.T, np.eye(3), atol=1e-12)
        np.testing.assert_allclose(R[:, 0], [0.0, -1.0, 0.0], atol=1e-12)  # face normal
        np.testing.assert_allclose(R[:, 2], [0.0, 0.0, 1.0], atol=1e-12)   # handle->head

    def test_parallel_up_hint_falls_back_deterministically(self):
        first = target_from_intercept([0.5, 0.0, 1.0], [0.0, 0.0, -1.0])
        second = target_from_intercept([0.5, 0.0, 1.0], [0.0, 0.0, -1.0])
        np.testing.assert_array_equal(np.asarray(first.rotation),
                                      np.asarray(second.rotation))
        R = np.asarray(first.rotation)
        np.testing.assert_allclose(R @ R.T, np.eye(3), atol=1e-12)


if __name__ == "__main__":
    unittest.main()
