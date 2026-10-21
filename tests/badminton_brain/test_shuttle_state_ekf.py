# -*- coding: utf-8 -*-
"""P7-05 acceptance: image-plane-first EKF with a gated 3D upgrade.

All data here is SYNTHETIC.  The consistency numbers below say "the filter's own
covariance is trustworthy under the model it declares", not "the tracker is
accurate on real shuttlecock images" - that would need real data this task does
not have and must not claim.

Checks:
  1. NEES / NIS consistency when the truth matches the declared model
  2. the filter is conservative (never over-confident) on deterministic CV truth
  3. outlier rejection through the chi-square gate, and the documented re-init
  4. missing measurements -> prediction only, covariance grows, COASTING -> LOST
  5. 3D upgrade refused without calibration / trustworthy depth, and correct
     position+velocity when both exist
  6. contract violations (NaN, stale timestamp, mode mismatch) raise

Run:
    python -m pytest tests/badminton_brain/test_shuttle_state_ekf.py -q
"""
from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from badminton_brain.estimation.shuttle_state_ekf import (  # noqa: E402
    COASTING,
    LOST,
    MAX_COAST_STEPS,
    MEASUREMENT_STD_M,
    MEASUREMENT_STD_M_MEASURED,
    MEASUREMENT_STD_PX,
    MODE_COURT_3D,
    MODE_IMAGE_2D,
    OUTLIER_STREAK_REINIT,
    PROCESS_ACCEL_STD_MP_S2,
    PROCESS_ACCEL_STD_PX_S2,
    TRACKING,
    CourtCameraModel,
    DepthNotTrustworthyError,
    ShuttleEKFError,
    ShuttleStateEKF,
    StateModeError,
    UncalibratedCameraError,
)
from badminton_brain.status import UNRESOLVED_STATUSES, AssetStatus, Param  # noqa: E402

DT = 0.02                 # 50 Hz, the rate the existing perception tests use
SIGMA_PX = 2.0            # declared bbox-centre noise proxy
ACC_PX = 2000.0           # declared image-plane acceleration proxy [px/s^2]
X_TRUE_0 = np.array([640.0, 360.0, 300.0, -120.0])   # u, v, u_dot, v_dot


def _q_sqrt(Q):
    evals, evecs = np.linalg.eigh(Q)
    return evecs @ np.diag(np.sqrt(np.maximum(evals, 0.0)))


def _cv_truth(steps, *, noisy_process, seed):
    """Simulate CV truth; noisy_process=True matches the filter's Q exactly."""
    rng = np.random.default_rng(seed)
    F = np.array([[1.0, 0.0, DT, 0.0],
                  [0.0, 1.0, 0.0, DT],
                  [0.0, 0.0, 1.0, 0.0],
                  [0.0, 0.0, 0.0, 1.0]])
    filt = ShuttleStateEKF(process_accel_std=ACC_PX, measurement_std=SIGMA_PX)
    Q = filt.process_noise(DT)
    Lq = _q_sqrt(Q)
    x = X_TRUE_0.copy()
    out = [x.copy()]
    for _ in range(steps):
        x = F @ x
        if noisy_process:
            x = x + Lq @ rng.standard_normal(4)
        out.append(x.copy())
    return np.array(out), rng


class TestConsistency(unittest.TestCase):
    """NEES/NIS: is the declared covariance the real error covariance?"""

    def test_nees_and_nis_consistent_under_matching_model(self):
        n_runs, n_steps = 80, 40
        nees_all, nis_all = [], []
        rejected = 0
        for run in range(n_runs):
            truth, rng = _cv_truth(n_steps, noisy_process=True, seed=1000 + run)
            P0 = np.diag([SIGMA_PX ** 2, SIGMA_PX ** 2, 100.0 ** 2, 100.0 ** 2])
            x0 = truth[0] + np.sqrt(np.diag(P0)) * rng.standard_normal(4)
            filt = ShuttleStateEKF(process_accel_std=ACC_PX, measurement_std=SIGMA_PX)
            filt.initialize(x0[:2], 0.0, velocity=x0[2:], covariance=P0)
            for k in range(1, n_steps + 1):
                filt.predict(DT)
                z = truth[k][:2] + SIGMA_PX * rng.standard_normal(2)
                result = filt.update(z, k * DT)
                if not result.accepted:
                    # The gate is a 99% chi-square gate: ~1% of true inliers are dropped
                    # BY DESIGN.  Count them instead of forbidding them.
                    rejected += 1
                    continue
                nis_all.append(result.nis)
                err = truth[k] - filt.state
                P = filt.covariance
                nees_all.append(float(err @ np.linalg.solve(P, err)))
        nees = np.array(nees_all)
        nis = np.array(nis_all)
        # Pre-registered: the 99% gate may drop up to 3% of inliers (1% nominal + slack for
        # 3200 correlated samples); the NEES/NIS windows are the chi-square means.
        self.assertLessEqual(rejected / float(n_runs * n_steps), 0.03,
                             msg=f"gate dropped {rejected} inliers, too many for a 99% gate")
        self.assertGreater(nees.size, 0.97 * n_runs * n_steps)
        self.assertTrue(3.4 <= float(nees.mean()) <= 4.6,
                        msg=f"mean NEES {nees.mean():.4f} outside [3.4, 4.6]")
        self.assertTrue(1.5 <= float(nis.mean()) <= 2.3,
                        msg=f"mean NIS {nis.mean():.4f} outside [1.5, 2.3]")
        print(f"[P7-05] mean NEES = {nees.mean():.4f} (dof 4), "
              f"mean NIS = {nis.mean():.4f} (dof 2), N = {nees.size}, "
              f"gated inliers = {rejected}/{n_runs * n_steps}")

    def test_no_over_confidence_on_deterministic_cv_truth(self):
        n_runs, n_steps = 40, 40
        nees_all = []
        for run in range(n_runs):
            truth, rng = _cv_truth(n_steps, noisy_process=False, seed=7000 + run)
            filt = ShuttleStateEKF(process_accel_std=ACC_PX, measurement_std=SIGMA_PX)
            filt.initialize(truth[0][:2], 0.0, velocity=truth[0][2:].copy(),
                            covariance=np.diag([SIGMA_PX ** 2, SIGMA_PX ** 2,
                                                 50.0 ** 2, 50.0 ** 2]))
            for k in range(1, n_steps + 1):
                filt.predict(DT)
                z = truth[k][:2] + SIGMA_PX * rng.standard_normal(2)
                filt.update(z, k * DT)
                err = truth[k] - filt.state
                nees_all.append(float(err @ np.linalg.solve(filt.covariance, err)))
        mean_nees = float(np.mean(nees_all))
        # With a noiseless truth the model Q only adds covariance, so the filter must be
        # conservative: NEES below the chi-square(4) mean.  It must never exceed it much.
        self.assertLess(mean_nees, 4.6, msg=f"over-confident: mean NEES {mean_nees:.4f}")
        self.assertGreater(mean_nees, 0.0)
        print(f"[P7-05] deterministic-CV mean NEES = {mean_nees:.4f} (conservative)")


class TestOutlierRejection(unittest.TestCase):
    def _converged_filter(self):
        truth, rng = _cv_truth(25, noisy_process=False, seed=42)
        filt = ShuttleStateEKF(process_accel_std=ACC_PX, measurement_std=SIGMA_PX)
        filt.initialize(truth[0][:2], 0.0, velocity=truth[0][2:].copy(),
                        covariance=np.diag([SIGMA_PX ** 2, SIGMA_PX ** 2,
                                             20.0 ** 2, 20.0 ** 2]))
        for k in range(1, 26):
            filt.step(truth[k][:2] + SIGMA_PX * rng.standard_normal(2), k * DT)
        return filt, truth, rng

    def test_single_outlier_is_gated_and_state_untouched(self):
        filt, truth, rng = self._converged_filter()
        before_x, before_P = filt.state, filt.covariance
        accepted_before = filt.accepted_updates
        filt.predict(DT)
        predicted_x, predicted_P = filt.state, filt.covariance
        outlier = predicted_x[:2] + np.array([40.0, -35.0])   # ~20 sigma
        result = filt.update(outlier, 26 * DT)
        self.assertFalse(result.accepted)
        self.assertEqual(result.reason, "GATE")
        self.assertGreater(result.mahalanobis_d2, result.gate_chi2)
        self.assertEqual(filt.accepted_updates, accepted_before)
        np.testing.assert_allclose(filt.state, predicted_x)   # measurement ignored
        np.testing.assert_allclose(filt.covariance, predicted_P)
        self.assertEqual(filt.steps_without_measurement, 1)
        self.assertTrue(np.all(np.isfinite(filt.state)))
        # the pre-gate state is still the honest 2D estimate
        self.assertLess(float(np.max(np.abs(before_x[:2] - truth[25][:2]))), 10.0)
        self.assertEqual(filt.status, COASTING)

    def test_rejection_streak_triggers_documented_reinit(self):
        filt, _truth, _rng = self._converged_filter()
        streak = int(OUTLIER_STREAK_REINIT.value)
        last = None
        for i in range(streak):
            filt.predict(DT)
            last = filt.state[:2] + np.array([45.0, -40.0])
            result = filt.update(last, (26 + i) * DT)
            self.assertFalse(result.accepted)
        self.assertEqual(result.reason, "REINIT")
        np.testing.assert_allclose(filt.state[:2], last)
        self.assertEqual(filt.steps_without_measurement, 0)


class TestMissingMeasurements(unittest.TestCase):
    def test_prediction_only_grows_covariance_and_reports_coasting(self):
        truth, rng = _cv_truth(20, noisy_process=False, seed=11)
        filt = ShuttleStateEKF(process_accel_std=ACC_PX, measurement_std=SIGMA_PX)
        filt.initialize(truth[0][:2], 0.0, velocity=truth[0][2:].copy(),
                        covariance=np.diag([SIGMA_PX ** 2, SIGMA_PX ** 2,
                                             10.0 ** 2, 10.0 ** 2]))
        for k in range(1, 21):
            filt.step(truth[k][:2] + SIGMA_PX * rng.standard_normal(2), k * DT)
        self.assertEqual(filt.status, TRACKING)
        traces = [float(np.trace(filt.covariance))]
        states = [filt.state.copy()]
        n_missing = int(MAX_COAST_STEPS.value) + 1
        for i in range(1, n_missing + 1):
            filt.predict(DT)
            traces.append(float(np.trace(filt.covariance)))
            states.append(filt.state.copy())
            expected = TRACKING if i == 0 else (COASTING if i <= MAX_COAST_STEPS.value
                                                else LOST)
            self.assertEqual(filt.status, expected, msg=f"step {i}")
        self.assertTrue(np.all(np.diff(traces) > 0.0),
                        msg=f"covariance must grow while coasting: {traces}")
        # Constant-velocity propagation is exact: no measurement was invented.
        for i in range(1, n_missing + 1):
            np.testing.assert_allclose(states[i][:2],
                                       states[i - 1][:2] + DT * states[i - 1][2:])
        self.assertTrue(np.all(np.isfinite(filt.covariance)))
        self.assertTrue(filt.is_lost)
        print(f"[P7-05] coasting trace growth: {traces[0]:.3f} -> {traces[-1]:.3f}")

    def test_reset_clears_everything(self):
        truth, _rng = _cv_truth(3, noisy_process=False, seed=5)
        filt = ShuttleStateEKF()
        filt.initialize(truth[0][:2], 0.0)
        filt.reset()
        self.assertEqual(filt.mode, MODE_IMAGE_2D)
        self.assertEqual(filt.status, LOST)
        with self.assertRaises(ShuttleEKFError):
            filt.predict(DT)


def _camera(calibration="CALIBRATED"):
    K = np.array([[700.0, 0.0, 639.5], [0.0, 700.0, 359.5], [0.0, 0.0, 1.0]])
    return CourtCameraModel(K=K, T_court_camera=np.eye(4),
                            image_size=(1280, 720), calibration=calibration)


def _project(X, Y, Z, camera):
    K = camera.K
    return np.array([K[0, 0] * X / Z + K[0, 2], K[1, 1] * Y / Z + K[1, 2]])


def _truth_3d(t):
    return np.array([1.0 + 2.0 * t, 0.5 - 1.0 * t, 3.0 + 0.5 * t])


class TestThreeDUpgrade(unittest.TestCase):
    """A 2D state may only become 3D with depth AND a calibration."""

    def _tracked_2d_filter(self, steps=40, dt=0.01, sigma_px=0.05):
        cam = _camera("CALIBRATED")
        filt = ShuttleStateEKF(measurement_std=sigma_px)
        rng = np.random.default_rng(3)
        z0 = _project(*_truth_3d(0.0), cam)
        filt.initialize(z0, 0.0, velocity=np.zeros(2))
        for k in range(1, steps + 1):
            t = k * dt
            z = _project(*_truth_3d(t), cam) + sigma_px * rng.standard_normal(2)
            filt.step(z, t)
        return filt, cam

    def test_upgrade_recovers_position_and_velocity(self):
        filt, cam = self._tracked_2d_filter()
        self.assertEqual(filt.mode, MODE_IMAGE_2D)
        depth = float(_truth_3d(0.40)[2])
        result = filt.upgrade_to_3d(depth_m=depth, depth_sigma_m=0.02, camera=cam,
                                   depth_source="STEREO_TRIANGULATED",
                                   depth_rate_mps=0.5, depth_rate_sigma_mps=0.05)
        self.assertEqual(result.mode, MODE_COURT_3D)
        self.assertTrue(result.velocity_from_depth_rate)
        truth = _truth_3d(0.40)
        np.testing.assert_allclose(filt.position, truth, atol=0.02)
        np.testing.assert_allclose(filt.velocity, np.array([2.0, -1.0, 0.5]), atol=0.10)
        eigen = np.linalg.eigvalsh(filt.covariance)
        self.assertGreaterEqual(float(np.min(eigen)), -1e-12)
        self.assertTrue(np.all(np.isfinite(filt.covariance)))
        print(f"[P7-05] 3D upgrade pos err {np.linalg.norm(filt.position - truth):.4f} m, "
              f"vel err {np.linalg.norm(filt.velocity - np.array([2.0, -1.0, 0.5])):.4f} m/s")

    def test_upgrade_without_depth_rate_declares_the_proxy(self):
        filt, cam = self._tracked_2d_filter()
        result = filt.upgrade_to_3d(depth_m=3.2, depth_sigma_m=0.02, camera=cam,
                                   depth_source="STEREO_TRIANGULATED")
        self.assertFalse(result.velocity_from_depth_rate)
        without = result.velocity_std_mps.copy()
        filt2, cam2 = self._tracked_2d_filter()
        with_rate = filt2.upgrade_to_3d(depth_m=3.2, depth_sigma_m=0.02, camera=cam2,
                                       depth_source="STEREO_TRIANGULATED",
                                       depth_rate_mps=0.5,
                                       depth_rate_sigma_mps=0.05)
        self.assertTrue(np.all(without > with_rate.velocity_std_mps))

    def test_refuses_uncalibrated_camera(self):
        filt, _cam = self._tracked_2d_filter()
        with self.assertRaises(UncalibratedCameraError):
            filt.upgrade_to_3d(depth_m=3.2, depth_sigma_m=0.02,
                               camera=_camera("TEMP_PROXY"),
                               depth_source="STEREO_TRIANGULATED")
        self.assertEqual(filt.mode, MODE_IMAGE_2D)
        # explicit, auditable opt-in is allowed for synthetic work and is reported
        result = filt.upgrade_to_3d(depth_m=3.2, depth_sigma_m=0.02,
                                    camera=_camera("UNCALIBRATED"),
                                    depth_source="STEREO_TRIANGULATED",
                                    depth_rate_mps=0.5,
                                    depth_rate_sigma_mps=0.05,
                                    allow_uncalibrated=True)
        self.assertEqual(result.camera_calibration, "UNCALIBRATED")

    def test_refuses_untrustworthy_depth(self):
        filt, cam = self._tracked_2d_filter()
        with self.assertRaises(DepthNotTrustworthyError):
            filt.upgrade_to_3d(depth_m=3.2, depth_sigma_m=0.02, camera=cam,
                               depth_source="MONOCULAR_GUESS",
                               depth_rate_mps=0.5, depth_rate_sigma_mps=0.05)
        with self.assertRaises(DepthNotTrustworthyError):
            filt.upgrade_to_3d(depth_m=3.2, depth_sigma_m=0.0, camera=cam,
                               depth_source="STEREO_TRIANGULATED")
        with self.assertRaises(DepthNotTrustworthyError):
            filt.upgrade_to_3d(depth_m=-1.0, depth_sigma_m=0.02, camera=cam,
                               depth_source="STEREO_TRIANGULATED")
        with self.assertRaises(DepthNotTrustworthyError):
            filt.upgrade_to_3d(depth_m=3.2, depth_sigma_m=0.02, camera=cam,
                               depth_source="STEREO_TRIANGULATED",
                               depth_rate_mps=0.5)   # sigma missing
        self.assertEqual(filt.mode, MODE_IMAGE_2D)

    def test_cannot_construct_directly_in_3d_and_modes_are_separate(self):
        with self.assertRaises(ShuttleEKFError):
            ShuttleStateEKF(mode=MODE_COURT_3D)

    def test_2d_measurement_rejected_after_upgrade(self):
        filt, cam = self._tracked_2d_filter()
        filt.upgrade_to_3d(depth_m=3.2, depth_sigma_m=0.02, camera=cam,
                           depth_source="STEREO_TRIANGULATED",
                           depth_rate_mps=0.5, depth_rate_sigma_mps=0.05)
        with self.assertRaises(StateModeError):
            filt.update(np.array([700.0, 400.0]))

    def test_3d_update_requires_measured_covariance(self):
        filt, cam = self._tracked_2d_filter()
        filt.upgrade_to_3d(depth_m=3.2, depth_sigma_m=0.02, camera=cam,
                           depth_source="STEREO_TRIANGULATED",
                           depth_rate_mps=0.5, depth_rate_sigma_mps=0.05)
        self.assertIsNone(MEASUREMENT_STD_M.value)
        with self.assertRaises(ShuttleEKFError):
            filt.update(filt.position + 0.001)
        result = filt.update(filt.position + np.array([0.005, -0.004, 0.003]),
                             measurement_covariance=np.diag([0.02 ** 2] * 3))
        self.assertTrue(result.accepted)


class TestContractViolations(unittest.TestCase):
    def test_nan_measurement_rejected(self):
        filt = ShuttleStateEKF()
        filt.initialize(np.array([640.0, 360.0]), 0.0)
        with self.assertRaises(ShuttleEKFError):
            filt.update(np.array([float("nan"), 360.0]))

    def test_stale_timestamp_rejected(self):
        filt = ShuttleStateEKF()
        filt.initialize(np.array([640.0, 360.0]), 1.0)
        with self.assertRaises(ShuttleEKFError):
            filt.update(np.array([641.0, 360.0]), 0.5)

    def test_bad_constructor_arguments(self):
        for kwargs in ({"dt_s": 0.0}, {"dt_s": float("nan")},
                       {"process_accel_std": 0.0}, {"measurement_std": -1.0},
                       {"max_coast_steps": 0}):
            with self.assertRaises(ShuttleEKFError, msg=str(kwargs)):
                ShuttleStateEKF(**kwargs)

    def test_noise_parameters_are_declared_and_statuses_are_honest(self):
        for param in (PROCESS_ACCEL_STD_PX_S2, PROCESS_ACCEL_STD_MP_S2,
                      MEASUREMENT_STD_PX):
            self.assertIsInstance(param, Param)
            self.assertIsNotNone(param.value)
            self.assertIsInstance(param.source, str)
            self.assertTrue(len(param.source) > 40, msg="justification must be documented")
            # A proxy must never masquerade as a measured/derived value.
            self.assertIn(param.status, UNRESOLVED_STATUSES)
            self.assertEqual(param.status, AssetStatus.TEMP_PARAMETERIZED_PROXY)
        self.assertIsNone(MEASUREMENT_STD_M.value)
        self.assertIn(MEASUREMENT_STD_M.status, UNRESOLVED_STATUSES)
        self.assertIn(MEASUREMENT_STD_M_MEASURED.status, UNRESOLVED_STATUSES)
        declared = ShuttleStateEKF().parameters()
        for name in ("process_accel_std", "measurement_std", "gate_chi2",
                     "max_coast_steps", "outlier_streak_reinit"):
            self.assertIsInstance(declared[name], Param, msg=name)


if __name__ == "__main__":
    unittest.main(verbosity=2)
