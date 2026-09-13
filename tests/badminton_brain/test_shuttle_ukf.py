# -*- coding: utf-8 -*-
"""T3: augmented-state Shuttle UKF (numpy only) - RED first, then GREEN.

Spec: docs/superpowers/plans/2026-09-13-brain-modules.md (T3 row),
docs/architecture/ROBOT_BRAIN.md S4.3 / S7 (augmented-state UKF: p, v, drag, wind),
docs/simulation/BADMINTON_ROBOT.md S4 (authenticity levels), S12 (TEMP policy).

Acceptance (plan T3):
  * synthetic quadratic-drag flight with noisy measurements: UKF position RMSE is
    clearly smaller than the raw-measurement RMSE (numbers printed),
  * the drag estimate converges even from a wrong initial prior (tolerance printed),
  * no NaN, covariance stays positive semi-definite,
  * reset(env_ids) isolates environments.
Ground truth physics: src/trajectory/shuttle_aerodynamics.py (the same module the
filter must reuse - no second physics implementation anywhere in this test either).
"""
from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'src'))

from badminton_brain.estimation.shuttle_ukf import (  # noqa: E402
    DRAG_K_MAX, DRAG_K_MIN, DRAG_K_PROCESS_STD, GRAVITY_PARAM, INITIAL_DRAG_K_PRIOR,
    INITIAL_DRAG_STD, INITIAL_POSITION_STD, INITIAL_VELOCITY_STD,
    POSITION_MEASUREMENT_STD, PROCESS_ACCEL_STD, SHUTTLE_DRAG_K_REFERENCE,
    VELOCITY_MEASUREMENT_STD, WIND_PROCESS_STD, ShuttleUKF,
)
from badminton_brain.status import UNRESOLVED_STATUSES, AssetStatus, Param  # noqa: E402
from badminton_brain.types import BrainBoundaryError, ShuttleMeasurement  # noqa: E402
from trajectory.shuttle_aerodynamics import rollout  # noqa: E402

K_TRUE = float(SHUTTLE_DRAG_K_REFERENCE.value)          # 1/6.5 from the aerodynamics module
K_PRIOR = float(INITIAL_DRAG_K_PRIOR.value)             # deliberately wrong (TEMP proxy prior)
DT_MEAS = 0.02                                          # 50 Hz stereo
DURATION = 1.0
SIGMA_POS = 0.01                                        # synthetic measurement noise [m]
P0 = np.array([1.0, 0.5, 1.5])
V0 = np.array([-8.0, -1.0, 4.0])
WIND_TRUE = np.array([1.0, 0.0, 0.0])


def synthetic_flight(*, wind=WIND_TRUE, sigma=SIGMA_POS, seed=20260913,
                     duration=DURATION, dt_meas=DT_MEAS):
    """Ground truth from shuttle_aerodynamics.rollout + noisy ShuttleMeasurement."""
    n_steps = int(round(duration / dt_meas))
    truth = rollout(P0, V0, duration_s=duration, dt_s=5e-4, k_per_m=K_TRUE,
                    gravity=GRAVITY_PARAM.value, wind=wind)
    times = dt_meas * np.arange(n_steps + 1)
    pos_true = np.stack([np.interp(times, truth['time'], truth['position'][:, i])
                         for i in range(3)], axis=1)
    vel_true = np.stack([np.interp(times, truth['time'], truth['velocity'][:, i])
                         for i in range(3)], axis=1)
    rng = np.random.default_rng(seed)
    noise = sigma * rng.standard_normal(pos_true.shape)
    vel_noise = sigma * rng.standard_normal(vel_true.shape)
    return {
        'times': times,
        'position': pos_true,
        'velocity': vel_true,
        'measurement_position': pos_true + noise,
        'measurement_velocity': vel_true + vel_noise,
        'noise': noise,
        'velocity_noise': vel_noise,
    }


def measurement_at(flight, k, n_envs=1):
    """(n_envs,) identical ShuttleMeasurement at sample k."""
    pos = np.tile(flight['measurement_position'][k], (n_envs, 1))
    vel = np.tile(flight['measurement_velocity'][k], (n_envs, 1))
    cov = np.tile(np.eye(3) * SIGMA_POS ** 2, (n_envs, 1, 1))
    return ShuttleMeasurement(position=pos, velocity=vel, covariance=cov,
                              timestamp=float(flight['times'][k]))


def rmse(a, b):
    d = np.asarray(a) - np.asarray(b)
    return float(np.sqrt(np.mean(np.sum(d * d, axis=-1))))


def run_filter(ukf, flight, *, n_envs=1, burn_in=5, fuse_velocity=False):
    """Drive the filter over the whole flight; return per-step filter/raw errors."""
    ukf.initialize(position=np.tile(flight['measurement_position'][0], (n_envs, 1)),
                   velocity=np.tile((flight['measurement_position'][1]
                                     - flight['measurement_position'][0]) / DT_MEAS, (n_envs, 1)),
                   timestamp=0.0)
    filt_err, raw_err, vel_err, k_hist = [], [], [], []
    for k in range(1, len(flight['times'])):
        ukf.predict(DT_MEAS)
        ukf.update(measurement_at(flight, k, n_envs))
        if k >= burn_in:
            filt_err.append(rmse(ukf.position, flight['position'][k]))
            raw_err.append(rmse(flight['measurement_position'][k], flight['position'][k]))
            vel_err.append(rmse(ukf.velocity, flight['velocity'][k]))
        k_hist.append(float(ukf.drag_k[0]))
    return {
        'filter_rmse': float(np.mean(filt_err)),
        'raw_rmse': float(np.mean(raw_err)),
        'velocity_rmse': float(np.mean(vel_err)),
        'k_final': float(ukf.drag_k[0]),
        'k_history': np.asarray(k_hist),
        'samples': len(filt_err),
    }


class ShuttleUkfTests(unittest.TestCase):
    # ---- 1. unscented transform primitives -----------------------------------
    def test_sigma_points_reproduce_mean_and_covariance_and_weights_sum_to_one(self) -> None:
        ukf = ShuttleUKF(num_envs=1)
        mean = np.array([1.5, -0.5, 2.0, 3.0, -1.0, 0.5, K_TRUE])
        cov = np.diag([0.04, 0.09, 0.01, 0.25, 0.16, 0.36, 0.0025])
        pts, wm, wc = ukf.sigma_points(mean, cov)

        self.assertEqual(pts.shape, (1, 2 * mean.size + 1, mean.size))
        self.assertAlmostEqual(float(np.sum(wm)), 1.0, places=12)
        # alpha = 1.0 in this filter, so sum(wc) = 1 + beta (Julier/Uhlmann scaled UT)
        self.assertAlmostEqual(float(np.sum(wc)), 1.0 + ukf.beta, places=12)
        self.assertTrue(np.all(wm > 0.0) and np.all(wc > 0.0))
        np.testing.assert_allclose(wm @ pts[0], mean, rtol=0.0, atol=1e-12)

        d = pts[0] - mean
        cov_rec = (wc[:, None, None] * (d[:, :, None] * d[:, None, :])).sum(axis=0)
        np.testing.assert_allclose(cov_rec, cov, rtol=0.0, atol=1e-12)

    # ---- 2. the process model IS the frozen aerodynamics ----------------------
    def test_predict_reuses_shuttle_aerodynamics_rk4_step_exactly(self) -> None:
        from trajectory.shuttle_aerodynamics import rk4_step
        ukf = ShuttleUKF(num_envs=1, process_accel_std=0.0, drag_process_std=0.0)
        # zero covariance -> all sigma points collapse on the mean -> pure RK4 step
        ukf.initialize(position=P0, velocity=V0, drag_k=K_TRUE,
                       covariance=np.zeros((7, 7)))
        dt = 0.031
        ukf.predict(dt)
        p_expected, v_expected = rk4_step(P0, V0, dt, k_per_m=K_TRUE,
                                          gravity=GRAVITY_PARAM.value,
                                          wind=np.zeros(3))
        np.testing.assert_allclose(ukf.position[0], p_expected, rtol=0.0, atol=1e-12)
        np.testing.assert_allclose(ukf.velocity[0], v_expected, rtol=0.0, atol=1e-12)

    # ---- 3. acceptance: position RMSE clearly below the raw measurement -------
    def test_position_rmse_is_clearly_below_raw_measurement_rmse(self) -> None:
        flight = synthetic_flight()
        ukf = ShuttleUKF(num_envs=1)
        res = run_filter(ukf, flight)
        print('\n[T3] position RMSE  filter = %.5f m   raw measurement = %.5f m   '
              'ratio = %.3f   (%d samples)' % (res['filter_rmse'], res['raw_rmse'],
                                               res['filter_rmse'] / res['raw_rmse'],
                                               res['samples']))
        self.assertGreater(res['samples'], 30)
        self.assertLess(res['filter_rmse'], 0.5 * res['raw_rmse'])

    # ---- 4. acceptance: drag estimate converges from a wrong prior ------------
    def test_drag_estimate_converges_from_a_wrong_initial_prior(self) -> None:
        flight = synthetic_flight()
        ukf = ShuttleUKF(num_envs=1)
        res = run_filter(ukf, flight)
        rel_err = abs(res['k_final'] - K_TRUE) / K_TRUE
        print('\n[T3] drag k  prior = %.5f (%.0f%% off)   estimate = %.5f   true = %.5f   '
              'rel.err = %.2f%%' % (K_PRIOR, 100.0 * abs(K_PRIOR - K_TRUE) / K_TRUE,
                                    res['k_final'], K_TRUE, 100.0 * rel_err))
        print('[T3] k trajectory (every 10th step): %s'
              % np.array2string(res['k_history'][::10], precision=4))
        self.assertGreater(abs(K_PRIOR - K_TRUE) / K_TRUE, 0.4)   # the prior really is wrong
        self.assertLess(rel_err, 0.15)

    # ---- 5. acceptance: no NaN, covariance stays PSD, gaps survive ------------
    def test_no_nan_and_psd_covariance_with_dropouts_and_wide_drag_prior(self) -> None:
        flight = synthetic_flight(seed=7)
        ukf = ShuttleUKF(num_envs=1, initial_drag_std=0.5)   # sigma points reach k < 0
        ukf.initialize(position=flight['measurement_position'][0][None, :],
                       velocity=np.array([[0.0, 0.0, 0.0]]))
        for k in range(1, len(flight['times'])):
            ukf.predict(5 * DT_MEAS if k % 5 == 0 else DT_MEAS)   # 0.1 s measurement gap
            if k % 5 == 0:
                ukf.update(measurement_at(flight, k))
            state = ukf.state
            cov = ukf.covariance
            self.assertTrue(np.all(np.isfinite(state)))
            self.assertTrue(np.all(np.isfinite(cov)))
            np.testing.assert_allclose(cov, cov.T, rtol=0.0, atol=1e-12)
            eig = np.linalg.eigvalsh(0.5 * (cov + cov.T))
            self.assertGreater(float(eig.min()), -1e-9, 'covariance left the PSD cone')
            self.assertGreaterEqual(float(ukf.drag_k[0]), 0.0)
        self.assertTrue(np.all(np.isfinite(ukf.position)))

    # ---- 6. acceptance: reset(env_ids) isolates environments ------------------
    def test_reset_isolates_environments(self) -> None:
        ukf = ShuttleUKF(num_envs=3)
        ukf.initialize(position=np.array([[1.0, 0.0, 1.0], [2.0, 0.0, 1.0], [3.0, 0.0, 1.0]]),
                       velocity=np.array([[1.0, 0.0, 0.0]] * 3), drag_k=K_TRUE)
        ukf.predict(0.05)
        before = ukf.state.copy()
        ukf.reset([0, 2])
        after = ukf.state
        # reset envs go back to the declared default prior
        np.testing.assert_allclose(after[0, :6], 0.0, atol=1e-15)
        np.testing.assert_allclose(after[2, :6], 0.0, atol=1e-15)
        np.testing.assert_allclose(after[0, 6], K_PRIOR, rtol=0.0, atol=1e-15)
        # untouched env keeps its value
        np.testing.assert_allclose(after[1], before[1], rtol=0.0, atol=1e-15)
        self.assertNotAlmostEqual(float(after[1, 0]), 2.0, places=6)
        # covariance of the reset envs is the default prior, not the evolved one
        fresh = ShuttleUKF(num_envs=3)
        np.testing.assert_allclose(ukf.covariance[0], fresh.covariance[0], rtol=0.0, atol=1e-15)
        np.testing.assert_allclose(ukf.covariance[2], fresh.covariance[2], rtol=0.0, atol=1e-15)
        self.assertFalse(np.allclose(ukf.covariance[1], fresh.covariance[1], atol=1e-9))

    def test_reset_all_when_env_ids_is_none(self) -> None:
        ukf = ShuttleUKF(num_envs=2)
        ukf.initialize(position=np.ones((2, 3)), velocity=np.ones((2, 3)))
        ukf.reset(None)
        np.testing.assert_allclose(ukf.state[:, :6], 0.0, atol=1e-15)
        with self.assertRaises(BrainBoundaryError):
            ukf.reset([5])

    # ---- 7. batch isolation + hostile input ----------------------------------
    def test_environments_do_not_share_state_and_bad_batches_are_rejected(self) -> None:
        flight = synthetic_flight(seed=11)
        ukf = ShuttleUKF(num_envs=2)
        ukf.initialize(position=np.tile(flight['measurement_position'][0], (2, 1)),
                       velocity=np.zeros((2, 3)))
        # env 0 is driven, env 1 receives only one very noisy update at the end
        for k in range(1, 25):
            ukf.predict(DT_MEAS)
            ukf.update(measurement_at(flight, k, 1), env_ids=[0])
        env0_p, env1_p = ukf.position[0].copy(), ukf.position[1].copy()
        self.assertLess(rmse(env0_p, flight['position'][24]), 0.2)
        self.assertGreater(rmse(env1_p, flight['position'][24]), 0.2)   # env 1 never moved

        with self.assertRaises(BrainBoundaryError):
            ukf.update(measurement_at(flight, 0, 3))                    # wrong batch size
        with self.assertRaises(BrainBoundaryError):
            ukf.predict(-0.01)
        with self.assertRaises(BrainBoundaryError):
            ukf.update(np.zeros((2, 3)))                                # not a ShuttleMeasurement

    # ---- 8. wind is part of the augmented state ------------------------------
    def test_wind_augmented_state_estimates_the_wind_and_keeps_low_rmse(self) -> None:
        flight = synthetic_flight(seed=5)
        ukf = ShuttleUKF(num_envs=1, enable_wind=True)
        self.assertEqual(ukf.state.shape, (1, 10))
        res = run_filter(ukf, flight)
        wind_err = float(np.linalg.norm(ukf.wind[0] - WIND_TRUE))
        print('\n[T3] wind estimate = %s   true = %s   |err| = %.3f m/s'
              % (np.array2string(ukf.wind[0], precision=3), WIND_TRUE, wind_err))
        self.assertLess(res['filter_rmse'], 0.5 * res['raw_rmse'])
        self.assertLess(wind_err, 0.5)

    # ---- 9. provenance: no invented "measured" number -------------------------
    def test_every_parameter_declares_its_provenance(self) -> None:
        params = ShuttleUKF.parameters()
        for name, param in params.items():
            self.assertIsInstance(param, Param, name)
            self.assertTrue(str(param.source).strip(), name)
            if param.status in UNRESOLVED_STATUSES:
                self.assertIsNone(param.value, '%s claims a value while unresolved' % name)
            else:
                self.assertIsNotNone(param.value, name)
        self.assertIs(params['positional_noise'], POSITION_MEASUREMENT_STD)
        self.assertIs(POSITION_MEASUREMENT_STD.status, AssetStatus.REQUIRES_MEASUREMENT)
        self.assertIsNone(POSITION_MEASUREMENT_STD.value)
        self.assertIsNone(VELOCITY_MEASUREMENT_STD.value)
        self.assertIs(SHUTTLE_DRAG_K_REFERENCE.status, AssetStatus.TRACEABLE_REFERENCE)
        for name in ('initial_drag_prior', 'drag_process_std', 'process_accel_std',
                     'wind_process_std', 'initial_position_std', 'initial_velocity_std',
                     'initial_drag_std', 'drag_k_min', 'drag_k_max'):
            self.assertIs(params[name].status, AssetStatus.TEMP_PARAMETERIZED_PROXY, name)
        # the runtime defaults really come from those declarations
        ukf = ShuttleUKF(num_envs=1)
        self.assertAlmostEqual(ukf.process_accel_std, float(PROCESS_ACCEL_STD.value))
        self.assertAlmostEqual(ukf.drag_process_std, float(DRAG_K_PROCESS_STD.value))
        self.assertAlmostEqual(ukf.wind_process_std, float(WIND_PROCESS_STD.value))
        self.assertAlmostEqual(ukf.initial_position_std, float(INITIAL_POSITION_STD.value))
        self.assertAlmostEqual(ukf.initial_velocity_std, float(INITIAL_VELOCITY_STD.value))
        self.assertAlmostEqual(ukf.initial_drag_std, float(INITIAL_DRAG_STD.value))
        self.assertAlmostEqual(float(DRAG_K_MIN.value), 0.0)
        self.assertGreater(float(DRAG_K_MAX.value), float(SHUTTLE_DRAG_K_REFERENCE.value))

    # ---- 10. measurement fusion uses ShuttleMeasurement.covariance ------------
    def test_measurement_update_uses_the_reported_covariance(self) -> None:
        """A 100x larger reported covariance must be trusted 100x less."""
        flight = synthetic_flight(seed=3)
        tight = run_filter(ShuttleUKF(num_envs=1), flight)
        ukf_loose = ShuttleUKF(num_envs=1)
        ukf_loose.initialize(position=np.tile(flight['measurement_position'][0], (1, 1)),
                             velocity=np.zeros((1, 3)))
        for k in range(1, 40):
            ukf_loose.predict(DT_MEAS)
            m = measurement_at(flight, k)
            m.covariance = m.covariance * 1e4
            ukf_loose.update(m)
        self.assertGreater(np.linalg.norm(ukf_loose.position[0] - flight['position'][39]),
                           np.linalg.norm(ukf_loose.position[0] - flight['measurement_position'][39]))
        self.assertLess(tight['filter_rmse'], 0.5 * tight['raw_rmse'])


if __name__ == '__main__':
    unittest.main(verbosity=2)
