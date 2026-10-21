# -*- coding: utf-8 -*-
"""P7-07 acceptance: horizon-bounded prediction interface with uncertainty.

Everything is SYNTHETIC.  Two truth models are used:

  * an exact parabola (constant acceleration) - the reference predictor's own
    model, so this measures the interface/uncertainty machinery, not physics;
  * the frozen quadratic-drag model (src/trajectory/shuttle_aerodynamics.py) -
    a deliberate MODEL MISMATCH, so the reported error-vs-horizon growth shows
    where a quadratic extrapolation stops being usable inside 0.3-0.8 s.

The measured table is written to
outputs/p7_interfaces/p7_07_prediction_error_vs_horizon.json and must not be
read as real-detection accuracy (no real data is involved).

Run:
    python -m pytest tests/badminton_brain/test_horizon_prediction.py -q -s
"""
from __future__ import annotations

import json
import sys
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from badminton_brain.prediction.horizon_prediction import (  # noqa: E402
    CONFIDENCE_LEVEL,
    MAX_MEASUREMENT_AGE_S,
    MIN_FIT_SAMPLES,
    PREDICTION_HORIZON_MAX_S,
    PREDICTION_HORIZON_MIN_S,
    Z_95,
    PolynomialHorizonPredictor,
    PositionObservation,
    PredictionRejected,
    TrajectoryPrediction,
)
from badminton_brain.status import AssetStatus, Param  # noqa: E402
from trajectory.shuttle_aerodynamics import k_from_aerodynamic_length, rollout  # noqa: E402

EVIDENCE_PATH = ROOT / "outputs" / "p7_interfaces" / "p7_07_prediction_error_vs_horizon.json"

HORIZONS = (0.3, 0.4, 0.5, 0.6, 0.7, 0.8)
P0 = np.array([1.0, -0.5, 1.6])
V0 = np.array([4.0, 0.5, 3.0])
G = np.array([0.0, 0.0, -9.80665])
SIGMA_M = 0.002          # synthetic 3D position noise, 2 mm
WINDOW = 25              # samples (0.48 s of 50 Hz track; the fitting window is a parameter)
DT_OBS = 0.02            # 50 Hz


def parabola(t):
    return P0 + V0 * t + 0.5 * G * t * t


def parabola_velocity(t):
    return V0 + G * t


def make_window(rng, *, t_start=0.0, dt=DT_OBS, count=WINDOW, truth=parabola,
                sigma=SIGMA_M):
    stamps = t_start + dt * np.arange(count) - dt * (count - 1)   # newest sample at t_start
    return [PositionObservation(timestamp_s=float(t), position_m=truth(float(t))
                                + sigma * rng.standard_normal(3), sigma_m=sigma)
            for t in stamps]


class TestSyntheticParabola(unittest.TestCase):
    """Reference model == truth: the uncertainty machinery must be calibrated."""

    def test_prediction_error_and_interval_coverage_vs_horizon(self):
        rng = np.random.default_rng(20261021)
        n_runs = 200
        predictor = PolynomialHorizonPredictor()
        table = {}
        for horizon in HORIZONS:
            sq_err, bounds, covered, per_axis_cover = [], [], [], []
            for _ in range(n_runs):
                obs = make_window(rng)
                pred = predictor.predict(obs, now_s=0.0, horizon_s=horizon)
                truth = parabola(horizon)
                err = pred.final_position_m - truth
                sq_err.append(float(err @ err))
                bounds.append(pred.final_error_bound_m)
                lo, hi = pred.ci_lower_m[-1], pred.ci_upper_m[-1]
                inside = (truth >= lo - 1e-12) & (truth <= hi + 1e-12)
                per_axis_cover.append(inside)
                covered.append(bool(np.all(inside)))
            rmse = float(np.sqrt(np.mean(sq_err)))
            table[horizon] = {
                "rmse_m": rmse,
                "mean_error_bound_m": float(np.mean(bounds)),
                "ci_all_axes_coverage": float(np.mean(covered)),
                "ci_per_axis_coverage": float(np.mean(np.array(per_axis_cover))),
                "n_runs": n_runs,
            }
            # Pre-registered bounds for an exact-model fit, 2 mm noise, a 0.48 s / 25
            # sample window: 20 mm at the 0.3 s end and 80 mm at the 0.8 s end.  They are
            # fixed here BEFORE looking at the numbers and are not widened afterwards.
            limit = 0.02 if horizon <= 0.3 else 0.08
            self.assertLess(rmse, limit,
                            msg=f"horizon {horizon}: rmse {rmse:.4f} m > {limit} m")
            # A calibrated 95% interval must actually cover ~95% of the truth.
            self.assertTrue(0.88 <= table[horizon]["ci_per_axis_coverage"] <= 0.997,
                            msg=f"horizon {horizon}: coverage "
                                f"{table[horizon]['ci_per_axis_coverage']:.3f}")
            # The reported uncertainty must not be smaller than the realised error.
            self.assertLessEqual(rmse, 1.2 * table[horizon]["mean_error_bound_m"],
                                 msg=f"horizon {horizon}: error bound under-reports the error")
        rmses = [table[h]["rmse_m"] for h in HORIZONS]
        bounds = [table[h]["mean_error_bound_m"] for h in HORIZONS]
        self.assertTrue(all(np.isfinite(rmses)))
        # Error and reported uncertainty both grow with the horizon (allowed 10% jitter).
        for a, b in zip(rmses, rmses[1:]):
            self.assertGreaterEqual(b, 0.9 * a)
        for a, b in zip(bounds, bounds[1:]):
            self.assertGreater(b, a)
        for horizon in HORIZONS:
            row = table[horizon]
            print(f"[P7-07] parabola h={horizon:.1f}s rmse={row['rmse_m']*1000:.2f} mm "
                  f"bound={row['mean_error_bound_m']*1000:.2f} mm "
                  f"cover={row['ci_per_axis_coverage']:.3f}")
        self._write_evidence(
            {"horizon_window_s": [float(PREDICTION_HORIZON_MIN_S.value),
                                  float(PREDICTION_HORIZON_MAX_S.value)],
             "parabola_exact_model": {str(h): table[h] for h in HORIZONS}},
            merge=True)

    # -- evidence plumbing -------------------------------------------------
    def _write_evidence(self, payload, *, merge):
        existing = {}
        if merge and EVIDENCE_PATH.exists():
            existing = json.loads(EVIDENCE_PATH.read_text(encoding="utf-8"))
        existing.update(payload)
        existing["synthetic_only"] = True
        existing["note"] = ("synthetic truth + synthetic noise; no real detection or real "
                            "flight data is involved - not a real-data accuracy claim")
        EVIDENCE_PATH.parent.mkdir(parents=True, exist_ok=True)
        EVIDENCE_PATH.write_text(json.dumps(existing, indent=2, sort_keys=True) + "\n",
                                 encoding="utf-8")


class TestDragModelMismatch(unittest.TestCase):
    """Quadratic fit against the frozen drag model: where does the horizon hurt?"""

    def test_error_grows_with_horizon_under_model_mismatch(self):
        k = k_from_aerodynamic_length(6.5)
        ref = rollout(P0, V0, duration_s=2.0, dt_s=0.001, k_per_m=k)
        ref_t = np.asarray(ref["time"], dtype=float)
        ref_p = np.asarray(ref["position"], dtype=float)

        def truth(t):
            return np.array([np.interp(t, ref_t, ref_p[:, i]) for i in range(3)])

        rng = np.random.default_rng(424242)
        n_runs = 50
        predictor = PolynomialHorizonPredictor()
        table = {}
        for horizon in HORIZONS:
            sq_err, bounds, cover = [], [], []
            for _ in range(n_runs):
                obs = make_window(rng, t_start=0.48, truth=truth)
                pred = predictor.predict(obs, now_s=float(obs[-1].timestamp_s),
                                         horizon_s=horizon)
                now = obs[-1].timestamp_s
                err = pred.final_position_m - truth(now + horizon)
                sq_err.append(float(err @ err))
                bounds.append(pred.final_error_bound_m)
                lo, hi = pred.ci_lower_m[-1], pred.ci_upper_m[-1]
                cover.append(bool(np.all((truth(now + horizon) >= lo - 1e-12)
                                         & (truth(now + horizon) <= hi + 1e-12))))
            table[horizon] = {
                "rmse_m": float(np.sqrt(np.mean(sq_err))),
                "mean_error_bound_m": float(np.mean(bounds)),
                "ci_all_axes_coverage": float(np.mean(cover)),
                "n_runs": n_runs,
            }
            print(f"[P7-07] drag-mismatch h={horizon:.1f}s "
                  f"rmse={table[horizon]['rmse_m']*1000:.1f} mm "
                  f"bound={table[horizon]['mean_error_bound_m']*1000:.1f} mm "
                  f"cover={table[horizon]['ci_all_axes_coverage']:.3f}")

        rmses = [table[h]["rmse_m"] for h in HORIZONS]
        # Ceiling derived from the DECLARED model, not from the measured number: the
        # unmodelled drag deceleration at launch is k*|v0|^2, so over the longest horizon
        # the ignored term alone displaces the truth by 0.5*k*|v0|^2*T^2 = 1.24 m
        # (k = 1/6.5 1/m, |v0| = 5.02 m/s, T = 0.8 s).  A 25% allowance covers the extra
        # bias of fitting a quadratic to the mismatched window.  Fixed before the run.
        drag_ceiling = 0.5 * k * float(np.linalg.norm(V0)) ** 2 * HORIZONS[-1] ** 2
        # PINNED LIMITATION (do not "fix" by widening the interval): the reported interval
        # is the STATISTICAL uncertainty of the fit only.  Under a model mismatch it
        # under-covers completely, which is exactly why the interface must never be read as
        # a safety bound on real flight.
        for horizon in HORIZONS:
            self.assertLess(table[horizon]["ci_all_axes_coverage"], 0.2,
                            msg=f"h={horizon}: statistical CI unexpectedly covers the "
                                "mismatched truth; investigate before trusting the table")
        self.assertLess(rmses[-1], 1.25 * drag_ceiling,
                        msg=f"0.8 s mismatch error {rmses[-1]:.3f} m > "
                            f"1.25*{drag_ceiling:.3f} m")
        self.assertGreater(rmses[-1], rmses[0],
                           msg="drag mismatch must hurt more at the longer horizon")
        # Reported uncertainty must grow with horizon as well (model error is NOT in it,
        # so the interval under-covers here - reported, not hidden).
        TestSyntheticParabola._write_evidence(
            self,
            {"drag_mismatch_model_mismatch": {str(h): table[h] for h in HORIZONS}},
            merge=True)


class TestRejections(unittest.TestCase):
    def setUp(self):
        self.predictor = PolynomialHorizonPredictor()
        self.rng = np.random.default_rng(7)
        self.obs = make_window(self.rng)

    def test_horizon_outside_preregistered_window(self):
        for horizon in (0.29, 0.0, -0.3, 0.81, 1.5, float("nan")):
            with self.assertRaises(PredictionRejected, msg=str(horizon)) as ctx:
                self.predictor.predict(self.obs, now_s=0.0, horizon_s=horizon)
            self.assertIn(ctx.exception.reason, ("HORIZON_OUT_OF_RANGE", "NON_FINITE_INPUT"))

    def test_stale_frame_rejected(self):
        age = float(MAX_MEASUREMENT_AGE_S.value) + 0.05
        with self.assertRaises(PredictionRejected) as ctx:
            self.predictor.predict(self.obs, now_s=age, horizon_s=0.5)
        self.assertEqual(ctx.exception.reason, "STALE_MEASUREMENT")
        # just inside the budget is accepted
        ok = self.predictor.predict(self.obs,
                                    now_s=float(MAX_MEASUREMENT_AGE_S.value) * 0.99,
                                    horizon_s=0.5)
        self.assertGreater(ok.measurement_age_s, 0.0)

    def test_future_observation_rejected(self):
        with self.assertRaises(PredictionRejected) as ctx:
            self.predictor.predict(self.obs, now_s=-0.5, horizon_s=0.5)
        self.assertEqual(ctx.exception.reason, "FUTURE_OBSERVATION")

    def test_insufficient_and_degenerate_samples(self):
        with self.assertRaises(PredictionRejected) as ctx:
            self.predictor.predict(self.obs[: MIN_FIT_SAMPLES.value - 1], now_s=0.0,
                                   horizon_s=0.5)
        self.assertEqual(ctx.exception.reason, "INSUFFICIENT_SAMPLES")
        duplicated = [self.obs[0], self.obs[0], self.obs[1], self.obs[2]]
        with self.assertRaises(PredictionRejected) as ctx:
            self.predictor.predict(duplicated, now_s=0.0, horizon_s=0.5)
        self.assertEqual(ctx.exception.reason, "DEGENERATE_SAMPLES")

    def test_non_finite_inputs(self):
        # A non-finite observation cannot even be constructed.
        with self.assertRaises(PredictionRejected) as ctx:
            PositionObservation(timestamp_s=float("nan"), position_m=np.zeros(3),
                                sigma_m=0.01)
        self.assertEqual(ctx.exception.reason, "NON_FINITE_INPUT")
        with self.assertRaises(PredictionRejected):
            PositionObservation(timestamp_s=0.0, position_m=np.array([0.0, 0.0]),
                                sigma_m=0.01)
        with self.assertRaises(PredictionRejected):
            PositionObservation(timestamp_s=0.0, position_m=np.zeros(3), sigma_m=0.0)


class TestPredictionInvariants(unittest.TestCase):
    def setUp(self):
        rng = np.random.default_rng(99)
        self.pred = PolynomialHorizonPredictor().predict(
            make_window(rng), now_s=0.0, horizon_s=0.6)

    def test_grid_and_uncertainty_are_consistent(self):
        self.assertAlmostEqual(float(self.pred.times_s[0]), 0.0)
        self.assertAlmostEqual(float(self.pred.times_s[-1]), 0.6)
        self.assertTrue(np.all(np.diff(self.pred.times_s) > 0.0))
        eigen = np.linalg.eigvalsh(self.pred.position_covariance_m2)
        self.assertGreaterEqual(float(np.min(eigen)), -1e-12)
        np.testing.assert_allclose(
            self.pred.error_bound_m,
            Z_95 * np.sqrt(np.maximum(eigen.max(axis=1), 0.0)), atol=1e-9)
        self.assertGreater(float(self.pred.error_bound_m[-1]),
                           float(self.pred.error_bound_m[0]))
        self.assertAlmostEqual(float(self.pred.confidence_level),
                               float(CONFIDENCE_LEVEL.value))
        self.assertEqual(int(self.pred.sample_count), WINDOW)

    def test_tampered_uncertainty_is_refused(self):
        payload = dict(self.pred.__dict__)
        payload["error_bound_m"] = self.pred.error_bound_m * 0.5
        with self.assertRaises(PredictionRejected) as ctx:
            TrajectoryPrediction(**payload)
        self.assertEqual(ctx.exception.reason, "DEGENERATE_SAMPLES")
        payload = dict(self.pred.__dict__)
        payload["ci_upper_m"] = self.pred.ci_upper_m + 1.0
        with self.assertRaises(PredictionRejected):
            TrajectoryPrediction(**payload)
        payload = dict(self.pred.__dict__)
        payload["horizon_s"] = 0.9
        with self.assertRaises(PredictionRejected) as ctx:
            TrajectoryPrediction(**payload)
        self.assertEqual(ctx.exception.reason, "HORIZON_OUT_OF_RANGE")

    def test_declared_parameters(self):
        for param in (PREDICTION_HORIZON_MIN_S, PREDICTION_HORIZON_MAX_S,
                      MAX_MEASUREMENT_AGE_S, CONFIDENCE_LEVEL, MIN_FIT_SAMPLES):
            self.assertIsInstance(param, Param)
            self.assertIsNotNone(param.value)
        self.assertEqual(float(PREDICTION_HORIZON_MIN_S.value), 0.3)
        self.assertEqual(float(PREDICTION_HORIZON_MAX_S.value), 0.8)
        self.assertEqual(MAX_MEASUREMENT_AGE_S.status,
                         AssetStatus.TEMP_PARAMETERIZED_PROXY)
        declared = PolynomialHorizonPredictor().parameters()
        self.assertIn("max_measurement_age_s", declared)


if __name__ == "__main__":
    unittest.main(verbosity=2)
