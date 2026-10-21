# -*- coding: utf-8 -*-
"""P7-07: a trajectory-prediction interface with an explicit horizon and uncertainty.

What P7-07 asks for
-------------------
"0.3-0.8 s unified horizon, error / confidence interval / missing-observation
handling and per-scenario replay" (master spec P7-07).  The existing prediction
module (prediction/physics_predictor.py) rolls an already-estimated 3D state
forward and publishes PredictedTrajectory, but it accepts ANY horizon, carries no
uncertainty and cannot express "this prediction is built on a stale frame".
This module is the missing *interface layer* around such a predictor:

  * the horizon is validated against a PRE-REGISTERED range (0.3 s - 0.8 s);
    a request outside it is rejected instead of silently extrapolated,
  * every sample carries a covariance, a 1.96-sigma error bound and a 95%
    confidence interval, so a consumer can see how much to trust it,
  * a frame whose measurement is older than the declared latency budget is
    rejected by name (STALE_MEASUREMENT) rather than predicted from.

Reference predictor
-------------------
PolynomialHorizonPredictor is the reference implementation used by the tests: a
weighted (per-sample sigma) second-order polynomial fit in the court frame,
i.e. the exact model for constant acceleration, propagated with its own
covariance through the extrapolation Jacobian.  Its bias for a real shuttle
grows with the horizon because the frozen aero model adds a quadratic drag term
k|v|^2 (k = 1/6.5 1/m, trajectory/shuttle_aerodynamics.py) that a quadratic fit
cannot represent; the test measures that growth against the frozen model and
writes the table to
outputs/p7_interfaces/p7_07_prediction_error_vs_horizon.json.  These numbers are
SYNTHETIC: they quantify the interface, not the real-world accuracy of any
detector.

Run the tests with:
    python -m pytest tests/badminton_brain/test_horizon_prediction.py -q
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Dict, Optional, Sequence, Tuple

import numpy as np

from ..status import AssetStatus, Param

__all__ = [
    "PREDICTION_HORIZON_MIN_S", "PREDICTION_HORIZON_MAX_S", "MAX_MEASUREMENT_AGE_S",
    "MAX_MEASUREMENT_AGE_S_MEASURED", "CONFIDENCE_LEVEL", "Z_95", "MIN_FIT_SAMPLES",
    "DEFAULT_SAMPLE_COUNT", "REJECTION_REASONS", "PREDICTION_MODEL_ID",
    "PredictionRejected", "PositionObservation", "TrajectoryPrediction",
    "PolynomialHorizonPredictor",
]

# Pre-registered horizon window.  Source: docs/architecture/ROBOT_BRAIN.md 4.3
# ("the predictor rolls the state forward about 0.3-0.8 s"), already quoted by
# prediction/physics_predictor.py DEFAULT_HORIZON_S = 0.6.  P7-07 promotes that
# prose range to an enforced contract, so the window is fixed BEFORE any
# measurement is taken (master spec section 4: thresholds are pre-registered,
# never widened after seeing results).
PREDICTION_HORIZON_MIN_S = Param(
    0.3, AssetStatus.TRACEABLE_REFERENCE,
    "ROBOT_BRAIN.md 4.3 lower bound; also the shortest horizon for which a hard-hit "
    "shuttle still has a plausible planning budget.  Pre-registered, not tuned.")
PREDICTION_HORIZON_MAX_S = Param(
    0.8, AssetStatus.TRACEABLE_REFERENCE,
    "ROBOT_BRAIN.md 4.3 upper bound.  Beyond it the constant-acceleration reference model "
    "is dominated by the unmodelled drag term and the interface would be reporting noise.")

# Latency budget for the newest measurement a prediction may be built on.
MAX_MEASUREMENT_AGE_S = Param(
    0.10, AssetStatus.TEMP_PARAMETERIZED_PROXY,
    "TEMP: 100 ms = 5 frames at the 50 Hz stereo rate used across the existing perception "
    "tests.  Justification: the loop budget of 0.3 s horizon minus the ~0.1 s perception + "
    "inference latency leaves at least 0.2 s for decision/planning, so data older than the "
    "latency budget itself cannot be part of a decision-grade prediction.  Proxy until the "
    "real end-to-end latency exists (P7-13).")
MAX_MEASUREMENT_AGE_S_MEASURED = Param(
    None, AssetStatus.REQUIRES_MEASUREMENT,
    "measured capture->detection->prediction latency of this rig (P7-13 latency budget); no "
    "such measurement exists yet, so the 100 ms proxy stands.")

CONFIDENCE_LEVEL = Param(
    0.95, AssetStatus.TRACEABLE_REFERENCE,
    "two-sided 95% interval, z = 1.959963984540054 (standard normal quantile).")

Z_95 = 1.959963984540054

MIN_FIT_SAMPLES = Param(
    4, AssetStatus.TEMP_PARAMETERIZED_PROXY,
    "a second-order polynomial has 3 parameters per axis, so 4 samples leave one redundancy "
    "degree of freedom; fewer samples would make the reported uncertainty meaningless.")

DEFAULT_SAMPLE_COUNT = 17

PREDICTION_MODEL_ID = "court_frame_constant_acceleration_polynomial/v1"

REJECTION_REASONS: tuple[str, ...] = (
    "HORIZON_OUT_OF_RANGE",
    "STALE_MEASUREMENT",
    "FUTURE_OBSERVATION",
    "INSUFFICIENT_SAMPLES",
    "DEGENERATE_SAMPLES",
    "NON_FINITE_INPUT",
    "BAD_REFERENCE_TIME",
)


class PredictionRejected(ValueError):
    """A prediction request that the P7-07 contract refuses, with a named reason."""

    def __init__(self, reason: str, detail: str) -> None:
        if reason not in REJECTION_REASONS:
            raise ValueError(f"unknown rejection reason {reason!r}")
        self.reason = reason
        self.detail = detail
        super().__init__(f"{reason}: {detail}")


@dataclass(frozen=True)
class PositionObservation:
    """One court-frame shuttle position sample with an isotropic per-axis sigma.

    Court frame, metres (badminton_brain.types.COURT_FRAME).  This is the
    *output* of a calibrated 3D path (P7-06); a 2D detection must never be
    dressed up as one of these.
    """

    timestamp_s: float
    position_m: Any
    sigma_m: float

    def __post_init__(self) -> None:
        t = float(self.timestamp_s)
        if not math.isfinite(t):
            raise PredictionRejected("NON_FINITE_INPUT", f"timestamp_s={self.timestamp_s!r}")
        pos = np.asarray(self.position_m, dtype=float)
        if pos.shape != (3,):
            raise PredictionRejected("NON_FINITE_INPUT",
                                     f"position_m must be (3,), got {pos.shape}")
        if not np.all(np.isfinite(pos)):
            raise PredictionRejected("NON_FINITE_INPUT", "position_m contains NaN/Inf")
        s = float(self.sigma_m)
        if not math.isfinite(s) or s <= 0.0:
            raise PredictionRejected("NON_FINITE_INPUT",
                                     f"sigma_m must be finite and > 0, got {self.sigma_m!r}")
        object.__setattr__(self, "timestamp_s", t)
        object.__setattr__(self, "position_m", pos)
        object.__setattr__(self, "sigma_m", s)


@dataclass(frozen=True)
class TrajectoryPrediction:
    """Predicted court-frame trajectory with an explicit horizon and uncertainty.

    times_s                    (T,) absolute seconds, times_s[0] == reference_timestamp_s,
                               times_s[-1] == reference_timestamp_s + horizon_s
    position_m                 (T, 3) metres
    velocity_mps               (T, 3) metres per second
    position_covariance_m2     (T, 3, 3) positive semi-definite per sample
    confidence_level           in (0, 1), normally CONFIDENCE_LEVEL
    error_bound_m              (T,) z * sqrt(max eigenvalue of the sample covariance),
                               recomputed and cross-checked on construction
    ci_lower_m / ci_upper_m    (T, 3) position -/+ z * per-axis sigma
    measurement_age_s          age of the newest observation at the reference time
    sample_count               number of observations the fit consumed
    """

    reference_timestamp_s: float
    horizon_s: float
    times_s: Any
    position_m: Any
    velocity_mps: Any
    position_covariance_m2: Any
    confidence_level: float
    error_bound_m: Any
    ci_lower_m: Any
    ci_upper_m: Any
    measurement_age_s: float
    sample_count: int
    model: str = PREDICTION_MODEL_ID

    def __post_init__(self) -> None:
        lo = float(PREDICTION_HORIZON_MIN_S.value)
        hi = float(PREDICTION_HORIZON_MAX_S.value)
        h = float(self.horizon_s)
        if not math.isfinite(h) or not (lo <= h <= hi):
            raise PredictionRejected(
                "HORIZON_OUT_OF_RANGE",
                f"horizon_s={self.horizon_s!r} is outside the pre-registered "
                f"[{lo}, {hi}] s window (ROBOT_BRAIN.md 4.3); extend the contract "
                "deliberately instead of extrapolating past it")
        t0 = float(self.reference_timestamp_s)
        if not math.isfinite(t0):
            raise PredictionRejected("BAD_REFERENCE_TIME",
                                     f"reference_timestamp_s={self.reference_timestamp_s!r}")
        times = np.asarray(self.times_s, dtype=float)
        if times.ndim != 1 or times.size < 2:
            raise PredictionRejected("DEGENERATE_SAMPLES",
                                     f"times_s must be 1-D with >= 2 samples, got {times.shape}")
        if not np.all(np.isfinite(times)):
            raise PredictionRejected("NON_FINITE_INPUT", "times_s contains NaN/Inf")
        if not np.all(np.diff(times) > 0.0):
            raise PredictionRejected("DEGENERATE_SAMPLES", "times_s must be strictly increasing")
        if abs(float(times[0]) - t0) > 1e-9 or abs(float(times[-1]) - (t0 + h)) > 1e-9:
            raise PredictionRejected(
                "HORIZON_OUT_OF_RANGE",
                f"times_s must span [{t0}, {t0 + h}] exactly, got "
                f"[{float(times[0])}, {float(times[-1])}]")
        pos = np.asarray(self.position_m, dtype=float)
        vel = np.asarray(self.velocity_mps, dtype=float)
        cov = np.asarray(self.position_covariance_m2, dtype=float)
        n = times.size
        if pos.shape != (n, 3) or vel.shape != (n, 3):
            raise PredictionRejected("DEGENERATE_SAMPLES",
                                     f"position/velocity must be (T,3) with T={n}")
        if cov.shape != (n, 3, 3):
            raise PredictionRejected("DEGENERATE_SAMPLES",
                                     f"position_covariance_m2 must be ({n},3,3)")
        for name, arr in (("position_m", pos), ("velocity_mps", vel),
                          ("position_covariance_m2", cov)):
            if not np.all(np.isfinite(arr)):
                raise PredictionRejected("NON_FINITE_INPUT", f"{name} contains NaN/Inf")
        sym = 0.5 * (cov + np.swapaxes(cov, 1, 2))
        if not np.allclose(cov, sym, atol=1e-12):
            raise PredictionRejected("NON_FINITE_INPUT", "position covariance must be symmetric")
        eigen = np.linalg.eigvalsh(sym)
        if float(np.min(eigen)) < -1e-12:
            raise PredictionRejected("NON_FINITE_INPUT",
                                     "position covariance must be positive semi-definite")
        level = float(self.confidence_level)
        if not (0.0 < level < 1.0):
            raise PredictionRejected("NON_FINITE_INPUT",
                                     f"confidence_level must be in (0,1), got {level}")
        z = _z_for_level(level)
        bound = np.asarray(self.error_bound_m, dtype=float)
        if bound.shape != (n,) or not np.all(np.isfinite(bound)):
            raise PredictionRejected("DEGENERATE_SAMPLES", f"error_bound_m must be ({n},)")
        recomputed = z * np.sqrt(np.maximum(np.max(eigen, axis=1), 0.0))
        if not np.allclose(bound, recomputed, atol=1e-9, rtol=1e-9):
            raise PredictionRejected(
                "DEGENERATE_SAMPLES",
                "error_bound_m is inconsistent with position_covariance_m2 "
                "(must equal z*sqrt(lambda_max))")
        sigma = np.sqrt(np.maximum(np.diagonal(sym, axis1=1, axis2=2), 0.0))
        lo_arr = np.asarray(self.ci_lower_m, dtype=float)
        hi_arr = np.asarray(self.ci_upper_m, dtype=float)
        if lo_arr.shape != (n, 3) or hi_arr.shape != (n, 3):
            raise PredictionRejected("DEGENERATE_SAMPLES", "ci bounds must be (T,3)")
        if not np.allclose(lo_arr, pos - z * sigma, atol=1e-9, rtol=1e-9):
            raise PredictionRejected("DEGENERATE_SAMPLES", "ci_lower_m inconsistent with cov")
        if not np.allclose(hi_arr, pos + z * sigma, atol=1e-9, rtol=1e-9):
            raise PredictionRejected("DEGENERATE_SAMPLES", "ci_upper_m inconsistent with cov")
        age = float(self.measurement_age_s)
        if not math.isfinite(age) or age < 0.0:
            raise PredictionRejected("STALE_MEASUREMENT",
                                     f"measurement_age_s must be finite and >= 0, got {age}")
        if isinstance(self.sample_count, bool) or int(self.sample_count) < MIN_FIT_SAMPLES.value:
            raise PredictionRejected(
                "INSUFFICIENT_SAMPLES",
                f"sample_count={self.sample_count!r} < {MIN_FIT_SAMPLES.value}")
        object.__setattr__(self, "times_s", times)
        object.__setattr__(self, "position_m", pos)
        object.__setattr__(self, "velocity_mps", vel)
        object.__setattr__(self, "position_covariance_m2", cov)
        object.__setattr__(self, "error_bound_m", bound)
        object.__setattr__(self, "ci_lower_m", lo_arr)
        object.__setattr__(self, "ci_upper_m", hi_arr)
        object.__setattr__(self, "reference_timestamp_s", t0)
        object.__setattr__(self, "horizon_s", h)
        object.__setattr__(self, "confidence_level", level)
        object.__setattr__(self, "measurement_age_s", age)
        object.__setattr__(self, "sample_count", int(self.sample_count))

    @property
    def final_position_m(self) -> np.ndarray:
        return self.position_m[-1]

    @property
    def final_error_bound_m(self) -> float:
        return float(self.error_bound_m[-1])


def _z_for_level(level: float) -> float:
    """Two-sided standard-normal quantile for the supported confidence levels."""
    table = {
        0.90: 1.6448536269514722,
        0.95: Z_95,
        0.99: 2.5758293035489004,
    }
    for key, value in table.items():
        if abs(level - key) < 1e-12:
            return value
    raise PredictionRejected(
        "NON_FINITE_INPUT",
        f"confidence_level {level} is not one of the declared levels {sorted(table)}; "
        "add the quantile to the table instead of approximating it")


class PolynomialHorizonPredictor:
    """Reference predictor: weighted quadratic fit + covariance propagation.

    The fit is exact for constant acceleration, which is why the synthetic
    parabola test can assert millimetre-level error while the drag-model test in
    the same file reports the model-mismatch growth.  Nothing here is calibrated
    against real data.
    """

    model = PREDICTION_MODEL_ID

    def __init__(self, *, sample_count: int = DEFAULT_SAMPLE_COUNT,
                 max_age_s: Optional[float] = None) -> None:
        count = int(sample_count)
        if count < 2:
            raise PredictionRejected("DEGENERATE_SAMPLES", "sample_count must be >= 2")
        self._samples = count
        self._max_age = float(max_age_s if max_age_s is not None
                              else MAX_MEASUREMENT_AGE_S.value)
        if not math.isfinite(self._max_age) or self._max_age <= 0.0:
            raise PredictionRejected("STALE_MEASUREMENT", "max_age_s must be finite and > 0")

    @property
    def max_age_s(self) -> float:
        return self._max_age

    def parameters(self) -> Dict[str, Param]:
        return {
            "horizon_min_s": PREDICTION_HORIZON_MIN_S,
            "horizon_max_s": PREDICTION_HORIZON_MAX_S,
            "max_measurement_age_s": Param(
                self._max_age, AssetStatus.TEMP_PARAMETERIZED_PROXY,
                "MAX_MEASUREMENT_AGE_S or constructor override"),
            "confidence_level": CONFIDENCE_LEVEL,
            "min_fit_samples": MIN_FIT_SAMPLES,
        }

    def predict(self, observations: Sequence[PositionObservation], *, now_s: float,
                horizon_s: float, max_age_s: Optional[float] = None) -> TrajectoryPrediction:
        """Predict from the observation window, or raise PredictionRejected."""
        lo = float(PREDICTION_HORIZON_MIN_S.value)
        hi = float(PREDICTION_HORIZON_MAX_S.value)
        h = float(horizon_s)
        if not math.isfinite(h):
            raise PredictionRejected("NON_FINITE_INPUT", f"horizon_s={horizon_s!r}")
        if not (lo <= h <= hi):
            raise PredictionRejected(
                "HORIZON_OUT_OF_RANGE",
                f"horizon_s={h} outside the pre-registered [{lo}, {hi}] s window")
        now = float(now_s)
        if not math.isfinite(now):
            raise PredictionRejected("BAD_REFERENCE_TIME", f"now_s={now_s!r}")
        limit = float(max_age_s if max_age_s is not None else self._max_age)
        obs = list(observations)
        if len(obs) < int(MIN_FIT_SAMPLES.value):
            raise PredictionRejected(
                "INSUFFICIENT_SAMPLES",
                f"{len(obs)} observation(s) < MIN_FIT_SAMPLES={MIN_FIT_SAMPLES.value}")
        for item in obs:
            if not isinstance(item, PositionObservation):
                raise PredictionRejected(
                    "NON_FINITE_INPUT",
                    f"observations must be PositionObservation, got {type(item).__name__}")
        stamps = np.array([o.timestamp_s for o in obs], dtype=float)
        if not np.all(np.isfinite(stamps)):
            raise PredictionRejected("NON_FINITE_INPUT", "observation timestamps not finite")
        if np.any(np.diff(stamps) <= 0.0):
            raise PredictionRejected(
                "DEGENERATE_SAMPLES",
                "observation timestamps must be strictly increasing (de-duplicate or sort "
                "the window before predicting)")
        newest = float(np.max(stamps))
        age = now - newest
        if age > limit:
            raise PredictionRejected(
                "STALE_MEASUREMENT",
                f"newest observation is {age:.6f} s old, budget is {limit:.6f} s: a stale "
                "frame must not be predicted from")
        if age < -1e-9:
            raise PredictionRejected(
                "FUTURE_OBSERVATION",
                f"newest observation is {-age:.6f} s in the future of now_s: "
                "the clock bases disagree")

        tau = stamps - now
        A = np.vstack([np.ones_like(tau), tau, tau ** 2]).T
        sigma = np.array([o.sigma_m for o in obs], dtype=float)
        W = 1.0 / sigma ** 2
        AtWA = A.T @ (W[:, None] * A)
        try:
            AtWA_inv = np.linalg.inv(AtWA)
        except np.linalg.LinAlgError as exc:  # pragma: no cover - defensive
            raise PredictionRejected("DEGENERATE_SAMPLES",
                                     "fit normal equations are singular") from exc
        cond = float(np.linalg.cond(AtWA))
        if not np.isfinite(cond) or cond > 1e12:
            raise PredictionRejected(
                "DEGENERATE_SAMPLES",
                f"fit normal equations are ill-conditioned (cond={cond:.3e}); the "
                "observation window is too short or too clustered in time")
        Y = np.stack([o.position_m for o in obs], axis=0)
        beta = AtWA_inv @ (W[:, None] * A).T @ Y          # (3 coefficients, 3 axes)
        if not np.all(np.isfinite(beta)):
            raise PredictionRejected("NON_FINITE_INPUT", "fit coefficients are not finite")

        offsets = np.linspace(0.0, h, self._samples)
        times = now + offsets
        Jj = np.stack([np.ones_like(offsets), offsets, offsets ** 2], axis=1)  # (T,3)
        Jv = np.stack([np.zeros_like(offsets), np.ones_like(offsets), 2.0 * offsets], axis=1)
        pos = Jj @ beta
        vel = Jv @ beta

        n = times.size
        cov = np.zeros((n, 3, 3))
        # Per-sample propagation: var_axis(t) = j(t)^T AtWA_inv j(t), the axes share the
        # same time design matrix and are fitted independently (diagonal covariance).
        var_pos = np.einsum("ij,jk,ik->i", Jj, AtWA_inv, Jj)
        for axis in range(3):
            cov[:, axis, axis] = np.maximum(var_pos, 0.0)
        z = _z_for_level(float(CONFIDENCE_LEVEL.value))
        eigen = np.linalg.eigvalsh(cov)
        bound = z * np.sqrt(np.maximum(np.max(eigen, axis=1), 0.0))
        sigma_pos = np.sqrt(np.maximum(np.diagonal(cov, axis1=1, axis2=2), 0.0))
        return TrajectoryPrediction(
            reference_timestamp_s=now,
            horizon_s=h,
            times_s=times,
            position_m=pos,
            velocity_mps=vel,
            position_covariance_m2=cov,
            confidence_level=float(CONFIDENCE_LEVEL.value),
            error_bound_m=bound,
            ci_lower_m=pos - z * sigma_pos,
            ci_upper_m=pos + z * sigma_pos,
            measurement_age_s=age,
            sample_count=len(obs),
            model=self.model,
        )
