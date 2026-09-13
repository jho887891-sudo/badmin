# -*- coding: utf-8 -*-
"""T3: augmented-state Unscented Kalman Filter for the shuttlecock (numpy only).

Spec: docs/architecture/ROBOT_BRAIN.md S4.3 / S7 (Augmented-State UKF -> Shuttle State
position / velocity / drag / wind), docs/superpowers/plans/2026-09-13-brain-modules.md (T3),
docs/simulation/BADMINTON_ROBOT.md S4 (authenticity levels) + S12 (TEMP policy).

State vector (Court Frame; S28: no env_origin ever appears here)

    x = [px, py, pz, vx, vy, vz, k]                     (7)    k = quadratic drag coefficient [1/m]
    x = [px, py, pz, vx, vy, vz, k, wx, wy, wz]         (10)   enable_wind=True

The process model is *not* implemented here: every sigma point is propagated with
trajectory.shuttle_aerodynamics.rk4_step (the frozen model p_dot = v,
v_dot = g - k |v - w| (v - w)).  This module only carries the estimator machinery:
sigma points, weights, prediction, measurement update and per-environment isolation.

Authenticity (BADMINTON_ROBOT.md S4/S12): every number this filter uses is declared as a
Param and returned by ShuttleUKF.parameters().  Values that were never measured on this robot
are either REQUIRES_MEASUREMENT (value None, e.g. the stereo measurement noise, which is taken
from ShuttleMeasurement.covariance at runtime instead) or an explicitly labelled
TEMP_PARAMETERIZED_PROXY whose source says what it stands in for.  No number here is presented
as a measurement.

Layer boundary (interfaces.py): this is an estimation-layer *component*; it does not produce
UnifiedState on its own (robot localization owns the robot half), so it is deliberately a plain
class rather than an EstimationModule.  The estimation-layer adapter lives in
badminton_brain/estimation/estimator.py and consumes this API: initialize(...) ->
predict(dt_s) / update(ShuttleMeasurement) (or step) -> position / velocity / drag_k / wind /
state / covariance -> reset(env_ids).
"""
from __future__ import annotations

import math
from typing import Dict, Iterable, Optional, Sequence, Tuple

import numpy as np

from ..status import AssetStatus, Param
from ..types import BrainBoundaryError, ShuttleMeasurement

try:  # src/ on sys.path -> namespace package (same import style as the prediction layer)
    from trajectory.shuttle_aerodynamics import rk4_step
except ImportError:  # pragma: no cover - direct-path import (callers that add src/trajectory)
    from shuttle_aerodynamics import rk4_step  # type: ignore

__all__ = [
    "GRAVITY_PARAM", "SHUTTLE_DRAG_K_REFERENCE", "INITIAL_DRAG_K_PRIOR", "DRAG_K_PROCESS_STD",
    "PROCESS_ACCEL_STD", "WIND_PROCESS_STD", "INITIAL_POSITION_STD", "INITIAL_VELOCITY_STD",
    "INITIAL_DRAG_STD", "INITIAL_WIND_STD", "POSITION_MEASUREMENT_STD",
    "VELOCITY_MEASUREMENT_STD", "FUSED_VELOCITY_MEASUREMENT_STD", "DRAG_K_MIN", "DRAG_K_MAX",
    "POSITION_SLICE", "VELOCITY_SLICE", "DRAG_INDEX", "WIND_SLICE", "ShuttleUKF",
]

# --------------------------------------------------------------------------------------
# Declared parameters.  Nothing in this file may use a number that is not declared here.
# --------------------------------------------------------------------------------------
GRAVITY_PARAM = Param(
    (0.0, 0.0, -9.80665), AssetStatus.TRACEABLE_REFERENCE,
    "standard gravity, identical to the default of trajectory/shuttle_aerodynamics.rk4_step")

SHUTTLE_DRAG_K_REFERENCE = Param(
    1.0 / 6.5, AssetStatus.TRACEABLE_REFERENCE,
    "Darbois Texier et al., Shuttlecock dynamics, Procedia Eng. 34 (2012): natural feather "
    "shuttle aerodynamic length L = 6.5 m -> k = 1/L (docs/SHUTTLECOCK.md S1); same value as "
    "shuttle_aerodynamics.k_from_aerodynamic_length(6.5). Literature reference for the task "
    "model, not a measurement of our shuttles.")

INITIAL_DRAG_K_PRIOR = Param(
    0.5 / 6.5, AssetStatus.TEMP_PARAMETERIZED_PROXY,
    "TEMP initial prior for the augmented drag state (equivalent to L = 13 m, i.e. deliberately "
    "50% away from the reference k): the per-shuttle drag coefficient of our shuttles has never "
    "been measured, so the filter starts from a wrong, clearly labelled prior and must converge "
    "(tests/badminton_brain/test_shuttle_ukf.py). Replace with a measured k once a flight-capture "
    "drag identification exists.")

DRAG_K_PROCESS_STD = Param(
    0.02, AssetStatus.TEMP_PARAMETERIZED_PROXY,
    "TEMP random-walk density [1/m per sqrt(s)] for the drag state; no measured drift rate of k "
    "exists (k is expected to be quasi-constant within one flight, this only keeps the gain alive).")

PROCESS_ACCEL_STD = Param(
    1.0, AssetStatus.TEMP_PARAMETERIZED_PROXY,
    "TEMP white-acceleration noise density [m/s^2 per sqrt(Hz)]: stands in for the unmodelled "
    "spin/lift/draft accelerations that the frozen v0.1 quadratic-drag model ignores "
    "(docs/SHUTTLECOCK.md S6). Replace with a measured innovation statistic.")

WIND_PROCESS_STD = Param(
    1.0, AssetStatus.TEMP_PARAMETERIZED_PROXY,
    "TEMP random-walk density [m/s per sqrt(s)] for the optional wind state; no hall measurement "
    "of the indoor airflow exists.")

INITIAL_POSITION_STD = Param(
    0.05, AssetStatus.TEMP_PARAMETERIZED_PROXY,
    "TEMP initial position uncertainty [m] (only used when initialize() gets no covariance); the "
    "real stereo error is REQUIRES_MEASUREMENT, see positional_noise.")

INITIAL_VELOCITY_STD = Param(
    1.0, AssetStatus.TEMP_PARAMETERIZED_PROXY,
    "TEMP initial velocity uncertainty [m/s] (covers a two-frame finite-difference bootstrap).")

INITIAL_DRAG_STD = Param(
    0.08, AssetStatus.TEMP_PARAMETERIZED_PROXY,
    "TEMP initial drag uncertainty [1/m]: large enough that the reference k lies inside the "
    "initial 1-sigma sigma-point spread of the wrong prior above.")

INITIAL_WIND_STD = Param(
    2.0, AssetStatus.TEMP_PARAMETERIZED_PROXY,
    "TEMP initial wind uncertainty [m/s] (indoor airflow, no measurement available).")

POSITION_MEASUREMENT_STD = Param(
    None, AssetStatus.REQUIRES_MEASUREMENT,
    "stereo triangulation noise is not characterised in this repository: the filter therefore "
    "takes R from ShuttleMeasurement.covariance (perception must report it) instead of inventing "
    "a constant here. Measure it with the stereo calibration / static-target test to resolve.")

VELOCITY_MEASUREMENT_STD = Param(
    None, AssetStatus.REQUIRES_MEASUREMENT,
    "no stereo velocity-noise characterisation exists; the perception velocity covariance is not "
    "part of the frozen ShuttleMeasurement contract.")

FUSED_VELOCITY_MEASUREMENT_STD = Param(
    0.1, AssetStatus.TEMP_PARAMETERIZED_PROXY,
    "TEMP velocity measurement std [m/s] used only when fuse_velocity=True (a conservative proxy "
    "standing in for the unknown stereo velocity noise).")

DRAG_K_MIN = Param(
    0.0, AssetStatus.TEMP_PARAMETERIZED_PROXY,
    "TEMP lower bound of the drag state: sigma points with k < 0 are unphysical and rk4_step "
    "rejects them; clamping is a numerical guard, not a measurement.")

DRAG_K_MAX = Param(
    1.0, AssetStatus.TEMP_PARAMETERIZED_PROXY,
    "TEMP upper bound of the drag state (equivalent to L = 1 m, far below any real shuttle): "
    "keeps a diverging sigma point from producing a non-physical flight model.")

# State layout
POSITION_SLICE = slice(0, 3)
VELOCITY_SLICE = slice(3, 6)
DRAG_INDEX = 6
WIND_SLICE = slice(7, 10)

_MEASUREMENT_JITTER = 1e-12


class ShuttleUKF:
    """Batched (num_envs isolated) augmented-state UKF for one shuttle per environment."""

    BASE_STATE_DIM = 7

    def __init__(
        self,
        num_envs: int = 1,
        *,
        enable_wind: bool = False,
        fuse_velocity: bool = True,
        alpha: float = 1.0,
        beta: float = 2.0,
        kappa: float = 0.0,
        gravity: Optional[Iterable[float]] = None,
        initial_k: Optional[float] = None,
        process_accel_std: Optional[float] = None,
        drag_process_std: Optional[float] = None,
        wind_process_std: Optional[float] = None,
        initial_position_std: Optional[float] = None,
        initial_velocity_std: Optional[float] = None,
        initial_drag_std: Optional[float] = None,
        initial_wind_std: Optional[float] = None,
        velocity_measurement_std: Optional[float] = None,
        drag_k_min: Optional[float] = None,
        drag_k_max: Optional[float] = None,
    ) -> None:
        num_envs = int(num_envs)
        if num_envs < 1:
            raise BrainBoundaryError(f"num_envs must be >= 1, got {num_envs}")
        self.num_envs = num_envs
        self.enable_wind = bool(enable_wind)
        self.fuse_velocity = bool(fuse_velocity)
        self.alpha = float(alpha)
        self.beta = float(beta)
        self.kappa = float(kappa)
        if not math.isfinite(self.alpha) or self.alpha <= 0.0:
            raise BrainBoundaryError("alpha must be finite and > 0")

        self.gravity = self._vector3(
            GRAVITY_PARAM.value if gravity is None else gravity, 'gravity')
        self.state_dim = self.BASE_STATE_DIM + (3 if self.enable_wind else 0)

        self.process_accel_std = self._positive(
            PROCESS_ACCEL_STD.value if process_accel_std is None else process_accel_std,
            'process_accel_std', allow_zero=True)
        self.drag_process_std = self._positive(
            DRAG_K_PROCESS_STD.value if drag_process_std is None else drag_process_std,
            'drag_process_std', allow_zero=True)
        self.wind_process_std = self._positive(
            WIND_PROCESS_STD.value if wind_process_std is None else wind_process_std,
            'wind_process_std', allow_zero=True)
        self.initial_position_std = self._positive(
            INITIAL_POSITION_STD.value if initial_position_std is None else initial_position_std,
            'initial_position_std')
        self.initial_velocity_std = self._positive(
            INITIAL_VELOCITY_STD.value if initial_velocity_std is None else initial_velocity_std,
            'initial_velocity_std')
        self.initial_drag_std = self._positive(
            INITIAL_DRAG_STD.value if initial_drag_std is None else initial_drag_std,
            'initial_drag_std')
        self.initial_wind_std = self._positive(
            INITIAL_WIND_STD.value if initial_wind_std is None else initial_wind_std,
            'initial_wind_std')
        self.velocity_measurement_std = self._positive(
            FUSED_VELOCITY_MEASUREMENT_STD.value if velocity_measurement_std is None
            else velocity_measurement_std, 'velocity_measurement_std')
        self.drag_k_min = float(DRAG_K_MIN.value if drag_k_min is None else drag_k_min)
        self.drag_k_max = float(DRAG_K_MAX.value if drag_k_max is None else drag_k_max)
        if not (math.isfinite(self.drag_k_min) and math.isfinite(self.drag_k_max)
                and 0.0 <= self.drag_k_min < self.drag_k_max):
            raise BrainBoundaryError("drag_k bounds must satisfy 0 <= min < max")
        self.initial_k = float(INITIAL_DRAG_K_PRIOR.value if initial_k is None else initial_k)
        if not math.isfinite(self.initial_k):
            raise BrainBoundaryError("initial_k must be finite")

        # UT weights (scaled unscented transform, Julier and Uhlmann)
        n = self.state_dim
        lam = self.alpha ** 2 * (n + self.kappa) - n
        if n + lam <= 0.0:
            raise BrainBoundaryError("alpha/kappa give a non-positive sigma spread")
        self._lambda = lam
        self._scale = math.sqrt(n + lam)
        wm = np.full(2 * n + 1, 1.0 / (2.0 * (n + lam)))
        wc = wm.copy()
        wm[0] = lam / (n + lam)
        wc[0] = lam / (n + lam) + (1.0 - self.alpha ** 2 + self.beta)
        self._wm = wm
        self._wc = wc

        # Prior snapshot restored by reset(env_ids)
        self._default_state = np.zeros((num_envs, n))
        self._default_state[:, DRAG_INDEX] = self.initial_k
        self._default_covariance = np.tile(self._unit_covariance(), (num_envs, 1, 1))

        self._state = self._default_state.copy()
        self._covariance = self._default_covariance.copy()
        self._last_timestamp: Optional[np.ndarray] = None
        self.last_innovation = np.zeros((0, 3))
        self.last_nis = np.zeros((0,))

    # ------------------------------------------------------------------ declarations
    @classmethod
    def parameters(cls) -> Dict[str, Param]:
        """Every number this filter uses, with its authenticity status (S4/S12)."""
        return {
            'gravity': GRAVITY_PARAM,
            'reference_drag_k': SHUTTLE_DRAG_K_REFERENCE,
            'initial_drag_prior': INITIAL_DRAG_K_PRIOR,
            'drag_process_std': DRAG_K_PROCESS_STD,
            'process_accel_std': PROCESS_ACCEL_STD,
            'wind_process_std': WIND_PROCESS_STD,
            'initial_position_std': INITIAL_POSITION_STD,
            'initial_velocity_std': INITIAL_VELOCITY_STD,
            'initial_drag_std': INITIAL_DRAG_STD,
            'initial_wind_std': INITIAL_WIND_STD,
            'positional_noise': POSITION_MEASUREMENT_STD,
            'velocity_measurement_noise': VELOCITY_MEASUREMENT_STD,
            'fused_velocity_noise': FUSED_VELOCITY_MEASUREMENT_STD,
            'drag_k_min': DRAG_K_MIN,
            'drag_k_max': DRAG_K_MAX,
        }

    # ------------------------------------------------------------------ sigma points
    def sigma_points(self, mean, covariance) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Return (points (M, 2n+1, n), wm (2n+1,), wc (2n+1,)) for the scaled UT.

        The matrix square root is an eigendecomposition with negative eigenvalues clipped to
        zero, so a covariance that lost a little positivity still yields usable sigma points.
        """
        mean_arr = np.atleast_2d(np.asarray(mean, dtype=float))
        cov_arr = np.asarray(covariance, dtype=float)
        if mean_arr.ndim != 2 or mean_arr.shape[-1] != self.state_dim:
            raise BrainBoundaryError(
                f"mean must have shape (M, {self.state_dim}), got {mean_arr.shape}")
        if cov_arr.ndim == 2:
            cov_arr = np.tile(cov_arr, (mean_arr.shape[0], 1, 1))
        if cov_arr.shape != (mean_arr.shape[0], self.state_dim, self.state_dim):
            raise BrainBoundaryError(
                f"covariance must have shape (M, {self.state_dim}, {self.state_dim}), "
                f"got {cov_arr.shape}")
        if not (np.isfinite(mean_arr).all() and np.isfinite(cov_arr).all()):
            raise BrainBoundaryError("sigma_points received NaN/Inf")

        n = self.state_dim
        m = mean_arr.shape[0]
        sym = 0.5 * (cov_arr + np.swapaxes(cov_arr, -1, -2))
        eigval, eigvec = np.linalg.eigh(sym)
        root = eigvec @ (np.sqrt(np.clip(eigval, 0.0, None))[:, :, None]
                         * np.swapaxes(eigvec, -1, -2))

        points = np.empty((m, 2 * n + 1, n))
        points[:, 0, :] = mean_arr
        points[:, 1:1 + n, :] = mean_arr[:, None, :] + self._scale * root
        points[:, 1 + n:, :] = mean_arr[:, None, :] - self._scale * root
        return points, self._wm.copy(), self._wc.copy()

    # ------------------------------------------------------------------ lifecycle
    def initialize(self, position, velocity=None, drag_k=None, covariance=None,
                   timestamp: Optional[float] = None,
                   env_ids: Optional[Sequence[int]] = None) -> None:
        """Set the prior for the selected environments (does not touch the others)."""
        index = self._env_index(env_ids)
        m = index.size
        pos = self._block(position, m, 3, 'position')
        vel = np.zeros((m, 3)) if velocity is None else self._block(velocity, m, 3, 'velocity')
        if drag_k is None:
            k = np.full(m, self.initial_k)
        else:
            k = np.asarray(drag_k, dtype=float)
            k = np.full(m, float(k)) if k.ndim == 0 else k.reshape(-1)
            if k.size != m:
                raise BrainBoundaryError(f"drag_k must be scalar or ({m},), got {k.shape}")
        wind = np.zeros((m, 3))

        state = np.concatenate((pos, vel, k[:, None], wind), axis=1)
        if not self.enable_wind:
            state = state[:, :self.BASE_STATE_DIM]
        state = self._clip_state(state)
        if not np.isfinite(state).all():
            raise BrainBoundaryError("initialize received NaN/Inf")

        if covariance is None:
            cov = np.tile(self._unit_covariance(), (m, 1, 1))
        else:
            cov = np.asarray(covariance, dtype=float)
            if cov.ndim == 2:
                cov = np.tile(cov, (m, 1, 1))
            if cov.shape != (m, self.state_dim, self.state_dim):
                raise BrainBoundaryError(
                    f"covariance must have shape ({m}, {self.state_dim}, {self.state_dim}), "
                    f"got {cov.shape}")
            if not np.isfinite(cov).all():
                raise BrainBoundaryError("initialize received a non-finite covariance")
            cov = 0.5 * (cov + np.swapaxes(cov, -1, -2))

        self._state[index] = state
        self._covariance[index] = cov
        self._touch_timestamp(index, timestamp)

    def reset(self, env_ids: Optional[Sequence[int]] = None) -> None:
        """Restore the selected environments to the declared prior (S29: isolation)."""
        index = self._env_index(env_ids)
        self._state[index] = self._default_state[index]
        self._covariance[index] = self._default_covariance[index]
        if self._last_timestamp is not None:
            self._last_timestamp[index] = np.nan
        self.last_innovation = np.zeros((0, 3))
        self.last_nis = np.zeros((0,))

    # ------------------------------------------------------------------ filter steps
    def predict(self, dt_s: float, env_ids: Optional[Sequence[int]] = None) -> None:
        """Propagate the sigma points with shuttle_aerodynamics.rk4_step."""
        dt = float(dt_s)
        if not math.isfinite(dt) or dt <= 0.0:
            raise BrainBoundaryError(f"predict dt must be finite and > 0, got {dt_s!r}")
        index = self._env_index(env_ids)
        q = self._process_noise(dt)

        for e in index:
            points, wm, wc = self.sigma_points(self._state[e][None, :], self._covariance[e])
            propagated = np.stack([self._propagate(pt, dt) for pt in points[0]])
            mean = wm @ propagated
            dev = propagated - mean
            cov = np.einsum('i,ij,ik->jk', wc, dev, dev) + q
            self._state[e] = self._clip_state(mean)
            self._covariance[e] = 0.5 * (cov + cov.T)
            if not (np.isfinite(self._state[e]).all() and np.isfinite(self._covariance[e]).all()):
                raise FloatingPointError(
                    f"shuttle UKF produced a non-finite state in environment {e}")

    def update(self, measurement: ShuttleMeasurement,
               env_ids: Optional[Sequence[int]] = None) -> None:
        """Fuse one batched ShuttleMeasurement (position, and velocity when enabled)."""
        if not isinstance(measurement, ShuttleMeasurement):
            raise BrainBoundaryError(
                f"update expects a ShuttleMeasurement, got {type(measurement).__name__}")
        index = self._env_index(env_ids)
        m = index.size
        pos = self._block(measurement.position, m, 3, 'measurement.position')
        vel = self._block(measurement.velocity, m, 3, 'measurement.velocity')
        cov = np.asarray(measurement.covariance, dtype=float)
        if cov.shape != (m, 3, 3):
            raise BrainBoundaryError(
                f"measurement.covariance must have shape ({m}, 3, 3), got {cov.shape}")

        obs_dim = 6 if self.fuse_velocity else 3
        innovations = np.zeros((m, obs_dim))
        nis = np.zeros(m)
        for j, e in enumerate(index):
            r_pos = 0.5 * (cov[j] + cov[j].T) + _MEASUREMENT_JITTER * np.eye(3)
            if not np.isfinite(r_pos).all():
                raise BrainBoundaryError("measurement covariance contains NaN/Inf")
            if self.fuse_velocity:
                z = np.concatenate((pos[j], vel[j]))
                r = np.zeros((6, 6))
                r[:3, :3] = r_pos
                r[3:, 3:] = (self.velocity_measurement_std ** 2) * np.eye(3)
            else:
                z, r = pos[j], r_pos

            points, wm, wc = self.sigma_points(self._state[e][None, :], self._covariance[e])
            pts = points[0]
            if self.fuse_velocity:
                z_pts = np.concatenate((pts[:, POSITION_SLICE], pts[:, VELOCITY_SLICE]), axis=1)
            else:
                z_pts = pts[:, POSITION_SLICE]

            z_mean = wm @ z_pts
            dz = z_pts - z_mean
            dx = pts - self._state[e]
            s = np.einsum('i,ij,ik->jk', wc, dz, dz) + r
            p_xz = np.einsum('i,ij,ik->jk', wc, dx, dz)
            try:
                gain = np.linalg.solve(s.T, p_xz.T).T
            except np.linalg.LinAlgError:
                gain = p_xz @ np.linalg.pinv(s)
            innovation = z - z_mean
            new_state = self._state[e] + gain @ innovation
            new_cov = self._covariance[e] - gain @ s @ gain.T
            new_cov = 0.5 * (new_cov + new_cov.T)
            self._state[e] = self._clip_state(new_state)
            self._covariance[e] = new_cov
            if not (np.isfinite(self._state[e]).all() and np.isfinite(self._covariance[e]).all()):
                raise FloatingPointError(
                    f"shuttle UKF produced a non-finite state in environment {e}")

            innovations[j] = innovation
            try:
                nis[j] = float(innovation @ np.linalg.solve(s, innovation))
            except np.linalg.LinAlgError:
                nis[j] = float('nan')
        self.last_innovation = innovations
        self.last_nis = nis

    def step(self, measurement: ShuttleMeasurement, dt_s: Optional[float] = None) -> None:
        """predict(dt) + update(measurement); dt is derived from the message timestamps."""
        if dt_s is None:
            index = self._env_index(None)
            if self._last_timestamp is None or not np.isfinite(self._last_timestamp[index]).all():
                dt = 0.0
            else:
                dt = float(np.min(measurement.timestamp - self._last_timestamp[index]))
            if dt < 0.0:
                raise BrainBoundaryError(
                    f"measurement timestamp went backwards by {-dt:.6f} s")
            if dt > 0.0:
                self.predict(dt)
        else:
            self.predict(float(dt_s))
        self.update(measurement)
        self._touch_timestamp(np.arange(self.num_envs), float(measurement.timestamp))

    # ------------------------------------------------------------------ accessors
    @property
    def state(self) -> np.ndarray:
        return self._state.copy()

    @property
    def covariance(self) -> np.ndarray:
        return self._covariance.copy()

    @property
    def position(self) -> np.ndarray:
        return self._state[:, POSITION_SLICE].copy()

    @property
    def velocity(self) -> np.ndarray:
        return self._state[:, VELOCITY_SLICE].copy()

    @property
    def drag_k(self) -> np.ndarray:
        return self._state[:, DRAG_INDEX].copy()

    @property
    def wind(self) -> Optional[np.ndarray]:
        if not self.enable_wind:
            return None
        return self._state[:, WIND_SLICE].copy()

    # ------------------------------------------------------------------ internals
    def _propagate(self, sigma_point: np.ndarray, dt: float) -> np.ndarray:
        """One RK4 step of the frozen flight model (physics lives in shuttle_aerodynamics)."""
        k = float(sigma_point[DRAG_INDEX])
        k = min(max(k, self.drag_k_min), self.drag_k_max)
        wind = sigma_point[WIND_SLICE] if self.enable_wind else np.zeros(3)
        p_next, v_next = rk4_step(sigma_point[POSITION_SLICE], sigma_point[VELOCITY_SLICE], dt,
                                  k_per_m=k, gravity=self.gravity, wind=wind)
        out = sigma_point.copy()
        out[POSITION_SLICE] = p_next
        out[VELOCITY_SLICE] = v_next
        out[DRAG_INDEX] = k
        return out

    def _clip_state(self, state: np.ndarray) -> np.ndarray:
        """Project the drag state into its declared bounds (1-D mean or 2-D batch)."""
        out = np.array(state, dtype=float, copy=True)
        if out.ndim == 1:
            out[DRAG_INDEX] = min(max(out[DRAG_INDEX], self.drag_k_min), self.drag_k_max)
        else:
            np.clip(out[:, DRAG_INDEX], self.drag_k_min, self.drag_k_max,
                    out=out[:, DRAG_INDEX])
        return out

    def _process_noise(self, dt: float) -> np.ndarray:
        """White-noise-acceleration discretisation + random walks on k (and wind)."""
        n = self.state_dim
        q = np.zeros((n, n))
        qa = self.process_accel_std ** 2
        eye = np.eye(3)
        q[POSITION_SLICE, POSITION_SLICE] = qa * (dt ** 3) / 3.0 * eye
        q[VELOCITY_SLICE, VELOCITY_SLICE] = qa * dt * eye
        q[POSITION_SLICE, VELOCITY_SLICE] = qa * (dt ** 2) / 2.0 * eye
        q[VELOCITY_SLICE, POSITION_SLICE] = qa * (dt ** 2) / 2.0 * eye
        q[DRAG_INDEX, DRAG_INDEX] = (self.drag_process_std ** 2) * dt
        if self.enable_wind:
            q[WIND_SLICE, WIND_SLICE] = (self.wind_process_std ** 2) * dt * eye
        return q

    def _unit_covariance(self) -> np.ndarray:
        n = self.state_dim
        cov = np.zeros((n, n))
        cov[POSITION_SLICE, POSITION_SLICE] = (self.initial_position_std ** 2) * np.eye(3)
        cov[VELOCITY_SLICE, VELOCITY_SLICE] = (self.initial_velocity_std ** 2) * np.eye(3)
        cov[DRAG_INDEX, DRAG_INDEX] = self.initial_drag_std ** 2
        if self.enable_wind:
            cov[WIND_SLICE, WIND_SLICE] = (self.initial_wind_std ** 2) * np.eye(3)
        return cov

    def _touch_timestamp(self, index: np.ndarray, timestamp: Optional[float]) -> None:
        if timestamp is None:
            return
        value = float(timestamp)
        if not math.isfinite(value):
            raise BrainBoundaryError("timestamp must be finite")
        if self._last_timestamp is None:
            self._last_timestamp = np.full(self.num_envs, np.nan)
        self._last_timestamp[index] = value

    def _env_index(self, env_ids: Optional[Sequence[int]]) -> np.ndarray:
        if env_ids is None:
            return np.arange(self.num_envs)
        arr = np.asarray(env_ids)
        if arr.ndim != 1 or arr.size == 0:
            raise BrainBoundaryError(f"env_ids must be a non-empty 1-D sequence, got {env_ids!r}")
        if not np.issubdtype(arr.dtype, np.integer):
            if not np.all(arr == np.round(arr.astype(float))):
                raise BrainBoundaryError(f"env_ids must be integers, got {env_ids!r}")
        arr = arr.astype(int)
        if arr.min() < 0 or arr.max() >= self.num_envs:
            raise BrainBoundaryError(
                f"env_ids {arr.tolist()} out of range for {self.num_envs} environments")
        if np.unique(arr).size != arr.size:
            raise BrainBoundaryError(f"env_ids contains duplicates: {arr.tolist()}")
        return arr

    @staticmethod
    def _block(values, rows: int, cols: int, name: str) -> np.ndarray:
        arr = np.asarray(values, dtype=float)
        if arr.ndim == 1 and arr.size == cols and rows == 1:
            arr = arr[None, :]      # a single environment may be addressed with a bare vector
        if arr.shape != (rows, cols):
            raise BrainBoundaryError(f"{name} must have shape ({rows}, {cols}), got {arr.shape}")
        if not np.isfinite(arr).all():
            raise BrainBoundaryError(f"{name} contains NaN/Inf")
        return arr

    @staticmethod
    def _vector3(values, name: str) -> np.ndarray:
        arr = np.asarray(values, dtype=float).reshape(-1)
        if arr.shape != (3,) or not np.isfinite(arr).all():
            raise BrainBoundaryError(f"{name} must be a finite 3-vector, got {values!r}")
        return arr

    @staticmethod
    def _positive(value, name: str, allow_zero: bool = False) -> float:
        out = float(value)
        if not math.isfinite(out) or out < 0.0 or (out == 0.0 and not allow_zero):
            raise BrainBoundaryError(
                f"{name} must be finite and {'>= 0' if allow_zero else '> 0'}")
        return out
