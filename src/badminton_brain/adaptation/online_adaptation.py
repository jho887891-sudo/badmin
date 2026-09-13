# -*- coding: utf-8 -*-
"""T10 online adaptation: slow drag / wind / delay estimation from prediction error.

Layer: adaptation (runs after the baseline chain, ROBOT_BRAIN.md S11/S12/S15).
Spec: docs/architecture/ROBOT_BRAIN.md S7 (shuttle prediction adaptive loop: innovation ->
      correction of position / velocity / drag / wind) and S11 (slow online loop:
      drag / wind / delay small-parameter adaptation).

This is *not* online learning of a network.  It is a bounded, slow parameter adaptation:

    Feedback.prediction_error (predicted - measured, Court Frame)
        -> Gauss-Newton / normalized-LMS step on the drag / wind / delay estimates
        -> clipped per step and clipped into declared parameter bounds
        -> residual ledger (last K prediction errors per environment)

Conventions (fixed, because the frozen Feedback message carries no richer channel):
  * residual e = predicted - measured position, the sign convention of Feedback.prediction_error.
    The confidence path is the pushed prediction: the application layer calls set_prediction(
    trajectory) every step (same push-before-process idiom as SafetyContext) and this module
    compares the prediction pushed for the previous step against the state of the current step,
    both in absolute simulation time (PredictedTrajectory.times[0] == trajectory.timestamp).
    A pushed prediction is usable only when its timestamp is strictly older than the current state
    and its horizon still covers it; the newest usable one wins, so a push before or after
    process() both work.  Feedback.prediction_error, when present (a late real-rig measurement),
    always takes precedence over the pushed prediction;
  * the observation interval is the INTER-STEP interval dt = state.timestamp - previous state
    timestamp, maintained by this module.  In the pipeline Feedback.timestamp equals the
    UnifiedState timestamp (both are the step time, S43), so the same-step difference is identically
    zero and carries no information.  The first sample for an environment only stores the state
    (no interval yet) and a replay of the same step gives dt = 0: in both cases the residual is
    recorded but no parameter is updated;
  * error shapes (N, 3) or (N, 1) are used as-is; a (N,) or scalar error is read as a magnitude
    along the measured shuttle velocity direction (the only direction a scalar can mean);
  * the predicted position over dt is p + v*dt + 0.5*a*dt^2, so the sensitivity of the residual to
    a parameter is 0.5*dt^2 * d(acceleration)/d(parameter):
        a = g - k_total * |r| * r,   r = v - wind,   k_total = k_base * drag_scale
        d(a)/d(drag_scale) = -k_base * |r| * r
        d(a)/d(wind)       =  k_total * (outer(r_hat, r) + |r| * I)
        d(a)/d(delay)      =  0 in acceleration terms: delay is the latency of the state that
                              enters the predictor, so it stretches the rollout instead of changing
                              the acceleration.  The residual sensitivity is therefore +v (assuming
                              one extra second of latency rolls the prediction one v-second further
                              along the velocity), and the residual vanishes at the true latency.
    The update is theta += -gain * J^T e / (norm + eps) per parameter: a Gauss-Newton step on
    0.5 * ||e||^2, and since it is normalized it is free of arbitrary unit scaling.  The norm is
    ||J||^2 for a scalar parameter and the mean squared singular value ||J||^2/3 for the 3-vector
    wind parameter, so that the gain means the same "fraction of the parameter error corrected per
    step" for every parameter instead of silently becoming gain/3 for the vector one.
  * drag scale, wind and delay are only jointly identifiable when the shuttle velocity direction
    varies over samples; a single fixed velocity direction cannot separate drag from a wind
    component along the velocity (documented, not hidden).
"""
from __future__ import annotations

import math
from typing import Any, Dict, Optional, Sequence, Tuple

import numpy as np

from ..interfaces import AdaptationModule
from ..status import AssetStatus, Param, UNRESOLVED_STATUSES
from ..types import BrainBoundaryError, Feedback, Layer, PredictedTrajectory, UnifiedState

#: literature feather-shuttle aerodynamic length used by src/trajectory/shuttle_aerodynamics.py
#: (gravity does not appear in the sensitivities: only the drag / wind / delay terms of
#:  a = g - k|v - wind|(v - wind) depend on the estimated parameters)
AERODYNAMIC_LENGTH_M = 6.5


class OnlineAdaptation(AdaptationModule):
    """Slow-loop correction of the shuttle drag scale, wind and prediction delay."""

    layer = Layer.ADAPTATION
    name = 'online_adaptation'
    is_implemented = True
    output_type = dict

    PARAMETERS: Tuple[str, ...] = ('drag_scale', 'wind', 'delay')

    def __init__(self, num_envs: Optional[int] = None, *,
                 estimate: Sequence[str] = PARAMETERS,
                 aerodynamic_length_m: float = AERODYNAMIC_LENGTH_M,
                 ledger_size: int = 16,
                 drag_scale_bounds: Tuple[float, float] = (0.25, 4.0),
                 wind_bounds_mps: Tuple[float, float] = (-5.0, 5.0),
                 delay_bounds_s: Tuple[float, float] = (-0.05, 0.05),
                 gain_drag_scale: float = 0.4,
                 gain_wind: float = 0.4,
                 gain_delay: float = 0.4,
                 max_step_drag_scale: float = 0.1,
                 max_step_wind_mps: float = 0.2,
                 max_step_delay_s: float = 0.005,
                 eps: float = 1e-9) -> None:
        unknown = [name for name in estimate if name not in self.PARAMETERS]
        if unknown:
            raise BrainBoundaryError(
                "unknown adaptation parameter(s): %s (supported: %s)"
                % (unknown, list(self.PARAMETERS)))
        if num_envs is not None and int(num_envs) < 1:
            raise BrainBoundaryError("num_envs must be >= 1")
        if int(ledger_size) < 1:
            raise BrainBoundaryError("ledger_size must be >= 1")
        if not (math.isfinite(float(aerodynamic_length_m)) and float(aerodynamic_length_m) > 0.0):
            raise BrainBoundaryError("aerodynamic_length_m must be finite and > 0")

        self.estimate_names = tuple(estimate)
        self.ledger_size = int(ledger_size)
        self.eps = float(eps)
        self.num_envs = int(num_envs) if num_envs is not None else None
        self.k_per_m = 1.0 / float(aerodynamic_length_m)

        self.drag_scale_bounds = (float(drag_scale_bounds[0]), float(drag_scale_bounds[1]))
        self.wind_bounds_mps = (float(wind_bounds_mps[0]), float(wind_bounds_mps[1]))
        self.delay_bounds_s = (float(delay_bounds_s[0]), float(delay_bounds_s[1]))
        self.gain_drag_scale = float(gain_drag_scale)
        self.gain_wind = float(gain_wind)
        self.gain_delay = float(gain_delay)
        self.max_step_drag_scale = float(max_step_drag_scale)
        self.max_step_wind_mps = float(max_step_wind_mps)
        self.max_step_delay_s = float(max_step_delay_s)

        self.params: Dict[str, Param] = {
            'aerodynamic_length_m': Param(
                float(aerodynamic_length_m), AssetStatus.TRACEABLE_REFERENCE,
                "literature feather-shuttle aerodynamic length L = 6.5 m "
                "(src/trajectory/shuttle_aerodynamics.py)"),
            'gain_drag_scale': Param(self.gain_drag_scale, AssetStatus.TEMP_PARAMETERIZED_PROXY,
                                     "TEMP: normalized LMS gain, tuning knob, not measured"),
            'gain_wind': Param(self.gain_wind, AssetStatus.TEMP_PARAMETERIZED_PROXY,
                               "TEMP: normalized LMS gain, tuning knob, not measured"),
            'gain_delay': Param(self.gain_delay, AssetStatus.TEMP_PARAMETERIZED_PROXY,
                                "TEMP: normalized LMS gain, tuning knob, not measured"),
            'max_step_drag_scale': Param(self.max_step_drag_scale,
                                         AssetStatus.TEMP_PARAMETERIZED_PROXY,
                                         "TEMP: per-step increment cap, engineering choice"),
            'max_step_wind_mps': Param(self.max_step_wind_mps, AssetStatus.TEMP_PARAMETERIZED_PROXY,
                                       "TEMP: per-step increment cap, engineering choice"),
            'max_step_delay_s': Param(self.max_step_delay_s, AssetStatus.TEMP_PARAMETERIZED_PROXY,
                                      "TEMP: per-step increment cap, engineering choice"),
            'drag_scale_bounds': Param(self.drag_scale_bounds,
                                       AssetStatus.TEMP_PARAMETERIZED_PROXY,
                                       "TEMP: plausible drag-scale band; no hall measurement yet"),
            'wind_bounds_mps': Param(self.wind_bounds_mps, AssetStatus.TEMP_PARAMETERIZED_PROXY,
                                     "TEMP: indoor-hall draft band; requires hall measurement"),
            'delay_bounds_s': Param(self.delay_bounds_s, AssetStatus.TEMP_PARAMETERIZED_PROXY,
                                    "TEMP: vision/actuation latency band; requires measurement"),
            'ledger_size': Param(self.ledger_size, AssetStatus.TEMP_PARAMETERIZED_PROXY,
                                 "TEMP: engineering choice for the residual window"),
        }

        self.drag_scale = np.zeros(0)
        self.wind = np.zeros((0, 3))
        self.delay_s = np.zeros(0)
        self.updates = np.zeros(0, dtype=int)
        self.limited = np.zeros(0, dtype=bool)
        self.clamped = np.zeros(0, dtype=bool)
        self._ledger = np.zeros((0, self.ledger_size, 3))
        self._counts = np.zeros(0, dtype=int)
        self._next = np.zeros(0, dtype=int)
        self._prev_timestamp = np.zeros(0)
        self._prev_position = np.zeros((0, 3))
        self._prev_velocity = np.zeros((0, 3))
        self._prev_valid = np.zeros(0, dtype=bool)
        self._prediction_slots: list = []
        self._last_dt = np.zeros(0)
        self._last_source: list = []
        if self.num_envs is not None:
            self._allocate(self.num_envs)

    # ------------------------------------------------------------------ storage

    def _allocate(self, num_envs: int) -> None:
        self.num_envs = int(num_envs)
        n = self.num_envs
        self.drag_scale = np.ones(n)
        self.wind = np.zeros((n, 3))
        self.delay_s = np.zeros(n)
        self._drag_scale_initial = self.drag_scale.copy()
        self._wind_initial = self.wind.copy()
        self._delay_initial = self.delay_s.copy()
        self.updates = np.zeros(n, dtype=int)
        self.limited = np.zeros(n, dtype=bool)
        self.clamped = np.zeros(n, dtype=bool)
        self._ledger = np.full((n, self.ledger_size, 3), np.nan)
        self._counts = np.zeros(n, dtype=int)
        self._next = np.zeros(n, dtype=int)
        # previous state per environment: the slow loop owns its own interval measurement
        self._prev_timestamp = np.full(n, np.nan)
        self._prev_position = np.full((n, 3), np.nan)
        self._prev_velocity = np.full((n, 3), np.nan)
        self._prev_valid = np.zeros(n, dtype=bool)
        self._prediction_slots = []
        self._last_dt = np.full(n, np.nan)
        self._last_source = ['none'] * n

    def _ensure_envs(self, num_envs: int) -> None:
        if self.num_envs is None:
            self._allocate(num_envs)
        elif self.num_envs != num_envs:
            raise BrainBoundaryError(
                "OnlineAdaptation was built for %d environments but received %d"
                % (self.num_envs, num_envs))

    # ------------------------------------------------------------------ inputs

    def _normalize_error(self, raw: Any, velocity: np.ndarray,
                         num_envs: int) -> Optional[np.ndarray]:
        if raw is None:
            return None
        arr = np.asarray(raw, dtype=float)
        if arr.ndim == 0:
            arr = np.full(num_envs, float(arr))
        if arr.ndim == 1:
            if arr.shape[0] != num_envs:
                raise BrainBoundaryError(
                    "prediction_error batch %d does not match the state batch %d"
                    % (arr.shape[0], num_envs))
            speed = np.linalg.norm(velocity, axis=1)
            unit = np.zeros_like(velocity)
            moving = speed > self.eps
            unit[moving] = velocity[moving] / speed[moving][:, None]
            return arr[:, None] * unit
        if arr.ndim == 2:
            if arr.shape[0] != num_envs:
                raise BrainBoundaryError(
                    "prediction_error batch %d does not match the state batch %d"
                    % (arr.shape[0], num_envs))
            if arr.shape[1] == 3:
                return arr
            if arr.shape[1] == 1:
                return np.repeat(arr, 3, axis=1)
            raise BrainBoundaryError(
                "prediction_error must be (N, 3), (N, 1) or (N,), got shape %s" % (arr.shape,))
        raise BrainBoundaryError(
            "prediction_error must be (N, 3), (N, 1) or (N,), got shape %s" % (arr.shape,))

    def _contact_gate(self, contact_detected: Any, num_envs: int) -> np.ndarray:
        """A contact is a model discontinuity: skip the parameter update for that environment."""
        if contact_detected is None:
            return np.zeros(num_envs, dtype=bool)
        arr = np.asarray(contact_detected)
        if arr.ndim == 0:
            return np.full(num_envs, bool(arr))
        flat = arr.reshape(arr.shape[0], -1)
        if flat.shape[0] != num_envs:
            raise BrainBoundaryError(
                "contact_detected batch %d does not match the state batch %d"
                % (flat.shape[0], num_envs))
        return np.any(flat.astype(bool), axis=1)

    # ------------------------------------------------------- pushed prediction

    def set_prediction(self, trajectory) -> None:
        """Push the current PredictedTrajectory (call it before process, once per control step).

        Pass None to clear the cache.  The module keeps the last two pushes: the newest one that is
        strictly older than the current state and still covers it is the one used for the residual.
        """
        if trajectory is None:
            self._prediction_slots = []
            return None
        prepared = self._prepare_prediction(trajectory)
        self._prediction_slots.insert(0, prepared)
        del self._prediction_slots[2:]
        return None

    def _prepare_prediction(self, trajectory: PredictedTrajectory) -> Dict[str, Any]:
        if not isinstance(trajectory, PredictedTrajectory):
            raise BrainBoundaryError(
                "set_prediction expects a PredictedTrajectory, got %s" % type(trajectory).__name__)
        times = np.asarray(trajectory.times, dtype=float)
        positions = np.asarray(trajectory.position, dtype=float)
        num_envs = int(positions.shape[0])
        self._ensure_envs(num_envs)
        if times.ndim != 1 or times.shape[0] < 2:
            raise BrainBoundaryError(
                "a pushed prediction needs at least two time samples, got shape %s" % (times.shape,))
        if positions.ndim != 3 or positions.shape[1] != times.shape[0] or positions.shape[2] != 3:
            raise BrainBoundaryError(
                "pushed prediction position must be (N, T, 3) matching times, got %s"
                % (positions.shape,))
        if not np.all(np.diff(times) > 0.0):
            raise BrainBoundaryError("pushed prediction times must be strictly increasing")
        return {'times': times, 'positions': positions,
                'timestamp': float(trajectory.timestamp),
                'valid': np.ones(num_envs, dtype=bool)}

    def _prediction_candidates(self, inline) -> list:
        """Newest first: an inline prediction for this call, then the two pushed slots."""
        candidates = []
        if inline is not None:
            candidates.append(self._prepare_prediction(inline))
        candidates.extend(self._prediction_slots)
        return candidates

    def _prediction_residual(self, candidates, env: int, position: np.ndarray,
                             now: float):
        """predicted - measured position [m] from the newest usable prediction, else None."""
        for slot in candidates:
            if not bool(slot['valid'][env]):
                continue
            times = slot['times']
            if not (slot['timestamp'] < now <= float(times[-1]) + 1e-12):
                continue        # made at this instant (nothing to compare) or already past its horizon
            predicted = np.array([float(np.interp(now, times, slot['positions'][env, :, axis]))
                                  for axis in range(3)])
            return predicted - np.asarray(position, dtype=float)
        return None

    # ------------------------------------------------------------------ ledger

    def _push(self, env: int, error: np.ndarray) -> None:
        index = int(self._next[env])
        self._ledger[env, index] = error
        self._next[env] = (index + 1) % self.ledger_size
        self._counts[env] = min(int(self._counts[env]) + 1, self.ledger_size)

    def residual_ledger(self) -> np.ndarray:
        """Last K prediction errors per environment as (N, K, 3), oldest first.

        Unfilled slots are NaN.  The newest error is always in column K-1.
        """
        n = self.num_envs or 0
        out = np.full((n, self.ledger_size, 3), np.nan)
        for env in range(n):
            count = int(self._counts[env])
            if count == 0:
                continue
            if count < self.ledger_size:
                out[env, self.ledger_size - count:] = self._ledger[env, :count]
            else:
                out[env] = np.roll(self._ledger[env], -int(self._next[env]), axis=0)
        return out

    def residual_norms(self) -> np.ndarray:
        """Norm of every ledger entry as (N, K), NaN where the slot is unfilled."""
        return np.linalg.norm(self.residual_ledger(), axis=2)

    def residual_mean(self, window: Optional[int] = None) -> np.ndarray:
        """Mean residual norm per environment over the newest window entries (NaN if none)."""
        norms = self.residual_norms()
        if window is not None:
            if int(window) < 1:
                raise BrainBoundaryError("window must be >= 1")
            norms = norms[:, -int(window):]
        valid = np.isfinite(norms)
        counts = valid.sum(axis=1)
        total = np.where(valid, norms, 0.0).sum(axis=1)
        return np.where(counts > 0, total / np.maximum(counts, 1), np.nan)

    def residual_counts(self) -> np.ndarray:
        """Number of ledger entries currently held per environment (<= ledger_size)."""
        return self._counts.astype(int).copy()

    # ------------------------------------------------------------------ update

    def process(self, feedback: Feedback, state: UnifiedState, prediction=None) -> Any:
        if not isinstance(feedback, Feedback):
            raise BrainBoundaryError(
                "OnlineAdaptation.process expects a Feedback, got %s" % type(feedback).__name__)
        if not isinstance(state, UnifiedState):
            raise BrainBoundaryError(
                "OnlineAdaptation.process expects a UnifiedState, got %s" % type(state).__name__)

        velocity = np.asarray(state.shuttle_velocity, dtype=float)
        position = np.asarray(state.shuttle_position, dtype=float)
        num_envs = int(velocity.shape[0])
        self._ensure_envs(num_envs)
        now = float(state.timestamp)

        # dt is the interval since the state this module saw last: the pipeline stamps Feedback with
        # the same step time as the UnifiedState (S43), so the same-step difference is always zero.
        dt = np.full(num_envs, np.nan)
        usable = np.zeros(num_envs, dtype=bool)
        jump = np.full(num_envs, np.nan)
        for env in range(num_envs):
            if not self._prev_valid[env]:
                continue                    # first sample: no interval yet
            interval = now - float(self._prev_timestamp[env])
            dt[env] = interval
            usable[env] = math.isfinite(interval) and interval > 0.0
            jump[env] = float(np.linalg.norm(position[env] - self._prev_position[env]))

        errors = self._normalize_error(feedback.prediction_error, velocity, num_envs)
        blocked = self._contact_gate(feedback.contact_detected, num_envs)
        candidates = self._prediction_candidates(prediction)

        residual = np.full(num_envs, np.nan)
        updated = np.zeros(num_envs, dtype=bool)
        limited = np.zeros(num_envs, dtype=bool)
        clamped = np.zeros(num_envs, dtype=bool)
        source = ['none'] * num_envs

        for env in range(num_envs):
            error = None
            if errors is not None:
                # an explicit residual (a late real-rig measurement) always takes precedence; a
                # corrupt one is rejected rather than silently replaced by the pushed prediction
                source[env] = 'feedback'
                if np.all(np.isfinite(errors[env])):
                    error = errors[env]
            else:
                predicted = self._prediction_residual(candidates, env, position[env], now)
                if predicted is not None:
                    source[env] = 'prediction'
                    error = predicted
            if error is None or not np.all(np.isfinite(error)):
                continue                    # no usable residual: record nothing, update nothing
            residual[env] = float(np.linalg.norm(error))
            self._push(env, error)
            if not usable[env] or blocked[env]:
                continue
            limited[env], clamped[env] = self._update(env, error, velocity[env], float(dt[env]))
            self.updates[env] += 1
            updated[env] = True

        # snapshot the state for the next interval, whatever the residual quality was
        self._prev_timestamp[:] = now
        self._prev_position[:] = position
        self._prev_velocity[:] = velocity
        self._prev_valid[:] = True

        self.limited = limited
        self.clamped = clamped
        self._last_dt = dt.copy()
        self._last_source = list(source)
        return self._correction(dt=dt, residual=residual, updated=updated, limited=limited,
                                clamped=clamped, state_jump=jump, residual_source=source)

    def _update(self, env: int, error: np.ndarray, velocity: np.ndarray,
                dt: float) -> Tuple[bool, bool]:
        """One bounded Gauss-Newton step; returns (step_capped, bound_engaged)."""
        names = self.estimate_names
        relative = velocity - self.wind[env]
        speed = float(np.linalg.norm(relative))
        k_total = self.k_per_m * float(self.drag_scale[env])
        half_dt2 = 0.5 * dt * dt

        raw_drag = raw_delay = 0.0
        raw_wind = np.zeros(3)

        if 'drag_scale' in names:
            # d(a)/d(drag_scale) = -k_per_m * |r| * r  (k, not k * drag_scale: the scale is linear)
            gradient = (-half_dt2 * self.k_per_m * speed) * relative
            denom = float(gradient @ gradient) + self.eps
            raw_drag = -self.gain_drag_scale * float(gradient @ error) / denom
        if 'wind' in names:
            unit = relative / speed if speed > self.eps else np.zeros(3)
            accel_jac = k_total * (np.outer(unit, relative) + speed * np.eye(3))
            jac = half_dt2 * accel_jac
            # mean squared singular value (||J||_F^2 / 3): same gain semantics as a scalar parameter
            denom = float(np.sum(jac * jac)) / 3.0 + self.eps
            raw_wind = -self.gain_wind * (jac.T @ error) / denom
        if 'delay' in names:
            # latency stretches the rollout: d(residual)/d(delay) = +v (see the module docstring)
            gradient = velocity
            denom = float(gradient @ gradient) + self.eps
            raw_delay = -self.gain_delay * float(gradient @ error) / denom

        wind_norm = float(np.linalg.norm(raw_wind))
        limited = (abs(raw_drag) > self.max_step_drag_scale
                   or wind_norm > self.max_step_wind_mps
                   or abs(raw_delay) > self.max_step_delay_s)

        drag_step = float(np.clip(raw_drag, -self.max_step_drag_scale, self.max_step_drag_scale))
        delay_step = float(np.clip(raw_delay, -self.max_step_delay_s, self.max_step_delay_s))
        wind_step = raw_wind
        if wind_norm > self.max_step_wind_mps and wind_norm > 0.0:
            wind_step = raw_wind * (self.max_step_wind_mps / wind_norm)

        clamped = False
        if 'drag_scale' in names:
            proposed = float(self.drag_scale[env]) + drag_step
            bounded = float(np.clip(proposed, *self.drag_scale_bounds))
            clamped = clamped or bounded != proposed
            if math.isfinite(bounded):
                self.drag_scale[env] = bounded
        if 'wind' in names:
            proposed = self.wind[env] + wind_step
            bounded = np.clip(proposed, self.wind_bounds_mps[0], self.wind_bounds_mps[1])
            clamped = clamped or bool(np.any(bounded != proposed))
            if np.all(np.isfinite(bounded)):
                self.wind[env] = bounded
        if 'delay' in names:
            proposed = float(self.delay_s[env]) + delay_step
            bounded = float(np.clip(proposed, *self.delay_bounds_s))
            clamped = clamped or bounded != proposed
            if math.isfinite(bounded):
                self.delay_s[env] = bounded
        return limited, clamped

    # ------------------------------------------------------------------ output

    def _correction(self, *, dt: np.ndarray, residual: np.ndarray, updated: np.ndarray,
                    limited: np.ndarray, clamped: np.ndarray, state_jump: np.ndarray,
                    residual_source) -> Dict[str, Any]:
        return {
            'num_envs': int(self.num_envs or 0),
            'drag_scale': self.drag_scale.copy(),
            'drag_k_per_m': self.k_per_m * self.drag_scale.copy(),
            'wind': self.wind.copy(),
            'delay_s': self.delay_s.copy(),
            'dt_s': np.asarray(dt, dtype=float).copy(),
            'state_jump_m': np.asarray(state_jump, dtype=float),
            'residual': np.asarray(residual, dtype=float),
            'residual_source': np.asarray(list(residual_source), dtype=object),
            'residual_mean': self.residual_mean(),
            'updated': np.asarray(updated, dtype=bool),
            'limited': np.asarray(limited, dtype=bool),
            'clamped': np.asarray(clamped, dtype=bool),
            'updates': self.updates.astype(int).copy(),
            'parameters': dict(self.params),
        }

    def correction(self) -> Dict[str, Any]:
        """Current correction without consuming a feedback (read-only query)."""
        n = self.num_envs or 0
        return self._correction(dt=np.full(n, np.nan), residual=np.full(n, np.nan),
                                updated=np.zeros(n, dtype=bool),
                                limited=np.zeros(n, dtype=bool),
                                clamped=np.zeros(n, dtype=bool),
                                state_jump=np.full(n, np.nan),
                                residual_source=['none'] * n)

    def diagnostics(self) -> Dict[str, Any]:
        """Slow-loop state for the application layer (read-only, no side effects)."""
        n = self.num_envs or 0
        available = np.zeros(n, dtype=bool)
        timestamp = np.full(n, np.nan)
        horizon_end = np.full(n, np.nan)
        for slot in self._prediction_slots:
            fresh = slot['valid'] & ~available
            timestamp[fresh] = slot['timestamp']
            horizon_end[fresh] = float(slot['times'][-1])
            available |= slot['valid']
        age = np.full(n, np.nan)
        covered = self._prev_valid & available
        age[covered] = self._prev_timestamp[covered] - timestamp[covered]
        return {
            'num_envs': n,
            'prediction_available': available,
            'prediction_timestamp': timestamp,
            'prediction_horizon_end_s': horizon_end,
            'prediction_age_s': age,
            'prediction_slots': len(self._prediction_slots),
            'previous_state_valid': self._prev_valid.copy(),
            'previous_timestamp': self._prev_timestamp.copy(),
            'last_dt_s': self._last_dt.copy(),
            'last_residual_source': list(self._last_source) if self._last_source
                                    else ['none'] * n,
            'residual_counts': self.residual_counts(),
            'residual_mean': self.residual_mean(),
            'updates': self.updates.astype(int).copy(),
            'estimates': self.estimates(),
        }

    def estimates(self) -> Dict[str, Any]:
        """Copy of the current slow estimates (drag scale / wind / delay)."""
        return {'drag_scale': self.drag_scale.copy(), 'wind': self.wind.copy(),
                'delay_s': self.delay_s.copy(), 'k_per_m': self.k_per_m}

    # ------------------------------------------------------------------ reset

    def reset(self, env_ids) -> None:
        """Clear estimates, counters and the residual ledger of the selected environments only."""
        if self.num_envs is None or self.num_envs == 0:
            return None
        if env_ids is None:
            ids = np.arange(self.num_envs)
        else:
            ids = np.asarray(list(env_ids), dtype=int).reshape(-1)
        if ids.size and (ids.min() < 0 or ids.max() >= self.num_envs):
            raise BrainBoundaryError(
                "reset env_ids %s out of range for %d environments"
                % (ids.tolist(), self.num_envs))
        for env in ids.tolist():
            self.drag_scale[env] = self._drag_scale_initial[env]
            self.wind[env] = self._wind_initial[env]
            self.delay_s[env] = self._delay_initial[env]
            self.updates[env] = 0
            self.limited[env] = False
            self.clamped[env] = False
            self._ledger[env] = np.nan
            self._counts[env] = 0
            self._next[env] = 0
            self._prev_timestamp[env] = np.nan
            self._prev_position[env] = np.nan
            self._prev_velocity[env] = np.nan
            self._prev_valid[env] = False
            for slot in self._prediction_slots:
                slot['valid'][env] = False
        self._prediction_slots = [slot for slot in self._prediction_slots
                                  if bool(np.any(slot['valid']))]
        return None

    def previous_state(self) -> Dict[str, np.ndarray]:
        """Last state seen per environment (interval bookkeeping and episode diagnostics)."""
        return {'timestamp': self._prev_timestamp.copy(),
                'position': self._prev_position.copy(),
                'velocity': self._prev_velocity.copy(),
                'valid': self._prev_valid.copy()}

    def measurement_requirements(self) -> Dict[str, Param]:
        """One Param(None, REQUIRES_MEASUREMENT, source) per still-unresolved quantity (S4/S12).

        Same API as the T5/T6/T7 modules: everything that must be measured before the slow loop can
        be trusted on the real rig.  The literature aerodynamic length is resolved and is therefore
        not listed.
        """
        return {name: Param(None, AssetStatus.REQUIRES_MEASUREMENT,
                            'measure ' + name + ': ' + param.source)
                for name, param in self.params.items()
                if param.status in UNRESOLVED_STATUSES}


__all__ = ["OnlineAdaptation", "AERODYNAMIC_LENGTH_M"]
