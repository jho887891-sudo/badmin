# -*- coding: utf-8 -*-
"""T5 hit feasibility gate: Global Feasibility Gate of the decision layer.

Spec: docs/architecture/04_HIT_DECISION.md S16 ("Gate only does cheap, deterministic,
explainable global screening") + plan row T5 (docs/superpowers/plans/2026-09-13-brain-modules.md).

Scope: this module answers exactly one question - "is this incoming shuttle worth playing at
all?" - and returns types.HitDecision(feasible, reason, timestamp).  It never searches for an
intercept, never generates motion and never touches Safety: intercept_search.py (T6) wraps this
gate in the DecisionModule that returns (HitDecision, BestIntercept | None).

Criteria, each independently testable (order = evaluation order):

 1. input validity / freshness  INVALID_STATE, INVALID_PREDICTION, STALE_STATE, STALE_PREDICTION
 2. shuttle speed window        SHUTTLE_TOO_FAST, SHUTTLE_TOO_SLOW
 3. approach direction          WRONG_DIRECTION            (S19: median(vx) over a short window)
 4. landing inside the court    OUT_OF_BOUNDS            (configs/court.yaml geometry)
 5. landing on our half         OUTSIDE_RESPONSIBILITY
 6. flight already over         NO_TIME_MARGIN
 7. workspace box + arrival     UNREACHABLE, NO_TIME_MARGIN, NO_LANDING_IN_HORIZON

Criteria 4 and 5 need a *landing*, not a predicted end point.  A predictor may clip its horizon:
the T4 predictor then reports the last sample projected onto the ground and sets the contract flag
landed_within_horizon = False (DEC-019).  The gate believes a landing only when that flag is true,
or - flag absent - when the samples themselves reach the ground plane z = 0; otherwise the landing
gates are suspended and the verdict comes from the samples alone (review finding D2: a ball at
z = 1.16 m still on the opponent side was reported as "lands on the opponent side").

Input validity is re-checked here, not trusted from the message constructors (S17): types.py only
validates when a message is built, and messages stay mutable, so EVERY array the gate reads
(state and prediction) must be finite.  A NaN would otherwise make every comparison below False
and the gate would report a bogus FEASIBLE (review finding C1).

Reason stability: each rejection is one of the fixed HitReason members; INVALID_PREDICTION covers
mismatched/empty/non-finite prediction samples and INVALID_STATE the same for the unified state.

Frames and geometry: everything is Court Frame (COORDINATE_SYSTEM.md S3): origin at the net
centre on the ground, +X toward the opponent, +Y robot-left, +Z up.  Court bounds come from
configs/court.yaml (13.40 x 6.10, singles 5.18) and are Params with status TRACEABLE_REFERENCE -
nothing is invented here.

Reachability model (cheap, no IK): the racket workspace is an axis-aligned box around the base
position (arm_reach_*_m, arm_z_*_m, base yaw ignored - TEMP simplification).  The base may
station anywhere inside station_*_range_m, so a shuttle point is geometrically reachable when
some legal base placement puts it in that box; the required arrival time is then
decision_latency + base minimum travel time (triangle/trapezoid profile, S62-S64 of
04_HIT_DECISION.md) + arm slew lower bound + safety margin.  The gate accepts when at least one
predicted sample is both inside the reachable volume and reachable in time.

Single source of truth (review ruling C3): the base minimum-travel-time model is implemented once
in decision/travel_model.py and imported here (base_min_travel_time_s / min_travel_times_s) and
by intercept_search.py - neither decision module may carry a private copy again.

Known unknowns (plan rule 3): every kinematic / timing threshold is
AssetStatus.TEMP_PARAMETERIZED_PROXY with its source, and measurement_requirements() returns the
matching Param(None, REQUIRES_MEASUREMENT, source=...) for each of them.  Court geometry is
TRACEABLE_REFERENCE to configs/court.yaml.  No value here is a measurement.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field, fields
from enum import Enum
from typing import Dict, Optional, Tuple

import numpy as np

from ..status import UNRESOLVED_STATUSES, AssetStatus, Param
from ..types import HitDecision, PredictedTrajectory, UnifiedState
# Review C3: the base minimum-travel-time model lives in decision/travel_model.py only; the names
# below are re-exports kept for API compatibility (base_min_travel_time_s is in __all__).
from .travel_model import min_travel_time_s as base_min_travel_time_s
from .travel_model import min_travel_times_s
from .travel_model import min_travel_times_s as _base_travel_times

_TEMP = AssetStatus.TEMP_PARAMETERIZED_PROXY
_TRACE = AssetStatus.TRACEABLE_REFERENCE

# Numerical guard only - NOT a physical threshold: the court ground plane is z = 0 in the Court
# Frame (COORDINATE_SYSTEM.md), so this only absorbs interpolation round-off.
_GROUND_EPS_M = 1e-6


class HitReason(str, Enum):
    """Stable reason vocabulary (04_HIT_DECISION.md S9/S14); the value string is the contract.

    Names come from the document's reason-code list.  SHUTTLE_TOO_SLOW is the symmetric case of
    the documented SHUTTLE_TOO_FAST and is required by plan row T5 (speed upper *and* lower
    bound); it is the only addition.
    FEASIBLE is the gate-level accept code (S150 uses FEASIBLE/INFEASIBLE for gate outcomes);
    T6 replaces it with BEST_INTERCEPT_FOUND once a real intercept is selected.
    NO_LANDING_IN_HORIZON is the other addition: it reports the doc's WAIT situation (S10 -
    information insufficient, a better prediction may still arrive).  It is returned when the
    prediction is still airborne at the end of its horizon, no sample is inside the racket box
    yet, and the landing is therefore unknown (review D2, DEC-019 flag landed_within_horizon).
    """
    FEASIBLE = 'FEASIBLE'
    INVALID_STATE = 'INVALID_STATE'
    INVALID_PREDICTION = 'INVALID_PREDICTION'
    STALE_STATE = 'STALE_STATE'
    STALE_PREDICTION = 'STALE_PREDICTION'
    SHUTTLE_TOO_FAST = 'SHUTTLE_TOO_FAST'
    SHUTTLE_TOO_SLOW = 'SHUTTLE_TOO_SLOW'
    WRONG_DIRECTION = 'WRONG_DIRECTION'
    OUT_OF_BOUNDS = 'OUT_OF_BOUNDS'
    OUTSIDE_RESPONSIBILITY = 'OUTSIDE_RESPONSIBILITY'
    UNREACHABLE = 'UNREACHABLE'
    NO_TIME_MARGIN = 'NO_TIME_MARGIN'
    NO_LANDING_IN_HORIZON = 'NO_LANDING_IN_HORIZON'


def _temp(value, source: str) -> Param:
    """TEMP proxy value: explicitly marked and sourced, never presented as measured."""
    return Param(value, _TEMP, source)


@dataclass
class FeasibilityLimits:
    """All gate thresholds, configurable and self-describing (plan rule 3).

    Values marked TEMP are placeholders that make the gate runnable and testable; the real
    numbers must come from measurement (see measurement_requirements()).  Court geometry is
    traceable to configs/court.yaml / COORDINATE_SYSTEM.md.
    """
    # --- shuttle state envelope -------------------------------------------------
    shuttle_speed_min_mps: Param = field(default_factory=lambda: _temp(
        0.5, 'TEMP proxy: below this the shuttle is effectively at rest and cannot be played; '
             'replace with the measured playable incoming-speed floor'))
    shuttle_speed_max_mps: Param = field(default_factory=lambda: _temp(
        25.0, 'TEMP proxy: playable incoming speed ceiling of the slow-speed rig; replace with the '
              'measured racket-exit/incoming envelope (04_HIT_DECISION.md S7 RobotCapabilityModel)'))
    vx_min_approach_mps: Param = field(default_factory=lambda: _temp(
        0.0, 'TEMP proxy: the short-window median(vx) must stay below -this value (approaching); '
             '0.0 means strictly incoming, a positive value would demand a minimum approach speed '
             'once the measured shuttle-velocity noise is known'))
    direction_window_s: Param = field(default_factory=lambda: _temp(
        0.10, 'TEMP proxy: length of the short future window used by the median(vx) approach test '
              '(04_HIT_DECISION.md S19); replace with the measured prediction reliability horizon'))
    # --- freshness / latency budget --------------------------------------------
    max_state_age_s: Param = field(default_factory=lambda: _temp(
        0.05, 'TEMP proxy: estimation freshness budget placeholder; replace with the measured '
              'UnifiedState latency budget (04_HIT_DECISION.md S18)'))
    max_prediction_age_s: Param = field(default_factory=lambda: _temp(
        0.05, 'TEMP proxy: prediction freshness budget placeholder; replace with the measured '
              'PredictedTrajectory latency budget (04_HIT_DECISION.md S18)'))
    decision_latency_s: Param = field(default_factory=lambda: _temp(
        0.02, 'TEMP proxy: decision -> command latency budget placeholder; replace with the '
              'measured brain-cycle latency'))
    arm_slew_time_s: Param = field(default_factory=lambda: _temp(
        0.15, 'TEMP proxy: PiPER slew-to-strike time lower bound placeholder; replace with the '
              'measured minimum-time arm motion'))
    safety_time_margin_s: Param = field(default_factory=lambda: _temp(
        0.05, 'TEMP proxy: robustness margin added on top of the arrival-time lower bound'))
    # --- base motion limits (triangle/trapezoid profile inputs) -----------------
    base_v_max_mps: Param = field(default_factory=lambda: _temp(
        1.0, 'TEMP proxy: Morph One (4 steer-drive modules) linear speed limit placeholder; the '
             'authoritative numbers must come from the measured base limits / morph_one kinematics'))
    base_a_max_mps2: Param = field(default_factory=lambda: _temp(
        1.5, 'TEMP proxy: Morph One linear acceleration limit placeholder; replace with the '
             'measured base acceleration limit'))
    # --- racket workspace box (axis-aligned about the base, base yaw ignored) ---
    arm_reach_x_m: Param = field(default_factory=lambda: _temp(
        0.60, 'TEMP proxy: racket workspace half extent in X placeholder; replace with the '
              'measured PiPER + racket TCP workspace (COORDINATE_SYSTEM.md racket contact)'))
    arm_reach_y_m: Param = field(default_factory=lambda: _temp(
        0.60, 'TEMP proxy: racket workspace half extent in Y placeholder; replace with the '
              'measured PiPER + racket TCP workspace'))
    arm_z_min_m: Param = field(default_factory=lambda: _temp(
        0.15, 'TEMP proxy: lowest racket contact height; derived from COORDINATE_SYSTEM.md '
              '(PiPER mount at z=0.30) and must be replaced by the measured workspace'))
    arm_z_max_m: Param = field(default_factory=lambda: _temp(
        1.40, 'TEMP proxy: highest racket contact height; derived from COORDINATE_SYSTEM.md '
              '(mount z=0.30) + placeholder reach 0.60 + racket 0.50, must be measured'))
    # --- legal base station area (Court Frame) ---------------------------------
    station_x_range_m: Param = field(default_factory=lambda: _temp(
        (-6.40, -0.30), 'TEMP proxy: legal base station X range = robot half court inset '
                        '(20 cm from the baseline, 30 cm from the net) - inset must be measured'))
    station_y_range_m: Param = field(default_factory=lambda: _temp(
        (-2.75, 2.75), 'TEMP proxy: legal base station Y range = doubles court inset 30 cm - '
                       'inset must be measured'))
    # --- court geometry (authoritative, not invented) ---------------------------
    court_half_length_m: Param = field(default_factory=lambda: Param(
        6.70, _TRACE, 'configs/court.yaml court.length_m 13.40 / 2 (COORDINATE_SYSTEM.md: X in [-6.70, 6.70])'))
    court_half_width_doubles_m: Param = field(default_factory=lambda: Param(
        3.05, _TRACE, 'configs/court.yaml court.width_m 6.10 / 2 (COORDINATE_SYSTEM.md: Y in [-3.05, 3.05])'))
    court_half_width_singles_m: Param = field(default_factory=lambda: Param(
        2.59, _TRACE, 'configs/court.yaml court.singles_width_m 5.18 / 2'))
    net_x_m: Param = field(default_factory=lambda: Param(
        0.0, _TRACE, 'COORDINATE_SYSTEM.md: net plane at X=0, origin at the net centre'))
    responsibility_x_margin_m: Param = field(default_factory=lambda: _temp(
        0.0, 'TEMP proxy: the shuttle must land on the robot side of the net (x <= net + margin)'))
    boundary_mode: str = 'doubles'

    # ------------------------------------------------------------------ helpers
    def param_limits(self) -> Dict[str, Param]:
        """Every threshold field, by name (used by tests and by final-mode reporting)."""
        return {f.name: getattr(self, f.name) for f in fields(self)
                if isinstance(getattr(self, f.name), Param)}

    def unresolved_limits(self) -> Tuple[str, ...]:
        """Fields that are not backed by a real measurement (TEMP/unknown) or have no value."""
        return tuple(sorted(name for name, p in self.param_limits().items()
                            if p.status in UNRESOLVED_STATUSES or p.value is None))

    def measurement_requirements(self) -> Dict[str, Param]:
        """One Param(None, REQUIRES_MEASUREMENT, source) per still-unresolved threshold."""
        return {name: Param(None, AssetStatus.REQUIRES_MEASUREMENT,
                            'measure ' + name + ': ' + p.source)
                for name, p in self.param_limits().items()
                if p.status in UNRESOLVED_STATUSES}

    def court_half_width_m(self) -> float:
        if self.boundary_mode == 'doubles':
            return float(self.court_half_width_doubles_m.value)
        if self.boundary_mode == 'singles':
            return float(self.court_half_width_singles_m.value)
        raise ValueError("boundary_mode must be 'doubles' or 'singles', got %r" % (self.boundary_mode,))


def _value(param: Param, name: str) -> float:
    """Read a limit; an unknown limit must never silently decide (plan rule 3)."""
    if param.value is None:
        raise ValueError('limit %r is unresolved (%s): %s' % (name, param.status.value, param.source))
    return float(param.value)


def _range(param: Param, name: str) -> Tuple[float, float]:
    if param.value is None:
        raise ValueError('limit %r is unresolved (%s): %s' % (name, param.status.value, param.source))
    low, high = param.value
    return float(low), float(high)


class HitFeasibilityGate:
    """Stateless, deterministic global feasibility gate -> types.HitDecision.

    evaluate() decides one environment (HitDecision is a per-environment message, so the
    decision layer emits one decision per step); evaluate_batch() maps over the envs.
    """

    def __init__(self, limits: Optional[FeasibilityLimits] = None) -> None:
        self.limits = limits if limits is not None else FeasibilityLimits()

    # ------------------------------------------------------------------ public API
    def evaluate(self, state, trajectory, env_id: int = 0,
                 now: Optional[float] = None) -> HitDecision:
        """Decide whether the incoming shuttle is playable.

        state       UnifiedState (estimation output, Court Frame)
        trajectory  PredictedTrajectory (prediction output, absolute times)
        env_id      which batch element to decide
        now         simulation time used for both the freshness ages (S18) and the arrival-time
                    budgets.  When omitted the gate degrades to
                    max(state.timestamp, trajectory.timestamp) - the newest message timestamp -
                    so a freshly published pair is never rejected as stale; a caller that owns a
                    real clock (the decision adapter's set_now/now_provider) must pass it,
                    otherwise staleness cannot be detected at all.
        """
        limits = self.limits
        if not isinstance(state, UnifiedState):
            return self._reject(HitReason.INVALID_STATE, 0.0)
        if not isinstance(trajectory, PredictedTrajectory):
            return self._reject(HitReason.INVALID_PREDICTION, float(state.timestamp))

        t_now = self._now(state, trajectory, now)
        if not math.isfinite(t_now):
            return self._reject(HitReason.INVALID_STATE, 0.0)

        reason = self._input_reason(state, trajectory, env_id)
        if reason is not None:
            return self._reject(reason, t_now)

        # 1. freshness (04_HIT_DECISION.md S18)
        if t_now - float(state.timestamp) > _value(limits.max_state_age_s, 'max_state_age_s'):
            return self._reject(HitReason.STALE_STATE, t_now)
        if t_now - float(trajectory.timestamp) > _value(limits.max_prediction_age_s,
                                                        'max_prediction_age_s'):
            return self._reject(HitReason.STALE_PREDICTION, t_now)

        # 2./3. incoming shuttle speed window and approach direction
        # The incoming speed is the shuttle speed at the start of the predicted window; the
        # peak over the window would only measure the free-fall acceleration near the ground.
        velocity = np.asarray(trajectory.velocity[env_id], dtype=float)
        incoming_speed = float(np.linalg.norm(velocity[0]))
        if not math.isfinite(incoming_speed):          # defence in depth behind _input_reason
            return self._reject(HitReason.INVALID_PREDICTION, t_now)
        if incoming_speed > _value(limits.shuttle_speed_max_mps, 'shuttle_speed_max_mps'):
            return self._reject(HitReason.SHUTTLE_TOO_FAST, t_now)
        if incoming_speed < _value(limits.shuttle_speed_min_mps, 'shuttle_speed_min_mps'):
            return self._reject(HitReason.SHUTTLE_TOO_SLOW, t_now)
        # S19: the approach verdict uses median(vx) over a short *future* window, so a single
        # noisy prediction sample cannot flip the decision.  The window starts at the first
        # predicted sample; a non-positive window degrades to that first sample.
        times = np.asarray(trajectory.times, dtype=float)
        window_end = float(times[0]) + max(_value(limits.direction_window_s,
                                                  'direction_window_s'), 0.0)
        window = velocity[times <= window_end]
        if window.shape[0] == 0:
            window = velocity[:1]
        median_vx = float(np.median(window[:, 0]))
        if median_vx >= -_value(limits.vx_min_approach_mps, 'vx_min_approach_mps'):
            return self._reject(HitReason.WRONG_DIRECTION, t_now)

        # 4./5. landing validity in the Court Frame (S23 Landing Gate) - but ONLY when the
        # prediction actually contains a landing.  A predictor with a clipped horizon reports the
        # last sample projected onto the ground together with landed_within_horizon = False
        # (DEC-019); judging that projection rejected balls that were still in the air (review D2).
        positions = np.asarray(trajectory.position[env_id], dtype=float)
        landing_known = self._landing_is_known(trajectory, env_id, positions)
        if landing_known:
            landing = np.asarray(trajectory.landing_point[env_id], dtype=float)
            half_length = _value(limits.court_half_length_m, 'court_half_length_m')
            half_width = limits.court_half_width_m()
            if abs(float(landing[0])) > half_length or abs(float(landing[1])) > half_width:
                return self._reject(HitReason.OUT_OF_BOUNDS, t_now)
            net_x = _value(limits.net_x_m, 'net_x_m')
            if float(landing[0]) > net_x + _value(limits.responsibility_x_margin_m,
                                                  'responsibility_x_margin_m'):
                return self._reject(HitReason.OUTSIDE_RESPONSIBILITY, t_now)

        # 6. the predicted flight must not be over already
        if float(trajectory.arrival_time[env_id]) <= t_now:
            return self._reject(HitReason.NO_TIME_MARGIN, t_now)

        # 7. workspace box + arrival-time lower bound (base travel + arm slew + latency)
        reason = self._workspace_and_time_reason(state, trajectory, env_id, t_now, landing_known)
        if reason is not None:
            return self._reject(reason, t_now)
        return HitDecision(True, HitReason.FEASIBLE.value, t_now)

    def evaluate_batch(self, state, trajectory, now: Optional[float] = None) -> Tuple[HitDecision, ...]:
        """One HitDecision per environment (batch element), in env order.

        The now argument has exactly the semantics of evaluate() and is forwarded unchanged to
        every environment; when omitted, every environment uses the same per-message fallback
        (max(state.timestamp, trajectory.timestamp)).
        """
        if not isinstance(state, UnifiedState) or state.shuttle_position is None:
            return (self.evaluate(state, trajectory, 0, now),)
        count = int(np.asarray(state.shuttle_position).shape[0])
        return tuple(self.evaluate(state, trajectory, env_id=i, now=now) for i in range(count))

    # ------------------------------------------------------------------ internals
    @staticmethod
    def _reject(reason: HitReason, timestamp: float) -> HitDecision:
        return HitDecision(False, reason.value, timestamp)

    @staticmethod
    def _now(state: UnifiedState, trajectory: PredictedTrajectory,
             now: Optional[float]) -> float:
        if now is not None:
            return float(now)
        return max(float(state.timestamp), float(trajectory.timestamp))

    def _input_reason(self, state: UnifiedState, trajectory: PredictedTrajectory,
                      env_id: int) -> Optional[HitReason]:
        arrays = (state.shuttle_position, state.shuttle_velocity, state.base_pose)
        if any(a is None for a in arrays) or any(not np.all(np.isfinite(np.asarray(a, dtype=float)))
                                                 for a in arrays):
            return HitReason.INVALID_STATE
        if not 0 <= env_id < int(np.asarray(state.shuttle_position).shape[0]):
            return HitReason.INVALID_STATE

        # Sample arrays get exactly the same judgement as the state arrays above: types.py only
        # checks finiteness when a message is *built*, and messages are mutable, so a poisoned
        # array must be caught here or every NaN comparison below would silently be False.
        samples = (trajectory.position, trajectory.velocity, trajectory.landing_point,
                   trajectory.arrival_time, trajectory.times)
        if any(a is None for a in samples) or any(
                not np.all(np.isfinite(np.asarray(a, dtype=float))) for a in samples):
            return HitReason.INVALID_PREDICTION
        horizon = int(np.asarray(trajectory.position).shape[1])
        if horizon < 1 or int(np.asarray(trajectory.times).shape[0]) != horizon:
            return HitReason.INVALID_PREDICTION
        if env_id >= int(np.asarray(trajectory.position).shape[0]):
            return HitReason.INVALID_PREDICTION
        return None

    def _landing_is_known(self, trajectory: PredictedTrajectory, env_id: int,
                          positions: np.ndarray) -> bool:
        """True when landing_point is a real landing, not a horizon-clipped projection.

        The contract flag landed_within_horizon (DEC-019) is authoritative when present and
        finite; when it is absent (None) or unusable, the samples decide: a landing exists only
        if the trajectory actually reaches the court ground plane z = 0.
        """
        flag = getattr(trajectory, 'landed_within_horizon', None)
        if flag is not None:
            try:
                values = np.asarray(flag, dtype=float).reshape(-1)
            except (TypeError, ValueError):
                values = np.empty(0, dtype=float)
            if values.shape[0] > env_id and math.isfinite(float(values[env_id])):
                return bool(values[env_id] > 0.0)
        return bool(float(positions[-1, 2]) <= _GROUND_EPS_M)

    def _workspace_and_time_reason(self, state: UnifiedState, trajectory: PredictedTrajectory,
                                   env_id: int, t_now: float,
                                   landing_known: bool = True) -> Optional[HitReason]:
        limits = self.limits
        positions = np.asarray(trajectory.position[env_id], dtype=float)
        times = np.asarray(trajectory.times, dtype=float)
        base = np.asarray(state.base_pose[env_id][:3], dtype=float)

        reach_x = _value(limits.arm_reach_x_m, 'arm_reach_x_m')
        reach_y = _value(limits.arm_reach_y_m, 'arm_reach_y_m')
        z_min = _value(limits.arm_z_min_m, 'arm_z_min_m')
        z_max = _value(limits.arm_z_max_m, 'arm_z_max_m')
        (station_x0, station_x1) = _range(limits.station_x_range_m, 'station_x_range_m')
        (station_y0, station_y1) = _range(limits.station_y_range_m, 'station_y_range_m')
        net_x = _value(limits.net_x_m, 'net_x_m')

        # base placements that put this sample inside the racket box
        low_x = np.maximum(positions[:, 0] - reach_x, station_x0)
        high_x = np.minimum(positions[:, 0] + reach_x, station_x1)
        low_y = np.maximum(positions[:, 1] - reach_y, station_y0)
        high_y = np.minimum(positions[:, 1] + reach_y, station_y1)
        in_box = ((low_x <= high_x) & (low_y <= high_y)
                  & (positions[:, 2] >= z_min) & (positions[:, 2] <= z_max)
                  & (positions[:, 0] <= net_x))   # the racket must not cross the net plane
        if not bool(np.any(in_box)):
            # Nothing in the racket box.  Without a landing the ball may still drop into reach
            # later, so the honest verdict is "not known yet" (WAIT), never a confident
            # UNREACHABLE built on a horizon-clipped projection (review D2).
            return HitReason.UNREACHABLE if landing_known else HitReason.NO_LANDING_IN_HORIZON

        # cheapest legal base placement: clamp the current base into the valid set
        target_x = np.clip(base[0], low_x, high_x)
        target_y = np.clip(base[1], low_y, high_y)
        distance = np.hypot(target_x - base[0], target_y - base[1])
        travel = _base_travel_times(distance,
                                    _value(limits.base_v_max_mps, 'base_v_max_mps'),
                                    _value(limits.base_a_max_mps2, 'base_a_max_mps2'))
        required = (travel
                    + _value(limits.decision_latency_s, 'decision_latency_s')
                    + _value(limits.arm_slew_time_s, 'arm_slew_time_s')
                    + _value(limits.safety_time_margin_s, 'safety_time_margin_s'))
        if bool(np.any(in_box & (times - t_now >= required))):
            return None
        return HitReason.NO_TIME_MARGIN


__all__ = ["FeasibilityLimits", "HitFeasibilityGate", "HitReason", "base_min_travel_time_s",
           "min_travel_times_s"]
