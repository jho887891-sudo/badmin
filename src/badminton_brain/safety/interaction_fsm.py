# -*- coding: utf-8 -*-
"""P7-10: the six-state safety FSM with a latching HOLD.

STATES AND THE ONLY LEGAL MOVES
-------------------------------
    SEARCH  --frame-->            TRACK
    TRACK   --prediction-->       PREDICT
    PREDICT --plan-->             PLAN
    PLAN    --command-->          EXECUTE
    EXECUTE --prediction-->       PREDICT   (next frame, re-plan)
    EXECUTE --feedback/cycle-->   PLAN      (iterate on the same plan)
    SEARCH/TRACK/PREDICT/PLAN/EXECUTE --anything invalid--> HOLD
    HOLD    --explicit reset with a FRESH valid frame+prediction--> TRACK
Every other move is a protocol error and is handled the same way as any other
failure: the FSM latches HOLD with a typed reason.

FAIL-CLOSED, LITERALLY
----------------------
Each public stage runs inside a guard.  Whatever happens - a stale frame, a
covariance above threshold, a reversed time-to-hit, an unreachable target, a
joint-limit margin violation, a command above the P5-05 caps, or an EXCEPTION
thrown anywhere in the stage - the transition goes to HOLD and a typed reason is
appended to the event log.  A stage never returns a command on failure.

HOLD LATCHES
------------
Entering HOLD clears the pending command, and NO command can be emitted while in
HOLD: emit() is only reachable from PLAN/EXECUTE, and leaving HOLD requires
reset() with a freshly validated frame AND prediction.  A stale or invalid reset
is refused (ResetRejected) and the latch stays.  There is no timer, no "auto
recovery" and no implicit path back to EXECUTE.

2-D IS NEVER PROMOTED
---------------------
FrameInput refuses to hold a 3-D position when detection_dim == 2, and
prediction/frames go through reachability.require_world_3d.  A 2-D frame that
reaches the prediction stage latches HOLD with DETECTION_NOT_3D; it does not
invent a depth.

UNCERTAINTY AND TIME ARE CARRIED, NOT RE-DERIVED
------------------------------------------------
The prediction's sigma and time-to-hit enter the FSM and are PROPAGATED stage by
stage (sigma grows at the declared rate, time-to-hit shrinks with the clock).
Every event records both, so the decision can be replayed.  A command is only
emitted when the PROPAGATED sigma is still under threshold and the PROPAGATED
time-to-hit is still positive.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

from ..timing.latency_budget import MAX_SOURCE_FRAME_AGE_S
from ..reachability.ik import IkSolver
from ..reachability.reachability import (
    P5_ACCELERATION_CAP_RAD_S2, P5_VELOCITY_CAP_RAD_S, Calibration,
    DetectionDimensionalityError, KinematicLimits, ReachabilityDecision,
    ReachabilityReason, ReachabilityVerdict, assess_reachability, require_world_3d,
)

__all__ = [
    "FsmState", "HoldReason", "STALE_FRAME_AGE_S", "ALLOWED_TRANSITIONS",
    "FsmError", "HoldLatched", "ResetRejected", "CommandSuppressed", "StageOrderError",
    "FrameInput", "PredictionInput", "PlanInput", "FsmEvent", "FsmLimits",
    "InteractionFsm",
]

#: Same pre-registered 0.100 s as P7-08 (timing/latency_budget.py).  Imported, not
#: copied: if P7-13 measures a different budget there must be exactly one place to
#: change it, and a test pins this equality.
STALE_FRAME_AGE_S = MAX_SOURCE_FRAME_AGE_S


class FsmState(str, Enum):
    SEARCH = "SEARCH"
    TRACK = "TRACK"
    PREDICT = "PREDICT"
    PLAN = "PLAN"
    EXECUTE = "EXECUTE"
    HOLD = "HOLD"


class HoldReason(str, Enum):
    CALIBRATION_MISSING = "CALIBRATION_MISSING"
    STALE_FRAME = "STALE_FRAME"
    COVARIANCE_ABOVE_THRESHOLD = "COVARIANCE_ABOVE_THRESHOLD"
    TIME_TO_HIT_INVALID = "TIME_TO_HIT_INVALID"
    UNREACHABLE_TARGET = "UNREACHABLE_TARGET"
    JOINT_LIMIT_MARGIN = "JOINT_LIMIT_MARGIN"
    COMMAND_OVER_CAP = "COMMAND_OVER_CAP"
    DETECTION_NOT_3D = "DETECTION_NOT_3D"
    STAGE_ORDER = "STAGE_ORDER"
    STAGE_EXCEPTION = "STAGE_EXCEPTION"


#: The transition table this FSM actually implements.  HOLD has no outgoing edge
#: except the explicit reset, which is the latch.
ALLOWED_TRANSITIONS: Dict[FsmState, Tuple[FsmState, ...]] = {
    FsmState.SEARCH: (FsmState.TRACK, FsmState.HOLD),
    FsmState.TRACK: (FsmState.PREDICT, FsmState.HOLD),
    FsmState.PREDICT: (FsmState.PLAN, FsmState.HOLD),
    FsmState.PLAN: (FsmState.EXECUTE, FsmState.HOLD),
    FsmState.EXECUTE: (FsmState.PREDICT, FsmState.PLAN, FsmState.HOLD),
    FsmState.HOLD: (FsmState.TRACK,),
}


class FsmError(RuntimeError):
    """Base class: the FSM refused a transition."""


class HoldLatched(FsmError):
    """A stage was called while HOLD is latched; nothing was executed."""


class ResetRejected(FsmError):
    """reset() was attempted with an input that is not fresh and valid."""


class CommandSuppressed(FsmError):
    """A command was refused because it cannot be emitted in the current state."""


class StageOrderError(FsmError):
    """A stage was called out of order (e.g. plan before any prediction)."""


def _finite(value: object, name: str) -> float:
    try:
        out = float(value)
    except (TypeError, ValueError) as exc:
        raise FsmError(f"{name} must be a real number, got {value!r}") from exc
    if not math.isfinite(out):
        raise FsmError(f"{name} must be finite, got {out!r}")
    return out


@dataclass(frozen=True)
class FrameInput:
    """One perception frame, with the dimensionality it actually has.

    A 2-D detection carries NO world position.  Constructing one that claims a
    3-D point raises DetectionDimensionalityError here, at the boundary, so no
    downstream stage can inherit a promoted detection.
    """

    frame_id: int
    capture_time_s: float
    received_at_s: float
    detection_dim: int
    position_m: Optional[Sequence[float]] = None
    sigma_m: Optional[float] = None

    def __post_init__(self) -> None:
        if isinstance(self.frame_id, bool) or not isinstance(self.frame_id, int):
            raise FsmError("frame_id must be an int")
        object.__setattr__(self, "capture_time_s",
                           _finite(self.capture_time_s, "capture_time_s"))
        object.__setattr__(self, "received_at_s",
                           _finite(self.received_at_s, "received_at_s"))
        if self.detection_dim not in (2, 3):
            raise FsmError(f"detection_dim must be 2 or 3, got {self.detection_dim!r}")
        if self.detection_dim == 2:
            if self.position_m is not None:
                raise DetectionDimensionalityError(
                    f"frame {self.frame_id}: a 2-D detection was given a world 3-D position; "
                    "this is exactly the promotion the contract forbids")
            if self.sigma_m is not None:
                raise DetectionDimensionalityError(
                    f"frame {self.frame_id}: a 2-D detection cannot carry a 3-D sigma")
            return
        point = require_world_3d(self.position_m, source=f"FrameInput(frame {self.frame_id})")
        sigma = _finite(self.sigma_m, "sigma_m")
        if sigma <= 0.0:
            raise FsmError(f"sigma_m must be > 0, got {sigma}")
        object.__setattr__(self, "position_m", point)
        object.__setattr__(self, "sigma_m", sigma)

    @property
    def age_s(self) -> float:
        return self.received_at_s - self.capture_time_s


@dataclass(frozen=True)
class PredictionInput:
    """A prediction handed to the FSM: intercept point, approach, sigma, time-to-hit.

    sigma_start_m is the uncertainty at reference_time_s; sigma_growth_mps is the
    declared growth rate (m/s) the FSM propagates with.  Both come from the
    prediction layer (P7-07 carries a per-sample covariance); the FSM never
    re-invents them.
    """

    prediction_id: str
    frame_id: int
    reference_time_s: float
    intercept_time_s: float
    intercept_point_m: Sequence[float]
    approach_direction: Sequence[float]
    sigma_start_m: float
    sigma_growth_mps: float

    def __post_init__(self) -> None:
        if not isinstance(self.prediction_id, str) or not self.prediction_id.strip():
            raise FsmError("prediction_id must be a non-empty string")
        object.__setattr__(self, "reference_time_s",
                           _finite(self.reference_time_s, "reference_time_s"))
        object.__setattr__(self, "intercept_time_s",
                           _finite(self.intercept_time_s, "intercept_time_s"))
        point = require_world_3d(self.intercept_point_m,
                                 source=f"PredictionInput({self.prediction_id})")
        object.__setattr__(self, "intercept_point_m", point)
        approach = np.asarray(self.approach_direction, dtype=float).reshape(-1)
        if approach.shape != (3,) or not np.all(np.isfinite(approach)):
            raise FsmError("approach_direction must be a finite 3-vector")
        if float(np.linalg.norm(approach)) < 1e-9:
            raise FsmError("approach_direction must be non-zero")
        object.__setattr__(self, "approach_direction", approach)
        sigma = _finite(self.sigma_start_m, "sigma_start_m")
        growth = _finite(self.sigma_growth_mps, "sigma_growth_mps")
        if sigma <= 0.0 or growth < 0.0:
            raise FsmError("sigma_start_m must be > 0 and sigma_growth_mps >= 0")
        object.__setattr__(self, "sigma_start_m", sigma)
        object.__setattr__(self, "sigma_growth_mps", growth)

    def sigma_at(self, at_s: float) -> float:
        """Uncertainty propagated to at_s (linear growth from the reference time)."""
        elapsed = max(0.0, float(at_s) - self.reference_time_s)
        return self.sigma_start_m + self.sigma_growth_mps * elapsed


@dataclass(frozen=True)
class PlanInput:
    """The planner's proposal: joint setpoints and the seed used to derive them."""

    planned_at_s: float
    joints_rad: Sequence[float]
    seed_rad: Sequence[float]
    duration_s: float = 0.0

    def __post_init__(self) -> None:
        object.__setattr__(self, "planned_at_s", _finite(self.planned_at_s, "planned_at_s"))
        object.__setattr__(self, "duration_s", _finite(self.duration_s, "duration_s"))
        joints = np.asarray(self.joints_rad, dtype=float).reshape(-1)
        seed = np.asarray(self.seed_rad, dtype=float).reshape(-1)
        if joints.size == 0 or seed.size != joints.size:
            raise FsmError("joints_rad and seed_rad must have the same non-zero width")
        if not (np.all(np.isfinite(joints)) and np.all(np.isfinite(seed))):
            raise FsmError("joints_rad/seed_rad contain NaN/Inf")
        if self.duration_s < 0.0:
            raise FsmError("duration_s must be >= 0")
        object.__setattr__(self, "joints_rad", joints)
        object.__setattr__(self, "seed_rad", seed)


@dataclass(frozen=True)
class FsmEvent:
    """One recorded transition.  The event log IS the safety evidence."""

    seq: int
    at_s: float
    from_state: FsmState
    to_state: FsmState
    trigger: str
    reason: Optional[HoldReason] = None
    detail: str = ""
    time_to_hit_s: Optional[float] = None
    sigma_m: Optional[float] = None

    def as_dict(self) -> Dict[str, Any]:
        return {
            "seq": self.seq, "at_s": self.at_s,
            "from": self.from_state.value, "to": self.to_state.value,
            "trigger": self.trigger,
            "reason": None if self.reason is None else self.reason.value,
            "detail": self.detail,
            "time_to_hit_s": self.time_to_hit_s,
            "sigma_m": self.sigma_m,
        }


@dataclass(frozen=True)
class FsmLimits:
    """Pre-registered decision thresholds (all proxies; see the P7 doc)."""

    max_source_frame_age_s: float = STALE_FRAME_AGE_S
    max_position_sigma_m: float = 0.05
    min_time_to_hit_s: float = 0.02
    joint_margin_rad: float = 0.05
    velocity_cap_rad_s: float = P5_VELOCITY_CAP_RAD_S
    acceleration_cap_rad_s2: float = P5_ACCELERATION_CAP_RAD_S2
    command_dt_s: float = 1.0 / 60.0

    def __post_init__(self) -> None:
        for name in ("max_source_frame_age_s", "max_position_sigma_m",
                     "min_time_to_hit_s", "joint_margin_rad", "velocity_cap_rad_s",
                     "acceleration_cap_rad_s2", "command_dt_s"):
            value = getattr(self, name)
            if not math.isfinite(float(value)) or float(value) <= 0.0:
                raise FsmError(f"{name} must be finite and > 0, got {value!r}")


class InteractionFsm:
    """SEARCH/TRACK/PREDICT/PLAN/EXECUTE/HOLD with a latching fail-closed HOLD."""

    def __init__(self, *, limits: Optional[FsmLimits] = None,
                 calibration: Optional[Calibration] = None,
                 ik_solver: Optional[IkSolver] = None,
                 kinematic_limits: Optional[KinematicLimits] = None,
                 state: FsmState = FsmState.SEARCH,
                 clock_s: float = 0.0) -> None:
        self.limits = limits or FsmLimits()
        self.calibration = calibration or Calibration()
        self.ik_solver = ik_solver
        self.kinematic_limits = kinematic_limits or KinematicLimits()
        self._state = FsmState(state)
        self._clock = float(clock_s)
        self._events: List[FsmEvent] = []
        self._hold_reason: Optional[HoldReason] = None
        self._hold_detail = ""
        self._pending_command: Optional[np.ndarray] = None
        self._emitted: List[Dict[str, Any]] = []
        self._suppressed = 0
        self._frame: Optional[FrameInput] = None
        self._prediction: Optional[PredictionInput] = None
        self._plan: Optional[PlanInput] = None
        self._last_accepted_joints: Optional[np.ndarray] = None
        self._previous_command: Optional[np.ndarray] = None
        self._verdict: Optional[ReachabilityVerdict] = None

    # ------------------------------------------------------------------ state
    @property
    def state(self) -> FsmState:
        return self._state

    @property
    def hold_reason(self) -> Optional[HoldReason]:
        return self._hold_reason

    @property
    def hold_detail(self) -> str:
        return self._hold_detail

    @property
    def holding(self) -> bool:
        return self._state is FsmState.HOLD

    @property
    def events(self) -> List[FsmEvent]:
        return list(self._events)

    @property
    def emitted_commands(self) -> List[Dict[str, Any]]:
        return list(self._emitted)

    @property
    def suppressed_commands(self) -> int:
        return self._suppressed

    @property
    def last_verdict(self) -> Optional[ReachabilityVerdict]:
        return self._verdict

    @property
    def pending_command(self) -> Optional[np.ndarray]:
        return None if self._pending_command is None else self._pending_command.copy()

    def _record(self, from_state: FsmState, to_state: FsmState, trigger: str,
                reason: Optional[HoldReason] = None, detail: str = "",
                time_to_hit_s: Optional[float] = None,
                sigma_m: Optional[float] = None) -> None:
        allowed = ALLOWED_TRANSITIONS.get(from_state, ())
        if to_state not in allowed and to_state is not from_state:
            raise StageOrderError(
                f"transition {from_state.value} -> {to_state.value} is not in the "
                f"transition table (allowed: {[s.value for s in allowed]})")
        self._events.append(FsmEvent(
            seq=len(self._events) + 1, at_s=round(self._clock, 9),
            from_state=from_state, to_state=to_state, trigger=trigger, reason=reason,
            detail=detail, time_to_hit_s=time_to_hit_s, sigma_m=sigma_m))
        self._state = to_state

    # ------------------------------------------------------------------- HOLD
    def enter_hold(self, reason: HoldReason, detail: str,
                   *, time_to_hit_s: Optional[float] = None,
                   sigma_m: Optional[float] = None) -> None:
        """Latch HOLD.  Idempotent in state, but every entry is recorded."""
        from_state = self._state
        self._pending_command = None
        self._plan = None
        self._hold_reason = HoldReason(reason)
        self._hold_detail = detail
        if from_state is FsmState.HOLD:
            self._events.append(FsmEvent(
                seq=len(self._events) + 1, at_s=round(self._clock, 9),
                from_state=from_state, to_state=FsmState.HOLD,
                trigger="FAIL_CLOSED", reason=self._hold_reason, detail=detail,
                time_to_hit_s=time_to_hit_s, sigma_m=sigma_m))
            self._state = FsmState.HOLD
            return
        self._record(from_state, FsmState.HOLD, "FAIL_CLOSED", self._hold_reason, detail,
                     time_to_hit_s=time_to_hit_s, sigma_m=sigma_m)

    def _fail_closed(self, reason: HoldReason, detail: str, **kwargs: Any) -> FsmState:
        self.enter_hold(reason, detail, **kwargs)
        return self._state

    # ------------------------------------------------------------------ stages
    def on_frame(self, frame: FrameInput, now_s: float) -> FsmState:
        """SEARCH -> TRACK.  Refuses a stale frame or a missing calibration."""
        stage = "on_frame"
        self._clock = max(self._clock, _finite(now_s, "now_s"))
        if self.holding:
            self._suppressed_stage(stage)
            return self._state
        try:
            if not self.calibration.usable():
                return self._fail_closed(
                    HoldReason.CALIBRATION_MISSING,
                    "no verified court_from_base extrinsic: a detected point cannot be tied "
                    "to a world frame, so tracking is refused")
            age = self._clock - frame.capture_time_s
            sigma = frame.sigma_m if frame.detection_dim == 3 else None
            if age > self.limits.max_source_frame_age_s:
                return self._fail_closed(
                    HoldReason.STALE_FRAME,
                    f"frame {frame.frame_id} is {age:.4f} s old > "
                    f"{self.limits.max_source_frame_age_s:.4f} s; a stale frame is not a "
                    "track", sigma_m=sigma)
            if age < 0.0 - 1e-9:
                return self._fail_closed(
                    HoldReason.STALE_FRAME,
                    f"frame {frame.frame_id} is {-age:.4f} s in the FUTURE: clock error",
                    sigma_m=sigma)
            if frame.detection_dim == 3 and frame.sigma_m is not None:
                if frame.sigma_m > self.limits.max_position_sigma_m:
                    return self._fail_closed(
                        HoldReason.COVARIANCE_ABOVE_THRESHOLD,
                        f"frame sigma {frame.sigma_m:.4f} m > "
                        f"{self.limits.max_position_sigma_m:.4f} m", sigma_m=frame.sigma_m)
            from_state = self._state
            if from_state is FsmState.SEARCH:
                self._record(from_state, FsmState.TRACK, "FRAME", sigma_m=sigma)
            elif from_state is FsmState.TRACK:
                self._events.append(FsmEvent(
                    seq=len(self._events) + 1, at_s=round(self._clock, 9),
                    from_state=from_state, to_state=FsmState.TRACK, trigger="FRAME",
                    detail=f"frame {frame.frame_id}", sigma_m=sigma))
            elif from_state is FsmState.EXECUTE:
                # Perception keeps running while the swing is being executed: the
                # frame is checked for staleness/sigma exactly as above and the
                # tracking input is updated in place.  No state change, so this can
                # never be a route out of HOLD (HOLD returns earlier).
                self._events.append(FsmEvent(
                    seq=len(self._events) + 1, at_s=round(self._clock, 9),
                    from_state=from_state, to_state=FsmState.EXECUTE, trigger="FRAME_UPDATE",
                    detail=f"frame {frame.frame_id}", sigma_m=sigma))
            else:
                return self._fail_closed(
                    HoldReason.STAGE_ORDER,
                    f"on_frame called in {from_state.value}; frames are only accepted in "
                    "SEARCH/TRACK", sigma_m=sigma)
            self._frame = frame
            return self._state
        except DetectionDimensionalityError as exc:
            return self._fail_closed(HoldReason.DETECTION_NOT_3D, str(exc))
        except FsmError as exc:
            return self._fail_closed(HoldReason.STAGE_EXCEPTION, f"{stage}: {exc}")
        except Exception as exc:  # noqa: BLE001 - fail closed by contract
            return self._fail_closed(
                HoldReason.STAGE_EXCEPTION, f"{stage}: {type(exc).__name__}: {exc}")

    def on_prediction(self, prediction: PredictionInput, now_s: float) -> FsmState:
        """TRACK -> PREDICT.  Carries sigma and time-to-hit into the decision."""
        stage = "on_prediction"
        self._clock = max(self._clock, _finite(now_s, "now_s"))
        if self.holding:
            self._suppressed_stage(stage)
            return self._state
        try:
            if self._state not in (FsmState.TRACK, FsmState.EXECUTE):
                return self._fail_closed(
                    HoldReason.STAGE_ORDER,
                    f"on_prediction called in {self._state.value}; a prediction requires a "
                    "TRACKed frame first")
            if self._frame is None or self._frame.detection_dim != 3:
                return self._fail_closed(
                    HoldReason.DETECTION_NOT_3D,
                    "no 3-D frame: a 2-D-only detection is never promoted to world 3-D")
            time_to_hit = prediction.intercept_time_s - self._clock
            sigma = prediction.sigma_at(self._clock)
            if not math.isfinite(time_to_hit) or time_to_hit <= 0.0:
                return self._fail_closed(
                    HoldReason.TIME_TO_HIT_INVALID,
                    f"time-to-hit {time_to_hit:.4f} s is zero/reversed (intercept "
                    f"{prediction.intercept_time_s:.4f} s vs now {self._clock:.4f} s); a "
                    "ball already past the intercept cannot be hit",
                    time_to_hit_s=time_to_hit, sigma_m=sigma)
            if time_to_hit < self.limits.min_time_to_hit_s:
                return self._fail_closed(
                    HoldReason.TIME_TO_HIT_INVALID,
                    f"time-to-hit {time_to_hit:.4f} s < {self.limits.min_time_to_hit_s:.4f} s "
                    "minimum planning time", time_to_hit_s=time_to_hit, sigma_m=sigma)
            if sigma > self.limits.max_position_sigma_m:
                return self._fail_closed(
                    HoldReason.COVARIANCE_ABOVE_THRESHOLD,
                    f"propagated sigma {sigma:.4f} m > "
                    f"{self.limits.max_position_sigma_m:.4f} m: the intercept is not "
                    "localised well enough to plan a swing",
                    time_to_hit_s=time_to_hit, sigma_m=sigma)
            self._record(self._state, FsmState.PREDICT, "PREDICTION",
                         detail=f"prediction {prediction.prediction_id}",
                         time_to_hit_s=time_to_hit, sigma_m=sigma)
            self._prediction = prediction
            return self._state
        except DetectionDimensionalityError as exc:
            return self._fail_closed(HoldReason.DETECTION_NOT_3D, str(exc))
        except FsmError as exc:
            return self._fail_closed(HoldReason.STAGE_EXCEPTION, f"{stage}: {exc}")
        except Exception as exc:  # noqa: BLE001
            return self._fail_closed(
                HoldReason.STAGE_EXCEPTION, f"{stage}: {type(exc).__name__}: {exc}")

    def on_plan(self, plan: PlanInput, now_s: float) -> FsmState:
        """PREDICT -> PLAN.  Reachability decides; UNKNOWN is not a licence to move."""
        stage = "on_plan"
        self._clock = max(self._clock, _finite(now_s, "now_s"))
        if self.holding:
            self._suppressed_stage(stage)
            return self._state
        try:
            if self._state is not FsmState.PREDICT or self._prediction is None:
                return self._fail_closed(
                    HoldReason.STAGE_ORDER,
                    f"on_plan called in {self._state.value} without a prediction")
            prediction = self._prediction
            # PROPAGATE the prediction's uncertainty and clock to this stage: the
            # planning duration is time the ball spends travelling.
            horizon_s = max(0.0, plan.planned_at_s - prediction.reference_time_s) \
                + plan.duration_s
            time_to_hit = prediction.intercept_time_s - prediction.reference_time_s - horizon_s
            sigma = prediction.sigma_at(prediction.reference_time_s + horizon_s)
            if time_to_hit <= 0.0:
                return self._fail_closed(
                    HoldReason.TIME_TO_HIT_INVALID,
                    f"after {horizon_s:.4f} s of planning the intercept is {-time_to_hit:.4f} s "
                    "in the past", time_to_hit_s=time_to_hit, sigma_m=sigma)
            if sigma > self.limits.max_position_sigma_m:
                return self._fail_closed(
                    HoldReason.COVARIANCE_ABOVE_THRESHOLD,
                    f"propagated sigma {sigma:.4f} m over {horizon_s:.4f} s exceeds "
                    f"{self.limits.max_position_sigma_m:.4f} m",
                    time_to_hit_s=time_to_hit, sigma_m=sigma)
            verdict = assess_reachability(
                intercept_point_m=prediction.intercept_point_m,
                approach_direction=prediction.approach_direction,
                limits=self.kinematic_limits,
                time_to_target_s=time_to_hit,
                time_margin_s=self.limits.min_time_to_hit_s,
                seed_rad=plan.seed_rad,
                solver=self.ik_solver,
                calibration=self.calibration,
                position_sigma_m=sigma,
            )
            self._verdict = verdict
            if verdict.decision is ReachabilityDecision.NOT_REACHABLE:
                return self._fail_closed(
                    HoldReason.UNREACHABLE_TARGET,
                    f"{verdict.reason.value}: {verdict.detail}",
                    time_to_hit_s=time_to_hit, sigma_m=sigma)
            if verdict.decision is ReachabilityDecision.UNKNOWN:
                return self._fail_closed(
                    HoldReason.UNREACHABLE_TARGET,
                    f"UNKNOWN is not a licence to move ({verdict.reason.value}): "
                    f"{verdict.detail}", time_to_hit_s=time_to_hit, sigma_m=sigma)
            if verdict.ik is None:
                return self._fail_closed(
                    HoldReason.STAGE_EXCEPTION,
                    "REACHABLE verdict without an IK result: the interface forbids it",
                    time_to_hit_s=time_to_hit, sigma_m=sigma)
            self._record(FsmState.PREDICT, FsmState.PLAN, "PLAN",
                         detail=f"{verdict.reason.value}: {verdict.detail}",
                         time_to_hit_s=time_to_hit, sigma_m=sigma)
            self._plan = plan
            return self._state
        except DetectionDimensionalityError as exc:
            return self._fail_closed(HoldReason.DETECTION_NOT_3D, str(exc))
        except FsmError as exc:
            return self._fail_closed(HoldReason.STAGE_EXCEPTION, f"{stage}: {exc}")
        except Exception as exc:  # noqa: BLE001
            return self._fail_closed(
                HoldReason.STAGE_EXCEPTION, f"{stage}: {type(exc).__name__}: {exc}")

    def on_execute_command(self, joints_rad: Sequence[float], now_s: float) -> FsmState:
        """PLAN/EXECUTE -> EXECUTE, if and only if the command passes the P5-05 envelope."""
        stage = "on_execute_command"
        self._clock = max(self._clock, _finite(now_s, "now_s"))
        if self.holding:
            self._suppressed_stage(stage)
            raise CommandSuppressed(
                f"FSM is in HOLD ({self._hold_reason.value}): no command may be emitted")
        try:
            if self._state not in (FsmState.PLAN, FsmState.EXECUTE):
                return self._fail_closed(
                    HoldReason.STAGE_ORDER,
                    f"on_execute_command called in {self._state.value}; a command needs a "
                    "PLAN")
            command = np.asarray(joints_rad, dtype=float).reshape(-1)
            if command.shape != self._plan_shape() or not np.all(np.isfinite(command)):
                return self._fail_closed(
                    HoldReason.STAGE_EXCEPTION,
                    f"command has shape {command.shape} (expected {self._plan_shape()}) or "
                    "is not finite")
            # joint-limit margin
            margin, index = self._joint_margin(command)
            if margin < self.limits.joint_margin_rad:
                return self._fail_closed(
                    HoldReason.JOINT_LIMIT_MARGIN,
                    f"joint {index} is {margin:.4f} rad from its stop, inside the "
                    f"{self.limits.joint_margin_rad:.4f} rad margin")
            # P5-05 velocity / acceleration caps on the setpoint step
            dt = self.limits.command_dt_s
            if self._previous_command is not None and self._previous_command.shape == command.shape:
                step = float(np.max(np.abs(command - self._previous_command)))
                velocity = step / dt
                if velocity > self.limits.velocity_cap_rad_s + 1e-12:
                    return self._fail_closed(
                        HoldReason.COMMAND_OVER_CAP,
                        f"setpoint rate {velocity:.4f} rad/s > P5-05 cap "
                        f"{self.limits.velocity_cap_rad_s:.4f} rad/s")
                acceleration = step / (dt * dt)
                if acceleration > self.limits.acceleration_cap_rad_s2 + 1e-12:
                    return self._fail_closed(
                        HoldReason.COMMAND_OVER_CAP,
                        f"setpoint acceleration {acceleration:.4f} rad/s^2 > P5-05 cap "
                        f"{self.limits.acceleration_cap_rad_s2:.4f} rad/s^2")
            # uncertainty still under threshold at emission time
            sigma = self._sigma_now()
            if sigma is not None and sigma > self.limits.max_position_sigma_m:
                return self._fail_closed(
                    HoldReason.COVARIANCE_ABOVE_THRESHOLD,
                    f"sigma {sigma:.4f} m at command time > "
                    f"{self.limits.max_position_sigma_m:.4f} m", sigma_m=sigma)
            time_to_hit = self._time_to_hit_now()
            if time_to_hit is not None and time_to_hit <= 0.0:
                return self._fail_closed(
                    HoldReason.TIME_TO_HIT_INVALID,
                    f"time-to-hit {time_to_hit:.4f} s at command time: the intercept passed "
                    "while planning", time_to_hit_s=time_to_hit, sigma_m=sigma)
            from_state = self._state
            self._record(from_state, FsmState.EXECUTE, "COMMAND",
                         detail="command accepted by the P5-05 envelope",
                         time_to_hit_s=time_to_hit, sigma_m=sigma)
            self._pending_command = None
            self._previous_command = command.copy()
            self._last_accepted_joints = command.copy()
            self._emitted.append({
                "at_s": round(self._clock, 9),
                "joints_rad": [float(v) for v in command],
                "state": FsmState.EXECUTE.value,
                "time_to_hit_s": time_to_hit, "sigma_m": sigma,
            })
            return self._state
        except CommandSuppressed:
            raise
        except FsmError as exc:
            return self._fail_closed(HoldReason.STAGE_EXCEPTION, f"{stage}: {exc}")
        except Exception as exc:  # noqa: BLE001
            return self._fail_closed(
                HoldReason.STAGE_EXCEPTION, f"{stage}: {type(exc).__name__}: {exc}")

    def on_cycle_complete(self, now_s: float) -> FsmState:
        """EXECUTE -> PLAN: iterate on the same plan (more frames of the same swing)."""
        self._clock = max(self._clock, _finite(now_s, "now_s"))
        if self.holding:
            self._suppressed_stage("on_cycle_complete")
            return self._state
        if self._state is not FsmState.EXECUTE:
            return self._fail_closed(HoldReason.STAGE_ORDER,
                                     f"on_cycle_complete in {self._state.value}")
        sigma = self._sigma_now()
        time_to_hit = self._time_to_hit_now()
        if sigma is not None and sigma > self.limits.max_position_sigma_m:
            return self._fail_closed(HoldReason.COVARIANCE_ABOVE_THRESHOLD,
                                     f"sigma {sigma:.4f} m at cycle end", sigma_m=sigma)
        if time_to_hit is not None and time_to_hit <= 0.0:
            return self._fail_closed(HoldReason.TIME_TO_HIT_INVALID,
                                     f"time-to-hit {time_to_hit:.4f} s at cycle end",
                                     time_to_hit_s=time_to_hit, sigma_m=sigma)
        self._record(FsmState.EXECUTE, FsmState.PLAN, "CYCLE", time_to_hit_s=time_to_hit,
                     sigma_m=sigma)
        return self._state

    # ------------------------------------------------------------------- reset
    def reset(self, *, frame: FrameInput, prediction: PredictionInput, now_s: float,
              operator: str) -> FsmState:
        """Leave HOLD, but only with a FRESH, fully valid frame AND prediction.

        A reset is an explicit human/operator act (operator is required and
        recorded).  A stale frame, a dead time-to-hit, an over-threshold sigma or a
        missing calibration means the latch stays and ResetRejected is raised.
        """
        if not isinstance(operator, str) or not operator.strip():
            raise ResetRejected("reset requires a named operator; an anonymous reset is "
                                "indistinguishable from an automatic one")
        if not self.holding:
            raise ResetRejected(f"reset called in {self._state.value}: the FSM is not holding")
        self._clock = max(self._clock, _finite(now_s, "now_s"))
        problems: List[str] = []
        if not self.calibration.usable():
            problems.append("calibration is still missing/unverified")
        age = self._clock - frame.capture_time_s
        if age > self.limits.max_source_frame_age_s:
            problems.append(f"reset frame is {age:.4f} s old > "
                            f"{self.limits.max_source_frame_age_s:.4f} s")
        if age < -1e-9:
            problems.append(f"reset frame is {-age:.4f} s in the future")
        if frame.detection_dim != 3:
            problems.append("reset frame is 2-D; a fresh 3-D detection is required")
        time_to_hit = prediction.intercept_time_s - self._clock
        if time_to_hit < self.limits.min_time_to_hit_s:
            problems.append(f"reset prediction time-to-hit {time_to_hit:.4f} s is below the "
                            f"{self.limits.min_time_to_hit_s:.4f} s minimum")
        sigma = prediction.sigma_at(self._clock)
        if sigma > self.limits.max_position_sigma_m:
            problems.append(f"reset prediction sigma {sigma:.4f} m > "
                            f"{self.limits.max_position_sigma_m:.4f} m")
        if problems:
            self._events.append(FsmEvent(
                seq=len(self._events) + 1, at_s=round(self._clock, 9),
                from_state=FsmState.HOLD, to_state=FsmState.HOLD, trigger="RESET_REJECTED",
                reason=self._hold_reason, detail="; ".join(problems),
                time_to_hit_s=time_to_hit, sigma_m=sigma))
            raise ResetRejected("; ".join(problems))

        previous_reason = self._hold_reason
        self._hold_reason = None
        self._hold_detail = ""
        self._frame = frame
        self._prediction = prediction
        self._plan = None
        self._pending_command = None
        self._previous_command = None
        self._events.append(FsmEvent(
            seq=len(self._events) + 1, at_s=round(self._clock, 9),
            from_state=FsmState.HOLD, to_state=FsmState.TRACK, trigger="RESET",
            detail=f"operator={operator.strip()} cleared {previous_reason.value if previous_reason else 'HOLD'}",
            time_to_hit_s=time_to_hit, sigma_m=sigma))
        self._state = FsmState.TRACK
        return self._state

    # ----------------------------------------------------------------- helpers
    def _suppressed_stage(self, stage: str) -> None:
        self._suppressed += 1
        self._events.append(FsmEvent(
            seq=len(self._events) + 1, at_s=round(self._clock, 9),
            from_state=FsmState.HOLD, to_state=FsmState.HOLD, trigger="SUPPRESSED",
            reason=self._hold_reason, detail=f"{stage} refused while HOLD is latched"))

    def _plan_shape(self) -> Tuple[int, ...]:
        if self._plan is not None:
            return tuple(np.asarray(self._plan.joints_rad, dtype=float).shape)
        if self._last_accepted_joints is not None:
            return tuple(self._last_accepted_joints.shape)
        return (self.kinematic_limits.lower_rad.__len__(),)

    def _joint_margin(self, command: np.ndarray) -> Tuple[float, int]:
        lower = np.asarray(self.kinematic_limits.lower_rad, dtype=float)
        upper = np.asarray(self.kinematic_limits.upper_rad, dtype=float)
        margin = np.minimum(command - lower, upper - command)
        index = int(np.argmin(margin))
        return float(margin[index]), index

    def _sigma_now(self) -> Optional[float]:
        if self._prediction is None:
            return None
        return self._prediction.sigma_at(self._clock)

    def _time_to_hit_now(self) -> Optional[float]:
        if self._prediction is None:
            return None
        return self._prediction.intercept_time_s - self._clock

    def report(self) -> Dict[str, Any]:
        """Machine-readable state + event log (the safety evidence)."""
        reasons: Dict[str, int] = {}
        for event in self._events:
            if event.reason is not None:
                reasons[event.reason.value] = reasons.get(event.reason.value, 0) + 1
        return {
            "schema": "badmin.p7.safety_fsm",
            "schema_version": 1,
            "state": self._state.value,
            "hold": {
                "latched": self.holding,
                "reason": None if self._hold_reason is None else self._hold_reason.value,
                "detail": self._hold_detail,
            },
            "events": [event.as_dict() for event in self._events],
            "hold_reasons_seen": dict(sorted(reasons.items())),
            "commands_emitted": len(self._emitted),
            "commands_suppressed": self._suppressed,
            "thresholds": {
                "max_source_frame_age_s": self.limits.max_source_frame_age_s,
                "max_position_sigma_m": self.limits.max_position_sigma_m,
                "min_time_to_hit_s": self.limits.min_time_to_hit_s,
                "joint_margin_rad": self.limits.joint_margin_rad,
                "velocity_cap_rad_s": self.limits.velocity_cap_rad_s,
                "acceleration_cap_rad_s2": self.limits.acceleration_cap_rad_s2,
            },
            "evidence_class": "OFFLINE-SYNTHETIC",
        }
