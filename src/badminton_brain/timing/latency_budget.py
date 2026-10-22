# -*- coding: utf-8 -*-
"""P7-08: per-stage latency budget and stale-frame rejection for the brain chain.

Why this module exists
----------------------
P7-04/P7-05/P7-07 (PR #27) already reject a PREDICTION whose measurement is older
than MAX_MEASUREMENT_AGE_S = 0.10 s.  That check lives inside the predictor and
only knows about measurement age.  Nothing in the repo knows how long the whole
chain took (capture -> detect -> filter -> predict -> command), and nothing stops
a *consumer* from re-using a prediction built from a frame that has since gone
stale.  A real-time racket swing that acts on a 0.3 s old ball position does not
miss politely; it commits the robot to a point the ball left long ago.

This module supplies the missing, deliberately small pieces:

  StageTimestamps   the five absolute timestamps of one frame's journey;
  latency_report    per-stage durations, end-to-end budget accounting;
  StalenessGuard    admit/reject a prediction for consumption, with a
                    pre-registered threshold and an explicit refusal record;
  Admission         the audit trail entry a consumer must store.

Pre-registered policy (fixed BEFORE any measurement, see
management/morph_one_rs/P7_AUDIT_AND_LATENCY.md)
-------------------------------------------------
  MAX_SOURCE_FRAME_AGE_S = 0.100 s
      A prediction may be consumed only while its SOURCE FRAME is at most this
      old.  Boundary: age <= threshold is admitted, age > threshold is REFUSED.
      Rationale: the same 0.10 s already used as MAX_MEASUREMENT_AGE_S by P7-07,
      derived from the 0.30 s minimum prediction horizon minus ~0.10 s of
      perception/inference and ~0.10 s of decision+planning.  It is a TEMP proxy
      until P7-13 measures the real chain.
  Stage budgets (sum = END_TO_END_BUDGET_S = 0.100 s):
      capture 0.010 | detect 0.030 | filter 0.005 | predict 0.010 | command 0.005
      plus a 0.040 s reserve that is NOT spendable by a stage (actuation/network
      jitter).  A stage over budget is reported, never rounded away.

What this does NOT prove
------------------------
No camera is read, no detector runs, no robot moves.  The numbers produced from
this module are budget ACCOUNTING on timestamps a caller supplies; the constants
are proxies, and MEASURED_* replacements remain UNKNOWN (P7-13).

Run:
    python -m pytest tests/badminton_brain/test_latency_budget.py -q
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

__all__ = [
    "STAGES", "MAX_SOURCE_FRAME_AGE_S", "STAGE_BUDGET_S", "END_TO_END_BUDGET_S",
    "RESERVE_S", "REJECT_AGE_EXCEEDED", "REJECT_FRAME_REUSE", "REJECT_FUTURE_FRAME",
    "REJECT_CLOCK_REGRESSION", "REJECT_UNKNOWN_PREDICTION",
    "LatencyError", "StaleFrameRejected", "StageTimestamps", "Admission",
    "StalenessGuard", "latency_report",
]

# Ordered stages of one frame's journey through the brain.  The order is part of
# the contract: a timestamp earlier than its predecessor is a clock error, not a
# fast stage.
STAGES: Tuple[str, ...] = ("capture", "detect", "filter", "predict", "command")

# PRE-REGISTERED (before measurement) - see the module docstring.
MAX_SOURCE_FRAME_AGE_S = 0.100
STAGE_BUDGET_S: Dict[str, float] = {
    "capture": 0.010,
    "detect": 0.030,
    "filter": 0.005,
    "predict": 0.010,
    "command": 0.005,
}
RESERVE_S = 0.040
END_TO_END_BUDGET_S = 0.100

REJECT_AGE_EXCEEDED = "SOURCE_FRAME_AGE_EXCEEDED"
REJECT_FRAME_REUSE = "SOURCE_FRAME_REUSE_AFTER_STALE"
REJECT_FUTURE_FRAME = "SOURCE_FRAME_IN_FUTURE"
REJECT_CLOCK_REGRESSION = "CLOCK_REGRESSION"
REJECT_UNKNOWN_PREDICTION = "UNKNOWN_PREDICTION_ID"

TOLERANCE_S = 1e-9


class LatencyError(ValueError):
    """A timestamp record cannot be interpreted (order, finiteness, count)."""


class StaleFrameRejected(RuntimeError):
    """A prediction was refused for consumption; carries the machine-readable reason."""

    def __init__(self, reason: str, detail: str, *, admission: Optional["Admission"] = None):
        self.reason = reason
        self.detail = detail
        self.admission = admission
        super().__init__(f"{reason}: {detail}")


def _seconds(value: Any, name: str) -> float:
    if isinstance(value, bool):
        raise LatencyError(f"{name} must be a real number, got bool")
    try:
        out = float(value)
    except (TypeError, ValueError) as exc:
        raise LatencyError(f"{name} must be a real number, got {value!r}") from exc
    if not math.isfinite(out):
        raise LatencyError(f"{name} must be finite, got {out!r}")
    return out


@dataclass(frozen=True)
class StageTimestamps:
    """Absolute timestamps (seconds, one shared clock) of one frame's five stages."""

    capture: float
    detect: float
    filter: float
    predict: float
    command: float
    frame_id: Optional[int] = None

    def __post_init__(self) -> None:
        values = {}
        for stage in STAGES:
            values[stage] = _seconds(getattr(self, stage), "StageTimestamps." + stage)
            object.__setattr__(self, stage, values[stage])
        previous = None
        for stage in STAGES:
            if previous is not None and values[stage] < values[previous] - TOLERANCE_S:
                raise LatencyError(
                    f"stage {stage!r} ({values[stage]}) is earlier than {previous!r} "
                    f"({values[previous]}): timestamps must be non-decreasing")
            previous = stage
        if self.frame_id is not None:
            if isinstance(self.frame_id, bool) or not isinstance(self.frame_id, int):
                raise LatencyError("frame_id must be an integer when given")
            if self.frame_id < 0:
                raise LatencyError("frame_id must be >= 0")

    @property
    def source_frame_timestamp_s(self) -> float:
        """The capture time of the frame the detections were computed from."""
        return self.capture

    def durations_s(self) -> Dict[str, float]:
        """Duration of each stage = t_stage - t_previous_stage.

        A timestamp marks the moment a stage FINISHED, so "detect" is the time
        from the (finished) capture stamp to the (finished) detection stamp.
        The first stage has no predecessor; its duration is reported as 0.0 here
        and latency_report marks it NOT_MEASURABLE instead of pretending it is 0.
        """
        out: Dict[str, float] = {}
        previous = None
        for stage in STAGES:
            value = getattr(self, stage)
            out[stage] = 0.0 if previous is None else round(value - getattr(self, previous), 9)
            previous = stage
        return out

    @property
    def end_to_end_s(self) -> float:
        return round(self.command - self.capture, 9)

    @property
    def decision_age_s(self) -> float:
        """Age of the source frame at command time (this is what staleness uses)."""
        return round(self.command - self.capture, 9)

    def to_json_dict(self) -> Dict[str, Any]:
        return {
            "frame_id": self.frame_id,
            "timestamps_s": {stage: getattr(self, stage) for stage in STAGES},
            "durations_s": self.durations_s(),
            "end_to_end_s": self.end_to_end_s,
        }


def latency_report(frames: Iterable[StageTimestamps], *,
                   stage_budget_s: Optional[Mapping[str, float]] = None,
                   end_to_end_budget_s: Optional[float] = None) -> Dict[str, Any]:
    """Budget accounting over a batch of frames (no measurement pretence)."""
    budget = dict(STAGE_BUDGET_S if stage_budget_s is None else stage_budget_s)
    total_budget = END_TO_END_BUDGET_S if end_to_end_budget_s is None else float(end_to_end_budget_s)
    missing = [stage for stage in STAGES if stage not in budget]
    if missing:
        raise LatencyError(f"stage budget is missing {missing}")
    if abs(sum(budget.values()) + RESERVE_S - total_budget) > 1e-6:
        raise LatencyError(
            "stage budgets plus the reserve must equal the end-to-end budget: "
            f"{sum(budget.values())} + {RESERVE_S} != {total_budget}")

    rows = list(frames)
    if not rows:
        raise LatencyError("latency_report needs at least one frame")
    per_stage: Dict[str, Any] = {}
    for stage in STAGES:
        if stage == STAGES[0]:
            # capture is the clock origin: its own duration cannot be recovered
            # from these five timestamps, so it is NOT silently reported as 0.
            per_stage[stage] = {
                "budget_s": float(budget[stage]), "n": 0, "mean_s": None,
                "max_s": None, "over_budget_frames": 0,
                "over_budget_max_s": 0.0,
                "status": "NOT_MEASURABLE_FROM_STAGE_TIMESTAMPS",
            }
            continue
        values = sorted(row.durations_s()[stage] for row in rows)
        budget_value = float(budget[stage])
        over = [value for value in values if value > budget_value + TOLERANCE_S]
        per_stage[stage] = {
            "budget_s": budget_value,
            "n": len(values),
            "mean_s": round(sum(values) / len(values), 9),
            "max_s": round(values[-1], 9),
            "over_budget_frames": len(over),
            "over_budget_max_s": round(max(over), 9) if over else 0.0,
            "status": "OVER_BUDGET" if over else "WITHIN_BUDGET",
        }
    end_to_end = sorted(row.end_to_end_s for row in rows)
    e2e_over = [value for value in end_to_end if value > total_budget + TOLERANCE_S]
    worst = max(rows, key=lambda row: row.end_to_end_s)
    return {
        "schema": "badmin.p7.latency_budget",
        "schema_version": 1,
        "pre_registered": {
            "max_source_frame_age_s": MAX_SOURCE_FRAME_AGE_S,
            "stage_budget_s": budget,
            "reserve_s": RESERVE_S,
            "end_to_end_budget_s": total_budget,
            "measured_replacements": "UNKNOWN (P7-13 has not measured a real chain)",
        },
        "evidence_class": "OFFLINE-SYNTHETIC",
        "frames": len(rows),
        "per_stage": per_stage,
        "end_to_end": {
            "budget_s": total_budget,
            "mean_s": round(sum(end_to_end) / len(end_to_end), 9),
            "p50_s": round(end_to_end[len(end_to_end) // 2], 9),
            "max_s": round(end_to_end[-1], 9),
            "max_frame_id": worst.frame_id,
            "over_budget_frames": len(e2e_over),
            "status": "OVER_BUDGET" if e2e_over else "WITHIN_BUDGET",
        },
        "accounting": {
            "sum_of_means_equals_mean_of_end_to_end": True,
            "note": "end_to_end = command - capture = sum of the four inter-stage deltas",
        },
        "caveats": [
            "Timestamps are supplied by the caller; no camera, detector or robot ran here.",
            "Budgets are pre-registered proxies, not measured latencies.",
        ],
    }


@dataclass(frozen=True)
class Admission:
    """The record a consumer must keep in order to act on a prediction."""

    prediction_id: str
    frame_id: int
    source_frame_timestamp_s: float
    consumed_at_s: float
    age_s: float
    threshold_s: float
    decision: str = "ADMITTED"

    def to_json_dict(self) -> Dict[str, Any]:
        return {
            "prediction_id": self.prediction_id,
            "frame_id": self.frame_id,
            "source_frame_timestamp_s": self.source_frame_timestamp_s,
            "consumed_at_s": self.consumed_at_s,
            "age_s": self.age_s,
            "threshold_s": self.threshold_s,
            "decision": self.decision,
        }


class StalenessGuard:
    """Refuse to hand a stale prediction to a consumer; never silently re-use one.

    Policy (pre-registered):
      * age = now - source_frame_timestamp_s;
      * age > threshold  -> REFUSED, reason SOURCE_FRAME_AGE_EXCEEDED;
      * age < 0          -> REFUSED, reason SOURCE_FRAME_IN_FUTURE (clock error);
      * a prediction already admitted once -> REFUSED, reason
        SOURCE_FRAME_REUSE_AFTER_STALE (a recycled prediction is never silently
        accepted, whether or not it is also stale).
    Every refusal raises StaleFrameRejected carrying the Admission record, so a
    caller cannot swallow the decision: the exception IS the audit trail.
    """

    def __init__(self, *, threshold_s: float = MAX_SOURCE_FRAME_AGE_S,
                 clock_tolerance_s: float = TOLERANCE_S):
        threshold = _seconds(threshold_s, "threshold_s")
        if threshold <= 0.0:
            raise LatencyError(f"threshold_s must be > 0, got {threshold}")
        self.threshold_s = threshold
        self.clock_tolerance_s = float(clock_tolerance_s)
        self._last_admitted: Dict[str, Admission] = {}
        self._admissions: List[Admission] = []
        self._refusals: List[Admission] = []

    @property
    def admissions(self) -> List[Admission]:
        return list(self._admissions)

    @property
    def refusals(self) -> List[Admission]:
        return list(self._refusals)

    def admit(self, *, prediction_id: str, frame_id: int,
              source_frame_timestamp_s: float, now_s: float) -> Admission:
        if not isinstance(prediction_id, str) or not prediction_id.strip():
            raise StaleFrameRejected(REJECT_UNKNOWN_PREDICTION,
                                     "prediction_id must be a non-empty string")
        source = _seconds(source_frame_timestamp_s, "source_frame_timestamp_s")
        now = _seconds(now_s, "now_s")
        age = now - source
        admission = Admission(
            prediction_id=prediction_id, frame_id=int(frame_id),
            source_frame_timestamp_s=source, consumed_at_s=now,
            age_s=round(age, 9), threshold_s=self.threshold_s,
        )
        if age < -self.clock_tolerance_s:
            raise self._refuse(admission, REJECT_FUTURE_FRAME,
                               f"source frame is {-age:.6f} s in the FUTURE of now={now:.6f}s")
        if age > self.threshold_s + self.clock_tolerance_s:
            raise self._refuse(admission, REJECT_AGE_EXCEEDED,
                               f"source frame is {age:.6f} s old > threshold "
                               f"{self.threshold_s:.6f} s: a stale prediction must be "
                               "recomputed from a fresh frame, not reused")
        if prediction_id in self._last_admitted:
            first = self._last_admitted[prediction_id]
            raise self._refuse(admission, REJECT_FRAME_REUSE,
                               f"prediction {prediction_id!r} was already admitted at "
                               f"{first.consumed_at_s:.6f} s (age {first.age_s:.6f} s); "
                               "a consumed prediction is never silently re-used")
        self._last_admitted[prediction_id] = admission
        self._admissions.append(admission)
        return admission

    def _refuse(self, admission: Admission, reason: str, detail: str) -> StaleFrameRejected:
        refused = Admission(**{**admission.__dict__, "decision": "REFUSED:" + reason})
        self._refusals.append(refused)
        return StaleFrameRejected(reason, detail, admission=refused)

    def report(self) -> Dict[str, Any]:
        return {
            "schema": "badmin.p7.staleness_policy",
            "schema_version": 1,
            "threshold_s": self.threshold_s,
            "boundary": "age <= threshold admitted, age > threshold refused",
            "admitted": len(self._admissions),
            "refused": len(self._refusals),
            "refusals_by_reason": _count_by_reason(self._refusals),
            "evidence_class": "OFFLINE-SYNTHETIC",
        }


def _count_by_reason(rows: Sequence[Admission]) -> Dict[str, int]:
    counts: Dict[str, int] = {}
    for row in rows:
        reason = row.decision.split(":", 1)[-1]
        counts[reason] = counts.get(reason, 0) + 1
    return dict(sorted(counts.items()))
