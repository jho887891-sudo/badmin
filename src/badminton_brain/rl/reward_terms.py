# -*- coding: utf-8 -*-
"""Candidate reward terms with OFFLINE computability semantics (P8-02).

This module only *scores recorded frames*.  It has no environment, no simulator import, no
optimiser and no loop: a term is a pure function of a recorded Frame, so any historical
frame that was recorded once can be re-scored today, on a CPU, without Isaac and without
replaying the episode.

The one rule that matters for honest offline evaluation
------------------------------------------------------
Missing data is MASKED, never zero.  A term that cannot be evaluated returns
TermResult(value=None) with a reason.  A term whose input was actually measured as "nothing
happened" returns TermResult(value=0.0).  Conflating those two is how offline reward
verification silently becomes fiction: a zero would look like a measured miss and would
drag the average down, while a mask says "this frame cannot answer this question".

Every term here is a CANDIDATE (P8-02).  None of them is registered as a project reward, no
weight has been tuned, and docs/reward_design.md (project API) is deliberately untouched.
TRAINING_STATUS = "PREPARATION_ONLY_NO_TRAINING": no PPO, no training run, no GPU.
"""
from __future__ import annotations

import hashlib
import math
from dataclasses import dataclass, fields as dataclass_fields
from typing import Any, Callable, Dict, Mapping, Optional, Tuple

import numpy as np

REWARD_SCHEMA_VERSION = "p8-rew-v0.1-candidate"
TRAINING_STATUS = "PREPARATION_ONLY_NO_TRAINING"
SCHEMA_STATUS = "CANDIDATE_NOT_PROJECT_API"
TERM_PHASE = "p8-02-offline-candidate"


# --------------------------------------------------------------------------------------
# recorded frame
# --------------------------------------------------------------------------------------
@dataclass(frozen=True)
class Frame:
    """One recorded frame.  Frozen: a reward term can never edit the record it scores.

    Every field is optional on purpose: an offline dataset really does have holes, and the
    holes must be visible as masks rather than as fabricated zeros.
    """
    t_s: Optional[float] = None
    shuttle_position: Optional[Any] = None
    shuttle_velocity: Optional[Any] = None
    racket_position: Optional[Any] = None
    racket_velocity: Optional[Any] = None
    intercept_point: Optional[Any] = None
    contact_detected: Optional[bool] = None
    collision_impulse_ns: Optional[float] = None
    joint_limit_excess_rad: Optional[float] = None
    base_tilt_rad: Optional[float] = None


def frame_fingerprint(frame: Frame) -> str:
    """Stable digest of a frame, used to prove term purity in tests and reports."""
    h = hashlib.sha256()
    for f in dataclass_fields(Frame):
        value = getattr(frame, f.name)
        h.update(f.name.encode("utf-8"))
        if value is None:
            h.update(b"<none>")
        elif isinstance(value, (bool, int, float)):
            h.update(repr(float(value)).encode("utf-8"))
        else:
            arr = np.asarray(value, dtype=float)
            h.update(str(arr.shape).encode("utf-8"))
            h.update(np.ascontiguousarray(arr).tobytes())
        h.update(b"|")
    return h.hexdigest()


# --------------------------------------------------------------------------------------
# results
# --------------------------------------------------------------------------------------
@dataclass(frozen=True)
class TermResult:
    """Either a measured number or an explicit mask.  A third state does not exist."""
    name: str
    value: Optional[float] = None
    reason: str = ""
    masked: bool = False

    def __post_init__(self) -> None:
        if self.value is None and not self.masked:
            raise ValueError("{0}: value=None requires masked=True (missing data must be "
                             "labelled, never implied)".format(self.name))
        if self.value is not None and self.masked:
            raise ValueError("{0}: value and masked=True are contradictory".format(self.name))
        if self.value is not None:
            numeric = float(self.value)
            if not math.isfinite(numeric):
                raise ValueError("{0}: reward term produced {1!r}; a NaN/Inf term must be "
                                 "masked instead".format(self.name, self.value))
            object.__setattr__(self, "value", numeric)

    @classmethod
    def masked_result(cls, name: str, reason: str) -> "TermResult":
        return cls(name=name, value=None, reason=reason, masked=True)

    @classmethod
    def measured(cls, name: str, value: float, reason: str = "") -> "TermResult":
        return cls(name=name, value=float(value), reason=reason, masked=False)


# --------------------------------------------------------------------------------------
# parameters (TEMP placeholders -- not tuned, not measured)
# --------------------------------------------------------------------------------------
@dataclass(frozen=True)
class RewardParams:
    distance_scale_m: float = 0.50
    velocity_scale_mps: float = 1.00
    collision_impulse_threshold_ns: float = 0.50
    base_tilt_threshold_rad: float = 0.20
    timeout_limit_s: float = 6.00
    param_source: str = "TEMP_PARAMETERIZED_PROXY: P8-02 candidate scales, no tuning performed"

    @classmethod
    def temp_proxy(cls) -> "RewardParams":
        return cls()


@dataclass(frozen=True)
class RewardWeights:
    distance: float = 1.0
    contact: float = 5.0
    velocity: float = 0.5
    collision: float = 2.0
    joint_limit: float = 1.0
    instability: float = 1.0
    timeout: float = 1.0
    weight_source: str = "PLACEHOLDER: P8-02 candidate weights, NOT tuned, NOT validated"

    @classmethod
    def temp_proxy(cls) -> "RewardWeights":
        return cls()


# --------------------------------------------------------------------------------------
# helpers
# --------------------------------------------------------------------------------------
def _vector(frame: Frame, name: str, size: int) -> Tuple[Optional[np.ndarray], str]:
    """Return (vector, "") or (None, reason).  Corrupt data raises instead of masking."""
    raw = getattr(frame, name)
    if raw is None:
        return None, "{} not recorded in this frame".format(name)
    arr = np.asarray(raw, dtype=float)
    if arr.shape != (size,):
        raise ValueError("{}: recorded shape {} is not ({},); a corrupt frame is an error, "
                         "not a mask".format(name, arr.shape, size))
    if not np.all(np.isfinite(arr)):
        raise ValueError("{}: recorded value contains NaN/Inf; a corrupt frame is an error, "
                         "not a mask".format(name))
    return arr, ""


def _scalar(frame: Frame, name: str) -> Tuple[Optional[float], str]:
    raw = getattr(frame, name)
    if raw is None:
        return None, "{} not recorded in this frame".format(name)
    value = float(raw)
    if not math.isfinite(value):
        raise ValueError("{}: recorded value is not finite; a corrupt frame is an error, not "
                         "a mask".format(name))
    return value, ""


# --------------------------------------------------------------------------------------
# candidate terms (pure functions of a recorded frame)
# --------------------------------------------------------------------------------------
def distance_to_intercept(frame: Frame, params: RewardParams = RewardParams()) -> TermResult:
    """-||shuttle_position - intercept_point|| / scale.  Dense shaping term."""
    name = "distance_to_intercept"
    shuttle, why_1 = _vector(frame, "shuttle_position", 3)
    target, why_2 = _vector(frame, "intercept_point", 3)
    if shuttle is None or target is None:
        return TermResult.masked_result(name, why_1 or why_2)
    distance = float(np.linalg.norm(shuttle - target))
    return TermResult.measured(name, -distance / float(params.distance_scale_m),
                               "distance={:.6f} m, scale={:.3f} m".format(
                                   distance, params.distance_scale_m))


def contact_success(frame: Frame, params: RewardParams = RewardParams()) -> TermResult:
    """+1.0 when the recorded frame reports racket-shuttle contact, 0.0 when it reports none.

    contact_detected = False is a MEASUREMENT (the contact sensor answered "no"), so it
    scores 0.0.  contact_detected = None means the sensor was not recorded, so the term is
    masked -- NOT 0.0.
    """
    name = "contact_success"
    if frame.contact_detected is None:
        return TermResult.masked_result(name, "contact_detected not recorded in this frame")
    if frame.contact_detected:
        return TermResult.measured(name, 1.0, "contact_detected=True")
    return TermResult.measured(name, 0.0, "contact_detected=False (measured, not a mask)")


def velocity_matching(frame: Frame, params: RewardParams = RewardParams()) -> TermResult:
    """-||racket_velocity - shuttle_velocity|| / scale; rewards matching the shuttle's velocity."""
    name = "velocity_matching"
    racket, why_1 = _vector(frame, "racket_velocity", 3)
    shuttle, why_2 = _vector(frame, "shuttle_velocity", 3)
    if racket is None or shuttle is None:
        return TermResult.masked_result(name, why_1 or why_2)
    mismatch = float(np.linalg.norm(racket - shuttle))
    return TermResult.measured(name, -mismatch / float(params.velocity_scale_mps),
                               "velocity mismatch={:.6f} m/s".format(mismatch))


def collision_penalty(frame: Frame, params: RewardParams = RewardParams()) -> TermResult:
    """-max(0, collision_impulse - threshold).  Zero impulse is measured, so 0.0 not masked."""
    name = "collision_penalty"
    impulse, why = _scalar(frame, "collision_impulse_ns")
    if impulse is None:
        return TermResult.masked_result(name, why)
    excess = max(0.0, impulse - float(params.collision_impulse_threshold_ns))
    return TermResult.measured(name, -excess,
                               "impulse={:.6f} N.s, threshold={:.3f} N.s".format(
                                   impulse, params.collision_impulse_threshold_ns))


def joint_limit_penalty(frame: Frame, params: RewardParams = RewardParams()) -> TermResult:
    """-joint_limit_excess_rad.  A recorded 0.0 means "checked, no excess"."""
    name = "joint_limit_penalty"
    excess, why = _scalar(frame, "joint_limit_excess_rad")
    if excess is None:
        return TermResult.masked_result(name, why)
    return TermResult.measured(name, -max(0.0, excess),
                               "joint limit excess={:.6f} rad".format(excess))


def instability_penalty(frame: Frame, params: RewardParams = RewardParams()) -> TermResult:
    """-max(0, base_tilt - threshold): chassis tilt beyond the proxy stability envelope."""
    name = "instability_penalty"
    tilt, why = _scalar(frame, "base_tilt_rad")
    if tilt is None:
        return TermResult.masked_result(name, why)
    excess = max(0.0, tilt - float(params.base_tilt_threshold_rad))
    return TermResult.measured(name, -excess,
                               "base tilt={:.6f} rad, threshold={:.3f} rad".format(
                                   tilt, params.base_tilt_threshold_rad))


def timeout_penalty(frame: Frame, params: RewardParams = RewardParams()) -> TermResult:
    """-1.0 once the recorded episode time passes the limit, else 0.0 (time is measured)."""
    name = "timeout_penalty"
    t_s, why = _scalar(frame, "t_s")
    if t_s is None:
        return TermResult.masked_result(name, why)
    if t_s >= float(params.timeout_limit_s):
        return TermResult.measured(name, -1.0, "t_s={:.6f} s >= limit {:.3f} s".format(
            t_s, params.timeout_limit_s))
    return TermResult.measured(name, 0.0, "t_s={:.6f} s < limit {:.3f} s".format(
        t_s, params.timeout_limit_s))


# --------------------------------------------------------------------------------------
# registry / aggregation
# --------------------------------------------------------------------------------------
@dataclass(frozen=True)
class RewardTermSpec:
    name: str
    fn: Callable[[Frame, RewardParams], TermResult]
    weight_key: str
    unit: str
    description: str


REWARD_TERMS: Tuple[RewardTermSpec, ...] = (
    RewardTermSpec("distance_to_intercept", distance_to_intercept, "distance", "1/m",
                   "dense shaping toward the intercept point"),
    RewardTermSpec("contact_success", contact_success, "contact", "1",
                   "sparse contact/success reward"),
    RewardTermSpec("velocity_matching", velocity_matching, "velocity", "s/m",
                   "racket velocity vs shuttle velocity match"),
    RewardTermSpec("collision_penalty", collision_penalty, "collision", "N.s",
                   "collision impulse above the proxy threshold"),
    RewardTermSpec("joint_limit_penalty", joint_limit_penalty, "joint_limit", "rad",
                   "joint range excess"),
    RewardTermSpec("instability_penalty", instability_penalty, "instability", "rad",
                   "chassis tilt beyond the proxy stability envelope"),
    RewardTermSpec("timeout_penalty", timeout_penalty, "timeout", "1",
                   "episode time beyond the recorded limit"),
)

REWARD_TERM_NAMES: Tuple[str, ...] = tuple(t.name for t in REWARD_TERMS)


@dataclass(frozen=True)
class RewardBreakdown:
    """Per-term results plus the weighted sum of the terms that could be evaluated."""
    results: Tuple[TermResult, ...]
    total: float
    masked_terms: Tuple[str, ...]
    complete: bool
    weights_source: str
    params_source: str

    def by_name(self) -> Dict[str, TermResult]:
        return {r.name: r for r in self.results}

    def weighted(self, weights: RewardWeights) -> Dict[str, float]:
        return {r.name: getattr(weights, spec.weight_key) * float(r.value)
                for r, spec in zip(self.results, REWARD_TERMS) if not r.masked}


def total_reward(frame: Frame, weights: RewardWeights = RewardWeights.temp_proxy(),
                 params: RewardParams = RewardParams.temp_proxy()) -> RewardBreakdown:
    """Weighted sum over evaluable terms; masked terms are excluded and named explicitly."""
    results = []
    total = 0.0
    masked = []
    for spec in REWARD_TERMS:
        result = spec.fn(frame, params)
        results.append(result)
        if result.masked:
            masked.append(result.name)
        else:
            total += getattr(weights, spec.weight_key) * float(result.value)
    return RewardBreakdown(
        results=tuple(results),
        total=float(total),
        masked_terms=tuple(masked),
        complete=(not masked),
        weights_source=weights.weight_source,
        params_source=params.param_source,
    )


@dataclass(frozen=True)
class ComputabilityRow:
    term: str
    per_frame: Tuple[str, ...]
    computable_frames: int
    masked_frames: int

    @property
    def all_computable(self) -> bool:
        return self.masked_frames == 0


def offline_computability_report(
        frames: Tuple[Frame, ...],
        params: RewardParams = RewardParams.temp_proxy(),
) -> Tuple[ComputabilityRow, ...]:
    """Offline computability matrix: for every candidate term, every recorded frame.

    This is the P8-02 deliverable: it proves each term is computable from recorded state on
    a CPU with no simulator, and it makes every mask visible with its reason.
    """
    rows = []
    for spec in REWARD_TERMS:
        cells = []
        computable = 0
        for frame in frames:
            result = spec.fn(frame, params)
            if result.masked:
                cells.append("masked: " + result.reason)
            else:
                computable += 1
                cells.append("{:.6f} ({})".format(float(result.value), result.reason))
        rows.append(ComputabilityRow(term=spec.name, per_frame=tuple(cells),
                                     computable_frames=computable,
                                     masked_frames=len(frames) - computable))
    return tuple(rows)


__all__ = [
    "REWARD_SCHEMA_VERSION", "TRAINING_STATUS", "SCHEMA_STATUS", "TERM_PHASE",
    "Frame", "frame_fingerprint", "TermResult", "RewardParams", "RewardWeights",
    "RewardTermSpec", "REWARD_TERMS", "REWARD_TERM_NAMES", "RewardBreakdown", "total_reward",
    "distance_to_intercept", "contact_success", "velocity_matching", "collision_penalty",
    "joint_limit_penalty", "instability_penalty", "timeout_penalty",
    "ComputabilityRow", "offline_computability_report",
]
