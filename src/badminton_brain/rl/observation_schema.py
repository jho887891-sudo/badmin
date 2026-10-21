# -*- coding: utf-8 -*-
"""Candidate whole-body policy observation schema (P8-01) -- SPEC + STRICT ENCODER ONLY.

Scope
-----
Preparation for training, NOT training.  This module defines a declarative observation
specification and a strict encoder for it.  It contains no environment, no rollout loop and
no optimiser; it cannot execute a policy and it has no path to hardware.

Status
------
  SCHEMA_STATUS  = "CANDIDATE_NOT_PROJECT_API"
  TRAINING_STATUS = "PREPARATION_ONLY_NO_TRAINING"
docs/observation_action.md declares Observation/Action to be *project API* ("any change must
update that document").  This candidate schema deliberately does NOT edit that document; it
is a proposal tracked by management/morph_one_rs/P8_TRAINING_GATE.md until P8-06 is
authorised by the user.  The final project API still belongs to the document owner.

Design rules (each is enforced by a test under tests/badminton_brain/rl/)
------------------------------------------------------------------------
* Every channel declares its unit, so degrees-vs-radians and mm-vs-metres cannot pass.
* Every channel declares a maximum age, so a stale frame is an error, not a quiet reuse.
* Every channel declares bounds, so a kilometre-scaled position or an exploded covariance
  fails instead of entering a network.
* Missing keys, unknown keys, wrong shape/dtype, NaN/Inf, out-of-bounds values, stale or
  future timestamps and unit confusion all raise; the encoder never "repairs" an observation.
* Values are wrapped in ObservationValue(value, unit, timestamp_s): a bare ndarray is
  refused, because without the unit the encoder could not check unit confusion at all.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Dict, Iterable, Mapping, Optional, Sequence, Tuple

import numpy as np

OBSERVATION_SCHEMA_VERSION = "p8-obs-v0.1-candidate"
SCHEMA_STATUS = "CANDIDATE_NOT_PROJECT_API"
TRAINING_STATUS = "PREPARATION_ONLY_NO_TRAINING"

#: Future timestamps up to this much ahead of "now" are tolerated (clock jitter only).
CLOCK_TOLERANCE_S = 0.05
#: PIER-like / Morph One-like TEMP proxies, mirrored from src/badminton_brain/safety/safety_shield.py
#: SafetyLimits.temp_proxy() so that the schema cannot silently disagree with the shield.
TEMP_JOINT_LIMIT_RAD = 2.6
TEMP_WORKSPACE_MIN = (-2.60, -1.50, 0.15)
TEMP_WORKSPACE_MAX = (-0.40, 1.50, 2.20)
TEMP_COMMAND_TIMEOUT_S = 0.20


# --------------------------------------------------------------------------------------
# error taxonomy: every rejection path is named, so tests can assert on the *reason*
# --------------------------------------------------------------------------------------
class ObservationSchemaError(ValueError):
    """Base class for every strict-observation rejection."""


class MissingFieldError(ObservationSchemaError):
    """A required channel was not supplied."""


class UnknownFieldError(ObservationSchemaError):
    """An undeclared channel was supplied (typos must not be silently dropped)."""


class ObservationShapeError(ObservationSchemaError):
    """A channel has the wrong shape (including a wrong history length)."""


class ObservationDTypeError(ObservationSchemaError):
    """A channel has the wrong dtype (strings, complex, float-in-a-flag, ...)."""


class ObservationNonFiniteError(ObservationSchemaError):
    """A channel contains NaN or Inf."""


class ObservationBoundsError(ObservationSchemaError):
    """A channel is outside its declared bounds."""


class ObservationStalenessError(ObservationSchemaError):
    """A channel timestamp is stale, missing or in the future."""


class ObservationUnitError(ObservationSchemaError):
    """A channel was supplied in a different unit than declared."""


class ObservationBatchError(ObservationSchemaError):
    """Batched and unbatched channels were mixed, or batch sizes disagree."""


# --------------------------------------------------------------------------------------
# declarative specification
# --------------------------------------------------------------------------------------
_UNIT_ALIASES = {
    "radians": "rad", "radian": "rad", "degrees": "deg", "degree": "deg", "deg": "deg",
    "meters": "m", "meter": "m", "metres": "m", "metre": "m", "mm": "mm", "cm": "cm",
    "seconds": "s", "second": "s", "sec": "s", "s": "s",
    "rad/s": "rad/s", "rads/s": "rad/s", "m/s": "m/s", "meters/s": "m/s",
    "quat": "1", "quaternion": "1", "dimensionless": "1", "bool": "1",
}


def normalise_unit(unit: Any) -> str:
    """Canonicalise a unit string so 'radians' and 'rad' compare equal but 'deg' does not."""
    text = str(unit).strip().lower().replace(" ", "")
    return _UNIT_ALIASES.get(text, text)


#: kept for internal call sites; the public alias above is the shared one used by action_schema
_norm_unit = normalise_unit


def _uniform(lo: float, hi: float, n: int) -> Tuple[Tuple[float, ...], Tuple[float, ...]]:
    return (tuple([float(lo)] * n), tuple([float(hi)] * n))


def _box(lo: Sequence[float], hi: Sequence[float]) -> Tuple[Tuple[float, ...], Tuple[float, ...]]:
    return (tuple(float(v) for v in lo), tuple(float(v) for v in hi))


@dataclass(frozen=True)
class ObservationField:
    """One channel of the candidate observation vector.

    dtype  : "float64" or "bool" (bools are encoded 0.0/1.0 so the vector stays one array,
             but a float such as 0.5 is rejected -- a flag is not a probability).
    history: how many stacked frames the channel must carry (oldest -> newest).
    """
    name: str
    shape: Tuple[int, ...]
    dtype: str
    unit: str
    bounds: Optional[Tuple[Tuple[float, ...], Tuple[float, ...]]]
    source: str
    max_age_s: float
    history: int = 1
    description: str = ""

    @property
    def elements(self) -> int:
        return int(np.prod(self.shape)) if self.shape else 1

    @property
    def width(self) -> int:
        """Flattened width, including history."""
        return self.elements * self.history

    @property
    def expected_shape(self) -> Tuple[int, ...]:
        return (self.history,) + tuple(self.shape) if self.history > 1 else tuple(self.shape)


_J = TEMP_JOINT_LIMIT_RAD
_WMIN, _WMAX = TEMP_WORKSPACE_MIN, TEMP_WORKSPACE_MAX

OBSERVATION_FIELDS: Tuple[ObservationField, ...] = (
    ObservationField(
        "base_position", (3,), "float64", "m", _uniform(-20.0, 20.0, 3),
        "Estimation.UnifiedState.base_pose[0:3] (court frame)", TEMP_COMMAND_TIMEOUT_S,
        description="chassis position in the court frame"),
    ObservationField(
        "base_orientation", (4,), "float64", "1", _uniform(-1.0, 1.0, 4),
        "Estimation.UnifiedState.base_pose[3:7] (court frame)", TEMP_COMMAND_TIMEOUT_S,
        description="unit quaternion (w, x, y, z); 1 = dimensionless"),
    ObservationField(
        "base_linear_velocity", (3,), "float64", "m/s", _uniform(-2.0, 2.0, 3),
        "Estimation.UnifiedState.base_twist[0:3]", TEMP_COMMAND_TIMEOUT_S),
    ObservationField(
        "base_angular_velocity", (3,), "float64", "rad/s", _uniform(-4.0, 4.0, 3),
        "Estimation.UnifiedState.base_twist[3:6]", TEMP_COMMAND_TIMEOUT_S),
    ObservationField(
        "joint_position", (6,), "float64", "rad", _uniform(-_J, _J, 6),
        "Estimation.UnifiedState.joint_pos (PiPER 6-DoF)", TEMP_COMMAND_TIMEOUT_S),
    ObservationField(
        "joint_velocity", (6,), "float64", "rad/s", _uniform(-4.0, 4.0, 6),
        "Estimation.UnifiedState.joint_vel", TEMP_COMMAND_TIMEOUT_S),
    ObservationField(
        "racket_position", (3,), "float64", "m", _box(_WMIN, _WMAX),
        "Estimation.UnifiedState.racket_contact_pose[0:3]", TEMP_COMMAND_TIMEOUT_S,
        description="racket contact point; box is the TEMP safety workspace proxy"),
    ObservationField(
        "racket_orientation", (4,), "float64", "1", _uniform(-1.0, 1.0, 4),
        "Estimation.UnifiedState.racket_contact_pose[3:7]", TEMP_COMMAND_TIMEOUT_S),
    ObservationField(
        "racket_linear_velocity", (3,), "float64", "m/s", _uniform(-20.0, 20.0, 3),
        "Estimation.UnifiedState.racket_contact_twist[0:3]", TEMP_COMMAND_TIMEOUT_S),
    ObservationField(
        "shuttle_position", (3,), "float64", "m", _box((-20.0, -20.0, -1.0), (20.0, 20.0, 12.0)),
        "Estimation shuttle filter (UKF) position", 0.15,
        description="court-frame shuttle estimate"),
    ObservationField(
        "shuttle_velocity", (3,), "float64", "m/s", _uniform(-60.0, 60.0, 3),
        "Estimation shuttle filter (UKF) velocity", 0.15, history=3,
        description="3 stacked frames (oldest -> newest); an estimate, not ground truth"),
    ObservationField(
        "shuttle_covariance", (9,), "float64", "m^2", _uniform(-2500.0, 2500.0, 9),
        "Estimation shuttle filter (UKF) 3x3 covariance, row-major flattened", 0.15,
        description="uncertainty of the shuttle estimate; unit m^2"),
    ObservationField(
        "time_to_hit_s", (1,), "float64", "s", _uniform(0.0, 5.0, 1),
        "Prediction/Decision BestIntercept.time_s", 0.15,
        description="only meaningful when feasibility_flags[3] (prediction_valid) == 1"),
    ObservationField(
        "feasibility_flags", (4,), "bool", "1", _uniform(0.0, 1.0, 4),
        "Decision.feasibility", 0.15,
        description="[reachable, in_workspace, timing_ok, prediction_valid]"),
    ObservationField(
        "prev_base_linear_command", (2,), "float64", "m/s", _uniform(-0.60, 0.60, 2),
        "previous action, linear part (vx, vy) in the base frame", TEMP_COMMAND_TIMEOUT_S),
    ObservationField(
        "prev_base_angular_command", (1,), "float64", "rad/s", _uniform(-1.00, 1.00, 1),
        "previous action, yaw rate wz in the base frame", TEMP_COMMAND_TIMEOUT_S),
    ObservationField(
        "prev_joint_delta_command", (6,), "float64", "rad", _uniform(-0.04, 0.04, 6),
        "previous action, per-joint delta (2.0 rad/s * 0.02 s proxy)", TEMP_COMMAND_TIMEOUT_S),
)

OBSERVATION_BY_NAME: Dict[str, ObservationField] = {f.name: f for f in OBSERVATION_FIELDS}


def _build_layout() -> Tuple[Dict[str, slice], int]:
    layout: Dict[str, slice] = {}
    offset = 0
    for f in OBSERVATION_FIELDS:
        layout[f.name] = slice(offset, offset + f.width)
        offset += f.width
    return layout, offset


FIELD_SLICES, OBSERVATION_DIM = _build_layout()


def observation_dim() -> int:
    """Total flattened observation dimension (including declared history)."""
    return OBSERVATION_DIM


def field_names() -> Tuple[str, ...]:
    return tuple(f.name for f in OBSERVATION_FIELDS)


# --------------------------------------------------------------------------------------
# value wrapper
# --------------------------------------------------------------------------------------
@dataclass(frozen=True)
class ObservationValue:
    """A channel value that can never hide its unit or its age."""
    value: Any
    unit: str
    timestamp_s: float


# --------------------------------------------------------------------------------------
# checks
# --------------------------------------------------------------------------------------
def _check_unit(field: ObservationField, unit: Any) -> None:
    got = _norm_unit(unit)
    want = _norm_unit(field.unit)
    if got != want:
        hint = ""
        if got in ("deg", "degrees") and want == "rad":
            hint = " (degrees supplied where radians are declared: convert with deg2rad)"
        elif got == "mm" and want == "m":
            hint = " (millimetres supplied where metres are declared: divide by 1000)"
        elif got == "cm" and want == "m":
            hint = " (centimetres supplied where metres are declared: divide by 100)"
        raise ObservationUnitError(
            "{0}: unit confusion, declared unit is {1!r} but the value was supplied in "
            "{2!r}{3}".format(field.name, field.unit, unit, hint))


def _check_timestamp(field: ObservationField, timestamp_s: Any, *, now_s: Optional[float],
                     enforce_staleness: bool) -> None:
    if timestamp_s is None:
        raise ObservationStalenessError(
            "{0}: timestamp_s is required (an untimestamped channel cannot be proven fresh)"
            .format(field.name))
    try:
        ts = float(timestamp_s)
    except (TypeError, ValueError):
        raise ObservationStalenessError(
            "{0}: timestamp_s must be a number, got {1!r}".format(field.name, timestamp_s))
    if not math.isfinite(ts):
        raise ObservationStalenessError(
            "{0}: timestamp_s must be finite, got {1!r}".format(field.name, ts))
    if not enforce_staleness:
        return
    if now_s is None:
        raise ObservationSchemaError(
            "now_s is required when enforce_staleness=True: staleness cannot be checked "
            "without a reference time (pass now_s or enforce_staleness=False)")
    try:
        now = float(now_s)
    except (TypeError, ValueError):
        raise ObservationSchemaError("now_s must be a number, got {0!r}".format(now_s))
    if not math.isfinite(now):
        raise ObservationSchemaError("now_s must be finite, got {0!r}".format(now))
    age = now - ts
    if age < -CLOCK_TOLERANCE_S:
        raise ObservationStalenessError(
            "{0}: timestamp {1:.6f} s is {2:.6f} s in the future of now_s={3:.6f} s "
            "(tolerance {4} s)".format(field.name, ts, -age, now, CLOCK_TOLERANCE_S))
    if age > field.max_age_s:
        raise ObservationStalenessError(
            "{0}: stale channel, age {1:.6f} s exceeds max_age_s={2} s (now_s={3:.6f})"
            .format(field.name, age, field.max_age_s, now))


def _as_numeric(field: ObservationField, value: Any) -> np.ndarray:
    if isinstance(value, np.ndarray) and value.dtype.kind == "O":
        raise ObservationDTypeError("{0}: object arrays are not a channel value"
                                    .format(field.name))
    try:
        arr = np.asarray(value)
    except Exception as exc:  # pragma: no cover - numpy raises for exotic inputs
        raise ObservationDTypeError("{0}: not array-like ({1})".format(field.name, exc))
    if arr.dtype.kind == "c":
        raise ObservationDTypeError(
            "{0}: complex values are not an observation (dtype {1})".format(field.name, arr.dtype))
    if arr.dtype.kind not in ("f", "i", "u", "b"):
        raise ObservationDTypeError(
            "{0}: dtype {1} is not numeric; expected {2}".format(field.name, arr.dtype, field.dtype))
    if field.dtype == "float64":
        if arr.dtype.kind == "b":
            raise ObservationDTypeError(
                "{0}: bool values in a float channel; expected {1}".format(field.name, field.dtype))
    else:  # bool channel
        if arr.dtype.kind not in ("b", "i", "u"):
            raise ObservationDTypeError(
                "{0}: bool channel requires bool or integer 0/1, got dtype {1} "
                "(a flag is not a probability)".format(field.name, arr.dtype))
    return arr


def _coerce_field(field: ObservationField, raw: Any, *, now_s: Optional[float],
                  enforce_staleness: bool) -> Tuple[np.ndarray, bool]:
    if not isinstance(raw, ObservationValue):
        raise ObservationSchemaError(
            "{0}: value must be ObservationValue(value, unit, timestamp_s), got {1}. A bare "
            "array carries no unit, so unit confusion could not be detected"
            .format(field.name, type(raw).__name__))
    _check_unit(field, raw.unit)
    _check_timestamp(field, raw.timestamp_s, now_s=now_s, enforce_staleness=enforce_staleness)
    arr = _as_numeric(field, raw.value)

    expected = field.expected_shape
    batched = False
    if arr.shape == expected:
        pass
    elif arr.ndim == len(expected) + 1 and tuple(arr.shape[1:]) == expected and arr.shape[0] >= 1:
        batched = True
    else:
        extra = ""
        if field.history > 1:
            extra = (" : this channel declares history={0}, so the value must be "
                     "{1} (or batched (N,)+{1}), never a single frame"
                     .format(field.history, expected))
        raise ObservationShapeError(
            "{0}: shape {1} does not match declared {2}{3}".format(field.name, arr.shape, expected, extra))

    values = arr.astype(np.float64, copy=False)
    if not np.all(np.isfinite(values)):
        bad = int(np.size(values) - np.count_nonzero(np.isfinite(values)))
        raise ObservationNonFiniteError(
            "{0}: {1} non-finite entr{2} (NaN/Inf); the encoder refuses to train on them"
            .format(field.name, bad, "y" if bad == 1 else "ies"))

    if field.dtype == "bool":
        allowed = (values == 0.0) | (values == 1.0)
        if not bool(np.all(allowed)):
            raise ObservationDTypeError(
                "{0}: bool channel contains a non-binary value ({1!r}); flags must be exactly "
                "0 or 1".format(field.name, float(values[~allowed].ravel()[0])))

    if field.bounds is not None:
        lo = np.tile(np.asarray(field.bounds[0], dtype=float), field.history)
        hi = np.tile(np.asarray(field.bounds[1], dtype=float), field.history)
        flat = values.reshape(values.shape[0], -1) if batched else values.reshape(-1)
        below = flat < lo
        above = flat > hi
        if bool(np.any(below)) or bool(np.any(above)):
            mask = (below | above).ravel()
            first_pos = int(np.argwhere(mask)[0][0])
            raise ObservationBoundsError(
                "{0}: value {1!r} at channel offset {2} is outside the declared bounds of this "
                "channel [{3!r}, {4!r}] (per-element limits apply); check the unit and the source"
                .format(field.name, float(flat.ravel()[first_pos]), first_pos,
                        float(lo.min()), float(hi.max())))

    if batched:
        return values.reshape(values.shape[0], -1), True
    return values.reshape(-1), False


def _coerce_fields(values: Mapping[str, Any], *, now_s: Optional[float],
                   enforce_staleness: bool) -> Tuple[np.ndarray, Optional[int], Tuple[str, ...]]:
    if not isinstance(values, Mapping):
        raise ObservationSchemaError(
            "build_observation expects a mapping of channel name -> ObservationValue, got {0}"
            .format(type(values).__name__))
    missing = [f.name for f in OBSERVATION_FIELDS if f.name not in values]
    if missing:
        raise MissingFieldError("missing observation channel(s): " + ", ".join(missing))
    unknown = sorted(set(values) - set(OBSERVATION_BY_NAME))
    if unknown:
        raise UnknownFieldError(
            "undeclared observation channel(s): " + ", ".join(unknown)
            + "; the schema is closed so a typo cannot silently shrink the observation")

    columns = []
    batch_sizes = []
    for field in OBSERVATION_FIELDS:
        flat, batched = _coerce_field(field, values[field.name], now_s=now_s,
                                      enforce_staleness=enforce_staleness)
        columns.append(flat)
        if batched:
            batch_sizes.append(int(flat.shape[0]))
    if batch_sizes and len(batch_sizes) != len(OBSERVATION_FIELDS):
        raise ObservationBatchError(
            "mixed batched/unbatched channels: only {0} of {1} channels carry a batch dimension"
            .format(len(batch_sizes), len(OBSERVATION_FIELDS)))
    if batch_sizes and len(set(batch_sizes)) > 1:
        raise ObservationBatchError(
            "inconsistent batch sizes {0}".format(sorted(set(batch_sizes))))
    batch = batch_sizes[0] if batch_sizes else None
    if batch is None:
        return np.concatenate(columns).astype(np.float64), None, field_names()
    return np.concatenate(columns, axis=1).astype(np.float64), batch, field_names()


# --------------------------------------------------------------------------------------
# public API
# --------------------------------------------------------------------------------------
def build_observation(values: Mapping[str, ObservationValue], *, now_s: Optional[float] = None,
                      enforce_staleness: bool = True) -> np.ndarray:
    """Encode a strict observation mapping into one float64 vector.

    Returns shape (OBSERVATION_DIM,) -- or (N, OBSERVATION_DIM) when every channel is batched.
    Raises ObservationSchemaError subclasses on every violation; it never clamps or fills.
    """
    obs, _batch, _names = _coerce_fields(values, now_s=now_s, enforce_staleness=enforce_staleness)
    return obs


@dataclass(frozen=True)
class ValidationReport:
    """What validate() actually proved; returned so callers cannot claim more than that."""
    schema_version: str
    dim: int
    batch_size: Optional[int]
    fields_checked: Tuple[str, ...]
    checks: Tuple[str, ...]
    staleness_enforced: bool
    training_status: str = TRAINING_STATUS


_CHECK_NAMES = ("keys", "units", "timestamps", "shape", "dtype", "finite", "bounds")


def validate(values_or_array: Any, *, now_s: Optional[float] = None,
             enforce_staleness: bool = True) -> ValidationReport:
    """Strictly validate either a channel mapping or an already-built observation array.

    Mapping input: runs the full strict encoder (keys, units, timestamps, shape, dtype,
    finite, bounds) and returns a report.  Array input: checks dim, dtype (float64 exactly),
    finiteness and per-channel bounds, then returns a report.
    """
    if isinstance(values_or_array, Mapping):
        obs, batch, names = _coerce_fields(values_or_array, now_s=now_s,
                                           enforce_staleness=enforce_staleness)
        return ValidationReport(
            schema_version=OBSERVATION_SCHEMA_VERSION,
            dim=int(obs.shape[-1]),
            batch_size=batch,
            fields_checked=names,
            checks=_CHECK_NAMES,
            staleness_enforced=bool(enforce_staleness),
        )

    arr = np.asarray(values_or_array)
    if arr.dtype != np.dtype("float64"):
        raise ObservationDTypeError(
            "observation array dtype must be float64, got {0} (float32 would change the "
            "bounds arithmetic silently)".format(arr.dtype))
    if arr.ndim == 1:
        if arr.shape[0] != OBSERVATION_DIM:
            raise ObservationShapeError(
                "observation length {0} does not match schema dim {1}"
                .format(arr.shape[0], OBSERVATION_DIM))
        batch = None
    elif arr.ndim == 2:
        if arr.shape[1] != OBSERVATION_DIM:
            raise ObservationShapeError(
                "observation width {0} does not match schema dim {1}"
                .format(arr.shape[1], OBSERVATION_DIM))
        if arr.shape[0] < 1:
            raise ObservationBatchError("batched observation needs at least one row")
        batch = int(arr.shape[0])
    else:
        raise ObservationShapeError(
            "observation must be (dim,) or (N, dim), got shape {0}".format(arr.shape))
    if not np.all(np.isfinite(arr)):
        raise ObservationNonFiniteError(
            "observation contains NaN/Inf ({0} entries)".format(int(np.size(arr) - np.count_nonzero(np.isfinite(arr)))))
    rows = arr.reshape(-1, arr.shape[-1])
    for field in OBSERVATION_FIELDS:
        if field.bounds is None:
            continue
        lo = np.tile(np.asarray(field.bounds[0], dtype=float), field.history)
        hi = np.tile(np.asarray(field.bounds[1], dtype=float), field.history)
        block = rows[:, FIELD_SLICES[field.name]]
        if bool(np.any(block < lo)) or bool(np.any(block > hi)):
            raise ObservationBoundsError(
                "{0}: built observation slice violates declared bounds".format(field.name))
    return ValidationReport(
        schema_version=OBSERVATION_SCHEMA_VERSION,
        dim=int(arr.shape[-1]),
        batch_size=batch,
        fields_checked=field_names(),
        checks=("dim", "dtype", "finite", "bounds"),
        staleness_enforced=False,
    )


def schema_rows() -> Tuple[Dict[str, Any], ...]:
    """Declarative table (for the gate document / audit); names, shapes, units, sources."""
    rows = []
    for f in OBSERVATION_FIELDS:
        rows.append({
            "name": f.name, "shape": f.shape, "dtype": f.dtype, "unit": f.unit,
            "bounds": f.bounds, "source": f.source, "max_age_s": f.max_age_s,
            "history": f.history, "slice": FIELD_SLICES[f.name], "description": f.description,
        })
    return tuple(rows)


__all__ = [
    "OBSERVATION_SCHEMA_VERSION", "SCHEMA_STATUS", "TRAINING_STATUS", "CLOCK_TOLERANCE_S",
    "ObservationField", "ObservationValue", "OBSERVATION_FIELDS", "OBSERVATION_BY_NAME",
    "OBSERVATION_DIM", "FIELD_SLICES", "observation_dim", "field_names", "build_observation",
    "validate", "ValidationReport", "schema_rows",
    "ObservationSchemaError", "MissingFieldError", "UnknownFieldError", "ObservationShapeError",
    "ObservationDTypeError", "ObservationNonFiniteError", "ObservationBoundsError",
    "ObservationStalenessError", "ObservationUnitError", "ObservationBatchError",
]
