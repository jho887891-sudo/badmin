# -*- coding: utf-8 -*-
"""Candidate whole-body policy action schema (P8-01) -- SIMULATION-ONLY INTERFACE.

!!! THIS ACTION INTERFACE IS SIMULATION-ONLY !!!
------------------------------------------------
Every object produced here is tagged simulation_only=True and target="SIMULATOR_ONLY".
SimulatedCommand.to_real_robot() raises RealRobotActionForbidden; there is no driver, no
topic, no serial port and no vendor SDK in this module, so no code path in this file can
move a physical robot.  P8-07 (real-hardware risk list, e-stop, watchdog, speed limits) is
documentation-only and PPO/training is NOT AUTHORISED (P8-06 needs explicit user permission
plus the P5/P6/P7 gates).

Action layout (9 required dims, matching docs/observation_action.md placeholder)
--------------------------------------------------------------------------------
  base linear velocity command   (2,) = (vx, vy)   [m/s]   body frame
  base angular velocity command  (1,) = (wz)       [rad/s] body frame
  arm joint position delta       (6,) = (dq1..dq6) [rad]   per control step
  optional racket target         (3,) position + (4,) orientation [m, dimensionless]

Strictness: every violation raises a named ActionSchemaError subclass.  Nothing is clamped
here on purpose -- clamping would silently hide a policy operating outside its envelope.
(The Safety shield is the component allowed to clamp, and only inside the simulator.)
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Dict, Mapping, Optional, Sequence, Tuple

import numpy as np

from ..status import AssetStatus, Param
from .observation_schema import normalise_unit

ACTION_SCHEMA_VERSION = "p8-act-v0.1-candidate"
SCHEMA_STATUS = "CANDIDATE_NOT_PROJECT_API"
TRAINING_STATUS = "PREPARATION_ONLY_NO_TRAINING"
SIMULATION_ONLY = True
TARGET = "SIMULATOR_ONLY"
DEFAULT_DT_S = 0.02  # 50 Hz placeholder control period (same proxy as SafetyLimits.control_dt)

REAL_ROBOT_FORBIDDEN_REASON = (
    "Refusing to send an action to a physical robot. This action schema is a SIMULATION-ONLY "
    "training-preparation interface (P8-01): it has no hardware driver, no e-stop, no watchdog "
    "and no measured Morph One / PiPER limits (all values are TEMP_PARAMETERIZED_PROXY). Real-"
    "robot execution is NOT AUTHORISED; P8-07 risk documentation and a separate explicit user "
    "authorisation are required first."
)


class ActionSchemaError(ValueError):
    """Base class for every strict-action rejection."""


class MissingActionFieldError(ActionSchemaError):
    """A required action component was not supplied."""


class UnknownActionFieldError(ActionSchemaError):
    """An undeclared action component was supplied."""


class ActionShapeError(ActionSchemaError):
    """An action component has the wrong shape."""


class ActionDTypeError(ActionSchemaError):
    """An action component has the wrong dtype."""


class ActionNonFiniteError(ActionSchemaError):
    """An action component contains NaN or Inf."""


class ActionBoundsError(ActionSchemaError):
    """An action component is outside its absolute bounds."""


class ActionUnitError(ActionSchemaError):
    """An action component was supplied in a different unit than declared."""


class ActionRateLimitError(ActionSchemaError):
    """The step-to-step change of the command exceeds the declared rate limit."""


class ActionContextError(ActionSchemaError):
    """Missing/invalid context (joint positions, dt, ...) needed to build a command."""


class RealRobotActionForbidden(RuntimeError):
    """Raised by every attempt to route a simulated action to real hardware."""


# --------------------------------------------------------------------------------------
# limits: TEMP proxies, each carrying its own provenance (S4 / S12)
# --------------------------------------------------------------------------------------
def _temp(value: Any, source: str) -> Param:
    return Param(value=value, status=AssetStatus.TEMP_PARAMETERIZED_PROXY, source=source)


@dataclass
class ActionLimits:
    """Simulation-only action envelope; nothing here is measured on hardware."""
    base_linear_velocity_max: Param
    base_angular_velocity_max: Param
    joint_velocity_max: Param
    base_linear_accel_max: Param
    base_angular_accel_max: Param
    joint_accel_max: Param
    control_dt_max: Param
    joint_position_min: Param
    joint_position_max: Param
    racket_workspace_min: Param
    racket_workspace_max: Param

    @classmethod
    def temp_proxy(cls) -> "ActionLimits":
        return cls(
            base_linear_velocity_max=_temp(0.60, "Morph One-like 0.6 m/s, mirrors SafetyLimits.temp_proxy"),
            base_angular_velocity_max=_temp(1.00, "Morph One-like 1.0 rad/s, mirrors SafetyLimits.temp_proxy"),
            joint_velocity_max=_temp(2.00, "PiPER-like 2.0 rad/s, mirrors SafetyLimits.temp_proxy"),
            base_linear_accel_max=_temp(0.50, "PLACEHOLDER chassis linear acceleration 0.5 m/s^2 (unmeasured)"),
            base_angular_accel_max=_temp(1.00, "PLACEHOLDER chassis yaw acceleration 1.0 rad/s^2 (unmeasured)"),
            joint_accel_max=_temp(2.00, "PLACEHOLDER joint acceleration 2.0 rad/s^2 (unmeasured)"),
            control_dt_max=_temp(0.10, "PLACEHOLDER max control period 100 ms (unmeasured)"),
            joint_position_min=_temp(np.full(6, -2.60), "PiPER-like +-2.6 rad joint range proxy"),
            joint_position_max=_temp(np.full(6, 2.60), "PiPER-like +-2.6 rad joint range proxy"),
            racket_workspace_min=_temp(np.array([-2.60, -1.50, 0.15]), "workspace box proxy (court frame)"),
            racket_workspace_max=_temp(np.array([-0.40, 1.50, 2.20]), "workspace box proxy (court frame)"),
        )


DEFAULT_LIMITS = ActionLimits.temp_proxy()


@dataclass(frozen=True)
class ActionField:
    name: str
    shape: Tuple[int, ...]
    dtype: str
    unit: str
    bounds: Tuple[Tuple[float, ...], Tuple[float, ...]]
    optional: bool
    source: str
    description: str = ""

    @property
    def elements(self) -> int:
        return int(np.prod(self.shape))


def action_fields(limits: ActionLimits = DEFAULT_LIMITS, dt_s: float = DEFAULT_DT_S) -> Tuple[ActionField, ...]:
    """Declarative action spec for one control step of length dt_s."""
    lin = float(limits.base_linear_velocity_max.value)
    ang = float(limits.base_angular_velocity_max.value)
    dq = float(limits.joint_velocity_max.value) * float(dt_s)
    wmin = np.asarray(limits.racket_workspace_min.value, dtype=float)
    wmax = np.asarray(limits.racket_workspace_max.value, dtype=float)
    return (
        ActionField("base_linear_velocity", (2,), "float64", "m/s",
                    (tuple([-lin] * 2), tuple([lin] * 2)), False,
                    "chassis (vx, vy) command, robot base frame",
                    "body-frame planar velocity command, |v| bounded by base_linear_velocity_max"),
        ActionField("base_angular_velocity", (1,), "float64", "rad/s",
                    (tuple([-ang]), tuple([ang])), False,
                    "chassis yaw-rate command wz, robot base frame",
                    "body-frame yaw rate, bounded by base_angular_velocity_max"),
        ActionField("joint_position_delta", (6,), "float64", "rad",
                    (tuple([-dq] * 6), tuple([dq] * 6)), False,
                    "per-joint position delta for one control step (<= joint_velocity_max * dt)",
                    "position delta, not a velocity and not an absolute target"),
        ActionField("racket_target_position", (3,), "float64", "m",
                    (tuple(float(v) for v in wmin), tuple(float(v) for v in wmax)), True,
                    "OPTIONAL racket contact-point target; workspace box proxy, court frame",
                    "optional; when supplied the orientation must be supplied too"),
        ActionField("racket_target_orientation", (4,), "float64", "1",
                    (tuple([-1.0] * 4), tuple([1.0] * 4)), True,
                    "OPTIONAL unit quaternion (w, x, y, z) for the racket; must accompany the position",
                    "optional; pair of position+orientation, both or neither"),
    )


ACTION_FIELDS: Tuple[ActionField, ...] = action_fields()
ACTION_BY_NAME: Dict[str, ActionField] = {f.name: f for f in ACTION_FIELDS}
REQUIRED_ACTION_NAMES: Tuple[str, ...] = tuple(f.name for f in ACTION_FIELDS if not f.optional)
OPTIONAL_ACTION_NAMES: Tuple[str, ...] = tuple(f.name for f in ACTION_FIELDS if f.optional)


def action_dim(include_optional: bool = False) -> int:
    """Required action dimension (9) or 9 + 7 = 16 with the optional racket target."""
    return sum(f.elements for f in ACTION_FIELDS if include_optional or not f.optional)


@dataclass(frozen=True)
class ActionValue:
    """One action component with its declared unit (so deg-vs-rad cannot pass)."""
    value: Any
    unit: str


# --------------------------------------------------------------------------------------
# strict validation
# --------------------------------------------------------------------------------------
def _check_unit(field: ActionField, unit: Any) -> None:
    got = normalise_unit(unit)
    want = normalise_unit(field.unit)
    if got != want:
        hint = ""
        if got == "deg" and want == "rad":
            hint = " (degrees supplied where radians are declared)"
        elif got == "mm" and want == "m":
            hint = " (millimetres supplied where metres are declared)"
        raise ActionUnitError("{0}: unit confusion, declared {1!r}, got {2!r}{3}"
                              .format(field.name, field.unit, unit, hint))


def _as_numeric(field: ActionField, value: Any) -> np.ndarray:
    try:
        arr = np.asarray(value)
    except Exception as exc:  # pragma: no cover
        raise ActionDTypeError("{0}: not array-like ({1})".format(field.name, exc))
    if arr.dtype.kind == "c":
        raise ActionDTypeError("{0}: complex values are not an action".format(field.name))
    if arr.dtype.kind not in ("f", "i", "u"):
        raise ActionDTypeError("{0}: dtype {1} is not numeric float; expected {2}"
                               .format(field.name, arr.dtype, field.dtype))
    return arr


def _coerce_action_field(field: ActionField, raw: Any) -> Optional[np.ndarray]:
    if raw is None:
        if field.optional:
            return None
        raise MissingActionFieldError("{0}: required action component is None".format(field.name))
    if not isinstance(raw, ActionValue):
        raise ActionSchemaError(
            "{0}: value must be ActionValue(value, unit), got {1} (a bare array carries no unit)"
            .format(field.name, type(raw).__name__))
    _check_unit(field, raw.unit)
    arr = _as_numeric(field, raw.value)
    if arr.shape != field.shape:
        raise ActionShapeError("{0}: shape {1} does not match declared {2}"
                               .format(field.name, arr.shape, field.shape))
    values = arr.astype(np.float64, copy=False)
    if not np.all(np.isfinite(values)):
        raise ActionNonFiniteError("{0}: contains NaN/Inf".format(field.name))
    lo = np.asarray(field.bounds[0], dtype=float)
    hi = np.asarray(field.bounds[1], dtype=float)
    if bool(np.any(values < lo)) or bool(np.any(values > hi)):
        idx = int(np.argmax((values < lo) | (values > hi)))
        raise ActionBoundsError(
            "{0}: component {1} value {2!r} outside bounds [{3!r}, {4!r}]; the action was "
            "rejected, not clamped".format(field.name, idx, float(values.reshape(-1)[idx]),
                                           float(lo.min()), float(hi.max())))
    return values.reshape(-1)


def _check_dt(dt_s: Any, limits: ActionLimits) -> float:
    try:
        dt = float(dt_s)
    except (TypeError, ValueError):
        raise ActionContextError("dt_s must be a number, got {0!r}".format(dt_s))
    if not math.isfinite(dt) or dt <= 0.0:
        raise ActionContextError("dt_s must be finite and > 0, got {0!r}".format(dt_s))
    max_dt = float(limits.control_dt_max.value)
    if dt > max_dt:
        raise ActionContextError(
            "dt_s={0!r} exceeds control_dt_max={1}: an action computed for a longer step "
            "cannot be applied as a single control step".format(dt, max_dt))
    return dt


def validate(action: Mapping[str, ActionValue], *, limits: ActionLimits = DEFAULT_LIMITS,
             dt_s: float = DEFAULT_DT_S) -> Dict[str, np.ndarray]:
    """Strictly validate an action mapping and return its flat float64 components."""
    dt = _check_dt(dt_s, limits)
    if not isinstance(action, Mapping):
        raise ActionSchemaError("action must be a mapping, got {0}".format(type(action).__name__))
    fields = action_fields(limits, dt)
    by_name = {f.name: f for f in fields}
    missing = [f.name for f in fields if not f.optional and f.name not in action]
    if missing:
        raise MissingActionFieldError("missing action component(s): " + ", ".join(missing))
    unknown = sorted(set(action) - set(by_name))
    if unknown:
        raise UnknownActionFieldError("undeclared action component(s): " + ", ".join(unknown))
    out: Dict[str, np.ndarray] = {}
    for field in fields:
        flat = _coerce_action_field(field, action.get(field.name))
        if flat is not None:
            out[field.name] = flat
    if ("racket_target_position" in out) != ("racket_target_orientation" in out):
        raise ActionSchemaError(
            "racket target must supply position and orientation together; got only "
            + ("position" if "racket_target_position" in out else "orientation"))
    return out


# --------------------------------------------------------------------------------------
# command object
# --------------------------------------------------------------------------------------
@dataclass(frozen=True)
class SimulatedCommand:
    """A command for a SIMULATOR ONLY. Constructing it cannot actuate anything.

    base_twist_body         : (3,) = [vx, vy, wz], robot base frame
    joint_position_target   : (6,) = joint_position (measured) + joint_position_delta
    """
    base_twist_body: np.ndarray
    joint_position_target: np.ndarray
    dt_s: float
    timestamp_s: Optional[float] = None
    racket_target_position: Optional[np.ndarray] = None
    racket_target_orientation: Optional[np.ndarray] = None
    previous_action_checked: bool = False
    simulation_only: bool = True
    target: str = TARGET

    def __post_init__(self) -> None:
        if self.simulation_only is not True or self.target != TARGET:
            raise RealRobotActionForbidden(
                "SimulatedCommand must stay simulation_only=True/'" + TARGET + "'; refusing to "
                "construct a command that claims another target")
        for name in ("base_twist_body", "joint_position_target"):
            arr = np.asarray(getattr(self, name), dtype=float)
            if not np.all(np.isfinite(arr)):
                raise ActionNonFiniteError("{0}: contains NaN/Inf".format(name))
        if np.asarray(self.base_twist_body).shape != (3,):
            raise ActionShapeError("base_twist_body must be (3,)")
        if np.asarray(self.joint_position_target).shape != (6,):
            raise ActionShapeError("joint_position_target must be (6,)")

    def to_real_robot(self, *_args: Any, **_kwargs: Any) -> None:
        """Always raises: there is no authorisation and no hardware path."""
        raise RealRobotActionForbidden(REAL_ROBOT_FORBIDDEN_REASON)

    def assert_simulation_only(self) -> None:
        if self.simulation_only is not True or self.target != TARGET:
            raise RealRobotActionForbidden(REAL_ROBOT_FORBIDDEN_REASON)


def to_command(action: Mapping[str, ActionValue], *, joint_position: Any,
               previous_action: Optional[Mapping[str, ActionValue]] = None,
               dt_s: float = DEFAULT_DT_S, now_s: Optional[float] = None,
               limits: ActionLimits = DEFAULT_LIMITS) -> SimulatedCommand:
    """Convert a validated action into a (simulation-only) command.

    joint_position is REQUIRED: a delta alone cannot name a target, and inventing the current
    joint positions would be exactly the kind of silent fabrication this project forbids.
    """
    dt = _check_dt(dt_s, limits)
    parts = validate(action, limits=limits, dt_s=dt)

    q = np.asarray(joint_position, dtype=float)
    if q.shape != (6,):
        raise ActionContextError("joint_position must be (6,), got shape {0}".format(q.shape))
    if not np.all(np.isfinite(q)):
        raise ActionNonFiniteError("joint_position contains NaN/Inf")
    qmin = np.asarray(limits.joint_position_min.value, dtype=float)
    qmax = np.asarray(limits.joint_position_max.value, dtype=float)
    if bool(np.any(q < qmin)) or bool(np.any(q > qmax)):
        raise ActionBoundsError(
            "joint_position outside the TEMP joint range [{0}, {1}]; refusing to build a command "
            "from an impossible state".format(float(qmin.min()), float(qmax.max())))

    q_target = q + parts["joint_position_delta"]

    checked = False
    if previous_action is not None:
        prev = validate(previous_action, limits=limits, dt_s=dt)
        lin_acc = float(limits.base_linear_accel_max.value)
        ang_acc = float(limits.base_angular_accel_max.value)
        jnt_acc = float(limits.joint_accel_max.value)
        checks = (
            ("base_linear_velocity", lin_acc * dt, "base_linear_accel_max"),
            ("base_angular_velocity", ang_acc * dt, "base_angular_accel_max"),
            ("joint_position_delta", jnt_acc * dt, "joint_accel_max"),
        )
        for name, allowed, limit_name in checks:
            delta = np.abs(parts[name] - prev[name])
            if bool(np.any(delta > allowed + 1e-12)):
                idx = int(np.argmax(delta))
                raise ActionRateLimitError(
                    "{0}: step-to-step change {1!r} at component {2} exceeds {3} * dt = {4!r} "
                    "(rate limit); the command was rejected, not smoothed"
                    .format(name, float(delta[idx]), idx, limit_name, float(allowed)))
        checked = True

    return SimulatedCommand(
        base_twist_body=np.concatenate([parts["base_linear_velocity"], parts["base_angular_velocity"]]),
        joint_position_target=q_target,
        dt_s=dt,
        timestamp_s=None if now_s is None else float(now_s),
        racket_target_position=parts.get("racket_target_position"),
        racket_target_orientation=parts.get("racket_target_orientation"),
        previous_action_checked=checked,
    )


def schema_rows(limits: ActionLimits = DEFAULT_LIMITS, dt_s: float = DEFAULT_DT_S) -> Tuple[Dict[str, Any], ...]:
    rows = []
    for f in action_fields(limits, dt_s):
        rows.append({
            "name": f.name, "shape": f.shape, "dtype": f.dtype, "unit": f.unit,
            "bounds": f.bounds, "optional": f.optional, "source": f.source,
            "description": f.description,
        })
    return tuple(rows)


__all__ = [
    "ACTION_SCHEMA_VERSION", "SCHEMA_STATUS", "TRAINING_STATUS", "SIMULATION_ONLY", "TARGET",
    "DEFAULT_DT_S", "REAL_ROBOT_FORBIDDEN_REASON", "ActionLimits", "DEFAULT_LIMITS",
    "ActionField", "ACTION_FIELDS", "REQUIRED_ACTION_NAMES", "OPTIONAL_ACTION_NAMES",
    "ActionValue", "action_dim", "action_fields", "validate", "to_command",
    "SimulatedCommand", "schema_rows",
    "ActionSchemaError", "MissingActionFieldError", "UnknownActionFieldError", "ActionShapeError",
    "ActionDTypeError", "ActionNonFiniteError", "ActionBoundsError", "ActionUnitError",
    "ActionRateLimitError", "ActionContextError", "RealRobotActionForbidden",
]
