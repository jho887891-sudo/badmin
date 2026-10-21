# -*- coding: utf-8 -*-
"""P5-05: a command safety envelope (position / velocity / acceleration / finiteness) plus HOLD.

The envelope is the last thing between a planner or an RL policy and the articulation. It answers
exactly one question - 'may this joint target be sent?' - and it answers it by RAISING, never by
quietly clipping: a clip turns an out-of-range command into a plausible-looking command that the
robot then executes, and the operator never learns that the planner asked for something impossible.

WHAT IS CHECKED, IN THIS ORDER (first failure wins, so the message names one cause at a time)
    1. width            one value per runtime DOF
    2. finiteness       NaN / +-inf are rejected BEFORE any comparison (NaN < lo is False, so a
                        range check alone cannot see a NaN)
    3. position         per-joint limits from the MODEL TABLE (badminton_scene.rest_pose), which is
                        the same table P5/P5B validated against the USD to 8.3e-8 rad
    4. velocity         |target - previous target| / dt  <=  v_max[joint]
    5. acceleration     |target - 2*prev + prevprev| / dt^2  (only when a full history exists)

    Steps 4 and 5 are a TARGET-STEP limiter, not a measurement: they bound how fast the commanded
    setpoint may move. The real joint velocity/torque of the arm is UNKNOWN here - no encoder data,
    no torque sensor, no GPU multi-env run. The limits themselves are POLICY CAPS, not datasheet
    numbers: see ENVELOPE_SOURCE.

HOLD
    hold() freezes the last ACCEPTED target object and refuses every new command until release().
    The frozen target is kept bit-identically: the same tuple of floats is handed back, no arithmetic
    is applied to it (a hold that recomputes a target is a slow drift, not a hold). Rejections while
    holding raise HoldActive with the held target in the message, and rejections always carry the
    offending joint name and value as attributes so a caller can log a structured event.

NO RUNTIME DEPENDENCIES
    Pure python (math only). The runtime names and the model limits are the only inputs.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, Iterable, Mapping, Optional, Sequence, Tuple

from .rest_pose import JointLimit, limits_for

#: Where the velocity / acceleration caps come from. They are NOT measurements and NOT datasheet
#: values: the URDF carries no velocity limit for these joints, and no unit has been characterised.
#: They exist so that the limiter is total (it must reject something), and they must be replaced by
#: measured or vendor values before anything but a standstill test command is authorised.
ENVELOPE_SOURCE = (
    "POLICY CAP - unverified. The URDF records no velocity limit for the B601-RS joints and no unit "
    "has been characterised (P5_CONTROLLER_AND_HOLD.md section 5); position limits alone come from "
    "the model table cross-checked against the USD"
)

#: Per-unit defaults, applied by envelope_for() unless the caller overrides them.
_REVOLUTE_VELOCITY = 2.0       # rad/s
_REVOLUTE_ACCELERATION = 20.0  # rad/s^2
_PRISMATIC_VELOCITY = 0.05     # m/s (a finger stroke per second)
_PRISMATIC_ACCELERATION = 0.5  # m/s^2

#: Nominal control period used when the caller does not pass dt to a rate check.
DEFAULT_DT = 1.0 / 60.0


class CommandRejected(RuntimeError):
    """Base class: the target was refused. Carries the offending joint / value / limit."""

    code = "command_rejected"

    def __init__(self, message: str, joint: Optional[str] = None, value: object = None,
                 limit: object = None, index: Optional[int] = None):
        super().__init__(message)
        self.joint = joint
        self.value = value
        self.limit = limit
        self.index = index

    def as_dict(self) -> Dict[str, object]:
        """Structured form for a log line / telemetry event."""
        return {"code": self.code, "joint": self.joint, "index": self.index,
                "value": self.value, "limit": self.limit, "message": str(self)}


class CommandWidthError(CommandRejected):
    """The target does not carry exactly one value per runtime DOF."""

    code = "command_width"


class NonFiniteCommand(CommandRejected):
    """The target contains NaN or an infinity."""

    code = "non_finite"


class PositionLimitViolation(CommandRejected):
    """The target leaves a joint's position limits (inclusive stops are allowed)."""

    code = "position_limit"


class VelocityLimitViolation(CommandRejected):
    """The target step needs more than the allowed setpoint rate."""

    code = "velocity_limit"


class AccelerationLimitViolation(CommandRejected):
    """The target step needs more than the allowed setpoint acceleration."""

    code = "acceleration_limit"


class UnknownCommandJoint(CommandRejected):
    """A named target mentions a joint the runtime does not have."""

    code = "unknown_joint"


class HoldActive(CommandRejected):
    """A new command arrived while HOLD is engaged."""

    code = "hold_active"


class HoldWithoutTarget(CommandRejected):
    """hold() was asked for before any target had been accepted - there is nothing to freeze."""

    code = "hold_without_target"


class NotHolding(CommandRejected):
    """release() was called while the gate was not holding."""

    code = "not_holding"


def _floats(payload: object, *, what: str = "joint target") -> Tuple[float, ...]:
    """Flatten a 1-D payload (list, tuple, numpy array, fake torch tensor) to floats.

    Only the interface that is actually needed is used: .tolist() when present, iteration otherwise.
    Nothing imports numpy or torch, so a shim (tests/simulation/arm_pose/fake_torch.py) drives the
    exact same code path as the real runtime.
    """
    if payload is None:
        raise CommandWidthError("%s is None; a command needs one value per runtime DOF" % what,
                                value=None)
    if hasattr(payload, "tolist"):
        payload = payload.tolist()
    values = []
    for element in payload:
        if isinstance(element, (list, tuple)) or hasattr(element, "tolist"):
            nested = element.tolist() if hasattr(element, "tolist") else element
            if isinstance(nested, (list, tuple)) and len(nested) == 1:
                element = nested[0]  # a single-env 2-D row is accepted, a wider batch is not
            else:
                raise CommandWidthError(
                    "%s is 2-D; this single-env gate takes one value per DOF" % what,
                    value=payload)
        values.append(float(element))
    return tuple(values)


@dataclass(frozen=True)
class Envelope:
    """The caps one arm's commands are checked against, plus where each group came from."""

    model: str
    limits: Mapping[str, JointLimit]
    velocity: Mapping[str, float]
    acceleration: Mapping[str, float]
    source: str = ENVELOPE_SOURCE

    def limit_of(self, name: str) -> JointLimit:
        return self.limits[name]


def envelope_for(model: object, velocity: Optional[Mapping[str, float]] = None,
                 acceleration: Optional[Mapping[str, float]] = None) -> Envelope:
    """Build the envelope for a model, position limits from that model's table."""
    spec_limits = limits_for(model)
    resolved = resolve_model_model_key(model)
    vel: Dict[str, float] = {}
    acc: Dict[str, float] = {}
    for name, limit in spec_limits.items():
        prismatic = limit.kind == "prismatic"
        vel[name] = (_PRISMATIC_VELOCITY if prismatic else _REVOLUTE_VELOCITY)
        acc[name] = (_PRISMATIC_ACCELERATION if prismatic else _REVOLUTE_ACCELERATION)
    if velocity:
        vel.update({str(name): float(value) for name, value in velocity.items()})
    if acceleration:
        acc.update({str(name): float(value) for name, value in acceleration.items()})
    return Envelope(model=resolved, limits=spec_limits, velocity=vel, acceleration=acc)


def resolve_model_model_key(model: object) -> str:
    """key of a model given the model object itself (no runtime names needed)."""
    from .rest_pose import _spec  # local: the resolver is the single source of truth

    return _spec(model).key


class SafetyGate:
    """Validate setpoints against one arm's envelope, and latch HOLD on top of it.

    The gate is positional in RUNTIME DOF order: it is constructed with the runtime joint names and
    never reorders anything. Name -> index resolution belongs to badminton_scene.dof_index (P5-04);
    keeping the two apart means the gate cannot accidentally validate one joint against another
    joint's limits, which is the whole failure mode this slice exists to prevent.
    """

    def __init__(self, model: object, names: Sequence[object],
                 envelope: Optional[Envelope] = None, dt: float = DEFAULT_DT):
        self.model = resolve_model_model_key(model)
        self.names: Tuple[str, ...] = tuple(str(name) for name in names)
        if not self.names:
            raise CommandWidthError("no runtime joint names; the gate has nothing to check", value=0)
        self.envelope = envelope if envelope is not None else envelope_for(self.model)
        missing = [name for name in self.names if name not in self.envelope.limits]
        if missing:
            raise UnknownCommandJoint(
                "the %s envelope has no limits for runtime joints %s"
                % (self.model, missing), value=missing)
        self.dt = float(dt)
        if not (self.dt > 0.0 and math.isfinite(self.dt)):
            raise CommandWidthError("dt must be a positive finite number, got %r" % (dt,),
                                    value=dt)
        self._last: Optional[Tuple[float, ...]] = None
        self._previous: Optional[Tuple[float, ...]] = None
        self._held: Optional[Tuple[float, ...]] = None
        self._rejected = 0

    # -- inspection --------------------------------------------------------
    @property
    def dof_count(self) -> int:
        return len(self.names)

    @property
    def holding(self) -> bool:
        return self._held is not None

    @property
    def held_target(self) -> Optional[Tuple[float, ...]]:
        """The frozen target object, byte-identical to the one that was accepted, or None."""
        return self._held

    @property
    def last_valid(self) -> Optional[Tuple[float, ...]]:
        return self._last

    @property
    def rejected_count(self) -> int:
        return self._rejected

    # -- validation --------------------------------------------------------
    def validate(self, joint_pos: object, *, previous: Optional[object] = None,
                 dt: Optional[float] = None) -> Tuple[float, ...]:
        """Check one target without changing any state. Returns the runtime-ordered tuple."""
        values = _floats(joint_pos)
        try:
            self._check(values, previous=previous, dt=dt)
        except CommandRejected:
            self._rejected += 1
            raise
        return values

    def validate_named(self, joint_pos: Mapping[str, object], *,
                       previous: Optional[object] = None, dt: Optional[float] = None) -> Tuple[float, ...]:
        """Check a {name: value} target; unknown or absent names both raise."""
        if not isinstance(joint_pos, Mapping):
            raise CommandWidthError(
                "validate_named() needs a {joint name: value} mapping, got %r"
                % (type(joint_pos).__name__,), value=joint_pos)
        extra = [str(name) for name in joint_pos if str(name) not in self.names]
        if extra:
            self._rejected += 1
            raise UnknownCommandJoint(
                "target names joints the runtime does not have: %s (runtime %s)"
                % (extra, list(self.names)), joint=extra[0], value=extra)
        missing = [name for name in self.names if name not in joint_pos]
        if missing:
            self._rejected += 1
            raise UnknownCommandJoint(
                "target has no value for runtime joints %s" % missing, joint=missing[0], value=None)
        ordered = tuple(joint_pos[name] for name in self.names)
        return self.validate(ordered, previous=previous, dt=dt)

    def accept(self, joint_pos: object, *, dt: Optional[float] = None) -> Tuple[float, ...]:
        """Validate, then latch: this target becomes the last valid target (and the rate baseline)."""
        if self.holding:
            self._rejected += 1
            raise HoldActive(
                "HOLD is engaged: refusing a new joint target while the %s gate holds %r. "
                "release() first." % (self.model, self._held), value=self._held)
        values = self.validate(joint_pos, previous=self._last, dt=dt)
        self._previous = self._last
        self._last = values
        return values

    def _check(self, values: Tuple[float, ...], *, previous: object, dt: Optional[float]) -> None:
        if len(values) != self.dof_count:
            raise CommandWidthError(
                "target has %d values for %d runtime DOFs %s"
                % (len(values), self.dof_count, list(self.names)),
                value=len(values), index=None)

        for position, value in enumerate(values):
            if not math.isfinite(value):
                raise NonFiniteCommand(
                    "%s = %r is not a finite number; a NaN/Inf setpoint would be written to the "
                    "articulation and propagate through the whole state"
                    % (self.names[position], value),
                    joint=self.names[position], value=value, index=position)

        for position, (name, value) in enumerate(zip(self.names, values)):
            limit = self.envelope.limits[name]
            if not (limit.lower <= value <= limit.upper):
                raise PositionLimitViolation(
                    "%s = %r is outside [%r, %r] %s of the %s model table"
                    % (name, value, limit.lower, limit.upper, limit.unit, self.model),
                    joint=name, value=value, limit=(limit.lower, limit.upper), index=position)

        if previous is None:
            return
        prev = _floats(previous, what="previous target")
        if len(prev) != self.dof_count:
            raise CommandWidthError(
                "previous target has %d values for %d runtime DOFs"
                % (len(prev), self.dof_count), value=len(prev))
        step_dt = self.dt if dt is None else float(dt)
        if not (step_dt > 0.0 and math.isfinite(step_dt)):
            raise CommandWidthError("dt must be a positive finite number, got %r" % (dt,),
                                    value=dt)
        self._check_velocity(values, prev, step_dt)
        self._check_acceleration(values, prev, step_dt)

    def _check_velocity(self, values, prev, dt) -> None:
        for position, (name, value, old) in enumerate(zip(self.names, values, prev)):
            rate = abs(value - old) / dt
            cap = self.envelope.velocity[name]
            if rate > cap:
                raise VelocityLimitViolation(
                    "%s would move %r -> %r in %g s = %.6g %s/s, above the %.6g cap (%s)"
                    % (name, old, value, dt, rate, "rad" if self.envelope.limits[name].unit == "rad" else "m",
                       cap, ENVELOPE_SOURCE),
                    joint=name, value=value, limit=cap, index=position)

    def _check_acceleration(self, values, prev, dt) -> None:
        if self._previous is None:
            return  # no history yet: an acceleration is undefined, not zero
        older = self._previous
        if len(older) != self.dof_count:
            return
        for position, (name, value, old, oldest) in enumerate(zip(self.names, values, prev, older)):
            accel = abs(value - 2.0 * old + oldest) / (dt * dt)
            cap = self.envelope.acceleration[name]
            if accel > cap:
                raise AccelerationLimitViolation(
                    "%s would change setpoint acceleration to %.6g (targets %r -> %r -> %r in %g s), "
                    "above the %.6g cap (%s)"
                    % (name, accel, oldest, old, value, dt, cap, ENVELOPE_SOURCE),
                    joint=name, value=value, limit=cap, index=position)

    def seed(self, joint_pos: object) -> Tuple[float, ...]:
        """Re-baseline the rate limiter to a known state (used after a reset / teleport).

        A reset moves the joints without commanding them, so the previous setpoint is no longer the
        physical one. Seeding clears the acceleration history instead of inventing one: the first
        command after a reset is rate-limited against the reset pose, not against a stale target.
        The target is validated, so a seed cannot smuggle in an out-of-range state.
        """
        values = _floats(joint_pos, what="seed target")
        self._check(values, previous=None, dt=self.dt)
        self._last = values
        self._previous = None
        return values

    # -- HOLD --------------------------------------------------------------
    def hold(self) -> Tuple[float, ...]:
        """Freeze the last accepted target, bit-identically, and refuse further commands."""
        if self.holding:
            raise HoldActive("HOLD is already engaged on %r" % (self._held,), value=self._held)
        if self._last is None:
            raise HoldWithoutTarget(
                "no target has been accepted yet; there is nothing to hold", value=None)
        self._held = self._last  # the same object: no copy, no recomputation, no drift
        return self._held

    def release(self) -> None:
        """Leave HOLD. The last valid target stays the rate baseline."""
        if not self.holding:
            raise NotHolding("release() called while the %s gate was not holding" % self.model)
        self._held = None


#: Public alias: the same 1-D payload flattener the gate uses, so callers (and the controller)
#: normalise payloads identically instead of growing a second, subtly different converter.
as_floats = _floats

__all__ = [
    "ENVELOPE_SOURCE", "DEFAULT_DT", "as_floats",
    "CommandRejected", "CommandWidthError", "NonFiniteCommand", "PositionLimitViolation",
    "VelocityLimitViolation", "AccelerationLimitViolation", "UnknownCommandJoint",
    "HoldActive", "HoldWithoutTarget", "NotHolding",
    "Envelope", "envelope_for", "SafetyGate",
]
