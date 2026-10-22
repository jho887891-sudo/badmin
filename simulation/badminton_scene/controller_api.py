# -*- coding: utf-8 -*-
"""R10-F1 (P5-09): the UNIFIED B601-RS controller API - one abstract interface, one adapter.

WHY A SECOND LAYER ON TOP OF badminton_scene.controller
    controller.py already owns the arm: name -> DOF index (P5-04), the safety envelope + HOLD
    latch (P5-05) and the model-aware reset path (P5B/P5-06).  What it does NOT have is a
    published CONTRACT: an abstract interface a planner, an RL policy, a test double or a second
    arm can be written against, and a set of TYPED refusals a caller can branch on without
    importing the safety module.

    This module is that contract, and nothing else:

        ArmController (ABC)          state / targets / reset / hold / IK hook
          B601RSControllerAdapter    the concrete adapter over B601RSController

THE ORDERING GUARANTEE (the reason this file exists)
    Every refusal is raised BEFORE the sink is touched.  The ABC owns all of the validation and
    calls exactly three hooks that a concrete subclass implements:

        _do_read()            read the articulation
        _do_write(values)     WRITE the joint setpoint      <- the only command write
        _do_reset(rows)       WRITE a reset pose            <- the only reset write

    A subclass cannot "validate a bit and write anyway": it never sees the raw target, only a
    fully validated tuple.  tests/r10_f/test_controller_api_contract.py counts _do_write and
    _do_reset calls on a fake adapter and proves the count is 0 on every refusal path -
    including the out-of-limit target the P7-10 task is explicitly about ("do not write an
    out-of-range target and then refuse").

TYPED ERROR PATHS (all derive from ControllerRefusal, all carry .code)
    UnknownJointTarget     a named target mentions a joint this arm does not have
    TargetWidthError       one value per runtime DOF was not supplied
    NonFiniteTarget        NaN / +-inf anywhere in the target
    OutOfLimitTarget       outside the model limit table  (REFUSED BEFORE ANY WRITE)
    StaleState             the cached reading is older than max_state_age_s
    StateUnavailable       there is no state to read (never padded with zeros)
    DoubleReset            a second reset with no progress in between
    HoldDuringMotion       hold() while a setpoint is still being executed
    HoldEngaged            a command/reset arrived while HOLD is latched
    NotHolding             release() while not holding
    HoldWithoutTarget      hold() before any target was accepted
    IkUnavailable          the IK hook has no solver behind it (declared, not faked)

WHAT THIS FILE STILL DOES NOT KNOW
    There is no torque data, no measured velocity limit and no IK solver for the B601-RS racket
    target (P5_CONTROLLER_AND_HOLD.md section 5).  Those stay UNKNOWN: the API has the hook, and
    the hook raises.

NO RUNTIME DEPENDENCIES
    Pure python + abc.  No torch, numpy, Isaac or pxr, so the contract is checkable on a plain
    CPU box; the adapter imports the existing controller, which imports torch lazily only on
    the reset path.
"""
from __future__ import annotations

import abc
import math
from dataclasses import dataclass
from typing import Callable, Dict, Mapping, Optional, Sequence, Tuple

__all__ = [
    "API_VERSION",
    "ControllerRefusal", "UnknownJointTarget", "TargetWidthError", "NonFiniteTarget",
    "OutOfLimitTarget", "StaleState", "StateUnavailable", "DoubleReset",
    "HoldDuringMotion", "HoldEngaged", "NotHolding", "HoldWithoutTarget", "IkUnavailable",
    "JointReading", "JointCommand", "HoldTicket",
    "ArmController", "B601RSControllerAdapter", "ENVELOPE_SOURCE_NOTE",
]

#: Bumped when a signature or a refusal code changes.  A caller can pin it.
API_VERSION = "r10f/arm-controller/v1"

ENVELOPE_SOURCE_NOTE = (
    "position limits: badminton_scene.rest_pose model table (official URDF, cross-checked "
    "against the converted USD to 8.3e-8 rad by P5). velocity/acceleration caps: P5-05 POLICY "
    "CAP, unverified - the URDF carries no velocity limit for these joints."
)


# --------------------------------------------------------------------- refusals
class ControllerRefusal(RuntimeError):
    """Base class for every typed refusal.  .code is stable and safe to branch on."""

    code = "controller_refusal"

    def __init__(self, message: str, **context: object) -> None:
        super().__init__(message)
        self.context: Dict[str, object] = dict(context)

    def as_dict(self) -> Dict[str, object]:
        payload: Dict[str, object] = {"code": self.code, "message": str(self)}
        payload.update(self.context)
        return payload


class UnknownJointTarget(ControllerRefusal):
    code = "unknown_joint"


class TargetWidthError(ControllerRefusal):
    code = "target_width"


class NonFiniteTarget(ControllerRefusal):
    code = "non_finite_target"


class OutOfLimitTarget(ControllerRefusal):
    """Outside the model limit table.  Raised BEFORE any write reaches the sink."""

    code = "out_of_limit_target"


class StaleState(ControllerRefusal):
    code = "stale_state"


class StateUnavailable(ControllerRefusal):
    code = "state_unavailable"


class DoubleReset(ControllerRefusal):
    code = "double_reset"


class HoldDuringMotion(ControllerRefusal):
    code = "hold_during_motion"


class HoldEngaged(ControllerRefusal):
    code = "hold_engaged"


class NotHolding(ControllerRefusal):
    code = "not_holding"


class HoldWithoutTarget(ControllerRefusal):
    code = "hold_without_target"


class IkUnavailable(ControllerRefusal):
    code = "ik_unavailable"
# ------------------------------------------------------------------- value types
@dataclass(frozen=True)
class JointReading:
    """One arm reading, addressable by joint NAME.  No torque field: there is no data."""

    names: Tuple[str, ...]
    joint_pos: Tuple[float, ...]
    joint_vel: Optional[Tuple[float, ...]]
    sampled_at_s: float
    source: str = ""

    def by_name(self) -> Dict[str, float]:
        return dict(zip(self.names, self.joint_pos))

    def __getitem__(self, name: str) -> float:
        if name not in self.names:
            raise UnknownJointTarget(
                "arm reading has no joint %r (runtime joints %s)" % (name, list(self.names)),
                joint=str(name), known=list(self.names))
        return self.joint_pos[self.names.index(name)]

    def as_dict(self) -> Dict[str, object]:
        return {"names": list(self.names), "joint_pos": [float(v) for v in self.joint_pos],
                "joint_vel": None if self.joint_vel is None
                else [float(v) for v in self.joint_vel],
                "sampled_at_s": float(self.sampled_at_s), "source": self.source}


@dataclass(frozen=True)
class JointCommand:
    """An accepted setpoint, plus the limit margin it was checked against."""

    names: Tuple[str, ...]
    joint_pos: Tuple[float, ...]
    accepted_at_s: float
    limit_margin: Tuple[float, ...]
    source: str = ""

    def as_dict(self) -> Dict[str, object]:
        return {"names": list(self.names), "joint_pos": [float(v) for v in self.joint_pos],
                "accepted_at_s": float(self.accepted_at_s),
                "limit_margin": [float(v) for v in self.limit_margin], "source": self.source}


@dataclass(frozen=True)
class HoldTicket:
    """The frozen target and when the latch engaged.  Bit-identical to the accepted command."""

    joint_pos: Tuple[float, ...]
    names: Tuple[str, ...]
    engaged_at_s: float
    reason: str = "OPERATOR_HOLD"


# ---------------------------------------------------------------------- interface
class ArmController(abc.ABC):
    """The published B601-RS contract.  Subclass and implement three hooks.

    The ABC performs, in this order and before touching a hook that writes:
        width -> finiteness -> per-joint limits -> HOLD / in-flight state
    so a refusal can never arrive after a write.
    """

    api_version = API_VERSION

    def __init__(self, joint_names: Sequence[object], *,
                 limits: Mapping[str, Tuple[float, float, str]],
                 model: str = "", max_state_age_s: float = 0.5,
                 clock: Optional[Callable[[], float]] = None) -> None:
        names = tuple(str(name) for name in joint_names)
        if not names:
            raise TargetWidthError("no joint names: the controller has no DOF to address", count=0)
        if len(set(names)) != len(names):
            raise UnknownJointTarget(
                "duplicate joint names %s: a duplicated DOF cannot be addressed by name"
                % (list(names),), known=list(names))
        missing = [name for name in names if name not in limits]
        if missing:
            raise UnknownJointTarget(
                "no limit entry for runtime joints %s (limits cover %s)"
                % (missing, sorted(limits)), joint=missing[0], known=sorted(limits))
        self.joint_names: Tuple[str, ...] = names
        self.model = str(model)
        self._limits: Dict[str, Tuple[float, float, str]] = {
            name: (float(limits[name][0]), float(limits[name][1]), str(limits[name][2]))
            for name in names}
        if not (math.isfinite(max_state_age_s) and max_state_age_s > 0.0):
            raise TargetWidthError("max_state_age_s must be finite and > 0",
                                   value=max_state_age_s)
        self.max_state_age_s = float(max_state_age_s)
        self._clock = clock

        # observable state (tests read these; nothing here writes to a robot)
        self.write_count = 0
        self.reset_count = 0

        self._reading: Optional[JointReading] = None
        self._last_command: Optional[JointCommand] = None
        self._pending: Optional[Tuple[float, ...]] = None
        self._motion_pending = False
        self._hold: Optional[HoldTicket] = None
        self._resets_since_progress = 0

    # -- hooks a subclass implements ---------------------------------------
    @abc.abstractmethod
    def _do_read(self, at_s: float):
        """Return (joint_pos, joint_vel_or_None), in RUNTIME order."""
        raise NotImplementedError

    @abc.abstractmethod
    def _do_write(self, values: Tuple[float, ...], at_s: float) -> None:
        """THE command write.  Only ever called with fully validated values."""
        raise NotImplementedError

    @abc.abstractmethod
    def _do_reset(self, rows: Tuple[Tuple[float, ...], ...], at_s: float) -> None:
        """THE reset write.  Only ever called with validated, in-limit rows."""
        raise NotImplementedError

    # -- introspection -----------------------------------------------------
    @property
    def dof_count(self) -> int:
        return len(self.joint_names)

    @property
    def holding(self) -> bool:
        return self._hold is not None

    @property
    def held(self) -> Optional[HoldTicket]:
        return self._hold

    @property
    def motion_pending(self) -> bool:
        return self._motion_pending

    @property
    def last_command(self) -> Optional[JointCommand]:
        return self._last_command

    def limits(self) -> Dict[str, Tuple[float, float, str]]:
        return dict(self._limits)

    def describe(self) -> Dict[str, object]:
        return {"api_version": self.api_version, "model": self.model,
                "joint_names": list(self.joint_names), "dof_count": self.dof_count,
                "holding": self.holding, "motion_pending": self._motion_pending,
                "limits": {name: list(spec) for name, spec in self._limits.items()},
                "envelope_source": ENVELOPE_SOURCE_NOTE,
                "write_count": self.write_count, "reset_count": self.reset_count}
# -- 1. state ----------------------------------------------------------
    def sample_state(self, at_s: float) -> JointReading:
        """Read the arm now.  at_s is the caller's clock (no hidden time.time())."""
        now = self._time(at_s)
        positions, velocities = self._do_read(now)
        values = tuple(float(value) for value in positions)
        self._check_width(values, "the runtime state")
        self._check_finite(values, "the runtime state")
        vels = None if velocities is None else tuple(float(v) for v in velocities)
        if vels is not None:
            self._check_width(vels, "the runtime velocity state")
        self._reading = JointReading(names=self.joint_names, joint_pos=values, joint_vel=vels,
                                     sampled_at_s=now,
                                     source="%s/runtime" % (self.model or "arm"))
        self._resets_since_progress = 0
        if self._pending is not None and values == self._pending:
            self._motion_pending = False
        return self._reading

    def state(self, at_s: float) -> JointReading:
        """The cached reading, refused if it is older than max_state_age_s."""
        now = self._time(at_s)
        if self._reading is None:
            raise StateUnavailable(
                "no state has been sampled: a fabricated zero pose would be indistinguishable "
                "from a real one downstream; call sample_state() first", model=self.model)
        age = now - self._reading.sampled_at_s
        if age > self.max_state_age_s:
            raise StaleState(
                "the cached state is %.4f s old (> %.4f s): using it would plan against a "
                "robot that has already moved" % (age, self.max_state_age_s),
                age_s=age, max_age_s=self.max_state_age_s)
        if age < -1e-9:
            raise StaleState("the cached state is stamped %.4f s in the FUTURE: clock error"
                             % (-age,), age_s=age)
        return self._reading

    # -- 2. joint targets --------------------------------------------------
    def command_joint_targets(self, targets: object, at_s: float) -> JointCommand:
        """Validate and submit one setpoint.  Raises BEFORE _do_write on any refusal."""
        now = self._time(at_s)
        if self.holding:
            raise HoldEngaged(
                "HOLD is latched at %r: refusing a new setpoint; release() first"
                % (self._hold.joint_pos if self._hold else None,),
                held_at=None if self._hold is None else self._hold.engaged_at_s)
        values = self._normalise(targets)
        # --- every refusal below happens before the sink is touched ---
        self._check_width(values, "the joint target")
        self._check_finite(values, "the joint target")
        self._check_limits(values, "joint target")
        self._do_write(values, now)
        self.write_count += 1
        self._pending = values
        self._motion_pending = True
        self._resets_since_progress = 0
        command = JointCommand(names=self.joint_names, joint_pos=values, accepted_at_s=now,
                               limit_margin=self._margins(values), source="controller_api")
        self._last_command = command
        return command

    def settle(self, at_s: float) -> Optional[JointReading]:
        """Declare the arm to have REACHED its setpoint (a supervisor's post-condition).

        Needed because hold() refuses while a setpoint is in flight: freezing a target the
        arm never reached is a slow drift with better branding.
        """
        now = self._time(at_s)
        self._motion_pending = False
        self._pending = None
        return self._reading

    # -- 3. reset ----------------------------------------------------------
    def reset(self, joint_pos: object, at_s: float) -> None:
        """Write a reset pose through the same checked path.  Never reinterprets None."""
        now = self._time(at_s)
        if self.holding:
            raise HoldEngaged(
                "HOLD is latched: refusing a reset while the arm is held; release() first",
                held_at=None if self._hold is None else self._hold.engaged_at_s)
        if self._resets_since_progress >= 1:
            raise DoubleReset(
                "reset() was already called and nothing has happened since (no accepted "
                "setpoint, no fresh state read). A reset loop hides the fault it is meant to "
                "clear: read the state first if the reset really did not take.", resets=1)
        rows = self._normalise_rows(joint_pos)
        for row_index, row in enumerate(rows):
            self._check_width(row, "reset row %d" % row_index)
            self._check_finite(row, "reset row %d" % row_index)
            self._check_limits(row, "reset row %d" % row_index)
        self._do_reset(rows, now)
        self.reset_count += 1
        self._resets_since_progress += 1
        self._motion_pending = False
        self._pending = None
        # A reset teleports the joints without commanding them: the cached reading now
        # describes where the arm USED to be.  Drop it rather than serve it as fresh.
        self._reading = None

    # -- 4. HOLD -----------------------------------------------------------
    def hold(self, at_s: float, *, reason: str = "OPERATOR_HOLD") -> HoldTicket:
        now = self._time(at_s)
        if self.holding:
            raise HoldEngaged("HOLD is already latched (reason %r)" % (self._hold.reason,),
                              reason=self._hold.reason)
        if self._motion_pending:
            raise HoldDuringMotion(
                "a setpoint is still being executed (pending %r): latching HOLD now would "
                "freeze a target the arm has not reached. Call settle() once the arm is at "
                "the setpoint, or hold on the MEASURED state instead."
                % (self._pending,),
                pending=None if self._pending is None else [float(v) for v in self._pending])
        if self._last_command is None:
            raise HoldWithoutTarget(
                "no setpoint has been accepted yet: there is nothing to freeze", model=self.model)
        self._hold = HoldTicket(joint_pos=self._last_command.joint_pos,
                                names=self._last_command.names, engaged_at_s=now,
                                reason=str(reason))
        return self._hold

    def release(self, at_s: float) -> None:
        self._time(at_s)
        if not self.holding:
            raise NotHolding("release() called while the controller was not holding")
        self._hold = None

    # -- 5. IK hook --------------------------------------------------------
    def solve_ik(self, target_pose: object, at_s: float,
                 seed: Optional[Sequence[float]] = None):
        """Declared hook, no implementation.  Raises instead of returning a plausible pose."""
        self._time(at_s)
        raise IkUnavailable(
            "solve_ik() has no solver behind it: this repo has no IK for the B601-RS racket "
            "target and no calibrated TCP frame to solve against. Returning the current pose "
            "or the seed would be indistinguishable from a solution downstream.",
            has_seed=seed is not None, target=str(type(target_pose).__name__))
# -- internals ---------------------------------------------------------
    def _time(self, at_s: object) -> float:
        try:
            value = float(at_s)
        except (TypeError, ValueError):
            raise TargetWidthError("at_s must be a real number, got %r" % (at_s,)) from None
        if not math.isfinite(value):
            raise TargetWidthError("at_s must be finite, got %r" % (value,))
        return value

    def _normalise(self, targets: object) -> Tuple[float, ...]:
        if isinstance(targets, Mapping):
            extra = [str(name) for name in targets if str(name) not in self.joint_names]
            if extra:
                raise UnknownJointTarget(
                    "target names joints this arm does not have: %s (runtime joints %s)"
                    % (extra, list(self.joint_names)),
                    joint=extra[0], known=list(self.joint_names))
            missing = [name for name in self.joint_names if name not in targets]
            if missing:
                raise UnknownJointTarget(
                    "target has no value for runtime joints %s" % (missing,),
                    joint=missing[0], known=list(self.joint_names))
            return tuple(float(targets[name]) for name in self.joint_names)
        if hasattr(targets, "tolist"):
            targets = targets.tolist()
        if not isinstance(targets, (list, tuple)):
            raise TargetWidthError(
                "target must be a sequence or a {name: value} mapping, got %r"
                % (type(targets).__name__,), value=type(targets).__name__)
        return tuple(float(value) for value in targets)

    def _normalise_rows(self, joint_pos: object) -> Tuple[Tuple[float, ...], ...]:
        if joint_pos is None:
            raise TargetWidthError(
                "reset() was given no joint_pos; this API does not reinterpret None as the "
                "rest pose - ask for the pose you mean")
        if isinstance(joint_pos, Mapping):
            joint_pos = self._normalise(joint_pos)
        if hasattr(joint_pos, "tolist"):
            joint_pos = joint_pos.tolist()
        if not isinstance(joint_pos, (list, tuple)) or len(joint_pos) == 0:
            raise TargetWidthError("joint_pos must be a non-empty sequence of joint values",
                                   value=type(joint_pos).__name__)
        first = joint_pos[0]
        if isinstance(first, (list, tuple)):
            return tuple(tuple(float(value) for value in row) for row in joint_pos)
        return (tuple(float(value) for value in joint_pos),)

    def _check_width(self, values: Tuple[float, ...], what: str) -> None:
        if len(values) != self.dof_count:
            raise TargetWidthError(
                "%s has %d values for %d runtime DOFs %s"
                % (what, len(values), self.dof_count, list(self.joint_names)),
                count=len(values), expected=self.dof_count, what=what)

    def _check_finite(self, values: Tuple[float, ...], what: str) -> None:
        for position, value in enumerate(values):
            if not math.isfinite(value):
                raise NonFiniteTarget(
                    "%s: %s = %r is not finite; a NaN/Inf setpoint propagates through the "
                    "whole state" % (what, self.joint_names[position], value),
                    joint=self.joint_names[position], index=position)

    def _check_limits(self, values: Tuple[float, ...], what: str) -> None:
        for position, (name, value) in enumerate(zip(self.joint_names, values)):
            lower, upper, unit = self._limits[name]
            if not (lower <= value <= upper):
                raise OutOfLimitTarget(
                    "%s: %s = %r is outside [%r, %r] %s of the %s model table; REFUSED BEFORE "
                    "ANY WRITE" % (what, name, value, lower, upper, unit, self.model or "arm"),
                    joint=name, value=value, lower=lower, upper=upper, unit=unit,
                    index=position, written=False)

    def _margins(self, values: Tuple[float, ...]) -> Tuple[float, ...]:
        out = []
        for name, value in zip(self.joint_names, values):
            lower, upper, _unit = self._limits[name]
            out.append(float(min(value - lower, upper - value)))
        return tuple(out)


# ------------------------------------------------------------------------ adapter
class B601RSControllerAdapter(ArmController):
    """The concrete adapter over badminton_scene.controller.B601RSController.

    The wrapped controller keeps owning the DOF map, the P5-05 envelope and the P5B reset path.
    This adapter adds the published contract (typed refusals, an explicit ordering guarantee, a
    caller clock, a state age) and translates the wrapped refusals into it.

    The adapter validates BEFORE delegating, so the wrapped object's reset/write hooks are not
    reached on a refusal.  That is deliberate defence in depth: the wrapped controller refuses
    too, but the count that matters (writes) must be zero even if it did not.
    """

    def __init__(self, controller: object, *, max_state_age_s: float = 0.5,
                 clock: Optional[Callable[[], float]] = None) -> None:
        self.controller = controller
        envelope = getattr(controller, "envelope", None)
        if envelope is None or not hasattr(envelope, "limits"):
            raise StateUnavailable("the wrapped controller exposes no envelope/limits; this "
                                   "adapter refuses to invent a limit table")
        names = tuple(str(name) for name in controller.joint_names)
        limits = {}
        for name in names:
            limit = envelope.limits[name]
            limits[name] = (limit.lower, limit.upper, limit.unit)
        super().__init__(names, limits=limits, model=str(getattr(controller, "model", "")),
                         max_state_age_s=max_state_age_s, clock=clock)

    # -- hooks -------------------------------------------------------------
    def _do_read(self, at_s: float):
        import badminton_scene.controller as _controller

        try:
            state = self.controller.read_state()
        except _controller.StateUnavailable as exc:
            raise StateUnavailable(str(exc), model=self.model) from exc
        return (tuple(state.joint_pos),
                None if state.joint_vel is None else tuple(state.joint_vel))

    def _do_write(self, values: Tuple[float, ...], at_s: float) -> None:
        import badminton_scene.safety as _safety

        try:
            self.controller.set_joint_targets(values)
        except (_safety.PositionLimitViolation, _safety.NonFiniteCommand,
                _safety.CommandWidthError) as exc:
            # This adapter's own check should have fired first; if the two limit tables ever
            # drift, the refusal still arrives typed and still arrives before execution.
            raise OutOfLimitTarget(str(exc), joint=getattr(exc, "joint", None),
                                   value=getattr(exc, "value", None), written=False) from exc
        except _safety.UnknownCommandJoint as exc:
            raise UnknownJointTarget(str(exc), joint=getattr(exc, "joint", None)) from exc
        except _safety.HoldActive as exc:
            raise HoldEngaged(str(exc)) from exc

    def _do_reset(self, rows: Tuple[Tuple[float, ...], ...], at_s: float) -> None:
        import badminton_scene.rest_pose as _rest
        import badminton_scene.safety as _safety

        payload = rows[0] if len(rows) == 1 else rows
        try:
            self.controller.reset(payload)
        except _rest.RestPoseLimitViolation as exc:
            raise OutOfLimitTarget(str(exc), written=False) from exc
        except _rest.UnknownJoint as exc:
            raise UnknownJointTarget(str(exc)) from exc
        except _safety.HoldActive as exc:
            raise HoldEngaged(str(exc)) from exc


