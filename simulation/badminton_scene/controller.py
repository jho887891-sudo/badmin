# -*- coding: utf-8 -*-
"""P5-09: the minimal reusable B601-RS controller API - state, targets, reset, HOLD, IK hook.

WHY A CONTROLLER AND NOT MORE SCRIPT CODE
    P5 and P5B proved two things about the arm separately: the static USD/limit table agrees with
    the model (P5), and a reset writes a model-aware, in-limit pose (P5B). Neither says anything
    about the object a planner, an RL policy or a real deployment would actually talk to. Without
    one, every caller re-derives name -> index (wrongly, from traversal order), re-checks limits (or
    forgets to) and re-implements 'stop where you are' as a special case. This is that object, and
    it is deliberately small enough to be reviewed in one sitting:

        read_state()            current joints, addressed by NAME (no guessed order)
        set_joint_targets()     validated setpoint (position + rate + finiteness), P5-04/P5-05
        reset()                 validated pose write through badminton_scene.reset (P5B), P5-06
        reset_to_rest()         the model-aware rest pose, same path
        hold() / release()      freeze the last accepted target (P5-05)
        solve_ik()              explicit hook: raises IkNotImplemented. There is no IK solver in
                                this repo; a stub that returned the current pose would be worse
                                than a stub that refuses.

Layering (each layer is independently tested, and each can be used without the one above it):
      rest_pose.py   the model tables (limits, rest poses) - no runtime dependencies
      dof_index.py   name -> runtime DOF index, from the RUNTIME name list (P5-04)
      safety.py      envelope + HOLD latch, in runtime DOF order (P5-05)
      controller.py  this file: binds one robot instance together and owns the sim writes

NO ISAAC, NO TORCH AT IMPORT
    Nothing here imports torch or numpy at module scope. The reset path imports torch lazily (or
    uses an injected tensor factory / reset function), which is what lets the whole API be driven
    by tests/simulation/arm_pose/fake_torch.py on a CPU box with no Isaac Lab installed.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Callable, Dict, Mapping, Optional, Sequence, Tuple

from .dof_index import JointIndex
from .rest_pose import (
    MODEL_REBOT_RS,
    ResetPoseError,
    RestPoseLimitViolation,
    UnknownJoint,
    limits_for,
    resolve_model,
    rest_joint_pos,
)
from .safety import (
    DEFAULT_DT,
    Envelope,
    HoldActive,
    SafetyGate,
    envelope_for,
)


class StateUnavailable(RuntimeError):
    """The robot object does not expose a joint state this controller can read."""


class IkNotImplemented(NotImplementedError):
    """solve_ik() is a declared hook with no implementation behind it (deliberately)."""


@dataclass(frozen=True)
class ArmState:
    """One arm reading, always addressable by joint name.

    joint_vel is None when the runtime does not expose it. Torque is not part of this type at all:
    this repo has no torque data for the arm (no sensor, no measured run), and an optional field
    that is always None reads like a measurement that is merely missing.
    """

    names: Tuple[str, ...]
    joint_pos: Tuple[float, ...]
    joint_vel: Optional[Tuple[float, ...]] = None
    source: str = ""

    def by_name(self) -> Dict[str, float]:
        return {name: value for name, value in zip(self.names, self.joint_pos)}

    def velocity_by_name(self) -> Optional[Dict[str, float]]:
        if self.joint_vel is None:
            return None
        return {name: value for name, value in zip(self.names, self.joint_vel)}

    def __getitem__(self, name: str) -> float:
        if name not in self.names:
            raise UnknownJoint(
                "arm state has no joint %r (runtime joints %s)" % (name, list(self.names)))
        return self.joint_pos[self.names.index(name)]


def _sequence(payload: object, what: str) -> Tuple[float, ...]:
    """1-D payload (list / tuple / numpy array / shim tensor) -> tuple of floats."""
    if hasattr(payload, "tolist"):
        payload = payload.tolist()
    if not isinstance(payload, (list, tuple)):
        raise StateUnavailable("%s is not a 1-D sequence: %r" % (what, type(payload).__name__))
    return tuple(float(value) for value in payload)


class B601RSController:
    """Bind one loaded articulation to the P5 tables and expose the four operations that matter."""

    model_key = MODEL_REBOT_RS

    def __init__(self, robot, *, model: object = None, envelope: Optional[Envelope] = None,
                 reset_fn: Optional[Callable] = None, tensor_factory: Optional[Callable] = None,
                 shuttle=None, dt: float = DEFAULT_DT):
        self.robot = robot
        self.shuttle = shuttle
        self.joint_names: Tuple[str, ...] = tuple(str(name) for name in robot.joint_names)
        # P5-04 + the P5B model resolver: both raise (unknown model / unknown / duplicate DOF)
        # instead of falling back to a positional guess. The index is built FIRST so that a
        # duplicated or absent runtime list is reported as what it is (a DOF-identity defect) rather
        # than as 'no known arm', which would send the reader looking at the model table instead of
        # at the articulation that reported the names.
        JointIndex.build(self.joint_names, model="runtime")
        self.model = resolve_model(self.joint_names, robot_key=model)
        self.index = JointIndex.build(self.joint_names, required=limits_for(self.model),
                                      model=self.model)
        self.envelope = envelope if envelope is not None else envelope_for(self.model)
        self.gate = SafetyGate(self.model, self.joint_names, envelope=self.envelope, dt=dt)
        self._reset_fn = reset_fn
        self._tensor_factory = tensor_factory
        self.commands: list = []   # (runtime-ordered target, dt) - the accepted history, for audit

    # -- identity ----------------------------------------------------------
    @property
    def dof_count(self) -> int:
        return self.index.dof_count

    @property
    def dof_index(self) -> Dict[str, int]:
        return self.index.mapping

    @property
    def holding(self) -> bool:
        return self.gate.holding

    @property
    def held_target(self) -> Optional[Tuple[float, ...]]:
        return self.gate.held_target

    def describe(self) -> Dict[str, object]:
        return {"model": self.model, "joint_names": list(self.joint_names),
                "dof_index": self.dof_index, "dof_count": self.dof_count,
                "holding": self.holding,
                "position_limits": {name: [limit.lower, limit.upper, limit.unit]
                                    for name, limit in self.envelope.limits.items()},
                "envelope_source": self.envelope.source}

    # -- 1. state ----------------------------------------------------------
    def read_state(self) -> ArmState:
        """Read the articulation state, addressed by name; order comes from the runtime list."""
        data = getattr(self.robot, "data", None)
        joint_pos = getattr(data, "joint_pos", None) if data is not None else None
        if joint_pos is None:
            raise StateUnavailable(
                "robot.data.joint_pos is not available; this controller will not invent a state "
                "(a fabricated zero pose is indistinguishable from a real one downstream)")
        positions = _sequence(joint_pos, "robot.data.joint_pos")
        if len(positions) != self.dof_count:
            raise StateUnavailable(
                "robot.data.joint_pos has %d values for %d runtime DOFs %s"
                % (len(positions), self.dof_count, list(self.joint_names)))
        joint_vel = getattr(data, "joint_vel", None)
        velocities = None
        if joint_vel is not None:
            velocities = _sequence(joint_vel, "robot.data.joint_vel")
            if len(velocities) != self.dof_count:
                raise StateUnavailable(
                    "robot.data.joint_vel has %d values for %d runtime DOFs"
                    % (len(velocities), self.dof_count))
        return ArmState(names=self.joint_names, joint_pos=positions, joint_vel=velocities,
                        source="/".join(name for name in (self.model, "runtime")))

    # -- 2. joint targets --------------------------------------------------
    def set_joint_targets(self, joint_pos: object, *, dt: Optional[float] = None) -> Tuple[float, ...]:
        """Validate and latch a setpoint. Mapping (by name) or runtime-ordered sequence."""
        if isinstance(joint_pos, Mapping):
            # Named form: unknown / absent names must fail with the offending NAME, which the
            # positional gate cannot know. validate_named() maps the names onto runtime order.
            ordered = self.gate.validate_named(joint_pos, dt=dt)
        else:
            ordered = joint_pos
        values = self.gate.accept(ordered, dt=dt)
        self.commands.append((values, dt))
        return values

    def seed(self, joint_pos: object) -> Tuple[float, ...]:
        """Re-baseline after a reset/teleport so the next command is rate-limited correctly."""
        return self.gate.seed(joint_pos)

    # -- 3. HOLD -----------------------------------------------------------
    def hold(self) -> Tuple[float, ...]:
        return self.gate.hold()

    def release(self) -> None:
        self.gate.release()

    # -- 4. reset ----------------------------------------------------------
    def reset(self, joint_pos: object, *, env_ids=None, dt: Optional[float] = None):
        """Write one POSITION target through badminton_scene.reset.reset_idx, after checking it.

        joint_pos is REQUIRED. reset_idx(None) means 'use the model-aware rest pose' by design (P5B);
        the controller has a separate, explicit reset_to_rest() for that, so passing None here is a
        caller bug and raises instead of being reinterpreted.
        """
        if self.holding:
            raise HoldActive(
                "HOLD is engaged: refusing a reset while the arm is held at %r; release() first"
                % (self.gate.held_target,), value=self.gate.held_target)
        rows = self._reset_rows(joint_pos)
        # Check the target HERE as well as inside reset_idx(): the controller must refuse before it
        # needs a tensor, a torch install or a sim handle, and it must report the offending joint.
        self._check_rows(rows)
        tensor = self._as_tensor(rows)
        reset_fn = self._resolver()
        result = reset_fn(env_ids, self.robot, self.shuttle, joint_pos=tensor)
        # The joints moved without being commanded, so the rate baseline has to follow them.
        self.seed(rows[0])
        return result

    def reset_to_rest(self, *, env_ids=None):
        """Write the model-aware rest pose (the P5B pose) through the same checked path."""
        target = rest_joint_pos(self.model, self.joint_names)
        values = tuple(target[name] for name in self.joint_names)
        return self.reset(values, env_ids=env_ids)

    def _reset_rows(self, joint_pos: object) -> Tuple[Tuple[float, ...], ...]:
        if joint_pos is None:
            raise ResetPoseError(
                "reset() was given no joint_pos. This controller does not reinterpret None as the "
                "rest pose: call reset_to_rest() for that, or pass the pose you mean.")
        if isinstance(joint_pos, Mapping):
            joint_pos = self.index.from_mapping(joint_pos)
        if hasattr(joint_pos, "tolist"):
            joint_pos = joint_pos.tolist()
        if not isinstance(joint_pos, (list, tuple)):
            raise ResetPoseError(
                "joint_pos is %r, not a sequence of joint values"
                % (type(joint_pos).__name__,))
        if len(joint_pos) == 0:
            raise ResetPoseError("joint_pos is empty; a reset needs one value per runtime DOF")
        first = joint_pos[0]
        if isinstance(first, (list, tuple)):
            rows = tuple(tuple(float(value) for value in row) for row in joint_pos)
            widths = {len(row) for row in rows}
            if len(widths) != 1:
                raise ResetPoseError(
                    "joint_pos rows have different widths %s" % sorted(widths))
            # Several rows are allowed (a multi-env write); every row must still be a full DOF
            # vector, and the rate baseline is re-seeded from row 0 - this controller drives one
            # arm, so rows are expected to be distinct environment copies of the same arm.
        else:
            rows = (tuple(float(value) for value in joint_pos),)
        width = len(rows[0])
        if width != self.dof_count:
            raise ResetPoseError(
                "joint_pos has %d columns but the %s robot has %d joints %s"
                % (width, self.model, self.dof_count, list(self.joint_names)))
        return rows

    def _check_rows(self, rows: Tuple[Tuple[float, ...], ...]) -> None:
        """Every value finite and inside its joint's model limits, before anything is written."""
        for row_index, row in enumerate(rows):
            for name, value in zip(self.joint_names, row):
                limit = self.envelope.limits[name]
                if not math.isfinite(value):
                    raise RestPoseLimitViolation(
                        "reset target row %d %s = %r is not a finite number"
                        % (row_index, name, value))
                if not (limit.lower <= value <= limit.upper):
                    raise RestPoseLimitViolation(
                        "reset target row %d %s = %r is outside [%r, %r] %s of the %s model table"
                        % (row_index, name, value, limit.lower, limit.upper, limit.unit,
                           self.model))

    def _as_tensor(self, rows: Tuple[Tuple[float, ...], ...]):
        factory = self._tensor_factory
        if factory is None:
            import torch  # lazy: the module stays importable without Isaac / torch installed

            factory = torch.tensor
        return factory([list(row) for row in rows])

    def _resolver(self) -> Callable:
        if self._reset_fn is not None:
            return self._reset_fn

        def _default(env_ids, robot, shuttle, joint_pos=None):
            from .reset import reset_idx  # lazy for the same reason as torch above

            return reset_idx(env_ids, robot, shuttle, joint_pos=joint_pos)

        return _default

    # -- 5. IK hook --------------------------------------------------------
    def solve_ik(self, target_pose: object, *, seed: Optional[Sequence[float]] = None):
        """Declared, unimplemented. Raises rather than returning a plausible-looking pose."""
        raise IkNotImplemented(
            "solve_ik() is not implemented: this repo has no IK solver for the B601-RS racket "
            "target and no calibrated TCP frame to solve against (P5_CONTROLLER_AND_HOLD.md "
            "section 5). Returning the current pose or a zero vector would be indistinguishable "
            "from a solved pose downstream, so this raises NotImplementedError instead.")


__all__ = [
    "ArmState", "B601RSController", "IkNotImplemented", "StateUnavailable",
]
