# -*- coding: utf-8 -*-
"""P5-07 / P5-08: an OFFLINE multi-environment harness for N independent B601-RS arms.

WHAT THIS IS, AND WHAT IT IS NOT
    This module drives N independent instances of the P5-09 controller through an INJECTED sim
    adapter (duck-typed: read_state / write_targets / reset). It is a harness and a contract, not a
    simulator: the bundled OfflineMockSim is a bookkeeping mock whose only physics is "the last
    accepted target becomes the joint position". Nothing here imports Isaac Sim, torch, pxr or a
    GPU, and no number produced by this file is a statement about Isaac throughput. See
    measure_offline_mock_throughput() and management/morph_one_rs/P5_MULTIENV_OFFLINE.md.

WHY IT EXISTS
    A single-arm controller can be correct and still be useless in a batched RL run: the failure
    modes of a multi-env job are not per-joint arithmetic errors, they are CROSS-ENV leaks -
    one shared RNG, one shared gate, one shared adapter row, one shared 'last target'. When env 3
    hits a limit and env 4's state moves, every reward in the batch is suspect and nothing in a
    per-env unit test can see it. So the one property this harness is built around is:

        an action in env A must leave env B's STATE DIGEST bit-identical.

    The digest (state_digest) covers the adapter's joint positions, the accepted setpoint, the HOLD
    latch, the step counter and the per-env RNG state. All floats are hashed as float.hex(), so
    "bit-identical" means what it says: not close, not equal to 1e-9, identical bits.

PER-ENV STATE (nothing is shared)
    seed        derived per env: sha256("<seed>|<env_id>"), never a shared Random object
    rng         random.Random with that derived seed; every stochastic draw goes through it
    rest pose   rest_pose.rest_joint_pos(MODEL_REBOT_RS, names) - the B601-RS pose (joint2 0.3,
                joint3 1.0, fingers mid-travel). The PiPER pair (joint2 1.5, joint3 -1.5) is NOT
                reachable on this arm: joint3 is [0, 3.14], so -1.5 rad is out of range. The harness
                validates the rest pose on construction and raises rather than writing it.
    gate        one SafetyGate (P5-05) per env, so HOLD is per env and cannot latch the batch
    adapter row one env_id per sim call; there is no batched row index to get wrong
    ledger      rejections / faults / writes are recorded per env

EVERY WRITE GOES THROUGH THE P5-05 ENVELOPE
    step() has exactly one order: SafetyGate.accept() first, adapter.write_targets() second. A
    rejected target is recorded in RejectedWrite with the typed code from safety.CommandRejected
    (non_finite / position_limit / unknown_joint / hold_active / velocity_limit / ...) and the
    adapter is NOT called. A test can therefore assert both halves: the typed reason exists AND the
    adapter's write count did not move.

FAULTS ARE INJECTED, CONTAINED AND TYPED
    nan_target, out_of_limit, unknown_joint, mid_run_hold, adapter_exception, timeout,
    restart_resume. Each is recorded as a FaultEvent with the env id, a typed code and a detail
    string, and each is contained: the other envs' digests do not change. Two of them are honest
    about their limits rather than papering over them:

      * adapter_exception  the gate has already latched the accepted setpoint, so the plant is BEHIND
                          the gate. The event is recorded, env.write_diverged is set, and resume()
                          re-seeds the rate baseline from the adapter's real state. A silent retry
                          would hide a plant/controller divergence, which is the thing to see.
      * timeout            the write runs in a worker thread with a deadline. A thread cannot be
                          killed, so a timed-out write MAY STILL LAND in the adapter afterwards; the
                          env is quarantined, the event says late_write_possible=True, and
                          wait_for_pending() joins the worker so a test can see what actually landed.
                          This is a genuine limitation of an in-process harness, not a feature.

RESTART AND RESUME
    restart_and_resume() rebuilds the controller from the per-env checkpoint (adapter pose, accepted
    target, HOLD latch, RNG state, counters) and re-materialises the pose through the adapter. The
    digest after a resume equals the digest before the restart, bit for bit. That is a claim about
    THIS harness's recorded state. A real Isaac restart also loses PhysX solver/warm-start state
    that is not recorded here at all - see the UNKNOWN list in the management report.

NO RUNTIME DEPENDENCIES
    stdlib only (hashlib, random, threading, time, tracemalloc for the measurement). Importable and
    runnable on a plain CPU box with no simulation stack installed.
"""
from __future__ import annotations

import hashlib
import math
import platform
import random
import sys
import threading
import time
import tracemalloc
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Mapping, Optional, Sequence, Tuple

from .controller import B601RSController
from .rest_pose import (
    MODEL_REBOT_RS,
    ResetPoseError,
    rest_joint_pos,
    validate_rest_pose,
)
from .safety import DEFAULT_DT, CommandRejected, HoldActive

#: The label that must travel with every number this module produces.
OFFLINE_LABEL = "OFFLINE-MOCK — not Isaac throughput"

# -- fault kinds (the injector's vocabulary, one string per failure mode) --------------------
FAULT_NAN_TARGET = "nan_target"
FAULT_OUT_OF_LIMIT = "out_of_limit"
FAULT_UNKNOWN_JOINT = "unknown_joint"
FAULT_MID_RUN_HOLD = "mid_run_hold"
FAULT_ADAPTER_EXCEPTION = "adapter_exception"
FAULT_TIMEOUT = "timeout"
FAULT_RESTART_RESUME = "restart_resume"

QUEUED_FAULTS = (
    FAULT_NAN_TARGET, FAULT_OUT_OF_LIMIT, FAULT_UNKNOWN_JOINT,
    FAULT_MID_RUN_HOLD, FAULT_ADAPTER_EXCEPTION, FAULT_TIMEOUT,
)
FAULT_KINDS = QUEUED_FAULTS + (FAULT_RESTART_RESUME,)

#: Typed reason recorded for each injected kind. Where the P5-05 gate owns the reason the gate's
#: own code is used instead, so a reader can join a fault line against a safety rejection.
FAULT_CODES = {
    FAULT_NAN_TARGET: "non_finite",
    FAULT_OUT_OF_LIMIT: "position_limit",
    FAULT_UNKNOWN_JOINT: "unknown_joint",
    FAULT_MID_RUN_HOLD: "hold_active",
    FAULT_ADAPTER_EXCEPTION: "adapter_exception",
    FAULT_TIMEOUT: "timeout",
    FAULT_RESTART_RESUME: "restart_resume",
}

#: Status of one env's controller relative to its plant.
STATUS_OK = "ok"
STATUS_QUARANTINED = "quarantined"


class MultiEnvError(RuntimeError):
    """Base class: the harness refuses an operation it cannot justify."""


class UnknownEnv(MultiEnvError):
    """An env id was used that this harness does not own."""


class EnvQuarantined(MultiEnvError):
    """The env is quarantined after a timeout; resume() it before commanding again."""


class UnknownFault(MultiEnvError):
    """inject() was given a fault kind this harness does not implement."""


# --------------------------------------------------------------------------- records
@dataclass(frozen=True)
class FaultEvent:
    """One injected fault, as a typed record. Never raised and dropped: always in a ledger."""

    env_id: int
    kind: str
    code: str
    detail: str
    step: int
    contained: bool = True
    late_write_possible: bool = False

    def as_dict(self) -> Dict[str, object]:
        return {"env_id": self.env_id, "kind": self.kind, "code": self.code,
                "detail": self.detail, "step": self.step, "contained": self.contained,
                "late_write_possible": self.late_write_possible}


@dataclass(frozen=True)
class RejectedWrite:
    """A target the P5-05 envelope refused BEFORE the adapter was touched."""

    env_id: int
    step: int
    code: str
    joint: Optional[str]
    index: Optional[int]
    value: object
    limit: object
    message: str

    def as_dict(self) -> Dict[str, object]:
        return {"env_id": self.env_id, "step": self.step, "code": self.code,
                "joint": self.joint, "index": self.index, "value": self.value,
                "limit": self.limit, "message": self.message}


@dataclass(frozen=True)
class StepResult:
    """What one step() did: accepted-and-written, or rejected, or faulted."""

    env_id: int
    step: int
    accepted: bool
    values: Optional[Tuple[float, ...]]
    code: str
    detail: str = ""

    def as_dict(self) -> Dict[str, object]:
        return {"env_id": self.env_id, "step": self.step, "accepted": self.accepted,
                "values": None if self.values is None else list(self.values),
                "code": self.code, "detail": self.detail}


@dataclass
class EnvCheckpoint:
    """Everything the harness can restore after a restart. Nothing else is restorable."""

    pose: Tuple[float, ...]
    target: Optional[Tuple[float, ...]]
    holding: bool
    rng_state: object
    attempts: int
    writes: int


# --------------------------------------------------------------------------- adapter
class OfflineMockSim:
    """The injected adapter: one joint vector per env, writes recorded, failures injectable.

    It is deliberately dumb. Its only 'physics' is that an accepted setpoint becomes the joint
    position, which is enough to detect cross-env leaks and NOT enough to say anything about
    dynamics, contact or real Isaac behaviour. It records non-finite writes instead of rejecting
    them, because the point of the harness is that the adapter must never be handed one.
    """

    def __init__(self, joint_names: Sequence[str], env_count: int,
                 initial_pose: Sequence[float], *, label: str = "offline-mock"):
        self.joint_names: Tuple[str, ...] = tuple(str(name) for name in joint_names)
        self.env_count = int(env_count)
        if self.env_count <= 0:
            raise MultiEnvError("env_count must be positive, got %r" % (env_count,))
        if len(initial_pose) != len(self.joint_names):
            raise MultiEnvError(
                "initial_pose has %d values for %d joints" % (len(initial_pose), len(self.joint_names)))
        pose = tuple(float(value) for value in initial_pose)
        self.label = label
        self._lock = threading.Lock()
        self._pos: Dict[int, Tuple[float, ...]] = {env: pose for env in range(self.env_count)}
        self._writes: Dict[int, List[Tuple[float, ...]]] = {env: [] for env in range(self.env_count)}
        self._resets: Dict[int, List[Tuple[float, ...]]] = {env: [] for env in range(self.env_count)}
        self._fail: Dict[int, BaseException] = {}
        self._delay: Dict[int, float] = {}
        #: Non-finite values this adapter was HANDED. Must stay empty - that is the assertion.
        self.nonfinite_writes: List[Tuple[int, Tuple[float, ...]]] = []
        self.call_count = 0

    # -- duck-typed adapter surface ----------------------------------------
    def read_state(self, env_id: int) -> Tuple[float, ...]:
        self._check_env(env_id)
        with self._lock:
            return self._pos[int(env_id)]

    def write_targets(self, env_id: int, targets: Sequence[float]) -> None:
        self._check_env(env_id)
        env_id = int(env_id)
        delay = self._delay.pop(env_id, 0.0)
        if delay > 0.0:
            time.sleep(delay)  # a slow adapter: the harness's deadline is what has to bound it
        failure = self._fail.pop(env_id, None)
        if failure is not None:
            # Raised BEFORE any mutation: a failed write leaves the plant where it was.
            raise failure
        values = tuple(float(value) for value in targets)
        if len(values) != len(self.joint_names):
            raise MultiEnvError("write_targets got %d values for %d joints"
                                % (len(values), len(self.joint_names)))
        with self._lock:
            self.call_count += 1
            self._writes[env_id].append(values)
            if not all(math.isfinite(value) for value in values):
                self.nonfinite_writes.append((env_id, values))
            self._pos[env_id] = values

    def reset(self, env_id: int, pose: Sequence[float]) -> None:
        self._check_env(env_id)
        values = tuple(float(value) for value in pose)
        if len(values) != len(self.joint_names):
            raise MultiEnvError("reset got %d values for %d joints"
                                % (len(values), len(self.joint_names)))
        with self._lock:
            self.call_count += 1
            self._resets[int(env_id)].append(values)
            self._pos[int(env_id)] = values

    # -- fault knobs -------------------------------------------------------
    def fail_next_write(self, env_id: int, exception: Optional[BaseException] = None) -> None:
        self._check_env(env_id)
        self._fail[int(env_id)] = exception or RuntimeError(
            "injected adapter failure for env %d" % int(env_id))

    def delay_next_write(self, env_id: int, seconds: float) -> None:
        self._check_env(env_id)
        self._delay[int(env_id)] = float(seconds)

    # -- inspection --------------------------------------------------------
    def writes(self, env_id: int) -> List[Tuple[float, ...]]:
        self._check_env(env_id)
        return list(self._writes[int(env_id)])

    def write_count(self, env_id: int) -> int:
        return len(self.writes(env_id))

    def resets(self, env_id: int) -> List[Tuple[float, ...]]:
        self._check_env(env_id)
        return list(self._resets[int(env_id)])

    def reset_count(self, env_id: int) -> int:
        return len(self.resets(env_id))

    def _check_env(self, env_id: int) -> None:
        try:
            known = int(env_id) in self._pos
        except (TypeError, ValueError):
            known = False
        if not known:
            raise UnknownEnv("env %r is not one of this adapter's %d envs"
                             % (env_id, self.env_count))


class EnvRobotProxy:
    """A per-env 'robot' object: joint_pos is read LIVE from that env's adapter row.

    Nothing is cached. A cached row would be exactly the shared-state bug this module exists to
    exclude: two envs reading one buffer look independent until the first reset.
    """

    def __init__(self, sim, env_id: int, joint_names: Sequence[str]):
        self._sim = sim
        self._env_id = int(env_id)
        self.joint_names = list(joint_names)

    @property
    def data(self):
        return _RobotData(self._sim.read_state(self._env_id))


class _RobotData:
    __slots__ = ("joint_pos", "joint_vel")

    def __init__(self, joint_pos: Sequence[float]):
        self.joint_pos = tuple(float(value) for value in joint_pos)
        self.joint_vel = None  # this mock has no velocity: it is not a measurement that is missing


def _plain_tensor(rows: Sequence[Sequence[float]]) -> Tuple[Tuple[float, ...], ...]:
    """tensor_factory stand-in: the controller's reset path without torch."""
    return tuple(tuple(float(value) for value in row) for row in rows)


def _hex(values: Optional[Sequence[float]]) -> str:
    """Exact, order-sensitive encoding of a float sequence (bit-identical, not 'close')."""
    if values is None:
        return "-"
    return ",".join(float(value).hex() for value in values)


def derive_seed(seed: int, env_id: int) -> int:
    """One deterministic 63-bit seed per (job seed, env id). No shared Random anywhere."""
    digest = hashlib.sha256(("%d|%d" % (int(seed), int(env_id))).encode("ascii")).digest()
    return int.from_bytes(digest[:8], "big") >> 1


# --------------------------------------------------------------------------- env state
@dataclass
class _EnvRuntime:
    env_id: int
    seed: int
    rng: random.Random
    robot: EnvRobotProxy
    controller: B601RSController
    rest: Tuple[float, ...]
    status: str = STATUS_OK
    attempts: int = 0
    writes: int = 0
    restarts: int = 0
    pending_fault: Optional[str] = None
    write_diverged: bool = False
    checkpoint: Optional[EnvCheckpoint] = None
    rejections: List[RejectedWrite] = field(default_factory=list)
    faults: List[FaultEvent] = field(default_factory=list)


class MultiEnvHarness:
    """N independent B601-RS controllers over one injected sim adapter."""

    def __init__(self, sim, *, model: object = MODEL_REBOT_RS,
                 env_count: Optional[int] = None, seeds: Optional[Sequence[int]] = None,
                 dt: float = DEFAULT_DT, write_timeout: float = 0.5,
                 timeout_delay: Optional[float] = None):
        self.sim = sim
        self.model = model
        self.dt = float(dt)
        self.write_timeout = float(write_timeout)
        # A timeout fault is only a timeout if the adapter really outlives the deadline: an explicit
        # delay under the deadline would silently turn the fault into a slow success.
        self.timeout_delay = (2.0 * self.write_timeout if timeout_delay is None
                              else max(float(timeout_delay), 2.0 * self.write_timeout))
        self.joint_names: Tuple[str, ...] = tuple(str(name) for name in sim.joint_names)

        adapter_count = getattr(sim, "env_count", None)
        if env_count is None:
            if seeds is None:
                raise MultiEnvError("give env_count or seeds; the harness will not guess the batch")
            env_count = len(seeds)
        self.env_count = int(env_count)
        if self.env_count <= 0:
            raise MultiEnvError("env_count must be positive, got %r" % (self.env_count,))
        if adapter_count is not None and int(adapter_count) != self.env_count:
            raise MultiEnvError(
                "the adapter owns %d envs but the harness was asked for %d"
                % (int(adapter_count), self.env_count))
        if seeds is None:
            seeds = tuple(1000 + env for env in range(self.env_count))
        self.seeds: Tuple[int, ...] = tuple(int(seed) for seed in seeds)
        if len(self.seeds) != self.env_count:
            raise MultiEnvError("%d seeds for %d envs" % (len(self.seeds), self.env_count))

        self._pending: List[threading.Thread] = []
        self._envs: Dict[int, _EnvRuntime] = {}
        for env_id in range(self.env_count):
            self._envs[env_id] = self._build(env_id, self.seeds[env_id])

    # -- construction ------------------------------------------------------
    def _rest_values(self) -> Tuple[float, ...]:
        """The B601-RS rest pose in runtime order, validated before it can be written."""
        pose = rest_joint_pos(self.model, self.joint_names)   # raises for a foreign joint set
        validate_rest_pose(self.model, pose)                 # raises on a stop / out of limits
        return tuple(float(pose[name]) for name in self.joint_names)

    def _build(self, env_id: int, seed: int) -> _EnvRuntime:
        rest = self._rest_values()
        robot = EnvRobotProxy(self.sim, env_id, self.joint_names)
        controller = B601RSController(
            robot, model=self.model, reset_fn=self._reset_fn(env_id),
            tensor_factory=_plain_tensor, dt=self.dt)
        controller.seed(rest)   # the rate baseline starts at the pose the arm actually has
        state = _EnvRuntime(
            env_id=env_id, seed=seed, rng=random.Random(derive_seed(seed, env_id)),
            robot=robot, controller=controller, rest=rest)
        self._checkpoint(state)
        return state

    def _reset_fn(self, env_id: int) -> Callable:
        """The controller's reset sink: validated upstream by P5B/P5-09, written here per env."""

        def _write(env_ids, robot, shuttle, joint_pos=None):
            if joint_pos is None:
                raise ResetPoseError("the harness reset sink needs an explicit pose")
            rows = tuple(tuple(float(value) for value in row) for row in joint_pos)
            if len(rows) != 1:
                raise ResetPoseError(
                    "env %d is one arm, but the reset carried %d rows" % (env_id, len(rows)))
            self.sim.reset(env_id, rows[0])
            return {"env_id": env_id, "written": list(rows[0])}

        return _write

    # -- lookup ------------------------------------------------------------
    def _env(self, env_id: int) -> _EnvRuntime:
        try:
            return self._envs[int(env_id)]
        except (KeyError, TypeError, ValueError):
            raise UnknownEnv("env %r is not one of this harness's %d envs"
                             % (env_id, self.env_count)) from None

    @property
    def env_ids(self) -> Tuple[int, ...]:
        return tuple(sorted(self._envs))

    # -- state -------------------------------------------------------------
    def read_state(self, env_id: int) -> Dict[str, object]:
        env = self._env(env_id)
        state = env.controller.read_state()
        return {"env_id": env.env_id, "names": list(state.names),
                "joint_pos": dict(state.by_name()), "holding": env.controller.holding,
                "status": env.status}

    def state_digest(self, env_id: int) -> str:
        """sha256 over the env's exact state: positions, setpoint, HOLD latch, counters, RNG."""
        env = self._env(env_id)
        payload = "|".join((
            "env=%d" % env.env_id,
            "seed=%d" % env.seed,
            "attempts=%d" % env.attempts,
            "writes=%d" % env.writes,
            "holding=%d" % (1 if env.controller.holding else 0),
            "status=%s" % env.status,
            "names=%s" % ",".join(self.joint_names),
            "pos=%s" % _hex(self.sim.read_state(env.env_id)),
            "target=%s" % _hex(env.controller.gate.last_valid),
            "rng=%s" % hashlib.sha256(repr(env.rng.getstate()).encode("utf-8")).hexdigest(),
        ))
        return hashlib.sha256(payload.encode("ascii")).hexdigest()

    def digests(self) -> Dict[int, str]:
        return {env_id: self.state_digest(env_id) for env_id in self.env_ids}

    def rest_pose(self, env_id: int) -> Tuple[float, ...]:
        return self._env(env_id).rest

    @property
    def rest_pose_values(self) -> Tuple[float, ...]:
        return self._rest_values()

    # -- stochastic targets (per-env RNG only) ------------------------------
    def random_target(self, env_id: int) -> Tuple[float, ...]:
        """A deterministic in-limit, rate-bounded target drawn from THIS env's own RNG.

        Each joint moves at most half its per-step velocity cap, so the sample is accepted by the
        P5-05 envelope by construction - which keeps the throughput measurement measuring the
        harness, not the rejection path.
        """
        env = self._env(env_id)
        envelope = env.controller.envelope
        base = env.controller.gate.last_valid
        if base is None:
            base = env.rest
        values = []
        for position, name in enumerate(self.joint_names):
            limit = envelope.limits[name]
            # Bounded by the velocity cap AND by the acceleration cap, because the gate checks the
            # second difference of the setpoint: a step that honours only the velocity cap is
            # rejected on every second call (20 rad/s^2 at dt = 1/60 s allows ~2.8e-3 rad/step).
            # 0.45 instead of 0.5 leaves room for the limit clamp below.
            span = min(0.5 * envelope.velocity[name] * self.dt,
                       0.45 * envelope.acceleration[name] * self.dt * self.dt)
            value = base[position] + env.rng.uniform(-span, span)
            values.append(min(max(value, limit.lower), limit.upper))
        return tuple(values)

    # -- 1. commands -------------------------------------------------------
    def step(self, env_id: int, target=None, *, dt: Optional[float] = None) -> StepResult:
        """Validate through P5-05, then (and only then) write to that env's adapter row."""
        env = self._env(env_id)
        if env.status == STATUS_QUARANTINED:
            raise EnvQuarantined(
                "env %d is quarantined after a timeout; call resume(%d) first"
                % (env.env_id, env.env_id))
        env.attempts += 1
        kind = env.pending_fault
        env.pending_fault = None
        payload = self._payload(env, kind, target)
        if kind == FAULT_MID_RUN_HOLD:
            env.controller.hold()   # the arm freezes here; the command below must be refused

        try:
            values = env.controller.set_joint_targets(payload, dt=dt)
        except CommandRejected as rejection:
            record = RejectedWrite(
                env_id=env.env_id, step=env.attempts, code=rejection.code,
                joint=rejection.joint, index=rejection.index, value=rejection.value,
                limit=rejection.limit, message=str(rejection))
            env.rejections.append(record)
            self._checkpoint(env)
            return StepResult(env_id=env.env_id, step=env.attempts, accepted=False, values=None,
                              code=record.code, detail=record.message)

        if kind == FAULT_ADAPTER_EXCEPTION:
            self.sim.fail_next_write(env.env_id)
        elif kind == FAULT_TIMEOUT:
            self.sim.delay_next_write(env.env_id, self.timeout_delay)

        status, outcome = self._guarded(lambda: self.sim.write_targets(env.env_id, values),
                                        self.write_timeout)
        if status == "timeout":
            event = FaultEvent(
                env_id=env.env_id, kind=FAULT_TIMEOUT, code=FAULT_CODES[FAULT_TIMEOUT],
                detail="write did not return within %.3f s; the worker thread cannot be killed, so "
                       "the write may still land afterwards" % self.write_timeout,
                step=env.attempts, late_write_possible=True)
            env.faults.append(event)
            env.status = STATUS_QUARANTINED
            return StepResult(env_id=env.env_id, step=env.attempts, accepted=False, values=None,
                              code=event.code, detail=event.detail)
        if status == "error":
            event = FaultEvent(
                env_id=env.env_id, kind=FAULT_ADAPTER_EXCEPTION,
                code=FAULT_CODES[FAULT_ADAPTER_EXCEPTION],
                detail="adapter raised %s: %s; the gate already latched the setpoint, so the plant "
                       "is behind the gate until resume()" % (type(outcome).__name__, outcome),
                step=env.attempts)
            env.faults.append(event)
            env.write_diverged = True
            self._checkpoint(env)
            return StepResult(env_id=env.env_id, step=env.attempts, accepted=False, values=None,
                              code=event.code, detail=event.detail)

        env.writes += 1
        self._checkpoint(env)
        return StepResult(env_id=env.env_id, step=env.attempts, accepted=True, values=values,
                          code="accepted")

    def _payload(self, env: _EnvRuntime, kind: Optional[str], target):
        """Build the payload for one step, applying the queued fault to it when there is one."""
        if kind is None:
            return self.random_target(env.env_id) if target is None else target
        if kind == FAULT_NAN_TARGET:
            base = list(env.controller.gate.last_valid or env.rest)
            index = self.joint_names.index("joint3") if "joint3" in self.joint_names else 1
            base[index] = float("nan")
            return tuple(base)
        if kind == FAULT_OUT_OF_LIMIT:
            base = list(env.controller.gate.last_valid or env.rest)
            name = "joint3" if "joint3" in self.joint_names else self.joint_names[1]
            limit = env.controller.envelope.limits[name]
            base[self.joint_names.index(name)] = limit.upper + 0.5
            return tuple(base)
        if kind == FAULT_UNKNOWN_JOINT:
            name = self.joint_names[2]
            return {name + "_typo": 0.1}
        if kind in (FAULT_MID_RUN_HOLD, FAULT_ADAPTER_EXCEPTION, FAULT_TIMEOUT):
            return self.random_target(env.env_id) if target is None else target
        raise UnknownFault("no payload rule for fault kind %r" % (kind,))

    # -- 2. HOLD -----------------------------------------------------------
    def hold(self, env_id: int) -> Tuple[float, ...]:
        env = self._env(env_id)
        held = env.controller.hold()
        self._checkpoint(env)
        return held

    def release(self, env_id: int) -> None:
        env = self._env(env_id)
        env.controller.release()
        self._checkpoint(env)

    @property
    def holding(self) -> Dict[int, bool]:
        return {env_id: self._envs[env_id].controller.holding for env_id in self.env_ids}

    # -- 3. reset ----------------------------------------------------------
    def reset(self, env_id: int, pose=None) -> Dict[str, object]:
        """Reset one env. A reset in env A may not move env B (asserted by the tests)."""
        env = self._env(env_id)
        if env.status == STATUS_QUARANTINED:
            raise EnvQuarantined(
                "env %d is quarantined after a timeout; call resume(%d) first"
                % (env.env_id, env.env_id))
        try:
            if pose is None:
                result = env.controller.reset_to_rest()
            else:
                result = env.controller.reset(pose)
        except CommandRejected as rejection:
            record = RejectedWrite(
                env_id=env.env_id, step=env.attempts, code=rejection.code,
                joint=rejection.joint, index=rejection.index, value=rejection.value,
                limit=rejection.limit, message=str(rejection))
            env.rejections.append(record)
            self._checkpoint(env)
            return {"env_id": env.env_id, "written": None, "code": record.code}
        env.writes += 1
        self._checkpoint(env)
        return {"env_id": env.env_id, "written": (result or {}).get("written"), "code": "accepted"}

    def reset_to_rest(self, env_id: int) -> Dict[str, object]:
        return self.reset(env_id, None)

    # -- 4. faults ---------------------------------------------------------
    def inject(self, env_id: int, kind: str) -> Dict[str, object]:
        """Queue one fault for the next step() of that env (restart_resume acts immediately)."""
        env = self._env(env_id)
        if kind not in FAULT_KINDS:
            raise UnknownFault("unknown fault kind %r; implemented kinds: %s"
                               % (kind, ", ".join(FAULT_KINDS)))
        if kind == FAULT_RESTART_RESUME:
            return self.restart_and_resume(env_id)
        if env.status == STATUS_QUARANTINED:
            raise EnvQuarantined(
                "env %d is quarantined; resume(%d) before injecting again"
                % (env.env_id, env.env_id))
        env.pending_fault = kind
        return {"env_id": env.env_id, "kind": kind, "queued": True,
                "code": FAULT_CODES[kind]}

    def faults(self, env_id: int) -> List[Dict[str, object]]:
        return [event.as_dict() for event in self._env(env_id).faults]

    def rejections(self, env_id: int) -> List[Dict[str, object]]:
        return [record.as_dict() for record in self._env(env_id).rejections]

    def events(self, env_id: int) -> List[Dict[str, object]]:
        """One ledger per env: every rejection and every fault, each with its typed code."""
        env = self._env(env_id)
        rows: List[Dict[str, object]] = [dict(kind="rejection", **record.as_dict())
                                         for record in env.rejections]
        rows.extend(dict(kind="fault", **event.as_dict()) for event in env.faults)
        return rows

    def status(self, env_id: int) -> str:
        return self._env(env_id).status

    def controller(self, env_id: int) -> B601RSController:
        """The per-env controller (for describe()/envelope inspection in tests and reports)."""
        return self._env(env_id).controller

    def write_diverged(self, env_id: int) -> bool:
        """True while the accepted setpoint is known NOT to have reached the plant."""
        return self._env(env_id).write_diverged

    def restarts(self, env_id: int) -> int:
        """How many times this env's controller was rebuilt from its checkpoint."""
        return self._env(env_id).restarts

    def wait_for_pending(self, timeout: Optional[float] = None) -> int:
        """Join worker threads left behind by a timed-out write. Returns how many were joined."""
        deadline = None if timeout is None else time.monotonic() + float(timeout)
        joined = 0
        still = []
        for thread in list(self._pending):
            budget = 1.0 if deadline is None else max(0.0, deadline - time.monotonic())
            thread.join(budget)
            if thread.is_alive():
                still.append(thread)
            else:
                joined += 1
        self._pending = still
        return joined

    def resume(self, env_id: int) -> Dict[str, object]:
        """Leave quarantine after a timeout, or re-sync after a failed write.

        The rate baseline is re-seeded from the adapter's REAL state, because after a failed or
        late write the setpoint the gate holds is not necessarily the position the plant has.
        """
        self.wait_for_pending()
        env = self._env(env_id)
        actual = tuple(float(value) for value in self.sim.read_state(env.env_id))
        env.controller.seed(actual)
        env.status = STATUS_OK
        emitted = env.write_diverged
        env.write_diverged = False
        self._checkpoint(env)
        return {"env_id": env.env_id, "resumed": True, "pose": list(actual),
                "reconciled_divergence": emitted}

    def restart_and_resume(self, env_id: int) -> Dict[str, object]:
        """Rebuild one env from its checkpoint. Digest before == digest after, bit for bit."""
        self.wait_for_pending()
        env = self._env(env_id)
        checkpoint = env.checkpoint
        if checkpoint is None:
            raise MultiEnvError("env %d has no checkpoint to resume from" % env.env_id)
        self.sim.reset(env.env_id, checkpoint.pose)
        robot = EnvRobotProxy(self.sim, env.env_id, self.joint_names)
        controller = B601RSController(
            robot, model=self.model, reset_fn=self._reset_fn(env.env_id),
            tensor_factory=_plain_tensor, dt=self.dt)
        if checkpoint.target is not None:
            controller.seed(checkpoint.target)
        if checkpoint.holding:
            controller.hold()
        env.controller = controller
        env.robot = robot
        env.rng.setstate(checkpoint.rng_state)
        env.attempts = checkpoint.attempts
        env.writes = checkpoint.writes
        env.status = STATUS_OK
        env.write_diverged = False
        env.pending_fault = None
        env.restarts += 1
        event = FaultEvent(
            env_id=env.env_id, kind=FAULT_RESTART_RESUME, code=FAULT_CODES[FAULT_RESTART_RESUME],
            detail="controller rebuilt from checkpoint (pose, setpoint, HOLD, RNG, counters); "
                   "PhysX solver/warm-start state is NOT recorded, so this is not a complete "
                   "restart model", step=env.attempts)
        env.faults.append(event)
        self._checkpoint(env)
        return event.as_dict()

    # -- 5. bookkeeping ----------------------------------------------------
    def _checkpoint(self, env: _EnvRuntime) -> None:
        env.checkpoint = EnvCheckpoint(
            pose=tuple(float(value) for value in self.sim.read_state(env.env_id)),
            target=None if env.controller.gate.last_valid is None
            else tuple(env.controller.gate.last_valid),
            holding=env.controller.holding,
            rng_state=env.rng.getstate(),
            attempts=env.attempts,
            writes=env.writes,
        )

    def _guarded(self, call: Callable, timeout: float) -> Tuple[str, object]:
        """Run one adapter call under a deadline. A hung call cannot be killed, only abandoned."""
        box: Dict[str, object] = {}

        def run() -> None:
            try:
                box["value"] = call()
            except BaseException as exc:  # noqa: BLE001 - the harness contains adapter failures
                box["error"] = exc

        thread = threading.Thread(target=run, name="p5-multienv-write", daemon=True)
        thread.start()
        thread.join(float(timeout))
        if thread.is_alive():
            self._pending.append(thread)
            return "timeout", None
        if "error" in box:
            return "error", box["error"]
        return "ok", box.get("value")

    def describe(self) -> Dict[str, object]:
        """A JSON-safe summary: one entry per env, plus the offline label."""
        return {
            "label": OFFLINE_LABEL,
            "model": self.model,
            "joint_names": list(self.joint_names),
            "env_count": self.env_count,
            "seeds": list(self.seeds),
            "rest_pose": list(self.rest_pose_values),
            "envs": [{
                "env_id": env_id,
                "status": self._envs[env_id].status,
                "holding": self._envs[env_id].controller.holding,
                "attempts": self._envs[env_id].attempts,
                "writes": self._envs[env_id].writes,
                "restarts": self._envs[env_id].restarts,
                "digest": self.state_digest(env_id),
                "faults": self.faults(env_id),
                "rejections": self.rejections(env_id),
            } for env_id in self.env_ids],
        }


# --------------------------------------------------------------------------- measurement
def build_mock_harness(*, env_count: int = 4, seed: int = 7, model: object = MODEL_REBOT_RS,
                       dt: float = DEFAULT_DT, write_timeout: float = 0.5) -> MultiEnvHarness:
    """A ready-to-drive harness over OfflineMockSim, with one seed per env."""
    names = tuple(rest_joint_pos(model))            # the model's joint names, model order
    rest = rest_joint_pos(model, names)
    initial = tuple(float(rest[name]) for name in names)
    sim = OfflineMockSim(names, env_count, initial)
    return MultiEnvHarness(sim, model=model, env_count=env_count,
                           seeds=tuple(seed + env for env in range(env_count)),
                           dt=dt, write_timeout=write_timeout)


def measure_offline_mock_throughput(*, env_count: int = 4, steps: int = 200, seed: int = 7,
                                    warmup: int = 10) -> Dict[str, object]:
    """Measure THE MOCK: env-steps/second and peak python allocations. Not Isaac anything.

    steps is per env; the reported rate is env-steps/second for the whole batch, plus the per-env
    rate. tracemalloc measures PYTHON allocations of THIS process only - it is not GPU memory, not
    PhysX memory, and not a memory model of the real simulator.
    """
    sim = OfflineMockSim(tuple(rest_joint_pos(MODEL_REBOT_RS)),
                         env_count, tuple(rest_joint_pos(MODEL_REBOT_RS).values()))
    harness = MultiEnvHarness(sim, env_count=env_count,
                              seeds=tuple(seed + env for env in range(env_count)))
    for _ in range(warmup):
        for env_id in harness.env_ids:
            harness.step(env_id)

    tracemalloc.start()
    baseline, _ = tracemalloc.get_traced_memory()
    started = time.perf_counter()
    accepted = 0
    for _ in range(steps):
        for env_id in harness.env_ids:
            accepted += 1 if harness.step(env_id).accepted else 0
    elapsed = time.perf_counter() - started
    current, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    env_steps = steps * env_count
    return {
        "label": OFFLINE_LABEL,
        "what_is_measured": "the OFFLINE mock harness (python bookkeeping + P5-05 validation); "
                            "no Isaac Sim, no PhysX, no GPU was involved",
        "env_count": env_count,
        "steps_per_env": steps,
        "env_steps": env_steps,
        "accepted_env_steps": accepted,
        "wall_seconds": round(elapsed, 6),
        "env_steps_per_second": round(env_steps / elapsed, 1) if elapsed > 0 else None,
        "steps_per_second_per_env": round(steps / elapsed, 1) if elapsed > 0 else None,
        "peak_python_bytes": int(peak),
        "peak_python_mib": round(peak / (1024.0 * 1024.0), 3),
        "python_allocated_bytes_during_run": int(peak - baseline),
        "isaac_throughput": "UNKNOWN - no Isaac run was made, on any GPU, at any env count",
        "gpu_memory": "UNKNOWN - no GPU was used or queried",
        "host": {"python": platform.python_version(), "platform": platform.platform(),
                 "machine": platform.machine()},
    }


def _main(argv: Optional[Sequence[str]] = None) -> int:
    import argparse
    import json
    from pathlib import Path

    parser = argparse.ArgumentParser(
        description="Measure the OFFLINE multi-env mock. %s" % OFFLINE_LABEL)
    parser.add_argument("--envs", type=int, default=4, help="env count to measure")
    parser.add_argument("--steps", type=int, default=200, help="steps per env")
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--out", type=str, default="", help="write the JSON here")
    args = parser.parse_args(list(argv) if argv is not None else None)

    result = measure_offline_mock_throughput(env_count=args.envs, steps=args.steps, seed=args.seed)
    text = json.dumps(result, indent=2, sort_keys=True)
    print(text)
    if args.out:
        path = Path(args.out)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text + "\n", encoding="utf-8")
        print("wrote %s" % path, file=sys.stderr)
    return 0


if __name__ == "__main__":  # pragma: no cover - exercised by the documented command
    raise SystemExit(_main())


__all__ = [
    "OFFLINE_LABEL", "FAULT_KINDS", "QUEUED_FAULTS", "FAULT_CODES",
    "FAULT_NAN_TARGET", "FAULT_OUT_OF_LIMIT", "FAULT_UNKNOWN_JOINT", "FAULT_MID_RUN_HOLD",
    "FAULT_ADAPTER_EXCEPTION", "FAULT_TIMEOUT", "FAULT_RESTART_RESUME",
    "STATUS_OK", "STATUS_QUARANTINED",
    "MultiEnvError", "UnknownEnv", "EnvQuarantined", "UnknownFault",
    "FaultEvent", "RejectedWrite", "StepResult", "EnvCheckpoint",
    "OfflineMockSim", "EnvRobotProxy", "MultiEnvHarness",
    "derive_seed", "build_mock_harness", "measure_offline_mock_throughput",
]
