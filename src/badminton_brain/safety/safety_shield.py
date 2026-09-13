# -*- coding: utf-8 -*-
"""Safety Shield (T8): WholeBodyTarget -> SafeCommand.

Spec: docs/architecture/06_SAFETY.md (Safety is an independent, deterministic, testable
barrier that upstream layers cannot bypass) and docs/simulation/BADMINTON_ROBOT.md
(S4 authenticity levels, S12 TEMP policy).

Implemented in this task (plan 2026-09-13-brain-modules.md, T8 row):
  * 6-axis joint position limits                     (06_SAFETY.md S42-S45)
  * joint velocity limits (per-step command clamp)   (S46, S51)
  * Morph One base twist limits, direction-preserving scaling (S65-S69)
  * racket_contact safe workspace box                (S77)
  * per-env e-stop input with latch + explicit reset (S38-S40, S19 priority)
  * communication timeout / future-dated command     (S25-S27, S31, S35-S37)
  * NaN/Inf -> zero command instead of an exception  (S24)

Deliberately NOT implemented here (other tasks / later revisions): safety projection QP,
predictive collision, geofence footprint, human separation, velocity-aware dynamic joint
envelope (real q, dq, a_brake and T_delay are not measured yet).

Honesty rules (S4/S12):
  * Every limit is a Param.  The default SafetyLimits carries value=None with status
    REQUIRES_MEASUREMENT: no PiPER / Morph One / court-workspace limit has been measured or
    taken from an official source in this repository.
  * The shield refuses to run (BrainBoundaryError) when a limit has no value.
  * SafetyLimits.temp_proxy() provides explicitly TEMP_PARAMETERIZED_PROXY placeholders for
    development and tests; using them raises a RuntimeWarning.
"""
from __future__ import annotations

import warnings
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, Optional, Sequence, Tuple, Union

import numpy as np

from ..interfaces import SafetyModule
from ..status import AssetStatus, Param
from ..types import BrainBoundaryError, SafeCommand, WholeBodyTarget

JOINT_COUNT = 6
BASE_TWIST_DIM = 3
RACKET_POSE_DIM = 7

# Safety actions (06_SAFETY.md S11); the log records the highest-priority action per env.
ACTION_PASS = "PASS"
ACTION_PROJECT = "PROJECT"
ACTION_HOLD = "HOLD"
ACTION_CONTROLLED_STOP = "CONTROLLED_STOP"
ACTION_PROTECTIVE_STOP = "PROTECTIVE_STOP"
ACTION_ESTOP_REQUEST = "ESTOP_REQUEST"

_ACTION_RANK = {
    ACTION_PASS: 0,
    ACTION_PROJECT: 1,
    ACTION_HOLD: 2,
    ACTION_CONTROLLED_STOP: 3,
    ACTION_PROTECTIVE_STOP: 4,
    ACTION_ESTOP_REQUEST: 5,
}

EPS = 1e-12

# --- provenance strings for the unmeasured limits ---------------------------------
_SRC_JOINT_POS = ("PiPER joint position limits: must come from the official PiPER URDF / driver "
                  "configuration for this unit; no PiPER joint limit has been measured in this repo")
_SRC_JOINT_VEL = ("PiPER joint velocity limits: must come from the PiPER specification / driver "
                  "limits; no PiPER joint speed limit has been measured in this repo")
_SRC_BASE_TWIST = ("Morph One base twist limits (vx, vy, wz): must come from the Morph One chassis "
                   "specification or a measured speed test; not measured in this repo")
_SRC_BASE_SPEED = ("Morph One combined translation speed limit sqrt(vx^2+vy^2) (06_SAFETY.md S66): "
                   "must come from the chassis specification / measured test; not measured")
_SRC_WORKSPACE = ("SafeRacketWorkspace box (06_SAFETY.md S77): must be derived from the PiPER "
                  "reachable envelope + racket geometry + court layout (CAD / measurement); not measured")
_SRC_DT = ("safety control period dt: must come from the measured real-time loop rate "
           "(06_SAFETY.md S7 control_dt); not measured")
_SRC_TIMEOUT = ("communication timeout: must come from the measured Brain->driver latency budget "
                "(06_SAFETY.md S26/S31); not measured")
_SRC_CLOCK = ("clock tolerance for future-dated commands (06_SAFETY.md S27): must come from the "
              "measured clock synchronisation error; not measured")

_TEMP_PREFIX = ("TEMP_PARAMETERIZED_PROXY engineering placeholder for development and tests only - "
                "NOT a measurement, replace with the measured value")


def _unmeasured(source: str) -> Param:
    return Param(None, AssetStatus.REQUIRES_MEASUREMENT, source)


def _temp(value: Any, what: str) -> Param:
    return Param(np.asarray(value, dtype=float), AssetStatus.TEMP_PARAMETERIZED_PROXY,
                 _TEMP_PREFIX + ": " + what)


@dataclass
class SafetyContext:
    """Optional per-step safety inputs (06_SAFETY.md S7 SafetyContext, T8 subset).

    estop / timeouts are per-env masks of shape (N,) (a scalar is broadcast).  A True entry
    means that environment is e-stopped / its communication watchdog expired.
    """
    now: Optional[float] = None
    estop: Any = None
    timeouts: Any = None
    racket_contact_pose: Any = None
    joint_position: Any = None
    joint_velocity: Any = None


@dataclass
class SafetyLimits:
    """Configurable safety limits; every entry records how it was obtained (S4/S12)."""

    joint_position_min: Param = field(default_factory=lambda: _unmeasured(_SRC_JOINT_POS))
    joint_position_max: Param = field(default_factory=lambda: _unmeasured(_SRC_JOINT_POS))
    joint_velocity_max: Param = field(default_factory=lambda: _unmeasured(_SRC_JOINT_VEL))
    base_twist_max: Param = field(default_factory=lambda: _unmeasured(_SRC_BASE_TWIST))
    base_translation_speed_max: Param = field(default_factory=lambda: _unmeasured(_SRC_BASE_SPEED))
    racket_workspace_min: Param = field(default_factory=lambda: _unmeasured(_SRC_WORKSPACE))
    racket_workspace_max: Param = field(default_factory=lambda: _unmeasured(_SRC_WORKSPACE))
    control_dt: Param = field(default_factory=lambda: _unmeasured(_SRC_DT))
    command_timeout_s: Param = field(default_factory=lambda: _unmeasured(_SRC_TIMEOUT))
    clock_tolerance_s: Param = field(default_factory=lambda: _unmeasured(_SRC_CLOCK))

    # -- introspective helpers -----------------------------------------------------
    def as_dict(self) -> Dict[str, Param]:
        return {name: getattr(self, name) for name in self.__dataclass_fields__}

    def unresolved(self) -> Tuple[str, ...]:
        """Names of limits not backed by a real measurement (common/status.py UNRESOLVED_STATUSES).

        An explicitly TEMP-marked proxy counts as unresolved: it is usable for development but
        it is still not the measured value (S4/S12).
        """
        return tuple(name for name, param in self.as_dict().items() if not param.is_resolved)

    def missing_values(self) -> Tuple[str, ...]:
        """Names of limits with no value at all: the shield cannot run with these."""
        return tuple(name for name, param in self.as_dict().items() if param.value is None)

    def values(self) -> Dict[str, Any]:
        missing = self.missing_values()
        if missing:
            raise BrainBoundaryError(
                "SafetyLimits has no value for: " + ", ".join(missing) +
                "; measure them or mark an explicit TEMP proxy (S4/S12)")
        out: Dict[str, Any] = {}
        for name, param in self.as_dict().items():
            arr = np.asarray(param.value, dtype=float).ravel()
            out[name] = float(arr[0]) if arr.size == 1 else arr
        return out

    @classmethod
    def temp_proxy(cls) -> "SafetyLimits":
        """Explicitly TEMP placeholders so the shield can run in development / tests."""
        return cls(
            joint_position_min=_temp(np.full(JOINT_COUNT, -2.6),
                                     "PiPER-like +-2.6 rad joint range; per-joint values unknown"),
            joint_position_max=_temp(np.full(JOINT_COUNT, 2.6),
                                     "PiPER-like +-2.6 rad joint range; per-joint values unknown"),
            joint_velocity_max=_temp(np.full(JOINT_COUNT, 2.0), "PiPER-like 2.0 rad/s joint speed"),
            base_twist_max=_temp(np.array([0.60, 0.60, 1.00]),
                                 "Morph One-like vx/vy 0.6 m/s, wz 1.0 rad/s"),
            base_translation_speed_max=_temp(0.60, "Morph One-like 0.6 m/s combined translation"),
            racket_workspace_min=_temp(np.array([-2.60, -1.50, 0.15]),
                                       "box around the nominal robot station, court frame"),
            racket_workspace_max=_temp(np.array([-0.40, 1.50, 2.20]),
                                       "box around the nominal robot station, court frame"),
            control_dt=_temp(0.02, "50 Hz placeholder control period"),
            command_timeout_s=_temp(0.20, "200 ms placeholder communication budget"),
            clock_tolerance_s=_temp(0.05, "50 ms placeholder clock tolerance"),
        )


class SafetyShield(SafetyModule):
    """WholeBodyTarget -> SafeCommand.  The only producer of executable commands."""

    name = "safety_shield"
    is_implemented = True
    output_type = SafeCommand

    def __init__(self, num_envs: Optional[int] = None, limits: Optional[SafetyLimits] = None,
                 forward_kinematics: Optional[Callable[[np.ndarray], np.ndarray]] = None) -> None:
        if limits is None:
            warnings.warn(
                "[SafetyShield] no limits supplied: using SafetyLimits.temp_proxy() "
                "(TEMP_PARAMETERIZED_PROXY development placeholders). Measured PiPER / Morph One "
                "limits are REQUIRES_MEASUREMENT (docs/simulation/BADMINTON_ROBOT.md S4/S12).",
                RuntimeWarning, stacklevel=2)
            limits = SafetyLimits.temp_proxy()
        missing = limits.missing_values()
        if missing:
            raise BrainBoundaryError(
                "SafetyShield refuses to run with unmeasured limits (value=None): "
                + ", ".join(missing)
                + "; supply measured values or an explicitly TEMP-marked proxy")
        unresolved = limits.unresolved()
        if unresolved:
            warnings.warn(
                "[SafetyShield] running with TEMP_PARAMETERIZED_PROXY (not measured) limits: "
                + ", ".join(unresolved)
                + "; these must be replaced by measured values (docs/simulation/BADMINTON_ROBOT.md S4/S12)",
                RuntimeWarning, stacklevel=2)
        self.limits = limits
        self._v = limits.values()
        self._validate_limits()
        self.forward_kinematics = forward_kinematics
        self.num_envs = int(num_envs) if num_envs is not None else None
        self.last_actions: Tuple[str, ...] = ()
        self._init_state(self.num_envs)

    # -- construction helpers ------------------------------------------------------
    def _validate_limits(self) -> None:
        shapes = {"joint_position_min": JOINT_COUNT, "joint_position_max": JOINT_COUNT,
                  "joint_velocity_max": JOINT_COUNT, "base_twist_max": BASE_TWIST_DIM,
                  "racket_workspace_min": 3, "racket_workspace_max": 3}
        for name, expected in shapes.items():
            arr = np.asarray(self._v[name], dtype=float).ravel()
            if arr.size != expected:
                raise BrainBoundaryError(
                    "SafetyLimits." + name + " must have " + str(expected) + " entries")
            if not np.all(np.isfinite(arr)):
                raise BrainBoundaryError("SafetyLimits." + name + " contains NaN/Inf")
            self._v[name] = arr
        if np.any(self._v["joint_position_min"] >= self._v["joint_position_max"]):
            raise BrainBoundaryError("joint_position_min must be < joint_position_max")
        if np.any(self._v["racket_workspace_min"] >= self._v["racket_workspace_max"]):
            raise BrainBoundaryError("racket_workspace_min must be < racket_workspace_max")
        if np.any(self._v["joint_velocity_max"] <= 0.0) or np.any(self._v["base_twist_max"] <= 0.0):
            raise BrainBoundaryError("velocity limits must be > 0")
        for name in ("control_dt", "command_timeout_s", "clock_tolerance_s",
                     "base_translation_speed_max"):
            self._v[name] = float(np.asarray(self._v[name], dtype=float).ravel()[0])
        if self._v["base_translation_speed_max"] <= 0.0:
            raise BrainBoundaryError("base_translation_speed_max must be > 0")
        if self._v["control_dt"] <= 0.0:
            raise BrainBoundaryError("control_dt must be > 0")
        if self._v["command_timeout_s"] < 0.0 or self._v["clock_tolerance_s"] < 0.0:
            raise BrainBoundaryError("command_timeout_s / clock_tolerance_s must be >= 0")

    def _init_state(self, num_envs: Optional[int]) -> None:
        n = num_envs or 0
        self._applied_q = np.zeros((n, JOINT_COUNT))
        self._has_reference = np.zeros(n, dtype=bool)
        self._estop_latched = np.zeros(n, dtype=bool)
        self._velocity_ref_warned = np.zeros(n, dtype=bool)

    def _ensure_envs(self, n: int) -> None:
        if self.num_envs is None:
            self.num_envs = n
            self._init_state(n)
        elif n != self.num_envs:
            raise BrainBoundaryError(
                "SafetyShield was built for " + str(self.num_envs) + " envs but received " + str(n))

    # -- reset ---------------------------------------------------------------------
    def reset(self, env_ids: Optional[Union[int, Sequence[int]]] = None) -> None:
        """Clear latch / command history for the selected envs (None = every env)."""
        if self.num_envs is None or self.num_envs == 0:
            return
        if env_ids is None:
            idx = np.arange(self.num_envs)
        elif isinstance(env_ids, (int, np.integer)):
            idx = np.array([int(env_ids)])
        else:
            idx = np.asarray(env_ids, dtype=int).ravel()
        if idx.size == 0:
            return
        if int(idx.min()) < 0 or int(idx.max()) >= self.num_envs:
            raise BrainBoundaryError("reset env_ids out of range [0, " + str(self.num_envs) + ")")
        self._estop_latched[idx] = False
        self._has_reference[idx] = False
        self._velocity_ref_warned[idx] = False
        self._applied_q[idx] = 0.0

    # -- helpers -------------------------------------------------------------------
    def _as_mask(self, name: str, value: Any, n: int) -> np.ndarray:
        if value is None:
            return np.zeros(n, dtype=bool)
        arr = np.asarray(value)
        if arr.ndim == 0:
            return np.full(n, bool(arr))
        if arr.shape != (n,):
            raise BrainBoundaryError(name + " must be (N,) = (" + str(n) + ",), got " + str(arr.shape))
        return arr.astype(bool)

    def _env_matrix(self, name: str, value: Any, n: int, dim: int) -> Optional[np.ndarray]:
        if value is None:
            return None
        arr = np.asarray(value, dtype=float)
        if arr.shape != (n, dim):
            raise BrainBoundaryError(name + " must be (N, " + str(dim) + ") = (" + str(n) + ", "
                                     + str(dim) + "), got " + str(arr.shape))
        return arr

    def _outside_workspace(self, xyz: np.ndarray) -> np.ndarray:
        lo = self._v["racket_workspace_min"]
        hi = self._v["racket_workspace_max"]
        return ((xyz < lo - EPS) | (xyz > hi + EPS)).any(axis=1)

    @staticmethod
    def _rank(action: str, candidate: str) -> str:
        return candidate if _ACTION_RANK[candidate] > _ACTION_RANK[action] else action

    # -- main entry point ----------------------------------------------------------
    def process(self, target: WholeBodyTarget, state: Any = None,
                context: Optional[SafetyContext] = None,
                now: Optional[float] = None) -> SafeCommand:
        if not isinstance(target, WholeBodyTarget):
            raise BrainBoundaryError(
                "SafetyShield.process expects a WholeBodyTarget, got " + type(target).__name__)
        try:
            twist = np.asarray(target.base_twist, dtype=float)
            joints = np.asarray(target.joint_position_target, dtype=float)
        except (TypeError, ValueError) as exc:
            raise BrainBoundaryError("WholeBodyTarget arrays must be numeric: " + str(exc)) from exc
        if twist.ndim != 2 or twist.shape[1] != BASE_TWIST_DIM:
            raise BrainBoundaryError(
                "WholeBodyTarget.base_twist must be (N, " + str(BASE_TWIST_DIM) + "), got "
                + str(twist.shape))
        if joints.ndim != 2 or joints.shape[1] != JOINT_COUNT:
            raise BrainBoundaryError(
                "WholeBodyTarget.joint_position_target must be (N, " + str(JOINT_COUNT)
                + "), got " + str(joints.shape))
        n = int(twist.shape[0])
        if joints.shape[0] != n:
            raise BrainBoundaryError("WholeBodyTarget arrays disagree on the batch size")
        self._ensure_envs(n)

        ctx = context if context is not None else SafetyContext()
        estop = self._as_mask("SafetyContext.estop", ctx.estop, n)
        watchdog = self._as_mask("SafetyContext.timeouts", ctx.timeouts, n)
        now_value = now if now is not None else ctx.now
        if state is not None:
            measured_q = self._env_matrix("state.joint_pos", getattr(state, "joint_pos", None),
                                          n, JOINT_COUNT)
            racket_pose = self._env_matrix("state.racket_contact_pose",
                                           getattr(state, "racket_contact_pose", None),
                                           n, RACKET_POSE_DIM)
        else:
            measured_q = self._env_matrix("SafetyContext.joint_position", ctx.joint_position,
                                          n, JOINT_COUNT)
            racket_pose = self._env_matrix("SafetyContext.racket_contact_pose",
                                           ctx.racket_contact_pose, n, RACKET_POSE_DIM)

        codes = [set() for _ in range(n)]
        actions = [ACTION_PASS] * n
        halted = np.zeros(n, dtype=bool)

        def halt(env_idx: int, code: str, action: str) -> None:
            halted[env_idx] = True
            codes[env_idx].add(code)
            actions[env_idx] = self._rank(actions[env_idx], action)

        # 1. e-stop input + latch (S38-S40)
        self._estop_latched |= estop
        for env_idx in np.flatnonzero(self._estop_latched):
            halt(int(env_idx), "estop_latched", ACTION_ESTOP_REQUEST)

        # 2. communication watchdog (S26/S31/S35-S37)
        for env_idx in np.flatnonzero(watchdog):
            halt(int(env_idx), "communication_timeout", ACTION_CONTROLLED_STOP)

        # 3. clock checks (S25-S27)
        if now_value is not None:
            t_now = float(now_value)
            if not np.isfinite(t_now):
                raise BrainBoundaryError("SafetyShield.process(now=...) must be finite")
            age = t_now - float(target.timestamp)
            if age > self._v["command_timeout_s"]:
                for env_idx in range(n):
                    halt(env_idx, "communication_timeout", ACTION_CONTROLLED_STOP)
            elif age < -self._v["clock_tolerance_s"]:
                for env_idx in range(n):
                    halt(env_idx, "command_timestamp_future", ACTION_PROTECTIVE_STOP)

        # 4. hard validity: NaN / Inf (S24)
        invalid = ~np.isfinite(twist).all(axis=1) | ~np.isfinite(joints).all(axis=1)
        if measured_q is not None:
            invalid |= ~np.isfinite(measured_q).all(axis=1)
        if racket_pose is not None:
            invalid |= ~np.isfinite(racket_pose).all(axis=1)
        for env_idx in np.flatnonzero(invalid):
            halt(int(env_idx), "invalid_non_finite", ACTION_PROTECTIVE_STOP)

        lo = self._v["joint_position_min"]
        hi = self._v["joint_position_max"]
        dq = self._v["joint_velocity_max"]
        dt = self._v["control_dt"]
        vmax = self._v["base_twist_max"]
        v_speed_max = self._v["base_translation_speed_max"]

        # 5. velocity reference: measured q, else the last safe command
        reference = np.zeros((n, JOINT_COUNT))
        if measured_q is not None:
            reference = measured_q.copy()
            have_reference = np.ones(n, dtype=bool)
        else:
            have_reference = self._has_reference.copy()
            reference[have_reference] = self._applied_q[have_reference]
        missing_ref = ~have_reference & ~halted
        if missing_ref.any():
            reference[missing_ref] = np.clip(joints[missing_ref], lo, hi)
            for env_idx in np.flatnonzero(missing_ref):
                if not self._velocity_ref_warned[env_idx]:
                    self._velocity_ref_warned[env_idx] = True
                    codes[env_idx].add("no_velocity_reference")
                    actions[env_idx] = self._rank(actions[env_idx], ACTION_PROJECT)

        # 6. current state beyond a hard limit -> protective stop (S45)
        outside = ((reference < lo - EPS) | (reference > hi + EPS)).any(axis=1) & ~halted
        for env_idx in np.flatnonzero(outside):
            bad = np.flatnonzero((reference[env_idx] < lo - EPS) | (reference[env_idx] > hi + EPS))
            halted[env_idx] = True
            for joint in bad:
                codes[env_idx].add("measured_joint_beyond_limit[j=" + str(int(joint)) + "]")
            actions[env_idx] = self._rank(actions[env_idx], ACTION_PROTECTIVE_STOP)

        # 7. joint projection: one step of the velocity limit, then the position limit
        over_velocity = np.abs(joints - reference) > dq * dt + EPS
        pre_position = np.clip(joints, reference - dq * dt, reference + dq * dt)
        q_out = np.clip(pre_position, lo, hi)
        over_hi = pre_position > hi + EPS
        over_lo = pre_position < lo - EPS

        # 8. base twist: direction-preserving scaling (S65-S69)
        abs_twist = np.abs(twist)
        scale = np.ones(n)
        for axis in range(BASE_TWIST_DIM):
            allowed = np.where(abs_twist[:, axis] > vmax[axis],
                               vmax[axis] / np.maximum(abs_twist[:, axis], EPS), 1.0)
            scale = np.minimum(scale, allowed)
        speed = np.hypot(twist[:, 0], twist[:, 1])
        scale = np.minimum(scale, np.where(speed > v_speed_max,
                                           v_speed_max / np.maximum(speed, EPS), 1.0))
        over_twist = scale < 1.0 - EPS
        twist_out = twist * scale[:, None]

        for env_idx in range(n):
            if halted[env_idx]:
                continue
            for joint in np.flatnonzero(over_velocity[env_idx]):
                codes[env_idx].add("joint_velocity_max[j=" + str(int(joint)) + "]")
            for joint in np.flatnonzero(over_hi[env_idx]):
                codes[env_idx].add("joint_position_max[j=" + str(int(joint)) + "]")
            for joint in np.flatnonzero(over_lo[env_idx]):
                codes[env_idx].add("joint_position_min[j=" + str(int(joint)) + "]")
            if over_twist[env_idx]:
                codes[env_idx].add("base_twist_max")

        # 9. racket contact workspace box (S77) -> reject (HOLD), never silently clamp
        reject = np.zeros(n, dtype=bool)
        if racket_pose is not None:
            reject |= self._outside_workspace(racket_pose[:, :3])
        if self.forward_kinematics is not None and (~halted).any():
            predicted = np.asarray(self.forward_kinematics(q_out), dtype=float)
            if predicted.ndim != 2 or predicted.shape[0] != n or predicted.shape[1] < 3:
                raise BrainBoundaryError(
                    "forward_kinematics must return (N, >=3) racket positions, got "
                    + str(predicted.shape))
            reject |= (self._outside_workspace(predicted[:, :3])
                       | ~np.isfinite(predicted[:, :3]).all(axis=1))
        reject &= ~halted
        for env_idx in np.flatnonzero(reject):
            codes[env_idx].add("racket_workspace")
            actions[env_idx] = self._rank(actions[env_idx], ACTION_HOLD)

        # 10. assemble: e-stop / timeout -> zero command; workspace -> hold the last safe q
        out_twist = np.where(halted[:, None], 0.0, twist_out)
        out_q = np.where(halted[:, None], 0.0, q_out)
        out_twist = np.where(reject[:, None], 0.0, out_twist)
        out_q = np.where(reject[:, None], reference, out_q)

        for env_idx in range(n):
            if not halted[env_idx] and codes[env_idx]:
                actions[env_idx] = self._rank(actions[env_idx], ACTION_PROJECT)

        self._applied_q = out_q.copy()
        self._has_reference[:] = True
        self.last_actions = tuple(actions)

        violations = tuple(sorted("env" + str(env) + ":" + code
                                  for env, code_set in enumerate(codes) for code in code_set))
        timestamp = float(now_value) if now_value is not None else float(target.timestamp)
        return SafeCommand(base_twist=out_twist, joint_position_target=out_q,
                           limited=bool(violations), violations=violations, timestamp=timestamp)


__all__ = ["SafetyContext", "SafetyLimits", "SafetyShield", "JOINT_COUNT", "BASE_TWIST_DIM",
           "ACTION_PASS", "ACTION_PROJECT", "ACTION_HOLD", "ACTION_CONTROLLED_STOP",
           "ACTION_PROTECTIVE_STOP", "ACTION_ESTOP_REQUEST"]
