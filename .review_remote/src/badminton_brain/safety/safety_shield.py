# -*- coding: utf-8 -*-
"""Safety Shield (T8): WholeBodyTarget -> SafeCommand.

Spec: docs/architecture/06_SAFETY.md (Safety is an independent, deterministic, testable
barrier that upstream layers cannot bypass) and docs/simulation/BADMINTON_ROBOT.md
(S4 authenticity levels, S12 TEMP policy).

Implemented (plan 2026-09-13-brain-modules.md, T8 row):
  * 6-axis joint position limits                          (06_SAFETY.md S42-S45)
  * joint velocity limits (per-step command projection)   (S46, S51)
  * base twist limits, direction-preserving scaling, with the two documented semantics
    - base_twist_axis_max: |vx| <= vx_max, |vy| <= vy_max, |wz| <= wz_max   (S65)
    - base_translation_speed_max: sqrt(vx^2 + vy^2) <= v_xy_max            (S66)
  * racket_contact safe workspace box, FK-verified HOLD fallback           (S77)
  * per-env e-stop input with latch and explicit reset                     (S38-S40, S19)
  * communication timeout / future-dated command                          (S25-S27, S31, S35-S37)
  * NaN/Inf -> zero command instead of an exception                        (S24)

Context policy (coordinator ruling, HIGH):
  The application layer (T11 full_brain / the real-time loop) must push the per-step
  SafetyContext through set_context() BEFORE calling process().  process() also accepts an
  inline context= for tests, which overrides the pushed one for that call.  SafetyContext
  carries the per-env e-stop mask, an external watchdog mask, the clock (now) and the
  measured state (joint_position / racket_contact_pose).
  An env for which no context information exists is NEVER silently treated as healthy:
    * require_context=True (default) -> violation code "no_context" for that env, action
      PROJECT.  The command still moves (absence of information is not a fault), but the
      caller sees the gap in violations / context_snapshot().
    * require_context=False -> the gap is only reported by context_snapshot()
      ("without_context_envs"), the caller explicitly opted out of the marking.
  context_snapshot() returns the source ("pushed" / "argument" / "none"), the epoch, the
  effective now, the active e-stop / watchdog / measured envs and the last verdict.

Enforcement priority (06_SAFETY.md S19, adapted to the T8 subset):
  hard stops (e-stop, watchdog/timeout, future clock, NaN/Inf, measured joint beyond a hard
  limit) > joint position limits > racket workspace box > base twist limits > joint velocity
  step projection.  A workspace-verified HOLD therefore takes precedence over the per-step
  joint velocity projection: freezing (or retreating to a verified feasible point) beats the
  soft velocity step, while the hard joint position limits are never violated.

NOT IMPLEMENTED (explicit; out of the T8 scope, no measured data available):
  * joint acceleration limit (S47) and jerk limit (S48)
  * torque / effort limit (S49): PiPER has no trustworthy commanded/measured effort yet
  * velocity-aware dynamic joint envelope + CBF (S51-S64): needs measured dq, a_brake, T_delay
  * predictive collision / swept-volume checks (S93-S110)
  * base geofence and footprint inflation (S70-S76)
  * racket-base / camera-mast self collision (S83-S92)
  * human separation and TTC (S103-S105)
  * safety projection QP (S115+): this revision uses documented, per-limit projections
  * software ESTOP_REQUEST is only a request to the platform emergency path (S41)

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

# Violation codes (documented vocabulary, one string per env in SafeCommand.violations).
CODE_ESTOP = "estop_latched"
CODE_TIMEOUT = "communication_timeout"
CODE_FUTURE = "command_timestamp_future"
CODE_NON_FINITE = "invalid_non_finite"
CODE_NO_CONTEXT = "no_context"
CODE_NO_REFERENCE = "no_velocity_reference"
CODE_MEASURED_JOINT = "measured_joint_beyond_limit"
CODE_JOINT_VEL = "joint_velocity_max"
CODE_JOINT_MIN = "joint_position_min"
CODE_JOINT_MAX = "joint_position_max"
CODE_AXIS_TWIST = "base_twist_axis_max"
CODE_SPEED_TWIST = "base_translation_speed_max"
CODE_WORKSPACE_STATE = "racket_workspace_state"
CODE_WORKSPACE_COMMAND = "racket_workspace_command"
CODE_WORKSPACE_REST = "racket_workspace_rest"
CODE_WORKSPACE_UNREACHABLE = "racket_workspace_unreachable"

EPS = 1e-12

# --- provenance strings for the unmeasured limits ---------------------------------
_SRC_JOINT_POS = ("PiPER joint position limits: must come from the official PiPER URDF / driver "
                  "configuration for this unit; no PiPER joint limit has been measured in this repo")
_SRC_JOINT_VEL = ("PiPER joint velocity limits: must come from the PiPER specification / driver "
                  "limits; no PiPER joint speed limit has been measured in this repo")
_SRC_BASE_AXIS = ("Morph One base twist axis limits (vx, vy, wz) (06_SAFETY.md S65): must come from "
                  "the Morph One chassis specification or a measured speed test; not measured")
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
    """Per-step safety inputs pushed by the application layer (06_SAFETY.md S7, T8 subset).

    estop / timeouts are per-env masks of shape (N,) (a scalar is broadcast).  A True entry
    means that environment is e-stopped / its communication watchdog expired.  now is the
    clock used for the command-age checks, and joint_position / racket_contact_pose are the
    measured state (used as the velocity reference and for the racket workspace check).
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
    base_twist_axis_max: Param = field(default_factory=lambda: _unmeasured(_SRC_BASE_AXIS))
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
            base_twist_axis_max=_temp(np.array([0.60, 0.60, 1.00]),
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
                 forward_kinematics: Optional[Callable[[np.ndarray], np.ndarray]] = None,
                 require_context: bool = True) -> None:
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
        self.require_context = bool(require_context)
        self.num_envs = int(num_envs) if num_envs is not None else None
        self.last_actions: Tuple[str, ...] = ()
        self._pushed_context: Optional[SafetyContext] = None
        self._pushed_valid: Optional[np.ndarray] = None
        self._context_epoch = 0
        self._last_snapshot: Dict[str, Any] = {
            "source": "none", "epoch": 0, "now": None, "estop_envs": (), "timeout_envs": (),
            "measured_joint_envs": (), "measured_racket_envs": (), "without_context_envs": (),
            "limited": False, "violations": (), "actions": (),
        }
        self._init_state(self.num_envs)

    # -- construction helpers ------------------------------------------------------
    def _validate_limits(self) -> None:
        shapes = {"joint_position_min": JOINT_COUNT, "joint_position_max": JOINT_COUNT,
                  "joint_velocity_max": JOINT_COUNT, "base_twist_axis_max": BASE_TWIST_DIM,
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
        if np.any(self._v["joint_velocity_max"] <= 0.0) or np.any(self._v["base_twist_axis_max"] <= 0.0):
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
        # Review finding 1: every stop / fallback pose must be feasible by construction.
        self._safe_rest_q = np.clip(np.zeros(JOINT_COUNT), self._v["joint_position_min"],
                                    self._v["joint_position_max"])
        # Review finding 3: without any measurement the reference is the limit box centre.
        self._limit_centre = 0.5 * (self._v["joint_position_min"] + self._v["joint_position_max"])

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
            self._pushed_valid = (np.ones(n, dtype=bool) if self._pushed_context is not None
                                  else None)
        elif n != self.num_envs:
            raise BrainBoundaryError(
                "SafetyShield was built for " + str(self.num_envs) + " envs but received " + str(n))

    # -- context push API (coordinator ruling, HIGH) -------------------------------
    def set_context(self, context: SafetyContext) -> None:
        """Push the per-step SafetyContext (call this before process() every control step)."""
        if not isinstance(context, SafetyContext):
            raise BrainBoundaryError(
                "set_context expects a SafetyContext, got " + type(context).__name__)
        self._pushed_context = SafetyContext(
            now=context.now,
            estop=None if context.estop is None else np.array(context.estop, dtype=bool, copy=True),
            timeouts=None if context.timeouts is None else np.array(context.timeouts, dtype=bool, copy=True),
            racket_contact_pose=(None if context.racket_contact_pose is None
                                 else np.array(context.racket_contact_pose, dtype=float, copy=True)),
            joint_position=(None if context.joint_position is None
                            else np.array(context.joint_position, dtype=float, copy=True)),
            joint_velocity=(None if context.joint_velocity is None
                            else np.array(context.joint_velocity, dtype=float, copy=True)),
        )
        self._context_epoch += 1
        self._pushed_valid = (np.ones(self.num_envs, dtype=bool)
                              if self.num_envs else None)

    def context_snapshot(self) -> Dict[str, Any]:
        """Diagnostics of the last process() call (source, masks, verdict)."""
        return dict(self._last_snapshot)

    # -- reset ---------------------------------------------------------------------
    def reset(self, env_ids: Optional[Union[int, Sequence[int]]] = None) -> None:
        """Clear latch / context / command history for the selected envs (None = every env)."""
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
        if self._pushed_valid is not None:
            self._pushed_valid[idx] = False        # the pushed context no longer covers them

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

    def _fk_pose(self, q: np.ndarray) -> np.ndarray:
        pose = np.asarray(self.forward_kinematics(q), dtype=float)
        if pose.ndim != 2 or pose.shape[0] != q.shape[0] or pose.shape[1] < 3:
            raise BrainBoundaryError(
                "forward_kinematics must return (N, >=3) racket positions, got " + str(pose.shape))
        return pose

    def _pose_inside(self, q: np.ndarray) -> np.ndarray:
        if self.forward_kinematics is None:
            return np.ones(q.shape[0], dtype=bool)
        pose = self._fk_pose(q)[:, :3]
        return ~(self._outside_workspace(pose) | ~np.isfinite(pose).all(axis=1))

    def _apply_verified_hold(self, q_final: np.ndarray, twist_final: np.ndarray,
                             offenders: np.ndarray, reference: np.ndarray,
                             primary: Dict[int, str], codes, actions) -> None:
        """Replace the command of the offending envs with an FK-verified HOLD point.

        Candidate order: freeze at the current reference (zero velocity step), then the last
        emitted safe command, then the joint-limit rest pose.  If no candidate yields a racket
        pose inside the box, the env is reported loudly as racket_workspace_unreachable.
        """
        pending = offenders.copy()
        n = q_final.shape[0]
        last_safe = np.where(self._has_reference[:, None], self._applied_q, self._safe_rest_q)
        rest = np.tile(self._safe_rest_q, (n, 1))
        for matrix, is_rest in ((reference, False), (last_safe, False), (rest, True)):
            if not pending.any():
                break
            if self.forward_kinematics is None:
                take = pending.copy()
            else:
                take = pending & self._pose_inside(matrix)
            for env_idx in np.flatnonzero(take):
                q_final[env_idx] = matrix[env_idx]
                twist_final[env_idx] = 0.0
                codes[env_idx].add(primary.get(int(env_idx), CODE_WORKSPACE_COMMAND))
                if is_rest:
                    codes[env_idx].add(CODE_WORKSPACE_REST)
                actions[env_idx] = self._rank(actions[env_idx], ACTION_HOLD)
            pending &= ~take
            if self.forward_kinematics is None:
                break
        for env_idx in np.flatnonzero(pending):
            q_final[env_idx] = reference[env_idx]
            twist_final[env_idx] = 0.0
            codes[env_idx].add(CODE_WORKSPACE_UNREACHABLE)
            actions[env_idx] = self._rank(actions[env_idx], ACTION_PROTECTIVE_STOP)

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
            raise BrainBoundaryError("WholeBodyTarget.base_twist must be (N, " + str(BASE_TWIST_DIM)
                                     + "), got " + str(twist.shape))
        if joints.ndim != 2 or joints.shape[1] != JOINT_COUNT:
            raise BrainBoundaryError("WholeBodyTarget.joint_position_target must be (N, "
                                     + str(JOINT_COUNT) + "), got " + str(joints.shape))
        n = int(twist.shape[0])
        if joints.shape[0] != n:
            raise BrainBoundaryError("WholeBodyTarget arrays disagree on the batch size")
        self._ensure_envs(n)

        # 0. context resolution: inline argument > pushed context (per env) > none
        if context is not None:
            src, valid, source_name = context, np.ones(n, dtype=bool), "argument"
        elif self._pushed_context is not None:
            src, source_name = self._pushed_context, "pushed"
            valid = self._pushed_valid if self._pushed_valid is not None else np.ones(n, dtype=bool)
        else:
            src, valid, source_name = None, np.zeros(n, dtype=bool), "none"
        has_context = valid.copy()

        estop = np.zeros(n, dtype=bool)
        watchdog = np.zeros(n, dtype=bool)
        if src is not None and src.estop is not None:
            estop = self._as_mask("context.estop", src.estop, n) & valid
        if src is not None and src.timeouts is not None:
            watchdog = self._as_mask("context.timeouts", src.timeouts, n) & valid
        effective_now: Optional[float] = None
        t_now = np.full(n, np.nan)
        if now is not None:
            effective_now = float(now)
        elif src is not None and src.now is not None:
            effective_now = float(src.now)
        if effective_now is not None:
            if not np.isfinite(effective_now):
                raise BrainBoundaryError("SafetyShield.process(now=...) must be finite")
            t_now[:] = effective_now if now is not None else np.where(valid, effective_now, np.nan)

        measured_q = np.zeros((n, JOINT_COUNT))
        measured_q_mask = np.zeros(n, dtype=bool)
        racket_pose = np.zeros((n, RACKET_POSE_DIM))
        racket_pose_mask = np.zeros(n, dtype=bool)
        if state is not None and getattr(state, "joint_pos", None) is not None:
            measured_q = self._env_matrix("state.joint_pos", state.joint_pos, n, JOINT_COUNT)
            measured_q_mask[:] = True
        elif src is not None and src.joint_position is not None:
            mat = self._env_matrix("context.joint_position", src.joint_position, n, JOINT_COUNT)
            measured_q[valid] = mat[valid]
            measured_q_mask = valid.copy()
        if state is not None and getattr(state, "racket_contact_pose", None) is not None:
            racket_pose = self._env_matrix("state.racket_contact_pose", state.racket_contact_pose,
                                           n, RACKET_POSE_DIM)
            racket_pose_mask[:] = True
        elif src is not None and src.racket_contact_pose is not None:
            mat = self._env_matrix("context.racket_contact_pose", src.racket_contact_pose,
                                   n, RACKET_POSE_DIM)
            racket_pose[valid] = mat[valid]
            racket_pose_mask = valid.copy()

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
            halt(int(env_idx), CODE_ESTOP, ACTION_ESTOP_REQUEST)

        # 2. communication watchdog (S26/S31/S35-S37)
        for env_idx in np.flatnonzero(watchdog):
            halt(int(env_idx), CODE_TIMEOUT, ACTION_CONTROLLED_STOP)

        # 3. clock checks (S25-S27), per env: only envs whose clock is known are judged
        age = t_now - float(target.timestamp)
        for env_idx in np.flatnonzero(np.isfinite(age) & (age > self._v["command_timeout_s"])):
            halt(int(env_idx), CODE_TIMEOUT, ACTION_CONTROLLED_STOP)
        for env_idx in np.flatnonzero(np.isfinite(age) & (age < -self._v["clock_tolerance_s"])):
            halt(int(env_idx), CODE_FUTURE, ACTION_PROTECTIVE_STOP)

        # 4. hard validity: NaN / Inf (S24)
        invalid = ~np.isfinite(twist).all(axis=1) | ~np.isfinite(joints).all(axis=1)
        invalid |= measured_q_mask & ~np.isfinite(measured_q).all(axis=1)
        invalid |= racket_pose_mask & ~np.isfinite(racket_pose).all(axis=1)
        for env_idx in np.flatnonzero(invalid):
            halt(int(env_idx), CODE_NON_FINITE, ACTION_PROTECTIVE_STOP)

        # 5. missing context must be visible, never silently healthy (coordinator ruling)
        for env_idx in np.flatnonzero(~has_context & ~halted):
            if self.require_context:
                codes[env_idx].add(CODE_NO_CONTEXT)
                actions[env_idx] = self._rank(actions[env_idx], ACTION_PROJECT)

        lo = self._v["joint_position_min"]
        hi = self._v["joint_position_max"]
        dq = self._v["joint_velocity_max"]
        dt = self._v["control_dt"]
        vmax_axis = self._v["base_twist_axis_max"]
        v_speed_max = self._v["base_translation_speed_max"]

        # 6. velocity reference: measured q, else the last safe command, else the limit box centre
        reference = np.zeros((n, JOINT_COUNT))
        have_ref = np.zeros(n, dtype=bool)
        if measured_q_mask.any():
            reference[measured_q_mask] = measured_q[measured_q_mask]
            have_ref |= measured_q_mask
        prev = self._has_reference & ~have_ref
        if prev.any():
            reference[prev] = self._applied_q[prev]
            have_ref |= prev
        missing_ref = ~have_ref & ~halted
        if missing_ref.any():
            reference[missing_ref] = self._limit_centre
            for env_idx in np.flatnonzero(missing_ref):
                if not self._velocity_ref_warned[env_idx]:
                    self._velocity_ref_warned[env_idx] = True
                    codes[env_idx].add(CODE_NO_REFERENCE)
                    actions[env_idx] = self._rank(actions[env_idx], ACTION_PROJECT)

        # 7. measured state beyond a hard limit -> protective stop (S45)
        outside = (measured_q_mask
                   & ((reference < lo - EPS) | (reference > hi + EPS)).any(axis=1)
                   & ~halted)
        for env_idx in np.flatnonzero(outside):
            bad = np.flatnonzero((reference[env_idx] < lo - EPS) | (reference[env_idx] > hi + EPS))
            halted[env_idx] = True
            for joint in bad:
                codes[env_idx].add(CODE_MEASURED_JOINT + "[j=" + str(int(joint)) + "]")
            actions[env_idx] = self._rank(actions[env_idx], ACTION_PROTECTIVE_STOP)

        # 8. joint projection: one step of the velocity limit, then the position limit
        over_velocity = np.abs(joints - reference) > dq * dt + EPS
        pre_position = np.clip(joints, reference - dq * dt, reference + dq * dt)
        q_projected = np.clip(pre_position, lo, hi)
        over_hi = pre_position > hi + EPS
        over_lo = pre_position < lo - EPS

        # 9. base twist: direction-preserving scaling with two documented limits (S65/S66)
        abs_twist = np.abs(twist)
        scale = np.ones(n)
        axis_over = np.zeros((n, BASE_TWIST_DIM), dtype=bool)
        for axis in range(BASE_TWIST_DIM):
            axis_over[:, axis] = abs_twist[:, axis] > vmax_axis[axis]
            allowed = np.where(axis_over[:, axis],
                               vmax_axis[axis] / np.maximum(abs_twist[:, axis], EPS), 1.0)
            scale = np.minimum(scale, allowed)
        speed = np.hypot(twist[:, 0], twist[:, 1])
        speed_over = speed > v_speed_max
        scale = np.minimum(scale, np.where(speed_over, v_speed_max / np.maximum(speed, EPS), 1.0))
        twist_projected = twist * scale[:, None]

        for env_idx in range(n):
            if halted[env_idx]:
                continue
            for joint in np.flatnonzero(over_velocity[env_idx]):
                codes[env_idx].add(CODE_JOINT_VEL + "[j=" + str(int(joint)) + "]")
            for joint in np.flatnonzero(over_hi[env_idx]):
                codes[env_idx].add(CODE_JOINT_MAX + "[j=" + str(int(joint)) + "]")
            for joint in np.flatnonzero(over_lo[env_idx]):
                codes[env_idx].add(CODE_JOINT_MIN + "[j=" + str(int(joint)) + "]")
            if axis_over[env_idx].any():
                codes[env_idx].add(CODE_AXIS_TWIST)
            if speed_over[env_idx]:
                codes[env_idx].add(CODE_SPEED_TWIST)

        # 10. assemble: hard stops -> zero twist + the feasible rest pose (review finding 1)
        q_final = np.where(halted[:, None], self._safe_rest_q, q_projected)
        twist_final = np.where(halted[:, None], 0.0, twist_projected)

        # 11. racket workspace: measured state and the commanded pose must both be inside
        offenders = np.zeros(n, dtype=bool)
        primary: Dict[int, str] = {}
        if racket_pose_mask.any():
            state_bad = (racket_pose_mask & ~halted
                         & self._outside_workspace(racket_pose[:, :3]))
            for env_idx in np.flatnonzero(state_bad):
                primary[int(env_idx)] = CODE_WORKSPACE_STATE
            offenders |= state_bad
        if self.forward_kinematics is not None:
            commanded_pose = self._fk_pose(q_final)[:, :3]
            command_bad = (~halted & ~offenders
                           & (self._outside_workspace(commanded_pose)
                              | ~np.isfinite(commanded_pose).all(axis=1)))
            for env_idx in np.flatnonzero(command_bad):
                primary[int(env_idx)] = CODE_WORKSPACE_COMMAND
            offenders |= command_bad
        if offenders.any():
            self._apply_verified_hold(q_final, twist_final, offenders, reference, primary,
                                      codes, actions)

        for env_idx in range(n):
            if not halted[env_idx] and codes[env_idx]:
                actions[env_idx] = self._rank(actions[env_idx], ACTION_PROJECT)

        self._applied_q = q_final.copy()
        self._has_reference[:] = True
        self.last_actions = tuple(actions)

        violations = tuple(sorted("env" + str(env) + ":" + code
                                  for env, code_set in enumerate(codes) for code in code_set))
        timestamp = float(effective_now) if effective_now is not None else float(target.timestamp)
        self._last_snapshot = {
            "source": source_name,
            "epoch": self._context_epoch,
            "now": effective_now,
            "estop_envs": tuple(int(e) for e in np.flatnonzero(estop)),
            "timeout_envs": tuple(int(e) for e in np.flatnonzero(watchdog)),
            "measured_joint_envs": tuple(int(e) for e in np.flatnonzero(measured_q_mask)),
            "measured_racket_envs": tuple(int(e) for e in np.flatnonzero(racket_pose_mask)),
            "without_context_envs": tuple(int(e) for e in np.flatnonzero(~has_context)),
            "limited": bool(violations),
            "violations": violations,
            "actions": tuple(actions),
        }
        return SafeCommand(base_twist=twist_final, joint_position_target=q_final,
                           limited=bool(violations), violations=violations, timestamp=timestamp)


__all__ = ["SafetyContext", "SafetyLimits", "SafetyShield", "JOINT_COUNT", "BASE_TWIST_DIM",
           "ACTION_PASS", "ACTION_PROJECT", "ACTION_HOLD", "ACTION_CONTROLLED_STOP",
           "ACTION_PROTECTIVE_STOP", "ACTION_ESTOP_REQUEST"]
