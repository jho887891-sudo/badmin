# -*- coding: utf-8 -*-
"""P8-04: two deterministic NON-RL baselines over the P8-01/P8-02 task interface.

Both baselines are pure functions of a VALIDATED P8-01 observation vector and return a P8-01
action mapping, so they run through exactly the same validation path a learned policy will have
to run through later.  Nothing here learns: no gradient, no dataset, no torch, no GPU.

  * ReachabilityFSMBaseline -- a finite-state policy driven by reachability/feasibility verdicts.
    It holds (fail-closed, zero action) whenever the verdict says the target cannot be served,
    and applies a saturated bang-bang approach otherwise.  The verdict source is
    LocalReachabilityFSMStub, which is DOCUMENTED AS A STUB: it merely relays the four
    Decision.feasibility flags carried by the observation.  The real P7 interaction FSM
    (branch p7/reachability-fsm, commit adaee81) is NOT an ancestor of this branch, so it is
    not importable here; p7_fsm_status() records that fact at runtime and a test fails loudly
    the day the module does become importable, so somebody writes the adapter instead of
    silently claiming P7 verdicts.
  * TrackingControllerBaseline -- a proportional controller on the racket-to-target error
    (base velocity for the planar part, one joint delta for the vertical part) with the P5-05
    safety envelope: every command is slew-limited to the declared acceleration limits and is
    then passed through SafetyShield by the evaluation harness, so it cannot leave the
    envelope even when the error is large.

Fair-comparison rules the harness enforces (see evaluation.py): same episode list, same seed
ledger, same observation and action validation path, same success/failure definition.
"""
from __future__ import annotations

import importlib
from dataclasses import dataclass
from typing import Any, Dict, Mapping, Optional, Protocol, Tuple

import numpy as np

from ..safety.safety_shield import SafetyShield
from .action_schema import DEFAULT_DT_S, DEFAULT_LIMITS, ActionLimits, ActionValue
from .observation_schema import FIELD_SLICES, ObservationValue, validate as validate_observation

BASELINE_VERSION = "p8-baseline-v0.1"
BASELINE_STATUS = "NON_RL_DETERMINISTIC_BASELINE_NO_LEARNING"
P7_FSM_MODULE = "badminton_brain.safety.interaction_fsm"
P7_FSM_EXPECTED_AVAILABLE = False  # true only after p7/reachability-fsm is merged into the base

#: The synthetic racket model used by the harness (documented, tunable, not a robot model).
SYNTHETIC_RACKET_MODEL = {
    "xy_offset_m": (0.30, 0.00),
    "z_base_m": 0.60,
    "z_amplitude_m": 0.40,
    "z_joint_index": 2,
}

FSM_TRACK = "TRACK"
FSM_HOLD_UNREACHABLE = "HOLD_UNREACHABLE"
FSM_HOLD_OUT_OF_WORKSPACE = "HOLD_OUT_OF_WORKSPACE"
FSM_HOLD_TIMING = "HOLD_TIMING"
FSM_HOLD_INVALID_PREDICTION = "HOLD_INVALID_PREDICTION"
FSM_STATES = (FSM_TRACK, FSM_HOLD_UNREACHABLE, FSM_HOLD_OUT_OF_WORKSPACE, FSM_HOLD_TIMING,
              FSM_HOLD_INVALID_PREDICTION)


class BaselineError(ValueError):
    """Base class for baseline refusals."""


class ObservationInterfaceError(BaselineError):
    """The policy was handed something that is not a validated P8-01 observation."""


def p7_fsm_status() -> Dict[str, Any]:
    """Record whether the real P7 interaction FSM can be imported on this branch."""
    try:
        importlib.import_module(P7_FSM_MODULE)
    except Exception as exc:  # ImportError, but a broken import must also be visible
        return {"module": P7_FSM_MODULE, "importable": False,
                "reason": "{0}: {1}".format(type(exc).__name__, exc),
                "verdict_source": "LocalReachabilityFSMStub"}
    return {"module": P7_FSM_MODULE, "importable": True,
            "reason": "module imported; the P7 verdict adapter has not been written yet",
            "verdict_source": "LocalReachabilityFSMStub"}


@dataclass(frozen=True)
class ReachabilityVerdict:
    """What the FSM says about the current target."""
    state: str
    reason: str
    may_act: bool

    def __post_init__(self) -> None:
        if self.state not in FSM_STATES:
            raise BaselineError("unknown FSM state {0!r}".format(self.state))


class LocalReachabilityFSMStub:
    """DOCUMENTED STUB -- it is a flag relay, NOT a reachability computation.

    It maps the observation's feasibility_flags = [reachable, in_workspace, timing_ok,
    prediction_valid] onto a fail-closed state machine.  It makes no geometric, kinematic or
    dynamic reachability statement of its own, and the report must never describe its numbers as
    P7 reachability performance.
    """
    name = "local-reachability-fsm-stub"

    def verdict(self, flags: Any) -> ReachabilityVerdict:
        reachable, in_workspace, timing_ok, prediction_valid = [bool(f) for f in flags]
        if not prediction_valid:
            return ReachabilityVerdict(FSM_HOLD_INVALID_PREDICTION,
                                       "prediction_valid=0: no usable intercept", False)
        if not reachable:
            return ReachabilityVerdict(FSM_HOLD_UNREACHABLE,
                                       "reachable=0: target judged unreachable", False)
        if not in_workspace:
            return ReachabilityVerdict(FSM_HOLD_OUT_OF_WORKSPACE,
                                       "in_workspace=0: target outside the racket workspace", False)
        if not timing_ok:
            return ReachabilityVerdict(FSM_HOLD_TIMING, "timing_ok=0: contact window missed", False)
        return ReachabilityVerdict(FSM_TRACK, "all feasibility flags set", True)


@dataclass(frozen=True)
class ObservationView:
    """A strictly validated P8-01 observation plus named access to its channels."""
    vector: np.ndarray

    def __post_init__(self) -> None:
        arr = np.asarray(self.vector)
        if arr.ndim != 1:
            raise ObservationInterfaceError(
                "a baseline consumes ONE observation vector, got shape {0}".format(arr.shape))
        try:
            report = validate_observation(arr)  # strict P8-01 path
        except Exception as exc:
            raise ObservationInterfaceError(
                "not a validated P8-01 observation: {0}: {1}"
                .format(type(exc).__name__, exc)) from exc
        if report.dim != arr.shape[0]:
            raise ObservationInterfaceError("observation validation dim mismatch")

    def field(self, name: str) -> np.ndarray:
        if name not in FIELD_SLICES:
            raise ObservationInterfaceError("unknown observation field {0!r}".format(name))
        return np.array(self.vector[FIELD_SLICES[name]], dtype=float)

    @property
    def racket_position(self) -> np.ndarray:
        return self.field("racket_position")

    @property
    def joint_position(self) -> np.ndarray:
        return self.field("joint_position")

    @property
    def shuttle_position(self) -> np.ndarray:
        return self.field("shuttle_position")

    @property
    def shuttle_velocity(self) -> np.ndarray:
        return self.field("shuttle_velocity").reshape(-1, 3)[-1]

    @property
    def time_to_hit_s(self) -> float:
        return float(self.field("time_to_hit_s")[0])

    @property
    def feasibility_flags(self) -> Tuple[bool, bool, bool, bool]:
        raw = self.field("feasibility_flags")
        return tuple(bool(v > 0.5) for v in raw)  # type: ignore[return-value]


def intercept_target(view: ObservationView) -> np.ndarray:
    """The shared target of BOTH baselines: linear extrapolation of the shuttle estimate.

    Identical for both baselines on purpose -- the comparison must not differ because one policy
    got a different target.  The synthetic episodes construct shuttle states whose extrapolation
    at time_to_hit_s IS the episode target, so the success metric is identical too.
    """
    return view.shuttle_position + view.shuttle_velocity * view.time_to_hit_s


def _slew(current: np.ndarray, desired: np.ndarray, max_delta: float) -> np.ndarray:
    delta = np.clip(desired - current, -max_delta, max_delta)
    return current + delta


def _cap_norm(vector: np.ndarray, max_norm: float) -> np.ndarray:
    """Keep |v| inside the combined-translation limit (P5-05 also limits the norm)."""
    vec = np.asarray(vector, dtype=float)
    norm = float(np.linalg.norm(vec))
    if norm > float(max_norm) > 0.0:
        return vec * (float(max_norm) / norm)
    return vec


class BaselinePolicy(Protocol):
    name: str

    def act(self, observation: ObservationView, *, joint_position: Any,
            previous_action: Optional[Mapping[str, ActionValue]] = None,
            dt_s: float = DEFAULT_DT_S) -> Dict[str, ActionValue]:
        ...


def _previous(previous_action: Optional[Mapping[str, ActionValue]], limits: ActionLimits,
              name: str, width: int) -> np.ndarray:
    if previous_action is None or name not in previous_action:
        return np.zeros(width)
    return np.asarray(previous_action[name].value, dtype=float).reshape(-1)


def _safety_limits(limits: ActionLimits, dt_s: float) -> Dict[str, float]:
    return {
        "max_linear_speed": float(limits.base_linear_velocity_max.value),
        "max_yaw_rate": float(limits.base_angular_velocity_max.value),
        "max_joint_delta": float(limits.joint_velocity_max.value) * dt_s,
        "linear_slew": float(limits.base_linear_accel_max.value) * dt_s,
        "yaw_slew": float(limits.base_angular_accel_max.value) * dt_s,
        "joint_slew": float(limits.joint_accel_max.value) * dt_s,
    }


class ReachabilityFSMBaseline:
    """Fail-closed finite-state baseline (see LocalReachabilityFSMStub for what its verdict is)."""
    name = "reachability-fsm"
    kind = "finite_state"

    def __init__(self, fsm: Optional[LocalReachabilityFSMStub] = None,
                 limits: ActionLimits = DEFAULT_LIMITS, speed_fraction: float = 0.9,
                 deadband_m: float = 0.05) -> None:
        self.fsm = fsm or LocalReachabilityFSMStub()
        self.limits = limits
        self.speed_fraction = float(speed_fraction)
        self.deadband_m = float(deadband_m)
        self.last_state: str = "INIT"
        self.last_reason: str = "not stepped yet"

    def act(self, observation: ObservationView, *, joint_position: Any,
            previous_action: Optional[Mapping[str, ActionValue]] = None,
            dt_s: float = DEFAULT_DT_S) -> Dict[str, ActionValue]:
        view = observation
        verdict = self.fsm.verdict(view.feasibility_flags)
        self.last_state, self.last_reason = verdict.state, verdict.reason
        lim = _safety_limits(self.limits, dt_s)
        prev_v = _previous(previous_action, self.limits, "base_linear_velocity", 2)
        prev_w = _previous(previous_action, self.limits, "base_angular_velocity", 1)
        prev_dq = _previous(previous_action, self.limits, "joint_position_delta", 6)
        if not verdict.may_act:
            v = _slew(prev_v, np.zeros(2), lim["linear_slew"])
            w = _slew(prev_w, np.zeros(1), lim["yaw_slew"])
            dq = _slew(prev_dq, np.zeros(6), lim["joint_slew"])
        else:
            target = intercept_target(view)
            err = target - view.racket_position
            v_des = np.where(np.abs(err[:2]) > self.deadband_m,
                             np.sign(err[:2]) * lim["max_linear_speed"] * self.speed_fraction, 0.0)
            v_des = _cap_norm(v_des, lim["max_linear_speed"] * self.speed_fraction)
            v = _slew(prev_v, v_des, lim["linear_slew"])
            w = _slew(prev_w, np.zeros(1), lim["yaw_slew"])
            dq_des = np.zeros(6)
            if abs(float(err[2])) > self.deadband_m:
                dq_des[int(SYNTHETIC_RACKET_MODEL["z_joint_index"])] = (
                    np.sign(float(err[2])) * min(lim["max_joint_delta"], lim["joint_slew"]))
            dq = _slew(prev_dq, dq_des, lim["joint_slew"])
        return {
            "base_linear_velocity": ActionValue(np.clip(v, -lim["max_linear_speed"],
                                                        lim["max_linear_speed"]), "m/s"),
            "base_angular_velocity": ActionValue(np.clip(w, -lim["max_yaw_rate"],
                                                         lim["max_yaw_rate"]), "rad/s"),
            "joint_position_delta": ActionValue(np.clip(dq, -lim["max_joint_delta"],
                                                        lim["max_joint_delta"]), "rad"),
        }


class TrackingControllerBaseline:
    """Proportional tracking of the racket-to-target error inside the P5-05 envelope."""
    name = "tracking-p"
    kind = "proportional"

    def __init__(self, limits: ActionLimits = DEFAULT_LIMITS, kp_linear: float = 3.0,
                 kp_joint: float = 0.20, deadband_m: float = 0.01) -> None:
        self.limits = limits
        self.kp_linear = float(kp_linear)
        self.kp_joint = float(kp_joint)
        self.deadband_m = float(deadband_m)
        self.last_state: str = "INIT"
        self.last_reason: str = "not stepped yet"

    def act(self, observation: ObservationView, *, joint_position: Any,
            previous_action: Optional[Mapping[str, ActionValue]] = None,
            dt_s: float = DEFAULT_DT_S) -> Dict[str, ActionValue]:
        view = observation
        lim = _safety_limits(self.limits, dt_s)
        target = intercept_target(view)
        err = target - view.racket_position
        prev_v = _previous(previous_action, self.limits, "base_linear_velocity", 2)
        prev_w = _previous(previous_action, self.limits, "base_angular_velocity", 1)
        prev_dq = _previous(previous_action, self.limits, "joint_position_delta", 6)
        v_des = np.clip(self.kp_linear * err[:2], -lim["max_linear_speed"], lim["max_linear_speed"])
        v_des = _cap_norm(v_des, lim["max_linear_speed"] * 0.92)
        if float(np.linalg.norm(err[:2])) < self.deadband_m:
            v_des = np.zeros(2)
        v = _slew(prev_v, v_des, lim["linear_slew"])
        w = _slew(prev_w, np.zeros(1), lim["yaw_slew"])
        dq_des = np.zeros(6)
        dq_des[int(SYNTHETIC_RACKET_MODEL["z_joint_index"])] = float(
            np.clip(self.kp_joint * float(err[2]), -lim["max_joint_delta"], lim["max_joint_delta"]))
        dq = _slew(prev_dq, dq_des, lim["joint_slew"])
        self.last_state = "TRACK"
        self.last_reason = "proportional correction on the racket-to-target error, flags ignored"
        return {
            "base_linear_velocity": ActionValue(np.clip(v, -lim["max_linear_speed"],
                                                        lim["max_linear_speed"]), "m/s"),
            "base_angular_velocity": ActionValue(np.clip(w, -lim["max_yaw_rate"],
                                                         lim["max_yaw_rate"]), "rad/s"),
            "joint_position_delta": ActionValue(np.clip(dq, -lim["max_joint_delta"],
                                                        lim["max_joint_delta"]), "rad"),
        }


def build_baselines(limits: ActionLimits = DEFAULT_LIMITS) -> Dict[str, Any]:
    """The two baselines, keyed by name, sharing the same action limits."""
    policies = (ReachabilityFSMBaseline(limits=limits), TrackingControllerBaseline(limits=limits))
    return {p.name: p for p in policies}


def new_safety_shield(require_context: bool = True) -> SafetyShield:
    """P5-05 safety envelope as actually used by the harness (TEMP proxy limits)."""
    from ..safety.safety_shield import SafetyLimits
    return SafetyShield(num_envs=1, limits=SafetyLimits.temp_proxy(),
                        require_context=require_context)


__all__ = [
    "BASELINE_VERSION", "BASELINE_STATUS", "P7_FSM_MODULE", "P7_FSM_EXPECTED_AVAILABLE",
    "SYNTHETIC_RACKET_MODEL", "FSM_STATES", "FSM_TRACK", "FSM_HOLD_UNREACHABLE",
    "FSM_HOLD_OUT_OF_WORKSPACE", "FSM_HOLD_TIMING", "FSM_HOLD_INVALID_PREDICTION",
    "BaselineError", "ObservationInterfaceError", "ObservationView", "ReachabilityVerdict",
    "LocalReachabilityFSMStub", "ReachabilityFSMBaseline", "TrackingControllerBaseline",
    "BaselinePolicy", "intercept_target", "p7_fsm_status", "build_baselines", "new_safety_shield",
]
