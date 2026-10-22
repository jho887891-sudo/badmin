# -*- coding: utf-8 -*-
"""P8-05: the TRAINING GATE -- an ENFORCEMENT object, not a document.

WHAT THIS MODULE IS
-------------------
A small, dependency-free state machine that decides whether a training run may start, keeps
the run inside a pre-registered budget, and defines the on-disk checkpoint contract.  It
refuses with typed exceptions; nothing here is a warning or a log line.

A run may start only if ALL of the following hold (each failure has its own exception):

  1. a pre-registration record EXISTS ......................... PreRegistrationMissingError
  2. it is COMPLETE (every field, every hyperparameter) ....... PreRegistrationIncompleteError
  3. its SHA-256 pin still matches its content (not mutated) .. PreRegistrationIntegrityError
  4. the authorization flag is explicitly True ................ AuthorizationMissingError
  5. P5/P6/P7 are DONE/PASS in the caller-supplied status map . UpstreamGateNotSatisfiedError
  6. the requested budget does not exceed the registered one .. BudgetExceedsRegisteredError
  7. the seed ledger hash equals the registered one ........... SeedLedgerMismatchError

(The evidence-based dependency gate that must also be satisfied lives in dependency_gate.py;
authorize_training_run below calls it and the flag alone can never bypass it.)

BUDGET ENFORCEMENT
------------------
BudgetAccountant takes an INJECTED clock (clock=..., default time.monotonic) so it can be
tested without sleeping a single second.  admit_step refuses BEFORE the step runs once the
wall-clock / GPU / episode / step budget is spent, latches BUDGET_EXHAUSTED and keeps
refusing afterwards (fail-closed, not fail-open).

CHECKPOINT CONTRACT
-------------------
* one file per checkpoint, written atomically (temp file in the same directory + os.replace);
* a SHA-256 over (metadata, payload bytes) -- a single flipped byte is refused on load;
* resume REFUSES when the pre-registration hash, the config hash or the seed ledger hash of
  the checkpoint differs from the caller's, because that would silently mix two experiments;
* STOP_REASONS is the closed vocabulary every stop must be recorded with.

NO TRAINING IS PERFORMED HERE.  There is no optimiser, no rollout, no torch, no GPU call --
this module only decides, counts and persists.  See management/morph_one_rs/
P8_TRAINING_GATE_ENFORCEMENT.md.
"""
from __future__ import annotations

import base64
import hashlib
import json
import math
import os
import time
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, Mapping, Optional, Sequence, Tuple

TRAINING_GATE_VERSION = "p8-training-gate-v0.1"
CHECKPOINT_FORMAT = "p8-training-checkpoint-v0.1"

#: The only authorization key this gate accepts.
AUTHORIZATION_FLAG = "ppo_training_authorized"
#: The only algorithm this gate registers.
SUPPORTED_ALGORITHM = "PPO"

#: Upstream gates that must be DONE/PASS before a PPO run may be authorized (queue P8-06).
REQUIRED_UPSTREAM_GATES: Tuple[str, ...] = ("P5", "P6", "P7")
PASSING_GATE_STATUSES: Tuple[str, ...] = ("DONE", "PASS")

#: Closed vocabulary of stop reasons.  Every stop record and checkpoint uses one of these.
STOP_REASONS: Tuple[str, ...] = (
    "BUDGET_EXHAUSTED",  # wall-clock / GPU / episode / step budget spent
    "PLATEAU",           # no improvement within the registered patience/plateau window
    "DIVERGED",          # non-finite or runaway metric
    "SAFETY",            # a safety guard (shield/limits/e-stop) fired
    "OPERATOR",          # a human stopped the run
    "CRASH",             # the process died; resume from the last checkpoint is allowed
)

#: Hyperparameters that MUST be pre-registered (no "we will tune it later").
REQUIRED_HYPERPARAMETERS: Tuple[str, ...] = (
    "learning_rate", "gamma", "gae_lambda", "clip_range", "entropy_coef",
    "n_steps", "batch_size", "n_epochs",
)
#: Stop conditions that MUST be pre-registered.
REQUIRED_STOP_CONDITIONS: Tuple[str, ...] = (
    "max_steps", "patience_evaluations", "plateau_window", "min_delta",
)
COMPARATORS: Tuple[str, ...] = (">=", ">", "<=", "<", "==")


# --------------------------------------------------------------------------------------
# exceptions: one per refusal so a caller can never mistake a refusal for a warning
# --------------------------------------------------------------------------------------
class TrainingGateError(ValueError):
    """Base class for every training-gate refusal."""


class PreRegistrationMissingError(TrainingGateError):
    """No pre-registration record exists; a run cannot start without one."""


class PreRegistrationIncompleteError(TrainingGateError):
    """The pre-registration record is missing required fields or they are invalid."""


class PreRegistrationIntegrityError(TrainingGateError):
    """The pre-registration content no longer matches its SHA-256 pin (mutated)."""


class AuthorizationMissingError(TrainingGateError):
    """The authorization flag is absent, not literally True, or has no accountable grantor."""


class UpstreamGateNotSatisfiedError(TrainingGateError):
    """At least one of P5/P6/P7 is not DONE/PASS in the caller-supplied status map."""

    def __init__(self, message: str, *, unsatisfied: Sequence[str] = ()) -> None:
        super().__init__(message)
        self.unsatisfied = tuple(unsatisfied)


class BudgetExceedsRegisteredError(TrainingGateError):
    """The requested budget is larger than the pre-registered budget."""

    def __init__(self, message: str, *, dimension: str = "") -> None:
        super().__init__(message)
        self.dimension = dimension


class SeedLedgerMismatchError(TrainingGateError):
    """The seed ledger hash differs from the pre-registered one."""


class BudgetExhaustedError(TrainingGateError):
    """A budget is exhausted: the step is REFUSED before it starts."""

    def __init__(self, message: str, *, dimension: str = "",
                 stop_reason: str = "BUDGET_EXHAUSTED") -> None:
        super().__init__(message)
        self.dimension = dimension
        self.stop_reason = stop_reason


class CheckpointIntegrityError(TrainingGateError):
    """The checkpoint file is missing, unreadable or its hash does not match."""


class CheckpointConfigMismatchError(TrainingGateError):
    """Resuming would mix two experiments (registration/config/ledger hash differs)."""

    def __init__(self, message: str, *, mismatches: Sequence[str] = ()) -> None:
        super().__init__(message)
        self.mismatches = tuple(mismatches)


class StopReasonError(TrainingGateError):
    """A stop reason outside STOP_REASONS."""


# --------------------------------------------------------------------------------------
# hashing helpers
# --------------------------------------------------------------------------------------
def _canonical(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


def _sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def is_sha256_hex(value: Any) -> bool:
    """True for a 64-character lowercase hex digest (the only pin format accepted)."""
    if not isinstance(value, str) or len(value) != 64:
        return False
    return all(c in "0123456789abcdef" for c in value)


def _is_number(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)


def _is_int(value: Any) -> bool:
    return isinstance(value, int) and not isinstance(value, bool)


# --------------------------------------------------------------------------------------
# the pre-registration record
# --------------------------------------------------------------------------------------
@dataclass(frozen=True)
class PreRegistration:
    """Everything that must be decided BEFORE a run, pinned by a SHA-256 over itself."""

    algorithm: str
    hyperparameters: Mapping[str, Any]
    seed_ledger_hash: str
    episode_budget: int
    wall_clock_budget_s: float
    gpu_budget_s: float
    success_metric: str
    acceptance: Mapping[str, Any]
    stop_conditions: Mapping[str, Any]
    evaluation_protocol_hash: str
    version: str = TRAINING_GATE_VERSION
    pinned_hash: Optional[str] = None

    # -- payload / hash ------------------------------------------------------------
    def payload(self) -> Dict[str, Any]:
        """Exactly the content the pin covers (everything except the pin itself)."""
        return {
            "version": self.version,
            "algorithm": self.algorithm,
            "hyperparameters": dict(self.hyperparameters),
            "seed_ledger_hash": self.seed_ledger_hash,
            # raw values, NOT coerced: a float episode_budget must fail validate() instead of
            # being silently truncated into a different (valid-looking) registration
            "episode_budget": self.episode_budget,
            "wall_clock_budget_s": self.wall_clock_budget_s,
            "gpu_budget_s": self.gpu_budget_s,
            "success_metric": self.success_metric,
            "acceptance": dict(self.acceptance),
            "stop_conditions": dict(self.stop_conditions),
            "evaluation_protocol_hash": self.evaluation_protocol_hash,
        }

    def registration_hash(self) -> str:
        return _sha256_text(_canonical(self.payload()))

    def to_dict(self) -> Dict[str, Any]:
        doc = self.payload()
        doc["registration_hash"] = self.registration_hash()
        return doc

    def to_json(self, *, indent: int = 2) -> str:
        return json.dumps(self.to_dict(), indent=indent, sort_keys=True) + "\n"

    def write(self, path: Any) -> Path:
        """Write the pinned record atomically (temp file + os.replace)."""
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        tmp = target.with_name(target.name + ".tmp-" + uuid.uuid4().hex)
        try:
            tmp.write_text(self.to_json(), encoding="utf-8")
            os.replace(tmp, target)
        except BaseException:
            _unlink_quietly(tmp)
            raise
        return target

    # -- completeness --------------------------------------------------------------
    def validate(self) -> None:
        """Raise PreRegistrationIncompleteError naming every missing/invalid field."""
        problems = []
        if self.version != TRAINING_GATE_VERSION:
            problems.append("version must be {0!r}, got {1!r}"
                            .format(TRAINING_GATE_VERSION, self.version))
        if not isinstance(self.algorithm, str) or self.algorithm.strip().upper() != SUPPORTED_ALGORITHM:
            problems.append("algorithm must be {0!r} (the only registered algorithm), got {1!r}"
                            .format(SUPPORTED_ALGORITHM, self.algorithm))
        if not isinstance(self.hyperparameters, Mapping) or not self.hyperparameters:
            problems.append("hyperparameters must be a non-empty mapping")
        else:
            missing = [k for k in REQUIRED_HYPERPARAMETERS if k not in self.hyperparameters]
            if missing:
                problems.append("hyperparameters missing " + ", ".join(missing))
            bad = [k for k, v in self.hyperparameters.items() if not _is_number(v)]
            if bad:
                problems.append("hyperparameters not finite numbers: " + ", ".join(sorted(bad)))
        if not _is_int(self.episode_budget) or self.episode_budget < 1:
            problems.append("episode_budget must be an int >= 1, got {0!r}"
                            .format(self.episode_budget))
        if not _is_number(self.wall_clock_budget_s) or self.wall_clock_budget_s <= 0:
            problems.append("wall_clock_budget_s must be a finite number > 0, got {0!r}"
                            .format(self.wall_clock_budget_s))
        if not _is_number(self.gpu_budget_s) or self.gpu_budget_s <= 0:
            problems.append("gpu_budget_s must be a finite number > 0, got {0!r}"
                            .format(self.gpu_budget_s))
        if not isinstance(self.success_metric, str) or not self.success_metric.strip():
            problems.append("success_metric must be a non-empty string")
        problems.extend(_acceptance_problems(self.acceptance))
        problems.extend(_stop_condition_problems(self.stop_conditions))
        for name in ("seed_ledger_hash", "evaluation_protocol_hash"):
            if not is_sha256_hex(getattr(self, name)):
                problems.append("{0} must be a 64-char lowercase sha256 hex digest, got {1!r}"
                                .format(name, getattr(self, name)))
        if problems:
            raise PreRegistrationIncompleteError(
                "pre-registration is incomplete, a run may not start: " + "; ".join(problems))

    def verify_pin(self) -> None:
        """Refuse content that no longer matches its pin (edit after registration)."""
        if self.pinned_hash is None:
            raise PreRegistrationIntegrityError(
                "pre-registration is not sha256-pinned; a pinned record must be written to "
                "JSON and re-loaded before a run may start")
        actual = self.registration_hash()
        if actual != self.pinned_hash:
            raise PreRegistrationIntegrityError(
                "pre-registration hash mismatch: pinned {0}, recomputed {1}; the record was "
                "edited after registration".format(self.pinned_hash, actual))

    # -- (de)serialisation ---------------------------------------------------------
    @classmethod
    def from_dict(cls, doc: Mapping[str, Any]) -> "PreRegistration":
        if not isinstance(doc, Mapping):
            raise PreRegistrationIncompleteError("pre-registration must be a JSON object")
        required = ("version", "algorithm", "hyperparameters", "seed_ledger_hash",
                    "episode_budget", "wall_clock_budget_s", "gpu_budget_s",
                    "success_metric", "acceptance", "stop_conditions",
                    "evaluation_protocol_hash", "registration_hash")
        missing = [k for k in required if k not in doc]
        if missing:
            raise PreRegistrationIncompleteError(
                "pre-registration document is missing field(s): " + ", ".join(missing))
        record = cls(
            version=str(doc["version"]), algorithm=str(doc["algorithm"]),
            hyperparameters=dict(doc["hyperparameters"]),
            seed_ledger_hash=str(doc["seed_ledger_hash"]),
            episode_budget=doc["episode_budget"],
            wall_clock_budget_s=doc["wall_clock_budget_s"],
            gpu_budget_s=doc["gpu_budget_s"],
            success_metric=str(doc["success_metric"]),
            acceptance=dict(doc["acceptance"]),
            stop_conditions=dict(doc["stop_conditions"]),
            evaluation_protocol_hash=str(doc["evaluation_protocol_hash"]),
            pinned_hash=str(doc["registration_hash"]),
        )
        record.validate()
        record.verify_pin()
        return record

    @classmethod
    def load(cls, path: Any) -> "PreRegistration":
        target = Path(path)
        if not target.exists():
            raise PreRegistrationMissingError(
                "no pre-registration record at {0}; a training run may not start".format(target))
        try:
            doc = json.loads(target.read_text(encoding="utf-8"))
        except (OSError, ValueError) as exc:
            raise PreRegistrationIncompleteError(
                "pre-registration at {0} is not readable JSON: {1}".format(target, exc))
        return cls.from_dict(doc)


def _acceptance_problems(acceptance: Any) -> list:
    if not isinstance(acceptance, Mapping):
        return ["acceptance must be a mapping with threshold/comparator"]
    problems = []
    threshold = acceptance.get("threshold")
    if not _is_number(threshold) or not (0.0 <= float(threshold) <= 1.0):
        problems.append("acceptance.threshold must be a finite number in [0, 1], got {0!r}"
                        .format(threshold))
    comparator = acceptance.get("comparator", ">=")
    if comparator not in COMPARATORS:
        problems.append("acceptance.comparator must be one of {0}, got {1!r}"
                        .format(COMPARATORS, comparator))
    return problems


def _stop_condition_problems(conditions: Any) -> list:
    if not isinstance(conditions, Mapping):
        return ["stop_conditions must be a mapping"]
    missing = [k for k in REQUIRED_STOP_CONDITIONS if k not in conditions]
    if missing:
        return ["stop_conditions missing " + ", ".join(missing)]
    problems = []
    for key, minimum in (("max_steps", 1), ("patience_evaluations", 1),
                         ("plateau_window", 2)):
        value = conditions.get(key)
        if not _is_int(value) or int(value) < minimum:
            problems.append("stop_conditions.{0} must be an int >= {1}, got {2!r}"
                            .format(key, minimum, value))
    min_delta = conditions.get("min_delta")
    if not _is_number(min_delta) or float(min_delta) < 0.0:
        problems.append("stop_conditions.min_delta must be a finite number >= 0, got {0!r}"
                        .format(min_delta))
    return problems


# --------------------------------------------------------------------------------------
# budget request / decision
# --------------------------------------------------------------------------------------
@dataclass(frozen=True)
class BudgetRequest:
    """What the caller wants to run NOW; compared against the registered budget."""
    episodes: int = 1
    wall_clock_s: float = 0.0
    gpu_s: float = 0.0
    steps: int = 0

    def validate(self) -> None:
        problems = []
        if not _is_int(self.episodes) or self.episodes < 0:
            problems.append("episodes must be an int >= 0, got {0!r}".format(self.episodes))
        if not _is_int(self.steps) or self.steps < 0:
            problems.append("steps must be an int >= 0, got {0!r}".format(self.steps))
        for name in ("wall_clock_s", "gpu_s"):
            value = getattr(self, name)
            if not _is_number(value) or float(value) < 0:
                problems.append("{0} must be a finite number >= 0, got {1!r}".format(name, value))
        if problems:
            raise BudgetExceedsRegisteredError(
                "invalid budget request: " + "; ".join(problems), dimension="request")


@dataclass(frozen=True)
class GateDecision:
    """The signed-off receipt of a passing check (never produced by a refusal)."""
    algorithm: str
    registration_hash: str
    seed_ledger_hash: str
    evaluation_protocol_hash: str
    episode_budget: int
    wall_clock_budget_s: float
    gpu_budget_s: float
    upstream_gate_status: Mapping[str, str]
    authorization_flag: bool = True

    def to_dict(self) -> Dict[str, Any]:
        return {"algorithm": self.algorithm, "registration_hash": self.registration_hash,
                "seed_ledger_hash": self.seed_ledger_hash,
                "evaluation_protocol_hash": self.evaluation_protocol_hash,
                "episode_budget": int(self.episode_budget),
                "wall_clock_budget_s": float(self.wall_clock_budget_s),
                "gpu_budget_s": float(self.gpu_budget_s),
                "upstream_gate_status": dict(self.upstream_gate_status),
                "authorization_flag": bool(self.authorization_flag),
                "training_gate_version": TRAINING_GATE_VERSION}


# --------------------------------------------------------------------------------------
# the gate itself
# --------------------------------------------------------------------------------------
@dataclass(frozen=True)
class TrainingGate:
    """Holds the pre-registration (or nothing) and answers check() with a refusal or a receipt."""
    registration: Optional[PreRegistration] = None
    missing_reason: str = ""

    @classmethod
    def from_file(cls, path: Any) -> "TrainingGate":
        target = Path(path)
        if not target.exists():
            return cls(registration=None,
                       missing_reason="pre-registration file not found: {0}".format(target))
        return cls(registration=PreRegistration.load(target))

    def check(self, *, request: BudgetRequest, seed_ledger_hash: str,
              upstream_gates: Mapping[str, str],
              authorization: Mapping[str, Any]) -> GateDecision:
        """REFUSE (typed exception) or return the decision receipt.  No warnings-only path."""
        # 1/2/3: the record must exist, be complete and still match its pin.
        if self.registration is None:
            raise PreRegistrationMissingError(
                "a training run was requested but no pre-registration record exists: {0}"
                .format(self.missing_reason or "none supplied"))
        self.registration.validate()
        self.registration.verify_pin()
        # 4: the authorization flag must be explicitly True and attributable.
        if not isinstance(authorization, Mapping):
            raise AuthorizationMissingError(
                "authorization must be a mapping carrying {0!r}; got {1!r}"
                .format(AUTHORIZATION_FLAG, type(authorization).__name__))
        if AUTHORIZATION_FLAG not in authorization:
            raise AuthorizationMissingError(
                "authorization flag {0!r} is not set; absent is not permission"
                .format(AUTHORIZATION_FLAG))
        if authorization.get(AUTHORIZATION_FLAG) is not True:
            raise AuthorizationMissingError(
                "authorization flag {0!r} must be literally True (got {1!r}); a string, a number "
                "or a default is not an authorization"
                .format(AUTHORIZATION_FLAG, authorization.get(AUTHORIZATION_FLAG)))
        granted_by = authorization.get("granted_by")
        if not isinstance(granted_by, str) or not granted_by.strip():
            raise AuthorizationMissingError(
                "authorization flag {0!r} has no accountable 'granted_by'; an anonymous flag is "
                "not an authorization".format(AUTHORIZATION_FLAG))
        # 5: upstream P5/P6/P7 must be DONE/PASS.
        if not isinstance(upstream_gates, Mapping):
            raise UpstreamGateNotSatisfiedError(
                "upstream gate statuses must be a mapping gate -> status, got {0!r}"
                .format(type(upstream_gates).__name__),
                unsatisfied=REQUIRED_UPSTREAM_GATES)
        unsatisfied = []
        for gate_id in REQUIRED_UPSTREAM_GATES:
            status = upstream_gates.get(gate_id)
            if not isinstance(status, str) or status.strip().upper() not in PASSING_GATE_STATUSES:
                unsatisfied.append("{0}={1!r}".format(gate_id, status))
        if unsatisfied:
            raise UpstreamGateNotSatisfiedError(
                "upstream gate(s) not DONE/PASS: {0}; required statuses are {1}"
                .format(", ".join(unsatisfied), PASSING_GATE_STATUSES),
                unsatisfied=tuple(item.split("=")[0] for item in unsatisfied))
        # 6: the requested budget must fit inside the registered one.
        request.validate()
        reg = self.registration
        max_steps = int(reg.stop_conditions["max_steps"])
        exceed = []
        if request.episodes > reg.episode_budget:
            exceed.append(("episode", request.episodes, reg.episode_budget))
        if float(request.wall_clock_s) > float(reg.wall_clock_budget_s):
            exceed.append(("wall_clock", request.wall_clock_s, reg.wall_clock_budget_s))
        if float(request.gpu_s) > float(reg.gpu_budget_s):
            exceed.append(("gpu", request.gpu_s, reg.gpu_budget_s))
        if request.steps > max_steps:
            exceed.append(("steps", request.steps, max_steps))
        if exceed:
            raise BudgetExceedsRegisteredError(
                "requested budget exceeds the pre-registered budget: " + "; ".join(
                    "{0} requested {1!r} > registered {2!r}".format(name, want, have)
                    for name, want, have in exceed),
                dimension=exceed[0][0])
        # 7: the seed ledger must be the registered one.
        if str(seed_ledger_hash) != reg.seed_ledger_hash:
            raise SeedLedgerMismatchError(
                "seed ledger hash {0!r} differs from the registered {1!r}; the run would use "
                "seeds that were never pre-registered"
                .format(str(seed_ledger_hash), reg.seed_ledger_hash))
        return GateDecision(
            algorithm=reg.algorithm, registration_hash=reg.registration_hash(),
            seed_ledger_hash=reg.seed_ledger_hash,
            evaluation_protocol_hash=reg.evaluation_protocol_hash,
            episode_budget=int(reg.episode_budget),
            wall_clock_budget_s=float(reg.wall_clock_budget_s),
            gpu_budget_s=float(reg.gpu_budget_s),
            upstream_gate_status={k: str(upstream_gates[k]) for k in REQUIRED_UPSTREAM_GATES})


@dataclass(frozen=True)
class RunAuthorization:
    """What the caller gets when BOTH the gate and the dependency gate passed."""
    decision: GateDecision
    dependency_requirements: Tuple[str, ...]

    def to_dict(self) -> Dict[str, Any]:
        doc = self.decision.to_dict()
        doc["dependency_requirements"] = list(self.dependency_requirements)
        doc["authorized"] = True
        return doc


def authorize_training_run(*, gate: TrainingGate, request: BudgetRequest,
                           seed_ledger_hash: str,
                           upstream_gates: Mapping[str, str],
                           authorization: Mapping[str, Any],
                           dependency_gate: Any) -> RunAuthorization:
    """The ONLY path to an authorization; the flag alone cannot bypass the dependency gate.

    dependency_gate must expose require_satisfied() (see dependency_gate.py).  It is a required
    keyword argument, so a caller cannot forget it: the evidence-based gate runs even when the
    flag is set and even when the caller claims P5/P6/P7 are DONE/PASS.
    """
    decision = gate.check(request=request, seed_ledger_hash=seed_ledger_hash,
                          upstream_gates=upstream_gates, authorization=authorization)
    dependency_gate.require_satisfied()
    ids = tuple(getattr(item, "requirement_id", str(item))
                for item in getattr(dependency_gate, "requirements", ()))
    return RunAuthorization(decision=decision, dependency_requirements=ids)


# --------------------------------------------------------------------------------------
# budget accountant (injected clock; never sleeps)
# --------------------------------------------------------------------------------------
@dataclass(frozen=True)
class BudgetSnapshot:
    wall_elapsed_s: float
    wall_remaining_s: float
    gpu_used_s: float
    gpu_remaining_s: float
    episodes_used: int
    episodes_remaining: int
    steps_used: int
    steps_remaining: int
    exhausted: bool
    stop_reason: Optional[str]

    def to_dict(self) -> Dict[str, Any]:
        return {"wall_elapsed_s": float(self.wall_elapsed_s),
                "wall_remaining_s": float(self.wall_remaining_s),
                "gpu_used_s": float(self.gpu_used_s),
                "gpu_remaining_s": float(self.gpu_remaining_s),
                "episodes_used": int(self.episodes_used),
                "episodes_remaining": int(self.episodes_remaining),
                "steps_used": int(self.steps_used),
                "steps_remaining": int(self.steps_remaining),
                "exhausted": bool(self.exhausted),
                "stop_reason": self.stop_reason}


@dataclass(frozen=True)
class StepAdmission:
    """A granted step: the caller may now run exactly this much work."""
    step_index: int
    episodes: int
    gpu_s: float
    snapshot: BudgetSnapshot


class BudgetAccountant:
    """Counts wall-clock / GPU / episode / step budget and refuses before a step starts."""

    def __init__(self, registration: PreRegistration, *,
                 clock: Optional[Callable[[], float]] = None,
                 gpu_used_s: float = 0.0) -> None:
        registration.validate()
        registration.verify_pin()
        self._registration = registration
        self._clock = clock if clock is not None else time.monotonic
        self._started_at = float(self._clock())
        self._steps_used = 0
        self._episodes_used = 0
        self._gpu_used_s = float(gpu_used_s)
        self._stop_reason: Optional[str] = None
        self._stopped = False

    # -- read-only views ------------------------------------------------------------
    @property
    def registration(self) -> PreRegistration:
        return self._registration

    @property
    def stop_reason(self) -> Optional[str]:
        return self._stop_reason

    def elapsed_s(self) -> float:
        return max(0.0, float(self._clock()) - self._started_at)

    def snapshot(self) -> BudgetSnapshot:
        elapsed = self.elapsed_s()
        wall_budget = float(self._registration.wall_clock_budget_s)
        gpu_budget = float(self._registration.gpu_budget_s)
        episodes = int(self._registration.episode_budget)
        max_steps = int(self._registration.stop_conditions["max_steps"])
        exhausted = (self._stopped or elapsed >= wall_budget
                     or self._gpu_used_s >= gpu_budget
                     or self._episodes_used >= episodes
                     or self._steps_used >= max_steps)
        return BudgetSnapshot(
            wall_elapsed_s=elapsed, wall_remaining_s=max(0.0, wall_budget - elapsed),
            gpu_used_s=self._gpu_used_s, gpu_remaining_s=max(0.0, gpu_budget - self._gpu_used_s),
            episodes_used=self._episodes_used,
            episodes_remaining=max(0, episodes - self._episodes_used),
            steps_used=self._steps_used, steps_remaining=max(0, max_steps - self._steps_used),
            exhausted=exhausted,
            stop_reason=self._stop_reason or ("BUDGET_EXHAUSTED" if exhausted else None))

    # -- refusals -------------------------------------------------------------------
    def _refuse(self, dimension: str, message: str) -> None:
        self._stopped = True
        if self._stop_reason is None:
            self._stop_reason = "BUDGET_EXHAUSTED"
        raise BudgetExhaustedError(
            "{0} (dimension={1}); the step is refused before it starts".format(message, dimension),
            dimension=dimension, stop_reason=self._stop_reason)

    def _assert_not_exhausted(self) -> None:
        snapshot = self.snapshot()
        if self._stopped:
            self._refuse("latched", "the accountant is latched stopped ({0})"
                         .format(self._stop_reason))
        if snapshot.wall_elapsed_s >= float(self._registration.wall_clock_budget_s):
            self._refuse("wall_clock",
                         "wall-clock budget exhausted: {0:.3f}s elapsed of {1:.3f}s registered"
                         .format(snapshot.wall_elapsed_s,
                                 self._registration.wall_clock_budget_s))
        if snapshot.gpu_used_s >= float(self._registration.gpu_budget_s):
            self._refuse("gpu", "GPU budget exhausted: {0:.3f}s used of {1:.3f}s registered"
                         .format(snapshot.gpu_used_s, self._registration.gpu_budget_s))
        if snapshot.episodes_used >= int(self._registration.episode_budget):
            self._refuse("episode", "episode budget exhausted: {0} used of {1} registered"
                         .format(snapshot.episodes_used, self._registration.episode_budget))
        if snapshot.steps_used >= int(self._registration.stop_conditions["max_steps"]):
            self._refuse("max_steps", "step budget exhausted: {0} steps used of {1} registered"
                         .format(snapshot.steps_used,
                                 self._registration.stop_conditions["max_steps"]))

    def admit_step(self, *, episodes: int = 1, gpu_s: float = 0.0,
                   step_wall_s: float = 0.0) -> StepAdmission:
        """Grant one step or refuse.  Refusal is latched and covers the step's own cost."""
        if not _is_int(episodes) or episodes < 1:
            raise BudgetExhaustedError(
                "episodes per step must be an int >= 1, got {0!r}".format(episodes),
                dimension="request")
        if not _is_number(gpu_s) or float(gpu_s) < 0:
            raise BudgetExhaustedError(
                "gpu_s per step must be a finite number >= 0, got {0!r}".format(gpu_s),
                dimension="request")
        if not _is_number(step_wall_s) or float(step_wall_s) < 0:
            raise BudgetExhaustedError(
                "step_wall_s must be a finite number >= 0, got {0!r}".format(step_wall_s),
                dimension="request")
        self._assert_not_exhausted()
        if self._episodes_used + int(episodes) > int(self._registration.episode_budget):
            self._refuse("episode", "the step needs {0} episode(s) but only {1} remain"
                         .format(episodes, self.snapshot().episodes_remaining))
        if self._gpu_used_s + float(gpu_s) > float(self._registration.gpu_budget_s):
            self._refuse("gpu", "the step needs {0:.3f}s GPU but only {1:.3f}s remain"
                         .format(float(gpu_s), self.snapshot().gpu_remaining_s))
        if self.elapsed_s() + float(step_wall_s) >= float(self._registration.wall_clock_budget_s):
            self._refuse("wall_clock",
                         "the step would reach the wall-clock budget ({0:.3f}s elapsed + "
                         "{1:.3f}s step >= {2:.3f}s registered)"
                         .format(self.elapsed_s(), float(step_wall_s),
                                 self._registration.wall_clock_budget_s))
        self._steps_used += 1
        self._episodes_used += int(episodes)
        self._gpu_used_s += float(gpu_s)
        return StepAdmission(step_index=self._steps_used - 1, episodes=int(episodes),
                             gpu_s=float(gpu_s), snapshot=self.snapshot())

    def stop(self, reason: str, detail: str = "") -> "StopRecord":
        """Record a stop with a reason from STOP_REASONS; unknown reasons are refused."""
        if reason not in STOP_REASONS:
            raise StopReasonError("stop reason {0!r} is outside the vocabulary {1}"
                                  .format(reason, STOP_REASONS))
        self._stopped = True
        if self._stop_reason is None:
            self._stop_reason = reason
        return StopRecord(reason=reason, detail=detail, steps_used=self._steps_used,
                          episodes_used=self._episodes_used, step_index=self._steps_used,
                          budget=self.snapshot())

    def lapse_as_crash(self, detail: str = "process stopped responding") -> "StopRecord":
        """A run that disappears is recorded as CRASH (resumable), never as success."""
        return self.stop("CRASH", detail)


# --------------------------------------------------------------------------------------
# stop criteria
# --------------------------------------------------------------------------------------
@dataclass(frozen=True)
class StopRecord:
    """One stop, in the closed vocabulary, with the budget state at the moment of stopping."""
    reason: str
    detail: str
    steps_used: int
    episodes_used: int
    step_index: int
    budget: Optional[BudgetSnapshot] = None

    def __post_init__(self) -> None:
        if self.reason not in STOP_REASONS:
            raise StopReasonError("stop reason {0!r} is outside the vocabulary {1}"
                                  .format(self.reason, STOP_REASONS))

    def to_dict(self) -> Dict[str, Any]:
        return {"reason": self.reason, "detail": self.detail,
                "steps_used": int(self.steps_used), "episodes_used": int(self.episodes_used),
                "step_index": int(self.step_index),
                "budget": self.budget.to_dict() if self.budget is not None else None}


@dataclass(frozen=True)
class StopCriteria:
    """Pre-registered stop conditions turned into a decision function."""
    max_steps: int
    patience_evaluations: int
    plateau_window: int
    min_delta: float
    metric: str = "success_rate"
    comparator: str = ">="
    threshold: float = 0.0

    @classmethod
    def from_registration(cls, registration: PreRegistration) -> "StopCriteria":
        registration.validate()
        conditions = registration.stop_conditions
        acceptance = registration.acceptance
        return cls(max_steps=int(conditions["max_steps"]),
                   patience_evaluations=int(conditions["patience_evaluations"]),
                   plateau_window=int(conditions["plateau_window"]),
                   min_delta=float(conditions["min_delta"]),
                   metric=str(registration.success_metric),
                   comparator=str(acceptance.get("comparator", ">=")),
                   threshold=float(acceptance["threshold"]))

    def meets_acceptance(self, value: float) -> bool:
        if not _is_number(value):
            return False
        number = float(value)
        return {">=": number >= self.threshold, ">": number > self.threshold,
                "<=": number <= self.threshold, "<": number < self.threshold,
                "==": number == self.threshold}[self.comparator]

    def evaluate(self, history: Sequence[float], *, steps_used: int,
                 diverged: bool = False) -> Optional[StopRecord]:
        """Return the stop record if the run must stop, else None (pure function)."""
        values = [float(v) for v in history]
        if diverged or any(not math.isfinite(v) for v in values):
            return StopRecord(reason="DIVERGED", detail="non-finite metric observed",
                              steps_used=int(steps_used), episodes_used=0,
                              step_index=int(steps_used))
        if int(steps_used) >= int(self.max_steps):
            return StopRecord(reason="BUDGET_EXHAUSTED",
                              detail="max_steps={0} reached".format(self.max_steps),
                              steps_used=int(steps_used), episodes_used=0,
                              step_index=int(steps_used))
        window = int(self.plateau_window)
        if window >= 2 and len(values) >= window:
            recent = values[-window:]
            if max(recent) - min(recent) < float(self.min_delta):
                return StopRecord(
                    reason="PLATEAU",
                    detail="spread {0:.6f} over the last {1} evaluations is below min_delta "
                           "{2:.6f}".format(max(recent) - min(recent), window, self.min_delta),
                    steps_used=int(steps_used), episodes_used=0, step_index=int(steps_used))
        patience = int(self.patience_evaluations)
        if patience and len(values) > patience:
            best_before = max(values[:-patience])
            if max(values[-patience:]) <= best_before:
                return StopRecord(
                    reason="PLATEAU",
                    detail="{0} consecutive evaluations without beating {1:.6f}"
                           .format(patience, best_before),
                    steps_used=int(steps_used), episodes_used=0, step_index=int(steps_used))
        return None


# --------------------------------------------------------------------------------------
# checkpoint contract
# --------------------------------------------------------------------------------------
@dataclass(frozen=True)
class CheckpointMetadata:
    """The identity of a checkpoint: which experiment, which config, which seeds, how far."""
    step: int
    episode: int
    registration_hash: str
    seed_ledger_hash: str
    evaluation_protocol_hash: str
    config_hash: str
    git_sha: str = ""
    stop_reason: Optional[str] = None

    def validate(self) -> None:
        problems = []
        if not _is_int(self.step) or self.step < 0:
            problems.append("step must be an int >= 0, got {0!r}".format(self.step))
        if not _is_int(self.episode) or self.episode < 0:
            problems.append("episode must be an int >= 0, got {0!r}".format(self.episode))
        for name in ("registration_hash", "seed_ledger_hash", "evaluation_protocol_hash",
                     "config_hash"):
            if not is_sha256_hex(getattr(self, name)):
                problems.append("{0} must be a 64-char sha256 hex digest, got {1!r}"
                                .format(name, getattr(self, name)))
        if self.stop_reason is not None and self.stop_reason not in STOP_REASONS:
            problems.append("stop_reason {0!r} is outside the vocabulary {1}"
                            .format(self.stop_reason, STOP_REASONS))
        if problems:
            raise StopReasonError("invalid checkpoint metadata: " + "; ".join(problems))

    def payload(self) -> Dict[str, Any]:
        return {"step": int(self.step), "episode": int(self.episode),
                "registration_hash": self.registration_hash,
                "seed_ledger_hash": self.seed_ledger_hash,
                "evaluation_protocol_hash": self.evaluation_protocol_hash,
                "config_hash": self.config_hash, "git_sha": self.git_sha,
                "stop_reason": self.stop_reason}

    def metadata_hash(self) -> str:
        return _sha256_text(_canonical(self.payload()))


def checkpoint_digest(metadata: CheckpointMetadata, payload: Mapping[str, bytes]) -> str:
    """SHA-256 over the metadata AND the payload bytes (one edited byte changes it)."""
    encoded = {name: base64.b64encode(bytes(blob)).decode("ascii")
               for name, blob in sorted(payload.items())}
    return _sha256_text(_canonical({"format": CHECKPOINT_FORMAT,
                                    "metadata": metadata.payload(), "payload": encoded}))


@dataclass(frozen=True)
class CheckpointRecord:
    path: str
    metadata: CheckpointMetadata
    checkpoint_hash: str
    payload_sha256: str
    payload_bytes: int

    def to_dict(self) -> Dict[str, Any]:
        return {"path": self.path, "checkpoint_hash": self.checkpoint_hash,
                "payload_sha256": self.payload_sha256, "payload_bytes": self.payload_bytes,
                "metadata": self.metadata.payload()}


def _unlink_quietly(path: Path) -> None:
    try:
        Path(path).unlink()
    except OSError:
        pass


def write_checkpoint(path: Any, *, metadata: CheckpointMetadata,
                     payload: Mapping[str, bytes], fsync: bool = True) -> CheckpointRecord:
    """Atomic write: full content to a temp file in the same dir, then os.replace."""
    metadata.validate()
    if not isinstance(payload, Mapping):
        raise CheckpointIntegrityError("checkpoint payload must be a mapping name -> bytes")
    encoded = {name: base64.b64encode(bytes(blob)).decode("ascii")
               for name, blob in sorted(payload.items())}
    digest = _sha256_text(_canonical({"payload": encoded}))
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    document = {"format": CHECKPOINT_FORMAT, "metadata": metadata.payload(),
                "payload": encoded, "payload_sha256": digest}
    document["checkpoint_hash"] = _sha256_text(
        _canonical({"format": CHECKPOINT_FORMAT, "metadata": metadata.payload(),
                    "payload": encoded}))
    blob = json.dumps(document, sort_keys=True, indent=2) + "\n"
    tmp = target.with_name(target.name + ".tmp-" + uuid.uuid4().hex)
    try:
        with open(tmp, "w", encoding="utf-8", newline="\n") as handle:
            handle.write(blob)
            handle.flush()
            if fsync:
                os.fsync(handle.fileno())
        os.replace(tmp, target)
    except BaseException:
        _unlink_quietly(tmp)
        raise
    total = sum(len(bytes(v)) for v in payload.values())
    return CheckpointRecord(path=str(target), metadata=metadata,
                            checkpoint_hash=document["checkpoint_hash"],
                            payload_sha256=digest, payload_bytes=int(total))


def read_checkpoint(path: Any) -> Tuple[CheckpointMetadata, Dict[str, bytes], CheckpointRecord]:
    """Load and VERIFY a checkpoint; any edit/truncation is refused, never repaired."""
    target = Path(path)
    if not target.exists():
        raise CheckpointIntegrityError("no checkpoint at {0}".format(target))
    try:
        document = json.loads(target.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        raise CheckpointIntegrityError(
            "checkpoint at {0} is not readable JSON: {1}".format(target, exc))
    for name in ("format", "metadata", "payload", "payload_sha256", "checkpoint_hash"):
        if name not in document:
            raise CheckpointIntegrityError(
                "checkpoint at {0} is missing field {1!r}".format(target, name))
    if document["format"] != CHECKPOINT_FORMAT:
        raise CheckpointIntegrityError(
            "checkpoint format {0!r} is not {1!r}"
            .format(document["format"], CHECKPOINT_FORMAT))
    try:
        meta_doc = document["metadata"]
        metadata = CheckpointMetadata(
            step=meta_doc["step"], episode=meta_doc["episode"],
            registration_hash=str(meta_doc["registration_hash"]),
            seed_ledger_hash=str(meta_doc["seed_ledger_hash"]),
            evaluation_protocol_hash=str(meta_doc["evaluation_protocol_hash"]),
            config_hash=str(meta_doc["config_hash"]),
            git_sha=str(meta_doc.get("git_sha", "")),
            stop_reason=meta_doc.get("stop_reason"))
        metadata.validate()
        payload = {name: base64.b64decode(encoded.encode("ascii"), validate=True)
                   for name, encoded in document["payload"].items()}
    except (KeyError, TypeError, ValueError) as exc:
        raise CheckpointIntegrityError(
            "checkpoint at {0} is malformed: {1}".format(target, exc))
    expected = _sha256_text(_canonical({"format": CHECKPOINT_FORMAT,
                                        "metadata": metadata.payload(),
                                        "payload": document["payload"]}))
    if document["checkpoint_hash"] != expected:
        raise CheckpointIntegrityError(
            "checkpoint hash mismatch at {0}: recorded {1}, recomputed {2}; the file was edited "
            "or corrupted after it was written"
            .format(target, document["checkpoint_hash"], expected))
    payload_digest = _sha256_text(
        _canonical({"payload": {name: base64.b64encode(blob).decode("ascii")
                                for name, blob in sorted(payload.items())}}))
    if document["payload_sha256"] != payload_digest:
        raise CheckpointIntegrityError(
            "checkpoint payload hash mismatch at {0}: recorded {1}, recomputed {2}"
            .format(target, document["payload_sha256"], payload_digest))
    record = CheckpointRecord(path=str(target), metadata=metadata, checkpoint_hash=expected,
                              payload_sha256=payload_digest,
                              payload_bytes=sum(len(blob) for blob in payload.values()))
    return metadata, payload, record


#: Stop reasons after which the run may NOT be resumed without a new decision.
NON_RESUMABLE_STOP_REASONS: Tuple[str, ...] = ("SAFETY", "DIVERGED")


def resume_checkpoint(path: Any, *, registration: PreRegistration,
                      seed_ledger_hash: str, config_hash: str) -> CheckpointRecord:
    """Verify the checkpoint and REFUSE to resume a different experiment."""
    metadata, _payload, record = read_checkpoint(path)
    registration.validate()
    mismatches = []
    if metadata.registration_hash != registration.registration_hash():
        mismatches.append("registration_hash")
    if metadata.seed_ledger_hash != str(seed_ledger_hash):
        mismatches.append("seed_ledger_hash")
    if metadata.config_hash != str(config_hash):
        mismatches.append("config_hash")
    if mismatches:
        raise CheckpointConfigMismatchError(
            "refusing to resume {0}: {1} differ(s) from the checkpoint; the run would continue "
            "one experiment with another experiment's state"
            .format(path, ", ".join(mismatches)), mismatches=tuple(mismatches))
    if metadata.stop_reason in NON_RESUMABLE_STOP_REASONS:
        raise CheckpointConfigMismatchError(
            "refusing to resume {0}: it was written after a {1} stop, which needs a new "
            "authorization decision, not a resume".format(path, metadata.stop_reason),
            mismatches=("stop_reason",))
    return record


__all__ = [
    "TRAINING_GATE_VERSION", "CHECKPOINT_FORMAT", "AUTHORIZATION_FLAG", "SUPPORTED_ALGORITHM",
    "REQUIRED_UPSTREAM_GATES", "PASSING_GATE_STATUSES", "STOP_REASONS",
    "REQUIRED_HYPERPARAMETERS", "REQUIRED_STOP_CONDITIONS", "COMPARATORS",
    "NON_RESUMABLE_STOP_REASONS",
    "TrainingGateError", "PreRegistrationMissingError", "PreRegistrationIncompleteError",
    "PreRegistrationIntegrityError", "AuthorizationMissingError",
    "UpstreamGateNotSatisfiedError", "BudgetExceedsRegisteredError", "SeedLedgerMismatchError",
    "BudgetExhaustedError", "CheckpointIntegrityError", "CheckpointConfigMismatchError",
    "StopReasonError",
    "is_sha256_hex", "PreRegistration", "BudgetRequest", "GateDecision", "TrainingGate",
    "RunAuthorization", "authorize_training_run", "BudgetSnapshot", "StepAdmission",
    "BudgetAccountant", "StopRecord", "StopCriteria", "CheckpointMetadata",
    "checkpoint_digest", "CheckpointRecord", "write_checkpoint", "read_checkpoint",
    "resume_checkpoint",
]
