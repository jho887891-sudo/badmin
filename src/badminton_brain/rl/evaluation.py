# -*- coding: utf-8 -*-
"""P8-04: the FAIR evaluation harness shared by every baseline (and, later, by PPO).

The harness is deliberately policy-agnostic.  One episode list -> one identical loop for each
baseline:

  1. build the P8-01 observation from the synthetic state and validate it with
     observation_schema.validate (the strict encoder path),
  2. ask the policy for a P8-01 action mapping,
  3. validate/to_command the action (bounds, units, rate limits),
  4. pass the resulting WholeBodyTarget through the P5-05 SafetyShield and record its verdict,
  5. apply the SAFE command to the synthetic state and score one identical success/failure rule.

Success = the racket contact point is within the episode tolerance of the episode target at some
step before the horizon.  Timeout = horizon exhausted without success.  Safety reject = the
shield did not return PASS (the baseline asked for something outside the envelope) -- the
episode ends there and is counted, never silently dropped.

SYNTHETIC DYNAMICS (documented, deliberately boring)
----------------------------------------------------
  base_position_{t+1} = base_position_t + dt * base_twist[0:2]      (planar, court == body frame)
  base origin starts at STATION_ORIGIN = (-1.50, 0.00, 0.00) so the initial racket point is
  already inside the racket workspace box of the P8-01 schema
  joint_position_{t+1} = safe joint_position_target                 (no arm dynamics at all)
  racket_position     = base_position + (0.30, 0.00, 0.60 + 0.40*sin(q2))
This is a kinematic toy with 3 actuated degrees of freedom.  It is NOT a robot model and it is
NOT physics; every number produced with it is labelled synthetic.
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from typing import Any, Dict, Mapping, Optional, Sequence, Tuple

import numpy as np

from ..safety.safety_shield import ACTION_PASS, SafetyContext
from ..types import WholeBodyTarget
from .action_schema import DEFAULT_DT_S, DEFAULT_LIMITS, ActionValue, to_command
from .baselines import (
    SYNTHETIC_RACKET_MODEL, ObservationView, new_safety_shield,
)
from .curriculum import CurriculumSchedule, CurriculumSplitViolation
from .observation_schema import ObservationValue, build_observation as _encode_observation
from .seed_ledger import SPLITS, SeedLedger

EVALUATION_VERSION = "p8-eval-v0.1"
SYNTHETIC_DYNAMICS = "documented kinematic toy (3 actuated DoF); not a robot model, not physics"
DEFAULT_HORIZON_STEPS = 400
DEFAULT_TOLERANCE_M = 0.05
SHUTTLE_VELOCITY = np.array([0.0, 0.0, -3.0])  # falling shuttle, synthetic
TIME_TO_HIT_S = 0.5

#: The synthetic robot station: the base origin is INSIDE the racket workspace box, so the
#: initial racket point (base + xy_offset) is legal for the P8-01 observation schema.
STATION_ORIGIN = np.array([-1.50, 0.00, 0.00])
WORKSPACE_MIN = np.asarray(DEFAULT_LIMITS.racket_workspace_min.value, dtype=float)
WORKSPACE_MAX = np.asarray(DEFAULT_LIMITS.racket_workspace_max.value, dtype=float)
_Z_INDEX = int(SYNTHETIC_RACKET_MODEL["z_joint_index"])
XY_OFFSET = np.asarray(SYNTHETIC_RACKET_MODEL["xy_offset_m"], dtype=float)


class EvaluationProtocolError(ValueError):
    """Base class for harness refusals."""


class LedgerEpisodeMismatchError(EvaluationProtocolError):
    """The seed ledger and the episode list do not describe the same experiment."""


class EpisodeDefinitionError(EvaluationProtocolError):
    """An episode is malformed (split unknown, target outside the workspace, ...)."""


def racket_position_from(base_position: Any, joint_position: Any) -> np.ndarray:
    """The synthetic racket model (documented in the module docstring)."""
    p = np.asarray(base_position, dtype=float).reshape(3)
    q = np.asarray(joint_position, dtype=float).reshape(6)
    z = float(SYNTHETIC_RACKET_MODEL["z_base_m"]) + float(
        SYNTHETIC_RACKET_MODEL["z_amplitude_m"]) * np.sin(q[_Z_INDEX])
    return np.array([p[0] + XY_OFFSET[0], p[1] + XY_OFFSET[1], p[2] + z])


@dataclass
class SyntheticState:
    base_position: np.ndarray = field(default_factory=lambda: np.zeros(3))
    joint_position: np.ndarray = field(default_factory=lambda: np.zeros(6))

    def copy(self) -> "SyntheticState":
        return SyntheticState(self.base_position.copy(), self.joint_position.copy())


@dataclass(frozen=True)
class Episode:
    """One reproducible episode: seed, split, curriculum level and the target it is judged on."""
    episode_id: str
    seed: int
    split: str
    level: str
    target: np.ndarray
    initial_base_position: np.ndarray
    initial_joint_position: np.ndarray
    horizon_steps: int = DEFAULT_HORIZON_STEPS
    tolerance_m: float = DEFAULT_TOLERANCE_M
    shuttle_velocity: np.ndarray = field(default_factory=lambda: SHUTTLE_VELOCITY.copy())
    time_to_hit_s: float = TIME_TO_HIT_S
    feasibility_flags: Tuple[bool, bool, bool, bool] = (True, True, True, True)

    def __post_init__(self) -> None:
        if self.split not in SPLITS:
            raise EpisodeDefinitionError("episode {0}: unknown split {1!r}"
                                         .format(self.episode_id, self.split))
        target = np.asarray(self.target, dtype=float).reshape(3)
        if not np.all(np.isfinite(target)):
            raise EpisodeDefinitionError("episode {0}: target is not finite".format(self.episode_id))
        if bool(np.any(target < WORKSPACE_MIN)) or bool(np.any(target > WORKSPACE_MAX)):
            raise EpisodeDefinitionError(
                "episode {0}: target {1} is outside the racket workspace [{2}, {3}]"
                .format(self.episode_id, target.tolist(), WORKSPACE_MIN.tolist(),
                        WORKSPACE_MAX.tolist()))
        if int(self.horizon_steps) < 1:
            raise EpisodeDefinitionError("episode {0}: horizon must be >= 1".format(self.episode_id))
        if not float(self.tolerance_m) > 0.0:
            raise EpisodeDefinitionError("episode {0}: tolerance must be > 0"
                                         .format(self.episode_id))

    @property
    def shuttle_position(self) -> np.ndarray:
        """Shuttle state whose linear extrapolation at time_to_hit_s equals the target."""
        return np.asarray(self.target, dtype=float) - np.asarray(self.shuttle_velocity,
                                                                dtype=float) * float(self.time_to_hit_s)

    def to_dict(self) -> Dict[str, Any]:
        return {"episode_id": self.episode_id, "seed": int(self.seed), "split": self.split,
                "level": self.level, "target": [float(v) for v in self.target],
                "initial_base_position": [float(v) for v in self.initial_base_position],
                "horizon_steps": int(self.horizon_steps),
                "tolerance_m": float(self.tolerance_m),
                "feasibility_flags": [bool(f) for f in self.feasibility_flags]}


@dataclass(frozen=True)
class EpisodeList:
    """Seeds + targets + the ledger hash they came from: the unit of a fair comparison."""
    episodes: Tuple[Episode, ...]
    ledger_hash: str
    split: str
    level: str

    def __post_init__(self) -> None:
        if self.split not in SPLITS:
            raise EpisodeDefinitionError("episode list split must be one of {0}, got {1!r}"
                                         .format(SPLITS, self.split))
        if not self.episodes:
            raise EpisodeDefinitionError("an episode list needs at least one episode")
        for ep in self.episodes:
            if ep.split != self.split:
                raise EpisodeDefinitionError(
                    "episode {0} is in split {1!r} but the list is split {2!r}"
                    .format(ep.episode_id, ep.split, self.split))

    def fingerprint(self) -> str:
        blob = json.dumps({"ledger_hash": self.ledger_hash, "split": self.split,
                           "level": self.level,
                           "episodes": [ep.to_dict() for ep in self.episodes]},
                          sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(blob.encode("utf-8")).hexdigest()

    def __len__(self) -> int:
        return len(self.episodes)


def episodes_from_ledger(ledger: SeedLedger, schedule: CurriculumSchedule, *,
                         split: str, level: str, count: int,
                         horizon_steps: int = DEFAULT_HORIZON_STEPS,
                         tolerance_m: float = DEFAULT_TOLERANCE_M) -> EpisodeList:
    """Build the episode list from the ledger; the level's split MUST match the seed split."""
    if split not in SPLITS:
        raise EpisodeDefinitionError("unknown split {0!r}".format(split))
    lv = schedule.level(level)
    if lv.split != split:
        raise CurriculumSplitViolation(
            "level {0!r} belongs to split {1!r}; refusing to draw it from the {2!r} seed set"
            .format(level, lv.split, split))
    seeds = ledger.seeds_for(split)
    n = int(count)
    if n < 1:
        raise EpisodeDefinitionError("count must be >= 1, got {0}".format(count))
    if n > len(seeds):
        raise EpisodeDefinitionError(
            "split {0!r} holds {1} seeds but {2} episodes were requested"
            .format(split, len(seeds), n))
    episodes = []
    for i in range(n):
        seed = int(seeds[i])
        params = schedule.sample(level, seed)
        target = np.array([params["target_x_m"], params["target_y_m"], params["target_z_m"]])
        distance = float(np.linalg.norm(target[:2]))
        reachable = distance <= float(params["reach_radius_m"])
        in_workspace = not (bool(np.any(target < WORKSPACE_MIN))
                            or bool(np.any(target > WORKSPACE_MAX)))
        episodes.append(Episode(
            episode_id="{0}/{1}/{2}".format(split, level, i), seed=seed, split=split,
            level=level, target=target,
            initial_base_position=STATION_ORIGIN.copy(), initial_joint_position=np.zeros(6),
            horizon_steps=int(horizon_steps), tolerance_m=float(tolerance_m),
            feasibility_flags=(bool(reachable), bool(in_workspace), True, True)))
    return EpisodeList(episodes=tuple(episodes), ledger_hash=ledger.ledger_hash(), split=split,
                       level=level)


def check_alignment(episodes: EpisodeList, ledger: SeedLedger) -> None:
    """REFUSE to evaluate when the ledger and the episode list disagree."""
    if episodes.ledger_hash != ledger.ledger_hash():
        raise LedgerEpisodeMismatchError(
            "episode list was built from ledger {0} but the supplied ledger hashes to {1}; the "
            "run would mix two different seed sets and is refused"
            .format(episodes.ledger_hash, ledger.ledger_hash()))
    try:
        allowed = set(ledger.seeds_for(episodes.split))
    except Exception as exc:
        raise LedgerEpisodeMismatchError(
            "ledger cannot supply split {0!r}: {1}".format(episodes.split, exc))
    for ep in episodes.episodes:
        if int(ep.seed) not in allowed:
            raise LedgerEpisodeMismatchError(
                "episode {0} uses seed {1} which is not in the {2!r} seed set of ledger {3}; "
                "mixed-split evaluation is refused"
                .format(ep.episode_id, ep.seed, episodes.split, ledger.ledger_hash()[:12]))


def assert_identical_episodes(a: EpisodeList, b: EpisodeList) -> None:
    """Fairness gate: two baselines are only comparable on the SAME episode list."""
    if a.fingerprint() != b.fingerprint():
        raise EvaluationProtocolError(
            "episode lists differ ({0} vs {1}); a baseline comparison requires the identical "
            "episode list".format(a.fingerprint()[:12], b.fingerprint()[:12]))


def build_observation(state: SyntheticState, episode: Episode, t_s: float) -> np.ndarray:
    """Encode one step of the synthetic state through the strict P8-01 observation schema."""
    q = np.asarray(state.joint_position, dtype=float)
    p = np.asarray(state.base_position, dtype=float)
    racket = racket_position_from(p, q)
    shuttle = episode.shuttle_position
    v = np.zeros(6)

    def ov(value: Any, unit: str) -> ObservationValue:
        return ObservationValue(value=value, unit=unit, timestamp_s=float(t_s))

    values = {
        "base_position": ov(p, "m"),
        "base_orientation": ov(np.array([1.0, 0.0, 0.0, 0.0]), "1"),
        "base_linear_velocity": ov(np.zeros(3), "m/s"),
        "base_angular_velocity": ov(np.zeros(3), "rad/s"),
        "joint_position": ov(q, "rad"),
        "joint_velocity": ov(v, "rad/s"),
        "racket_position": ov(racket, "m"),
        "racket_orientation": ov(np.array([1.0, 0.0, 0.0, 0.0]), "1"),
        "racket_linear_velocity": ov(np.zeros(3), "m/s"),
        "shuttle_position": ov(shuttle, "m"),
        "shuttle_velocity": ov(np.tile(np.asarray(episode.shuttle_velocity, dtype=float), (3, 1)),
                               "m/s"),
        "shuttle_covariance": ov(np.zeros(9), "m^2"),
        "time_to_hit_s": ov(np.array([float(episode.time_to_hit_s)]), "s"),
        "feasibility_flags": ov(np.array(episode.feasibility_flags, dtype=bool), "1"),
        "prev_base_linear_command": ov(np.zeros(2), "m/s"),
        "prev_base_angular_command": ov(np.zeros(1), "rad/s"),
        "prev_joint_delta_command": ov(np.zeros(6), "rad"),
    }
    return _encode_observation(values, now_s=float(t_s))


@dataclass(frozen=True)
class EpisodeOutcome:
    episode_id: str
    seed: int
    result: str  # "success" | "timeout" | "safety_reject" | "action_reject"
    steps: int
    shield_action: str
    policy_state: str
    min_distance_m: float
    final_distance_m: float


@dataclass(frozen=True)
class EvaluationReport:
    """Counts only -- there is no training signal and no RL number anywhere in here."""
    policy_name: str
    policy_kind: str
    split: str
    level: str
    ledger_hash: str
    episode_fingerprint: str
    episodes_total: int
    success: int
    timeout: int
    safety_reject: int
    action_reject: int
    success_rate: float
    mean_min_distance_m: float
    dt_s: float
    synthetic: bool = True
    status: str = "NON_RL_BASELINE_NO_LEARNING"

    def to_dict(self, include_outcomes: bool = False,
                outcomes: Sequence[EpisodeOutcome] = ()) -> Dict[str, Any]:
        doc: Dict[str, Any] = {
            "policy_name": self.policy_name, "policy_kind": self.policy_kind,
            "split": self.split, "level": self.level, "ledger_hash": self.ledger_hash,
            "episode_fingerprint": self.episode_fingerprint,
            "episodes_total": int(self.episodes_total), "success": int(self.success),
            "timeout": int(self.timeout), "safety_reject": int(self.safety_reject),
            "action_reject": int(self.action_reject),
            "success_rate": float(self.success_rate),
            "mean_min_distance_m": float(self.mean_min_distance_m), "dt_s": float(self.dt_s),
            "synthetic": bool(self.synthetic), "status": self.status,
            "dynamics": SYNTHETIC_DYNAMICS,
        }
        if include_outcomes:
            doc["outcomes"] = [
                {"episode_id": o.episode_id, "seed": int(o.seed), "result": o.result,
                 "steps": int(o.steps), "shield_action": o.shield_action,
                 "policy_state": o.policy_state, "min_distance_m": float(o.min_distance_m),
                 "final_distance_m": float(o.final_distance_m)} for o in outcomes]
        return doc

    def to_json(self, *, include_outcomes: bool = False,
                outcomes: Sequence[EpisodeOutcome] = (), indent: int = 2) -> str:
        return json.dumps(self.to_dict(include_outcomes=include_outcomes, outcomes=outcomes),
                          indent=indent, sort_keys=True)


def _apply(state: SyntheticState, twist: np.ndarray, joint_target: np.ndarray,
           dt_s: float) -> None:
    v = np.asarray(twist, dtype=float).reshape(3)
    state.base_position = np.array([state.base_position[0] + dt_s * float(v[0]),
                                    state.base_position[1] + dt_s * float(v[1]),
                                    state.base_position[2]])
    state.joint_position = np.asarray(joint_target, dtype=float).reshape(6)


def run_episode(policy: Any, episode: Episode, *, dt_s: float = DEFAULT_DT_S) -> EpisodeOutcome:
    """Run one episode through the shared validation path; see the module docstring."""
    state = SyntheticState(np.asarray(episode.initial_base_position, dtype=float).copy(),
                           np.asarray(episode.initial_joint_position, dtype=float).copy())
    shield = new_safety_shield()
    previous_action: Optional[Mapping[str, ActionValue]] = None
    min_distance = float(np.linalg.norm(racket_position_from(state.base_position,
                                                            state.joint_position)
                                       - episode.target))
    result, shield_label, steps = "timeout", "NONE", 0
    for k in range(int(episode.horizon_steps)):
        t_s = float(k) * float(dt_s)
        observation = build_observation(state, episode, t_s)
        view = ObservationView(observation)
        action = policy.act(view, joint_position=state.joint_position,
                            previous_action=previous_action, dt_s=dt_s)
        try:
            command = to_command(action, joint_position=state.joint_position,
                                 previous_action=previous_action, dt_s=dt_s)
        except Exception:
            result = "action_reject"
            steps = k
            break
        previous_action = action
        racket = racket_position_from(state.base_position, state.joint_position)
        pose7 = np.concatenate([racket, np.array([1.0, 0.0, 0.0, 0.0])])
        context = SafetyContext(now=t_s, joint_position=state.joint_position.reshape(1, 6),
                               racket_contact_pose=pose7.reshape(1, 7))
        target = WholeBodyTarget(base_twist=command.base_twist_body.reshape(1, 3),
                                 joint_position_target=command.joint_position_target.reshape(1, 6),
                                 horizon_s=float(dt_s), timestamp=t_s)
        safe = shield.process(target, state=None, context=context, now=t_s)
        shield_label = shield.last_actions[0] if shield.last_actions else "UNKNOWN"
        steps = k + 1
        if shield_label != ACTION_PASS:
            result = "safety_reject"
            break
        _apply(state, np.asarray(safe.base_twist[0], dtype=float),
               np.asarray(safe.joint_position_target[0], dtype=float), dt_s)
        distance = float(np.linalg.norm(
            racket_position_from(state.base_position, state.joint_position) - episode.target))
        min_distance = min(min_distance, distance)
        if distance <= float(episode.tolerance_m):
            result = "success"
            break
    final_distance = float(np.linalg.norm(
        racket_position_from(state.base_position, state.joint_position) - episode.target))
    return EpisodeOutcome(episode_id=episode.episode_id, seed=int(episode.seed), result=result,
                          steps=int(steps), shield_action=str(shield_label),
                          policy_state=str(getattr(policy, "last_state", "")),
                          min_distance_m=min_distance, final_distance_m=final_distance)


def evaluate(policy: Any, episodes: EpisodeList, ledger: SeedLedger, *,
             dt_s: float = DEFAULT_DT_S) -> Tuple[EvaluationReport, Tuple[EpisodeOutcome, ...]]:
    """Evaluate one policy on one episode list; refuses a mismatched ledger."""
    check_alignment(episodes, ledger)
    outcomes = tuple(run_episode(policy, ep, dt_s=dt_s) for ep in episodes.episodes)
    counts: Dict[str, int] = {"success": 0, "timeout": 0, "safety_reject": 0, "action_reject": 0}
    for outcome in outcomes:
        if outcome.result not in counts:
            raise EvaluationProtocolError("unknown episode result {0!r}".format(outcome.result))
        counts[outcome.result] += 1
    total = len(outcomes)
    report = EvaluationReport(
        policy_name=str(getattr(policy, "name", type(policy).__name__)),
        policy_kind=str(getattr(policy, "kind", "unknown")),
        split=episodes.split, level=episodes.level, ledger_hash=episodes.ledger_hash,
        episode_fingerprint=episodes.fingerprint(), episodes_total=int(total),
        success=counts["success"], timeout=counts["timeout"],
        safety_reject=counts["safety_reject"], action_reject=counts["action_reject"],
        success_rate=(float(counts["success"]) / float(total)) if total else 0.0,
        mean_min_distance_m=float(np.mean([o.min_distance_m for o in outcomes])) if outcomes else 0.0,
        dt_s=float(dt_s))
    return report, outcomes


def evaluate_baselines(policies: Mapping[str, Any], episodes: EpisodeList, ledger: SeedLedger, *,
                       dt_s: float = DEFAULT_DT_S) -> Dict[str, Tuple[EvaluationReport,
                                                                     Tuple[EpisodeOutcome, ...]]]:
    """Same episodes, same ledger, same path: one report per baseline."""
    check_alignment(episodes, ledger)
    return {name: evaluate(policy, episodes, ledger, dt_s=dt_s)
            for name, policy in policies.items()}


__all__ = [
    "EVALUATION_VERSION", "SYNTHETIC_DYNAMICS", "DEFAULT_HORIZON_STEPS", "DEFAULT_TOLERANCE_M",
    "SHUTTLE_VELOCITY", "TIME_TO_HIT_S", "STATION_ORIGIN", "WORKSPACE_MIN", "WORKSPACE_MAX",
    "EvaluationProtocolError", "LedgerEpisodeMismatchError", "EpisodeDefinitionError",
    "SyntheticState", "Episode", "EpisodeList", "EpisodeOutcome", "EvaluationReport",
    "racket_position_from", "build_observation", "episodes_from_ledger", "check_alignment",
    "assert_identical_episodes", "run_episode", "evaluate", "evaluate_baselines",
]
