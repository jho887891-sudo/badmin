# -*- coding: utf-8 -*-
"""P8-06: the ALGORITHM DEPENDENCY GATE -- what PPO would need, and whether it exists yet.

WHAT THIS MODULE IS
-------------------
A machine-checkable declaration of the upstream work PPO depends on, plus a refusal.  Every
requirement names

  * the queue item it comes from (P5-04, P5-09, P6-02, P7-10, ...),
  * what PPO would actually need from it (safe controller + HOLD, verified kinematics/TCP,
    fail-closed FSM + reachability, ...),
  * an evidence artifact path that must exist ON THE BRANCH UNDER TEST,
  * the branch/commit where that artifact was last seen, so the pointer is falsifiable, and
  * a recorded status that only a human may raise, together with the command and exit code
    that justified it.

require_satisfied() raises and NAMES every unsatisfied requirement.  It never consults the
authorization flag: an authorization cannot create evidence, so it cannot bypass this gate.

STATUS OF THE TABLE (recorded 2026-10-22 on branch p8/seeds-baselines @ b74994a)
------------------------------------------------------------------------------
All eleven requirements are MISSING_EVIDENCE.  The P5/P6/P7 deliverables exist only on
unmerged branches (p5/controller-hold, p5/multienv-offline, p6/racket-camera-frames,
p6/dynamic-fk, p7/reachability-fsm, p7/closed-loop-wiring); none of them is merged into this
branch, so none of the artifacts resolves here:

    $ git merge-base --is-ancestor p5/controller-hold HEAD   ->  exit code 1
    $ git merge-base --is-ancestor p6/dynamic-fk HEAD        ->  exit code 1
    $ git merge-base --is-ancestor p7/reachability-fsm HEAD  ->  exit code 1

Therefore PPO is NOT AUTHORIZED, and this module is built to keep saying so.  A test asserts
that refusal (tests/badminton_brain/rl/test_rl_dependency_gate.py) -- it is the expected
state, not a bug.

NO TRAINING IS PERFORMED HERE, and no status is invented: an artifact that is absent yields
MISSING_EVIDENCE; an artifact that is present without a recorded acceptance (a real command
and its real exit code) yields UNKNOWN; only a recorded SATISFIED + a present artifact yields
SATISFIED.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, Optional, Sequence, Tuple

DEPENDENCY_GATE_VERSION = "p8-dependency-gate-v0.1"

#: What this module says about PPO today.  It is a constant, not a computed hope.
PPO_TRAINING_STATUS = "NOT_AUTHORIZED"

#: Requirement statuses.  Only SATISFIED lets a requirement through.
STATUS_SATISFIED = "SATISFIED"
STATUS_MISSING = "MISSING_EVIDENCE"
STATUS_UNKNOWN = "UNKNOWN"
STATUSES: Tuple[str, ...] = (STATUS_SATISFIED, STATUS_MISSING, STATUS_UNKNOWN)

#: Repo root of this checkout: src/badminton_brain/rl/dependency_gate.py -> repo.
DEFAULT_ROOT = Path(__file__).resolve().parents[3]


class DependencyGateError(RuntimeError):
    """Base class for every dependency-gate refusal (a hard stop, not a warning)."""


class EvidenceContradictionError(DependencyGateError):
    """The table claims SATISFIED for an artifact that is not on this branch."""


class RequirementMissingError(DependencyGateError):
    """At least one P5/P6/P7 requirement PPO needs is missing; names every one of them."""

    def __init__(self, message: str, *, missing: Sequence[str] = (),
                 reasons: Optional[Dict[str, str]] = None) -> None:
        super().__init__(message)
        self.missing = tuple(missing)
        self.reasons = dict(reasons or {})


@dataclass(frozen=True)
class Requirement:
    """One thing PPO needs from P5/P6/P7, with the evidence that would prove it."""

    requirement_id: str
    upstream: str
    title: str
    needs_ppo: str
    evidence_path: str
    evidence_branch: str
    evidence_commit: str
    declared_status: str = STATUS_MISSING
    note: str = ""

    def __post_init__(self) -> None:
        if self.declared_status not in STATUSES:
            raise EvidenceContradictionError(
                "requirement {0}: declared_status {1!r} is not one of {2}"
                .format(self.requirement_id, self.declared_status, STATUSES))
        if self.upstream not in ("P5", "P6", "P7"):
            raise EvidenceContradictionError(
                "requirement {0}: upstream must be P5/P6/P7, got {1!r}"
                .format(self.requirement_id, self.upstream))

    def evidence_present(self, root: Optional[Any] = None) -> bool:
        base = Path(root) if root is not None else DEFAULT_ROOT
        return (base / self.evidence_path).exists()

    def resolve(self, root: Optional[Any] = None) -> Tuple[str, str]:
        """(status, reason) computed from the real tree; never optimistic."""
        if not self.evidence_present(root):
            if self.declared_status == STATUS_SATISFIED:
                raise EvidenceContradictionError(
                    "requirement {0} is declared SATISFIED but {1} does not exist on this "
                    "branch; a status without evidence is refused"
                    .format(self.requirement_id, self.evidence_path))
            return STATUS_MISSING, (
                "evidence artifact absent from this branch: {0} (seen on {1} @ {2}, unmerged)"
                .format(self.evidence_path, self.evidence_branch, self.evidence_commit[:7]))
        if self.declared_status == STATUS_SATISFIED:
            return STATUS_SATISFIED, "artifact present and the table records an accepted result"
        if self.declared_status == STATUS_MISSING:
            return STATUS_UNKNOWN, (
                "artifact {0} is present but the table still records MISSING_EVIDENCE; a "
                "recorded acceptance (real command + real exit code) is required"
                .format(self.evidence_path))
        return STATUS_UNKNOWN, (
            "artifact {0} is present but no acceptance has been recorded".format(
                self.evidence_path))

    def to_dict(self, root: Optional[Any] = None) -> Dict[str, Any]:
        status, reason = self.resolve(root)
        return {"requirement_id": self.requirement_id, "upstream": self.upstream,
                "title": self.title, "needs_ppo": self.needs_ppo,
                "evidence_path": self.evidence_path, "evidence_branch": self.evidence_branch,
                "evidence_commit": self.evidence_commit,
                "declared_status": self.declared_status,
                "status": status, "reason": reason, "note": self.note}


#: Everything PPO would need from P5/P6/P7.  The IDs and wording follow the queue file
#: management/morph_one_rs/EXECUTION_QUEUE_CONTINUOUS_P0_P8.md (branch
#: origin/chatgpt/p2-wc-continuous-queue-20261021), sections P5/P6/P7.
DEFAULT_REQUIREMENTS: Tuple[Requirement, ...] = (
    Requirement(
        requirement_id="P5-04", upstream="P5",
        title="explicit joint name -> runtime DOF index map + rad/deg gain acceptance",
        needs_ppo="the policy's 6 joint targets must index the SAME DOFs as the controller; a "
                  "stage-order index would silently command the wrong joint",
        evidence_path="simulation/badminton_scene/dof_index.py",
        evidence_branch="p5/controller-hold", evidence_commit="2afd6f2",
        note="exists only on the unmerged P5 branch; no acceptance exit code recorded here"),
    Requirement(
        requirement_id="P5-05", upstream="P5",
        title="motion-command limits + velocity/acceleration/torque protection",
        needs_ppo="PPO explores, so every sampled action must be bounded by a safe controller "
                  "that cannot command an unsafe target",
        evidence_path="management/morph_one_rs/P5_CONTROLLER_AND_HOLD.md",
        evidence_branch="p5/controller-hold", evidence_commit="2afd6f2",
        note="documents the P5 envelope/HOLD work; unmerged, no gate statement on this branch"),
    Requirement(
        requirement_id="P5-06", upstream="P5",
        title="single-env reset negatives (limit/name/NaN/missing joint) fail closed",
        needs_ppo="episodes must start from a verified reset; a silent clamp would train on "
                  "states the real robot can never reach",
        evidence_path="management/morph_one_rs/P5B_RESET_POSE.md",
        evidence_branch="p5/controller-hold", evidence_commit="2afd6f2",
        note="reset-pose evidence lives on the P5 branch only; unmerged"),
    Requirement(
        requirement_id="P5-07", upstream="P5",
        title="N=2 multi-env instance isolation + randomised reset",
        needs_ppo="vectorised rollouts need provably isolated environments, otherwise the "
                  "gradient mixes episodes that share state",
        evidence_path="management/morph_one_rs/P5_MULTIENV_OFFLINE.md",
        evidence_branch="p5/multienv-offline", evidence_commit="e084efd",
        note="offline multi-env harness is on p5/multienv-offline only; unmerged"),
    Requirement(
        requirement_id="P5-09", upstream="P5",
        title="minimal reusable B601-RS controller API (state, targets, reset, hold, IK hook)",
        needs_ppo="PPO needs a stable controller interface with an explicit HOLD; without it "
                  "the policy output cannot be applied deterministically",
        evidence_path="simulation/badminton_scene/controller.py",
        evidence_branch="p5/controller-hold", evidence_commit="2afd6f2",
        note="controller.py exists only on the unmerged P5 branch"),
    Requirement(
        requirement_id="P6-02", upstream="P6",
        title="racket -> flange -> arm -> base frame tree with relative transforms",
        needs_ppo="the observation's racket pose and the reward's contact point must come from "
                  "one verified chain, otherwise the policy optimises a wrong geometry",
        evidence_path="outputs/simulation/rebot_b601_rs/morph_one/p6_frames/frame_tree.json",
        evidence_branch="p6/racket-camera-frames", evidence_commit="d7672d5",
        note="frame_tree.json is committed only on the unmerged P6 branch"),
    Requirement(
        requirement_id="P6-05", upstream="P6",
        title="multi-frame consistency (camera->base->world->racket, m/mm, handedness)",
        needs_ppo="unit/handedness regressions must exist before a policy is trained on metric "
                  "observations",
        evidence_path="management/morph_one_rs/P6_RACKET_CAMERA_FRAMES.md",
        evidence_branch="p6/racket-camera-frames", evidence_commit="d7672d5",
        note="P6 frame report is on the unmerged P6 branch; no merge into this branch"),
    Requirement(
        requirement_id="P6-09", upstream="P6",
        title="whole-machine integration manifest + multi-env capability gate (opens P7)",
        needs_ppo="P6-09 is the queue's own release condition for the closed loop PPO will be "
                  "trained against",
        evidence_path="management/morph_one_rs/P6_INTEGRATION_MANIFEST.md",
        evidence_branch="(none found)", evidence_commit="unknown",
        note="UNKNOWN: no integration-manifest deliverable was found on any branch inspected; "
             "this requirement may not even be started"),
    Requirement(
        requirement_id="P7-09", upstream="P7",
        title="racket reachable workspace + joint limits + IK + candidate hit times",
        needs_ppo="the action space and the reward's feasibility mask must match the real "
                  "reachable set; otherwise PPO learns physically impossible hits",
        evidence_path="management/morph_one_rs/P7_REACHABILITY_AND_FSM.md",
        evidence_branch="p7/reachability-fsm", evidence_commit="adaee81",
        note="reachability + IK live on the unmerged P7 branch"),
    Requirement(
        requirement_id="P7-10", upstream="P7",
        title="fail-closed safety FSM (SEARCH/TRACK/PREDICT/PLAN/EXECUTE/HOLD)",
        needs_ppo="PPO acts inside the FSM; the FSM must hold on low confidence/timeout/"
                  "unreachable, so a bad policy action ends in HOLD and not in motion",
        evidence_path="management/morph_one_rs/P7_REACHABILITY_AND_FSM.md",
        evidence_branch="p7/reachability-fsm", evidence_commit="adaee81",
        note="FSM doc is on the unmerged P7 branch; no merge, no acceptance record here"),
    Requirement(
        requirement_id="P7-12", upstream="P7",
        title="low-speed hitting repeatability statistics (trials, contact rate, returns)",
        needs_ppo="a training success metric is meaningless without a measured non-RL hit "
                  "baseline on the same protocol",
        evidence_path="outputs/badminton_brain/p7_closed_loop/wired_episodes.json",
        evidence_branch="p7/closed-loop-wiring", evidence_commit="a3d6de0",
        note="closed-loop episodes are committed only on the unmerged P7 branch"),
)


@dataclass(frozen=True)
class AlgorithmDependencyGate:
    """The dependency gate: a table plus a refusal that names every missing requirement."""

    requirements: Tuple[Requirement, ...] = DEFAULT_REQUIREMENTS
    declared_authorization: str = PPO_TRAINING_STATUS

    # -- inspection -----------------------------------------------------------------
    def resolve_table(self, root: Optional[Any] = None) -> Tuple[Dict[str, Any], ...]:
        return tuple(req.to_dict(root) for req in self.requirements)

    def unsatisfied(self, root: Optional[Any] = None) -> Tuple[Requirement, ...]:
        return tuple(req for req in self.requirements
                     if req.resolve(root)[0] != STATUS_SATISFIED)

    def allows_ppo(self, root: Optional[Any] = None) -> bool:
        return not self.unsatisfied(root)

    def summary(self, root: Optional[Any] = None) -> Dict[str, Any]:
        table = self.resolve_table(root)
        counts: Dict[str, int] = {status: 0 for status in STATUSES}
        for row in table:
            counts[row["status"]] = counts.get(row["status"], 0) + 1
        return {"version": DEPENDENCY_GATE_VERSION,
                "ppo_training_status": PPO_TRAINING_STATUS,
                "declared_authorization": self.declared_authorization,
                "allows_ppo": self.allows_ppo(root),
                "counts": counts, "requirements": [dict(row) for row in table]}

    # -- the refusal ----------------------------------------------------------------
    def require_satisfied(self, root: Optional[Any] = None) -> None:
        """Raise RequirementMissingError naming EVERY missing requirement.  No bypass."""
        missing = self.unsatisfied(root)
        if not missing:
            return
        reasons: Dict[str, str] = {}
        lines = []
        for req in missing:
            status, reason = req.resolve(root)
            reasons[req.requirement_id] = "{0}: {1}".format(status, reason)
            lines.append("{0} [{1}] {2} -- {3}".format(req.requirement_id, status,
                                                        req.title, reason))
        raise RequirementMissingError(
            "PPO training is NOT AUTHORIZED: {0} of {1} upstream requirement(s) are "
            "unsatisfied (PPO_TRAINING_STATUS={2}; an authorization flag cannot create "
            "evidence):\n  - {3}".format(len(missing), len(self.requirements),
                                          PPO_TRAINING_STATUS, "\n  - ".join(lines)),
            missing=tuple(req.requirement_id for req in missing), reasons=reasons)


def default_gate() -> AlgorithmDependencyGate:
    """The gate as it stands in this repository: PPO is refused."""
    return AlgorithmDependencyGate()


def format_table(gate: Optional[AlgorithmDependencyGate] = None,
                 root: Optional[Any] = None) -> str:
    """Human-readable status table (used by the report and by the tests)."""
    active = gate if gate is not None else default_gate()
    rows = active.resolve_table(root)
    width = max(len(r["requirement_id"]) for r in rows) if rows else 2
    lines = ["{0:<{w}}  {1:<6}  {2:<16}  {3}".format("ID", "UP", "STATUS", "NEEDS (PPO)",
                                                     w=width)]
    for row in rows:
        lines.append("{0:<{w}}  {1:<6}  {2:<16}  {3}".format(
            row["requirement_id"], row["upstream"], row["status"], row["title"], w=width))
    return "\n".join(lines)


__all__ = [
    "DEPENDENCY_GATE_VERSION", "PPO_TRAINING_STATUS", "STATUS_SATISFIED", "STATUS_MISSING",
    "STATUS_UNKNOWN", "STATUSES", "DEFAULT_ROOT", "DependencyGateError",
    "EvidenceContradictionError", "RequirementMissingError", "Requirement",
    "DEFAULT_REQUIREMENTS", "AlgorithmDependencyGate", "default_gate", "format_table",
]
