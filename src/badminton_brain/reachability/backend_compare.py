# -*- coding: utf-8 -*-
"""R10-F2 (P7-09): verified-8DOF vs proxy-6DOF reachability, with provenance on EVERY verdict.

THE PROBLEM THIS SOLVES
    src/badminton_brain/reachability/backends.py already refuses to substitute one backend for
    another (select_backend("verified") raises instead of returning the proxy).  What it did not
    have is a *comparison*: a table in which every verdict carries the backend that produced it,
    the frame that verdict is about, and the provenance of that frame.  Without those three labels
    a REACHABLE from the 6-DOF proxy DH and a REACHABLE from the P6 8-DOF FK look identical in a
    report, and the second one silently lends its credibility to the first.

WHAT A LABEL MUST CARRY (and what a test pins)
    backend_id      proxy_dh | verified_p6_fk
    frame_name      racket_face_proxy | gripper_end
    target_kind     RACKET_FACE_PROXY | ARM_END_FRAME
    source_class    PROXY_DH_ENGINEERING_V0_1 | VERIFIED_ASSET_KINEMATICS
    provenance      the backend's own provenance() dict (incl. the FK sha256 when verified)

    A verdict object that carries TWO backend ids raises BackendMixError.  That is the explicit
    test the task asks for: "any object carrying two backend ids raises".  Nothing in this module
    ever averages, merges or reconciles the two backends.

WHY THE COMPARISON IS NOT A RANKING
    The two backends answer DIFFERENT QUESTIONS.  The proxy answers for a 6-DOF racket-face point
    that does not exist on the real asset; the verified backend answers for the authored
    gripper_end frame on link6, whose racket face / racket mount transform is UNKNOWN in the P6
    frame contract (edges 13/14/15).  So the report compares SAID-REACHABLE rates over the same
    target set and says so; it never declares one of them "the true reachability".

NO PHYSICS CLAIM
    This is kinematics only.  No contact, no hit rate, no ball.  See harness/claim_layers.py.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

from .backends import (BACKEND_PROXY, BACKEND_VERIFIED, BackendDampedLeastSquaresIk,
                       BackendMixError, KinematicsBackend, VerifiedFkUnavailable,
                       kinematic_limits_for, select_backend)
from .ik import IkTarget
from .reachability import (Calibration, ReachabilityDecision, ReachabilityReason,
                           ReachabilityVerdict, assess_reachability, target_from_intercept)

__all__ = [
    "PROVENANCE_FIELDS", "VerdictProvenance", "ComparisonScenario", "LabelledVerdict",
    "single_backend_id", "label", "compare_backends", "format_table",
]

#: A label missing any of these is not a label.
PROVENANCE_FIELDS: Tuple[str, ...] = ("backend", "source_class", "frame_name", "target_kind",
                                      "dof", "limit_source")


@dataclass(frozen=True)
class VerdictProvenance:
    """Where one verdict came from.  Frozen: a label cannot be edited after the fact."""

    backend: str
    source_class: str
    frame_name: str
    target_kind: str
    dof: int
    limit_source: str
    detail: Dict[str, Any] = field(default_factory=dict)

    @classmethod
    def of(cls, backend: KinematicsBackend) -> "VerdictProvenance":
        raw = dict(backend.provenance())
        missing = [name for name in PROVENANCE_FIELDS if name not in raw]
        if missing:
            raise BackendMixError(
                "backend %r produced a provenance without %s; an unlabelled verdict is not "
                "accepted" % (getattr(backend, "backend_id", "?"), missing))
        return cls(backend=str(raw["backend"]), source_class=str(raw["source_class"]),
                   frame_name=str(raw["frame_name"]), target_kind=str(raw["target_kind"]),
                   dof=int(raw["dof"]), limit_source=str(raw["limit_source"]),
                   detail={k: v for k, v in raw.items() if k not in PROVENANCE_FIELDS})

    def as_dict(self) -> Dict[str, Any]:
        return {"backend": self.backend, "source_class": self.source_class,
                "frame_name": self.frame_name, "target_kind": self.target_kind,
                "dof": self.dof, "limit_source": self.limit_source, "detail": self.detail}


def single_backend_id(*ids: Optional[str]) -> str:
    """Exactly one backend id, or raise.  Never picks the first, never merges."""
    seen = [str(value) for value in ids if value is not None and str(value).strip() != ""]
    if not seen:
        raise BackendMixError("no backend id was supplied: an unlabelled verdict is refused")
    unique = sorted(set(seen))
    if len(unique) > 1:
        raise BackendMixError(
            "one object carries %d backend ids %s: a verdict belongs to exactly ONE backend, "
            "and mixing them is how a proxy number becomes a verified claim" % (len(unique),
                                                                              unique))
    return unique[0]


@dataclass(frozen=True)
class ComparisonScenario:
    """One target the two backends are both asked about (same numbers, same clock)."""

    name: str
    intercept_point_m: Tuple[float, float, float]
    approach_direction: Tuple[float, float, float] = (0.0, 0.0, -1.0)
    time_to_target_s: float = 0.30
    position_sigma_m: Optional[float] = None
    note: str = ""

    def as_dict(self) -> Dict[str, Any]:
        return {"name": self.name, "intercept_point_m": [float(v) for v in
                                                         self.intercept_point_m],
                "approach_direction": [float(v) for v in self.approach_direction],
                "time_to_target_s": float(self.time_to_target_s),
                "position_sigma_m": self.position_sigma_m, "note": self.note}


@dataclass(frozen=True)
class LabelledVerdict:
    """A reachability verdict, bound to exactly one backend and one frame.

    backend_ids is a tuple on purpose: the field exists so that a caller CAN hand in two ids,
    and __post_init__ then raises.  A single-id field would make the mistake unrepresentable
    and the test that the task asks for unexpressible.
    """

    scenario: str
    backend_ids: Tuple[str, ...]
    provenance: VerdictProvenance
    verdict: ReachabilityVerdict
    seed_rad: Tuple[float, ...]
    limits: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        backend_id = single_backend_id(*self.backend_ids)
        if backend_id != self.provenance.backend:
            raise BackendMixError(
                "labelled verdict for %r says backend %r but its provenance says %r"
                % (self.scenario, backend_id, self.provenance.backend))

    @property
    def backend(self) -> str:
        return self.provenance.backend

    def as_dict(self) -> Dict[str, Any]:
        return {
            "scenario": self.scenario,
            "backend": self.backend,
            "backend_ids": list(self.backend_ids),
            "provenance": self.provenance.as_dict(),
            "seed_rad": [float(v) for v in self.seed_rad],
            "decision": self.verdict.decision.value,
            "reason": self.verdict.reason.value,
            "detail": self.verdict.detail,
            "frame_name": self.provenance.frame_name,
            "target_kind": self.provenance.target_kind,
            "evidence_class": self.verdict.evidence_class,
            "joint_margin_rad": self.verdict.joint_margin_rad,
            "required_slew_s": self.verdict.required_slew_s,
        }


def label(backend: KinematicsBackend, scenario: ComparisonScenario,
          verdict: ReachabilityVerdict, seed: Sequence[float]) -> LabelledVerdict:
    return LabelledVerdict(scenario=scenario.name,
                           backend_ids=(str(backend.backend_id),),
                           provenance=VerdictProvenance.of(backend), verdict=verdict,
                           seed_rad=tuple(float(v) for v in seed))
def _scenario_target(scenario: ComparisonScenario) -> IkTarget:
    return target_from_intercept(scenario.intercept_point_m, scenario.approach_direction)


def compare_backends(scenarios: Sequence[ComparisonScenario], *,
                     backends: Sequence[str] = (BACKEND_PROXY, BACKEND_VERIFIED),
                     verified_kwargs: Optional[Mapping[str, Any]] = None,
                     time_margin_s: float = 0.02,
                     seed_fn=None) -> Dict[str, Any]:
    """Ask each backend the SAME questions, one backend at a time, and never mix the answers.

    A backend that cannot be constructed contributes an explicit UNAVAILABLE section with the
    refusal text - it does NOT contribute proxy numbers under a verified heading.
    """
    verified_kwargs = dict(verified_kwargs or {})
    report: Dict[str, Any] = {
        "schema": "badmin.r10f.backend_compare",
        "schema_version": 1,
        "layer": "TRAJECTORY_REACHABLE",
        "note": "kinematics only: no contact, no ball, no hit rate. The proxy answers for a "
                "6-DOF racket-face point that does not exist on the real asset; the verified "
                "backend answers for the authored gripper_end frame of link6. The two are "
                "compared, never merged.",
        "scenarios": [scenario.as_dict() for scenario in scenarios],
        "backends": {},
    }
    for name in backends:
        try:
            backend = select_backend(str(name), **verified_kwargs) if str(name) == BACKEND_VERIFIED \
                else select_backend(str(name))
        except VerifiedFkUnavailable as exc:
            report["backends"][str(name)] = {
                "backend": str(name), "status": "UNAVAILABLE", "refusal": str(exc),
                "verdicts": [], "note": "no number is produced for an unavailable backend, and "
                                        "the other backend is NOT substituted",
            }
            continue
        provenance = VerdictProvenance.of(backend)
        limits = kinematic_limits_for(backend)
        solver = BackendDampedLeastSquaresIk(backend)
        scenarios_out: List[Dict[str, Any]] = []
        labelled: List[LabelledVerdict] = []
        decided = {decision.value: 0 for decision in ReachabilityDecision}
        for scenario in scenarios:
            seed = ([0.0] * backend.dof if seed_fn is None
                    else [float(v) for v in seed_fn(backend, scenario)])
            if len(seed) != backend.dof:
                raise BackendMixError(
                    "seed width %d does not match backend %r dof %d"
                    % (len(seed), backend.backend_id, backend.dof))
            verdict = assess_reachability(
                target=_scenario_target(scenario), limits=limits,
                time_to_target_s=scenario.time_to_target_s, time_margin_s=time_margin_s,
                seed_rad=seed, solver=solver, calibration=Calibration(court_from_base=_EYE,
                                                                     verified=True),
                position_sigma_m=scenario.position_sigma_m)
            entry = label(backend, scenario, verdict, seed)
            labelled.append(entry)
            decided[verdict.decision.value] += 1
            scenarios_out.append(entry.as_dict())
        single_backend_id(*[entry.backend for entry in labelled])   # one backend per section
        report["backends"][str(name)] = {
            "backend": str(name), "status": "OK",
            "provenance": provenance.as_dict(),
            "limits": {"dof": limits.dof, "joint_names": list(limits.joint_names),
                       "joint_margin_rad": limits.joint_margin_rad,
                       "workspace_min_m": list(limits.workspace_min_m),
                       "workspace_max_m": list(limits.workspace_max_m)},
            "decisions": decided,
            "verdicts": scenarios_out,
        }
    return report


_EYE = ((1.0, 0.0, 0.0, 0.0), (0.0, 1.0, 0.0, 0.0), (0.0, 0.0, 1.0, 0.0),
        (0.0, 0.0, 0.0, 1.0))


def format_table(report: Mapping[str, Any]) -> str:
    """A markdown table: one row per (backend, scenario), every cell labelled."""
    lines = ["| backend | frame | scenario | decision | reason | margin rad |",
             "|---|---|---|---|---|---|"]
    for backend_id, section in report.get("backends", {}).items():
        if section.get("status") != "OK":
            lines.append("| %s | UNKNOWN | - | UNAVAILABLE | %s | - |"
                         % (backend_id, section.get("refusal", "")[:60]))
            continue
        frame = section["provenance"]["frame_name"]
        for verdict in section["verdicts"]:
            margin = verdict.get("joint_margin_rad")
            lines.append("| %s | %s | %s | %s | %s | %s |"
                         % (backend_id, frame, verdict["scenario"], verdict["decision"],
                            verdict["reason"],
                            "null" if margin is None else "%.4f" % float(margin)))
    return "\n".join(lines)
