# -*- coding: utf-8 -*-
"""R10-F4 (P7-11/12): split the closed-loop claim into THREE separately labelled layers.

WHY THE SPLIT
    The P7 harness is simulation-only.  It is easy, and wrong, to read "the episode ran end to
    end" as "the robot returns a shuttle".  That single sentence hides three claims with three
    completely different evidence requirements:

      TRAJECTORY_REACHABLE   IK converged, joint-limit margin respected, the setpoint passed the
                             P5-05 caps.  Evidence: the FSM event log + the labelled reachability
                             verdict.  This layer IS supported by committed runs.
      GEOMETRIC_CONTACT      the racket proxy surface and the ball sphere came within a surface
                             distance, i.e. the PLAN puts the tool near the ball.  Evidence: the
                             PROXY geometry only (racket_cfg.py + a sphere).  Supported as a
                             PROXY statistic, never as a hit.
      PHYSICAL_HIT           the racket actually imparted an impulse and the shuttle actually
                             left on a return trajectory.  Requires a CONTACT MODEL plus a
                             measured/calibrated TCP and racket face, neither of which exists
                             here.  Status: BLOCKED/IMPOSSIBLE today.

    The layers are NOT a progression that a single run upgrades.  A run can complete layers 1
    and 2 with enormous margin on layer 3; layer 3 stays BLOCKED because no contact model exists.

hit_rate IS NULL BY CONTRACT
    hit_rate has no referent without layer 3.  This module therefore REFUSES a non-null hit rate
    rather than merely documenting that it should be null: audit_report() raises
    HitRateFabrication.  A report that repairs the null (0.0, an "estimated" 0.62, a proxy
    contact rate copied into the field) fails the test that guards this file.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Mapping, Optional, Sequence

__all__ = [
    "LAYER_TRAJECTORY_REACHABLE", "LAYER_GEOMETRIC_CONTACT", "LAYER_PHYSICAL_HIT",
    "LAYER_ORDER", "LAYERS", "LayerStatus", "ClaimLayer", "ClaimAuditError",
    "HitRateFabrication", "audit_report", "three_layer_claim", "HIT_RATE_NULL_NOTE",
]

LAYER_TRAJECTORY_REACHABLE = "TRAJECTORY_REACHABLE"
LAYER_GEOMETRIC_CONTACT = "GEOMETRIC_CONTACT"
LAYER_PHYSICAL_HIT = "PHYSICAL_HIT"

LAYER_ORDER: tuple = (LAYER_TRAJECTORY_REACHABLE, LAYER_GEOMETRIC_CONTACT, LAYER_PHYSICAL_HIT)

HIT_RATE_NULL_NOTE = (
    "hit_rate is None by contract: without a calibrated frame, a measured contact model and a "
    "physics engine there is no referent for a hit rate. A proxy surface-distance statistic is "
    "not a hit and must never be written into this field."
)


class LayerStatus:
    SUPPORTED = "SUPPORTED"
    PROXY_ONLY = "PROXY_ONLY"
    BLOCKED = "BLOCKED"
    NOT_EVALUATED = "NOT_EVALUATED"


class ClaimAuditError(RuntimeError):
    """The report does not support the claim it is being read as."""


class HitRateFabrication(ClaimAuditError):
    """A non-null hit rate was reported.  There is no evidence layer that could produce it."""

    def __init__(self, value: object, where: str) -> None:
        super().__init__(
            "%s reports hit_rate=%r. %s" % (where, value, HIT_RATE_NULL_NOTE))
        self.value = value
        self.where = where


@dataclass(frozen=True)
class ClaimLayer:
    """One layer: what it claims, what it needs, and what this repository can show for it."""

    layer: str
    claim: str
    required_evidence: Sequence[str]
    status: str
    blockers: Sequence[str] = ()
    evidence_paths: Sequence[str] = ()

    def as_dict(self) -> Dict[str, Any]:
        return {"layer": self.layer, "claim": self.claim,
                "required_evidence": list(self.required_evidence), "status": self.status,
                "blockers": list(self.blockers),
                "evidence_paths": list(self.evidence_paths)}


LAYERS: tuple = (
    ClaimLayer(
        layer=LAYER_TRAJECTORY_REACHABLE,
        claim="the arm can be COMMANDED to the planned joint vector: IK converged on the "
              "declared backend, the joint-limit margin holds, and the setpoint passed the "
              "P5-05 position/rate caps",
        required_evidence=("labelled reachability verdict (backend + frame + provenance)",
                           "FSM event log showing PLAN and the envelope acceptance",
                           "joint vector within the model limit table with margin"),
        status=LayerStatus.SUPPORTED,
        blockers=(),
        evidence_paths=("outputs/badminton_brain/p7_closed_loop/backend_compare.json",
                        "outputs/badminton_brain/p7_closed_loop/wired_episodes.json"),
    ),
    ClaimLayer(
        layer=LAYER_GEOMETRIC_CONTACT,
        claim="the racket PROXY surface passes within a declared distance of the ball sphere",
        required_evidence=("racket collision proxy (P6 racket_collision_proxy.json)",
                           "ball sphere radius",
                           "surface distance statistic per episode"),
        status=LayerStatus.PROXY_ONLY,
        blockers=("the racket collision proxy is a simplified mesh, not the real racket",
                  "the ball is a sphere in a kinematic replay, not a simulated body",
                  "no swept volume, no relative velocity at closest approach"),
        evidence_paths=("outputs/badminton_brain/p7_closed_loop/wired_episodes.json",),
    ),
    ClaimLayer(
        layer=LAYER_PHYSICAL_HIT,
        claim="the racket imparts an impulse and the shuttle leaves on a return trajectory",
        required_evidence=("a CONTACT MODEL (friction/restitution, string-bed compliance)",
                           "a calibrated TCP and racket FACE frame",
                           "a physics engine stepping the shuttle as a body",
                           "measured or simulated post-impact shuttle state"),
        status=LayerStatus.BLOCKED,
        blockers=("no contact model for the racket/string bed in this repository",
                  "the racket FACE frame is UNKNOWN in the P6 frame contract (edges 13/14/15)",
                  "no calibrated D455 extrinsic / no measured TCP",
                  "no physics engine in the closed-loop path (simulation-only replay)"),
        evidence_paths=(),
    ),
)


def forbid_hit_rate(value: object, *, where: str = "a closed-loop report") -> None:
    """Raise if a hit rate was reported.  None is the only accepted value."""
    if value is not None:
        raise HitRateFabrication(value, where)


def _walk_hit_rates(node: object, path: str) -> List[tuple]:
    """Every (path, value) pair that names a hit rate, at any depth."""
    found: List[tuple] = []
    if isinstance(node, Mapping):
        for key, value in node.items():
            key_path = "%s.%s" % (path, key) if path else str(key)
            if str(key) == "hit_rate":
                found.append((key_path, value))
            else:
                found.extend(_walk_hit_rates(value, key_path))
    elif isinstance(node, (list, tuple)):
        for index, value in enumerate(node):
            found.extend(_walk_hit_rates(value, "%s[%d]" % (path, index)))
    return found


@dataclass(frozen=True)
class ClaimAudit:
    """The result of auditing one closed-loop report against the three layers."""

    where: str
    hit_rate_paths: Sequence[str]
    real_hit_rate_claimed: bool
    layers: Sequence[ClaimLayer]
    warnings: Sequence[str] = ()

    @property
    def layers_supported(self) -> Sequence[str]:
        return [layer.layer for layer in self.layers if layer.status == LayerStatus.SUPPORTED]

    def as_dict(self) -> Dict[str, Any]:
        return {"where": self.where, "hit_rate_paths": list(self.hit_rate_paths),
                "hit_rate": None, "real_hit_rate_claimed": self.real_hit_rate_claimed,
                "layers": [layer.as_dict() for layer in self.layers],
                "warnings": list(self.warnings),
                "note": HIT_RATE_NULL_NOTE}


def audit_report(report: Mapping[str, Any], *, where: str = "report") -> ClaimAudit:
    """Audit a closed-loop report: hit_rate must be null, and no layer may be upgraded.

    Raises HitRateFabrication for ANY non-null hit_rate, at any depth - including an aggregate
    rate that some summary layer added above the per-episode nulls.
    """
    if not isinstance(report, Mapping):
        raise ClaimAuditError("%s is not a mapping (%r)" % (where, type(report).__name__))
    found = _walk_hit_rates(report, "")
    if not found:
        raise ClaimAuditError(
            "%s carries no hit_rate field at all: the null is part of the contract, and its "
            "absence makes 'null' indistinguishable from 'forgotten'" % where)
    for path, value in found:
        forbid_hit_rate(value, where="%s (%s)" % (where, path))
    claimed = report.get("real_hit_rate_claimed") is True
    if claimed:
        raise HitRateFabrication("real_hit_rate_claimed=True",
                                 "%s (a claim without a hit rate is still a claim)" % where)
    warnings: List[str] = []
    if "PHYSICAL_HIT" in str(report).upper() and report.get("physical_hit_verified") is True:
        raise ClaimAuditError(
            "%s asserts physical_hit_verified=True; no contact model exists in this repository"
            % where)
    return ClaimAudit(where=where, hit_rate_paths=tuple(path for path, _ in found),
                      real_hit_rate_claimed=claimed, layers=LAYERS, warnings=tuple(warnings))


def three_layer_claim(*, audited: Sequence[ClaimAudit] = ()) -> Dict[str, Any]:
    """The three layers as a reportable structure.  PHYSICAL_HIT is BLOCKED, always."""
    return {
        "schema": "badmin.r10f.claim_layers",
        "schema_version": 1,
        "note": "three separately labelled claims; completing one does not promote another. "
                "PHYSICAL_HIT is BLOCKED because no contact model and no calibrated racket face "
                "exist in this repository.",
        "hit_rate": None,
        "real_hit_rate_claimed": False,
        "layers": [layer.as_dict() for layer in LAYERS],
        "audited_reports": [audit.as_dict() for audit in audited],
    }
