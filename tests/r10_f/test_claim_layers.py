# -*- coding: utf-8 -*-
"""R10-F4 (P7-11/12): the three-layer claim split, and hit_rate stays NULL - or the test fails.

Run:
    python -m pytest tests/r10_f/test_claim_layers.py -q
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
for _entry in (ROOT / "src", ROOT / "simulation", ROOT):
    if str(_entry) not in sys.path:
        sys.path.insert(0, str(_entry))

from src.badminton_brain.harness import claim_layers as layers  # noqa: E402

EPISODES = (ROOT / "outputs" / "badminton_brain" / "p7_closed_loop" / "wired_episodes.json")
BACKEND_COMPARE = ROOT / "outputs" / "badminton_brain" / "p7_closed_loop" / "backend_compare.json"


def test_the_three_layers_are_distinct_and_ordered():
    assert layers.LAYER_ORDER == ("TRAJECTORY_REACHABLE", "GEOMETRIC_CONTACT", "PHYSICAL_HIT")
    assert len({layer.layer for layer in layers.LAYERS}) == 3
    statuses = {layer.layer: layer.status for layer in layers.LAYERS}
    assert statuses["TRAJECTORY_REACHABLE"] == layers.LayerStatus.SUPPORTED
    assert statuses["GEOMETRIC_CONTACT"] == layers.LayerStatus.PROXY_ONLY
    assert statuses["PHYSICAL_HIT"] == layers.LayerStatus.BLOCKED


def test_physical_hit_names_its_blockers_and_carries_no_evidence():
    physical = [layer for layer in layers.LAYERS if layer.layer == "PHYSICAL_HIT"][0]
    assert physical.evidence_paths == ()
    assert len(physical.blockers) >= 3
    joined = " ".join(physical.blockers).lower()
    assert "contact model" in joined
    assert "calibrat" in joined


def test_a_non_null_hit_rate_fails_the_test():
    with pytest.raises(layers.HitRateFabrication):
        layers.forbid_hit_rate(0.0)
    with pytest.raises(layers.HitRateFabrication):
        layers.forbid_hit_rate(0.62)
    layers.forbid_hit_rate(None)


def test_a_report_that_repairs_the_null_is_refused():
    fabricated = {"schema": "x", "episodes": [{"hit_rate": 0.5}]}
    with pytest.raises(layers.HitRateFabrication):
        layers.audit_report(fabricated, where="fabricated")
    aggregate = {"episodes": [{"hit_rate": None}], "summary": {"hit_rate": 0.31}}
    with pytest.raises(layers.HitRateFabrication):
        layers.audit_report(aggregate, where="aggregate")     # even ONE non-null anywhere
    claim = {"hit_rate": None, "real_hit_rate_claimed": True}
    with pytest.raises(layers.HitRateFabrication):
        layers.audit_report(claim, where="claim")
    with pytest.raises(layers.ClaimAuditError):
        layers.audit_report({"no_hit_rate_field": 1}, where="silent")


def test_the_committed_episode_report_keeps_every_hit_rate_null():
    if not EPISODES.is_file():
        pytest.skip("committed P7 episode report not present in this worktree")
    report = json.loads(EPISODES.read_text(encoding="utf-8"))
    audit = layers.audit_report(report, where="wired_episodes.json")
    assert audit.hit_rate_paths                       # the nulls are present, not absent
    assert audit.real_hit_rate_claimed is False
    assert "PHYSICAL_HIT" not in audit.layers_supported


def test_the_three_layer_report_is_null_and_blocked():
    audit = None
    if EPISODES.is_file():
        audit = layers.audit_report(json.loads(EPISODES.read_text(encoding="utf-8")),
                                    where="wired_episodes.json")
    claim = layers.three_layer_claim(audited=[audit] if audit else ())
    assert claim["hit_rate"] is None
    assert claim["real_hit_rate_claimed"] is False
    statuses = {layer["layer"]: layer["status"] for layer in claim["layers"]}
    assert statuses["PHYSICAL_HIT"] == layers.LayerStatus.BLOCKED
    layers.forbid_hit_rate(claim["hit_rate"])


def test_a_physical_hit_assertion_without_a_contact_model_is_refused():
    with pytest.raises(layers.ClaimAuditError):
        layers.audit_report({"hit_rate": None, "physical_hit_verified": True},
                            where="overclaim")
