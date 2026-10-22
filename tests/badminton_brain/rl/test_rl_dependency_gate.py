# -*- coding: utf-8 -*-
"""P8-06 tests: the algorithm dependency gate REFUSES PPO today -- and that is the point.

The first test is the deliverable: with the repository as it stands, require_satisfied() raises
and names every missing P5/P6/P7 requirement.  The remaining tests prove the gate is a real
check (it CAN pass with evidence) and that no authorization flag, no caller-supplied status map
and no missing file can make it pass without evidence.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "src"))

from badminton_brain.rl.dependency_gate import (  # noqa: E402
    DEFAULT_REQUIREMENTS, PPO_TRAINING_STATUS, STATUS_MISSING, STATUS_SATISFIED, STATUS_UNKNOWN,
    AlgorithmDependencyGate, DependencyGateError, EvidenceContradictionError, Requirement,
    RequirementMissingError, default_gate, format_table,
)

EXPECTED_IDS = ("P5-04", "P5-05", "P5-06", "P5-07", "P5-09", "P6-02", "P6-05", "P6-09",
                "P7-09", "P7-10", "P7-12")


def test_the_gate_refuses_ppo_today_and_names_every_missing_requirement():
    gate = default_gate()
    assert gate.allows_ppo() is False
    assert PPO_TRAINING_STATUS == "NOT_AUTHORIZED"
    with pytest.raises(RequirementMissingError) as excinfo:
        gate.require_satisfied()
    error = excinfo.value
    assert error.missing == EXPECTED_IDS
    message = str(error)
    assert "NOT AUTHORIZED" in message
    for requirement_id in EXPECTED_IDS:
        assert requirement_id in message
    assert "P5-05" in error.reasons and "P7-10" in error.reasons
    assert "MISSING_EVIDENCE" in error.reasons["P5-05"]


def test_every_declared_requirement_is_missing_evidence_on_this_branch():
    for row in default_gate().resolve_table(ROOT):
        assert row["status"] == STATUS_MISSING, row
        assert "unmerged" in row["reason"] or "absent" in row["reason"]


def test_the_gate_covers_all_three_upstream_gates():
    upstreams = {req.upstream for req in DEFAULT_REQUIREMENTS}
    assert upstreams == {"P5", "P6", "P7"}
    assert len(DEFAULT_REQUIREMENTS) == len(EXPECTED_IDS)


def test_a_requirement_declared_satisfied_without_evidence_is_a_contradiction():
    fabricated = Requirement(
        requirement_id="P5-05", upstream="P5", title="limits", needs_ppo="bounds",
        evidence_path="management/morph_one_rs/DOES_NOT_EXIST.md",
        evidence_branch="p5/controller-hold", evidence_commit="2afd6f2",
        declared_status=STATUS_SATISFIED)
    gate = AlgorithmDependencyGate(requirements=(fabricated,))
    with pytest.raises(EvidenceContradictionError):
        gate.require_satisfied(ROOT)


def test_an_unknown_status_word_is_refused():
    with pytest.raises(EvidenceContradictionError):
        Requirement(requirement_id="P5-05", upstream="P5", title="t", needs_ppo="n",
                    evidence_path="x", evidence_branch="b", evidence_commit="c",
                    declared_status="PROBABLY_FINE")


def test_an_upstream_gate_outside_p5_p6_p7_is_refused():
    with pytest.raises(EvidenceContradictionError):
        Requirement(requirement_id="P9-01", upstream="P9", title="t", needs_ppo="n",
                    evidence_path="x", evidence_branch="b", evidence_commit="c")


def test_an_artifact_without_a_recorded_acceptance_stays_unknown(tmp_path):
    (tmp_path / "evidence.md").write_text("we think this works\n", encoding="utf-8")
    uncertain = Requirement(
        requirement_id="P6-02", upstream="P6", title="frame tree", needs_ppo="one chain",
        evidence_path="evidence.md", evidence_branch="p6/racket-camera-frames",
        evidence_commit="d7672d5")
    status, reason = uncertain.resolve(tmp_path)
    assert status == STATUS_UNKNOWN
    assert "acceptance" in reason
    gate = AlgorithmDependencyGate(requirements=(uncertain,))
    with pytest.raises(RequirementMissingError) as excinfo:
        gate.require_satisfied(tmp_path)
    assert excinfo.value.missing == ("P6-02",)


def test_the_gate_can_pass_when_evidence_and_an_acceptance_are_both_recorded(tmp_path):
    (tmp_path / "evidence.md").write_text("P6-02 frame tree, gate run, exit code 0\n",
                                          encoding="utf-8")
    accepted = Requirement(
        requirement_id="P6-02", upstream="P6", title="frame tree", needs_ppo="one chain",
        evidence_path="evidence.md", evidence_branch="p6/racket-camera-frames",
        evidence_commit="d7672d5", declared_status=STATUS_SATISFIED,
        note="recorded after the P6 branch was merged and its command returned exit code 0")
    gate = AlgorithmDependencyGate(requirements=(accepted,))
    assert gate.allows_ppo(tmp_path) is True
    gate.require_satisfied(tmp_path)  # must not raise
    assert gate.summary(tmp_path)["counts"][STATUS_SATISFIED] == 1


def test_an_authorization_flag_does_not_bypass_the_gate():
    gate = AlgorithmDependencyGate(requirements=DEFAULT_REQUIREMENTS,
                                   declared_authorization="AUTHORIZED")
    assert gate.declared_authorization == "AUTHORIZED"
    with pytest.raises(RequirementMissingError):
        gate.require_satisfied(ROOT)


def test_the_table_is_human_readable_and_not_empty():
    text = format_table()
    for requirement_id in EXPECTED_IDS:
        assert requirement_id in text
    assert "MISSING_EVIDENCE" in text


def test_the_refusal_is_a_hard_error_not_a_warning():
    with pytest.raises(DependencyGateError):
        default_gate().require_satisfied()
