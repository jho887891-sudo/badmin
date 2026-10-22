"""Tests for the R10-C3 assertion-provenance registry."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from tools.queue import assertion_registry as ar

REPO = Path(__file__).resolve().parents[2]
REGISTRY = REPO / ar.REGISTRY_PATH


def load():
    return json.loads(REGISTRY.read_text(encoding="utf-8"))


def check_payload(tmp_path, payload):
    path = tmp_path / "registry.json"
    path.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
    return ar.check_registry(REPO, path)


def entry(**overrides):
    base = {
        "id": "AS-T-01",
        "task": "P0-07",
        "status": "ASSERTED",
        "claim": "a claim",
        "source": {"file": "tools/queue/evidence_contract.py", "line": 128,
                   "quote": "def classify_run(run_dir: Path):"},
        "spec_clause": "C1",
        "unit": "boolean (契约判定)",
        "evidence": {"trace_path": "tests/fixtures/r10c/good/F1_r01"},
        "threshold": {"value": "violations == 0", "unit": "violation count",
                      "pre_registered": True, "registered_commit": "14f339f"},
    }
    base.update(overrides)
    return base


def test_the_committed_registry_satisfies_the_contract():
    assert ar.check_registry(REPO, REGISTRY) == []


def test_registry_covers_the_new_p0_and_p5_assertions():
    payload = load()
    tasks = {e["task"] for e in payload["entries"]}
    assert {"P0-07", "P5-07", "P5-08"} <= tasks
    assert any(e["status"] == "ASSERTED" for e in payload["entries"])
    assert any(e["status"] == "UNKNOWN" for e in payload["entries"])


def test_asserted_entry_without_a_line_fails(tmp_path):
    payload = {"entries": [entry(source={"file": "tools/queue/evidence_contract.py"})]}
    problems = check_payload(tmp_path, payload)
    assert any(p.code == "ASSERTED_SOURCE_INVALID" for p in problems)


def test_asserted_entry_without_a_spec_clause_fails(tmp_path):
    payload = {"entries": [entry(spec_clause=None)]}
    assert any(p.code == "SPEC_CLAUSE_INVALID" for p in check_payload(tmp_path, payload))


def test_asserted_entry_without_a_unit_fails(tmp_path):
    payload = {"entries": [entry(unit=None)]}
    assert any(p.code == "ASSERTED_UNIT_MISSING" for p in check_payload(tmp_path, payload))


def test_asserted_entry_without_evidence_fails(tmp_path):
    payload = {"entries": [entry(evidence={})]}
    assert any(p.code == "ASSERTED_EVIDENCE_INVALID" for p in check_payload(tmp_path, payload))


def test_asserted_entry_whose_trace_path_is_missing_fails(tmp_path):
    payload = {"entries": [entry(evidence={"trace_path": "tests/fixtures/r10c/ghost"})]}
    assert any(p.code == "ASSERTED_EVIDENCE_INVALID" for p in check_payload(tmp_path, payload))


def test_asserted_entry_without_a_preregistered_threshold_fails(tmp_path):
    threshold = {"value": 1, "unit": "N", "pre_registered": False, "registered_commit": "14f339f"}
    payload = {"entries": [entry(threshold=threshold)]}
    assert any(p.code == "THRESHOLD_NOT_PREREGISTERED"
               for p in check_payload(tmp_path, payload))


def test_asserted_threshold_needs_a_value_and_unit(tmp_path):
    payload = {"entries": [entry(threshold={"value": None, "unit": "",
                                            "pre_registered": True,
                                            "registered_commit": "14f339f"})]}
    assert any(p.code == "ASSERTED_THRESHOLD_INVALID"
               for p in check_payload(tmp_path, payload))


def test_quote_must_really_be_on_the_declared_line(tmp_path):
    payload = {"entries": [entry(source={"file": "tools/queue/evidence_contract.py",
                                         "line": 128, "quote": "def check_file_index"})]}
    assert any(p.code == "ASSERTED_QUOTE_MISMATCH"
               for p in check_payload(tmp_path, payload))


def test_line_beyond_the_file_fails(tmp_path):
    payload = {"entries": [entry(source={"file": "tools/queue/evidence_contract.py",
                                         "line": 99999, "quote": "x"})]}
    assert any(p.code == "ASSERTED_SOURCE_INVALID" for p in check_payload(tmp_path, payload))


def test_unknown_entry_carrying_a_number_fails(tmp_path):
    payload = {"entries": [entry(status="UNKNOWN", unit=None, threshold=None, evidence=None,
                                 claim="命中率约 6% 的漂移")]}
    assert any(p.code == "UNKNOWN_CARRIES_NUMBER"
               for p in check_payload(tmp_path, payload))


def test_unknown_entry_carrying_a_value_fails(tmp_path):
    payload = {"entries": [entry(status="UNKNOWN", claim="未知量",
                                 evidence=None, threshold=None, unit="N")]}
    assert any(p.code == "UNKNOWN_CARRIES_VALUE"
               for p in check_payload(tmp_path, payload))


def test_unknown_entry_may_still_name_task_and_clause_ids(tmp_path):
    """Task ids like P5-07 are identifiers, not measurements."""
    payload = {"entries": [entry(status="UNKNOWN", unit=None, threshold=None, evidence=None,
                                 claim="P5-07 的 N 环境真实 Run 数未知",
                                 spec_clause="B2")]}
    assert check_payload(tmp_path, payload) == []


def test_duplicate_ids_fail(tmp_path):
    payload = {"entries": [entry(), entry()]}
    assert any(p.code == "ID_DUPLICATE" for p in check_payload(tmp_path, payload))


def test_repository_task_ids_are_stripped_before_the_number_check():
    assert ar.has_number("P5-07 的 Run 数未知") is False
    assert ar.has_number("约 6% 偏差") is True


def test_cli_exit_codes(tmp_path, capsys):
    assert ar.main(["--repo", str(REPO)]) == 0
    assert ar.main(["--repo", str(REPO), "--registry", str(tmp_path / "missing.json")]) == 3
    broken = tmp_path / "broken.json"
    broken.write_text(json.dumps({"entries": [entry(unit=None)]}), encoding="utf-8")
    assert ar.main(["--repo", str(REPO), "--registry", str(broken)]) == 1
