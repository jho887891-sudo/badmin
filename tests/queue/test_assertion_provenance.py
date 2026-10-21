"""A09: assertion provenance registry, loader, source verification and generator."""

from __future__ import annotations

import json
import subprocess
from pathlib import Path

import pytest

from tools.queue import assertion_provenance as a09

REGISTRY = a09.DEFAULT_REGISTRY
REPO = Path(__file__).resolve().parents[2]

#: the gates the task names, seeded from the real P2/P3 evidence
REAL_GATES = (
    "contact_detected",
    "settles",
    "no_tip_over",
    "no_long_term_penetration",
    "static_weight_supported",
    "mount_stable_under_load",
    "arm_does_not_touch_ground",
    "t0_world_position_within_tol",
    "t0_world_rotation_within_tol",
    "settle_drift_within_tol",
    "max_drift_within_tol",
)


def ref_available(ref: str, path: str) -> bool:
    try:
        return subprocess.run(["git", "cat-file", "-e", "%s:%s" % (ref, path)],
                              cwd=str(REPO), capture_output=True).returncode == 0
    except OSError:
        return False


# ------------------------------------------------------------------ registry
def test_registry_loads_and_covers_every_real_gate():
    registry = a09.ProvenanceRegistry.load(REGISTRY)
    assert set(REAL_GATES) <= set(registry.gates())
    for gate in REAL_GATES:
        entry = registry.entry(gate)
        assert entry["unit"], gate
        assert entry["data_file"], gate
        assert entry["acceptance"], gate
        assert entry["acceptance_decision"] == "pass_when_true", gate
        assert entry["sources"] and entry["spec_clauses"], gate
        for row in entry["sources"] + entry["spec_clauses"]:
            assert isinstance(row["line"], int) and row["line"] > 0, gate
            assert row["path"].endswith(".py") or row["path"].endswith(".md"), gate


def test_registry_cites_the_real_spec_documents():
    registry = a09.ProvenanceRegistry.load(REGISTRY)
    cited = set()
    for gate in REAL_GATES:
        for row in registry.entry(gate)["spec_clauses"]:
            cited.add(row["path"])
    assert "management/morph_one_rs/PHASE_P2_G5_LANDING_GATE.md" in cited
    assert "management/morph_one_rs/PHASE_P2_WC_G6_PREREGISTRATION.md" in cited
    assert "management/morph_one_rs/PHASE_P3_DYNAMIC_MOUNT_GATE.md" in cited


def test_registry_entries_are_structurally_complete():
    registry = a09.ProvenanceRegistry.load(REGISTRY)
    assert registry.verify_entries() == []


def test_incomplete_entry_is_reported():
    registry = a09.ProvenanceRegistry({"entries": {"contact_detected": {
        "unit": "N", "sources": [{"path": "a.py", "line": 1}], "spec_clauses": []}}})
    problems = registry.verify_entries()
    assert {p["code"] for p in problems} >= {"ENTRY_INCOMPLETE"}
    missing = {p["reason"] for p in problems}
    assert any("data_file" in reason for reason in missing)


@pytest.mark.skipif(not ref_available("dev/p2-wheel-contact-g1g4",
                                      "management/morph_one_rs/PHASE_P2_G5_LANDING_GATE.md"),
                    reason="P2 gate ref not present in this clone")
def test_every_cited_source_line_really_exists():
    """Real self-check: every cited file+line+anchor is verified against the repo."""
    registry = a09.ProvenanceRegistry.load(REGISTRY)
    problems = registry.verify_sources()
    assert problems == [], problems


def test_anchor_mismatch_is_reported(tmp_path):
    fake = tmp_path / "gate.py"
    fake.write_text("line one\nline two\n", encoding="utf-8")
    registry = a09.ProvenanceRegistry({"entries": {"contact_detected": {
        "unit": "N", "data_file": "raw_poses.jsonl", "acceptance": "peak > 1 N",
        "acceptance_decision": "pass_when_true",
        "sources": [{"path": fake.name, "line": 1, "line_contains": "not_here"}],
        "spec_clauses": [{"path": fake.name, "line": 2,
                          "line_contains": "line two"}]}}})
    resolver = a09.SourceResolver(repo_root=tmp_path)
    problems = registry.verify_sources(resolver=resolver)
    assert [p["code"] for p in problems] == ["SOURCE_ANCHOR_MISMATCH"]
    assert "no longer contains" in problems[0]["reason"]


def test_out_of_range_line_is_reported(tmp_path):
    fake = tmp_path / "gate.py"
    fake.write_text("only one line\n", encoding="utf-8")
    registry = a09.ProvenanceRegistry({"entries": {"settles": {
        "unit": "m/s", "data_file": "raw_poses.jsonl", "acceptance": "x",
        "acceptance_decision": "pass_when_true",
        "sources": [{"path": fake.name, "line": 99}],
        "spec_clauses": [{"path": fake.name, "line": 1}]}}})
    problems = registry.verify_sources(resolver=a09.SourceResolver(repo_root=tmp_path))
    assert [p["code"] for p in problems] == ["SOURCE_LINE_OUT_OF_RANGE"]


def test_unresolvable_source_is_reported(tmp_path):
    registry = a09.ProvenanceRegistry({"entries": {"settles": {
        "unit": "m/s", "data_file": "raw_poses.jsonl", "acceptance": "x",
        "acceptance_decision": "pass_when_true",
        "sources": [{"path": "does_not_exist.py", "line": 1, "ref": "deadbeef"}],
        "spec_clauses": [{"path": "also_missing.md", "line": 1}]}}})
    problems = registry.verify_sources(resolver=a09.SourceResolver(repo_root=tmp_path,
                                                                  runner=lambda r, p: None))
    assert {"SOURCE_UNRESOLVED"} <= {p["code"] for p in problems}


# ------------------------------------------------------------------ generator
def real_result_gates():
    return {gate: True for gate in REAL_GATES} | {"all_finite": True}


def test_every_real_gate_is_registered_and_generates():
    registry = a09.ProvenanceRegistry.load(REGISTRY)
    payload, problems = a09.generate({"run_id": "G7B_grav_r08", "gate_result": "PASS",
                                      "gates": real_result_gates()}, registry)
    assert problems == []
    assert payload["totals"]["observed"] == len(REAL_GATES) + 1
    assert payload["totals"]["registered"] == len(REAL_GATES) + 1
    row = [r for r in payload["gates"] if r["gate"] == "contact_detected"][0]
    assert row["unit"] == "N"
    assert row["recorded_decision"] is True
    assert row["sources"][0]["path"] == "scripts/simulation/p2_landing_gates.py"


def test_unregistered_gate_fails():
    registry = a09.ProvenanceRegistry.load(REGISTRY)
    payload, problems = a09.generate({"gates": {"contact_detected": True,
                                                "made_up_gate": True}}, registry)
    assert [p["code"] for p in problems] == ["UNREGISTERED_GATE"]
    assert problems[0]["gate"] == "made_up_gate"
    assert payload["totals"]["registered"] == 1


def test_failing_gate_is_not_a_provenance_problem():
    """G5_r01 legitimately recorded False; that is a FAIL, not a tracing defect."""
    registry = a09.ProvenanceRegistry.load(REGISTRY)
    payload, problems = a09.generate({"run_id": "G5_r01", "gate_result": "FAIL",
                                      "gates": {"t0_world_position_within_tol": False,
                                                "settle_drift_within_tol": False}}, registry)
    assert problems == []
    assert payload["gates"][0]["recorded_decision"] is False


def test_non_boolean_gate_fails_closed():
    registry = a09.ProvenanceRegistry.load(REGISTRY)
    _payload, problems = a09.generate({"gates": {"settles": None}}, registry)
    assert [p["code"] for p in problems] == ["NONBOOLEAN_GATE"]


def test_diagnostics_gates_are_scanned_too():
    registry = a09.ProvenanceRegistry.load(REGISTRY)
    payload = a09.generate({"gates": {"settles": True},
                            "diagnostics": {"static_gates_as_diagnostics":
                                            {"settle_drift_within_tol": False}}}, registry)
    gates = {row["gate"] for row in payload[0]["gates"]}
    # a demoted static gate keeps its label but resolves to the same registry entry
    assert gates == {"settles", "diagnostics.settle_drift_within_tol"}
    assert payload[1] == []
    row = [r for r in payload[0]["gates"]
           if r["gate"] == "diagnostics.settle_drift_within_tol"][0]
    assert row["recorded_decision"] is False
    assert row["sources"][0]["path"] == "scripts/simulation/morph_one_p2_real_chassis.py"


def test_unknown_polarity_fails():
    registry = a09.ProvenanceRegistry({"entries": {"settles": {
        "unit": "m/s", "data_file": "raw_poses.jsonl", "acceptance": "x",
        "acceptance_decision": "whatever", "sources": [{"path": "a.py", "line": 1}],
        "spec_clauses": [{"path": "a.md", "line": 1}]}}})
    _payload, problems = a09.generate({"gates": {"settles": True}}, registry)
    assert [p["code"] for p in problems] == ["DECISION_POLARITY_UNKNOWN"]


# ------------------------------------------------------------------ CLI
def test_cli_self_check_passes(capsys):
    assert a09.main(["--self-check"]) == 0
    assert "PASS (A09_REGISTRY)" in capsys.readouterr().out


def test_cli_unregistered_gate_exits_two(tmp_path, capsys):
    result = tmp_path / "result.json"
    result.write_text(json.dumps({"gates": {"settles": True, "invented_gate": True}}),
                      encoding="utf-8")
    assert a09.main(["--result-json", str(result)]) == 2
    captured = capsys.readouterr()
    assert "FAIL (A09_PROVENANCE)" in captured.err
    assert "UNREGISTERED_GATE" in captured.err


def test_cli_registered_result_exits_zero_with_artifact(tmp_path, capsys):
    result = tmp_path / "result.json"
    result.write_text(json.dumps({"run_id": "G6A_r01", "gates": real_result_gates()}),
                      encoding="utf-8")
    out = tmp_path / "provenance.json"
    assert a09.main(["--result-json", str(result), "--out", str(out)]) == 0
    assert "PASS (A09_PROVENANCE)" in capsys.readouterr().out
    payload = json.loads(out.read_text(encoding="utf-8"))
    assert payload["totals"]["problems"] == 0
    assert len(payload["gates"]) == len(REAL_GATES) + 1


def test_cli_list_prints_gates(capsys):
    assert a09.main(["--list"]) == 0
    assert "contact_detected" in capsys.readouterr().out
