"""Tests for the R10-C1/C2 typed evidence contracts."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from tools.queue import evidence_contract as ec

FIXTURES = Path(__file__).resolve().parents[1] / "fixtures" / "r10c"


def codes(violations):
    return sorted({v.code for v in violations})


def fixture(*parts) -> Path:
    return FIXTURES.joinpath(*parts)


# --------------------------------------------------------------------- C1 both ways


def test_fault_run_without_result_json_satisfies_its_contract():
    """A declared fault run legitimately has no result.json: it must PASS."""
    violations = ec.verify_run(fixture("good", "F1_r01"))
    assert violations == [], codes(violations)
    assert not (fixture("good", "F1_r01") / "result.json").exists()


def test_normal_run_with_full_set_satisfies_its_contract():
    assert ec.verify_run(fixture("good", "N1_r01")) == []


def test_ordinary_run_missing_result_json_still_fails():
    """Missing result.json must never be read as proof of a fault."""
    violations = ec.verify_run(fixture("bad", "B1_r01_no_result"))
    assert "KIND_UNDECLARED_UNPROVEN" in codes(violations)


def test_fault_run_missing_its_required_files_fails():
    violations = ec.verify_run(fixture("bad", "B2_r01_fault_no_log"))
    assert {"FAULT_NO_RAW_LOG", "RAW_LOG_MISSING"} <= set(codes(violations))


def test_a_run_claiming_both_kinds_is_a_contradiction():
    violations = ec.verify_run(fixture("bad", "B3_r01_kind_conflict"))
    assert "KIND_CONFLICT" in codes(violations)


def test_fault_record_needs_a_typed_code():
    violations = ec.verify_run(fixture("bad", "B4_r01_untyped_fault"))
    assert "FAULT_CODE_NOT_TYPED" in codes(violations)


def test_fault_classification_never_comes_from_a_missing_file(tmp_path):
    kind, _ = ec.classify_run(fixture("bad", "B1_r01_no_result"))
    assert kind == "UNDECLARED"
    kind, _ = ec.classify_run(fixture("good", "F1_r01"))
    assert kind == "FAULT"


# --------------------------------------------------------------------- C2 integrity


def test_parameter_hash_is_recomputed_from_the_manifest_config():
    violations = ec.verify_run(fixture("bad", "B5_r01_param_hash"))
    assert "PARAM_HASH_MISMATCH" in codes(violations)


def test_environment_fingerprint_is_required():
    violations = ec.verify_run(fixture("bad", "B6_r01_no_env"))
    assert "ENV_FINGERPRINT_MISSING" in codes(violations)


def test_tampered_file_breaks_sha256_integrity():
    violations = ec.verify_run(fixture("bad", "B7_r01_sha_tamper"))
    assert "SHA256_MISMATCH" in codes(violations)


def test_stale_file_outside_the_index_fails():
    violations = ec.verify_run(fixture("bad", "B8_r01_stale"))
    assert "STALE_UNINDEXED_FILE" in codes(violations)


def test_artifact_from_another_run_fails():
    violations = ec.verify_run(fixture("bad", "B9_r01_cross_run"))
    assert "CROSS_RUN_REFERENCE" in codes(violations)


def test_duplicate_run_id_across_evidence_fails():
    violations = ec.verify_set([
        fixture("bad_dup", "a", "DUP_r01"),
        fixture("bad_dup", "b", "DUP_r01"),
    ])
    assert "RUN_ID_DUPLICATE" in codes(violations)


def test_raw_log_outside_the_run_directory_fails():
    violations = ec.verify_run(fixture("bad", "B11_r01_log_outside"))
    assert "LOG_OUTSIDE_RUN" in codes(violations)


def test_parameter_hash_matches_the_shared_helper():
    manifest = json.loads((fixture("good", "N1_r01") / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["config_sha256"] == ec.parameter_hash(manifest["config"])


# ------------------------------------------------------------------------- CLI


def test_cli_zero_on_a_satisfied_fixture_set(capsys):
    assert ec.main(["--root", str(fixture("good"))]) == 0
    assert "satisfied" in capsys.readouterr().out


def test_cli_one_on_violations(capsys):
    assert ec.main(["--root", str(fixture("bad"))]) == 1
    assert "VIOLATION" in capsys.readouterr().out


def test_cli_three_when_the_input_is_unusable(tmp_path, capsys):
    assert ec.main(["--run-dir", str(tmp_path / "nope")]) == 3
    assert "error:" in capsys.readouterr().err
    assert ec.main([]) == 3


def test_json_report_is_machine_readable(capsys):
    assert ec.main(["--root", str(fixture("good")), "--json"]) == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["ok"] is True
    assert payload["contracts"]["FAULT"] == ["fault_record.json", "gate_runner_summary.json"]
