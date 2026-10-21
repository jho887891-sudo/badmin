"""A06: unique run ids, immutable evidence ledger, stale result.json fails closed."""

from __future__ import annotations

import json
import shutil
from pathlib import Path

import pytest

from tools.queue import run_evidence as a06


def make_run(base: Path, run_id: str, gate_result="PASS", gate_run_id=None,
             manifest_run_id=None):
    run_dir = base / run_id
    run_dir.mkdir(parents=True)
    (run_dir / "manifest.json").write_text(json.dumps({
        "run_id": manifest_run_id or run_id,
        "gravity_m_s2": [0, 0, -9.81],
        "command": "--run-id %s --gravity-z -9.81" % (manifest_run_id or run_id),
    }, indent=1), encoding="utf-8")
    (run_dir / "result.json").write_text(json.dumps({
        "run_id": gate_run_id or run_id,
        "gate_result": gate_result,
        "gates": {"all_finite": True, "settles": True},
    }, indent=1), encoding="utf-8")
    (run_dir / "raw_poses.jsonl").write_text('{"step": 0, "z": 0.334}\n', encoding="utf-8")
    (run_dir / "child.stdout.log").write_text("ok\n", encoding="utf-8")
    return run_dir


def codes(problems):
    return [problem["code"] for problem in problems]


# ------------------------------------------------------------------ contract 1
def test_clean_runs_directory_passes(tmp_path):
    make_run(tmp_path, "G5L_r01")
    make_run(tmp_path, "G5L_r02")
    runs, problems = a06.check_runs(tmp_path)
    assert problems == []
    assert sorted(runs) == ["G5L_r01", "G5L_r02"]
    assert runs["G5L_r01"]["gate_result"] == "PASS"


def test_case_only_duplicate_run_id_directories_fail(tmp_path):
    """Case-only duplicates: checked on names, so it also holds on a case-sensitive FS."""
    problems = a06.check_unique_run_ids([tmp_path / "G5L_r01", tmp_path / "G5L_R01"])
    assert "DUPLICATE_RUN_ID" in codes(problems)
    duplicate = [p for p in problems if p["code"] == "DUPLICATE_RUN_ID"][0]
    assert "G5L_r01" in duplicate["reason"] and "G5L_R01" in duplicate["reason"]


def test_two_directories_claiming_one_run_id_fail(tmp_path):
    make_run(tmp_path, "runA", gate_run_id="G5L_r01", manifest_run_id="G5L_r01")
    make_run(tmp_path, "runB", gate_run_id="G5L_r01", manifest_run_id="G5L_r01")
    _runs, problems = a06.check_runs(tmp_path)
    assert "DUPLICATE_RUN_ID" in codes(problems)


# ------------------------------------------------------------------ contract 3
def test_stale_inherited_result_fails_closed(tmp_path):
    """A real stale case: G5L_r01 holds the result.json of a different run."""
    source = make_run(tmp_path, "G5_r01")
    target = tmp_path / "G5L_r01"
    target.mkdir()
    shutil.copy(source / "manifest.json", target / "manifest.json")
    # fix the manifest run id, but inherit the result.json of G5_r01
    manifest = json.loads((target / "manifest.json").read_text(encoding="utf-8"))
    manifest["run_id"] = "G5L_r01"
    manifest["command"] = "--run-id G5L_r01 --gravity-z -9.81"
    (target / "manifest.json").write_text(json.dumps(manifest, indent=1), encoding="utf-8")
    shutil.copy(source / "result.json", target / "result.json")
    (target / "raw_poses.jsonl").write_text("{}\n", encoding="utf-8")

    _runs, problems = a06.check_runs(target.parent)
    stale = [p for p in problems if p["code"] == "STALE_RESULT"]
    assert stale, problems
    assert any("inherited from a different run" in p["reason"] for p in stale)
    assert any("'G5_r01'" in p["reason"] for p in stale)


def test_stale_result_against_expect_run_id(tmp_path):
    make_run(tmp_path, "G6A_r01")
    _runs, problems = a06.check_runs(tmp_path, expect_run_id="G6A_r02")
    assert "RUN_ID_MISMATCH" in codes(problems)


def test_result_manifest_digest_mismatch_is_stale(tmp_path):
    run_dir = make_run(tmp_path, "G7B_grav_r08")
    result = json.loads((run_dir / "result.json").read_text(encoding="utf-8"))
    result["manifest_sha256"] = "sha256:" + "0" * 64
    (run_dir / "result.json").write_text(json.dumps(result), encoding="utf-8")
    _runs, problems = a06.check_runs(tmp_path)
    assert "STALE_RESULT" in codes(problems)


def test_missing_result_is_not_a_pass(tmp_path):
    run_dir = make_run(tmp_path, "G5L_r01")
    (run_dir / "result.json").unlink()
    _runs, problems = a06.check_runs(tmp_path)
    assert "RESULT_MISSING" in codes(problems)


# ------------------------------------------------------------------ contract 2
def test_ledger_round_trip_is_green(tmp_path):
    make_run(tmp_path, "G5L_r01")
    make_run(tmp_path, "G5L_r02")
    ledger = a06.build_ledger(tmp_path)
    assert ledger["totals"]["runs"] == 2
    assert ledger["totals"]["files"] == 8
    assert a06.verify_ledger(tmp_path, ledger) == []


def test_overwritten_evidence_fails(tmp_path):
    run_dir = make_run(tmp_path, "G5L_r01")
    ledger = a06.build_ledger(tmp_path)
    (run_dir / "raw_poses.jsonl").write_text('{"step": 0, "z": 0.5}\n', encoding="utf-8")
    problems = a06.verify_ledger(tmp_path, ledger)
    assert "EVIDENCE_MODIFIED" in codes(problems)
    modified = [p for p in problems if p["code"] == "EVIDENCE_MODIFIED"][0]
    assert "raw_poses.jsonl" in modified["reason"]
    assert "overwritten" in modified["reason"]


def test_deleted_evidence_fails(tmp_path):
    run_dir = make_run(tmp_path, "G5L_r01")
    ledger = a06.build_ledger(tmp_path)
    (run_dir / "raw_poses.jsonl").unlink()
    assert "EVIDENCE_REMOVED" in codes(a06.verify_ledger(tmp_path, ledger))
    shutil.rmtree(run_dir)
    assert "EVIDENCE_REMOVED" in codes(a06.verify_ledger(tmp_path, ledger))


def test_new_evidence_after_the_freeze_fails(tmp_path):
    run_dir = make_run(tmp_path, "G5L_r01")
    ledger = a06.build_ledger(tmp_path)
    (run_dir / "notes.md").write_text("appended later\n", encoding="utf-8")
    problems = a06.verify_ledger(tmp_path, ledger)
    assert "EVIDENCE_UNLISTED" in codes(problems)
    # the escape hatch is explicit and never the default
    assert a06.verify_ledger(tmp_path, ledger, allow_unlisted=True) == []


def test_unlisted_run_directory_fails(tmp_path):
    make_run(tmp_path, "G5L_r01")
    ledger = a06.build_ledger(tmp_path)
    make_run(tmp_path, "G5L_r02")
    assert "EVIDENCE_UNLISTED" in codes(a06.verify_ledger(tmp_path, ledger))


# ------------------------------------------------------------------ CLI
def test_cli_ledger_and_check_exit_codes(tmp_path, capsys):
    run_dir = make_run(tmp_path, "G5L_r01")
    ledger_path = tmp_path / "ledger.json"
    assert a06.main(["ledger", "--runs-root", str(tmp_path), "--out", str(ledger_path)]) == 0
    assert "runs 1 files 4" in capsys.readouterr().out
    assert a06.main(["check", "--runs-root", str(tmp_path), "--ledger", str(ledger_path)]) == 0

    (run_dir / "result.json").write_text(json.dumps({"run_id": "G5_r01", "gate_result": "PASS"}),
                                         encoding="utf-8")
    assert a06.main(["check", "--runs-root", str(tmp_path), "--ledger", str(ledger_path)]) == 2
    captured = capsys.readouterr()
    assert "FAIL (A06_RUN_EVIDENCE)" in captured.err
    assert "EVIDENCE_MODIFIED" in captured.err
    assert "STALE_RESULT" in captured.err


def test_cli_missing_runs_root_exits_one(tmp_path, capsys):
    assert a06.main(["check", "--runs-root", str(tmp_path / "nope")]) == 1
    assert "RUNS_ROOT_MISSING" in capsys.readouterr().err
