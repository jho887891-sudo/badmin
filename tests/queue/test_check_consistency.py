"""Tests for the A03 queue/repo consistency gate."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from tools.queue import check_consistency as cc
from tools.queue import queue_state as qs

REPO = Path(__file__).resolve().parents[2]
FULL_SHA = "0123456789abcdef0123456789abcdef01234567"


def make_task(task_id, status="UNKNOWN", deps=(), **fields):
    entry = {
        "task_id": task_id,
        "stage": qs.stage_of(task_id),
        "title": task_id,
        "dependencies": list(deps),
        "status": status,
        "owner": None,
        "head_sha": None,
        "artifact": None,
        "latest_run_id": None,
        "next_action": None,
        "blocker": None,
        "last_error": None,
    }
    entry.update(fields)
    return entry


def make_state(tasks):
    return {"schema_version": qs.SCHEMA_VERSION, "whole_project_done": False, "tasks": tasks}


def test_done_with_missing_artifact_is_inconsistent(tmp_path):
    state = make_state(
        [make_task("A01", "DONE", artifact="ghost.json", head_sha=FULL_SHA, owner="me")]
    )
    problems = cc.check_done_evidence(state, cc.RepoView(tmp_path))
    assert any("does not exist" in problem for problem in problems)


def test_artifact_resolves_through_the_commit(tmp_path):
    """A run that lives on another branch is fine when head_sha really has it."""
    state = qs.load_state(REPO / qs.STATE_PATH)
    view = cc.RepoView(REPO)
    pinned = [t for t in state["tasks"] if t["status"] in ("PARTIAL", "RUNNING") and t["artifact"]]
    assert pinned, "expected at least one seeded evidence-carrying slot"
    for task in pinned:
        assert view.resolve(task["artifact"], task["head_sha"]) is not None, task["task_id"]


def test_duplicate_concurrent_claim_is_inconsistent():
    state = make_state(
        [
            make_task("P2-05", "RUNNING", artifact="tools/queue/queue_state.py"),
            make_task("P2-06", "RUNNING", artifact="tools/queue/queue_state.py"),
        ]
    )
    problems = cc.check_duplicate_claims(state)
    assert any("duplicate artifact claim" in problem for problem in problems)


def test_running_slot_claiming_an_unknown_slots_artifact_conflicts():
    """A RUNNING slot must not sit on the artifact another slot already owns."""
    state = make_state(
        [
            make_task("P2-05", "RUNNING", artifact="tools/queue/queue_state.py"),
            make_task("P2-06", "UNKNOWN", artifact="tools/queue/queue_state.py"),
        ]
    )
    problems = cc.check_duplicate_claims(state)
    assert any("duplicate artifact claim" in problem for problem in problems)


def test_two_unknown_slots_may_share_a_planned_artifact():
    state = make_state(
        [
            make_task("P2-05", "UNKNOWN", artifact="outputs/plan.md"),
            make_task("P2-06", "UNKNOWN", artifact="outputs/plan.md"),
        ]
    )
    assert cc.check_duplicate_claims(state) == []


def test_incomplete_run_directory_is_inconsistent(tmp_path):
    run = tmp_path / cc.EVIDENCE_ROOT / "GX_r01"
    run.mkdir(parents=True)
    (run / "result.json").write_text("{}", encoding="utf-8")
    state = make_state(
        [
            make_task(
                "P2-01",
                "PARTIAL",
                artifact="%s/GX_r01/result.json" % cc.EVIDENCE_ROOT,
                head_sha=FULL_SHA,
            )
        ]
    )
    problems = cc.check_run_integrity(state, cc.RepoView(tmp_path))
    assert any("missing manifest.json" in problem for problem in problems)
    assert any("missing raw_poses.jsonl" in problem for problem in problems)


def test_missing_run_directory_is_inconsistent(tmp_path):
    state = make_state(
        [
            make_task(
                "P2-01",
                "DONE",
                artifact="%s/GONE_r01/result.json" % cc.EVIDENCE_ROOT,
                head_sha=FULL_SHA,
                owner="me",
            )
        ]
    )
    problems = cc.check_run_integrity(state, cc.RepoView(tmp_path))
    assert any("run directory" in problem and "missing" in problem for problem in problems)


def test_unknown_latest_run_id_is_inconsistent(tmp_path):
    state = make_state([make_task("P2-05", "RUNNING", artifact="a.json", latest_run_id="NOPE_r99")])
    problems = cc.check_run_ids(state, cc.RepoView(tmp_path))
    assert any("NOPE_r99" in problem for problem in problems)


def test_registry_disagreement_is_reported(tmp_path):
    state = make_state([make_task("P3-01", "UNKNOWN", deps=["A01"])])
    registry = {
        "task_count": 1,
        "tasks": [{"task_id": "P3-01", "dependencies": ["A01"]}],
    }
    path = tmp_path / "registry.json"
    path.write_text(json.dumps(registry), encoding="utf-8")
    problems = cc.check_registry(state, REPO, path)
    assert any("disagree with the policy" in problem for problem in problems)


def test_cycle_is_reported_as_fail_closed():
    state = make_state([make_task("A01", deps=["A02"]), make_task("A02", deps=["A01"])])
    problems = cc.check_structure(state, REPO)
    assert any(problem.startswith("FAIL CLOSED") for problem in problems)


def test_committed_state_is_consistent():
    """The real acceptance: the committed queue must pass the gate with exit 0."""
    state = qs.load_state(REPO / qs.STATE_PATH)
    problems = cc.check_consistency(state, REPO)
    assert problems == []


def test_cli_exit_code_on_committed_state(capsys):
    assert cc.main(["--repo", str(REPO)]) == 0
    assert "consistent" in capsys.readouterr().out


def test_cli_exit_code_on_broken_state(tmp_path, capsys):
    state = make_state([make_task("A01", "DONE", artifact="ghost.json", head_sha=FULL_SHA, owner="m")])
    path = tmp_path / "state.json"
    qs.save_state(state, path)
    assert cc.main(["--repo", str(REPO), "--state", str(path)]) == 1
    assert "INCONSISTENT" in capsys.readouterr().out
