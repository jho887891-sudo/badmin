"""Tests for the A02 queue state schema, topology and READY selector.

Pure CPU: python stdlib + pytest only. Nothing here imports Isaac/Kit.
"""

from __future__ import annotations

import json
import subprocess
from pathlib import Path

import pytest

from tools.queue import queue_state as qs

REPO = Path(__file__).resolve().parents[2]
FULL_SHA = "0123456789abcdef0123456789abcdef01234567"


def make_task(task_id, status="UNKNOWN", deps=(), **fields):
    entry = {
        "task_id": task_id,
        "stage": qs.stage_of(task_id),
        "title": "title of %s" % task_id,
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


def synthetic_spec(count=96):
    lines = []
    for index in range(count):
        if index < 10:
            task_id = "A%02d" % (index + 1)
        else:
            stage_index = index - 10
            task_id = "P0-%02d" % (stage_index + 1)
        lines.append("- [ ] **%s**：任务 %d 的标题" % (task_id, index + 1))
    return "\n".join(lines)


# ------------------------------------------------------------ spec / registry


def test_parse_spec_reads_96_slots():
    tasks = qs.parse_spec(synthetic_spec())
    assert len(tasks) == 96
    assert tasks[0]["task_id"] == "A01"
    assert tasks[0]["stage"] == "A"
    assert tasks[10]["task_id"] == "P0-01"
    assert tasks[10]["stage"] == "P0"
    assert tasks[0]["title"].startswith("任务 1")


def test_parse_spec_rejects_wrong_count():
    with pytest.raises(qs.SpecError):
        qs.parse_spec(synthetic_spec(95))


def test_parse_spec_rejects_duplicate_ids():
    lines = synthetic_spec().splitlines()
    lines[1] = lines[0]
    with pytest.raises(qs.SpecError):
        qs.parse_spec("\n".join(lines))


def test_stage_of():
    assert qs.stage_of("A01") == "A"
    assert qs.stage_of("P2-01") == "P2"
    assert qs.stage_of("P7-13") == "P7"


def test_dependency_policy_matches_spec_rules():
    assert qs.dependencies_for("P3-01") == ["P2-10"]
    assert qs.dependencies_for("P4-11") == ["P1-09", "P2-10", "P3-08"]
    assert qs.dependencies_for("P6-01") == ["P2-10"]
    assert qs.dependencies_for("P6-09") == ["P2-10", "P5-09", "P6-08"]
    assert qs.dependencies_for("P7-01") == ["P5-09", "P6-09"]
    assert qs.dependencies_for("P8-06") == ["P7-13"]
    assert qs.dependencies_for("P8-01") == []
    assert qs.dependencies_for("A01") == []
    assert qs.dependencies_for("P2-10") == []


def test_generated_state_is_schema_complete():
    spec_tasks = qs.parse_spec(synthetic_spec())
    state = qs.generate_state(spec_tasks, None, {"ref": "test", "sha": None})
    assert state["schema_version"] == qs.SCHEMA_VERSION
    assert len(state["tasks"]) == 96
    assert state["whole_project_done"] is False
    for task in state["tasks"]:
        for field in qs.REQUIRED_TASK_FIELDS:
            assert field in task, "%s missing %s" % (task["task_id"], field)
        assert task["status"] in qs.STATUSES
    assert qs.validate_state(state) == []


def test_generate_preserves_recorded_facts():
    spec_tasks = qs.parse_spec(synthetic_spec())
    before = qs.generate_state(spec_tasks, None, {})
    before["tasks"][0].update(
        {"status": "RUNNING", "owner": "main", "artifact": "x.json", "next_action": "go"}
    )
    after = qs.generate_state(spec_tasks, before, {})
    assert after["tasks"][0]["status"] == "RUNNING"
    assert after["tasks"][0]["owner"] == "main"
    assert after["tasks"][0]["artifact"] == "x.json"
    assert after["tasks"][1]["status"] == "UNKNOWN"


# --------------------------------------------------------------- READY rules


def test_ready_requires_every_dependency_done():
    state = make_state(
        [
            make_task("A01", "DONE", artifact="a.json", head_sha=FULL_SHA, owner="me"),
            make_task("P2-01", "UNKNOWN"),
            make_task("P3-01", "UNKNOWN", deps=["P2-01"]),
        ]
    )
    queue = qs.compute_queue(state)
    assert queue["derived"]["A01"] == "DONE"
    assert queue["derived"]["P2-01"] == "READY"
    assert queue["derived"]["P3-01"] == "BLOCKED"
    assert queue["blocked"][0]["blocking_chain"] == ["P3-01", "P2-01"]
    # the chain root is not DONE yet; READY means "do this one first"
    assert queue["blocked"][0]["root_cause_status"] == "READY"
    ready_ids = [item["task_id"] for item in queue["ready"]]
    assert ready_ids == ["P2-01"]


def test_multi_level_blocked_chain():
    state = make_state(
        [
            make_task("P1-09", "PARTIAL"),
            make_task("P2-10", "RUNNING"),
            make_task("P3-08", "UNKNOWN", deps=["P2-10"]),
            make_task("P4-01", "UNKNOWN", deps=["P1-09", "P2-10", "P3-08"]),
        ]
    )
    queue = qs.compute_queue(state)
    blocked = {item["task_id"]: item for item in queue["blocked"]}
    assert blocked["P4-01"]["blocking_chain"] == ["P4-01", "P1-09"]
    assert queue["derived"]["P3-08"] == "BLOCKED"


def test_failed_dependency_propagates_with_root_status():
    state = make_state(
        [
            make_task("P2-01", "FAILED"),
            make_task("P3-01", "UNKNOWN", deps=["P2-01"]),
        ]
    )
    queue = qs.compute_queue(state)
    assert queue["derived"]["P3-01"] == "BLOCKED"
    assert queue["blocked"][0]["root_cause_status"] == "FAILED"


def test_declared_ready_with_pending_dependency_is_blocked():
    state = make_state(
        [
            make_task("P2-10", "UNKNOWN"),
            make_task("P3-01", "READY", deps=["P2-10"]),
        ]
    )
    assert qs.compute_queue(state)["derived"]["P3-01"] == "BLOCKED"


def test_status_counts_cover_all_slots():
    spec_tasks = qs.parse_spec(synthetic_spec())
    state = qs.generate_state(spec_tasks, None, {})
    counts = qs.compute_queue(state)["counts"]
    assert sum(counts.values()) == 96


# ---------------------------------------------------- cycle detection / schem


def test_cycle_fails_closed():
    state = make_state(
        [
            make_task("A01", deps=["A02"]),
            make_task("A02", deps=["A01"]),
        ]
    )
    with pytest.raises(qs.QueueCycleError):
        qs.compute_queue(state)
    problems = qs.validate_state(state)
    assert any("cycle" in problem for problem in problems)


def test_unknown_dependency_raises():
    state = make_state([make_task("P3-01", deps=["P9-99"])])
    with pytest.raises(qs.QueueError):
        qs.compute_queue(state)


def test_validate_state_flags_done_without_evidence():
    state = make_state([make_task("A01", "DONE")] + [make_task("A%02d" % i) for i in range(2, 11)])
    state["tasks"] = state["tasks"][:1]
    problems = qs.validate_state(state)
    assert any("DONE without artifact" in problem for problem in problems)
    assert any("DONE without head_sha" in problem for problem in problems)


def test_illegal_status_is_rejected():
    state = make_state([make_task("A01", "SORT_OF_DONE")])
    problems = qs.validate_state(state)
    assert any("illegal status" in problem for problem in problems)


# ------------------------------------------------------------------ writes


def test_atomic_write_leaves_no_temp_files(tmp_path):
    path = tmp_path / "nested" / "state.json"
    qs.atomic_write_text(path, "{}\n")
    assert path.read_text(encoding="utf-8") == "{}\n"
    assert [p.name for p in path.parent.iterdir()] == ["state.json"]


def test_save_state_roundtrip(tmp_path):
    state = make_state([make_task("A01", "READY")])
    path = tmp_path / "state.json"
    qs.save_state(state, path)
    assert qs.load_state(path) == state


# --------------------------------------------------------------- transitions


def test_done_requires_artifact_and_sha(tmp_path):
    artifact = tmp_path / "evidence.json"
    artifact.write_text("{}", encoding="utf-8")
    state = make_state([make_task("A01", "RUNNING")])
    task = state["tasks"][0]
    errors = qs.check_transition(task, "DONE", {}, repo=tmp_path)
    assert any("requires --artifact" in error for error in errors)
    errors = qs.check_transition(task, "DONE", {"artifact": "evidence.json"}, repo=tmp_path)
    assert any("requires --head-sha" in error for error in errors)


def test_done_rejects_missing_artifact(tmp_path):
    state = make_state([make_task("A01", "RUNNING")])
    errors = qs.check_transition(
        state["tasks"][0], "DONE", {"artifact": "nope.json", "head_sha": FULL_SHA}, repo=tmp_path
    )
    assert any("artifact does not exist" in error for error in errors)


def test_done_transition_is_terminal_and_evidence_immutable(tmp_path):
    artifact = tmp_path / "evidence.json"
    artifact.write_text("{}", encoding="utf-8")
    state = make_state(
        [make_task("A01", "DONE", artifact="evidence.json", head_sha=FULL_SHA, owner="me")]
    )
    task = state["tasks"][0]
    assert any(
        "terminal" in error for error in qs.check_transition(task, "DONE", {}, repo=None)
    )
    assert qs.ALLOWED_TRANSITIONS["DONE"] == set()
    errors = qs.check_transition(task, "RUNNING", {}, repo=tmp_path)
    assert any("illegal transition" in error for error in errors)
    errors = qs.check_transition(task, "DONE", {"artifact": "other.json"}, repo=tmp_path)
    assert any("refusing to replace" in error for error in errors)


def test_illegal_transition_matrix():
    state = make_state([make_task("P2-05", "RUNNING")])
    errors = qs.check_transition(state["tasks"][0], "UNKNOWN", {}, repo=None)
    assert any("illegal transition" in error for error in errors)


def test_set_ready_requires_done_dependencies():
    state = make_state(
        [
            make_task("P2-10", "UNKNOWN"),
            make_task("P3-01", "UNKNOWN", deps=["P2-10"]),
        ]
    )
    with pytest.raises(qs.QueueError):
        qs.set_status(state, "P3-01", "READY", {})


def test_set_status_persists_updates():
    state = make_state([make_task("A01", "UNKNOWN")])
    task = qs.set_status(
        state, "A01", "RUNNING", {"owner": "main", "artifact": "tools/queue/queue_state.py"}
    )
    assert task["status"] == "RUNNING"
    assert task["owner"] == "main"


# ------------------------------------------------------------------ real repo


def test_committed_state_validates_and_is_cycle_free():
    state = qs.load_state(REPO / qs.STATE_PATH)
    assert qs.validate_state(state) == []
    queue = qs.compute_queue(state)
    assert len(queue["order"]) == 96
    assert all(task["status"] in qs.STATUSES for task in state["tasks"])


def test_committed_registry_matches_state():
    registry = json.loads((REPO / qs.REGISTRY_PATH).read_text(encoding="utf-8"))
    state = qs.load_state(REPO / qs.STATE_PATH)
    assert registry["task_count"] == 96
    assert [t["task_id"] for t in registry["tasks"]] == [t["task_id"] for t in state["tasks"]]


def test_committed_state_has_no_unproven_done():
    state = qs.load_state(REPO / qs.STATE_PATH)
    for task in state["tasks"]:
        if task["status"] == "DONE":
            assert task["artifact"] and task["head_sha"] and task["owner"]
            assert qs.artifact_present(REPO, task["artifact"], task["head_sha"])


# ----------------------------------------------------------------------- CLI


def test_cli_ready_json(tmp_path, capsys):
    state = make_state([make_task("A01", "UNKNOWN"), make_task("P3-01", "UNKNOWN", deps=["P2-10"])])
    state["tasks"].append(make_task("P2-10", "UNKNOWN"))
    path = tmp_path / "state.json"
    qs.save_state(state, path)
    code = qs.main(["--repo", str(REPO), "--state", str(path), "--ready"])
    assert code == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["rule"].startswith("READY")
    assert [item["task_id"] for item in payload["ready"]] == ["A01", "P2-10"]
    assert payload["blocked"][0]["blocking_chain"] == ["P3-01", "P2-10"]


def test_cli_set_refuses_done_without_sha(tmp_path, capsys):
    path = tmp_path / "state.json"
    qs.save_state(make_state([make_task("A01", "RUNNING")]), path)
    code = qs.main(["--repo", str(REPO), "--state", str(path), "--set", "A01=DONE",
                    "--artifact", "tools/queue/queue_state.py"])
    assert code == 1
    assert "refused" in capsys.readouterr().err


def test_cli_set_rejects_bad_status(tmp_path, capsys):
    path = tmp_path / "state.json"
    qs.save_state(make_state([make_task("A01", "UNKNOWN")]), path)
    code = qs.main(["--repo", str(REPO), "--state", str(path), "--set", "A01=MOSTLY_DONE"])
    assert code == 2


def test_cli_validate_and_status(tmp_path, capsys):
    path = tmp_path / "state.json"
    qs.save_state(make_state([make_task("A01", "UNKNOWN")]), path)
    assert qs.main(["--repo", str(REPO), "--state", str(path), "--status"]) == 0
    assert "A01" in capsys.readouterr().out


def test_cli_cycle_exit_code(tmp_path):
    path = tmp_path / "state.json"
    qs.save_state(
        make_state([make_task("A01", deps=["A02"]), make_task("A02", deps=["A01"])]), path
    )
    assert qs.main(["--repo", str(REPO), "--state", str(path), "--ready"]) == 3
