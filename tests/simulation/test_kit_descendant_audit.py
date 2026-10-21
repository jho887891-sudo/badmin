# -*- coding: utf-8 -*-
"""A04 regression: the post-run descendant audit must SEE survivors and must never kill.

Target: tools/simulation/kit_descendant_audit.py

Contract:: exit 0 = no recorded PID is alive; exit 2 = at least one live survivor
(reported with the exact liveness command used); exit 3 = the run directory carries
no PID records at all (UNKNOWN, fail-closed); exit 1 = usage error.  The audit is
read-only: it may probe liveness but it must never signal a process.
"""
from __future__ import annotations

import json
import subprocess
import sys
import time
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
AUDIT = ROOT / "tools" / "simulation" / "kit_descendant_audit.py"
SYS = sys.executable
MARKER = "isaac_kit_probe_marker"


def _run(*args, timeout=60):
    return subprocess.run([SYS, str(AUDIT)] + [str(a) for a in args],
                          capture_output=True, text=True, timeout=timeout)


def _report(completed):
    return json.loads((completed.stdout or "").strip())


def _spawn_sleeper(sleep_s=60):
    return subprocess.Popen([SYS, "-c", "import time; time.sleep(%s)" % sleep_s, MARKER],
                            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)


def _dead_pid():
    p = subprocess.Popen([SYS, "-c", "pass"])
    p.wait(timeout=60)
    return p.pid


def _write_summary(run_dir, payload):
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "gate_runner_summary.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")


def test_clean_run_with_dead_pids_exits_zero(tmp_path):
    run_dir = tmp_path / "run_clean"
    dead = _dead_pid()
    _write_summary(run_dir, {"run_id": "R-CLEAN", "descendant_kill": {
        "child_pid": dead, "killed_pids": [dead], "surviving_pids": [],
        "descendants_before_kill": [dead]}})
    c = _run("--run-dir", run_dir)
    assert c.returncode == 0, c.stdout + c.stderr
    rep = _report(c)
    assert rep["verdict"] == "CLEAN"
    assert rep["survivor_count"] == 0
    assert rep["pid_count"] >= 1
    assert all(item["alive"] is False for item in rep["pids"])


def test_live_survivor_is_reported_and_nothing_is_killed(tmp_path):
    run_dir = tmp_path / "run_survivor"
    _write_summary(run_dir, {"run_id": "R-LEAK", "descendant_kill": {
        "child_pid": _dead_pid(), "killed_pids": [_dead_pid()], "surviving_pids": []}})
    helper = _spawn_sleeper()
    try:
        # an extra record file, as a crashed runner would leave behind
        (run_dir / "recorded_descendants.json").write_text(
            json.dumps({"grandchild_pids": [helper.pid]}), encoding="utf-8")
        before = helper.poll()
        assert before is None
        c = _run("--run-dir", run_dir, "--marker", MARKER)
        assert c.returncode == 2, c.stdout + c.stderr
        rep = _report(c)
        assert rep["verdict"] == "SURVIVORS"
        assert rep["survivor_count"] == 1
        survivor = [i for i in rep["pids"] if i["alive"]][0]
        assert survivor["pid"] == helper.pid
        assert survivor["marker_found"] is True
        assert survivor["classification"].startswith("SURVIVOR")
        assert survivor["liveness_probe"], "the exact liveness command must be reported"
        for item in rep["pids"]:
            assert "liveness_probe" in item
        assert helper.poll() is None, "the audit must never kill anything"
    finally:
        if helper.poll() is None:
            helper.kill()
            helper.wait(timeout=20)


def test_alive_pid_without_marker_is_still_fail_closed(tmp_path):
    run_dir = tmp_path / "run_recycled"
    helper = _spawn_sleeper()
    try:
        (run_dir).mkdir(parents=True, exist_ok=True)
        (run_dir / "pids.json").write_text(json.dumps({"child_pids": [helper.pid]}), encoding="utf-8")
        c = _run("--run-dir", run_dir, "--marker", "a_marker_that_is_absent")
        assert c.returncode == 2, c.stdout + c.stderr
        rep = _report(c)
        item = [i for i in rep["pids"] if i["pid"] == helper.pid][0]
        assert item["alive"] is True
        assert item["marker_found"] is False
        assert "RECYCLED" in item["classification"] or "VERIFY" in item["classification"]
    finally:
        if helper.poll() is None:
            helper.kill()
            helper.wait(timeout=20)


def test_real_spawn_and_exit_is_audited_as_dead(tmp_path):
    """A genuinely spawned-and-exited process: its PID is a real PID record."""
    p = subprocess.Popen([SYS, "-c", "print('done')", MARKER],
                         stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    out, _err = p.communicate(timeout=60)
    assert out.strip() == "done"
    run_dir = tmp_path / "run_real"
    _write_summary(run_dir, {"run_id": "R-REAL", "descendant_kill": {
        "child_pid": p.pid, "descendants_before_kill": [p.pid]}})
    c = _run("--run-dir", run_dir, "--marker", MARKER)
    assert c.returncode == 0, c.stdout + c.stderr
    rep = _report(c)
    assert rep["pid_count"] == 1
    assert rep["pids"][0]["pid"] == p.pid
    assert rep["pids"][0]["alive"] is False


def test_run_dir_without_pid_records_is_unknown_and_nonzero(tmp_path):
    run_dir = tmp_path / "run_empty"
    run_dir.mkdir(parents=True)
    (run_dir / "result.json").write_text(json.dumps({"gate_result": "PASS"}), encoding="utf-8")
    c = _run("--run-dir", run_dir)
    assert c.returncode == 3, c.stdout + c.stderr
    rep = _report(c)
    assert rep["verdict"] == "NO_PID_RECORDS"
    assert rep["pid_count"] == 0


def test_missing_run_dir_is_a_usage_error(tmp_path):
    c = _run("--run-dir", tmp_path / "does_not_exist")
    assert c.returncode == 1, c.stdout + c.stderr


def test_json_out_matches_stdout(tmp_path):
    run_dir = tmp_path / "run_json"
    _write_summary(run_dir, {"run_id": "R-JSON", "descendant_kill": {"child_pid": _dead_pid()}})
    out_path = tmp_path / "audit.json"
    c = _run("--run-dir", run_dir, "--json-out", out_path)
    assert c.returncode == 0, c.stdout + c.stderr
    assert json.loads(out_path.read_text(encoding="utf-8"))["verdict"] == "CLEAN"
