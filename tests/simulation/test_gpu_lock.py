# -*- coding: utf-8 -*-
"""A04 regression: one Isaac instance, one GPU host, one lock - and one honest audit trail.

Target: tools/simulation/gpu_lock.py

Contract under test (fail-closed, never-kill-another-user's-process)::

    * acquire is ATOMIC: O_CREAT|O_EXCL on the lock file, so exactly one of two
      real processes wins (evidence comes from actual subprocess contention, not
      from an in-process double call);
    * a live holder is REFUSED, a provably dead holder is TAKEN OVER, a recycled
      PID (marker mismatch) is never kept alive and never killed;
    * a lock file never carries a foreign host's PID and never survives an
      exception raised inside the holder;
    * --dry-run projects the decision and creates nothing.

No Isaac Sim and no GPU are touched: the "Isaac" process is a tiny generated
Python sleeper whose argv contains the expected marker.
"""
from __future__ import annotations

import json
import os
import socket
import subprocess
import sys
import time
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
LOCK_TOOL = ROOT / "tools" / "simulation" / "gpu_lock.py"
SYS = sys.executable
MARKER = "isaac_kit_probe_marker"


def _env():
    env = dict(os.environ)
    env["PYTHONPATH"] = str(ROOT / "tools" / "simulation") + os.pathsep + env.get("PYTHONPATH", "")
    env["PYTHONIOENCODING"] = "utf-8"
    return env


def _run(*args, timeout=60):
    return subprocess.run([SYS, str(LOCK_TOOL)] + [str(a) for a in args],
                          capture_output=True, text=True, timeout=timeout, env=_env())


def _payload(completed):
    text = (completed.stdout or "").strip()
    assert text, "lock tool printed nothing; stderr=%r" % (completed.stderr,)
    return json.loads(text)


def _wait_for(path, timeout=30.0):
    deadline = time.time() + timeout
    while time.time() < deadline:
        if Path(path).exists():
            return True
        time.sleep(0.02)
    return False


@pytest.fixture()
def lock_path(tmp_path):
    return tmp_path / "isaac_gpu0.lock"


def _spawn_holder(sleep_s=60):
    """A live process whose argv contains MARKER (stands in for Isaac Kit)."""
    return subprocess.Popen([SYS, "-c", "import time; time.sleep(%s)" % sleep_s, MARKER],
                            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)


def _spawn_lock_holder(lock_path, run_id="R-HOLDER", hold_s=30):
    return subprocess.Popen(
        [SYS, str(LOCK_TOOL), "--lock-path", str(lock_path), "--run-id", run_id,
         "--marker", MARKER, "--hold-seconds", str(hold_s)],
        stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, env=_env(),
    )


def _stop(proc):
    if proc is not None and proc.poll() is None:
        proc.kill()
        try:
            proc.wait(timeout=20)
        except subprocess.TimeoutExpired:  # pragma: no cover - defensive
            pass


def _fabricate_lock(lock_path, pid, run_id="R-OLD", heartbeat_at=None,
                    marker=MARKER, timeout_s=3600.0, token="fabricated-token", host=None):
    lock_path.write_text(json.dumps({
        "schema_version": 1, "host": host or socket.gethostname(), "pid": pid,
        "pid_start_time": None, "run_id": run_id, "expected_marker": marker,
        "acquired_at": "2026-10-20T00:00:00+00:00",
        "heartbeat_at": heartbeat_at or time.strftime("%Y-%m-%dT%H:%M:%S+00:00", time.gmtime()),
        "heartbeat_timeout_s": timeout_s, "token": token, "history": [],
    }), encoding="utf-8")
    return lock_path.read_text(encoding="utf-8")


# --------------------------------------------------------------------------- 1
def test_lock_file_content_round_trip_via_live_holder(lock_path):
    holder = _spawn_lock_holder(lock_path, run_id="R-ROUNDTRIP", hold_s=30)
    try:
        assert _wait_for(lock_path), "holder never created the lock"
        doc = json.loads(lock_path.read_text(encoding="utf-8"))
        assert doc["host"] == socket.gethostname()
        assert doc["pid"] > 0
        assert doc["run_id"] == "R-ROUNDTRIP"
        assert doc["expected_marker"] == MARKER
        for key in ("pid_start_time", "acquired_at", "heartbeat_at", "token", "history"):
            assert key in doc
        assert doc["schema_version"] == 1
        assert doc["history"] == []
        # heartbeat timestamps parse as ISO and are recent
        import datetime
        beat = datetime.datetime.fromisoformat(doc["heartbeat_at"])
        age = (datetime.datetime.now(datetime.timezone.utc) - beat).total_seconds()
        assert -5 <= age <= 120
        # status query reports the live holder and must not touch the file
        before = lock_path.read_text(encoding="utf-8")
        status = _payload(_run("--lock-path", lock_path, "--run-id", "R-OTHER", "--status"))
        assert status["action"] == "REFUSE"
        assert status["reason"] == "LIVE_HOLDER"
        assert status["holder"]["run_id"] == "R-ROUNDTRIP"
        # a status query is read-only: same holder/token, no takeover history
        after = json.loads(lock_path.read_text(encoding="utf-8"))
        assert after["token"] == json.loads(before)["token"]
        assert after["pid"] == json.loads(before)["pid"]
        assert after["history"] == []
    finally:
        _stop(holder)


# --------------------------------------------------------------------------- 2
def test_second_real_process_is_refused_while_first_holds(lock_path):
    holder = _spawn_lock_holder(lock_path, run_id="R-HOLDER", hold_s=4)
    try:
        assert _wait_for(lock_path), "holder never created the lock"
        held = json.loads(lock_path.read_text(encoding="utf-8"))
        assert held["run_id"] == "R-HOLDER"

        second = _run("--lock-path", lock_path, "--run-id", "R-SECOND", "--marker", MARKER)
        assert second.returncode == 2, "second contender must be refused, got rc=%d" % second.returncode
        out = json.loads((second.stdout or "").strip())
        assert out["action"] == "REFUSE"
        assert out["reason"] == "LIVE_HOLDER"
        assert out["holder"]["pid"] == held["pid"]
        assert out["holder"]["run_id"] == "R-HOLDER"
        assert out["holder_alive"] is True
        assert out["holder_cmdline_marker_found"] is True
        # the loser wrote nothing over the winner's record
        assert json.loads(lock_path.read_text(encoding="utf-8"))["run_id"] == "R-HOLDER"
        # the winner keeps working, then releases on its own exit
        assert holder.wait(timeout=60) == 0, holder.stderr.read()
    finally:
        _stop(holder)
    assert not lock_path.exists(), "graceful exit must remove the lock"


@pytest.mark.skipif(os.name == "nt", reason="POSIX signal delivery")
def test_signal_releases_the_lock(lock_path):
    import signal as signal_module
    holder = _spawn_lock_holder(lock_path, run_id="R-SIGNAL", hold_s=60)
    try:
        assert _wait_for(lock_path)
        holder.send_signal(signal_module.SIGTERM)
        assert holder.wait(timeout=60) == 128 + int(signal_module.SIGTERM)
    finally:
        _stop(holder)
    assert not lock_path.exists(), "SIGTERM handler must release the lock"


# --------------------------------------------------------------------------- 3
def test_takeover_of_lock_whose_recorded_pid_is_provably_dead(lock_path):
    p = subprocess.Popen([SYS, "-c", "pass"])
    p.wait(timeout=60)
    dead_pid = p.pid
    _fabricate_lock(lock_path, dead_pid, run_id="R-CRASHED",
                    heartbeat_at="2026-10-20T00:00:00+00:00", token="dead-token")

    out = _payload(_run("--lock-path", lock_path, "--run-id", "R-NEW", "--marker", MARKER))
    assert out["action"] == "TAKEOVER", out
    assert out["reason"] == "HOLDER_DEAD"
    # the CLI releases on exit, so the written record is asserted from its own output
    doc = out["record"]
    assert doc["run_id"] == "R-NEW"
    assert doc["token"] != "dead-token"
    assert len(doc["history"]) == 1
    entry = doc["history"][0]
    assert entry["reason"] == "HOLDER_DEAD"
    assert entry["prior"]["pid"] == dead_pid
    assert entry["prior"]["run_id"] == "R-CRASHED"
    assert entry["checks"]["alive"] is False


# --------------------------------------------------------------------------- 4
def test_refuses_to_steal_a_live_holder_with_the_expected_marker(lock_path):
    helper = _spawn_holder()
    try:
        before = _fabricate_lock(lock_path, helper.pid, run_id="R-FOREIGN-USER",
                                 marker=MARKER, token="live-token")
        out = _run("--lock-path", lock_path, "--run-id", "R-THIEF", "--marker", MARKER)
        assert out.returncode == 2, out.stdout + out.stderr
        payload = json.loads((out.stdout or "").strip())
        assert payload["action"] == "REFUSE"
        assert payload["reason"] == "LIVE_HOLDER"
        assert payload["holder"]["pid"] == helper.pid
        assert payload["holder_cmdline_marker_found"] is True
        assert lock_path.read_text(encoding="utf-8") == before, "refusal must not touch the lock"
        assert helper.poll() is None, "a refused contender must never kill the holder"
    finally:
        _stop(helper)


# --------------------------------------------------------------------------- 5
def test_recycled_pid_without_marker_is_taken_over_and_not_killed(lock_path):
    helper = _spawn_holder()
    try:
        _fabricate_lock(lock_path, helper.pid, run_id="R-OLD",
                        marker="isaac_the_marker_that_is_absent", token="recycled-token")
        out = _payload(_run("--lock-path", lock_path, "--run-id", "R-RECYCLED", "--marker", MARKER))
        assert out["action"] == "TAKEOVER"
        assert out["reason"] == "PID_RECYCLED_MARKER_MISMATCH"
        assert out["holder_alive"] is True
        assert out["holder_cmdline_marker_found"] is False
        assert helper.poll() is None, "takeover of a recycled PID must not kill the stranger"
        doc = out["record"]
        assert doc["run_id"] == "R-RECYCLED"
        assert doc["history"][0]["checks"]["marker_found"] is False
    finally:
        _stop(helper)


# --------------------------------------------------------------------------- 6
def test_foreign_host_lock_is_refused_not_stolen(lock_path):
    before = _fabricate_lock(lock_path, 999999, run_id="R-REMOTE",
                             heartbeat_at="2026-10-20T00:00:00+00:00",
                             host="some-other-gpu-host", timeout_s=30.0)
    c = _run("--lock-path", lock_path, "--run-id", "R-LOCAL", "--marker", MARKER)
    assert c.returncode == 2, c.stdout + c.stderr
    payload = json.loads((c.stdout or "").strip())
    assert payload["reason"] == "FOREIGN_HOST"
    assert lock_path.read_text(encoding="utf-8") == before


# --------------------------------------------------------------------------- 7
def test_stale_heartbeat_of_live_holder_is_taken_over_with_history(lock_path):
    helper = _spawn_holder()
    try:
        old = "2026-10-20T00:00:00+00:00"
        _fabricate_lock(lock_path, helper.pid, run_id="R-BACKGROUND", heartbeat_at=old,
                        timeout_s=5.0, token="stale-token")
        out = _payload(_run("--lock-path", lock_path, "--run-id", "R-AFTER-STALE",
                            "--marker", MARKER, "--heartbeat-timeout", "5"))
        assert out["action"] == "TAKEOVER"
        assert out["reason"] == "HEARTBEAT_STALE"
        doc = out["record"]
        assert doc["history"][0]["reason"] == "HEARTBEAT_STALE"
        assert doc["history"][0]["prior"]["run_id"] == "R-BACKGROUND"
        assert doc["history"][0]["checks"]["heartbeat_stale"] is True
        assert helper.poll() is None, "stale-heartbeat takeover must never kill the holder"
    finally:
        _stop(helper)


# --------------------------------------------------------------------------- 8
def test_dry_run_projects_decision_and_creates_nothing(lock_path):
    c = _run("--lock-path", lock_path, "--run-id", "R-DRY", "--marker", MARKER, "--dry-run")
    assert c.returncode == 0, c.stdout + c.stderr
    out = json.loads((c.stdout or "").strip())
    assert out["dry_run"] is True
    assert out["action"] == "ACQUIRE"
    assert not lock_path.exists(), "dry-run must not create the lock"

    helper = _spawn_holder()
    try:
        before = _fabricate_lock(lock_path, helper.pid, run_id="R-LIVE", token="t")
        c2 = _run("--lock-path", lock_path, "--run-id", "R-DRY2", "--marker", MARKER, "--dry-run")
        assert c2.returncode == 2, c2.stdout + c2.stderr
        out2 = json.loads((c2.stdout or "").strip())
        assert out2["action"] == "REFUSE"
        assert out2["reason"] == "LIVE_HOLDER"
        assert lock_path.read_text(encoding="utf-8") == before
    finally:
        _stop(helper)


# --------------------------------------------------------------------------- 9
def test_release_on_exception_inside_holder(lock_path, tmp_path):
    helper = tmp_path / "raise_inside.py"
    helper.write_text(
        "import sys\n"
        "sys.path.insert(0, %r)\n"
        "from gpu_lock import GpuLock\n"
        "lock = GpuLock(%r, run_id='R-EXC', marker=%r)\n"
        "lock.acquire()\n"
        "assert __import__('pathlib').Path(%r).exists()\n"
        "raise RuntimeError('experiment blew up')\n"
        % (str(ROOT / "tools" / "simulation"), str(lock_path), MARKER, str(lock_path)),
        encoding="utf-8",
    )
    c = subprocess.run([SYS, str(helper)], capture_output=True, text=True, timeout=60, env=_env())
    assert c.returncode != 0, "the generated holder must fail"
    assert "experiment blew up" in c.stderr
    assert not lock_path.exists(), "atexit/context release must drop the lock on exception"


# --------------------------------------------------------------------------- 10
def test_simulated_crashed_holder_is_recoverable(lock_path, tmp_path):
    crasher = tmp_path / "crasher.py"
    crasher.write_text(
        "import os, sys\n"
        "sys.path.insert(0, %r)\n"
        "from gpu_lock import GpuLock\n"
        "lock = GpuLock(%r, run_id='R-CRASH', marker=%r)\n"
        "lock.acquire()\n"
        "print(lock.pid, flush=True)\n"
        "os._exit(9)\n"
        % (str(ROOT / "tools" / "simulation"), str(lock_path), MARKER),
        encoding="utf-8",
    )
    c = subprocess.run([SYS, str(crasher)], capture_output=True, text=True, timeout=60, env=_env())
    assert c.returncode == 9, c.stdout + c.stderr
    crashed_pid = int(c.stdout.strip().splitlines()[0])
    assert lock_path.exists(), "os._exit skips atexit: the lock must survive the crash"

    # a recovering holder that KEEPS the lock, so the takeover is verified ON DISK
    recoverer = subprocess.Popen(
        [SYS, str(LOCK_TOOL), "--lock-path", str(lock_path), "--run-id", "R-RECOVER",
         "--marker", MARKER, "--hold-seconds", "20"],
        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, env=_env(),
    )
    try:
        deadline = time.time() + 30
        doc = None
        while time.time() < deadline:
            try:
                doc = json.loads(lock_path.read_text(encoding="utf-8"))
            except (OSError, ValueError):
                doc = None
            if doc and doc.get("run_id") == "R-RECOVER":
                break
            time.sleep(0.05)
        assert doc is not None and doc["run_id"] == "R-RECOVER", "crashed lock was not recovered on disk"
        assert len(doc["history"]) == 1
        assert doc["history"][0]["reason"] == "HOLDER_DEAD"
        assert doc["history"][0]["prior"]["pid"] == crashed_pid
        assert doc["history"][0]["prior"]["run_id"] == "R-CRASH"
    finally:
        recoverer.kill()
        recoverer.wait(timeout=30)


# --------------------------------------------------------------------------- 11
def test_release_from_another_pid_is_refused(lock_path):
    """A stale owner must not delete its successor's lock."""
    holder = _spawn_lock_holder(lock_path, run_id="R-OWNER", hold_s=20)
    try:
        assert _wait_for(lock_path)
        doc = json.loads(lock_path.read_text(encoding="utf-8"))
        out = _run("--lock-path", lock_path, "--run-id", "R-OWNER", "--release",
                   "--token", doc["token"])
        assert out.returncode == 3, out.stdout + out.stderr
        payload = json.loads((out.stdout or "").strip())
        assert payload["released"] is False
        assert payload["reason"] == "NOT_OWNER"
        assert json.loads(lock_path.read_text(encoding="utf-8"))["token"] == doc["token"]
    finally:
        _stop(holder)


# --------------------------------------------------------------------------- 12
def test_posix_permissions_are_user_only(lock_path):
    if os.name == "nt":
        pytest.skip("POSIX permission bits are not meaningful on Windows")
    holder = _spawn_lock_holder(lock_path, run_id="R-PERM", hold_s=20)
    try:
        assert _wait_for(lock_path)
        assert (lock_path.stat().st_mode & 0o077) == 0
    finally:
        _stop(holder)
