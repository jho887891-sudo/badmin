# -*- coding: utf-8 -*-
"""A04: an exclusive GPU lock for exactly ONE Isaac instance on a shared host.

Why this exists
---------------
Isaac Sim is single-instance per GPU and cannot be trusted to fail closed: a crashed
Kit leaves stale artefacts, and two runs on the same GPU silently corrupt each other's
evidence.  The lock belongs OUTSIDE the runner: it is taken before the runner starts
and it is a plain file, so a crashed holder is always recoverable by the next run.

Contract
--------
* acquire is ATOMIC: os.open(O_CREAT|O_EXCL|O_WRONLY, 0o600).  Two real processes
  cannot both create the file; the loser gets FileExistsError and is told to refuse.
* the lock file carries schema_version, host, pid, pid_start_time, run_id,
  expected_marker, acquired_at, heartbeat_at, heartbeat_timeout_s, token, history.
* REFUSE when the recorded holder is on the same host AND its PID is alive AND its
  cmdline still contains expected_marker (a live owner), or when the holder claims a
  DIFFERENT host (we cannot prove liveness there, so we never steal).
* TAKEOVER only when the recorded PID is provably dead (kill -0 / tasklist says gone),
  when its cmdline no longer carries the marker and/or its start time differs
  (PID RECYCLED: the number now belongs to a stranger), or when a live matching
  holder's heartbeat is older than heartbeat_timeout_s.
* NEITHER path ever signals another process.  The lock is advisory exclusion, not a
  liveness killer; the runner's own descendant kill stays where it is.
* release happens through the context manager, atexit, SIGINT/SIGTERM, and requires
  the ownership token so a stale owner cannot delete its successor's lock.
* takeover is also race-safe: the old file is first renamed aside (atomic, exactly one
  contender wins), then a fresh O_EXCL file is created; the rename's content becomes
  history[].  A crash between the two steps leaves the lock ABSENT, i.e. recoverable.

Exit codes
----------
  0  acquired (or, with --dry-run, the projection is ACQUIRE/TAKEOVER)
  1  usage / internal error
  2  refused: a live (or unprovable) holder keeps the lock (dry-run included)
  3  release refused: we do not own the lock (wrong pid/token)

Usage
-----
    python tools/simulation/gpu_lock.py --lock-path /tmp/isaac_gpu0.lock \
        --run-id R1 --hold-seconds 30
    python tools/simulation/gpu_lock.py --lock-path ... --run-id R1 --dry-run
    python tools/simulation/gpu_lock.py --lock-path ... --run-id R1 --status
    python tools/simulation/gpu_lock.py --lock-path ... --run-id R1 --release --token <t>

Residual UNKNOWN (mirrored in management/morph_one_rs/A04_GPU_LOCK_AND_CLEANUP.md):
the lock cannot see a Kit service-host process that never touches the lock, and no
real two-Isaac-instance concurrency test has been run on the GPU host.
"""

from __future__ import annotations

import argparse
import atexit
import json
import os
import signal
import shutil
import socket
import subprocess
import sys
import threading
import time
import uuid
from datetime import datetime, timezone

SCHEMA_VERSION = 1
DEFAULT_HEARTBEAT_TIMEOUT_S = 30.0
DEFAULT_MARKER = "isaac"
LOCK_MODE = 0o600

EXIT_ACQUIRED = 0
EXIT_ERROR = 1
EXIT_REFUSED = 2
EXIT_RELEASE_REFUSED = 3

ACTION_ACQUIRE = "ACQUIRE"
ACTION_TAKEOVER = "TAKEOVER"
ACTION_REFUSE = "REFUSE"

REASON_NO_LOCK = "NO_LOCK"
REASON_LOCK_UNREADABLE = "LOCK_UNREADABLE"
REASON_FOREIGN_HOST = "FOREIGN_HOST"
REASON_LIVE_HOLDER = "LIVE_HOLDER"
REASON_LIVE_UNVERIFIABLE = "LIVE_HOLDER_UNVERIFIABLE"
REASON_SELF_HELD = "SELF_HELD"
REASON_PID_RECYCLED_MARKER = "PID_RECYCLED_MARKER_MISMATCH"
REASON_PID_RECYCLED_START = "PID_RECYCLED_START_TIME_MISMATCH"
REASON_HEARTBEAT_STALE = "HEARTBEAT_STALE"
REASON_HOLDER_DEAD = "HOLDER_DEAD"

REQUIRED_FIELDS = (
    "schema_version", "host", "pid", "pid_start_time", "run_id", "expected_marker",
    "acquired_at", "heartbeat_at", "heartbeat_timeout_s", "token", "history",
)


# --------------------------------------------------------------------------- probes
def pid_alive(pid, os_name=None):
    """Is this PID still running?  Never signals the process.

    On Windows os.kill(pid, 0) is NOT a probe: CPython maps it to TerminateProcess, so
    tasklist is used instead (same primitive as tools/simulation/p2_gate_runner.py).
    """
    os_name = os_name or os.name
    try:
        pid = int(pid)
    except (TypeError, ValueError):
        return False
    if pid <= 0:
        return False
    if os_name == "nt":
        try:
            completed = subprocess.run(
                ["tasklist", "/FI", "PID eq %d" % pid, "/FO", "CSV", "/NH"],
                capture_output=True, text=True, timeout=30,
            )
        except (OSError, subprocess.SubprocessError):
            return False
        return ('"%d"' % pid) in (completed.stdout or "")
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    except OSError:
        return False
    return True


def liveness_probe_command(pid, os_name=None):
    """The exact command / syscall the liveness decision used (auditable evidence)."""
    os_name = os_name or os.name
    if os_name == "nt":
        return 'tasklist /FI "PID eq %d" /FO CSV /NH' % int(pid)
    return "os.kill(%d, 0)" % int(pid)


def read_cmdline(pid, os_name=None):
    """Best-effort command line of *pid*; None when it cannot be read.

    POSIX: /proc/<pid>/cmdline (NUL separated).  Windows: wmic, then CIM via
    PowerShell.  Both are read-only queries.
    """
    os_name = os_name or os.name
    try:
        pid = int(pid)
    except (TypeError, ValueError):
        return None
    if pid <= 0:
        return None
    if os_name != "nt":
        try:
            with open("/proc/%d/cmdline" % pid, "rb") as handle:
                raw = handle.read()
        except OSError:
            return None
        if not raw:
            return None
        return raw.replace(b"\x00", b" ").decode("utf-8", "replace").strip()
    if shutil.which("wmic"):
        try:
            completed = subprocess.run(
                ["wmic", "process", "where", "processid=%d" % pid, "get", "commandline", "/value"],
                capture_output=True, text=True, timeout=30,
            )
            for line in (completed.stdout or "").splitlines():
                key, _, value = line.partition("=")
                if key.strip().lower() == "commandline" and value.strip():
                    return value.strip()
        except (OSError, subprocess.SubprocessError):
            pass
    powershell = shutil.which("powershell") or shutil.which("pwsh")
    if powershell:
        try:
            completed = subprocess.run(
                [powershell, "-NoProfile", "-NonInteractive", "-Command",
                 "(Get-CimInstance Win32_Process -Filter 'ProcessId=%d').CommandLine" % pid],
                capture_output=True, text=True, timeout=60,
            )
            text = (completed.stdout or "").strip()
            if text:
                return text
        except (OSError, subprocess.SubprocessError):
            pass
    return None


def cmdline_probe_command(pid, os_name=None):
    os_name = os_name or os.name
    if os_name != "nt":
        return "read /proc/%d/cmdline" % int(pid)
    return "wmic process where processid=%d get commandline /value" % int(pid)


def pid_start_time(pid, os_name=None):
    """Kernel start time of *pid* (Linux /proc, else None): a recycled-PID tripwire."""
    os_name = os_name or os.name
    if os_name == "nt":
        return None
    try:
        with open("/proc/%d/stat" % int(pid), "r", encoding="utf-8", errors="replace") as handle:
            fields = handle.read().rsplit(")", 1)[-1].split()
        return float(fields[19])  # field 22 of /proc/pid/stat, 0-based after the comm split
    except (OSError, IndexError, ValueError):
        return None


def utc_now():
    return datetime.now(timezone.utc)


def iso(moment):
    return moment.isoformat()


def parse_iso(text):
    if not isinstance(text, str) or not text:
        raise ValueError("not an ISO timestamp: %r" % (text,))
    value = datetime.fromisoformat(text.replace("Z", "+00:00"))
    if value.tzinfo is None:
        value = value.replace(tzinfo=timezone.utc)
    return value


# --------------------------------------------------------------------- decision core
def _read_lock(path):
    with open(str(path), "r", encoding="utf-8") as handle:
        text = handle.read()
    if not text.strip():
        raise ValueError("empty lock file")
    doc = json.loads(text)
    if not isinstance(doc, dict):
        raise ValueError("lock file is not a JSON object")
    missing = [field for field in REQUIRED_FIELDS if field not in doc]
    if missing:
        raise ValueError("missing fields: %s" % ", ".join(missing))
    return doc


def inspect_holder(doc, marker, now=None, os_name=None):
    """Everything the decision needs to know about the recorded holder."""
    os_name = os_name or os.name
    now = now or utc_now()
    pid = doc.get("pid")
    alive = pid_alive(pid, os_name=os_name)
    cmdline = read_cmdline(pid, os_name=os_name) if alive else None
    marker_found = None if cmdline is None else (str(marker) in cmdline)
    start_now = pid_start_time(pid, os_name=os_name) if alive else None
    try:
        heartbeat_age = (now - parse_iso(doc.get("heartbeat_at"))).total_seconds()
    except (TypeError, ValueError):
        heartbeat_age = None
    timeout = doc.get("heartbeat_timeout_s")
    try:
        timeout = float(timeout)
    except (TypeError, ValueError):
        timeout = DEFAULT_HEARTBEAT_TIMEOUT_S
    stale = heartbeat_age is not None and heartbeat_age > timeout
    return {
        "pid": pid,
        "host": doc.get("host"),
        "run_id": doc.get("run_id"),
        "alive": alive,
        "cmdline": cmdline,
        "marker_found": marker_found,
        "recorded_start_time": doc.get("pid_start_time"),
        "current_start_time": start_now,
        "start_time_mismatch": bool(
            doc.get("pid_start_time") is not None and start_now is not None
            and abs(float(doc["pid_start_time"]) - float(start_now)) > 1e-6
        ),
        "heartbeat_age_s": heartbeat_age,
        "heartbeat_timeout_s": timeout,
        "heartbeat_stale": stale,
        "liveness_probe": liveness_probe_command(pid, os_name=os_name) if pid else None,
        "cmdline_probe": cmdline_probe_command(pid, os_name=os_name) if pid else None,
    }


def decide(lock_path, run_id, marker=DEFAULT_MARKER, host=None, now=None,
           heartbeat_timeout_s=None, os_name=None):
    """Pure decision for *lock_path*: ACQUIRE / TAKEOVER / REFUSE with a reason."""
    os_name = os_name or os.name
    host = host or socket.gethostname()
    now = now or utc_now()
    decision = {"action": None, "reason": None, "holder": None, "checks": {},
                "lock_path": str(lock_path), "run_id": run_id, "marker": marker,
                "host": host, "now": iso(now)}

    if not os.path.exists(str(lock_path)):
        decision["action"] = ACTION_ACQUIRE
        decision["reason"] = REASON_NO_LOCK
        return decision

    try:
        doc = _read_lock(lock_path)
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        # An unreadable/empty lock is never treated as a live holder: O_CREAT|O_EXCL
        # creates the file before its owner writes the payload, so a crash in that
        # window must be recoverable.  The takeover is recorded, not silent.
        decision["action"] = ACTION_TAKEOVER
        decision["reason"] = REASON_LOCK_UNREADABLE
        decision["checks"] = {"read_error": str(exc), "holder_alive": None}
        return decision

    # The marker to verify is the one the HOLDER recorded: that is what proves the live
    # cmdline still belongs to the process that took the lock (a recycled PID does not).
    holder_marker = doc.get("expected_marker") or marker
    holder = inspect_holder(doc, holder_marker, now=now, os_name=os_name)
    decision["holder"] = {key: doc.get(key) for key in
                          ("host", "pid", "run_id", "acquired_at", "heartbeat_at",
                           "pid_start_time", "token")}
    decision["checks"] = holder
    decision["holder_alive"] = holder["alive"]
    decision["holder_cmdline_marker_found"] = holder["marker_found"]
    decision["holder_heartbeat_age_s"] = holder["heartbeat_age_s"]

    if holder["host"] != host:
        # Another machine: we cannot prove its PID alive or dead.  Never steal.
        decision["action"] = ACTION_REFUSE
        decision["reason"] = REASON_FOREIGN_HOST
        return decision

    if int(doc.get("pid") or 0) == os.getpid():
        decision["action"] = ACTION_REFUSE
        decision["reason"] = REASON_SELF_HELD
        return decision

    if not holder["alive"]:
        decision["action"] = ACTION_TAKEOVER
        decision["reason"] = REASON_HOLDER_DEAD
        return decision

    # The PID is alive.  Is it still OUR Isaac, or a recycled stranger?
    if holder["marker_found"] is False:
        decision["action"] = ACTION_TAKEOVER
        decision["reason"] = REASON_PID_RECYCLED_MARKER
        return decision
    if holder["start_time_mismatch"]:
        decision["action"] = ACTION_TAKEOVER
        decision["reason"] = REASON_PID_RECYCLED_START
        return decision
    if holder["heartbeat_stale"]:
        decision["action"] = ACTION_TAKEOVER
        decision["reason"] = REASON_HEARTBEAT_STALE
        return decision
    if holder["marker_found"] is None:
        # A live PID we cannot interrogate: fail closed rather than steal.
        decision["action"] = ACTION_REFUSE
        decision["reason"] = REASON_LIVE_UNVERIFIABLE
        return decision

    decision["action"] = ACTION_REFUSE
    decision["reason"] = REASON_LIVE_HOLDER
    return decision


# ------------------------------------------------------------------------- the lock
class GpuLock:
    """One exclusive Isaac-on-this-GPU lock.  Use as a context manager."""

    def __init__(self, lock_path, run_id, marker=DEFAULT_MARKER,
                 heartbeat_timeout_s=DEFAULT_HEARTBEAT_TIMEOUT_S, host=None,
                 os_name=None, dry_run=False):
        self.lock_path = str(lock_path)
        self.run_id = str(run_id)
        self.marker = str(marker)
        self.heartbeat_timeout_s = float(heartbeat_timeout_s)
        self.host = host or socket.gethostname()
        self.os_name = os_name or os.name
        self.dry_run = bool(dry_run)
        self.pid = os.getpid()
        self.token = None
        self.record = None
        self._released = True
        self._atexit_registered = False
        self._signal_handlers = {}
        self._heartbeat_thread = None
        self._stop_heartbeat = threading.Event()

    # -- introspection ------------------------------------------------------------
    def decide(self, now=None):
        return decide(self.lock_path, self.run_id, marker=self.marker, host=self.host,
                      now=now, heartbeat_timeout_s=self.heartbeat_timeout_s,
                      os_name=self.os_name)

    def read(self):
        return _read_lock(self.lock_path)

    # -- acquire ------------------------------------------------------------------
    def _new_record(self, history=None):
        now = utc_now()
        return {
            "schema_version": SCHEMA_VERSION,
            "host": self.host,
            "pid": self.pid,
            "pid_start_time": pid_start_time(self.pid, os_name=self.os_name),
            "run_id": self.run_id,
            "expected_marker": self.marker,
            "acquired_at": iso(now),
            "heartbeat_at": iso(now),
            "heartbeat_timeout_s": self.heartbeat_timeout_s,
            "token": self.token,
            "history": list(history or []),
        }

    def _create_exclusive(self, record):
        flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
        if hasattr(os, "O_BINARY"):
            flags |= os.O_BINARY
        handle = os.open(self.lock_path, flags, LOCK_MODE)
        try:
            os.write(handle, json.dumps(record, indent=2, sort_keys=True).encode("utf-8"))
            os.fsync(handle)
        finally:
            os.close(handle)

    def _takeover(self, decision):
        """Race-safe: rename the stale file aside first (only one contender wins),
        then create the new lock with O_EXCL.  The moved-aside content becomes the
        single history entry.  A crash in between leaves the lock ABSENT = recoverable."""
        prior = None
        tombstone = None
        if os.path.exists(self.lock_path):
            try:
                prior = _read_lock(self.lock_path)
            except (OSError, ValueError, json.JSONDecodeError):
                prior = None
            stamp = utc_now().strftime("%Y%m%dT%H%M%S")
            tombstone = "%s.takeover-%s-%d" % (self.lock_path, stamp, self.pid)
            try:
                os.rename(self.lock_path, tombstone)
            except FileNotFoundError:
                # somebody else won the rename: re-decide instead of double-stealing
                again = self.decide()
                if again["action"] == ACTION_REFUSE:
                    raise LockRefused(self.lock_path, again)
                return self._takeover(again)
            except OSError as exc:
                raise LockBusy(self.lock_path, decision, detail=str(exc))
        history = list((prior or {}).get("history") or [])
        entry = {
            "taken_over_at": iso(utc_now()),
            "by_pid": self.pid,
            "by_run_id": self.run_id,
            "reason": decision["reason"],
            "prior": {key: (prior or {}).get(key) for key in
                      ("host", "pid", "run_id", "acquired_at", "heartbeat_at",
                       "pid_start_time", "token")},
            "checks": {key: decision.get("checks", {}).get(key) for key in
                       ("alive", "marker_found", "heartbeat_age_s", "heartbeat_stale",
                        "start_time_mismatch", "liveness_probe")},
            "note": "takeover never signals the previous holder",
        }
        history.append(entry)
        record = self._new_record(history=history)
        try:
            self._create_exclusive(record)
        except FileExistsError:
            again = self.decide()
            raise LockRefused(self.lock_path, again)
        finally:
            if tombstone and os.path.exists(tombstone):
                try:
                    os.remove(tombstone)
                except OSError:
                    pass
        return record

    def acquire(self, now=None):
        decision = self.decide(now=now)
        if self.dry_run:
            decision["dry_run"] = True
            return decision
        if decision["action"] == ACTION_REFUSE:
            raise LockRefused(self.lock_path, decision)
        self.token = uuid.uuid4().hex
        try:
            if decision["action"] == ACTION_TAKEOVER:
                self.record = self._takeover(decision)
            else:
                self.record = self._new_record()
                try:
                    self._create_exclusive(self.record)
                except FileExistsError:
                    raise LockBusy(self.lock_path, self.decide())
        except (LockRefused, LockBusy):
            raise
        self._released = False
        self._install_cleanup()
        decision.update({"dry_run": False, "token": self.token, "record": self.record})
        return decision

    def _install_cleanup(self):
        if not self._atexit_registered:
            atexit.register(self.release)
            self._atexit_registered = True
        for signum in (getattr(signal, "SIGINT", None), getattr(signal, "SIGTERM", None)):
            if signum is None:
                continue
            try:
                previous = signal.getsignal(signum)
                signal.signal(signum, self._make_handler(signum, previous))
                self._signal_handlers[signum] = previous
            except (ValueError, OSError):  # not the main thread, or unsupported
                pass

    def _make_handler(self, signum, previous):
        def _handler(signum, frame):
            try:
                self.release()
            finally:
                if callable(previous) and previous not in (signal.SIG_DFL, signal.SIG_IGN):
                    return previous(signum, frame)
                raise SystemExit(128 + int(signum))
        return _handler

    # -- heartbeat / release ------------------------------------------------------
    def heartbeat(self, now=None):
        if self._released or self.dry_run:
            return False
        try:
            doc = _read_lock(self.lock_path)
        except (OSError, ValueError, json.JSONDecodeError):
            return False
        if doc.get("token") != self.token:
            return False
        doc["heartbeat_at"] = iso(now or utc_now())
        tmp = self.lock_path + ".hb-%d" % self.pid
        with open(tmp, "w", encoding="utf-8") as handle:
            json.dump(doc, handle, indent=2, sort_keys=True)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(tmp, self.lock_path)
        return True

    def start_heartbeat(self, interval_s=None):
        interval = float(interval_s or max(1.0, self.heartbeat_timeout_s / 3.0))
        self._stop_heartbeat.clear()

        def _beat():
            while not self._stop_heartbeat.wait(interval):
                self.heartbeat()

        self._heartbeat_thread = threading.Thread(target=_beat, name="gpu-lock-heartbeat",
                                                  daemon=True)
        self._heartbeat_thread.start()
        return self._heartbeat_thread

    def stop_heartbeat(self):
        self._stop_heartbeat.set()
        if self._heartbeat_thread is not None:
            self._heartbeat_thread.join(timeout=5.0)
            self._heartbeat_thread = None

    def release(self):
        """Remove the lock IFF it is still mine (same pid AND ownership token)."""
        if self._released or self.dry_run or not self.token:
            if self._atexit_registered:
                try:
                    atexit.unregister(self.release)
                except Exception:  # pragma: no cover - defensive
                    pass
                self._atexit_registered = False
            self._released = True
            return False
        self.stop_heartbeat()
        owned = False
        try:
            doc = _read_lock(self.lock_path)
            owned = doc.get("token") == self.token and int(doc.get("pid") or 0) == self.pid
        except (OSError, ValueError, json.JSONDecodeError):
            owned = False
        if not owned:
            return False
        try:
            os.remove(self.lock_path)
        except FileNotFoundError:
            return False
        finally:
            self._released = True
            if self._atexit_registered:
                try:
                    atexit.unregister(self.release)
                except Exception:  # pragma: no cover
                    pass
                self._atexit_registered = False
        return True

    def __enter__(self):
        self.acquire()
        return self

    def __exit__(self, exc_type, exc, tb):
        self.release()
        return False


class LockRefused(RuntimeError):
    def __init__(self, lock_path, decision):
        self.lock_path = str(lock_path)
        self.decision = decision
        super().__init__("GPU lock refused (%s): %s" % (decision.get("reason"), lock_path))


class LockBusy(RuntimeError):
    def __init__(self, lock_path, decision, detail=None):
        self.lock_path = str(lock_path)
        self.decision = decision
        super().__init__("GPU lock race lost%s: %s" % (" (%s)" % detail if detail else "",
                                                       lock_path))


# ------------------------------------------------------------------------------- CLI
def _emit(payload):
    sys.stdout.write(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    sys.stdout.flush()


def main(argv=None):
    parser = argparse.ArgumentParser(description="Exclusive lock for one Isaac instance per GPU")
    parser.add_argument("--lock-path", required=True, help="lock file path (created O_EXCL)")
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--marker", default=DEFAULT_MARKER,
                        help="token that must appear in the holder's cmdline")
    parser.add_argument("--heartbeat-timeout", type=float, default=DEFAULT_HEARTBEAT_TIMEOUT_S)
    parser.add_argument("--host", default=None, help="override hostname (tests/audits)")
    parser.add_argument("--hold-seconds", type=float, default=0.0,
                        help="hold the lock for N seconds, then release")
    parser.add_argument("--dry-run", action="store_true",
                        help="report the projected decision, create nothing")
    parser.add_argument("--status", action="store_true", help="report the current lock decision")
    parser.add_argument("--release", action="store_true", help="release my own lock")
    parser.add_argument("--token", default=None, help="ownership token for --release")
    args = parser.parse_args(argv)

    try:
        if args.release:
            lock = GpuLock(args.lock_path, args.run_id, marker=args.marker,
                           heartbeat_timeout_s=args.heartbeat_timeout, host=args.host)
            lock.token = args.token
            lock._released = False
            try:
                doc = lock.read()
            except (OSError, ValueError, json.JSONDecodeError) as exc:
                _emit({"action": "RELEASE", "released": False, "reason": "LOCK_UNREADABLE",
                       "lock_path": args.lock_path, "error": str(exc)})
                return EXIT_RELEASE_REFUSED
            mine = doc.get("token") == args.token and int(doc.get("pid") or 0) == os.getpid()
            if not mine:
                _emit({"action": "RELEASE", "released": False, "reason": "NOT_OWNER",
                       "lock_path": args.lock_path,
                       "holder": {key: doc.get(key) for key in ("pid", "run_id", "token")},
                       "by_pid": os.getpid()})
                return EXIT_RELEASE_REFUSED
            lock.token = args.token
            released = lock.release()
            _emit({"action": "RELEASE", "released": released, "reason": "RELEASED",
                   "lock_path": args.lock_path})
            return EXIT_ACQUIRED

        if args.status:
            lock = GpuLock(args.lock_path, args.run_id, marker=args.marker,
                           heartbeat_timeout_s=args.heartbeat_timeout, host=args.host,
                           dry_run=True)
            decision = lock.decide()
            decision["status_query"] = True
            _emit(decision)
            return EXIT_ACQUIRED

        lock = GpuLock(args.lock_path, args.run_id, marker=args.marker,
                       heartbeat_timeout_s=args.heartbeat_timeout, host=args.host,
                       dry_run=args.dry_run)
        try:
            decision = lock.acquire()
        except LockRefused as exc:
            payload = dict(exc.decision)
            payload["action"] = ACTION_REFUSE
            payload["dry_run"] = bool(args.dry_run)
            _emit(payload)
            return EXIT_REFUSED
        except LockBusy as exc:
            payload = dict(exc.decision or {})
            payload["action"] = ACTION_REFUSE
            payload["reason"] = payload.get("reason") or "RACE_LOST"
            payload["error"] = str(exc)
            _emit(payload)
            return EXIT_REFUSED
        if args.dry_run:
            _emit(decision)
            return EXIT_ACQUIRED if decision["action"] != ACTION_REFUSE else EXIT_REFUSED

        payload = dict(decision)
        payload.update({"token": lock.token, "record": lock.record, "dry_run": False})
        _emit(payload)
        try:
            if args.hold_seconds and args.hold_seconds > 0:
                deadline = time.time() + args.hold_seconds
                while time.time() < deadline:
                    time.sleep(min(1.0, max(0.05, deadline - time.time())))
                    lock.heartbeat()
            return EXIT_ACQUIRED
        finally:
            lock.release()
    except SystemExit:
        raise
    except Exception as exc:  # pragma: no cover - defensive
        _emit({"action": "ERROR", "reason": type(exc).__name__, "error": str(exc)})
        return EXIT_ERROR


if __name__ == "__main__":
    sys.exit(main())
