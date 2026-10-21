"""P0 outer gate runner: a FAIL must never look like a PASS.

Kit can terminate the child process with status 0 even when Python raised (observed twice in
this project).  Therefore the acceptance decision is taken HERE, in a plain Python parent
process, from two independent inputs:

  1. the child process return code,
  2. the JSON gate written by the experiment itself.

The outer process exits non-zero unless BOTH the child returned 0 AND the JSON gate says PASS.
The original child status is preserved in the summary so the Kit behaviour stays auditable.

Fail-closed details (each one is regression tested in tests/simulation/test_p2_gate_runner.py):

  * STRICT RUN-DIRECTORY CONTRACT (Issue #13 audit, 2026-10-21T06:15): one --out-dir is one
    run.  If the directory already holds evidence of an earlier attempt -- result.json,
    gate_runner_summary.json, manifest.json, or this runner's own .p2_gate_run_owner.json
    marker -- the runner REFUSES: no child is started, no gate_runner_summary.json is
    written, a gate_runner_refusal.json record is left for the audit trail, and the process
    exits non-zero with a distinct code per refusal reason (see REFUSAL_EXIT_CODES).  A reused
    directory therefore can never yield PASS from stale files.  The only way into a used
    directory is the explicit escape hatch --allow-reused-out-dir, which is recorded in the
    summary as allow_reused_out_dir=true and keeps the result.json quarantine below.
  * --timeout is enforced and the WHOLE process tree is signalled, not just the direct child:
    POSIX uses start_new_session=True plus os.killpg(SIGTERM, then SIGKILL); Windows uses
    CREATE_NEW_PROCESS_GROUP plus "taskkill /F /T".  The kill is verified and recorded under
    descendant_kill in the summary.  Isaac Kit grandchildren that detach into their own
    process group/session remain UNKNOWN and are described in summary["residual_risk"].
  * result.json is read fail-closed: unreadable/unparseable JSON, a JSON document that is not
    an object, a missing gate_result key, a non-string gate_result and a run_id that does not
    match --run-id all become an INVALID gate and outer exit 1 instead of raising out of
    main().

Usage:
    python p2_gate_runner.py --script <experiment.py> --run-id R1 --out-dir <dir> -- --lift 0.4
"""

from __future__ import annotations

import argparse
import json
import math
import os
import subprocess
import sys
import time

TIMEOUT_RETURNCODE = 124  # conventional "timed out" status, as in timeout(1)

EXIT_PASS = 0
EXIT_FAIL = 1

# One distinct exit code per refusal reason so a caller can tell "the runner refused" from
# "the gate failed".  Every refusal is a hard, child-not-started, non-zero outcome.
REFUSAL_EXIT_CODES = {
    "OUT_DIR_RUN_ID_MISMATCH": 2,         # evidence in --out-dir belongs to another run_id
    "OUT_DIR_RESUME_SAME_RUN_ID": 3,      # same run_id, earlier attempt already left evidence
    "OUT_DIR_NAME_CONFLICT": 4,           # basename != run_id while the marker says "reused"
    "OUT_DIR_NOT_EMPTY_UNRECOGNIZED": 5,  # leftovers, no parseable run_id to compare
}

EVIDENCE_FILES = ("result.json", "gate_runner_summary.json", "manifest.json")
MARKER_NAME = ".p2_gate_run_owner.json"
REFUSAL_RECORD_NAME = "gate_runner_refusal.json"
QUARANTINE_PREFIX = "result.json.prev-"

GATE_STATUS_OK = "OK"
GATE_STATUS_MISSING = "MISSING"
GATE_STATUS_INVALID = "INVALID"

_RESIDUAL_RISK = (
    "kill covers what the OS still sees under the child (taskkill /T tree on Windows, "
    "killpg of the child's own session on POSIX).  Isaac Kit grandchildren that leave the "
    "process group/session (double detach, service-hosted workers) are outside that reach: "
    "UNKNOWN, not claimed as cleaned."
)


def _sanitize(text):
    return "".join(c if (c.isalnum() or c in "-._") else "_" for c in str(text))


def _as_text(value):
    """TimeoutExpired may carry bytes even when text mode was requested."""
    if value is None:
        return ""
    if isinstance(value, bytes):
        return value.decode("utf-8", "replace")
    return value


def _json_safe(value, path=""):
    """Return (safe_value, non_finite_paths).

    Python's json.dump writes bare NaN/Infinity tokens, which are NOT valid JSON (RFC 8259)
    even though json.load tolerates them.  A summary that cannot be parsed by a strict
    reader is not evidence, so every non-finite number is replaced by its string form and
    the offending paths are reported; the caller marks such a run INVALID.
    """
    bad = []
    if isinstance(value, float):
        if math.isfinite(value):
            return value, bad
        if math.isnan(value):
            bad.append(path or "$")
            return "NaN", bad
        bad.append(path or "$")
        return ("Infinity" if value > 0 else "-Infinity"), bad
    if isinstance(value, dict):
        out = {}
        for key, item in value.items():
            safe, sub = _json_safe(item, "%s.%s" % (path, key))
            out[key] = safe
            bad.extend(sub)
        return out, bad
    if isinstance(value, (list, tuple)):
        out = []
        for index, item in enumerate(value):
            safe, sub = _json_safe(item, "%s[%d]" % (path, index))
            out.append(safe)
            bad.extend(sub)
        return out, bad
    return value, bad


# ---------------------------------------------------------------------------------------
# run-directory inspection (read-only: refusal must not modify the evidence it refuses on)
# ---------------------------------------------------------------------------------------
def _read_json_run_id(path):
    """Best-effort run_id of a JSON evidence file.  Returns None when absent/unparseable."""
    try:
        with open(path, encoding="utf-8") as handle:
            payload = json.load(handle)
    except (OSError, ValueError, UnicodeDecodeError):
        return None
    if not isinstance(payload, dict):
        return None
    run_id = payload.get("run_id")
    if isinstance(run_id, str) and run_id.strip():
        return run_id
    return None


def inspect_out_dir(out_dir):
    """Return (evidence, leftovers, marker) without touching the directory.

    evidence: [{"file", "run_id" or None, "size_bytes"}] for the recognised run artifacts.
    leftovers: other regular files (logs, quarantined result.json.prev-*, ...).
    marker: the parsed owner marker, or None.
    """
    evidence = []
    leftovers = []
    marker = None
    if not os.path.isdir(out_dir):
        return evidence, leftovers, marker
    for name in sorted(os.listdir(out_dir)):
        full = os.path.join(out_dir, name)
        if not os.path.isfile(full):
            continue
        try:
            size = os.path.getsize(full)
        except OSError:
            size = None
        if name == REFUSAL_RECORD_NAME:
            continue  # our own refusal record: not evidence of a child run
        if name == MARKER_NAME:
            marker_run_id = _read_json_run_id(full)
            marker = {"file": name, "run_id": marker_run_id, "size_bytes": size}
            evidence.append(marker)
            continue
        if name in EVIDENCE_FILES:
            evidence.append({"file": name, "run_id": _read_json_run_id(full),
                             "size_bytes": size})
            continue
        leftovers.append({"file": name, "size_bytes": size})
    return evidence, leftovers, marker


def out_dir_name_matches_run_id(out_dir, run_id):
    """True when the directory basename is the run_id (case-insensitive, sanitised)."""
    basename = os.path.basename(os.path.normpath(os.path.abspath(out_dir)))
    return _sanitize(basename).lower() == _sanitize(run_id).lower()


def decide_out_dir_use(out_dir, run_id):
    """The strict contract: may this --out-dir be used for --run-id?

    Returns a dict with code (None == fresh, go ahead), exit_code, message, evidence,
    leftovers, marker and the name-match flag.
    """
    evidence, leftovers, marker = inspect_out_dir(out_dir)
    name_matches = out_dir_name_matches_run_id(out_dir, run_id)
    plan = {
        "out_dir": out_dir,
        "out_dir_basename": os.path.basename(os.path.normpath(out_dir)),
        "requested_run_id": run_id,
        "out_dir_name_matches_run_id": name_matches,
        "evidence": evidence,
        "leftovers": leftovers,
        "marker_run_id": (marker or {}).get("run_id"),
        "code": None,
        "exit_code": EXIT_PASS,
        "message": "fresh --out-dir, no earlier attempt detected",
    }
    if not evidence and not leftovers:
        return plan

    foreign = [item for item in evidence
               if item["run_id"] is not None and item["run_id"] != run_id]
    if foreign:
        plan["code"] = "OUT_DIR_RUN_ID_MISMATCH"
        plan["message"] = (
            "--out-dir already holds evidence of a DIFFERENT run: %s.  One run_id one "
            "directory (see management/morph_one_rs/P2_GATE_RUNNER_STRICT_OUTDIR.md)."
            % ", ".join("%s(run_id=%s)" % (item["file"], item["run_id"]) for item in foreign))
    elif evidence:
        marker_says_used = marker is not None or any(
            item["file"] in EVIDENCE_FILES for item in evidence)
        if marker_says_used and not name_matches:
            plan["code"] = "OUT_DIR_NAME_CONFLICT"
            plan["message"] = (
                "--out-dir basename %r does not name the run %r while a previous attempt "
                "left its marker/evidence in that directory."
                % (plan["out_dir_basename"], run_id))
        else:
            plan["code"] = "OUT_DIR_RESUME_SAME_RUN_ID"
            plan["message"] = (
                "--out-dir already holds evidence of an earlier attempt of run_id %r "
                "(resume).  Refused by default; a partial-write resume needs the explicit "
                "--allow-reused-out-dir escape hatch." % run_id)
    else:
        plan["code"] = "OUT_DIR_NOT_EMPTY_UNRECOGNIZED"
        plan["message"] = (
            "--out-dir is not empty but holds no recognised run artifact (%s); refusing to "
            "guess whether it belongs to %r."
            % (", ".join(item["file"] for item in leftovers[:5]), run_id))
    plan["exit_code"] = REFUSAL_EXIT_CODES[plan["code"]]
    return plan


def _write_refusal_record(out_dir, payload):
    """Leave an auditable refusal next to the evidence (never overwrites run artifacts)."""
    path = os.path.join(out_dir, REFUSAL_RECORD_NAME)
    try:
        with open(path, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, ensure_ascii=False, indent=1)
    except OSError as exc:  # pragma: no cover - only on an unwritable directory
        return None, repr(exc)
    return path, None


def _refuse(plan, argv_note):
    payload = {
        "refused": True,
        # P0-04 spec-level code: one stable name for "this --out-dir already belongs to a
        # run", with the granular refusal_code kept for the audit trail.
        "error_code": "OUT_DIR_REUSED",
        "refusal_code": plan["code"],
        "refusal_exit": plan["exit_code"],
        "message": plan["message"],
        "requested_run_id": plan["requested_run_id"],
        "out_dir": plan["out_dir"],
        "out_dir_basename": plan["out_dir_basename"],
        "out_dir_name_matches_run_id": plan["out_dir_name_matches_run_id"],
        "marker_run_id": plan["marker_run_id"],
        "evidence": plan["evidence"],
        "leftovers": plan["leftovers"],
        "child_started": False,
        "allow_reused_out_dir": False,
        "argv": argv_note,
        "created_at_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "note": ("strict run-directory contract: a FAIL (and a stale PASS) must never look "
                 "like a PASS, so an already-used --out-dir is refused before any child "
                 "starts"),
    }
    record_path, record_error = _write_refusal_record(plan["out_dir"], payload)
    payload["refusal_record"] = record_path
    payload["refusal_record_error"] = record_error
    sys.stderr.write("[p2_gate_runner] REFUSED (%s, exit %d): %s\n"
                     % (plan["code"], plan["exit_code"], plan["message"]))
    sys.stderr.flush()
    print(json.dumps(payload, ensure_ascii=False, indent=1))
    return plan["exit_code"]


# ---------------------------------------------------------------------------------------
# process-tree kill (timeout honesty)
# ---------------------------------------------------------------------------------------
def child_popen_kwargs(os_name=None):
    """Platform flags that put the child (and its descendants) in their own killable group."""
    os_name = os_name or os.name
    if os_name == "nt":
        # A new process group is what lets CTRL_BREAK be scoped, but the tree kill below is
        # driven by taskkill /T; the flag still keeps the child out of the parent's console
        # group so no console signal can leak into the pytest process.
        return {"creationflags": getattr(subprocess, "CREATE_NEW_PROCESS_GROUP", 0x00000200)}
    return {"start_new_session": True}


def build_kill_tree_plan(pid, os_name=None, pgid=None):
    """Describe how a child *tree* is killed on this platform (pure function, testable)."""
    os_name = os_name or os.name
    if os_name == "nt":
        return {
            "method": "windows_taskkill_tree",
            "pid": pid,
            "argv": ["taskkill", "/F", "/T", "/PID", str(pid)],
            "covers_descendants": True,
        }
    return {
        "method": "posix_killpg",
        "pid": pid,
        "pgid": pgid if pgid is not None else pid,
        "signals": ["SIGTERM", "SIGKILL"],
        "covers_descendants": True,
    }


def pid_alive(pid, os_name=None):
    """Is this PID still running?  Uses tasklist on Windows (os.kill(pid, 0) would kill)."""
    os_name = os_name or os.name
    try:
        if os_name == "nt":
            completed = subprocess.run(
                ["tasklist", "/FI", "PID eq %d" % int(pid), "/FO", "CSV", "/NH"],
                capture_output=True, text=True, timeout=60)
            return ('"%d"' % int(pid)) in (completed.stdout or "")
        try:
            os.kill(int(pid), 0)
        except ProcessLookupError:
            return False
        except PermissionError:
            return True
        return True
    except (OSError, subprocess.SubprocessError, ValueError):
        return False


def process_group_alive(pgid, os_name=None):
    os_name = os_name or os.name
    if os_name == "nt":
        return False
    try:
        os.killpg(int(pgid), 0)
    except ProcessLookupError:
        return False
    except (PermissionError, OSError):
        return True
    return True


def list_descendant_pids(pid, os_name=None):
    """Best-effort descendant PIDs of *pid* (Windows only; POSIX relies on the process group).

    One process snapshot, then a local tree walk.  Returns [] when the snapshot is
    unavailable -- callers must then report the gap honestly instead of claiming a clean
    tree (see surviving_pids / descendant_cleanup_verified in kill_process_tree).
    """
    os_name = os_name or os.name
    if os_name != "nt":
        return []
    script = (
        "Get-CimInstance Win32_Process | Select-Object ProcessId,ParentProcessId | "
        "ConvertTo-Json -Compress")
    try:
        completed = subprocess.run(
            ["powershell", "-NoProfile", "-NonInteractive", "-Command", script],
            capture_output=True, text=True, timeout=120)
        raw = (completed.stdout or "").strip()
        if not raw:
            return []
        data = json.loads(raw)
    except (OSError, subprocess.SubprocessError, ValueError):
        return []
    if isinstance(data, dict):
        data = [data]
    if not isinstance(data, list):
        return []
    children = {}
    for item in data:
        if not isinstance(item, dict):
            continue
        try:
            parent = int(item["ParentProcessId"])
            child_pid = int(item["ProcessId"])
        except (KeyError, TypeError, ValueError):
            continue
        children.setdefault(parent, []).append(child_pid)
    seen, stack = [], [int(pid)]
    while stack:
        current = stack.pop()
        for child_pid in children.get(current, []):
            if child_pid not in seen:
                seen.append(child_pid)
                stack.append(child_pid)
    return seen


def _wait_gone(pid, deadline_s, os_name=None):
    deadline = time.time() + deadline_s
    while time.time() < deadline:
        if not pid_alive(pid, os_name=os_name):
            return True
        time.sleep(0.05)
    return not pid_alive(pid, os_name=os_name)


def kill_process_tree(proc, grace_s=2.0, os_name=None):
    """Kill the child AND its descendants; return an auditable record.  Never raises."""
    os_name = os_name or os.name
    pid = proc.pid
    plan = build_kill_tree_plan(pid, os_name=os_name)
    # Concrete PID evidence (P0-03): snapshot the live descendants BEFORE the kill, then
    # prove afterwards which of those PIDs is gone and which is still running.
    descendants_before = list_descendant_pids(pid, os_name=os_name)
    record = {"pid": pid, "child_pid": pid, "os": os_name, "plan": plan, "attempted": True,
              "errors": [], "child_terminated": None, "descendants_terminated": None,
              "descendants_before_kill": list(descendants_before),
              "descendant_snapshot_available": bool(descendants_before) or os_name != "nt",
              "killed_pids": [pid] + list(descendants_before),
              "surviving_pids": []}
    if os_name == "nt":
        try:
            completed = subprocess.run(plan["argv"], capture_output=True, text=True,
                                       timeout=120)
            record["taskkill_returncode"] = completed.returncode
            record["taskkill_stdout"] = (completed.stdout or "").strip()
            record["taskkill_stderr"] = (completed.stderr or "").strip()
        except (OSError, subprocess.SubprocessError) as exc:
            record["errors"].append("taskkill: %r" % (exc,))
        if not _wait_gone(pid, 5.0, os_name=os_name):
            # taskkill refused or the process is immune: fall back to a hard single kill and
            # report honestly that the guarantee is weaker.
            try:
                proc.kill()
                record["method"] = "windows_taskkill_tree+fallback_proc_kill"
            except OSError as exc:
                record["errors"].append("fallback kill: %r" % (exc,))
        else:
            record["method"] = "windows_taskkill_tree"
    else:
        import signal
        pgid = plan["pgid"]
        for sig_name in ("SIGTERM", "SIGKILL"):
            try:
                os.killpg(pgid, getattr(signal, sig_name))
            except ProcessLookupError:
                break
            except OSError as exc:
                record["errors"].append("killpg %s: %r" % (sig_name, exc))
            if sig_name == "SIGTERM":
                try:
                    proc.wait(timeout=grace_s)
                except subprocess.TimeoutExpired:
                    pass
        record["method"] = "posix_killpg"
    try:
        proc.wait(timeout=30)
    except (subprocess.TimeoutExpired, OSError):
        pass
    record["child_terminated"] = not pid_alive(pid, os_name=os_name)
    # verify every PID we claim to have killed, not just the direct child
    record["surviving_pids"] = [int(item) for item in record["killed_pids"]
                               if pid_alive(item, os_name=os_name)]
    record["surviving_pids_verified_at"] = round(time.time(), 3)
    if os_name == "nt":
        record["descendants_terminated"] = (not record["surviving_pids"]
                                           and record["child_terminated"])
    else:
        record["descendants_terminated"] = not process_group_alive(plan["pgid"], os_name=os_name)
    record["verified_terminated"] = bool(record["child_terminated"]
                                         and record["descendants_terminated"]
                                         and not record["surviving_pids"])
    # Honest label: True only when we actually saw a descendant list AND every PID in it is
    # dead.  A missing snapshot or a survivor makes it False/None, never a silent True.
    if os_name == "nt" and not record["descendant_snapshot_available"]:
        record["descendant_cleanup_verified"] = None
        record["descendant_cleanup_note"] = (
            "descendant snapshot unavailable: no PID list to verify, tree kill unverified")
    else:
        record["descendant_cleanup_verified"] = bool(record["verified_terminated"])
    record["residual_risk"] = _RESIDUAL_RISK
    return record


# ---------------------------------------------------------------------------------------
# result.json quarantine + fail-closed gate reading
# ---------------------------------------------------------------------------------------
def _quarantine_stale_result(out_dir, result_path, run_id):
    """Move a previous run's result.json aside so the gate only reads THIS child's file.

    Without this a reused --out-dir is fail-OPEN: a child that exits 0 without writing
    result.json (exactly what Kit does when it swallows a failure) would be judged PASS by
    a stale file from an earlier successful run.  Only reachable through the explicit
    --allow-reused-out-dir escape hatch, because the strict contract refuses a used
    directory before this point.
    """
    if not os.path.isfile(result_path):
        return None
    safe_run_id = "".join(c if c.isalnum() else "_" for c in str(run_id)) or "run"
    stamp = time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())
    target = os.path.join(out_dir, QUARANTINE_PREFIX + "%s-%s.json" % (stamp, safe_run_id))
    os.replace(result_path, target)
    return target


GATE_VALUES = ("PASS", "FAIL")  # the ONLY two accepted gate_result strings


def _invalid(reason, gate="INVALID", metrics=None, run_id=None, gates=None):
    return {"json_gate": gate, "json_gate_status": GATE_STATUS_INVALID,
            "json_gate_reason": reason, "json_metrics": metrics or {},
            "result_payload_run_id": run_id, "gates": gates or {},
            "gates_all_true": False, "non_finite_metrics": []}


def read_json_gate(result_path, requested_run_id=None):
    """Read result.json fail-closed.  NEVER raises; the verdict is data, not an exception.

    Every check below is a fail-closed structural assertion (P0-02):
      * absent / unreadable / non-object / truncated document -> INVALID (specific label),
      * gate_result must be the STRING "PASS" or "FAIL": null, 0, 1, true, "pass", "NaN",
        a list or an object are all INVALID -- no truthiness, no case folding,
      * "gates" must be a non-empty object of INDEPENDENT BOOLEAN assertions; an absent,
        empty or non-boolean gates object is INVALID (a missing gate is not a passing gate),
      * gate_result == "PASS" while a gates entry is false/non-boolean is a CONFLICT and is
        INVALID, so a self-contradicting child can never be accepted,
      * NaN/Infinity anywhere in the payload (metrics included) is INVALID; the metrics are
        stringified so the summary we write stays strictly valid JSON,
      * a run_id that disagrees with --run-id is INVALID.
    """
    if not os.path.isfile(result_path):
        return {"json_gate": "MISSING", "json_gate_status": GATE_STATUS_MISSING,
                "json_gate_reason": "result.json was not written by the child",
                "json_metrics": {}, "result_payload_run_id": None, "gates": {},
                "gates_all_true": False, "non_finite_metrics": []}
    try:
        with open(result_path, encoding="utf-8") as handle:
            text = handle.read()
    except (OSError, UnicodeDecodeError) as exc:
        return _invalid("INVALID_JSON: result.json unreadable (%r)" % (exc,),
                        gate="INVALID_JSON")
    if not text.strip():
        return _invalid("INVALID_JSON: result.json is empty", gate="INVALID_JSON")
    try:
        payload = json.loads(text)
    except ValueError as exc:
        return _invalid("INVALID_JSON: json.loads failed (%s)" % (exc,), gate="INVALID_JSON")
    if not isinstance(payload, dict):
        return _invalid("INVALID_JSON: top-level document is %s, expected a JSON object"
                        % type(payload).__name__, gate="INVALID_JSON")

    safe_payload, non_finite = _json_safe(payload)
    metrics = safe_payload.get("metrics", {}) if isinstance(safe_payload, dict) else {}
    if not isinstance(metrics, dict):
        metrics = {}
    payload_run_id = safe_payload.get("run_id") if isinstance(safe_payload, dict) else None
    payload_run_id = payload_run_id if isinstance(payload_run_id, str) else None
    raw_gates = payload.get("gates")
    gates = raw_gates if isinstance(raw_gates, dict) else {}
    gates_all_true = bool(gates) and all(value is True for value in gates.values())

    def fail(reason, gate):
        outcome = _invalid(reason, gate=gate, metrics=metrics, run_id=payload_run_id,
                           gates=safe_payload.get("gates") if isinstance(gates, dict) else {})
        outcome["non_finite_metrics"] = non_finite
        return outcome

    if "gate_result" not in payload:
        return fail("MISSING_GATE_RESULT: result.json has no gate_result key",
                    "MISSING_GATE_RESULT")
    gate = payload["gate_result"]
    if not isinstance(gate, str):
        return fail("INVALID_GATE_RESULT: gate_result is %s (%r), expected the string "
                    "\"PASS\" or \"FAIL\"" % (type(gate).__name__, gate),
                    "INVALID_GATE_RESULT")
    if gate not in GATE_VALUES:
        # exactly PASS/FAIL: "pass", "PASS ", "true", "NaN" or "" are not accepted, and the
        # rejected spelling is preserved verbatim so the audit sees what the child wrote
        return fail("INVALID_GATE_RESULT: gate_result %r is not one of %s (case and exact "
                    "spelling are load-bearing)" % (gate, list(GATE_VALUES)),
                    "INVALID_GATE_RESULT")
    if not isinstance(raw_gates, dict) or not gates:
        return fail("GATES_MISSING: result.json must carry a non-empty \"gates\" object of "
                    "independent boolean assertions; an absent/empty gates object is INVALID",
                    "GATES_MISSING")
    non_boolean = sorted(str(key) for key, value in gates.items()
                         if not isinstance(value, bool))
    if non_boolean:
        return fail("GATES_NOT_BOOLEAN: gate(s) %s are not boolean true/false; a non-boolean "
                    "gate is not a passing gate" % (", ".join(non_boolean),),
                    "GATES_NOT_BOOLEAN")
    if gate == "PASS" and not gates_all_true:
        return fail("GATE_CONFLICT: gate_result is PASS but gate(s) %s are false"
                    % (", ".join(sorted(str(key) for key, value in gates.items()
                                        if value is not True)),),
                    "GATE_CONFLICT")
    if non_finite:
        return fail("NON_FINITE_METRICS: NaN/Infinity found at %s (stringified in the "
                    "summary so the summary itself stays valid JSON)"
                    % (", ".join(non_finite[:8]),), "NON_FINITE_METRICS")
    if requested_run_id is not None and payload_run_id is not None \
            and payload_run_id != requested_run_id:
        return fail("RUN_ID_MISMATCH: result.json run_id %r != requested %r"
                    % (payload_run_id, requested_run_id), "RUN_ID_MISMATCH")
    return {"json_gate": gate, "json_gate_status": GATE_STATUS_OK, "json_gate_reason": "",
            "json_metrics": metrics, "result_payload_run_id": payload_run_id,
            "gates": safe_payload.get("gates", {}), "gates_all_true": gates_all_true,
            "non_finite_metrics": []}


# ---------------------------------------------------------------------------------------
# metadata consistency (P0-05): the manifest must not lie about what was actually run
# ---------------------------------------------------------------------------------------
GRAVITY_TOL = 1e-6
DT_TOL = 1e-9


def _as_float(value):
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    value = float(value)
    return value if math.isfinite(value) else None


def check_manifest_consistency(out_dir, gravity_z=None, steps=None, dt=None):
    """Compare manifest.json against the arguments the runner actually passed.

    A real bug we hit: the manifest recorded gravity [0,0,0] while the command ran with
    --gravity-z -9.81, so every "zero-gravity" claim in that run was false.  The manifest is
    the provenance record, so a disagreement with the command line is INVALID:
      * gravity_m_s2          vs --gravity-z   (list form compares its Z component)
      * steps / dt            vs --steps, --dt
    An absent manifest makes no claim (metadata_consistent True, status MANIFEST_ABSENT), an
    UNREADABLE manifest is fail-closed (False), and a comparison requested without the
    matching manifest field is also fail-closed (False) -- we never assume agreement.
    """
    path = os.path.join(out_dir, "manifest.json")
    result = {"manifest_path": path, "manifest_present": False, "metadata_status": "MANIFEST_ABSENT",
              "metadata_consistent": True, "mismatches": [], "checked": {},
              "requested": {"gravity_z": gravity_z, "steps": steps, "dt": dt},
              "note": "no manifest.json: no provenance claim was made, nothing to contradict"}
    if not os.path.isfile(path):
        return result
    result["manifest_present"] = True
    try:
        with open(path, encoding="utf-8") as handle:
            manifest = json.load(handle)
    except (OSError, ValueError, UnicodeDecodeError) as exc:
        result["metadata_consistent"] = False
        result["metadata_status"] = "MANIFEST_UNREADABLE"
        result["mismatches"].append("manifest.json unreadable: %r" % (exc,))
        result["note"] = "an unreadable provenance record cannot be trusted"
        return result
    if not isinstance(manifest, dict):
        result["metadata_consistent"] = False
        result["metadata_status"] = "MANIFEST_NOT_OBJECT"
        result["mismatches"].append("manifest.json top-level is %s" % type(manifest).__name__)
        return result

    statuses = []

    if gravity_z is not None:
        raw = manifest.get("gravity_m_s2")
        recorded = None
        if isinstance(raw, (list, tuple)) and len(raw) >= 3:
            recorded = _as_float(raw[2])
            result["checked"]["gravity_vector"] = list(raw)
        else:
            recorded = _as_float(raw)
        result["checked"]["gravity_m_s2"] = recorded
        result["checked"]["gravity_z_requested"] = gravity_z
        if recorded is None:
            statuses.append("GRAVITY_UNVERIFIABLE")
            result["mismatches"].append(
                "manifest has no readable gravity_m_s2 while --gravity-z=%s was passed"
                % (gravity_z,))
        elif abs(recorded - float(gravity_z)) > GRAVITY_TOL:
            statuses.append("GRAVITY_MISMATCH")
            result["mismatches"].append(
                "manifest gravity_m_s2=%r disagrees with --gravity-z=%r"
                % (recorded, gravity_z))

    if steps is not None:
        recorded = manifest.get("steps")
        result["checked"]["steps"] = recorded
        if isinstance(recorded, bool) or not isinstance(recorded, int):
            statuses.append("STEPS_UNVERIFIABLE")
            result["mismatches"].append(
                "manifest has no integer steps while --steps=%s was passed" % (steps,))
        elif int(recorded) != int(steps):
            statuses.append("STEPS_MISMATCH")
            result["mismatches"].append("manifest steps=%r != --steps=%r"
                                        % (recorded, steps))

    if dt is not None:
        recorded = _as_float(manifest.get("dt"))
        result["checked"]["dt"] = recorded
        if recorded is None:
            statuses.append("DT_UNVERIFIABLE")
            result["mismatches"].append(
                "manifest has no numeric dt while --dt=%s was passed" % (dt,))
        elif abs(recorded - float(dt)) > DT_TOL:
            statuses.append("DT_MISMATCH")
            result["mismatches"].append("manifest dt=%r != --dt=%r" % (recorded, dt))

    result["metadata_consistent"] = not result["mismatches"]
    result["metadata_status"] = ",".join(statuses) if statuses else "CONSISTENT"
    result["note"] = ("manifest agrees with the arguments the runner passed"
                      if result["metadata_consistent"] else
                      "manifest disagrees with the command actually executed -> INVALID")
    return result


# ---------------------------------------------------------------------------------------
# JSONL integrity (P0-06): per-step completeness, ordering, finiteness, contact honesty
# ---------------------------------------------------------------------------------------
CONTACT_KEYS = ("contact", "contact_force", "contact_normal_force", "touch", "touching",
                "in_contact", "contact_status")
LIVE_CONTACT_KEYS = ("contact", "contact_force", "contact_normal_force", "touch", "touching",
                     "in_contact")


def _is_contact_key(key):
    lowered = str(key).lower()
    return any(token in lowered for token in CONTACT_KEYS)


def _is_live_contact_key(key):
    lowered = str(key).lower()
    return lowered in LIVE_CONTACT_KEYS or "force" in lowered


def check_pose_jsonl(path, expected_count=None):
    """Validate raw_poses.jsonl frame by frame.  NEVER raises; returns a report dict.

    Checks: every line is a JSON object; "step" is an int and STRICTLY increasing (a
    duplicated or out-of-order frame is a rewound/merged run, not a longer run); the frame
    count matches the expected count; every number is finite (NaN/Inf is not a measurement);
    and every contact field is either a finite number or explicitly null.  A null contact is
    "unavailable" and is COUNTED SEPARATELY from a numeric 0 -- contact_status=unavailable
    with a numeric 0 force, or a 0 written where the sensor never reported, is exactly the
    silent zero this check exists to catch.
    """
    report = {
        "path": path, "present": False, "expected_count": expected_count,
        "line_count": 0, "expected_count_matches": None, "steps": [],
        "first_step": None, "last_step": None, "monotonic_steps": None,
        "duplicate_steps": [], "invalid_json_lines": [], "non_object_lines": [],
        "missing_step_lines": [], "non_integer_step_lines": [],
        "non_finite": [], "contact_numeric_count": 0, "contact_null_count": 0,
        "contact_invalid": [], "contact_silently_zero": [],
        "contact_status_unavailable_count": 0, "errors": [], "valid": False,
    }
    if not path or not os.path.isfile(path):
        report["errors"].append("raw_poses.jsonl is absent")
        return report
    report["present"] = True
    try:
        with open(path, encoding="utf-8") as handle:
            raw_lines = handle.read().splitlines()
    except (OSError, UnicodeDecodeError) as exc:
        report["errors"].append("raw_poses.jsonl unreadable: %r" % (exc,))
        return report

    steps = []
    for index, line in enumerate(raw_lines, start=1):
        if not line.strip():
            continue                      # trailing blank line is tolerated, not counted
        report["line_count"] += 1
        try:
            record = json.loads(line)
        except ValueError as exc:
            report["invalid_json_lines"].append({"line": index, "error": str(exc)})
            continue
        if not isinstance(record, dict):
            report["non_object_lines"].append({"line": index,
                                               "type": type(record).__name__})
            continue
        if "step" not in record:
            report["missing_step_lines"].append(index)
        else:
            step = record["step"]
            if isinstance(step, bool) or not isinstance(step, int):
                report["non_integer_step_lines"].append({"line": index, "step": step})
            else:
                if steps and step <= steps[-1]:
                    report["duplicate_steps"].append({"line": index, "step": step,
                                                      "previous": steps[-1]})
                steps.append(step)

        # two passes: contact_status may come AFTER the contact fields in the record
        unavailable = False
        status_value = record.get("contact_status")
        if isinstance(status_value, str):
            if status_value.strip().lower() in ("unavailable", "none", "unknown",
                                                "no_sensor"):
                unavailable = True
                report["contact_status_unavailable_count"] += 1
        for key, value in record.items():
            if str(key).lower() == "contact_status":
                continue
            if _is_contact_key(key) and _is_live_contact_key(key):
                if value is None:
                    report["contact_null_count"] += 1
                elif isinstance(value, bool) or not isinstance(value, (int, float)):
                    report["contact_invalid"].append({"line": index, "field": key,
                                                      "value": value,
                                                      "why": "not numeric and not null"})
                elif not math.isfinite(float(value)):
                    report["contact_invalid"].append({"line": index, "field": key,
                                                      "value": str(value),
                                                      "why": "non-finite"})
                else:
                    report["contact_numeric_count"] += 1
                    if unavailable and float(value) == 0.0:
                        report["contact_silently_zero"].append(
                            {"line": index, "field": key, "value": 0.0,
                             "why": "contact_status=unavailable yet a 0 was written"})
            # every other numeric field must be finite and JSON-safe
            if isinstance(value, float) and not math.isfinite(value):
                report["non_finite"].append({"line": index, "field": key,
                                             "value": repr(value)})
            elif isinstance(value, dict):
                _, bad = _json_safe(value, key)
                for item in bad:
                    report["non_finite"].append({"line": index, "field": item,
                                                 "value": "non-finite"})
            elif isinstance(value, list):
                _, bad = _json_safe(value, key)
                for item in bad:
                    report["non_finite"].append({"line": index, "field": item,
                                                 "value": "non-finite"})

    report["steps"] = steps
    report["first_step"] = steps[0] if steps else None
    report["last_step"] = steps[-1] if steps else None
    report["monotonic_steps"] = bool(steps) and not report["duplicate_steps"] \
        and not report["missing_step_lines"] and not report["non_integer_step_lines"]
    if expected_count is not None:
        report["expected_count_matches"] = report["line_count"] == int(expected_count)
    problems = []
    if report["line_count"] == 0:
        problems.append("no frames")
    if report["invalid_json_lines"]:
        problems.append("%d unparseable line(s)" % len(report["invalid_json_lines"]))
    if report["non_object_lines"]:
        problems.append("%d non-object line(s)" % len(report["non_object_lines"]))
    if report["missing_step_lines"]:
        problems.append("%d line(s) without step" % len(report["missing_step_lines"]))
    if report["non_integer_step_lines"]:
        problems.append("%d non-integer step(s)" % len(report["non_integer_step_lines"]))
    if report["duplicate_steps"]:
        problems.append("%d non-monotonic step(s)" % len(report["duplicate_steps"]))
    if report["non_finite"]:
        problems.append("%d non-finite value(s)" % len(report["non_finite"]))
    if report["contact_invalid"]:
        problems.append("%d invalid contact field(s)" % len(report["contact_invalid"]))
    if report["contact_silently_zero"]:
        problems.append("%d silent zero contact(s)" % len(report["contact_silently_zero"]))
    if expected_count is not None and not report["expected_count_matches"]:
        problems.append("frame count %d != expected %d"
                        % (report["line_count"], int(expected_count)))
    report["errors"] = problems
    report["valid"] = not problems
    return report


# ---------------------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------------------
def main(argv=None):
    parser = argparse.ArgumentParser(description="outer PASS/FAIL gate runner")
    # --check-jsonl is a standalone integrity check, so the child arguments are only
    # required when a child is actually going to be started (validated below).
    parser.add_argument("--script")
    parser.add_argument("--run-id")
    parser.add_argument("--out-dir")
    parser.add_argument("--timeout", type=float, default=1800.0)
    parser.add_argument("--allow-reused-out-dir", action="store_true",
                        help="escape hatch: write into an --out-dir that already holds "
                             "evidence (recorded in the summary; stale result.json is "
                             "quarantined before the child starts)")
    parser.add_argument("--gravity-z", type=float, default=None,
                        help="the gravity-z actually passed to the experiment; compared "
                             "against manifest.json gravity_m_s2 (P0-05)")
    parser.add_argument("--steps", type=int, default=None,
                        help="step count actually requested; compared against manifest steps")
    parser.add_argument("--dt", type=float, default=None,
                        help="dt actually requested; compared against manifest dt")
    parser.add_argument("--jsonl", default=None,
                        help="raw_poses.jsonl to validate (default: <out-dir>/raw_poses.jsonl)")
    parser.add_argument("--expected-frames", type=int, default=None,
                        help="expected frame count for the JSONL integrity check")
    parser.add_argument("--require-jsonl", action="store_true",
                        help="an absent raw_poses.jsonl is INVALID instead of merely absent")
    parser.add_argument("--check-jsonl", default=None, metavar="PATH",
                        help="standalone mode: validate a raw_poses.jsonl and exit 0/1 "
                             "(no child is started)")
    args, extra = parser.parse_known_args(argv)
    if extra and extra[0] == "--":
        extra = extra[1:]

    if args.check_jsonl:
        # P0-06 CLI: a pure integrity check, usable on any finished run directory.
        report = check_pose_jsonl(os.path.abspath(args.check_jsonl), args.expected_frames)
        print(json.dumps(report, ensure_ascii=False, indent=1))
        return EXIT_PASS if report["valid"] else EXIT_FAIL
    missing_required = [name for name in ("script", "run_id", "out_dir")
                        if not getattr(args, name)]
    if missing_required:
        parser.error("the following arguments are required: %s"
                     % ", ".join("--" + name.replace("_", "-")
                                 for name in missing_required))

    out_dir = os.path.abspath(args.out_dir)
    argv_note = " ".join(sys.argv)
    plan = decide_out_dir_use(out_dir, args.run_id)
    if plan["code"] is not None and not args.allow_reused_out_dir:
        return _refuse(plan, argv_note)

    os.makedirs(out_dir, exist_ok=True)
    result_path = os.path.join(out_dir, "result.json")
    quarantined = _quarantine_stale_result(out_dir, result_path, args.run_id)
    prior_evidence = plan["evidence"]
    prior_attempt = 1
    for item in prior_evidence:
        if item["file"] == MARKER_NAME:
            prior_attempt += 1
    marker_path = os.path.join(out_dir, MARKER_NAME)
    with open(marker_path, "w", encoding="utf-8") as handle:
        json.dump({"run_id": args.run_id,
                   "run_id_sanitized": _sanitize(args.run_id),
                   "out_dir_basename": plan["out_dir_basename"],
                   "out_dir_name_matches_run_id": plan["out_dir_name_matches_run_id"],
                   "attempt": prior_attempt,
                   "allow_reused_out_dir": bool(args.allow_reused_out_dir),
                   "started_at_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())},
                  handle, ensure_ascii=False, indent=1)

    command = [sys.executable, os.path.abspath(args.script),
               "--run-id", args.run_id, "--out-dir", out_dir] + list(extra)
    started = time.time()
    timed_out = False
    kill_record = None
    proc = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                            text=True, **child_popen_kwargs())
    child_pid = proc.pid
    try:
        stdout_text, stderr_text = proc.communicate(timeout=args.timeout)
        child_returncode = proc.returncode
    except subprocess.TimeoutExpired as exc:
        timed_out = True
        child_returncode = TIMEOUT_RETURNCODE
        kill_record = kill_process_tree(proc)
        try:
            rest_out, rest_err = proc.communicate(timeout=30)
        except subprocess.TimeoutExpired:  # pragma: no cover - pipes stuck open
            rest_out, rest_err = "", ""
        stdout_text = _as_text(rest_out) or _as_text(exc.stdout)
        stderr_text = _as_text(rest_err) or _as_text(exc.stderr)
    elapsed = round(time.time() - started, 3)
    if timed_out:
        stderr_text = _as_text(stderr_text) + (
            "\n[p2_gate_runner] child exceeded --timeout=%s s and its process tree was killed "
            "(%s)\n" % (args.timeout, (kill_record or {}).get("method")))

    child_stdout = os.path.join(out_dir, "gate_child.stdout.log")
    child_stderr = os.path.join(out_dir, "gate_child.stderr.log")
    with open(child_stdout, "w", encoding="utf-8") as handle:
        handle.write(stdout_text)
    with open(child_stderr, "w", encoding="utf-8") as handle:
        handle.write(stderr_text)

    gate_report = read_json_gate(result_path, requested_run_id=args.run_id)
    gate = gate_report["json_gate"]

    # P0-05: the manifest must agree with the command we actually ran (gravity bug)
    metadata = check_manifest_consistency(out_dir, gravity_z=args.gravity_z,
                                          steps=args.steps, dt=args.dt)
    gravity_mismatch = "GRAVITY" in metadata["metadata_status"]
    if not metadata["metadata_consistent"]:
        gate_status = GATE_STATUS_INVALID
        if gravity_mismatch:
            gate = "INVALID_METADATA"
        gate_report = dict(gate_report)
        gate_report["json_gate_status"] = gate_status
        gate_report["json_gate_reason"] = (
            (gate_report["json_gate_reason"] + " | " if gate_report["json_gate_reason"] else "")
            + "METADATA_INCONSISTENT: " + "; ".join(metadata["mismatches"]))

    # P0-06: per-frame JSONL integrity (absent is only INVALID when --require-jsonl)
    jsonl_path = os.path.abspath(args.jsonl) if args.jsonl \
        else os.path.join(out_dir, "raw_poses.jsonl")
    jsonl_report = check_pose_jsonl(jsonl_path, args.expected_frames)
    if not jsonl_report["present"] and not args.require_jsonl:
        jsonl_valid = True
        jsonl_status = "JSONL_ABSENT"
    else:
        jsonl_valid = bool(jsonl_report["valid"])
        jsonl_status = "JSONL_VALID" if jsonl_valid else "JSONL_INVALID"
    if not jsonl_valid:
        gate_report = dict(gate_report)
        gate_report["json_gate_status"] = GATE_STATUS_INVALID

    child_ok = child_returncode == 0
    gates_ok = (gate_report["json_gate_status"] == GATE_STATUS_OK) and gate == "PASS"
    metadata_ok = bool(metadata["metadata_consistent"])
    outer_exit = EXIT_PASS if (child_ok and gates_ok and metadata_ok and jsonl_valid) \
        else EXIT_FAIL
    summary = {
        "run_id": args.run_id,
        "command": " ".join(command),
        "child_returncode": child_returncode,
        "timed_out": timed_out,
        "timeout_s": args.timeout,
        "quarantined_stale_result": quarantined,
        "json_gate": gate,
        "json_gate_status": gate_report["json_gate_status"],
        "json_gate_reason": gate_report["json_gate_reason"],
        "json_metrics": gate_report["json_metrics"],
        "result_payload_run_id": gate_report["result_payload_run_id"],
        "json_gates": gate_report.get("gates", {}),
        "json_gates_all_true": gate_report.get("gates_all_true", False),
        "non_finite_metrics": gate_report.get("non_finite_metrics", []),
        "metadata_consistent": metadata_ok,
        "metadata_check": metadata,
        "jsonl_status": jsonl_status,
        "jsonl_valid": jsonl_valid,
        "jsonl_path": jsonl_path,
        "jsonl_check": jsonl_report,
        "child_pid": child_pid,
        "killed_pids": [] if kill_record is None else kill_record.get("killed_pids", []),
        "surviving_pids": [] if kill_record is None else kill_record.get("surviving_pids", []),
        "descendant_cleanup_verified": (None if kill_record is None
                                        else kill_record.get("descendant_cleanup_verified")),
        "child_ok": child_ok,
        "gates_ok": gates_ok,
        "outer_exit": outer_exit,
        "elapsed_s": elapsed,
        "child_stdout_log": child_stdout,
        "child_stderr_log": child_stderr,
        "out_dir": out_dir,
        "out_dir_basename": plan["out_dir_basename"],
        "out_dir_name_matches_run_id": plan["out_dir_name_matches_run_id"],
        "run_attempt": prior_attempt,
        "allow_reused_out_dir": bool(args.allow_reused_out_dir),
        "reused_out_dir": plan["code"] is not None,
        "out_dir_reuse_code": plan["code"],
        "prior_evidence": prior_evidence,
        "out_dir_marker": marker_path,
        "descendant_kill": kill_record,
        "residual_risk": _RESIDUAL_RISK if timed_out else None,
        "note": ("outer exit is non-zero unless the child returned 0 AND the JSON gate is PASS; "
                 "the raw child status is kept so Kit swallowing failures stays visible"),
    }
    summary, summary_non_finite = _json_safe(summary)
    summary["summary_non_finite_paths"] = summary_non_finite
    # allow_nan=False: a summary holding a bare NaN token is not parseable by a strict JSON
    # reader, so it would not be evidence at all.  _json_safe already stringified every
    # non-finite number and json_gate_status is forced INVALID when any existed.
    with open(os.path.join(out_dir, "gate_runner_summary.json"), "w", encoding="utf-8") as handle:
        json.dump(summary, handle, ensure_ascii=False, indent=1, allow_nan=False)
    print(json.dumps({k: summary[k] for k in
                      ("run_id", "child_returncode", "json_gate", "json_gate_status",
                       "outer_exit", "timed_out", "elapsed_s")},
                     ensure_ascii=False, indent=1))
    return outer_exit


if __name__ == "__main__":
    raise SystemExit(main())
