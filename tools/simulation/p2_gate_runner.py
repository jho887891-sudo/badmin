"""P0 outer gate runner: a FAIL must never look like a PASS.

Kit can terminate the child process with status 0 even when Python raised (observed twice in
this project).  Therefore the acceptance decision is taken HERE, in a plain Python parent
process, from two independent inputs:

  1. the child process return code,
  2. the JSON gate written by the experiment itself.

The outer process exits non-zero unless BOTH the child returned 0 AND the JSON gate says PASS.
The original child status is preserved in the summary so the Kit behaviour stays auditable.

Fail-closed details (each one is regression tested in tests/simulation/test_p2_gate_runner.py):

  * a result.json left over from an earlier run in the same --out-dir is moved aside before
    the child starts, so a child that writes nothing can never inherit an old PASS;
  * --timeout is enforced; a child that overruns it is killed and reported as
    child_returncode=124 (the conventional timeout status) with timed_out=true.

Usage:
    python p2_gate_runner.py --script <experiment.py> --run-id R1 --out-dir <dir> -- --lift 0.4
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time

TIMEOUT_RETURNCODE = 124  # conventional "timed out" status, as in timeout(1)


def _as_text(value):
    """TimeoutExpired may carry bytes even when text mode was requested."""
    if value is None:
        return ""
    if isinstance(value, bytes):
        return value.decode("utf-8", "replace")
    return value


def _quarantine_stale_result(out_dir, result_path, run_id):
    """Move a previous run's result.json aside so the gate only reads THIS child's file.

    Without this a reused --out-dir is fail-OPEN: a child that exits 0 without writing
    result.json (exactly what Kit does when it swallows a failure) would be judged PASS by
    a stale file from an earlier successful run.
    """
    if not os.path.isfile(result_path):
        return None
    safe_run_id = "".join(c if c.isalnum() else "_" for c in str(run_id)) or "run"
    stamp = time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())
    target = os.path.join(out_dir, "result.json.prev-%s-%s.json" % (stamp, safe_run_id))
    os.replace(result_path, target)
    return target


def main(argv=None):
    parser = argparse.ArgumentParser(description="outer PASS/FAIL gate runner")
    parser.add_argument("--script", required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--timeout", type=float, default=1800.0)
    args, extra = parser.parse_known_args(argv)
    if extra and extra[0] == "--":
        extra = extra[1:]

    out_dir = os.path.abspath(args.out_dir)
    os.makedirs(out_dir, exist_ok=True)
    result_path = os.path.join(out_dir, "result.json")
    quarantined = _quarantine_stale_result(out_dir, result_path, args.run_id)
    command = [sys.executable, os.path.abspath(args.script),
               "--run-id", args.run_id, "--out-dir", out_dir] + list(extra)
    started = time.time()
    timed_out = False
    try:
        completed = subprocess.run(command, capture_output=True, text=True, timeout=args.timeout)
        child_returncode = completed.returncode
        stdout_text = completed.stdout or ""
        stderr_text = completed.stderr or ""
    except subprocess.TimeoutExpired as exc:
        timed_out = True
        child_returncode = TIMEOUT_RETURNCODE
        stdout_text = _as_text(exc.stdout)
        stderr_text = _as_text(exc.stderr) + (
            "\n[p2_gate_runner] child exceeded --timeout=%s s and was killed\n" % args.timeout)
    elapsed = round(time.time() - started, 3)

    child_stdout = os.path.join(out_dir, "gate_child.stdout.log")
    child_stderr = os.path.join(out_dir, "gate_child.stderr.log")
    with open(child_stdout, "w", encoding="utf-8") as handle:
        handle.write(stdout_text)
    with open(child_stderr, "w", encoding="utf-8") as handle:
        handle.write(stderr_text)

    gate = "MISSING"
    metrics = {}
    if os.path.isfile(result_path):
        with open(result_path, encoding="utf-8") as handle:
            payload = json.load(handle)
        gate = payload.get("gate_result", "MISSING")
        metrics = payload.get("metrics", {})

    child_ok = child_returncode == 0
    gates_ok = gate == "PASS"
    outer_exit = 0 if (child_ok and gates_ok) else 1
    summary = {
        "run_id": args.run_id,
        "command": " ".join(command),
        "child_returncode": child_returncode,
        "timed_out": timed_out,
        "timeout_s": args.timeout,
        "quarantined_stale_result": quarantined,
        "json_gate": gate,
        "json_metrics": metrics,
        "child_ok": child_ok,
        "gates_ok": gates_ok,
        "outer_exit": outer_exit,
        "elapsed_s": elapsed,
        "child_stdout_log": child_stdout,
        "child_stderr_log": child_stderr,
        "note": ("outer exit is non-zero unless the child returned 0 AND the JSON gate is PASS; "
                 "the raw child status is kept so Kit swallowing failures stays visible"),
    }
    with open(os.path.join(out_dir, "gate_runner_summary.json"), "w", encoding="utf-8") as handle:
        json.dump(summary, handle, ensure_ascii=False, indent=1)
    print(json.dumps({k: summary[k] for k in
                      ("run_id", "child_returncode", "json_gate", "outer_exit",
                       "timed_out", "elapsed_s")},
                     ensure_ascii=False, indent=1))
    return outer_exit


if __name__ == "__main__":
    raise SystemExit(main())
