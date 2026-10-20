"""External experiment runner: preserve Kit status and enforce explicit JSON gates.

No Isaac Sim dependency. The outer process is the authority for CI exit status.
A true PASS requires both a zero child return code and explicit passing JSON gates.
Use a NEW report path and output directory for each run; never overwrite evidence.

Example:
 python scripts/simulation/p0_subprocess_gate_runner.py \
   --report /tmp/run123/result.json --output-dir /tmp/run123/logs -- \
   /path/to/isaac-python scripts/simulation/morph_one_p2_init_diagnostics.py \
   --out /tmp/run123/result.json
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
from typing import Any


def inspect_gate(report: Path) -> dict[str, Any]:
    """Require at least one explicit PASS/FAIL indicator; fail closed on conflicts."""
    try:
        payload = json.loads(report.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        return {"status": "INVALID", "reason": f"Cannot read gate JSON: {exc}"}
    if not isinstance(payload, dict):
        return {"status": "INVALID", "reason": "Top-level JSON must be an object"}

    checks: list[tuple[str, bool]] = []
    for section in ("gates", "acceptance"):
        if section in payload:
            values = payload[section]
            if not isinstance(values, dict) or not values:
                return {"status": "INVALID", "reason": f"{section} must be a nonempty object"}
            for name, passed in values.items():
                if not isinstance(passed, bool):
                    return {"status": "INVALID",
                            "reason": f"{section}.{name} must be boolean"}
                checks.append((f"{section}.{name}", passed))

    if "gate_result" in payload:
        result = payload["gate_result"]
        if not isinstance(result, str) or result not in ("PASS", "FAIL"):
            return {"status": "INVALID", "reason": "gate_result must be PASS or FAIL"}
        checks.append(("gate_result", result == "PASS"))

    if not checks:
        return {"status": "INVALID",
                "reason": "Missing explicit gate_result, gates or acceptance"}

    failed = [name for name, passed in checks if not passed]
    return {"status": "FAIL" if failed else "PASS",
            "failed_checks": failed, "checked_fields": len(checks)}


def stop_group(proc: subprocess.Popen[Any]) -> None:
    """Terminate the launched process group when a Kit experiment times out."""
    if proc.poll() is not None:
        return
    if os.name == "posix":
        try:
            os.killpg(proc.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
    else:
        proc.kill()
    proc.wait()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--timeout-sec", type=float, default=900)
    parser.add_argument("command", nargs=argparse.REMAINDER)
    args = parser.parse_args(argv)

    command = args.command[1:] if args.command[:1] == ["--"] else args.command
    if not command:
        parser.error("Pass experiment command after --")
    if args.timeout_sec <= 0:
        parser.error("--timeout-sec must be positive")

    # Fail instead of deleting or overwriting any previous experiment/evidence.
    output_dir = args.output_dir.resolve()
    report = args.report.resolve()
    if report.exists() or output_dir.exists():
        parser.error("Existing evidence detected; select a fresh report/output directory")

    output_dir.mkdir(parents=True, exist_ok=False)
    report.parent.mkdir(parents=True, exist_ok=True)
    stdout_path = output_dir / "child.stdout.log"
    stderr_path = output_dir / "child.stderr.log"
    summary_path = output_dir / "gate_runner_summary.json"

    started = datetime.now(timezone.utc).isoformat()
    child_exit = None
    timed_out = False
    spawn_error = None
    with stdout_path.open("x", encoding="utf-8") as out, \
         stderr_path.open("x", encoding="utf-8") as err:
        try:
            proc = subprocess.Popen(command, stdout=out, stderr=err,
                                    start_new_session=(os.name == "posix"))
            try:
                child_exit = proc.wait(timeout=args.timeout_sec)
            except subprocess.TimeoutExpired:
                timed_out = True
                stop_group(proc)
                child_exit = proc.returncode
        except OSError as exc:
            spawn_error = f"{type(exc).__name__}: {exc}"

    gate = inspect_gate(report)
    outer_exit = (124 if timed_out else
                  0 if child_exit == 0 and gate["status"] == "PASS" else 1)
    summary = {
        "child_command": command,
        "child_returncode_observed": child_exit,
        "child_timed_out": timed_out,
        "spawn_error": spawn_error,
        "gate": gate,
        "outer_exit_code": outer_exit,
        "report_path": str(report),
        "stdout_path": str(stdout_path),
        "stderr_path": str(stderr_path),
        "started_utc": started,
        "finished_utc": datetime.now(timezone.utc).isoformat(),
    }
    with summary_path.open("x", encoding="utf-8") as handle:
        handle.write(json.dumps(summary, indent=2, ensure_ascii=False) + "\n")
    print(json.dumps(summary, indent=2, ensure_ascii=False), flush=True)
    return outer_exit


if __name__ == "__main__":
    sys.exit(main())
