# -*- coding: utf-8 -*-
"""A04: post-run audit of the PIDs a run recorded - READ ONLY, never kills anything.

Motivation
-----------
The runner kills the child's process tree (POSIX killpg / Windows taskkill /T) and records
WHY it believes the tree is gone.  But Kit grandchildren that double-detach or are hosted
by a service are outside that reach, so a run directory alone cannot prove the GPU is free.
This tool re-checks, after the run, every PID the evidence carries, with the exact probe
command in the report.  It signals nothing: the only liveness primitive is
os.kill(pid, 0) on POSIX (no signal delivered) or "tasklist /FI ..." on Windows, which is
also why it is safe on a shared host that other users are logged into.

What is scanned (run directory)
-------------------------------
  * every JSON file in the run directory (one level of sub-directories included), and
  * every *.txt file holding bare PIDs / comma-separated PID lists.
From those documents only PID-bearing keys are taken: pid, pids, child_pid(s),
grandchild_pids, descendant(s)_pid(s), descendants_before_kill, descendants_after_kill,
killed_pids, surviving_pids, survivor_pids, recorded_pids.  Numbers under any other key
(return codes, timings, sizes) are NOT treated as PIDs.

Exit codes
----------
  0  CLEAN            no recorded PID is alive
  1  USAGE/ERROR      bad arguments, missing run dir
  2  SURVIVORS        at least one recorded PID is still alive (fail-closed)
  3  NO_PID_RECORDS   the run directory carries no PID evidence at all = UNKNOWN

Usage
-----
    python tools/simulation/kit_descendant_audit.py --run-dir outputs/.../run_001
    python tools/simulation/kit_descendant_audit.py --run-dir ... --json-out audit.json
    python tools/simulation/kit_descendant_audit.py --run-dir ... --marker isaac_kit
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from gpu_lock import (  # noqa: E402  (same directory, shared read-only probes)
    cmdline_probe_command,
    liveness_probe_command,
    pid_alive,
    read_cmdline,
)

SCHEMA_VERSION = 1
DEFAULT_MARKER = "isaac"
MAX_PIDS = 512
MAX_FILE_BYTES = 4 * 1024 * 1024

EXIT_CLEAN = 0
EXIT_USAGE = 1
EXIT_SURVIVORS = 2
EXIT_NO_PID_RECORDS = 3

PID_KEYS = {
    "pid", "pids",
    "child_pid", "child_pids",
    "grandchild_pid", "grandchild_pids",
    "descendant_pid", "descendant_pids",
    "descendants_before_kill", "descendants_after_kill",
    "killed_pids", "surviving_pids", "survivor_pids", "recorded_pids",
}

CLASS_DEAD = "DEAD"
CLASS_SURVIVOR_OURS = "SURVIVOR_OURS"
CLASS_SURVIVOR_RECYCLED = "SURVIVOR_RECYCLED_OR_UNRELATED"
CLASS_SURVIVOR_UNVERIFIED = "SURVIVOR_UNVERIFIABLE"


def _coerce_pids(value):
    """PIDs out of an int / numeric string / list; anything else is ignored."""
    found = []
    if isinstance(value, bool):
        return found
    if isinstance(value, int):
        found.append(value)
    elif isinstance(value, str):
        text = value.strip()
        if text.isdigit():
            found.append(int(text))
    elif isinstance(value, (list, tuple)):
        for item in value:
            found.extend(_coerce_pids(item))
    return found


def _walk_pid_keys(node, out, trail=""):
    """Collect PID values from known keys only (never from return codes/timings)."""
    if isinstance(node, dict):
        for key, value in node.items():
            key_text = str(key).strip().lower()
            path = "%s.%s" % (trail, key) if trail else str(key)
            if key_text in PID_KEYS:
                for pid in _coerce_pids(value):
                    if 0 < pid <= 2 ** 31:
                        out.append((pid, path))
            elif isinstance(value, (dict, list)):
                _walk_pid_keys(value, out, path)
    elif isinstance(node, list):
        for index, item in enumerate(node):
            if isinstance(item, (dict, list)):
                _walk_pid_keys(item, out, "%s[%d]" % (trail, index))


def _text_file_pids(text, out, trail):
    for line in text.splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        for token in line.replace(",", " ").replace(";", " ").split():
            if token.isdigit():
                pid = int(token)
                if 0 < pid <= 2 ** 31:
                    out.append((pid, trail))


def discover_pid_records(run_dir, max_bytes=MAX_FILE_BYTES):
    """Return [(pid, source_path, json_path)] found under *run_dir*."""
    records = []
    candidates = sorted(run_dir.rglob("*"))
    for path in candidates:
        if not path.is_file():
            continue
        if path.suffix.lower() not in (".json", ".txt", ".csv", ".log"):
            continue
        try:
            if path.stat().st_size > max_bytes:
                continue
            text = path.read_text(encoding="utf-8", errors="replace")
        except OSError:
            continue
        suffix = path.suffix.lower()
        if suffix == ".json":
            try:
                doc = json.loads(text)
            except (ValueError, json.JSONDecodeError):
                continue
            found = []
            _walk_pid_keys(doc, found)
            for pid, json_path in found:
                records.append((pid, str(path), json_path))
        else:
            found = []
            _text_file_pids(text, found, path.name)
            for pid, json_path in found:
                records.append((pid, str(path), json_path))
    return records


def audit_run_dir(run_dir, marker=DEFAULT_MARKER, os_name=None, now=None,
                  max_pids=MAX_PIDS):
    os_name = os_name or os.name
    now = now or datetime.now(timezone.utc)
    run_dir = Path(run_dir)
    records = discover_pid_records(run_dir)
    sources = sorted({source for _pid, source, _trail in records})

    pids = []
    seen = set()
    for pid, source, trail in records:
        if pid in seen:
            for item in pids:
                if item["pid"] == pid:
                    item["records"].append({"source": source, "path": trail})
            continue
        if len(pids) >= max_pids:
            break
        seen.add(pid)
        alive = pid_alive(pid, os_name=os_name)
        cmdline = read_cmdline(pid, os_name=os_name) if alive else None
        marker_found = None if cmdline is None else (str(marker) in cmdline)
        if not alive:
            classification = CLASS_DEAD
        elif marker_found is True:
            classification = CLASS_SURVIVOR_OURS
        elif marker_found is False:
            classification = CLASS_SURVIVOR_RECYCLED
        else:
            classification = CLASS_SURVIVOR_UNVERIFIED
        pids.append({
            "pid": pid,
            "alive": bool(alive),
            "classification": classification,
            "marker_found": marker_found,
            "cmdline": cmdline,
            "liveness_probe": liveness_probe_command(pid, os_name=os_name),
            "cmdline_probe": cmdline_probe_command(pid, os_name=os_name),
            "records": [{"source": source, "path": trail}],
        })

    pids.sort(key=lambda item: item["pid"])
    survivors = [item for item in pids if item["alive"]]
    if not pids:
        verdict = "NO_PID_RECORDS"
    elif survivors:
        verdict = "SURVIVORS"
    else:
        verdict = "CLEAN"
    report = {
        "schema_version": SCHEMA_VERSION,
        "tool": "kit_descendant_audit",
        "run_dir": str(run_dir),
        "marker": marker,
        "os": os_name,
        "host": __import__("socket").gethostname(),
        "checked_at": now.isoformat(),
        "verdict": verdict,
        "pid_count": len(pids),
        "survivor_count": len(survivors),
        "survivors": [{"pid": item["pid"], "classification": item["classification"],
                       "liveness_probe": item["liveness_probe"]} for item in survivors],
        "sources": sources,
        "pids": pids,
        "kills_performed": 0,
        "note": ("read-only audit: no process was signalled; a live survivor needs a human "
                 "decision because the audit must not kill another user's process"),
    }
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(
        description="Post-run audit of recorded child/descendant PIDs (read-only)")
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--marker", default=DEFAULT_MARKER,
                        help="token expected in the survivor's cmdline")
    parser.add_argument("--json-out", default=None)
    parser.add_argument("--os", dest="os_name", default=None,
                        help="override the platform probe (nt/posix); diagnostics only")
    parser.add_argument("--max-pids", type=int, default=MAX_PIDS)
    args = parser.parse_args(argv)

    run_dir = Path(args.run_dir)
    if not run_dir.is_dir():
        sys.stderr.write("kit_descendant_audit: --run-dir is not a directory: %s\n" % run_dir)
        print(json.dumps({"schema_version": SCHEMA_VERSION, "verdict": "USAGE_ERROR",
                          "error": "run dir not found: %s" % run_dir}, indent=2))
        return EXIT_USAGE

    report = audit_run_dir(run_dir, marker=args.marker, os_name=args.os_name,
                           max_pids=args.max_pids)
    text = json.dumps(report, indent=2, sort_keys=True)
    sys.stdout.write(text + "\n")
    sys.stdout.flush()
    if args.json_out:
        out_path = Path(args.json_out)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(text + "\n", encoding="utf-8")

    if report["verdict"] == "NO_PID_RECORDS":
        return EXIT_NO_PID_RECORDS
    if report["verdict"] == "SURVIVORS":
        return EXIT_SURVIVORS
    return EXIT_CLEAN


if __name__ == "__main__":
    sys.exit(main())
