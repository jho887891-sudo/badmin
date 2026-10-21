#!/usr/bin/env python
"""A06: unique RUN_ID, immutable raw evidence, stale/inherited result.json fails closed.

Three contracts, all enforced *after the fact* on committed evidence and on CPU only:

1. RUN_ID UNIQUENESS.  Inside one runs directory every run_id is unique.  A run_id is
   the directory name; it must equal the run_id recorded in manifest.json and in
   result.json.  Two directories claiming the same run_id (including names that differ
   only by case) are a hard failure.

2. EVIDENCE IMMUTABILITY.  A committed ledger ({"runs": {run_id: {path: sha256}}}) is
   the authority.  A file whose content hash changed, a file that disappeared, or a
   new file that the ledger never saw makes the check fail.  Nothing is ever repaired
   in place: historical evidence is read-only.

3. STALE / INHERITED RESULT FAILS CLOSED.  A result.json whose run_id disagrees with
   its own directory, with its manifest, or with the requested --expect-run-id, or
   whose recorded manifest digest disagrees with the manifest next to it, is INVALID.
   A residual summary from a different run must never be able to produce a PASS.

Relation to p2_gate_runner.py (commit ce278c2, branch test/p0-gate-runner-strict-outdir)
--------------------------------------------------------------------------------------
The runner already owns the *write-time* side of exactly these rules; this module
does not re-implement them, it cites them:

* decide_out_dir_use  -> OUT_DIR_RUN_ID_MISMATCH / OUT_DIR_RESUME_SAME_RUN_ID /
  OUT_DIR_NAME_CONFLICT / OUT_DIR_NOT_EMPTY_UNRECOGNIZED (p2_gate_runner.py:185-238)
* read_json_gate      -> RUN_ID_MISMATCH when result.json run_id != --run-id
  (p2_gate_runner.py:602-605)
* _quarantine_stale_result -> the explicit rename escape hatch (p2_gate_runner.py:489)

The runner refuses *before* a child starts and needs the evidence on the same host.
This module is the audit half: it runs on a clean clone with no ISAAC, no GPU and no
prior knowledge, and re-derives uniqueness/immutability from content hashes alone,
so a reviewer who trusts neither the producer nor the runner can still check it.

Usage
-----
    python tools/queue/run_evidence.py ledger build --runs-root <dir> --out ledger.json
    python tools/queue/run_evidence.py check --runs-root <dir> --ledger ledger.json
    python tools/queue/run_evidence.py check --runs-root <dir> --expect-run-id G5L_r01

Exit codes: 0 = valid, 2 = contract violation, 1 = usage or IO error.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

SCHEMA_VERSION = 1
LEDGER_NAME = "evidence_ledger.json"

#: files whose presence makes a directory a run directory
RUN_MARKERS = ("manifest.json", "result.json")

#: files hashed by the ledger; logs are included so they cannot be rewritten either
LEDGER_SUFFIXES = (".json", ".jsonl", ".usda", ".usd", ".usdc", ".log", ".md", ".txt", ".csv")


# --------------------------------------------------------------------- helpers
def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_json(path: Path):
    """(payload, error) - never raises, because bad evidence must be reported."""
    try:
        with open(path, "r", encoding="utf-8") as handle:
            return json.load(handle), None
    except FileNotFoundError:
        return None, "missing"
    except (OSError, json.JSONDecodeError) as exc:
        return None, "unreadable: %s" % exc


def discover_run_dirs(runs_root: Path) -> list:
    """One level deep: every child directory that holds manifest.json or result.json."""
    root = Path(runs_root)
    if not root.is_dir():
        return []
    found = []
    for child in sorted(root.iterdir()):
        if not child.is_dir():
            continue
        if any((child / marker).is_file() for marker in RUN_MARKERS):
            found.append(child)
    return found


# ------------------------------------------------------------------ contract 1
def check_unique_run_ids(run_dirs) -> list:
    """No two directories may share a run_id; run_id must equal the directory name."""
    problems: list = []
    seen: dict = {}
    for run_dir in run_dirs:
        run_dir = Path(run_dir)
        key = run_dir.name.lower()
        if key in seen:
            problems.append({
                "code": "DUPLICATE_RUN_ID",
                "run_id": run_dir.name,
                "reason": "run_id %r is used by both %s and %s"
                          % (run_dir.name, seen[key].as_posix(), run_dir.as_posix()),
            })
        else:
            seen[key] = run_dir

    # two different directories may also claim the same run_id in their JSON
    claims: dict = {}
    for run_dir in run_dirs:
        for name in RUN_MARKERS:
            payload, error = load_json(Path(run_dir) / name)
            if error is not None or not isinstance(payload, dict):
                continue
            claimed = payload.get("run_id")
            if not isinstance(claimed, str) or not claimed:
                continue
            previous = claims.get(claimed)
            if previous is not None and Path(previous[0]).resolve() != Path(run_dir).resolve():
                problems.append({
                    "code": "DUPLICATE_RUN_ID",
                    "run_id": claimed,
                    "reason": "%s claims run_id %r, already claimed by %s/%s"
                              % ((Path(run_dir) / name).as_posix(), claimed,
                                 Path(previous[0]).as_posix(), previous[1]),
                })
            else:
                claims.setdefault(claimed, (run_dir, name))
    return problems


# ------------------------------------------------------------------ contract 3
def check_run_identity(run_dir, expect_run_id=None) -> list:
    """Stale / inherited evidence fails closed: every run_id must agree."""
    run_dir = Path(run_dir)
    problems: list = []
    manifest, manifest_error = load_json(run_dir / "manifest.json")
    result, result_error = load_json(run_dir / "result.json")
    manifest_run_id = None
    if isinstance(manifest, dict):
        manifest_run_id = manifest.get("run_id")
    result_run_id = None
    if isinstance(result, dict):
        result_run_id = result.get("run_id")

    if result_error == "missing":
        problems.append({
            "code": "RESULT_MISSING",
            "run_id": run_dir.name,
            "reason": "%s has no result.json" % run_dir.as_posix(),
        })
    elif result_error is not None:
        problems.append({
            "code": "RESULT_INVALID",
            "run_id": run_dir.name,
            "reason": "%s is %s" % ((run_dir / "result.json").as_posix(), result_error),
        })
    if manifest_error is not None:
        problems.append({
            "code": "MANIFEST_MISSING",
            "run_id": run_dir.name,
            "reason": "%s is %s" % ((run_dir / "manifest.json").as_posix(), manifest_error),
        })

    for label, value in (("result.json", result_run_id), ("manifest.json", manifest_run_id)):
        if isinstance(value, str) and value and value != run_dir.name:
            problems.append({
                "code": "STALE_RESULT",
                "run_id": run_dir.name,
                "reason": "%s in %s carries run_id %r, which is not this directory: the "
                          "result was inherited from a different run and must never be "
                          "read as this run's verdict"
                          % (label, run_dir.as_posix(), value),
            })
    if isinstance(result_run_id, str) and isinstance(manifest_run_id, str) \
            and result_run_id != manifest_run_id:
        problems.append({
            "code": "STALE_RESULT",
            "run_id": run_dir.name,
            "reason": "result.json run_id %r != manifest.json run_id %r in %s"
                      % (result_run_id, manifest_run_id, run_dir.as_posix()),
        })
    if expect_run_id is not None and run_dir.name != expect_run_id:
        problems.append({
            "code": "RUN_ID_MISMATCH",
            "run_id": run_dir.name,
            "reason": "directory %s is not the requested --expect-run-id %r"
                      % (run_dir.as_posix(), expect_run_id),
        })

    # a result that records its own manifest digest must agree with the manifest present
    recorded = None
    if isinstance(result, dict):
        recorded = (result.get("manifest_sha256") or result.get("inputs", {}).get("manifest_sha256")
                    if isinstance(result.get("inputs"), dict) else result.get("manifest_sha256"))
    if isinstance(recorded, str) and recorded and (run_dir / "manifest.json").is_file():
        actual = sha256_file(run_dir / "manifest.json")
        if recorded.replace("sha256:", "") != actual:
            problems.append({
                "code": "STALE_RESULT",
                "run_id": run_dir.name,
                "reason": "result.json records manifest_sha256 %s but manifest.json is %s: "
                          "the result does not belong to this manifest"
                          % (recorded, actual),
            })
    return problems


# ------------------------------------------------------------------ contract 2
def build_ledger(runs_root, suffixes=LEDGER_SUFFIXES) -> dict:
    """Committed hash manifest: the immutable record of what the evidence was."""
    root = Path(runs_root)
    runs = {}
    for run_dir in discover_run_dirs(root):
        files = {}
        for path in sorted(run_dir.rglob("*")):
            if not path.is_file() or path.suffix.lower() not in suffixes:
                continue
            files[path.relative_to(run_dir).as_posix()] = {
                "sha256": sha256_file(path),
                "bytes": path.stat().st_size,
            }
        runs[run_dir.name] = files
    return {
        "schema_version": SCHEMA_VERSION,
        "task": "A06",
        "runs_root": root.as_posix(),
        "runs": runs,
        "totals": {
            "runs": len(runs),
            "files": sum(len(files) for files in runs.values()),
        },
    }


def verify_ledger(runs_root, ledger, allow_unlisted=False,
                  suffixes=LEDGER_SUFFIXES) -> list:
    """Compare the evidence on disk against the committed ledger.  Never repairs."""
    root = Path(runs_root)
    problems: list = []
    expected_runs = (ledger or {}).get("runs") or {}
    present = {run_dir.name: run_dir for run_dir in discover_run_dirs(root)}

    for run_id in sorted(expected_runs):
        if run_id not in present:
            problems.append({
                "code": "EVIDENCE_REMOVED",
                "run_id": run_id,
                "reason": "run directory %s was recorded in the ledger but is gone"
                          % run_id,
            })
            continue
        run_dir = present[run_id]
        expected_files = expected_runs[run_id] or {}
        actual_files = {}
        for path in sorted(run_dir.rglob("*")):
            if not path.is_file() or path.suffix.lower() not in suffixes:
                continue
            actual_files[path.relative_to(run_dir).as_posix()] = path
        for rel, record in sorted(expected_files.items()):
            if rel not in actual_files:
                problems.append({
                    "code": "EVIDENCE_REMOVED",
                    "run_id": run_id,
                    "reason": "%s/%s was recorded with sha256 %s but is gone"
                              % (run_id, rel, record.get("sha256")),
                })
                continue
            actual = sha256_file(actual_files[rel])
            if actual != record.get("sha256"):
                problems.append({
                    "code": "EVIDENCE_MODIFIED",
                    "run_id": run_id,
                    "reason": "%s/%s sha256 is %s but the ledger recorded %s: committed "
                              "evidence was overwritten"
                              % (run_id, rel, actual, record.get("sha256")),
                })
        if not allow_unlisted:
            for rel in sorted(set(actual_files) - set(expected_files)):
                problems.append({
                    "code": "EVIDENCE_UNLISTED",
                    "run_id": run_id,
                    "reason": "%s/%s is not in the committed ledger: the evidence "
                              "directory grew after it was frozen" % (run_id, rel),
                })

    if not allow_unlisted:
        for run_id in sorted(set(present) - set(expected_runs)):
            problems.append({
                "code": "EVIDENCE_UNLISTED",
                "run_id": run_id,
                "reason": "run directory %s is not in the committed ledger" % run_id,
            })
    return problems


# ------------------------------------------------------------------ one call
def check_runs(runs_root, ledger=None, expect_run_id=None, allow_unlisted=False) -> tuple:
    run_dirs = discover_run_dirs(Path(runs_root))
    problems = check_unique_run_ids(run_dirs)
    for run_dir in run_dirs:
        problems.extend(check_run_identity(run_dir, expect_run_id=expect_run_id))
    if ledger is not None:
        problems.extend(verify_ledger(Path(runs_root), ledger, allow_unlisted=allow_unlisted))
    runs = {}
    for run_dir in run_dirs:
        result, _error = load_json(run_dir / "result.json")
        runs[run_dir.name] = {
            "run_id": run_dir.name,
            "path": run_dir.as_posix(),
            "gate_result": (result or {}).get("gate_result")
            if isinstance(result, dict) else None,
            "result_run_id": (result or {}).get("run_id")
            if isinstance(result, dict) else None,
        }
    return runs, problems


# ----------------------------------------------------------------------- CLI
def main(argv=None) -> int:
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding="utf-8", errors="replace")
        except (AttributeError, ValueError):
            pass
    parser = argparse.ArgumentParser(description="A06 run id + immutable evidence")
    parser.add_argument("mode", choices=("check", "ledger"))
    parser.add_argument("action", nargs="?", default="build", choices=("build", "verify"),
                        help="ledger sub-action (build writes, verify compares)")
    parser.add_argument("--runs-root", required=True)
    parser.add_argument("--ledger", default=None)
    parser.add_argument("--out", default=None)
    parser.add_argument("--expect-run-id", default=None)
    parser.add_argument("--allow-unlisted", action="store_true")
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args(argv)

    root = Path(args.runs_root)
    if not root.is_dir():
        print("FAIL (RUNS_ROOT_MISSING): %s" % root.as_posix(), file=sys.stderr)
        return 1

    if args.mode == "ledger":
        if args.action == "build" or args.out:
            ledger = build_ledger(root)
            target = Path(args.out) if args.out else (root / LEDGER_NAME)
            target.write_text(json.dumps(ledger, indent=2, sort_keys=True) + chr(10),
                              encoding="utf-8")
            print("ledger: %s runs %d files %d"
                  % (target.as_posix(), ledger["totals"]["runs"], ledger["totals"]["files"]))
            return 0
        ledger, error = load_json(Path(args.ledger)) if args.ledger else (None, "no --ledger")
        if error is not None:
            print("FAIL (LEDGER_UNREADABLE): %s" % error, file=sys.stderr)
            return 1
        problems = verify_ledger(root, ledger, allow_unlisted=args.allow_unlisted)
    else:
        ledger = None
        if args.ledger:
            ledger, error = load_json(Path(args.ledger))
            if error is not None:
                print("FAIL (LEDGER_UNREADABLE): %s" % error, file=sys.stderr)
                return 1
        runs, problems = check_runs(root, ledger=ledger, expect_run_id=args.expect_run_id,
                                    allow_unlisted=args.allow_unlisted)
        if args.json:
            print(json.dumps({"runs": runs, "problems": problems}, indent=2,
                             sort_keys=True, ensure_ascii=False))

    if problems:
        print("FAIL (A06_RUN_EVIDENCE): %d violation(s) under %s"
              % (len(problems), root.as_posix()), file=sys.stderr)
        for problem in problems:
            print("  - %s [%s]: %s" % (problem["code"], problem.get("run_id"),
                                       problem["reason"]), file=sys.stderr)
        return 2
    print("PASS (A06_RUN_EVIDENCE): %s" % root.as_posix())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
