#!/usr/bin/env python
"""R10-C1/C2: typed evidence contracts for gate/fault runs (pure CPU).

The queue gate (tools/queue/check_consistency.py) asks one question: "is the
artifact there?"  This module asks the sharper one: "which *kind* of run is
this, and does its evidence set actually prove that claim?"

C1 - two disjoint contracts
    NORMAL  : manifest.json + result.json + raw_poses.jsonl + gate_runner_summary.json
    FAULT   : fault_record.json (typed code) + gate_runner_summary.json
              + at least one raw child log
    A run is FAULT only because it *declares* one (fault_record.json with a
    typed code), never because result.json happens to be missing.  So:
      * a real fault run is not rejected for lacking result.json;
      * an ordinary run that lacks result.json is still rejected.
    Declaring both result.json and fault_record.json is a contradiction and
    fails closed.

C2 - integrity of the evidence set
    * RUN_ID unique across the scanned set, equal to the directory name and to
      every run_id embedded in the run's own JSON;
    * parameter hash recomputed from the manifest's config block;
    * environment fingerprint present and non-empty;
    * raw child logs present, non-empty and *inside* the run directory;
    * every indexed file's SHA256 recomputed;
    * no stale/unindexed files, no cross-run references.

Exit codes: 0 = every run satisfies its contract, 1 = violations, 3 = the
checker itself could not run (fail closed).  Never starts Isaac/Kit; the
standard library only.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
from pathlib import Path

SCHEMA_VERSION = "1.0"

KIND_NORMAL = "NORMAL"
KIND_FAULT = "FAULT"

NORMAL_REQUIRED = ("manifest.json", "result.json", "raw_poses.jsonl", "gate_runner_summary.json")
FAULT_REQUIRED = ("fault_record.json", "gate_runner_summary.json")
CHILD_LOG_RE = re.compile(r"^(child|gate_child)_[A-Za-z0-9_.-]*\.(log|txt)$")
MANIFEST_INDEX_SELF = ("manifest.json",)

# A3 fault taxonomy: a fault record must name one of these, not a free string.
FAULT_CODES = (
    "EXPECTED_FAULT",
    "UNEXPECTED_FAULT",
    "CRASH",
    "TIMEOUT",
    "SIGSEGV",
    "KIT_PYTHON_ERROR",
)
FAULT_CLASSES = ("EXPECTED", "UNEXPECTED")

ENV_FINGERPRINT_KEYS = ("python_version", "platform", "sim_backend")

# violation codes -> human explanation (one place, so the report cannot drift)
CODES = {
    "KIND_UNDECLARED_UNPROVEN": "no result.json and no fault_record.json: a missing result never proves a fault",
    "NORMAL_INCOMPLETE": "an ordinary run is missing a required file",
    "FAULT_INCOMPLETE": "a fault run is missing a required file",
    "FAULT_RECORD_INVALID": "fault_record.json is not a typed fault record",
    "FAULT_CODE_NOT_TYPED": "fault record code is not in the A3 typed taxonomy",
    "KIND_CONFLICT": "result.json and fault_record.json coexist: the run claims to be both kinds",
    "FAULT_NO_RAW_LOG": "a fault run carries no raw child log, so the fault is not reproducible",
    "RUN_ID_DUPLICATE": "the same RUN_ID is declared by more than one run directory",
    "RUN_ID_DIR_MISMATCH": "declared RUN_ID differs from the run directory name",
    "CROSS_RUN_REFERENCE": "an artifact inside the run belongs to another RUN_ID",
    "PARAM_HASH_MISSING": "manifest declares no config block / config_sha256",
    "PARAM_HASH_MISMATCH": "config_sha256 does not match the recomputed config hash",
    "ENV_FINGERPRINT_MISSING": "environment fingerprint absent or incomplete",
    "RAW_LOG_MISSING": "declared raw child log is absent or empty",
    "LOG_OUTSIDE_RUN": "raw child log path points outside the run directory",
    "SHA256_INDEX_MISSING": "manifest carries no file index, so integrity cannot be checked",
    "FILE_MISSING": "indexed file is absent from the run directory",
    "SHA256_MISMATCH": "file content does not match the indexed SHA256",
    "STALE_UNINDEXED_FILE": "file present in the run directory is not in the manifest index",
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_json(value) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def parameter_hash(config) -> str:
    """SHA256 over the canonical JSON of the manifest config block."""
    return hashlib.sha256(canonical_json(config).encode("utf-8")).hexdigest()


class Violation:
    __slots__ = ("code", "run_id", "detail")

    def __init__(self, code: str, run_id: str, detail: str):
        if code not in CODES:
            raise KeyError("unknown violation code: %s" % code)
        self.code = code
        self.run_id = run_id
        self.detail = detail

    def __str__(self) -> str:
        return "%s [%s] %s: %s" % (self.code, self.run_id, CODES[self.code], self.detail)


# ------------------------------------------------------------------ C1: kinds


def read_json(path: Path):
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def classify_run(run_dir: Path):
    """Return ('FAULT' | 'NORMAL' | 'UNDECLARED', reasons).

    The only way to be FAULT is a fault_record.json.  A missing result.json on
    its own yields UNDECLARED, which the contract rule then rejects.
    """
    has_fault = (run_dir / "fault_record.json").is_file()
    has_result = (run_dir / "result.json").is_file()
    if has_fault and has_result:
        return "CONFLICT", ["result.json and fault_record.json coexist"]
    if has_fault:
        return KIND_FAULT, []
    if has_result:
        return KIND_NORMAL, []
    return "UNDECLARED", ["neither result.json nor fault_record.json"]


def check_contract_kind(run_dir: Path, violations: list) -> str:
    run_id = run_dir.name
    kind, _ = classify_run(run_dir)
    if kind == "CONFLICT":
        violations.append(Violation("KIND_CONFLICT", run_id,
                                    "both result.json and fault_record.json are present"))
        return kind
    if kind == "UNDECLARED":
        violations.append(Violation("KIND_UNDECLARED_UNPROVEN", run_id,
                                    "result.json absent and no fault_record.json declares a fault"))
        return kind
    if kind == KIND_FAULT:
        record = check_fault_record(run_dir, run_id, violations)
        for required in FAULT_REQUIRED:
            if not (run_dir / required).is_file():
                violations.append(Violation(
                    "FAULT_INCOMPLETE", run_id,
                    "fault run requires %s (result.json is deliberately NOT required)" % required))
        logs = sorted(p.name for p in run_dir.iterdir() if p.is_file() and CHILD_LOG_RE.match(p.name))
        if not logs:
            violations.append(Violation("FAULT_NO_RAW_LOG", run_id,
                                        "expected at least one child_*.log / gate_child_*.log"))
        if record is not None:
            for name in logs:
                if (run_dir / name).stat().st_size == 0:
                    violations.append(Violation("RAW_LOG_MISSING", run_id,
                                                "%s is empty" % name))
        # C1 negative direction: a fault run must not silently carry success data
        return kind
    for required in NORMAL_REQUIRED:
        if not (run_dir / required).is_file():
            violations.append(Violation(
                "NORMAL_INCOMPLETE", run_id,
                "ordinary run requires %s" % required))
    return kind


def check_fault_record(run_dir: Path, run_id: str, violations: list):
    path = run_dir / "fault_record.json"
    try:
        record = read_json(path)
    except (OSError, ValueError) as exc:
        violations.append(Violation("FAULT_RECORD_INVALID", run_id, "unreadable JSON: %s" % exc))
        return None
    problems = []
    if not isinstance(record, dict):
        problems.append("not a JSON object")
    else:
        if record.get("run_id") != run_id:
            problems.append("run_id %r != directory %r" % (record.get("run_id"), run_id))
        code = record.get("fault_code")
        if not isinstance(code, str) or not code.strip():
            problems.append("fault_code missing or not a string")
        elif code not in FAULT_CODES:
            violations.append(Violation(
                "FAULT_CODE_NOT_TYPED", run_id,
                "fault_code %r not in %s" % (code, list(FAULT_CODES))))
        if record.get("fault_class") not in FAULT_CLASSES:
            problems.append("fault_class %r not in %s" % (record.get("fault_class"), list(FAULT_CLASSES)))
        if "outer_exit" not in record:
            problems.append("outer_exit missing: a typed fault must record the process exit")
        if "child_returncode" not in record:
            problems.append("child_returncode missing: swallowed-child-rc faults must stay visible")
    for problem in problems:
        violations.append(Violation("FAULT_RECORD_INVALID", run_id, problem))
    return record if isinstance(record, dict) else None


# ------------------------------------------------------------------ C2: parts


def _run_id_fields(run_dir: Path):
    """(filename, declared_run_id) for every JSON file that carries a run_id."""
    found = []
    for path in sorted(run_dir.glob("*.json")):
        try:
            payload = read_json(path)
        except (OSError, ValueError):
            continue
        if isinstance(payload, dict) and isinstance(payload.get("run_id"), str):
            found.append((path.name, payload["run_id"]))
    return found


def check_run_id(run_dir: Path, violations: list):
    run_id = run_dir.name
    for name, declared in _run_id_fields(run_dir):
        if declared != run_id:
            violations.append(Violation(
                "CROSS_RUN_REFERENCE", run_id,
                "%s declares run_id %r" % (name, declared)))


def check_parameter_hash(run_dir: Path, manifest: dict, violations: list):
    run_id = run_dir.name
    config = manifest.get("config")
    declared = manifest.get("config_sha256")
    if not isinstance(config, dict) or not config:
        violations.append(Violation("PARAM_HASH_MISSING", run_id,
                                    "manifest has no non-empty config block"))
        return
    if not isinstance(declared, str) or not declared.strip():
        violations.append(Violation("PARAM_HASH_MISSING", run_id,
                                    "manifest has no config_sha256"))
        return
    recomputed = parameter_hash(config)
    if recomputed != declared.lower():
        violations.append(Violation(
            "PARAM_HASH_MISMATCH", run_id,
            "declared %s, recomputed %s" % (declared, recomputed)))


def check_env_fingerprint(run_dir: Path, manifest: dict, violations: list):
    run_id = run_dir.name
    fingerprint = manifest.get("env_fingerprint")
    if not isinstance(fingerprint, dict) or not fingerprint:
        violations.append(Violation("ENV_FINGERPRINT_MISSING", run_id,
                                    "manifest has no env_fingerprint object"))
        return
    missing = [key for key in ENV_FINGERPRINT_KEYS
               if not isinstance(fingerprint.get(key), str) or not fingerprint.get(key).strip()]
    if missing:
        violations.append(Violation("ENV_FINGERPRINT_MISSING", run_id,
                                    "env_fingerprint lacks %s" % missing))


def check_raw_logs(run_dir: Path, violations: list):
    """Declared child logs must exist inside the run dir and be non-empty."""
    run_id = run_dir.name
    summary_path = run_dir / "gate_runner_summary.json"
    if not summary_path.is_file():
        return
    try:
        summary = read_json(summary_path)
    except (OSError, ValueError) as exc:
        violations.append(Violation("RAW_LOG_MISSING", run_id,
                                    "gate_runner_summary.json unreadable: %s" % exc))
        return
    if not isinstance(summary, dict):
        violations.append(Violation("RAW_LOG_MISSING", run_id,
                                    "gate_runner_summary.json is not an object"))
        return
    declared = [summary.get(key) for key in ("child_stdout_log", "child_stderr_log")]
    declared = [value for value in declared if isinstance(value, str) and value.strip()]
    if not declared:
        violations.append(Violation(
            "RAW_LOG_MISSING", run_id,
            "gate_runner_summary.json declares no child_stdout_log/child_stderr_log"))
        return
    for value in declared:
        name = Path(value)
        if name.is_absolute() or len(name.parts) > 1:
            violations.append(Violation(
                "LOG_OUTSIDE_RUN", run_id,
                "%r is not a run-relative log path" % value))
            continue
        target = run_dir / name
        if not target.is_file() or target.stat().st_size == 0:
            violations.append(Violation("RAW_LOG_MISSING", run_id,
                                        "%s absent or empty" % name))


def check_file_index(run_dir: Path, manifest: dict, violations: list):
    run_id = run_dir.name
    index = manifest.get("files")
    if not isinstance(index, dict) or not index:
        violations.append(Violation("SHA256_INDEX_MISSING", run_id,
                                    "manifest carries no files{path: sha256} index"))
        return
    for rel, expected in sorted(index.items()):
        target = run_dir / rel
        if not target.is_file():
            violations.append(Violation("FILE_MISSING", run_id,
                                        "indexed file %s is absent" % rel))
            continue
        actual = sha256_file(target)
        if not isinstance(expected, str) or actual != expected.lower():
            violations.append(Violation(
                "SHA256_MISMATCH", run_id,
                "%s declared %s, actual %s" % (rel, expected, actual)))
    for path in sorted(run_dir.iterdir()):
        if not path.is_file() or path.name in MANIFEST_INDEX_SELF:
            continue
        if path.name not in index:
            violations.append(Violation(
                "STALE_UNINDEXED_FILE", run_id,
                "%s is not in the manifest index (stale or foreign)" % path.name))


def verify_run(run_dir: Path, violations=None) -> list:
    """Full C1+C2 verdict for one run directory."""
    violations = [] if violations is None else violations
    if not run_dir.is_dir():
        raise FileNotFoundError("run directory not found: %s" % run_dir)
    kind = check_contract_kind(run_dir, violations)
    manifest_path = run_dir / "manifest.json"
    if manifest_path.is_file():
        try:
            manifest = read_json(manifest_path)
        except (OSError, ValueError) as exc:
            violations.append(Violation("PARAM_HASH_MISSING", run_dir.name,
                                        "manifest.json unreadable: %s" % exc))
            manifest = None
        if isinstance(manifest, dict):
            check_run_id(run_dir, violations)
            check_parameter_hash(run_dir, manifest, violations)
            check_env_fingerprint(run_dir, manifest, violations)
            check_raw_logs(run_dir, violations)
            check_file_index(run_dir, manifest, violations)
            if isinstance(manifest.get("run_id"), str) and manifest["run_id"] != run_dir.name:
                violations.append(Violation(
                    "RUN_ID_DIR_MISMATCH", run_dir.name,
                    "manifest run_id %r" % manifest["run_id"]))
    return violations


def scan_discover(run_dir: Path):
    return run_dir.is_dir()


def verify_set(run_dirs) -> list:
    """Verify several runs and enforce RUN_ID uniqueness across them."""
    violations = []
    seen = {}
    for run_dir in run_dirs:
        verify_run(run_dir, violations)
        seen.setdefault(run_dir.name, []).append(str(run_dir))
    for run_id, paths in sorted(seen.items()):
        if len(paths) > 1:
            violations.append(Violation(
                "RUN_ID_DUPLICATE", run_id,
                "declared by %d directories: %s" % (len(paths), ", ".join(paths))))
    return violations


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(
        description="R10-C1/C2 typed evidence contract verifier (pure CPU).")
    parser.add_argument("--run-dir", action="append", default=[],
                        help="one run directory (repeatable)")
    parser.add_argument("--root", action="append", default=[],
                        help="parent directory whose immediate subdirectories are runs")
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args(argv)
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding="utf-8", errors="replace")
        except (AttributeError, ValueError):
            pass

    run_dirs = [Path(value) for value in args.run_dir]
    try:
        for root in args.root:
            base = Path(root)
            if not base.is_dir():
                print("error: root not found: %s" % root, file=sys.stderr)
                return 3
            run_dirs.extend(sorted(p for p in base.iterdir() if p.is_dir()))
        if not run_dirs:
            print("error: nothing to check (use --run-dir or --root)", file=sys.stderr)
            return 3
        missing = [d for d in run_dirs if not d.is_dir()]
        if missing:
            print("error: run directory not found: %s" % missing[0], file=sys.stderr)
            return 3
        violations = verify_set(run_dirs)
    except OSError as exc:
        print("error: %s" % exc, file=sys.stderr)
        return 3

    if args.json:
        print(json.dumps({
            "contracts": {"NORMAL": list(NORMAL_REQUIRED), "FAULT": list(FAULT_REQUIRED)},
            "runs": [str(d) for d in run_dirs],
            "ok": not violations,
            "violations": [{"code": v.code, "run_id": v.run_id, "detail": v.detail}
                           for v in violations],
        }, ensure_ascii=False, indent=2))
    else:
        for violation in violations:
            print("VIOLATION %s" % violation)
        if violations:
            print("%d violation(s) over %d run(s)" % (len(violations), len(run_dirs)))
        else:
            print("evidence contracts satisfied: %d run(s)" % len(run_dirs))
    return 0 if not violations else 1


if __name__ == "__main__":
    raise SystemExit(main())
