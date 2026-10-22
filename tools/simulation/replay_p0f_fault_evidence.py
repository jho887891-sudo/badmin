# -*- coding: utf-8 -*-
"""Offline replay of the six committed P0-07 fault runs through the R10-A classifier.

WHY: the classifier in tools/simulation/p2_gate_runner.py is a pure function of collected
evidence, so the claims made about the six real Kit fault runs can be re-checked against
the committed artifacts instead of being asserted from memory.

SOURCE OF TRUTH: branch p3/gate-v1.1 @ 6593736 (or whatever --ref resolves to), directory

    outputs/simulation/rebot_b601_rs/morph_one/p2_wc_gates/P0F_*/

The evidence is extracted with "git show <ref>:<path>" into a temporary directory, so the
committed run directories are NEVER modified, moved or written to.

Usage:
    python tools/simulation/replay_p0f_fault_evidence.py                 # table + exit code
    python tools/simulation/replay_p0f_fault_evidence.py --json          # machine readable
    python tools/simulation/replay_p0f_fault_evidence.py --ref p3/gate-v1.1 --repo <path>

Exit code: 0 only when EVERY run is classified exactly as expected; 1 otherwise.
"""
import argparse
import importlib.util
import json
import os
import shutil
import subprocess
import sys
import tempfile

P0F_RUNS = (
    "P0F_exception_r01",
    "P0F_exception_r02",
    "P0F_exception_r03",
    "P0F_jointlimit_r01",
    "P0F_jointlimit_r02",
    "P0F_jointlimit_r03",
)
EVIDENCE_SUBDIR = os.path.join("outputs", "simulation", "rebot_b601_rs", "morph_one",
                               "p2_wc_gates")
# Pinned commit, NOT the moving branch tip: p3/gate-v1.1 has moved on (R10-B work).
EVIDENCE_COMMIT = "65937365adaacb49b3d7e3e24ecdf38fda1ccf21"
DEFAULT_REF = EVIDENCE_COMMIT
EVIDENCE_FILES = ("gate_runner_summary.json", "outer.log", "fault_record.json",
                  "failure.json", "result.json", "manifest.json", "raw_poses.jsonl",
                  "gate_child.stdout.log", "gate_child.stderr.log")

# Expected verdicts, justified in management/morph_one_rs/R10_A_RUNNER_FAULT_SEMANTICS.md.
# committed_only: what the six runs COMMITTED artifacts can prove.
EXPECTED_COMMITTED = {
    "P0F_exception_r01": ("INCOMPLETE", "EXPECTED_FAULT_NOT_OBSERVED"),
    "P0F_exception_r02": ("EXPECTED_FAULT", "FAULT_RECORD_TYPED"),
    "P0F_exception_r03": ("EXPECTED_FAULT", "FAULT_RECORD_TYPED"),
    "P0F_jointlimit_r01": ("CRASH", "CRASH_ABNORMAL_TERMINATION"),
    "P0F_jointlimit_r02": ("EXPECTED_FAULT", "FAULT_RECORD_TYPED"),
    "P0F_jointlimit_r03": ("EXPECTED_FAULT", "FAULT_RECORD_TYPED"),
}
# With the ignored local stderr logs overlaid, the r01 traceback becomes evidence.
EXPECTED_WITH_LOGS = dict(EXPECTED_COMMITTED)
EXPECTED_WITH_LOGS["P0F_exception_r01"] = ("EXPECTED_FAULT", "TRACEBACK_ONLY")
EXPECTED = EXPECTED_COMMITTED


def load_runner(repo_root):
    """Import the gate runner by path (tools/ is not an installed package)."""
    path = os.path.join(repo_root, "tools", "simulation", "p2_gate_runner.py")
    spec = importlib.util.spec_from_file_location("p2_gate_runner_replay", path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def git_blob(repo_root, ref, relative_path):
    """Return the text of one committed blob, or None when the path does not exist."""
    try:
        completed = subprocess.run(["git", "-C", repo_root, "show",
                                    "%s:%s" % (ref, relative_path)],
                                   stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    except (OSError, subprocess.SubprocessError):
        return None
    if completed.returncode != 0:
        return None
    return completed.stdout.decode("utf-8", "replace")


def resolve_ref(repo_root, ref):
    completed = subprocess.run(["git", "-C", repo_root, "rev-parse", "--verify",
                                "%s^{commit}" % ref],
                               stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    if completed.returncode != 0:
        return None
    return completed.stdout.decode("utf-8", "replace").strip()


def materialize_run(repo_root, ref, run_id, target_dir):
    """Copy one committed run directory into target_dir.  Returns the written file names."""
    written = []
    os.makedirs(target_dir, exist_ok=True)
    for name in EVIDENCE_FILES:
        relative = "/".join([EVIDENCE_SUBDIR.replace(os.sep, "/"), run_id, name])
        text = git_blob(repo_root, ref, relative)
        if text is None:
            continue
        mode = "wb" if name.endswith(".jsonl") else "w"
        if mode == "wb":
            with open(os.path.join(target_dir, name), "wb") as handle:
                handle.write(text.encode("utf-8"))
        else:
            with open(os.path.join(target_dir, name), "w", encoding="utf-8",
                      newline="\n") as handle:
                handle.write(text)
        written.append(name)
    return written


def build_verdict(runner, run_dir, run_id, command="", timed_out=False):
    """Rebuild exactly what main() would collect, then classify.  No child is started."""
    stderr_path = os.path.join(run_dir, "gate_child.stderr.log")
    stderr_text = ""
    if os.path.isfile(stderr_path):
        with open(stderr_path, encoding="utf-8", errors="replace") as handle:
            stderr_text = handle.read()
    gate = runner.read_json_gate(os.path.join(run_dir, "result.json"),
                                 requested_run_id=run_id)
    metadata = runner.check_manifest_consistency(run_dir)
    jsonl = runner.check_pose_jsonl(os.path.join(run_dir, "raw_poses.jsonl"), None)
    jsonl_valid = bool(jsonl["valid"]) if jsonl["present"] else True
    fault_record = runner.read_fault_record(run_dir)
    summary = {}
    summary_path = os.path.join(run_dir, "gate_runner_summary.json")
    if os.path.isfile(summary_path):
        try:
            with open(summary_path, encoding="utf-8") as handle:
                summary = json.load(handle)
        except (OSError, ValueError):
            summary = {}
    if not command:
        command = summary.get("command", "")
    child_returncode = summary.get("child_returncode")
    evidence = runner.RunEvidence(
        run_id=run_id,
        child_returncode=child_returncode,
        timed_out=bool(summary.get("timed_out", timed_out)),
        stderr_text=stderr_text,
        gate=gate,
        metadata_consistent=bool(metadata["metadata_consistent"]),
        jsonl_valid=jsonl_valid,
        fault_record=fault_record,
        declared_faults=tuple(runner.declared_faults_from_command(command)),
    )
    verdict = runner.classify_run(evidence)
    return evidence, verdict, {
        "traceback_tier": verdict.traceback.get("tier"),
        "command": command,
        "child_returncode": child_returncode,
        "jsonl_present": jsonl["present"],
        "jsonl_valid": jsonl_valid,
        "jsonl_bytes": (os.path.getsize(os.path.join(run_dir, "raw_poses.jsonl"))
                        if jsonl["present"] else None),
        "result_json_present": os.path.isfile(os.path.join(run_dir, "result.json")),
        "fault_record_name": fault_record.get("name"),
        "declared_faults": list(evidence.declared_faults),
    }


LOG_OVERLAY_FILES = ("gate_child.stderr.log", "gate_child.stdout.log", "outer.log")


def overlay_logs(logs_dir, run_id, target_dir):
    """Copy the gate_child.*.log / outer.log files that git does not store (.gitignore:13).

    Returns the list of copied names; [] when the directory or the files are absent.  These
    files are NOT verifiable against the commit -- that limitation is reported as an UNKNOWN
    and the mode is printed with the table.
    """
    copied = []
    source_dir = os.path.join(logs_dir, EVIDENCE_SUBDIR, run_id)
    if not os.path.isdir(source_dir):
        return copied
    for name in LOG_OVERLAY_FILES:
        source = os.path.join(source_dir, name)
        if not os.path.isfile(source):
            continue
        with open(source, "rb") as src, open(os.path.join(target_dir, name), "wb") as dst:
            dst.write(src.read())
        copied.append(name)
    return copied


def replay(repo_root, ref=DEFAULT_REF, runs=P0F_RUNS, logs_dir=None):
    """Classify every committed run.  Returns (rows, all_expected, resolved_sha)."""
    sha = resolve_ref(repo_root, ref)
    if sha is None:
        raise RuntimeError("ref %r does not resolve in %r" % (ref, repo_root))
    runner = load_runner(repo_root)
    rows = []
    work_root = tempfile.mkdtemp(prefix="p0f_replay_")
    try:
        for run_id in runs:
            target = os.path.join(work_root, run_id)
            written = materialize_run(repo_root, sha, run_id, target)
            if not written:
                rows.append({"run_id": run_id, "error": "no evidence found at ref",
                             "files": []})
                continue
            overlay = overlay_logs(logs_dir, run_id, target) if logs_dir else []
            _, verdict, detail = build_verdict(runner, target, run_id)
            expected_table = EXPECTED_WITH_LOGS if overlay else EXPECTED_COMMITTED
            expected_outcome, expected_reason = expected_table[run_id]
            row = {"run_id": run_id,
                   "outcome": verdict.outcome.value,
                   "reason_code": verdict.reason_code,
                   "rule": verdict.rule,
                   "exit_code": verdict.exit_code,
                   "expected_outcome": expected_outcome,
                   "expected_reason_code": expected_reason,
                   "match": (verdict.outcome.value == expected_outcome
                             and verdict.reason_code == expected_reason),
                   "logs_overlaid": overlay,
                   "files": written}
            row.update(detail)
            rows.append(row)
    finally:
        shutil.rmtree(work_root, ignore_errors=True)
    return rows, all(row.get("match") for row in rows), sha


def format_table(rows):
    header = ("RUN_ID", "child_rc", "fault_record", "traceback", "outcome", "reason_code",
              "exit", "verdict")
    lines = ["%-22s %-9s %-16s %-20s %-16s %-32s %-5s %s" % header]
    lines.append("-" * 132)
    for row in rows:
        if row.get("error"):
            lines.append("%-22s %s" % (row["run_id"], row["error"]))
            continue
        lines.append("%-22s %-9s %-16s %-20s %-16s %-32s %-5s %s" % (
            row["run_id"], row["child_returncode"], row.get("fault_record_name") or "-",
            row.get("traceback_tier") or "-", row["outcome"],
            row["reason_code"], row["exit_code"],
            "MATCH" if row["match"] else "MISMATCH"))
    return "\n".join(lines)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", default=os.path.dirname(os.path.dirname(
        os.path.dirname(os.path.abspath(__file__)))))
    parser.add_argument("--ref", default=DEFAULT_REF)
    parser.add_argument("--logs-dir", default=os.environ.get("P0F_LOGS_DIR"),
                        help="checkout root that holds the ignored gate_child.*.log files "
                             "(optional; switches the run to the with-logs expectations)")
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args(argv)
    repo_root = os.path.abspath(args.repo)
    logs_dir = os.path.abspath(args.logs_dir) if args.logs_dir else None
    rows, all_expected, sha = replay(repo_root, args.ref, logs_dir=logs_dir)
    mode = "committed_only" if not any(row.get("logs_overlaid") for row in rows)         else "with_local_ignored_logs"
    if args.json:
        print(json.dumps({"ref": args.ref, "sha": sha, "mode": mode, "rows": rows,
                          "all_expected": all_expected}, ensure_ascii=False, indent=1))
    else:
        print("P0-07 fault evidence replay  ref=%s sha=%s  mode=%s" % (args.ref, sha, mode))
        print(format_table(rows))
        print("")
        print("all six classified as expected: %s" % ("YES" if all_expected else "NO"))
        print("none of them is SUCCESS: %s"
              % ("YES" if not any(row.get("outcome") == "SUCCESS" for row in rows) else "NO"))
    return 0 if all_expected else 1


if __name__ == "__main__":
    raise SystemExit(main())
