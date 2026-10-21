# -*- coding: utf-8 -*-
"""P0 regression: the outer PASS/FAIL gate must never let a FAIL look like a PASS.

Target: tools/simulation/p2_gate_runner.py

Contract under test::

    outer exit 0  <=>  child return code == 0  AND  result.json:gate_result == "PASS"

The load-bearing case is **child rc=0 with gate_result=FAIL**: Isaac Kit has twice
returned status 0 from a Python run that failed, so the parent process -- not the
child status -- has to decide acceptance.

Second contract layer (Issue #13 audit 2026-10-21T06:15: "--out-dir" reuse and timeout
honesty), all of it fail-closed:

  1. ONE RUN_ID = ONE EVIDENCE DIRECTORY.  An --out-dir that already holds evidence of an
     earlier attempt (result.json / gate_runner_summary.json / manifest.json /
     .p2_gate_run_owner.json with a different run_id) is REFUSED before any child starts,
     with a distinct non-zero exit code and a gate_runner_refusal.json record.  A reused
     directory can therefore never inherit a stale PASS.  The escape hatch
     --allow-reused-out-dir is the only way in, it is recorded in the summary, and it keeps
     the result.json quarantine.
  2. --timeout kills the child's whole process tree (POSIX killpg of the child's own
     session, Windows taskkill /F /T), and the kill is verified against a real grandchild
     process.  Kit grandchildren that detach remain UNKNOWN and are labelled as such.
  3. result.json is parsed fail-closed: malformed JSON, a non-object document, a missing
     gate_result, or a non-string gate_result is an INVALID gate with outer exit 1 -- never
     an exception escaping main().

No Isaac Sim and no GPU are touched: the "experiment" is a small generated Python
script that writes result.json and exits with a chosen code.  The only real
subprocesses are that fake child, its grandchild probe, and the gate runner itself.
"""
from __future__ import annotations

import importlib.util
import json
import os
import subprocess
import sys
import time
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
RUNNER_PATH = ROOT / "tools" / "simulation" / "p2_gate_runner.py"

# Tokens are substituted, not .format()ed, so the JSON braces in the payload stay literal.
# Raw string: the template carries a literal backslash-n that the GENERATED child must see.
_CHILD_TEMPLATE = r'''# -*- coding: utf-8 -*-
"""Generated fake P2 experiment: no Isaac Sim, no GPU, no torch.

Writes result.json (unless told not to) and exits with a fixed code so the outer
gate contract can be exercised without any simulator.  It can also write raw bytes
into result.json (malformed-JSON cases) and spawn a grandchild probe (timeout cases).
"""
import argparse
import json
import os
import subprocess
import sys
import time


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--fake-gate", default="__GATE__")
    parser.add_argument("--fake-write", default="__WRITE__")
    parser.add_argument("--fake-raw-result-file", default="")
    parser.add_argument("--fake-marker", default="")
    parser.add_argument("--fake-grandchild-script", default="")
    parser.add_argument("--fake-grandchild-pidfile", default="")
    parser.add_argument("--fake-grandchild-beat", default="")
    parser.add_argument("--fake-gates-json", default=__GATES__)
    parser.add_argument("--fake-extra-metrics-json", default="")
    parser.add_argument("--fake-manifest-json", default="")
    parser.add_argument("--fake-jsonl", default="")
    args, extra = parser.parse_known_args()
    out_dir = os.path.abspath(args.out_dir)
    os.makedirs(out_dir, exist_ok=True)
    if args.fake_marker:
        with open(args.fake_marker, "w", encoding="utf-8") as handle:
            handle.write("fake child started pid=%d\n" % os.getpid())
    if args.fake_grandchild_script:
        grandchild = subprocess.Popen([sys.executable, args.fake_grandchild_script,
                                       args.fake_grandchild_beat])
        with open(args.fake_grandchild_pidfile, "w", encoding="utf-8") as handle:
            handle.write(str(grandchild.pid))
        print("fake child: grandchild pid=%d" % grandchild.pid, flush=True)
    if args.fake_raw_result_file:
        with open(args.fake_raw_result_file, "rb") as src:
            raw = src.read()
        with open(os.path.join(out_dir, "result.json"), "wb") as dst:
            dst.write(raw)
    elif args.fake_write == "1":
        metrics = {"fake_metric": 1.5}
        if args.fake_extra_metrics_json:
            metrics.update(json.loads(args.fake_extra_metrics_json))
        payload = {
            "run_id": args.run_id,
            "gate_result": args.fake_gate,
            "gates": json.loads(args.fake_gates_json),
            "metrics": metrics,
            "child_extra_argv": list(extra),
        }
        with open(os.path.join(out_dir, "result.json"), "w", encoding="utf-8") as handle:
            json.dump(payload, handle, ensure_ascii=False, indent=1)
    if args.fake_manifest_json:
        with open(os.path.join(out_dir, "manifest.json"), "w", encoding="utf-8") as handle:
            handle.write(args.fake_manifest_json)
    if args.fake_jsonl:
        with open(os.path.join(out_dir, "raw_poses.jsonl"), "w", encoding="utf-8") as handle:
            handle.write(args.fake_jsonl)
    print("fake child: gate=%s write=%s" % (args.fake_gate, args.fake_write), flush=True)
    print("fake child stderr marker", file=sys.stderr, flush=True)
    time.sleep(__SLEEP__)
    return __RC__


if __name__ == "__main__":
    raise SystemExit(main())
'''

_GRANDCHILD_TEMPLATE = '''# -*- coding: utf-8 -*-
"""Grandchild probe: appends to a heartbeat file forever, so "is it still running?"
is measurable from the outside (file growth), not just inferable from a PID."""
import sys
import time

beat = sys.argv[1]
while True:
    with open(beat, "a", encoding="utf-8") as handle:
        handle.write("x")
        handle.flush()
    time.sleep(0.1)
'''


@pytest.fixture(scope="module")
def runner():
    """Import the gate runner by path (tools/ is not an installed package)."""
    assert RUNNER_PATH.is_file(), "gate runner missing: %s" % RUNNER_PATH
    spec = importlib.util.spec_from_file_location("p2_gate_runner_under_test", RUNNER_PATH)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def write_fake_child(tmp_path, returncode=0, gate="PASS", write_result=True, sleep_s=0.0):
    script = tmp_path / "fake_experiment.py"
    script.write_text(
        _CHILD_TEMPLATE
        .replace("__RC__", str(int(returncode)))
        .replace("__GATE__", gate)
        .replace("__WRITE__", "1" if write_result else "0")
        .replace("__GATES__", repr(json.dumps({"fake_self_check": True})))
        .replace("__SLEEP__", repr(float(sleep_s))),
        encoding="utf-8",
    )
    return script


def invoke(runner, script, out_dir, run_id="R1", extra=(), timeout=None,
           allow_reused_out_dir=False, gravity_z=None, steps=None, dt=None,
           expected_frames=None, require_jsonl=False):
    """Call main() in-process so the returned outer exit code is directly assertable."""
    argv = ["--script", str(script), "--run-id", run_id, "--out-dir", str(out_dir)]
    if timeout is not None:
        argv += ["--timeout", str(timeout)]
    if allow_reused_out_dir:
        argv += ["--allow-reused-out-dir"]
    if gravity_z is not None:
        argv += ["--gravity-z", str(gravity_z)]
    if steps is not None:
        argv += ["--steps", str(steps)]
    if dt is not None:
        argv += ["--dt", repr(float(dt))]
    if expected_frames is not None:
        argv += ["--expected-frames", str(expected_frames)]
    if require_jsonl:
        argv += ["--require-jsonl"]
    if extra:
        argv += ["--"] + [str(item) for item in extra]
    return runner.main(argv)


def read_summary(out_dir):
    path = Path(out_dir) / "gate_runner_summary.json"
    assert path.is_file(), "gate_runner_summary.json not written: %s" % path
    return json.loads(path.read_text(encoding="utf-8"))


def read_refusal(out_dir):
    path = Path(out_dir) / "gate_runner_refusal.json"
    assert path.is_file(), "gate_runner_refusal.json not written: %s" % path
    return json.loads(path.read_text(encoding="utf-8"))


def pid_alive(pid):
    """Independent liveness probe (never imported from the runner under test).

    os.kill(pid, 0) is NOT usable on Windows: os.kill there terminates the process.
    """
    if os.name == "nt":
        completed = subprocess.run(
            ["tasklist", "/FI", "PID eq %d" % int(pid), "/FO", "CSV", "/NH"],
            capture_output=True, text=True)
        return ('"%d"' % int(pid)) in (completed.stdout or "")
    try:
        os.kill(int(pid), 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


# --------------------------------------------------------------------------------------
# 1. Truth table of the two independent inputs (the core contract)
# --------------------------------------------------------------------------------------
@pytest.mark.parametrize(
    "child_rc, json_gate, expected_outer",
    [
        (0, "PASS", 0),   # genuine success
        (0, "FAIL", 1),   # Kit swallowed the failure -> MUST still be rejected
        (1, "PASS", 1),   # the JSON says PASS but the child died -> rejected
        (1, "FAIL", 1),   # both inputs agree on failure
    ],
    ids=["rc0_gatePASS", "rc0_gateFAIL_kit_swallowed", "rc1_gatePASS", "rc1_gateFAIL"],
)
def test_outer_exit_is_pass_only_when_child_ok_and_gate_pass(
        runner, tmp_path, child_rc, json_gate, expected_outer):
    out_dir = tmp_path / ("out_%d_%s" % (child_rc, json_gate))
    script = write_fake_child(tmp_path, returncode=child_rc, gate=json_gate)

    outer_exit = invoke(runner, script, out_dir)

    assert outer_exit == expected_outer
    summary = read_summary(out_dir)
    assert summary["child_returncode"] == child_rc
    assert summary["json_gate"] == json_gate
    assert summary["child_ok"] is (child_rc == 0)
    assert summary["gates_ok"] is (json_gate == "PASS")
    assert summary["outer_exit"] == expected_outer


def test_kit_swallowed_failure_is_rejected_even_though_the_child_exited_zero(runner, tmp_path):
    """The single most important assertion in this file (see module docstring)."""
    out_dir = tmp_path / "swallowed"
    script = write_fake_child(tmp_path, returncode=0, gate="FAIL")

    outer_exit = invoke(runner, script, out_dir)

    assert outer_exit == 1
    summary = read_summary(out_dir)
    assert summary["child_returncode"] == 0      # Kit behaviour stays visible/auditable
    assert summary["json_gate"] == "FAIL"
    assert summary["outer_exit"] == 1


# --------------------------------------------------------------------------------------
# 2. Missing result.json -> fail closed, marker "MISSING"
# --------------------------------------------------------------------------------------
def test_missing_result_json_exits_one_and_marks_gate_missing(runner, tmp_path):
    out_dir = tmp_path / "no_result"
    script = write_fake_child(tmp_path, returncode=0, write_result=False)

    outer_exit = invoke(runner, script, out_dir)

    assert outer_exit == 1
    summary = read_summary(out_dir)
    assert summary["json_gate"] == "MISSING"
    assert summary["child_returncode"] == 0
    assert summary["json_metrics"] == {}
    assert summary["outer_exit"] == 1


# --------------------------------------------------------------------------------------
# 3. The raw child status is preserved verbatim in the summary
# --------------------------------------------------------------------------------------
@pytest.mark.parametrize("child_rc", [0, 1, 3, 7])
def test_summary_preserves_raw_child_returncode(runner, tmp_path, child_rc):
    out_dir = tmp_path / ("preserve_%d" % child_rc)
    script = write_fake_child(tmp_path, returncode=child_rc, gate="PASS")

    invoke(runner, script, out_dir)

    summary = read_summary(out_dir)
    assert summary["child_returncode"] == child_rc
    assert summary["run_id"] == "R1"
    # summary["command"] is the CHILD command line (this is what gets audited),
    # and the raw return code must be the child's own value, not a boolean or a cap.
    assert str(Path(script).resolve()) in summary["command"]
    assert summary["timed_out"] is False
    assert summary["timeout_s"] == 1800.0  # documented default survives untouched
    assert summary["outer_exit"] == (0 if child_rc == 0 else 1)
    # nothing was killed and nothing was reused: the audit fields must say so
    assert summary["descendant_kill"] is None
    assert summary["residual_risk"] is None
    assert summary["allow_reused_out_dir"] is False
    assert summary["reused_out_dir"] is False
    assert summary["run_attempt"] == 1


# --------------------------------------------------------------------------------------
# 4. The child's own streams are kept as evidence
# --------------------------------------------------------------------------------------
def test_child_stdout_and_stderr_are_captured_to_logs(runner, tmp_path):
    out_dir = tmp_path / "logs"
    script = write_fake_child(tmp_path, returncode=0, gate="PASS")

    invoke(runner, script, out_dir)

    summary = read_summary(out_dir)
    stdout_log = Path(summary["child_stdout_log"]).read_text(encoding="utf-8")
    stderr_log = Path(summary["child_stderr_log"]).read_text(encoding="utf-8")
    assert "fake child: gate=PASS write=1" in stdout_log
    assert "fake child stderr marker" in stderr_log
    # the JSON metrics written by the child survive into the summary
    assert summary["json_metrics"] == {"fake_metric": 1.5}


# --------------------------------------------------------------------------------------
# 5. Arguments after "--" reach the child unchanged (the documented calling convention)
# --------------------------------------------------------------------------------------
def test_arguments_after_separator_are_forwarded_to_the_child(runner, tmp_path):
    out_dir = tmp_path / "forward"
    script = write_fake_child(tmp_path, returncode=0, gate="PASS")

    outer_exit = invoke(runner, script, out_dir, extra=["--lift", "0.40", "--write-data"])

    assert outer_exit == 0
    payload = json.loads((out_dir / "result.json").read_text(encoding="utf-8"))
    assert payload["child_extra_argv"] == ["--lift", "0.40", "--write-data"]


# --------------------------------------------------------------------------------------
# 6. The real process exit code (not just the return value of main())
# --------------------------------------------------------------------------------------
@pytest.mark.parametrize(
    "child_rc, json_gate, expected_exit",
    [(0, "PASS", 0), (0, "FAIL", 1), (1, "PASS", 1)],
    ids=["rc0_gatePASS", "rc0_gateFAIL", "rc1_gatePASS"],
)
def test_real_process_exit_code(tmp_path, child_rc, json_gate, expected_exit):
    out_dir = tmp_path / ("cli_%d_%s" % (child_rc, json_gate))
    script = write_fake_child(tmp_path, returncode=child_rc, gate=json_gate)
    argv = [sys.executable, str(RUNNER_PATH), "--script", str(script),
            "--run-id", "CLI1", "--out-dir", str(out_dir)]

    completed = subprocess.run(argv, capture_output=True, text=True)

    assert completed.returncode == expected_exit, completed.stderr
    printed = json.loads(completed.stdout)
    assert printed["run_id"] == "CLI1"
    assert printed["child_returncode"] == child_rc
    assert printed["json_gate"] == json_gate
    assert printed["outer_exit"] == expected_exit
    assert printed["timed_out"] is False


# --------------------------------------------------------------------------------------
# 7. --timeout is enforced (regression: the option used to be parsed and then ignored,
#    so a hung Kit child blocked the gate forever and wrote no summary at all)
# --------------------------------------------------------------------------------------
def test_timeout_is_enforced_and_reported(runner, tmp_path):
    out_dir = tmp_path / "timeout"
    # The child writes a PASS result.json and THEN hangs: only the timeout can reject it.
    script = write_fake_child(tmp_path, returncode=0, gate="PASS", sleep_s=20.0)

    outer_exit = invoke(runner, script, out_dir, timeout=1.0)

    assert outer_exit == 1
    summary = read_summary(out_dir)
    assert summary["timed_out"] is True
    assert summary["child_returncode"] == 124
    assert summary["json_gate"] == "PASS"      # the JSON gate alone would have said PASS
    assert summary["outer_exit"] == 1
    assert summary["elapsed_s"] < 30.0
    assert "exceeded --timeout" in Path(summary["child_stderr_log"]).read_text(encoding="utf-8")
    # the kill is not just attempted, it is verified and auditable
    kill = summary["descendant_kill"]
    assert kill["attempted"] is True
    assert kill["child_terminated"] is True
    assert kill["verified_terminated"] is True
    assert "UNKNOWN" in summary["residual_risk"]


# --------------------------------------------------------------------------------------
# 8. A stale result.json must not pass a child that writes nothing (used to be fail-OPEN:
#    out_dir reused from an earlier successful run -> old PASS inherited by a silent child)
#    Since the strict contract this path is only reachable via --allow-reused-out-dir.
# --------------------------------------------------------------------------------------
def test_stale_result_json_is_quarantined_and_cannot_pass_a_silent_child(runner, tmp_path):
    out_dir = tmp_path / "stale"
    out_dir.mkdir()
    stale = {"run_id": "OLD", "gate_result": "PASS", "metrics": {"old_metric": 1}}
    (out_dir / "result.json").write_text(json.dumps(stale), encoding="utf-8")
    script = write_fake_child(tmp_path, returncode=0, write_result=False)

    outer_exit = invoke(runner, script, out_dir, allow_reused_out_dir=True)

    assert outer_exit == 1
    summary = read_summary(out_dir)
    assert summary["json_gate"] == "MISSING"
    assert summary["child_returncode"] == 0
    backup = Path(summary["quarantined_stale_result"])
    assert backup.is_file()
    assert json.loads(backup.read_text(encoding="utf-8")) == stale
    assert not (out_dir / "result.json").exists()
    # the escape hatch is recorded, never silent
    assert summary["allow_reused_out_dir"] is True
    assert summary["reused_out_dir"] is True
    assert summary["out_dir_reuse_code"] == "OUT_DIR_RUN_ID_MISMATCH"
    assert any(item["file"] == "result.json" and item["run_id"] == "OLD"
               for item in summary["prior_evidence"])


def test_fresh_run_quarantines_nothing(runner, tmp_path):
    out_dir = tmp_path / "fresh"
    script = write_fake_child(tmp_path, returncode=0, gate="PASS")

    assert invoke(runner, script, out_dir) == 0

    summary = read_summary(out_dir)
    assert summary["quarantined_stale_result"] is None
    assert summary["reused_out_dir"] is False
    assert summary["out_dir_reuse_code"] is None
    assert summary["prior_evidence"] == []


# ======================================================================================
# 9. STRICT RUN-DIRECTORY CONTRACT (Issue #13 audit 2026-10-21T06:15)
#    One run_id = one evidence directory.  A used directory is REFUSED before any child
#    starts, with a distinct exit code, and the refusal never rewrites the old evidence.
# ======================================================================================
@pytest.mark.parametrize(
    "kind",
    ["result.json", "gate_runner_summary.json", "manifest.json", ".p2_gate_run_owner.json"],
)
def test_out_dir_with_foreign_evidence_is_refused_before_any_child_starts(runner, tmp_path, kind):
    """Every evidence kind with a different run_id locks the directory (exit 2)."""
    run_id = "R1"
    # basename != run_id on purpose: this is the reuse case the audit complained about
    out_dir = tmp_path / ("reused_%s" % kind.replace(".", "_"))
    out_dir.mkdir(parents=True, exist_ok=True)
    foreign = out_dir / kind
    foreign.write_text(json.dumps({"run_id": "SOME_OTHER_RUN", "gate_result": "PASS",
                                   "metrics": {"old_metric": 1}}), encoding="utf-8")
    evidence_bytes = foreign.read_bytes()
    script = write_fake_child(tmp_path, returncode=0, gate="PASS")
    child_marker = tmp_path / "child_ran.txt"

    outer_exit = invoke(runner, script, out_dir, run_id=run_id,
                        extra=["--fake-marker", str(child_marker)])

    assert outer_exit == 2, "a foreign run_id must refuse with the mismatch code"
    # 1. the child NEVER ran
    assert not child_marker.exists(), "the gate runner started the child despite refusing"
    # 2. no fresh logs, and the old evidence is byte-identical
    if kind == "gate_runner_summary.json":
        # this kind IS the pre-existing file: it must survive untouched, not disappear
        assert (out_dir / "gate_runner_summary.json").read_bytes() == evidence_bytes
    else:
        assert not (out_dir / "gate_runner_summary.json").exists()
    assert not (out_dir / "gate_child.stdout.log").exists()
    assert not (out_dir / "gate_child.stderr.log").exists()
    assert foreign.read_bytes() == evidence_bytes
    # 3. the refusal itself is auditable
    refusal = read_refusal(out_dir)
    assert refusal["refused"] is True
    assert refusal["refusal_code"] == "OUT_DIR_RUN_ID_MISMATCH"
    assert refusal["refusal_exit"] == 2
    assert refusal["child_started"] is False
    assert refusal["requested_run_id"] == run_id
    assert "SOME_OTHER_RUN" in refusal["message"]
    assert any(item["file"] == kind and item["run_id"] == "SOME_OTHER_RUN"
               for item in refusal["evidence"])


def test_foreign_pass_result_json_cannot_be_inherited_as_pass(runner, tmp_path):
    """The exact fail-open the audit raised: an old PASS sitting in a reused --out-dir."""
    out_dir = tmp_path / "old_pass_dir"
    out_dir.mkdir()
    stale = {"run_id": "OLD_PASS_RUN", "gate_result": "PASS", "metrics": {"m": 1}}
    (out_dir / "result.json").write_text(json.dumps(stale), encoding="utf-8")
    # a child that exits 0 and writes nothing is exactly the Kit failure mode
    script = write_fake_child(tmp_path, returncode=0, write_result=False)

    outer_exit = invoke(runner, script, out_dir, run_id="R1")

    assert outer_exit != 0
    assert outer_exit == 2
    assert not (out_dir / "gate_runner_summary.json").exists()
    # the stale PASS is untouched and still visible -- but it never became this run's verdict
    assert json.loads((out_dir / "result.json").read_text(encoding="utf-8")) == stale


def test_same_run_id_partial_resume_is_refused_with_its_own_code(runner, tmp_path):
    """Resume of a partially written SAME run_id is refused, with a distinct error code."""
    out_dir = tmp_path / "R1"          # basename == run_id, so this is a resume, not a rename
    out_dir.mkdir()
    partial = {"run_id": "R1", "gate_result": "FAIL"}
    (out_dir / "result.json").write_text(json.dumps(partial), encoding="utf-8")
    script = write_fake_child(tmp_path, returncode=0, gate="PASS")
    child_marker = tmp_path / "child_ran.txt"

    outer_exit = invoke(runner, script, out_dir, run_id="R1",
                        extra=["--fake-marker", str(child_marker)])

    assert outer_exit == 3
    assert not child_marker.exists()
    assert not (out_dir / "gate_runner_summary.json").exists()
    refusal = read_refusal(out_dir)
    assert refusal["refusal_code"] == "OUT_DIR_RESUME_SAME_RUN_ID"
    assert refusal["refusal_exit"] == 3
    assert refusal["out_dir_name_matches_run_id"] is True
    assert "allow-reused-out-dir" in refusal["message"]


def test_out_dir_basename_conflict_with_reuse_marker_is_refused(runner, tmp_path):
    """basename != run_id while the marker says the directory was reused (exit 4)."""
    out_dir = tmp_path / "not_the_run_id"
    out_dir.mkdir()
    (out_dir / ".p2_gate_run_owner.json").write_text(
        json.dumps({"run_id": "R1", "attempt": 1}), encoding="utf-8")
    script = write_fake_child(tmp_path, returncode=0, gate="PASS")
    child_marker = tmp_path / "child_ran.txt"

    outer_exit = invoke(runner, script, out_dir, run_id="R1",
                        extra=["--fake-marker", str(child_marker)])

    assert outer_exit == 4
    assert not child_marker.exists()
    refusal = read_refusal(out_dir)
    assert refusal["refusal_code"] == "OUT_DIR_NAME_CONFLICT"
    assert refusal["refusal_exit"] == 4
    assert refusal["out_dir_basename"] == "not_the_run_id"
    assert refusal["out_dir_name_matches_run_id"] is False


def test_unrecognized_leftovers_in_out_dir_are_refused(runner, tmp_path):
    """Non-empty directory with no parseable run artifact: refuse rather than guess."""
    out_dir = tmp_path / "leftovers"
    out_dir.mkdir()
    (out_dir / "gate_child.stdout.log").write_text("old run output\n", encoding="utf-8")
    script = write_fake_child(tmp_path, returncode=0, gate="PASS")
    child_marker = tmp_path / "child_ran.txt"

    outer_exit = invoke(runner, script, out_dir, run_id="R1",
                        extra=["--fake-marker", str(child_marker)])

    assert outer_exit == 5
    assert not child_marker.exists()
    refusal = read_refusal(out_dir)
    assert refusal["refusal_code"] == "OUT_DIR_NOT_EMPTY_UNRECOGNIZED"
    assert refusal["refusal_exit"] == 5
    assert [item["file"] for item in refusal["leftovers"]] == ["gate_child.stdout.log"]


def test_allow_reused_out_dir_is_the_only_way_in_and_is_recorded(runner, tmp_path):
    """Escape hatch: same directory, explicit flag, and the fresh PASS is auditable."""
    out_dir = tmp_path / "escaped"
    out_dir.mkdir()
    (out_dir / "result.json").write_text(
        json.dumps({"run_id": "OLD", "gate_result": "PASS"}), encoding="utf-8")
    script = write_fake_child(tmp_path, returncode=0, gate="PASS")

    outer_exit = invoke(runner, script, out_dir, allow_reused_out_dir=True)

    assert outer_exit == 0
    summary = read_summary(out_dir)
    assert summary["allow_reused_out_dir"] is True
    assert summary["reused_out_dir"] is True
    assert summary["run_attempt"] == 1          # no marker existed yet -> first attempt
    assert summary["quarantined_stale_result"] is not None
    assert json.loads((out_dir / "result.json").read_text(encoding="utf-8"))["run_id"] == "R1"


def test_real_process_exit_code_for_refusal(tmp_path):
    """The refusal must be a real non-zero OS exit code, not just a return value."""
    out_dir = tmp_path / "cli_refused"
    out_dir.mkdir()
    stale = {"run_id": "OLD_CLI", "gate_result": "PASS"}
    (out_dir / "result.json").write_text(json.dumps(stale), encoding="utf-8")
    script = write_fake_child(tmp_path, returncode=0, gate="PASS")
    argv = [sys.executable, str(RUNNER_PATH), "--script", str(script),
            "--run-id", "CLI1", "--out-dir", str(out_dir)]

    completed = subprocess.run(argv, capture_output=True, text=True)

    assert completed.returncode == 2, completed.stderr
    printed = json.loads(completed.stdout)
    assert printed["refused"] is True
    assert printed["refusal_code"] == "OUT_DIR_RUN_ID_MISMATCH"
    assert printed["child_started"] is False
    assert "REFUSED" in completed.stderr
    assert not (out_dir / "gate_runner_summary.json").exists()
    assert json.loads((out_dir / "result.json").read_text(encoding="utf-8")) == stale


def test_second_attempt_in_the_same_dir_is_refused_even_after_a_success(runner, tmp_path):
    """A PASS directory is still a used directory: rerunning into it refuses (exit 3)."""
    out_dir = tmp_path / "R7"          # basename == run_id so the code is "resume"
    script = write_fake_child(tmp_path, returncode=0, gate="PASS")
    assert invoke(runner, script, out_dir, run_id="R7") == 0

    child_marker = tmp_path / "child_ran_2nd.txt"
    second = invoke(runner, script, out_dir, run_id="R7",
                    extra=["--fake-marker", str(child_marker)])

    assert second == 3
    assert not child_marker.exists()
    refusal = read_refusal(out_dir)
    assert refusal["refusal_code"] == "OUT_DIR_RESUME_SAME_RUN_ID"
    # the successful first-run evidence is untouched
    assert json.loads((out_dir / "gate_runner_summary.json").read_text(
        encoding="utf-8"))["outer_exit"] == 0


# ======================================================================================
# 10. FAIL-CLOSED ON MALFORMED result.json (audit: an exception used to escape main())
# ======================================================================================
_MALFORMED_CASES = [
    (b"{not json at all", "INVALID_JSON", "not JSON"),
    (b"", "INVALID_JSON", "empty file"),
    (b'{"gate_result": "PASS"', "INVALID_JSON", "truncated JSON"),
    (b"[1, 2, 3]", "INVALID_JSON", "top-level array"),
    (b'"PASS"', "INVALID_JSON", "top-level string"),
    (b"\x00\xff\xfe\x01binary", "INVALID_JSON", "undecodable bytes"),
    (b'{"metrics": {"m": 1}}', "MISSING_GATE_RESULT", "no gate_result key"),
    (b'{"gate_result": true}', "INVALID_GATE_RESULT", "boolean true is not PASS"),
    (b'{"gate_result": false}', "INVALID_GATE_RESULT", "boolean false"),
    (b'{"gate_result": 1}', "INVALID_GATE_RESULT", "number"),
    (b'{"gate_result": null}', "INVALID_GATE_RESULT", "null"),
    (b'{"gate_result": ["PASS"]}', "INVALID_GATE_RESULT", "list"),
    (b'{"gate_result": {"v": "PASS"}}', "INVALID_GATE_RESULT", "object"),
]


@pytest.mark.parametrize(
    "raw_bytes, expected_gate, case_id",
    _MALFORMED_CASES,
    ids=[case[2] for case in _MALFORMED_CASES],
)
def test_malformed_result_json_is_fail_closed(runner, tmp_path, raw_bytes, expected_gate, case_id):
    """No exception, no PASS: the malformed document becomes an INVALID gate + exit 1."""
    out_dir = tmp_path / ("malformed_%s" % expected_gate.lower())
    raw = tmp_path / "raw_result.bin"
    raw.write_bytes(raw_bytes)
    script = write_fake_child(tmp_path, returncode=0, write_result=False)

    outer_exit = invoke(runner, script, out_dir,
                        extra=["--fake-raw-result-file", str(raw)])

    assert outer_exit == 1, "malformed result.json must never yield PASS (%s)" % case_id
    summary = read_summary(out_dir)
    assert summary["child_returncode"] == 0
    assert summary["json_gate"] == expected_gate
    assert summary["json_gate_status"] == "INVALID"
    assert summary["gates_ok"] is False
    assert summary["outer_exit"] == 1
    assert summary["json_gate_reason"], "the invalid gate must explain itself"


def test_result_json_with_a_foreign_run_id_is_fail_closed(runner, tmp_path):
    """A child that writes another run's verdict must not be trusted."""
    out_dir = tmp_path / "payload_foreign_run"
    raw = tmp_path / "raw_result.json"
    raw.write_text(json.dumps({"run_id": "OTHER_RUN", "gate_result": "PASS",
                               "gates": {"self_check": True}}),
                   encoding="utf-8")
    script = write_fake_child(tmp_path, returncode=0, write_result=False)

    outer_exit = invoke(runner, script, out_dir, run_id="R1",
                        extra=["--fake-raw-result-file", str(raw)])

    assert outer_exit == 1
    summary = read_summary(out_dir)
    assert summary["json_gate"] == "RUN_ID_MISMATCH"
    assert summary["json_gate_status"] == "INVALID"
    assert summary["result_payload_run_id"] == "OTHER_RUN"
    assert "OTHER_RUN" in summary["json_gate_reason"]


def test_result_json_without_run_id_is_still_judged_on_its_gate(runner, tmp_path):
    """run_id is optional in result.json; when absent only the gate string decides."""
    out_dir = tmp_path / "payload_no_run_id"
    raw = tmp_path / "raw_result.json"
    raw.write_text(json.dumps({"gate_result": "PASS", "gates": {"self_check": True}}),
                   encoding="utf-8")
    script = write_fake_child(tmp_path, returncode=0, write_result=False)

    outer_exit = invoke(runner, script, out_dir, extra=["--fake-raw-result-file", str(raw)])

    assert outer_exit == 0
    summary = read_summary(out_dir)
    assert summary["json_gate"] == "PASS"
    assert summary["json_gate_status"] == "OK"
    assert summary["result_payload_run_id"] is None


def test_non_pass_gate_string_is_recorded_verbatim_and_rejected(runner, tmp_path):
    """No coercion: an unknown gate string keeps its name and still exits 1."""
    out_dir = tmp_path / "unknown_gate"
    raw = tmp_path / "raw_result.json"
    raw.write_text(json.dumps({"gate_result": "UNKNOWN"}), encoding="utf-8")
    script = write_fake_child(tmp_path, returncode=0, write_result=False)

    outer_exit = invoke(runner, script, out_dir, extra=["--fake-raw-result-file", str(raw)])

    assert outer_exit == 1
    summary = read_summary(out_dir)
    # the gate is structurally INVALID (only PASS/FAIL are accepted) and the child's own
    # spelling is preserved in the reason so the audit sees exactly what was written
    assert summary["json_gate"] == "INVALID_GATE_RESULT"
    assert summary["json_gate_status"] == "INVALID"
    assert "UNKNOWN" in summary["json_gate_reason"]
    assert summary["gates_ok"] is False


# ======================================================================================
# 11. TIMEOUT HONESTY: the kill must reach the child's descendants, on a real process tree
# ======================================================================================
def test_timeout_kills_the_child_and_its_grandchild(runner, tmp_path):
    """Real (non-Isaac) process tree: child -> grandchild, both must be dead afterwards."""
    out_dir = tmp_path / "tree_timeout"
    grandchild = tmp_path / "grandchild_probe.py"
    grandchild.write_text(_GRANDCHILD_TEMPLATE, encoding="utf-8")
    pidfile = tmp_path / "grandchild.pid"
    beat = tmp_path / "grandchild.heartbeat"

    # The child spawns the grandchild, records its pid, then hangs past the timeout.
    script = write_fake_child(tmp_path, returncode=0, gate="PASS", sleep_s=120.0)
    outer_exit = invoke(
        runner, script, out_dir, timeout=6.0,
        extra=["--fake-grandchild-script", str(grandchild),
               "--fake-grandchild-pidfile", str(pidfile),
               "--fake-grandchild-beat", str(beat)])

    assert outer_exit == 1
    summary = read_summary(out_dir)
    assert summary["timed_out"] is True
    assert summary["child_returncode"] == 124
    kill = summary["descendant_kill"]
    assert kill["attempted"] is True
    assert kill["child_terminated"] is True
    assert kill["descendants_terminated"] is True
    assert kill["verified_terminated"] is True
    assert kill["plan"]["covers_descendants"] is True
    if os.name == "nt":
        assert kill["method"].startswith("windows_taskkill_tree")
        assert kill["plan"]["argv"][:4] == ["taskkill", "/F", "/T", "/PID"]
        assert kill["taskkill_returncode"] == 0
    else:
        assert kill["method"] == "posix_killpg"
        assert kill["plan"]["signals"] == ["SIGTERM", "SIGKILL"]
    # Kit is explicitly not claimed as covered
    assert "UNKNOWN" in summary["residual_risk"]
    assert "Isaac Kit" in summary["residual_risk"]

    assert pidfile.is_file(), "the grandchild was never spawned"
    grandchild_pid = int(pidfile.read_text(encoding="utf-8").strip())
    assert grandchild_pid > 0
    assert beat.is_file(), "the grandchild never started writing heartbeats"
    alive_bytes = beat.stat().st_size
    assert alive_bytes > 0, "the grandchild produced no heartbeat before the kill"
    # 1. the grandchild PID is gone
    assert not pid_alive(grandchild_pid), (
        "grandchild pid %d survived the timeout kill" % grandchild_pid)
    # 2. and it stopped writing: the heartbeat file no longer grows
    time.sleep(0.8)
    assert beat.stat().st_size == alive_bytes, "the grandchild is still writing heartbeats"


def test_kill_tree_plan_covers_descendants_on_both_platforms(runner):
    """The platform plan is a pure function, so the POSIX path is pinned too."""
    windows = runner.build_kill_tree_plan(4321, os_name="nt")
    assert windows["method"] == "windows_taskkill_tree"
    assert windows["argv"] == ["taskkill", "/F", "/T", "/PID", "4321"]
    assert windows["covers_descendants"] is True

    posix = runner.build_kill_tree_plan(4321, os_name="posix")
    assert posix["method"] == "posix_killpg"
    assert posix["pgid"] == 4321
    assert posix["signals"] == ["SIGTERM", "SIGKILL"]
    assert posix["covers_descendants"] is True


def test_child_popen_kwargs_put_the_child_in_its_own_group(runner):
    """Windows: CREATE_NEW_PROCESS_GROUP.  POSIX: start_new_session (own process group)."""
    windows = runner.child_popen_kwargs(os_name="nt")
    assert windows["creationflags"] == getattr(subprocess, "CREATE_NEW_PROCESS_GROUP", 0x200)
    assert "start_new_session" not in windows

    posix = runner.child_popen_kwargs(os_name="posix")
    assert posix == {"start_new_session": True}


def test_pid_alive_probe_distinguishes_running_from_finished():
    """Sanity-check the liveness probe the timeout tests depend on."""
    proc = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(30)"])
    try:
        assert pid_alive(proc.pid) is True
    finally:
        proc.kill()
        proc.wait(timeout=30)
    assert pid_alive(proc.pid) is False

# ======================================================================================
# 12. P0-02 EXTRA: gate_result domain, gates dict, NaN/Infinity metrics
#     (null / 0 / "pass" / NaN, empty-or-absent gates, non-finite metrics)
# ======================================================================================
_GATE_DOMAIN_CASES = [
    (b'{"gate_result": null, "gates": {"g": true}}', "null", "GATE_NULL"),
    (b'{"gate_result": 0, "gates": {"g": true}}', "zero", "GATE_ZERO"),
    (b'{"gate_result": 1, "gates": {"g": true}}', "one", "GATE_ONE"),
    (b'{"gate_result": "pass", "gates": {"g": true}}', "lowercase", "GATE_LOWERCASE"),
    (b'{"gate_result": "Pass", "gates": {"g": true}}', "mixedcase", "GATE_MIXEDCASE"),
    (b'{"gate_result": "NaN", "gates": {"g": true}}', "nan_string", "GATE_NAN_STRING"),
    (b'{"gate_result": "", "gates": {"g": true}}', "empty_string", "GATE_EMPTY_STRING"),
    (b'{"gate_result": "PASS ", "gates": {"g": true}}', "trailing_space", "GATE_PADDED"),
    (b'{"gate_result": NaN, "gates": {"g": true}}', "bare_nan", "GATE_BARE_NAN"),
    (b'{"gate_result": "PASS"}', "gates_absent", "GATES_ABSENT"),
    (b'{"gate_result": "PASS", "gates": {}}', "gates_empty", "GATES_EMPTY"),
    (b'{"gate_result": "PASS", "gates": []}', "gates_list", "GATES_LIST"),
    (b'{"gate_result": "PASS", "gates": {"g": 1}}', "gates_non_bool", "GATES_NON_BOOLEAN"),
    (b'{"gate_result": "PASS", "gates": {"g": false}}', "gates_conflict",
     "GATES_CONFLICT"),
]


@pytest.mark.parametrize("raw_bytes, case_id, expect_token", _GATE_DOMAIN_CASES,
                         ids=[case[1] for case in _GATE_DOMAIN_CASES])
def test_gate_domain_and_gates_dict_are_fail_closed(runner, tmp_path, raw_bytes, case_id,
                                                    expect_token):
    """Only the exact string PASS/FAIL plus a non-empty all-true boolean gates object pass."""
    out_dir = tmp_path / ("domain_%s" % case_id)
    raw = tmp_path / "raw_result.json"
    raw.write_bytes(raw_bytes)
    script = write_fake_child(tmp_path, returncode=0, write_result=False)

    outer_exit = invoke(runner, script, out_dir, extra=["--fake-raw-result-file", str(raw)])

    assert outer_exit == 1, "%s must never be accepted" % case_id
    summary = read_summary(out_dir)
    assert summary["json_gate_status"] == "INVALID"
    assert summary["gates_ok"] is False
    assert summary["json_gate_reason"], "an invalid gate must explain itself"
    assert expect_token.split("_")[0] in summary["json_gate"] or "GATE" in summary["json_gate"]


def test_exact_pass_with_all_true_boolean_gates_is_accepted(runner, tmp_path):
    """Positive control for section 12: the strict reader still accepts a clean PASS."""
    out_dir = tmp_path / "domain_positive"
    raw = tmp_path / "raw_result.json"
    raw.write_text(json.dumps({"gate_result": "PASS",
                               "gates": {"a": True, "b": True},
                               "metrics": {"m": 3}}), encoding="utf-8")
    script = write_fake_child(tmp_path, returncode=0, write_result=False)

    outer_exit = invoke(runner, script, out_dir, extra=["--fake-raw-result-file", str(raw)])

    assert outer_exit == 0
    summary = read_summary(out_dir)
    assert summary["json_gate"] == "PASS"
    assert summary["json_gates_all_true"] is True
    assert summary["json_gates"] == {"a": True, "b": True}


@pytest.mark.parametrize("token", ["NaN", "Infinity", "-Infinity"])
def test_non_finite_metrics_keep_the_summary_valid_json_and_mark_the_run_invalid(
        runner, tmp_path, token):
    """A NaN metric must not corrupt the summary file; it makes the run INVALID."""
    out_dir = tmp_path / ("nonfinite_%s" % token.strip("-"))
    raw = tmp_path / "raw_result.json"
    # raw token: json.dumps(inf) would emit Infinity, which is what a careless child writes
    raw.write_text('{"gate_result": "PASS", "gates": {"g": true}, '
                   '"metrics": {"m": %s, "ok": 1.0}}' % token, encoding="utf-8")
    script = write_fake_child(tmp_path, returncode=0, write_result=False)

    outer_exit = invoke(runner, script, out_dir, extra=["--fake-raw-result-file", str(raw)])

    assert outer_exit == 1
    summary_path = out_dir / "gate_runner_summary.json"
    text = summary_path.read_text(encoding="utf-8")
    # strict JSON: parse_constant only fires for the bare NaN/Infinity tokens, so a summary
    # that parses under it contains no non-JSON number literal anywhere
    strict = json.loads(text, parse_constant=_reject_constant)
    assert strict["json_gate_status"] == "INVALID"
    assert strict["non_finite_metrics"], "the offending metric path must be recorded"
    assert strict["json_metrics"]["m"] == token
    assert strict["json_metrics"]["ok"] == 1.0
    assert strict["gates_ok"] is False
    assert strict["outer_exit"] == 1


def _reject_constant(token):
    raise AssertionError("summary contains the non-JSON constant %r" % token)


# ======================================================================================
# 13. P0-05 METADATA CONSISTENCY: manifest gravity vs the gravity the command used
# ======================================================================================
def _manifest(**overrides):
    payload = {"run_id": "R1", "gravity_m_s2": -9.81, "steps": 120, "dt": 1.0 / 60.0}
    payload.update(overrides)
    return json.dumps(payload)


def test_manifest_gravity_conflict_makes_the_run_invalid(runner, tmp_path):
    """The real bug: manifest said gravity [0,0,0] while the command used -9.81."""
    out_dir = tmp_path / "gravity_conflict"
    script = write_fake_child(tmp_path, returncode=0, gate="PASS")

    outer_exit = invoke(runner, script, out_dir, gravity_z=-9.81,
                        steps=120, dt=1.0 / 60.0,
                        extra=["--fake-manifest-json",
                               _manifest(gravity_m_s2=[0.0, 0.0, 0.0])])

    assert outer_exit == 1
    summary = read_summary(out_dir)
    assert summary["metadata_consistent"] is False
    assert summary["metadata_check"]["metadata_status"] == "GRAVITY_MISMATCH"
    assert summary["json_gate"] == "INVALID_METADATA"
    assert summary["json_gate_status"] == "INVALID"
    assert "gravity" in summary["json_gate_reason"].lower()
    assert summary["child_returncode"] == 0
    assert summary["gates_ok"] is False


def test_manifest_agreeing_with_the_command_is_accepted(runner, tmp_path):
    """Positive control: consistent metadata keeps the PASS."""
    out_dir = tmp_path / "gravity_ok"
    script = write_fake_child(tmp_path, returncode=0, gate="PASS")

    outer_exit = invoke(runner, script, out_dir, gravity_z=-9.81,
                        steps=120, dt=1.0 / 60.0,
                        extra=["--fake-manifest-json", _manifest()])

    assert outer_exit == 0
    summary = read_summary(out_dir)
    assert summary["metadata_consistent"] is True
    assert summary["metadata_check"]["metadata_status"] == "CONSISTENT"
    assert summary["metadata_check"]["checked"]["gravity_m_s2"] == -9.81


@pytest.mark.parametrize(
    "manifest_overrides, expected_status",
    [({"steps": 60}, "STEPS_MISMATCH"),
     ({"dt": 1.0 / 30.0}, "DT_MISMATCH"),
     ({"gravity_m_s2": None}, "GRAVITY_UNVERIFIABLE")],
    ids=["steps", "dt", "gravity_missing"],
)
def test_manifest_steps_dt_and_missing_gravity_are_fail_closed(
        runner, tmp_path, manifest_overrides, expected_status):
    out_dir = tmp_path / ("meta_%s" % expected_status.lower())
    script = write_fake_child(tmp_path, returncode=0, gate="PASS")

    outer_exit = invoke(runner, script, out_dir, gravity_z=-9.81, steps=120,
                        dt=1.0 / 60.0,
                        extra=["--fake-manifest-json", _manifest(**manifest_overrides)])

    assert outer_exit == 1
    summary = read_summary(out_dir)
    assert summary["metadata_consistent"] is False
    assert summary["metadata_check"]["metadata_status"] == expected_status


def test_unreadable_manifest_is_fail_closed(runner, tmp_path):
    out_dir = tmp_path / "manifest_garbage"
    script = write_fake_child(tmp_path, returncode=0, gate="PASS")

    outer_exit = invoke(runner, script, out_dir, gravity_z=-9.81,
                        extra=["--fake-manifest-json", "{not json"])

    assert outer_exit == 1
    summary = read_summary(out_dir)
    assert summary["metadata_consistent"] is False
    assert summary["metadata_check"]["metadata_status"] == "MANIFEST_UNREADABLE"


def test_absent_manifest_makes_no_claim_and_keeps_the_pass(runner, tmp_path):
    out_dir = tmp_path / "manifest_absent"
    script = write_fake_child(tmp_path, returncode=0, gate="PASS")

    assert invoke(runner, script, out_dir, gravity_z=-9.81) == 0

    summary = read_summary(out_dir)
    assert summary["metadata_consistent"] is True
    assert summary["metadata_check"]["metadata_status"] == "MANIFEST_ABSENT"


# ======================================================================================
# 14. P0-06 JSONL INTEGRITY: ordering, count, finiteness, contact null vs 0
# ======================================================================================
def _pose(step, contact=None, force=None, extra=None):
    record = {"step": step, "t": step / 60.0, "z": 0.5, "contact": contact}
    if force is not None:
        record["contact_force"] = force
    if extra:
        record.update(extra)
    return json.dumps(record)


def _jsonl(tmp_path, lines, name="raw_poses.jsonl"):
    path = tmp_path / name
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def test_jsonl_valid_file_passes(runner, tmp_path):
    path = _jsonl(tmp_path, [_pose(i, contact=None) for i in range(5)])
    report = runner.check_pose_jsonl(str(path), 5)
    assert report["valid"] is True
    assert report["line_count"] == 5
    assert report["monotonic_steps"] is True
    assert report["expected_count_matches"] is True
    assert report["contact_null_count"] == 5
    assert report["contact_numeric_count"] == 0


@pytest.mark.parametrize(
    "lines, expected_key",
    [
        ([_pose(0), _pose(2), _pose(1)], "duplicate_steps"),
        ([_pose(0), _pose(0)], "duplicate_steps"),
        (["{not json", _pose(1)], "invalid_json_lines"),
        (["[1, 2]", _pose(1)], "non_object_lines"),
        ([json.dumps({"t": 0.0})], "missing_step_lines"),
        ([json.dumps({"step": "0"})], "non_integer_step_lines"),
        ([_pose(0, extra={"z": float("nan")})], "non_finite"),
        ([_pose(0, contact="touching")], "contact_invalid"),
        ([_pose(0, extra={"contact_status": "unavailable", "contact": 0.0})],
         "contact_silently_zero"),
    ],
    ids=["rewound_steps", "duplicate_steps", "bad_json", "not_object", "no_step",
         "string_step", "nan_value", "contact_not_numeric", "silent_zero_in_unavailable"],
)
def test_jsonl_integrity_violations_are_rejected(runner, tmp_path, lines, expected_key):
    path = _jsonl(tmp_path, lines)
    report = runner.check_pose_jsonl(str(path), None)
    assert report["valid"] is False
    assert report[expected_key], "expected %s to be recorded" % expected_key


def test_jsonl_absent_and_count_mismatch_are_invalid(runner, tmp_path):
    missing = runner.check_pose_jsonl(str(tmp_path / "nope.jsonl"), 3)
    assert missing["valid"] is False
    assert missing["present"] is False

    path = _jsonl(tmp_path, [_pose(i) for i in range(3)])
    report = runner.check_pose_jsonl(str(path), 4)
    assert report["valid"] is False
    assert report["expected_count_matches"] is False


def test_invalid_jsonl_makes_the_run_invalid(runner, tmp_path):
    """outer 0 requires a valid JSONL as well, not just a PASS gate."""
    out_dir = tmp_path / "jsonl_bad"
    script = write_fake_child(tmp_path, returncode=0, gate="PASS")

    outer_exit = invoke(runner, script, out_dir,
                        extra=["--fake-jsonl", _pose(1) + "\n" + _pose(1) + "\n"])

    assert outer_exit == 1
    summary = read_summary(out_dir)
    assert summary["jsonl_status"] == "JSONL_INVALID"
    assert summary["jsonl_valid"] is False
    assert summary["jsonl_check"]["duplicate_steps"]
    assert summary["outer_exit"] == 1


def test_valid_jsonl_with_expected_frame_count_keeps_the_pass(runner, tmp_path):
    out_dir = tmp_path / "jsonl_ok"
    script = write_fake_child(tmp_path, returncode=0, gate="PASS")
    lines = "\n".join(_pose(i, contact=None) for i in range(4)) + "\n"

    outer_exit = invoke(runner, script, out_dir,
                        extra=["--fake-jsonl", lines])

    assert outer_exit == 0
    summary = read_summary(out_dir)
    assert summary["jsonl_status"] == "JSONL_VALID"
    assert summary["jsonl_check"]["line_count"] == 4


def test_require_jsonl_rejects_an_absent_file(runner, tmp_path):
    out_dir = tmp_path / "jsonl_required"
    script = write_fake_child(tmp_path, returncode=0, gate="PASS")

    outer_exit = invoke(runner, script, out_dir, require_jsonl=True)

    assert outer_exit == 1
    summary = read_summary(out_dir)
    assert summary["jsonl_status"] == "JSONL_INVALID"
    args = summary["metadata_check"]["requested"]
    assert args["gravity_z"] is None


def test_jsonl_cli_mode_exit_codes(tmp_path):
    """The documented CLI: --check-jsonl PATH exits 0/1 on its own (no child started)."""
    good = _jsonl(tmp_path, [_pose(0), _pose(1)], name="good.jsonl")
    bad = _jsonl(tmp_path, [_pose(1), _pose(0)], name="bad.jsonl")

    ok = subprocess.run([sys.executable, str(RUNNER_PATH), "--check-jsonl", str(good),
                         "--expected-frames", "2"], capture_output=True, text=True)
    assert ok.returncode == 0, ok.stderr
    assert json.loads(ok.stdout)["valid"] is True

    broken = subprocess.run([sys.executable, str(RUNNER_PATH), "--check-jsonl", str(bad)],
                            capture_output=True, text=True)
    assert broken.returncode == 1, broken.stderr
    assert json.loads(broken.stdout)["duplicate_steps"]


# ======================================================================================
# 15. P0-03 PID EVIDENCE: the summary carries the real PIDs and a verified cleanup flag
# ======================================================================================
def test_timeout_summary_records_real_pids_of_the_killed_tree(runner, tmp_path):
    out_dir = tmp_path / "pid_evidence"
    grandchild = tmp_path / "grandchild_probe.py"
    grandchild.write_text(_GRANDCHILD_TEMPLATE, encoding="utf-8")
    pidfile = tmp_path / "grandchild.pid"
    beat = tmp_path / "grandchild.heartbeat"
    child_pidfile = tmp_path / "child.pid"

    script = write_fake_child(tmp_path, returncode=0, gate="PASS", sleep_s=120.0)
    outer_exit = invoke(
        runner, script, out_dir, timeout=8.0,
        extra=["--fake-marker", str(child_pidfile),
               "--fake-grandchild-script", str(grandchild),
               "--fake-grandchild-pidfile", str(pidfile),
               "--fake-grandchild-beat", str(beat)])

    assert outer_exit == 1
    summary = read_summary(out_dir)
    assert isinstance(summary["child_pid"], int) and summary["child_pid"] > 0
    assert summary["killed_pids"], "the killed PID list must not be empty"
    assert summary["child_pid"] in summary["killed_pids"]
    assert summary["surviving_pids"] == []
    grandchild_pid = int(pidfile.read_text(encoding="utf-8").strip())
    assert not pid_alive(grandchild_pid)
    # every pid we claim to have killed is really gone
    for pid in summary["killed_pids"]:
        assert not pid_alive(pid), "killed_pid %d is still alive" % pid
    assert summary["descendant_cleanup_verified"] in (True, None)
    assert "UNKNOWN" in summary["residual_risk"]


def test_non_timeout_run_claims_no_kill_and_no_cleanup(runner, tmp_path):
    out_dir = tmp_path / "no_kill"
    script = write_fake_child(tmp_path, returncode=0, gate="PASS")

    assert invoke(runner, script, out_dir) == 0

    summary = read_summary(out_dir)
    assert isinstance(summary["child_pid"], int)
    assert summary["killed_pids"] == []
    assert summary["surviving_pids"] == []
    assert summary["descendant_cleanup_verified"] is None
    assert summary["descendant_kill"] is None


# ======================================================================================
# 16. P0-04 spec-level error code: OUT_DIR_REUSED
# ======================================================================================
def test_refusal_carries_the_spec_error_code_out_dir_reused(runner, tmp_path):
    out_dir = tmp_path / "reused_code"
    out_dir.mkdir()
    (out_dir / "result.json").write_text(
        json.dumps({"run_id": "OTHER", "gate_result": "PASS"}), encoding="utf-8")
    script = write_fake_child(tmp_path, returncode=0, gate="PASS")
    child_marker = tmp_path / "child_ran.txt"

    outer_exit = invoke(runner, script, out_dir, run_id="R1",
                        extra=["--fake-marker", str(child_marker)])

    assert outer_exit != 0
    assert not child_marker.exists()
    refusal = read_refusal(out_dir)
    assert refusal["error_code"] == "OUT_DIR_REUSED"
    assert refusal["refusal_code"] == "OUT_DIR_RUN_ID_MISMATCH"
    assert refusal["child_started"] is False

