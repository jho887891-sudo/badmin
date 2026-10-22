# -*- coding: utf-8 -*-
"""R10-A regression: the runner must report a TYPED outcome, and no fault may look like a PASS.

Target: tools/simulation/p2_gate_runner.py (A2/A3/A4 of EXECUTION_BATCH_R10_32_20261022)

Contract under test (the five normative outcomes, summary["outcome"]):

    SUCCESS           exit 0   -- the only outcome allowed to exit 0
    EXPECTED_FAULT    exit 10  -- a DECLARED fault was injected and observed: the test
                                  PASSED, the simulation did NOT succeed
    UNEXPECTED_FAULT  exit 1   -- fault/failure signal without a matching declaration
    CRASH             exit 1   -- abnormal termination (signal / Kit crash reporter)
    INCOMPLETE        exit 1   -- no fault signal, but no acceptable completion certificate

The load-bearing fact from P0-07: this Kit build swallows Python exceptions and still
reports child rc 0 (4 of 6 real fault runs), so a traceback or a typed fault_record.json is
a HARD non-success signal and the child return code alone proves nothing.

No Isaac Sim, no GPU, no network: the child is a generated Python script; the P0F replay
reads the COMMITTED fault evidence through "git show" into a temp dir (never written back).
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
REPLAY_PATH = ROOT / "tools" / "simulation" / "replay_p0f_fault_evidence.py"

_CHILD_TEMPLATE = r'''# -*- coding: utf-8 -*-
"""Generated fake experiment (no Isaac, no GPU).  Prints/crashes/writes on demand."""
import argparse
import json
import os
import sys
import time


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--fake-gate", default="PASS")
    parser.add_argument("--fake-write", default="1")
    parser.add_argument("--fake-gates-json", default='{"fake_self_check": true}')
    parser.add_argument("--fake-fault-record", default="")
    parser.add_argument("--fake-traceback", default="")
    parser.add_argument("--fake-kit-noise", default="0")
    parser.add_argument("--fake-raw-result", default="")
    args, extra = parser.parse_known_args()
    out_dir = os.path.abspath(args.out_dir)
    os.makedirs(out_dir, exist_ok=True)

    if args.fake_kit_noise == "1":
        # The exact false-positive shape seen in the real P0F stderr logs: an
        # "ExceptionName: message" line at column 0 WITHOUT a traceback header.
        sys.stderr.write(
            "/home/dgut/runs/g/scripts/simulation/p3_dynamic_mount.py:298: DeprecationWarning: "
            "The function 'write_root_pose_to_sim' will be deprecated in a future release.\n"
            "  robot.write_root_pose_to_sim(\n")
    if args.fake_traceback:
        sys.stderr.write(
            "Traceback (most recent call last):\n"
            "  File \"" + __file__ + "\", line 568, in <module>\n"
            "    raise SystemExit(main())\n"
            "  File \"" + __file__ + "\", line 434, in main\n"
            "    raise RuntimeError(\"INJECTED_PYTHON_EXCEPTION at step 60\")\n"
            "RuntimeError: " + args.fake_traceback + "\n")
    if args.fake_fault_record:
        with open(os.path.join(out_dir, "fault_record.json"), "w", encoding="utf-8") as fh:
            fh.write(args.fake_fault_record)
    if args.fake_raw_result:
        with open(args.fake_raw_result, "rb") as src:
            raw = src.read()
        with open(os.path.join(out_dir, "result.json"), "wb") as dst:
            dst.write(raw)
    elif args.fake_write == "1":
        payload = {"run_id": args.run_id, "gate_result": args.fake_gate,
                   "gates": json.loads(args.fake_gates_json),
                   "metrics": {"fake_metric": 1.5},
                   "child_extra_argv": list(extra)}
        with open(os.path.join(out_dir, "result.json"), "w", encoding="utf-8") as fh:
            json.dump(payload, fh, ensure_ascii=False, indent=1)
    sys.stderr.flush()
    print("fake child done gate=%s" % args.fake_gate, flush=True)
    time.sleep(__SLEEP__)
    return __RC__


if __name__ == "__main__":
    raise SystemExit(main())
'''

TYPED_FAULT_RECORD = json.dumps({
    "run_id": "R1", "fault": "python_exception", "code": "INJECTED_PYTHON_EXCEPTION",
    "detail": {"step": 60}, "sim_steps_taken": 60,
    "note": "typed refusal: the runner must surface this as a non-zero child code",
}, ensure_ascii=False, indent=1)
JOINT_FAULT_RECORD = json.dumps({
    "run_id": "R1", "fault": "joint_limit", "code": "JOINT_LIMIT_REFUSED",
    "detail": {"joint": "joint5", "target_rad": 2.07, "upper_limit_rad": 1.57,
               "targets_written": 0}, "sim_steps_taken": 0,
}, ensure_ascii=False, indent=1)


@pytest.fixture(scope="module")
def runner():
    """Import the gate runner by path (tools/ is not an installed package)."""
    assert RUNNER_PATH.is_file(), "gate runner missing: %s" % RUNNER_PATH
    spec = importlib.util.spec_from_file_location("p2_gate_runner_r10a", RUNNER_PATH)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def write_fake_child(tmp_path, returncode=0, gate="PASS", write_result=True, sleep_s=0.0,
                     fault_record="", traceback_text="", kit_noise=False, gates=None,
                     raw_result_bytes=None):
    script = tmp_path / "fake_fault_experiment.py"
    script.write_text(
        _CHILD_TEMPLATE
        .replace("__RC__", str(int(returncode)))
        .replace("__SLEEP__", repr(float(sleep_s))),
        encoding="utf-8")
    extra = ["--fake-gate", gate,
             "--fake-write", "1" if write_result else "0",
             "--fake-fault-record", fault_record,
             "--fake-traceback", traceback_text,
             "--fake-kit-noise", "1" if kit_noise else "0",
             "--fake-gates-json", json.dumps(gates if gates is not None
                                             else {"fake_self_check": True})]
    if raw_result_bytes is not None:
        raw_path = tmp_path / "raw_result.bin"
        raw_path.write_bytes(raw_result_bytes)
        extra += ["--fake-raw-result", str(raw_path)]
    return script, extra


def invoke(runner, script, out_dir, run_id="R1", child_extra=(), timeout=None,
           expect_fault=None, allow_reused_out_dir=False):
    """Call main() in-process so the returned outer exit code is directly assertable."""
    argv = ["--script", str(script), "--run-id", run_id, "--out-dir", str(out_dir)]
    if timeout is not None:
        argv += ["--timeout", str(timeout)]
    if allow_reused_out_dir:
        argv += ["--allow-reused-out-dir"]
    for item in (expect_fault or []):
        argv += ["--expect-fault", item]
    if child_extra:
        argv += ["--"] + [str(item) for item in child_extra]
    return runner.main(argv)


def read_summary(out_dir):
    path = Path(out_dir) / "gate_runner_summary.json"
    assert path.is_file(), "gate_runner_summary.json not written: %s" % path
    return json.loads(path.read_text(encoding="utf-8"))


def run_case(runner, tmp_path, case_id, **kwargs):
    """Run one fake child and return (outer_exit, summary)."""
    child_kwargs = {key: kwargs.pop(key) for key in list(kwargs)
                    if key in ("returncode", "gate", "write_result", "sleep_s",
                               "fault_record", "traceback_text", "kit_noise", "gates",
                               "raw_result_bytes")}
    child_extra = list(kwargs.pop("child_extra", ()))
    out_dir = tmp_path / case_id
    script, extra = write_fake_child(tmp_path, **child_kwargs)
    outer = invoke(runner, script, out_dir, child_extra=child_extra + extra, **kwargs)
    return outer, read_summary(out_dir)


# --------------------------------------------------------------------------------------
# A3: the vocabulary itself
# --------------------------------------------------------------------------------------
def test_outcome_vocabulary_is_exactly_five_and_only_success_exits_zero(runner):
    assert [item.value for item in runner.Outcome] == [
        "SUCCESS", "EXPECTED_FAULT", "UNEXPECTED_FAULT", "CRASH", "INCOMPLETE"]
    exits = {item.value: runner.OUTCOME_EXIT_CODES[item] for item in runner.Outcome}
    assert exits == {"SUCCESS": 0, "EXPECTED_FAULT": 10, "UNEXPECTED_FAULT": 1,
                     "CRASH": 1, "INCOMPLETE": 1}
    # A fault-injection test PASS must be separable from a simulation SUCCESS by exit code
    # alone, otherwise a CI consumer cannot tell them apart.
    assert [value for value, code in exits.items() if code == 0] == ["SUCCESS"]
    assert runner.EXIT_EXPECTED_FAULT != 0 and runner.EXIT_EXPECTED_FAULT != 1


@pytest.mark.parametrize(
    "overrides, expected_outcome, expected_reason",
    [
        ({}, "SUCCESS", "ALL_CHECKS_GREEN"),
        ({"gate": "FAIL"}, "INCOMPLETE", "GATE_SAYS_FAIL"),
        ({"returncode": 1, "write_result": False}, "UNEXPECTED_FAULT", "CHILD_NONZERO_EXIT"),
        ({"returncode": 1}, "UNEXPECTED_FAULT",
         "CHILD_NONZERO_EXIT_WITH_PASS_ARTIFACT"),
        ({"write_result": False}, "INCOMPLETE", "RESULT_JSON_MISSING"),
        ({"traceback_text": "INJECTED_PYTHON_EXCEPTION at step 60"},
         "UNEXPECTED_FAULT", "TRACEBACK_WITHOUT_EXPECTATION"),
        ({"fault_record": TYPED_FAULT_RECORD}, "UNEXPECTED_FAULT",
         "FAULT_RECORD_WITHOUT_EXPECTATION"),
        ({"raw_result_bytes": b'{"run_id": "R1", "gate_result": '}, "INCOMPLETE",
         "RESULT_JSON_INVALID"),
        ({"gates": {"fake_self_check": False}}, "INCOMPLETE", "RESULT_JSON_INVALID"),
    ],
    ids=["green", "gate_fail", "rc1_silent", "rc1_with_pass_artifact", "no_result",
         "traceback", "fault_record",
         "corrupt_json", "false_gate"],
)
def test_every_case_that_is_not_green_is_fail_closed(
        runner, tmp_path, overrides, expected_outcome, expected_reason):
    outer, summary = run_case(runner, tmp_path, "case_%s" % expected_reason,
                              **dict(overrides))
    assert summary["outcome"] == expected_outcome
    assert summary["outcome_reason_code"] == expected_reason
    assert summary["outcome_exit"] == outer
    assert (outer == 0) == (expected_outcome == "SUCCESS")
    if expected_outcome != "SUCCESS":
        assert outer != 0, "a non-SUCCESS outcome must never exit 0: %r" % summary["outcome"]


def test_classifier_is_pure_and_total(runner):
    """classify_run() takes data only: no path, no I/O, and it never raises."""
    evidence = runner.RunEvidence(
        run_id="R1", child_returncode=0, timed_out=False, stderr_text="",
        gate={"json_gate": "MISSING", "json_gate_status": "MISSING", "gates_all_true": False},
        metadata_consistent=True, jsonl_valid=True, fault_record={"present": False})
    verdict = runner.classify_run(evidence)
    assert verdict.outcome is runner.Outcome.INCOMPLETE
    assert verdict.reason_code == "RESULT_JSON_MISSING"
    # garbage in, typed verdict out (never an exception escaping the runner)
    weird = runner.RunEvidence(
        run_id="R1", child_returncode=None, timed_out=False, stderr_text=None,
        gate=None, metadata_consistent=None, jsonl_valid=None,
        fault_record={"present": True, "typed": False})
    assert runner.classify_run(weird).outcome is runner.Outcome.UNEXPECTED_FAULT


# --------------------------------------------------------------------------------------
# A2: traceback signature + typed fault artifact (the Kit false-success cases)
# --------------------------------------------------------------------------------------
@pytest.mark.parametrize(
    "stderr_text, detected, tier",
    [
        ("", False, None),
        # FALSE POSITIVE CONTROL: exactly the Kit deprecation lines from P0F_exception_r01
        ("/home/x/p3_dynamic_mount.py:298: DeprecationWarning: The function 'f' is deprecated.\n"
         "  robot.write_root_pose_to_sim(\n", False, None),
        ("OSError: [Errno 9] Bad file descriptor\n", False, None),
        ("Traceback (most recent call last):\n",
         False, None),
        ("Traceback (most recent call last):\n"
         '  File "/x/p3_dynamic_mount.py", line 568, in <module>\n'
         "    raise SystemExit(main())\n"
         '  File "/x/p3_dynamic_mount.py", line 434, in main\n'
         '    raise RuntimeError("INJECTED_PYTHON_EXCEPTION at step 60")\n'
         "RuntimeError: INJECTED_PYTHON_EXCEPTION at step 60\n",
         True, "TRACEBACK_PAIR"),
        ("Traceback (most recent call last):\n"
         '  File "/x/p3_dynamic_mount.py", line 434, in main\n',
         True, "TRACEBACK_TRUNCATED"),
        ("Fatal Python error: Segmentation fault\n", True, "FATAL_PYTHON_ERROR"),
    ],
    ids=["empty", "kit_deprecation_only", "bare_exception_line", "bare_header",
         "pair", "truncated", "fatal_python_error"],
)
def test_traceback_signature_requires_the_header_and_a_terminator(
        runner, stderr_text, detected, tier):
    report = runner.detect_traceback(stderr_text)
    assert report["detected"] is detected
    assert report["tier"] == tier


@pytest.mark.parametrize(
    "declared, fault, code, matches",
    [
        ("python_exception", "python_exception", "INJECTED_PYTHON_EXCEPTION", True),
        ("joint_limit", "joint_limit", "JOINT_LIMIT_REFUSED", True),
        ("JOINT_LIMIT", "joint_limit", "JOINT_LIMIT_REFUSED", True),
        ("python_exception", "joint_limit", "JOINT_LIMIT_REFUSED", False),
        ("", "python_exception", "X", False),
    ],
    ids=["exact", "joint", "case_folded", "mismatch", "empty_declaration"],
)
def test_declared_fault_token_matching(runner, declared, fault, code, matches):
    tokens = {runner.normalize_fault_token(fault), runner.normalize_fault_token(code)}
    assert runner._fault_matches(declared, tokens, "") is matches


@pytest.mark.parametrize(
    "extra, expected",
    [
        (["--inject-fault", "python_exception"], ["python_exception"]),
        (["--inject-fault=joint_limit"], ["joint_limit"]),
        (["--fault-injection", "python_exception"], ["python_exception"]),
        (["--profile", "static"], []),
        ([], []),
    ],
    ids=["space", "equals", "alias", "unrelated", "none"],
)
def test_declared_faults_are_read_from_forwarded_child_arguments(runner, extra, expected):
    assert runner.declared_faults_from_extra(extra) == expected
    # the same rule must work on a RECORDED command line (offline evidence replay)
    assert runner.declared_faults_from_command("python x.py " + " ".join(extra)) == expected


def test_child_rc_zero_with_a_traceback_is_never_success(runner, tmp_path):
    """THE A2 case: a valid PASS artifact, rc 0, and a traceback on stderr."""
    outer, summary = run_case(runner, tmp_path, "traceback_rc0",
                              traceback_text="INJECTED_PYTHON_EXCEPTION at step 60")
    assert summary["child_returncode"] == 0            # Kit lied ...
    assert summary["child_ok"] is True                 # ... and the old signal says "ok"
    assert summary["json_gate"] == "PASS"              # ... and the artifact says PASS
    assert summary["outcome"] == "UNEXPECTED_FAULT"    # ... and the outcome still refuses
    assert summary["outcome_reason_code"] == "TRACEBACK_WITHOUT_EXPECTATION"
    assert outer == 1


def test_child_rc_zero_with_a_typed_fault_record_is_never_success(runner, tmp_path):
    outer, summary = run_case(runner, tmp_path, "fault_record_rc0", write_result=False,
                              fault_record=TYPED_FAULT_RECORD)
    assert summary["child_returncode"] == 0
    assert summary["outcome"] == "UNEXPECTED_FAULT"
    assert summary["outcome_reason_code"] == "FAULT_RECORD_WITHOUT_EXPECTATION"
    assert summary["fault_verdict"]["fault_record"]["typed"] is True
    assert outer == 1


def test_a_fault_record_and_a_passing_result_json_cannot_both_be_true(runner, tmp_path):
    outer, summary = run_case(runner, tmp_path, "fault_and_pass",
                              fault_record=TYPED_FAULT_RECORD,
                              expect_fault=["python_exception"])
    assert summary["outcome"] == "UNEXPECTED_FAULT"
    assert summary["outcome_reason_code"] == "FAULT_RUN_CLAIMED_PASS"
    assert outer == 1


def test_untyped_fault_record_is_still_a_fault_signal(runner, tmp_path):
    outer, summary = run_case(runner, tmp_path, "fault_untyped", write_result=False,
                              fault_record='{"run_id": "R1", "note": "half a record"}')
    assert summary["outcome"] == "UNEXPECTED_FAULT"
    assert summary["fault_verdict"]["fault_record"]["typed"] is False
    assert summary["fault_verdict"]["fault_record"]["error"]
    assert outer == 1


def test_a_corrupt_fault_record_cannot_degrade_into_silence(runner, tmp_path):
    outer, summary = run_case(runner, tmp_path, "fault_corrupt", write_result=False,
                              fault_record='{"run_id": "R1", "fault": ')
    assert summary["outcome"] == "UNEXPECTED_FAULT"
    assert "unreadable" in summary["fault_verdict"]["fault_record"]["error"]
    assert outer == 1


def test_kit_deprecation_noise_alone_does_not_break_a_real_success(runner, tmp_path):
    outer, summary = run_case(runner, tmp_path, "noise_ok", kit_noise=True)
    assert summary["outcome"] == "SUCCESS"
    assert outer == 0


# --------------------------------------------------------------------------------------
# A3: fault-injection TEST pass vs simulation SUCCESS, end to end
# --------------------------------------------------------------------------------------
def test_declared_fault_observed_from_the_forwarded_inject_flag_exits_ten(runner, tmp_path):
    outer, summary = run_case(
        runner, tmp_path, "expected_fault", write_result=False, returncode=0,
        fault_record=TYPED_FAULT_RECORD,
        child_extra=["--inject-fault", "python_exception",
                     "--fake-fault-record", TYPED_FAULT_RECORD])
    assert summary["declared_faults"] == ["python_exception"]
    assert summary["declared_fault_source"] == "forwarded_child_args"
    assert summary["outcome"] == "EXPECTED_FAULT"
    assert summary["outcome_reason_code"] == "FAULT_RECORD_TYPED"
    assert outer == 10
    assert outer != 0, "a fault test PASS must not share the SUCCESS exit code"


def test_declared_fault_proven_by_traceback_only_is_recorded_as_weaker_evidence(
        runner, tmp_path):
    outer, summary = run_case(
        runner, tmp_path, "traceback_declared", write_result=False,
        traceback_text="INJECTED_PYTHON_EXCEPTION at step 60",
        child_extra=["--fake-traceback", "INJECTED_PYTHON_EXCEPTION at step 60"],
        expect_fault=["python_exception"])
    assert summary["declared_fault_source"] == "cli"
    assert summary["outcome"] == "EXPECTED_FAULT"
    assert summary["outcome_reason_code"] == "TRACEBACK_ONLY"
    assert outer == 10


def test_declared_fault_that_never_happened_is_incomplete_not_success(runner, tmp_path):
    outer, summary = run_case(runner, tmp_path, "declared_absent",
                              expect_fault=["python_exception"])
    assert summary["outcome"] == "INCOMPLETE"
    assert summary["outcome_reason_code"] == "EXPECTED_FAULT_NOT_OBSERVED"
    assert outer == 1


def test_declared_fault_that_does_not_match_the_record_is_unexpected(runner, tmp_path):
    outer, summary = run_case(runner, tmp_path, "declared_mismatch", write_result=False,
                              fault_record=JOINT_FAULT_RECORD,
                              expect_fault=["python_exception"])
    assert summary["outcome"] == "UNEXPECTED_FAULT"
    assert summary["outcome_reason_code"] == "FAULT_RECORD_MISMATCH"
    assert outer == 1


def test_a_crash_during_a_declared_fault_is_a_crash_not_an_expected_fault(runner, tmp_path):
    outer, summary = run_case(
        runner, tmp_path, "declared_crash", returncode=139, write_result=False,
        fault_record=TYPED_FAULT_RECORD, expect_fault=["python_exception"])
    assert summary["outcome"] == "CRASH"
    assert summary["outcome_reason_code"] == "CRASH_ABNORMAL_TERMINATION"
    assert outer == 1


def test_foreign_fault_record_cannot_certify_this_run(runner, tmp_path):
    foreign = json.loads(TYPED_FAULT_RECORD)
    foreign["run_id"] = "SOMEONE_ELSE"
    out_dir = tmp_path / "foreign_record"
    out_dir.mkdir()
    (out_dir / "fault_record.json").write_text(json.dumps(foreign), encoding="utf-8")
    (out_dir / ".p2_gate_run_owner.json").write_text(
        json.dumps({"run_id": "foreign_record"}), encoding="utf-8")
    script, extra = write_fake_child(tmp_path, write_result=False)
    outer = invoke(runner, script, out_dir, run_id="foreign_record",
                   child_extra=extra, expect_fault=["python_exception"],
                   allow_reused_out_dir=True)
    # the stale record is quarantined, so the child's silence is what is judged
    summary = read_summary(out_dir)
    assert summary["quarantined_stale_fault_record"] is not None
    assert summary["outcome"] == "INCOMPLETE"
    assert summary["outcome_reason_code"] == "EXPECTED_FAULT_NOT_OBSERVED"
    assert outer == 1


# --------------------------------------------------------------------------------------
# A4: fail-closed regressions (timeout / SIGSEGV / missing result / corrupt / stale / rc+PASS)
# --------------------------------------------------------------------------------------
def test_timeout_is_incomplete_and_exit_one(runner, tmp_path):
    outer, summary = run_case(runner, tmp_path, "timeout_case", sleep_s=30.0, timeout=1.0)
    assert summary["timed_out"] is True
    assert summary["child_returncode"] == runner.TIMEOUT_RETURNCODE
    assert summary["outcome"] == "INCOMPLETE"
    assert summary["outcome_reason_code"] == "TIMEOUT"
    assert outer == 1


@pytest.mark.parametrize("returncode", [139, 134, 137],
                         ids=["sigsegv", "sigabrt", "sigkill"])
def test_signal_exit_is_a_crash_even_with_a_passing_artifact(runner, tmp_path, returncode):
    outer, summary = run_case(runner, tmp_path, "crash_%d" % returncode,
                              returncode=returncode)
    assert summary["outcome"] == "CRASH"
    assert summary["outcome_reason_code"] == "CRASH_ABNORMAL_TERMINATION"
    assert summary["fault_verdict"]["crash"]["crash"] is True
    assert outer == 1


def test_kit_crash_reporter_output_is_a_crash(runner, tmp_path):
    out_dir = tmp_path / "kit_crash"
    script, extra = write_fake_child(tmp_path, returncode=0, write_result=True)
    # simulate the real P0F_jointlimit_r01 stderr tail with an rc that Kit swallowed to 0
    crash_log = tmp_path / "crash_stderr.txt"
    crash_log.write_text(
        "2026-10-22T04:15:55Z [Warning] [carb.crashreporter-breakpad.plugin] [crash] "
        "A crash has occurred.\n"
        "2026-10-22T04:15:55Z [Warning] [carb.crashreporter-breakpad.plugin] "
        "terminating this process with exit code 139.\n", encoding="utf-8")
    outer = invoke(runner, script, out_dir, child_extra=extra)
    summary = read_summary(out_dir)
    assert summary["outcome"] == "SUCCESS"  # unrelated run: no crash text was involved
    # now the same text, fed through the pure detector
    crash = runner.detect_crash(0, crash_log.read_text(encoding="utf-8"))
    assert crash["crash"] is True
    assert "STDERR_KIT_CRASH_REPORTER_LOG" in crash["signals"]
    assert outer == 0


def test_missing_result_json_is_incomplete(runner, tmp_path):
    outer, summary = run_case(runner, tmp_path, "no_result", write_result=False)
    assert summary["outcome"] == "INCOMPLETE"
    assert summary["outcome_reason_code"] == "RESULT_JSON_MISSING"
    assert outer == 1


def test_corrupted_result_json_is_incomplete_and_the_summary_stays_valid_json(
        runner, tmp_path):
    outer, summary = run_case(runner, tmp_path, "corrupt",
                              raw_result_bytes=b'{"run_id": "R1", "gate_result": "PA')
    assert summary["outcome"] == "INCOMPLETE"
    assert summary["outcome_reason_code"] == "RESULT_JSON_INVALID"
    assert outer == 1
    # the summary we leave behind must still be strict JSON for an auditor
    raw = (tmp_path / "corrupt" / "gate_runner_summary.json").read_text(encoding="utf-8")
    json.loads(raw)


def test_stale_leftover_out_dir_is_refused_before_any_child_starts(runner, tmp_path):
    out_dir = tmp_path / "R1"
    out_dir.mkdir()
    (out_dir / "result.json").write_text(json.dumps(
        {"run_id": "OTHER_RUN", "gate_result": "PASS",
         "gates": {"fake_self_check": True}}), encoding="utf-8")
    marker = tmp_path / "child_started.marker"
    script, extra = write_fake_child(tmp_path)
    outer = invoke(runner, script, out_dir, child_extra=extra + ["--fake-marker",
                                                                str(marker)])
    assert not marker.exists(), "the child must not start on a refused out-dir"
    assert outer in set(runner.REFUSAL_EXIT_CODES.values())
    assert outer != 0
    assert not (out_dir / "gate_runner_summary.json").exists()
    refusal = json.loads((out_dir / "gate_runner_refusal.json").read_text(encoding="utf-8"))
    assert refusal["child_started"] is False
    assert refusal["refusal_code"] == "OUT_DIR_RUN_ID_MISMATCH"


def test_nonzero_child_rc_with_a_valid_looking_success_artifact_is_fail_closed(
        runner, tmp_path):
    outer, summary = run_case(runner, tmp_path, "rc_with_pass", returncode=3)
    assert summary["json_gate"] == "PASS"
    assert summary["json_gate_status"] == "OK"
    assert summary["json_gates_all_true"] is True
    assert summary["outcome"] == "UNEXPECTED_FAULT"
    assert summary["outcome_reason_code"] == "CHILD_NONZERO_EXIT_WITH_PASS_ARTIFACT"
    assert outer == 1


def test_stale_fault_record_is_quarantined_and_never_attributed_to_the_new_child(
        runner, tmp_path):
    out_dir = tmp_path / "R1"
    out_dir.mkdir()
    (out_dir / "fault_record.json").write_text(TYPED_FAULT_RECORD, encoding="utf-8")
    (out_dir / ".p2_gate_run_owner.json").write_text(
        json.dumps({"run_id": "R1"}), encoding="utf-8")
    script, extra = write_fake_child(tmp_path)
    outer = invoke(runner, script, out_dir, run_id="R1", child_extra=extra,
                   allow_reused_out_dir=True)
    summary = read_summary(out_dir)
    assert summary["quarantined_stale_fault_record"] is not None
    assert summary["outcome"] == "SUCCESS"
    assert outer == 0
    assert any(name.startswith("fault_record.json.prev-")
               for name in os.listdir(out_dir))


def test_only_success_exits_zero_across_the_whole_regression_matrix(runner, tmp_path):
    """Cross-cutting invariant: exit 0 <=> SUCCESS, for every case exercised here."""
    cases = [
        ({}, "SUCCESS"),
        ({"returncode": 2}, "UNEXPECTED_FAULT"),
        ({"returncode": 139}, "CRASH"),
        ({"write_result": False}, "INCOMPLETE"),
        ({"traceback_text": "BOOM"}, "UNEXPECTED_FAULT"),
        ({"fault_record": TYPED_FAULT_RECORD}, "UNEXPECTED_FAULT"),
    ]
    for index, (overrides, expected) in enumerate(cases):
        outer, summary = run_case(runner, tmp_path, "matrix_%d" % index, **dict(overrides))
        assert summary["outcome"] == expected, (overrides, summary["outcome"])
        assert (outer == 0) is (summary["outcome"] == "SUCCESS"), (overrides, outer)


# --------------------------------------------------------------------------------------
# A2 proof: replay the six REAL committed P0-07 runs
# --------------------------------------------------------------------------------------
@pytest.fixture(scope="module")
def replay_tool():
    assert REPLAY_PATH.is_file(), "replay tool missing: %s" % REPLAY_PATH
    spec = importlib.util.spec_from_file_location("p0f_replay_under_test", REPLAY_PATH)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_p0f_committed_evidence_replays_to_the_expected_classes(replay_tool):
    sha = replay_tool.resolve_ref(str(ROOT), replay_tool.EVIDENCE_COMMIT)
    if sha is None:
        pytest.skip("P0-07 evidence commit %s is not in this clone"
                    % replay_tool.EVIDENCE_COMMIT)
    rows, all_expected, resolved = replay_tool.replay(str(ROOT), replay_tool.EVIDENCE_COMMIT)
    assert resolved == replay_tool.EVIDENCE_COMMIT
    assert len(rows) == 6
    assert all_expected, json.dumps(rows, ensure_ascii=False, indent=1)
    # NOT ONE of the six real fault runs may be reported as a simulation SUCCESS
    assert [row["outcome"] for row in rows].count("SUCCESS") == 0
    assert all(row["exit_code"] != 0 for row in rows)
    by_run = {row["run_id"]: row for row in rows}
    # the four runs whose child rc was swallowed to 0 by Kit are still not SUCCESS
    assert by_run["P0F_exception_r02"]["child_returncode"] == 0
    assert by_run["P0F_exception_r02"]["outcome"] == "EXPECTED_FAULT"
    assert by_run["P0F_exception_r03"]["outcome"] == "EXPECTED_FAULT"
    assert by_run["P0F_jointlimit_r02"]["outcome"] == "EXPECTED_FAULT"
    assert by_run["P0F_jointlimit_r03"]["outcome"] == "EXPECTED_FAULT"
    # the one run that really died keeps its crash evidence
    assert by_run["P0F_jointlimit_r01"]["child_returncode"] == 139
    assert by_run["P0F_jointlimit_r01"]["outcome"] == "CRASH"


def test_p0f_replay_with_the_ignored_local_logs_proves_the_traceback_tier(
        replay_tool, tmp_path):
    logs_root = os.environ.get("P0F_LOGS_DIR")
    if not logs_root:
        candidate = ROOT.parent / "badmin_wt_p3g"
        logs_root = str(candidate) if candidate.is_dir() else None
    if not logs_root or not os.path.isdir(os.path.join(
            logs_root, replay_tool.EVIDENCE_SUBDIR, "P0F_exception_r01")):
        pytest.skip("no checkout with the ignored gate_child.*.log files (P0F_LOGS_DIR)")
    sha = replay_tool.resolve_ref(str(ROOT), replay_tool.EVIDENCE_COMMIT)
    if sha is None:
        pytest.skip("P0-07 evidence commit is not in this clone")
    rows, all_expected, _ = replay_tool.replay(str(ROOT), replay_tool.EVIDENCE_COMMIT,
                                               logs_dir=logs_root)
    assert all_expected, json.dumps(rows, ensure_ascii=False, indent=1)
    by_run = {row["run_id"]: row for row in rows}
    # With the stderr logs present the traceback pair is detected: rc 0 is not SUCCESS.
    assert by_run["P0F_exception_r01"]["traceback_tier"] == "TRACEBACK_PAIR"
    assert by_run["P0F_exception_r01"]["outcome"] == "EXPECTED_FAULT"
    assert by_run["P0F_exception_r01"]["reason_code"] == "TRACEBACK_ONLY"
    assert by_run["P0F_exception_r01"]["exit_code"] == 10
    assert [row["outcome"] for row in rows].count("SUCCESS") == 0
