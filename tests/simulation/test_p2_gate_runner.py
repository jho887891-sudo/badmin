# -*- coding: utf-8 -*-
"""P0 regression: the outer PASS/FAIL gate must never let a FAIL look like a PASS.

Target: ``tools/simulation/p2_gate_runner.py``.

Contract under test::

    outer exit 0  <=>  child return code == 0  AND  result.json:gate_result == "PASS"

The load-bearing case is **child rc=0 with gate_result=FAIL**: Isaac Kit has twice
returned status 0 from a Python run that failed, so the parent process -- not the
child status -- has to decide acceptance.

No Isaac Sim and no GPU are touched: the "experiment" is a small generated Python
script that writes ``result.json`` and exits with a chosen code.  The only real
subprocesses are that fake child and (in one test) the gate runner itself.
"""
from __future__ import annotations

import importlib.util
import json
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
RUNNER_PATH = ROOT / "tools" / "simulation" / "p2_gate_runner.py"

# Tokens are substituted, not .format()ed, so the JSON braces in the payload stay literal.
_CHILD_TEMPLATE = '''# -*- coding: utf-8 -*-
"""Generated fake P2 experiment: no Isaac Sim, no GPU, no torch.

Writes result.json (unless told not to) and exits with a fixed code so the outer
gate contract can be exercised without any simulator.
"""
import argparse
import json
import os
import sys
import time


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--fake-gate", default="__GATE__")
    parser.add_argument("--fake-write", default="__WRITE__")
    args, extra = parser.parse_known_args()
    out_dir = os.path.abspath(args.out_dir)
    os.makedirs(out_dir, exist_ok=True)
    if args.fake_write == "1":
        payload = {
            "run_id": args.run_id,
            "gate_result": args.fake_gate,
            "metrics": {"fake_metric": 1.5},
            "child_extra_argv": list(extra),
        }
        with open(os.path.join(out_dir, "result.json"), "w", encoding="utf-8") as handle:
            json.dump(payload, handle, ensure_ascii=False, indent=1)
    print("fake child: gate=%s write=%s" % (args.fake_gate, args.fake_write), flush=True)
    print("fake child stderr marker", file=sys.stderr, flush=True)
    time.sleep(__SLEEP__)
    return __RC__


if __name__ == "__main__":
    raise SystemExit(main())
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
        .replace("__SLEEP__", repr(float(sleep_s))),
        encoding="utf-8",
    )
    return script


def invoke(runner, script, out_dir, run_id="R1", extra=(), timeout=None):
    """Call main() in-process so the returned outer exit code is directly assertable."""
    argv = ["--script", str(script), "--run-id", run_id, "--out-dir", str(out_dir)]
    if timeout is not None:
        argv += ["--timeout", str(timeout)]
    if extra:
        argv += ["--"] + [str(item) for item in extra]
    return runner.main(argv)


def read_summary(out_dir):
    path = Path(out_dir) / "gate_runner_summary.json"
    assert path.is_file(), "gate_runner_summary.json not written: %s" % path
    return json.loads(path.read_text(encoding="utf-8"))


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


# --------------------------------------------------------------------------------------
# 8. A stale result.json must not pass a child that writes nothing (used to be fail-OPEN:
#    out_dir reused from an earlier successful run -> old PASS inherited by a silent child)
# --------------------------------------------------------------------------------------
def test_stale_result_json_is_quarantined_and_cannot_pass_a_silent_child(runner, tmp_path):
    out_dir = tmp_path / "stale"
    out_dir.mkdir()
    stale = {"run_id": "OLD", "gate_result": "PASS", "metrics": {"old_metric": 1}}
    (out_dir / "result.json").write_text(json.dumps(stale), encoding="utf-8")
    script = write_fake_child(tmp_path, returncode=0, write_result=False)

    outer_exit = invoke(runner, script, out_dir)

    assert outer_exit == 1
    summary = read_summary(out_dir)
    assert summary["json_gate"] == "MISSING"
    assert summary["child_returncode"] == 0
    backup = Path(summary["quarantined_stale_result"])
    assert backup.is_file()
    assert json.loads(backup.read_text(encoding="utf-8")) == stale
    assert not (out_dir / "result.json").exists()


def test_fresh_run_quarantines_nothing(runner, tmp_path):
    out_dir = tmp_path / "fresh"
    script = write_fake_child(tmp_path, returncode=0, gate="PASS")

    assert invoke(runner, script, out_dir) == 0

    assert read_summary(out_dir)["quarantined_stale_result"] is None
