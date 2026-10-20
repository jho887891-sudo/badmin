"""P0 external gate-runner tests. Standard library only; no Isaac Sim import."""

import json
from pathlib import Path
import subprocess
import sys

RUNNER = (Path(__file__).resolve().parents[2] /
          "scripts" / "simulation" / "p0_subprocess_gate_runner.py")


def run_case(tmp_path, report_payload, child_exit=0):
    report_path = tmp_path / "result.json"
    logs_dir = tmp_path / "logs"
    script = (
        "import json,sys; "
        "open(sys.argv[1],'w').write(sys.argv[2]); "
        "print('stdout-marker'); "
        "print('stderr-marker',file=sys.stderr); "
        "sys.exit(int(sys.argv[3]))"
    )
    result = subprocess.run(
        [sys.executable, str(RUNNER),
         "--report", str(report_path), "--output-dir", str(logs_dir),
         "--", sys.executable, "-c", script,
         str(report_path), json.dumps(report_payload), str(child_exit)],
        capture_output=True, text=True, timeout=20,
    )
    summary = json.loads((logs_dir / "gate_runner_summary.json").read_text())
    return result, summary, logs_dir


def test_explicit_pass_and_zero_process_exit(tmp_path):
    result, summary, logs = run_case(
        tmp_path, {"gates": {"setup": True}, "gate_result": "PASS"})
    assert result.returncode == 0
    assert summary["outer_exit_code"] == 0
    assert summary["child_returncode_observed"] == 0
    assert summary["gate"]["status"] == "PASS"
    assert "stdout-marker" in (logs / "child.stdout.log").read_text()
    assert "stderr-marker" in (logs / "child.stderr.log").read_text()


def test_json_fail_overrides_zero_kit_like_process_exit(tmp_path):
    result, summary, _ = run_case(
        tmp_path, {"acceptance": {"settle": False}}, child_exit=0)
    assert result.returncode != 0
    assert summary["child_returncode_observed"] == 0
    assert summary["gate"]["failed_checks"] == ["acceptance.settle"]


def test_nonzero_child_overrides_pass_json(tmp_path):
    result, summary, _ = run_case(
        tmp_path, {"gate_result": "PASS"}, child_exit=7)
    assert result.returncode != 0
    assert summary["child_returncode_observed"] == 7


def test_conflicting_gate_fields_fail(tmp_path):
    result, summary, _ = run_case(
        tmp_path, {"gate_result": "PASS", "gates": {"settle": False}})
    assert result.returncode != 0
    assert summary["gate"]["status"] == "FAIL"


def test_no_gate_fields_fail_closed(tmp_path):
    result, summary, _ = run_case(tmp_path, {"telemetry": {"z": 0}})
    assert result.returncode != 0
    assert summary["gate"]["status"] == "INVALID"


def test_nonboolean_gate_is_invalid(tmp_path):
    result, summary, _ = run_case(
        tmp_path, {"gates": {"settle": "false"}})
    assert result.returncode != 0
    assert summary["gate"]["status"] == "INVALID"


def test_stale_report_is_never_used(tmp_path):
    report = tmp_path / "result.json"
    report.write_text('{"gate_result":"PASS"}', encoding="utf-8")
    logs = tmp_path / "logs"
    result = subprocess.run(
        [sys.executable, str(RUNNER), "--report", str(report),
         "--output-dir", str(logs), "--", sys.executable, "-c", "print('ran')"],
        capture_output=True, text=True, timeout=20,
    )
    assert result.returncode != 0
    assert "Existing evidence" in result.stderr
    assert report.read_text() == '{"gate_result":"PASS"}'
    assert not (logs / "gate_runner_summary.json").exists()
