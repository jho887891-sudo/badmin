"""Tests for scripts/simulation/p3_summary.py: the auto-generated P3 evidence summary.

Every exit code asserted here is a REAL subprocess exit code of the script, not a
mocked return value.  Synthetic fixtures live in tmp_path; the committed evidence
under outputs/.../p2_wc_gates is also exercised as-is.
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[2]
SCRIPT = REPO / "scripts" / "simulation" / "p3_summary.py"
EVIDENCE = REPO / "outputs" / "simulation" / "rebot_b601_rs" / "morph_one" / "p2_wc_gates"

sys.path.insert(0, str(REPO / "scripts" / "simulation"))

from p3_summary import (  # noqa: E402
    GENERATIONS,
    classify_generation,
)

QUAT_PROBE = {
    "value": "xyzw",
    "probe_measured_yaw_rad": {"wxyz": 1.1e-07, "xyzw": 0.199999865},
}

DIRECTED_PASS = {
    "commanded_displacement_m": 0.3,
    "achieved_along_command_max_m": 0.3004,
    "achieved_along_command_final_m": 0.3003,
    "best_frame": 180,
    "achieved_ratio": 1.0014,
    "final_ratio": 1.0012,
    "cross_track_m": 1.4e-04,
    "pass": True,
}


# --------------------------------------------------------------------------- #
# fixture construction
# --------------------------------------------------------------------------- #
def make_run(root: Path, run_id: str, *, profile: str = "translate_xy",
             generation: str = "v1.1", gate_result: str = "PASS",
             outer_exit: int = 0, gates: dict | None = None,
             negative_control: dict | None = None,
             freeze_axis: str | None = None,
             achieved_ratio: float = 1.0,
             poses: str = "pose\n",
             steps: int = 3,
             with_result: bool = True,
             with_gate_runner: bool = True) -> Path:
    """Write one minimal but structurally faithful run directory."""
    run_dir = root / run_id
    run_dir.mkdir(parents=True)

    if gates is None:
        gates = {"all_finite": True, "mount_stable": True, "joint_hold": True,
                 "arm_does_not_touch_ground": True,
                 "translation_actually_measured": gate_result == "PASS"}
    metrics = {
        "steps": steps,
        "mount_position_deviation_max_m": 3e-08,
        "mount_rotation_deviation_max_rad": 9e-08,
        "peak_lin_speed_m_s": 0.0,
        "peak_jerk_m_s3": 0.0,
        "arm_peak_contact_n": 0.0,
        "command_amplitude": {"x_m": 0.3, "y_m": 0.0, "z_m": 0.0, "yaw_rad": 0.0},
    }
    # A negative control is only meaningful when the frozen component is NON-ZERO.  The
    # real P3_neg_frozen_y_r01 froze Y on a command whose Y was 0, so the freeze was a no-op
    # and the run proved nothing; the fixture therefore mirrors the meaningful case by
    # default and leaves the vacuous case to an explicit test.
    frozen = (negative_control or {}).get("frozen_axis")
    if frozen in ("x", "y", "z", "yaw"):
        metrics["command_amplitude"][{"x": "x_m", "y": "y_m", "z": "z_m",
                                      "yaw": "yaw_rad"}[frozen]] = 0.1
    diagnostics = {
        "quat_order_for_write_root_pose": dict(QUAT_PROBE),
        "root_to_chassis_offset_m": 0.261476607,
        "root_pose_is_offset_compensated": True,
    }
    if generation == "v1.1":
        metrics["directed_translation"] = {**DIRECTED_PASS,
                                          "achieved_ratio": achieved_ratio,
                                          "pass": gate_result == "PASS"}
        diagnostics["negative_control"] = negative_control
    elif generation == "v1.0" and negative_control is not None:
        raise AssertionError("negative_control is a v1.1-only artefact")

    result = {
        "run_id": run_id, "profile": profile, "scene_variant": "P3_DYNAMIC_MOUNT",
        "drive_mode": "DIAGNOSTIC_ROOT_POSE (NOT wheel drive)",
        "gate_result": gate_result, "metrics": metrics,
        "thresholds": {"mount_pos_m": 0.001, "motion_ratio": 0.9},
        "gates": gates, "diagnostics": diagnostics,
    }
    if with_result:
        (run_dir / "result.json").write_text(json.dumps(result), encoding="utf-8")
    (run_dir / "manifest.json").write_text(json.dumps({
        "run_id": run_id, "profile": profile, "steps": steps,
        "drive_mode": "DIAGNOSTIC_ROOT_POSE (NOT wheel drive)",
        **({"freeze_axis": freeze_axis} if freeze_axis is not None else {}),
    }), encoding="utf-8")
    if with_gate_runner:
        (run_dir / "gate_runner_summary.json").write_text(json.dumps({
            "run_id": run_id, "child_returncode": 0, "child_ok": True,
            "gates_ok": gate_result == "PASS", "json_gate": gate_result,
            "outer_exit": outer_exit, "elapsed_s": 1.0, "timed_out": False,
        }), encoding="utf-8")
    (run_dir / "raw_poses.jsonl").write_text(poses, encoding="utf-8")
    return run_dir


def run_script(*args: str) -> subprocess.CompletedProcess:
    return subprocess.run([sys.executable, str(SCRIPT), *args],
                          cwd=str(REPO), capture_output=True, text=True, encoding="utf-8")


def summary_of(root: Path, *extra: str) -> dict:
    out = root.parent / (root.name + "_summary.json")
    md = root.parent / (root.name + "_summary.md")
    proc = run_script("--gates-dir", str(root), "--json-out", str(out),
                      "--md-out", str(md), *extra)
    assert proc.returncode == 0, proc.stdout + proc.stderr
    return json.loads(out.read_text(encoding="utf-8"))


# --------------------------------------------------------------------------- #
# committed evidence
# --------------------------------------------------------------------------- #
@pytest.fixture(scope="module")
def committed(tmp_path_factory):
    out_dir = tmp_path_factory.mktemp("committed")
    json_out = out_dir / "p3_summary.json"
    md_out = out_dir / "p3_summary.md"
    proc = run_script("--gates-dir", str(EVIDENCE), "--json-out", str(json_out),
                      "--md-out", str(md_out))
    data = json.loads(json_out.read_text(encoding="utf-8"))
    return proc, data, md_out


def test_committed_evidence_exits_zero(committed):
    proc, data, _ = committed
    assert proc.returncode == 0, proc.stdout + proc.stderr
    assert data["totals"]["errors"] == 0
    # invariant, not a snapshot: the evidence set grows every round, so pin the properties
    # that must hold for ANY size instead of the count of one afternoon.
    assert data["totals"]["runs"] >= 12
    by_generation = data["totals"]["by_generation"]
    assert set(by_generation) == set(GENERATIONS)
    assert all(count > 0 for count in by_generation.values())
    assert sum(by_generation.values()) == data["totals"]["runs"]


def test_committed_evidence_generation_split_is_exact(committed):
    _, data, _ = committed
    by_gen = {gen: {run["run_id"] for run in data["runs"] if run["generation"] == gen}
              for gen in GENERATIONS}
    assert {
        "P3_compound_r09", "P3_compound_r11", "P3_static_r01", "P3_static_r09",
        "P3_static_r11", "P3_xy_r09", "P3_yaw_r09", "P3_yaw_r11"} <= by_gen["v1.0"]
    assert {
        "P3_neg_frozen_xy_r02", "P3_neg_frozen_z_r02", "P3_xy_r13", "P3_z_r13",
        "P3_yaw_r15", "P3_yaw_r16", "P3_static_r15", "P3_static_r16"} <= by_gen["v1.1"]
    assert not by_gen["v1.0"] & by_gen["v1.1"]


def test_committed_negative_controls_verified(committed):
    _, data, _ = committed
    controls = {c["run_id"]: c for c in data["negative_controls"]}
    assert len(controls) >= 2
    for control in controls.values():
        # meaningful controls must be observed failing; vacuous ones are reported as
        # warnings (they prove nothing) and are never counted as verified
        if control["verified"]:
            assert control["gate_result"] == "FAIL"
            assert control["outer_exit"] == 1
            # the attributable motion gate depends on the frozen axis: translation for
            # x/y/z, yaw for yaw.  It must be among the FALSE gates either way.
            assert ({"translation_actually_measured", "yaw_actually_measured"}
                    & set(control["gates_false"]))
        else:
            assert control["vacuous"] is True
    assert any(control["verified"] for control in controls.values())


def test_committed_coverage_gaps_are_reported(committed):
    _, data, _ = committed
    gaps = {(gap["generation"], gap["profile"]) for gap in data["coverage_gaps"]}
    # translate_z has zero v1.0 runs: the amendment cites P3_z_r09, which is not on
    # this branch, and P3_z_r13 is v1.1.  translate_xy has exactly one PASS per generation.
    assert ("v1.0", "translate_z") in gaps
    for gap in data["coverage_gaps"]:
        assert gap["independent_pass"] < gap["required"] == 2


def test_committed_directed_fields_present_only_for_v11(committed):
    _, data, _ = committed
    for run in data["runs"]:
        directed = run["directed_translation"]
        if run["generation"] == "v1.0":
            # the v1.0 gate never wrote the directed block
            assert directed is None
        elif directed is not None:
            # v1.1 writes it for the translation profiles; the yaw/static profiles have no
            # translation command, so absence is legitimate there
            assert set(directed) >= {"achieved_ratio", "best_frame", "cross_track_m"}


def test_markdown_carries_caveat_and_table(committed):
    _, data, md_out = committed
    text = md_out.read_text(encoding="utf-8")
    assert "NOT dynamics or stability evidence" in text
    assert "root_to_chassis_offset_m" in text
    for run in data["runs"]:
        assert run["run_id"] in text


def test_summary_is_byte_deterministic(tmp_path):
    a = tmp_path / "a.json"
    b = tmp_path / "b.json"
    assert run_script("--gates-dir", str(EVIDENCE), "--json-out", str(a),
                      "--md-out", str(tmp_path / "a.md")).returncode == 0
    assert run_script("--gates-dir", str(EVIDENCE), "--json-out", str(b),
                      "--md-out", str(tmp_path / "b.md")).returncode == 0
    assert a.read_bytes() == b.read_bytes()


# --------------------------------------------------------------------------- #
# generation classification
# --------------------------------------------------------------------------- #
def test_classify_directed_translation_is_v11():
    classified = classify_generation({"metrics": {"directed_translation": {}},
                                      "diagnostics": {}})
    assert classified["generation"] == "v1.1"
    assert classified["basis"] == ["metrics.directed_translation"]


def test_classify_negative_control_only_is_v11():
    classified = classify_generation({"metrics": {},
                                      "diagnostics": {"negative_control": {"frozen_axis": "z"}}})
    assert classified["generation"] == "v1.1"


def test_classify_quat_probe_alone_stays_v10_and_is_flagged():
    classified = classify_generation({"metrics": {"peak_jerk_m_s3": 0.0},
                                      "diagnostics": {"quat_order_for_write_root_pose":
                                                      dict(QUAT_PROBE)}})
    assert classified["generation"] == "v1.0"
    assert classified["quat_probe_two_entries"] is True
    assert classified["ambiguous_legacy_quat_probe"] is True


def test_classify_spec_quat_rule_flag_marks_v11():
    classified = classify_generation({"metrics": {},
                                      "diagnostics": {"quat_order_for_write_root_pose":
                                                      dict(QUAT_PROBE)}},
                                     spec_quat_rule=True)
    assert classified["generation"] == "v1.1"


def test_classify_single_entry_quat_probe_is_not_a_marker():
    classified = classify_generation({
        "metrics": {},
        "diagnostics": {"quat_order_for_write_root_pose": {
            "probe_measured_yaw_rad": {"xyzw": 0.2}}}}, spec_quat_rule=True)
    assert classified["generation"] == "v1.0"
    assert classified["quat_probe_two_entries"] is False


# --------------------------------------------------------------------------- #
# coverage audit
# --------------------------------------------------------------------------- #
def test_single_run_profile_is_a_coverage_gap(tmp_path):
    root = tmp_path / "gates"
    make_run(root, "P3_xy_r01", profile="translate_xy")
    data = summary_of(root)
    gaps = {(g["generation"], g["profile"]) for g in data["coverage_gaps"]}
    assert ("v1.1", "translate_xy") in gaps
    assert data["coverage"]["v1.1"]["translate_xy"]["independent_pass"] == 1
    assert data["coverage"]["v1.1"]["translate_xy"]["gap"] is True


def test_two_independent_passes_close_the_gap(tmp_path):
    root = tmp_path / "gates"
    make_run(root, "P3_xy_r01", profile="translate_xy", poses="a\n")
    make_run(root, "P3_xy_r02", profile="translate_xy", poses="b\n")
    data = summary_of(root)
    gaps = {(gap["generation"], gap["profile"]) for gap in data["coverage_gaps"]}
    assert ("v1.1", "translate_xy") not in gaps
    # the same profile still has zero runs under the other generation: coverage is
    # always reported per generation, never pooled
    assert gaps == {("v1.0", "translate_xy")}
    assert data["coverage"]["v1.1"]["translate_xy"]["independent_pass"] == 2


def test_duplicate_raw_poses_is_not_independent_evidence(tmp_path):
    root = tmp_path / "gates"
    make_run(root, "P3_xy_r01", profile="translate_xy", poses="same\n")
    make_run(root, "P3_xy_r02", profile="translate_xy", poses="same\n")
    data = summary_of(root)
    assert data["diagnostics"]["duplicate_evidence"] == [
        {"run_id": "P3_xy_r02", "duplicate_of": "P3_xy_r01"}]
    assert data["coverage"]["v1.1"]["translate_xy"]["independent_pass"] == 1
    assert data["coverage_gaps"]


def test_failing_runs_do_not_count_towards_coverage(tmp_path):
    root = tmp_path / "gates"
    make_run(root, "P3_xy_r01", profile="translate_xy", poses="a\n")
    make_run(root, "P3_xy_r02", profile="translate_xy", poses="b\n",
             gate_result="FAIL", outer_exit=1)
    data = summary_of(root)
    assert data["coverage"]["v1.1"]["translate_xy"]["independent_pass"] == 1
    assert data["coverage"]["v1.1"]["translate_xy"]["fails"] == ["P3_xy_r02"]


def test_negative_control_run_is_not_counted_as_coverage(tmp_path):
    root = tmp_path / "gates"
    make_run(root, "P3_neg_r01", profile="translate_z", negative_control={"frozen_axis": "z"},
             freeze_axis="z", gate_result="FAIL", outer_exit=1,
             gates={"all_finite": True, "mount_stable": True, "joint_hold": True,
                    "arm_does_not_touch_ground": True,
                    "translation_actually_measured": False})
    data = summary_of(root)
    assert data["coverage"]["v1.1"]["translate_z"]["runs"] == []
    assert data["coverage"]["v1.1"]["translate_z"]["independent_pass"] == 0


# --------------------------------------------------------------------------- #
# integrity failures must exit non-zero (real subprocess exit codes)
# --------------------------------------------------------------------------- #
def test_negative_control_that_passed_fails_the_script(tmp_path):
    root = tmp_path / "gates"
    make_run(root, "P3_neg_frozen_z_r01", profile="translate_z",
             negative_control={"frozen_axis": "z"}, freeze_axis="z",
             gate_result="PASS", outer_exit=0,
             gates={"all_finite": True, "mount_stable": True, "joint_hold": True,
                    "arm_does_not_touch_ground": True,
                    "translation_actually_measured": True})
    json_out = tmp_path / "s.json"
    proc = run_script("--gates-dir", str(root), "--json-out", str(json_out),
                      "--md-out", str(tmp_path / "s.md"))
    assert proc.returncode == 1
    data = json.loads(json_out.read_text(encoding="utf-8"))
    codes = [error["code"] for error in data["errors"]]
    assert codes == ["negative_control_passed"]
    assert data["negative_controls"][0]["verified"] is False
    assert "negative control did not fail" in proc.stderr


def test_negative_control_with_wrong_outer_exit_fails_the_script(tmp_path):
    root = tmp_path / "gates"
    make_run(root, "P3_neg_frozen_z_r01", profile="translate_z",
             negative_control={"frozen_axis": "z"}, freeze_axis="z",
             gate_result="FAIL", outer_exit=0,
             gates={"all_finite": True, "mount_stable": True, "joint_hold": True,
                    "arm_does_not_touch_ground": True,
                    "translation_actually_measured": False})
    proc = run_script("--gates-dir", str(root), "--json-out", str(tmp_path / "s.json"),
                      "--md-out", str(tmp_path / "s.md"))
    assert proc.returncode == 1


def test_correctly_failing_negative_control_keeps_exit_zero(tmp_path):
    root = tmp_path / "gates"
    make_run(root, "P3_neg_frozen_z_r01", profile="translate_z",
             negative_control={"frozen_axis": "z"}, freeze_axis="z",
             gate_result="FAIL", outer_exit=1,
             gates={"all_finite": True, "mount_stable": True, "joint_hold": True,
                    "arm_does_not_touch_ground": True,
                    "translation_actually_measured": False})
    proc = run_script("--gates-dir", str(root), "--json-out", str(tmp_path / "s.json"),
                      "--md-out", str(tmp_path / "s.md"))
    assert proc.returncode == 0, proc.stdout + proc.stderr


def test_missing_result_json_fails_the_script(tmp_path):
    root = tmp_path / "gates"
    make_run(root, "P3_xy_r01", profile="translate_xy", with_result=False)
    json_out = tmp_path / "s.json"
    proc = run_script("--gates-dir", str(root), "--json-out", str(json_out),
                      "--md-out", str(tmp_path / "s.md"))
    assert proc.returncode == 1
    data = json.loads(json_out.read_text(encoding="utf-8"))
    assert [error["code"] for error in data["errors"]] == ["missing_json"]
    assert data["totals"]["runs"] == 0


def test_missing_gate_runner_summary_fails_the_script(tmp_path):
    root = tmp_path / "gates"
    make_run(root, "P3_xy_r01", profile="translate_xy", with_gate_runner=False)
    json_out = tmp_path / "s.json"
    proc = run_script("--gates-dir", str(root), "--json-out", str(json_out),
                      "--md-out", str(tmp_path / "s.md"))
    assert proc.returncode == 1
    data = json.loads(json_out.read_text(encoding="utf-8"))
    assert [error["code"] for error in data["errors"]] == ["missing_gate_runner_summary"]
    assert data["runs"][0]["outer_exit"] is None


def test_missing_manifest_is_tolerated(tmp_path):
    root = tmp_path / "gates"
    run_dir = make_run(root, "P3_xy_r01", profile="translate_xy")
    (run_dir / "manifest.json").unlink()
    data = summary_of(root)
    assert data["totals"]["runs"] == 1
    assert data["runs"][0]["profile"] == "translate_xy"


def test_no_run_directories_exits_two(tmp_path):
    root = tmp_path / "empty"
    root.mkdir()
    proc = run_script("--gates-dir", str(root), "--json-out", str(tmp_path / "s.json"),
                      "--md-out", str(tmp_path / "s.md"))
    assert proc.returncode == 2


# --------------------------------------------------------------------------- #
# aggregates must never be pooled across generations
# --------------------------------------------------------------------------- #
def test_aggregates_are_per_generation_and_never_pooled(tmp_path):
    root = tmp_path / "gates"
    # same profile, both generations, deliberately different ratios
    make_run(root, "P3_xy_r01", profile="translate_xy", generation="v1.0",
             achieved_ratio=0.42, poses="old\n")
    make_run(root, "P3_xy_r13", profile="translate_xy", generation="v1.1",
             achieved_ratio=1.0, poses="new\n")
    data = summary_of(root)

    assert set(data["aggregates"]) == set(GENERATIONS)
    for pooled in ("overall", "all", "total", "combined"):
        assert pooled not in data["aggregates"]
    assert "achieved_ratio" not in data["aggregates"]["v1.0"]
    assert data["aggregates"]["v1.0"]["runs"] == 1
    assert data["aggregates"]["v1.1"]["runs"] == 1
    assert data["aggregates"]["v1.0"]["directed_runs"] == 0
    assert data["aggregates"]["v1.1"]["directed_achieved_ratio_min"] == 1.0
    # the v1.0 run must not leak into the v1.1 aggregate or vice versa
    assert data["aggregates"]["v1.0"]["run_ids"] == ["P3_xy_r01"]
    assert data["aggregates"]["v1.1"]["run_ids"] == ["P3_xy_r13"]


def test_generation_aggregate_metrics_come_only_from_own_runs(tmp_path):
    root = tmp_path / "gates"
    old = make_run(root, "P3_xy_r01", profile="translate_xy", generation="v1.0",
                   poses="old\n")
    new = make_run(root, "P3_xy_r13", profile="translate_xy", generation="v1.1",
                   poses="new\n")
    payload = json.loads((old / "result.json").read_text(encoding="utf-8"))
    payload["metrics"]["mount_position_deviation_max_m"] = 9.0
    (old / "result.json").write_text(json.dumps(payload), encoding="utf-8")
    payload = json.loads((new / "result.json").read_text(encoding="utf-8"))
    payload["metrics"]["mount_position_deviation_max_m"] = 1e-8
    (new / "result.json").write_text(json.dumps(payload), encoding="utf-8")

    data = summary_of(root)
    assert data["aggregates"]["v1.0"]["mount_position_deviation_max_m"] == 9.0
    assert data["aggregates"]["v1.1"]["mount_position_deviation_max_m"] == 1e-8
    # no global max that would silently mix them
    assert "mount_position_deviation_max_m" not in data["totals"]


def test_diagnostic_drive_caveat_is_always_emitted(tmp_path):
    root = tmp_path / "gates"
    make_run(root, "P3_xy_r13", profile="translate_xy")
    data = summary_of(root)
    assert "NOT dynamics or stability evidence" in data["caveats"][0]
    assert any("NOT dynamics or stability evidence" in warning for warning in data["warnings"])


def test_offset_and_frames_are_reported(tmp_path):
    root = tmp_path / "gates"
    make_run(root, "P3_xy_r13", profile="translate_xy", steps=3)
    data = summary_of(root)
    diag = data["diagnostics"]
    assert diag["root_to_chassis_offset_m_values"] == [0.261476607]
    assert diag["raw_poses_frames_observed"] == [1]  # fixture has one line
    assert diag["raw_poses_frames_expected"] == [4]
    assert any("FRAME COUNT" in warning for warning in data["warnings"])
