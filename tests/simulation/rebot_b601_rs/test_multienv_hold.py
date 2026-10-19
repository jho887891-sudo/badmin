# -*- coding: utf-8 -*-
"""Multi-env gate: every env must hold its OWN pose, on ALL joints, for the whole measured window.

Evidence: scripts/simulation/check_multienv_hold.py on the GPU host, merged by
tools/merge_multienv_evidence.py into MULTIENV_hold.json.

Design rules this file is held to (from the code review of 70843e7):
  * the hold-duration assertion must require a real minimum, not compare a value with itself;
  * a run whose verdict is not all_envs_hold makes the whole acceptance FAIL - no 'continue';
  * limits are checked for all eight DOFs over the full time series, not joint2/joint3 at the end;
  * an independent reset must be proven to restore the rest pose AND to leave the other envs alone;
  * the recorded velocity series must be consistent with the recorded position series.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "simulation"))

from badminton_scene.scene_layout import REBOT_RS
from src.simulation.rebot_b601_rs.joint_limits import JOINT_LIMITS  # noqa: E402

EVIDENCE = ROOT / "outputs" / "simulation" / "rebot_b601_rs" / "MULTIENV_hold.json"
REQUIRED_COUNTS = [2, 4, 16, 32, 64, 128]
MIN_HOLD_SECONDS = 2.0
MAX_HOLD_ERROR_RAD = 0.05
MAX_VEL_POS_MISMATCH_RAD = 0.002  # 2-second static hold, arm joints only
LIMIT_TOL = 1e-6


@pytest.fixture(scope="module")
def ev():
    assert EVIDENCE.is_file(), (
        "missing multi-env evidence; run scripts/simulation/check_multienv_hold.py on the GPU host")
    return json.loads(EVIDENCE.read_text(encoding="utf-8"))


def test_all_required_env_counts_are_present(ev):
    """No skipping: a climb that stops early leaves counts missing and the gate fails."""
    counts = [run["num_envs"] for run in ev["runs"]]
    assert counts == REQUIRED_COUNTS, counts


def test_verified_physx_configuration_is_used(ev):
    expected = ev["physics_configuration"]
    assert expected["solver_type"] == 1
    assert expected["enable_external_forces_every_iteration"] is True
    assert expected["scope"].startswith("static hold")
    for run in ev["runs"]:
        actual = run["physics_configuration"]
        assert actual["solver_type"] == 1
        assert actual["enable_external_forces_every_iteration"] is True


def test_no_run_reports_a_failed_hold(ev):
    """Explicit and unconditional: one bad run fails the suite instead of being skipped."""
    failed = [run["num_envs"] for run in ev["runs"] if not run["verdict"]["all_envs_hold"]]
    assert failed == [], "runs that did not hold: " + str(failed)


def test_the_hold_window_is_at_least_two_seconds(ev):
    for run in ev["runs"]:
        assert run["hold_seconds"] >= MIN_HOLD_SECONDS, run["num_envs"]
        assert run["steps_measured"] * run["physics_dt"] >= MIN_HOLD_SECONDS, run["num_envs"]


def test_every_env_holds_its_own_target(ev):
    for run in ev["runs"]:
        assert len(run["envs"]) == run["num_envs"], run["num_envs"]
        assert run["verdict"]["all_envs_finite"] is True, run["num_envs"]
        for env in run["envs"]:
            assert env["worst_window_hold_error_rad"] <= MAX_HOLD_ERROR_RAD, (
                run["num_envs"], env["env_id"], env["worst_window_hold_error_rad"])
            assert env["hold_error_rad"] <= env["worst_window_hold_error_rad"] + 2e-6
        assert run["verdict"]["max_window_hold_error_rad"] <= MAX_HOLD_ERROR_RAD


def test_targets_actually_differ_between_environments(ev):
    for run in ev["runs"]:
        distinct = {tuple(round(v, 6) for v in env["target"]) for env in run["envs"]}
        assert len(distinct) >= 2, run["num_envs"]


def test_all_eight_dofs_stay_inside_limits_for_the_whole_run(ev):
    """Running extrema over every step and every env - not the final joint2/joint3 snapshot."""
    for run in ev["runs"]:
        trace = run["limit_trace"]
        assert set(trace) == set(run["joint_names"]) and len(trace) == 8, run["num_envs"]
        for name, rec in trace.items():
            lo, hi = rec["limit"]
            assert rec["inside"] is True, (run["num_envs"], name, rec)
            assert rec["observed_min"] >= lo - LIMIT_TOL, (run["num_envs"], name, rec)
            assert rec["observed_max"] <= hi + LIMIT_TOL, (run["num_envs"], name, rec)
            # the official table is the reference, not whatever the run recorded
            assert abs(lo - JOINT_LIMITS[name].angle.lower) <= 1e-6, name
            assert abs(hi - JOINT_LIMITS[name].angle.upper) <= 1e-6, name
        assert run["first_limit_violation"] == {}, (run["num_envs"], run["first_limit_violation"])


def test_recorded_positions_are_internally_consistent(ev):
    """Whatever the velocity channel says, the position series must agree with itself."""
    for run in ev["runs"]:
        assert run["probe_series_rows"] == run["steps_measured"], run["num_envs"]
        for name, rec in run["velocity_position_consistency"].items():
            # three values rounded to 1e-6 in the evidence, so allow 3e-6 of rounding slack
            assert abs(rec["end"] - rec["start"] - rec["measured_delta_rad"]) <= 3e-6, (
                run["num_envs"], name, rec)


def test_arm_velocity_integral_matches_position_during_static_hold(ev):
    """TGS external-forces-per-iteration must make measured qdot usable at REST.

    This is deliberately NOT a dynamic control certification. For moving-arm
    trajectories use the separate step probe and its segment-wise gate.
    """
    for run in ev["runs"]:
        assert run["verdict"]["velocity_position_consistent"] is True, run["num_envs"]
        assert run["hold_seconds"] >= MIN_HOLD_SECONDS
        for name in ("joint1", "joint2", "joint3", "joint4", "joint5", "joint6"):
            mismatch = run["velocity_position_consistency"][name]["mismatch_rad"]
            assert abs(mismatch) <= MAX_VEL_POS_MISMATCH_RAD, (
                run["num_envs"], name, mismatch)



def test_independent_reset_restores_rest_and_leaves_others_alone(ev):
    for run in ev["runs"]:
        info = run["independent_reset"]
        assert info["reset_env"] in range(run["num_envs"]), run["num_envs"]
        # 1. the reset env really goes back to the rest pose ...
        assert info["env_immediate_error_vs_rest_rad"] <= 1e-4, (run["num_envs"], info)
        assert info["env_after_60_steps_error_vs_rest_rad"] <= MAX_HOLD_ERROR_RAD, (
            run["num_envs"], info)
        # 2. ... and its drive target is the rest pose, not the old target
        expected = [REBOT_RS["rest_joint_pos"].get(name, 0.0)
                    for name in run["joint_names"]]
        actual = info["env_target_after_reset"]
        assert len(actual) == len(expected)
        assert all(abs(a - b) <= 1e-8 for a, b in zip(actual, expected)), (
            run["num_envs"], actual, expected)
        assert info["env_target_matches_rest"] is True, run["num_envs"]
        assert info["env_target_after_60_steps_matches_rest"] is True, run["num_envs"]
        # 3. the other envs keep their targets ...
        assert info["others_target_unchanged"] is True, run["num_envs"]
        assert info["others_target_unchanged_after_steps"] is True, run["num_envs"]
        # 4. ... and their states do not move
        for env_id, drift in info["other_env_drift_rad"].items():
            assert drift <= 1e-3, (run["num_envs"], env_id, drift)
        assert run["verdict"]["independent_reset_ok"] is True, run["num_envs"]


def test_performance_is_recorded_for_every_run(ev):
    for run in ev["runs"]:
        perf = run["performance"]
        assert perf["wall_seconds"] > 0, run["num_envs"]
        assert perf["steps_per_second"] > 0, run["num_envs"]
        assert perf["num_envs"] == run["num_envs"]
