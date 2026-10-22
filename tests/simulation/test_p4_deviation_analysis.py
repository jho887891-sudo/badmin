"""Tests for scripts/simulation/p4_deviation_analysis.py (R10-D).

Two layers:
  1. synthetic fixtures that prove the arithmetic is right (least squares,
     per-wheel radius, closed decomposition, staircase detection);
  2. a round-trip on the COMMITTED P4 evidence, which is skipped when the
     outputs tree is not present (e.g. a shallow clone).
"""

import json
import math
import os
import sys

import pytest

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, os.path.join(REPO, "scripts", "simulation"))

import p4_deviation_analysis as P4D  # noqa: E402

RUNS_ROOT = os.path.join(REPO, "outputs", "simulation", "rebot_b601_rs", "morph_one",
                         "p2_wc_gates")
DT = P4D.DT


# --------------------------------------------------------------------- fixtures
def make_run(v_cmd=0.30, r_ff=0.101459, r_plant=0.10298, kp=2.0, vy_meas_zero=True,
             steps=600, ramp_steps=180, hold_end=480):
    """Synthesise one closed-loop run exactly as the controller would produce it."""
    omega_ff = v_cmd / r_ff
    trim_limit = 0.05 * omega_ff
    rows = []
    y = 0.0
    v = 0.0
    trim = 0.0
    for k in range(steps + 1):
        if k > 0:
            phase = min(1.0, k / ramp_steps)
            vy_target = -v_cmd * phase
            if k > hold_end:
                vy_target = 0.0
                trim = 0.0
            if k == ramp_steps:
                trim = 0.0
            vy_meas = -v if not vy_meas_zero else 0.0
            trim += 0.5 * kp * DT * (vy_meas - vy_target)
            trim = max(-trim_limit, min(trim_limit, trim))
            omega = omega_ff + trim
            v = omega * r_plant            # the plant: chassis speed follows the wheel
            y += -v * DT                   # omega > 0 -> travel along -Y
        rows.append({
            "run_id": "SYNTH", "profile": "velocity_control", "step": k,
            "t_s": k * DT,
            "chassis_pos": [0.0, y, 0.347],
            "chassis_yaw_rad": 0.0,
            "wheel_omega_rad_s": {"wheel1": omega if k else 0.0,
                                  "wheel2": omega if k else 0.0,
                                  "wheel3": omega if k else 0.0,
                                  "wheel4": omega if k else 0.0},
            "wheel_pos_rad": {"wheel1": -y / r_plant, "wheel2": -y / r_plant,
                              "wheel3": -y / r_plant, "wheel4": -y / r_plant},
            "wheel_target_rad_s": {"wheel1": 0.0, "wheel2": 0.0,
                                   "wheel3": 0.0, "wheel4": 0.0},
            "contact_by_sensor_n": {"chassis": 0.0, "wheels": 400.0},
            "root_writes": 0, "finite": True,
        })
    result = {
        "run_id": "SYNTH", "profile": "velocity_control", "gate_result": "PASS",
        "proxy_parameters": {"wheel_radius_m": 0.103},
        "metrics": {
            "wheel_targets_rad_s_per_wheel": [0.0, 0.0, 0.0, 0.0],
            "velocity_control": {
                "commanded_speed_m_s": v_cmd, "commanded_after_clamp_m_s": v_cmd,
                "clamped": False, "steady_window_steps": [ramp_steps, hold_end],
                "steady_mean_speed_m_s": 0.0, "steady_relative_error": 0.0,
                "peak_speed_m_s": 0.0, "since_zero_max_speed_m_s": 0.0,
                "distance_m": 0.0, "kp": kp, "nominal_radius_m": r_ff},
            "wheel_delta_theta_rad": {}, "measured_rolling_radius_m": {},
            "measured_rolling_radius_rel_error": [],
        },
        "thresholds": {"speed_tracking_rel": 0.15},
        "gates": {"all_finite": True, "no_root_pose_write": True},
        "diagnostics": {"root_writes": 0, "drive_profile": "synthetic",
                        "friction_coefficient": 0.8, "payload_kg": 0.0},
    }
    return {"run_id": "SYNTH", "dir": None, "result": result, "rows": rows}


# ------------------------------------------------------------------------ maths
def test_lstsq_recovers_a_known_constant_speed():
    rows = [{"t_s": k * DT, "chassis_pos": [0.0, -0.4 * k * DT, 0.0]} for k in range(50)]
    slope, rms, n = P4D.lstsq_speed(rows, 0, 49)
    assert n == 50
    assert slope == pytest.approx(-0.4, rel=1e-12)
    assert rms == pytest.approx(0.0, abs=1e-12)


def test_consecutive_speeds_are_the_per_step_differences():
    rows = [{"chassis_pos": [0.0, -0.25 * k * DT, 0.0]} for k in range(11)]
    speeds = P4D.consecutive_speeds(rows)
    assert len(speeds) == 10
    assert all(v == pytest.approx(-0.25, rel=1e-12) for v in speeds)


def test_per_wheel_radius_is_distance_over_joint_angle():
    run = make_run(r_plant=0.10298)
    entry = P4D.analyse_velocity_run(run, 0.103)
    for wheel in P4D.WHEELS:
        assert entry["per_wheel"][wheel]["r_eff_m"] == pytest.approx(0.10298, rel=1e-9)
        assert entry["per_wheel"][wheel]["sign_consistent"] is True
    assert entry["r_eff_spread_rel"] < 1e-9


def test_decomposition_closes_and_finds_both_factors():
    """A run with dead feedback must show trim = +5 % and radius = r_plant / r_ff."""
    run = make_run(r_plant=0.10298, vy_meas_zero=True)
    entry = P4D.analyse_velocity_run(run, 0.103)
    dec = entry["decomposition"]
    assert dec["closure"] == pytest.approx(1.0, abs=1e-12)
    assert dec["factor_product"] == pytest.approx(dec["measured_ratio"], rel=1e-12)
    # over the whole pre-registered window the trim is still winding up, so the window
    # average sits below the +5 % clamp while the plateau reaches it
    assert 1.03 < dec["factor_trim"] < 1.05
    assert entry["feedback"]["plateau"]["factor_trim"] == pytest.approx(1.05, abs=2e-3)
    # the synthetic generator has a one-sample seam between the position integral and
    # the omega mean, so r_implied differs from the plant radius by ~1e-4
    assert dec["factor_radius"] == pytest.approx(0.10298 / 0.101459, rel=2e-4)
    assert dec["total_rel"] == pytest.approx(dec["factor_product"] - 1.0, rel=1e-9)


def test_dead_feedback_is_detected_as_a_staircase():
    """vy_meas == 0 -> the trim rises at exactly 0.5*kp*v_cmd per second."""
    run = make_run(v_cmd=0.30, vy_meas_zero=True)
    entry = P4D.analyse_velocity_run(run, 0.103)
    fb = entry["feedback"]
    assert fb["ramp_rate_ratio"] == pytest.approx(1.0, rel=2e-3)
    assert fb["steps_to_saturate_expected"] == pytest.approx(59.14, rel=0.01)
    assert fb["model_rms_rel_error"] < 2e-3


def test_live_feedback_does_not_saturate():
    """With a live measurement the trim never reaches the clamp.

    The live loop is slow: d(trim)/dk = -dt*r_plant*(trim - trim*), a time constant of
    ~1165 steps (~10 s), so the run has to be long enough to see the settled value.
    """
    run = make_run(v_cmd=0.30, vy_meas_zero=False, steps=12000, hold_end=11800)
    entry = P4D.analyse_velocity_run(run, 0.103)
    fb = entry["feedback"]
    assert fb["ramp_rate_ratio"] < 0.05         # the plant feedback kills the staircase
    assert abs(entry["decomposition"]["trim_added_rad_s"]) < 0.4 * entry["decomposition"][
        "trim_limit_rad_s"]
    # the settled instantaneous speed is exact; the window average still carries the
    # start-up transient, so it stays an order of magnitude below the dead-feedback case
    assert entry["steady_relative_error_recomputed"] < 2e-3


def test_geometric_radius_projection_is_reported():
    run = make_run(r_plant=0.10298)
    entry = P4D.analyse_velocity_run(run, 0.103)
    odom = entry["odometry"]
    # odometry uses r_ff where the plant uses r_plant, so the sign is inverted w.r.t.
    # the factor_radius convention: err = r_ff/r_plant - 1
    assert odom["feedforward_odom_error_rel"] == pytest.approx(
        0.101459 / 0.10298 - 1.0, rel=1e-5)
    # slip is defined against the radius implied by the SAME window, so the synthetic
    # one-sample seam cancels instead of dominating a 2e-4 quantity
    assert odom["slip_estimate_geometric"] == pytest.approx(
        1.0 - entry["r_implied_from_v_over_omega_m"] / 0.103, rel=1e-12)
    assert odom["slip_estimate_geometric"] == pytest.approx(1.0 - 0.10298 / 0.103, abs=5e-4)


# ------------------------------------------------------- committed-evidence layer
pytestmark_data = pytest.mark.skipif(
    not os.path.isdir(RUNS_ROOT), reason="committed P4 evidence not present")


@pytestmark_data
def test_committed_matrix_has_36_runs():
    groups = P4D.matrix_groups(RUNS_ROOT)
    assert len(groups) == 36
    radii = [g["measured_rolling_radius_mean_m"] for g in groups
             if g["measured_rolling_radius_mean_m"]]
    assert min(radii) == pytest.approx(0.101030, abs=1e-6)
    assert max(radii) == pytest.approx(0.101611, abs=1e-6)
    span_rel = (max(radii) - min(radii)) / P4D.mean(radii)
    assert span_rel < 0.01          # ~0.6 %: cannot explain a 6 % deviation


@pytestmark_data
def test_committed_velocity_runs_reproduce_and_close():
    for run_id in ("P4V_030_r13", "P4V_030_r14", "P4V_050_r13", "P4V_050_r14"):
        run = P4D.read_run(RUNS_ROOT, run_id)
        entry = P4D.analyse_velocity_run(run, 0.103)
        assert entry["speed"]["from_endpoints_m_s"] == pytest.approx(
            entry["speed"]["committed_steady_mean_m_s"], abs=5e-7)
        assert entry["steady_relative_error_recomputed"] == pytest.approx(
            entry["steady_relative_error_committed"], abs=5e-7)
        assert entry["decomposition"]["closure"] == pytest.approx(1.0, abs=1e-9)
        assert entry["feedback"]["ramp_rate_ratio"] == pytest.approx(1.0, rel=5e-3)
        assert 0.06 < entry["steady_relative_error_recomputed"] < 0.07


@pytestmark_data
def test_committed_thresholds_are_untouched():
    run = P4D.read_run(RUNS_ROOT, "P4V_030_r13")
    assert run["result"]["thresholds"]["speed_tracking_rel"] == 0.2
    vc = run["result"]["metrics"]["velocity_control"]
    assert vc["nominal_radius_m"] == pytest.approx(0.101459, abs=1e-9)
    assert vc["steady_relative_error"] == pytest.approx(0.060367703, abs=1e-9)


@pytestmark_data
def test_main_exits_zero_on_the_committed_tree(tmp_path):
    out = tmp_path / "report.json"
    md = tmp_path / "report.md"
    rc = P4D.main(["--runs-root", RUNS_ROOT, "--out", str(out), "--markdown", str(md)])
    assert rc == 0
    report = json.loads(out.read_text(encoding="utf-8"))
    assert report["verdict"] == "CONSISTENT"
    assert report["problems"] == []
    assert report["d2"]["run_count"] == 36
    assert report["d3"]["root_writes_total"] == 0
    assert "P4V_030_r13" in md.read_text(encoding="utf-8")
