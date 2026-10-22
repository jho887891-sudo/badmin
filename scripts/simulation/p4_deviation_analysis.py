#!/usr/bin/env python3
"""R10-D: independent recomputation of the P4-09 ~6 % speed deviation.

CPU-only analysis.  Reads the COMMITTED raw_poses.jsonl / result.json of the P4
evidence runs and recomputes, from scratch and without reusing any of the
numbers stored in result.json:

  D1  chassis speed (consecutive differences AND least-squares fit), per-wheel
      joint displacement, per-wheel effective rolling radius, achieved/commanded
      ratio, per-wheel implied radius r = v/omega, odometry vs ground truth and
      a slip estimate; then a CLOSED decomposition of the deviation.
  D2  consistency audit of the existing 36-run single-factor matrix
      (friction / payload / wheel speed / no-drive / reverse) and a bound on how
      much of the deviation the recorded rolling-radius spread can explain.
  D3  sign / dimension / saturation audit: four-wheel sign convention for
      straight roll, differential signs for a turn, commanded vs measured wheel
      omega, the +-50 rad/s clamp, and root_writes == 0 (no diagnostic root-pose
      driving).
  D4  per-run per-gate odometry / stop / limit acceptance table plus an explicit
      P4_REAL gap list.

Nothing is written into the run directories.  Thresholds are NOT modified: the
pre-registered 15 % gate is reported as-is and the analysis only explains the
6 %.

Exit codes: 0 = all recomputations reproduced the committed numbers and every
audit invariant held; 2 = at least one mismatch / invariant violation
(the mismatches are printed and stored under "problems").
"""

from __future__ import annotations

import argparse
import json
import math
import os

DT = 1.0 / 120.0
WHEELS = ("wheel1", "wheel2", "wheel3", "wheel4")
VELOCITY_PROFILE = "velocity_control"
STRAIGHT_PROFILES = ("roll_neg_y", "roll_pos_y", "velocity_control")
MATRIX_PREFIXES = ("P4f_", "P4L_", "P4S_", "P4N_", "P4R_")
RADIUS_TOL = 1e-9          # closure tolerance of the algebraic decomposition
REPRO_TOL = 5e-7           # recomputed vs committed metric (committed rounded to 9 dp)


# --------------------------------------------------------------------------- io
def load_jsonl(path):
    rows = []
    with open(path, "r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def load_json(path):
    with open(path, "r", encoding="utf-8") as handle:
        return json.load(handle)


def read_run(runs_root, run_id):
    directory = os.path.join(runs_root, run_id)
    result_path = os.path.join(directory, "result.json")
    poses_path = os.path.join(directory, "raw_poses.jsonl")
    return {
        "run_id": run_id,
        "dir": directory,
        "result": load_json(result_path) if os.path.exists(result_path) else None,
        "rows": load_jsonl(poses_path) if os.path.exists(poses_path) else None,
    }


# ------------------------------------------------------------------ kinematics
def consecutive_speeds(rows, axis=1, dt=DT):
    """v_k = (x_k - x_(k-1)) / dt, index k = 1 .. n-1."""
    return [(rows[k]["chassis_pos"][axis] - rows[k - 1]["chassis_pos"][axis]) / dt
            for k in range(1, len(rows))]


def lstsq_speed(rows, i0, i1, axis=1):
    """Ordinary least squares slope of position(axis) against t over [i0, i1].

    Returns (slope, rms_residual, n) so a reviewer can see the fit quality.
    """
    xs = [rows[k]["t_s"] for k in range(i0, i1 + 1)]
    ys = [rows[k]["chassis_pos"][axis] for k in range(i0, i1 + 1)]
    n = len(xs)
    mean_x = sum(xs) / n
    mean_y = sum(ys) / n
    sxx = sum((x - mean_x) ** 2 for x in xs)
    sxy = sum((x - mean_x) * (y - mean_y) for x, y in zip(xs, ys))
    slope = sxy / sxx if sxx else 0.0
    resid = [y - (mean_y + slope * (x - mean_x)) for x, y in zip(xs, ys)]
    rms = math.sqrt(sum(r * r for r in resid) / n)
    return slope, rms, n


def slope_of(rows, i0, i1, block, key):
    """Least squares slope of one telemetry channel against t."""
    xs = [rows[k]["t_s"] for k in range(i0, i1 + 1)]
    ys = [rows[k][block][key] for k in range(i0, i1 + 1)]
    n = len(xs)
    mx = sum(xs) / n
    my = sum(ys) / n
    sxx = sum((x - mx) ** 2 for x in xs)
    sxy = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
    return sxy / sxx if sxx else 0.0


def mean(values):
    return sum(values) / len(values) if values else 0.0


def stdev(values):
    if len(values) < 2:
        return 0.0
    m = mean(values)
    return math.sqrt(sum((v - m) ** 2 for v in values) / (len(values) - 1))


# ------------------------------------------------ D1: one velocity-control run
def analyse_velocity_run(run, r_geom):
    """Independent recomputation for one P4V_* run.

    Every quantity is derived from raw_poses.jsonl plus the two frozen constants
    the run itself declares in result.json (the feed-forward radius and the
    pre-registered steady window).  No committed metric is reused as an input.
    """
    rows = run["rows"]
    result = run["result"]
    vc = result["metrics"]["velocity_control"]
    r_ff = float(vc["nominal_radius_m"])
    i0, i1 = (int(x) for x in vc["steady_window_steps"])
    v_cmd = float(vc["commanded_after_clamp_m_s"])
    kp = float(vc["kp"])

    n = len(rows)
    if i1 >= n:
        raise SystemExit("%s: steady window %s outside %d frames" % (run["run_id"], (i0, i1), n))

    # (a) chassis speed -------------------------------------------------------
    speeds = consecutive_speeds(rows)
    window_speeds = speeds[i0:i1]                    # same slicing as the gate
    v_consecutive = mean([abs(v) for v in window_speeds])
    dy = rows[i1]["chassis_pos"][1] - rows[i0]["chassis_pos"][1]
    span = rows[i1]["t_s"] - rows[i0]["t_s"]
    v_endpoint = abs(dy / span)
    v_lstsq, v_lstsq_rms, v_lstsq_n = lstsq_speed(rows, i0, i1)
    v_lstsq = abs(v_lstsq)

    # (b) per-wheel joint displacement and effective radius -------------------
    per_wheel = {}
    for wheel in WHEELS:
        dtheta = rows[i1]["wheel_pos_rad"][wheel] - rows[i0]["wheel_pos_rad"][wheel]
        omega_mean = mean([rows[k]["wheel_omega_rad_s"][wheel] for k in range(i0, i1 + 1)])
        # a straight roll needs travel and rotation of OPPOSITE sign
        # (omega > 0 about +X drives the chassis along -Y)
        sign_ok = (dy * dtheta) < 0.0
        r_eff = abs(dy) / abs(dtheta) if dtheta else None
        per_wheel[wheel] = {
            "delta_theta_rad": dtheta,
            "omega_mean_rad_s": omega_mean,
            "sign_consistent": bool(sign_ok),
            "r_eff_m": r_eff,
            "r_from_v_over_omega_m": (abs(dy) / span) / abs(omega_mean) if omega_mean else None,
            "achieved_over_commanded": abs(dy) / span / v_cmd if v_cmd else None,
        }

    r_values = [per_wheel[w]["r_eff_m"] for w in WHEELS if per_wheel[w]["r_eff_m"]]
    r_spread = max(r_values) - min(r_values)
    r_spread_rel = r_spread / mean(r_values)

    # (c) ratio and implied radius -------------------------------------------
    ratio = abs(dy) / span / v_cmd
    omega_mean_all = mean([per_wheel[w]["omega_mean_rad_s"] for w in WHEELS])
    r_implied = abs(dy) / span / abs(omega_mean_all)
    r_pure_radius_hypothesis = v_cmd / abs(omega_mean_all)

    # (d) odometry vs ground truth + slip ------------------------------------
    dtheta_mean = mean([abs(per_wheel[w]["delta_theta_rad"]) for w in WHEELS])
    odom_ff = r_ff * dtheta_mean                  # what the feed-forward radius predicts
    odom_geom = r_geom * dtheta_mean              # what the P1 v2 geometric radius predicts
    slip_geom = 1.0 - abs(dy) / span / (abs(omega_mean_all) * r_geom)

    # ---- closed decomposition of the deviation ------------------------------
    omega_ff = v_cmd / r_ff                       # the feed-forward wheel speed
    trim_mean = omega_mean_all - omega_ff          # what the integral trim added
    factor_trim = omega_mean_all / omega_ff
    factor_radius = r_implied / r_ff
    closure = factor_trim * factor_radius / ratio  # must be 1 by algebra

    # ---- feedback liveness test (is vy_meas really used?) ------------------
    # During the hold phase the law is  trim += 0.5*kp*dt*(vy_meas - vy_target).
    # If vy_meas == 0 the trim is a pure staircase of slope 0.5*kp*dt*v_cmd.
    # skip the first few steps: the trim is reset AT i0, so omega(i0) still carries
    # the value produced by the previous (ramp-phase) target.
    ramp_a, ramp_b = i0 + 5, min(i1 - 1, i0 + 55)
    ramp_rate_per_s = slope_of(rows, ramp_a, ramp_b, "wheel_omega_rad_s", "wheel1")
    # the staircase slope, expressed both per second and per physics step
    staircase_per_s = 0.5 * kp * v_cmd
    staircase_per_step = 0.5 * kp * DT * v_cmd
    trim_limit = 0.05 * omega_ff
    plateau_target = 1.05 * omega_ff
    steps_to_limit = trim_limit / staircase_per_step if staircase_per_step else None

    # plateau window = the part of the hold window after the staircase has provably
    # saturated (i0 + ceil(limit / staircase slope)); deterministic, not curve-fitted.
    plateau_start = i0 + int(math.ceil(steps_to_limit)) if steps_to_limit else None
    if plateau_start is None or plateau_start >= i1:
        plateau = None
    else:
        p_dy = rows[i1]["chassis_pos"][1] - rows[plateau_start]["chassis_pos"][1]
        p_dt = rows[i1]["t_s"] - rows[plateau_start]["t_s"]
        p_dtheta = mean([rows[i1]["wheel_pos_rad"][w] - rows[plateau_start]["wheel_pos_rad"][w]
                         for w in WHEELS])
        p_omega = mean([mean([rows[k]["wheel_omega_rad_s"][w]
                              for k in range(plateau_start, i1 + 1)]) for w in WHEELS])
        p_v = abs(p_dy) / p_dt
        p_ratio = p_v / v_cmd
        p_r = abs(p_dy) / abs(p_dtheta)
        plateau = {
            "steps": [plateau_start, i1],
            "duration_s": p_dt,
            "speed_m_s": p_v,
            "achieved_over_commanded": p_ratio,
            "steady_relative_error": abs(p_ratio - 1.0),
            "r_eff_m": p_r,
            "omega_mean_rad_s": p_omega,
            "factor_trim": p_omega / omega_ff,
            "factor_radius": p_r / r_ff,
        }

    # ---- reconstruction of the recorded trajectory from the law only -------
    # omega_pred(k) = omega_ff + min(limit, 0.5*kp*dt*v_cmd*(k - i0))  [vy_meas = 0]
    pred_err = []
    for k in range(i0, i1 + 1):
        pred = omega_ff + min(trim_limit, staircase_per_step * (k - i0))
        actual = rows[k]["wheel_omega_rad_s"]["wheel1"]
        pred_err.append(abs(actual - pred) / omega_ff)
    model_rms = math.sqrt(sum(e * e for e in pred_err) / len(pred_err))
    model_max = max(pred_err)

    return {
        "run_id": run["run_id"],
        "profile": result["profile"],
        "gate_result": result["gate_result"],
        "commanded_speed_m_s": v_cmd,
        "commanded_raw_speed_m_s": float(vc["commanded_speed_m_s"]),
        "clamped": bool(vc["clamped"]),
        "steady_window_steps": [i0, i1],
        "speed": {
            "from_consecutive_positions_m_s": v_consecutive,
            "from_endpoints_m_s": v_endpoint,
            "from_least_squares_m_s": v_lstsq,
            "lstsq_rms_residual_m": v_lstsq_rms,
            "lstsq_samples": v_lstsq_n,
            "committed_steady_mean_m_s": float(vc["steady_mean_speed_m_s"]),
        },
        "achieved_over_commanded": ratio,
        "steady_relative_error_recomputed": abs(ratio - 1.0),
        "steady_relative_error_committed": float(vc["steady_relative_error"]),
        "per_wheel": per_wheel,
        "r_eff_spread_m": r_spread,
        "r_eff_spread_rel": r_spread_rel,
        "r_eff_mean_m": mean(r_values),
        "r_eff_stdev_m": stdev(r_values),
        "r_implied_from_v_over_omega_m": r_implied,
        "r_pure_radius_hypothesis_m": r_pure_radius_hypothesis,
        "r_feedforward_m": r_ff,
        "r_geometric_m": r_geom,
        "odometry": {
            "delta_theta_mean_rad": dtheta_mean,
            "ground_truth_distance_m": abs(dy),
            "feedforward_odom_m": odom_ff,
            "feedforward_odom_error_mm": (odom_ff - abs(dy)) * 1000.0,
            "feedforward_odom_error_rel": odom_ff / abs(dy) - 1.0,
            "geometric_odom_m": odom_geom,
            "geometric_odom_error_mm": (odom_geom - abs(dy)) * 1000.0,
            "geometric_odom_error_rel": odom_geom / abs(dy) - 1.0,
            "slip_estimate_geometric": slip_geom,
        },
        "decomposition": {
            "omega_feedforward_rad_s": omega_ff,
            "omega_mean_measured_rad_s": omega_mean_all,
            "trim_added_rad_s": trim_mean,
            "trim_limit_rad_s": trim_limit,
            "trim_fraction_of_limit": trim_mean / trim_limit if trim_limit else None,
            "factor_trim": factor_trim,
            "factor_radius": factor_radius,
            "factor_product": factor_trim * factor_radius,
            "measured_ratio": ratio,
            "closure": closure,
            "trim_contribution_rel": factor_trim - 1.0,
            "radius_contribution_rel": factor_radius - 1.0,
            "total_rel": ratio - 1.0,
        },
        "feedback": {
            "assumption": "trim += 0.5*kp*dt*(vy_meas - vy_target); if vy_meas == 0 the trim "
                          "is a staircase of slope 0.5*kp*dt*v_cmd",
            "ramp_rate_measured_rad_s2": ramp_rate_per_s,
            "ramp_rate_expected_if_vy_meas_zero_rad_s2": staircase_per_s,
            "ramp_rate_measured_rad_s_per_step": ramp_rate_per_s * DT,
            "ramp_rate_expected_rad_s_per_step": staircase_per_step,
            "ramp_rate_ratio": (ramp_rate_per_s / staircase_per_s
                                if staircase_per_s else None),
            "steps_to_saturate_expected": steps_to_limit,
            "plateau_target_omega_rad_s": plateau_target,
            "model_rms_rel_error": model_rms,
            "model_max_rel_error": model_max,
            "plateau": plateau,
        },
        "committed": {
            "wheel_delta_theta_rad": result["metrics"].get("wheel_delta_theta_rad"),
            "measured_rolling_radius_m": result["metrics"].get("measured_rolling_radius_m"),
            "measured_rolling_radius_rel_error":
                result["metrics"].get("measured_rolling_radius_rel_error"),
        },
        "gates": result["gates"],
        "thresholds": result["thresholds"],
        "diagnostics": {
            "drive_mode": result["diagnostics"].get("drive_mode"),
            "root_writes": result["diagnostics"].get("root_writes"),
            "friction_coefficient": result["diagnostics"].get("friction_coefficient"),
            "payload_kg": result["diagnostics"].get("payload_kg"),
        },
    }


# ------------------------------------------------- D2: single-factor matrix
def matrix_groups(runs_root):
    """Group the 36 single-factor runs by their declared single knob."""
    names = sorted(os.listdir(runs_root)) if os.path.isdir(runs_root) else []
    groups = []
    for run_id in names:
        if not run_id.startswith(MATRIX_PREFIXES):
            continue
        result_path = os.path.join(runs_root, run_id, "result.json")
        if not os.path.exists(result_path):
            continue
        result = load_json(result_path)
        diag = result["diagnostics"]
        metrics = result["metrics"]
        targets = metrics.get("wheel_targets_rad_s_per_wheel") or [0, 0, 0, 0]
        knob = {
            "friction": diag.get("friction_coefficient"),
            "payload_kg": diag.get("payload_kg"),
            "wheel_target_rad_s": targets[0] if targets else 0.0,
            "profile": result["profile"],
        }
        radii = [v for v in (metrics.get("measured_rolling_radius_m") or {}).values()
                 if v is not None]
        groups.append({
            "run_id": run_id,
            "profile": result["profile"],
            "knob": knob,
            "gate_result": result["gate_result"],
            "root_writes": diag.get("root_writes"),
            "displacement_m": metrics.get("chassis_displacement_m"),
            "yaw_change_rad": metrics.get("yaw_change_rad"),
            "measured_rolling_radius_mean_m": mean(radii) if radii else None,
            "measured_rolling_radius_m": metrics.get("measured_rolling_radius_m"),
            "steps": metrics.get("steps"),
        })
    return groups


def steady_radius(runs_root, run_id, skip=60):
    """Recompute the radius over a steady sub-window, dropping the spin-up.

    The committed |dY|/|dtheta| uses the whole run, which includes the
    0 -> target wheel spin-up of the first ~0.5 s; that transient is a ratio
    artefact, not a radius.  Recomputing after 'skip' frames separates the two.
    """
    path = os.path.join(runs_root, run_id, "raw_poses.jsonl")
    if not os.path.exists(path):
        return None
    rows = load_jsonl(path)
    if len(rows) <= skip + 2:
        return None
    i0, i1 = skip, len(rows) - 1
    dy = rows[i1]["chassis_pos"][1] - rows[i0]["chassis_pos"][1]
    dtheta = mean([rows[i1]["wheel_pos_rad"][w] - rows[i0]["wheel_pos_rad"][w] for w in WHEELS])
    return {
        "window_steps": [i0, i1],
        "delta_y_m": dy,
        "delta_theta_mean_rad": dtheta,
        "r_eff_m": abs(dy) / abs(dtheta) if dtheta else None,
    }


def subwindow_radii(runs_root, run_id, edges):
    """r_eff over consecutive sub-windows - shows where the ratio changes."""
    path = os.path.join(runs_root, run_id, "raw_poses.jsonl")
    if not os.path.exists(path):
        return []
    rows = load_jsonl(path)
    out = []
    for a, b in zip(edges[:-1], edges[1:]):
        if b >= len(rows):
            continue
        dy = rows[b]["chassis_pos"][1] - rows[a]["chassis_pos"][1]
        dtheta = mean([rows[b]["wheel_pos_rad"][w] - rows[a]["wheel_pos_rad"][w] for w in WHEELS])
        out.append({
            "steps": [a, b],
            "delta_y_m": dy,
            "delta_theta_mean_rad": dtheta,
            "r_eff_m": abs(dy) / abs(dtheta) if dtheta else None,
        })
    return out


# ------------------------------------------------- D3: sign/dimension/saturation
def audit_signs(runs_root, r_geom, clamp=50.0):
    findings = []
    names = sorted(os.listdir(runs_root)) if os.path.isdir(runs_root) else []
    for run_id in names:
        if not run_id.startswith("P4"):
            continue
        run = read_run(runs_root, run_id)
        if run["result"] is None or run["rows"] is None or not run["rows"]:
            continue
        rows = run["rows"]
        result = run["result"]
        targets = result["metrics"].get("wheel_targets_rad_s_per_wheel") or [0, 0, 0, 0]
        tail = [mean([r["wheel_omega_rad_s"][w] for r in rows[-20:]]) for w in WHEELS]
        max_abs_omega = max(abs(r["wheel_omega_rad_s"][w]) for r in rows for w in WHEELS)
        root_writes = sum(1 for r in rows if r.get("root_writes", 0) != 0)
        logged_targets = sorted({round(r["wheel_target_rad_s"][w], 6)
                                 for r in rows for w in WHEELS})
        signs = {(1 if t > 1e-9 else -1 if t < -1e-9 else 0) for t in targets}
        entry = {
            "run_id": run_id,
            "profile": result["profile"],
            "commanded_targets": targets,
            "tail_omega": dict(zip(WHEELS, tail)),
            "all_wheels_same_sign": len(signs) == 1,
            "differential": len(signs) > 1,
            "max_abs_omega_rad_s": max_abs_omega,
            "clamp_rad_s": clamp,
            "at_clamp": bool(abs(max_abs_omega - clamp) < 1e-6),
            "over_clamp": bool(max_abs_omega > clamp),
            "root_writes_nonzero_frames": root_writes,
            "logged_target_values": logged_targets,
            # the closed loop rebuilds its target tensor inside the loop but the JSONL
            # keeps writing the profile-level target list, which stays [0,0,0,0]; a
            # reviewer reading only raw_poses would conclude the wheels were never driven
            "logged_target_is_stale_stub": bool(
                len(logged_targets) == 1 and abs(logged_targets[0]) < 1e-12
                and (any(abs(t) > 1e-12 for t in targets)
                     or max_abs_omega > 0.01)),
            "logs_joint_displacement": "wheel_pos_rad" in rows[0],
            "slip_free_projection": None,
            "radius_consistent": None,
        }
        disp = result["metrics"].get("chassis_displacement_m") or [0, 0, 0]
        if (result["profile"] in STRAIGHT_PROFILES and any(abs(t) > 1e-9 for t in targets)
                and "wheel_pos_rad" in rows[0]):
            dtheta = rows[-1]["wheel_pos_rad"]["wheel1"] - rows[0]["wheel_pos_rad"]["wheel1"]
            entry["slip_free_projection"] = r_geom * dtheta
            entry["radius_consistent"] = (
                abs(abs(disp[1]) / abs(dtheta) - r_geom) / r_geom if dtheta else None)
        findings.append(entry)
    return findings


# ------------------------------------------------------- D4: acceptance table
GATE_ORDER = ("all_finite", "no_root_pose_write", "contact_detected",
              "steady_speed_tracks_command", "stops_after_zero_command",
              "speed_limit_respected", "no_motion_without_drive",
              "chassis_displaced_along_command", "wheel_speed_tracks_target",
              "yaw_changed", "measured_rolling_radius_consistent")


def acceptance_table(runs_root, analysed):
    table = []
    for entry in analysed:
        run_id = entry["run_id"]
        run = read_run(runs_root, run_id)
        rows = run["rows"]
        result = run["result"]
        vc = result["metrics"]["velocity_control"]
        speeds = consecutive_speeds(rows)   # speeds[k-1] is the step-k speed
        # The pre-registration says "max speed within 0.5 s after the command reaches
        # zero" -> steps 481..541.  The committed gate instead takes the LAST 0.5 s of
        # the run (steps 541..600).  Both windows are reported so a reviewer can see
        # which convention each number belongs to.
        zero_step = int(round(4.0 / DT))
        first_half = max(abs(v) for v in speeds[zero_step:zero_step + int(0.5 / DT)])
        since_zero = max(abs(v) for v in speeds[-int(0.5 / DT):])
        settle = None
        for k in range(zero_step, len(speeds)):
            if abs(speeds[k]) <= 0.02:
                settle = (k + 1 - zero_step) * DT
                break
        peak = max(abs(v) for v in speeds)
        v_cmd = float(vc["commanded_after_clamp_m_s"])
        table.append({
            "run_id": run_id,
            "commanded_m_s": v_cmd,
            "measured_m_s": entry["speed"]["from_endpoints_m_s"],
            "peak_m_s": peak,
            "since_zero_max_m_s": since_zero,
            "since_zero_first_half_max_m_s": first_half,
            "stop_settle_time_s": settle,
            "committed_since_zero_max_m_s": float(vc["since_zero_max_speed_m_s"]),
            "steady_relative_error": entry["steady_relative_error_recomputed"],
            "prereg_tolerance": 0.15,
            "prereg_pass": entry["steady_relative_error_recomputed"] <= 0.15,
            "stop_tolerance_m_s": 0.02,
            "stop_pass": since_zero <= 0.02,
            "stop_pass_literal_prereg_window": first_half <= 0.02,
            "limit_rhs_m_s": 1.10 * v_cmd + 0.02,
            "limit_pass": peak <= 1.10 * v_cmd + 0.02,
            "odom_error_rel": entry["odometry"]["feedforward_odom_error_rel"],
            "odom_error_mm": entry["odometry"]["feedforward_odom_error_mm"],
            "gates": {g: result["gates"].get(g) for g in GATE_ORDER
                      if g in result["gates"]},
            "committed_gate_result": result["gate_result"],
        })
    return table


P4_REAL_GAPS = [
    {"gap": "loaded rolling radius",
     "why": "0.103 m is the unloaded P1 v2 torus equator; the loaded radius of the real "
            "wheel is not measured", "status": "UNKNOWN"},
    {"gap": "contact tread",
     "why": "the proxy uses a CYLINDER r=0.103, not the crowned toroidal tread, so the "
            "contact patch and pressure distribution are wrong", "status": "PROXY"},
    {"gap": "mass and inertia",
     "why": "16 kg chassis + 1 kg/wheel are TEST_PROXY values; no vendor mass data "
            "(P4-02 only proves the proxy is self-consistent)", "status": "PROXY"},
    {"gap": "friction levels",
     "why": "0.2 / 0.8 / 1.4 are arbitrary gear values on the ground material, not "
            "measured tyre-road coefficients", "status": "PROXY"},
    {"gap": "motor model",
     "why": "joint velocity targets, no torque/current limit, no back-EMF, no battery "
            "sag, no thermal derating", "status": "MISSING"},
    {"gap": "encoder and latency",
     "why": "joint state is read at the physics rate with no encoder quantisation, noise "
            "or communication delay", "status": "MISSING"},
    {"gap": "closed-loop feedback tap",
     "why": "the chassis speed fed back to the trim is identically zero (see D1); a real "
            "loop needs a live measurement", "status": "DEFECT"},
    {"gap": "slope / side-slip / one wheel airborne",
     "why": "only flat straight-line and turn cases were run", "status": "MISSING"},
    {"gap": "emergency stop",
     "why": "the stop test only shows that a zero command decays; there is no E-stop "
            "hardware or safety relay", "status": "MISSING"},
    {"gap": "wheel_target_rad_s logging",
     "why": "the closed-loop runs log a stale 0 target, so the applied wheel command is "
            "not in raw_poses.jsonl", "status": "DEFECT"},
]


# ---------------------------------------------------------------------- report
def render_markdown(report):
    out = []
    lines = out.append
    lines("### D1 - velocity-control recomputation")
    lines("")
    lines("| run | cmd m/s | v consec | v endpoints | v lstsq | ratio | err | r_eff mean | spread |")
    lines("|---|---|---|---|---|---|---|---|---|")
    for e in report["d1"]:
        s = e["speed"]
        lines("| %s | %.4f | %.9f | %.9f | %.9f | %.9f | %.6f | %.6f | %.3f %% |" % (
            e["run_id"], e["commanded_speed_m_s"],
            s["from_consecutive_positions_m_s"], s["from_endpoints_m_s"],
            s["from_least_squares_m_s"], e["achieved_over_commanded"],
            e["steady_relative_error_recomputed"], e["r_eff_mean_m"],
            100.0 * e["r_eff_spread_rel"]))
    lines("")
    lines("| run | per-wheel r_eff (m) | per-wheel r = v/omega (m) |")
    lines("|---|---|---|")
    for e in report["d1"]:
        a = ", ".join("%.6f" % e["per_wheel"][w]["r_eff_m"] for w in WHEELS)
        b = ", ".join("%.6f" % e["per_wheel"][w]["r_from_v_over_omega_m"] for w in WHEELS)
        lines("| %s | %s | %s |" % (e["run_id"], a, b))
    lines("")
    lines("| run | omega_ff | omega mean | trim | limit | f_trim | f_radius | product | ratio |")
    lines("|---|---|---|---|---|---|---|---|---|")
    for e in report["d1"]:
        d = e["decomposition"]
        lines("| %s | %.6f | %.6f | %+.6f | %.6f | %.6f | %.6f | %.6f | %.6f |" % (
            e["run_id"], d["omega_feedforward_rad_s"], d["omega_mean_measured_rad_s"],
            d["trim_added_rad_s"], d["trim_limit_rad_s"], d["factor_trim"],
            d["factor_radius"], d["factor_product"], d["measured_ratio"]))
    lines("")
    lines("| run | plateau window | plateau v | plateau err | plateau r_eff | f_trim | f_radius |")
    lines("|---|---|---|---|---|---|---|")
    for e in report["d1"]:
        p = e["feedback"]["plateau"]
        if p is None:
            lines("| %s | n/a | n/a | n/a | n/a | n/a | n/a |" % e["run_id"])
            continue
        lines("| %s | %d-%d | %.9f | %.6f | %.6f | %.6f | %.6f |" % (
            e["run_id"], p["steps"][0], p["steps"][1], p["speed_m_s"],
            p["steady_relative_error"], p["r_eff_m"], p["factor_trim"], p["factor_radius"]))
    lines("")
    lines("| run | ramp rate measured | expected if vy_meas==0 | ratio | steps to saturate | model RMS |")
    lines("|---|---|---|---|---|---|")
    for e in report["d1"]:
        f = e["feedback"]
        lines("| %s | %.6f | %.6f | %.4f | %.2f | %.5f |" % (
            e["run_id"], f["ramp_rate_measured_rad_s_per_step"],
            f["ramp_rate_expected_rad_s_per_step"], f["ramp_rate_ratio"],
            f["steps_to_saturate_expected"], f["model_rms_rel_error"]))
    lines("")
    lines("### D4 - acceptance table")
    lines("")
    lines("| run | cmd | measured | peak | stop first-0.5s | stop last-0.5s | settle s | err | <=15% | stop | limit | odom err mm | odom err % |")
    lines("|---|---|---|---|---|---|---|---|---|---|---|---|---|")
    for t in report["d4"]["acceptance"]:
        lines("| %s | %.4f | %.9f | %.9f | %.6f | %.6f | %.3f | %.6f | %s | %s | %s | %+.3f | %+.3f %% |" % (
            t["run_id"], t["commanded_m_s"], t["measured_m_s"], t["peak_m_s"],
            t["since_zero_first_half_max_m_s"], t["since_zero_max_m_s"],
            t["stop_settle_time_s"] if t["stop_settle_time_s"] is not None else -1.0,
            t["steady_relative_error"],
            "PASS" if t["prereg_pass"] else "FAIL",
            "PASS" if t["stop_pass"] else "FAIL",
            "PASS" if t["limit_pass"] else "FAIL",
            t["odom_error_mm"], 100.0 * t["odom_error_rel"]))
    lines("")
    return "\n".join(out)


def audit_matrix_knobs(groups):
    """Report, per profile, which knobs vary across the runs of that group."""
    by_profile = {}
    for g in groups:
        by_profile.setdefault(g["profile"], []).append(g)
    out = {}
    for profile, entries in sorted(by_profile.items()):
        varying = {}
        for knob in ("friction", "payload_kg", "wheel_target_rad_s"):
            varying[knob] = sorted({g["knob"][knob] for g in entries})
        out[profile] = {"runs": [g["run_id"] for g in entries], "knobs": varying}
    return out


def main(argv=None):
    parser = argparse.ArgumentParser(description="R10-D P4 deviation analysis")
    parser.add_argument("--runs-root", required=True,
                        help="directory holding the P4* run directories")
    parser.add_argument("--velocity-runs", nargs="+",
                        default=["P4V_030_r13", "P4V_030_r14",
                                 "P4V_050_r13", "P4V_050_r14",
                                 "P4V_clamp_r13", "P4V_clamp_r14"])
    parser.add_argument("--reference-open-loop", default="P4f_base_r11")
    parser.add_argument("--out", default=None, help="write the full JSON report here")
    parser.add_argument("--markdown", default=None, help="write the markdown tables here")
    args = parser.parse_args(argv)

    root = os.path.abspath(args.runs_root)
    problems = []

    probe = read_run(root, args.velocity_runs[0])
    if probe["result"] is None:
        raise SystemExit("missing result.json for %s" % args.velocity_runs[0])
    r_geom = float(probe["result"]["proxy_parameters"]["wheel_radius_m"])
    thresholds = probe["result"]["thresholds"]

    # ---- D1
    analysed = []
    for run_id in args.velocity_runs:
        run = read_run(root, run_id)
        if run["result"] is None or run["rows"] is None:
            problems.append("%s: run directory incomplete" % run_id)
            continue
        entry = analyse_velocity_run(run, r_geom)
        analysed.append(entry)
        s = entry["speed"]
        if abs(s["from_endpoints_m_s"] - s["committed_steady_mean_m_s"]) > REPRO_TOL:
            problems.append("%s: recomputed steady mean %.9f != committed %.9f" % (
                run_id, s["from_endpoints_m_s"], s["committed_steady_mean_m_s"]))
        if abs(entry["steady_relative_error_recomputed"]
               - entry["steady_relative_error_committed"]) > REPRO_TOL:
            problems.append("%s: recomputed relative error %.9f != committed %.9f" % (
                run_id, entry["steady_relative_error_recomputed"],
                entry["steady_relative_error_committed"]))
        if abs(entry["decomposition"]["closure"] - 1.0) > RADIUS_TOL:
            problems.append("%s: decomposition does not close (%.12f)" % (
                run_id, entry["decomposition"]["closure"]))
        if entry["r_eff_spread_rel"] > 5e-3:
            problems.append("%s: per-wheel r_eff spread %.4f %% > 0.5 %%" % (
                run_id, 100.0 * entry["r_eff_spread_rel"]))
        if not all(entry["per_wheel"][w]["sign_consistent"] for w in WHEELS):
            problems.append("%s: a wheel rotates with the travel direction" % run_id)
        if entry["diagnostics"]["root_writes"] not in (0, None):
            problems.append("%s: root_writes=%s" % (run_id, entry["diagnostics"]["root_writes"]))

    # ---- D1 cross-check on the open-loop reference (where the radius came from)
    reference = read_run(root, args.reference_open_loop)
    reference_block = None
    if reference["result"] is not None and reference["rows"] is not None:
        rows = reference["rows"]
        n = len(rows) - 1
        dy = rows[n]["chassis_pos"][1] - rows[0]["chassis_pos"][1]
        dtheta_full = mean([rows[n]["wheel_pos_rad"][w] - rows[0]["wheel_pos_rad"][w]
                            for w in WHEELS])
        steady = steady_radius(root, args.reference_open_loop, skip=60)
        sub = subwindow_radii(root, args.reference_open_loop,
                              [0, 1, 30, 60, 120, 180, 240, 300, 360])
        committed = [v for v in reference["result"]["metrics"]
                     ["measured_rolling_radius_m"].values() if v]
        reference_block = {
            "run_id": args.reference_open_loop,
            "full_run_r_eff_m": abs(dy) / abs(dtheta_full),
            "committed_r_eff_m": mean(committed),
            "steady_window_r_eff_m": steady["r_eff_m"] if steady else None,
            "steady_window_steps": steady["window_steps"] if steady else None,
            "subwindow_radii": sub,
        }
        if abs(reference_block["full_run_r_eff_m"]
               - reference_block["committed_r_eff_m"]) > 1e-3:
            problems.append("%s: full-run r_eff %.6f does not reproduce the committed %.6f" % (
                args.reference_open_loop, reference_block["full_run_r_eff_m"],
                reference_block["committed_r_eff_m"]))

    # ---- D2
    groups = matrix_groups(root)
    radii = [g["measured_rolling_radius_mean_m"] for g in groups
             if g["measured_rolling_radius_mean_m"]]
    radius_span = (max(radii) - min(radii)) if radii else 0.0
    radius_span_rel = radius_span / mean(radii) if radii else 0.0
    measured_deviation = (mean([e["steady_relative_error_recomputed"] for e in analysed])
                          if analysed else 0.0)
    steady_recomputed = {}
    for g in groups:
        if g["profile"] in ("roll_neg_y", "roll_pos_y"):
            steady_recomputed[g["run_id"]] = steady_radius(root, g["run_id"], skip=60)
    steady_values = [v["r_eff_m"] for v in steady_recomputed.values() if v and v["r_eff_m"]]
    steady_span_rel = ((max(steady_values) - min(steady_values)) / mean(steady_values)
                       if steady_values else 0.0)
    d2 = {
        "runs": groups,
        "run_count": len(groups),
        "recorded_radius_min_m": min(radii) if radii else None,
        "recorded_radius_max_m": max(radii) if radii else None,
        "recorded_radius_span_m": radius_span,
        "recorded_radius_span_rel": radius_span_rel,
        "measured_relative_deviation": measured_deviation,
        "explanation_ratio": measured_deviation / radius_span_rel if radius_span_rel else None,
        "steady_window_radius": steady_recomputed,
        "steady_radius_span_rel": steady_span_rel,
        "knob_consistency": audit_matrix_knobs(groups),
    }
    if radius_span_rel >= measured_deviation:
        problems.append("D2: the recorded rolling-radius spread alone could explain the "
                        "deviation (%.4f %% >= %.4f %%)" % (100 * radius_span_rel,
                                                            100 * measured_deviation))

    # ---- D3
    signs = audit_signs(root, r_geom)
    d3 = {
        "findings": signs,
        "clamp_rad_s": 50.0,
        "runs_at_clamp": [f["run_id"] for f in signs if f["at_clamp"]],
        "runs_over_clamp": [f["run_id"] for f in signs if f["over_clamp"]],
        "root_writes_total": sum(f["root_writes_nonzero_frames"] for f in signs),
        "stale_target_logs": [f["run_id"] for f in signs if f["logged_target_is_stale_stub"]],
        "straight_roll_same_sign": {
            f["run_id"]: f["all_wheels_same_sign"] for f in signs
            if f["profile"] in STRAIGHT_PROFILES and any(f["commanded_targets"])},
        "turn_differential": {f["run_id"]: f["differential"] for f in signs
                              if f["profile"] in ("turn", "spin_single")},
    }
    if d3["runs_over_clamp"]:
        problems.append("D3: wheel speed exceeded the +-50 rad/s clamp: %s"
                        % d3["runs_over_clamp"])
    if d3["root_writes_total"]:
        problems.append("D3: %d frames reported a non-zero root write" % d3["root_writes_total"])
    for run_id, same in d3["straight_roll_same_sign"].items():
        if not same:
            problems.append("D3: %s is a straight roll but the four wheels do not share a sign"
                            % run_id)

    # ---- D4
    d4 = {"acceptance": acceptance_table(root, analysed),
          "prereg_tolerance": thresholds.get("speed_tracking_rel"),
          "stop_window_convention": {
              "prereg": "PHASE_P4_09_PREREG.md: max speed within 0.5 s after the command "
                        "reaches zero (steps 481..541)",
              "implementation": "p4_proxy_wheel_drive.py: max over the LAST 0.5 s of the run "
                                "(steps 541..600)",
              "note": "the two windows overlap in conclusion: the ONLY sample above 0.02 m/s "
                      "inside the literal window is the single step-481 boundary sample, "
                      "which still averages the pre-zero motion; the chassis is below "
                      "0.02 m/s after the settle times listed per run in the table",
              "boundary_samples_m_s": {t["run_id"]: t["since_zero_first_half_max_m_s"]
                                       for t in acceptance_table(root, analysed)},
              "settle_time_s": {t["run_id"]: t["stop_settle_time_s"]
                                for t in acceptance_table(root, analysed)},
          },
          "p4_real_gaps": P4_REAL_GAPS}
    for entry in d4["acceptance"]:
        if not entry["stop_pass"]:
            problems.append("D4: %s did not stop within 0.02 m/s" % entry["run_id"])
        if not entry["limit_pass"]:
            problems.append("D4: %s exceeded the speed limit envelope" % entry["run_id"])
        if entry["gates"].get("no_root_pose_write") is False:
            problems.append("D4: %s no_root_pose_write is False" % entry["run_id"])

    report = {
        "analysis": "R10-D P4 dynamics deviation root cause",
        "runs_root": root,
        "r_geometric_m": r_geom,
        "dt_s": DT,
        "d1": analysed,
        "d1_open_loop_reference": reference_block,
        "d2": d2,
        "d3": d3,
        "d4": d4,
        "problems": problems,
        "verdict": "CONSISTENT" if not problems else "VIOLATION",
    }

    markdown = render_markdown(report)
    print(markdown)
    if args.markdown:
        with open(args.markdown, "w", encoding="utf-8") as handle:
            handle.write(markdown)
    if args.out:
        with open(args.out, "w", encoding="utf-8") as handle:
            json.dump(report, handle, ensure_ascii=False, indent=1)
    print(json.dumps({"verdict": report["verdict"], "problems": problems,
                      "run_count_d2": d2["run_count"]}, ensure_ascii=False, indent=1))
    return 0 if not problems else 2


if __name__ == "__main__":
    raise SystemExit(main())
