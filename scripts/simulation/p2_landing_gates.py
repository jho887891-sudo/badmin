"""Pre-registered landing / load-bearing gates for P2 gravity runs.

Separate from the ORIGINAL P2 static gates (t0 world pose + 60-step drift).  A free fall
cannot satisfy a "world pose must not change" gate, so the landing run gets its own
booleans.  Thresholds are frozen in
management/morph_one_rs/PHASE_P2_WC_G6_PREREGISTRATION.md (G6) and
management/morph_one_rs/PHASE_P2_G5_LANDING_GATE.md (G5 landing) and are never tuned to
the observed data.

Pure functions over already-recorded per-step telemetry: no Isaac, no GPU, unit testable.
"""

from __future__ import annotations

import math

TAIL_STEPS = 30              # 0.25 s at dt=1/120
CONTACT_FORCE_THRESHOLD_N = 1.0
MAX_TILT_RAD = 0.05
TAIL_SPEED_MAX_M_S = 0.02
TAIL_Z_SPAN_MAX_M = 0.002
TAIL_PENETRATION_MAX_M = 0.002
WEIGHT_RELATIVE_TOL = 0.05   # |mean tail contact - m*g| / (m*g) <= 5 %


def wheel_bottom_from_chassis(chassis_pos, rotation_row_z, wheel_local_centres, half_height):
    """World z of every wheel bottom: chassis z + (third row of R) . p_local - half_height."""
    bottoms = []
    for frame_pos in chassis_pos:
        bottoms.append([frame_pos[2] + float(rotation_row_z[0] * centre[0]
                                            + rotation_row_z[1] * centre[1]
                                            + rotation_row_z[2] * centre[2]) - half_height
                        for centre in wheel_local_centres])
    return bottoms


def compute_landing_gates(frames, wheel_bottom, tail_steps=TAIL_STEPS,
                          gravity_z=-9.81, mass_kg=None):
    """Frames are the recorded per-step dicts (step, pos, lin_vel, tilt_rad, force)."""
    forces = [frame.get("max_contact_force_n") for frame in frames]
    known = [value for value in forces if value is not None]
    peak = max(known) if known else None
    first_contact = next((frame["step"] for frame in frames
                          if frame.get("max_contact_force_n") is not None
                          and frame["max_contact_force_n"] > CONTACT_FORCE_THRESHOLD_N), None)
    tail = frames[-min(tail_steps, len(frames)):]
    tail_bottom = wheel_bottom[-min(tail_steps, len(wheel_bottom)):]
    tail_speed = max(float(math.sqrt(sum(v * v for v in frame["lin_vel"]))) for frame in tail)
    tail_z_span = max(frame["pos"][2] for frame in tail) - min(frame["pos"][2] for frame in tail)
    tail_penetration = max(0.0, -min(min(row) for row in tail_bottom))
    tilts = [frame["tilt_rad"] for frame in frames if frame.get("tilt_rad") is not None]
    max_tilt = max(tilts) if tilts else None
    tail_forces = [frame.get("max_contact_force_n") for frame in tail
                   if frame.get("max_contact_force_n") is not None]
    mean_tail_force = (sum(tail_forces) / len(tail_forces)) if tail_forces else None

    gates = {
        "all_finite": all(frame.get("finite", True) for frame in frames),
        "contact_detected": bool(peak is not None and peak > CONTACT_FORCE_THRESHOLD_N),
        "settles": bool(tail_speed < TAIL_SPEED_MAX_M_S and tail_z_span < TAIL_Z_SPAN_MAX_M),
        "no_long_term_penetration": bool(tail_penetration < TAIL_PENETRATION_MAX_M),
    }
    if max_tilt is not None:
        gates["no_tip_over"] = bool(max_tilt < MAX_TILT_RAD)
    metrics = {
        "peak_contact_force_n": peak,
        "first_contact_step": first_contact,
        "tail_mean_contact_force_n": (round(mean_tail_force, 6)
                                      if mean_tail_force is not None else None),
        "tail_speed_max_m_s": round(tail_speed, 9),
        "tail_z_span_m": round(tail_z_span, 9),
        "tail_max_penetration_m": round(tail_penetration, 9),
        "max_tilt_rad": (round(max_tilt, 9) if max_tilt is not None else None),
        "final_chassis_z_m": round(frames[-1]["pos"][2], 9),
    }
    if mass_kg is not None and gravity_z is not None:
        expected_weight = mass_kg * abs(gravity_z)
        metrics["expected_static_weight_n"] = round(expected_weight, 6)
        if mean_tail_force is not None and expected_weight > 0.0:
            relative = abs(mean_tail_force - expected_weight) / expected_weight
            metrics["tail_weight_relative_error"] = round(relative, 9)
            metrics["thresholds_weight_relative"] = WEIGHT_RELATIVE_TOL
            gates["static_weight_supported"] = bool(relative <= WEIGHT_RELATIVE_TOL)
    return gates, metrics
