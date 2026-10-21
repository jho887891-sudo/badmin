"""Unit tests for the pre-registered P2 landing/load-bearing gates (pure CPU)."""

from __future__ import annotations

import math
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..",
                                                "scripts", "simulation")))

from p2_landing_gates import compute_landing_gates  # noqa: E402

WHEEL_BOTTOM = [[0.0, 0.0, 0.0, 0.0]] * 120


def make_frames(count=120, force=196.2, z=0.334, speed=0.0, tilt=0.0, penetration=0.0):
    frames = []
    for step in range(count):
        frames.append({
            "step": step,
            "pos": [0.0, 0.0, z],
            "lin_vel": [speed, 0.0, 0.0],
            "tilt_rad": tilt,
            "finite": True,
            "max_contact_force_n": force,
        })
    return frames


def bottoms(count=120, value=None):
    return [[value if value is not None else 0.0] * 4 for _ in range(count)]


def test_settled_weight_supported_run_passes():
    gates, metrics = compute_landing_gates(make_frames(), bottoms(), gravity_z=-9.81,
                                           mass_kg=20.0)
    assert gates["contact_detected"]
    assert gates["settles"]
    assert gates["no_tip_over"]
    assert gates["no_long_term_penetration"]
    assert gates["static_weight_supported"]
    assert abs(metrics["tail_mean_contact_force_n"] - 196.2) < 1e-6


def test_peak_force_alone_is_not_support():
    # A single huge impulse then a long silent tail must NOT be called supported.
    frames = make_frames(force=0.0)
    frames[10]["max_contact_force_n"] = 2693.5
    gates, metrics = compute_landing_gates(frames, bottoms(), gravity_z=-9.81, mass_kg=20.0)
    assert gates["contact_detected"]
    assert not gates["static_weight_supported"]
    assert metrics["peak_contact_force_n"] == 2693.5


def test_freefall_without_contact_fails():
    frames = make_frames(force=0.0, z=0.40, speed=0.5)
    frames[0]["max_contact_force_n"] = None
    gates, _ = compute_landing_gates(frames, bottoms(), gravity_z=-9.81, mass_kg=20.0)
    assert not gates["contact_detected"]


def test_no_contact_sensor_readings_keeps_gates_boolean_false_not_pass():
    frames = make_frames(force=None)
    gates, metrics = compute_landing_gates(frames, bottoms(), gravity_z=-9.81, mass_kg=20.0)
    assert gates["contact_detected"] is False
    assert metrics["peak_contact_force_n"] is None


def test_tip_over_detected():
    gates, _ = compute_landing_gates(make_frames(tilt=0.2), bottoms(), gravity_z=-9.81)
    assert not gates["no_tip_over"]


def test_rising_body_does_not_settle():
    frames = make_frames(speed=0.05)
    gates, _ = compute_landing_gates(frames, bottoms(), gravity_z=-9.81)
    assert not gates["settles"]


def test_penetration_detected():
    gates, _ = compute_landing_gates(make_frames(), bottoms(value=-0.01), gravity_z=-9.81)
    assert not gates["no_long_term_penetration"]


def test_weight_gate_absent_without_mass():
    gates, metrics = compute_landing_gates(make_frames(), bottoms(), gravity_z=-9.81)
    assert "static_weight_supported" not in gates
    assert "expected_static_weight_n" not in metrics


def test_nan_frame_breaks_all_finite():
    frames = make_frames()
    frames[5]["finite"] = False
    gates, _ = compute_landing_gates(frames, bottoms(), gravity_z=-9.81)
    assert not gates["all_finite"]
