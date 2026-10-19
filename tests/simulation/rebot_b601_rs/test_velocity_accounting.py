# -*- coding: utf-8 -*-
"""No Isaac Sim needed: regression tests for velocity/probe time-window accounting."""
from __future__ import annotations

import math
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "simulation"))

from badminton_scene.velocity_accounting import summarize_velocity_window


def test_first_step_is_included_in_both_integral_and_displacement():
    # Old implementation silently omitted v(step=1) and started at q(step=1).
    r = summarize_velocity_window(
        q0=0.0, positions=[0.2, 0.3], velocities=[2.0, 1.0], dt=0.1)
    assert r["q_start"] == 0.0
    assert r["q_end"] == pytest.approx(0.3)
    assert r["integrated_physx_velocity"] == pytest.approx(0.3)
    assert r["measured_delta"] == pytest.approx(0.3)
    assert r["integral_minus_position"] == pytest.approx(0.0, abs=1e-12)
    assert r["accounting"]["net_matches_segment_sum"]
    assert r["accounting"]["integral_matches_segment_sum"]


def test_outbound_and_return_windows_tile_exactly_without_overlap():
    # switch at physics step 4: outbound steps 1..3, return steps 4..5.
    r = summarize_velocity_window(
        q0=0.3, positions=[0.35, 0.4, 0.4, 0.35, 0.3],
        velocities=[0.5, 0.5, 0.0, -0.5, -0.5],
        dt=0.1, switch_step=4)
    out, ret = r["segments"]
    assert (out["start_step"], out["end_step"]) == (0, 3)
    assert (ret["start_step"], ret["end_step"]) == (3, 5)
    assert out["actual_delta_rad"] == pytest.approx(0.1)
    assert ret["actual_delta_rad"] == pytest.approx(-0.1)
    assert out["integrated_physx_velocity_rad"] == pytest.approx(0.1)
    assert ret["integrated_physx_velocity_rad"] == pytest.approx(-0.1)
    assert r["measured_delta"] == pytest.approx(0.0, abs=1e-12)
    assert r["integrated_physx_velocity"] == pytest.approx(0.0, abs=1e-12)


def test_gravity_ghost_velocity_not_accepted_as_correct():
    # A bad physics configuration must still produce a measurable discrepancy.
    r = summarize_velocity_window(
        q0=0.286, positions=[0.286, 0.286, 0.286],
        velocities=[0.0675743, 0.0675743, 0.0675743], dt=1 / 240)
    assert r["measured_delta"] == 0.0
    assert r["integral_minus_position"] > 0.0008


@pytest.mark.parametrize("data", [
    (0.0, [0.1], [0.1], 0.1, None),
    (0.0, [0.1, 0.2], [0.1], 0.1, None),
    (0.0, [0.1, 0.2], [0.1, 0.2], 0.0, None),
    (0.0, [0.1, float("nan")], [0.1, 0.2], 0.1, None),
    (0.0, [0.1, 0.2], [0.1, 0.2], 0.1, 1),
])
def test_rejects_invalid_windows(data):
    q0, positions, velocities, dt, switch = data
    with pytest.raises(ValueError):
        summarize_velocity_window(q0, positions, velocities, dt, switch_step=switch)
