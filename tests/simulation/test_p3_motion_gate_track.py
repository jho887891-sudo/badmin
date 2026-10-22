"""Regression tests for the P3 v1.1 directed motion gate (DeepSeek extension).

ChatGPT PR #31 compared only the first and last frame; these cases pin the per-frame
track semantics and the negative-control wiring that the GPU run uses.
"""

from __future__ import annotations

import os
import sys

import pytest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..",
                                                "scripts", "simulation")))

from p3_motion_gate import directed_displacement_track  # noqa: E402

RATIO = 0.9


def track(cmds, positions):
    return directed_displacement_track(cmds, positions, RATIO)


def test_z_motion_of_full_amplitude_passes():
    positions = [[0.0, 0.0, 0.40], [0.0, 0.0, 0.45], [0.0, 0.0, 0.50]]
    result = track([0.0, 0.0, 0.10], positions)
    assert result["pass"] is True
    assert result["achieved_along_command_max_m"] == pytest.approx(0.10, abs=1e-12)
    assert result["achieved_ratio"] == pytest.approx(1.0, abs=1e-12)


def test_z_command_with_no_z_motion_fails_even_if_x_moves():
    # this is the fail-open PR #31 found: hypot(x, y) never looked at Z
    positions = [[0.0, 0.0, 0.40], [0.10, 0.0, 0.40], [0.20, 0.0, 0.40]]
    result = track([0.0, 0.0, 0.10], positions)
    assert result["pass"] is False
    assert result["achieved_along_command_max_m"] == pytest.approx(0.0, abs=1e-12)


def test_reversed_direction_fails():
    positions = [[0.0, 0.0, 0.40], [0.0, 0.0, 0.35], [0.0, 0.0, 0.30]]
    result = track([0.0, 0.0, 0.10], positions)
    assert result["pass"] is False
    # max projection over frames is the START frame (0.0), so a purely backwards profile
    # cannot pass; the final projection is the negative one and is reported separately.
    assert result["achieved_ratio"] == pytest.approx(0.0, abs=1e-12)
    assert result["final_ratio"] < 0.0


def test_wrong_axis_same_magnitude_fails():
    positions = [[0.0, 0.0, 0.40], [0.10, 0.0, 0.40]]
    result = track([0.0, 0.10, 0.0], positions)
    assert result["pass"] is False


def test_transient_motion_that_returns_still_counts():
    # per-frame track: reaches the target mid-run, then returns to the start
    positions = [[0.0, 0.0, 0.40], [0.0, 0.0, 0.50], [0.0, 0.0, 0.40]]
    result = track([0.0, 0.0, 0.10], positions)
    assert result["pass"] is True
    assert result["best_frame"] == 1
    assert result["final_ratio"] == pytest.approx(0.0, abs=1e-12)


def test_partial_motion_below_ratio_fails():
    positions = [[0.0, 0.0, 0.0], [0.0, 0.0, 0.089]]
    result = track([0.0, 0.0, 0.10], positions)
    assert result["pass"] is False


def test_zero_command_is_not_applicable():
    assert track([0.0, 0.0, 0.0], [[0.0, 0.0, 0.0]]) is None


def test_empty_positions_raise():
    with pytest.raises(ValueError):
        track([0.0, 0.0, 0.1], [])


def test_nonfinite_position_raises():
    with pytest.raises(ValueError):
        track([0.0, 0.0, 0.1], [[0.0, 0.0, 0.0], [0.0, float("nan"), 0.1]])


def test_bad_ratio_raises():
    with pytest.raises(ValueError):
        directed_displacement_track([0.0, 0.0, 0.1], [[0.0, 0.0, 0.0]], 0.0)


def test_diagonal_command_projects_each_axis():
    positions = [[0.0, 0.0, 0.0], [0.2, -0.15, 0.05]]
    result = track([0.2, -0.15, 0.05], positions)
    assert result["pass"] is True
    assert result["achieved_ratio"] == pytest.approx(1.0, abs=1e-12)
    assert result["cross_track_m"] == pytest.approx(0.0, abs=1e-12)

def test_numpy_positions_are_accepted():
    numpy = pytest.importorskip("numpy")
    positions = numpy.array([[0.0, 0.0, 0.40], [0.0, 0.0, 0.50]])
    result = track([0.0, 0.0, 0.10], positions)
    assert result["pass"] is True
    assert result["achieved_ratio"] == pytest.approx(1.0, abs=1e-12)


def test_numpy_positions_with_no_motion_fail():
    numpy = pytest.importorskip("numpy")
    positions = numpy.array([[0.0, 0.0, 0.40], [0.10, 0.0, 0.40]])
    assert track([0.0, 0.0, 0.10], positions)["pass"] is False
