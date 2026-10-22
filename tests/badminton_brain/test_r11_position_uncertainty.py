"""A01: reject mixed-state covariance as position variance."""
from pathlib import Path
import sys

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
from badminton_brain.prediction.position_uncertainty import (  # noqa: E402
    CHI2_2D_95, position_uncertainty,
)


def test_large_velocity_uncertainty_does_not_affect_position_sigma():
    state = np.diag([9.0, 4.0, 10**10, 10**8])
    result = position_uncertainty(state)
    assert result["position_sigma_major_px"] == pytest.approx(3.0)
    assert result["position_sigma_minor_px"] == pytest.approx(2.0)
    assert result["ellipse95"]["major_radius_px"] == pytest.approx(7.343240492, rel=1e-5)
    assert result["velocity_covariance_px2_s2"] == [[1e10, 0.0], [0.0, 1e8]]


def test_rotated_ellipse_preserves_xy_covariance():
    a = np.deg2rad(30)
    rot = np.array([[np.cos(a), -np.sin(a)], [np.sin(a), np.cos(a)]])
    P = np.eye(4)
    P[:2, :2] = rot @ np.diag([25.0, 100.0]) @ rot.T
    r = position_uncertainty(P)
    assert r["position_sigma_major_px"] == pytest.approx(10.0)
    assert r["position_sigma_minor_px"] == pytest.approx(5.0)
    assert r["ellipse95"]["major_radius_px"] == pytest.approx(10 * np.sqrt(CHI2_2D_95))
    assert abs(r["ellipse95"]["angle_deg_image_xy"] % 180 - 120) < 1e-10


@pytest.mark.parametrize("value", [np.full((4, 4), np.nan), np.eye(3),
                                    np.diag([-1, 1, 1, 1])])
def test_invalid_covariance_fails_closed(value):
    with pytest.raises(ValueError):
        position_uncertainty(value)
