# -*- coding: utf-8 -*-
"""The safety shield's per-joint proxy must track the official B601-RS limit table.

safety_shield.py carries its own numeric arrays so the brain layer stays importable without the
simulation stack. This test is what keeps the copy honest: if the RS table changes, this fails.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "simulation"))

from src.badminton_brain.safety.safety_shield import SafetyLimits  # noqa: E402
from src.simulation.rebot_b601_rs.joint_limits import JOINT_LIMITS, ARM_JOINTS  # noqa: E402


def test_safety_proxy_limits_equal_the_official_rs_ranges():
    limits = SafetyLimits.temp_proxy()
    lo = np.asarray(limits.joint_position_min.value, dtype=float)
    hi = np.asarray(limits.joint_position_max.value, dtype=float)
    expected_lo = np.array([JOINT_LIMITS[n].angle.lower for n in ARM_JOINTS])
    expected_hi = np.array([JOINT_LIMITS[n].angle.upper for n in ARM_JOINTS])
    np.testing.assert_allclose(lo, expected_lo, atol=1e-12)
    np.testing.assert_allclose(hi, expected_hi, atol=1e-12)


def test_safety_proxy_is_asymmetric_where_the_arm_is():
    """joint2/joint3 are [0, 3.14] on this arm: a symmetric proxy would allow an invalid command."""
    limits = SafetyLimits.temp_proxy()
    lo = np.asarray(limits.joint_position_min.value, dtype=float)
    hi = np.asarray(limits.joint_position_max.value, dtype=float)
    assert list(lo) != list(-hi), "the B601-RS range is not symmetric"
    assert lo[1] == 0.0 and lo[2] == 0.0


def test_velocity_proxy_is_not_the_suspicious_urdf_number():
    limits = SafetyLimits.temp_proxy()
    v = np.asarray(limits.joint_velocity_max.value, dtype=float)
    assert np.all(v < 50.0), "the URDF 50/40 values are not verified motor speeds"
    assert np.all(v > 0.0)
