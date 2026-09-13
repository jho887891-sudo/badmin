#!/usr/bin/env python3
from __future__ import annotations

import math
import unittest

import numpy as np

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src" / "trajectory"))

from shuttle_aerodynamics import (
    acceleration,
    k_from_aerodynamic_length,
    rk4_step,
    rollout,
    terminal_speed,
)


class ShuttleAerodynamicsTests(unittest.TestCase):
    def test_default_literature_aerodynamic_length_maps_to_k(self) -> None:
        self.assertAlmostEqual(k_from_aerodynamic_length(6.5), 1.0 / 6.5, places=15)

    def test_zero_relative_air_velocity_has_gravity_only(self) -> None:
        g = np.array([0.0, 0.0, -9.80665])
        wind = np.array([1.0, -2.0, 0.5])
        a = acceleration(np.zeros(3), wind.copy(), k_per_m=1.0 / 6.5, gravity=g, wind=wind)
        np.testing.assert_allclose(a, g, rtol=0.0, atol=1e-14)

    def test_drag_always_opposes_relative_velocity(self) -> None:
        g = np.zeros(3)
        wind = np.array([0.4, -0.2, 0.1])
        v = np.array([3.0, -4.0, 2.0])
        rel = v - wind
        a = acceleration(np.zeros(3), v, k_per_m=1.0 / 6.5, gravity=g, wind=wind)
        self.assertLess(float(np.dot(a, rel)), 0.0)

    def test_terminal_speed_makes_vertical_acceleration_nearly_zero(self) -> None:
        k = 1.0 / 6.5
        g = 9.80665
        vt = terminal_speed(k, g)
        a = acceleration(
            np.zeros(3),
            np.array([0.0, 0.0, -vt]),
            k_per_m=k,
            gravity=np.array([0.0, 0.0, -g]),
            wind=np.zeros(3),
        )
        self.assertAlmostEqual(a[2], 0.0, places=12)

    def test_gravity_only_rk4_matches_closed_form(self) -> None:
        p0 = np.array([1.2, 0.0, 1.8])
        v0 = np.array([-3.0, 0.0, 1.5])
        dt = 0.4
        g = np.array([0.0, 0.0, -9.80665])
        p1, v1 = rk4_step(p0, v0, dt, k_per_m=0.0, gravity=g, wind=np.zeros(3))
        expected_p = p0 + v0 * dt + 0.5 * g * dt * dt
        expected_v = v0 + g * dt
        np.testing.assert_allclose(p1, expected_p, rtol=0.0, atol=1e-12)
        np.testing.assert_allclose(v1, expected_v, rtol=0.0, atol=1e-12)

    def test_quadratic_drag_reduces_high_horizontal_speed(self) -> None:
        p0 = np.zeros(3)
        v0 = np.array([15.0, 0.0, 0.0])
        result = rollout(
            p0,
            v0,
            duration_s=0.5,
            dt_s=0.001,
            k_per_m=1.0 / 6.5,
            gravity=np.zeros(3),
            wind=np.zeros(3),
        )
        self.assertLess(np.linalg.norm(result["velocity"][-1]), np.linalg.norm(v0))
        self.assertGreater(result["position"][-1, 0], 0.0)

    def test_rollout_outputs_are_finite(self) -> None:
        result = rollout(
            np.array([1.2, 0.0, 1.8]),
            np.array([-3.0, 0.2, 1.5]),
            duration_s=1.0,
            dt_s=0.002,
            k_per_m=1.0 / 6.5,
            gravity=np.array([0.0, 0.0, -9.80665]),
            wind=np.zeros(3),
        )
        self.assertTrue(np.isfinite(result["time"]).all())
        self.assertTrue(np.isfinite(result["position"]).all())
        self.assertTrue(np.isfinite(result["velocity"]).all())


if __name__ == "__main__":
    unittest.main()
