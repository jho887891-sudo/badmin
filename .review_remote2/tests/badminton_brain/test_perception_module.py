# -*- coding: utf-8 -*-
"""Perception adapter tests (coordinator T11 wiring)."""
from __future__ import annotations
import sys
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'src'))

from badminton_brain.interfaces import PerceptionModule  # noqa: E402
from badminton_brain.perception.perception_module import StereoPerceptionModule  # noqa: E402
from badminton_brain.status import AssetStatus  # noqa: E402
from badminton_brain.types import (  # noqa: E402
    BrainBoundaryError, Layer, RobotSensorState, ShuttleMeasurement,
)

N = 2


def sensors(t: float = 0.0) -> RobotSensorState:
    return RobotSensorState(base_pose=np.tile(np.array([-1.6, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0]), (N, 1)),
                            joint_pos=np.zeros((N, 6)), joint_vel=np.zeros((N, 6)), timestamp=t)


def truth_provider(n: int, t: float) -> np.ndarray:
    base = np.array([1.0, 0.2, 1.8])
    return np.stack([base + np.array([-2.0 * t, 0.0, -0.5 * t]) for _ in range(n)])


class PerceptionModuleTests(unittest.TestCase):
    def test_is_a_perception_module(self) -> None:
        mod = StereoPerceptionModule(num_envs=N, truth_provider=truth_provider)
        self.assertIsInstance(mod, PerceptionModule)
        self.assertIs(mod.layer, Layer.PERCEPTION)

    def test_without_any_source_it_refuses_to_pretend(self) -> None:
        mod = StereoPerceptionModule(num_envs=N)
        with self.assertRaises(BrainBoundaryError):
            mod.process(sensors())

    def test_simulated_path_returns_triangulated_measurement_close_to_truth(self) -> None:
        mod = StereoPerceptionModule(num_envs=N, truth_provider=truth_provider)
        out = mod.process(sensors(0.0))
        self.assertIsInstance(out, ShuttleMeasurement)
        self.assertEqual(out.position.shape, (N, 3))
        error = np.linalg.norm(out.position - truth_provider(N, 0.0), axis=1)
        self.assertTrue(np.all(error < 0.05), 'triangulation error too large: %s' % (error,))
        self.assertTrue(np.all(np.isfinite(out.velocity)))

    def test_ground_truth_is_never_passed_through(self) -> None:
        mod = StereoPerceptionModule(num_envs=N, truth_provider=truth_provider)
        out = mod.process(sensors(0.0))
        exact = np.allclose(out.position, truth_provider(N, 0.0), atol=0.0, rtol=0.0)
        self.assertFalse(exact, 'measurement must come from noisy correspondences, not from truth')

    def test_velocity_uses_the_previous_frame(self) -> None:
        mod = StereoPerceptionModule(num_envs=N, truth_provider=truth_provider)
        mod.process(sensors(0.0))
        out = mod.process(sensors(0.1))
        self.assertTrue(np.all(np.isfinite(out.velocity)))
        self.assertGreater(float(np.abs(out.velocity[0, 0])), 0.0)

    def test_readiness_reports_uncalibrated_hardware(self) -> None:
        mod = StereoPerceptionModule(num_envs=N, truth_provider=truth_provider)
        report = mod.readiness()
        self.assertEqual(report['camera_calibration'], AssetStatus.REQUIRES_CALIBRATION.value)
        self.assertTrue(report['simulated_path'])
        self.assertFalse(report['real_path'])

    def test_measure_fn_path_is_used_when_injected(self) -> None:
        calls = []

        def measure_fn(t):
            calls.append(t)
            return np.zeros((N, 3))

        mod = StereoPerceptionModule(num_envs=N, measure_fn=measure_fn)
        out = mod.process(sensors(0.25))
        self.assertEqual(calls, [0.25])
        np.testing.assert_allclose(out.position, np.zeros((N, 3)))


if __name__ == '__main__':
    unittest.main()
