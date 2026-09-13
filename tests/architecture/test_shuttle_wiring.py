# -*- coding: utf-8 -*-
"""Estimator/shuttle-filter wiring tests (coordinator, after T3 landed)."""
from __future__ import annotations
import sys
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'src'))

from badminton_brain.types import RobotSensorState, ShuttleMeasurement  # noqa: E402

N = 2


class ShuttleFilterWiringTests(unittest.TestCase):
    def test_estimator_uses_the_t3_ukf_when_available(self) -> None:
        from badminton_brain.estimation.estimator import EkfEstimatorModule
        mod = EkfEstimatorModule(num_envs=N)
        self.assertIn('Bridge', mod.shuttle_source, 'the T3 UKF bridge must be injected by default')

    def test_passthrough_is_explicit_when_the_filter_is_disabled(self) -> None:
        from badminton_brain.estimation.estimator import EkfEstimatorModule
        mod = EkfEstimatorModule(num_envs=N, use_ukf=False)
        self.assertIn('TEMP', mod.shuttle_source)

    def test_filtered_output_stays_finite_through_a_short_track(self) -> None:
        from badminton_brain.estimation.estimator import EkfEstimatorModule
        mod = EkfEstimatorModule(num_envs=N)
        for step in range(4):
            t = 0.05 * step
            perception = ShuttleMeasurement(
                position=np.tile(np.array([1.0 - 2.0 * t, 0.2, 1.8 - 0.5 * t]), (N, 1)),
                velocity=np.tile(np.array([-2.0, 0.0, -0.5]), (N, 1)),
                covariance=np.tile(np.eye(3) * 1e-4, (N, 1, 1)), timestamp=t)
            sensors = RobotSensorState(base_pose=np.tile(np.array([-1.6, 0, 0, 0, 0, 0, 1.0]), (N, 1)),
                                       joint_pos=np.zeros((N, 6)), joint_vel=np.zeros((N, 6)),
                                       timestamp=t, odom_twist=np.zeros((N, 3)), imu_yaw_rate=np.zeros((N,)))
            out = mod.process(perception, sensors)
            self.assertTrue(np.all(np.isfinite(out.shuttle_position)))
            self.assertEqual(out.shuttle_position.shape, (N, 3))


if __name__ == '__main__':
    unittest.main()
