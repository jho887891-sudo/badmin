# -*- coding: utf-8 -*-
"""Coordinator ruling: RobotSensorState gains optional odometry/IMU channels and the
estimation layer gains an EstimationModule adapter (T2 module was otherwise unpluggable)."""
from __future__ import annotations
import sys
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'src'))

from badminton_brain.interfaces import EstimationModule  # noqa: E402
from badminton_brain.types import (  # noqa: E402
    BrainBoundaryError, Layer, RobotSensorState, ShuttleMeasurement, UnifiedState,
)

N = 3


class SensorChannelTests(unittest.TestCase):
    def test_optional_odom_and_imu_channels_exist_and_validate(self) -> None:
        s = RobotSensorState(base_pose=np.zeros((N, 7)), joint_pos=np.zeros((N, 6)),
                             joint_vel=np.zeros((N, 6)), timestamp=0.0,
                             odom_twist=np.zeros((N, 3)), imu_yaw_rate=np.zeros((N,)))
        self.assertEqual(s.odom_twist.shape, (N, 3))
        self.assertEqual(s.imu_yaw_rate.shape, (N,))

    def test_channels_default_to_none_so_existing_callers_keep_working(self) -> None:
        s = RobotSensorState(base_pose=np.zeros((N, 7)), joint_pos=np.zeros((N, 6)),
                             joint_vel=np.zeros((N, 6)), timestamp=0.0)
        self.assertIsNone(s.odom_twist)
        self.assertIsNone(s.imu_yaw_rate)

    def test_wrong_channel_shapes_are_rejected(self) -> None:
        with self.assertRaises(BrainBoundaryError):
            RobotSensorState(base_pose=np.zeros((N, 7)), joint_pos=np.zeros((N, 6)),
                             joint_vel=np.zeros((N, 6)), timestamp=0.0,
                             odom_twist=np.zeros((N, 2)))
        with self.assertRaises(BrainBoundaryError):
            RobotSensorState(base_pose=np.zeros((N, 7)), joint_pos=np.zeros((N, 6)),
                             joint_vel=np.zeros((N, 6)), timestamp=0.0,
                             imu_yaw_rate=np.zeros((N, 1)))


class EstimationAdapterTests(unittest.TestCase):
    def test_adapter_is_an_estimation_module(self) -> None:
        from badminton_brain.estimation.estimator import EkfEstimatorModule
        mod = EkfEstimatorModule(num_envs=N)
        self.assertIsInstance(mod, EstimationModule)
        self.assertIs(mod.layer, Layer.ESTIMATION)

    def test_adapter_produces_unified_state(self) -> None:
        from badminton_brain.estimation.estimator import EkfEstimatorModule
        mod = EkfEstimatorModule(num_envs=N)
        sensors = RobotSensorState(base_pose=np.tile(np.array([-1.6, 0, 0, 0, 0, 0, 1.0]), (N, 1)),
                                   joint_pos=np.zeros((N, 6)), joint_vel=np.zeros((N, 6)), timestamp=0.0,
                                   odom_twist=np.zeros((N, 3)), imu_yaw_rate=np.zeros((N,)))
        perception = ShuttleMeasurement(position=np.tile(np.array([1.0, 0.0, 1.8]), (N, 1)),
                                        velocity=np.tile(np.array([-3.0, 0.0, 1.5]), (N, 1)),
                                        covariance=np.tile(np.eye(3) * 1e-4, (N, 1, 1)), timestamp=0.0)
        out = mod.process(perception, sensors)
        self.assertIsInstance(out, UnifiedState)
        self.assertEqual(out.shuttle_position.shape, (N, 3))
        self.assertEqual(out.base_pose.shape, (N, 7))

    def test_adapter_reset_only_touches_selected_envs(self) -> None:
        from badminton_brain.estimation.estimator import EkfEstimatorModule
        mod = EkfEstimatorModule(num_envs=N)
        before = mod.state_snapshot()
        mod.reset([1])
        after = mod.state_snapshot()
        self.assertFalse(np.array_equal(before[1], after[1]))
        self.assertTrue(np.array_equal(before[0], after[0]))
        self.assertTrue(np.array_equal(before[2], after[2]))


if __name__ == '__main__':
    unittest.main()
