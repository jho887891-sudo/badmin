# -*- coding: utf-8 -*-
"""Regression tests for invalid rows in the existing shuttle UKF integration.

These tests use only deterministic synthetic measurements. They do not load a
video, train YOLO, start Isaac Sim, or command robot motion.
"""
from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from badminton_brain.estimation.estimator import EkfEstimatorModule  # noqa: E402
from badminton_brain.estimation.shuttle_ukf import ShuttleUKF  # noqa: E402
from badminton_brain.types import RobotSensorState, ShuttleMeasurement  # noqa: E402


def _measurement(position, timestamp, valid_mask):
    position = np.asarray(position, dtype=float)
    velocity = np.zeros_like(position)
    covariance = np.tile(np.eye(3) * 1e-4, (position.shape[0], 1, 1))
    return ShuttleMeasurement(
        position=position,
        velocity=velocity,
        covariance=covariance,
        timestamp=float(timestamp),
        valid_mask=np.asarray(valid_mask, dtype=float),
    )


def _robot_sensors(timestamp):
    return RobotSensorState(
        base_pose=np.tile(np.array([-1.6, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0]), (1, 1)),
        joint_pos=np.zeros((1, 6)),
        joint_vel=np.zeros((1, 6)),
        timestamp=float(timestamp),
        odom_twist=np.zeros((1, 3)),
        imu_yaw_rate=np.zeros((1,)),
    )


class UKFValidityIntegrationTests(unittest.TestCase):
    def test_ukf_does_not_fuse_rows_marked_invalid(self):
        ukf = ShuttleUKF(num_envs=2, fuse_velocity=False)
        ukf.initialize(
            position=np.array([[1.0, 0.0, 1.0], [1.0, 0.0, 1.0]]),
            velocity=np.zeros((2, 3)),
            drag_k=0.1,
            covariance=np.eye(7) * 0.02,
            timestamp=0.0,
        )
        before = ukf.state.copy()
        before_covariance = ukf.covariance.copy()

        ukf.update(_measurement(
            [[1.1, 0.0, 1.0], [100.0, 100.0, 100.0]],
            timestamp=0.1,
            valid_mask=[1.0, 0.0],
        ))

        # Env 0 has a valid observation and should update. Env 1 has no
        # measurement; UKF.update alone must not pull it toward the bogus row.
        self.assertFalse(np.allclose(ukf.state[0], before[0]))
        np.testing.assert_allclose(ukf.state[1], before[1], rtol=0.0, atol=1e-12)
        np.testing.assert_allclose(
            ukf.covariance[1], before_covariance[1], rtol=0.0, atol=1e-12
        )

    def test_all_invalid_frame_returns_ukf_prediction_not_zero_measurement(self):
        estimator = EkfEstimatorModule(num_envs=1, use_ukf=True)
        first = estimator.process(
            _measurement([[1.0, 0.0, 1.0]], timestamp=0.0, valid_mask=[1.0]),
            _robot_sensors(0.0),
        )
        self.assertGreater(float(np.linalg.norm(first.shuttle_position[0])), 0.1)

        dropout = estimator.process(
            _measurement([[0.0, 0.0, 0.0]], timestamp=0.1, valid_mask=[0.0]),
            _robot_sensors(0.1),
        )

        self.assertTrue(np.isfinite(dropout.shuttle_position).all())
        self.assertGreater(
            float(np.linalg.norm(dropout.shuttle_position[0])),
            0.1,
            "an invalid zero-filled measurement must not replace the UKF state",
        )

    def test_estimator_accepts_bare_ukf_without_argument_collision(self):
        ukf = ShuttleUKF(num_envs=1, fuse_velocity=False)
        ukf.initialize(
            position=np.array([[1.0, 0.0, 1.0]]),
            velocity=np.zeros((1, 3)),
            drag_k=0.1,
            covariance=np.eye(7) * 0.02,
            timestamp=0.0,
        )
        estimator = EkfEstimatorModule(num_envs=1, shuttle_filter=ukf, use_ukf=False)
        estimator.process(
            _measurement([[1.0, 0.0, 1.0]], timestamp=0.0, valid_mask=[1.0]),
            _robot_sensors(0.0),
        )
        dropout = estimator.process(
            _measurement([[0.0, 0.0, 0.0]], timestamp=0.1, valid_mask=[0.0]),
            _robot_sensors(0.1),
        )
        self.assertTrue(np.isfinite(dropout.shuttle_position).all())
        self.assertGreater(
            float(np.linalg.norm(dropout.shuttle_position[0])),
            0.1,
            "bare UKF injection must predict across an invalid observation",
        )


if __name__ == "__main__":
    unittest.main()
