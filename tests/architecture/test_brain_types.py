# -*- coding: utf-8 -*-
"""Robot Brain message contracts (docs/architecture/ROBOT_BRAIN.md data flow)."""
from __future__ import annotations
import sys
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'src'))

from badminton_brain.types import (  # noqa: E402
    BASELINE_ORDER, BrainBoundaryError, Feedback, HitDecision, Layer,
    PredictedTrajectory, RobotSensorState, SafeCommand, ShuttleMeasurement,
    UnifiedState, WholeBodyTarget, check_batched, check_court_frame,
)


class LayerTests(unittest.TestCase):
    def test_baseline_order_matches_document(self) -> None:
        self.assertEqual(BASELINE_ORDER,
                         (Layer.PERCEPTION, Layer.ESTIMATION, Layer.PREDICTION,
                          Layer.DECISION, Layer.PLANNING, Layer.SAFETY, Layer.EXECUTION))
        self.assertIn(Layer.ADAPTATION, list(Layer))

    def test_layer_values_are_snake_case_identifiers(self) -> None:
        for layer in Layer:
            self.assertEqual(layer.value, layer.value.lower())


class MessageContractTests(unittest.TestCase):
    def test_shuttle_measurement_shape_and_frame(self) -> None:
        m = ShuttleMeasurement(position=np.zeros((4, 3)), velocity=np.zeros((4, 3)),
                               covariance=np.tile(np.eye(3), (4, 1, 1)), timestamp=1.25)
        self.assertEqual(m.frame, 'court')
        self.assertEqual(m.timestamp, 1.25)
        self.assertEqual(m.position.shape, (4, 3))

    def test_messages_refuse_non_court_frames(self) -> None:
        with self.assertRaises(BrainBoundaryError):
            RobotSensorState(frame='world', base_pose=np.zeros((2, 7)), joint_pos=np.zeros((2, 6)),
                             joint_vel=np.zeros((2, 6)), timestamp=0.0)

    def test_no_message_exposes_env_origin(self) -> None:
        for cls in (ShuttleMeasurement, RobotSensorState, UnifiedState, PredictedTrajectory,
                    HitDecision, WholeBodyTarget, SafeCommand, Feedback):
            fields = set(getattr(cls, '__dataclass_fields__', {}))
            self.assertNotIn('env_origin', fields, msg=cls.__name__)
            self.assertNotIn('env_origins', fields, msg=cls.__name__)

    def test_check_court_frame_helper(self) -> None:
        m = ShuttleMeasurement(position=np.zeros((1, 3)), velocity=np.zeros((1, 3)),
                               covariance=np.zeros((1, 3, 3)), timestamp=0.0)
        check_court_frame(m)
        m.frame = 'world'
        with self.assertRaises(BrainBoundaryError):
            check_court_frame(m)

    def test_check_batched_helper(self) -> None:
        check_batched('position', np.zeros((4, 3)), 3)
        with self.assertRaises(BrainBoundaryError):
            check_batched('position', np.zeros((4, 2)), 3)
        with self.assertRaises(BrainBoundaryError):
            check_batched('position', np.zeros((3,)), 3)

    def test_timestamp_must_be_finite_and_records_sim_time(self) -> None:
        with self.assertRaises(BrainBoundaryError):
            HitDecision(feasible=True, reason='ok', timestamp=float('nan'))


class StageOutputTypingTests(unittest.TestCase):
    def test_unified_state_carries_both_robot_and_shuttle(self) -> None:
        u = UnifiedState(base_pose=np.zeros((2, 7)), base_twist=np.zeros((2, 6)),
                         joint_pos=np.zeros((2, 6)), joint_vel=np.zeros((2, 6)),
                         racket_contact_pose=np.zeros((2, 7)), racket_contact_twist=np.zeros((2, 6)),
                         shuttle_position=np.zeros((2, 3)), shuttle_velocity=np.zeros((2, 3)),
                         timestamp=0.0)
        self.assertEqual(u.base_pose.shape, (2, 7))
        self.assertEqual(u.shuttle_position.shape, (2, 3))

    def test_predicted_trajectory_has_time_axis(self) -> None:
        t = PredictedTrajectory(times=np.linspace(0, 0.5, 6), position=np.zeros((2, 6, 3)),
                                velocity=np.zeros((2, 6, 3)), landing_point=np.zeros((2, 3)),
                                arrival_time=np.zeros((2,)), timestamp=0.0)
        self.assertEqual(t.position.shape[0], 2)
        self.assertEqual(t.times.ndim, 1)


if __name__ == '__main__':
    unittest.main()
