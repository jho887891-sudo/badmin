# -*- coding: utf-8 -*-
"""CPU-only tests for the Isaac RGB-D perception adapter using camera doubles."""
from __future__ import annotations

import hashlib
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from badminton_brain.perception.isaac_rgbd_yolo import (  # noqa: E402
    IsaacRgbdPerceptionModule,
    UltralyticsShuttleDetector,
    read_camera_frame,
)
from badminton_brain.types import RobotSensorState  # noqa: E402


class Proxy:
    def __init__(self, value):
        self.torch = np.asarray(value)


class FakeCameraData:
    def __init__(self, *, rgb=None, depth=None, intrinsic=None, position=None, quaternion=None):
        if rgb is None:
            rgb = np.zeros((1, 80, 100, 4), dtype=np.uint8)
        if depth is None:
            depth = np.full((1, 80, 100, 1), 2.0, dtype=np.float32)
        if intrinsic is None:
            intrinsic = np.array([[[100.0, 0.0, 50.0],
                                   [0.0, 100.0, 40.0],
                                   [0.0, 0.0, 1.0]]])
        if position is None:
            position = np.array([[1.0, 2.0, 3.0]])
        if quaternion is None:
            quaternion = np.array([[0.0, 0.0, 0.0, 1.0]])
        self.output = {
            "rgb": Proxy(rgb),
            "distance_to_image_plane": Proxy(depth),
        }
        self.intrinsic_matrices = Proxy(intrinsic)
        self.pos_w = Proxy(position)
        self.quat_w_opengl = Proxy(quaternion)


class FakeCamera:
    def __init__(self, **kwargs):
        self.data = FakeCameraData(**kwargs)


class FakeDetector:
    def __init__(self, detections=None):
        self.detections = detections or []
        self.last_image = None

    def detect(self, image_rgb):
        self.last_image = np.asarray(image_rgb).copy()
        return list(self.detections)


def sensor_state(timestamp):
    return RobotSensorState(
        base_pose=np.array([[0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0]]),
        joint_pos=np.zeros((1, 6)),
        joint_vel=np.zeros((1, 6)),
        timestamp=float(timestamp),
    )


class IsaacRgbdPerceptionTests(unittest.TestCase):
    def setUp(self):
        self.camera = FakeCamera()
        self.detector = FakeDetector([
            {"xyxy": [45.0, 35.0, 55.0, 45.0], "confidence": 0.9, "class_id": 0}
        ])
        self.module = IsaacRgbdPerceptionModule(self.camera, self.detector)

    def test_reads_rgb_depth_intrinsics_and_pose(self):
        frame = read_camera_frame(self.camera)
        self.assertEqual(frame["rgb"].shape, (80, 100, 3))
        self.assertEqual(frame["depth_m"].shape, (80, 100))
        np.testing.assert_allclose(frame["intrinsic_matrix_px"][0, 0], 100.0)
        np.testing.assert_allclose(frame["position_world_m"], [1.0, 2.0, 3.0])

    def test_accepts_unbatched_rgb_and_depth_buffers(self):
        self.camera.data.output["rgb"] = Proxy(np.zeros((80, 100, 3), dtype=np.uint8))
        self.camera.data.output["distance_to_image_plane"] = Proxy(
            np.full((80, 100), 2.0, dtype=np.float32)
        )
        frame = read_camera_frame(self.camera)
        self.assertEqual(frame["rgb"].shape, (80, 100, 3))
        self.assertEqual(frame["depth_m"].shape, (80, 100))

    def test_depth_backprojection_returns_world_frame_measurement(self):
        result = self.module.process(sensor_state(0.0))
        self.assertEqual(result.position.shape, (1, 3))
        np.testing.assert_allclose(result.position[0], [1.0, 2.0, 1.0], atol=1e-8)
        np.testing.assert_allclose(result.valid_mask, [1.0])
        self.assertEqual(self.module.last_diagnostic["reason"], "detection_and_depth_valid")

    def test_no_detection_is_a_masked_miss_not_origin_measurement(self):
        self.detector.detections = []
        result = self.module.process(sensor_state(0.0))
        np.testing.assert_allclose(result.valid_mask, [0.0])
        np.testing.assert_allclose(result.position, np.zeros((1, 3)))
        self.assertEqual(self.module.last_diagnostic["reason"], "no_target_detection")

    def test_invalid_depth_is_a_masked_miss(self):
        self.camera.data.output["distance_to_image_plane"] = Proxy(
            np.full((1, 80, 100, 1), np.inf, dtype=np.float32)
        )
        result = self.module.process(sensor_state(0.0))
        np.testing.assert_allclose(result.valid_mask, [0.0])
        self.assertEqual(self.module.last_diagnostic["reason"], "no_valid_depth_in_center_patch")

    def test_wrong_class_is_not_accepted_as_shuttle(self):
        self.detector.detections = [
            {"xyxy": [45.0, 35.0, 55.0, 45.0], "confidence": 0.99, "class_id": 1}
        ]
        result = self.module.process(sensor_state(0.0))
        np.testing.assert_allclose(result.valid_mask, [0.0])
        self.assertEqual(self.module.last_diagnostic["reason"], "no_target_detection")

    def test_missing_class_id_is_not_accepted_when_target_class_is_required(self):
        self.detector.detections = [
            {"xyxy": [45.0, 35.0, 55.0, 45.0], "confidence": 0.99}
        ]
        result = self.module.process(sensor_state(0.0))
        np.testing.assert_allclose(result.valid_mask, [0.0])
        self.assertEqual(self.module.last_diagnostic["reason"], "no_target_detection")

    def test_nonfinite_rgb_buffer_is_rejected(self):
        rgb = np.zeros((1, 80, 100, 3), dtype=np.float32)
        rgb[0, 0, 0, 0] = np.nan
        self.camera.data.output["rgb"] = Proxy(rgb)
        with self.assertRaisesRegex(Exception, "rgb buffer contains NaN/Inf"):
            self.module.process(sensor_state(0.0))

    def test_velocity_uses_two_valid_timestamped_measurements(self):
        self.module.process(sensor_state(0.0))
        self.detector.detections = [
            {"xyxy": [55.0, 35.0, 65.0, 45.0], "confidence": 0.9, "class_id": 0}
        ]
        result = self.module.process(sensor_state(0.1))
        self.assertEqual(self.module.last_diagnostic["velocity_status"],
                         "FINITE_DIFFERENCE_BETWEEN_VALID_RGBD_MEASUREMENTS")
        self.assertAlmostEqual(float(result.velocity[0, 0]), 2.0, places=6)

    def test_covariance_is_symmetric_positive_semidefinite(self):
        result = self.module.process(sensor_state(0.0))
        covariance = result.covariance[0]
        np.testing.assert_allclose(covariance, covariance.T, atol=1e-12)
        self.assertGreaterEqual(float(np.linalg.eigvalsh(covariance).min()), -1e-12)

    def test_requires_single_environment(self):
        with self.assertRaises(ValueError):
            IsaacRgbdPerceptionModule(self.camera, self.detector, num_envs=2)

    def test_expected_checkpoint_sha256_mismatch_fails_before_model_loading(self):
        with tempfile.TemporaryDirectory() as tmp:
            checkpoint = Path(tmp) / "weights.pt"
            checkpoint.write_bytes(b"frozen-test-bytes")
            actual = hashlib.sha256(b"frozen-test-bytes").hexdigest()
            self.assertEqual(len(actual), 64)
            with self.assertRaisesRegex(ValueError, "checkpoint SHA256 mismatch"):
                UltralyticsShuttleDetector(checkpoint, expected_sha256="0" * 64)
            with self.assertRaisesRegex(ValueError, "64-character hexadecimal"):
                UltralyticsShuttleDetector(checkpoint, expected_sha256="not-a-sha")

    def test_detector_receives_rgb_without_alpha(self):
        image = np.zeros((1, 80, 100, 4), dtype=np.uint8)
        image[0, :, :, 0] = 200
        self.camera.data.output["rgb"] = Proxy(image)
        self.module.process(sensor_state(0.0))
        self.assertEqual(self.detector.last_image.shape, (80, 100, 3))
        self.assertEqual(int(self.detector.last_image[0, 0, 0]), 200)


if __name__ == "__main__":
    unittest.main()
