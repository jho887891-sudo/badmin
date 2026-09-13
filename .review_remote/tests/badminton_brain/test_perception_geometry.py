# -*- coding: utf-8 -*-
"""T1 - perception geometry (pure numpy): acceptance tests.

Spec:
  * docs/architecture/ROBOT_BRAIN.md S11/S15 - perception layer, court frame output
  * docs/architecture/COORDINATE_SYSTEM.md - court frame origin/axes, camera rig baseline
  * docs/simulation/BADMINTON_ROBOT.md S4/S12 - every number declares its authenticity;
    a guess may never be presented as a measurement

Acceptance (docs/superpowers/plans/2026-09-13-brain-modules.md, task T1):
  1. synthetic stereo pair -> triangulated 3D point within 1e-6 m of ground truth
  2. everything outside the ROI is rejected (pixel ROI and court play volume)
  3. no calibration constant is invented: fx/fy/cx/cy and the stereo extrinsics
     (baseline, left/right camera poses) are Param values with an explicit status + source

Run: cd /home/T7/ojh/robot_sim && ./env_isaaclab/bin/python tests/badminton_brain/test_perception_geometry.py
"""
from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'src'))

from badminton_brain.perception.stereo_geometry import (  # noqa: E402
    CAMERA_FRAME_LEFT, CAMERA_FRAME_RIGHT, COURT_FRAME, ENGINEERING_BASELINE_M,
    ENGINEERING_IMAGE_HEIGHT_PX, ENGINEERING_IMAGE_WIDTH_PX, ENGINEERING_LEFT_Y,
    ENGINEERING_RIGHT_Y, ENGINEERING_RIG_PITCH_DEG, ENGINEERING_RIG_WORLD, IMAGE_FRAME_LEFT,
    IMAGE_FRAME_RIGHT, CameraIntrinsics, CourtBoxROI, ImageROI, StereoExtrinsics,
    baseline_stereo_setup, court_box_gate, peak_pixel, roi_gate, roi_select,
    subpixel_centroid, triangulate,
)
from badminton_brain.perception.synthetic_detector import (  # noqa: E402
    DEFAULT_SEED, SyntheticStereoDetector,
)
from badminton_brain.status import UNRESOLVED_STATUSES, AssetStatus, Param  # noqa: E402
from badminton_brain.types import BrainBoundaryError  # noqa: E402

# Ground truth shuttle positions (court frame) inside the ENGINEERING_V0_1 perception ROI
# (scene_layout.PERCEPTION_ROI), roughly 1.5-3.0 m in front of the camera rig.
GT_POINTS_COURT = np.array([
    [0.30, -0.40, 1.35],
    [0.60, 0.25, 1.10],
    [1.20, 0.00, 1.60],
    [0.10, 0.55, 0.90],
    [1.35, -0.70, 2.10],
    [-0.10, 0.30, 1.45],
])

TRIANGULATION_TOL_M = 1e-6      # acceptance bound of task T1
CENTROID_TOL_PX = 0.05          # subpixel centroid, Gaussian blob on a 13x13 window
PATCH_CHAIN_TOL_M = 5e-3        # patch -> centroid -> triangulate (integer pixel grid)
NOISE_SIGMA_PX = 0.5            # TEMP proxy used by the detector default


def stereo_pair(points_court, noise_sigma_px: float = 0.0, seed: int = DEFAULT_SEED):
    """Project court-frame points into both images (noise-free unless asked)."""
    detector = SyntheticStereoDetector(noise_sigma_px=noise_sigma_px, seed=seed)
    detection = detector.detect(points_court, add_noise=noise_sigma_px > 0.0)
    return detector, detection


class TestCalibrationIsExplicit(unittest.TestCase):
    """Acceptance 3: frames and authenticity statuses are never implicit."""

    def test_intrinsics_declare_frames_and_status(self) -> None:
        K = CameraIntrinsics.engineering_baseline()
        self.assertEqual(K.frame, IMAGE_FRAME_LEFT)
        self.assertEqual(K.camera_frame, CAMERA_FRAME_LEFT)
        self.assertEqual((K.width_px, K.height_px),
                         (ENGINEERING_IMAGE_WIDTH_PX, ENGINEERING_IMAGE_HEIGHT_PX))
        for name in ('fx', 'fy', 'cx', 'cy'):
            value = getattr(K, name)
            self.assertIsInstance(value, Param, name)
            self.assertEqual(value.status, AssetStatus.TEMP_PARAMETERIZED_PROXY, name)
            self.assertIsNotNone(value.value, name)
            self.assertTrue(value.source.strip(), name + ' must state where the value comes from')
        matrix = K.matrix()
        self.assertEqual(matrix.shape, (3, 3))
        self.assertAlmostEqual(matrix[0, 0], K.fx.value, places=12)
        self.assertAlmostEqual(matrix[1, 1], K.fy.value, places=12)
        self.assertAlmostEqual(matrix[0, 2], K.cx.value, places=12)
        self.assertAlmostEqual(matrix[1, 2], K.cy.value, places=12)

    def test_uncalibrated_intrinsics_carry_no_value(self) -> None:
        K = CameraIntrinsics.uncalibrated()
        for name in ('fx', 'fy', 'cx', 'cy'):
            value = getattr(K, name)
            self.assertEqual(value.status, AssetStatus.REQUIRES_CALIBRATION, name)
            self.assertIsNone(value.value, name)
        with self.assertRaises(BrainBoundaryError):
            K.matrix()

    def test_measurement_cannot_be_faked(self) -> None:
        with self.assertRaises(ValueError):
            Param(700.0, AssetStatus.REQUIRES_CALIBRATION, 'pretend measured focal length')
        with self.assertRaises(ValueError):
            Param(0.29, AssetStatus.REQUIRES_MEASUREMENT, '')

    def test_extrinsics_declare_frames_and_baseline(self) -> None:
        ext = StereoExtrinsics.engineering_baseline()
        self.assertEqual(ext.pose_frame, COURT_FRAME)
        self.assertEqual(ext.left_frame, CAMERA_FRAME_LEFT)
        self.assertEqual(ext.right_frame, CAMERA_FRAME_RIGHT)
        self.assertEqual(ext.baseline_m.status, AssetStatus.TEMP_PARAMETERIZED_PROXY)
        self.assertAlmostEqual(ext.baseline_m.value, ENGINEERING_BASELINE_M, places=12)
        self.assertTrue(ext.baseline_m.source.strip())
        # the unmeasured real-rig quantities exist but must stay empty
        self.assertEqual(ext.baseline_measured_m.status, AssetStatus.REQUIRES_MEASUREMENT)
        self.assertIsNone(ext.baseline_measured_m.value)
        self.assertIn(ext.mount_pitch_measured_deg.status,
                      {AssetStatus.REQUIRES_MEASUREMENT, AssetStatus.REQUIRES_CALIBRATION})
        self.assertIsNone(ext.mount_pitch_measured_deg.value)
        self.assertIn(ext.relative_rotation_measured_deg.status, UNRESOLVED_STATUSES)
        self.assertIsNone(ext.relative_rotation_measured_deg.value)

        T_left_right = ext.T_left_right()
        self.assertEqual(T_left_right.shape, (4, 4))
        self.assertTrue(np.allclose(T_left_right[3], [0.0, 0.0, 0.0, 1.0], atol=1e-12))
        # baseline implied by the transform must equal the declared baseline Param
        implied = float(np.linalg.norm(T_left_right[:3, 3]))
        self.assertAlmostEqual(implied, ext.baseline_m.value, places=12)
        # extrinsic consistency: |p_left_pose - p_right_pose| == baseline in the court frame
        left = ext.left_camera_pose_court.value
        right = ext.right_camera_pose_court.value
        self.assertAlmostEqual(float(np.linalg.norm(left[:3, 3] - right[:3, 3])),
                               ext.baseline_m.value, places=12)
        # the poses reproduce the frozen scene layout rig (-1.40, 0, 1.20), pitch -4 deg, y +-0.145
        mid = 0.5 * (left[:3, 3] + right[:3, 3])
        self.assertTrue(np.allclose(mid, ENGINEERING_RIG_WORLD, atol=1e-12))
        self.assertAlmostEqual(float(left[1, 3]), ENGINEERING_LEFT_Y, places=12)
        self.assertAlmostEqual(float(right[1, 3]), ENGINEERING_RIGHT_Y, places=12)
        self.assertAlmostEqual(float(left[2, 3]), float(right[2, 3]), places=12)
        self.assertIsNotNone(ENGINEERING_RIG_PITCH_DEG)

    def test_parameters_report_lists_every_calibration_item(self) -> None:
        setup = baseline_stereo_setup()
        params = {}
        params.update(setup.intrinsics_left.parameters())
        params.update(setup.extrinsics.parameters())
        for name in ('fx', 'fy', 'cx', 'cy', 'baseline_m', 'left_camera_pose_court',
                     'right_camera_pose_court', 'left_right_transform'):
            self.assertIn(name, params, name)
            self.assertIsInstance(params[name], Param, name)
            self.assertTrue(params[name].source.strip(), name)
        for name, param in params.items():
            if param.status in (AssetStatus.REQUIRES_MEASUREMENT, AssetStatus.UNKNOWN,
                                AssetStatus.REQUIRES_CALIBRATION):
                self.assertIsNone(param.value, name + ' claims a value while unmeasured')

    def test_boxes_declare_status_and_source(self) -> None:
        roi = ImageROI(u_min=400.0, u_max=900.0, v_min=100.0, v_max=520.0,
                       frame=IMAGE_FRAME_LEFT, source='unit-test window',
                       status=AssetStatus.TEMP_PARAMETERIZED_PROXY)
        self.assertEqual(roi.frame, IMAGE_FRAME_LEFT)
        self.assertTrue(roi.source.strip())
        volume = CourtBoxROI.perception_volume()
        self.assertEqual(volume.frame, COURT_FRAME)
        self.assertEqual(volume.status, AssetStatus.TRACEABLE_REFERENCE)
        self.assertIn('scene_layout', volume.source)
        with self.assertRaises(ValueError):
            ImageROI(u_min=900.0, u_max=400.0, v_min=0.0, v_max=10.0, source='inverted')


class TestTriangulation(unittest.TestCase):
    """Acceptance 1: exact analytic stereo pair -> 3D within 1e-6 m."""

    def test_synthetic_stereo_pair_triangulates_within_1e_6_m(self) -> None:
        detector, detection = stereo_pair(GT_POINTS_COURT, noise_sigma_px=0.0)
        self.assertTrue(np.all(detection.valid), 'ground truth points must be visible')
        points = triangulate(detection.uv_left, detection.uv_right,
                             detector.intrinsics_left, detector.extrinsics.T_left_right())
        self.assertEqual(points.shape, GT_POINTS_COURT.shape)
        error = np.abs(points - GT_POINTS_COURT)
        print('  [T1] max |triangulated - ground truth| = {:.3e} m'.format(float(error.max())))
        self.assertLess(float(error.max()), TRIANGULATION_TOL_M)
        # the court frame is the declared contract of the brain (types.COURT_FRAME)
        self.assertEqual(COURT_FRAME, 'court')
        # the court-frame pose is really applied: an identity pose gives different numbers
        camera_frame = triangulate(detection.uv_left, detection.uv_right,
                                   detector.intrinsics_left, detector.extrinsics.T_left_right(),
                                   T_court_left=np.eye(4))
        self.assertGreater(float(np.abs(camera_frame - points).max()), 0.5)

    def test_camera_frame_triangulation_round_trips(self) -> None:
        detector, detection = stereo_pair(GT_POINTS_COURT, noise_sigma_px=0.0)
        K = detector.intrinsics_left
        T_left_right = detector.extrinsics.T_left_right()
        T_court_left = detector.extrinsics.left_camera_pose_court.value
        points_cam = triangulate(detection.uv_left, detection.uv_right, K, T_left_right,
                                 T_court_left=np.eye(4))
        # reproject the camera-frame result analytically: must reproduce the input pixels
        R = T_court_left[:3, :3]
        t = T_court_left[:3, 3]
        expected_cam = (R.T @ (GT_POINTS_COURT - t).T).T
        self.assertLess(float(np.abs(points_cam - expected_cam).max()), TRIANGULATION_TOL_M)
        uv = (K.matrix() @ points_cam.T).T
        uv = uv[:, :2] / uv[:, 2:3]
        self.assertLess(float(np.abs(uv - detection.uv_left).max()), 1e-9)

    def test_single_point_keeps_its_shape(self) -> None:
        detector, detection = stereo_pair(GT_POINTS_COURT[:1], noise_sigma_px=0.0)
        one = triangulate(detection.uv_left[0], detection.uv_right[0],
                          detector.intrinsics_left, detector.extrinsics.T_left_right())
        self.assertEqual(one.shape, (3,))
        self.assertAlmostEqual(float(np.linalg.norm(one - GT_POINTS_COURT[0])), 0.0, places=6)

    def test_invalid_inputs_are_rejected(self) -> None:
        detector, detection = stereo_pair(GT_POINTS_COURT, noise_sigma_px=0.0)
        K = detector.intrinsics_left
        T = detector.extrinsics.T_left_right()
        with self.assertRaises(BrainBoundaryError):        # batch mismatch
            triangulate(detection.uv_left, detection.uv_right[:-1], K, T)
        with self.assertRaises(BrainBoundaryError):        # not (..., 2)
            triangulate(np.zeros((3, 3)), np.zeros((3, 3)), K, T)
        with self.assertRaises(BrainBoundaryError):        # NaN measurement
            triangulate(np.full((2, 2), np.nan), detection.uv_right[:2], K, T)
        with self.assertRaises(BrainBoundaryError):        # scaled (non-rigid) extrinsics
            triangulate(detection.uv_left, detection.uv_right, K, 2.0 * T)
        with self.assertRaises(BrainBoundaryError):        # degenerate: identical rays
            triangulate(detection.uv_left, detection.uv_left, K, T)
        with self.assertRaises(BrainBoundaryError):        # invalid intrinsics
            triangulate(detection.uv_left, detection.uv_right,
                        CameraIntrinsics.uncalibrated(), T)

    def test_valid_mask_marks_rows_nan(self) -> None:
        detector, detection = stereo_pair(GT_POINTS_COURT, noise_sigma_px=0.0)
        valid = np.array([True, False, True, False, True, True])
        points = triangulate(detection.uv_left, detection.uv_right,
                             detector.intrinsics_left, detector.extrinsics.T_left_right(),
                             valid=valid)
        self.assertTrue(np.all(np.isnan(points[~valid])))
        self.assertTrue(np.all(np.isfinite(points[valid])))
        self.assertLess(float(np.abs(points[valid] - GT_POINTS_COURT[valid]).max()),
                        TRIANGULATION_TOL_M)


class TestRoiGating(unittest.TestCase):
    """Acceptance 2: outside the ROI -> rejected."""

    def setUp(self) -> None:
        self.roi = ImageROI(u_min=400.0, u_max=900.0, v_min=100.0, v_max=520.0,
                            frame=IMAGE_FRAME_LEFT, source='unit-test window',
                            status=AssetStatus.TEMP_PARAMETERIZED_PROXY)

    def test_pixel_roi_rejects_outside_and_nan(self) -> None:
        uv = np.array([
            [650.0, 300.0],       # inside
            [399.9, 300.0],       # left of the ROI
            [900.1, 300.0],       # right of the ROI
            [650.0, 99.9],        # above the ROI
            [650.0, 520.1],       # below the ROI
            [np.nan, 300.0],      # no measurement
        ])
        mask = roi_gate(uv, self.roi)
        self.assertEqual(mask.dtype, np.bool_)
        self.assertEqual(list(mask), [True, False, False, False, False, False])
        inside, same_mask = roi_select(uv, self.roi)
        self.assertEqual(inside.shape, (1, 2))
        self.assertTrue(np.allclose(inside[0], [650.0, 300.0]))
        self.assertTrue(np.array_equal(same_mask, mask))

    def test_roi_bounds_are_inclusive(self) -> None:
        corners = np.array([[400.0, 100.0], [900.0, 520.0], [400.0, 520.0], [900.0, 100.0]])
        self.assertTrue(np.all(roi_gate(corners, self.roi)))

    def test_single_point_gate_returns_a_scalar(self) -> None:
        self.assertTrue(bool(roi_gate(np.array([650.0, 300.0]), self.roi)))
        self.assertFalse(bool(roi_gate(np.array([10.0, 300.0]), self.roi)))

    def test_court_play_volume_gate(self) -> None:
        volume = CourtBoxROI.perception_volume()
        points = np.array([
            [0.40, 0.00, 1.40],     # inside the perception volume
            [-1.00, 0.00, 1.40],    # behind the robot / outside x range
            [3.00, 0.00, 1.40],     # beyond the far x bound
            [0.40, 2.50, 1.40],     # outside y
            [0.40, 0.00, 0.05],     # below z
        ])
        self.assertEqual(list(court_box_gate(points, volume)),
                         [True, False, False, False, False])

    def test_detector_roi_option_rejects_outside_detections(self) -> None:
        detector, _ = stereo_pair(GT_POINTS_COURT, noise_sigma_px=0.0)
        narrow = ImageROI(u_min=600.0, u_max=700.0, v_min=200.0, v_max=400.0,
                          frame=IMAGE_FRAME_LEFT, source='unit-test narrow window',
                          status=AssetStatus.TEMP_PARAMETERIZED_PROXY)
        detection = detector.detect(GT_POINTS_COURT, add_noise=False, roi=narrow)
        self.assertTrue(np.any(~detection.valid), 'the narrow ROI must reject points')
        self.assertTrue(np.all(roi_gate(detection.uv_left[detection.valid], narrow)))


class TestSubpixelCentroid(unittest.TestCase):
    """Subpixel centroid: exact on discrete weights, bounded error on Gaussian blobs."""

    def test_exact_on_a_linear_ramp(self) -> None:
        patch = np.array([[0.0, 0.0, 0.0],
                          [0.0, 0.0, 1.0],
                          [0.0, 0.0, 2.0]])
        uv = subpixel_centroid(patch)
        self.assertTrue(np.allclose(uv, [2.0, 5.0 / 3.0], atol=1e-12), uv)

    def test_symmetric_peak_is_the_pixel_centre(self) -> None:
        patch = np.zeros((5, 5))
        patch[2, 3] = 1.0
        self.assertTrue(np.allclose(subpixel_centroid(patch), [3.0, 2.0], atol=1e-12))
        patch = np.array([[0.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 0.0]])
        self.assertTrue(np.allclose(subpixel_centroid(patch), [1.0, 1.0], atol=1e-12))
        self.assertEqual(tuple(peak_pixel(patch)), (1, 1))

    def test_gaussian_blob_centre_is_subpixel_accurate(self) -> None:
        detector = SyntheticStereoDetector(noise_sigma_px=0.0)
        truth_uv = np.array([611.37, 288.62])
        patch, origin = detector.render_patch(truth_uv, sigma_px=1.2, half_size=6)
        self.assertEqual(patch.shape, (13, 13))
        uv = subpixel_centroid(patch, origin_uv=origin)
        error = float(np.abs(uv - truth_uv).max())
        print('  [T1] gaussian centroid error = {:.3e} px (origin {} )'.format(error, origin))
        self.assertLess(error, CENTROID_TOL_PX)
        # background subtraction keeps the estimate unbiased
        noisy_background = patch + 0.4
        uv_bg = subpixel_centroid(noisy_background, threshold=0.4, origin_uv=origin)
        self.assertLess(float(np.abs(uv_bg - truth_uv).max()), CENTROID_TOL_PX * 2.0)

    def test_unusable_patches_are_rejected(self) -> None:
        with self.assertRaises(BrainBoundaryError):
            subpixel_centroid(np.zeros((4, 4)))                  # no intensity
        with self.assertRaises(BrainBoundaryError):
            subpixel_centroid(np.zeros((4, 4)) - 1.0)            # only negative intensity
        with self.assertRaises(BrainBoundaryError):
            subpixel_centroid(np.zeros(4))                       # not a 2-D patch
        with self.assertRaises(BrainBoundaryError):
            subpixel_centroid(np.full((3, 3), np.nan))           # NaN patch
        with self.assertRaises(BrainBoundaryError):
            subpixel_centroid(np.zeros((0, 0)))                  # empty patch


class TestSyntheticDetector(unittest.TestCase):
    """Deterministic analytic test double - never a learned detector."""

    def test_same_seed_reproduces_identical_pixels(self) -> None:
        first = SyntheticStereoDetector(noise_sigma_px=NOISE_SIGMA_PX, seed=DEFAULT_SEED)
        second = SyntheticStereoDetector(noise_sigma_px=NOISE_SIGMA_PX, seed=DEFAULT_SEED)
        a = first.detect(GT_POINTS_COURT)
        b = second.detect(GT_POINTS_COURT)
        np.testing.assert_array_equal(a.uv_left, b.uv_left)
        np.testing.assert_array_equal(a.uv_right, b.uv_right)
        first.reset()
        c = first.detect(GT_POINTS_COURT)
        np.testing.assert_array_equal(a.uv_left, c.uv_left)
        other = SyntheticStereoDetector(noise_sigma_px=NOISE_SIGMA_PX, seed=DEFAULT_SEED + 1)
        d = other.detect(GT_POINTS_COURT)
        self.assertFalse(np.array_equal(a.uv_left, d.uv_left))

    def test_noise_is_zero_mean_and_bounded_by_sigma(self) -> None:
        count = 400
        rng = np.random.default_rng(7)
        points = np.repeat(GT_POINTS_COURT[2:3], count, axis=0)
        points[:, 1] += np.linspace(-0.3, 0.3, count)      # keep them all inside the image
        detector = SyntheticStereoDetector(noise_sigma_px=NOISE_SIGMA_PX, seed=DEFAULT_SEED)
        exact = SyntheticStereoDetector(noise_sigma_px=0.0, seed=DEFAULT_SEED)
        noisy = detector.detect(points)
        clean = exact.detect(points)
        self.assertTrue(np.all(noisy.valid))
        residual = noisy.uv_left - clean.uv_left
        print('  [T1] pixel noise: mean = {:.4f} px, std = {:.4f} px'.format(
            float(residual.mean()), float(residual.std())))
        self.assertLess(abs(float(residual.mean())), 0.25 * NOISE_SIGMA_PX)
        self.assertLess(float(residual.std()), 3.0 * NOISE_SIGMA_PX)
        self.assertGreater(float(residual.std()), 0.2 * NOISE_SIGMA_PX)
        self.assertFalse(np.array_equal(residual, np.zeros_like(residual)))
        self.assertGreater(count, 0)

    def test_noise_free_projection_is_analytic(self) -> None:
        detector, detection = stereo_pair(GT_POINTS_COURT, noise_sigma_px=0.0)
        K = detector.intrinsics_left.matrix()
        T = detector.extrinsics.left_camera_pose_court.value
        expected = (K @ (T[:3, :3].T @ (GT_POINTS_COURT - T[:3, 3]).T)).T
        expected = expected[:, :2] / expected[:, 2:3]
        self.assertLess(float(np.abs(detection.uv_left - expected).max()), 1e-9)

    def test_points_behind_or_outside_the_view_are_invalid(self) -> None:
        detector = SyntheticStereoDetector(noise_sigma_px=0.0)
        points = np.array([
            [0.40, 0.00, 1.40],     # visible
            [-2.60, 0.00, 1.20],    # behind both cameras
            [0.40, 9.00, 1.40],     # far outside the image to the side
        ])
        detection = detector.detect(points)
        self.assertEqual(list(detection.valid), [True, False, False])
        self.assertTrue(np.all(np.isnan(detection.uv_left[1:])))
        self.assertTrue(np.all(np.isnan(detection.uv_right[1:])))
        # both images see the point, at a horizontal disparity fx * baseline / depth
        self.assertFalse(np.allclose(detection.uv_left[0], detection.uv_right[0]))
        left_pose = detector.extrinsics.left_camera_pose_court.value
        depth = float((left_pose[:3, :3].T @ (points[0] - left_pose[:3, 3]))[2])
        expected_disparity = -detector.intrinsics_left.fx.value * ENGINEERING_BASELINE_M / depth
        self.assertAlmostEqual(float(detection.uv_right[0, 0] - detection.uv_left[0, 0]),
                               expected_disparity, places=6)
        self.assertGreater(abs(expected_disparity), 100.0)

    def test_detection_declares_itself_synthetic(self) -> None:
        detector = SyntheticStereoDetector(noise_sigma_px=0.0)
        detection = detector.detect(GT_POINTS_COURT)
        self.assertTrue(detection.is_synthetic)
        self.assertIn('synthetic', detection.source.lower())
        self.assertNotIn('yolo', detection.source.lower())
        self.assertEqual(detection.uv_left_frame, IMAGE_FRAME_LEFT)
        self.assertEqual(detection.uv_right_frame, IMAGE_FRAME_RIGHT)
        self.assertEqual(detection.left_camera_frame, CAMERA_FRAME_LEFT)
        self.assertEqual(detection.ground_truth_frame, COURT_FRAME)
        self.assertIn('synthetic', detector.describe().lower())

    def test_detector_noise_parameter_states_authenticity(self) -> None:
        detector = SyntheticStereoDetector(noise_sigma_px=NOISE_SIGMA_PX, seed=DEFAULT_SEED)
        params = detector.parameters()
        self.assertIn('pixel_noise_sigma_px', params)
        self.assertEqual(params['pixel_noise_sigma_px'].status,
                         AssetStatus.TEMP_PARAMETERIZED_PROXY)
        self.assertTrue(params['pixel_noise_sigma_px'].source.strip())
        self.assertEqual(params['pixel_noise_sigma_measured_px'].status,
                         AssetStatus.REQUIRES_MEASUREMENT)
        self.assertIsNone(params['pixel_noise_sigma_measured_px'].value)


class TestPerceptionChain(unittest.TestCase):
    """The chain the perception layer really runs: image patch -> uv -> 3D."""

    def test_patch_to_3d_chain(self) -> None:
        detector, detection = stereo_pair(GT_POINTS_COURT, noise_sigma_px=0.0)
        uv_left = np.zeros_like(detection.uv_left)
        uv_right = np.zeros_like(detection.uv_right)
        for i in range(len(GT_POINTS_COURT)):
            patch_l, origin_l = detector.render_patch(detection.uv_left[i], sigma_px=1.5,
                                                      half_size=6)
            patch_r, origin_r = detector.render_patch(detection.uv_right[i], sigma_px=1.5,
                                                      half_size=6)
            uv_left[i] = subpixel_centroid(patch_l, origin_uv=origin_l)
            uv_right[i] = subpixel_centroid(patch_r, origin_uv=origin_r)
        volume = CourtBoxROI.perception_volume()
        points = triangulate(uv_left, uv_right, detector.intrinsics_left,
                             detector.extrinsics.T_left_right())
        inside = court_box_gate(points, volume)
        self.assertTrue(np.all(inside), 'every ground truth point is inside the play volume')
        error = float(np.abs(points - GT_POINTS_COURT).max())
        print('  [T1] patch -> centroid -> 3D error = {:.3e} m'.format(error))
        self.assertLess(error, PATCH_CHAIN_TOL_M)
        self.assertLess(float(np.abs(uv_left - detection.uv_left).max()), CENTROID_TOL_PX)

    def test_roi_rejects_the_ball_entering_late(self) -> None:
        """A detection outside the ROI must never reach triangulation as a valid point."""
        detector = SyntheticStereoDetector(noise_sigma_px=0.0)
        roi = ImageROI(u_min=560.0, u_max=760.0, v_min=160.0, v_max=420.0,
                       frame=IMAGE_FRAME_LEFT, source='unit-test gate',
                       status=AssetStatus.TEMP_PARAMETERIZED_PROXY)
        outside = np.array([[-0.60, 0.00, 2.00], [1.30, 0.00, 0.60]])
        detection = detector.detect(outside, roi=roi)
        self.assertFalse(np.any(detection.valid))
        self.assertTrue(np.all(np.isnan(detection.uv_left)))
        # an out-of-ROI detection can never be pushed into triangulation as a measurement
        with self.assertRaises(BrainBoundaryError):
            triangulate(detection.uv_left, detection.uv_right, detector.intrinsics_left,
                        detector.extrinsics.T_left_right())


if __name__ == '__main__':
    unittest.main(verbosity=2)
