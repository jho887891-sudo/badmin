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

Review ruling D4 / observation 2 (broad review): a LEGAL sensor event must never crash the
closed loop.  Losing the target out of view is reported as an invalid detection (valid=False,
NaN uv); a detection window with nothing above the threshold (occlusion, underexposure, the
shuttle outside the window) is reported the same way by subpixel_centroid, which returns the
sentinel CentroidResult(uv=NaN, valid=False).  Only genuinely illegal input raises.

Review finding D4 (broad review): a target leaving the field of view is a NORMAL sensor event.
project()/detect() must report it as an invalid detection (valid=False, uv=NaN) and keep
running instead of throwing through pipeline.step; only genuinely invalid input (wrong shape,
empty batch, NaN/Inf coordinates) stays a BrainBoundaryError.  The discipline that a NaN
correspondence is never triangulated stays in stereo_geometry.triangulate().

Run: cd /home/T7/ojh/robot_sim && ./env_isaaclab/bin/python tests/badminton_brain/test_perception_geometry.py
"""
from __future__ import annotations

import sys
import unittest
import warnings
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'src'))

from badminton_brain.perception.stereo_geometry import (  # noqa: E402
    CAMERA_FRAME_LEFT, CAMERA_FRAME_RIGHT, COURT_FRAME, ENGINEERING_BASELINE_M,
    ENGINEERING_IMAGE_HEIGHT_PX, ENGINEERING_IMAGE_WIDTH_PX, ENGINEERING_LEFT_Y,
    ENGINEERING_RIGHT_Y, ENGINEERING_RIG_PITCH_DEG, ENGINEERING_RIG_WORLD, IMAGE_FRAME_LEFT,
    IMAGE_FRAME_RIGHT, CameraIntrinsics, CentroidResult, CourtBoxROI, ImageROI,
    StereoExtrinsics,
    baseline_stereo_setup, court_box_gate, peak_pixel, roi_gate, roi_select,
    subpixel_centroid, triangulate,
)
from badminton_brain.perception.stereo_geometry import CentroidResult  # noqa: E402, F401
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
    """Subpixel centroid: exact on discrete weights, bounded error on Gaussian blobs.

    Two distinct semantics, both tested below:
      * a legal but unusable WINDOW (nothing above the threshold: occlusion, underexposure,
        the shuttle outside the window) -> the sentinel, uv = NaN and valid = False (D4);
      * genuinely ILLEGAL input (shape, NaN/Inf, illegal parameters) -> BrainBoundaryError.
    """

    def test_exact_on_a_linear_ramp(self) -> None:
        patch = np.array([[0.0, 0.0, 0.0],
                          [0.0, 0.0, 1.0],
                          [0.0, 0.0, 2.0]])
        result = subpixel_centroid(patch)
        self.assertTrue(result.valid)
        self.assertTrue(np.allclose(result.uv, [2.0, 5.0 / 3.0], atol=1e-12), result.uv)
        self.assertAlmostEqual(result.weight_sum, 3.0, places=12)
        self.assertEqual(result.threshold, 0.0)
        self.assertEqual(result.peak, (2, 2), 'brightest sample is u=2 (col), v=2 (row)')

    def test_symmetric_peak_is_the_pixel_centre(self) -> None:
        patch = np.zeros((5, 5))
        patch[2, 3] = 1.0
        result = subpixel_centroid(patch)
        self.assertTrue(result.valid)
        self.assertTrue(np.allclose(result.uv, [3.0, 2.0], atol=1e-12))
        patch = np.array([[0.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 0.0]])
        result = subpixel_centroid(patch)
        self.assertTrue(result.valid)
        self.assertTrue(np.allclose(result.uv, [1.0, 1.0], atol=1e-12))
        self.assertEqual(tuple(peak_pixel(patch)), (1, 1))
        self.assertEqual(result.peak, (1, 1))

    def test_gaussian_blob_centre_is_subpixel_accurate(self) -> None:
        detector = SyntheticStereoDetector(noise_sigma_px=0.0)
        truth_uv = np.array([611.37, 288.62])
        patch, origin = detector.render_patch(truth_uv, sigma_px=1.2, half_size=6)
        self.assertEqual(patch.shape, (13, 13))
        result = subpixel_centroid(patch, origin_uv=origin)
        self.assertTrue(result.valid)
        error = float(np.abs(result.uv - truth_uv).max())
        print('  [T1] gaussian centroid error = {:.3e} px (origin {} )'.format(error, origin))
        self.assertLess(error, CENTROID_TOL_PX)
        # background subtraction keeps the estimate unbiased
        noisy_background = patch + 0.4
        result_bg = subpixel_centroid(noisy_background, threshold=0.4, origin_uv=origin)
        self.assertTrue(result_bg.valid)
        self.assertLess(float(np.abs(result_bg.uv - truth_uv).max()), CENTROID_TOL_PX * 2.0)

    def test_blank_window_returns_the_sentinel_not_an_exception(self) -> None:
        """D4 principle: a window with nothing to measure is a sensor event, not a failure."""
        cases = (
            ('all dark', np.zeros((5, 5)), 0.0),
            ('only negative intensity', np.zeros((5, 5)) - 2.0, 0.0),
            ('threshold above the peak', np.array([[0.1, 0.2], [0.3, 0.15]]), 0.5),
            ('single dark sample', np.zeros((1, 1)), 0.0),
        )
        for name, patch, threshold in cases:
            result = subpixel_centroid(patch, threshold=threshold)      # must not raise
            self.assertFalse(result.valid, name)
            self.assertEqual(result.uv.shape, (2,), name)
            self.assertTrue(np.all(np.isnan(result.uv)), name)
            self.assertEqual(result.weight_sum, 0.0, name)
        sentinel = subpixel_centroid(np.zeros((4, 4)), origin_uv=(600.0, 300.0))
        self.assertFalse(sentinel.valid)
        self.assertTrue(np.all(np.isnan(sentinel.uv)), 'never a fake centre such as (0, 0)')
        self.assertEqual(tuple(sentinel.origin_uv), (600.0, 300.0))

    def test_sentinel_and_estimate_share_one_convention(self) -> None:
        """Exactly the detector's convention: valid rows are finite, invalid rows are NaN."""
        detector = SyntheticStereoDetector(noise_sigma_px=0.0)
        truth_uv = np.array([611.37, 288.62])
        patch, origin = detector.render_patch(truth_uv, sigma_px=1.2, half_size=6)
        measured = subpixel_centroid(patch, origin_uv=origin)
        blank = subpixel_centroid(np.zeros_like(patch), origin_uv=origin)
        self.assertTrue(measured.valid)
        self.assertFalse(blank.valid)
        self.assertTrue(np.all(np.isfinite(measured.uv)))
        self.assertTrue(np.all(np.isnan(blank.uv)))
        # a batch built from both behaves like a SyntheticDetection.valid mask
        batch = np.stack([measured.uv, blank.uv])
        finite = np.isfinite(batch).all(axis=1)
        self.assertTrue(np.array_equal(finite, np.array([measured.valid, blank.valid])))
        with self.assertRaises(BrainBoundaryError):        # the sentinel is never a measurement
            triangulate(batch, np.zeros((2, 2)), np.eye(3) * 700.0, np.eye(4))

    def test_illegal_patches_are_still_rejected(self) -> None:
        """The sentinel must not swallow programming errors."""
        with self.assertRaises(BrainBoundaryError):
            subpixel_centroid(np.zeros(4))                       # not a 2-D patch
        with self.assertRaises(BrainBoundaryError):
            subpixel_centroid(np.full((3, 3), np.nan))           # NaN patch
        with self.assertRaises(BrainBoundaryError):
            subpixel_centroid(np.array([[0.0, np.inf], [1.0, 2.0]]))   # Inf patch
        with self.assertRaises(BrainBoundaryError):
            subpixel_centroid(np.zeros((0, 0)))                  # empty patch
        with self.assertRaises(BrainBoundaryError):
            subpixel_centroid(np.zeros((3, 3)), origin_uv=(1.0, 2.0, 3.0))   # bad origin shape
        with self.assertRaises(BrainBoundaryError):
            subpixel_centroid(np.zeros((3, 3)), origin_uv=(np.nan, 0.0))     # NaN origin
        with self.assertRaises(BrainBoundaryError):
            subpixel_centroid(np.zeros((3, 3)), threshold=float('nan'))      # illegal parameter
        with self.assertRaises(BrainBoundaryError):
            subpixel_centroid(np.zeros((3, 3)), threshold=float('inf'))


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
            result_l = subpixel_centroid(patch_l, origin_uv=origin_l)
            result_r = subpixel_centroid(patch_r, origin_uv=origin_r)
            self.assertTrue(result_l.valid and result_r.valid)
            uv_left[i] = result_l.uv
            uv_right[i] = result_r.uv
        volume = CourtBoxROI.perception_volume()
        points = triangulate(uv_left, uv_right, detector.intrinsics_left,
                             detector.extrinsics.T_left_right())
        inside = court_box_gate(points, volume)
        self.assertTrue(np.all(inside), 'every ground truth point is inside the play volume')
        error = float(np.abs(points - GT_POINTS_COURT).max())
        print('  [T1] patch -> centroid -> 3D error = {:.3e} m'.format(error))
        self.assertLess(error, PATCH_CHAIN_TOL_M)
        self.assertLess(float(np.abs(uv_left - detection.uv_left).max()), CENTROID_TOL_PX)

    def test_a_blank_window_is_skipped_not_thrown(self) -> None:
        """The front-end flow of the ruling: one target is hidden in this frame (occlusion /
        underexposure).  It must be skipped like an invalid detection, and the rest of the
        frame must still produce a measurement."""
        detector, detection = stereo_pair(GT_POINTS_COURT[:2], noise_sigma_px=0.0)
        uv_left = np.full_like(detection.uv_left, np.nan)
        results = []
        for i in range(len(detection.uv_left)):
            patch, origin = detector.render_patch(detection.uv_left[i], sigma_px=1.5,
                                                  half_size=6)
            if i == 1:                                  # the second target is hidden
                patch = np.zeros_like(patch)
            results.append(subpixel_centroid(patch, origin_uv=origin))
        self.assertEqual([r.valid for r in results], [True, False])
        self.assertTrue(np.all(np.isnan(results[1].uv)))
        uv_left[0] = results[0].uv
        keep = np.array([r.valid for r in results])
        self.assertLess(float(np.abs(uv_left[keep] - detection.uv_left[keep]).max()),
                        CENTROID_TOL_PX)
        points = triangulate(uv_left[keep], detection.uv_right[keep],
                             detector.intrinsics_left, detector.extrinsics.T_left_right())
        self.assertLess(float(np.abs(points - GT_POINTS_COURT[:1]).max()), PATCH_CHAIN_TOL_M)
        # pushing the sentinel row through anyway is still refused (measurement discipline)
        with self.assertRaises(BrainBoundaryError):
            triangulate(uv_left, detection.uv_right, detector.intrinsics_left,
                        detector.extrinsics.T_left_right())

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



class TestOutOfViewIsASensorEvent(unittest.TestCase):
    """Review finding D4: losing the target is a sensor event, not a pipeline failure.

    detect()/project() report it as valid=False with NaN uv and keep running; the refusal to
    triangulate a NaN correspondence stays where it belongs, in triangulate().
    """

    # a shuttle that has left the coverage on every axis: behind the rig, beside it, above it
    OUT_OF_VIEW = np.array([
        [-3.00, 0.00, 1.20],     # behind both cameras (negative camera depth)
        [0.40, 9.00, 1.40],      # far outside the images, to the side
        [0.40, 0.00, 40.00],     # far above the visible cone
    ])

    def _detector(self, noise_sigma_px: float = NOISE_SIGMA_PX) -> SyntheticStereoDetector:
        return SyntheticStereoDetector(noise_sigma_px=noise_sigma_px, seed=DEFAULT_SEED)

    def test_out_of_view_detection_is_invalid_not_an_exception(self) -> None:
        detector = self._detector()
        detection = detector.detect(self.OUT_OF_VIEW)          # D4: must not raise
        self.assertEqual(detection.count, len(self.OUT_OF_VIEW))
        self.assertFalse(np.any(detection.valid))
        self.assertEqual(detection.valid_count, 0)
        self.assertTrue(np.all(np.isnan(detection.uv_left)))
        self.assertTrue(np.all(np.isnan(detection.uv_right)))
        uv_left, uv_right = detection.valid_pairs()
        self.assertEqual(uv_left.shape, (0, 2))
        self.assertEqual(uv_right.shape, (0, 2))
        self.assertTrue(detection.is_synthetic)

    def test_project_itself_reports_the_empty_view(self) -> None:
        detector = self._detector(noise_sigma_px=0.0)
        detection = detector.project(self.OUT_OF_VIEW)         # D4: must not raise either
        self.assertEqual(detection.valid_count, 0)
        self.assertTrue(np.all(np.isnan(detection.uv_left)))
        self.assertTrue(np.all(np.isnan(detection.uv_right)))

    def test_invalid_input_is_still_rejected_loudly(self) -> None:
        """Only genuinely illegal input keeps raising: the D4 fix must not swallow bugs."""
        detector = self._detector()
        with self.assertRaises(BrainBoundaryError):
            detector.detect(np.zeros((0, 3)))                      # empty batch
        with self.assertRaises(BrainBoundaryError):
            detector.detect(np.zeros((4, 2)))                      # not (N, 3)
        with self.assertRaises(BrainBoundaryError):
            detector.detect(np.array([[np.nan, 0.0, 1.4]]))        # NaN coordinate
        with self.assertRaises(BrainBoundaryError):
            detector.detect(np.array([[0.4, 0.0, np.inf]]))        # Inf coordinate
        # and the geometry functions keep their own input discipline
        with self.assertRaises(BrainBoundaryError):
            triangulate(np.zeros((2, 3)), np.zeros((2, 3)), detector.intrinsics_left,
                        detector.extrinsics.T_left_right())

    def test_camera_plane_point_is_flagged_without_a_warning(self) -> None:
        """z_cam == 0 divides by zero: it must become an invalid row, never a warning/crash."""
        detector = self._detector(noise_sigma_px=0.0)
        left_pose = detector.extrinsics.T_court_left()
        on_the_plane = (left_pose[:3, 3] + 0.5 * left_pose[:3, 0]).reshape(1, 3)   # z_cam == 0
        with warnings.catch_warnings():
            warnings.simplefilter('error')                     # any numpy warning fails the test
            detection = detector.detect(on_the_plane)
        self.assertFalse(bool(detection.valid[0]))
        self.assertTrue(np.all(np.isnan(detection.uv_left)))
        self.assertTrue(np.all(np.isnan(detection.uv_right)))

    def test_a_mixed_frame_keeps_the_visible_point(self) -> None:
        detector = self._detector(noise_sigma_px=0.0)
        points = np.array([[0.40, 0.00, 1.40], [-3.00, 0.00, 1.20], [1.20, 0.00, 1.60]])
        detection = detector.detect(points)
        self.assertEqual(list(detection.valid), [True, False, True])
        self.assertTrue(np.all(np.isnan(detection.uv_left[~detection.valid])))
        self.assertTrue(np.all(np.isfinite(detection.uv_left[detection.valid])))
        recovered = triangulate(detection.uv_left[detection.valid],
                               detection.uv_right[detection.valid],
                               detector.intrinsics_left, detector.extrinsics.T_left_right())
        self.assertLess(float(np.abs(recovered - points[detection.valid]).max()),
                        TRIANGULATION_TOL_M)

    def test_the_detector_keeps_running_when_the_target_leaves_and_returns(self) -> None:
        """The pipeline.step sequence D4 crashed on: visible -> gone -> visible again."""
        detector = self._detector()
        visible = np.array([[0.40, 0.00, 1.40]])
        first = detector.detect(visible)
        self.assertTrue(bool(first.valid[0]))
        gone = detector.detect(self.OUT_OF_VIEW)               # target leaves the field of view
        self.assertEqual(gone.valid_count, 0)
        back = detector.detect(visible)                        # ... and comes back
        self.assertTrue(bool(back.valid[0]))
        self.assertTrue(np.all(np.isfinite(back.uv_left)))
        self.assertTrue(np.all(np.isfinite(back.uv_right)))

    def test_a_sweep_of_poses_outside_the_view_never_raises(self) -> None:
        detector = self._detector()
        grid = np.array([[x, y, z]
                         for x in np.linspace(-6.0, 3.0, 7)
                         for y in np.linspace(-6.0, 6.0, 7)
                         for z in (-1.0, 0.0, 1.4, 6.0)])
        detection = detector.detect(grid)                      # must not raise anywhere
        self.assertEqual(detection.count, grid.shape[0])
        inner = detection.valid
        # a valid row is a finite pair; the rows that are not valid never carry a fake pixel
        self.assertTrue(np.all(np.isfinite(detection.uv_left[inner])))
        self.assertTrue(np.all(np.isfinite(detection.uv_right[inner])))
        self.assertTrue(np.all(np.isnan(detection.uv_left[~inner]) |
                               np.isnan(detection.uv_right[~inner])))

    def test_triangulate_still_refuses_the_invalid_pair(self) -> None:
        """Requirement 2: the NaN correspondence is never turned into a measurement."""
        detector = self._detector()
        detection = detector.detect(self.OUT_OF_VIEW)
        with self.assertRaises(BrainBoundaryError):
            triangulate(detection.uv_left, detection.uv_right, detector.intrinsics_left,
                        detector.extrinsics.T_left_right())
        with self.assertRaises(BrainBoundaryError):            # the mask does not excuse NaN uv
            triangulate(detection.uv_left, detection.uv_right, detector.intrinsics_left,
                        detector.extrinsics.T_left_right(), valid=detection.valid)
        # the only discipline-preserving way to consume this frame: filter first (as the
        # perception adapter does), which leaves nothing to triangulate
        uv_left, uv_right = detection.valid_pairs()
        self.assertEqual(uv_left.shape[0], 0)


if __name__ == '__main__':
    unittest.main(verbosity=2)
