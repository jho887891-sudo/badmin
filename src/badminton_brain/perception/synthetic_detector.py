# -*- coding: utf-8 -*-
"""Deterministic synthetic stereo detector - an analytic TEST DOUBLE, not a detector model.

This module does NOT load weights, does not run a network and never pretends to be a learned
detector (no YOLO, no training data, no image realism claim).  It is the analytic inverse of
stereo_geometry: it takes court-frame ground truth, projects it through the pinhole model of
the calibrated rig and adds Gaussian pixel noise from a fixed seed, so that perception
geometry can be tested end to end without a camera, without Isaac and without weights.

Because the ground truth of a synthetic frame is known exactly, the 1e-6 m triangulation
acceptance of task T1 is meaningful.  It says NOTHING about real camera accuracy: the real
intrinsics/extrinsics are still TEMP or unmeasured (see stereo_geometry
calibration_parameters()).

Authenticity: the noise level is a Param.  The default is a TEMP_PARAMETERIZED_PROXY
engineering guess and the real pixel noise of the assembled rig is an explicit
REQUIRES_MEASUREMENT slot with value None.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field, replace
from typing import Any, Dict, Optional, Tuple

import numpy as np

from ..status import AssetStatus, Param
from ..types import COURT_FRAME, BrainBoundaryError
from .stereo_geometry import (
    CAMERA_FRAME_LEFT, CAMERA_FRAME_RIGHT, IMAGE_FRAME_LEFT, IMAGE_FRAME_RIGHT,
    BaselineStereoSetup, CameraIntrinsics, ImageROI, baseline_stereo_setup,
)

DEFAULT_SEED = 20260913

SYNTHETIC_SOURCE = ('synthetic analytic pinhole projection with deterministic Gaussian pixel '
                    'noise (test double: no learned weights, no image data, no camera)')

_NOISE_SOURCE = ('TEMP engineering proxy for rig pixel noise: a plausible detector centroid '
                 'scatter for a slow shuttle in the play volume; the real value must be '
                 'identified on the assembled rig (see pixel_noise_sigma_measured_px)')


@dataclass(frozen=True)
class SyntheticDetection:
    """One synthetic frame pair: what a stereo detector would have reported.

    points_court is the ground truth the frame was generated from.  It exists ONLY for test
    error evaluation and must never be consumed by estimation/prediction (that would be
    cheating the estimator with truth).
    """
    points_court: np.ndarray
    uv_left: np.ndarray
    uv_right: np.ndarray
    valid: np.ndarray
    points_left_cam: np.ndarray
    points_right_cam: np.ndarray
    uv_left_frame: str = IMAGE_FRAME_LEFT
    uv_right_frame: str = IMAGE_FRAME_RIGHT
    left_camera_frame: str = CAMERA_FRAME_LEFT
    right_camera_frame: str = CAMERA_FRAME_RIGHT
    ground_truth_frame: str = COURT_FRAME
    is_synthetic: bool = True
    source: str = SYNTHETIC_SOURCE

    @property
    def count(self) -> int:
        return int(self.valid.shape[0])

    @property
    def valid_count(self) -> int:
        return int(np.count_nonzero(self.valid))

    def valid_pairs(self) -> Tuple[np.ndarray, np.ndarray]:
        """The correspondence pairs a downstream triangulator may use."""
        return self.uv_left[self.valid], self.uv_right[self.valid]


class SyntheticStereoDetector:
    """Court-frame points -> (uv_left, uv_right) with optional deterministic pixel noise."""

    def __init__(self, intrinsics: Optional[CameraIntrinsics] = None,
                 extrinsics: Optional[BaselineStereoSetup] = None,
                 noise_sigma_px: Any = 0.5,
                 seed: int = DEFAULT_SEED) -> None:
        setup = baseline_stereo_setup()
        self.intrinsics_left = intrinsics if intrinsics is not None \
            else setup.intrinsics_left
        self.intrinsics_right = replace(self.intrinsics_left, frame=IMAGE_FRAME_RIGHT,
                                        camera_frame=CAMERA_FRAME_RIGHT)
        if extrinsics is None:
            self.extrinsics = setup.extrinsics
        elif isinstance(extrinsics, BaselineStereoSetup):
            self.extrinsics = extrinsics.extrinsics
        else:
            self.extrinsics = extrinsics
        if isinstance(noise_sigma_px, Param):
            self.pixel_noise_sigma_px = noise_sigma_px
        else:
            sigma = float(noise_sigma_px)
            if sigma < 0.0:
                raise ValueError("noise_sigma_px must be >= 0")
            self.pixel_noise_sigma_px = Param(
                sigma, AssetStatus.TEMP_PARAMETERIZED_PROXY, _NOISE_SOURCE)
        self.pixel_noise_sigma_measured_px = Param(
            None, AssetStatus.REQUIRES_MEASUREMENT,
            "measured stereo detection noise for the assembled rig: record the shuttle "
            "detector scatter (or the residual of a static target) at the working distance; "
            "the synthetic noise is only an engineering proxy")
        self.seed = int(seed)
        self._rng = np.random.default_rng(self.seed)
        self._projected = 0

    # -- housekeeping -------------------------------------------------------
    def reset(self) -> None:
        """Restart the noise stream: the next detect() reproduces the first one."""
        self._rng = np.random.default_rng(self.seed)
        self._projected = 0

    @property
    def sigma_px(self) -> float:
        value = self.pixel_noise_sigma_px.value
        return 0.0 if value is None else float(value)

    def describe(self) -> str:
        return (f"synthetic stereo detector (test double): seed={self.seed}, "
                f"noise_sigma={self.sigma_px} px, "
                f"baseline={self.extrinsics.baseline_from_transform():.6f} m, "
                f"image={self.intrinsics_left.width_px}x{self.intrinsics_left.height_px}, "
                f"frames {IMAGE_FRAME_LEFT}/{IMAGE_FRAME_RIGHT}")

    def parameters(self) -> Dict[str, Param]:
        params: Dict[str, Param] = {}
        for name, param in self.intrinsics_left.parameters().items():
            params['intrinsics_left.' + name] = param
        for name, param in self.intrinsics_right.parameters().items():
            params['intrinsics_right.' + name] = param
        params.update(self.extrinsics.parameters())
        params['pixel_noise_sigma_px'] = self.pixel_noise_sigma_px
        params['pixel_noise_sigma_measured_px'] = self.pixel_noise_sigma_measured_px
        return params

    # -- geometry -----------------------------------------------------------
    def _check_points(self, points_court: Any) -> np.ndarray:
        points = np.asarray(points_court, dtype=float)
        if points.ndim != 2 or points.shape[1] != 3:
            raise BrainBoundaryError(
                f"points_court must be (N, 3) in the court frame, got {points.shape}")
        if points.shape[0] == 0:
            raise BrainBoundaryError("points_court must contain at least one point")
        if not np.all(np.isfinite(points)):
            raise BrainBoundaryError("points_court contains NaN/Inf")
        return points

    def _project_one(self, T_court_cam: np.ndarray, K: np.ndarray,
                     points: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        R = T_court_cam[:3, :3]
        t = T_court_cam[:3, 3]
        cam = (R.T @ (points - t).T).T
        z = cam[:, 2]
        with np.errstate(divide='ignore', invalid='ignore'):
            uv = (K @ cam.T).T
            uv = uv[:, :2] / z[:, None]
        visible = (z > 0.0) & np.isfinite(uv).all(axis=1)
        visible &= (uv[:, 0] >= 0.0) & (uv[:, 0] <= self.intrinsics_left.width_px - 1)
        visible &= (uv[:, 1] >= 0.0) & (uv[:, 1] <= self.intrinsics_left.height_px - 1)
        uv = np.where(visible[:, None], uv, np.nan)
        return cam, uv, visible

    def project(self, points_court: Any) -> SyntheticDetection:
        """Exact analytic projection (no noise): the ground-truth stereo pair."""
        points = self._check_points(points_court)
        K_l = self.intrinsics_left.matrix()
        K_r = self.intrinsics_right.matrix()
        cam_l, uv_l, vis_l = self._project_one(self.extrinsics.T_court_left(), K_l, points)
        cam_r, uv_r, vis_r = self._project_one(self.extrinsics.T_court_right(), K_r, points)
        if not np.any(vis_l):
            raise BrainBoundaryError(
                "no ground truth point is visible in the left image: the scene is outside the "
                "camera view (check the rig pose and the point cloud)")
        return SyntheticDetection(
            points_court=points, uv_left=uv_l, uv_right=uv_r, valid=(vis_l & vis_r),
            points_left_cam=cam_l, points_right_cam=cam_r)

    def detect(self, points_court: Any, add_noise: bool = True,
               roi: Optional[ImageROI] = None) -> SyntheticDetection:
        """Project and (optionally) add deterministic pixel noise; a pixel ROI gates the left
        image, and anything it rejects is reported as an invalid detection."""
        detection = self.project(points_court)
        uv_l = detection.uv_left.copy()
        uv_r = detection.uv_right.copy()
        sigma = self.sigma_px
        if add_noise and sigma > 0.0:
            count = uv_l.shape[0]
            noise_l = self._rng.normal(0.0, sigma, size=(count, 2))
            noise_r = self._rng.normal(0.0, sigma, size=(count, 2))
            uv_l = np.where(np.isfinite(uv_l), uv_l + noise_l, uv_l)
            uv_r = np.where(np.isfinite(uv_r), uv_r + noise_r, uv_r)
            self._projected += 1
        valid = detection.valid.copy()
        if roi is not None:
            if roi.frame != IMAGE_FRAME_LEFT:
                raise BrainBoundaryError(
                    f"the detector gates uv_left ({IMAGE_FRAME_LEFT}) but the ROI frame is "
                    f"{roi.frame!r}")
            keep = valid & roi.contains(uv_l)
            uv_l = np.where(keep[:, None], uv_l, np.nan)
            uv_r = np.where(keep[:, None], uv_r, np.nan)
            valid = keep
        return SyntheticDetection(
            points_court=detection.points_court, uv_left=uv_l, uv_right=uv_r, valid=valid,
            points_left_cam=detection.points_left_cam,
            points_right_cam=detection.points_right_cam)

    # -- image synthesis ----------------------------------------------------
    def render_patch(self, uv_px: Any, sigma_px: float = 1.2, half_size: int = 6,
                     amplitude: float = 1.0, background: float = 0.0
                     ) -> Tuple[np.ndarray, Tuple[int, int]]:
        """Gaussian shuttle blob centred on a fractional pixel -> (patch, origin_uv).

        patch[i, j] is the pixel at uv = (origin_u + j, origin_v + i), which is exactly the
        convention subpixel_centroid() expects in origin_uv.  Deterministic: no noise here.
        """
        uv = np.asarray(uv_px, dtype=float).reshape(-1)
        if uv.shape != (2,) or not np.all(np.isfinite(uv)):
            raise BrainBoundaryError(f"uv_px must be two finite numbers, got {uv_px!r}")
        sigma = float(sigma_px)
        if sigma <= 0.0:
            raise BrainBoundaryError("sigma_px must be positive")
        h = int(half_size)
        if h < 1:
            raise BrainBoundaryError("half_size must be >= 1")
        u0 = int(math.floor(float(uv[0]) - h))
        v0 = int(math.floor(float(uv[1]) - h))
        cols = u0 + np.arange(2 * h + 1, dtype=float)
        rows = v0 + np.arange(2 * h + 1, dtype=float)
        du = cols[None, :] - float(uv[0])
        dv = rows[:, None] - float(uv[1])
        patch = float(amplitude) * np.exp(-(du ** 2 + dv ** 2) / (2.0 * sigma ** 2))
        patch = patch + float(background)
        return patch, (u0, v0)


__all__ = ["DEFAULT_SEED", "SYNTHETIC_SOURCE", "SyntheticDetection", "SyntheticStereoDetector"]
