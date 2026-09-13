# -*- coding: utf-8 -*-
"""Perception layer geometry (ROBOT_BRAIN.md S11/S15): pixels -> 3D court-frame points.

stereo_geometry holds the calibration-aware geometry (frames and authenticity explicit);
synthetic_detector is the deterministic analytic test double used to test it without a
camera, without weights and without Isaac.  Nothing in this package predicts a trajectory or
decides anything: it measures, as the perception layer must.
"""
from .stereo_geometry import (
    CAMERA_FRAME_LEFT, CAMERA_FRAME_RIGHT, COURT_FRAME, ENGINEERING_BASELINE_M,
    ENGINEERING_IMAGE_HEIGHT_PX, ENGINEERING_IMAGE_WIDTH_PX, ENGINEERING_LEFT_Y,
    ENGINEERING_RIG_PITCH_DEG, ENGINEERING_RIG_WORLD, ENGINEERING_RIGHT_Y, IMAGE_FRAME_LEFT,
    IMAGE_FRAME_RIGHT, BaselineStereoSetup, CameraIntrinsics, CentroidResult, CourtBoxROI,
    ImageROI, StereoExtrinsics, baseline_stereo_setup, calibration_parameters, compose,
    court_box_gate, forward_camera_pose_court, forward_camera_rotation, intrinsics_matrix,
    peak_pixel, rigid_inverse, roi_gate, roi_select, subpixel_centroid, triangulate,
)
from .synthetic_detector import (
    DEFAULT_SEED, SYNTHETIC_SOURCE, SyntheticDetection, SyntheticStereoDetector,
)

__all__ = [
    "CAMERA_FRAME_LEFT", "CAMERA_FRAME_RIGHT", "COURT_FRAME", "IMAGE_FRAME_LEFT",
    "IMAGE_FRAME_RIGHT", "ENGINEERING_RIG_WORLD", "ENGINEERING_RIG_PITCH_DEG",
    "ENGINEERING_BASELINE_M", "ENGINEERING_LEFT_Y", "ENGINEERING_RIGHT_Y",
    "ENGINEERING_IMAGE_WIDTH_PX", "ENGINEERING_IMAGE_HEIGHT_PX",
    "CameraIntrinsics", "StereoExtrinsics", "BaselineStereoSetup", "ImageROI", "CourtBoxROI",
    "CentroidResult",
    "baseline_stereo_setup", "calibration_parameters", "triangulate", "intrinsics_matrix",
    "roi_gate", "roi_select", "court_box_gate", "subpixel_centroid", "peak_pixel",
    "forward_camera_rotation", "forward_camera_pose_court", "rigid_inverse", "compose",
    "SyntheticStereoDetector", "SyntheticDetection", "DEFAULT_SEED", "SYNTHETIC_SOURCE",
]
