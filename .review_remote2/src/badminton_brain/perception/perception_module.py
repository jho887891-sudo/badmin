# -*- coding: utf-8 -*-
"""Perception adapter: stereo geometry (T1) -> PerceptionModule.

Coordinator ruling (2026-09-13): the frozen PerceptionModule signature receives only a
RobotSensorState (no images), so the observation source is injected at construction.
SIMULATED path: a truth provider supplies court-frame shuttle positions (simulation only),
T1 SyntheticStereoDetector projects them with TEMP calibration and adds pixel noise, and this
adapter triangulates the noisy correspondences back to 3D.  The ground truth never leaves this
file - only triangulated measurements are returned (T1 module rule).  REAL path: inject
measure_fn (stereo frames + measured calibration + trained detector).
"""
from __future__ import annotations

from typing import Any, Callable, Optional, Sequence

import numpy as np

from ..interfaces import PerceptionModule
from ..status import AssetStatus, Param
from ..types import BrainBoundaryError, RobotSensorState, ShuttleMeasurement
from .stereo_geometry import triangulate

DETECTOR_SOURCE = Param(
    'synthetic', AssetStatus.TEMP_PARAMETERIZED_PROXY,
    'T1 deterministic synthetic stereo detector; a trained detector with measured calibration '
    'is REQUIRES_CALIBRATION',
)

CALIBRATION_STATUS = Param(
    None, AssetStatus.REQUIRES_CALIBRATION,
    'intrinsics/baseline/rig pose of the assembled stereo head are not measured yet',
)


class StereoPerceptionModule(PerceptionModule):
    """Turns (simulated) stereo observations into a Court-Frame ShuttleMeasurement."""

    name = 'stereo_perception'
    is_implemented = True

    def __init__(self, num_envs: int = 1, *, truth_provider=None, measure_fn=None, detector=None,
                 measurement_noise_m: float = 1e-3) -> None:
        self.num_envs = int(num_envs)
        self.truth_provider = truth_provider
        self.measure_fn = measure_fn
        self.detector = detector if detector is not None else self._default_detector()
        self.detector_source = DETECTOR_SOURCE
        self.measurement_noise_m = float(measurement_noise_m)
        self.last_valid_mask = None
        self._prev_position = None
        self._prev_timestamp = None

    def process(self, sensors: RobotSensorState) -> ShuttleMeasurement:
        if not isinstance(sensors, RobotSensorState):
            raise BrainBoundaryError('StereoPerceptionModule.process expects a RobotSensorState')
        timestamp = float(sensors.timestamp)
        position = self._measure(timestamp)
        if position.ndim != 2 or position.shape[1] != 3:
            raise BrainBoundaryError('measurement must be (N, 3), got %s' % (position.shape,))
        velocity = self._velocity(position, timestamp)
        covariance = np.tile(np.eye(3) * self.measurement_noise_m ** 2, (position.shape[0], 1, 1))
        return ShuttleMeasurement(position=position, velocity=velocity, covariance=covariance,
                                  timestamp=timestamp)

    def reset(self, env_ids: Sequence[int]) -> None:
        self._prev_position = None
        self._prev_timestamp = None
        self.last_valid_mask = None
        reset = getattr(self.detector, 'reset', None)
        if callable(reset):
            reset()

    def _measure(self, timestamp: float) -> np.ndarray:
        if self.measure_fn is not None:
            return np.asarray(self.measure_fn(timestamp), dtype=float)
        if self.truth_provider is None:
            raise BrainBoundaryError(
                'no stereo source: construct with truth_provider=... (simulated) or measure_fn=... (real)')
        truth = np.asarray(self.truth_provider(self.num_envs, timestamp), dtype=float)
        if truth.ndim != 2 or truth.shape[1] != 3:
            raise BrainBoundaryError('truth_provider must return (N, 3), got %s' % (truth.shape,))
        detection = self.detector.detect(truth, add_noise=True)
        uv_left, uv_right = detection.valid_pairs()
        valid = np.asarray(detection.valid, dtype=bool)
        self.last_valid_mask = valid
        position = np.zeros_like(truth)
        if uv_left.shape[0] > 0:
            triangulated = triangulate(uv_left, uv_right, self.detector.intrinsics_left,
                                       self.detector.extrinsics.T_left_right(),
                                       T_court_left=self.detector.extrinsics.T_court_left())
            position[valid] = np.asarray(triangulated, dtype=float).reshape(-1, 3)
        return position

    def _velocity(self, position: np.ndarray, timestamp: float) -> np.ndarray:
        previous = self._prev_position
        previous_t = self._prev_timestamp
        self._prev_position = np.array(position, dtype=float, copy=True)
        self._prev_timestamp = timestamp
        if previous is None or previous_t is None or timestamp <= previous_t:
            return np.zeros_like(position)
        return (position - previous) / (timestamp - previous_t)

    def _default_detector(self):
        from .synthetic_detector import SyntheticStereoDetector
        return SyntheticStereoDetector()

    def readiness(self) -> dict:
        """What is still missing before this can run on real hardware."""
        return {
            'detector': self.detector_source.status.value,
            'simulated_path': self.truth_provider is not None,
            'real_path': self.measure_fn is not None,
            'camera_calibration': CALIBRATION_STATUS.status.value,
            'note': 'the synthetic path is an engineering proxy; it never claims measured accuracy',
        }


__all__ = ["StereoPerceptionModule", "DETECTOR_SOURCE", "CALIBRATION_STATUS"]

