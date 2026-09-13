# -*- coding: utf-8 -*-
"""Estimation layer adapter (coordinator ruling, 2026-09-13).

T2 delivered a batched 6-state EKF (:class:`RobotLocalization`) with its own API;
the frozen EstimationModule interface expects
``process(ShuttleMeasurement, RobotSensorState) -> UnifiedState``.
This adapter is the only place that bridges the two, so the EKF stays interface-agnostic
and no contract has to bend.

Shuttle filtering: if a shuttle filter (T3 UKF) is injected it is used; otherwise the
measurement is passed through and that shortcut is marked TEMP, never presented as an estimate.
"""
from __future__ import annotations

from typing import Any, Optional, Sequence, Tuple

import numpy as np

from ..interfaces import EstimationModule
from ..status import AssetStatus, Param
from ..types import BrainBoundaryError, Layer, RobotSensorState, ShuttleMeasurement, UnifiedState
from .robot_localization import RobotLocalization

RACKET_CONTACT_OFFSET = Param(
    (0.0, 0.0, 0.0), AssetStatus.TEMP_PARAMETERIZED_PROXY,
    "identity placeholder until measured T_link6_tcp / T_tcp_contact are available",
)

ABSOLUTE_POSE_NOISE = Param(
    1e-4, AssetStatus.TEMP_PARAMETERIZED_PROXY,
    "engineering baseline for the odometry/pose fusion measurement noise",
)


def _quat_from_yaw(yaw: np.ndarray) -> np.ndarray:
    half = yaw * 0.5
    q = np.zeros((yaw.shape[0], 4), dtype=float)
    q[:, 0] = np.cos(half)
    q[:, 3] = np.sin(half)
    return q


class EkfEstimatorModule(EstimationModule):
    """Robot EKF (+ optional shuttle filter) -> UnifiedState."""

    name = 'ekf_estimator'
    is_implemented = True

    def __init__(self, num_envs: int = 1, *, localization: Optional[RobotLocalization] = None,
                 shuttle_filter: Any = None, use_ukf: bool = True) -> None:
        self.num_envs = int(num_envs)
        self.localization = localization or RobotLocalization(num_envs=self.num_envs)
        if shuttle_filter is None and use_ukf:
            try:
                from .shuttle_ukf import ShuttleEstimatorBridge
                shuttle_filter = ShuttleEstimatorBridge(num_envs=self.num_envs)
            except Exception:  # the UKF is optional; passthrough is labelled TEMP below
                shuttle_filter = None
        self.shuttle_filter = shuttle_filter
        self._last_timestamp: Optional[float] = None
        if self.shuttle_filter is None:
            self.shuttle_source = 'TEMP: measurement passthrough (T3 UKF not available)'
        else:
            self.shuttle_source = type(self.shuttle_filter).__name__

    # ---- EstimationModule API ----
    def process(self, perception: ShuttleMeasurement, sensors: RobotSensorState) -> UnifiedState:
        if not isinstance(perception, ShuttleMeasurement) or not isinstance(sensors, RobotSensorState):
            raise BrainBoundaryError('EkfEstimatorModule.process expects (ShuttleMeasurement, RobotSensorState)')

        dt = 0.0 if self._last_timestamp is None else float(sensors.timestamp) - self._last_timestamp
        if dt > 0.0:
            odom = sensors.odom_twist
            if odom is None:
                odom = np.zeros((self.num_envs, 3), dtype=float)
            imu = sensors.imu_yaw_rate
            self.localization.predict(dt, odom, imu)
        self._last_timestamp = float(sensors.timestamp)

        pose_xyyaw = self._pose_xyyaw_from_base_pose(sensors.base_pose)
        covariance = np.tile(np.eye(3) * float(ABSOLUTE_POSE_NOISE.value), (self.num_envs, 1, 1))
        self.localization.update_pose_measurement(pose_xyyaw, covariance)

        base_pose = np.asarray(self.localization.base_pose(), dtype=float)
        base_twist = np.asarray(self.localization.base_twist(), dtype=float)
        shuttle_position, shuttle_velocity = self._shuttle_estimate(perception, sensors)

        return UnifiedState(
            base_pose=base_pose,
            base_twist=base_twist,
            joint_pos=sensors.joint_pos,
            joint_vel=sensors.joint_vel,
            racket_contact_pose=self._racket_contact_pose(base_pose),
            racket_contact_twist=base_twist,
            shuttle_position=shuttle_position,
            shuttle_velocity=shuttle_velocity,
            timestamp=float(sensors.timestamp),
        )

    def reset(self, env_ids: Sequence[int]) -> None:
        """Reset only the selected environments (UKF filter reset included)."""
        self.localization.reset(env_ids)
        if self.shuttle_filter is not None and hasattr(self.shuttle_filter, 'reset'):
            self.shuttle_filter.reset(env_ids)

    # ---- helpers ----
    def _pose_xyyaw_from_base_pose(self, base_pose: Any) -> np.ndarray:
        pose = np.asarray(base_pose, dtype=float)
        yaw = np.arctan2(2.0 * (pose[:, 0] * pose[:, 3] + pose[:, 1] * pose[:, 2]),
                         1.0 - 2.0 * (pose[:, 2] ** 2 + pose[:, 3] ** 2))
        return np.stack([pose[:, 0], pose[:, 1], yaw], axis=1)

    def _shuttle_estimate(self, perception: ShuttleMeasurement, sensors: RobotSensorState):
        if self.shuttle_filter is None:
            return np.asarray(perception.position, dtype=float), np.asarray(perception.velocity, dtype=float)
        estimate = self.shuttle_filter.update(perception, sensors)
        return np.asarray(estimate['position'], dtype=float), np.asarray(estimate['velocity'], dtype=float)

    def _racket_contact_pose(self, base_pose: np.ndarray) -> np.ndarray:
        """Court-frame racket-contact pose under a TEMP identity offset from robot_base."""
        offset = np.asarray(RACKET_CONTACT_OFFSET.value, dtype=float)
        pose = np.array(base_pose, dtype=float, copy=True)
        pose[:, :3] = pose[:, :3] + offset
        return pose

    def state_snapshot(self) -> Tuple[np.ndarray, ...]:
        """Per-environment (state, covariance) pairs, for reset-isolation tests."""
        state = np.asarray(self.localization._state, dtype=float) if hasattr(self.localization, '_state') else None
        covariance = np.asarray(self.localization._covariance, dtype=float) if hasattr(self.localization, '_covariance') else None
        if state is None or covariance is None:
            raise BrainBoundaryError('RobotLocalization does not expose _state/_covariance for snapshots')
        return tuple(np.concatenate([state[i], covariance[i].reshape(-1)]) for i in range(state.shape[0]))


__all__ = ["EkfEstimatorModule", "RACKET_CONTACT_OFFSET", "ABSOLUTE_POSE_NOISE"]
