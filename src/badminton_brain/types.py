# -*- coding: utf-8 -*-
"""Robot Brain message contracts (docs/architecture/ROBOT_BRAIN.md data flow).

Data flow: Perception -> ShuttleMeasurement / RobotSensorState
           Estimation -> UnifiedState
           Prediction -> PredictedTrajectory
           Decision   -> HitDecision / BestIntercept
           Planning   -> WholeBodyTarget
           Safety     -> SafeCommand
           Execution  -> Robot/Environment (+ Feedback)

Architecture baseline order is the document's baseline (ROBOT_BRAIN.md S15) and is
enforced by the pipeline.  Every message carries a simulation timestamp (S43) and lives
in the Court Frame (COORDINATE_SYSTEM.md S3): no message may expose env_origin (S28).
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Iterable, Optional, Sequence, Tuple

import numpy as np


class BrainBoundaryError(RuntimeError):
    """Raised when a message or module crosses a layer boundary it must not cross."""


class Layer(str, Enum):
    """Layers of the modular architecture (ROBOT_BRAIN.md S11/S12)."""
    PERCEPTION = 'perception'
    ESTIMATION = 'estimation'
    PREDICTION = 'prediction'
    DECISION = 'decision'
    PLANNING = 'planning'
    SAFETY = 'safety'
    EXECUTION = 'execution'
    ADAPTATION = 'adaptation'


COURT_FRAME = 'court'

# Architecture Baseline (ROBOT_BRAIN.md S15): the main closed loop, in order.
BASELINE_ORDER: Tuple[Layer, ...] = (
    Layer.PERCEPTION,
    Layer.ESTIMATION,
    Layer.PREDICTION,
    Layer.DECISION,
    Layer.PLANNING,
    Layer.SAFETY,
    Layer.EXECUTION,
)


def _as_array(value: Any) -> Optional[np.ndarray]:
    if value is None:
        return None
    return np.asarray(value, dtype=float)


def check_batched(name: str, array: Any, expected_last_dim: int) -> np.ndarray:
    """Require a batched tensor of shape (N, ..., expected_last_dim) with N >= 1."""
    arr = _as_array(array)
    if arr is None:
        raise BrainBoundaryError(f"{name} must not be None")
    if arr.ndim < 2:
        raise BrainBoundaryError(f"{name} must be batched, got shape {arr.shape}")
    if arr.shape[-1] != expected_last_dim:
        raise BrainBoundaryError(
            f"{name} last dimension must be {expected_last_dim}, got shape {arr.shape}")
    if arr.shape[0] < 1:
        raise BrainBoundaryError(f"{name} must have at least one batch element")
    if not np.all(np.isfinite(arr)):
        raise BrainBoundaryError(f"{name} contains NaN/Inf")
    return arr


def check_per_env_scalar(name: str, array: Any) -> np.ndarray:
    """Require a per-environment scalar tensor of shape (N,)."""
    arr = _as_array(array)
    if arr is None:
        raise BrainBoundaryError(f"{name} must not be None")
    if arr.ndim != 1 or arr.shape[0] < 1:
        raise BrainBoundaryError(f"{name} must be (N,), got shape {arr.shape}")
    if not np.all(np.isfinite(arr)):
        raise BrainBoundaryError(f"{name} contains NaN/Inf")
    return arr


def check_court_frame(message: Any) -> None:
    """S28: Robot Brain only ever sees Court Frame quantities."""
    frame = getattr(message, 'frame', None)
    if frame != COURT_FRAME:
        raise BrainBoundaryError(
            f"{type(message).__name__}.frame must be '{COURT_FRAME}', got {frame!r}")


def _check_timestamp(timestamp: float, owner: str) -> float:
    value = float(timestamp)
    if not math.isfinite(value):
        raise BrainBoundaryError(f"{owner}.timestamp must be finite (simulation time)")
    return value


def _check_batch_dim(values: Sequence[Optional[np.ndarray]], owner: str) -> int:
    dims = {int(v.shape[0]) for v in values if v is not None}
    if len(dims) > 1:
        raise BrainBoundaryError(f"{owner}: inconsistent batch sizes {sorted(dims)}")
    return dims.pop() if dims else 0


@dataclass
class ShuttleMeasurement:
    """Perception output: shuttle 3D measurement in the Court Frame (batched)."""
    position: Any = None
    velocity: Any = None
    covariance: Any = None
    timestamp: float = 0.0
    frame: str = COURT_FRAME
    # DEC-023: which rows are real detections.  An out-of-view shuttle must never be
    # consumed as a measurement at the origin.
    valid_mask: Any = None

    def __post_init__(self) -> None:
        check_court_frame(self)
        self.timestamp = _check_timestamp(self.timestamp, 'ShuttleMeasurement')
        if self.valid_mask is not None:
            self.valid_mask = check_per_env_scalar('ShuttleMeasurement.valid_mask',
                                                   np.asarray(self.valid_mask, dtype=float))
        self.position = check_batched('ShuttleMeasurement.position', self.position, 3)
        self.velocity = check_batched('ShuttleMeasurement.velocity', self.velocity, 3)
        self.covariance = check_batched('ShuttleMeasurement.covariance', self.covariance, 3)
        _check_batch_dim([self.position, self.velocity, self.covariance], 'ShuttleMeasurement')


@dataclass
class RobotSensorState:
    """Perception output: robot proprioception + odometry in the Court Frame."""
    frame: str = COURT_FRAME
    base_pose: Any = None
    joint_pos: Any = None
    joint_vel: Any = None
    timestamp: float = 0.0
    # Optional proprioception channels (coordinator ruling 2026-09-13, DEC-013): the
    # estimation layer needs odometry and a yaw-rate source, but they must stay optional so
    # that existing callers keep working unchanged.
    # SEMANTICS (pinned after the T2 review flagged the hazard): odom_twist is the
    # PER-STEP INCREMENT supplied by the wheel odometry, [dx_body, dy_body, dyaw] since the
    # previous call - NOT a velocity despite the field name; the EKF integrates it once.
    # imu_yaw_rate is the yaw rate in rad/s for the same interval.
    odom_twist: Any = None
    imu_yaw_rate: Any = None

    def __post_init__(self) -> None:
        check_court_frame(self)
        self.timestamp = _check_timestamp(self.timestamp, 'RobotSensorState')
        self.base_pose = check_batched('RobotSensorState.base_pose', self.base_pose, 7)
        self.joint_pos = check_batched('RobotSensorState.joint_pos', self.joint_pos, 6)
        self.joint_vel = check_batched('RobotSensorState.joint_vel', self.joint_vel, 6)
        if self.odom_twist is not None:
            self.odom_twist = check_batched('RobotSensorState.odom_twist', self.odom_twist, 3)
        if self.imu_yaw_rate is not None:
            self.imu_yaw_rate = check_per_env_scalar('RobotSensorState.imu_yaw_rate', self.imu_yaw_rate)
        _check_batch_dim([self.base_pose, self.joint_pos, self.joint_vel,
                          self.odom_twist, self.imu_yaw_rate], 'RobotSensorState')


@dataclass
class UnifiedState:
    """Estimation output: one consistent Court-Frame state for the whole robot + shuttle."""
    base_pose: Any = None
    base_twist: Any = None
    joint_pos: Any = None
    joint_vel: Any = None
    racket_contact_pose: Any = None
    racket_contact_twist: Any = None
    shuttle_position: Any = None
    shuttle_velocity: Any = None
    timestamp: float = 0.0
    frame: str = COURT_FRAME

    def __post_init__(self) -> None:
        check_court_frame(self)
        self.timestamp = _check_timestamp(self.timestamp, 'UnifiedState')
        self.base_pose = check_batched('UnifiedState.base_pose', self.base_pose, 7)
        self.base_twist = check_batched('UnifiedState.base_twist', self.base_twist, 6)
        self.joint_pos = check_batched('UnifiedState.joint_pos', self.joint_pos, 6)
        self.joint_vel = check_batched('UnifiedState.joint_vel', self.joint_vel, 6)
        self.racket_contact_pose = check_batched('UnifiedState.racket_contact_pose', self.racket_contact_pose, 7)
        self.racket_contact_twist = check_batched('UnifiedState.racket_contact_twist', self.racket_contact_twist, 6)
        self.shuttle_position = check_batched('UnifiedState.shuttle_position', self.shuttle_position, 3)
        self.shuttle_velocity = check_batched('UnifiedState.shuttle_velocity', self.shuttle_velocity, 3)
        _check_batch_dim([self.base_pose, self.base_twist, self.joint_pos, self.joint_vel,
                          self.racket_contact_pose, self.racket_contact_twist,
                          self.shuttle_position, self.shuttle_velocity], 'UnifiedState')


@dataclass
class PredictedTrajectory:
    """Prediction output: (N, T, 3) shuttle trajectory over the prediction horizon."""
    times: Any = None
    position: Any = None
    velocity: Any = None
    landing_point: Any = None
    arrival_time: Any = None
    # DEC-019: promoted from the predictor private attribute to the contract, because a
    # horizon-truncated landing point must never be consumed as a real bounce.
    landed_within_horizon: Any = None
    timestamp: float = 0.0
    frame: str = COURT_FRAME

    def __post_init__(self) -> None:
        check_court_frame(self)
        self.timestamp = _check_timestamp(self.timestamp, 'PredictedTrajectory')
        self.times = np.asarray(self.times, dtype=float) if self.times is not None else None
        if self.times is None or self.times.ndim != 1:
            raise BrainBoundaryError('PredictedTrajectory.times must be 1-D (T,)')
        if not np.all(np.isfinite(self.times)):
            raise BrainBoundaryError('PredictedTrajectory.times contains NaN/Inf')
        self.position = check_batched('PredictedTrajectory.position', self.position, 3)
        self.velocity = check_batched('PredictedTrajectory.velocity', self.velocity, 3)
        if self.position.ndim != 3 or self.position.shape[1] != self.times.shape[0]:
            raise BrainBoundaryError('PredictedTrajectory.position must be (N, T, 3) matching times')
        self.landing_point = check_batched('PredictedTrajectory.landing_point', self.landing_point, 3)
        self.arrival_time = check_per_env_scalar('PredictedTrajectory.arrival_time', self.arrival_time)
        if self.landed_within_horizon is not None:
            self.landed_within_horizon = check_per_env_scalar(
                'PredictedTrajectory.landed_within_horizon',
                np.asarray(self.landed_within_horizon, dtype=float))


@dataclass
class HitDecision:
    """Decision output: whether the shuttle should be played at all (no planning here)."""
    feasible: bool = False
    reason: str = ''
    timestamp: float = 0.0
    frame: str = COURT_FRAME

    def __post_init__(self) -> None:
        check_court_frame(self)
        self.timestamp = _check_timestamp(self.timestamp, 'HitDecision')


@dataclass
class BestIntercept:
    """Decision output: where and when to meet the shuttle (no motion generation)."""
    position: Any = None
    time_s: Any = None
    racket_pose: Any = None
    score: Any = None
    timestamp: float = 0.0
    frame: str = COURT_FRAME

    def __post_init__(self) -> None:
        check_court_frame(self)
        self.timestamp = _check_timestamp(self.timestamp, 'BestIntercept')
        if self.position is not None:
            self.position = check_batched('BestIntercept.position', self.position, 3)
        if self.racket_pose is not None:
            self.racket_pose = check_batched('BestIntercept.racket_pose', self.racket_pose, 7)


@dataclass
class WholeBodyTarget:
    """Planning/policy output: base twist + arm joint targets. Never executed directly."""
    base_twist: Any = None
    joint_position_target: Any = None
    # SCALAR by contract: while BestIntercept.time_s is per-environment, this horizon is the
    # batch-min deadline (the most conservative), so every environment is served by a
    # horizon that is valid for it.  Changing this to (N,) would break Execution/Safety.
    horizon_s: float = 0.0
    timestamp: float = 0.0
    frame: str = COURT_FRAME

    def __post_init__(self) -> None:
        check_court_frame(self)
        self.timestamp = _check_timestamp(self.timestamp, 'WholeBodyTarget')
        self.base_twist = check_batched('WholeBodyTarget.base_twist', self.base_twist, 3)
        self.joint_position_target = check_batched('WholeBodyTarget.joint_position_target',
                                                   self.joint_position_target, 6)


@dataclass
class SafeCommand:
    """Safety output: the only message Execution is allowed to consume.

    NOTE (DEC-014 / ISSUE-010): ``base_twist`` is expressed in the robot_base (body) frame
    [vx_body, vy_body, wz] while this message ``frame`` stays 'court' (the court frame is the
    canonical global reference for poses and timestamps).  The court->body rotation is done by
    the planner, so any driver or dataset that reads this message must NOT reinterpret
    ``base_twist`` as a court-frame velocity.
    """
    base_twist: Any = None
    joint_position_target: Any = None
    limited: bool = False
    violations: tuple = ()
    timestamp: float = 0.0
    frame: str = COURT_FRAME

    def __post_init__(self) -> None:
        check_court_frame(self)
        self.timestamp = _check_timestamp(self.timestamp, 'SafeCommand')
        self.base_twist = check_batched('SafeCommand.base_twist', self.base_twist, 3)
        self.joint_position_target = check_batched('SafeCommand.joint_position_target',
                                                   self.joint_position_target, 6)


@dataclass
class Feedback:
    """Closed-loop return path (S8): prediction vs reality + dataset record handles."""
    timestamp: float = 0.0
    frame: str = COURT_FRAME
    # Prediction residual: predicted shuttle position/velocity minus the measured one
    # (metres / metres per second).  Consumed by the adaptation layer.
    prediction_error: Any = None
    contact_detected: Any = None
    # Execution tracking residual (DEC-015): how far the executed command stayed from the
    # requested one.  Kept separate so the two meanings cannot be confused again.
    tracking_residual: Any = None

    def __post_init__(self) -> None:
        check_court_frame(self)
        self.timestamp = _check_timestamp(self.timestamp, 'Feedback')
        if self.tracking_residual is not None:
            self.tracking_residual = check_per_env_scalar(
                'Feedback.tracking_residual', self.tracking_residual)


__all__ = ["BrainBoundaryError", "Layer", "COURT_FRAME", "BASELINE_ORDER",
           "ShuttleMeasurement", "RobotSensorState", "UnifiedState", "PredictedTrajectory",
           "HitDecision", "BestIntercept", "WholeBodyTarget", "SafeCommand", "Feedback",
           "check_batched", "check_court_frame"]
