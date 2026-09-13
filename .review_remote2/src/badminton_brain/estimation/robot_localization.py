# -*- coding: utf-8 -*-
"""T2 - robot localization: 6-D EKF over wheel odometry + IMU + absolute pose (pure numpy).

Layer: estimation (ROBOT_BRAIN.md S11/S12).  Court frame only (COORDINATE_SYSTEM.md S3/S28).

State (COORDINATE_SYSTEM.md S5.2, SI units):
    x = [x, y, yaw, vx, vy, wz]
    x, y   [m]      robot_base origin position in the court frame (net centre below / ground)
    yaw    [rad]    base yaw, robot +X measured from court +X towards court +Y, kept in (-pi, pi]
    vx, vy [m/s]    body-frame linear velocity of robot_base (+X forward, +Y left)
    vy     body-frame velocity towards the robot +Y (left) axis
    wz     [rad/s]  body-frame yaw rate

Measurements
    predict(dt, odom, imu)                odom = per-step increments [dx, dy, dyaw] measured in
                                          robot_base (wheel odometry); imu column 0 = gyro yaw rate
                                          [rad/s].  The gyro is the yaw-rate source whenever it is
                                          present, the wheel yaw increment is the fallback.
    update_pose_measurement(pose, cov)    absolute (visual / landmark) pose measurement
                                          [x, y, yaw] in the court frame with its covariance.

Output (frozen message contract in badminton_brain/types.py)
    base_pose()  -> (N, 7)  = [x, y, z, qx, qy, qz, qw]; quaternion order x,y,z,w per
                             COORDINATE_SYSTEM.md S2.4; planar yaw only (qx = qy = 0)
    base_twist() -> (N, 6)  = [vx, vy, vz, wx, wy, wz], planar base -> vz = wx = wy = 0

Authenticity (BADMINTON_ROBOT.md S4/S12): every numeric parameter is a Param.  Sensor noise is not
identified on the real Morph One yet, so the runnable values are TEMP_PARAMETERIZED_PROXY with an
explicit source, and the identification requirement is carried as REQUIRES_MEASUREMENT entries.
reset(env_ids) (S29) only ever touches the selected environments.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Optional, Sequence, Tuple

import numpy as np

from ..status import AssetStatus, Param
from ..types import BrainBoundaryError

STATE_NAMES: Tuple[str, ...] = ('x', 'y', 'yaw', 'vx', 'vy', 'wz')
STATE_DIM = 6
POSE_DIM = 3
INPUT_DIM = 3


def wrap_angle(angle: Any) -> np.ndarray:
    """Wrap an angle (scalar or array) into [-pi, pi)."""
    return (np.asarray(angle, dtype=float) + np.pi) % (2.0 * np.pi) - np.pi


@dataclass(frozen=True)
class LocalizationNoiseConfig:
    """Noise description of the localizer inputs.  No number may hide its origin (S4/S12)."""

    odom_translation_noise_m: Param = field(default_factory=lambda: Param(
        0.002, AssetStatus.TEMP_PARAMETERIZED_PROXY,
        "temporary simulation proxy: 1-sigma wheel-odometry translation noise per 10 ms step; "
        "the real Morph One odometry noise has not been identified (see "
        "odom_translation_noise_measured_m)"))
    odom_yaw_noise_rad: Param = field(default_factory=lambda: Param(
        0.002, AssetStatus.TEMP_PARAMETERIZED_PROXY,
        "temporary simulation proxy: 1-sigma wheel-odometry yaw-increment noise per 10 ms step; "
        "used only when no IMU yaw rate is supplied"))
    gyro_noise_rad_s: Param = field(default_factory=lambda: Param(
        0.005, AssetStatus.TEMP_PARAMETERIZED_PROXY,
        "temporary simulation proxy: 1-sigma IMU gyro yaw-rate noise [rad/s]"))
    accel_noise_m_s2: Param = field(default_factory=lambda: Param(
        0.5, AssetStatus.TEMP_PARAMETERIZED_PROXY,
        "temporary simulation proxy: 1-sigma unmodelled body acceleration [m/s^2] that drives the "
        "continuously-white-acceleration part of Q (omni base slipping, pushing, unmodelled forces)"))
    yaw_accel_noise_rad_s2: Param = field(default_factory=lambda: Param(
        0.5, AssetStatus.TEMP_PARAMETERIZED_PROXY,
        "temporary simulation proxy: 1-sigma unmodelled yaw angular acceleration [rad/s^2]"))
    odom_translation_noise_measured_m: Param = field(default_factory=lambda: Param(
        None, AssetStatus.REQUIRES_MEASUREMENT,
        "Morph One wheel-odometry translation noise must be identified on the real robot "
        "(BADMINTON_ROBOT.md S4): drive a known straight line and compare against the court "
        "measurement system"))
    gyro_noise_measured_rad_s: Param = field(default_factory=lambda: Param(
        None, AssetStatus.REQUIRES_MEASUREMENT,
        "IMU gyro noise density / Allan deviation of the Morph One IMU is not measured yet"))

    def entries(self) -> Dict[str, Param]:
        return {name: getattr(self, name) for name in self.__dataclass_fields__}


@dataclass(frozen=True)
class LocalizationConfig:
    """Fixed configuration of the localizer (scene / experiment layer, never a mechanical constant)."""

    home_pose_xyyaw: Param = field(default_factory=lambda: Param(
        (-1.60, 0.0, 0.0), AssetStatus.TRACEABLE_REFERENCE,
        "BADMINTON_ROBOT.md S31 engineering baseline (scene/experiment config layer); "
        "overridden per experiment, never hardcoded in the robot model"))
    base_height_m: Param = field(default_factory=lambda: Param(
        0.0, AssetStatus.TRACEABLE_REFERENCE,
        "COORDINATE_SYSTEM.md S3.6 ground plane z = 0 and robot_base origin on the ground; a planar "
        "base therefore reports z = 0 in the court frame"))
    initial_covariance: Param = field(default_factory=lambda: Param(
        np.diag([0.25, 0.25, 0.05, 0.25, 0.25, 0.05]), AssetStatus.TEMP_PARAMETERIZED_PROXY,
        "temporary simulation proxy for the start-up uncertainty at the home pose; the real "
        "start-up covariance (homing repeatability) is not measured (REQUIRES_MEASUREMENT)"))
    noise: LocalizationNoiseConfig = field(default_factory=LocalizationNoiseConfig)

    def entries(self) -> Dict[str, Param]:
        out = {'home_pose_xyyaw': self.home_pose_xyyaw,
               'base_height_m': self.base_height_m,
               'initial_covariance': self.initial_covariance}
        out.update(self.noise.entries())
        return out


class RobotLocalization:
    """Stateful batched EKF: [x, y, yaw, vx, vy, wz] per environment, court frame.

    Design notes
      * The odometry increment is the control input; the velocity block of the transition Jacobian
        is therefore zero (velocity states follow the odometry/IMU input directly) and the input
        noise is mapped through G.  Unmodelled acceleration enters Q as a
        continuously-white-acceleration term, so the pose covariance still grows between updates.
      * The pose measurement Jacobian is the constant selection H = [I3 | 0], so the measurement is
        linear and no relinearisation loop is needed; the yaw innovation is wrapped.
      * Every environment is independent (per-env dt, per-env noise draw); nothing couples envs and
        reset() selects rows, so reset isolation is exact.
    """

    name = 'robot_localization'
    is_implemented = True

    def __init__(self, num_envs: int, config: Optional[LocalizationConfig] = None) -> None:
        num_envs = int(num_envs)
        if num_envs < 1:
            raise BrainBoundaryError(f"num_envs must be >= 1, got {num_envs}")
        self.num_envs = num_envs
        self.config = config or LocalizationConfig()
        self._home_pose = self._resolve_vector('home_pose_xyyaw', self.config.home_pose_xyyaw, POSE_DIM)
        self._base_height = float(self._resolve_scalar('base_height_m', self.config.base_height_m))
        self._initial_covariance = self._resolve_covariance(self.config.initial_covariance)
        self._noise = {
            'odom_translation_noise_m': self._resolve_scalar(
                'odom_translation_noise_m', self.config.noise.odom_translation_noise_m, positive=True),
            'odom_yaw_noise_rad': self._resolve_scalar(
                'odom_yaw_noise_rad', self.config.noise.odom_yaw_noise_rad, positive=True),
            'gyro_noise_rad_s': self._resolve_scalar(
                'gyro_noise_rad_s', self.config.noise.gyro_noise_rad_s, positive=True),
            'accel_noise_m_s2': self._resolve_scalar(
                'accel_noise_m_s2', self.config.noise.accel_noise_m_s2, positive=True),
            'yaw_accel_noise_rad_s2': self._resolve_scalar(
                'yaw_accel_noise_rad_s2', self.config.noise.yaw_accel_noise_rad_s2, positive=True),
        }
        self._state = np.zeros((num_envs, STATE_DIM), dtype=float)
        self._state[:, :POSE_DIM] = self._home_pose
        self._covariance = np.tile(self._initial_covariance, (num_envs, 1, 1))

    # ------------------------------------------------------------------ parameters

    @staticmethod
    def _resolve_scalar(name: str, param: Param, positive: bool = False) -> float:
        if not isinstance(param, Param):
            raise BrainBoundaryError(f"{name} must be a Param, got {type(param).__name__}")
        value = param.value
        if value is None:
            raise BrainBoundaryError(
                f"{name} has no value (status={param.status.value}); supply a value with a declared "
                "origin (BADMINTON_ROBOT.md S4/S12)")
        value = float(value)
        if not np.isfinite(value) or (positive and value <= 0.0):
            raise BrainBoundaryError(f"{name} must be finite and positive, got {value}")
        return value

    @staticmethod
    def _resolve_vector(name: str, param: Param, dim: int) -> np.ndarray:
        if not isinstance(param, Param):
            raise BrainBoundaryError(f"{name} must be a Param, got {type(param).__name__}")
        if param.value is None:
            raise BrainBoundaryError(f"{name} has no value (status={param.status.value})")
        out = np.asarray(param.value, dtype=float).reshape(-1)
        if out.shape != (dim,) or not np.all(np.isfinite(out)):
            raise BrainBoundaryError(f"{name} must be {dim} finite numbers, got {param.value!r}")
        return out

    @staticmethod
    def _resolve_covariance(param: Param) -> np.ndarray:
        if not isinstance(param, Param):
            raise BrainBoundaryError(f"initial_covariance must be a Param, got {type(param).__name__}")
        if param.value is None:
            raise BrainBoundaryError(f"initial_covariance has no value (status={param.status.value})")
        cov = np.asarray(param.value, dtype=float)
        if cov.shape != (STATE_DIM, STATE_DIM):
            raise BrainBoundaryError(f"initial_covariance must be ({STATE_DIM}, {STATE_DIM})")
        return RobotLocalization._symmetrize_checked('initial_covariance', cov)

    def parameters(self) -> Dict[str, Param]:
        """Every numeric parameter this module uses, with its authenticity status (S4/S12)."""
        return dict(self.config.entries())

    # ------------------------------------------------------------------ state access

    @property
    def state(self) -> np.ndarray:
        """(N, 6) snapshot of [x, y, yaw, vx, vy, wz]."""
        return self._state.copy()

    @property
    def covariance(self) -> np.ndarray:
        """(N, 6, 6) snapshot of the state covariance."""
        return self._covariance.copy()

    def pose_xyyaw(self) -> np.ndarray:
        """(N, 3) = [x, y, yaw] in the court frame."""
        return self._state[:, :POSE_DIM].copy()

    def base_pose(self) -> np.ndarray:
        """(N, 7) = [x, y, z, qx, qy, qz, qw] in the court frame (types.UnifiedState.base_pose)."""
        yaw = self._state[:, 2]
        pose = np.zeros((self.num_envs, 7), dtype=float)
        pose[:, 0:2] = self._state[:, 0:2]
        pose[:, 2] = self._base_height
        pose[:, 5] = np.sin(0.5 * yaw)          # qz (planar rotation about court +Z)
        pose[:, 6] = np.cos(0.5 * yaw)          # qw
        return pose

    def base_twist(self) -> np.ndarray:
        """(N, 6) = [vx, vy, vz, wx, wy, wz] body twist (types.UnifiedState.base_twist)."""
        twist = np.zeros((self.num_envs, 6), dtype=float)
        twist[:, 0] = self._state[:, 3]
        twist[:, 1] = self._state[:, 4]
        twist[:, 5] = self._state[:, 5]
        return twist

    # ------------------------------------------------------------------ validation

    def _check_dt(self, dt: Any) -> np.ndarray:
        arr = np.asarray(dt, dtype=float)
        if arr.ndim == 0:
            arr = np.full((self.num_envs,), float(arr))
        if arr.shape != (self.num_envs,):
            raise BrainBoundaryError(
                f"dt must be a scalar or ({self.num_envs},), got shape {np.shape(dt)}")
        if not np.all(np.isfinite(arr)) or np.any(arr <= 0.0):
            raise BrainBoundaryError(f"dt must be finite and positive, got {np.shape(dt)} of values")
        return arr

    def _check_batched(self, name: str, value: Any, last_dim: int) -> np.ndarray:
        arr = np.asarray(value, dtype=float)
        if arr.ndim != 2 or arr.shape[0] != self.num_envs or arr.shape[1] != last_dim:
            raise BrainBoundaryError(
                f"{name} must be ({self.num_envs}, {last_dim}), got shape {arr.shape}")
        if not np.all(np.isfinite(arr)):
            raise BrainBoundaryError(f"{name} contains NaN/Inf")
        return arr

    def _check_imu(self, imu: Any) -> Optional[np.ndarray]:
        if imu is None:
            return None
        arr = np.asarray(imu, dtype=float)
        if arr.ndim == 1:
            arr = arr.reshape(-1, 1)
        if arr.ndim != 2 or arr.shape[0] != self.num_envs or arr.shape[1] < 1:
            raise BrainBoundaryError(
                f"imu must be ({self.num_envs}, k>=1) with column 0 = yaw rate, got shape {arr.shape}")
        if not np.all(np.isfinite(arr)):
            raise BrainBoundaryError("imu contains NaN/Inf")
        return arr[:, 0]

    @staticmethod
    def _symmetrize_checked(name: str, cov: np.ndarray) -> np.ndarray:
        symmetric = 0.5 * (cov + np.swapaxes(cov, -1, -2))
        asym = float(np.max(np.abs(symmetric - cov))) if cov.size else 0.0
        if asym > 1e-6 * max(1.0, float(np.max(np.abs(symmetric)))):
            raise BrainBoundaryError(f"{name} must be symmetric (max asymmetry {asym:.3e})")
        if not np.all(np.isfinite(symmetric)):
            raise BrainBoundaryError(f"{name} contains NaN/Inf")
        min_eig = float(np.min(np.linalg.eigvalsh(symmetric)))
        if min_eig < -1e-9 * max(1.0, float(np.max(np.abs(symmetric)))):
            raise BrainBoundaryError(f"{name} must be positive semidefinite (min eigenvalue {min_eig:.3e})")
        return symmetric

    def _check_env_ids(self, env_ids: Any) -> Optional[np.ndarray]:
        if env_ids is None:
            return None
        ids = np.asarray(list(env_ids), dtype=int).reshape(-1)
        if ids.size and (int(ids.min()) < 0 or int(ids.max()) >= self.num_envs):
            raise BrainBoundaryError(
                f"env_ids must be within [0, {self.num_envs}), got {ids.tolist()}")
        return ids

    # ------------------------------------------------------------------ transitions

    def predict(self, dt: Any, odom: Any, imu: Any = None) -> None:
        """Propagate the state with one odometry increment (and IMU yaw rate when supplied).

        Parameters
        ----------
        dt   : float or (N,) seconds since the previous prediction
        odom : (N, 3) measured increments [dx, dy, dyaw] in robot_base (wheel odometry)
        imu  : None or (N, k>=1); column 0 is the gyro yaw rate [rad/s].  When given it replaces the
               wheel yaw increment as the yaw-rate source.
        """
        dt_col = self._check_dt(dt)
        odom_arr = self._check_batched('odom', odom, INPUT_DIM)
        wz_imu = self._check_imu(imu)

        sigma_v = self._noise['odom_translation_noise_m'] / dt_col
        if wz_imu is None:
            wz_u = odom_arr[:, 2] / dt_col
            sigma_wz = self._noise['odom_yaw_noise_rad'] / dt_col
        else:
            wz_u = wz_imu
            sigma_wz = np.full((self.num_envs,), self._noise['gyro_noise_rad_s'])
        vx_u = odom_arr[:, 0] / dt_col
        vy_u = odom_arr[:, 1] / dt_col

        yaw = self._state[:, 2]
        c, s = np.cos(yaw), np.sin(yaw)
        body_x = vx_u * c - vy_u * s
        body_y = vx_u * s + vy_u * c

        state = self._state
        state[:, 0] += body_x * dt_col
        state[:, 1] += body_y * dt_col
        state[:, 2] = wrap_angle(yaw + wz_u * dt_col)
        state[:, 3] = vx_u
        state[:, 4] = vy_u
        state[:, 5] = wz_u

        # transition Jacobian: pose rows carry the yaw dependence, velocity rows are driven by the input
        F = np.zeros((self.num_envs, STATE_DIM, STATE_DIM), dtype=float)
        F[:, 0, 0] = 1.0
        F[:, 1, 1] = 1.0
        F[:, 2, 2] = 1.0
        F[:, 0, 2] = -body_y * dt_col
        F[:, 1, 2] = body_x * dt_col

        # input noise mapped through G = df/du
        G = np.zeros((self.num_envs, STATE_DIM, INPUT_DIM), dtype=float)
        G[:, 0, 0] = c * dt_col
        G[:, 0, 1] = -s * dt_col
        G[:, 1, 0] = s * dt_col
        G[:, 1, 1] = c * dt_col
        G[:, 2, 2] = dt_col
        G[:, 3, 0] = 1.0
        G[:, 4, 1] = 1.0
        G[:, 5, 2] = 1.0
        u_cov = np.zeros((self.num_envs, INPUT_DIM, INPUT_DIM), dtype=float)
        u_cov[:, 0, 0] = sigma_v ** 2
        u_cov[:, 1, 1] = sigma_v ** 2
        u_cov[:, 2, 2] = sigma_wz ** 2
        Q = G @ u_cov @ np.swapaxes(G, 1, 2)

        # unmodelled acceleration: continuous white acceleration discretised over dt
        sigma_a = self._noise['accel_noise_m_s2']
        sigma_alpha = self._noise['yaw_accel_noise_rad_s2']
        half_dt2 = 0.5 * dt_col ** 2
        Q[:, 0, 0] += (sigma_a * half_dt2) ** 2
        Q[:, 1, 1] += (sigma_a * half_dt2) ** 2
        Q[:, 2, 2] += (sigma_alpha * half_dt2) ** 2
        Q[:, 3, 3] += (sigma_a * dt_col) ** 2
        Q[:, 4, 4] += (sigma_a * dt_col) ** 2
        Q[:, 5, 5] += (sigma_alpha * dt_col) ** 2

        P = self._covariance
        P_pred = F @ P @ np.swapaxes(F, 1, 2) + Q
        self._covariance = 0.5 * (P_pred + np.swapaxes(P_pred, 1, 2))

    def update_pose_measurement(self, pose_xyyaw: Any, covariance: Any) -> None:
        """Fuse an absolute court-frame pose measurement [x, y, yaw] with a (N,3,3) covariance."""
        measurement = self._check_batched('pose_xyyaw', pose_xyyaw, POSE_DIM)
        cov = np.asarray(covariance, dtype=float)
        if cov.ndim == 2:
            cov = np.tile(cov, (self.num_envs, 1, 1))
        if cov.shape != (self.num_envs, POSE_DIM, POSE_DIM):
            raise BrainBoundaryError(
                f"covariance must be (3, 3) or ({self.num_envs}, 3, 3), got shape {np.shape(covariance)}")
        R = self._symmetrize_checked('pose measurement covariance', cov)

        innovation = measurement - self._state[:, :POSE_DIM]
        innovation[:, 2] = wrap_angle(innovation[:, 2])

        P = self._covariance
        S = P[:, :POSE_DIM, :POSE_DIM] + R
        K = P[:, :, :POSE_DIM] @ np.linalg.inv(S)         # (N, 6, 3)

        self._state = self._state + (K @ innovation[:, :, None])[:, :, 0]
        self._state[:, 2] = wrap_angle(self._state[:, 2])

        H = np.zeros((POSE_DIM, STATE_DIM), dtype=float)
        H[0, 0] = H[1, 1] = H[2, 2] = 1.0
        A = np.eye(STATE_DIM) - K @ H                     # (N, 6, 6)
        P_new = A @ P @ np.swapaxes(A, 1, 2) + K @ R @ np.swapaxes(K, 1, 2)
        self._covariance = 0.5 * (P_new + np.swapaxes(P_new, 1, 2))

    # ------------------------------------------------------------------ reset / init

    def initialize(self, pose_xyyaw: Any, covariance: Any = None, env_ids: Any = None) -> None:
        """Set the selected environments to a known pose (velocity zeroed) and covariance."""
        ids = self._check_env_ids(env_ids)
        rows = np.arange(self.num_envs) if ids is None else ids
        if rows.size == 0:
            return
        pose = np.asarray(pose_xyyaw, dtype=float)
        if pose.ndim == 1:
            pose = np.tile(pose.reshape(1, POSE_DIM), (rows.size, 1))
        if pose.shape != (rows.size, POSE_DIM) or not np.all(np.isfinite(pose)):
            raise BrainBoundaryError(
                f"pose_xyyaw must be ({rows.size}, 3) finite, got shape {np.shape(pose_xyyaw)}")
        if covariance is None:
            cov = np.tile(self._initial_covariance, (rows.size, 1, 1))
        else:
            cov = np.asarray(covariance, dtype=float)
            if cov.ndim == 2:
                cov = np.tile(cov, (rows.size, 1, 1))
            if cov.shape != (rows.size, STATE_DIM, STATE_DIM):
                raise BrainBoundaryError(
                    f"covariance must be (6, 6) or ({rows.size}, 6, 6), got shape {np.shape(covariance)}")
            cov = self._symmetrize_checked('covariance', cov)

        self._state[rows, 0] = pose[:, 0]
        self._state[rows, 1] = pose[:, 1]
        self._state[rows, 2] = wrap_angle(pose[:, 2])
        self._state[rows, 3:] = 0.0
        self._covariance[rows] = cov

    def reset(self, env_ids: Sequence[int] = None) -> None:
        """S29: reset only the selected environments; every other row stays bit-identical."""
        ids = self._check_env_ids(env_ids)
        rows = np.arange(self.num_envs) if ids is None else ids
        if rows.size == 0:
            return
        home = np.tile(self._home_pose, (rows.size, 1))
        cov = np.tile(self._initial_covariance, (rows.size, 1, 1))
        self._state[rows, 0] = home[:, 0]
        self._state[rows, 1] = home[:, 1]
        self._state[rows, 2] = home[:, 2]
        self._state[rows, 3:] = 0.0
        self._covariance[rows] = cov


__all__ = ["RobotLocalization", "LocalizationConfig", "LocalizationNoiseConfig", "wrap_angle",
           "STATE_NAMES", "STATE_DIM"]
