# -*- coding: utf-8 -*-
"""P7-05: a shuttlecock EKF that is DEFINED in the image plane first.

Requirement (master spec P7-05/P7-06)
------------------------------------
P7-05 asks for an EKF/sequential-filter state definition with explicit process
and measurement noise and outlier rejection, and P7-06 says the 3D
position/velocity/covariance state may only be entered when trustworthy depth
and a calibration exist: a 2D bbox is never 3D.  The existing filter in this
package (estimation/shuttle_ukf.py) is a *3D court-frame* UKF with an augmented
drag state; it cannot run at all without a stereo point, so it cannot serve as
the 2D-first estimator.  This module adds the missing mode:

    MODE_IMAGE_2D   x = [u, v, u_dot, v_dot]          px, px/s     (4)
    MODE_COURT_3D   x = [px, py, pz, vx, vy, vz]      m,  m/s      (6)

The filter starts in MODE_IMAGE_2D, consumes only DetectionObservation values
(badminton_brain.perception... see perception/shuttle_detection/detection_output.py),
and can only reach MODE_COURT_3D through upgrade_to_3d(), which refuses to run
without (a) a depth measurement from a *trustworthy* source and (b) a
*calibrated* camera model.  There is no other way to obtain a 3D state from this
class, so "2D presented as 3D" is not merely discouraged, it is unreachable.

Noise parameters (all declared as Param, nothing hidden)
-------------------------------------------------------
Process noise uses the piecewise-constant-acceleration discretisation
(Bar-Shalom, Li & Kirubarajan, "Estimation with Applications to Tracking and
Navigation", 2001, section 6.2.2):

    Q_axis(dt) = a^2 * [[dt^4/4, dt^3/2], [dt^3/2, dt^2]]

with a the per-axis acceleration std.  The values are engineering proxies,
justified numerically in their Param source and in
management/morph_one_rs/P7_DETECTION_EKF_INTERFACES.md; the *measured*
counterparts exist as Param(None, REQUIRES_MEASUREMENT) and are what a real
characterisation would have to fill in.

Behaviour when measurements stop arriving (documented, tested)
--------------------------------------------------------------
Calling predict(dt) without a following update is prediction-only: the state
follows the constant-velocity model exactly and the covariance grows with the Q
above, so the next gate widens.  After MAX_COAST_STEPS consecutive predictions
without an accepted update the tracker reports status "LOST"; it never invents a
measurement and never resets itself silently.

Run the tests with:
    python -m pytest tests/badminton_brain/test_shuttle_state_ekf.py -q
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Dict, Optional

import numpy as np

from ..status import AssetStatus, Param

__all__ = [
    "MODE_IMAGE_2D", "MODE_COURT_3D", "TRACKING", "COASTING", "LOST",
    "DEPTH_SOURCES", "TRUSTWORTHY_DEPTH_SOURCES", "CALIBRATION_STATES",
    "PROCESS_ACCEL_STD_PX_S2", "MEASUREMENT_STD_PX", "MEASUREMENT_STD_PX_MEASURED",
    "PROCESS_ACCEL_STD_MP_S2", "MEASUREMENT_STD_M", "MEASUREMENT_STD_M_MEASURED",
    "INITIAL_POSITION_STD_PX", "INITIAL_VELOCITY_STD_PX_S",
    "INITIAL_POSITION_STD_M", "INITIAL_VELOCITY_STD_MPS",
    "GATE_CHI2_2DOF_099", "GATE_CHI2_3DOF_099", "MAX_COAST_STEPS",
    "OUTLIER_STREAK_REINIT", "DEPTH_RATE_STD_PROXY_MPS",
    "ShuttleEKFError", "UncalibratedCameraError", "DepthNotTrustworthyError",
    "StateModeError", "CourtCameraModel", "UpdateResult", "UpgradeResult",
    "ShuttleStateEKF",
]

MODE_IMAGE_2D = "IMAGE_PLANE_2D"
MODE_COURT_3D = "COURT_3D"

TRACKING = "TRACKING"
COASTING = "COASTING"
LOST = "LOST"

# Depth provenance vocabulary.  Only the first entries carry real parallax; a
# monocular depth *guess* is named here only so that it can be refused by name.
DEPTH_SOURCES = ("STEREO_TRIANGULATED", "STRUCTURED_LIGHT", "LIDAR", "MONOCULAR_GUESS")
TRUSTWORTHY_DEPTH_SOURCES = ("STEREO_TRIANGULATED", "STRUCTURED_LIGHT", "LIDAR")

# Camera calibration provenance.  Only CALIBRATED may open the 3D state.
CALIBRATION_STATES = ("CALIBRATED", "TEMP_PROXY", "UNCALIBRATED")

# ---------------------------------------------------------------------------
# Declared parameters
# ---------------------------------------------------------------------------
PROCESS_ACCEL_STD_PX_S2 = Param(
    2000.0, AssetStatus.TEMP_PARAMETERIZED_PROXY,
    "TEMP per-axis image-plane acceleration std [px/s^2] for the constant-velocity model. "
    "Justification (order of magnitude, not a measurement): the frozen quadratic-drag model "
    "(trajectory/shuttle_aerodynamics.py) decelerates a shuttle by k*|v|^2 with "
    "k = 1/6.5 1/m; at a representative 4.5 m/s that is 3.1 m/s^2, and a 1280x720 rig camera "
    "with fx ~ 700 px (engineering baseline in perception/stereo_geometry.py) sees roughly "
    "700 px/m, so 3.1 m/s^2 ~ 2200 px/s^2.  2000 px/s^2 is that magnitude, rounded down. "
    "Perspective magnification near the image border makes the true image acceleration larger, "
    "so this proxy is optimistic there; replace with a measured innovation statistic.")

MEASUREMENT_STD_PX = Param(
    2.0, AssetStatus.TEMP_PARAMETERIZED_PROXY,
    "TEMP bbox-centre measurement std [px].  Justification: the small-target bin of this "
    "project is 4-6 px (management/shuttle_detection/closure_questions_interim.md, "
    "the 4-6 px row of the size-bucket table), so 2.0 px is about one third of the upper "
    "extent of that bin - a plausible centre jitter for a target only a few pixels wide. "
    "It is a proxy: no bbox jitter measurement exists for our rig.  Callers that have a "
    "per-detection pixel_covariance must pass it instead; that value wins.")

MEASUREMENT_STD_PX_MEASURED = Param(
    None, AssetStatus.REQUIRES_MEASUREMENT,
    "the real bbox-centre jitter of the deployed detector on our rig (repeated static-target "
    "frames, or annotation-residual statistics per size bucket) has never been measured. "
    "Filling this in is the only thing that may replace MEASUREMENT_STD_PX.")

PROCESS_ACCEL_STD_MP_S2 = Param(
    3.0, AssetStatus.TEMP_PARAMETERIZED_PROXY,
    "TEMP per-axis acceleration std [m/s^2] for COURT_3D.  Justification: the same drag "
    "deceleration bound as above, k*|v|^2 = (1/6.5)*4.5^2 = 3.1 m/s^2, so 3 m/s^2 covers the "
    "whole unmodelled drag range of a slow rally shuttle without an explicit drag state. "
    "shuttle_ukf.py uses 1.0 m/s^2 *plus* an augmented drag state; this filter has no drag "
    "state, hence the larger number.  Proxy, not a measurement.")

MEASUREMENT_STD_M = Param(
    None, AssetStatus.REQUIRES_MEASUREMENT,
    "stereo/triangulation noise in metres is not characterised in this repository (identical "
    "status to POSITION_MEASUREMENT_STD in estimation/shuttle_ukf.py).  A COURT_3D update "
    "therefore REFUSES to invent R: the caller must pass measurement_covariance from the "
    "depth/triangulation pipeline.")

MEASUREMENT_STD_M_MEASURED = Param(
    None, AssetStatus.REQUIRES_CALIBRATION,
    "needs the stereo calibration session plus a static-target residual study; cannot be "
    "derived from any file currently in the repository.")

INITIAL_POSITION_STD_PX = Param(
    8.0, AssetStatus.TEMP_PARAMETERIZED_PROXY,
    "TEMP initial image-plane position std [px] when initialize() is given no covariance: "
    "twice the 4-6 px small-target bin, i.e. deliberately loose.")

INITIAL_VELOCITY_STD_PX_S = Param(
    600.0, AssetStatus.TEMP_PARAMETERIZED_PROXY,
    "TEMP initial image-plane velocity std [px/s]: a shuttle crossing the frame in ~1 s covers "
    "~1280 px, so 600 px/s is a loose bound on the first-difference velocity.")

INITIAL_POSITION_STD_M = Param(
    0.1, AssetStatus.TEMP_PARAMETERIZED_PROXY,
    "TEMP initial court-frame position std [m] used only by the 3D upgrade when the caller "
    "supplies no position covariance; the stereo error itself is REQUIRES_MEASUREMENT.")

INITIAL_VELOCITY_STD_MPS = Param(
    1.0, AssetStatus.TEMP_PARAMETERIZED_PROXY,
    "TEMP initial court-frame velocity std [m/s] for the 3D upgrade (same order as "
    "INITIAL_VELOCITY_STD in shuttle_ukf.py).")

# Two-sided 99% chi-square quantiles with 2 and 3 degrees of freedom.
GATE_CHI2_2DOF_099 = Param(
    9.2103, AssetStatus.TRACEABLE_REFERENCE,
    "chi-square(2) 0.99 quantile (standard table), used as the 2D Mahalanobis gate; 1% of "
    "inlier measurements are expected to be dropped by this gate by construction.")

GATE_CHI2_3DOF_099 = Param(
    11.3449, AssetStatus.TRACEABLE_REFERENCE,
    "chi-square(3) 0.99 quantile (standard table), used as the 3D Mahalanobis gate.")

MAX_COAST_STEPS = Param(
    5, AssetStatus.TEMP_PARAMETERIZED_PROXY,
    "TEMP coasting allowance: 5 frames at the 50 Hz stereo rate used by the existing tests "
    "(dt = 0.02 s) = 0.1 s.  Beyond that the tracker reports LOST instead of pretending to "
    "know the shuttle.  Derived from the sampling rate, not measured on the real pipeline.")

OUTLIER_STREAK_REINIT = Param(
    3, AssetStatus.TEMP_PARAMETERIZED_PROXY,
    "TEMP re-initialisation threshold: 3 consecutive gated rejections are read as 'the "
    "predictions are wrong, not the measurements' and the tracker re-initialises from the "
    "latest measurement (and says so in the UpdateResult).  Chosen with the same 0.1 s "
    "coasting budget as MAX_COAST_STEPS.")

DEPTH_RATE_STD_PROXY_MPS = Param(
    2.0, AssetStatus.TEMP_PARAMETERIZED_PROXY,
    "TEMP depth-rate std [m/s] used by the 3D upgrade when the caller supplies a depth but no "
    "depth rate: without an observed d/dt the court-frame velocity cannot be upgraded "
    "honestly, so the missing rate is declared with this proxy and inflates the velocity "
    "covariance.  Pass depth_rate_mps/depth_rate_sigma_mps to remove the proxy.")


class ShuttleEKFError(RuntimeError):
    """Base class of the tracker's contract violations."""


class UncalibratedCameraError(ShuttleEKFError):
    """A 3D upgrade was attempted with a camera whose calibration is not CALIBRATED."""


class DepthNotTrustworthyError(ShuttleEKFError):
    """A 3D upgrade was attempted with a depth source that cannot support 3D."""


class StateModeError(ShuttleEKFError):
    """A measurement does not match the filter mode (2D observation into 3D, etc.)."""


@dataclass(frozen=True)
class CourtCameraModel:
    """Pinhole camera in the court frame, with its own calibration provenance.

    K            3x3 intrinsics (fx, fy, cx, cy, 0 skew)
    T_court_camera  rigid 4x4, p_court = T @ p_camera (OpenCV axes: +Z optical)
    image_size   (width_px, height_px)
    calibration  one of CALIBRATION_STATES
    """

    K: Any
    T_court_camera: Any
    image_size: tuple
    calibration: str = "UNCALIBRATED"

    def __post_init__(self) -> None:
        K = np.asarray(self.K, dtype=float)
        if K.shape != (3, 3) or not np.all(np.isfinite(K)):
            raise ShuttleEKFError("CourtCameraModel.K must be a finite 3x3 matrix")
        if K[0, 0] <= 0.0 or K[1, 1] <= 0.0:
            raise ShuttleEKFError("CourtCameraModel.K focal lengths must be positive")
        if abs(float(K[2, 0])) > 0.0 or abs(float(K[2, 1])) > 0.0 or abs(float(K[2, 2]) - 1.0) > 0.0:
            raise ShuttleEKFError("CourtCameraModel.K last row must be [0, 0, 1]")
        T = np.asarray(self.T_court_camera, dtype=float)
        if T.shape != (4, 4) or not np.all(np.isfinite(T)):
            raise ShuttleEKFError("CourtCameraModel.T_court_camera must be a finite 4x4 matrix")
        if not np.allclose(T[3], (0.0, 0.0, 0.0, 1.0), atol=1e-12):
            raise ShuttleEKFError("CourtCameraModel.T_court_camera last row must be [0,0,0,1]")
        R = T[:3, :3]
        if not np.allclose(R.T @ R, np.eye(3), atol=1e-9):
            raise ShuttleEKFError("CourtCameraModel.T_court_camera rotation is not orthonormal")
        size = tuple(int(v) for v in self.image_size)
        if len(size) != 2 or size[0] <= 1 or size[1] <= 1:
            raise ShuttleEKFError("CourtCameraModel.image_size must be (width_px, height_px) > 1")
        if self.calibration not in CALIBRATION_STATES:
            raise ShuttleEKFError(
                f"CourtCameraModel.calibration must be one of {CALIBRATION_STATES}, "
                f"got {self.calibration!r}")
        object.__setattr__(self, "K", K)
        object.__setattr__(self, "T_court_camera", T)
        object.__setattr__(self, "image_size", size)

    @property
    def is_calibrated(self) -> bool:
        return self.calibration == "CALIBRATED"


@dataclass(frozen=True)
class UpdateResult:
    """Outcome of one measurement update (never a silent no-op)."""

    accepted: bool
    reason: str            # 'UPDATED' | 'GATE' | 'REINIT'
    mahalanobis_d2: float
    nis: float             # == mahalanobis_d2 for a linear/Gaussian update
    gate_chi2: float
    steps_without_measurement: int
    status: str


@dataclass(frozen=True)
class UpgradeResult:
    """Outcome of a 2D -> 3D upgrade (what was upgraded and what was not)."""

    mode: str
    depth_source: str
    camera_calibration: str
    velocity_from_depth_rate: bool
    position_std_m: np.ndarray
    velocity_std_mps: np.ndarray


class ShuttleStateEKF:
    """Constant-velocity EKF with an image-plane (2D) base state and a gated 3D upgrade."""

    def __init__(
        self,
        mode: str = MODE_IMAGE_2D,
        *,
        dt_s: float = 0.02,
        process_accel_std: Optional[float] = None,
        measurement_std: Optional[float] = None,
        gate_chi2: Optional[float] = None,
        initial_position_std: Optional[float] = None,
        initial_velocity_std: Optional[float] = None,
        max_coast_steps: Optional[int] = None,
        outlier_streak_reinit: Optional[int] = None,
    ) -> None:
        if mode not in (MODE_IMAGE_2D, MODE_COURT_3D):
            raise ShuttleEKFError(f"mode must be {MODE_IMAGE_2D!r} or {MODE_COURT_3D!r}")
        dt = float(dt_s)
        if not math.isfinite(dt) or dt <= 0.0:
            raise ShuttleEKFError("dt_s must be a positive finite number")
        if mode == MODE_COURT_3D:
            raise ShuttleEKFError(
                "COURT_3D must be reached through upgrade_to_3d() from a 2D filter: a filter "
                "may not be constructed directly in 3D, because that is exactly how a 2D bbox "
                "would become a fake 3D state (P7-06)")
        self.mode = MODE_IMAGE_2D
        self.dt_s = dt
        self._accel_std = float(process_accel_std if process_accel_std is not None
                                else PROCESS_ACCEL_STD_PX_S2.value)
        self._meas_std = float(measurement_std if measurement_std is not None
                               else MEASUREMENT_STD_PX.value)
        self._gate_chi2 = float(gate_chi2 if gate_chi2 is not None
                                else GATE_CHI2_2DOF_099.value)
        self._initial_pos_std = float(initial_position_std
                                      if initial_position_std is not None
                                      else INITIAL_POSITION_STD_PX.value)
        self._initial_vel_std = float(initial_velocity_std
                                      if initial_velocity_std is not None
                                      else INITIAL_VELOCITY_STD_PX_S.value)
        self._max_coast = int(max_coast_steps if max_coast_steps is not None
                              else MAX_COAST_STEPS.value)
        self._reinit_streak = int(outlier_streak_reinit
                                  if outlier_streak_reinit is not None
                                  else OUTLIER_STREAK_REINIT.value)
        if self._accel_std <= 0.0 or self._meas_std <= 0.0:
            raise ShuttleEKFError("process/measurement std must be > 0")
        if self._max_coast < 1 or self._reinit_streak < 1:
            raise ShuttleEKFError("max_coast_steps and outlier_streak_reinit must be >= 1")
        self._x: Optional[np.ndarray] = None
        self._P: Optional[np.ndarray] = None
        self._t: Optional[float] = None
        self._misses = 0
        self._rejects = 0
        self._accepted = 0

    # -- state ------------------------------------------------------------
    @property
    def state_dim(self) -> int:
        return 4 if self.mode == MODE_IMAGE_2D else 6

    @property
    def position_dim(self) -> int:
        return 2 if self.mode == MODE_IMAGE_2D else 3

    @property
    def state(self) -> np.ndarray:
        if self._x is None:
            raise ShuttleEKFError("filter is not initialised")
        return self._x.copy()

    @property
    def covariance(self) -> np.ndarray:
        if self._P is None:
            raise ShuttleEKFError("filter is not initialised")
        return self._P.copy()

    @property
    def position(self) -> np.ndarray:
        return self.state[: self.position_dim]

    @property
    def velocity(self) -> np.ndarray:
        d = self.position_dim
        return self.state[d: 2 * d]

    @property
    def timestamp_s(self) -> Optional[float]:
        return self._t

    @property
    def steps_without_measurement(self) -> int:
        return self._misses

    @property
    def accepted_updates(self) -> int:
        return self._accepted

    @property
    def status(self) -> str:
        if self._x is None:
            return LOST
        if self._misses == 0:
            return TRACKING
        if self._misses <= self._max_coast:
            return COASTING
        return LOST

    @property
    def is_coasting(self) -> bool:
        return self.status == COASTING

    @property
    def is_lost(self) -> bool:
        return self.status == LOST

    def parameters(self) -> Dict[str, Param]:
        """Every number this instance runs on, with its declared status/source."""
        return {
            "process_accel_std": Param(
                self._accel_std, AssetStatus.TEMP_PARAMETERIZED_PROXY,
                "resolved from PROCESS_ACCEL_STD_PX_S2 / PROCESS_ACCEL_STD_MP_S2 or the "
                "constructor override"),
            "measurement_std": Param(
                self._meas_std, AssetStatus.TEMP_PARAMETERIZED_PROXY,
                "resolved from MEASUREMENT_STD_PX or the constructor override; a per-update "
                "measurement_covariance always wins over this default"),
            "gate_chi2": Param(self._gate_chi2, AssetStatus.TRACEABLE_REFERENCE,
                               "99% chi-square gate quantile for the current mode"),
            "max_coast_steps": Param(self._max_coast, AssetStatus.TEMP_PARAMETERIZED_PROXY,
                                     "MAX_COAST_STEPS or override"),
            "outlier_streak_reinit": Param(
                self._reinit_streak, AssetStatus.TEMP_PARAMETERIZED_PROXY,
                "OUTLIER_STREAK_REINIT or override"),
        }

    # -- model ------------------------------------------------------------
    def transition_matrix(self, dt_s: float) -> np.ndarray:
        dt = float(dt_s)
        if not math.isfinite(dt) or dt < 0.0:
            raise ShuttleEKFError("dt must be finite and >= 0")
        n = self.position_dim
        F = np.eye(2 * n)
        F[:n, n:] = np.eye(n) * dt
        return F

    def process_noise(self, dt_s: float) -> np.ndarray:
        """Piecewise-constant-acceleration Q (Bar-Shalom 6.2.2), per axis."""
        dt = float(dt_s)
        if not math.isfinite(dt) or dt < 0.0:
            raise ShuttleEKFError("dt must be finite and >= 0")
        n = self.position_dim
        a2 = self._accel_std ** 2
        block = a2 * np.array([[dt ** 4 / 4.0, dt ** 3 / 2.0],
                               [dt ** 3 / 2.0, dt ** 2]])
        Q = np.zeros((2 * n, 2 * n))
        for i in range(n):
            Q[np.ix_([i, n + i], [i, n + i])] = block
        return Q

    def measurement_matrix(self) -> np.ndarray:
        n = self.position_dim
        return np.hstack([np.eye(n), np.zeros((n, n))])

    def measurement_covariance(self) -> np.ndarray:
        """Default R: isotropic measurement_std^2 (TEMP proxy).

        In COURT_3D this raises: MEASUREMENT_STD_M is REQUIRES_MEASUREMENT, so a 3D update
        must receive the depth/triangulation covariance explicitly.
        """
        if self.mode == MODE_COURT_3D:
            raise ShuttleEKFError(
                "COURT_3D measurement noise is REQUIRES_MEASUREMENT: pass "
                "measurement_covariance from the depth/triangulation pipeline instead of "
                "inventing a metre-valued default")
        n = self.position_dim
        return np.eye(n) * self._meas_std ** 2

    # -- lifecycle --------------------------------------------------------
    def reset(self) -> None:
        self._x = None
        self._P = None
        self._t = None
        self._misses = 0
        self._rejects = 0
        self._accepted = 0
        self.mode = MODE_IMAGE_2D

    def _check_measurement(self, measurement: Any) -> np.ndarray:
        arr = np.asarray(measurement, dtype=float)
        n = self.position_dim
        if arr.shape != (n,):
            raise StateModeError(
                f"a {self.mode} filter needs a ({n},) measurement, got shape {arr.shape}"
                + (" (a 2D bbox cannot be fed to a 3D state, and vice versa)"))
        if not np.all(np.isfinite(arr)):
            raise ShuttleEKFError("measurement contains NaN/Inf")
        return arr

    def initialize(self, measurement: Any, timestamp_s: float,
                   *, velocity: Optional[Any] = None,
                   covariance: Optional[Any] = None) -> None:
        z = self._check_measurement(measurement)
        t = float(timestamp_s)
        if not math.isfinite(t):
            raise ShuttleEKFError("timestamp_s must be finite")
        n = self.position_dim
        if velocity is None:
            vel = np.zeros(n)
        else:
            vel = np.asarray(velocity, dtype=float)
            if vel.shape != (n,) or not np.all(np.isfinite(vel)):
                raise ShuttleEKFError(f"velocity must be a finite ({n},) vector")
        if covariance is None:
            P = np.diag([self._initial_pos_std ** 2] * n
                        + [self._initial_vel_std ** 2] * n)
        else:
            P = np.asarray(covariance, dtype=float)
            if P.shape != (2 * n, 2 * n) or not np.all(np.isfinite(P)):
                raise ShuttleEKFError(f"covariance must be a finite ({2 * n},{2 * n}) matrix")
            if float(np.min(np.linalg.eigvalsh(0.5 * (P + P.T)))) < -1e-12:
                raise ShuttleEKFError("covariance must be positive semi-definite")
            P = 0.5 * (P + P.T)
        self._x = np.concatenate([z, vel])
        self._P = P
        self._t = t
        self._misses = 0
        self._rejects = 0

    def predict(self, dt_s: Optional[float] = None) -> np.ndarray:
        """Prediction-only step: covariance grows, no measurement is invented."""
        if self._x is None:
            raise ShuttleEKFError("filter is not initialised")
        dt = self.dt_s if dt_s is None else float(dt_s)
        F = self.transition_matrix(dt)
        self._x = F @ self._x
        self._P = F @ self._P @ F.T + self.process_noise(dt)
        self._P = 0.5 * (self._P + self._P.T)
        self._t = (0.0 if self._t is None else self._t) + dt
        self._misses += 1
        return self._x.copy()

    def update(self, measurement: Any, timestamp_s: Optional[float] = None,
               *, measurement_covariance: Optional[Any] = None) -> UpdateResult:
        """Gated measurement update.  Returns the outcome; never silently ignores."""
        if self._x is None:
            raise ShuttleEKFError("filter is not initialised; call initialize() first")
        z = self._check_measurement(measurement)
        if timestamp_s is not None:
            t = float(timestamp_s)
            if not math.isfinite(t):
                raise ShuttleEKFError("timestamp_s must be finite")
            if self._t is not None and t < self._t - 1e-12:
                raise ShuttleEKFError(
                    f"non-monotonic timestamp: {t} < current {self._t}; a stale frame must "
                    "be rejected by the caller (see prediction horizon module P7-07), not "
                    "rewound here")
            self._t = t
        if measurement_covariance is None:
            R = self.measurement_covariance()
        else:
            R = np.asarray(measurement_covariance, dtype=float)
            n = self.position_dim
            if R.shape != (n, n) or not np.all(np.isfinite(R)):
                raise ShuttleEKFError(f"measurement_covariance must be a finite ({n},{n}) matrix")
            if not np.allclose(R, R.T, atol=1e-12):
                raise ShuttleEKFError("measurement_covariance must be symmetric")
            if float(np.min(np.linalg.eigvalsh(0.5 * (R + R.T)))) < -1e-12:
                raise ShuttleEKFError("measurement_covariance must be positive semi-definite")
            if float(np.min(np.diag(R))) <= 0.0:
                raise ShuttleEKFError("measurement_covariance diagonal must be > 0")
            R = 0.5 * (R + R.T)

        H = self.measurement_matrix()
        Hx = H @ self._x
        S = H @ self._P @ H.T + R
        innovation = z - Hx
        try:
            S_inv = np.linalg.inv(S)
        except np.linalg.LinAlgError as exc:  # pragma: no cover - defensive
            raise ShuttleEKFError("innovation covariance is singular") from exc
        d2 = float(innovation @ S_inv @ innovation)

        if d2 > self._gate_chi2:
            self._rejects += 1
            # _misses is "prediction steps since the last accepted update": the preceding
            # predict() already counted this frame, so a gated update must not count again.
            self._misses = max(self._misses, 1)
            if self._rejects >= self._reinit_streak:
                # Documented behaviour: after a streak of gated rejections the prediction,
                # not the measurement, is assumed wrong -> re-initialise and say so.
                n = self.position_dim
                self._x[:n] = z
                self._P[:n, :n] = R
                self._P[n:, :] = 0.0
                self._P[:, n:] = 0.0
                self._P[n:, n:] = np.eye(n) * self._initial_vel_std ** 2
                self._x[n:] = 0.0
                self._misses = 0
                self._rejects = 0
                return UpdateResult(False, "REINIT", d2, d2, self._gate_chi2,
                                    self._misses, self.status)
            return UpdateResult(False, "GATE", d2, d2, self._gate_chi2,
                                self._misses, self.status)

        K = self._P @ H.T @ S_inv
        self._x = self._x + K @ innovation
        I_KH = np.eye(2 * self.position_dim) - K @ H
        # Joseph form keeps P symmetric positive semi-definite under round-off.
        self._P = I_KH @ self._P @ I_KH.T + K @ R @ K.T
        self._P = 0.5 * (self._P + self._P.T)
        self._misses = 0
        self._rejects = 0
        self._accepted += 1
        return UpdateResult(True, "UPDATED", d2, d2, self._gate_chi2, self._misses, self.status)

    def step(self, measurement: Any, timestamp_s: float,
             *, measurement_covariance: Optional[Any] = None) -> UpdateResult:
        """predict(dt from timestamps) then update(); the normal per-frame call."""
        if self._x is None:
            self.initialize(measurement, timestamp_s)
            return UpdateResult(True, "UPDATED", 0.0, 0.0, self._gate_chi2, 0, self.status)
        dt = float(timestamp_s) - float(self._t) if self._t is not None else self.dt_s
        if dt < 0.0:
            raise ShuttleEKFError(
                f"non-monotonic timestamp: {timestamp_s} < current {self._t}")
        if dt > 0.0:
            self.predict(dt)
        return self.update(measurement, timestamp_s,
                           measurement_covariance=measurement_covariance)

    # -- 2D -> 3D upgrade -------------------------------------------------
    def upgrade_to_3d(self, *, depth_m: float, depth_sigma_m: float,
                      camera: CourtCameraModel, depth_source: str,
                      depth_rate_mps: Optional[float] = None,
                      depth_rate_sigma_mps: Optional[float] = None,
                      allow_uncalibrated: bool = False) -> UpgradeResult:
        """Convert the image-plane state into court-frame 3D, or refuse.

        Refusals (typed, never a silent 2D->3D assumption):
          StateModeError           filter is already 3D, or state uninitialised
          DepthNotTrustworthyError depth_source not in TRUSTWORTHY_DEPTH_SOURCES
                                   (e.g. MONOCULAR_GUESS) or depth/sigma not finite/positive
          UncalibratedCameraError  camera.calibration != CALIBRATED and
                                   allow_uncalibrated=False

        The conversion is the exact first-order propagation of the pinhole model
        p_cam = d * ((u-cx)/fx, (v-cy)/fy, 1), p_court = R p_cam + t, with the
        depth (and, if supplied, the depth rate) as extra uncertain inputs, so the
        3D covariance inherits BOTH the image-plane covariance and the depth error.
        """
        if self.mode != MODE_IMAGE_2D:
            raise StateModeError("upgrade_to_3d() only applies to an image-plane filter")
        if self._x is None:
            raise ShuttleEKFError("filter is not initialised; nothing to upgrade")
        if depth_source not in DEPTH_SOURCES:
            raise DepthNotTrustworthyError(
                f"unknown depth_source {depth_source!r}; expected one of {DEPTH_SOURCES}")
        if depth_source not in TRUSTWORTHY_DEPTH_SOURCES:
            raise DepthNotTrustworthyError(
                f"depth_source {depth_source!r} cannot support a 3D state: P7-06 forbids "
                "turning a 2D/estimated depth into a 3D position")
        d = float(depth_m)
        sd = float(depth_sigma_m)
        if not math.isfinite(d) or d <= 0.0:
            raise DepthNotTrustworthyError(f"depth_m must be finite and > 0, got {d}")
        if not math.isfinite(sd) or sd <= 0.0:
            raise DepthNotTrustworthyError(f"depth_sigma_m must be finite and > 0, got {sd}")
        if not camera.is_calibrated and not allow_uncalibrated:
            raise UncalibratedCameraError(
                f"camera calibration is {camera.calibration!r}: P7-06 requires CALIBRATED "
                "intrinsics and extrinsics before any 3D state exists (allow_uncalibrated="
                "True is an explicit, auditable opt-in used only for synthetic tests)")

        u, v, ud, vd = (float(x) for x in self._x)
        K = camera.K
        fx, fy, cx, cy = float(K[0, 0]), float(K[1, 1]), float(K[0, 2]), float(K[1, 2])
        R_c = camera.T_court_camera[:3, :3]
        t_c = camera.T_court_camera[:3, 3]

        if depth_rate_mps is None:
            dd = 0.0
            sdd = float(DEPTH_RATE_STD_PROXY_MPS.value)
            from_rate = False
        else:
            dd = float(depth_rate_mps)
            if not math.isfinite(dd):
                raise DepthNotTrustworthyError("depth_rate_mps must be finite")
            if depth_rate_sigma_mps is None:
                raise DepthNotTrustworthyError(
                    "depth_rate_sigma_mps is required when depth_rate_mps is supplied")
            sdd = float(depth_rate_sigma_mps)
            if not math.isfinite(sdd) or sdd <= 0.0:
                raise DepthNotTrustworthyError("depth_rate_sigma_mps must be finite and > 0")
            from_rate = True

        ax, ay = (u - cx) / fx, (v - cy) / fy
        p_cam = np.array([ax * d, ay * d, d])
        v_cam = np.array([ud * d / fx + ax * dd, vd * d / fy + ay * dd, dd])
        # Jacobian wrt [u, v, u_dot, v_dot, d, d_dot]  (6 inputs -> 6 state entries)
        J = np.zeros((6, 6))
        ex = R_c @ np.array([1.0, 0.0, 0.0])
        ey = R_c @ np.array([0.0, 1.0, 0.0])
        ed = R_c @ np.array([ax, ay, 1.0])
        J[0:3, 0] = ex * d / fx
        J[0:3, 1] = ey * d / fy
        J[0:3, 4] = ed
        J[3:6, 0] = ex * dd / fx
        J[3:6, 1] = ey * dd / fy
        J[3:6, 2] = ex * d / fx
        J[3:6, 3] = ey * d / fy
        J[3:6, 4] = R_c @ np.array([ud / fx, vd / fy, 0.0])
        J[3:6, 5] = ed

        P_aug = np.zeros((6, 6))
        P_aug[:4, :4] = self._P
        P_aug[4, 4] = sd ** 2
        P_aug[5, 5] = sdd ** 2
        P3 = J @ P_aug @ J.T
        P3 = 0.5 * (P3 + P3.T)
        x3 = np.concatenate([R_c @ p_cam + t_c, R_c @ v_cam])
        if not np.all(np.isfinite(x3)) or not np.all(np.isfinite(P3)):
            raise ShuttleEKFError("3D upgrade produced non-finite state/covariance")

        self._x = x3
        self._P = P3
        self.mode = MODE_COURT_3D
        self._misses = 0
        self._rejects = 0
        pos_std = np.sqrt(np.maximum(np.diag(P3)[:3], 0.0))
        vel_std = np.sqrt(np.maximum(np.diag(P3)[3:], 0.0))
        return UpgradeResult(self.mode, depth_source, camera.calibration, from_rate,
                             pos_std, vel_std)
