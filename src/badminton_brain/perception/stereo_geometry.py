# -*- coding: utf-8 -*-
"""Stereo perception geometry (pure numpy): image pixels -> court-frame 3D points.

Spec:
  * docs/architecture/ROBOT_BRAIN.md S11/S15 - perception layer; the brain only ever sees
    court-frame quantities (src/badminton_brain/types.py COURT_FRAME)
  * docs/architecture/COORDINATE_SYSTEM.md - court frame origin/axes and the camera rig
    (rig (-1.40, 0, 1.20), pitch -4 deg, baseline 0.29, left y=+0.145 / right y=-0.145)
  * docs/simulation/BADMINTON_ROBOT.md S4 (authenticity levels) / S12 (TEMP policy)

Frames are ALWAYS explicit, never implicit:

  court                  the only frame the brain sees.  Origin on the ground under the net
                         centre, +X towards the opponent, +Y robot-left, +Z up.
  camera_left/right      metric 3D frames of the rig cameras, OpenCV pinhole convention:
                         +X = image right, +Y = image down, +Z = optical axis (forward).
  image_left/right       pixel frames: uv = (u, v) = (column, row), origin at the CENTRE of
                         the top-left pixel, u right, v down.  A projection returns the uv
                         of that pixel centre; no half-pixel offset is hidden anywhere.

Transform naming: T_A_B maps a point expressed in frame B into frame A,
p_A = T_A_B @ p_B.  Hence T_left_right = T_{left<-right} and T_court_left is the pose of the
left camera in the court frame.  All transforms are homogeneous 4x4, rigid (checked).

Authenticity (BADMINTON_ROBOT.md S4/S12): the real stereo rig of this project is NOT
calibrated - there is no lens/sensor calibration and no measured mounting.  Therefore every
calibration quantity is a Param: the values this module runs on are
TEMP_PARAMETERIZED_PROXY engineering baselines with an explicit source, and the corresponding
measured/calibrated quantities exist as value=None entries with status REQUIRES_MEASUREMENT /
REQUIRES_CALIBRATION (see BaselineStereoSetup.parameters() / calibration_parameters()).
Nothing here may ever be read as a measurement of the real camera, and no test may claim the
synthetic 1e-6 m result as a real calibration accuracy.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Dict, Mapping, Optional, Sequence, Tuple

import numpy as np

from ..status import AssetStatus, Param
from ..types import COURT_FRAME, BrainBoundaryError

# ---------------------------------------------------------------------------
# Frame names (explicit, part of the module contract)
# ---------------------------------------------------------------------------
IMAGE_FRAME_LEFT = 'image_left'
IMAGE_FRAME_RIGHT = 'image_right'
CAMERA_FRAME_LEFT = 'camera_left'
CAMERA_FRAME_RIGHT = 'camera_right'

IMAGE_FRAMES = (IMAGE_FRAME_LEFT, IMAGE_FRAME_RIGHT)
CAMERA_FRAMES = (CAMERA_FRAME_LEFT, CAMERA_FRAME_RIGHT)

# ---------------------------------------------------------------------------
# ENGINEERING_V0_1 rig baseline - single source: simulation/badminton_scene/scene_layout.py
# (CAMERA block) and docs/architecture/COORDINATE_SYSTEM.md.  These are placeholder rig
# numbers, NOT measurements of the real mounting.
# ---------------------------------------------------------------------------
ENGINEERING_RIG_WORLD = np.array([-1.40, 0.0, 1.20])   # rig midpoint, court frame
ENGINEERING_RIG_PITCH_DEG = -4.0                       # scene_layout CAMERA['pitch_deg']
ENGINEERING_BASELINE_M = 0.29                          # scene_layout CAMERA['baseline']
ENGINEERING_LEFT_Y = 0.145                             # scene_layout CAMERA['left_y']
ENGINEERING_RIGHT_Y = -0.145                           # scene_layout CAMERA['right_y']
ENGINEERING_IMAGE_WIDTH_PX = 1280                      # TEMP: no sensor config exists yet
ENGINEERING_IMAGE_HEIGHT_PX = 720

_RIG_SOURCE = ("ENGINEERING_V0_1 placeholder rig, simulation/badminton_scene/scene_layout.py "
               "CAMERA (rig_world=(-1.40,0,1.20), pitch_deg=-4.0, baseline=0.29, "
               "left_y=+0.145, right_y=-0.145) and docs/architecture/COORDINATE_SYSTEM.md; "
               "not a measured mounting")
_INTRINSICS_SOURCE = ("TEMP engineering baseline for a 1280x720 rig camera: no intrinsic "
                      "calibration of the real lens/sensor has been performed and the USD "
                      "camera sensor has no render configuration yet; replace with the "
                      "chessboard calibration result (see the *_measured entries)")


# ---------------------------------------------------------------------------
# small rigid-transform helpers
# ---------------------------------------------------------------------------
def _as_transform(name: str, value: Any) -> np.ndarray:
    """Validate a homogeneous 4x4 rigid transform (rotation + translation only)."""
    arr = np.asarray(value, dtype=float)
    if arr.shape != (4, 4):
        raise BrainBoundaryError(f"{name} must be a 4x4 homogeneous transform, got {arr.shape}")
    if not np.all(np.isfinite(arr)):
        raise BrainBoundaryError(f"{name} contains NaN/Inf")
    if not np.allclose(arr[3], (0.0, 0.0, 0.0, 1.0), atol=1e-12):
        raise BrainBoundaryError(f"{name} last row must be [0, 0, 0, 1], got {arr[3]}")
    R = arr[:3, :3]
    if not np.allclose(R.T @ R, np.eye(3), atol=1e-9):
        raise BrainBoundaryError(f"{name} rotation is not orthonormal (scale/shear present)")
    if abs(float(np.linalg.det(R)) - 1.0) > 1e-9:
        raise BrainBoundaryError(f"{name} rotation determinant must be +1 (reflection)")
    return arr


def rigid_inverse(T: Any) -> np.ndarray:
    """Inverse of a rigid 4x4 transform (exact, no generic matrix inversion)."""
    arr = _as_transform('T', T)
    out = np.eye(4)
    out[:3, :3] = arr[:3, :3].T
    out[:3, 3] = -arr[:3, :3].T @ arr[:3, 3]
    return out


def compose(*transforms: Any) -> np.ndarray:
    """T_A_C = compose(T_A_B, T_B_C)."""
    out = np.eye(4)
    for T in transforms:
        out = out @ _as_transform('T', T)
    return _as_transform('compose()', out)


def forward_camera_rotation(pitch_deg: float) -> np.ndarray:
    """Rotation of a forward-facing rig camera looking from the robot side.

    Image +X (right) maps to court -Y and image +Y (down) maps to court -Z, so a shuttle to
    the robot's right appears on the right of the image.  pitch_deg > 0 tilts the optical
    axis DOWN (nose-down); pitch_deg < 0 tilts it up.

    The sign convention of scene_layout.CAMERA['pitch_deg'] is not documented anywhere in the
    docs; this module applies it as nose-down-positive, which makes the frozen -4.0 deg tilt
    the rig slightly UP instead of down.  That ambiguity is a REQUIRES_MEASUREMENT item
    (StereoExtrinsics.mount_pitch_measured_deg), never a measured fact.
    """
    angle = math.radians(float(pitch_deg))
    x_axis = np.array([0.0, -1.0, 0.0])
    z_axis = np.array([math.cos(angle), 0.0, -math.sin(angle)])
    y_axis = np.cross(z_axis, x_axis)
    return np.column_stack([x_axis, y_axis, z_axis])


def forward_camera_pose_court(x: float, y: float, z: float, pitch_deg: float) -> np.ndarray:
    """Pose (court frame) of a forward-facing camera at (x, y, z) with a given pitch."""
    T = np.eye(4)
    T[:3, :3] = forward_camera_rotation(pitch_deg)
    T[:3, 3] = np.array([float(x), float(y), float(z)])
    return _as_transform('camera pose', T)


# ---------------------------------------------------------------------------
# calibration containers - every number carries its authenticity
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class CameraIntrinsics:
    """Pinhole intrinsics of one rig camera (skew assumed zero, TEMP).

    frame        : pixel frame these intrinsics address (image_left / image_right)
    camera_frame : 3D camera frame they are defined in (camera_left / camera_right)
    """
    fx: Param
    fy: Param
    cx: Param
    cy: Param
    width_px: int = ENGINEERING_IMAGE_WIDTH_PX
    height_px: int = ENGINEERING_IMAGE_HEIGHT_PX
    frame: str = IMAGE_FRAME_LEFT
    camera_frame: str = CAMERA_FRAME_LEFT
    skew: Param = field(default_factory=lambda: Param(
        0.0, AssetStatus.TEMP_PARAMETERIZED_PROXY,
        "assumed zero pixel skew (rectangular pixels); a real calibration must report it"))

    def __post_init__(self) -> None:
        if self.frame not in IMAGE_FRAMES:
            raise ValueError(f"CameraIntrinsics.frame must be one of {IMAGE_FRAMES}, "
                             f"got {self.frame!r}")
        if self.camera_frame not in CAMERA_FRAMES:
            raise ValueError(f"CameraIntrinsics.camera_frame must be one of {CAMERA_FRAMES}, "
                             f"got {self.camera_frame!r}")
        if int(self.width_px) <= 1 or int(self.height_px) <= 1:
            raise ValueError("CameraIntrinsics.width_px/height_px must be > 1")
        for name in ('fx', 'fy', 'cx', 'cy', 'skew'):
            if not isinstance(getattr(self, name), Param):
                raise ValueError(f"CameraIntrinsics.{name} must be a Param, got "
                                 f"{type(getattr(self, name)).__name__}")
        if self.fx.value is not None and float(self.fx.value) <= 0.0:
            raise ValueError("CameraIntrinsics.fx must be positive when resolved")
        if self.fy.value is not None and float(self.fy.value) <= 0.0:
            raise ValueError("CameraIntrinsics.fy must be positive when resolved")

    @property
    def is_calibrated(self) -> bool:
        return all(getattr(self, n).value is not None for n in ('fx', 'fy', 'cx', 'cy'))

    def matrix(self) -> np.ndarray:
        """3x3 pinhole matrix K.  Raises when the intrinsics are still uncalibrated."""
        if not self.is_calibrated:
            missing = [n for n in ('fx', 'fy', 'cx', 'cy') if getattr(self, n).value is None]
            raise BrainBoundaryError(
                "CameraIntrinsics are not calibrated (missing " + ", ".join(missing) +
                "); a value must never be invented - run the chessboard calibration")
        K = np.array([
            [float(self.fx.value), float(self.skew.value), float(self.cx.value)],
            [0.0, float(self.fy.value), float(self.cy.value)],
            [0.0, 0.0, 1.0],
        ], dtype=float)
        if not np.all(np.isfinite(K)):
            raise BrainBoundaryError("CameraIntrinsics.matrix() is not finite")
        return K

    def parameters(self) -> Dict[str, Param]:
        params = {n: getattr(self, n) for n in ('fx', 'fy', 'cx', 'cy', 'skew')}
        for name in ('fx', 'fy', 'cx', 'cy'):
            params[name + '_measured'] = Param(
                None, AssetStatus.REQUIRES_CALIBRATION,
                f"calibrated focal/principal point of the real {self.frame} camera "
                "(chessboard/planar-target calibration on the assembled rig); the value "
                "currently in use is a TEMP engineering baseline")
        return params

    @classmethod
    def engineering_baseline(cls, frame: str = IMAGE_FRAME_LEFT,
                             camera_frame: str = CAMERA_FRAME_LEFT,
                             width_px: int = ENGINEERING_IMAGE_WIDTH_PX,
                             height_px: int = ENGINEERING_IMAGE_HEIGHT_PX,
                             fx: float = 700.0, fy: Optional[float] = None,
                             cx: Optional[float] = None, cy: Optional[float] = None
                             ) -> "CameraIntrinsics":
        """TEMP engineering baseline: a plausible rig camera, NOT a calibration result."""
        fy = fx if fy is None else fy
        cx = (width_px - 1) / 2.0 if cx is None else cx
        cy = (height_px - 1) / 2.0 if cy is None else cy
        return cls(
            fx=Param(fx, AssetStatus.TEMP_PARAMETERIZED_PROXY, _INTRINSICS_SOURCE),
            fy=Param(fy, AssetStatus.TEMP_PARAMETERIZED_PROXY, _INTRINSICS_SOURCE),
            cx=Param(cx, AssetStatus.TEMP_PARAMETERIZED_PROXY,
                     _INTRINSICS_SOURCE + "; principal point assumed at the image centre "
                     "(pixel-centre convention: (width-1)/2)"),
            cy=Param(cy, AssetStatus.TEMP_PARAMETERIZED_PROXY,
                     _INTRINSICS_SOURCE + "; principal point assumed at the image centre "
                     "(pixel-centre convention: (height-1)/2)"),
            width_px=int(width_px), height_px=int(height_px),
            frame=frame, camera_frame=camera_frame)

    @classmethod
    def uncalibrated(cls, frame: str = IMAGE_FRAME_LEFT,
                     camera_frame: str = CAMERA_FRAME_LEFT,
                     width_px: int = ENGINEERING_IMAGE_WIDTH_PX,
                     height_px: int = ENGINEERING_IMAGE_HEIGHT_PX) -> "CameraIntrinsics":
        """Explicit 'no calibration exists' intrinsics: all four values are None."""
        source = (f"intrinsics of the real {frame} camera have not been calibrated "
                  "(no chessboard/planar-target session on the assembled rig yet)")
        return cls(
            fx=Param(None, AssetStatus.REQUIRES_CALIBRATION, source),
            fy=Param(None, AssetStatus.REQUIRES_CALIBRATION, source),
            cx=Param(None, AssetStatus.REQUIRES_CALIBRATION, source),
            cy=Param(None, AssetStatus.REQUIRES_CALIBRATION, source),
            width_px=int(width_px), height_px=int(height_px),
            frame=frame, camera_frame=camera_frame)


@dataclass(frozen=True)
class StereoExtrinsics:
    """Stereo rig extrinsics.  Poses are expressed in the court frame (the brain contract).

    baseline_m                 : |t| of left->right, TEMP baseline until measured
    left_camera_pose_court     : T_court_left  (left camera pose in the court frame)
    right_camera_pose_court    : T_court_right
    left_right_transform       : T_left_right, so that p_left = T_left_right @ p_right
    """
    baseline_m: Param
    left_camera_pose_court: Param
    right_camera_pose_court: Param
    left_right_transform: Param
    pose_frame: str = COURT_FRAME
    left_frame: str = CAMERA_FRAME_LEFT
    right_frame: str = CAMERA_FRAME_RIGHT
    baseline_measured_m: Param = field(default_factory=lambda: Param(
        None, AssetStatus.REQUIRES_MEASUREMENT,
        "real camera separation of the assembled rig (calibration target or calliper); the "
        "value used for geometry is the ENGINEERING_V0_1 placeholder 0.29 m"))
    mount_pitch_measured_deg: Param = field(default_factory=lambda: Param(
        None, AssetStatus.REQUIRES_MEASUREMENT,
        "real mount pitch of the camera rig (and the sign convention of "
        "scene_layout.CAMERA['pitch_deg'], which the docs never state); the geometry runs on "
        "-4.0 deg interpreted as nose-down-positive"))
    left_camera_pose_measured_court: Param = field(default_factory=lambda: Param(
        None, AssetStatus.REQUIRES_MEASUREMENT,
        "measured pose of the left camera in the court frame after rig assembly; the geometry "
        "runs on the placeholder rig pose from scene_layout.py CAMERA"))
    relative_rotation_measured_deg: Param = field(default_factory=lambda: Param(
        None, AssetStatus.REQUIRES_CALIBRATION,
        "relative rotation between the two cameras (rectification) from stereo calibration; "
        "here both cameras are assumed perfectly parallel"))

    def __post_init__(self) -> None:
        if self.pose_frame != COURT_FRAME:
            raise BrainBoundaryError(
                f"StereoExtrinsics.pose_frame must be '{COURT_FRAME}' (brain contract), "
                f"got {self.pose_frame!r}")
        if self.left_frame == self.right_frame:
            raise BrainBoundaryError("StereoExtrinsics.left_frame/right_frame must differ")
        for name in ('baseline_m', 'left_camera_pose_court', 'right_camera_pose_court',
                     'left_right_transform'):
            if not isinstance(getattr(self, name), Param):
                raise ValueError(f"StereoExtrinsics.{name} must be a Param")
        if self.baseline_m.value is not None and float(self.baseline_m.value) <= 0.0:
            raise ValueError("StereoExtrinsics.baseline_m must be positive when resolved")
        if self.left_camera_pose_court.value is not None:
            T_left = _as_transform('left_camera_pose_court', self.left_camera_pose_court.value)
            T_right = _as_transform('right_camera_pose_court',
                                    self.right_camera_pose_court.value)
            T_lr = _as_transform('left_right_transform', self.left_right_transform.value)
            measured = float(np.linalg.norm(T_left[:3, 3] - T_right[:3, 3]))
            if self.baseline_m.value is not None and \
                    abs(measured - float(self.baseline_m.value)) > 1e-9:
                raise BrainBoundaryError(
                    f"declared baseline {float(self.baseline_m.value)} contradicts the camera "
                    f"poses ({measured}); one of them is wrong")
            implied = compose(rigid_inverse(T_left), T_right)
            if not np.allclose(implied, T_lr, atol=1e-9):
                raise BrainBoundaryError(
                    "left_right_transform contradicts left/right camera poses in the court frame")

    def T_left_right(self) -> np.ndarray:
        if self.left_right_transform.value is None:
            raise BrainBoundaryError("StereoExtrinsics.left_right_transform is unmeasured")
        return _as_transform('T_left_right', self.left_right_transform.value)

    def T_right_left(self) -> np.ndarray:
        return rigid_inverse(self.T_left_right())

    def T_court_left(self) -> np.ndarray:
        if self.left_camera_pose_court.value is None:
            raise BrainBoundaryError("StereoExtrinsics.left_camera_pose_court is unmeasured")
        return _as_transform('T_court_left', self.left_camera_pose_court.value)

    def T_court_right(self) -> np.ndarray:
        if self.right_camera_pose_court.value is None:
            raise BrainBoundaryError("StereoExtrinsics.right_camera_pose_court is unmeasured")
        return _as_transform('T_court_right', self.right_camera_pose_court.value)

    def baseline_from_transform(self) -> float:
        """Baseline implied by T_left_right (the number the geometry actually uses)."""
        return float(np.linalg.norm(self.T_left_right()[:3, 3]))

    def parameters(self) -> Dict[str, Param]:
        return {
            'baseline_m': self.baseline_m,
            'left_camera_pose_court': self.left_camera_pose_court,
            'right_camera_pose_court': self.right_camera_pose_court,
            'left_right_transform': self.left_right_transform,
            'baseline_measured_m': self.baseline_measured_m,
            'mount_pitch_measured_deg': self.mount_pitch_measured_deg,
            'left_camera_pose_measured_court': self.left_camera_pose_measured_court,
            'relative_rotation_measured_deg': self.relative_rotation_measured_deg,
        }

    @classmethod
    def engineering_baseline(cls, pitch_deg: float = ENGINEERING_RIG_PITCH_DEG,
                             baseline_m: float = ENGINEERING_BASELINE_M,
                             rig_world: Any = ENGINEERING_RIG_WORLD,
                             left_y: float = ENGINEERING_LEFT_Y,
                             right_y: float = ENGINEERING_RIGHT_Y) -> "StereoExtrinsics":
        """TEMP rig extrinsics: parallel cameras, placeholder mounting, zero relative rotation."""
        rig = np.asarray(rig_world, dtype=float).reshape(3)
        left_pose = forward_camera_pose_court(rig[0], rig[1] + left_y, rig[2], pitch_deg)
        right_pose = forward_camera_pose_court(rig[0], rig[1] + right_y, rig[2], pitch_deg)
        T_left_right = compose(rigid_inverse(left_pose), right_pose)
        measured_baseline = float(np.linalg.norm(T_left_right[:3, 3]))
        if abs(measured_baseline - float(baseline_m)) > 1e-9:
            raise BrainBoundaryError(
                f"left_y/right_y give a {measured_baseline} m baseline but {baseline_m} m was "
                "declared: the rig constants contradict each other")
        return cls(
            baseline_m=Param(baseline_m, AssetStatus.TEMP_PARAMETERIZED_PROXY, _RIG_SOURCE),
            left_camera_pose_court=Param(
                left_pose, AssetStatus.TEMP_PARAMETERIZED_PROXY,
                _RIG_SOURCE + "; left camera pose = rig midpoint + left_y"),
            right_camera_pose_court=Param(
                right_pose, AssetStatus.TEMP_PARAMETERIZED_PROXY,
                _RIG_SOURCE + "; right camera pose = rig midpoint + right_y"),
            left_right_transform=Param(
                T_left_right, AssetStatus.TEMP_PARAMETERIZED_PROXY,
                _RIG_SOURCE + "; both cameras assumed perfectly parallel (identity rotation), "
                "so T_left_right is the pure translation of the right camera seen from the "
                "left one"),
        )


@dataclass(frozen=True)
class BaselineStereoSetup:
    """The rig configuration the perception layer runs on (TEMP until the rig is calibrated)."""
    intrinsics_left: CameraIntrinsics
    intrinsics_right: CameraIntrinsics
    extrinsics: StereoExtrinsics

    def T_court_left(self) -> np.ndarray:
        return self.extrinsics.T_court_left()

    def T_court_right(self) -> np.ndarray:
        return self.extrinsics.T_court_right()

    def T_left_right(self) -> np.ndarray:
        return self.extrinsics.T_left_right()

    def parameters(self) -> Dict[str, Param]:
        params: Dict[str, Param] = {}
        for name, param in self.intrinsics_left.parameters().items():
            params['left.' + name] = param
        for name, param in self.intrinsics_right.parameters().items():
            params['right.' + name] = param
        params.update(self.extrinsics.parameters())
        return params

    def unresolved(self) -> Tuple[str, ...]:
        from ..status import UNRESOLVED_STATUSES
        return tuple(sorted(n for n, p in self.parameters().items()
                            if p.status in UNRESOLVED_STATUSES))


def baseline_stereo_setup() -> BaselineStereoSetup:
    """Build the ENGINEERING_V0_1 stereo rig (TEMP values, every one of them a Param)."""
    return BaselineStereoSetup(
        intrinsics_left=CameraIntrinsics.engineering_baseline(IMAGE_FRAME_LEFT,
                                                              CAMERA_FRAME_LEFT),
        intrinsics_right=CameraIntrinsics.engineering_baseline(IMAGE_FRAME_RIGHT,
                                                               CAMERA_FRAME_RIGHT),
        extrinsics=StereoExtrinsics.engineering_baseline())


def calibration_parameters() -> Dict[str, Param]:
    """Every calibration quantity of the rig with its authenticity (S4/S12), incl. the empty
    measured slots, so that nothing unknown can hide behind a number."""
    params = baseline_stereo_setup().parameters()
    params['intrinsics_uncalibrated_left.fx'] = CameraIntrinsics.uncalibrated(
        IMAGE_FRAME_LEFT, CAMERA_FRAME_LEFT).fx
    return params


# ---------------------------------------------------------------------------
# region-of-interest gating
# ---------------------------------------------------------------------------
def _check_uv(name: str, value: Any) -> np.ndarray:
    arr = np.asarray(value, dtype=float)
    if arr.ndim < 1 or arr.shape[-1] != 2:
        raise BrainBoundaryError(f"{name} must have shape (..., 2), got {arr.shape}")
    return arr


def _lead_mask(mask_flat: np.ndarray, shape: Tuple[int, ...]) -> Any:
    if shape == ():
        return bool(mask_flat[0])
    return mask_flat.reshape(shape)


@dataclass(frozen=True)
class ImageROI:
    """Rectangular pixel window; bounds are INCLUSIVE (u_min <= u <= u_max in pixels)."""
    u_min: float
    u_max: float
    v_min: float
    v_max: float
    frame: str = IMAGE_FRAME_LEFT
    source: str = ''
    status: AssetStatus = AssetStatus.TEMP_PARAMETERIZED_PROXY

    def __post_init__(self) -> None:
        for name in ('u_min', 'u_max', 'v_min', 'v_max'):
            if not math.isfinite(float(getattr(self, name))):
                raise ValueError(f"ImageROI.{name} must be finite")
        if float(self.u_min) >= float(self.u_max) or float(self.v_min) >= float(self.v_max):
            raise ValueError(f"ImageROI is empty: u[{self.u_min}, {self.u_max}] "
                             f"v[{self.v_min}, {self.v_max}]")
        if self.frame not in IMAGE_FRAMES:
            raise ValueError(f"ImageROI.frame must be one of {IMAGE_FRAMES}, got {self.frame!r}")
        if not str(self.source).strip():
            raise ValueError("ImageROI.source is required: state where the ROI comes from")
        if not isinstance(self.status, AssetStatus):
            raise ValueError("ImageROI.status must be an AssetStatus")

    def contains(self, uv: Any) -> np.ndarray:
        arr = _check_uv('uv', uv)
        ok = np.isfinite(arr).all(axis=-1)
        ok &= (arr[..., 0] >= self.u_min) & (arr[..., 0] <= self.u_max)
        ok &= (arr[..., 1] >= self.v_min) & (arr[..., 1] <= self.v_max)
        return ok.astype(bool)

    def parameters(self) -> Dict[str, Param]:
        return {name: Param(float(getattr(self, name)), self.status, self.source)
                for name in ('u_min', 'u_max', 'v_min', 'v_max')}


def roi_gate(uv_left: Any, roi: ImageROI) -> np.ndarray:
    """Boolean mask of the detections that fall inside the ROI (NaN -> False)."""
    if not isinstance(roi, ImageROI):
        raise BrainBoundaryError(f"roi must be an ImageROI, got {type(roi).__name__}")
    return roi.contains(uv_left)


def roi_select(uv_left: Any, roi: ImageROI) -> Tuple[np.ndarray, np.ndarray]:
    """Return (points inside the ROI, mask in the input's leading shape)."""
    arr = _check_uv('uv_left', uv_left)
    mask = np.asarray(roi_gate(arr, roi))
    flat = arr.reshape(-1, 2)[mask.reshape(-1)]
    return flat, mask


@dataclass(frozen=True)
class CourtBoxROI:
    """Axis-aligned box in the court frame (play volume gate for the perception layer)."""
    x_min: float
    x_max: float
    y_min: float
    y_max: float
    z_min: float
    z_max: float
    frame: str = COURT_FRAME
    source: str = ''
    status: AssetStatus = AssetStatus.TRACEABLE_REFERENCE

    def __post_init__(self) -> None:
        for name in ('x_min', 'x_max', 'y_min', 'y_max', 'z_min', 'z_max'):
            if not math.isfinite(float(getattr(self, name))):
                raise ValueError(f"CourtBoxROI.{name} must be finite")
        for lo, hi in (('x_min', 'x_max'), ('y_min', 'y_max'), ('z_min', 'z_max')):
            if float(getattr(self, lo)) >= float(getattr(self, hi)):
                raise ValueError(f"CourtBoxROI is empty: {lo} >= {hi}")
        if self.frame != COURT_FRAME:
            raise ValueError(f"CourtBoxROI.frame must be '{COURT_FRAME}', got {self.frame!r}")
        if not str(self.source).strip():
            raise ValueError("CourtBoxROI.source is required: state where the box comes from")
        if not isinstance(self.status, AssetStatus):
            raise ValueError("CourtBoxROI.status must be an AssetStatus")

    def contains(self, points_court: Any) -> np.ndarray:
        arr = np.asarray(points_court, dtype=float)
        if arr.ndim < 1 or arr.shape[-1] != 3:
            raise BrainBoundaryError(f"points must have shape (..., 3), got {arr.shape}")
        ok = np.isfinite(arr).all(axis=-1)
        ok &= (arr[..., 0] >= self.x_min) & (arr[..., 0] <= self.x_max)
        ok &= (arr[..., 1] >= self.y_min) & (arr[..., 1] <= self.y_max)
        ok &= (arr[..., 2] >= self.z_min) & (arr[..., 2] <= self.z_max)
        return ok.astype(bool)

    def parameters(self) -> Dict[str, Param]:
        return {name: Param(float(getattr(self, name)), self.status, self.source)
                for name in ('x_min', 'x_max', 'y_min', 'y_max', 'z_min', 'z_max')}

    @classmethod
    def perception_volume(cls) -> "CourtBoxROI":
        """ENGINEERING_V0_1 perception volume (scene_layout.PERCEPTION_ROI, spec 15)."""
        return cls(
            x_min=-0.60, x_max=1.40, y_min=-1.00, y_max=1.00, z_min=0.30, z_max=2.80,
            source="simulation/badminton_scene/scene_layout.py PERCEPTION_ROI "
                   "(world_x/world_y/z, spec 15, ENGINEERING_V0_1)")


def court_box_gate(points_court: Any, roi: CourtBoxROI) -> np.ndarray:
    """Boolean mask of the 3D points inside the court-frame play volume (NaN -> False)."""
    if not isinstance(roi, CourtBoxROI):
        raise BrainBoundaryError(f"roi must be a CourtBoxROI, got {type(roi).__name__}")
    return roi.contains(points_court)


# ---------------------------------------------------------------------------
# subpixel centroid
# ---------------------------------------------------------------------------
def _check_patch(image_patch: Any) -> np.ndarray:
    patch = np.asarray(image_patch, dtype=float)
    if patch.ndim != 2:
        raise BrainBoundaryError(f"image_patch must be 2-D (rows, cols), got shape {patch.shape}")
    if patch.size == 0:
        raise BrainBoundaryError("image_patch is empty")
    if not np.all(np.isfinite(patch)):
        raise BrainBoundaryError("image_patch contains NaN/Inf")
    return patch


def peak_pixel(image_patch: Any) -> Tuple[int, int]:
    """Integer peak of the patch as (u, v) = (column, row); ties resolve to the first."""
    patch = _check_patch(image_patch)
    row, col = np.unravel_index(int(np.argmax(patch)), patch.shape)
    return int(col), int(row)


def subpixel_centroid(image_patch: Any, threshold: float = 0.0,
                      origin_uv: Sequence[float] = (0.0, 0.0)) -> np.ndarray:
    """Intensity-weighted centroid of a patch, in the uv frame of the patch.

    The patch pixel (row i, column j) sits at uv = origin_uv + (j, i), i.e. u is the column
    and v the row, with the integer value at the CENTRE of that pixel (image convention of
    this module).  threshold is subtracted first and negative residuals are clipped, which is
    the background rejection used for a bright shuttle on a darker background.  Raises
    BrainBoundaryError when the patch cannot carry a measurement (empty, NaN, no intensity).
    """
    patch = _check_patch(image_patch)
    origin = np.asarray(origin_uv, dtype=float).reshape(-1)
    if origin.shape != (2,) or not np.all(np.isfinite(origin)):
        raise BrainBoundaryError(f"origin_uv must be two finite numbers, got {origin_uv!r}")
    weights = np.clip(patch - float(threshold), 0.0, None)
    total = float(weights.sum())
    if total <= 0.0:
        raise BrainBoundaryError(
            f"image_patch has no intensity above threshold={threshold}: no centroid exists")
    rows = np.arange(patch.shape[0], dtype=float)
    cols = np.arange(patch.shape[1], dtype=float)
    v = float((weights.sum(axis=1) * rows).sum() / total)
    u = float((weights.sum(axis=0) * cols).sum() / total)
    return np.array([origin[0] + u, origin[1] + v])


# ---------------------------------------------------------------------------
# triangulation
# ---------------------------------------------------------------------------
def intrinsics_matrix(K: Any, name: str = 'K') -> np.ndarray:
    """Accept a CameraIntrinsics, a 3x3 matrix, a {fx,fy,cx,cy[,skew]} mapping or a
    4/5-sequence (fx, fy, cx, cy[, skew])."""
    if isinstance(K, CameraIntrinsics):
        return K.matrix()
    if isinstance(K, Mapping):
        missing = [k for k in ('fx', 'fy', 'cx', 'cy') if k not in K]
        if missing:
            raise BrainBoundaryError(f"{name} mapping is missing {missing}")
        matrix = np.array([[float(K['fx']), float(K.get('skew', 0.0)), float(K['cx'])],
                           [0.0, float(K['fy']), float(K['cy'])],
                           [0.0, 0.0, 1.0]], dtype=float)
    else:
        arr = np.asarray(K, dtype=float)
        if arr.shape == (4,) or arr.shape == (5,):
            fx, fy, cx, cy = (float(x) for x in arr[:4])
            skew = float(arr[4]) if arr.shape == (5,) else 0.0
            matrix = np.array([[fx, skew, cx], [0.0, fy, cy], [0.0, 0.0, 1.0]], dtype=float)
        elif arr.shape == (3, 3):
            matrix = arr.astype(float)
        else:
            raise BrainBoundaryError(
                f"{name} must be a CameraIntrinsics, a 3x3 matrix, a mapping with "
                f"fx/fy/cx/cy or a (4,)/(5,) sequence, got shape {arr.shape}")
    if not np.all(np.isfinite(matrix)):
        raise BrainBoundaryError(f"{name} contains NaN/Inf")
    if matrix[2, 2] == 0.0:
        raise BrainBoundaryError(f"{name}[2, 2] must be non-zero")
    if matrix[0, 0] <= 0.0 or matrix[1, 1] <= 0.0:
        raise BrainBoundaryError(f"{name} focal lengths must be positive")
    return matrix


def triangulate(uv_left: Any, uv_right: Any, K: Any, T_left_right: Any,
                K_right: Any = None, T_court_left: Any = None,
                valid: Any = None) -> np.ndarray:
    """Triangulate correspondence pairs into 3D points (DLT, batched, pure numpy).

    Input
      uv_left / uv_right : (..., 2) pixel coordinates (u = column, v = row) in image_left and
                           image_right.  They must be finite; use the valid mask for the
                           detections a gate rejected.
      K                  : intrinsics of the LEFT camera (CameraIntrinsics / 3x3 / mapping /
                           (fx, fy, cx, cy[, skew])).  K_right defaults to the same camera.
      T_left_right       : rigid 4x4, p_left = T_left_right @ p_right.
      T_court_left       : rigid 4x4 pose of the left camera in the court frame.  Default is
                           the ENGINEERING_V0_1 rig pose, so the result is in the COURT FRAME
                           (the brain contract); pass np.eye(4) to stay in camera_left.
      valid              : optional (...,) mask; False rows come back as NaN.

    Output
      (..., 3) points, in the court frame by default.

    Accuracy note: with an exact synthetic pair the result is exact to floating point.  The
    real stereo error of this rig is unknown until the cameras are calibrated; nothing here
    claims a measured accuracy.
    """
    uv_l = _check_uv('uv_left', uv_left)
    uv_r = _check_uv('uv_right', uv_right)
    if uv_l.shape[:-1] != uv_r.shape[:-1]:
        raise BrainBoundaryError(
            f"uv_left and uv_right must share their leading shape, got {uv_l.shape} and "
            f"{uv_r.shape}")
    if not np.all(np.isfinite(uv_l)) or not np.all(np.isfinite(uv_r)):
        raise BrainBoundaryError(
            "uv_left/uv_right contain NaN/Inf: gate them out with an ROI or pass valid=")
    K_l = intrinsics_matrix(K, 'K')
    K_r = intrinsics_matrix(K, 'K_right') if K_right is not None else K_l
    T_lr = _as_transform('T_left_right', T_left_right)
    T_rl = rigid_inverse(T_lr)
    if T_court_left is None:
        T_court_left = baseline_stereo_setup().T_court_left()
    T_court = _as_transform('T_court_left', T_court_left)
    if valid is not None:
        mask = np.asarray(valid, dtype=bool)
        if mask.shape != uv_l.shape[:-1]:
            raise BrainBoundaryError(
                f"valid mask shape {mask.shape} does not match uv leading shape "
                f"{uv_l.shape[:-1]}")

    lead = uv_l.shape[:-1]
    flat_l = uv_l.reshape(-1, 2)
    flat_r = uv_r.reshape(-1, 2)
    count = flat_l.shape[0]

    # DLT: every correspondence contributes two rows per camera.
    P_l = K_l @ np.hstack([np.eye(3), np.zeros((3, 1))])
    P_r = K_r @ T_rl[:3, :]
    A = np.empty((count, 4, 4), dtype=float)
    A[:, 0, :] = flat_l[:, 0:1] * P_l[2] - P_l[0]
    A[:, 1, :] = flat_l[:, 1:2] * P_l[2] - P_l[1]
    A[:, 2, :] = flat_r[:, 0:1] * P_r[2] - P_r[0]
    A[:, 3, :] = flat_r[:, 1:2] * P_r[2] - P_r[1]
    _, singular, Vh = np.linalg.svd(A)
    if count:
        scale = singular[:, 0]
        # rank must be 3: a rank-2 system means both rays are the same line (no parallax).
        bad = singular[:, 2] <= 1e-9 * np.maximum(scale, 1e-12)
        if np.any(bad):
            raise BrainBoundaryError(
                "degenerate triangulation: zero parallax for correspondence(s) "
                f"{np.flatnonzero(bad)[:5].tolist()} (the two rays coincide)")
    homogeneous = Vh[:, -1, :]
    weight = homogeneous[:, 3]
    if np.any(np.abs(weight) < 1e-12):
        raise BrainBoundaryError("degenerate triangulation: point at infinity (w = 0)")
    points_left = homogeneous[:, :3] / weight[:, None]

    R = T_court[:3, :3]
    t = T_court[:3, 3]
    points = (R @ points_left.T).T + t
    if not np.all(np.isfinite(points)):
        raise BrainBoundaryError("triangulation produced non-finite points")
    if valid is not None:
        points[~mask.reshape(-1)] = np.nan
    return points.reshape(lead + (3,))


__all__ = [
    "COURT_FRAME", "IMAGE_FRAME_LEFT", "IMAGE_FRAME_RIGHT", "CAMERA_FRAME_LEFT",
    "CAMERA_FRAME_RIGHT", "ENGINEERING_RIG_WORLD", "ENGINEERING_RIG_PITCH_DEG",
    "ENGINEERING_BASELINE_M", "ENGINEERING_LEFT_Y", "ENGINEERING_RIGHT_Y",
    "ENGINEERING_IMAGE_WIDTH_PX", "ENGINEERING_IMAGE_HEIGHT_PX",
    "CameraIntrinsics", "StereoExtrinsics", "BaselineStereoSetup", "ImageROI", "CourtBoxROI",
    "baseline_stereo_setup", "calibration_parameters", "roi_gate", "roi_select",
    "court_box_gate", "subpixel_centroid", "peak_pixel", "triangulate", "intrinsics_matrix",
    "forward_camera_rotation", "forward_camera_pose_court", "rigid_inverse", "compose",
]
