# -*- coding: utf-8 -*-
"""BadmintonRobot configuration schema (Phase 1).

Source of truth: docs/simulation/BADMINTON_ROBOT.md
  S4  asset authenticity levels (AssetStatus)
  S12 TEMP parameter policy (value + status + source)
  S13 PiPER frozen joint order, S14 PiPER mount, S15/S16 racket, S18 stereo camera
  S31 initial pose belongs to the scene/experiment config layer (not hardcoded in the class)
Frame names follow docs/architecture/COORDINATE_SYSTEM.md S2.5 (canonical snake_case).

Nothing here invents a real physical value: every field carries an explicit status.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field, replace
from enum import Enum
from typing import Any, Dict, Optional, Tuple


class AssetStatus(str, Enum):
    """S4 authenticity levels."""
    VERIFIED_OFFICIAL = "VERIFIED_OFFICIAL"
    VERIFIED_MEASURED = "VERIFIED_MEASURED"
    DERIVED_FROM_MEASUREMENT = "DERIVED_FROM_MEASUREMENT"
    TRACEABLE_REFERENCE = "TRACEABLE_REFERENCE"
    TEMP_PARAMETERIZED_PROXY = "TEMP_PARAMETERIZED_PROXY"
    UNKNOWN = "UNKNOWN"
    REQUIRES_MEASUREMENT = "REQUIRES_MEASUREMENT"
    REQUIRES_CALIBRATION = "REQUIRES_CALIBRATION"


UNRESOLVED_STATUSES = frozenset({
    AssetStatus.TEMP_PARAMETERIZED_PROXY,
    AssetStatus.UNKNOWN,
    AssetStatus.REQUIRES_MEASUREMENT,
    AssetStatus.REQUIRES_CALIBRATION,
})


@dataclass
class Param:
    """A value that can never hide how it was obtained (S4 / S12)."""
    value: Any = None
    status: AssetStatus = AssetStatus.REQUIRES_MEASUREMENT
    source: str = ""

    def __post_init__(self) -> None:
        if not isinstance(self.status, AssetStatus):
            raise ValueError(f"status must be an AssetStatus, got {self.status!r}")
        if not str(self.source).strip():
            raise ValueError("Param.source is required: state where the value comes from")
        unresolved = {AssetStatus.REQUIRES_MEASUREMENT, AssetStatus.UNKNOWN, AssetStatus.REQUIRES_CALIBRATION}
        if self.value is not None and self.status in unresolved:
            raise ValueError(
                f"value provided while status={self.status.value}; "
                "either mark it TEMP_PARAMETERIZED_PROXY or leave value=None"
            )

    @property
    def is_resolved(self) -> bool:
        return self.status not in UNRESOLVED_STATUSES


class WheelId(str, Enum):
    """Four steer+drive wheel modules (S3.4 / S8 keep the real topology)."""
    FL = "FL"
    FR = "FR"
    RL = "RL"
    RR = "RR"


STEER_DRIVE_JOINT_NAMES: Tuple[str, ...] = tuple(
    f"{w.value.lower()}_{kind}"
    for w in WheelId
    for kind in ("steer", "drive")
)


class DriveMode(str, Enum):
    """S9 two supported chassis drive modes."""
    BODY_TWIST_ACTUATOR = "BODY_TWIST_ACTUATOR"
    STEER_DRIVE_WHEEL_MODEL = "STEER_DRIVE_WHEEL_MODEL"


@dataclass
class MorphOneGeometry:
    length_m: Param = field(default_factory=lambda: Param(0.70, AssetStatus.TEMP_PARAMETERIZED_PROXY, "engineering baseline chassis box"))
    width_m: Param = field(default_factory=lambda: Param(0.55, AssetStatus.TEMP_PARAMETERIZED_PROXY, "engineering baseline chassis box"))
    wheel_radius_m: Param = field(default_factory=lambda: Param(0.06, AssetStatus.TEMP_PARAMETERIZED_PROXY, "engineering baseline wheel radius"))
    wheel_width_m: Param = field(default_factory=lambda: Param(0.04, AssetStatus.TEMP_PARAMETERIZED_PROXY, "engineering baseline wheel width"))
    wheel_positions_robot: Param = field(default_factory=lambda: Param(
        None, AssetStatus.REQUIRES_MEASUREMENT, "wheel center coordinates in robot_base"))


@dataclass
class MorphOneMassProperties:
    total_mass_kg: Param = field(default_factory=lambda: Param(None, AssetStatus.REQUIRES_MEASUREMENT, "whole-base mass not measured"))
    com_robot: Param = field(default_factory=lambda: Param(None, AssetStatus.REQUIRES_MEASUREMENT, "base COM not measured"))
    inertia_robot: Param = field(default_factory=lambda: Param(None, AssetStatus.REQUIRES_MEASUREMENT, "base inertia not measured"))


@dataclass
class MorphOneCfg:
    """Four-steer four-drive chassis config (topology is frozen, values are not)."""
    asset_status: AssetStatus = AssetStatus.TEMP_PARAMETERIZED_PROXY
    geometry: MorphOneGeometry = field(default_factory=MorphOneGeometry)
    mass_properties: MorphOneMassProperties = field(default_factory=MorphOneMassProperties)
    joint_names: Dict[str, str] = field(default_factory=lambda: {n: n for n in STEER_DRIVE_JOINT_NAMES})
    max_steer_angle_rad: Param = field(default_factory=lambda: Param(math.pi, AssetStatus.TEMP_PARAMETERIZED_PROXY, "continuous steering assumed"))
    max_steer_rate_rad_s: Param = field(default_factory=lambda: Param(6.0, AssetStatus.TEMP_PARAMETERIZED_PROXY, "engineering baseline steer rate"))
    max_wheel_speed_rad_s: Param = field(default_factory=lambda: Param(40.0, AssetStatus.TEMP_PARAMETERIZED_PROXY, "engineering baseline wheel speed"))


@dataclass
class PiperCfg:
    """S13/S14: the Stage-0 PiPER asset is a frozen dependency."""
    usd_path: str = "assets/piper_stage0/assets/piper_no_gripper.usd"
    joint_names: Tuple[str, ...] = ("joint1", "joint2", "joint3", "joint4", "joint5", "joint6")
    link6_prim_path: str = "/Robot/Piper/Geometry/link6"
    mount_translation: Param = field(default_factory=lambda: Param(
        (0.0, 0.0, 0.30), AssetStatus.TEMP_PARAMETERIZED_PROXY, "engineering baseline mount"))
    mount_quaternion: Param = field(default_factory=lambda: Param(
        (1.0, 0.0, 0.0, 0.0), AssetStatus.TEMP_PARAMETERIZED_PROXY, "engineering baseline mount"))

    @property
    def num_joints(self) -> int:
        return len(self.joint_names)


@dataclass
class RacketCfg:
    """S15/S16: fixed racket, +X face normal, +Z handle->head."""
    usd_path: str = "assets/third_party/racket_visual.usd"
    racket_root_prim_path: str = "/Robot/Racket"
    face_normal_local: Tuple[float, float, float] = (1.0, 0.0, 0.0)
    t_link6_tcp: Param = field(default_factory=lambda: Param(
        None, AssetStatus.REQUIRES_MEASUREMENT, "adapter transform T_link6_tcp not measured"))
    t_tcp_contact: Param = field(default_factory=lambda: Param(
        None, AssetStatus.REQUIRES_MEASUREMENT, "string-bed contact centre not measured"))


@dataclass
class StereoCameraCfg:
    """S18: engineering baseline stereo rig (TEMP, replaceable)."""
    asset_status: AssetStatus = AssetStatus.TEMP_PARAMETERIZED_PROXY
    baseline_m: Param = field(default_factory=lambda: Param(0.29, AssetStatus.TEMP_PARAMETERIZED_PROXY, "engineering baseline baseline"))
    left_y_m: float = 0.145
    right_y_m: float = -0.145
    center_offset_robot_xyz: Tuple[float, float, float] = (0.20, 0.0, 1.20)
    pitch_deg: float = -4.0
    frame_left: str = "camera_left"
    frame_right: str = "camera_right"
    frame_center: str = "camera_center"


@dataclass
class FrameConventionCfg:
    """S3.1 frozen Court Frame + COORDINATE_SYSTEM S2.5/S2.6 naming."""
    world_frame: str = "court"
    up_axis: str = "Z"
    handedness: str = "right"
    transform_naming: str = "T_A_B (B to A)"
    canonical_frames: Tuple[str, ...] = (
        "court", "robot_base",
        "piper_base", "piper_link1", "piper_link2", "piper_link3",
        "piper_link4", "piper_link5", "piper_link6",
        "racket_tcp", "racket_contact", "racket_contact_frame",
        "camera_rig", "camera_center", "camera_left", "camera_right",
        "camera_left_optical", "camera_right_optical", "imu_link",
    )


@dataclass
class BadmintonRobotCfg:
    morph_one: MorphOneCfg = field(default_factory=MorphOneCfg)
    piper: PiperCfg = field(default_factory=PiperCfg)
    racket: RacketCfg = field(default_factory=RacketCfg)
    stereo_camera: StereoCameraCfg = field(default_factory=StereoCameraCfg)
    drive_mode: DriveMode = DriveMode.BODY_TWIST_ACTUATOR
    frame_convention: FrameConventionCfg = field(default_factory=FrameConventionCfg)
    # S31: initial pose is scene/experiment configuration, never baked into the robot class.
    initial_base_pose_xyz: Tuple[float, float, float] = (-1.60, 0.0, 0.0)
    initial_base_pose_source: str = "engineering baseline, scene/experiment config layer"

    def copy(self) -> "BadmintonRobotCfg":
        return replace(self)


def make_default_robot_cfg() -> BadmintonRobotCfg:
    """Fresh default config; every unresolved field stays explicitly unresolved."""
    return BadmintonRobotCfg()


__all__ = [
    "AssetStatus", "Param", "UNRESOLVED_STATUSES",
    "WheelId", "STEER_DRIVE_JOINT_NAMES", "DriveMode",
    "MorphOneGeometry", "MorphOneMassProperties", "MorphOneCfg",
    "PiperCfg", "RacketCfg", "StereoCameraCfg", "FrameConventionCfg",
    "BadmintonRobotCfg", "make_default_robot_cfg",
]
