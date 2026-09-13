# -*- coding: utf-8 -*-
"""Config validator (Phase 1).  Spec: BADMINTON_ROBOT.md S33 / S34 / S49."""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import List, Tuple

from ..badminton_robot_cfg import (
    UNRESOLVED_STATUSES,
    AssetStatus,
    BadmintonRobotCfg,
    Param,
    STEER_DRIVE_JOINT_NAMES,
)

PIPER_JOINT_ORDER: Tuple[str, ...] = ("joint1", "joint2", "joint3", "joint4", "joint5", "joint6")


class ValidationError(Exception):
    pass


@dataclass
class ValidationReport:
    ok: bool = True
    warnings: List[str] = field(default_factory=list)
    errors: List[str] = field(default_factory=list)


def _check_quat(name: str, quat, errors: List[str]) -> None:
    if quat is None:
        return
    if len(quat) != 4:
        errors.append(f"{name}: quaternion must have 4 components, got {len(quat)}")
        return
    norm = math.sqrt(sum(float(v) * float(v) for v in quat))
    if not math.isfinite(norm) or abs(norm - 1.0) > 1e-6:
        errors.append(f"{name}: quaternion is not normalized (|q|={norm})")


def _collect_unresolved(cfg: BadmintonRobotCfg) -> List[str]:
    out: List[str] = []
    for path, param in (
        ("morph_one.geometry.length_m", cfg.morph_one.geometry.length_m),
        ("morph_one.geometry.width_m", cfg.morph_one.geometry.width_m),
        ("morph_one.geometry.wheel_radius_m", cfg.morph_one.geometry.wheel_radius_m),
        ("morph_one.geometry.wheel_width_m", cfg.morph_one.geometry.wheel_width_m),
        ("morph_one.geometry.wheel_positions_robot", cfg.morph_one.geometry.wheel_positions_robot),
        ("morph_one.mass_properties.total_mass_kg", cfg.morph_one.mass_properties.total_mass_kg),
        ("morph_one.mass_properties.com_robot", cfg.morph_one.mass_properties.com_robot),
        ("morph_one.mass_properties.inertia_robot", cfg.morph_one.mass_properties.inertia_robot),
        ("morph_one.max_steer_angle_rad", cfg.morph_one.max_steer_angle_rad),
        ("morph_one.max_steer_rate_rad_s", cfg.morph_one.max_steer_rate_rad_s),
        ("morph_one.max_wheel_speed_rad_s", cfg.morph_one.max_wheel_speed_rad_s),
        ("piper.mount_translation", cfg.piper.mount_translation),
        ("piper.mount_quaternion", cfg.piper.mount_quaternion),
        ("racket.t_link6_tcp", cfg.racket.t_link6_tcp),
        ("racket.t_tcp_contact", cfg.racket.t_tcp_contact),
        ("stereo_camera.baseline_m", cfg.stereo_camera.baseline_m),
    ):
        if isinstance(param, Param) and param.status in UNRESOLVED_STATUSES:
            out.append(f"{path}={param.status.value}")
    return out


def validate_robot_cfg(cfg: BadmintonRobotCfg, *, mode: str = "development") -> ValidationReport:
    """Validate the config; final mode refuses any unresolved/TEMP parameter (S34)."""
    if mode not in ("development", "final"):
        raise ValueError("mode must be 'development' or 'final'")
    errors: List[str] = []
    warnings: List[str] = []

    # PiPER (S13 / S53)
    if tuple(cfg.piper.joint_names) != PIPER_JOINT_ORDER:
        errors.append(f"piper.joint_names must be exactly {PIPER_JOINT_ORDER}, got {tuple(cfg.piper.joint_names)}")

    # Morph One topology (S8 / S63.5)
    for joint in STEER_DRIVE_JOINT_NAMES:
        if joint not in cfg.morph_one.joint_names:
            errors.append(f"morph_one.joint_names is missing '{joint}' (four-steer four-drive topology is frozen)")

    # steering / wheel limits
    if cfg.morph_one.max_steer_angle_rad.value is not None:
        v = float(cfg.morph_one.max_steer_angle_rad.value)
        if not math.isfinite(v) or v <= 0.0:
            errors.append("morph_one.max_steer_angle_rad must be finite and > 0")
    for name, param in (("max_steer_rate_rad_s", cfg.morph_one.max_steer_rate_rad_s),
                        ("max_wheel_speed_rad_s", cfg.morph_one.max_wheel_speed_rad_s)):
        if param.value is not None and not (math.isfinite(float(param.value)) and float(param.value) > 0.0):
            errors.append(f"morph_one.{name} must be finite and > 0")

    # quaternions (S33.7)
    _check_quat("piper.mount_quaternion", cfg.piper.mount_quaternion.value, errors)
    if isinstance(cfg.racket.t_link6_tcp.value, dict):
        _check_quat("racket.t_link6_tcp.quaternion", cfg.racket.t_link6_tcp.value.get("quaternion_xyzw"), errors)
    _check_quat("racket.face_normal_quaternion", None, errors)

    # stereo camera (S18 / S55)
    baseline = cfg.stereo_camera.baseline_m.value
    if baseline is None or not (math.isfinite(float(baseline)) and float(baseline) > 0.0):
        errors.append("stereo_camera.baseline_m must be finite and > 0")
    if not cfg.stereo_camera.left_y_m > cfg.stereo_camera.right_y_m:
        errors.append("stereo_camera.left_y_m must be greater than right_y_m")
    if baseline is not None and abs((cfg.stereo_camera.left_y_m - cfg.stereo_camera.right_y_m) - float(baseline)) > 1e-9:
        errors.append("stereo_camera baseline must equal left_y_m - right_y_m")

    # NaN / Inf sweep over numeric param values (S33.13 / S45)
    for path, param in _all_params(cfg):
        if param.value is None:
            continue
        for v in _flatten_numeric(param.value):
            if not math.isfinite(v):
                errors.append(f"{path}: non-finite value {v}")

    unresolved = _collect_unresolved(cfg)
    if unresolved:
        if mode == "final":
            errors.append(
                "final mode requires resolved parameters; still TEMP/UNKNOWN: " + ", ".join(unresolved)
            )
        else:
            warnings.extend(f"TEMP/REQUIRES_MEASUREMENT: {u}" for u in unresolved)

    if errors:
        raise ValidationError("; ".join(errors))
    return ValidationReport(ok=True, warnings=warnings, errors=[])


def _all_params(cfg: BadmintonRobotCfg):
    for name in ("length_m", "width_m", "wheel_radius_m", "wheel_width_m", "wheel_positions_robot"):
        yield f"morph_one.geometry.{name}", getattr(cfg.morph_one.geometry, name)
    for name in ("total_mass_kg", "com_robot", "inertia_robot"):
        yield f"morph_one.mass_properties.{name}", getattr(cfg.morph_one.mass_properties, name)
    yield "piper.mount_translation", cfg.piper.mount_translation
    yield "piper.mount_quaternion", cfg.piper.mount_quaternion
    yield "racket.t_link6_tcp", cfg.racket.t_link6_tcp
    yield "racket.t_tcp_contact", cfg.racket.t_tcp_contact
    yield "stereo_camera.baseline_m", cfg.stereo_camera.baseline_m


def _flatten_numeric(value):
    if isinstance(value, (int, float)):
        yield float(value)
    elif isinstance(value, dict):
        for v in value.values():
            yield from _flatten_numeric(v)
    elif isinstance(value, (list, tuple)):
        for v in value:
            yield from _flatten_numeric(v)


__all__ = ["ValidationError", "ValidationReport", "validate_robot_cfg", "PIPER_JOINT_ORDER"]
