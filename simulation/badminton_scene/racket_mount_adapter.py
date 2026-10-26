# -*- coding: utf-8 -*-
"""Explicit, fail-closed visual racket attachment for the B601-RS scene.

This attaches the racket visual beneath the authored gripper_end frame. It does
not create collision geometry, calibrate the mount, or assert a physical sweet spot.
"""
from __future__ import annotations

from dataclasses import dataclass
import math
from pathlib import Path
from typing import Mapping, Optional, Sequence


class RacketMountError(ValueError):
    """The racket mount or visual asset cannot be accepted safely."""


@dataclass(frozen=True)
class RacketMountTransform:
    translation_m: tuple[float, float, float]
    quaternion_wxyz: tuple[float, float, float, float]
    status: str
    source: str


def gripper_end_prim_path(arm_prim_path: str) -> str:
    """Canonical gripper_end path in the converted B601-RS USD hierarchy."""
    root = str(arm_prim_path).rstrip("/")
    if not root:
        raise RacketMountError("arm_prim_path must be non-empty")
    return root + "/Geometry/base_link/link1/link2/link3/link4/link5/link6/gripper_end"


def racket_mount_prim_path(arm_prim_path: str) -> str:
    return gripper_end_prim_path(arm_prim_path) + "/RacketMount"


def validate_mount_transform(record: Optional[Mapping[str, object]]) -> RacketMountTransform:
    """Validate an explicit mount record; UNKNOWN and implicit identity are rejected."""
    if not isinstance(record, Mapping):
        raise RacketMountError(
            "racket_mount_transform is required when include_racket=True; no default transform exists")
    status = str(record.get("status", "UNKNOWN")).strip().upper()
    if status not in {"MEASURED", "SIMULATION_ASSUMPTION"}:
        raise RacketMountError(
            "mount status must be MEASURED or explicit SIMULATION_ASSUMPTION, got %r" % status)
    translation = record.get("translation_m")
    quaternion = record.get("quaternion_wxyz")
    source = str(record.get("source", "")).strip()
    if not isinstance(translation, (list, tuple)) or len(translation) != 3:
        raise RacketMountError("translation_m must contain exactly three metre values")
    if not isinstance(quaternion, (list, tuple)) or len(quaternion) != 4:
        raise RacketMountError("quaternion_wxyz must contain exactly four values in wxyz order")
    t = tuple(float(v) for v in translation)
    q = tuple(float(v) for v in quaternion)
    if not all(math.isfinite(v) for v in t + q):
        raise RacketMountError("mount transform contains NaN or Inf")
    norm = math.sqrt(sum(v * v for v in q))
    if abs(norm - 1.0) > 1e-6:
        raise RacketMountError(
            "quaternion_wxyz must already be unit length; refusing silent normalization (norm=%.9g)" % norm)
    if not source:
        raise RacketMountError("mount transform requires a provenance/source string")
    return RacketMountTransform(t, q, status, source)


def resolve_racket_visual_path(asset_path: Optional[str] = None) -> Path:
    """Resolve the committed visual USD and fail before Isaac Sim starts if it is absent."""
    path = Path(asset_path) if asset_path is not None else (
        Path(__file__).resolve().parents[2] / "assets" / "third_party" / "racket_visual.usd")
    path = path.expanduser().resolve()
    if not path.is_file():
        raise RacketMountError("racket visual USD does not exist: %s" % path)
    return path


def attach_racket_visual(stage, *, arm_prim_path: str,
                         mount_transform: Optional[Mapping[str, object]],
                         asset_path: Optional[str] = None) -> str:
    """Reference the racket visual under gripper_end using an explicit local transform.

    Visual-only: no RigidBodyAPI, collision shapes, or fixed joint is authored here.
    Those must be implemented and validated separately before contact dynamics are claimed.
    """
    mount = validate_mount_transform(mount_transform)
    asset = resolve_racket_visual_path(asset_path)
    gripper_path = gripper_end_prim_path(arm_prim_path)
    mount_path = gripper_path + "/RacketMount"
    visual_path = mount_path + "/RacketVisual"

    gripper = stage.GetPrimAtPath(gripper_path)
    if gripper is None or not gripper.IsValid():
        raise RacketMountError("B601-RS gripper_end prim not found: %s" % gripper_path)
    existing = stage.GetPrimAtPath(mount_path)
    if existing is not None and existing.IsValid():
        raise RacketMountError("refusing to overwrite existing racket mount prim: %s" % mount_path)

    # Import USD only at call time so pure validation tests do not require Isaac Sim / pxr.
    from pxr import Gf, Sdf, UsdGeom

    mount_xform = UsdGeom.Xform.Define(stage, mount_path)
    mount_xform.AddTranslateOp().Set(Gf.Vec3d(*mount.translation_m))
    orient = mount_xform.AddOrientOp()
    orient.Set(Gf.Quatd(mount.quaternion_wxyz[0], Gf.Vec3d(*mount.quaternion_wxyz[1:])))
    visual_xform = UsdGeom.Xform.Define(stage, visual_path)
    visual_xform.GetPrim().GetReferences().AddReference(str(asset), Sdf.Path("/RacketVisual"))
    visual_xform.GetPrim().SetCustomDataByKey("badminton_mount_status", mount.status)
    visual_xform.GetPrim().SetCustomDataByKey("badminton_mount_source", mount.source)
    return visual_path


__all__ = [
    "RacketMountError", "RacketMountTransform", "attach_racket_visual",
    "gripper_end_prim_path", "racket_mount_prim_path",
    "resolve_racket_visual_path", "validate_mount_transform",
]
