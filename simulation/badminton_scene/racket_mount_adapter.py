# -*- coding: utf-8 -*-
"""Explicit, fail-closed racket attachment for the B601-RS scene.

The visual asset and optional 43-shape collision proxy are attached beneath the
authored gripper_end frame. The proxy is engineering geometry, not a manufacturer
collision model; this module does not calibrate the mount or assert a physical
sweet spot, mass, inertia, or validated contact dynamics.
"""
from __future__ import annotations

from dataclasses import dataclass
import json
import math
from pathlib import Path
from typing import Mapping, Optional


class RacketMountError(ValueError):
    """The racket mount, visual asset, or collision proxy cannot be accepted safely."""


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


def resolve_collision_proxy_path(proxy_path: Optional[str] = None) -> Path:
    """Resolve the committed JSON proxy without requiring pxr / Isaac Sim."""
    path = Path(proxy_path) if proxy_path is not None else (
        Path(__file__).resolve().parents[2] / "outputs" / "simulation" /
        "rebot_b601_rs" / "morph_one" / "p6_frames" / "racket_collision_proxy.json")
    path = path.expanduser().resolve()
    if not path.is_file():
        raise RacketMountError("racket collision proxy JSON does not exist: %s" % path)
    return path


def load_racket_collision_proxy(proxy_path: Optional[str] = None) -> dict:
    """Load and validate all primitive records before mutating a USD stage."""
    path = resolve_collision_proxy_path(proxy_path)
    try:
        doc = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise RacketMountError("cannot read racket collision proxy %s: %s" % (path, exc)) from exc
    if doc.get("schema") != "p6_racket_collision_proxy/1":
        raise RacketMountError("unexpected racket collision proxy schema")
    if "NOT a manufacturer collision model" not in str(doc.get("kind", "")):
        raise RacketMountError("proxy kind must remain explicitly marked as engineering-only")
    primitives = doc.get("primitives")
    if not isinstance(primitives, list) or not primitives:
        raise RacketMountError("collision proxy must contain a non-empty primitives list")
    if int(doc.get("primitive_count", -1)) != len(primitives):
        raise RacketMountError("primitive_count does not match the primitives list")
    ids = [str(p.get("id", "")) for p in primitives]
    if any(not item for item in ids) or len(set(ids)) != len(ids):
        raise RacketMountError("collision proxy primitive ids must be non-empty and unique")

    for item in primitives:
        kind = item.get("type")
        geometry = item.get("geometry") or {}
        if kind == "capsule":
            start = geometry.get("axis_start_m")
            end = geometry.get("axis_end_m")
            radius = geometry.get("radius_m")
            if not _finite_vector(start, 3) or not _finite_vector(end, 3):
                raise RacketMountError("capsule %s needs finite 3D endpoints" % item["id"])
            if not _finite_positive(radius):
                raise RacketMountError("capsule %s radius must be positive" % item["id"])
            if math.dist(start, end) <= 1e-9:
                raise RacketMountError("capsule %s has a degenerate axis" % item["id"])
        elif kind == "box":
            center = geometry.get("center_m")
            half = geometry.get("half_extents_m")
            if not _finite_vector(center, 3) or not _finite_vector(half, 3):
                raise RacketMountError("box %s needs finite 3D centre and half-extents" % item["id"])
            if any(float(v) <= 0.0 for v in half):
                raise RacketMountError("box %s half-extents must be positive" % item["id"])
        else:
            raise RacketMountError("unsupported collision primitive type %r" % kind)
    return doc


def _finite_vector(value, size: int) -> bool:
    return (
        isinstance(value, (list, tuple)) and len(value) == size
        and all(isinstance(v, (int, float)) and math.isfinite(float(v)) for v in value)
    )


def _finite_positive(value) -> bool:
    return isinstance(value, (int, float)) and math.isfinite(float(value)) and float(value) > 0.0


def _quat_wxyz_from_z_axis(direction):
    """Return a unit wxyz quaternion rotating local +Z onto direction."""
    dx, dy, dz = (float(v) for v in direction)
    norm = math.sqrt(dx * dx + dy * dy + dz * dz)
    if not math.isfinite(norm) or norm <= 1e-12:
        raise RacketMountError("capsule axis direction is degenerate")
    x, y, z = dx / norm, dy / norm, dz / norm
    dot = max(-1.0, min(1.0, z))
    if dot >= 1.0 - 1e-12:
        return (1.0, 0.0, 0.0, 0.0)
    if dot <= -1.0 + 1e-12:
        return (0.0, 1.0, 0.0, 0.0)
    # axis = +Z cross direction
    ax, ay, az = -y, x, 0.0
    axis_norm = math.sqrt(ax * ax + ay * ay + az * az)
    ax, ay, az = ax / axis_norm, ay / axis_norm, az / axis_norm
    half = 0.5 * math.acos(dot)
    s = math.sin(half)
    return (math.cos(half), ax * s, ay * s, az * s)


def _set_orient(op, quaternion_wxyz, Gf, UsdGeom) -> None:
    """Set an orient op using its declared USD precision (float or double)."""
    w, x, y, z = (float(v) for v in quaternion_wxyz)
    if op.GetPrecision() == UsdGeom.XformOp.PrecisionDouble:
        op.Set(Gf.Quatd(w, Gf.Vec3d(x, y, z)))
    else:
        op.Set(Gf.Quatf(w, Gf.Vec3f(x, y, z)))


def capsule_spine_height_from_endpoints(axis_start, axis_end, radius: float) -> float:
    """Convert endpoint-to-endpoint length to USD capsule cylinder-spine height.

    USD UsdGeomCapsule.height excludes the two hemispherical caps. The proxy JSON
    stores axis_start_m/axis_end_m as the OUTER endpoints of the fitted shape,
    so subtract both cap radii. If the segment is shorter than its diameter, use
    a zero-height capsule (a sphere-like conservative proxy) rather than a
    negative height. This may over-cover very short taper segments, never under-cover.
    """
    if not _finite_vector(axis_start, 3) or not _finite_vector(axis_end, 3):
        raise RacketMountError("capsule endpoints must be finite 3D vectors")
    if not _finite_positive(radius):
        raise RacketMountError("capsule radius must be positive")
    return max(math.dist(axis_start, axis_end) - 2.0 * float(radius), 0.0)


def _author_collision_proxy(stage, parent_path: str, primitives: list) -> str:
    """Author primitive colliders under the link6 rigid body via gripper_end.

    No RigidBodyAPI is applied to the proxy: its shapes inherit the arm's link6
    rigid body through the gripper_end Xform hierarchy. No mass/inertia is added.
    """
    from pxr import Gf, UsdGeom, UsdPhysics

    proxy_path = parent_path + "/CollisionProxy"
    UsdGeom.Xform.Define(stage, proxy_path)
    for item in primitives:
        path = proxy_path + "/" + str(item["id"])
        geometry = item["geometry"]
        if item["type"] == "capsule":
            a = tuple(float(v) for v in geometry["axis_start_m"])
            b = tuple(float(v) for v in geometry["axis_end_m"])
            direction = tuple(b[i] - a[i] for i in range(3))
            radius = float(geometry["radius_m"])
            height = capsule_spine_height_from_endpoints(a, b, radius)
            centre = tuple((a[i] + b[i]) / 2.0 for i in range(3))
            quat = _quat_wxyz_from_z_axis(direction)
            capsule = UsdGeom.Capsule.Define(stage, path)
            capsule.CreateAxisAttr().Set(UsdGeom.Tokens.z)
            capsule.CreateHeightAttr().Set(height)
            capsule.CreateRadiusAttr().Set(radius)
            # OpenUSD requires an authored extent when height/radius are authored.
            # The extent is local to the capsule's +Z spine before its xform rotation.
            half_z = height / 2.0 + radius
            capsule.CreateExtentAttr().Set([
                Gf.Vec3f(-radius, -radius, -half_z),
                Gf.Vec3f(radius, radius, half_z),
            ])
            capsule.AddTranslateOp().Set(Gf.Vec3d(*centre))
            _set_orient(capsule.AddOrientOp(), quat, Gf, UsdGeom)
            prim = capsule.GetPrim()
        else:
            cube = UsdGeom.Cube.Define(stage, path)
            cube.CreateSizeAttr().Set(2.0)
            cube.AddTranslateOp().Set(Gf.Vec3d(*geometry["center_m"]))
            # AddScaleOp() defaults to double precision on this newly defined prim.
            cube.AddScaleOp().Set(Gf.Vec3d(*geometry["half_extents_m"]))
            prim = cube.GetPrim()
        UsdPhysics.CollisionAPI.Apply(prim)
        prim.SetCustomDataByKey("badminton_evidence_class", "PROXY_DERIVED")
        prim.SetCustomDataByKey("badminton_proxy_schema", "p6_racket_collision_proxy/1")
    return proxy_path


def attach_racket_visual(stage, *, arm_prim_path: str,
                         mount_transform: Optional[Mapping[str, object]],
                         asset_path: Optional[str] = None,
                         include_collision: bool = False,
                         collision_proxy_path: Optional[str] = None) -> str:
    """Attach racket visual and optionally its engineering collision proxy.

    The visual and collision shapes share the explicit mount transform. When
    include_collision=True, 43 primitive colliders are authored under link6's
    rigid-body hierarchy. They are not a separate rigid body and do not add the
    racket's unknown mass/inertia. This is contact geometry, not a validated
    physical racket model.
    """
    mount = validate_mount_transform(mount_transform)
    asset = resolve_racket_visual_path(asset_path)
    proxy_doc = load_racket_collision_proxy(collision_proxy_path) if include_collision else None
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
    _set_orient(orient, mount.quaternion_wxyz, Gf, UsdGeom)
    visual_xform = UsdGeom.Xform.Define(stage, visual_path)
    visual_xform.GetPrim().GetReferences().AddReference(str(asset), Sdf.Path("/RacketVisual"))
    visual_xform.GetPrim().SetCustomDataByKey("badminton_mount_status", mount.status)
    visual_xform.GetPrim().SetCustomDataByKey("badminton_mount_source", mount.source)
    if proxy_doc is not None:
        _author_collision_proxy(stage, mount_path, proxy_doc["primitives"])
    return visual_path


__all__ = [
    "RacketMountError", "RacketMountTransform", "attach_racket_visual",
    "gripper_end_prim_path", "racket_mount_prim_path",
    "resolve_racket_visual_path", "resolve_collision_proxy_path",
    "load_racket_collision_proxy", "validate_mount_transform",
    "capsule_spine_height_from_endpoints",
]
