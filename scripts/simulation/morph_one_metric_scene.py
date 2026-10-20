"""Build and measure a metres-stage assembly of the converted Morph One visual USD.

Scope
-----
This module only does scene composition and measurement: reference the visual
USD into a metres stage, optionally scale it, and read back what the composed
stage actually contains.  It adds no collider, no mass, no joint and no arm
attachment, and it does not claim any physical property of the chassis.

Why the measurement transforms points instead of trusting one bbox call
----------------------------------------------------------------------
UsdGeom.BBoxCache already returns world bounds, but a bbox can hide a wrong
transform.  So the primary measurement here walks the composed stage through
instance proxies, reads every mesh's points and pushes each point through that
prim's local-to-world matrix.  The bbox-cache value is reported next to it and
the two must agree; if they disagree the report says so.

Nothing in the module computes an "expected" size from a formula: the expected
envelope is an input used only for comparison, and every measured number comes
from prim data in the stage.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Sequence

__all__ = [
    "build_metric_scene",
    "measure_scene",
    "world_bbox_from_points",
    "envelope_within",
    "ratio_between",
    "write_json",
]

DEFAULT_ENVELOPE_M = (0.580, 0.373, 0.803)


def envelope_within(size, expected, tolerance_ratio: float = 0.02, floor_m: float = 0.002) -> bool:
    """True when every axis is within max(relative tolerance, absolute floor)."""
    if size is None or expected is None:
        return False
    if len(size) != len(expected):
        return False
    for measured, target in zip(size, expected):
        allowed = max(abs(target) * tolerance_ratio, floor_m)
        if abs(measured - target) > allowed:
            return False
    return True


def ratio_between(numerator, denominator) -> "list[float] | None":
    if not numerator or not denominator:
        return None
    return [
        round(numerator[i] / denominator[i], 9) if denominator[i] else None
        for i in range(len(numerator))
    ]


def build_metric_scene(scene_path, visual_usd, scale=None, meters_per_unit: float = 1.0) -> str:
    """Author a metres stage that references the visual USD, optionally scaled."""
    from pxr import Gf, Usd, UsdGeom

    scene_path = str(scene_path)
    Path(scene_path).parent.mkdir(parents=True, exist_ok=True)
    stage = Usd.Stage.CreateNew(scene_path)
    UsdGeom.SetStageMetersPerUnit(stage, meters_per_unit)
    UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)

    root = UsdGeom.Xform.Define(stage, "/MorphOne")
    stage.SetDefaultPrim(root.GetPrim())
    visual = UsdGeom.Xform.Define(stage, "/MorphOne/Visual")
    visual.GetPrim().GetReferences().AddReference(str(visual_usd))
    if scale is not None:
        visual.AddScaleOp().Set(Gf.Vec3f(float(scale[0]), float(scale[1]), float(scale[2])))
    stage.GetRootLayer().Save()
    return scene_path


def world_bbox_from_points(stage, mesh_limit: int = 0) -> dict:
    """World-space bbox by transforming every mesh point of every instance."""
    from pxr import Gf, Usd, UsdGeom

    xform_cache = UsdGeom.XformCache(Usd.TimeCode.Default())
    low = [float("inf")] * 3
    high = [float("-inf")] * 3
    mesh_instances = 0
    points_used = 0
    per_mesh = []

    proxies = Usd.PrimRange(
        stage.GetPseudoRoot(), Usd.TraverseInstanceProxies(Usd.PrimDefaultPredicate)
    )
    for prim in proxies:
        if prim.GetTypeName() != "Mesh":
            continue
        points = UsdGeom.Mesh(prim).GetPointsAttr().Get() or []
        if not points:
            continue
        matrix = xform_cache.GetLocalToWorldTransform(prim)
        mesh_low = [float("inf")] * 3
        mesh_high = [float("-inf")] * 3
        for point in points:
            world = matrix.Transform(Gf.Vec3d(point[0], point[1], point[2]))
            for axis in range(3):
                low[axis] = min(low[axis], world[axis])
                high[axis] = max(high[axis], world[axis])
                mesh_low[axis] = min(mesh_low[axis], world[axis])
                mesh_high[axis] = max(mesh_high[axis], world[axis])
        points_used += len(points)
        mesh_instances += 1
        if mesh_limit and len(per_mesh) < mesh_limit:
            per_mesh.append(
                {
                    "path": str(prim.GetPath()),
                    "points": len(points),
                    "world_size": [round(mesh_high[i] - mesh_low[i], 9) for i in range(3)],
                    "world_min": [round(v, 9) for v in mesh_low],
                }
            )

    if mesh_instances == 0:
        return {"mesh_instances": 0, "size": None, "per_mesh_sample": []}
    return {
        "mesh_instances": mesh_instances,
        "points_used": points_used,
        "min": [round(v, 9) for v in low],
        "max": [round(v, 9) for v in high],
        "size": [round(high[i] - low[i], 9) for i in range(3)],
        "per_mesh_sample": per_mesh,
    }


def measure_scene(scene_path, limit_prims: int = 5) -> dict:
    """Describe a composed scene: metadata, prim census, transforms and bbox."""
    from pxr import Usd, UsdGeom, UsdPhysics

    stage = Usd.Stage.Open(str(scene_path))
    if stage is None:
        return {"scene": str(scene_path), "opened": False}

    stage_mpu = UsdGeom.GetStageMetersPerUnit(stage)
    up_axis = str(UsdGeom.GetStageUpAxis(stage))

    type_counts: dict = {}
    prim_count = 0
    for prim in stage.TraverseAll():
        prim_count += 1
        type_name = prim.GetTypeName()
        type_counts[type_name] = type_counts.get(type_name, 0) + 1

    from_points = world_bbox_from_points(stage, mesh_limit=limit_prims)

    cache = UsdGeom.BBoxCache(Usd.TimeCode.Default(), ["default", "render"])
    bound = cache.ComputeWorldBound(stage.GetPseudoRoot()).ComputeAlignedRange()
    bbox_size = None
    if not bound.IsEmpty():
        minimum, maximum = bound.GetMin(), bound.GetMax()
        bbox_size = [round(maximum[i] - minimum[i], 9) for i in range(3)]

    visual_prim = stage.GetPrimAtPath("/MorphOne/Visual")
    transform = UsdGeom.XformCache(Usd.TimeCode.Default()).GetLocalToWorldTransform(visual_prim)
    diagonal = [round(transform[i][i], 9) for i in range(3)]
    authored_scale = []
    if visual_prim:
        for attribute in visual_prim.GetAttributes():
            if attribute.GetName().startswith("xformOp:scale"):
                authored_scale.append([float(v) for v in (attribute.Get() or [])])

    physx = {}
    for api_name in ("CollisionAPI", "RigidBodyAPI", "MassAPI", "ArticulationRootAPI"):
        api_type = getattr(UsdPhysics, api_name)
        physx[api_name] = sum(1 for prim in stage.TraverseAll() if prim.HasAPI(api_type))

    sizes_match = (
        from_points.get("size") is not None
        and bbox_size is not None
        and all(abs(from_points["size"][i] - bbox_size[i]) <= 1e-6 for i in range(3))
    )

    return {
        "scene": str(scene_path),
        "opened": True,
        "stage_meters_per_unit": stage_mpu,
        "up_axis": up_axis,
        "prim_count": prim_count,
        "prim_type_counts": dict(sorted(type_counts.items(), key=lambda kv: -kv[1])[:8]),
        "authored_scale_ops": authored_scale,
        "composed_local_to_world_diagonal": diagonal,
        "world_bbox_from_points_stage_units": from_points.get("size"),
        "world_bbox_from_points_meters": (
            [round(v * stage_mpu, 9) for v in from_points["size"]]
            if from_points.get("size")
            else None
        ),
        "world_bbox_from_cache_stage_units": bbox_size,
        "bbox_methods_agree": sizes_match,
        "mesh_instances": from_points.get("mesh_instances", 0),
        "points_used": from_points.get("points_used", 0),
        "per_mesh_sample": from_points.get("per_mesh_sample", []),
        "physx_apis": physx,
    }


def write_json(path, payload) -> str:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
    return str(path)


def evaluate(report: dict, expected_m: Sequence[float] = DEFAULT_ENVELOPE_M) -> dict:
    """Attach pass/fail decisions to a measurement report."""
    size_m = report.get("world_bbox_from_points_meters")
    checks = {
        "opened": bool(report.get("opened")),
        "bbox_methods_agree": bool(report.get("bbox_methods_agree")),
        "has_meshes": report.get("mesh_instances", 0) > 0,
        "no_physics_apis": all(
            value == 0 for value in (report.get("physx_apis") or {}).values()
        ),
    }
    enriched = dict(report)
    enriched["measured_envelope_m"] = size_m
    enriched["expected_envelope_m"] = list(expected_m)
    enriched["matches_expected_envelope"] = envelope_within(size_m, list(expected_m))
    enriched["checks"] = checks
    return enriched
