"""Audit a converted USD: units, geometry extents, and PhysX contamination.

A CAD->USD conversion is only useful if the result can be trusted, so this tool
re-opens the produced stage and reports facts rather than a log line:

* stage metadata (metersPerUnit, up axis, kilograms per unit),
* prim/mesh/instance counts, including the geometry the converter stores inside
  an abstract "Prototypes" scope (the default Stage.Traverse() skips it),
* per-mesh local extents and the largest absolute local coordinate, which is how
  an extreme outlier in the source CAD would show up,
* whether any PhysX schema (CollisionAPI / RigidBodyAPI / MassAPI /
  ArticulationRootAPI) was applied, i.e. whether unusual geometry could reach
  the physics engine at all.

Requires pxr (ships with Isaac Sim); it does not need a running Kit app.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

__all__ = ["audit_usd"]


def audit_usd(usd_path: str, top_n: int = 8) -> dict:
    from pxr import Usd, UsdGeom, UsdPhysics

    stage = Usd.Stage.Open(usd_path)
    if stage is None:
        return {"opened": False, "usd_path": usd_path}

    cache = UsdGeom.BBoxCache(Usd.TimeCode.Default(), ["default", "render"])
    all_prims = list(stage.TraverseAll())
    meshes = [prim for prim in all_prims if prim.GetTypeName() == "Mesh"]

    rows = []
    max_abs_local = 0.0
    for prim in meshes:
        mesh = UsdGeom.Mesh(prim)
        points = mesh.GetPointsAttr().Get() or []
        if not points:
            continue
        xs = [p[0] for p in points]
        ys = [p[1] for p in points]
        zs = [p[2] for p in points]
        local_extent = [max(xs) - min(xs), max(ys) - min(ys), max(zs) - min(zs)]
        for point in points:
            max_abs_local = max(max_abs_local, abs(point[0]), abs(point[1]), abs(point[2]))
        world_extent = None
        bound = cache.ComputeWorldBound(prim).ComputeAlignedRange()
        if not bound.IsEmpty():
            minimum, maximum = bound.GetMin(), bound.GetMax()
            world_extent = [round(maximum[i] - minimum[i], 3) for i in range(3)]
        rows.append(
            {
                "path": prim.GetPath().pathString,
                "points": len(points),
                "local_extent": [round(v, 3) for v in local_extent],
                "world_extent": world_extent,
            }
        )

    physx = {}
    for api_name in ("CollisionAPI", "RigidBodyAPI", "MassAPI", "ArticulationRootAPI"):
        api_type = getattr(UsdPhysics, api_name)
        physx[api_name] = sum(1 for prim in all_prims if prim.HasAPI(api_type))

    rendered = 0
    for prim in Usd.PrimRange(
        stage.GetPseudoRoot(), Usd.TraverseInstanceProxies(Usd.PrimDefaultPredicate)
    ):
        if prim.GetTypeName() == "Mesh":
            rendered += 1

    ordered = sorted(rows, key=lambda row: -max(row["local_extent"]))
    root_bound = cache.ComputeWorldBound(stage.GetPseudoRoot()).ComputeAlignedRange()
    stage_bbox = None
    if not root_bound.IsEmpty():
        minimum, maximum = root_bound.GetMin(), root_bound.GetMax()
        stage_bbox = {
            "min": [round(v, 3) for v in minimum],
            "max": [round(v, 3) for v in maximum],
            "size": [round(maximum[i] - minimum[i], 3) for i in range(3)],
        }

    # OpenUSD references do not auto-convert length metrics. A STEP-authored
    # millimetre stage with raw coordinates near 580 must be scaled 0.001
    # when referenced into an Isaac Lab metre-scale stage. Merely reading
    # metersPerUnit=0.001 is NOT a guarantee of correct physical dimensions.
    source_mpu = float(UsdGeom.GetStageMetersPerUnit(stage))
    bbox_size_m = (
        [round(v * source_mpu, 9) for v in stage_bbox["size"]]
        if stage_bbox is not None else None
    )

    return {
        "opened": True,
        "usd_path": usd_path,
        "assembly_unit_warning": (
            "USD references do not automatically convert metersPerUnit; "
            "apply scale=source_meters_per_unit/target_meters_per_unit "
            "and verify a composed test scene before adding collisions"
        ),
        "raw_bbox_size_in_source_meters": bbox_size_m,
        "required_xform_scale_for_metre_stage": source_mpu,
        "stage": {
            "meters_per_unit": UsdGeom.GetStageMetersPerUnit(stage),
            "up_axis": UsdGeom.GetStageUpAxis(stage),
            "kilograms_per_unit": stage.GetMetadata("kilogramsPerUnit"),
        },
        "prims": len(all_prims),
        "meshes": len(meshes),
        "rendered_mesh_instances": rendered,
        "instanceable_prims": sum(1 for prim in all_prims if prim.IsInstanceable()),
        "abstract_prims": sum(1 for prim in all_prims if prim.IsAbstract()),
        "total_mesh_points": sum(row["points"] for row in rows),
        "mesh_local_extent_max": (ordered[0]["local_extent"] if ordered else None),
        "mesh_local_extent_min": (ordered[-1]["local_extent"] if ordered else None),
        "max_abs_local_coordinate": round(max_abs_local, 3),
        "meshes_with_local_extent_over_1000": sum(
            1 for row in ordered if max(row["local_extent"]) > 1000.0
        ),
        "physx_apis": physx,
        "stage_world_bbox": stage_bbox,
        "largest_meshes": ordered[:top_n],
        "smallest_meshes": ordered[-3:],
    }


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="Audit a converted USD stage")
    parser.add_argument("--usd", required=True)
    parser.add_argument("--report", default="", help="JSON output path")
    parser.add_argument("--top-n", type=int, default=8)
    args = parser.parse_args(argv)

    report = audit_usd(args.usd, top_n=args.top_n)
    text = json.dumps(report, indent=1)
    if args.report:
        path = Path(args.report)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text + "\n", encoding="utf-8")
    print(text)
    return 0 if report.get("opened") else 1


if __name__ == "__main__":
    raise SystemExit(main())
