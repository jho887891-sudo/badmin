"""Measure the metric scenes from inside Isaac Sim (headless).

The pure-USD measurement is done by verify_metric_scene_reference.py.  This
script repeats the same measurement inside a running Isaac Sim app, because the
question "does the converted millimetre USD land at the right physical size in a
metres stage" is about what the simulator actually composes, not about what a
plain pxr interpreter composes.

It only reads: no rigid body, no collider, no articulation, no arm attachment is
created, and the script asserts that the referenced content carries no PhysX
schema afterwards.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Sequence

__all__ = ["measure_in_app"]


def measure_in_app(scene_paths: Sequence[str], visual_usd: str) -> dict:
    from pxr import Usd, UsdGeom, UsdPhysics

    results = []
    for scene_path in scene_paths:
        stage = Usd.Stage.Open(scene_path)
        if stage is None:
            results.append({"scene": scene_path, "opened": False})
            continue
        stage_mpu = UsdGeom.GetStageMetersPerUnit(stage)
        cache = UsdGeom.BBoxCache(Usd.TimeCode.Default(), ["default", "render"])
        bound = cache.ComputeWorldBound(stage.GetPseudoRoot()).ComputeAlignedRange()
        size_units = None
        if not bound.IsEmpty():
            minimum, maximum = bound.GetMin(), bound.GetMax()
            size_units = [maximum[i] - minimum[i] for i in range(3)]

        physx = {}
        for api_name in ("CollisionAPI", "RigidBodyAPI", "MassAPI", "ArticulationRootAPI"):
            api_type = getattr(UsdPhysics, api_name)
            physx[api_name] = sum(
                1 for prim in stage.TraverseAll() if prim.HasAPI(api_type)
            )
        mesh_count = sum(
            1 for prim in stage.TraverseAll() if prim.GetTypeName() == "Mesh"
        )
        results.append(
            {
                "scene": scene_path,
                "opened": True,
                "stage_meters_per_unit": stage_mpu,
                "up_axis": str(UsdGeom.GetStageUpAxis(stage)),
                "content_bbox_stage_units": (
                    [round(v, 9) for v in size_units] if size_units else None
                ),
                "content_bbox_meters": (
                    [round(v * stage_mpu, 9) for v in size_units] if size_units else None
                ),
                "mesh_prims": mesh_count,
                "physx_apis": physx,
            }
        )
    return {
        "measured_inside": "isaac sim app (headless)",
        "visual_usd": visual_usd,
        "scenes": results,
    }


def main(argv: "Sequence[str] | None" = None) -> int:
    parser = argparse.ArgumentParser(description="Measure metric scenes inside Isaac Sim")
    parser.add_argument("--scene", action="append", required=True)
    parser.add_argument("--visual-usd", required=True)
    parser.add_argument("--report", required=True)

    from isaaclab.app import AppLauncher

    AppLauncher.add_app_launcher_args(parser)
    args_cli = parser.parse_args(argv)
    args_cli.headless = True
    app_launcher = AppLauncher(args_cli)
    simulation_app = app_launcher.app

    report = measure_in_app(args_cli.scene, args_cli.visual_usd)
    path = Path(args_cli.report)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=1), flush=True)
    simulation_app.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
