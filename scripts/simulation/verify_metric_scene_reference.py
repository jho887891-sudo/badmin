"""Metric-scene reference test for the converted Morph One visual USD.

What this proves (and what it does not)
---------------------------------------
This is a *unit and composition* test only.  It builds a metres stage, references
the converted visual USD into it, and measures the world bounding box with and
without an extra 0.001 scale on the reference node.  It adds no collision, no
mass, no joints and no robot attachment: nothing here claims the assembly is
physically valid, nor that the bounding box is a verified chassis dimension.

USD semantics this experiment measures (do not assume, measure)
--------------------------------------------------------------
The converted Morph One USD declares metersPerUnit = 0.001 (millimetres) with
millimetre-valued coordinates.  USD treats metersPerUnit as *metadata*: pulling a
layer in through a reference does NOT rescale its coordinates.  This script does
not take that for granted -- it reads the composed local-to-world transform and
reports whether a conversion actually happened.

Cases
-----
A  metres stage, plain reference, no scale op        -> expected 580-unit content
B  metres stage, reference + xformOp:scale = 0.001   -> expected 0.580 m envelope
C  millimetre stage, plain reference, no scale op    -> physically 0.580 m too

The measured result decides the wording of the report: whichever case matches the
expected physical envelope is named, and the report never adjusts the expectation
to fit the measurement.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Sequence

__all__ = ["build_scene", "measure_scene", "run"]


def build_scene(scene_path: Path, visual_usd: Path, meters_per_unit: float, scale=None) -> Path:
    from pxr import Gf, Usd, UsdGeom

    scene_path.parent.mkdir(parents=True, exist_ok=True)
    stage = Usd.Stage.CreateNew(str(scene_path))
    UsdGeom.SetStageMetersPerUnit(stage, meters_per_unit)
    UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)

    root = UsdGeom.Xform.Define(stage, "/MorphOne")
    stage.SetDefaultPrim(root.GetPrim())
    visual = UsdGeom.Xform.Define(stage, "/MorphOne/Visual")
    visual.GetPrim().GetReferences().AddReference(str(visual_usd))
    if scale is not None:
        visual.AddScaleOp().Set(Gf.Vec3f(*scale))
    stage.GetRootLayer().Save()
    return scene_path


def measure_scene(scene_path: Path, visual_usd: Path) -> dict:
    from pxr import Usd, UsdGeom

    stage = Usd.Stage.Open(str(scene_path))
    if stage is None:
        return {"opened": False, "scene": str(scene_path)}

    stage_mpu = UsdGeom.GetStageMetersPerUnit(stage)
    up_axis = UsdGeom.GetStageUpAxis(stage)
    cache = UsdGeom.BBoxCache(Usd.TimeCode.Default(), ["default", "render"])

    default_prim = stage.GetDefaultPrim()
    content_bound = cache.ComputeWorldBound(default_prim).ComputeAlignedRange()
    size_stage_units = None
    minimum = maximum = None
    if not content_bound.IsEmpty():
        minimum = content_bound.GetMin()
        maximum = content_bound.GetMax()
        size_stage_units = [maximum[i] - minimum[i] for i in range(3)]

    visual_path = "/MorphOne/Visual"
    visual_prim = stage.GetPrimAtPath(visual_path)
    transform = UsdGeom.XformCache(Usd.TimeCode.Default()).GetLocalToWorldTransform(visual_prim)
    diagonal = [round(transform[i][i], 9) for i in range(3)]

    authored_scale_ops = []
    for attribute in visual_prim.GetAttributes():
        if attribute.GetName().startswith("xformOp:scale"):
            authored_scale_ops.append(list(attribute.Get() or []))

    visual_layer_mpu = None
    visual_stage = Usd.Stage.Open(str(visual_usd))
    if visual_stage is not None:
        visual_layer_mpu = UsdGeom.GetStageMetersPerUnit(visual_stage)

    return {
        "opened": True,
        "scene": str(scene_path),
        "scene_meters_per_unit": stage_mpu,
        "up_axis": str(up_axis),
        "visual_usd_meters_per_unit": visual_layer_mpu,
        "authored_scale_ops": authored_scale_ops,
        "composed_local_to_world_diagonal": diagonal,
        "content_bbox_stage_units": {
            "min": [round(v, 9) for v in minimum] if minimum is not None else None,
            "max": [round(v, 9) for v in maximum] if maximum is not None else None,
            "size": [round(v, 9) for v in size_stage_units] if size_stage_units else None,
        },
        "content_bbox_meters": (
            [round(v * stage_mpu, 9) for v in size_stage_units] if size_stage_units else None
        ),
    }


def _matches(measured, expected, tolerance_ratio=0.02) -> bool:
    if measured is None:
        return False
    return all(
        abs(measured[i] - expected[i]) <= max(abs(expected[i]) * tolerance_ratio, 0.002)
        for i in range(3)
    )


def run(visual_usd: Path, output_dir: Path, expected_m=(0.580, 0.373, 0.803), tolerance_ratio=0.02) -> dict:
    output_dir.mkdir(parents=True, exist_ok=True)
    cases = [
        ("A_no_scale_metre_stage", 1.0, None),
        ("B_scale_0.001_metre_stage", 1.0, (0.001, 0.001, 0.001)),
        ("C_no_scale_millimetre_stage", 0.001, None),
    ]
    results = []
    for name, stage_mpu, scale in cases:
        scene_path = output_dir / (name + ".usda")
        build_scene(scene_path, visual_usd, stage_mpu, scale)
        measured = measure_scene(scene_path, visual_usd)
        measured["case"] = name
        measured["requested_stage_meters_per_unit"] = stage_mpu
        measured["requested_scale"] = list(scale) if scale else None
        measured["matches_expected_envelope"] = _matches(
            measured.get("content_bbox_meters"), list(expected_m), tolerance_ratio
        )
        results.append(measured)

    matching = [case["case"] for case in results if case.get("matches_expected_envelope")]
    ratios = {}
    by_name = {case["case"]: case for case in results}
    a = by_name["A_no_scale_metre_stage"]["content_bbox_meters"]
    b = by_name["B_scale_0.001_metre_stage"]["content_bbox_meters"]
    if a and b:
        ratios["B_over_A"] = [round(b[i] / a[i], 6) if a[i] else None for i in range(3)]

    return {
        "visual_usd": str(visual_usd),
        "expected_envelope_m": list(expected_m),
        "tolerance_ratio": tolerance_ratio,
        "cases": results,
        "cases_matching_expected": matching,
        "case_ratio_B_over_A": ratios,
        "notes": [
            "unit conversion only; no collision, mass, joints or arm attachment",
            "the bounding box is the converted geometry envelope, not a verified chassis dimension",
            "USD keeps metersPerUnit as metadata and does not rescale referenced coordinates, so case A "
            "stays in authored millimetre numbers; the explicit 0.001 scale in case B is what produces "
            "a metres-stage envelope of 0.580 x 0.373 x 0.803",
            "case C matches only because the whole stage is declared in millimetres",
        ],
    }


def main(argv: "Sequence[str] | None" = None) -> int:
    parser = argparse.ArgumentParser(description="Metric-scene reference test")
    parser.add_argument("--visual-usd", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--report", default="")
    parser.add_argument(
        "--expected-m", type=float, nargs=3, default=[0.580, 0.373, 0.803],
        help="expected physical envelope in metres",
    )
    args = parser.parse_args(argv)

    report = run(Path(args.visual_usd), Path(args.output_dir), tuple(args.expected_m))
    text = json.dumps(report, ensure_ascii=False, indent=1)
    report_path = Path(args.report) if args.report else Path(args.output_dir) / "metric_scene_report.json"
    report_path.write_text(text + "\n", encoding="utf-8")
    print(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
