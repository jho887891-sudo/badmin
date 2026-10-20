"""B3: does scale 0.001 + a +90 deg X rotation turn the CAD Y-up chassis into a
Z-up world pose, and where do the four candidate wheel solids end up?

Read-only: the source asset is opened, never written.  Numbers are measurements of
the converted asset, not vendor statements about the real coordinate system.

Usage:
    python morph_one_orientation_probe.py --chassis-usd <usda> --mapping <json> --out <json>
"""

from __future__ import annotations

import argparse
import json
import math
import os

from pxr import Gf, Usd, UsdGeom

# STEP solids that sit on the four candidate drive axes (workstream B evidence)
WHEEL_EIDS = (435700, 759491, 773650, 871375)
SCALE = 0.001


def rot_x(vec, degrees):
    rad = math.radians(degrees)
    c, s = math.cos(rad), math.sin(rad)
    x, y, z = vec
    return [x, c * y - s * z, s * y + c * z]


def analyse(usd_path, mapping_path, degrees):
    stage = Usd.Stage.Open(usd_path)
    if stage is None:
        raise SystemExit("cannot open %s" % usd_path)
    cache = UsdGeom.BBoxCache(Usd.TimeCode.Default(), ["default", "render"])
    default_prim = stage.GetDefaultPrim()

    def world_bbox(prim):
        box = cache.ComputeWorldBound(prim).ComputeAlignedRange()
        return [list(box.GetMin()), list(box.GetMax())]

    whole = world_bbox(default_prim)

    pairs = {}
    try:
        with open(mapping_path, encoding="utf-8") as handle:
            for pair in json.load(handle).get("pairs", []):
                pairs[int(pair["step_eid"])] = pair.get("usd_path")
    except FileNotFoundError:
        pairs = {}

    rows = []
    for eid in WHEEL_EIDS:
        path = pairs.get(eid)
        prim = stage.GetPrimAtPath(path) if path else None
        if not prim or not prim.IsValid():
            rows.append({"step_eid": eid, "usd_path": path, "found": False})
            continue
        low, high = world_bbox(prim)
        centre = [(low[i] + high[i]) / 2.0 for i in range(3)]
        size = [high[i] - low[i] for i in range(3)]
        scaled_centre = [v * SCALE for v in centre]
        rows.append({
            "step_eid": eid,
            "usd_path": path,
            "found": True,
            "cad_centre_mm": [round(v, 3) for v in centre],
            "cad_bbox_size_mm": [round(v, 3) for v in size],
            "world_centre_m_after_scale": [round(v, 6) for v in scaled_centre],
            "world_centre_m_after_scale_and_rotation": [round(v, 6) for v in rot_x(scaled_centre, degrees)],
            "world_low_z_m_after_scale_and_rotation": round(
                min(rot_x([low[0] * SCALE, low[1] * SCALE, low[2] * SCALE], degrees)[2],
                    rot_x([high[0] * SCALE, high[1] * SCALE, high[2] * SCALE], degrees)[2]), 6),
        })

    corners = [[whole[0][0], whole[0][1], whole[0][2]], [whole[1][0], whole[1][1], whole[1][2]]]
    corners += [[whole[0][0], whole[1][1], whole[0][2]], [whole[0][0], whole[0][1], whole[1][2]]]
    rotated = [rot_x([c[0] * SCALE, c[1] * SCALE, c[2] * SCALE], degrees) for c in corners]
    rot_min = [min(v[i] for v in rotated) for i in range(3)]
    rot_max = [max(v[i] for v in rotated) for i in range(3)]

    return {
        "chassis_usd": os.path.abspath(usd_path),
        "mapping": os.path.abspath(mapping_path),
        "asset_meters_per_unit": UsdGeom.GetStageMetersPerUnit(stage),
        "asset_up_axis": UsdGeom.GetStageUpAxis(stage),
        "applied_scale": SCALE,
        "applied_rotation_deg_about_x": degrees,
        "cad_bbox_mm": {"min": [round(v, 3) for v in whole[0]],
                        "max": [round(v, 3) for v in whole[1]],
                        "size": [round(whole[1][i] - whole[0][i], 3) for i in range(3)]},
        "cad_most_negative_axis": "xyz"[min(range(3), key=lambda i: whole[0][i])],
        "up_axis_by_smallest_extent": "xyz"[min(
            range(3), key=lambda i: whole[1][i] - whole[0][i])],
        "world_up_axis_by_smallest_extent": "xyz"[min(
            range(3), key=lambda i: rot_max[i] - rot_min[i])],
        "world_bbox_m": {"min": [round(v, 6) for v in rot_min],
                         "max": [round(v, 6) for v in rot_max],
                         "size": [round(rot_max[i] - rot_min[i], 6) for i in range(3)]},
        "world_bbox_size_permuted": [
            round(rot_max[i] - rot_min[i], 6) for i in range(3)
        ] != [round((whole[1][i] - whole[0][i]) * SCALE, 6) for i in range(3)],
        "wheel_solids": rows,
        "note": ("an up axis and a rotation sign inferred from CAD are CANDIDATE conventions; "
                 "they are not the vendor's declared robot frame"),
    }


def main(argv=None):
    parser = argparse.ArgumentParser(description="Orientation probe for the converted chassis")
    parser.add_argument("--chassis-usd", required=True)
    parser.add_argument("--mapping", default="")
    parser.add_argument("--degrees", type=float, default=90.0)
    parser.add_argument("--out", default="")
    args = parser.parse_args(argv)
    report = analyse(args.chassis_usd, args.mapping, args.degrees)
    text = json.dumps(report, ensure_ascii=False, indent=1)
    if args.out:
        directory = os.path.dirname(os.path.abspath(args.out))
        if directory:
            os.makedirs(directory, exist_ok=True)
        with open(args.out, "w", encoding="utf-8") as handle:
            handle.write(text)
    print(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
