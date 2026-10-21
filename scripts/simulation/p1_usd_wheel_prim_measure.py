"""Read-only pxr measurement of the four candidate wheel prims (CPU only).

Context
-------
outputs/simulation/rebot_b601_rs/morph_one/physx_validation/orientation_plus90.json
reports wheel_solids bbox 48.791 x 205.991 x 205.991 mm for the four STEP eids
435700 / 759491 / 773650 / 871375.  This tool re-measures those numbers from the
converted asset with pxr alone and records, for each prim:

  * stage metersPerUnit and upAxis, so "mm" is a *derived* statement, not a guess
  * the prim's own point count, face count and shell (connected component) count
  * the axis-aligned bbox of the mesh points in PRIM space and in WORLD space,
    plus the same box scaled by metersPerUnit
  * the disc axis (smallest-extent axis) and the radial statistics of the point
    cloud about that axis, which is what separates "tyre outer radius" from
    "hub radius" or "merged multi-entity mesh"

No GPU is touched: this only uses pxr (Usd / UsdGeom / Gf).  Nothing is written
back to the asset.

Usage (on the GPU host, CPU only):

    /home/T7/ojh/robot_sim/env_isaaclab/bin/python \
        p1_usd_wheel_prim_measure.py \
        --usda /home/T7/dgut/runs/cad_dev/usd/M1-4WD4S_model2601_dev.usda \
        --mapping /home/dgut/runs/wheels/mapping.json \
        --eids 435700,759491,773650,871375 \
        --out-json p1_usd_wheel_prim_measure.json \
        --out-csv p1_usd_wheel_prim_measure.csv
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import sys

from pxr import Gf, Usd, UsdGeom


def _mapping_pairs(path):
    with open(path, "r", encoding="utf-8") as handle:
        data = json.load(handle)
    if isinstance(data, dict):
        for key in ("pairs", "mapping", "entries", "parts"):
            if key in data and isinstance(data[key], list):
                return data[key]
        raise SystemExit("no list of pairs found in %s (keys: %s)" % (path, list(data.keys())))
    if isinstance(data, list):
        return data
    raise SystemExit("unsupported mapping layout in %s" % path)


def _eid_of(entry):
    for key in ("step_eid", "eid", "step_id", "id"):
        if key in entry:
            try:
                return int(entry[key])
            except (TypeError, ValueError):
                continue
    return None


def _component_labels(points, counts, indices):
    """Vertex-connectivity component label per point (union-find, no welding)."""
    parent = list(range(len(points)))

    def find(a):
        while parent[a] != a:
            parent[a] = parent[parent[a]]
            a = parent[a]
        return a

    def union(a, b):
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[rb] = ra

    cursor = 0
    for count in counts:
        group = indices[cursor : cursor + int(count)]
        cursor += int(count)
        if not group:
            continue
        first = int(group[0])
        for other in group[1:]:
            union(first, int(other))
    labels = [find(i) for i in range(len(points))]
    return labels


def _component_stats(points, labels, meters_per_unit, outer_radius_mm, keep=5):
    """Per-shell vertex counts and boxes, in mm, plus how many reach the rim."""
    groups = {}
    for index, label in enumerate(labels):
        groups.setdefault(label, []).append(index)
    entries = []
    reaches = 0
    for label, members in groups.items():
        lo = [math.inf] * 3
        hi = [-math.inf] * 3
        for index in members:
            p = points[index]
            for k in range(3):
                if p[k] < lo[k]:
                    lo[k] = p[k]
                if p[k] > hi[k]:
                    hi[k] = p[k]
        size_mm = [(hi[k] - lo[k]) * meters_per_unit * 1000.0 for k in range(3)]
        if max(size_mm) >= outer_radius_mm:
            reaches += 1
        entries.append({"vertex_count": len(members), "bbox_size_mm": [round(v, 4) for v in size_mm]})
    entries.sort(key=lambda item: -item["vertex_count"])
    counts = sorted(entry["vertex_count"] for entry in entries)
    def pct(values, q):
        if not values:
            return None
        pos = (len(values) - 1) * q
        low = int(math.floor(pos))
        high = int(math.ceil(pos))
        if low == high:
            return values[low]
        return values[low] * (1.0 - (pos - low)) + values[high] * (pos - low)
    return {
        "count": len(entries),
        "vertex_count_min": counts[0] if counts else None,
        "vertex_count_median": pct(counts, 0.5),
        "vertex_count_max": counts[-1] if counts else None,
        "components_spanning_outer_diameter": reaches,
        "largest_components": entries[:keep],
        "note": (
            "vertex-connected components of an unwelded tessellation are BREP face patches, "
            "not physical parts; a component count above 1 is therefore not proof of a merged mesh"
        ),
    }


def _bbox_of(points):
    lo = [math.inf] * 3
    hi = [-math.inf] * 3
    for p in points:
        for k in range(3):
            if p[k] < lo[k]:
                lo[k] = p[k]
            if p[k] > hi[k]:
                hi[k] = p[k]
    return lo, hi


def _round_box(lo, hi, scale=1.0):
    return {
        "min": [round(lo[k] * scale, 6) for k in range(3)],
        "max": [round(hi[k] * scale, 6) for k in range(3)],
        "size": [round((hi[k] - lo[k]) * scale, 6) for k in range(3)],
    }


def measure_prim(stage, usd_path, meters_per_unit, xform_cache, bbox_cache):
    prim = stage.GetPrimAtPath(usd_path)
    result = {
        "usd_path": usd_path,
        "prim_valid": bool(prim),
        "type_name": prim.GetTypeName() if prim else "",
    }
    if not prim:
        return result
    if prim.GetTypeName() != "Mesh":
        result["note"] = "prim exists but is not a UsdGeom.Mesh; nothing measured"
        return result

    mesh = UsdGeom.Mesh(prim)
    points_attr = mesh.GetPointsAttr()
    points = points_attr.Get() if points_attr else None
    if not points:
        result["note"] = "mesh has no points"
        return result

    counts = list(mesh.GetFaceVertexCountsAttr().Get() or [])
    indices = list(mesh.GetFaceVertexIndicesAttr().Get() or [])

    local = [tuple(float(v) for v in p) for p in points]

    matrix = xform_cache.GetLocalToWorldTransform(prim)
    world = [matrix.Transform(Gf.Vec3d(p[0], p[1], p[2])) for p in local]
    world = [tuple(float(v) for v in p) for p in world]

    llo, lhi = _bbox_of(local)
    wlo, whi = _bbox_of(world)

    bound = bbox_cache.ComputeWorldBound(prim).ComputeAlignedRange()
    bmin = bound.GetMin()
    bmax = bound.GetMax()

    extent_attr = mesh.GetExtentAttr()
    extent = extent_attr.Get() if extent_attr and extent_attr.HasAuthoredValue() else None

    # disc axis = smallest-extent axis of the world box
    world_size = [whi[k] - wlo[k] for k in range(3)]
    axis = min(range(3), key=lambda k: world_size[k])
    plane = [k for k in range(3) if k != axis]
    centre = [(whi[k] + wlo[k]) / 2.0 for k in range(3)]
    radii = []
    for p in world:
        dy = p[plane[0]] - centre[plane[0]]
        dz = p[plane[1]] - centre[plane[1]]
        radii.append(math.sqrt(dy * dy + dz * dz))
    radii.sort()

    def pct(values, q):
        if not values:
            return None
        pos = (len(values) - 1) * q
        low = int(math.floor(pos))
        high = int(math.ceil(pos))
        if low == high:
            return values[low]
        return values[low] * (1.0 - (pos - low)) + values[high] * (pos - low)

    outer = sum(1 for r in radii if r >= radii[-1] - 0.5)

    labels = _component_labels(local, counts, indices)
    component_stats = _component_stats(
        world,
        labels,
        meters_per_unit,
        outer_radius_mm=radii[-1] * meters_per_unit * 1000.0,
    )

    result.update(
        {
            "point_count": len(local),
            "face_count": len(counts),
            "face_vertex_index_count": len(indices),
            "shell_count_vertex_connectivity": component_stats["count"],
            "components": component_stats,
            "authored_extent_stage_units": (
                [[round(float(v), 6) for v in extent[0]], [round(float(v), 6) for v in extent[1]]]
                if extent
                else None
            ),
            "prim_to_world_matrix": [list(row) for row in matrix],
            "local_bbox_stage_units": _round_box(llo, lhi),
            "world_bbox_stage_units": _round_box(wlo, whi),
            "world_bbox_mm": _round_box(wlo, whi, meters_per_unit * 1000.0),
            "world_bbox_m": _round_box(wlo, whi, meters_per_unit),
            "pxr_compute_world_bound_mm": {
                "min": [round(float(bmin[k]) * meters_per_unit * 1000.0, 6) for k in range(3)],
                "max": [round(float(bmax[k]) * meters_per_unit * 1000.0, 6) for k in range(3)],
                "size": [
                    round(float(bmax[k] - bmin[k]) * meters_per_unit * 1000.0, 6) for k in range(3)
                ],
            },
            "disc_axis_index": axis,
            "disc_axis": "xyz"[axis],
            "radial_stats_world_mm": {
                "min": round(radii[0] * meters_per_unit * 1000.0, 6),
                "p50": round(pct(radii, 0.5) * meters_per_unit * 1000.0, 6),
                "p90": round(pct(radii, 0.9) * meters_per_unit * 1000.0, 6),
                "max": round(radii[-1] * meters_per_unit * 1000.0, 6),
                "points_within_0p5mm_of_max": outer,
            },
            "outer_diameter_from_points_mm": round(2.0 * radii[-1] * meters_per_unit * 1000.0, 6),
            "local_to_world_scale_ratio": [
                round(
                    (whi[k] - wlo[k]) / (lhi[k] - llo[k]), 9
                )
                if (lhi[k] - llo[k]) > 0
                else None
                for k in range(3)
            ],
        }
    )
    return result


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="pxr re-measurement of candidate wheel prims")
    parser.add_argument("--usda", required=True)
    parser.add_argument("--mapping", required=True)
    parser.add_argument("--eids", default="435700,759491,773650,871375")
    parser.add_argument("--out-json", default="")
    parser.add_argument("--out-csv", default="")
    args = parser.parse_args(argv)

    wanted = [int(t) for t in args.eids.split(",") if t.strip()]
    pairs = _mapping_pairs(args.mapping)
    by_eid = {}
    for entry in pairs:
        eid = _eid_of(entry)
        if eid is not None:
            by_eid.setdefault(eid, entry)

    stage = Usd.Stage.Open(args.usda)
    if not stage:
        raise SystemExit("could not open %s" % args.usda)
    meters_per_unit = UsdGeom.GetStageMetersPerUnit(stage)
    up_axis = UsdGeom.GetStageUpAxis(stage)
    xform_cache = UsdGeom.XformCache(Usd.TimeCode.Default())
    bbox_cache = UsdGeom.BBoxCache(Usd.TimeCode.Default(), ["default", "render"])

    print("usda                 : %s" % os.path.abspath(args.usda))
    print("stage metersPerUnit  : %r" % meters_per_unit)
    print("stage upAxis         : %s" % up_axis)
    print("mapping pairs        : %d (unique eids %d)" % (len(pairs), len(by_eid)))
    print("")

    solids = []
    for eid in wanted:
        entry = by_eid.get(eid) or {}
        usd_path = entry.get("usd_path") or ""
        record = {
            "step_eid": eid,
            "usd_path": usd_path,
            "mapping_entry_keys": sorted(entry.keys()),
            "mapping_grade": entry.get("grade") or entry.get("usd_mapping_grade"),
        }
        if not usd_path:
            record["note"] = "no usd_path for this eid in the mapping file"
            solids.append(record)
            continue
        measured = measure_prim(stage, usd_path, meters_per_unit, xform_cache, bbox_cache)
        record.update(measured)
        solids.append(record)

    report = {
        "tool": "p1_usd_wheel_prim_measure.py",
        "usda": os.path.abspath(args.usda),
        "mapping": os.path.abspath(args.mapping),
        "stage_meters_per_unit": meters_per_unit,
        "stage_up_axis": str(up_axis),
        "pxr_version": list(Usd.GetVersion()),
        "requested_eids": wanted,
        "solids": solids,
        "bbox_definition": (
            "world_bbox_* is the axis-aligned box of the mesh point set transformed by the prim's "
            "local-to-world matrix; pxr_compute_world_bound_mm is UsdGeom.BBoxCache for cross-check"
        ),
    }

    header = "eid      pts     faces  shells  world bbox (mm)                  r_max_mm  axis"
    print(header)
    print("-" * len(header))
    rows = []
    for solid in solids:
        bbox = solid.get("world_bbox_mm") or {}
        size = bbox.get("size") or [None, None, None]
        radii = solid.get("radial_stats_world_mm") or {}
        print(
            "%-8d %-7d %-6d %-7d %-32s %-9s %s"
            % (
                solid["step_eid"],
                solid.get("point_count", -1),
                solid.get("face_count", -1),
                solid.get("shell_count_vertex_connectivity", -1),
                "%.3f x %.3f x %.3f" % tuple(size),
                radii.get("max"),
                solid.get("disc_axis"),
            )
        )
        rows.append(
            {
                "step_eid": solid["step_eid"],
                "usd_path": solid.get("usd_path", ""),
                "prim_valid": solid.get("prim_valid"),
                "type_name": solid.get("type_name", ""),
                "point_count": solid.get("point_count"),
                "face_count": solid.get("face_count"),
                "shell_count": solid.get("shell_count_vertex_connectivity"),
                "bbox_x_mm": round(size[0], 6) if size[0] is not None else None,
                "bbox_y_mm": round(size[1], 6) if size[1] is not None else None,
                "bbox_z_mm": round(size[2], 6) if size[2] is not None else None,
                "disc_axis": solid.get("disc_axis"),
                "r_min_mm": radii.get("min"),
                "r_p50_mm": radii.get("p50"),
                "r_p90_mm": radii.get("p90"),
                "r_max_mm": radii.get("max"),
                "outer_diameter_mm": solid.get("outer_diameter_from_points_mm"),
                "points_within_0p5mm_of_max": radii.get("points_within_0p5mm_of_max"),
            }
        )
    print("")

    if args.out_json:
        os.makedirs(os.path.dirname(os.path.abspath(args.out_json)), exist_ok=True)
        with open(args.out_json, "w", encoding="utf-8") as handle:
            json.dump(report, handle, ensure_ascii=False, indent=2, default=str)
            handle.write("\n")
        print("wrote %s" % args.out_json)
    if args.out_csv:
        os.makedirs(os.path.dirname(os.path.abspath(args.out_csv)), exist_ok=True)
        fields = [
            "step_eid",
            "usd_path",
            "prim_valid",
            "type_name",
            "point_count",
            "face_count",
            "shell_count",
            "bbox_x_mm",
            "bbox_y_mm",
            "bbox_z_mm",
            "disc_axis",
            "r_min_mm",
            "r_p50_mm",
            "r_p90_mm",
            "r_max_mm",
            "outer_diameter_mm",
            "points_within_0p5mm_of_max",
        ]
        with open(args.out_csv, "w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=fields)
            writer.writeheader()
            for row in rows:
                writer.writerow({key: row.get(key) for key in fields})
        print("wrote %s" % args.out_csv)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
