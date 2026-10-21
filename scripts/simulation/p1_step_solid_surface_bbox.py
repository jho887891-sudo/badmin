"""Kernel-free STEP-side re-measurement of the four candidate wheel solids.

Why this file exists
--------------------
wheel_parts.csv reports, for the four unnamed "NONE_*" wheel-like solids
(STEP eids 435700 / 759491 / 773650 / 871375), a size of
48.653 x 180.782 x 157.783 mm, while the converted USD meshes that the mapping
file points at measure 48.791 x 205.991 x 205.991 mm.  Before anyone calls
either number "the wheel diameter" we have to know *what each number measures*.

step_solid_geometry.vertex_bbox measures the axis-aligned box of the solid's
BREP VERTEX_POINT set only.  A STEP BREP does not tessellate its analytic
surfaces: a cylindrical tyre face is stored as CYLINDRICAL_SURFACE plus two
CIRCLE edges carrying a numeric radius, so the extremity of that face is *not*
a vertex.  A vertex-only box therefore under-reports the true extent of any
solid whose widest feature is a cylindrical face.

This tool brackets the truth instead of guessing it:

    vertex_bbox  <=  true geometric bbox  <=  surface_envelope_bbox

* vertex_bbox - the same measure wheel_parts.csv uses (lower bracket).
* surface_envelope_bbox - vertices, plus the full 360 deg envelope of every
  CIRCLE / ELLIPSE and the axial shell of every CYLINDRICAL_SURFACE, plus every
  B-spline control point (convex-hull bound).  Because an arc may be trimmed,
  this is an *upper* bracket, never a measurement of the solid itself.
* all_points_bbox - every CARTESIAN_POINT in the closure (loosest upper
  bracket; includes LINE origins and placement locations that need not lie on
  the surface).

A number that falls inside the bracket is *consistent* with the solid; it is
still not proof of the physical wheel diameter.  Units are read from the file's
own GLOBAL_UNIT_ASSIGNED_CONTEXT and are reported, never assumed.

Usage (pure CPU, standard library only):

    python scripts/simulation/p1_step_solid_surface_bbox.py
        --step _scratch_downloads/M1-4WD4S_model2601.step
        --eids 435700,759491,773650,871375
        --out-json outputs/simulation/rebot_b601_rs/morph_one/p1_wheel_identity/step_remeasure.json
        --out-csv outputs/simulation/rebot_b601_rs/morph_one/p1_wheel_identity/step_remeasure.csv
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import re
import sys
from typing import Dict, List, Sequence, Tuple

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from scripts.simulation.step_geometry_census import (  # noqa: E402
    SOLID_TYPES,
    entity_name,
    forward_closure,
    parse_step,
)
from scripts.simulation.step_solid_geometry import vertex_bbox  # noqa: E402

_FLOAT_RE = re.compile(r"[-+]?(?:[0-9]+[.]?[0-9]*|[.][0-9]+)(?:[EeDd][-+]?[0-9]+)?")
_NUM = r"[-+]?(?:[0-9]+[.]?[0-9]*|[.][0-9]+)(?:[EeDd][-+]?[0-9]+)?"
_TRIPLE_RE = re.compile(
    r"[(]\s*(" + _NUM + r")\s*,\s*(" + _NUM + r")\s*,\s*(" + _NUM + r")\s*[)]"
)

#: Entity types whose control points bound the surface (convex-hull property).
SPLINE_TYPES = frozenset(
    {
        "B_SPLINE_CURVE_WITH_KNOTS",
        "B_SPLINE_CURVE",
        "RATIONAL_B_SPLINE_CURVE",
        "B_SPLINE_SURFACE_WITH_KNOTS",
        "B_SPLINE_SURFACE",
        "RATIONAL_B_SPLINE_SURFACE",
        "BEZIER_CURVE",
        "BEZIER_SURFACE",
        "QUASI_UNIFORM_CURVE",
    }
)
CIRCULAR_SURFACE_TYPES = frozenset({"CIRCLE", "CYLINDRICAL_SURFACE"})


class Box:
    """Mutable axis-aligned box in file units."""

    __slots__ = ("lo", "hi")

    def __init__(self) -> None:
        self.lo = [math.inf, math.inf, math.inf]
        self.hi = [-math.inf, -math.inf, -math.inf]

    def empty(self) -> bool:
        return any(not math.isfinite(v) for v in self.lo)

    def add_point(self, p: Sequence[float]) -> None:
        for k in range(3):
            if p[k] < self.lo[k]:
                self.lo[k] = float(p[k])
            if p[k] > self.hi[k]:
                self.hi[k] = float(p[k])

    def add_interval(self, axis: int, low: float, high: float) -> None:
        if low < self.lo[axis]:
            self.lo[axis] = float(low)
        if high > self.hi[axis]:
            self.hi[axis] = float(high)

    def merge(self, other):
        if other.empty():
            return
        for k in range(3):
            self.add_interval(k, other.lo[k], other.hi[k])

    def as_dict(self) -> dict:
        if self.empty():
            return {}
        return {
            "min": [round(v, 4) for v in self.lo],
            "max": [round(v, 4) for v in self.hi],
            "size": [round(self.hi[k] - self.lo[k], 4) for k in range(3)],
            "centre": [round((self.hi[k] + self.lo[k]) / 2.0, 4) for k in range(3)],
        }


def _floats(body: str) -> List[float]:
    out = []
    for token in _FLOAT_RE.findall(body):
        try:
            out.append(float(token.replace("D", "E").replace("d", "e")))
        except ValueError:
            continue
    return out


def _unit(values):
    norm = math.sqrt(sum(float(v) * float(v) for v in values))
    if norm <= 0.0 or not math.isfinite(norm):
        return None
    return (values[0] / norm, values[1] / norm, values[2] / norm)


def _point_of(entity):
    if entity is None or entity.etype != "CARTESIAN_POINT":
        return None
    triple = _TRIPLE_RE.search(entity.body)
    if triple is None:
        return None
    return tuple(float(g.replace("D", "E").replace("d", "e")) for g in triple.groups())


def _placement(entities, placement_id):
    """(location point, axis direction, reference direction) or Nones."""
    placement = entities.get(placement_id) if placement_id is not None else None
    if placement is None:
        return None, None, None
    location = None
    dirs = []
    for ref in placement.refs:
        target = entities.get(ref)
        if target is None:
            continue
        if target.etype == "CARTESIAN_POINT" and location is None:
            location = _point_of(target)
        elif target.etype == "DIRECTION":
            triple = _TRIPLE_RE.search(target.body)
            if triple is None:
                continue
            vector = _unit([float(g.replace("D", "E").replace("d", "e")) for g in triple.groups()])
            if vector is not None:
                dirs.append(vector)
    axis = dirs[0] if dirs else None
    ref_dir = dirs[1] if len(dirs) > 1 else None
    return location, axis, ref_dir


def _cross(a: Sequence[float], b: Sequence[float]):
    return (
        a[1] * b[2] - a[2] * b[1],
        a[2] * b[0] - a[0] * b[2],
        a[0] * b[1] - a[1] * b[0],
    )


def _envelope_of_curve(box, centre, normal, radii, ref_dir=None) -> bool:
    """Extend box by the full envelope of a circle / cylinder / ellipse.

    Circle (radii == (R,)):    extent_k = R * sqrt(1 - n_k^2)
    Ellipse (radii == (a, b)): extent_k = sqrt(a^2 d_k^2 + b^2 e_k^2)
                               with d the major direction and e = n x d
    """
    if centre is None or normal is None:
        return False
    if len(radii) == 1:
        radius = radii[0]
        for k in range(3):
            half = abs(radius) * math.sqrt(max(0.0, 1.0 - normal[k] * normal[k]))
            box.add_interval(k, centre[k] - half, centre[k] + half)
        return True
    if len(radii) >= 2:
        a, b = radii[0], radii[1]
        if ref_dir is None:
            return False
        dot = sum(ref_dir[k] * normal[k] for k in range(3))
        major = _unit([ref_dir[k] - dot * normal[k] for k in range(3)])
        if major is None:
            return False
        minor = _unit(_cross(normal, major))
        if minor is None:
            return False
        for k in range(3):
            half = math.sqrt(max(0.0, (a * major[k]) ** 2 + (b * minor[k]) ** 2))
            box.add_interval(k, centre[k] - half, centre[k] + half)
        return True
    return False


def measure_solid(model, solid_id: int, coords) -> dict:
    """Lower bracket, upper bracket and feature inventory for one solid."""
    closure = forward_closure(model.entities, [solid_id])
    entities = model.entities

    vertex_box = Box()
    vertex_count = 0
    for eid in closure:
        entity = entities[eid]
        if entity.etype != "VERTEX_POINT" or not entity.refs:
            continue
        point = coords.get(entity.refs[0])
        if point is None:
            continue
        vertex_box.add_point(point)
        vertex_count += 1

    envelope_box = Box()
    envelope_box.merge(vertex_box)

    all_points_box = Box()
    for eid in closure:
        point = _point_of(entities[eid])
        if point is not None:
            all_points_box.add_point(point)

    circle_radii: List[float] = []
    circle_axes: List[dict] = []
    cylinder_radii: List[float] = []
    ellipse_count = 0
    spline_points = 0
    type_counts: Dict[str, int] = {}

    for eid in sorted(closure):
        entity = entities[eid]
        type_counts[entity.etype] = type_counts.get(entity.etype, 0) + 1
        if entity.etype in CIRCULAR_SURFACE_TYPES:
            values = _floats(entity.body)
            if not values or values[-1] <= 0.0:
                continue
            radius = values[-1]
            placement_id = entity.refs[0] if entity.refs else None
            centre, normal, _ref = _placement(entities, placement_id)
            used = _envelope_of_curve(envelope_box, centre, normal, [radius])
            if entity.etype == "CIRCLE":
                circle_radii.append(radius)
            else:
                cylinder_radii.append(radius)
            if used and entity.etype == "CIRCLE":
                circle_axes.append(
                    {
                        "entity_id": eid,
                        "radius_mm": round(radius, 4),
                        "centre_mm": [round(float(v), 4) for v in centre],
                        "axis": [round(float(v), 6) for v in normal],
                    }
                )
        elif entity.etype == "ELLIPSE":
            values = _floats(entity.body)
            if len(values) >= 2 and values[-1] > 0.0 and values[-2] > 0.0:
                ellipse_count += 1
                placement_id = entity.refs[0] if entity.refs else None
                centre, normal, ref_dir = _placement(entities, placement_id)
                _envelope_of_curve(
                    envelope_box, centre, normal, [values[-2], values[-1]], ref_dir
                )
        elif entity.etype in SPLINE_TYPES:
            for ref in entity.refs:
                point = _point_of(entities.get(ref))
                if point is None:
                    continue
                envelope_box.add_point(point)
                spline_points += 1

    circle_axes.sort(key=lambda item: -item["radius_mm"])
    return {
        "step_eid": solid_id,
        "step_type": entities[solid_id].etype,
        "step_name": entity_name(entities[solid_id]) or "NONE",
        "closure_entities": len(closure),
        "vertex_points": vertex_count,
        "cartesian_points_in_closure": type_counts.get("CARTESIAN_POINT", 0),
        "circle_count": len(circle_radii),
        "cylindrical_surface_count": len(cylinder_radii),
        "ellipse_count": ellipse_count,
        "spline_control_points": spline_points,
        "max_circle_radius_mm": round(max(circle_radii), 4) if circle_radii else None,
        "max_cylinder_radius_mm": round(max(cylinder_radii), 4) if cylinder_radii else None,
        "top_circles": circle_axes[:5],
        "vertex_bbox": vertex_box.as_dict(),
        "surface_envelope_bbox": envelope_box.as_dict(),
        "all_points_bbox": all_points_box.as_dict(),
        "type_counts": dict(sorted(type_counts.items(), key=lambda kv: -kv[1])[:12]),
    }


def read_length_unit(header: str, body: str) -> dict:
    """Report the file's declared length unit instead of assuming millimetres."""
    schema = re.search(r"FILE_SCHEMA\s*[(]\s*[(]\s*'([^']*)'", header)
    return {
        "si_unit_tokens": sorted(set(re.findall(r"SI_UNIT\s*[(]([^)]*)[)]", body)))[:10],
        "conversion_based_units": sorted(
            set(re.findall(r"CONVERSION_BASED_UNIT\s*[(]\s*'([^']*)'", body))
        )[:10],
        "global_unit_context": sorted(
            set(re.findall(r"GLOBAL_UNIT_ASSIGNED_CONTEXT\s*[(]([^)]*)[)]", body))
        )[:5],
        "file_schema": schema.group(1) if schema else "",
    }


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="STEP-side bracket re-measurement")
    parser.add_argument("--step", required=True, help="path to the M1-4WD4S STEP file")
    parser.add_argument(
        "--eids",
        default="435700,759491,773650,871375",
        help="comma separated STEP entity ids of the solids to measure",
    )
    parser.add_argument("--out-json", default="", help="write the full report here")
    parser.add_argument("--out-csv", default="", help="write the comparison table here")
    args = parser.parse_args(argv)

    wanted = [int(token) for token in args.eids.split(",") if token.strip()]
    model = parse_step(args.step)
    coords = dict(zip(model.point_ids, model.points))
    body_text = "".join(entity.to_record() for entity in model.entities.values())

    solids = {}
    for eid in wanted:
        entity = model.entities.get(eid)
        if entity is None or not (set(entity.components) & SOLID_TYPES):
            raise SystemExit("eid %d is not a BREP solid in %s" % (eid, args.step))
        solids[eid] = measure_solid(model, eid, coords)
        published = vertex_bbox(model, eid, coords=coords)
        solids[eid]["vertex_bbox_helper"] = (
            {
                "min": [round(float(v), 4) for v in published["min"]],
                "max": [round(float(v), 4) for v in published["max"]],
                "size": [round(float(v), 4) for v in published["size"]],
                "vertex_points": published["vertex_points"],
            }
            if published
            else None
        )

    report = {
        "tool": "p1_step_solid_surface_bbox.py",
        "step_path": os.path.abspath(args.step),
        "step_size_bytes": model.size_bytes,
        "file_schema": model.file_schema,
        "units": read_length_unit(model.header, body_text),
        "parse_warnings": len(model.parse_warnings),
        "requested_eids": wanted,
        "solids": [solids[eid] for eid in wanted],
        "bracket_meaning": (
            "vertex_bbox is a lower bracket and surface_envelope_bbox an upper bracket of the "
            "true axis-aligned bbox; the true bbox lies between them. Neither is a wheel diameter."
        ),
    }

    print("STEP file      : %s (%d bytes)" % (report["step_path"], model.size_bytes))
    print("file_schema    : %s" % report["file_schema"])
    print("declared units : %s" % json.dumps(report["units"], ensure_ascii=False))
    print("parse warnings : %d" % report["parse_warnings"])
    print("")
    header = "eid      verts  circ  cyl  maxR_mm  vertex bbox (mm)              envelope bbox (mm)"
    print(header)
    print("-" * len(header))
    rows = []
    for solid in report["solids"]:
        vb = solid["vertex_bbox"]["size"]
        eb = solid["surface_envelope_bbox"]["size"]
        print(
            "%-8d %-6d %-5d %-4d %-8s %-30s %s"
            % (
                solid["step_eid"],
                solid["vertex_points"],
                solid["circle_count"],
                solid["cylindrical_surface_count"],
                solid["max_circle_radius_mm"],
                "%.3f x %.3f x %.3f" % tuple(vb),
                "%.3f x %.3f x %.3f" % tuple(eb),
            )
        )
        rows.append(
            {
                "step_eid": solid["step_eid"],
                "step_name": solid["step_name"],
                "vertex_points": solid["vertex_points"],
                "circle_count": solid["circle_count"],
                "cylindrical_surface_count": solid["cylindrical_surface_count"],
                "max_circle_radius_mm": solid["max_circle_radius_mm"],
                "vertex_bbox_x_mm": round(vb[0], 4),
                "vertex_bbox_y_mm": round(vb[1], 4),
                "vertex_bbox_z_mm": round(vb[2], 4),
                "envelope_bbox_x_mm": round(eb[0], 4),
                "envelope_bbox_y_mm": round(eb[1], 4),
                "envelope_bbox_z_mm": round(eb[2], 4),
            }
        )
    print("")

    if args.out_json:
        os.makedirs(os.path.dirname(os.path.abspath(args.out_json)), exist_ok=True)
        with open(args.out_json, "w", encoding="utf-8") as handle:
            json.dump(report, handle, ensure_ascii=False, indent=2)
            handle.write("\n")
        print("wrote %s" % args.out_json)
    if args.out_csv:
        os.makedirs(os.path.dirname(os.path.abspath(args.out_csv)), exist_ok=True)
        fields = [
            "step_eid",
            "step_name",
            "vertex_points",
            "circle_count",
            "cylindrical_surface_count",
            "max_circle_radius_mm",
            "vertex_bbox_x_mm",
            "vertex_bbox_y_mm",
            "vertex_bbox_z_mm",
            "envelope_bbox_x_mm",
            "envelope_bbox_y_mm",
            "envelope_bbox_z_mm",
        ]
        with open(args.out_csv, "w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=fields)
            writer.writeheader()
            for row in rows:
                writer.writerow({key: row[key] for key in fields})
        print("wrote %s" % args.out_csv)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
