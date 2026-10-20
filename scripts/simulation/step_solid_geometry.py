"""Per-solid geometry facts extracted from a STEP file without a CAD kernel.

Why this exists
---------------
The census tool answers "what entities are in the file" and "which solid owns a
suspicious point".  This tool answers the next question: how big is each part,
which group does it belong to, and what round features does it carry?

It is deliberately conservative:

* extents are measured from BREP VERTEX_POINTs only, never from LINE origins or
  B-spline control points (those may legally sit far away from the geometry),
* every classification is marked as a *hint* derived from a stated ratio rule,
* parts whose name is not exported are reported as UNKNOWN instead of guessed,
* no wheel radius, steer axis, wheelbase, mass or inertia is invented here.

Units: values are reported in the file's own length unit (millimetres for the
Morph One export) and never silently rescaled.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import re
import statistics
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

from scripts.simulation.step_geometry_census import (
    SOLID_TYPES,
    Entity,
    StepModel,
    entity_name,
    forward_closure,
    parse_step,
)

__all__ = [
    "vertex_bbox",
    "shape_hint",
    "part_group",
    "part_root",
    "circle_facts",
    "shape_fingerprint",
    "solid_geometry",
    "build_report",
]

_FLOAT_RE = re.compile(r"[-+]?(?:[0-9]+\.?[0-9]*|\.[0-9]+)(?:[EeDd][-+]?[0-9]+)?")
# Reuse the STEP numeric token (including E/D scientific notation) for ALL
# three vector components.  Otherwise SolidWorks DIRECTION values with a
# near-zero component such as -4.789422297650225240E-14 silently disappear.
_TRIPLE_RE = re.compile(
    r"\(\s*(" + _FLOAT_RE.pattern + r")\s*,\s*("
    + _FLOAT_RE.pattern + r")\s*,\s*(" + _FLOAT_RE.pattern + r")\s*\)"
)

SHAPE_DISC = "disc_like"
SHAPE_SLAB = "slab_like"
SHAPE_ROD = "rod_like"
SHAPE_BLOCK = "block_like"
SHAPE_UNKNOWN = "unknown"

UNKNOWN_NAME = "NONE"


def _coords(model: StepModel) -> Dict[int, Tuple[float, float, float]]:
    return dict(zip(model.point_ids, model.points))


def _direction_vector(entity: Entity) -> "Tuple[float, float, float] | None":
    """DIRECTION('',(x,y,z)) -> normalised tuple, or None."""
    match = _TRIPLE_RE.search(entity.body)
    if match is None:
        return None
    values = [float(g.replace("D", "E").replace("d", "e")) for g in match.groups()]
    scale = max(abs(v) for v in values)
    if scale == 0:
        return None
    return tuple(round(v / scale, 3) for v in values)  # type: ignore[return-value]


def _last_float(body: str) -> "float | None":
    matches = _FLOAT_RE.findall(body)
    if not matches:
        return None
    try:
        return float(matches[-1].replace("D", "E").replace("d", "e"))
    except ValueError:
        return None


def vertex_bbox(
    model: StepModel,
    solid_id: int,
    coords: "Dict[int, Tuple[float, float, float]] | None" = None,
) -> "dict | None":
    """Axis-aligned bbox measured from the solid's BREP vertices only.

    Vertices are the only points guaranteed to lie on the model surface, so this
    is the honest kernel-free extent: LINE origins and B-spline control points
    are excluded by construction.
    """
    if coords is None:
        coords = _coords(model)
    closure = forward_closure(model.entities, [solid_id])
    xs: List[float] = []
    ys: List[float] = []
    zs: List[float] = []
    for eid in closure:
        entity = model.entities[eid]
        if entity.etype != "VERTEX_POINT" or not entity.refs:
            continue
        point = coords.get(entity.refs[0])
        if point is None:
            continue
        xs.append(point[0])
        ys.append(point[1])
        zs.append(point[2])
    if not xs:
        return None
    return {
        "min": [min(xs), min(ys), min(zs)],
        "max": [max(xs), max(ys), max(zs)],
        "size": [max(xs) - min(xs), max(ys) - min(ys), max(zs) - min(zs)],
        "vertex_points": len(xs),
    }


def shape_hint(size: Sequence[float]) -> dict:
    """Ratio-based hint, never a claim: disc / slab / rod / block.

    The rule is stated so a reviewer can disagree with it.  With extents sorted
    descending as a >= b >= c:

      disc_like : b/a >= 0.80 and c/a <= 0.35   (round plate: two similar, one thin)
      rod_like  : b/a <= 0.60 and c/a <= 0.60   (long and thin in two directions)
      slab_like : c/a <= 0.35                   (flat but not round)
      block_like: otherwise
    """
    values = sorted((abs(float(v)) for v in size), reverse=True)
    largest, middle, smallest = values[0], values[1], values[2]
    if largest <= 0:
        return {"hint": SHAPE_UNKNOWN, "ratios": {"b_over_a": 0.0, "c_over_a": 0.0}}
    ratios = {"b_over_a": round(middle / largest, 4), "c_over_a": round(smallest / largest, 4)}
    if ratios["b_over_a"] >= 0.80 and ratios["c_over_a"] <= 0.35:
        hint = SHAPE_DISC
    elif ratios["b_over_a"] <= 0.60 and ratios["c_over_a"] <= 0.60:
        hint = SHAPE_ROD
    elif ratios["c_over_a"] <= 0.35:
        hint = SHAPE_SLAB
    else:
        hint = SHAPE_BLOCK
    return {"hint": hint, "ratios": ratios}


#: Segment of an assembly path that names the physical module.
#: "底盘模块-1/舵轮前组件-2/舵轮转动组件-2/减震器-1/part" -> 舵轮前组件-2
#: while a flat "主体模块-1/part" keeps 主体模块-1.
MODULE_SEGMENT_INDEX = 0
DEEP_MODULE_SEGMENT_INDEX = 1


def part_group(name: str) -> str:
    """Structural group a part belongs to, taken from its exported assembly path.

    SolidWorks writes "root/module/sub/module/part".  Two shapes occur in the
    Morph One export: a flat "root/part" and a deep "root/module/.../part".  The
    module is therefore segment 1 for a deep path and segment 0 for a flat one.
    A part whose name was not exported yields UNKNOWN.
    """
    if not name or name.strip() == UNKNOWN_NAME:
        return "UNKNOWN"
    segments = [segment.strip() for segment in name.split("/") if segment.strip()]
    if len(segments) <= 1:
        return "UNKNOWN"
    if len(segments) >= 3:
        return segments[DEEP_MODULE_SEGMENT_INDEX]
    return segments[MODULE_SEGMENT_INDEX]


def part_root(name: str) -> str:
    """Top-level assembly (e.g. 'M1.2-00V3 底盘模块-1') or UNKNOWN."""
    if not name or name.strip() == UNKNOWN_NAME:
        return "UNKNOWN"
    segments = [segment.strip() for segment in name.split("/") if segment.strip()]
    if len(segments) <= 1:
        return "UNKNOWN"
    return segments[0]


def circle_facts(model: StepModel, solid_id: int) -> dict:
    """Circular features of a solid: count, radius statistics, dominant axis."""
    closure = forward_closure(model.entities, [solid_id])
    radii: List[float] = []
    axes: Counter = Counter()
    for eid in closure:
        entity = model.entities[eid]
        if entity.etype != "CIRCLE":
            continue
        radius = _last_float(entity.body)
        if radius is not None and radius > 0:
            radii.append(radius)
        if entity.refs:
            placement = model.entities.get(entity.refs[0])
            if placement is not None:
                # Try every direction the placement references: the first
                # DIRECTION is the axis, but a degenerate one must not mask a
                # valid reference direction further along.
                for ref in placement.refs:
                    target = model.entities.get(ref)
                    if target is None or target.etype != "DIRECTION":
                        continue
                    vector = _direction_vector(target)
                    if vector is not None:
                        axes["(%g,%g,%g)" % vector] += 1
                        break
    facts: dict = {"circles": len(radii)}
    if radii:
        facts["circle_radius_mm"] = {
            "min": round(min(radii), 4),
            "median": round(statistics.median(radii), 4),
            "max": round(max(radii), 4),
        }
    if axes:
        facts["dominant_circle_axes"] = axes.most_common(3)
    return facts


def shape_fingerprint(
    model: StepModel,
    solid_id: int,
    coords: "Dict[int, Tuple[float, float, float]] | None" = None,
    tolerance: float = 0.1,
) -> dict:
    """A vertex-and-circle *candidate* duplicate fingerprint, not a shape proof.

    Coordinates are shifted by the bbox centre then quantized to tolerance,
    and sorted circle radii are included. This is translation-invariant but
    NOT rotation-invariant, and different topology may share the same
    quantized vertex/radius multiset. Equal hashes flag likely repeats for
    follow-up CAD kernel verification; unequal hashes do not disprove a rotated
    or mirrored copy. Do not call this a BREP equality certificate.
    """
    if coords is None:
        coords = _coords(model)
    bbox = vertex_bbox(model, solid_id, coords=coords)
    closure = forward_closure(model.entities, [solid_id])
    centre = (
        [(bbox["min"][i] + bbox["max"][i]) / 2.0 for i in range(3)] if bbox else [0.0, 0.0, 0.0]
    )
    vertices = []
    radii = []
    for eid in sorted(closure):
        entity = model.entities[eid]
        if entity.etype == "VERTEX_POINT" and entity.refs:
            point = coords.get(entity.refs[0])
            if point is None:
                continue
            vertices.append(
                tuple(round((point[i] - centre[i]) / tolerance) for i in range(3))
            )
        elif entity.etype == "CIRCLE":
            radius = _last_float(entity.body)
            if radius is not None:
                radii.append(round(radius / tolerance))
    vertices.sort()
    radii.sort()
    digest = hashlib.sha256(repr((vertices, radii)).encode("utf-8")).hexdigest()
    return {
        "fingerprint": digest[:16],
        "vertices": len(vertices),
        "circles": len(radii),
    }


def solid_geometry(model: StepModel, max_solids: int = 0) -> List[dict]:
    coords = _coords(model)
    solids = [
        entity
        for entity in model.entities.values()
        if set(entity.components) & SOLID_TYPES
    ]
    solids.sort(key=lambda e: e.eid)
    if max_solids:
        solids = solids[:max_solids]

    rows: List[dict] = []
    for solid in solids:
        name = entity_name(solid)
        bbox = vertex_bbox(model, solid.eid, coords=coords)
        closure = forward_closure(model.entities, [solid.eid])
        row = {
            "eid": solid.eid,
            "type": solid.etype,
            "name": name,
            "name_leaf": name.split("/")[-1].strip() if name else "",
            "part_group": part_group(name),
            "part_root": part_root(name),
            "named": bool(name and name.strip() != UNKNOWN_NAME),
            "closure_entities": len(closure),
            "cell_points": sum(
                1 for eid in closure if model.entities[eid].etype == "CARTESIAN_POINT"
            ),
            "vertex_points": sum(
                1 for eid in closure if model.entities[eid].etype == "VERTEX_POINT"
            ),
        }
        if bbox is not None:
            row["vertex_bbox_mm"] = {
                "min": [round(v, 3) for v in bbox["min"]],
                "max": [round(v, 3) for v in bbox["max"]],
                "size": [round(v, 3) for v in bbox["size"]],
            }
            row["shape"] = shape_hint(bbox["size"])
        else:
            row["vertex_bbox_mm"] = None
            row["shape"] = {"hint": SHAPE_UNKNOWN, "ratios": {}}
        row.update(circle_facts(model, solid.eid))
        row["shape_fingerprint"] = shape_fingerprint(model, solid.eid, coords=coords)
        rows.append(row)
    return rows


def _union_bbox(rows: Sequence[dict]) -> "dict | None":
    boxes = [row["vertex_bbox_mm"] for row in rows if row.get("vertex_bbox_mm")]
    if not boxes:
        return None
    minimum = [min(b["min"][i] for b in boxes) for i in range(3)]
    maximum = [max(b["max"][i] for b in boxes) for i in range(3)]
    return {
        "min": [round(v, 3) for v in minimum],
        "max": [round(v, 3) for v in maximum],
        "size": [round(maximum[i] - minimum[i], 3) for i in range(3)],
    }


def build_report(model: StepModel, max_solids: int = 0) -> dict:
    rows = solid_geometry(model, max_solids=max_solids)
    groups: Dict[str, List[dict]] = defaultdict(list)
    for row in rows:
        groups[row["part_group"]].append(row)

    group_rows = []
    for group, members in sorted(groups.items(), key=lambda kv: -len(kv[1])):
        union = _union_bbox(members)
        shapes = Counter(member["shape"]["hint"] for member in members)
        group_rows.append(
            {
                "part_group": group,
                "solids": len(members),
                "named": sum(1 for m in members if m["named"]),
                "union_vertex_bbox_mm": union,
                "shape_hints": dict(shapes.most_common()),
                "total_vertex_points": sum(m["vertex_points"] for m in members),
            }
        )

    fingerprint_groups: Dict[str, List[dict]] = defaultdict(list)
    for row in rows:
        fingerprint_groups[row["shape_fingerprint"]["fingerprint"]].append(row)
    repeated = []
    for fingerprint, members in fingerprint_groups.items():
        if len(members) < 2:
            continue
        repeated.append(
            {
                "fingerprint": fingerprint,
                "copies": len(members),
                "solids": [m["eid"] for m in members],
                "names": sorted({m["name_leaf"] or "NONE" for m in members})[:3],
                "named": sum(1 for m in members if m["named"]),
                "vertex_bbox_mm": members[0]["vertex_bbox_mm"],
                "circle_radius_mm": members[0].get("circle_radius_mm"),
            }
        )
    repeated.sort(key=lambda item: (-item["copies"], item["fingerprint"]))

    unnamed = [row for row in rows if not row["named"]]
    return {
        "input": {
            "path": str(model.path),
            "size_bytes": model.size_bytes,
            "file_schema": model.file_schema,
        },
        "units_note": "all sizes are in the file length unit (millimetres for Morph One)",
        "solids_total": len(rows),
        "solids_named": sum(1 for row in rows if row["named"]),
        "solids_unnamed": len(unnamed),
        "unnamed_solids": [
            {
                "eid": row["eid"],
                "closure_entities": row["closure_entities"],
                "vertex_bbox_mm": row["vertex_bbox_mm"],
            }
            for row in sorted(unnamed, key=lambda r: -r["closure_entities"])[:10]
        ],
        "scene_vertex_bbox_mm": _union_bbox(rows),
        "candidate_duplicate_shape_groups": repeated,
        # Back-compat alias for downstream consumers. Groups are NOT proven identical.
        "identical_shape_groups": repeated,
        "repeated_shape_groups": sum(1 for item in repeated if item["copies"] >= 2),
        "groups": group_rows,
        "solids": rows,
        "caveats": [
            "extents come from BREP vertices only; they are lower bounds of the true extent",
            "shape hints are ratio rules, not semantic labels",
            "matching fingerprints are duplicate candidates only, NOT exact shape equality; rotated/mirrored copies may differ",
            "no wheel radius / steer axis / wheelbase / mass / inertia is derived here",
        ],
    }


def main(argv: "Sequence[str] | None" = None) -> int:
    parser = argparse.ArgumentParser(description="Per-solid STEP geometry facts")
    parser.add_argument("--input", required=True, help="STEP file to analyse")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument(
        "--max-solids", type=int, default=0, help="limit analysis to N solids (0 = all)"
    )
    args = parser.parse_args(argv)

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    model = parse_step(args.input)
    report = build_report(model, max_solids=args.max_solids)

    json_path = out_dir / "step_solid_geometry.json"
    json_path.write_text(json.dumps(report, ensure_ascii=False, indent=1), encoding="utf-8")

    csv_path = out_dir / "step_solid_geometry.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            [
                "eid",
                "type",
                "named",
                "part_group",
                "closure_entities",
                "vertex_points",
                "size_x_mm",
                "size_y_mm",
                "size_z_mm",
                "shape_hint",
                "circles",
                "circle_radius_median_mm",
                "name",
            ]
        )
        for row in report["solids"]:
            size = (row["vertex_bbox_mm"] or {}).get("size") or ["", "", ""]
            radius = (row.get("circle_radius_mm") or {}).get("median", "")
            writer.writerow(
                [
                    row["eid"],
                    row["type"],
                    row["named"],
                    row["part_group"],
                    row["closure_entities"],
                    row["vertex_points"],
                    size[0],
                    size[1],
                    size[2],
                    row["shape"]["hint"],
                    row.get("circles", 0),
                    radius,
                    row["name"],
                ]
            )

    print(
        json.dumps(
            {
                "json": str(json_path),
                "csv": str(csv_path),
                "solids": report["solids_total"],
                "named": report["solids_named"],
                "unnamed": report["solids_unnamed"],
                "groups": len(report["groups"]),
                "scene_vertex_bbox_mm": (report["scene_vertex_bbox_mm"] or {}).get("size"),
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
