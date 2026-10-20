"""Map the 282 STEP solids to the 282 USD meshes with explicit evidence grades.

The two counts being equal proves nothing, so this tool never reports "1:1 by
count".  Each pair needs two independent signals, and every decision is graded.

Signal 1: name lineage
    The converter derived prim names from the STEP solid names, so normalising
    both sides gives the same key for most parts.  The converter's spelling
    differs in ways that must be normalised away:

      punctuation and Chinese characters disappear;
      GB\u2571T becomes GBuT, i.e. multi-byte characters leak extra lowercase
            letters, so lowercase is dropped from the key on both sides;
      several parts sharing a name get a counter suffix (NONE, NONE_1 ...).

Signal 2: geometry sandwich
    A STEP solid gives two honest bounds:

      lower bound  the bbox of its BREP vertices, which lie on the surface;
      upper bound  the bbox of every Cartesian point in the solid's closure,
                   i.e. including control points and placement origins.  A spline
                   surface stays inside the convex hull of its control points,
                   so the tessellated mesh cannot leave this box.

    A correct pair must satisfy  lower <= mesh <= upper  on every axis.  This is
    a necessary condition, not a similarity score, and it is what finally
    separates curved parts: the wheel-like solid's vertex bbox is
    48.7 x 180.8 x 157.8 mm while its real envelope is 48.79 x 205.99 x 205.99 mm,
    because a 103 mm radius arc bulges far outside the vertex extremes.

Grades
    VERIFIED   identical key, the key group holds exactly one solid and one mesh,
               and the sandwich holds.
    CANDIDATE  identical key and sandwich holds, but several parts share that key
               (or the name is literally NONE), so the pairing also relies on
               picking the geometrically closest partner inside the group.
    UNKNOWN    not paired; reported with the reason and the closest key ratio.

Nothing here labels a part physically: no wheel, no axle, no chassis dimension.
"""

from __future__ import annotations

import argparse
import csv
import difflib
import json
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

# Running this file directly puts the package directory on sys.path instead of
# the repository root, so make the package importable from either entry point.
_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

__all__ = [
    "name_key",
    "prim_core",
    "sandwich",
    "match_parts",
    "load_solid_facts",
    "load_mesh_facts",
    "GRADE_VERIFIED",
    "GRADE_CANDIDATE",
    "GRADE_UNKNOWN",
]

GRADE_VERIFIED = "VERIFIED"  # reserved for future per-solid source-ID / exact correspondence
GRADE_STRONG_CANDIDATE = "STRONG_CANDIDATE"
GRADE_CANDIDATE = "CANDIDATE"
GRADE_UNKNOWN = "UNKNOWN"

#: Empirical pairing tolerance for tessLOD=2. It has NOT been calibrated
#: against a manufacturer's conversion error bound. A pair that passes is a
#: candidate match, never independently proven source-to-mesh identity.
CONTAIN_TOL_MM = 2.5
CONTAIN_TOL_REL = 0.01
UNNAMED_KEY = "NONE"


def name_key(text: str) -> str:
    """Key shared by a STEP name and the converter's prim name for that part.

    Only uppercase letters and digits survive, and lowercase is dropped *before*
    anything else: the converter inserts lowercase letters when it re-encodes
    multi-byte characters (GB\u2571T becomes GBuT), so uppercasing first would
    keep that stray U and break the match.  Dropping lowercase loses the word
    "solid" on both sides equally and keeps every distinguishing digit.
    """
    return re.sub(r"[^A-Z0-9]+", "", text or "")


def prim_core(name: str) -> str:
    """Strip the converter prefix and its disambiguation suffixes from a prim name."""
    core = name[4:] if name.startswith("tn__") else name
    for _ in range(4):
        head, _, tail = core.rpartition("_")
        if not head:
            break
        if len(tail) >= 8 or tail.isdigit():
            core = head
            continue
        break
    return core


def _tol(extent: float) -> float:
    return max(abs(extent) * CONTAIN_TOL_REL, CONTAIN_TOL_MM)


def sandwich(lower_min: Sequence[float], lower_max: Sequence[float],
             upper_min: Sequence[float], upper_max: Sequence[float],
             mesh_min: Sequence[float], mesh_max: Sequence[float]) -> dict:
    """Check  lower <= mesh <= upper  on every axis.

    The analytic BREP vertex bbox must sit *inside* the tessellated mesh
    bbox, up to a declared tolerance. The mesh bbox must sit inside the
    proposed upper bound. Neither test independently certifies CAD identity,
    and the heuristic tolerance is not a proven tessLOD chord-error bound.
    """
    below = []
    above = []
    for axis in range(3):
        extent = max(abs(lower_max[axis] - lower_min[axis]), abs(upper_max[axis] - upper_min[axis]))
        tolerance = _tol(extent)
        # lower box must be inside the mesh: mesh_min <= lower_min and
        # mesh_max >= lower_max.  A curved surface may bulge outside the vertex
        # box, so only the reverse direction is a violation.
        below.append(
            max(0.0, (mesh_min[axis] - lower_min[axis]), (lower_max[axis] - mesh_max[axis]))
        )
        above.append(
            max(0.0, (upper_min[axis] - mesh_min[axis]), (mesh_max[axis] - upper_max[axis]))
        )
        if below[-1] > tolerance:
            below[-1] = below[-1]
        else:
            below[-1] = 0.0
        if above[-1] > tolerance:
            above[-1] = above[-1]
        else:
            above[-1] = 0.0

    bulge = [
        round(max(0.0, (mesh_max[axis] - lower_max[axis])), 3) for axis in range(3)
    ]
    cost = sum(below) + sum(above)
    return {
        "below_lower_bound_mm": [round(value, 3) for value in below],
        "above_upper_bound_mm": [round(value, 3) for value in above],
        "exceeds_vertex_bbox_mm": bulge,
        "ok": all(value == 0.0 for value in below) and all(value == 0.0 for value in above),
        "cost": round(cost, 4),
    }


def pair_cost(lower_min, lower_max, upper_min, upper_max, mesh_min, mesh_max) -> float:
    """Rank partners inside one name-key group.

    The sandwich already says whether a pair is admissible; this score only
    orders admissible pairs, so the 35 parts that share the literal name NONE are
    separated by where they sit (minimum corner) and by how much size the mesh
    adds on top of the vertex box.
    """
    total = 0.0
    for axis in range(3):
        extent = max(
            abs(lower_max[axis] - lower_min[axis]),
            abs(upper_max[axis] - upper_min[axis]),
            1.0,
        )
        scale = max(0.02 * extent, 1.0)
        total += abs(float(mesh_min[axis]) - float(lower_min[axis])) / scale
        total += abs(float(mesh_max[axis]) - float(lower_max[axis])) / scale * 0.5
    return round(total, 6)


def match_parts(solids: Sequence[dict], meshes: Sequence[dict]) -> dict:
    """Pair solids with meshes by name key, then by the geometry sandwich."""
    solid_keys: Dict[str, List[dict]] = defaultdict(list)
    for solid in solids:
        solid_keys[name_key(solid["name"])].append(solid)
    mesh_keys: Dict[str, List[dict]] = defaultdict(list)
    for mesh in meshes:
        mesh_keys[name_key(mesh["core"])].append(mesh)

    all_keys = sorted(set(solid_keys) | set(mesh_keys))
    pairs: List[dict] = []
    rejected: List[dict] = []
    rejected_count = 0

    for key in all_keys:
        key_solids = solid_keys.get(key, [])
        key_meshes = mesh_keys.get(key, [])
        if not key_solids or not key_meshes:
            continue

        scored = []
        for solid in key_solids:
            for mesh in key_meshes:
                result = sandwich(
                    solid["lower_min"], solid["lower_max"],
                    solid["upper_min"], solid["upper_max"],
                    mesh["min"], mesh["max"],
                )
                cost = pair_cost(
                    solid["lower_min"], solid["lower_max"],
                    solid["upper_min"], solid["upper_max"],
                    mesh["min"], mesh["max"],
                )
                scored.append((0 if result["ok"] else 1, cost, solid["eid"],
                               mesh["path"], solid, mesh, result))
        scored.sort(key=lambda item: (item[0], item[1], item[2], item[3]))

        # Global greedy selection can strand a constrained solid: A has two
        # possible partners while B has only one, but greedy may grab B's one.
        # Use deterministic augmenting paths to maximize valid pair *count*.
        # This is NOT a globally minimum-cost matching, nor proof of identity.
        feasible = defaultdict(list)
        scored_by_pair = {}
        for record in scored:
            _, cost, eid, mesh_path, solid, mesh, result = record
            if result["ok"]:
                feasible[eid].append((cost, mesh_path))
                scored_by_pair[(eid, mesh_path)] = record
            else:
                rejected_count += 1
                if len(rejected) < 50:
                    rejected.append({
                    "step_eid": eid,
                    "step_name": solid["name"],
                    "usd_path": mesh_path,
                    "reason": "geometry sandwich fails",
                    "detail": result,
                })
        for options in feasible.values():
            options.sort(key=lambda pair: (pair[0], pair[1]))

        match_mesh_to_solid = {}
        def _augment(eid, seen_meshes):
            for _, mp in feasible[eid]:
                if mp in seen_meshes:
                    continue
                seen_meshes.add(mp)
                prior = match_mesh_to_solid.get(mp)
                if prior is None or _augment(prior, seen_meshes):
                    match_mesh_to_solid[mp] = eid
                    return True
            return False

        for solid in sorted(key_solids, key=lambda item:
                            (len(feasible[item["eid"]]), item["eid"])):
            _augment(solid["eid"], set())

        chosen_pairs = set((eid, mp) for mp, eid in match_mesh_to_solid.items())
        group_size = (len(key_solids), len(key_meshes))
        taken_solids: set = set()
        taken_meshes: set = set()
        for _, _, solid_eid, mesh_path, solid, mesh, result in scored:
            if (solid_eid, mesh_path) not in chosen_pairs:
                continue
            if solid_eid in taken_solids or mesh_path in taken_meshes:
                raise AssertionError("invalid duplicate from bipartite matching")
            unique_in_group = group_size == (1, 1)
            named = bool(key) and key != UNNAMED_KEY
            # A unique normalized name plus a broad geometric bbox is not
            # a verifiable STEP entity identifier. Never promote to VERIFIED
            # without independent per-solid provenance or topology validation.
            grade = (GRADE_STRONG_CANDIDATE if (named and unique_in_group)
                     else GRADE_CANDIDATE)
            reasons = []
            if named:
                reasons.append("name key identical on both sides")
            else:
                reasons.append("name field is literally NONE on both sides")
            if unique_in_group:
                reasons.append("unique in its name-key group")
            else:
                reasons.append(
                    "%d solids / %d meshes share this key; partner chosen by geometry"
                    % group_size
                )
            reasons.append("geometry sandwich passes within empirical tolerance; identity NOT proven")

            taken_solids.add(solid_eid)
            taken_meshes.add(mesh_path)
            pairs.append(
                {
                    "step_eid": solid_eid,
                    "step_name": solid["name"],
                    "usd_path": mesh_path,
                    "usd_core": mesh["core"],
                    "usd_points": mesh.get("points"),
                    "grade": grade,
                    "reasons": reasons,
                    "name_key": key,
                    "group_solids": group_size[0],
                    "group_meshes": group_size[1],
                    "step_vertex_bbox_mm": {
                        "min": [round(float(v), 3) for v in solid["lower_min"]],
                        "max": [round(float(v), 3) for v in solid["lower_max"]],
                    },
                    "step_closure_bbox_mm": {
                        "min": [round(float(v), 3) for v in solid["upper_min"]],
                        "max": [round(float(v), 3) for v in solid["upper_max"]],
                    },
                    "usd_bbox_mm": {
                        "min": [round(float(v), 3) for v in mesh["min"]],
                        "max": [round(float(v), 3) for v in mesh["max"]],
                    },
                    "mesh_exceeds_vertex_bbox_mm": result["exceeds_vertex_bbox_mm"],
                    "sandwich": result,
                }
            )

    matched_solids = {pair["step_eid"] for pair in pairs}
    matched_meshes = {pair["usd_path"] for pair in pairs}
    grades = Counter(pair["grade"] for pair in pairs)

    unmatched_solids = []
    for solid in solids:
        if solid["eid"] in matched_solids:
            continue
        key = name_key(solid["name"])
        closest_ratio = 0.0
        closest_key = ""
        for mesh in meshes:
            ratio = difflib.SequenceMatcher(None, key, name_key(mesh["core"])).ratio()
            if ratio > closest_ratio:
                closest_ratio = ratio
                closest_key = name_key(mesh["core"])
        unmatched_solids.append(
            {
                "eid": solid["eid"],
                "name": solid["name"],
                "name_key": key,
                "closest_mesh_key": closest_key,
                "closest_key_ratio": round(closest_ratio, 4),
                "vertex_bbox_mm": {
                    "min": [round(float(v), 3) for v in solid["lower_min"]],
                    "max": [round(float(v), 3) for v in solid["lower_max"]],
                },
            }
        )

    unmatched_meshes = [
        {
            "path": mesh["path"],
            "core": mesh["core"],
            "name_key": name_key(mesh["core"]),
            "bbox_mm": {
                "min": [round(float(v), 3) for v in mesh["min"]],
                "max": [round(float(v), 3) for v in mesh["max"]],
            },
        }
        for mesh in meshes
        if mesh["path"] not in matched_meshes
    ]

    return {
        "solids": len(solids),
        "meshes": len(meshes),
        "pairs": pairs,
        "grade_counts": dict(grades),
        "grade_policy": ("VERIFIED requires independently traced per-entity source identity; "
                         "unique normalized name and broad bbox yield STRONG_CANDIDATE only; "
                         "ambiguous/unnamed matches yield CANDIDATE"),
        "pairs_with_name_evidence": sum(
            1 for pair in pairs if pair["name_key"] != UNNAMED_KEY
        ),
        "pairs_without_name_evidence": sum(
            1 for pair in pairs if pair["name_key"] == UNNAMED_KEY
        ),
        "unmatched_solids": unmatched_solids,
        "unmatched_meshes": unmatched_meshes,
        "rejected_pairs": rejected[:50],
        "rejected_pair_count": rejected_count,
        "pairing_algorithm": "deterministic maximum-cardinality augmenting-path (not min-cost)",
        # This heuristic never establishes a source-ID-level bijection,
        # even in the degenerate case of one solid and one similarly named mesh.
        "bijection_proven": False,
        "curved_parts_where_mesh_exceeds_vertex_bbox": sorted(
            [
                {
                    "step_eid": pair["step_eid"],
                    "step_name": pair["step_name"],
                    "usd_path": pair["usd_path"],
                    "grade": pair["grade"],
                    "exceeds_mm": pair["mesh_exceeds_vertex_bbox_mm"],
                }
                for pair in pairs
                if max(pair["mesh_exceeds_vertex_bbox_mm"]) > CONTAIN_TOL_MM
            ],
            key=lambda item: -max(item["exceeds_mm"]),
        )[:20],
        "ambiguous_name_keys": [
            {
                "key": key,
                "solids": len(solid_keys.get(key, [])),
                "meshes": len(mesh_keys.get(key, [])),
                "note": (
                    "name field is literally NONE on both sides"
                    if key == UNNAMED_KEY
                    else "identical name key; pairing decided by geometry only"
                ),
            }
            for key in all_keys
            if len(solid_keys.get(key, [])) > 1 or len(mesh_keys.get(key, [])) > 1
        ],
    }


# ------------------------------------------------------------------- extraction


def load_solid_facts(step_path: str) -> List[dict]:
    """Per-solid name plus an honest lower and upper bbox bound from the STEP file.

    lower: BREP vertices, which lie on the surface.
    upper: every Cartesian point in the solid's closure.  This is deliberately
           the loosest bound that is still guaranteed: a circular arc between two
           vertices bulges outside the vertex box, and a spline surface stays
           inside the hull of its control points, so anything tighter than "all
           referenced points" would reject correct pairs.  The bound therefore
           only rejects pairs that are impossible; ranking inside a name-key
           group is done by pair_cost, not by this bound.
    """
    from scripts.simulation.step_geometry_census import SOLID_TYPES, entity_name, parse_step

    model = parse_step(step_path)
    coords = dict(zip(model.point_ids, model.points))
    rows: List[dict] = []

    for entity in model.entities.values():
        if not (set(entity.components) & SOLID_TYPES):
            continue
        closure = _closure(model, entity.eid)
        vertex_box = _box_of(
            coords, (model.entities[eid] for eid in closure), only_vertices=True
        )
        if vertex_box is None:
            continue
        upper_box = _box_from_points(
            [
                coords[eid]
                for eid in closure
                if model.entities[eid].etype == "CARTESIAN_POINT" and eid in coords
            ]
        )
        if upper_box is None:
            upper_box = vertex_box
        # the guaranteed bound must at least cover the vertices
        upper_box = (
            [min(upper_box[0][axis], vertex_box[0][axis]) for axis in range(3)],
            [max(upper_box[1][axis], vertex_box[1][axis]) for axis in range(3)],
        )
        # Circular arcs and cylinders bulge far outside the vertex box: a screw
        # head whose vertices span 4.3 mm is really an 8.5 mm cylinder.  Grow the
        # upper bound by each circle radius around its centre, which stays a
        # guaranteed (if slightly loose) bound.
        for radius, centre in _circles_of(model, closure, coords):
            for axis in range(3):
                upper_box[0][axis] = min(upper_box[0][axis], centre[axis] - radius)
                upper_box[1][axis] = max(upper_box[1][axis], centre[axis] + radius)
        rows.append(
            {
                "eid": entity.eid,
                "name": entity_name(entity),
                "lower_min": list(vertex_box[0]),
                "lower_max": list(vertex_box[1]),
                "upper_min": list(upper_box[0]),
                "upper_max": list(upper_box[1]),
                "vertex_points": sum(
                    1 for eid in closure if model.entities[eid].etype == "VERTEX_POINT"
                ),
                "closure_points": len(closure),
            }
        )
    rows.sort(key=lambda row: row["eid"])
    return rows


def _closure(model, solid_id: int):
    from scripts.simulation.step_geometry_census import forward_closure

    return forward_closure(model.entities, [solid_id])


def _box_of(coords, entities, only_vertices: bool):
    xs: List[float] = []
    ys: List[float] = []
    zs: List[float] = []
    for entity in entities:
        if only_vertices and entity.etype != "VERTEX_POINT":
            continue
        if not entity.refs:
            continue
        point = coords.get(entity.refs[0])
        if point is None:
            continue
        xs.append(point[0])
        ys.append(point[1])
        zs.append(point[2])
    if not xs:
        return None
    return (
        [min(xs), min(ys), min(zs)],
        [max(xs), max(ys), max(zs)],
    )


def _hull_point_ids(model, closure, coords) -> List[int]:
    """Vertex points plus B-spline control points of the solid."""
    from scripts.simulation.step_geometry_census import _BSPLINE_TYPES

    point_ids: set = set()
    for eid in closure:
        entity = model.entities[eid]
        if entity.etype == "VERTEX_POINT" and entity.refs:
            if entity.refs[0] in coords:
                point_ids.add(entity.refs[0])
        elif set(entity.components) & _BSPLINE_TYPES:
            for ref in entity.refs:
                target = model.entities.get(ref)
                if target is not None and target.etype == "CARTESIAN_POINT" and ref in coords:
                    point_ids.add(ref)
    return sorted(point_ids)


def _circles_of(model, closure, coords):
    """(radius, centre) for every CIRCLE in a solid's closure."""
    from scripts.simulation.step_solid_geometry import _last_float

    found = []
    for eid in closure:
        entity = model.entities[eid]
        if entity.etype != "CIRCLE" or not entity.refs:
            continue
        radius = _last_float(entity.body)
        if radius is None or radius <= 0:
            continue
        placement = model.entities.get(entity.refs[0])
        if placement is None or not placement.refs:
            continue
        centre = coords.get(placement.refs[0])
        if centre is None:
            continue
        found.append((float(radius), centre))
    return found


def _box_from_points(points: Sequence[Sequence[float]]):
    if not points:
        return None
    xs = [p[0] for p in points]
    ys = [p[1] for p in points]
    zs = [p[2] for p in points]
    return ([min(xs), min(ys), min(zs)], [max(xs), max(ys), max(zs)])


def load_mesh_facts(usd_path: str) -> List[dict]:
    """Per-mesh prim core name, point count and bbox in the USD's world frame.

    The converter writes mesh points in a part-local frame and keeps the assembly
    placement on the prototype Xform above the mesh, so the *local* point bbox is
    centred near the origin and cannot be compared with STEP assembly coordinates.
    The bbox below is therefore computed by pushing every point through the prim's
    local-to-world matrix.  The local bbox is kept as well, for reference only.
    """
    from pxr import Gf, Usd, UsdGeom

    stage = Usd.Stage.Open(usd_path)
    if stage is None:
        raise RuntimeError("cannot open USD: %s" % usd_path)
    cache = UsdGeom.XformCache(Usd.TimeCode.Default())
    rows: List[dict] = []
    for prim in stage.TraverseAll():
        if prim.GetTypeName() != "Mesh":
            continue
        points = UsdGeom.Mesh(prim).GetPointsAttr().Get() or []
        if not points:
            continue
        world = cache.GetLocalToWorldTransform(prim)
        world_points = [
            world.Transform(Gf.Vec3d(float(p[0]), float(p[1]), float(p[2]))) for p in points
        ]
        xs = [p[0] for p in world_points]
        ys = [p[1] for p in world_points]
        zs = [p[2] for p in world_points]
        lx = [p[0] for p in points]
        ly = [p[1] for p in points]
        lz = [p[2] for p in points]
        parent = prim.GetParent()
        rows.append(
            {
                "path": str(prim.GetPath()),
                "core": prim_core(parent.GetName()),
                "parent_name": parent.GetName(),
                "points": len(points),
                "min": [min(xs), min(ys), min(zs)],
                "max": [max(xs), max(ys), max(zs)],
                "local_min": [min(lx), min(ly), min(lz)],
                "local_max": [max(lx), max(ly), max(lz)],
            }
        )
    rows.sort(key=lambda row: row["path"])
    return rows


def _write_csv(path: Path, pairs: Sequence[dict]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            [
                "grade", "step_eid", "step_name", "name_key", "usd_path", "usd_points",
                "vertex_min_x", "vertex_min_y", "vertex_min_z",
                "vertex_max_x", "vertex_max_y", "vertex_max_z",
                "usd_min_x", "usd_min_y", "usd_min_z",
                "usd_max_x", "usd_max_y", "usd_max_z",
                "bulge_x", "bulge_y", "bulge_z", "group_solids", "group_meshes", "reasons",
            ]
        )
        for pair in pairs:
            writer.writerow(
                [
                    pair["grade"], pair["step_eid"], pair["step_name"], pair["name_key"],
                    pair["usd_path"], pair["usd_points"],
                    *pair["step_vertex_bbox_mm"]["min"], *pair["step_vertex_bbox_mm"]["max"],
                    *pair["usd_bbox_mm"]["min"], *pair["usd_bbox_mm"]["max"],
                    *pair["mesh_exceeds_vertex_bbox_mm"],
                    pair["group_solids"], pair["group_meshes"],
                    " | ".join(pair["reasons"]),
                ]
            )


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="Map STEP solids to USD meshes")
    parser.add_argument("--step", required=True)
    parser.add_argument("--usd", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--report", default="")
    args = parser.parse_args(argv)

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    solids = load_solid_facts(args.step)
    meshes = load_mesh_facts(args.usd)
    result = match_parts(solids, meshes)
    result["inputs"] = {
        "step": str(Path(args.step).resolve()),
        "usd": str(Path(args.usd).resolve()),
    }
    result["notes"] = [
        "STRONG_CANDIDATE uses a unique normalized name key plus approximate bbox evidence; no matches are VERIFIED",
        "282 solids and 282 meshes do not by themselves prove a one-to-one mapping",
        "no part is labelled physically; shapes are never promoted to wheel axes",
    ]
    report_path = Path(args.report) if args.report else out_dir / "step_to_usd_mapping.json"
    report_path.write_text(json.dumps(result, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
    csv_path = out_dir / "step_to_usd_mapping.csv"
    _write_csv(csv_path, result["pairs"])
    print(
        json.dumps(
            {
                "report": str(report_path),
                "csv": str(csv_path),
                "solids": result["solids"],
                "meshes": result["meshes"],
                "pairs": len(result["pairs"]),
                "grade_counts": result["grade_counts"],
                "pairs_with_name_evidence": result["pairs_with_name_evidence"],
                "pairs_without_name_evidence": result["pairs_without_name_evidence"],
                "bijection_proven": result["bijection_proven"],
                "unmatched_solids": len(result["unmatched_solids"]),
                "unmatched_meshes": len(result["unmatched_meshes"]),
                "rejected_pair_count": result["rejected_pair_count"],
            },
            ensure_ascii=False,
            indent=1,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
