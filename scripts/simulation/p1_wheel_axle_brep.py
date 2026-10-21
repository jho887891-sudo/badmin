"""P1 v2: per-face BREP surface classification for the four wheel solids.

Kernel-free (standard library only).  For each ADVANCED_FACE of the four wheel
solids (STEP eids 435700 / 759491 / 773650 / 871375) this records the analytic
surface type, radius, the axial band along the wheel spin axis, the radial band
about that axis, and an area integrated from the analytic surface plus the
angular coverage of the face's own circular edges.

No CAD kernel, no GPU, read-only on the STEP file.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
from typing import Dict, List, Optional, Sequence, Tuple

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from scripts.simulation.p1_step_solid_surface_bbox import (  # noqa: E402
    _floats,
    _placement,
    _point_of,
    _unit,
)
from scripts.simulation.step_geometry_census import (  # noqa: E402
    forward_closure,
    parse_step,
)

WHEEL_EIDS = (435700, 759491, 773650, 871375)
SURFACE_TYPES = frozenset(
    {
        "PLANE",
        "CYLINDRICAL_SURFACE",
        "CONICAL_SURFACE",
        "TOROIDAL_SURFACE",
        "SPHERICAL_SURFACE",
        "B_SPLINE_SURFACE_WITH_KNOTS",
        "B_SPLINE_SURFACE",
        "RATIONAL_B_SPLINE_SURFACE",
        "SURFACE_OF_REVOLUTION",
    }
)
BOUND_TYPES = frozenset({"FACE_OUTER_BOUND", "FACE_BOUND"})
TWO_PI = 2.0 * math.pi


def _dot(a, b):
    return sum(float(a[i]) * float(b[i]) for i in range(3))


def _sub(a, b):
    return tuple(float(a[i]) - float(b[i]) for i in range(3))


def _cross(a, b):
    return (
        a[1] * b[2] - a[2] * b[1],
        a[2] * b[0] - a[0] * b[2],
        a[0] * b[1] - a[1] * b[0],
    )


def _flags(body: str) -> List[bool]:
    out = []
    idx = 0
    while True:
        at = body.find(".", idx)
        if at < 0:
            break
        token = body[at : at + 3]
        if token in (".T.", ".F."):
            out.append(token == ".T.")
            idx = at + 3
        else:
            idx = at + 1
    return out


def _surface(entities, surface_id) -> dict:
    entity = entities.get(surface_id)
    if entity is None:
        return {"type": "MISSING"}
    out = {"type": entity.etype, "entity_id": entity.eid}
    values = _floats(entity.body)
    placement_id = entity.refs[0] if entity.refs else None
    location, axis, ref_dir = _placement(entities, placement_id)
    out["location"] = list(location) if location else None
    out["axis"] = list(axis) if axis else None
    out["ref_dir"] = list(ref_dir) if ref_dir else None
    if entity.etype == "CYLINDRICAL_SURFACE" and values:
        out["radius"] = values[-1]
    elif entity.etype == "CONICAL_SURFACE" and len(values) >= 2:
        out["radius"] = values[-2]
        out["semi_angle_rad"] = values[-1]
    elif entity.etype == "TOROIDAL_SURFACE" and len(values) >= 2:
        out["major_radius"] = values[-2]
        out["minor_radius"] = values[-1]
    elif entity.etype == "SPHERICAL_SURFACE" and values:
        out["radius"] = values[-1]
    return out


def _arc_delta(entities, coords, curve_id, p_start, p_end, same_sense) -> Optional[float]:
    """Swept angle of an arc edge, counter-clockwise about the circle axis."""
    curve = entities.get(curve_id)
    if curve is None or curve.etype != "CIRCLE":
        return None
    location, axis, ref_dir = _placement(entities, curve.refs[0] if curve.refs else None)
    values = _floats(curve.body)
    if location is None or axis is None or not values:
        return None
    radius = values[-1]
    u = ref_dir if ref_dir is not None else (1.0, 0.0, 0.0)
    if abs(_dot(u, axis)) > 0.999:
        u = (0.0, 1.0, 0.0)
    u = _unit([u[i] - _dot(u, axis) * axis[i] for i in range(3)])
    if u is None:
        return None
    v = _unit(_cross(axis, u))
    if v is None:
        return None
    d1 = _sub(p_start, location)
    d2 = _sub(p_end, location)
    a1 = math.atan2(_dot(d1, v), _dot(d1, u))
    a2 = math.atan2(_dot(d2, v), _dot(d2, u))
    if same_sense:
        delta = (a2 - a1) % TWO_PI
    else:
        delta = (a1 - a2) % TWO_PI
    return delta if delta > 0.0 else TWO_PI


def _arc_samples(entities, coords, curve_id, p_start, p_end, count=8):
    """Intermediate points on an arc edge (polyline approximation of the arc)."""
    curve = entities.get(curve_id)
    if curve is None or curve.etype != "CIRCLE":
        return []
    location, axis, ref_dir = _placement(entities, curve.refs[0] if curve.refs else None)
    values = _floats(curve.body)
    if location is None or axis is None or not values:
        return []
    radius = values[-1]
    u = ref_dir if ref_dir is not None else (1.0, 0.0, 0.0)
    if abs(_dot(u, axis)) > 0.999:
        u = (0.0, 1.0, 0.0)
    u = _unit([u[i] - _dot(u, axis) * axis[i] for i in range(3)])
    if u is None:
        return []
    v = _unit(_cross(axis, u))
    if v is None:
        return []
    d1 = _sub(p_start, location)
    d2 = _sub(p_end, location)
    a1 = math.atan2(_dot(d1, v), _dot(d1, u))
    a2 = math.atan2(_dot(d2, v), _dot(d2, u))
    delta = (a2 - a1) % TWO_PI
    if delta > math.pi:
        delta -= TWO_PI
    out = []
    for k in range(1, count):
        t = a1 + delta * (k / float(count))
        out.append(
            tuple(
                location[i] + radius * (math.cos(t) * u[i] + math.sin(t) * v[i])
                for i in range(3)
            )
        )
    return out


def _vertex_point(entities, coords, vp_id):
    """CARTESIAN_POINT coordinates of a VERTEX_POINT."""
    vertex = entities.get(vp_id)
    if vertex is None or vertex.etype != "VERTEX_POINT" or not vertex.refs:
        return None
    return coords.get(vertex.refs[0])


def _loop_of(entities, coords, bound_id):
    """Ordered (points, arcs, vertex_points) for one bound loop."""
    bound = entities.get(bound_id)
    if bound is None or not bound.refs:
        return [], [], []
    loop = entities.get(bound.refs[0])
    if loop is None:
        return [], [], []
    points: List[tuple] = []
    arcs: List[dict] = []
    verts: List[tuple] = []
    for oe_id in loop.refs:
        oe = entities.get(oe_id)
        if oe is None or oe.etype != "ORIENTED_EDGE" or not oe.refs:
            continue
        ec = entities.get(oe.refs[0])
        if ec is None or len(ec.refs) < 3:
            continue
        start = _vertex_point(entities, coords, ec.refs[0])
        end = _vertex_point(entities, coords, ec.refs[1])
        if start is None or end is None:
            continue
        oe_flags = _flags(oe.body)
        reversed_ = bool(oe_flags) and oe_flags[0] is False
        if reversed_:
            start, end = end, start
        ec_flags = _flags(ec.body)
        same_sense = ec_flags[0] if ec_flags else True
        curve = entities.get(ec.refs[2])
        if curve is None:
            continue
        if curve.etype == "CIRCLE":
            if reversed_:
                same_sense = not same_sense
            delta = _arc_delta(entities, coords, curve.eid, start, end, same_sense)
            values = _floats(curve.body)
            arcs.append(
                {
                    "radius": values[-1] if values else None,
                    "delta_rad": delta,
                    "centre": _placement(entities, curve.refs[0])[0],
                }
            )
        points.append(start)
        points.extend(_arc_samples(entities, coords, curve.eid, start, end))
        points.append(end)
        verts.extend([start, end])
    return points, arcs, verts


def _project(point, origin, axis):
    d = _sub(point, origin)
    h = _dot(d, axis)
    radial = [d[i] - h * axis[i] for i in range(3)]
    return h, math.sqrt(sum(c * c for c in radial))


def _newell_area(points) -> Optional[float]:
    if len(points) < 3:
        return None
    nx = ny = nz = 0.0
    n = len(points)
    for i in range(n):
        a = points[i]
        b = points[(i + 1) % n]
        nx += (a[1] - b[1]) * (a[2] + b[2])
        ny += (a[2] - b[2]) * (a[0] + b[0])
        nz += (a[0] - b[0]) * (a[1] + b[1])
    return 0.5 * math.sqrt(nx * nx + ny * ny + nz * nz)


def measure_solid(model, solid_id: int, coords, centre, axis) -> dict:
    entities = model.entities
    closure = forward_closure(entities, [solid_id])
    faces = []
    for eid in sorted(closure):
        entity = entities[eid]
        if entity.etype != "ADVANCED_FACE":
            continue
        surface_id = None
        bound_ids = []
        for ref in entity.refs:
            target = entities.get(ref)
            if target is None:
                continue
            if target.etype in SURFACE_TYPES:
                surface_id = ref
            elif target.etype in BOUND_TYPES:
                bound_ids.append(ref)
        surface = _surface(entities, surface_id)
        pts: List[tuple] = []
        arcs: List[dict] = []
        verts: List[tuple] = []
        for bound_id in bound_ids:
            p, a, v = _loop_of(entities, coords, bound_id)
            pts.extend(p)
            arcs.extend(a)
            verts.extend(v)
        if not verts:
            faces.append({"face_eid": eid, "surface": surface, "status": "no_loop"})
            continue
        hs = []
        rs = []
        for v in verts:
            h, r = _project(v, centre, axis)
            hs.append(h)
            rs.append(r)
        band = [min(hs), max(hs)]
        radial = [min(rs), max(rs)]
        # Angular coverage of the face is measured about the SURFACE's own axis
        # (a bolt-hole cylinder is parallel to the wheel axis but offset from it).
        surface_axis = tuple(surface["axis"]) if surface.get("axis") else axis
        surface_loc = tuple(surface["location"]) if surface.get("location") else centre
        per_circle: Dict[tuple, float] = {}
        arc_radii = []
        for arc in arcs:
            if arc["delta_rad"] is None or arc["radius"] is None:
                continue
            if arc["centre"] is None:
                continue
            d = _sub(arc["centre"], surface_loc)
            offset = math.sqrt(max(0.0, sum(c * c for c in d) - _dot(d, surface_axis) ** 2))
            if offset > 1e-6:
                continue
            key = tuple(round(float(c), 4) for c in arc["centre"])
            per_circle[key] = per_circle.get(key, 0.0) + arc["delta_rad"]
            arc_radii.append(arc["radius"])
        coverage = min(TWO_PI, max(per_circle.values())) if per_circle else 0.0
        surf_h = []
        surf_radial = []
        for v in verts:
            h, r = _project(v, surface_loc, surface_axis)
            surf_h.append(h)
            surf_radial.append(r)
        surf_dh = max(surf_h) - min(surf_h)
        area = None
        method = None
        stype = surface["type"]
        dh = band[1] - band[0]
        minor_band = None
        if stype == "CYLINDRICAL_SURFACE" and coverage > 0.0:
            area = surface["radius"] * coverage * abs(surf_dh)
            method = "cylinder r*dtheta*dh (arc coverage about the face axis)"
        elif stype == "CONICAL_SURFACE" and coverage > 0.0 and arc_radii:
            r1, r2 = min(arc_radii), max(arc_radii)
            slant = math.hypot(abs(surf_dh), abs(r2 - r1))
            area = 0.5 * (r1 + r2) * coverage * slant
            method = "cone mean_radius*dtheta*slant (arc coverage about the face axis)"
        elif stype == "TOROIDAL_SURFACE" and coverage > 0.0:
            major = surface["major_radius"]
            minor = surface["minor_radius"]
            thetas = [math.atan2(h, r - major) for h, r in zip(surf_h, surf_radial)]
            t1, t2 = min(thetas), max(thetas)
            minor_band = [round(t1, 6), round(t2, 6)]
            area = coverage * minor * (
                major * (t2 - t1) + minor * (math.sin(t2) - math.sin(t1))
            )
            method = "torus integral r_minor*[R*dtheta_minor + r*sin] * dtheta_major"
        elif stype == "PLANE":
            area = _newell_area(pts)
            method = "planar newell polygon (arcs sampled 8x)"
        elif stype == "SPHERICAL_SURFACE" and coverage > 0.0:
            area = 2.0 * math.pi * surface["radius"] ** 2 * (
                1.0 - math.cos(min(math.pi, abs(surf_dh) / max(1e-9, surface["radius"])))
            )
            method = "sphere zone (approx)"
        faces.append(
            {
                "face_eid": eid,
                "surface": surface,
                "vertex_count": len(verts),
                "arc_count": len(arcs),
                "angular_coverage_rad": coverage if coverage > 0 else None,
                "axial_band_mm": [round(band[0], 6), round(band[1], 6)],
                "axial_width_mm": round(abs(dh), 6),
                "radial_band_mm": [round(radial[0], 6), round(radial[1], 6)],
                "area_mm2": round(area, 4) if area is not None else None,
                "area_method": method,
                "minor_parameter_band_rad": minor_band,
                "touches_r103_circle": any(
                    a["radius"] is not None and abs(a["radius"] - 103.0) < 1e-6 for a in arcs
                ),
            }
        )
    # group by surface type + nominal radius
    groups: Dict[str, dict] = {}
    for face in faces:
        surface = face.get("surface", {})
        stype = surface.get("type", "UNKNOWN")
        radius = surface.get("radius", surface.get("major_radius"))
        key = "%s|r=%s" % (
            stype,
            "none" if radius is None else round(float(radius), 3),
        )
        group = groups.setdefault(
            key,
            {
                "key": key,
                "surface_type": stype,
                "radius_mm": radius,
                "face_count": 0,
                "area_mm2": 0.0,
                "area_unknown_faces": 0,
                "axial_band_mm": [math.inf, -math.inf],
                "radial_band_mm": [math.inf, -math.inf],
                "touches_r103_circle": False,
            },
        )
        group["face_count"] += 1
        if face.get("area_mm2") is not None:
            group["area_mm2"] += face["area_mm2"]
        else:
            group["area_unknown_faces"] += 1
        if "axial_band_mm" in face:
            group["axial_band_mm"][0] = min(group["axial_band_mm"][0], face["axial_band_mm"][0])
            group["axial_band_mm"][1] = max(group["axial_band_mm"][1], face["axial_band_mm"][1])
            group["radial_band_mm"][0] = min(group["radial_band_mm"][0], face["radial_band_mm"][0])
            group["radial_band_mm"][1] = max(group["radial_band_mm"][1], face["radial_band_mm"][1])
        group["touches_r103_circle"] = group["touches_r103_circle"] or face.get(
            "touches_r103_circle", False
        )
    for group in groups.values():
        group["area_mm2"] = round(group["area_mm2"], 4)
        group["axial_band_mm"] = [
            round(group["axial_band_mm"][0], 6),
            round(group["axial_band_mm"][1], 6),
        ]
        group["axial_width_mm"] = round(group["axial_band_mm"][1] - group["axial_band_mm"][0], 6)
        group["radial_band_mm"] = [
            round(group["radial_band_mm"][0], 6),
            round(group["radial_band_mm"][1], 6),
        ]
    return {
        "step_eid": solid_id,
        "face_count": len(faces),
        "surface_type_counts": _count_types(faces),
        "faces": faces,
        "groups": sorted(groups.values(), key=lambda g: -g["area_mm2"]),
    }


def _count_types(faces) -> Dict[str, int]:
    out: Dict[str, int] = {}
    for face in faces:
        stype = face.get("surface", {}).get("type", "UNKNOWN")
        out[stype] = out.get(stype, 0) + 1
    return out


def main(argv=None):
    parser = argparse.ArgumentParser(description="Kernel-free BREP face census for wheel solids")
    parser.add_argument("--step", required=True)
    parser.add_argument("--eids", default=",".join(str(e) for e in WHEEL_EIDS))
    parser.add_argument("--out-json", required=True)
    parser.add_argument("--centres-json", default="", help="map step_eid -> [x,y,z] centre_mm")
    args = parser.parse_args(argv)

    model = parse_step(args.step)
    coords = dict(zip(model.point_ids, model.points))
    centres: Dict[int, Sequence[float]] = {}
    if args.centres_json:
        with open(args.centres_json, encoding="utf-8") as handle:
            centres = {int(k): v for k, v in json.load(handle).items()}
    eids = [int(e) for e in args.eids.split(",") if e.strip()]
    solids = []
    for eid in eids:
        entity = model.entities.get(eid)
        if entity is None:
            solids.append({"step_eid": eid, "status": "missing"})
            continue
        centre = centres.get(eid)
        if centre is None:
            raise SystemExit("no centre supplied for eid %d (--centres-json)" % eid)
        solids.append(measure_solid(model, eid, coords, tuple(centre), (1.0, 0.0, 0.0)))
    report = {
        "tool": "p1_wheel_axle_brep.py",
        "step_path": os.path.abspath(args.step),
        "step_size_bytes": model.size_bytes,
        "file_schema": model.file_schema,
        "parse_warnings": len(model.parse_warnings),
        "reference_axes": {
            "axial_band_along": "CAD +x, through each wheel centre (the verified spin axis)",
            "radial_band_about": "per-wheel line {centre + t*(1,0,0)} in the CAD frame",
            "centres_source": "per-wheel measured centre_mm from the P1 contract",
        },
        "solids": solids,
    }
    os.makedirs(os.path.dirname(os.path.abspath(args.out_json)), exist_ok=True)
    with open(args.out_json, "w", encoding="utf-8") as handle:
        json.dump(report, handle, ensure_ascii=False, indent=1)
    print(json.dumps({str(s.get("step_eid")): {
        "faces": s.get("face_count"),
        "types": s.get("surface_type_counts"),
    } for s in solids}, ensure_ascii=False, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
