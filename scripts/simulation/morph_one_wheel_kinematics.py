"""Candidate steering and drive axes for the Morph One chassis, from real geometry.

What this is, and what it is not
--------------------------------
This tool extracts *candidate* axes of rotation from the STEP geometry and groups
the parts that carry them.  A shared cylinder axis is evidence of a manufactured
rotation feature; it is NOT proof of a kinematic joint, and it is not a motor
output shaft.  A wheel module contains many co-axial holes, bearing seats and
screw bosses, so every axis below carries a confidence level and the parts that
produced it.

Inputs
    the STEP file  (eid, decoded Chinese part name, BREP circles and cylinders)
    optionally the audited STEP->USD mapping JSON, to attach a mesh path per part

Outputs
    part table         eid -> Chinese name -> USD mesh path -> candidate module
    axis clusters      direction, point on axis, axial extent, radii, parts
    role candidates    steering (near-vertical) / drive (near-horizontal) / other
    consistency report how many complete wheel modules exist, and their symmetry

Nothing here writes to the CAD files, and no joint, mass or inertia is created.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import re
import sys
from collections import Counter, defaultdict
from typing import Dict, Iterable, List, Sequence, Tuple

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

import numpy as np

from scripts.simulation.step_geometry_census import (
    SOLID_TYPES,
    entity_name,
    forward_closure,
    parse_step,
)
from scripts.simulation.step_solid_geometry import _last_float, vertex_bbox

# The audited step_solid_geometry._TRIPLE_RE has no exponent group, so it rejects
# components such as -4.789422297650225240E-14 and _direction_vector() returns
# None for almost every DIRECTION in this SolidWorks export (measured: entity
# 254973 parses to None).  This module therefore parses triples itself and does
# not import that helper.
_NUM = r"[-+]?(?:[0-9]+\.?[0-9]*|\.[0-9]+)(?:[EeDd][-+]?[0-9]+)?"
_TRIPLE_AGENT_RE = re.compile(
    r"\(\s*(" + _NUM + r")\s*,\s*(" + _NUM + r")\s*,\s*(" + _NUM + r")\s*\)"
)

AXIS_ANGLE_TOL_DEG = 0.5
AXIS_LINE_TOL_MM = 0.5
VERTICAL_DOT = 0.99
HORIZONTAL_DOT = 0.10
Z = np.array([0.0, 0.0, 1.0])
# A CAD model is not necessarily Z-up: this chassis is authored Y-up (the wheel
# solids share one vertical coordinate and touch the model minimum there).  The
# up axis is therefore detected and reported instead of being assumed.
UP_AXIS_CHOICES = ("auto", "x", "y", "z")
UP_INDEX = {"x": 0, "y": 1, "z": 2}
# Co-axial features separated by more than this along the shared line belong to
# different physical stacks (e.g. the two wheels of one axle line).  The value is
# comfortably larger than the widest solid measured on this chassis (48.7 mm) and
# far smaller than the corner spacing (about 500 mm), so it separates axle lines
# without splitting the two faces of a single wheel.
AXIAL_GAP_MM = 100.0
CORNER_POSITION_TOL_MM = 2.0
CORNER_RADIUS_TOL_MM = 1.0


# --------------------------------------------------------------------- features
def _unit(vec) -> "np.ndarray | None":
    v = np.asarray(vec, dtype=float)
    n = float(np.linalg.norm(v))
    if n <= 0.0 or not np.isfinite(n):
        return None
    return v / n


def _canonical(direction: np.ndarray) -> np.ndarray:
    """Deterministic sign so the same axis is not clustered twice."""
    idx = int(np.argmax(np.abs(direction)))
    return direction if direction[idx] >= 0 else -direction


def direction_vector(entity) -> "np.ndarray | None":
    """DIRECTION body -> unit vector, with exponent notation supported."""
    if entity is None:
        return None
    match = _TRIPLE_AGENT_RE.search(entity.body)
    if match is None:
        return None
    values = [float(g.replace("D", "E").replace("d", "e")) for g in match.groups()]
    return _unit(values)


def _direction_of(model, entity) -> "np.ndarray | None":
    """Axis direction of an AXIS2_PLACEMENT_3D: its first DIRECTION reference."""
    if entity is None:
        return None
    for ref in entity.refs:
        target = model.entities.get(ref)
        if target is None or target.etype != "DIRECTION":
            continue
        vector = direction_vector(target)
        if vector is not None:
            return vector
    return None


def _centre_of(model, entity, coords) -> "np.ndarray | None":
    if entity is None or not entity.refs:
        return None
    point = coords.get(entity.refs[0])
    return None if point is None else np.asarray(point, dtype=float)


def circular_features(model, solid_id: int, coords) -> List[dict]:
    """Every CIRCLE and CYLINDRICAL_SURFACE reachable from one solid."""
    out: List[dict] = []
    for eid in forward_closure(model.entities, [solid_id]):
        entity = model.entities[eid]
        if entity.etype not in ("CIRCLE", "CYLINDRICAL_SURFACE"):
            continue
        radius = _last_float(entity.body)
        if radius is None or radius <= 0:
            continue
        placement = model.entities.get(entity.refs[0]) if entity.refs else None
        direction = _direction_of(model, placement)
        centre = _centre_of(model, placement, coords)
        if direction is None or centre is None:
            continue
        out.append(
            {
                "entity_id": eid,
                "kind": entity.etype,
                "radius_mm": round(float(radius), 4),
                "direction": [round(float(v), 6) for v in _canonical(direction)],
                "point_mm": [round(float(v), 4) for v in centre],
            }
        )
    return out


# ---------------------------------------------------------------------- clusters
def _line_distance(point: np.ndarray, origin: np.ndarray, direction: np.ndarray) -> float:
    delta = point - origin
    return float(np.linalg.norm(delta - np.dot(delta, direction) * direction))


def cluster_axes(features: Sequence[dict]) -> List[dict]:
    """Group circular features that share one physical axis line."""
    cos_tol = math.cos(math.radians(AXIS_ANGLE_TOL_DEG))
    clusters: List[dict] = []
    for feature in features:
        d = np.asarray(feature["direction"], dtype=float)
        p = np.asarray(feature["point_mm"], dtype=float)
        placed = False
        for cluster in clusters:
            if abs(float(np.dot(d, cluster["direction"]))) < cos_tol:
                continue
            if _line_distance(p, cluster["origin"], cluster["direction"]) > AXIS_LINE_TOL_MM:
                continue
            cluster["members"].append(feature)
            if np.dot(d, cluster["direction"]) < 0:
                d = -d
            weights = np.array([m["radius_mm"] for m in cluster["members"]], dtype=float)
            points = np.array([m["point_mm"] for m in cluster["members"]], dtype=float)
            cluster["direction"] = _canonical(
                _unit(np.average(np.array([m["direction"] for m in cluster["members"]]),
                                 axis=0, weights=weights))
            )
            cluster["origin"] = points.mean(axis=0)
            placed = True
            break
        if not placed:
            clusters.append(
                {
                    "direction": _canonical(d),
                    "origin": p.copy(),
                    "members": [feature],
                }
            )
    for cluster in clusters:
        points = np.array([m["point_mm"] for m in cluster["members"]], dtype=float)
        projections = points @ cluster["direction"]
        cluster["extent_along_axis_mm"] = round(float(projections.max() - projections.min()), 3)
        cluster["max_radius_mm"] = round(max(m["radius_mm"] for m in cluster["members"]), 3)
        cluster["min_radius_mm"] = round(min(m["radius_mm"] for m in cluster["members"]), 3)
        cluster["feature_count"] = len(cluster["members"])
    return clusters


def _role_of(direction: np.ndarray, up_axis: str = "z") -> str:
    """Classify an axis against the *model's* up axis (default Z for compatibility).

    "steering_candidate_vertical" means parallel to the up axis, "drive_candidate_
    horizontal" means lying in the ground plane.  Both are geometry statements
    only, not claims that a joint exists.
    """
    unit = _unit(direction)
    if unit is None:
        return "oblique_or_other"
    axis = np.zeros(3, dtype=float)
    axis[UP_INDEX.get(str(up_axis).strip().lower(), 2)] = 1.0
    along_up = abs(float(np.dot(unit, axis)))
    if along_up >= VERTICAL_DOT:
        return "steering_candidate_vertical"
    if along_up <= HORIZONTAL_DOT:
        return "drive_candidate_horizontal"
    return "oblique_or_other"


def _confidence(cluster: dict, parts: Sequence[int], named: int) -> str:
    n = cluster["feature_count"]
    if n >= 3 and len(parts) >= 2 and named >= 1 and cluster["max_radius_mm"] >= 5.0:
        return "HIGH"
    if n >= 2 and cluster["max_radius_mm"] >= 3.0:
        return "MEDIUM"
    return "LOW"



# ----------------------------------------------------------------- chassis frame
def detect_up_axis(parts: Sequence[dict], requested: str = "auto") -> dict:
    """Report which CAD axis is plausibly vertical, with the evidence used.

    The only automatic rule used here is "the axis with the smallest overall
    extent", which is a convention hint and nothing more; it is reported together
    with the measured extents so a reader can reject it.
    """
    choice = str(requested or "auto").strip().lower()
    if choice not in UP_AXIS_CHOICES:
        raise ValueError("up axis must be one of %s, got %r" % (UP_AXIS_CHOICES, requested))
    boxes = [p["vertex_bbox_mm"] for p in parts if p.get("vertex_bbox_mm")]
    extents = None
    if boxes:
        lo = [min(b["min"][i] for b in boxes) for i in range(3)]
        hi = [max(b["max"][i] for b in boxes) for i in range(3)]
        extents = [round(float(hi[i] - lo[i]), 3) for i in range(3)]
    if choice == "auto":
        if extents is None:
            choice = "z"
            method = "default_no_geometry"
        else:
            choice = "xyz"[int(np.argmin(extents))]
            method = "smallest_overall_extent"
    else:
        method = "explicit_request"
    return {
        "axis": choice,
        "index": UP_INDEX[choice],
        "method": method,
        "model_extents_mm": extents,
        "caveat": (
            "an up axis inferred from CAD extents is a convention hint, not a gravity "
            "direction and not a mounting instruction"
        ),
    }


def horizontal_indices(up_index: int) -> List[int]:
    return [i for i in range(3) if i != up_index]


def axial_segments(cluster: dict) -> List[dict]:
    """Split one candidate axis line into the physically separate stacks on it.

    Two wheels on one axle line have a large empty stretch between them; without
    this split a report would silently present them as a single axle.
    """
    members = list(cluster.get("members") or ())
    if not members:
        return []
    direction = np.asarray(cluster["direction"], dtype=float)
    keyed = []
    for member in members:
        point = np.asarray(member["point_mm"], dtype=float)
        keyed.append((float(np.dot(point, direction)), member))
    keyed.sort(key=lambda item: item[0])
    runs: List[List[Tuple[float, dict]]] = [[keyed[0]]]
    for previous, current in zip(keyed, keyed[1:]):
        if current[0] - previous[0] > AXIAL_GAP_MM:
            runs.append([current])
        else:
            runs[-1].append(current)
    out = []
    for run in runs:
        points = np.array([member["point_mm"] for _, member in run], dtype=float)
        out.append(
            {
                "point_mm": [round(float(v), 4) for v in points.mean(axis=0)],
                "projection_min_mm": round(float(run[0][0]), 4),
                "projection_max_mm": round(float(run[-1][0]), 4),
                "radius_max_mm": round(max(member["radius_mm"] for _, member in run), 3),
                "feature_count": len(run),
                "members": [member for _, member in run],
            }
        )
    return out


def _corner_key(point: Sequence[float], up_index: int) -> str:
    return "".join("+" if float(point[i]) >= 0.0 else "-" for i in horizontal_indices(up_index))


def _corner_keys(up_index: int) -> List[str]:
    keys = [""]
    for _ in horizontal_indices(up_index):
        keys = [prefix + sign for prefix in keys for sign in ("+", "-")]
    return keys


def _mirror_point(point: Sequence[float], axis_index: int) -> np.ndarray:
    out = np.asarray(point, dtype=float).copy()
    out[axis_index] = -out[axis_index]
    return out


CONFIDENCE_RANK = {"LOW": 0, "MEDIUM": 1, "HIGH": 2}


def _best_segment(candidates: Sequence[dict]) -> Dict[str, object]:
    """Strongest candidate line in a bucket, with the evidence that ranked it."""
    if not candidates:
        return {}
    best = max(
        candidates,
        key=lambda s: (
            CONFIDENCE_RANK.get(s.get("confidence"), 0),
            int(s.get("feature_count", 0)),
            float(s.get("radius_max_mm") or 0.0),
        ),
    )
    return {
        "point_mm": best["point_mm"],
        "direction": best["direction"],
        "radius_max_mm": best["radius_max_mm"],
        "feature_count": best["feature_count"],
        "confidence": best.get("confidence"),
        "axial_span_mm": [best["projection_min_mm"], best["projection_max_mm"]],
        "step_eids": list(best["step_eids"])[:8],
    }


def corner_summary(segments: Sequence[dict], up_index: int, parts_by_eid: Dict[int, dict]) -> List[dict]:
    """Per-corner candidate wheel geometry: steering-like and drive-like lines."""
    rows = []
    for key in _corner_keys(up_index):
        bucket = [s for s in segments if s["corner"] == key]
        steering = [s for s in bucket if s["role"] == "steering_candidate_vertical"]
        drive = [s for s in bucket if s["role"] == "drive_candidate_horizontal"]
        owners = sorted({eid for s in bucket for eid in s["step_eids"]})
        named = sum(1 for eid in owners if parts_by_eid.get(eid, {}).get("named"))
        best_steering = _best_segment(steering)
        best_drive = _best_segment(drive)
        for candidate in (best_steering, best_drive):
            if candidate:
                candidate["part_names"] = [
                    parts_by_eid.get(eid, {}).get("name_leaf") for eid in candidate["step_eids"]
                ][:4]
        rows.append(
            {
                "corner": key,
                "steering_candidates": len(steering),
                "drive_candidates": len(drive),
                "drive_axial_span_mm": (
                    [
                        min(s["projection_min_mm"] for s in drive),
                        max(s["projection_max_mm"] for s in drive),
                    ]
                    if drive
                    else None
                ),
                "best_steering": best_steering,
                "best_drive": best_drive,
                "max_steering_circle_radius_mm": (
                    max(s["radius_max_mm"] for s in steering) if steering else None
                ),
                "max_drive_circle_radius_mm": (
                    max(s["radius_max_mm"] for s in drive) if drive else None
                ),
                "candidate_steering_axis_point_mm": best_steering.get("point_mm"),
                "candidate_drive_axis_point_mm": best_drive.get("point_mm"),
                "candidate_drive_direction": best_drive.get("direction"),
                "parts": len(owners),
                "named_parts": named,
                "modules": sorted({parts_by_eid.get(eid, {}).get("module") for eid in owners} - {None}),
                "part_names": [parts_by_eid.get(eid, {}).get("name_leaf") for eid in owners][:6],
            }
        )
    return rows


def symmetry_check(corners: Sequence[dict], up_index: int) -> List[dict]:
    """Compare mirrored corner pairs using their candidate steering axis points."""
    horizontal = horizontal_indices(up_index)
    pairs = []
    for first_index in range(len(horizontal)):
        axis_index = horizontal[first_index]
        for left in corners:
            for right in corners:
                if left["corner"] == right["corner"]:
                    continue
                expected = "".join(
                    ("-" if ch == "+" else "+") if position == first_index else ch
                    for position, ch in enumerate(left["corner"])
                )
                if expected != right["corner"]:
                    continue
                pairs.append((left, right, axis_index))
    seen = set()
    rows = []
    for left, right, axis_index in pairs:
        token = (frozenset((left["corner"], right["corner"])), axis_index)
        if token in seen:
            continue
        seen.add(token)
        a = left["candidate_steering_axis_point_mm"]
        b = right["candidate_steering_axis_point_mm"]
        row = {
            "corner_a": left["corner"],
            "corner_b": right["corner"],
            "mirror_axis": "xyz"[axis_index],
            "comparable": bool(a and b),
        }
        if a and b:
            mirrored = _mirror_point(a, axis_index)
            residual = [round(float(mirrored[i] - float(b[i])), 4) for i in range(3)]
            # The chosen point on an axis is arbitrary along that axis.
            # For a near-vertical steering candidate, a height difference in
            # those anchor points does NOT shift the steering axis footprint.
            # Preserve the full anchor residual for diagnostics, but compare
            # the horizontal projection for wheel-layout mirror symmetry.
            ground_residual = list(residual)
            ground_residual[up_index] = 0.0
            row["mirror_residual_mm"] = residual
            row["mirror_residual_norm_mm"] = round(float(np.linalg.norm(residual)), 4)
            row["ground_projection_residual_mm"] = ground_residual
            row["ground_projection_residual_norm_mm"] = round(
                float(np.linalg.norm(ground_residual)), 4
            )
            row["anchor_up_offset_mm"] = residual[up_index]
            row["symmetry_metric"] = "horizontal_axis_footprint"
            radius_a = left["max_steering_circle_radius_mm"]
            radius_b = right["max_steering_circle_radius_mm"]
            row["radius_a_mm"] = radius_a
            row["radius_b_mm"] = radius_b
            row["radius_delta_mm"] = (
                round(abs(float(radius_a) - float(radius_b)), 4)
                if radius_a is not None and radius_b is not None
                else None
            )
            row["consistent"] = bool(
                row["ground_projection_residual_norm_mm"] <= CORNER_POSITION_TOL_MM
                and (row["radius_delta_mm"] is None or row["radius_delta_mm"] <= CORNER_RADIUS_TOL_MM)
            )
        else:
            row["consistent"] = False
        rows.append(row)
    return rows


# ------------------------------------------------------------------------ groups
def module_of(name: str) -> str:
    """Second assembly segment, e.g. 'M1.2.2-00V3 舵轮前组件-1'."""
    if not name or name.strip() == "NONE":
        return "UNKNOWN"
    parts = [p.strip() for p in name.split("/") if p.strip()]
    if len(parts) >= 3:
        return parts[1]
    if len(parts) == 2:
        return parts[0]
    return "UNKNOWN"


def load_mapping(path: str) -> Dict[int, dict]:
    if not path or not os.path.isfile(path):
        return {}
    data = json.load(open(path))
    out: Dict[int, dict] = {}
    for pair in data.get("pairs", []):
        out[int(pair["step_eid"])] = {
            "usd_path": pair.get("usd_path"),
            "grade": pair.get("grade"),
            "usd_points": pair.get("usd_points"),
        }
    return out


def analyse(step_path: str, mapping_path: str, up_axis: str = "auto") -> dict:
    model = parse_step(step_path)
    coords = dict(zip(model.point_ids, model.points))
    mapping = load_mapping(mapping_path)

    solids = [e for e in model.entities.values() if set(e.components) & SOLID_TYPES]
    solids.sort(key=lambda e: e.eid)

    parts: List[dict] = []
    feature_by_solid: Dict[int, List[dict]] = {}
    for solid in solids:
        name = entity_name(solid)
        bbox = vertex_bbox(model, solid.eid)
        features = circular_features(model, solid.eid, coords)
        feature_by_solid[solid.eid] = features
        parts.append(
            {
                "step_eid": solid.eid,
                "step_type": solid.etype,
                "name": name,
                "name_leaf": name.split("/")[-1].strip() if name else "",
                "module": module_of(name),
                "named": bool(name and name.strip() != "NONE"),
                "usd_path": (mapping.get(solid.eid) or {}).get("usd_path"),
                "usd_mapping_grade": (mapping.get(solid.eid) or {}).get("grade"),
                "vertex_bbox_mm": (
                    {
                        "min": [round(float(v), 3) for v in bbox["min"]],
                        "max": [round(float(v), 3) for v in bbox["max"]],
                        "size": [round(float(v), 3) for v in bbox["size"]],
                    }
                    if bbox
                    else None
                ),
                "circular_features": len(features),
                "max_circle_radius_mm": (
                    round(max(f["radius_mm"] for f in features), 3) if features else None
                ),
            }
        )

    frame = detect_up_axis(parts, up_axis)
    up_index = int(frame["index"])
    parts_by_eid = {p["step_eid"]: p for p in parts}

    # tag every feature with its owning parts before clustering
    tagged: List[dict] = []
    for eid, features in feature_by_solid.items():
        for feature in features:
            item = dict(feature)
            item["step_eid"] = eid
            tagged.append(item)

    clusters = cluster_axes(tagged)
    cluster_rows: List[dict] = []
    for cluster in clusters:
        direction = cluster["direction"]
        owners = sorted({m["step_eid"] for m in cluster["members"]})
        owner_names = sorted({p["name"] for p in parts if p["step_eid"] in owners})
        named_owners = sum(1 for p in parts if p["step_eid"] in owners and p["named"])
        role = _role_of(direction, frame["axis"])
        modules = sorted({p["module"] for p in parts if p["step_eid"] in owners})
        up_axis_vector = np.zeros(3, dtype=float)
        up_axis_vector[up_index] = 1.0
        confidence = _confidence(cluster, owners, named_owners)
        split = axial_segments(cluster)
        for segment in split:
            segment["role"] = role
            segment["confidence"] = confidence
            segment["direction"] = [round(float(v), 6) for v in direction]
            segment["step_eids"] = sorted({m["step_eid"] for m in segment.pop("members")})
            segment["corner"] = _corner_key(segment["point_mm"], up_index)
        cluster_rows.append(
            {
                "role": role,
                "_segments": split,
                "axial_segment_count": len(split),
                "vertical_dot": round(abs(float(np.dot(direction, up_axis_vector))), 6),
                "direction": [round(float(v), 6) for v in direction],
                "axis_point_mm": [round(float(v), 4) for v in cluster["origin"]],
                "axial_extent_mm": cluster["extent_along_axis_mm"],
                "radius_mm": [cluster["min_radius_mm"], cluster["max_radius_mm"]],
                "feature_count": cluster["feature_count"],
                "feature_kinds": dict(Counter(m["kind"] for m in cluster["members"])),
                "step_eids": owners,
                "modules": modules,
                "part_names": [n for n in owner_names][:6],
                "confidence": confidence,
                "confidence_basis": {
                    "features": cluster["feature_count"],
                    "distinct_parts": len(owners),
                    "named_parts": named_owners,
                    "max_radius_mm": cluster["max_radius_mm"],
                },
            }
        )
    cluster_rows.sort(
        key=lambda row: (
            {"steering_candidate_vertical": 0, "drive_candidate_horizontal": 1}.get(row["role"], 2),
            -row["feature_count"],
            row["axis_point_mm"][up_index],
        )
    )
    segments: List[dict] = [segment for row in cluster_rows for segment in row.pop("_segments")]
    corner_rows = corner_summary(segments, up_index, parts_by_eid)
    symmetry_rows = symmetry_check(corner_rows, up_index)

    # ---- module consistency
    modules: Dict[str, List[dict]] = defaultdict(list)
    for part in parts:
        modules[part["module"]].append(part)
    module_rows = []
    for name, members in sorted(modules.items(), key=lambda kv: -len(kv[1])):
        boxes = [m["vertex_bbox_mm"] for m in members if m["vertex_bbox_mm"]]
        union = None
        if boxes:
            lo = [min(b["min"][i] for b in boxes) for i in range(3)]
            hi = [max(b["max"][i] for b in boxes) for i in range(3)]
            union = {
                "min": [round(v, 3) for v in lo],
                "max": [round(v, 3) for v in hi],
                "size": [round(hi[i] - lo[i], 3) for i in range(3)],
                "center": [round((hi[i] + lo[i]) / 2.0, 3) for i in range(3)],
            }
        module_rows.append(
            {
                "module": name,
                "solids": len(members),
                "named_solids": sum(1 for m in members if m["named"]),
                "union_vertex_bbox_mm": union,
                "step_eids": sorted(m["step_eid"] for m in members),
            }
        )

    return {
        "inputs": {
            "step": os.path.abspath(step_path),
            "mapping": os.path.abspath(mapping_path) if mapping_path else None,
        },
        "chassis_frame": frame,
        "tolerances": {
            "axis_angle_deg": AXIS_ANGLE_TOL_DEG,
            "axis_line_mm": AXIS_LINE_TOL_MM,
            "vertical_dot": VERTICAL_DOT,
            "horizontal_dot": HORIZONTAL_DOT,
            "axial_gap_mm": AXIAL_GAP_MM,
            "corner_position_tol_mm": CORNER_POSITION_TOL_MM,
            "corner_radius_tol_mm": CORNER_RADIUS_TOL_MM,
        },
        "parts": parts,
        "modules": module_rows,
        "axis_clusters": cluster_rows,
        "axis_segments": [
            {key: value for key, value in segment.items() if key != "members"}
            for segment in segments
        ],
        "corners": corner_rows,
        "symmetry": symmetry_rows,
        "notes": [
            "a shared cylinder axis is a manufactured rotation feature, not a proven joint",
            "a cylindrical surface is not necessarily a motor output shaft",
            "no joint, mass, inertia or wheel radius is asserted by this report",
            "role names describe the axis relative to the detected up axis, nothing more",
            "co-axial features far apart along one line are reported as separate axial segments",
            "corners are the four sign combinations of the two horizontal axes, not named wheels",
        ],
    }


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="Candidate wheel axes from real STEP geometry")
    parser.add_argument("--step", required=True)
    parser.add_argument("--mapping", default="")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--up-axis", default="auto", choices=list(UP_AXIS_CHOICES),
                        help="which CAD axis is vertical; 'auto' uses the smallest extent")
    args = parser.parse_args(argv)

    out_dir = args.output_dir
    os.makedirs(out_dir, exist_ok=True)
    report = analyse(args.step, args.mapping, up_axis=args.up_axis)

    json_path = os.path.join(out_dir, "wheel_kinematics.json")
    with open(json_path, "w", encoding="utf-8") as handle:
        json.dump(report, handle, ensure_ascii=False, indent=1)

    parts_csv = os.path.join(out_dir, "wheel_parts.csv")
    with open(parts_csv, "w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["step_eid", "step_type", "named", "module", "name_leaf",
                         "usd_path", "usd_mapping_grade", "size_x_mm", "size_y_mm",
                         "size_z_mm", "circular_features", "max_circle_radius_mm", "name"])
        for part in report["parts"]:
            size = (part["vertex_bbox_mm"] or {}).get("size") or ["", "", ""]
            writer.writerow([
                part["step_eid"], part["step_type"], part["named"], part["module"],
                part["name_leaf"], part["usd_path"], part["usd_mapping_grade"],
                size[0], size[1], size[2], part["circular_features"],
                part["max_circle_radius_mm"], part["name"],
            ])

    axes_csv = os.path.join(out_dir, "wheel_axis_candidates.csv")
    with open(axes_csv, "w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["role", "confidence", "up_parallel_dot", "dir_x", "dir_y", "dir_z",
                         "point_x_mm", "point_y_mm", "point_z_mm", "axial_extent_mm",
                         "axial_segments", "radius_min_mm", "radius_max_mm", "features",
                         "modules", "step_eids"])
        for row in report["axis_clusters"]:
            writer.writerow([
                row["role"], row["confidence"], row["vertical_dot"],
                *row["direction"], *row["axis_point_mm"], row["axial_extent_mm"],
                row["axial_segment_count"], row["radius_mm"][0], row["radius_mm"][1],
                row["feature_count"],
                ";".join(row["modules"]), ";".join(str(e) for e in row["step_eids"]),
            ])

    roles = Counter(row["role"] for row in report["axis_clusters"])
    print(json.dumps({
        "json": json_path, "parts_csv": parts_csv, "axes_csv": axes_csv,
        "solids": len(report["parts"]),
        "modules": [m["module"] for m in report["modules"]][:8],
        "up_axis": report["chassis_frame"]["axis"],
        "up_axis_method": report["chassis_frame"]["method"],
        "clusters": len(report["axis_clusters"]),
        "segments": len(report["axis_segments"]),
        "roles": dict(roles),
        "high_confidence": sum(1 for r in report["axis_clusters"] if r["confidence"] == "HIGH"),
        "corners_with_steering_candidate": sum(
            1 for c in report["corners"] if c["steering_candidates"] > 0
        ),
        "symmetry_pairs": len(report["symmetry"]),
        "symmetry_consistent": sum(1 for s in report["symmetry"] if s["consistent"]),
    }, ensure_ascii=False, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
