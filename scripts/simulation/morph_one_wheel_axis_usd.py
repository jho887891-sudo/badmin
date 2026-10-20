"""Standalone visualization of candidate wheel axes, written as plain USD text.

Why a text writer instead of Isaac Sim
--------------------------------------
The visualization only needs line primitives whose coordinates are already known
from the STEP analysis, so it is written directly as USDA text.  That keeps the
artifact reproducible on a CPU-only machine, reviewable in a diff, and impossible
to confuse with the CAD source: nothing here opens, edits or re-exports the STEP
file or the converted USD asset.

What the picture means
----------------------
Every curve is a *candidate* axis derived from co-axial BREP circles and cylinders,
drawn in the CAD frame in millimetres with the detected up axis declared in the
layer metadata.  A drawn axis is evidence of a manufactured rotation feature; it
is not a joint, not a motor shaft, and no mass, inertia or wheel radius is implied.

Usage
    python scripts/simulation/morph_one_wheel_axis_usd.py \
        --report outputs/.../wheel_kinematics.json --output outputs/.../wheel_axes.usda
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys

ALNUM = re.compile(r"[^0-9A-Za-z_]+")
CONFIDENCE_ORDER = {"LOW": 0, "MEDIUM": 1, "HIGH": 2}
ROLE_COLOURS = {
    "steering_candidate_vertical": (0.85, 0.15, 0.15),
    "drive_candidate_horizontal": (0.15, 0.35, 0.85),
    "oblique_or_other": (0.55, 0.55, 0.55),
}
BOX_COLOUR = (0.65, 0.65, 0.65)
BOX_WIDTH = 1.0
AXIS_WIDTH = 1.5
CIRCLE_SAMPLES = 48
DEFAULT_MIN_CONFIDENCE = "MEDIUM"
DEFAULT_MAX_PER_CORNER = 8


def sanitize(name: str) -> str:
    """Turn any label into a valid USD identifier."""
    cleaned = ALNUM.sub("_", str(name)).strip("_")
    if not cleaned:
        return "_"
    if cleaned[0].isdigit():
        cleaned = "_" + cleaned
    return cleaned


def corner_label(key: str) -> str:
    """'+' / '-' sign pattern -> a stable identifier such as 'Corner_Plus_Minus'."""
    words = {"+": "Plus", "-": "Minus"}
    return "Corner_" + "_".join(words.get(ch, sanitize(ch)) for ch in str(key))


def _fmt(value: float) -> str:
    return ("%.6f" % float(value)).rstrip("0").rstrip(".") or "0"


def _vec(values) -> str:
    return "(" + ", ".join(_fmt(v) for v in values) + ")"


def _unit(direction):
    norm = sum(float(v) * float(v) for v in direction) ** 0.5
    if norm <= 0.0:
        return None
    return [float(v) / norm for v in direction]


def model_bounds(parts):
    """Union of the per-part vertex bounding boxes, in millimetres."""
    boxes = [p["vertex_bbox_mm"] for p in parts if p.get("vertex_bbox_mm")]
    if not boxes:
        return None
    low = [min(b["min"][i] for b in boxes) for i in range(3)]
    high = [max(b["max"][i] for b in boxes) for i in range(3)]
    return low, high


def select_segments(report, min_confidence=DEFAULT_MIN_CONFIDENCE,
                    max_per_corner=DEFAULT_MAX_PER_CORNER, min_feature_count=3,
                    min_radius_mm=5.0):
    """Keep the strongest candidate axes per corner, deterministically ordered."""
    threshold = CONFIDENCE_ORDER.get(str(min_confidence).upper())
    if threshold is None:
        raise ValueError("unknown confidence %r" % (min_confidence,))
    kept = []
    for segment in report.get("axis_segments", []):
        if CONFIDENCE_ORDER.get(segment.get("confidence"), 0) < threshold:
            continue
        if int(segment.get("feature_count", 0)) < int(min_feature_count):
            continue
        if float(segment.get("radius_max_mm") or 0.0) < float(min_radius_mm):
            continue
        kept.append(segment)
    by_corner = {}
    for segment in kept:
        by_corner.setdefault(segment["corner"], []).append(segment)
    selected = []
    for key in sorted(by_corner):
        items = sorted(
            by_corner[key],
            key=lambda s: (
                -CONFIDENCE_ORDER.get(s.get("confidence"), 0),
                -int(s.get("feature_count", 0)),
                -float(s.get("radius_max_mm") or 0.0),
                tuple(s.get("point_mm") or (0.0, 0.0, 0.0)),
            ),
        )
        selected.extend(items[: int(max_per_corner)])
    return selected


def _curve_block(indent: str, name: str, points, colour, width):
    lines = [
        indent + 'def BasisCurves "%s"' % sanitize(name),
        indent + "{",
        indent + '    uniform token type = "linear"',
        indent + "    int[] curveVertexCounts = [%d]" % len(points),
        indent + "    point3f[] points = [" + ", ".join(_vec(p) for p in points) + "]",
        indent + "    float[] widths = [%s] (" % _fmt(width),
        indent + '        interpolation = "constant"',
        indent + "    )",
        indent + "    color3f[] primvars:displayColor = [%s] (" % _vec(colour),
        indent + '        interpolation = "constant"',
        indent + "    )",
        indent + "}",
    ]
    return lines


def _box_edges(low, high):
    x0, y0, z0 = low
    x1, y1, z1 = high
    corners = [
        (x0, y0, z0), (x1, y0, z0), (x1, y1, z0), (x0, y1, z0),
        (x0, y0, z1), (x1, y0, z1), (x1, y1, z1), (x0, y1, z1),
    ]
    pairs = [(0, 1), (1, 2), (2, 3), (3, 0), (4, 5), (5, 6), (6, 7), (7, 4),
             (0, 4), (1, 5), (2, 6), (3, 7)]
    points = []
    for a, b in pairs:
        points.extend([corners[a], corners[b]])
    return points


def _circle_points(point, direction, radius, samples=CIRCLE_SAMPLES):
    unit = _unit(direction)
    if unit is None:
        return []
    seed = [1.0, 0.0, 0.0]
    if abs(unit[0]) > 0.9:
        seed = [0.0, 1.0, 0.0]
    u = [
        seed[1] * unit[2] - seed[2] * unit[1],
        seed[2] * unit[0] - seed[0] * unit[2],
        seed[0] * unit[1] - seed[1] * unit[0],
    ]
    u = _unit(u)
    v = _unit([
        unit[1] * u[2] - unit[2] * u[1],
        unit[2] * u[0] - unit[0] * u[2],
        unit[0] * u[1] - unit[1] * u[0],
    ])
    import math

    points = []
    for index in range(samples + 1):
        angle = 2.0 * math.pi * index / samples
        cos_a, sin_a = math.cos(angle), math.sin(angle)
        points.append([
            float(point[i]) + float(radius) * (cos_a * u[i] + sin_a * v[i])
            for i in range(3)
        ])
    return points


def build_usda(report, min_confidence=DEFAULT_MIN_CONFIDENCE,
               max_per_corner=DEFAULT_MAX_PER_CORNER, source_note=""):
    """Render the whole layer as text, deterministically."""
    frame = report.get("chassis_frame") or {}
    up_axis = str(frame.get("axis", "z")).upper()
    bounds = model_bounds(report.get("parts") or [])
    selected = select_segments(report, min_confidence, max_per_corner)

    header = [
        "#usda 1.0",
        "(",
        '    defaultPrim = "MorphOneWheelAxisCandidates"',
        "    metersPerUnit = 0.001",
        '    upAxis = "%s"' % up_axis,
        '    doc = "Candidate wheel axes derived from STEP BREP circles and cylinders. '
        'Millimetre CAD coordinates, no source geometry is modified and no joint is created."',
        ")",
        "",
        'def Xform "MorphOneWheelAxisCandidates"',
        "{",
        '    custom string evidence = "%s"' % str(source_note).replace('"', "'"),
        '    custom string up_axis_basis = "%s"' % str(frame.get("method", "unknown")),
    ]
    if bounds:
        low, high = bounds
        header.append('    custom float3 model_bounds_min_mm = %s' % _vec(low))
        header.append('    custom float3 model_bounds_max_mm = %s' % _vec(high))

    body = []
    if bounds:
        body.extend(_curve_block("    ", "ChassisBoundingBox",
                                 _box_edges(bounds[0], bounds[1]), BOX_COLOUR, BOX_WIDTH))

    body.append('    def Scope "Corners"')
    body.append("    {")
    groups = {}
    for segment in selected:
        groups.setdefault(segment["corner"], []).append(segment)
    counts = {"axes": 0, "rings": 0}
    for key in sorted(groups):
        body.append('        def Scope "%s"' % corner_label(key))
        body.append("        {")
        body.append('            custom string corner_signs = "%s"' % str(key))
        for index, segment in enumerate(groups[key], start=1):
            direction = segment.get("direction") or [0.0, 0.0, 1.0]
            point = segment.get("point_mm") or [0.0, 0.0, 0.0]
            unit = _unit(direction) or [0.0, 0.0, 1.0]
            span_low = float(segment.get("projection_min_mm", 0.0))
            span_high = float(segment.get("projection_max_mm", 0.0))
            if abs(span_high - span_low) < 1.0:
                span_low, span_high = -20.0, 20.0
            base = float(unit[0]) * float(point[0]) + float(unit[1]) * float(point[1]) \
                + float(unit[2]) * float(point[2])
            endpoints = []
            for projection in (span_low, span_high):
                delta = projection - base
                endpoints.append([float(point[i]) + delta * float(unit[i]) for i in range(3)])
            colour = ROLE_COLOURS.get(segment.get("role"), ROLE_COLOURS["oblique_or_other"])
            role_word = sanitize(segment.get("role", "axis"))
            name = "%s_%02d_%s_%s" % (role_word, index, segment.get("confidence", "LOW"),
                                      int(segment.get("feature_count", 0)))
            body.extend(_curve_block("            ", name, endpoints, colour, AXIS_WIDTH))
            body.append('            custom string %s_parts = "%s"' % (
                sanitize(name),
                ",".join(str(e) for e in (segment.get("step_eids") or [])[:8]),
            ))
            counts["axes"] += 1
            radius = segment.get("radius_max_mm")
            if radius and float(radius) >= 5.0:
                ring = _circle_points(point, unit, float(radius))
                if ring:
                    body.extend(_curve_block(
                        "            ",
                        "%s_%02d_measured_ring_r%s" % (role_word, index, _fmt(radius)),
                        ring,
                        colour,
                        AXIS_WIDTH,
                    ))
                    counts["rings"] += 1
        body.append("        }")
    body.append("    }")
    body.append("}")

    text = "\n".join(header + body) + "\n"
    summary = {
        "up_axis": up_axis,
        "corners": sorted(groups),
        "axes": counts["axes"],
        "rings": counts["rings"],
        "selected_segments": len(selected),
        "min_confidence": min_confidence,
        "max_per_corner": max_per_corner,
    }
    return text, summary


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="Write candidate wheel axes as USDA text")
    parser.add_argument("--report", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--min-confidence", default=DEFAULT_MIN_CONFIDENCE,
                        choices=["LOW", "MEDIUM", "HIGH"])
    parser.add_argument("--max-per-corner", type=int, default=DEFAULT_MAX_PER_CORNER)
    args = parser.parse_args(argv)

    with open(args.report, encoding="utf-8") as handle:
        report = json.load(handle)
    note = os.path.basename(args.report)
    text, summary = build_usda(report, args.min_confidence, args.max_per_corner, note)
    directory = os.path.dirname(os.path.abspath(args.output))
    if directory:
        os.makedirs(directory, exist_ok=True)
    with open(args.output, "w", encoding="utf-8", newline="\n") as handle:
        handle.write(text)
    summary["output"] = os.path.abspath(args.output)
    summary["bytes"] = os.path.getsize(args.output)
    summary["lines"] = text.count("\n")
    print(json.dumps(summary, ensure_ascii=False, indent=1))
    return 0


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
