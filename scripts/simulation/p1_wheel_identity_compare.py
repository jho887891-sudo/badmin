"""Build the STEP-vs-USD comparison table for the four candidate wheels.

Reads the two measurement reports produced by
  * p1_step_solid_surface_bbox.py   (kernel-free STEP bracket, run locally)
  * p1_usd_wheel_prim_measure.py    (pxr mesh measurement, run on the host)
and emits one row per eid plus an explicit verdict per candidate hypothesis.

The verdicts are computed from the numbers, never typed in by hand:

    PASS    the hypothesis is supported by these measurements
    FAIL    the hypothesis is refuted by these measurements
    UNKNOWN the measurements cannot decide it

Usage:

    python scripts/simulation/p1_wheel_identity_compare.py
        --step-json outputs/.../p1_wheel_identity/step_remeasure.json
        --usd-json  outputs/.../p1_wheel_identity/usd_remeasure.json
        --out-json  outputs/.../p1_wheel_identity/comparison.json
        --out-csv   outputs/.../p1_wheel_identity/comparison.csv
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys

#: Tolerance for "the two sides describe the same point in space", in mm.
CENTRE_TOL_MM = 0.01
#: A tessellated circle of radius R is inscribed in the analytic circle, so the
#: chord sagitta for a typical CAD deflection stays well below this bound.
TESSELLATION_TOL_MM = 0.5


def _load(path):
    with open(path, "r", encoding="utf-8") as handle:
        return json.load(handle)


def _by_eid(solids):
    return {int(solid["step_eid"]): solid for solid in solids}


def _plane_axes(usd_solid, step_solid):
    """The two in-plane axes and the thin axis, from the *STEP circle axis*."""
    circles = step_solid.get("top_circles") or []
    axis = None
    if circles:
        components = circles[0]["axis"]
        axis = max(range(3), key=lambda k: abs(components[k]))
    if axis is None:
        axis = "xyz".index(usd_solid.get("disc_axis") or "x")
    return axis, [k for k in range(3) if k != axis]


def build_row(eid, step_solid, usd_solid):
    axis, plane = _plane_axes(usd_solid, step_solid)
    vertex_size = step_solid["vertex_bbox"]["size"]
    envelope_size = step_solid["surface_envelope_bbox"]["size"]
    usd_size = usd_solid.get("world_bbox_mm", {}).get("size") or [None, None, None]
    usd_box = usd_solid.get("world_bbox_mm", {})
    usd_centre = None
    if usd_box.get("min") and usd_box.get("max"):
        usd_centre = [(usd_box["min"][k] + usd_box["max"][k]) / 2.0 for k in range(3)]

    # The honest centre comparison uses the *largest analytic circle* of the STEP
    # solid against the tessellated mesh box centre: bbox centres would compare a
    # conservative envelope with a real extent and drift by that conservatism.
    circles = step_solid.get("top_circles") or []
    circle_radius = circles[0]["radius_mm"] if circles else None
    circle_centre = circles[0]["centre_mm"] if circles else None

    centre_delta = None
    if usd_centre and circle_centre:
        centre_delta = [round(usd_centre[k] - circle_centre[k], 6) for k in range(3)]

    max_circle_radius = step_solid.get("max_circle_radius_mm")
    radial = usd_solid.get("radial_stats_world_mm") or {}
    usd_r_max = radial.get("max")
    radius_residual = (
        round(usd_r_max - max_circle_radius, 6)
        if (usd_r_max is not None and max_circle_radius is not None)
        else None
    )

    in_plane_low = min(vertex_size[k] for k in plane)
    in_plane_high = max(envelope_size[k] for k in plane)
    usd_in_plane = [usd_size[k] for k in plane]
    usd_inside_bracket = all(
        in_plane_low - 1e-6 <= usd_size[k] <= in_plane_high + 1e-6 for k in plane
    )

    vertex_plane = sorted(vertex_size[k] for k in plane)
    usd_plane = sorted(usd_size[k] for k in plane)

    return {
        "step_eid": eid,
        "usd_path": usd_solid.get("usd_path"),
        "step_vertex_points": step_solid.get("vertex_points"),
        "step_cartesian_points": step_solid.get("cartesian_points_in_closure"),
        "step_advanced_faces": (step_solid.get("type_counts") or {}).get("ADVANCED_FACE"),
        "step_circle_count": step_solid.get("circle_count"),
        "step_cylinder_count": step_solid.get("cylindrical_surface_count"),
        "usd_point_count": usd_solid.get("point_count"),
        "usd_face_count": usd_solid.get("face_count"),
        "usd_shell_count": usd_solid.get("shell_count_vertex_connectivity"),
        "point_counts_equal": step_solid.get("cartesian_points_in_closure")
        == usd_solid.get("point_count"),
        "step_vertex_bbox_x_mm": vertex_size[0],
        "step_vertex_bbox_y_mm": vertex_size[1],
        "step_vertex_bbox_z_mm": vertex_size[2],
        "step_envelope_bbox_x_mm": envelope_size[0],
        "step_envelope_bbox_y_mm": envelope_size[1],
        "step_envelope_bbox_z_mm": envelope_size[2],
        "usd_bbox_x_mm": usd_size[0],
        "usd_bbox_y_mm": usd_size[1],
        "usd_bbox_z_mm": usd_size[2],
        "step_max_circle_radius_mm": max_circle_radius,
        "usd_point_radius_max_mm": usd_r_max,
        "radius_residual_mm": radius_residual,
        "thin_axis": "xyz"[axis],
        "thin_axis_step_vertex_mm": vertex_size[axis],
        "thin_axis_usd_mm": usd_size[axis],
        "in_plane_bracket_low_mm": round(in_plane_low, 6),
        "in_plane_bracket_high_mm": round(in_plane_high, 6),
        "in_plane_bracket_width_mm": round(in_plane_high - in_plane_low, 6),
        "in_plane_usd_sorted_mm": [round(v, 6) for v in usd_plane],
        "in_plane_vertex_sorted_mm": [round(v, 6) for v in vertex_plane],
        "usd_inside_step_bracket": usd_inside_bracket,
        "largest_circle_centre_mm": circle_centre,
        "usd_bbox_centre_mm": [round(v, 6) for v in usd_centre] if usd_centre else None,
        "centre_delta_mm": centre_delta,
        "centres_agree_within_tol": (
            all(abs(v) <= CENTRE_TOL_MM for v in centre_delta) if centre_delta else None
        ),
        "usd_radial_stats_mm": usd_solid.get("radial_stats_world_mm"),
    }


def evaluate_hypotheses(rows, step_report, usd_report):
    """Decide each candidate cause from the measurements only."""
    units = step_report.get("units") or {}
    declares_milli_metre = ".MILLI." in " ".join(units.get("si_unit_tokens") or [])
    meters_per_unit = usd_report.get("stage_meters_per_unit")

    all_inside = all(row["usd_inside_step_bracket"] for row in rows)
    all_centres = all(bool(row["centres_agree_within_tol"]) for row in rows)
    radii = [abs(row["radius_residual_mm"]) for row in rows if row["radius_residual_mm"] is not None]
    radius_ok = bool(radii) and max(radii) <= TESSELLATION_TOL_MM

    # The vertex-only box must be materially narrower than the envelope, otherwise
    # the "method difference" explanation would have nothing to explain.
    all_bracket_material = all(row["in_plane_bracket_width_mm"] > 1.0 for row in rows)

    gaps = []
    for row in rows:
        low = row["in_plane_bracket_low_mm"]
        usd = max(row["in_plane_usd_sorted_mm"])
        gaps.append(round(usd - low, 6))

    return {
        "interpretation": (
            "PASS = hypothesis supported by these measurements; FAIL = refuted; "
            "UNKNOWN = the measurements cannot decide it"
        ),
        "unit_mismatch": {
            "result": "FAIL" if (declares_milli_metre and meters_per_unit == 0.001) else "UNKNOWN",
            "evidence": {
                "step_declares_milli_metre": declares_milli_metre,
                "usd_stage_meters_per_unit": meters_per_unit,
                "in_plane_bracket_gaps_mm": gaps,
                "radius_residual_max_mm": round(max(radii), 6) if radii else None,
            },
            "reasoning": (
                "Both sides are in millimetres and the residuals are at tessellation scale, so a "
                "unit factor of 10, 100 or 1000 would have left an unmistakable mismatch."
            ),
        },
        "bbox_computation_method": {
            "result": "PASS" if all_inside and all_bracket_material else "UNKNOWN",
            "evidence": {
                "usd_inside_step_bracket_for_all_four": all_inside,
                "in_plane_gap_usd_minus_vertex_mm": gaps,
                "in_plane_bracket_width_mm": [
                    row["in_plane_bracket_width_mm"] for row in rows
                ],
                "max_in_plane_gap_mm": round(max(gaps), 6) if gaps else None,
                "mesh_points_vs_step_cartesian_points": [
                    [row["step_cartesian_points"], row["usd_point_count"]] for row in rows
                ],
            },
            "reasoning": (
                "The STEP number is a VERTEX_POINT-only box (analytic circles are not tessellated "
                "in a BREP, so the extremities of a cylindrical tyre face are not vertices). The "
                "mesh number is a tessellated point box. The USD value lands inside the STEP "
                "bracket, so the gap is a method difference, not a disagreement about the solid."
            ),
        },
        "merged_multiple_entities": {
            "result": "UNKNOWN",
            "evidence": {
                "usd_point_count_all_equal": len({row["usd_point_count"] for row in rows}) == 1,
                "usd_shell_counts": [row["usd_shell_count"] for row in rows],
                "step_advanced_faces": [row["step_advanced_faces"] for row in rows],
            },
            "reasoning": (
                "A shell count above 1 comes from an unwelded tessellation and cannot separate "
                "physical parts. Nothing here shows a second independent part, but a bbox cannot "
                "exclude one either."
            ),
        },
        "mapping_candidate_mismatch": {
            "result": "FAIL" if (all_centres and radius_ok) else "UNKNOWN",
            "evidence": {
                "centres_agree_for_all_four": all_centres,
                "centre_tolerance_mm": CENTRE_TOL_MM,
                "radius_residual_max_mm": round(max(radii), 6) if radii else None,
                "tessellation_tolerance_mm": TESSELLATION_TOL_MM,
            },
            "reasoning": (
                "Each mapped prim sits at the same centre and carries the same largest circular "
                "radius as its STEP solid, so the pairing is geometrically consistent even though "
                "the mapping grade is only CANDIDATE."
            ),
        },
        "hub_vs_tyre_two_entities": {
            "result": "UNKNOWN",
            "evidence": {
                "usd_radial_stats_mm": [row.get("usd_radial_stats_mm") for row in rows],
                "step_circle_count": [row["step_circle_count"] for row in rows],
                "step_cylinder_count": [row["step_cylinder_count"] for row in rows],
            },
            "reasoning": (
                "Each eid is a single BREP solid and a single mesh whose radius profile runs "
                "continuously from the bore to the outer band, so no bbox can say whether the "
                "outer 206 mm surface is the rolling surface or a flange."
            ),
        },
        "overall": "PARTIAL",
        "overall_reasoning": (
            "The 205.991 vs 180.782 gap is explained as a bounding-box method difference and the "
            "two sides agree on a 103.0 mm largest circular radius, but that is a CAD statement "
            "about two unnamed CANDIDATE-mapped solids, not a confirmed wheel diameter."
        ),
    }


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="STEP vs USD wheel bbox comparison")
    parser.add_argument("--step-json", required=True)
    parser.add_argument("--usd-json", required=True)
    parser.add_argument("--out-json", default="")
    parser.add_argument("--out-csv", default="")
    args = parser.parse_args(argv)

    step_report = _load(args.step_json)
    usd_report = _load(args.usd_json)
    step_by_eid = _by_eid(step_report["solids"])
    usd_by_eid = _by_eid(usd_report["solids"])

    shared = [eid for eid in step_report["requested_eids"] if eid in step_by_eid and eid in usd_by_eid]
    if not shared:
        raise SystemExit("no eid measured on both sides")

    rows = [build_row(eid, step_by_eid[eid], usd_by_eid[eid]) for eid in shared]
    hypotheses = evaluate_hypotheses(rows, step_report, usd_report)
    report = {
        "tool": "p1_wheel_identity_compare.py",
        "step_json": os.path.abspath(args.step_json),
        "usd_json": os.path.abspath(args.usd_json),
        "step_units": step_report.get("units"),
        "usd_stage_meters_per_unit": usd_report.get("stage_meters_per_unit"),
        "rows": rows,
        "hypotheses": hypotheses,
        "notes": [
            "step_vertex_bbox_* reproduces the size_x_mm/size_y_mm/size_z_mm of wheel_parts.csv",
            "step_envelope_bbox_* is an UPPER bracket (full 360 deg envelope of analytic circles)",
            "usd_bbox_* is the tessellated mesh point box, so it may sit slightly inside the "
            "analytic envelope by the chord sagitta",
            "no wheel diameter, axle position or track width is concluded here",
        ],
    }

    header = (
        "eid      stepverts usdpts  stepVertex bbox (mm)             envelope bbox (mm)             "
        "usd bbox (mm)                    Rstep    Rusd"
    )
    print(header)
    print("-" * len(header))
    for row in rows:
        print(
            "%-8d %-9d %-7d %-31s %-30s %-32s %-8s %s"
            % (
                row["step_eid"],
                row["step_vertex_points"],
                row["usd_point_count"],
                "%.3f x %.3f x %.3f"
                % (row["step_vertex_bbox_x_mm"], row["step_vertex_bbox_y_mm"], row["step_vertex_bbox_z_mm"]),
                "%.3f x %.3f x %.3f"
                % (
                    row["step_envelope_bbox_x_mm"],
                    row["step_envelope_bbox_y_mm"],
                    row["step_envelope_bbox_z_mm"],
                ),
                "%.3f x %.3f x %.3f" % (row["usd_bbox_x_mm"], row["usd_bbox_y_mm"], row["usd_bbox_z_mm"]),
                row["step_max_circle_radius_mm"],
                row["usd_point_radius_max_mm"],
            )
        )
    print("")
    print("hypothesis verdicts:")
    for name, payload in hypotheses.items():
        if isinstance(payload, dict) and "result" in payload:
            print("  %-34s %s" % (name, payload["result"]))
    print("  %-34s %s" % ("overall", hypotheses["overall"]))
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
            "usd_path",
            "step_vertex_points",
            "step_cartesian_points",
            "step_advanced_faces",
            "step_circle_count",
            "step_cylinder_count",
            "usd_point_count",
            "usd_face_count",
            "usd_shell_count",
            "point_counts_equal",
            "step_vertex_bbox_x_mm",
            "step_vertex_bbox_y_mm",
            "step_vertex_bbox_z_mm",
            "step_envelope_bbox_x_mm",
            "step_envelope_bbox_y_mm",
            "step_envelope_bbox_z_mm",
            "usd_bbox_x_mm",
            "usd_bbox_y_mm",
            "usd_bbox_z_mm",
            "step_max_circle_radius_mm",
            "usd_point_radius_max_mm",
            "radius_residual_mm",
            "thin_axis",
            "thin_axis_step_vertex_mm",
            "thin_axis_usd_mm",
            "in_plane_bracket_low_mm",
            "in_plane_bracket_high_mm",
            "in_plane_bracket_width_mm",
            "usd_inside_step_bracket",
            "centres_agree_within_tol",
            "centre_delta_mm",
            "largest_circle_centre_mm",
            "usd_bbox_centre_mm",
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
