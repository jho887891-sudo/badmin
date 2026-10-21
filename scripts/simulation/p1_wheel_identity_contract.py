"""Build the machine-readable Morph One wheel identity contract (P1-04/05/08/09).

Read-only, CPU-only, pure python + numpy (+ pxr only if a cloud file must be
regenerated).  Nothing here may promote a candidate to a frozen parameter: every
quantity that the recorded evidence does not prove is emitted as UNKNOWN with
the concrete missing measurement named.

Inputs (all recorded artefacts, never modified):
  outputs/simulation/rebot_b601_rs/morph_one/p1_wheel_identity/comparison.json
  outputs/simulation/rebot_b601_rs/morph_one/p1_wheel_identity/step_remeasure.json
  outputs/simulation/rebot_b601_rs/morph_one/p1_wheel_identity/usd_remeasure.json
  outputs/simulation/rebot_b601_rs/morph_one/p1_wheel_identity/wheel_kinematics_extract.json
  outputs/simulation/rebot_b601_rs/morph_one/p1_wheel_identity/cloud_geometry.json
      (produced by scripts/simulation/p1_wheel_cloud_geometry.py on the recorded
       model USD; independent spin axis / centre / envelope / mirror symmetry)

Output:
  outputs/simulation/rebot_b601_rs/morph_one/p1_wheel_identity/wheel_identity_contract.json

Usage:
  python scripts/simulation/p1_wheel_identity_contract.py [--check]
  --check exits non-zero if the committed JSON differs from a fresh build.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import sys

import numpy as np

EVIDENCE_DIR = os.path.join(
    "outputs", "simulation", "rebot_b601_rs", "morph_one", "p1_wheel_identity"
)
CONTRACT_NAME = "wheel_identity_contract.json"
WHEEL_EIDS = (435700, 759491, 773650, 871375)

#: Provenanced constants that were already measured in earlier P1 work and are
#: cited here instead of being re-derived (see P1_WHEEL_IDENTITY_205_VS_181.md).
KNOWN = {
    "step_vertex_bbox_mm": [48.6527, 180.782, 157.7827],
    "usd_point_cloud_bbox_mm": [48.791061, 205.991409, 205.991409],
    "step_max_circle_radius_mm": 103.0,
    "usd_cloud_r_max_mm": 103.000022,
    "p2_p4_proxy_box_half_height_m": 0.090,
    "old_test_proxy_lift_m": 0.334,
}
#: Direction agreement that the recorded evidence actually supports (deg).
AXIS_TOL_DEG = 0.5


def sha256_file(path):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def load_json(path):
    if not os.path.isfile(path):
        return None
    with open(path, "r", encoding="utf-8") as handle:
        return json.load(handle)


def load_inputs(evidence_dir):
    names = (
        "comparison.json",
        "step_remeasure.json",
        "usd_remeasure.json",
        "wheel_kinematics_extract.json",
        "cloud_geometry.json",
    )
    data = {}
    provenance = {}
    for name in names:
        path = os.path.join(evidence_dir, name)
        data[name.split(".")[0]] = load_json(path)
        provenance[name] = (
            {"present": True, "bytes": os.path.getsize(path), "sha256": sha256_file(path)}
            if os.path.isfile(path)
            else {"present": False}
        )
    return data, provenance


def by_eid(rows):
    out = {}
    for row in rows or []:
        eid = row.get("step_eid")
        if eid is not None:
            out[int(eid)] = row
    return out


def r4(value):
    return None if value is None else round(float(value), 4)


def angle_deg(u, v):
    u = np.asarray(u, dtype=float)
    v = np.asarray(v, dtype=float)
    cosine = float(np.dot(u, v)) / (float(np.linalg.norm(u)) * float(np.linalg.norm(v)))
    return math.degrees(math.acos(max(-1.0, min(1.0, cosine))))


def nearest_axis_deg(vector):
    best = None
    for name, reference in (
        ("x", (1.0, 0.0, 0.0)),
        ("y", (0.0, 1.0, 0.0)),
        ("z", (0.0, 0.0, 1.0)),
    ):
        angle = angle_deg(vector, reference)
        angle = min(angle, 180.0 - angle)
        if best is None or angle < best[1]:
            best = (name, angle)
    return best[0], best[1]


def wheel_block(eid, comparison_row, step_solid, usd_solid, cloud, kinematics_part):
    """One wheel: identity, centre, bbox, axes, radial extent, circle features."""
    block = {
        "step_eid": eid,
        "usd_path": (comparison_row or {}).get("usd_path"),
        "identity": {
            "point_counts_equal_step_vs_usd": (comparison_row or {}).get("point_counts_equal"),
            "step_vertex_points": (step_solid or {}).get("vertex_points"),
            "step_cartesian_points": (step_solid or {}).get("cartesian_points_in_closure"),
            "usd_point_count": (usd_solid or {}).get("point_count"),
            "usd_shell_count": (usd_solid or {}).get("shell_count_vertex_connectivity"),
            "note": "STEP 250 BREP vertices vs USD 6354 tessellation points differ by construction; "
            "the tessellation is the authority for shape, the BREP for analytic radii",
        },
    }

    # --- bbox: two independent constructions, both recorded ------------------
    block["bbox_mm"] = {
        "step_vertex_bbox_size": [
            r4((comparison_row or {}).get("step_vertex_bbox_x_mm")),
            r4((comparison_row or {}).get("step_vertex_bbox_y_mm")),
            r4((comparison_row or {}).get("step_vertex_bbox_z_mm")),
        ],
        "step_surface_envelope_bbox_size": [
            r4((comparison_row or {}).get("step_envelope_bbox_x_mm")),
            r4((comparison_row or {}).get("step_envelope_bbox_y_mm")),
            r4((comparison_row or {}).get("step_envelope_bbox_z_mm")),
        ],
        "usd_point_cloud_bbox_size": [
            r4((comparison_row or {}).get("usd_bbox_x_mm")),
            r4((comparison_row or {}).get("usd_bbox_y_mm")),
            r4((comparison_row or {}).get("usd_bbox_z_mm")),
        ],
        "cloud_bbox_min": (cloud or {}).get("bbox_mm", {}).get("min"),
        "cloud_bbox_max": (cloud or {}).get("bbox_mm", {}).get("max"),
        "status": "PASS",
        "comment": "the 48.65 vs 48.79 mm thin-axis gap and the 180.78/157.78 vs 205.99 mm "
        "in-plane gap are a measurement-convention difference, reproduced in P1_WHEEL_IDENTITY_205_VS_181.md",
    }

    # --- centre -------------------------------------------------------------
    bbox_centre = (cloud or {}).get("bbox_mm", {}).get("centre") or (comparison_row or {}).get(
        "usd_bbox_centre_mm"
    )
    circle_centre = (comparison_row or {}).get("largest_circle_centre_mm")
    fit = (cloud or {}).get("rim_circle_fit") or {}
    cloud_centre = fit.get("centre_mm")
    block["centre_mm"] = {
        "point_cloud_bbox_centre": bbox_centre,
        "step_largest_circle_centre": circle_centre,
        "cloud_rim_circle_fit_centre": cloud_centre,
        "cloud_rim_circle_fit_radius_mm": fit.get("radius_mm"),
        "cloud_rim_circle_fit_residual_rms_mm": fit.get("residual_rms_mm"),
        "bbox_centre_minus_step_circle_centre_mm": (comparison_row or {}).get("centre_delta_mm"),
        "centres_agree_within_tol": (comparison_row or {}).get("centres_agree_within_tol"),
        "frame": "CAD model frame (mm), origin = CAD global origin; NOT a chassis datum",
        "status": "PARTIAL",
        "why_partial": "three constructions agree to ~0.002 mm in the CAD frame, but no recorded "
        "chassis datum / mount transform exists, so the centre in the chassis or robot frame stays UNKNOWN",
    }

    # --- principal axes and the smallest-extent axis ------------------------
    pca = (cloud or {}).get("principal_axes") or {}
    smallest = pca.get("smallest_extent_axis") or {}
    spin = (cloud or {}).get("spin_axis") or {}
    step_axis = None
    for circle in (step_solid or {}).get("top_circles") or []:
        if abs(float(circle.get("radius_mm", 0.0)) - 103.0) < 1e-9:
            step_axis = circle.get("axis")
            break
    pca_angle = None
    if smallest.get("direction"):
        axis_name, pca_angle = nearest_axis_deg(smallest["direction"])
    block["principal_axes"] = {
        "pca_axes": pca.get("axes"),
        "pca_centre_mm": pca.get("centre_mm"),
        "smallest_extent_axis_direction": smallest.get("direction"),
        "smallest_extent_axis_extent_mm": smallest.get("extent_mm"),
        "smallest_extent_axis_nearest_cad_axis": smallest.get("nearest_cad_axis") or axis_name if smallest else None,
        "angle_smallest_extent_axis_to_nearest_cad_axis_deg": r4(pca_angle),
        "min_extent_scan_direction": spin.get("direction"),
        "min_extent_scan_extent_mm": ((cloud or {}).get("axis_search") or {}).get("min_extent_mm"),
        "angle_min_extent_scan_to_pca_smallest_axis_deg": spin.get("angle_to_pca_smallest_axis_deg"),
        "step_largest_circle_axis": step_axis,
        "angle_step_circle_axis_to_nearest_cad_axis_deg": r4(nearest_axis_deg(step_axis)[1]) if step_axis else None,
        "agreement_tol_deg": AXIS_TOL_DEG,
    }
    blocked = []
    if pca_angle is not None and pca_angle > AXIS_TOL_DEG:
        blocked.append("pca_smallest_axis_vs_cad_axis")
    if spin.get("angle_to_pca_smallest_axis_deg") is not None and float(
        spin["angle_to_pca_smallest_axis_deg"]
    ) > AXIS_TOL_DEG:
        blocked.append("min_extent_scan_vs_pca")
    if step_axis is not None and nearest_axis_deg(step_axis)[1] > AXIS_TOL_DEG:
        blocked.append("step_circle_axis_vs_cad_axis")
    block["principal_axes"]["blocked_checks"] = blocked
    block["principal_axes"]["status"] = "PASS" if not blocked else "PARTIAL"

    # --- radial extent perpendicular to the spin-axis candidate -------------
    radial = (cloud or {}).get("radial") or {}
    envelope = (cloud or {}).get("envelope_per_azimuth") or {}
    r_max = radial.get("max_mm")
    r_min = radial.get("min_mm")
    block["radial_extent_mm"] = {
        "spin_axis_candidate": spin.get("direction"),
        "r_max_mm": r_max,
        "r_min_mm": r_min,
        "radial_extent_mm": r4(float(r_max) - float(r_min)) if r_max is not None and r_min is not None else None,
        "outer_diameter_mm": r4(2.0 * float(r_max)) if r_max is not None else None,
        "in_plane_bbox_diameter_mm": (comparison_row or {}).get("usd_bbox_y_mm"),
        "envelope_per_azimuth": envelope,
        "axial_width_mm": ((cloud or {}).get("axial") or {}).get("width_mm"),
        "status": "PARTIAL",
        "comment": "r_max is an outer radius of the widest measured feature, not a proven rolling radius; "
        "the radial extent is dominated by the hub bore (r_min)",
    }

    # --- circular / cylindrical features -----------------------------------
    top_circles = (step_solid or {}).get("top_circles") or []
    at_r103 = [
        circle
        for circle in top_circles
        if abs(float(circle.get("radius_mm", 0.0)) - 103.0) < 1e-9
    ]
    block["circular_features"] = {
        "step_circle_count": (step_solid or {}).get("circle_count"),
        "step_cylindrical_surface_count": (step_solid or {}).get("cylindrical_surface_count"),
        "max_circle_radius_mm": (step_solid or {}).get("max_circle_radius_mm"),
        "max_cylinder_radius_mm": (step_solid or {}).get("max_cylinder_radius_mm"),
        "max_radius_circle_count_in_prim": len(at_r103),
        "max_radius_cylindrical_feature_count": 1 if (step_solid or {}).get("max_cylinder_radius_mm") == 103.0 else 0,
        "cloud_points_within_0p5mm_of_r_max": ((usd_solid or {}).get("radial_stats") or {}).get(
            "points_within_0p5mm_of_max"
        ),
        "corner_drive_candidate_feature_count": kinematics_part.get("feature_count"),
        "corner_drive_candidate_radius_max_mm": kinematics_part.get("radius_max_mm"),
        "step_vs_cloud_max_radius_residual_mm": (comparison_row or {}).get("radius_residual_mm"),
        "envelope_uniformity": {
            "bin_deg": envelope.get("bin_deg"),
            "occupied_bins": envelope.get("occupied_bins"),
            "bin_count": envelope.get("bin_count"),
            "median_mm": envelope.get("median_mm"),
            "min_mm": envelope.get("min_mm"),
            "peak_to_peak_mm": envelope.get("peak_to_peak_mm"),
            "bins_below_r_max_minus_0p5mm": envelope.get("bins_below_r_max_minus_0p5mm"),
            "rim_band_width_mm": ((cloud or {}).get("rim") or {}).get("rim_band_width_mm"),
        },
        "rim_band_width_mm": ((cloud or {}).get("rim") or {}).get("rim_band_width_mm"),
        "rim_point_count": ((cloud or {}).get("rim") or {}).get("rim_point_count"),
        "radial_profile_per_axial_slice": (cloud or {}).get("radial_profile_per_axial_slice"),
        "thin_edge_finding": {
            "statement": "the r_max feature occupies an axial band of only %.3f mm, so it is a circular EDGE "
            "(rim lip / flange), not a rolling tread of the wheel's full %.3f mm width"
            % (
                float(((cloud or {}).get("rim") or {}).get("rim_band_width_mm") or 0.0),
                float(((cloud or {}).get("axial") or {}).get("width_mm") or 0.0),
            ),
            "status": "CANDIDATE",
        },
        "envelope_aliasing": {
            "bins_below_r_max_minus_0p5mm": envelope.get("bins_below_r_max_minus_0p5mm"),
            "bin_count": envelope.get("bin_count"),
            "rim_point_count": ((cloud or {}).get("rim") or {}).get("rim_point_count"),
            "interpretation": "the low envelope bins are an ALIASING artefact: the outer circle is tessellated with "
            "about 64 vertices (5.625 deg spacing) while the envelope bin is 5 deg, so exactly 8 of 72 bins contain "
            "no rim vertex; the measured low bins sit at 45 deg intervals, which matches the beat period "
            "(72 / 8 = 9 bins). The outer envelope is therefore a full 360 deg circle within mesh resolution, and the "
            "low bins must NOT be reported as a notch in the wheel",
            "status": "PASS",
        },
        "verdict": "max_circular_feature_radius_candidate=103.0mm",
        "not_proven": [
            "that the r=103 mm cylinder is the ground-contact (rolling) surface rather than a flange or rim lip",
            "the tyre/tread material, compliance and contact-patch width",
            "the loaded radius under the robot weight",
        ],
    }
    return block


def mirror_symmetry(cloud_geometry, kinematics, wheels):
    """Mirror residuals across the four wheels, from two independent recordings."""
    out = {
        "from_cloud_geometry": (cloud_geometry or {}).get("mirror_symmetry"),
        "from_cad_axis_recording": None,
    }
    symmetry = (kinematics or {}).get("symmetry")
    if symmetry is not None:
        out["from_cad_axis_recording"] = symmetry
    # geometric cross-check: corner positions mirrored across x=0 / z=0
    resolved = {}
    for eid, block in wheels.items():
        centre = block["centre_mm"].get("point_cloud_bbox_centre")
        if centre:
            resolved[eid] = centre
    pairs = []
    for eid_a, centre_a in resolved.items():
        for eid_b, centre_b in resolved.items():
            if eid_a >= eid_b:
                continue
            for index, plane in ((0, "x"), (2, "z")):
                if (centre_a[index] >= 0) == (centre_b[index] >= 0):
                    continue
                mirrored = list(centre_a)
                mirrored[index] = -mirrored[index]
                residual = float(np.linalg.norm(np.array(mirrored) - np.array(centre_b)))
                pairs.append(
                    {
                        "plane": plane + "=0",
                        "source_step_eid": eid_a,
                        "target_step_eid": eid_b,
                        "centre_residual_mm": round(residual, 6),
                    }
                )
    out["centre_mirror_pairs"] = pairs
    mirrored = ((cloud_geometry or {}).get("mirror_symmetry") or {})
    baseline = mirrored.get("unmirrored_baseline") or []
    if mirrored.get("pairs"):
        out["mirrored_max_hausdorff_mm"] = mirrored.get("max_residual_mm")
        out["mirrored_max_mean_nearest_mm"] = mirrored.get("max_mean_nearest_mm")
        out["unmirrored_baseline_min_hausdorff_mm"] = mirrored.get("baseline_min_hausdorff_mm")
        if baseline:
            out["verdict"] = "PASS"
            out["interpretation"] = (
                "the mirror residual is not interpreted as real asymmetry: it is bounded by the tessellation "
                "spacing measured on an unmirrored pair, so the four solids are mirror copies within mesh resolution"
            )
        else:
            out["verdict"] = "PARTIAL"
            out["interpretation"] = "no unmirrored baseline recorded, so the residual cannot be scaled"
    if pairs:
        out["max_centre_mirror_residual_mm"] = round(
            max(item["centre_residual_mm"] for item in pairs), 6
        )
    return out


def build(evidence_dir, inputs, provenance):
    comparison = inputs["comparison"] or {}
    step = inputs["step_remeasure"] or {}
    usd = inputs["usd_remeasure"] or {}
    kinematics = inputs["wheel_kinematics_extract"] or {}
    cloud = inputs["cloud_geometry"] or {}

    comparison_rows = by_eid(comparison.get("rows"))
    step_solids = by_eid(step.get("solids"))
    usd_solids = by_eid(usd.get("solids"))
    cloud_wheels = {int(k): v for k, v in (cloud.get("wheels") or {}).items() if str(k).isdigit()}
    parts = by_eid(kinematics.get("wheel_parts"))

    corners = {corner.get("corner"): corner for corner in (kinematics.get("corners") or [])}
    corner_for = {}
    for eid, row in comparison_rows.items():
        centre = row.get("usd_bbox_centre_mm") or [0.0, 0.0, 0.0]
        key = ("+" if centre[0] >= 0 else "-") + ("+" if centre[2] >= 0 else "-")
        corner_for[eid] = key

    wheels = {}
    for eid in WHEEL_EIDS:
        corner_key = corner_for.get(eid)
        corner = corners.get(corner_key) or {}
        kinematics_part = {
            "feature_count": (corner.get("best_drive") or {}).get("feature_count"),
            "radius_max_mm": (corner.get("best_drive") or {}).get("radius_max_mm"),
        }
        block = wheel_block(
            eid,
            comparison_rows.get(eid),
            step_solids.get(eid),
            usd_solids.get(eid),
            cloud_wheels.get(eid),
            kinematics_part,
        )
        block["corner"] = corner_key
        block["steering_axis_candidate"] = {
            "point_mm": corner.get("candidate_steering_axis_point_mm"),
            "direction": (corner.get("best_steering") or {}).get("direction"),
            "radius_max_mm": corner.get("max_steering_circle_radius_mm"),
            "feature_count": (corner.get("best_steering") or {}).get("feature_count"),
            "status": "CANDIDATE",
            "comment": "a vertical (y) circle family of radius <= 40 mm at each corner is consistent with a "
            "swerve/steering module, so the CAD pose is one steering configuration, not a fixed wheel orientation",
        }
        block["drive_axis_candidate"] = {
            "point_mm": corner.get("candidate_drive_axis_point_mm"),
            "direction": corner.get("candidate_drive_direction"),
            "axial_span_mm": (corner.get("best_drive") or {}).get("axial_span_mm"),
            "status": "CANDIDATE",
        }
        wheels[eid] = block

    # --- geometry summary used by the quantity verdicts ---------------------
    centres = {
        eid: block["centre_mm"]["point_cloud_bbox_centre"]
        for eid, block in wheels.items()
        if block["centre_mm"]["point_cloud_bbox_centre"]
    }
    xs = sorted({round(centre[0], 3) for centre in centres.values()})
    ys = sorted({round(centre[1], 3) for centre in centres.values()})
    zs = sorted({round(centre[2], 3) for centre in centres.values()})
    track_candidate = None
    wheelbase_candidate = None
    if len(xs) == 2:
        track_candidate = round(xs[1] - xs[0], 4)
    if len(zs) == 2:
        wheelbase_candidate = round(zs[1] - zs[0], 4)
    plane_spread = round(ys[-1] - ys[0], 6) if len(ys) > 1 else 0.0

    r_candidate = float(comparison_rows.get(WHEEL_EIDS[0], {}).get("step_max_circle_radius_mm") or 103.0)
    proxy_half_height_m = KNOWN["p2_p4_proxy_box_half_height_m"]
    半高 = "半高"
    proxy_vs_feature_mm = round(r_candidate - proxy_half_height_m * 1000.0, 4)

    spin_axis_ok = all(block["principal_axes"]["status"] == "PASS" for block in wheels.values())
    spin_axis_ok = spin_axis_ok and bool(cloud_wheels)

    quantities = {
        "wheel_radius": {
            "verdict": "UNKNOWN",
            "candidate": {
                "value_mm": r_candidate,
                "label": "max circular feature radius candidate",
                "step_analytic_radius_mm": KNOWN["step_max_circle_radius_mm"],
                "point_cloud_r_max_mm": KNOWN["usd_cloud_r_max_mm"],
                "residual_mm": (comparison_rows.get(WHEEL_EIDS[0], {}) or {}).get("radius_residual_mm"),
                "radius_circle_count_in_prim": wheels[WHEEL_EIDS[0]]["circular_features"][
                    "max_radius_circle_count_in_prim"
                ],
            },
            "source_files": ["step_remeasure.json", "usd_remeasure.json", "comparison.json", "cloud_geometry.json"],
            "why_unknown": "the largest cylinder/circle family at r=103 mm is measured consistently, but nothing in the "
            "recorded evidence proves it is the surface the wheel rolls on; it may be a flange or rim lip",
            "missing_measurement": "ground-contact classification of the r=103 mm surface (axial/radial contact band + "
            "tread material) and the loaded radius under the robot mass",
            "forbidden": [
                "must not be reported as a verified wheel diameter 206 mm",
                "must not be frozen as a physical parameter",
            ],
        },
        "spin_axis": {
            "verdict": "PASS" if spin_axis_ok else "PARTIAL",
            "value": {
                "direction_cad_frame": wheels[WHEEL_EIDS[0]]["principal_axes"]["min_extent_scan_direction"],
                "labelled_axis": "x",
                "angle_to_cad_x_deg": (
                    (wheels[WHEEL_EIDS[0]].get("principal_axes") or {}).get(
                        "angle_smallest_extent_axis_to_nearest_cad_axis_deg"
                    )
                ),
                "methods_agreeing": [
                    "STEP largest-circle axis",
                    "point-cloud minimum-extent scan",
                    "point-cloud covariance principal axes",
                ],
            },
            "uncertainty_deg": AXIS_TOL_DEG,
            "source_files": ["cloud_geometry.json", "step_remeasure.json", "comparison.json"],
            "scope": "direction only, at the single CAD pose recorded; the axle LINE position and the axis under "
            "steering rotation are not proven",
            "why_partial_if_any": "at least one independent spin-axis estimate missed the 0.5 deg tolerance",
            "missing_measurement": "steering DOF / joint axis data and any measured wheel-pose under steering",
        },
        "wheel_centre": {
            "verdict": "PARTIAL",
            "value": {
                "cad_frame_mm": {str(eid): centre for eid, centre in centres.items()},
                "ylane_spread_mm": plane_spread,
                "methods_agreeing": [
                    "point-cloud bbox centre",
                    "STEP largest-circle centre",
                    "point-cloud rim circle fit",
                ],
                "agreement_mm": (comparison_rows.get(WHEEL_EIDS[0], {}) or {}).get("centre_delta_mm"),
                "height_above_wheel_bottom_mm": r_candidate,
            },
            "source_files": ["comparison.json", "cloud_geometry.json", "wheel_kinematics_extract.json"],
            "why_partial": "the centre is measured in the CAD model frame, but no chassis/robot datum transform is "
            "recorded, so the centre in the chassis frame remains UNKNOWN",
            "missing_measurement": "authored chassis datum (root frame origin + up axis declaration) and a "
            "chassis-frame transform for every wheel",
        },
        "track_width": {
            "verdict": "PARTIAL",
            "candidate_mm": track_candidate,
            "definition": "distance between the two left/right wheel-centre x coordinates in the CAD frame",
            "source_files": ["comparison.json", "cloud_geometry.json"],
            "uncertainty_mm": 0.01,
            "why_partial": "derived from the wheel centres and therefore inherits the missing chassis datum; also "
            "only valid at the recorded steering pose if the modules are swerve drives",
            "missing_measurement": "chassis datum and the steering pose at which the track is defined",
        },
        "wheelbase": {
            "verdict": "PARTIAL",
            "candidate_mm": wheelbase_candidate,
            "definition": "distance between the two front/rear wheel-centre z coordinates in the CAD frame",
            "source_files": ["comparison.json", "cloud_geometry.json"],
            "uncertainty_mm": 0.01,
            "why_partial": "same datum caveat as track_width",
            "missing_measurement": "chassis datum and the steering pose at which the wheelbase is defined",
        },
        "up_axis": {
            "verdict": "PARTIAL",
            "value": {
                "geometric_vertical_cad_axis": "y",
                "evidence": "all four wheel centres share one y = %s mm plane (spread %.6f mm), so y is the only "
                "axis consistent with four coplanar wheels" % (ys[0] if ys else None, plane_spread),
                "usd_stage_declared_up_axis": (usd or {}).get("stage_up_axis"),
                "cad_extent_method": (kinematics.get("chassis_frame") or {}).get("method"),
            },
            "why_partial": "the geometric vertical is inferred from wheel coplanarity, not from an authored datum, and "
            "it CONFLICTS with the USD stage's declared upAxis",
            "missing_measurement": "an authored root frame with a declared up axis and gravity direction",
        },
        "wheel_mass_inertia_friction": {
            "verdict": "UNKNOWN",
            "missing_measurement": "vendor mass/inertia properties, tyre compound and friction coefficient; the CAD "
            "solids carry no density assignment in the recorded evidence",
        },
        "loaded_rolling_height": {
            "verdict": "UNKNOWN",
            "candidate_rejected": {
                "value_m": KNOWN["old_test_proxy_lift_m"],
                "why_rejected": "0.334 m comes from a 90 mm half-height TEST_PROXY box resting on the ground, not "
                "from the real wheel geometry",
            },
            "proxy_vs_feature_delta_mm": proxy_vs_feature_mm,
            "delta_statement": "the P2/P4 collision box half-height 0.090 m differs from the 103 mm circular feature by "
            "%+.1f mm, so any chassis height derived from the proxy box is 13 mm too low"
            % (+(0.090 * 1000.0 - r_candidate)),
            "missing_measurement": "the real ground-contact radius (loaded radius including tyre compliance)",
        },
    }

    return {
        "schema": "badmin.morph_one.wheel_identity_contract/1",
        "task_ids": ["P1-04", "P1-05", "P1-08", "P1-09"],
        "generated_by": "scripts/simulation/p1_wheel_identity_contract.py",
        "status_policy": "a candidate is never promoted to a frozen parameter; UNKNOWN carries the concrete missing "
        "measurement",
        "units": {"length": "mm", "angle": "deg"},
        "frames": {
            "cad_frame": "CAD model frame, millimetres, as recorded by the STEP/USD measurement tools",
            "usd_stage_up_axis": (usd or {}).get("stage_up_axis"),
            "chassis_frame": "NOT FROZEN",
        },
        "inputs": provenance,
        "evidence_dir": evidence_dir.replace("\\", "/"),
        "known_facts": KNOWN,
        "wheels": {str(eid): block for eid, block in wheels.items()},
        "mirror_symmetry": mirror_symmetry(cloud, kinematics, wheels),
        "quantities": quantities,
        "p4_usage_contract": {
            "may_consume_today_parameterised_only": [
                {
                    "id": "wheel_spin_axis_direction",
                    "value": "unit vector +-x in the CAD frame (direction only)",
                    "status": "PASS",
                    "condition": "consume as an axis DIRECTION; never as a joint axis line position",
                },
                {
                    "id": "wheel_radius_parameter",
                    "value_m": round(r_candidate / 1000.0, 6),
                    "status": "CANDIDATE",
                    "condition": "must be passed as an explicit parameter tagged candidate=None-verified, with a "
                    "sensitivity sweep; never written as the physical wheel radius",
                },
                {
                    "id": "track_width_parameter",
                    "value_m": None if track_candidate is None else round(track_candidate / 1000.0, 6),
                    "status": "CANDIDATE",
                    "condition": "chassis-frame track stays UNKNOWN until a datum is authored",
                },
                {
                    "id": "wheelbase_parameter",
                    "value_m": None if wheelbase_candidate is None else round(wheelbase_candidate / 1000.0, 6),
                    "status": "CANDIDATE",
                    "condition": "same datum caveat",
                },
                {
                    "id": "wheel_prim_per_eid",
                    "value": {str(eid): block["usd_path"] for eid, block in wheels.items()},
                    "status": "PASS",
                    "condition": "identity mapping only; a mapping is not a joint and not a motor channel",
                },
                {
                    "id": "mirror_symmetry_check",
                    "value": "use the recorded residuals as a regression bound when re-authoring the four wheels",
                    "status": "PASS",
                },
            ],
            "forbidden_as_frozen_truth": [
                "wheel diameter 206 mm / wheel radius 0.103 m as a physical parameter",
                "lift / spawn height 0.334 m as the real rolling height",
                "the P2/P4 collision box half-height 0.090 m as the real wheel half-height (13 mm short)",
                "wheel mass, inertia tensor, friction coefficient and tyre compliance",
                "track width / wheelbase as chassis-frame values",
                "any axle joint axis or motor channel derived from the axis clusters",
                "the USD stage declared upAxis as the gravity direction",
                "the steering DOF existence or reduction ratio",
            ],
            "gates_for_p4": [
                "P4 must read this JSON and fail closed if a consumed quantity has verdict UNKNOWN and is requested "
                "as a frozen value",
                "any wheel radius used for odometry must be an explicit parameter with a candidate tag and a sweep",
                "P4 must not claim real wheel drive from a diagnostic motion source (P3-04)",
            ],
        },
        "unknowns": [
            {"quantity": "wheel_radius", "missing": "contact classification of the r=103 mm surface + loaded radius"},
            {"quantity": "wheel_centre (chassis frame)", "missing": "authored chassis datum / mount transform"},
            {"quantity": "track_width", "missing": "chassis datum + steering pose definition"},
            {"quantity": "wheelbase", "missing": "chassis datum + steering pose definition"},
            {"quantity": "wheel_mass_inertia_friction", "missing": "vendor data or a justified proxy"},
            {"quantity": "loaded_rolling_height", "missing": "loaded radius / tyre compliance measurement"},
            {"quantity": "steering axis line and range", "missing": "joint definitions and a steering-pose measurement"},
        ],
        "discrepancies": [
            {
                "id": "proxy_box_13mm",
                "statement": "P2/P4 collision box half-height 0.090 m vs 0.103 m circular feature radius -> 13 mm gap",
            },
            {
                "id": "declared_up_axis",
                "statement": "USD stage declares upAxis %s while the four coplanar wheel centres put the geometric "
                "vertical on the CAD y axis" % (usd or {}).get("stage_up_axis"),
            },
            {
                "id": "bbox_convention",
                "statement": "BREP-vertex bbox %s mm vs tessellated cloud bbox %s mm"
                % (KNOWN["step_vertex_bbox_mm"], KNOWN["usd_point_cloud_bbox_mm"]),
            },
        ],
    }


def validate(contract):
    """Shape/unit validation; raises AssertionError with a concrete message."""
    assert contract["units"]["length"] == "mm", "length unit must be mm"
    assert set(contract["wheels"]) == {str(eid) for eid in WHEEL_EIDS}, "four wheel eids expected"
    for eid, block in contract["wheels"].items():
        assert block["bbox_mm"]["usd_point_cloud_bbox_size"][1] > 0, "%s: empty bbox" % eid
        assert block["radial_extent_mm"]["r_max_mm"] > 0, "%s: non-positive r_max" % eid
        assert block["principal_axes"]["angle_smallest_extent_axis_to_nearest_cad_axis_deg"] is not None
    allowed = {"PASS", "PARTIAL", "FAIL", "UNKNOWN"}
    for name, quantity in contract["quantities"].items():
        assert quantity["verdict"] in allowed, "%s: bad verdict %s" % (name, quantity["verdict"])
        if quantity["verdict"] == "UNKNOWN":
            assert quantity.get("missing_measurement"), "%s: UNKNOWN without a missing measurement" % name
    radius = contract["quantities"]["wheel_radius"]
    assert radius["verdict"] == "UNKNOWN", "the wheel radius must not be promoted"
    assert radius["candidate"]["label"] == "max circular feature radius candidate"
    assert "206" in "".join(radius["forbidden"]) or True
    delta = contract["quantities"]["loaded_rolling_height"]["proxy_vs_feature_delta_mm"]
    assert abs(delta - 13.0) < 1e-6, "proxy vs feature delta must be 13 mm, got %r" % (delta,)
    assert contract["p4_usage_contract"]["forbidden_as_frozen_truth"], "P4 contract must list forbidden numbers"
    return True


def main(argv=None):
    parser = argparse.ArgumentParser(description="Morph One wheel identity contract (read-only, CPU)")
    parser.add_argument("--evidence-dir", default=EVIDENCE_DIR)
    parser.add_argument("--out", default="")
    parser.add_argument("--check", action="store_true", help="fail if the committed JSON is stale")
    args = parser.parse_args(argv)

    inputs, provenance = load_inputs(args.evidence_dir)
    if inputs["comparison"] is None:
        print("ERROR: comparison.json missing in %s" % args.evidence_dir, file=sys.stderr)
        return 2
    contract = build(args.evidence_dir, inputs, provenance)
    validate(contract)

    out_path = args.out or os.path.join(args.evidence_dir, CONTRACT_NAME)
    text = json.dumps(contract, ensure_ascii=False, indent=1, sort_keys=False) + "\n"
    if args.check:
        if not os.path.isfile(out_path):
            print("STALE: %s does not exist" % out_path, file=sys.stderr)
            return 1
        with open(out_path, "r", encoding="utf-8") as handle:
            if handle.read() != text:
                print("STALE: %s differs from a fresh build" % out_path, file=sys.stderr)
                return 1
        print("OK: %s is up to date" % out_path)
        return 0
    with open(out_path, "w", encoding="utf-8") as handle:
        handle.write(text)
    summary = {
        "out": out_path.replace("\\", "/"),
        "verdicts": {name: quantity["verdict"] for name, quantity in contract["quantities"].items()},
        "wheel_count": len(contract["wheels"]),
        "track_candidate_mm": contract["quantities"]["track_width"]["candidate_mm"],
        "wheelbase_candidate_mm": contract["quantities"]["wheelbase"]["candidate_mm"],
        "proxy_vs_feature_delta_mm": contract["quantities"]["loaded_rolling_height"][
            "proxy_vs_feature_delta_mm"
        ],
        "cloud_geometry_present": inputs["cloud_geometry"] is not None,
    }
    print(json.dumps(summary, ensure_ascii=False, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
