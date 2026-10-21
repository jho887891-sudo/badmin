"""P1 v2: per-wheel AXLE geometry in CAD and chassis frames (evidence-derived).

Reads the P1 (PR #25) contract and the v2 BREP face census, then writes
wheel_geometry_v2.json.  Every field carries a verdict (verified / candidate /
UNKNOWN), a unit, a frame, a source hash and a residual.  No value is promoted
from candidate to verified by this script: promotion needs a new measurement.

Pure CPU, standard library only.

    python scripts/simulation/p1_wheel_axle_geometry.py            # build
    python scripts/simulation/p1_wheel_axle_geometry.py --check    # staleness
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import sys

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

EVID = "outputs/simulation/rebot_b601_rs/morph_one/p1_wheel_identity"
CONTRACT = EVID + "/wheel_identity_contract.json"
FACES = EVID + "/wheel_brep_faces_v2.json"
CENTRES = EVID + "/wheel_centres_cad_mm.json"
OUT = EVID + "/wheel_geometry_v2.json"

WHEEL_ORDER = ("435700", "759491", "773650", "871375")
SCALE = 0.001          # mm -> m, the documented CAD->USD scale
ROT_DEG_X = 90.0       # documented CAD->USD rotation about x

#: P4 currently authors wheels with a 90 mm half-height proxy box (PR #25 finding)
P2_P4_PROXY_HALF_HEIGHT_M = 0.090


def sha256_of(path: str) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def rot_x(vec, degrees):
    """R_x: the documented CAD->USD rotation (see morph_one_orientation_probe)."""
    rad = math.radians(degrees)
    c, s = math.cos(rad), math.sin(rad)
    x, y, z = vec
    return [x, c * y - s * z, s * y + c * z]


def cad_mm_to_chassis_m(point_mm):
    return rot_x([float(v) * SCALE for v in point_mm], ROT_DEG_X)


def chassis_m_to_cad_mm(point_m):
    rad = math.radians(-ROT_DEG_X)
    c, s = math.cos(rad), math.sin(rad)
    x, y, z = point_m
    back = [x / SCALE, (c * y - s * z) / SCALE, (s * y + c * z) / SCALE]
    return back


def _r(value, nd=6):
    return round(float(value), nd)


def build(repo_root: str) -> dict:
    with open(os.path.join(repo_root, CONTRACT), encoding="utf-8") as handle:
        contract = json.load(handle)
    with open(os.path.join(repo_root, FACES), encoding="utf-8") as handle:
        faces = json.load(handle)
    with open(os.path.join(repo_root, CENTRES), encoding="utf-8") as handle:
        centres_in = json.load(handle)

    sources = {
        "contract": {"path": CONTRACT, "sha256": sha256_of(os.path.join(repo_root, CONTRACT))},
        "brep_faces_v2": {"path": FACES, "sha256": sha256_of(os.path.join(repo_root, FACES))},
        "centres_cad_mm": {"path": CENTRES, "sha256": sha256_of(os.path.join(repo_root, CENTRES))},
    }
    step_sources = contract.get("inputs", {}).get("step_remeasure.json", {})
    sources["step_remeasure_recorded"] = step_sources

    faces_by_eid = {str(s["step_eid"]): s for s in faces["solids"]}

    wheels = {}
    for eid in WHEEL_ORDER:
        wheel = contract["wheels"][eid]
        centre_cad = [float(v) for v in wheel["centre_mm"]["cloud_rim_circle_fit_centre"]]
        centre_step = [float(v) for v in wheel["centre_mm"]["step_largest_circle_centre"]]
        axis_pca = [float(v) for v in wheel["principal_axes"]["smallest_extent_axis_direction"]]
        axis_scan = [float(v) for v in wheel["radial_extent_mm"]["spin_axis_candidate"]]
        axis_step = [float(v) for v in wheel["principal_axes"]["step_largest_circle_axis"]]
        centre_chassis = cad_mm_to_chassis_m(centre_cad)
        roundtrip_mm = chassis_m_to_cad_mm(cad_mm_to_chassis_m(centre_cad))
        roundtrip = [_r(roundtrip_mm[i] - centre_cad[i], 9) for i in range(3)]
        residual_vs_step = [centre_cad[i] - centre_step[i] for i in range(3)]
        angle_pca_step = math.degrees(
            math.acos(max(-1.0, min(1.0, sum(axis_pca[i] * axis_step[i] for i in range(3)))))
        )
        angle_scan_step = math.degrees(
            math.acos(max(-1.0, min(1.0, sum(axis_scan[i] * axis_step[i] for i in range(3)))))
        )
        direction_chassis = rot_x(axis_step, ROT_DEG_X)
        # An axle LINE has no intrinsic sign: the stored BREP placement axis is
        # (+1,0,0) on the +x wheels and (-1,0,0) on the -x wheels (mirrored
        # placements).  The outward sign therefore comes from the measured
        # centre, never from the stored axis.
        outward_sign = 1.0 if centre_chassis[0] > 0.0 else -1.0
        stored_axis_parallel = 0.0 if axis_step[0] * outward_sign > 0.0 else 180.0
        faces_v2 = faces_by_eid.get(eid, {})
        wheels[eid] = {
            "usd_path": wheel["usd_path"],
            "axle_line_cad": {
                "point_mm": [_r(v, 6) for v in centre_cad],
                "direction_unit": [_r(v, 6) for v in axis_step],
                "line_form": "point + t * direction, t in millimetres, unrestricted",
                "verdict": "candidate",
                "why": "direction is measured three independent ways; the point is a "
                       "measured wheel centre, but the CAD global origin is not a chassis datum",
                "unit": "mm",
                "frame": "CAD model frame (global origin, +x = spin axis family)",
                "residual": {
                    "centre_vs_step_analytic_circle_mm": [_r(v, 6) for v in residual_vs_step],
                    "centre_norm_mm": _r(math.sqrt(sum(v * v for v in residual_vs_step))),
                    "axis_angle_pca_vs_step_deg": _r(angle_pca_step, 4),
                    "axis_angle_minscan_vs_step_deg": _r(angle_scan_step, 4),
                },
                "stored_brep_axis_sign": _r(axis_step[0], 6),
                "sign_note": "the stored CIRCLE placement axis is (+1,0,0) on the +x wheels and "
                             "(-1,0,0) on the -x wheels: an axle line is unsigned, and the flip is "
                             "itself mirror evidence, never a spin direction",
                "source": {"file": CONTRACT, "sha256": sources["contract"]["sha256"],
                           "locator": "wheels.%s.centre_mm/principal_axes" % eid},
            },
            "axle_line_chassis": {
                "point_m": [_r(v, 9) for v in centre_chassis],
                "direction_unit": [_r(v, 6) for v in direction_chassis],
                "verdict": "candidate",
                "why": "the CAD->USD mapping (0.001 scale + %g deg about x) is documented and "
                       "applied, but the chassis frame origin/up axis are not vendor-authorized, "
                       "so chassis-frame values stay candidates" % ROT_DEG_X,
                "unit": "m",
                "frame": "chassis body frame = R_x(%g deg) . (0.001 * CAD mm), Z-up" % ROT_DEG_X,
                "residual": {
                    "transform_roundtrip_mm": roundtrip,
                    "path_residual_mm": _r(math.sqrt(sum(v * v for v in residual_vs_step))),
                    "note": "the mapping is a similarity (orthogonal + uniform 0.001), so it "
                            "cannot change the path residual: the two paths differ by the CAD-frame "
                            "0.002 mm centre offset, which is 2e-06 m after the transform",
                },
                "source": {"file": CONTRACT, "sha256": sources["contract"]["sha256"],
                           "locator": "frames.chassis_frame = NOT FROZEN + documented mapping"},
            },
            "outward_axle_direction": {
                "direction_unit_chassis": [_r(outward_sign, 6), 0.0, 0.0],
                "side": "right (+x)" if outward_sign > 0 else "left (-x)",
                "spin_sign": int(outward_sign),
                "spin_convention": "omega_vector = spin_sign * (v_roll / r) * x_hat gives a wheel "
                                   "rolling with +v_roll along chassis -y; the forward direction "
                                   "itself is UNKNOWN",
                "verdict": "candidate",
                "unit": "unit vector",
                "frame": "chassis body frame",
                "residual": {
                    "angle_to_measured_pca_axis_deg": _r(angle_pca_step, 4),
                    "stored_axis_parallel_vs_outward_deg": stored_axis_parallel,
                    "mirror_residual_deg": 0.0,
                    "note": "all four solids are parallel to chassis x to 0.0043 deg (PCA) / "
                            "0.165 deg (min-span scan); the outward sign is assigned exactly by "
                            "the sign of the measured chassis x of the centre",
                },
            },
            "rolling_surface": _rolling_surface(faces_v2, sources),
        }

    return {
        "schema": "badmin.morph_one.wheel_geometry_v2/1",
        "generated_by": "scripts/simulation/p1_wheel_axle_geometry.py",
        "task_ids": ["P1-Rnext-6", "P1-Rnext-7"],
        "status_policy": contract["status_policy"],
        "units": {"length": "mm", "length_chassis": "m", "angle": "deg", "area": "mm^2"},
        "frames": {
            "cad": "CAD model frame, millimetres, as recorded by the STEP/USD measurement tools",
            "chassis": "chassis body frame = R_x(%g deg) . (0.001 * CAD mm); origin = CAD global "
                       "origin, which is NOT a calibrated chassis datum" % ROT_DEG_X,
            "chassis_mapping_evidence": "scripts/simulation/morph_one_orientation_probe.py "
                                        "(applied by p2_chassis_frame_math.py consumers)",
            "up_axis_note": "the mapping sends the geometric vertical (CAD y) to chassis z, which "
                            "agrees with the USD stage upAxis = Z; this RESOLVES the PR #25 "
                            "up-axis conflict as a candidate, and still is not a vendor statement",
        },
        "sources": sources,
        "wheels": wheels,
        "chassis_aggregates": _aggregates(wheels, centres_in),
        "part_identity": _part_identity(faces_by_eid),
        "p4_consumption_contract_v2": _p4_contract(wheels),
        "unknowns": _unknowns(),
    }


def _rolling_surface(faces_v2: dict, sources: dict) -> dict:
    groups = faces_v2.get("groups", [])
    tread = [g for g in groups if g["key"].startswith("TOROIDAL_SURFACE") and g["touches_r103_circle"]]
    tread_group = tread[0] if tread else None
    torus_faces = [f for f in faces_v2.get("faces", [])
                   if f.get("surface", {}).get("type") == "TOROIDAL_SURFACE"
                   and f.get("touches_r103_circle")]
    surface = torus_faces[0]["surface"] if torus_faces else {}
    major = surface.get("major_radius")
    minor = surface.get("minor_radius")
    area = round(sum(f["area_mm2"] for f in torus_faces), 4) if torus_faces else None
    minor_band = [min(f["minor_parameter_band_rad"][0] for f in torus_faces),
                  max(f["minor_parameter_band_rad"][1] for f in torus_faces)] if torus_faces else None
    cylinders = [g for g in groups if g["key"].startswith("CYLINDRICAL_SURFACE")]
    max_cylinder = max((g["radius_mm"] for g in cylinders if g["radius_mm"]), default=None)
    return {
        "verdict": "verified",
        "unit": "surface type / mm / mm^2",
        "frame": "CAD BREP analytic surfaces (no tessellation)",
        "residual": {
            "equator_vs_measured_circle_mm": _r(major + minor - 103.0, 9) if major else None,
            "max_cylinder_radius_in_solid_mm": max_cylinder,
            "note": "the surface type is analytic, so there is no tessellation residual; the only "
                    "residual is between the torus equator and the independently measured circle",
        },
        "classification": {
            "value": "TOROIDAL_SURFACE crowned tread band (two torus faces per axial half, "
                     "180 deg major each)",
            "verdict": "verified",
            "why": "every face bounding the 103 mm edge circle is an analytic TOROIDAL_SURFACE "
                   "with major radius 77.738 mm and minor radius 25.262 mm; no cylindrical face "
                   "of that radius exists in the solid",
            "unit": "surface type",
            "frame": "CAD BREP (analytic surface entities, no tessellation involved)",
            "residual": {
                "equator_recompute_mm": _r(major + minor, 6) if major else None,
                "equator_measured_circle_radius_mm": 103.0,
                "equator_residual_mm": _r(major + minor - 103.0, 6) if major else None,
                "max_cylinder_radius_in_solid_mm": max_cylinder,
                "cylinder_minus_103_mm": _r(max_cylinder - 103.0, 6) if max_cylinder else None,
            },
            "source": {"file": FACES, "sha256": sources["brep_faces_v2"]["sha256"],
                       "locator": "solids[*].groups (touches_r103_circle)"},
        },
        "tread_geometry": {
            "major_radius_mm": _r(major, 6) if major else None,
            "minor_radius_mm": _r(minor, 6) if minor else None,
            "radial_band_mm": tread_group["radial_band_mm"] if tread_group else None,
            "axial_band_mm": tread_group["axial_band_mm"] if tread_group else None,
            "axial_band_width_mm": _r(tread_group["axial_band_mm"][1] - tread_group["axial_band_mm"][0], 6)
            if tread_group else None,
            "minor_parameter_band_rad": [_r(v, 6) for v in minor_band] if minor_band else None,
            "minor_parameter_band_deg": [_r(math.degrees(v), 4) for v in minor_band] if minor_band else None,
            "area_mm2": area,
            "face_count": len(torus_faces),
            "verdict": "verified",
            "unit": "mm / rad / mm^2",
            "frame": "CAD model frame, axial band along the wheel spin axis through the wheel centre",
            "residual": {
                "area_method": "torus integral r*[R*dtheta_minor + r*sin] * dtheta_major, "
                               "dtheta from the face's own circular edges",
                "area_method_limit": "the major-angle coverage comes from circle-arc sums; a face "
                                     "trimmed by a spline rather than circles would be under-counted",
            },
            "source": {"file": FACES, "sha256": sources["brep_faces_v2"]["sha256"],
                       "locator": "solids[0].faces (TOROIDAL_SURFACE, touches_r103_circle)"},
        },
        "face_group_table_mm2": [
            {
                "group": g["key"],
                "surface_type": g["surface_type"],
                "radius_mm": g["radius_mm"],
                "face_count": g["face_count"],
                "area_mm2": g["area_mm2"],
                "area_unknown_faces": g["area_unknown_faces"],
                "axial_band_mm": g["axial_band_mm"],
                "axial_width_mm": g["axial_band_mm"][1] - g["axial_band_mm"][0],
                "radial_band_mm": g["radial_band_mm"],
                "touches_r103_circle": g["touches_r103_circle"],
            }
            for g in groups
        ],
        "rejected_hypotheses": [
            {"hypothesis": "the rolling surface is the 103 mm edge's adjacent cylinder",
             "verdict": "rejected",
             "evidence": "the largest cylindrical face radius in each wheel solid is 68.5 mm; "
                         "there is no cylinder at 92.5 mm or 103 mm"},
            {"hypothesis": "the rolling surface is a 49 mm wide flat cylindrical tread",
             "verdict": "rejected",
             "evidence": "the 103 mm circle is the equator of a torus; the tread band spans a "
                         "41.0 mm axial band but the radius drops to 92.5 mm at its edges"},
            {"hypothesis": "the wheel is a plain disc with a 206 mm flange",
             "verdict": "rejected",
             "evidence": "radial band 92.5 -> 103.0 mm on a 41 mm wide torus; the flange "
                         "hypothesis predicted a planar annulus, and the total planar area per "
                         "wheel is only 1883.2 mm^2"},
        ],
    }


def _aggregates(wheels: dict, centres_in: dict) -> dict:
    xs = [wheels[e]["axle_line_chassis"]["point_m"][0] for e in WHEEL_ORDER]
    ys = [wheels[e]["axle_line_chassis"]["point_m"][1] for e in WHEEL_ORDER]
    zs = [wheels[e]["axle_line_chassis"]["point_m"][2] for e in WHEEL_ORDER]
    track = max(xs) - min(xs)
    wheelbase = max(ys) - min(ys)
    height = max(zs) - min(zs)
    cad_x = [centres_in[e][0] for e in WHEEL_ORDER]
    cad_z = [centres_in[e][2] for e in WHEEL_ORDER]
    return {
        "track_width": {
            "value_m": _r(track, 9),
            "verdict": "candidate",
            "why": "measured relative geometry in the chassis frame; the frame origin and the "
                   "steering pose remain unauthorised",
            "uncertainty_m": 4e-6,
            "unit": "m",
            "frame": "chassis body frame, lateral = chassis x (CAD x)",
            "residual": {"cad_x_span_mm": _r(max(cad_x) - min(cad_x), 6),
                         "centre_fit_residual_mm": 0.002},
            "source": {"file": "contract + v2", "locator": "wheels[*].axle_line_chassis.point_m"},
        },
        "wheelbase": {
            "value_m": _r(wheelbase, 9),
            "verdict": "candidate",
            "why": "same as track_width; note the wheelbase is along chassis y, which is CAD z, "
                   "not chassis z",
            "uncertainty_m": 4e-6,
            "unit": "m",
            "frame": "chassis body frame, fore-aft = chassis y (CAD z)",
            "residual": {"cad_z_span_mm": _r(max(cad_z) - min(cad_z), 6),
                         "centre_fit_residual_mm": 0.002},
            "source": {"file": "contract + v2", "locator": "wheels[*].axle_line_chassis.point_m"},
        },
        "wheel_centre_plane_spread": {
            "value_m": _r(height, 9),
            "verdict": "verified",
            "why": "all four measured wheel centres share one chassis z plane to the measurement "
                   "resolution; this is what lets a single lift constant be well defined",
            "unit": "m",
            "frame": "chassis body frame",
            "residual": {"spread_m": _r(height, 9)},
        },
        "unloaded_rolling_height_from_geometry": {
            "value_m": _r(-min(zs) + 0.103, 9),
            "verdict": "candidate",
            "why": "chassis z of the wheel centre plus the 103 mm unloaded equatorial radius; "
                   "the loaded rolling radius stays UNKNOWN",
            "unit": "m",
            "frame": "chassis body frame",
            "residual": {
                "proxy_half_height_m": P2_P4_PROXY_HALF_HEIGHT_M,
                "proxy_lift_m": _r(-min(zs) + P2_P4_PROXY_HALF_HEIGHT_M, 9),
                "difference_m": _r(0.103 - P2_P4_PROXY_HALF_HEIGHT_M, 9),
            },
        },
    }


def _part_identity(faces_by_eid: dict) -> dict:
    rows = {}
    for eid in WHEEL_ORDER:
        torus = [f for f in faces_by_eid[eid]["faces"]
                 if f.get("surface", {}).get("type") == "TOROIDAL_SURFACE"
                 and f.get("touches_r103_circle")]
        bands = sorted([tuple(f["axial_band_mm"]) for f in torus])
        radii = sorted([f["surface"]["major_radius"] for f in torus])
        rows[eid] = {
            "tread_torus_face_count": len(torus),
            "tread_axial_bands_mm": [list(b) for b in bands],
            "tread_major_radii_mm": [_r(v, 6) for v in radii],
            "tread_area_mm2": _r(sum(f["area_mm2"] for f in torus), 4),
            "face_count": faces_by_eid[eid]["face_count"],
        }
    right = [rows["435700"], rows["759491"]]
    left = [rows["773650"], rows["871375"]]
    max_band_delta = 0.0
    for a, b in zip(right, left):
        for band_a in a["tread_axial_bands_mm"]:
            mirrored = [-band_a[1], -band_a[0]]
            best = min(
                max(abs(mirrored[0] - band_b[0]), abs(mirrored[1] - band_b[1]))
                for band_b in b["tread_axial_bands_mm"]
            )
            max_band_delta = max(max_band_delta, best)
    return {
        "verdict": "mirrored pairs (left / right), not four identical parts",
        "status": "verified at the BREP face-signature level, candidate for the physical part number",
        "why": "the tread torus is split into two 180 deg faces whose axial bands are "
               "[-20.502, -0.002] and [-0.002, 20.498] mm on the +x wheels and sign-flipped on "
               "the -x wheels, so the solids are mirror placements, not the same orientation",
        "max_mirror_band_residual_mm": _r(max_band_delta, 6),
        "mesh_resolution_limit_mm": 2.9877,
        "why_not_four_identical": "identical parts would show identical axial-band signs in the "
                                  "same CAD frame; they do not",
        "per_wheel": rows,
        "cross_checks": {
            "pr25_hausdorff_mm": 2.0872,
            "pr25_mesh_p95_mm": 2.9877,
            "pr25_verdict": "within mesh resolution: no real asymmetry provable from the cloud",
            "v2_adds": "analytic face-level sign flip, which the cloud could not resolve",
        },
    }


def _p4_contract(wheels: dict) -> dict:
    return {
        "may_use_as_TEST_PROXY_parameters": [
            {"id": "wheel_axle_direction_chassis", "value": "(1,0,0) unit, sign per side from "
             "outward_axle_direction.spin_sign", "verdict": "candidate",
             "constraint": "direction only; use spin_sign, never a hard-coded +/-"},
            {"id": "wheel_centre_chassis", "value": "(+/-0.250002, +/-0.297000, -0.244000) m",
             "verdict": "candidate",
             "constraint": "relative geometry only; do not treat the origin as a chassis datum"},
            {"id": "track_width", "value": "0.500004 m", "verdict": "candidate",
             "constraint": "relative geometry only, in the documented chassis mapping"},
            {"id": "wheelbase", "value": "0.594000 m", "verdict": "candidate",
             "constraint": "relative geometry only; the wheelbase runs along chassis y"},
            {"id": "wheel_unloaded_radius", "value": "0.103 m (torus equator R+r)",
             "verdict": "candidate",
             "constraint": "TEST_PROXY only; a loaded rolling radius is UNKNOWN"},
            {"id": "tread_axial_band", "value": "0.041 m wide, centred on the wheel plane",
             "verdict": "candidate",
             "constraint": "drives contact-patch width proxies; not a tyre model"},
            {"id": "wheel_spin_sign", "value": "spin_sign = +1 on +x wheels, -1 on -x wheels",
             "verdict": "candidate",
             "constraint": "sign convention relative to the outward axle; the forward "
                           "direction remains UNKNOWN"},
        ],
        "forbidden_as_real_values": [
            "wheel_radius as a frozen physical parameter (0.103 m stays a TEST_PROXY)",
            "loaded_rolling_height / any tyre deflection or contact-patch length",
            "lift = 0.334 m (it is 13 mm below the geometric 0.347 m and was never measured)",
            "the 0.090 m proxy collision half-height as a real tread half-width",
            "wheel mass, inertia tensor, friction coefficients, tread compliance",
            "chassis-frame origin as a calibrated datum, or chassis z as an authorised gravity axis",
            "any axle joint axis / motor channel inferred from the axis-cluster recording",
            "steering freedom, steering range or steering-axis position",
            "the toroidal tread as proof of the tyre material or of a real contact model",
        ],
        "gates_for_p4": [
            "read a field's verdict before using it; UNKNOWN must fail closed",
            "every candidate consumed must be an explicit parameter with a sensitivity sweep",
            "the spin sign must come from outward_axle_direction.spin_sign, not from a constant",
            "no number in this file may be cited as a measured physical wheel property",
        ],
    }


def _unknowns() -> list:
    return [
        {"quantity": "loaded_rolling_radius",
         "missing_measurement": "a loaded contact test (or tyre stiffness + vehicle load) to "
                                "convert the 103 mm unloaded equator into a rolling radius"},
        {"quantity": "contact_patch_geometry",
         "missing_measurement": "the torus only gives the unloaded crown; patch size needs a "
                                "contact/deflection measurement"},
        {"quantity": "chassis_frame_datum",
         "missing_measurement": "a vendor chassis frame declaration or a physical datum "
                                "calibration (origin + authorised up axis)"},
        {"quantity": "forward_direction",
         "missing_measurement": "which chassis y sign is the front of the vehicle"},
        {"quantity": "steering_axis_position_and_range",
         "missing_measurement": "a joint definition or a measured steering pose sweep"},
        {"quantity": "wheel_mass_inertia_friction",
         "missing_measurement": "vendor data or a measured part"},
        {"quantity": "tyre_material",
         "missing_measurement": "material specification; the BREP carries no material"},
    ]


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="Build the P1 wheel axle geometry v2 record")
    parser.add_argument("--check", action="store_true")
    parser.add_argument("--root", default=_REPO_ROOT)
    args = parser.parse_args(argv)
    report = build(args.root)
    text = json.dumps(report, ensure_ascii=False, indent=1) + "\n"
    out_path = os.path.join(args.root, OUT)
    if args.check:
        with open(out_path, encoding="utf-8") as handle:
            current = handle.read()
        if current != text:
            print("STALE: %s differs from a fresh build" % OUT)
            return 1
        print("OK: %s is up to date" % OUT)
        return 0
    with open(out_path, "w", encoding="utf-8", newline="\n") as handle:
        handle.write(text)
    print("wrote %s (%d bytes)" % (OUT, len(text.encode("utf-8"))))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
