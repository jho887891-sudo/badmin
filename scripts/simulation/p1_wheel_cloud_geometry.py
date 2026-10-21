"""Measure the rolling surface, spin axis and centre of the Morph One wheel solids
from the tessellated (USD) point cloud, with pxr + numpy and no GPU.

Why a second, independent measurement
-------------------------------------
outputs/simulation/rebot_b601_rs/morph_one/p1_wheel_identity/ established that the
206 mm circle in the STEP BREP is real (analytic radius 103.0 mm, mesh r_max
103.000022 mm) but did NOT establish that the radius-103 feature is the surface
the wheel rolls on.  This tool answers exactly that question from the tessellated
surface, which is the closest thing to "the shape" available without a CAD kernel:

  * the direction of minimum extent (an independent spin-axis estimate that uses
    the point cloud only, never the STEP axis);
  * the radius as a function of azimuth about that axis (r_env(theta)), because a
    wheel that rolls on a cylinder has a constant envelope radius while a roller
    /mecanum /sprocket surface does not;
  * the axial band over which the outer surface exists, i.e. the width of the
    material that can touch the ground;
  * a least-squares circle fit of the rim points, which yields the wheel CENTRE
    independently of any BREP placement entity.

Only pxr (Usd/UsdGeom/Gf) and numpy are used; nothing is written back to the
asset, no Isaac Sim, no rendering, no CUDA.

Usage (on the GPU host, CPU only):

    /home/T7/ojh/robot_sim/env_isaaclab/bin/python p1_wheel_cloud_geometry.py \
        --usda /home/T7/dgut/runs/cad_dev/usd/M1-4WD4S_model2601_dev.usda \
        --mapping /home/dgut/runs/wheels/mapping.json \
        --eids 435700,759491,773650,871375 \
        --out-json cloud_geometry.json --out-csv cloud_geometry.csv
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os

import numpy as np
from pxr import Gf, Usd, UsdGeom

#: Azimuth bin width used for the envelope-radius test, in degrees.
AZIMUTH_BIN_DEG = 5.0
#: Points within this distance of the maximum radius form the "rim" set.
RIM_TOL_MM = 0.05
#: Directions sampled for the minimum-extent axis search (hemisphere).
AXIS_GRID = 4001


def mapping_paths(path):
    with open(path, "r", encoding="utf-8") as handle:
        data = json.load(handle)
    pairs = data["pairs"] if isinstance(data, dict) and "pairs" in data else data
    out = {}
    for entry in pairs:
        eid = entry.get("step_eid", entry.get("eid"))
        if eid is None:
            continue
        out.setdefault(int(eid), entry.get("usd_path"))
    return out


def fibonacci_directions(count):
    """Deterministic near-uniform hemisphere directions (sign is irrelevant)."""
    index = np.arange(count, dtype=float) + 0.5
    z = index / count
    radius = np.sqrt(np.maximum(0.0, 1.0 - z * z))
    theta = math.pi * (1.0 + 5.0 ** 0.5) * index
    dirs = np.stack([radius * np.cos(theta), radius * np.sin(theta), z], axis=1)
    dirs /= np.linalg.norm(dirs, axis=1)[:, None]
    return dirs


def min_extent_axis(points, grid=AXIS_GRID):
    """Direction along which the point slab is thinnest, plus its extent.

    A wheel is thin along its spin axis and long across it, so the direction of
    minimum extent is an independent estimate of the spin axis.  The search is a
    deterministic Fibonacci grid refined by three shrinking local scans, so the
    answer does not depend on a starting guess.
    """
    centred = points - points.mean(axis=0)
    dirs = fibonacci_directions(grid)
    projections = centred @ dirs.T
    extents = projections.max(axis=0) - projections.min(axis=0)
    best = dirs[int(np.argmin(extents))]
    best_extent = float(extents.min())
    step = 0.02  # rad, about 1.15 deg
    for _ in range(3):
        for axis in range(3):
            for sign in (-1.0, 1.0):
                trial = best.copy()
                trial[axis] += sign * step
                trial /= np.linalg.norm(trial)
                projection = centred @ trial
                extent = float(projection.max() - projection.min())
                if extent < best_extent:
                    best_extent = extent
                    best = trial
        step *= 0.25
    return best / np.linalg.norm(best), best_extent, dirs, extents


def plane_basis(axis):
    helper = np.array([0.0, 0.0, 1.0])
    if abs(float(np.dot(helper, axis))) > 0.9:
        helper = np.array([1.0, 0.0, 0.0])
    e1 = np.cross(axis, helper)
    e1 /= np.linalg.norm(e1)
    e2 = np.cross(axis, e1)
    return e1, e2


def circle_fit(uv):
    """Algebraic (Kasa) circle fit: centre and radius in the 2-D plane."""
    x = uv[:, 0] - uv[:, 0].mean()
    y = uv[:, 1] - uv[:, 1].mean()
    matrix = np.stack([x, y, np.ones_like(x)], axis=1)
    rhs = -(x * x + y * y)
    solution, *_ = np.linalg.lstsq(matrix, rhs, rcond=None)
    cx, cy = -solution[0] / 2.0, -solution[1] / 2.0
    radius = math.sqrt(max(0.0, cx * cx + cy * cy - solution[2]))
    residual = np.hypot(x - cx, y - cy) - radius
    return (
        np.array([cx + uv[:, 0].mean(), cy + uv[:, 1].mean()]),
        float(radius),
        float(np.sqrt(np.mean(residual * residual))),
        float(np.abs(residual).max()),
    )


def angle_deg(u, v):
    cosine = float(np.dot(u, v)) / (float(np.linalg.norm(u)) * float(np.linalg.norm(v)))
    return math.degrees(math.acos(max(-1.0, min(1.0, cosine))))


def percentiles(values, qs):
    return {str(q): round(float(np.percentile(values, q)), 6) for q in qs}


def principal_axes(points):
    """Covariance eigen-decomposition: axes ordered by decreasing extent.

    For a solid of revolution the smallest-extent principal axis is the spin
    axis, so this is a second, purely statistical spin-axis estimate that is
    independent of both the STEP axis placement and the min-extent scan.
    """
    centred = points - points.mean(axis=0)
    covariance = np.cov(centred, rowvar=False)
    values, vectors = np.linalg.eigh(covariance)
    order = np.argsort(values)[::-1]
    values = values[order]
    vectors = vectors[:, order]
    axes = []
    for index in range(3):
        vector = vectors[:, index]
        vector = vector / np.linalg.norm(vector)
        if float(vector[0]) < 0:
            vector = -vector
        projection = centred @ vector
        axes.append(
            {
                "axis_index": index,
                "direction": [round(float(v), 6) for v in vector],
                "extent_mm": round(float(projection.max() - projection.min()), 4),
                "variance_mm2": round(float(values[index]), 4),
                "sigma_mm": round(float(math.sqrt(max(0.0, values[index]))), 4),
            }
        )
    return {"centre_mm": [round(float(v), 4) for v in points.mean(axis=0)], "axes": axes}


def nearest_cad_axis_deg(vector):
    """Angle in degrees between a direction and the nearest signed CAD axis."""
    vector = np.asarray(vector, dtype=float)
    best = None
    for name, reference in (
        ("x", np.array([1.0, 0.0, 0.0])),
        ("y", np.array([0.0, 1.0, 0.0])),
        ("z", np.array([0.0, 0.0, 1.0])),
    ):
        cosine = abs(float(np.dot(vector, reference))) / float(np.linalg.norm(vector))
        angle = math.degrees(math.acos(max(-1.0, min(1.0, cosine))))
        if best is None or angle < best[1]:
            best = (name, angle)
    return best[0], best[1]


def hausdorff_mm(source, target, chunk=256):
    """Symmetric nearest-neighbour distance between two point sets (mm).

    Chunked brute force: the clouds here are ~6k points, so this avoids a scipy
    dependency while staying exact.
    """
    def one_way(a, b):
        worst = 0.0
        total = 0.0
        count = 0
        for start in range(0, a.shape[0], chunk):
            block = a[start : start + chunk]
            distances = np.linalg.norm(block[:, None, :] - b[None, :, :], axis=2)
            nearest = distances.min(axis=1)
            worst = max(worst, float(nearest.max()))
            total += float(nearest.sum())
            count += int(nearest.size)
        return worst, total / float(max(1, count))

    forward, forward_mean = one_way(source, target)
    backward, backward_mean = one_way(target, source)
    return {
        "hausdorff_mm": round(max(forward, backward), 4),
        "mean_nearest_mm": round(0.5 * (forward_mean + backward_mean), 4),
        "forward_max_mm": round(forward, 4),
        "backward_max_mm": round(backward, 4),
    }


def analyse(points_mm, axis=None, centre=None):
    """All wheel-shape quantities for one point cloud, in millimetres."""
    points = np.asarray(points_mm, dtype=float)
    result = {"point_count": int(points.shape[0])}
    lo = points.min(axis=0)
    hi = points.max(axis=0)
    result["bbox_mm"] = {
        "min": [round(float(v), 4) for v in lo],
        "max": [round(float(v), 4) for v in hi],
        "size": [round(float(hi[i] - lo[i]), 4) for i in range(3)],
        "centre": [round(float((hi[i] + lo[i]) / 2.0), 4) for i in range(3)],
    }

    pca = principal_axes(points)
    smallest = pca["axes"][2]
    smallest_name, smallest_angle = nearest_cad_axis_deg(smallest["direction"])
    pca["smallest_extent_axis"] = {
        "direction": smallest["direction"],
        "extent_mm": smallest["extent_mm"],
        "nearest_cad_axis": smallest_name,
        "angle_to_nearest_cad_axis_deg": round(smallest_angle, 6),
        "method": "covariance eigenvector with the smallest eigenvalue",
    }
    result["principal_axes"] = pca

    if axis is None:
        axis, extent, _, grid_extents = min_extent_axis(points)
        result["axis_search"] = {
            "method": "minimum projected extent over %d Fibonacci directions + 3 local refinements" % AXIS_GRID,
            "min_extent_mm": round(extent, 4),
            "grid_extent_min_mm": round(float(grid_extents.min()), 4),
            "grid_extent_p01_mm": round(float(np.percentile(grid_extents, 1)), 4),
        }
    axis = np.asarray(axis, dtype=float)
    axis = axis / np.linalg.norm(axis)
    if float(axis[0]) < 0:
        axis = -axis
    if centre is None:
        centre = (lo + hi) / 2.0
    centre = np.asarray(centre, dtype=float)

    min_extent_name, min_extent_angle = nearest_cad_axis_deg(axis)
    result["spin_axis"] = {
        "direction": [round(float(v), 9) for v in axis],
        "nearest_cad_axis": min_extent_name,
        "angle_to_nearest_cad_axis_deg": round(min_extent_angle, 6),
        "angle_to_pca_smallest_axis_deg": round(
            angle_deg(axis, np.asarray(smallest["direction"], dtype=float)), 6
        ),
        "angle_to_x_deg": round(angle_deg(axis, np.array([1.0, 0.0, 0.0])), 6),
        "angle_to_y_deg": round(angle_deg(axis, np.array([0.0, 1.0, 0.0])), 6),
        "angle_to_z_deg": round(angle_deg(axis, np.array([0.0, 0.0, 1.0])), 6),
    }

    delta = points - centre
    axial = delta @ axis
    radial = np.linalg.norm(delta - np.outer(axial, axis), axis=1)

    result["axial"] = {
        "min_mm": round(float(axial.min()), 4),
        "max_mm": round(float(axial.max()), 4),
        "width_mm": round(float(axial.max() - axial.min()), 4),
        "mid_mm": round(float((axial.max() + axial.min()) / 2.0), 4),
    }
    result["radial"] = {
        "min_mm": round(float(radial.min()), 4),
        "max_mm": round(float(radial.max()), 4),
        "extent_mm": round(float(radial.max() - radial.min()), 4),
        "percentiles_mm": percentiles(radial, (1, 10, 50, 90, 99)),
    }

    r_max = float(radial.max())
    rim = radial >= r_max - RIM_TOL_MM
    result["rim"] = {
        "r_max_mm": round(r_max, 6),
        "rim_point_count": int(rim.sum()),
        "rim_axial_min_mm": round(float(axial[rim].min()), 4),
        "rim_axial_max_mm": round(float(axial[rim].max()), 4),
        "rim_band_width_mm": round(float(axial[rim].max() - axial[rim].min()), 4),
    }

    # envelope radius per azimuth bin: the surface that can touch the ground
    e1, e2 = plane_basis(axis)
    uv = np.stack([delta @ e1, delta @ e2], axis=1)
    azimuth = np.degrees(np.arctan2(uv[:, 1], uv[:, 0]))
    bins = int(round(360.0 / AZIMUTH_BIN_DEG))
    index = np.clip((azimuth + 180.0) / AZIMUTH_BIN_DEG, 0, bins - 1e-9).astype(int)
    envelope = np.full(bins, np.nan)
    for bin_index in range(bins):
        mask = index == bin_index
        if mask.any():
            envelope[bin_index] = float(radial[mask].max())
    occupied = int(np.sum(~np.isnan(envelope)))
    finite = envelope[~np.isnan(envelope)]
    result["envelope_per_azimuth"] = {
        "bin_deg": AZIMUTH_BIN_DEG,
        "bin_count": bins,
        "occupied_bins": occupied,
        "coverage": round(occupied / float(bins), 6),
        "min_mm": round(float(finite.min()), 6),
        "median_mm": round(float(np.median(finite)), 6),
        "max_mm": round(float(finite.max()), 6),
        "peak_to_peak_mm": round(float(finite.max() - finite.min()), 6),
        "bins_below_r_max_minus_0p5mm": int(np.sum(finite < r_max - 0.5)),
        "bin_centre_deg_from_180": [round(-180.0 + AZIMUTH_BIN_DEG * (i + 0.5), 3) for i in range(bins)],
        "per_bin_r_max_mm": [None if np.isnan(v) else round(float(v), 4) for v in envelope],
    }

    # radial extent per axial slice: a wheel keeps its radius across the tread, a
    # sphere-like blob does not
    edges = np.linspace(axial.min(), axial.max(), 9)
    slices = []
    for position in range(8):
        mask = (axial >= edges[position]) & (axial <= edges[position + 1])
        if not mask.any():
            slices.append(None)
            continue
        slices.append(
            {
                "axial_mm": [round(float(edges[position] - centre @ axis), 3), round(float(edges[position + 1] - centre @ axis), 3)],
                "r_max_mm": round(float(radial[mask].max()), 3),
                "r_min_mm": round(float(radial[mask].min()), 3),
                "points": int(mask.sum()),
            }
        )
    result["radial_profile_per_axial_slice"] = slices

    rim_uv = uv[rim]
    centre_uv, fit_radius, fit_rms, fit_max = circle_fit(rim_uv)
    fitted_centre = centre + centre_uv[0] * e1 + centre_uv[1] * e2
    result["rim_circle_fit"] = {
        "centre_mm": [round(float(v), 4) for v in fitted_centre],
        "radius_mm": round(fit_radius, 6),
        "residual_rms_mm": round(fit_rms, 6),
        "residual_max_mm": round(fit_max, 6),
        "centre_offset_from_bbox_centre_mm": round(float(np.linalg.norm(fitted_centre - centre)), 4),
    }
    return result


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="Wheel rolling surface from the USD point cloud (pxr, CPU)")
    parser.add_argument("--usda", required=True)
    parser.add_argument("--mapping", required=True)
    parser.add_argument("--eids", default="435700,759491,773650,871375")
    parser.add_argument("--out-json", default="")
    parser.add_argument("--out-csv", default="")
    parser.add_argument("--dump-dir", default="", help="optional directory for full point clouds (resources, keep off-repo)")
    args = parser.parse_args(argv)

    paths = mapping_paths(args.mapping)
    stage = Usd.Stage.Open(args.usda)
    if not stage:
        raise SystemExit("cannot open %s" % args.usda)
    meters_per_unit = UsdGeom.GetStageMetersPerUnit(stage)
    up_axis = UsdGeom.GetStageUpAxis(stage)
    xform = UsdGeom.XformCache()

    if args.dump_dir:
        os.makedirs(args.dump_dir, exist_ok=True)

    wheels = {}
    clouds = {}
    rows = []
    for eid in [int(t) for t in args.eids.split(",") if t.strip()]:
        usd_path = paths.get(eid)
        if not usd_path:
            wheels[str(eid)] = {"error": "eid not in mapping"}
            continue
        prim = stage.GetPrimAtPath(usd_path)
        if not prim or prim.GetTypeName() != "Mesh":
            wheels[str(eid)] = {"error": "prim missing or not a Mesh", "usd_path": usd_path}
            continue
        mesh = UsdGeom.Mesh(prim)
        points = mesh.GetPointsAttr().Get()
        matrix = xform.GetLocalToWorldTransform(prim)
        world = np.array(
            [[float(v) for v in matrix.Transform(Gf.Vec3d(float(p[0]), float(p[1]), float(p[2])))] for p in points],
            dtype=float,
        )
        world_mm = world * meters_per_unit * 1000.0
        if args.dump_dir:
            np.savetxt(
                os.path.join(args.dump_dir, "wheel_%d_points_mm.csv" % eid),
                world_mm,
                delimiter=",",
                fmt="%.6f",
                header="x_mm,y_mm,z_mm",
                comments="",
            )
        entry = analyse(world_mm)
        entry.update(
            {
                "step_eid": eid,
                "usd_path": usd_path,
                "face_count": int(len(mesh.GetFaceVertexCountsAttr().Get() or [])),
                "triangle_count": int(len(mesh.GetFaceVertexIndicesAttr().Get() or []) // 3),
                "local_to_world_scale": [
                    round(float(matrix.ExtractRotationMatrix().GetRow(i).GetLength()), 9) for i in range(3)
                ],
            }
        )
        clouds[eid] = world_mm
        wheels[str(eid)] = entry
        rows.append(
            {
                "step_eid": eid,
                "points": entry["point_count"],
                "bbox_mm": "x".join("%.3f" % v for v in entry["bbox_mm"]["size"]),
                "axis": entry["spin_axis"]["direction"],
                "angle_to_x_deg": entry["spin_axis"]["angle_to_x_deg"],
                "r_max_mm": entry["rim"]["r_max_mm"],
                "r_min_mm": entry["radial"]["min_mm"],
                "envelope_coverage": entry["envelope_per_azimuth"]["coverage"],
                "envelope_peak_to_peak_mm": entry["envelope_per_azimuth"]["peak_to_peak_mm"],
                "rim_band_mm": entry["rim"]["rim_band_width_mm"],
                "fit_radius_mm": entry["rim_circle_fit"]["radius_mm"],
                "fit_rms_mm": entry["rim_circle_fit"]["residual_rms_mm"],
            }
        )

    # mirror-symmetry residuals across the four wheels (chassis mirror planes)
    symmetry = {"pairs": [], "note": "source cloud is mirrored across the named plane, then compared with the counterpart cloud; residual is an exact nearest-neighbour Hausdorff distance in mm"}
    corner_of = {}
    for eid, cloud in clouds.items():
        centre = (cloud.min(axis=0) + cloud.max(axis=0)) / 2.0
        corner_of[eid] = {
            "x": 1 if centre[0] >= 0 else -1,
            "z": 1 if centre[2] >= 0 else -1,
        }
    for eid_a in sorted(clouds):
        for eid_b in sorted(clouds):
            if eid_a >= eid_b:
                continue
            sign_a, sign_b = corner_of[eid_a], corner_of[eid_b]
            for plane, index in (("x", 0), ("z", 2)):
                other = "z" if plane == "x" else "x"
                if sign_a[plane] == sign_b[plane]:
                    continue
                if sign_a[other] != sign_b[other]:
                    continue  # diagonal neighbours are not mirrored images
                mirrored = np.array(clouds[eid_a], copy=True)
                mirrored[:, index] = -mirrored[:, index]
                residual = hausdorff_mm(mirrored, clouds[eid_b])
                residual.update(
                    {
                        "plane": plane + "=0",
                        "source_step_eid": eid_a,
                        "target_step_eid": eid_b,
                        "source_corner": "%+d%+d" % (sign_a["x"], sign_a["z"]),
                        "target_corner": "%+d%+d" % (sign_b["x"], sign_b["z"]),
                    }
                )
                symmetry["pairs"].append(residual)
    # baseline: same corner signs on both planes, no mirroring -> pure tessellation spacing
    baseline = []
    for eid_a in sorted(clouds):
        for eid_b in sorted(clouds):
            if eid_a >= eid_b or corner_of[eid_a] != corner_of[eid_b]:
                continue
            residual = hausdorff_mm(clouds[eid_a], clouds[eid_b])
            residual.update({"kind": "unmirrored_baseline", "source_step_eid": eid_a, "target_step_eid": eid_b})
            baseline.append(residual)
    symmetry["unmirrored_baseline"] = baseline
    symmetry["baseline_min_hausdorff_mm"] = (
        round(min(item["hausdorff_mm"] for item in baseline), 4) if baseline else None
    )
    if symmetry["pairs"]:
        symmetry["max_residual_mm"] = round(max(item["hausdorff_mm"] for item in symmetry["pairs"]), 4)
        symmetry["max_mean_nearest_mm"] = round(
            max(item["mean_nearest_mm"] for item in symmetry["pairs"]), 4
        )
        symmetry["pair_count"] = len(symmetry["pairs"])

    report = {
        "inputs": {
            "usda": os.path.abspath(args.usda),
            "mapping": os.path.abspath(args.mapping),
            "eids": [int(t) for t in args.eids.split(",") if t.strip()],
        },
        "stage": {
            "meters_per_unit": float(meters_per_unit),
            "up_axis": str(up_axis),
            "note": "world coordinates are stage units * metersPerUnit * 1000 = millimetres",
        },
        "parameters": {"rim_tol_mm": RIM_TOL_MM, "azimuth_bin_deg": AZIMUTH_BIN_DEG, "axis_grid": AXIS_GRID},
        "mirror_symmetry": symmetry,
        "wheels": wheels,
        "notes": [
            "the minimum-extent direction is an independent spin-axis estimate from the cloud alone",
            "a constant envelope radius over a full 360 deg is what makes a surface a rolling surface",
            "the rim circle fit gives the wheel centre without using any BREP placement entity",
            "no GPU, no Isaac Sim, no rendering: pxr + numpy only",
        ],
    }

    if args.out_json:
        with open(args.out_json, "w", encoding="utf-8") as handle:
            json.dump(report, handle, ensure_ascii=False, indent=1)
    if args.out_csv and rows:
        with open(args.out_csv, "w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            for row in rows:
                writer.writerow(
                    {
                        **row,
                        "axis": ";".join("%.6f" % v for v in row["axis"]),
                    }
                )
    summary = {
        "stage": report["stage"],
        "wheels": {
            eid: {
                "points": w.get("point_count"),
                "bbox_mm": (w.get("bbox_mm") or {}).get("size"),
                "axis": (w.get("spin_axis") or {}).get("direction"),
                "angle_to_x_deg": (w.get("spin_axis") or {}).get("angle_to_x_deg"),
                "r_max_mm": (w.get("rim") or {}).get("r_max_mm"),
                "r_min_mm": (w.get("radial") or {}).get("min_mm"),
                "envelope": (w.get("envelope_per_azimuth") or {}).get("coverage"),
                "envelope_ptp_mm": (w.get("envelope_per_azimuth") or {}).get("peak_to_peak_mm"),
                "rim_band_mm": (w.get("rim") or {}).get("rim_band_width_mm"),
                "fit_radius_mm": (w.get("rim_circle_fit") or {}).get("radius_mm"),
                "fit_rms_mm": (w.get("rim_circle_fit") or {}).get("residual_rms_mm"),
                "centre_mm": (w.get("rim_circle_fit") or {}).get("centre_mm"),
                "pca_smallest_axis": (w.get("principal_axes") or {}).get("smallest_extent_axis"),
                "axial_width_mm": (w.get("axial") or {}).get("width_mm"),
                "envelope_min_mm": (w.get("envelope_per_azimuth") or {}).get("min_mm"),
            }
            for eid, w in wheels.items()
        },
    }
    print(json.dumps(summary, ensure_ascii=False, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
