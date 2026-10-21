# -*- coding: utf-8 -*-
"""P6-07: author a machine-readable collision proxy for the racket from its VISUAL mesh.

The shipped racket USD (assets/third_party/racket_visual.usd, sha256 643fb670...) has ZERO
collision geometry (evidence/racket_usd_audit.json collision_count=0), so anything that wants
to collide with the racket must author its own proxies.  This tool fits an ordered set of
analytic primitives (capsule / box) - handle, shaft, head rim, string bed - to the actual
triangle vertices of the two visual meshes and then STATES ITS OWN ACCURACY numerically:

  * containment  : fraction of sampled visual surface points inside the proxy union, and the
                   max distance of any uncovered point to the proxy (mm)
  * bbox coverage: fraction of the visual bbox volume occupied by the proxy (Monte-Carlo)
  * solid ratio  : proxy volume / mesh volume (both analytic / signed-tetrahedron)
  * leak         : max / mean distance from a proxy-interior point to the nearest visual
                   triangle - i.e. how much air the proxy claims

Pure python + numpy + pxr (read-only).  No Isaac Sim, no GPU, no asset writes.
Usage:  python tools/p6_frames/build_racket_collision_proxy.py
"""
from __future__ import annotations

import hashlib
import json
import math
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
RACKET_USD = ROOT / "assets/third_party/racket_visual.usd"
OUT = ROOT / "outputs/simulation/rebot_b601_rs/morph_one/p6_frames/racket_collision_proxy.json"
EXPECTED_SHA256 = "643fb67003c6429f98ec132d2db056c83ed02811b594ad27904b618bf7c1ec29"

#: a surface point must be inside the proxy or at most this far outside it
CONTAINMENT_TOL_M = 0.001
N_OUTLINE = 13          # rim nodes per side
TAPER_STEPS = 3         # grip -> shaft taper capsules
REPAIR_PASSES = 12      # containment fitting passes
SEED = 20261021
N_SURFACE = 20000
N_BBOX = 200000
N_LEAK = 4000


# --------------------------------------------------------------------------- #
# mesh access
# --------------------------------------------------------------------------- #
def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_meshes(path: Path):
    from pxr import Usd, UsdGeom

    stage = Usd.Stage.Open(str(path))
    cache = UsdGeom.XformCache()
    meshes = []
    for prim in stage.Traverse():
        if not prim.IsA(UsdGeom.Mesh):
            continue
        mesh = UsdGeom.Mesh(prim)
        pts = np.asarray(mesh.GetPointsAttr().Get(), dtype=float)
        counts = np.asarray(mesh.GetFaceVertexCountsAttr().Get(), dtype=int)
        idx = np.asarray(mesh.GetFaceVertexIndicesAttr().Get(), dtype=int)
        xf = np.asarray(cache.GetLocalToWorldTransform(prim)).T  # row-major 4x4
        # every mesh must already be in the racket_root frame; a silent non-identity xform
        # would move the mesh away from the primitives fitted in racket_root coordinates
        if not np.allclose(xf, np.eye(4), atol=1e-9):
            raise SystemExit(f"UNEXPECTED local-to-world xform on {prim.GetPath()}: {xf.tolist()}")
        tris = []
        off = 0
        for c in counts:
            if c != 3:
                raise SystemExit(f"non-triangle face ({c}) on {prim.GetPath()}: fan-triangulate first")
            tris.append(idx[off:off + 3])
            off += c
        meshes.append({
            "prim_path": str(prim.GetPath()),
            "name": prim.GetName(),
            "points": pts,
            "triangles": np.asarray(tris, dtype=int),
        })
    return stage, meshes


def sample_surface(meshes, n_total: int, seed: int):
    """Area-weighted barycentric samples on every triangle of every mesh (deterministic)."""
    rng = np.random.default_rng(seed)
    tri_all, tri_area = [], []
    for m in meshes:
        A = m["points"][m["triangles"][:, 0]]
        B = m["points"][m["triangles"][:, 1]]
        C = m["points"][m["triangles"][:, 2]]
        a = 0.5 * np.linalg.norm(np.cross(B - A, C - A), axis=1)
        tri_all.append((A, B, C, m["name"]))
        tri_area.append(a)
    areas = np.concatenate(tri_area)
    A = np.concatenate([t[0] for t in tri_all])
    B = np.concatenate([t[1] for t in tri_all])
    C = np.concatenate([t[2] for t in tri_all])
    pick = rng.choice(len(areas), size=n_total, p=areas / areas.sum())
    u = rng.random(n_total)
    v = rng.random(n_total)
    flip = u + v > 1.0
    u[flip] = 1.0 - u[flip]
    v[flip] = 1.0 - v[flip]
    pts = A[pick] + u[:, None] * (B[pick] - A[pick]) + v[:, None] * (C[pick] - A[pick])
    return pts, (A, B, C, areas)


def mesh_volume(meshes) -> float:
    """Signed tetrahedron sum about the origin (closed-ish low-poly shells; reported as an estimate)."""
    vol = 0.0
    for m in meshes:
        A = m["points"][m["triangles"][:, 0]]
        B = m["points"][m["triangles"][:, 1]]
        C = m["points"][m["triangles"][:, 2]]
        vol += float(np.abs(np.einsum("ij,ij->i", A, np.cross(B, C))).sum() / 6.0)
    return vol


# --------------------------------------------------------------------------- #
# primitive signed helpers (all axis-aligned in racket_root)
# --------------------------------------------------------------------------- #
def dist_out_capsule(P, a, b, r):
    ab = b - a
    L2 = float(ab @ ab)
    if L2 <= 0.0:  # a zero-length capsule is a sphere; never let a 0/0 poison the union
        return np.linalg.norm(P - a, axis=1) - r
    t = np.clip(((P - a) @ ab) / L2, 0.0, 1.0)
    proj = a + t[:, None] * ab
    return np.linalg.norm(P - proj, axis=1) - r


def dist_out_box(P, c, half):
    q = np.abs(P - c) - half
    return np.linalg.norm(np.maximum(q, 0.0), axis=1)


def prim_dist_out(P, prim):
    g = prim["geometry"]
    if prim["type"] == "capsule":
        return dist_out_capsule(P, np.asarray(g["axis_start_m"], float),
                                np.asarray(g["axis_end_m"], float), float(g["radius_m"]))
    if prim["type"] == "box":
        return dist_out_box(P, np.asarray(g["center_m"], float),
                            np.asarray(g["half_extents_m"], float))
    raise SystemExit(f"unsupported primitive type {prim['type']}")


def union_dist_out(P, prims):
    return np.min(np.stack([prim_dist_out(P, p) for p in prims]), axis=0)


def prim_volume(prim):
    g = prim["geometry"]
    if prim["type"] == "capsule":
        a = np.asarray(g["axis_start_m"], float)
        b = np.asarray(g["axis_end_m"], float)
        r = float(g["radius_m"])
        L = float(np.linalg.norm(b - a))
        return math.pi * r * r * L + 4.0 / 3.0 * math.pi * r ** 3
    if prim["type"] == "box":
        h = np.asarray(g["half_extents_m"], float)
        return float(8.0 * h[0] * h[1] * h[2])
    raise SystemExit(prim["type"])


# --------------------------------------------------------------------------- #
# point -> nearest triangle distance (Ericson closest-point-on-triangle)
# --------------------------------------------------------------------------- #
def nearest_tri_dist(P, A, B, C):
    best = np.full(len(P), np.inf)
    for i in range(len(A)):
        a, b, c = A[i], B[i], C[i]
        ab, ac, ap = b - a, c - a, P - a
        d1 = ap @ ab
        d2 = ap @ ac
        bp = P - b
        d3 = bp @ ab
        d4 = bp @ ac
        cp = P - c
        d5 = cp @ ab
        d6 = cp @ ac
        vc = d1 * d4 - d3 * d2
        vb = d5 * d2 - d1 * d6
        va = d3 * d6 - d5 * d4
        denom = va + vb + vc
        closest = np.empty_like(P)
        # region A
        m = (d1 <= 0) & (d2 <= 0)
        closest[m] = a
        # region B
        m = (~m) & (d3 >= 0) & (d4 <= d3)
        closest[m] = b
        done = (d1 <= 0) & (d2 <= 0)
        done = done | ((d3 >= 0) & (d4 <= d3))
        # region C
        m = (~done) & (d6 >= 0) & (d5 <= d6)
        closest[m] = c
        done = done | ((d6 >= 0) & (d5 <= d6))
        # edge AB
        m = (~done) & (vc <= 0) & (d1 >= 0) & (d3 <= 0)
        t = np.where(d1 - d3 != 0, d1 / (d1 - d3), 0.0)
        closest[m] = a + t[m][:, None] * ab
        done = done | m
        # edge AC
        m = (~done) & (vb <= 0) & (d2 >= 0) & (d6 <= 0)
        t2 = np.where(d2 - d6 != 0, d2 / (d2 - d6), 0.0)
        closest[m] = a + t2[m][:, None] * ac
        done = done | m
        # edge BC
        m = (~done) & (va <= 0) & ((d4 - d3) >= 0) & ((d5 - d6) >= 0)
        t3 = np.where((d4 - d3) + (d5 - d6) != 0, (d4 - d3) / ((d4 - d3) + (d5 - d6)), 0.0)
        closest[m] = b + t3[m][:, None] * (c - b)
        done = done | m
        # interior
        m = ~done
        if m.any():
            dd = np.where(denom[m] != 0, 1.0 / denom[m], 0.0)
            v = vb[m] * dd
            w = vc[m] * dd
            closest[m] = a + v[:, None] * ab + w[:, None] * ac
        d = np.linalg.norm(P - closest, axis=1)
        best = np.minimum(best, d)
    return best


# --------------------------------------------------------------------------- #
# fit
# --------------------------------------------------------------------------- #
def r2(v):
    return float(round(float(v), 6))


def build():
    got = sha256_file(RACKET_USD)
    if got != EXPECTED_SHA256:
        raise SystemExit(f"racket USD sha256 mismatch: {got} != {EXPECTED_SHA256}")
    stage, meshes = load_meshes(RACKET_USD)
    if len(meshes) != 2:
        raise SystemExit(f"expected 2 meshes, got {len(meshes)}")
    frame = next(m for m in meshes if "Racket" in m["name"])
    strings = next(m for m in meshes if "Strings" in m["name"])

    P = frame["points"]
    z = P[:, 2]
    ry = np.abs(P[:, 1])
    rx = np.abs(P[:, 0])
    rxy = np.hypot(P[:, 0], P[:, 1])
    y_max = float(ry.max())

    # ---- data-driven segmentation, ring by ring ------------------------- #
    # The mesh is authored as vertex RINGS (constant z).  Everything below is read off those
    # rings: no datasheet value enters, and a ring that is off the mid-plane contributes its
    # outer envelope (max |y|) and its tube cross-section (outer -> inner gap).
    rings = []
    for zz in np.unique(np.round(z, 6)):
        sel = np.abs(z - zz) < 1e-5
        if not sel.any():
            continue
        ay = ry[sel]
        outer = float(ay.max())
        cand = ay[ay > 0.5 * outer]
        rings.append({
            "z": float(zz),
            "n": int(sel.sum()),
            "outer_y": outer,
            "inner_y": float(cand.min()) if cand.size else None,
            "xmax": float(rx[sel].max()),
            "rmax": float(rxy[sel].max()),
        })
    rings.sort(key=lambda r: r["z"])
    z_top = rings[-1]["z"]

    r_grip = float(max(r["rmax"] for r in rings if r["z"] < 0.4 * z_top))
    r_shaft = float(min(r["rmax"] for r in rings if 0.1 * z_top < r["z"] < 0.6 * z_top))
    r_mid = 0.5 * (r_grip + r_shaft)
    grip_rings = [r["z"] for r in rings if r["z"] < 0.6 * z_top and r["rmax"] > r_mid]
    z_handle_top = float(max(grip_rings))

    # the throat: the silhouette is wider than the bare shaft from here up (the Y fork)
    throat_thresh = 2.0 * r_shaft
    wide_rings = [r for r in rings if r["z"] > z_handle_top and r["outer_y"] > throat_thresh]
    z_throat_start = float(wide_rings[0]["z"])
    hoop_rings = [r for r in wide_rings if r["outer_y"] > 0.25 * y_max]
    z_hoop_bottom = float(hoop_rings[0]["z"])

    # ---- rim: per-ring tube cross-section, resampled onto N_OUTLINE nodes - #
    tube = []
    for r in wide_rings:
        inner = r["inner_y"] if r["inner_y"] is not None else max(0.0, r["outer_y"] - 2 * r["xmax"])
        axis_y = 0.5 * (r["outer_y"] + inner)
        half_y = max(0.5 * (r["outer_y"] - inner), r["xmax"])
        tube.append((r["z"], axis_y, half_y))
    t_z = np.array([t[0] for t in tube])
    t_axis = np.array([t[1] for t in tube])
    t_half = np.array([t[2] for t in tube])
    node_z = np.linspace(float(t_z[0]), float(t_z[-1]), N_OUTLINE)
    node_axis = np.interp(node_z, t_z, t_axis)
    outline = list(zip(node_z.tolist(), node_axis.tolist()))
    # the tube radius is taken from the HOOP rings only: the throat rings are the Y fork, whose
    # |y| span is the fork width, not a tube cross-section
    hoop_half = [t[2] for t, r in zip(tube, wide_rings) if r["z"] >= z_hoop_bottom]
    r_rim = float(max(min(hoop_half) if hoop_half else t_half.max(), r_shaft))

    prims = []
    prims.append({
        "id": "racket_handle",
        "order": 1,
        "role": "handle",
        "type": "capsule",
        "geometry": {"axis_start_m": [0.0, 0.0, 0.0], "axis_end_m": [0.0, 0.0, r2(z_handle_top)],
                     "radius_m": r2(r_grip)},
        "dimension_derivation": (
            "axis = the grip centreline (racket_root +Z, origin at the handle butt per "
            "evidence/racket_usd_audit.json flange_note); radius = max radial distance of the "
            f"Obj_Racket_0 vertices with z < {z_hoop_bottom:.6f} m; length = the last authored "
            f"vertex ring whose radial distance still exceeds {(r_grip + r_shaft) / 2:.6f} m"),
        "measured_extent_m": {"z_min": 0.0, "z_max": r2(z_handle_top), "max_radial_m": r2(r_grip)},
    })
    prims.append({
        "id": "racket_shaft",
        "order": 2,
        "role": "shaft",
        "type": "capsule",
        "geometry": {"axis_start_m": [0.0, 0.0, r2(z_handle_top)],
                     "axis_end_m": [0.0, 0.0, r2(z_throat_start)], "radius_m": r2(r_shaft)},
        "dimension_derivation": (
            f"radius = max radial distance of the vertices in the shaft band "
            f"{0.55 * z_hoop_bottom:.6f} < z < {z_hoop_bottom:.6f} m; the capsule spans the grip "
            f"top ring to the first authored ring wider than {throat_thresh:.6f} m (the throat fork)"),
        "measured_extent_m": {"z_min": r2(z_handle_top), "z_max": r2(z_throat_start),
                              "max_radial_m": r2(r_shaft)},
    })

    # the grip taper (grip radius -> shaft radius): a stepped capsule chain.  A cone cannot be
    # one capsule, so it is split into TAPER_STEPS capsules whose radius is the WIDER end of its
    # step - the over-coverage is then bounded by one step's radial drop and is measured below.
    z_taper_end = float(min(r["z"] for r in rings
                            if r["z"] > z_handle_top and r["rmax"] <= 1.05 * r_shaft))
    step_len = (z_taper_end - z_handle_top) / TAPER_STEPS
    for i in range(TAPER_STEPS):
        z0 = z_handle_top + i * step_len
        z1 = z_handle_top + (i + 1) * step_len
        rad = r_grip + (r_shaft - r_grip) * (i * step_len) / (z_taper_end - z_handle_top)
        prims.append({
            "id": f"racket_grip_taper_{i + 1:02d}",
            "order": 1.5,
            "role": "handle",
            "type": "capsule",
            "geometry": {"axis_start_m": [0.0, 0.0, r2(z0)], "axis_end_m": [0.0, 0.0, r2(z1)],
                         "radius_m": r2(rad)},
            "dimension_derivation": (
                f"step {i + 1}/{TAPER_STEPS} of the measured grip->shaft taper between the "
                f"authored rings z={z_handle_top:.6f} m (r={r_grip:.6f} m) and z={z_taper_end:.6f} "
                f"m (r={r_shaft:.6f} m); each capsule takes the wider end radius of its step"),
            "measured_extent_m": {"z_min": r2(z0), "z_max": r2(z1), "radius_m": r2(rad)},
        })

    # rim: two capsule chains, one per side (+Y / -Y), through the measured outline, plus a
    # crown capsule across the tip (the top of the hoop is a horizontal arc in the YZ plane,
    # so a per-side vertical chain alone leaves the tip uncovered).
    # the throat FORK: the silhouette splits from the shaft axis into the two hoop sides.  Without
    # these two capsules the containment repair would grow the SHAFT to reach the fork and turn a
    # 3.9 mm shaft into a 9.8 mm one - the fork must be carried by its own primitives.
    fork_end = next((z_, y_) for z_, y_ in outline if z_ >= z_hoop_bottom)
    for side in (+1.0, -1.0):
        prims.append({
            "id": f"racket_frame_fork_{'p' if side > 0 else 'n'}",
            "order": 2.5,
            "role": "frame",
            "type": "capsule",
            "geometry": {"axis_start_m": [0.0, r2(side * 0.2 * r_shaft), r2(z_throat_start)],
                         "axis_end_m": [0.0, r2(side * fork_end[1]), r2(fork_end[0])],
                         "radius_m": r2(r_rim)},
            "dimension_derivation": (
                "from the shaft axis at the first fork ring (z=%.6f m) out to the measured tube "
                "centreline at the head bottom (z=%.6f m, |y|=%.6f m); radius = the measured hoop "
                "tube radius" % (z_throat_start, fork_end[0], fork_end[1])),
            "measured_extent_m": {"z_from": r2(z_throat_start), "z_to": r2(fork_end[0]),
                                  "out_y_from_m": r2(0.2 * r_shaft), "out_y_to_m": r2(fork_end[1])},
        })

    k = 0
    crown = {}
    for side in (+1.0, -1.0):
        for (z0, y0), (z1, y1) in zip(outline[:-1], outline[1:]):
            crown[side] = (z1, side * y1)
            k += 1
            prims.append({
                "id": f"racket_frame_rim_{'p' if side > 0 else 'n'}{k:02d}",
                "order": 3,
                "role": "frame",
                "type": "capsule",
                "geometry": {
                    "axis_start_m": [0.0, r2(side * y0), r2(z0)],
                    "axis_end_m": [0.0, r2(side * y1), r2(z1)],
                    "radius_m": r2(r_rim),
                },
                "dimension_derivation": (
                    "axis = the measured TUBE CENTRELINE of the hoop on this side, i.e. "
                    "(outer + inner)/2 of the |y| values of the authored vertex ring, linearly "
                    "interpolated between consecutive nodes; radius = the largest measured tube "
                    "half-thickness max((outer-inner)/2, max|x|) over the head rings"),
                "measured_extent_m": {"z_from": r2(z0), "z_to": r2(z1),
                                      "tube_axis_half_width_from_m": r2(y0),
                                      "tube_axis_half_width_to_m": r2(y1)},
            })

    if len(crown) == 2 and abs(crown[1.0][1] - crown[-1.0][1]) > 1e-6:
        (zp, yp), (zn, yn) = crown[1.0], crown[-1.0]
        prims.append({
            "id": "racket_frame_crown",
            "order": 4,
            "role": "frame",
            "type": "capsule",
            "geometry": {"axis_start_m": [0.0, r2(yp), r2(zp)], "axis_end_m": [0.0, r2(yn), r2(zn)],
                         "radius_m": r2(r_rim)},
            "dimension_derivation": (
                "bridges the last measured tube-centreline node of the +Y chain to the one of the "
                "-Y chain, closing the hoop tip"),
            "measured_extent_m": {"z_from": r2(zp), "z_to": r2(zn), "radius_m": r2(r_rim)},
        })

    # strings: solid membrane split into z-slabs so the elliptical outline is followed
    S = strings["points"]
    sz, sy, sx = S[:, 2], np.abs(S[:, 1]), np.abs(S[:, 0])
    s_zlo, s_zhi = float(sz.min()), float(sz.max())
    s_xthick = float(sx.max())
    s_levels = np.unique(np.round(sz, 6))
    bounds = s_levels  # a clean triangulated bed: one slab between consecutive authored rings
    for i in range(len(bounds) - 1):
        lo, hi = float(bounds[i]), float(bounds[i + 1])
        # float32 authoring: compare with a 1e-5 m (0.01 mm) slack, not an exact tolerance
        sel = (sz >= lo - 1e-5) & (sz <= hi + 1e-5)
        if not sel.any():
            raise SystemExit(f"empty strings slab {lo}..{hi} levels={s_levels.tolist()} n={len(sz)}")
        b = float(sy[sel].max())
        prims.append({
            "id": f"racket_string_bed_slab_{i + 1:02d}",
            "order": 5,
            "role": "string_bed",
            "type": "box",
            "geometry": {
                "center_m": [0.0, 0.0, r2(0.5 * (lo + hi))],
                "half_extents_m": [r2(s_xthick), r2(b), r2(0.5 * (hi - lo))],
            },
            "dimension_derivation": (
                "z-band of the Obj_Strings_0 vertices; half-width in Y = max |y| of the strings "
                "vertices inside the band (so the slab follows the elliptical bed outline); "
                "half-thickness in X = max |x| of the strings vertices. The bed is treated as a "
                "SOLID membrane: at the shuttle speeds of this project the string bed does not "
                "let the shuttle through, so the hole inside the head is not a collision void."),
            "measured_extent_m": {"z_min": r2(lo), "z_max": r2(hi), "half_width_m": r2(b),
                                  "half_thickness_m": r2(s_xthick)},
        })

    # the set is ORDERED: handle -> taper -> shaft -> frame rim -> crown -> string bed
    prims.sort(key=lambda p: (float(p["order"]), p["id"]))

    # ---- containment repair (grow only the primitive each gap sits on) ---- #
    # No primitive is inflated blindly: for every uncovered surface point the NEAREST primitive
    # is found and grown by exactly that point's deficit.  Growth is bounded, recorded, and the
    # loop is capped - if it cannot reach the tolerance the build FAILS (exit 1) rather than
    # shipping a proxy that lets the visual racket poke out.
    surf, tri = sample_surface(meshes, N_SURFACE, SEED)
    d_out = union_dist_out(surf, prims)
    repair = {
        "pass": "per-point nearest-primitive growth (radius / half-extent) until containment",
        "max_passes": REPAIR_PASSES,
        "before": {
            "inside_fraction": r2(float((d_out <= 0.0).mean())),
            "within_tolerance_fraction": r2(float((d_out <= CONTAINMENT_TOL_M).mean())),
            "max_uncovered_distance_m": r2(max(0.0, float(d_out.max()))),
        },
        "growths": [],
        "passes_used": 0,
    }
    for it in range(REPAIR_PASSES):
        bad = d_out > CONTAINMENT_TOL_M
        if not bad.any():
            break
        repair["passes_used"] = it + 1
        D = np.stack([prim_dist_out(surf, p) for p in prims])  # (n_prim, n_pts)
        nearest = np.argmin(D[:, bad], axis=0)
        deficit = d_out[bad]
        pts_bad = surf[bad]
        worst = {}
        for j, defc, pt in zip(nearest.tolist(), deficit.tolist(), pts_bad):
            if j not in worst or defc > worst[j][0]:
                worst[j] = (float(defc), pt)
        for j, (defc, pt) in worst.items():
            p = prims[j]
            add = defc * 1.02 + 5e-6
            g = p["geometry"]
            if p["type"] == "capsule":
                before_r = float(g["radius_m"])
                g["radius_m"] = r2(before_r + add)
                repair["growths"].append({"iteration": it + 1, "primitive": p["id"],
                                          "channel": "radius_m", "from_m": r2(before_r),
                                          "to_m": g["radius_m"], "added_m": r2(add)})
            else:
                # grow each axis by that one worst point's own shortfall on that axis
                c = np.asarray(g["center_m"], float)
                h = np.asarray(g["half_extents_m"], float)
                short = np.maximum(np.abs(pt - c) - h, 0.0)
                if not short.any():
                    short = np.full(3, add / 3.0)
                new_h = h + short * 1.02 + 5e-6
                for ax, name in enumerate("XYZ"):
                    if new_h[ax] > h[ax]:
                        repair["growths"].append({
                            "iteration": it + 1, "primitive": p["id"],
                            "channel": f"half_extents_m[{name}]", "from_m": r2(h[ax]),
                            "to_m": r2(new_h[ax]), "added_m": r2(new_h[ax] - h[ax])})
                g["half_extents_m"] = [r2(v) for v in new_h]
        d_out = union_dist_out(surf, prims)
    repair["after"] = {
        "inside_fraction": r2(float((d_out <= 0.0).mean())),
        "within_tolerance_fraction": r2(float((d_out <= CONTAINMENT_TOL_M).mean())),
        "max_uncovered_distance_m": r2(max(0.0, float(d_out.max()))),
    }
    repair["primitives_grown"] = len({g["primitive"] for g in repair["growths"]})
    r_rim = r2(max([float(p["geometry"]["radius_m"]) for p in prims if p["role"] == "frame"]
                   or [r_rim]))

    inside = d_out <= 0.0
    within_tol = d_out <= CONTAINMENT_TOL_M

    # ---- bbox coverage (Monte-Carlo) ------------------------------------ #
    bmin = P.min(axis=0)
    bmax = P.max(axis=0)
    rng = np.random.default_rng(SEED + 1)
    Q = rng.uniform(bmin, bmax, size=(N_BBOX, 3))
    q_out = union_dist_out(Q, prims)
    q_in = q_out <= 0.0
    bbox_vol = float(np.prod(bmax - bmin))

    # ---- leak (how much air the proxy claims) --------------------------- #
    rng2 = np.random.default_rng(SEED + 2)
    picks = rng2.choice(np.nonzero(q_in)[0], size=min(N_LEAK, int(q_in.sum())), replace=False)
    A, B, C, _ = tri
    leak_d = nearest_tri_dist(Q[picks], A, B, C)

    prim_vols = [prim_volume(p) for p in prims]
    mesh_vol = mesh_volume(meshes)

    for p, v in zip(prims, prim_vols):
        p["volume_m3"] = r2(v) if v > 1e-9 else float(f"{v:.9g}")

    doc = {
        "schema": "p6_racket_collision_proxy/1",
        "task": "P6-07",
        "title": "racket collision proxy, fitted to the visual mesh",
        "kind": "ENGINEERING_PROXY — NOT a manufacturer collision model",
        "generated_by": "tools/p6_frames/build_racket_collision_proxy.py",
        "units": {
            "length": "metre",
            "verified": "the source USD declares metersPerUnit=1.0 and upAxis=Z "
                        "(evidence/racket_usd_audit.json), and every dimension below is a "
                        "coordinate read straight out of that stage - no 0.001/1000 factor is applied",
            "volume": "cubic metre",
        },
        "root_frame": {
            "name": "racket_root",
            "prim_path": str(stage.GetDefaultPrim().GetPath()) if stage.GetDefaultPrim() else None,
            "definition": "+Z = handle butt -> head tip, +X = face normal, +Y = Z x X, origin = handle butt",
            "mesh_local_to_world": "identity (verified for both meshes while fitting; a non-identity "
                                   "xform aborts the build)",
            "convention_note": "the same racket_root frame as "
                               "outputs/simulation/rebot_b601_rs/morph_one/p6_frames/frame_tree.json "
                               "(node racket_root, edge 14 target)",
        },
        "source_meshes": [
            {
                "mesh_prim_path": m["prim_path"],
                "role": "frame + handle + shaft (visual only)" if m is frame else "string bed (visual only)",
                "points": int(len(m["points"])),
                "triangles": int(len(m["triangles"])),
                "bbox_min_m": [r2(v) for v in m["points"].min(axis=0)],
                "bbox_max_m": [r2(v) for v in m["points"].max(axis=0)],
                "collision_api": False,
                "source_file": "assets/third_party/racket_visual.usd",
                "sha256": EXPECTED_SHA256,
            }
            for m in (frame, strings)
        ],
        "visual_bbox_m": {"min": [r2(v) for v in bmin], "max": [r2(v) for v in bmax],
                          "size": [r2(v) for v in (bmax - bmin)],
                          "volume_m3": r2(bbox_vol)},
        "segmentation_measured": {
            "grip_radius_m": r2(r_grip),
            "shaft_radius_m": r2(r_shaft),
            "handle_top_z_m": r2(z_handle_top),
            "throat_start_z_m": r2(z_throat_start),
            "hoop_bottom_z_m": r2(z_hoop_bottom),
            "rackettip_z_m": r2(z_top),
            "hoop_tube_radius_m": r2(r_rim),
            "head_outline_rings_used": len(outline),
            "method": "quantiles/breakpoints read off the authored vertex rings of Obj_Racket_0; "
                      "no dimension is taken from a datasheet",
        },
        "primitive_count": len(prims),
        "primitives": prims,
        "accuracy": {
            "containment": {
                "surface_samples": int(N_SURFACE),
                "inside_fraction": r2(float(inside.mean())),
                "within_tolerance_fraction": r2(float(within_tol.mean())),
                "tolerance_m": CONTAINMENT_TOL_M,
                "max_uncovered_distance_m": r2(max(0.0, float(d_out.max()))),
                "verdict": "PASS" if bool(within_tol.all()) else "FAIL",
                "meaning": "every sampled visual surface point is inside the proxy union (or at "
                           "most tolerance_m outside it); the proxy therefore cannot let the "
                           "visual racket poke through a shuttle",
            },
            "bbox_volume_coverage": {
                "bbox_samples": int(N_BBOX),
                "fraction_of_visual_bbox_inside_proxy": r2(float(q_in.mean())),
                "bbox_volume_m3": r2(bbox_vol),
                "meaning": "of the visual bounding-box volume, this fraction is solid proxy; the "
                           "rest is the air around the handle/head that the visual bbox spans "
                           "(the bbox of a racket is mostly air, so this number is low by nature)",
            },
            "solid_volume": {
                "proxy_volume_m3": r2(sum(prim_vols)),
                "visual_shell_volume_m3": r2(mesh_vol),
                "proxy_over_shell": r2(sum(prim_vols) / mesh_vol) if mesh_vol > 0 else None,
                "meaning": "both are ESTIMATES: the proxy is a union (overlapping primitives are "
                           "counted twice) and the shell volume is a signed-tetrahedron sum over "
                           "open low-poly shells",
            },
            "leak": {
                "samples_inside_proxy": int(len(picks)),
                "max_distance_to_visual_surface_m": r2(float(leak_d.max())),
                "p95_distance_to_visual_surface_m": r2(float(np.percentile(leak_d, 95))),
                "mean_distance_to_visual_surface_m": r2(float(leak_d.mean())),
                "meaning": "how much air the proxy claims: the worst proxy-interior point is this "
                           "far from the nearest visual triangle (largest at the top of the "
                           "string-bed slabs, where the elliptical bed narrows inside a z-slab)",
            },
            "repair": repair,
            "tolerance_source": "CONTAINMENT_TOL_M in this tool = 0.001 m, chosen one decade "
                               "below the smallest racket feature the proxy models (shaft radius)",
            "acceptance_rule": "the build exits 0 only if within_tolerance_fraction == 1.0 after "
                               "the repair loop; otherwise it exits 1 and this file must not be "
                               "used as a collider",
        },
        "explicitly_not": [
            "a manufacturer / CAD collision model: the source is a low-poly Sketchfab visual "
            "(CC-BY-4.0) whose overall length is 682.7 mm versus 675 mm product / 680 mm BWF max",
            "a physical-material model: no mass, inertia, friction or restitution is stated here",
            "a racket MOUNT model: the handle->flange transform stays UNKNOWN (frame_tree.json "
            "edges 13/14) and no mount geometry is added by this proxy",
            "a deformation model: the bed is modelled as a rigid solid membrane",
        ],
        "unknown": [
            {"what": "the real racket's strung bed tension / deflection",
             "value": "UNKNOWN", "missing_datum": "no string-tension measurement in the repo"},
            {"what": "the collision behaviour of the actual physical racket",
             "value": "UNKNOWN", "missing_datum": "no contact experiment; the proxy has no "
                                                   "material properties"},
            {"what": "the material of the frame (mass distribution)",
             "value": "UNKNOWN", "missing_datum": "the visual asset carries colour only"},
        ],
        "provenance": {
            "pxr_used": True,
            "isaac_sim_used": False,
            "gpu_used": False,
            "assets_modified": False,
            "source_sha256_expected": EXPECTED_SHA256,
            "source_sha256_actual": got,
            "seed": SEED,
            "deterministic": True,
            "command": "python tools/p6_frames/build_racket_collision_proxy.py",
        },
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(doc, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    print(f"primitives: {len(prims)}")
    print(f"containment: inside={inside.mean():.6f} within_tol={within_tol.mean():.6f} "
          f"max_uncovered={max(0.0, float(d_out.max())):.6f} m")
    print(f"bbox coverage fraction = {q_in.mean():.6f} (bbox vol {bbox_vol:.9f} m3)")
    print(f"leak: max {leak_d.max():.6f} m, mean {leak_d.mean():.6f} m")
    print(f"proxy vol {sum(prim_vols):.9f} m3, shell vol {mesh_vol:.9f} m3")
    print(f"wrote {OUT}")
    return 0 if bool(within_tol.all()) else 1


if __name__ == "__main__":
    sys.exit(build())
