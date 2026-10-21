# -*- coding: utf-8 -*-
"""P6-07/P6-02: collision-proxy + merged-mount-chain regression tests.

Two failure modes, both of which produce plausible-looking output instead of an exception:

  1. a collision proxy that silently LETS THE VISUAL RACKET POKE OUT of it (the mesh is
     visual-only, so nothing else will complain) - the containment tests here re-measure it
     against the real USD mesh whenever pxr is available;
  2. a "UNKNOWN" boundary that has quietly acquired numbers (a guessed racket mount offset
     composes, runs and flies the racket a few centimetres off the wrist).

Pure python + numpy.  pxr is used READ-ONLY when importable and the numeric re-measurement is
skipped (pytest.skip, never a silent pass) when it is not.
"""
from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[2]
P6 = ROOT / "outputs/simulation/rebot_b601_rs/morph_one/p6_frames"
FRAME_TREE = P6 / "frame_tree.json"
PROXY = P6 / "racket_collision_proxy.json"
CHAIN = P6 / "mounted_chain.json"
P3_TREE = P6 / "evidence/p3_mount_frame_tree_at_0bbed19.json"
RACKET_USD = ROOT / "assets/third_party/racket_visual.usd"

RACKET_SHA256 = "643fb67003c6429f98ec132d2db056c83ed02811b594ad27904b618bf7c1ec29"
UNKNOWN_EDGE_IDS = [1, 2, 13, 14, 15]


def load(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def sha256_norm(path: Path) -> str:
    """sha256 over LF-normalised bytes: the committed JSON is CRLF in this checkout and LF in
    the object store, and a hash that only matches in one of them is not evidence."""
    return hashlib.sha256(path.read_bytes().replace(b"\r\n", b"\n")).hexdigest()


def resolve(doc, pointer: str):
    cur = doc
    for tok in pointer.strip("/").split("/"):
        if not tok:
            continue
        cur = cur[int(tok)] if isinstance(cur, list) else cur[tok]
    return cur


def quat_wxyz_to_mat(q):
    w, x, y, z = [float(v) for v in q]
    n = math.sqrt(w * w + x * x + y * y + z * z)
    assert n > 0.0
    w, x, y, z = w / n, x / n, y / n, z / n
    return np.array([[1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
                     [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
                     [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)]])


def frame_of(prim):
    g = prim["geometry"]
    if prim["type"] == "capsule":
        return np.array(g["axis_start_m"], float), np.array(g["axis_end_m"], float), float(g["radius_m"])
    if prim["type"] == "box":
        return np.array(g["center_m"], float), np.array(g["half_extents_m"], float), None
    raise AssertionError(f"unknown primitive type {prim['type']}")


def dist_out(P, prim):
    if prim["type"] == "capsule":
        a, b, r = frame_of(prim)
        ab = b - a
        L2 = float(ab @ ab)
        if L2 <= 0.0:
            return np.linalg.norm(P - a, axis=1) - r
        t = np.clip(((P - a) @ ab) / L2, 0.0, 1.0)
        return np.linalg.norm(P - (a + t[:, None] * ab), axis=1) - r
    c, h, _ = frame_of(prim)
    q = np.abs(P - c) - h
    return np.linalg.norm(np.maximum(q, 0.0), axis=1)


def union_dist_out(P, prims):
    return np.min(np.stack([dist_out(P, p) for p in prims]), axis=0)


# --------------------------------------------------------------------------- #
# 1. P6-07 collision proxy: primitives are real, nonzero and evidence-backed
# --------------------------------------------------------------------------- #
class TestProxyPrimitivesAreNonDegenerate:
    def test_schema_and_kind_are_explicit(self):
        doc = load(PROXY)
        assert doc["schema"] == "p6_racket_collision_proxy/1"
        assert doc["task"] == "P6-07"
        # it must never be mistaken for a manufacturer collision model
        assert "NOT a manufacturer collision model" in doc["kind"]

    def test_source_is_pinned_to_the_real_asset_bytes(self):
        doc = load(PROXY)
        assert doc["provenance"]["source_sha256_actual"] == RACKET_SHA256
        assert doc["provenance"]["source_sha256_actual"] == sha256_file(RACKET_USD)
        for m in doc["source_meshes"]:
            assert m["sha256"] == RACKET_SHA256
            assert m["collision_api"] is False
            assert m["points"] > 0 and m["triangles"] > 0

    def test_every_primitive_is_non_degenerate(self):
        doc = load(PROXY)
        assert doc["primitive_count"] == len(doc["primitives"])
        assert len(doc["primitives"]) >= 5
        for p in doc["primitives"]:
            assert p["type"] in ("capsule", "box"), p["id"]
            assert p["volume_m3"] > 1e-12, p["id"]
            if p["type"] == "capsule":
                a, b, r = frame_of(p)
                assert r > 1e-5, p["id"]
                # a capsule must have a real axis: a zero-length one is a sphere wearing the
                # wrong name and was already caught once during the fitting of this proxy
                assert float(np.linalg.norm(b - a)) > 1e-4, p["id"]
            else:
                c, h, _ = frame_of(p)
                assert (h > 1e-5).all(), p["id"]
                assert 4.0 / 3.0 * math.pi * float(np.prod(h)) > 1e-9, p["id"]

    def test_primitive_order_covers_handle_then_frame_then_string_bed(self):
        doc = load(PROXY)
        roles = [p["role"] for p in doc["primitives"]]
        order = [p["order"] for p in doc["primitives"]]
        assert order == sorted(order), "primitives must be an ORDERED set"
        assert roles.count("handle") >= 1 and roles.count("frame") >= 2
        assert roles.count("string_bed") >= 2
        assert roles.index("handle") < roles.index("frame") < roles.index("string_bed")

    def test_every_primitive_cites_its_evidence(self):
        doc = load(PROXY)
        for p in doc["primitives"]:
            assert p["dimension_derivation"].strip(), p["id"]
            assert p["measured_extent_m"], p["id"]
            assert p["geometry"], p["id"]
        # the segmentation that produced the numbers is recorded, not implied
        seg = doc["segmentation_measured"]
        for key in ("grip_radius_m", "shaft_radius_m", "handle_top_z_m", "throat_start_z_m",
                    "hoop_bottom_z_m", "rackettip_z_m", "hoop_tube_radius_m"):
            assert seg[key] > 0.0, key


# --------------------------------------------------------------------------- #
# 2. P6-07 accuracy: containment, coverage and leak are numbers, not adjectives
# --------------------------------------------------------------------------- #
class TestProxyAccuracyIsMeasured:
    def test_containment_verdict_passes_and_is_stated_with_a_tolerance(self):
        doc = load(PROXY)
        acc = doc["accuracy"]["containment"]
        assert acc["verdict"] == "PASS"
        assert acc["tolerance_m"] == pytest.approx(0.001)
        assert acc["within_tolerance_fraction"] == pytest.approx(1.0)
        assert acc["max_uncovered_distance_m"] <= acc["tolerance_m"] + 1e-9
        assert acc["surface_samples"] >= 10000

    def test_coverage_numbers_are_self_consistent(self):
        doc = load(PROXY)
        bb = doc["visual_bbox_m"]
        vol = float(np.prod(np.array(bb["size"], float)))
        assert bb["volume_m3"] == pytest.approx(vol, rel=1e-3)  # stored to 6 decimals
        cov = doc["accuracy"]["bbox_volume_coverage"]["fraction_of_visual_bbox_inside_proxy"]
        # a racket is mostly air: a proxy claiming most of the bbox would be a solid block
        assert 0.0 < cov < 0.25
        solid = doc["accuracy"]["solid_volume"]
        assert solid["proxy_volume_m3"] > 0.0
        assert solid["proxy_over_shell"] == pytest.approx(
            solid["proxy_volume_m3"] / solid["visual_shell_volume_m3"], rel=1e-3)
        leak = doc["accuracy"]["leak"]
        assert 0.0 < leak["max_distance_to_visual_surface_m"] < 0.05
        assert leak["mean_distance_to_visual_surface_m"] <= leak["max_distance_to_visual_surface_m"]
        assert leak["p95_distance_to_visual_surface_m"] <= leak["max_distance_to_visual_surface_m"]

    def test_dimensions_are_metres_of_a_real_racket(self):
        doc = load(PROXY)
        bb = doc["visual_bbox_m"]
        length = bb["max"][2] - bb["min"][2]
        # 0.5 m .. 1 m: neither millimetres-as-metres (682.7) nor metres-as-millimetres (0.00068)
        assert 0.5 < length < 1.0
        assert bb["min"][2] == pytest.approx(0.0, abs=1e-9)  # origin = handle butt
        for p in doc["primitives"]:
            for key, vals in p["geometry"].items():
                if key == "radius_m":
                    assert 1e-5 < float(vals) < 0.05
                else:
                    assert all(abs(float(v)) < 1.0 for v in vals), (p["id"], key)

    def test_containment_recomputed_against_the_real_mesh(self):
        """Re-measure containment from the committed primitives against the real USD triangles."""
        pxr = pytest.importorskip("pxr", reason="pxr (usd-core) not installed: cannot re-measure")
        from pxr import Usd, UsdGeom

        doc = load(PROXY)
        stage = Usd.Stage.Open(str(RACKET_USD))
        tris = []
        for prim in stage.Traverse():
            if not prim.IsA(UsdGeom.Mesh):
                continue
            mesh = UsdGeom.Mesh(prim)
            pts = np.asarray(mesh.GetPointsAttr().Get(), dtype=float)
            counts = np.asarray(mesh.GetFaceVertexCountsAttr().Get(), dtype=int)
            idx = np.asarray(mesh.GetFaceVertexIndicesAttr().Get(), dtype=int)
            off = 0
            for c in counts:
                assert c == 3
                tris.append(pts[idx[off:off + c]])
                off += c
        T = np.asarray(tris)  # (n_tri, 3, 3)
        area = 0.5 * np.linalg.norm(np.cross(T[:, 1] - T[:, 0], T[:, 2] - T[:, 0]), axis=1)
        rng = np.random.default_rng(20261021)
        pick = rng.choice(len(area), size=12000, p=area / area.sum())
        u, v = rng.random(12000), rng.random(12000)
        flip = u + v > 1.0
        u[flip], v[flip] = 1.0 - u[flip], 1.0 - v[flip]
        P = T[pick, 0] + u[:, None] * (T[pick, 1] - T[pick, 0]) + v[:, None] * (T[pick, 2] - T[pick, 0])
        d = union_dist_out(P, doc["primitives"])
        tol = doc["accuracy"]["containment"]["tolerance_m"]
        assert float(d.max()) <= tol, (
            f"re-measured containment FAILS: {int((d > tol).sum())} of {len(d)} surface samples "
            f"are outside the proxy by up to {float(d.max()):.6f} m (tolerance {tol} m)")
        # the recorded metric has to agree with the re-measurement, not merely be a nice number
        assert float((d <= tol).mean()) >= doc["accuracy"]["containment"]["within_tolerance_fraction"] - 1e-6


# --------------------------------------------------------------------------- #
# 3. P6-02 merged chain: one file, pointers instead of duplicated numbers
# --------------------------------------------------------------------------- #
class TestMergedChainHasNoDuplicatedNumbers:
    def test_chain_covers_the_whole_path_in_order(self):
        doc = load(CHAIN)
        order = doc["chain_order"]
        assert order[0] == "world"
        assert order[-1] == "racket_contact_frame"
        for n in ("chassis", "arm_base", "base_link", "gripper_end", "racket_mount", "racket_root"):
            assert n in order
        assert doc["summary"]["edges_total"] == 15
        assert doc["summary"]["edges_total"] == len(doc["edges"])

    def test_numeric_edges_are_pointers_and_the_pointers_verify(self):
        doc = load(CHAIN)
        sources = {s["ref"]["file"]: s for s in doc["sources"]}
        for e in doc["edges"]:
            if e["status"] == "UNKNOWN":
                assert e["value_ref"] is None
                continue
            r = e["value_ref"]
            assert r["sha256"] == sources[r["file"]]["sha256_file"]
            path = ROOT / r["file"]
            assert sha256_norm(path) == r["sha256"], f"pointer to {r['file']} is stale"
            node = resolve(load(path), r["json_pointer"])
            # no number is restated next to the pointer - that is what forbids two truths
            assert "translation_m" not in e and "rotation_quat" not in e
            assert "values" not in e and "rotation_matrix" not in e
            assert node["translation_m"] is not None

    def test_p3_anchor_is_pinned_and_is_identity(self):
        doc = load(CHAIN)
        e2 = next(e for e in doc["edges"] if e["edge"] == 2)
        anchor = e2["p3_01_anchor"]
        assert anchor["p3_commit"] == "0bbed19"
        p3 = load(ROOT / anchor["value_ref"]["file"])
        assert sha256_norm(ROOT / anchor["value_ref"]["file"]) == anchor["value_ref"]["sha256"]
        tf = resolve(p3, anchor["value_ref"]["json_pointer"])
        assert tf["position_m"]["value"] == [0.0, 0.0, 0.0]
        assert tf["rotation_quat_wxyz"]["value"] == [1.0, 0.0, 0.0, 0.0]
        node = resolve(p3, anchor["anchor_ref"]["json_pointer"])
        for side in ("local_pos0", "local_pos1"):
            assert node[side]["value"] == [0.0, 0.0, 0.0]
        for side in ("local_rot0", "local_rot1"):
            assert node[side]["value"] == [1.0, 0.0, 0.0, 0.0]
        assert anchor["verified_in_this_merge"]["local_rot0_equals_local_rot1"] is True
        assert anchor["verified_in_this_merge"]["matches_p3_derived_transform"] is True
        # ...and the PHYSICAL mount is still unknown, despite the identity above
        assert e2["status"] == "UNKNOWN"

    def test_p3_copy_is_byte_identical_to_the_delivered_branch(self):
        doc = load(CHAIN)
        p3 = next(s for s in doc["sources"] if s.get("p3_commit") == "0bbed19")
        assert sha256_norm(ROOT / p3["ref"]["file"]) == p3["sha256_file"]
        assert p3["ref"]["sha256_basis"] == "LF-normalised bytes"
        assert doc["naming_reconciliation"]["p6_arm_root_prim == p3_arm_base"]


# --------------------------------------------------------------------------- #
# 4. the UNKNOWN guard: this is the test that fails if someone guesses
# --------------------------------------------------------------------------- #
class TestUnknownEdgesStayUnknown:
    def _numlike(self, value, path="$"):
        """Yield every path at which a numeric scalar/matrix hides inside a node."""
        if isinstance(value, bool):
            return
        if isinstance(value, (int, float)):
            yield path, value
        elif isinstance(value, dict):
            for k, v in value.items():
                yield from self._numlike(v, f"{path}.{k}")
        elif isinstance(value, (list, tuple)):
            for i, v in enumerate(value):
                yield from self._numlike(v, f"{path}[{i}]")

    def test_the_same_five_edges_are_still_unknown(self):
        doc = load(CHAIN)
        unknown = [e["edge"] for e in doc["edges"] if e["status"] == "UNKNOWN"]
        assert unknown == UNKNOWN_EDGE_IDS
        assert doc["summary"]["unknown_edge_ids"] == UNKNOWN_EDGE_IDS
        assert doc["summary"]["unknown"] == 5
        assert doc["summary"]["asserted"] == 10
        for e in doc["edges"]:
            if e["status"] != "UNKNOWN":
                continue
            assert e["missing_datum"] and e["would_come_from"]
            assert e["value_ref"] is None

    def test_an_unknown_edge_may_not_carry_any_number(self):
        """Fill edge 13/14/15 with a plausible offset and this test fails."""
        doc = load(CHAIN)
        forbidden = {"translation_m", "rotation_quat", "rotation_matrix", "values",
                     "translation", "quaternion", "rotation", "offset", "position"}
        for e in doc["edges"]:
            if e["status"] != "UNKNOWN":
                continue
            payload = {k: v for k, v in e.items()
                       if k not in ("edge", "from", "to", "edge_kind", "status", "value_ref",
                                    "missing_datum", "would_come_from", "unknown_guard",
                                    "p3_01_anchor")}
            assert payload == {}, f"edge {e['edge']} gained a payload: {sorted(payload)}"
            for key in forbidden:
                assert key not in e, f"edge {e['edge']} carries a forbidden numeric key {key}"
            # and nothing numeric may be smuggled in through the free-text fields
            for key in ("missing_datum", "would_come_from", "unknown_guard"):
                nums = list(self._numlike(e[key]))
                assert nums == [], f"edge {e['edge']}.{key} smuggles numbers: {nums}"

    def test_the_arm_edges_are_not_reinterpreted_as_poses(self):
        doc = load(CHAIN)
        for e in doc["edges"]:
            if e["edge"] in (4, 5, 6, 7, 8, 9):
                assert e["status"] == "ASSERTED_BY_ASSET"
                assert e["value_ref"]["json_pointer"] == f"/edges/{e['edge'] - 1}"
        issue = doc["composition"]["open_issue"]
        assert issue["status"] == "OPEN"
        assert issue["max_abs_delta_m"] > 0.001  # recorded, never quietly rounded to zero


# --------------------------------------------------------------------------- #
# 5. composition: the mounted chain is a proper rigid transform
# --------------------------------------------------------------------------- #
class TestMountedChainComposes:
    def test_composition_a_is_a_proper_rotation(self):
        doc = load(CHAIN)
        A = doc["composition"]["A_usdgeom_xformops_authoritative"]
        T = np.array(A["T"], float)
        R = T[:3, :3]
        assert R.shape == (3, 3)
        assert float(np.linalg.det(R)) == pytest.approx(1.0, abs=1e-9)
        assert R.T @ R == pytest.approx(np.eye(3), abs=1e-9)
        assert np.array_equal(T[3], np.array([0.0, 0.0, 0.0, 1.0]))
        assert A["det_R"] == pytest.approx(1.0)

    def test_composition_a_reproduces_the_pxr_end_pose(self):
        doc = load(CHAIN)
        A = doc["composition"]["A_usdgeom_xformops_authoritative"]
        p3 = load(ROOT / A["vs_pxr_end_pose"]["p3_gripper_end_world_rest_pose_ref"]["file"])
        end = resolve(p3, A["vs_pxr_end_pose"]["p3_gripper_end_world_rest_pose_ref"]["json_pointer"])
        T = np.array(A["T"], float)
        assert T[:3, 3] == pytest.approx(np.array(end["world_position"], float), abs=1e-9)
        assert A["vs_pxr_end_pose"]["max_abs_delta_m"] <= 1e-6

    def test_composition_a_recomputed_from_the_p3_hops(self):
        """Independently compose the hops the file points at, in the file's own convention."""
        doc = load(CHAIN)
        A = doc["composition"]["A_usdgeom_xformops_authoritative"]
        p3 = load(ROOT / A["hops_ref"]["file"])
        hops = resolve(p3, A["hops_ref"]["json_pointer"])
        T = np.eye(4)
        for h in hops:
            lt = h["local_transform"]
            M = np.eye(4)
            M[:3, :3] = quat_wxyz_to_mat(lt["quat_wxyz"])
            M[:3, 3] = np.array(lt["position"], float)
            T = T @ M
        assert len(hops) == A["hops_used"]
        assert T == pytest.approx(np.array(A["T"], float), abs=1e-9)
        assert float(np.linalg.det(T[:3, :3])) == pytest.approx(1.0, abs=1e-9)
        assert T[:3, 3] == pytest.approx(
            np.array(load(ROOT / A["vs_pxr_end_pose"]["p3_gripper_end_world_rest_pose_ref"]["file"])
                     ["arm_chain_rest_pose"]["gripper_end_world_rest_pose"]["world_position"], float),
            abs=1e-6)

    def test_composition_b_is_a_proper_rotation_but_is_only_diagnostic(self):
        doc = load(CHAIN)
        B = doc["composition"]["B_p6_physics_joint_anchors"]
        T = np.array(B["T"], float)
        assert float(np.linalg.det(T[:3, :3])) == pytest.approx(1.0, abs=1e-9)
        assert B["vs_A"]["max_abs_delta_m"] == pytest.approx(
            doc["composition"]["open_issue"]["max_abs_delta_m"], abs=1e-12)
        # it must not be presented as the chain of record
        assert "authoritative" not in "B_p6_physics_joint_anchors"
        assert doc["composition"]["open_issue"]["status"] == "OPEN"
