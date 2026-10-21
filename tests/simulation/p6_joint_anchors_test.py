# -*- coding: utf-8 -*-
"""P6-02-OI-1: the joint body1 anchors (physics:localPos1/localRot1) and the composition gate.

THE BUG THESE TESTS CLOSE
-------------------------
mounted_chain.json composed the arm chain from the P6 joint edges knowing only the body0 side of
each USD joint.  A USD PhysicsJoint pins its joint frame TWICE:

    X0 = (physics:localPos0, physics:localRot0)   in body0
    X1 = (physics:localPos1, physics:localRot1)   in body1
    T_body0_body1 = X0 * inv(X1)

Dropping X1 is harmless only when X1 is the identity.  Here X1 is the identity for joint2 and
root_joint, and a 180 deg flip about +X -- (0, -1, 0, 0) wxyz -- for joint1/joint3/joint4/
joint5/joint6.  Composition B v1 therefore landed 0.33242 m (and exactly pi rad) away from the
pxr end-effector pose, while still passing every det(R) = +1 check.  That is the shape of
failure this file exists to catch: not an exception, three plausible numbers and a symmetric
matrix.

FOUR GATES
----------
  1. DRIFT: the joint-edge composition must reproduce the pxr reference within 1e-6 m and
     1e-6 rad, recomputed here from the committed JSON (never read back from the file).
  2. PROVENANCE: every anchor must carry both body sides, a source line number, and a sha256
     that still matches the file it points at.
  3. HISTORY: mounted_chain.json is immutable; its 0.33242 stays recorded and is not rounded.
  4. UNKNOWN: the five UNKNOWN edges must still carry no number.  Filling in a guessed mount
     offset is the failure mode, and it fails here loudly.

Pure python + numpy.  No pxr, no Isaac, no GPU, no network.
"""
from __future__ import annotations

import hashlib
import json
import math
import os
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[2]
P6 = ROOT / "outputs/simulation/rebot_b601_rs/morph_one/p6_frames"
FRAME_TREE = P6 / "frame_tree.json"
BASE_CHAIN = P6 / "mounted_chain.json"
ANCHORS = P6 / "joint_anchors.json"
CHAIN_V2 = P6 / "mounted_chain_v2.json"

UNKNOWN_EDGE_IDS = [1, 2, 13, 14, 15]
TOL_M = 1e-6
TOL_RAD = 1e-6

# The body1 side as authored by the USD layer. Restated here as an EXPECTATION, so that a
# silent edit of joint_anchors.json cannot make the composition pass by changing both sides.
EXPECTED_LOCAL_ROT1_WXYZ = {
    "root_joint": [1.0, 0.0, 0.0, 0.0],
    "joint1": [0.0, -1.0, 0.0, 0.0],
    "joint2": [1.0, 0.0, 0.0, 0.0],
    "joint3": [0.0, -1.0, 0.0, 0.0],
    "joint4": [0.0, -1.0, 0.0, 0.0],
    "joint5": [0.0, -1.0, 0.0, 0.0],
    "joint6": [0.0, -1.0, 0.0, 0.0],
}
HISTORICAL_B_V1_DELTA_M = 0.33242


def load(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def sha256_norm(path):
    return hashlib.sha256(Path(path).read_bytes().replace(b"\r\n", b"\n")).hexdigest()


def resolve(doc, pointer):
    cur = doc
    for tok in str(pointer).strip("/").split("/"):
        if not tok:
            continue
        cur = cur[int(tok)] if isinstance(cur, list) else cur[tok]
    return cur


# --------------------------------------------------------------------------- #
# Independent linear algebra.  Deliberately NOT imported from
# tools/p6_frames/build_joint_anchors.py: a test that shares the generator's code
# cannot catch the generator being wrong.
# --------------------------------------------------------------------------- #
def quat_to_mat(q):
    w, x, y, z = [float(v) for v in q]
    n = math.sqrt(w * w + x * x + y * y + z * z)
    assert n > 0.0, "degenerate quaternion"
    w, x, y, z = w / n, x / n, y / n, z / n
    return np.array([[1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
                     [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
                     [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)]])


def homog(pos, quat):
    T = np.eye(4)
    T[:3, :3] = quat_to_mat(quat)
    T[:3, 3] = np.asarray(pos, float)
    return T


def inv_h(T):
    out = np.eye(4)
    out[:3, :3] = T[:3, :3].T
    out[:3, 3] = -T[:3, :3].T @ T[:3, 3]
    return out


def angle_rad(Ra, Rb):
    """Geodesic angle via a relative quaternion: well conditioned near zero."""
    D = Rb.T @ Ra
    tr = float(np.trace(D))
    if tr > 0.0:
        s = math.sqrt(tr + 1.0) * 2.0
        w = 0.25 * s
        v = np.array([(D[2, 1] - D[1, 2]) / s, (D[0, 2] - D[2, 0]) / s, (D[1, 0] - D[0, 1]) / s])
    else:
        i = int(np.argmax(np.diag(D)))
        j, k = (i + 1) % 3, (i + 2) % 3
        s = math.sqrt(1.0 + D[i, i] - D[j, j] - D[k, k]) * 2.0
        w = (D[k, j] - D[j, k]) / s
        v = np.zeros(3)
        v[i] = 0.25 * s
        v[j] = (D[j, i] + D[i, j]) / s
        v[k] = (D[k, i] + D[i, k]) / s
    return 2.0 * math.atan2(float(np.linalg.norm(v)), abs(float(w)))


# --------------------------------------------------------------------------- #
# Composition helpers, driven ONLY by committed JSON
# --------------------------------------------------------------------------- #
def compose_chain(anchors, tree, base_edges, drop_body1):
    """Compose chassis -> ... -> gripper_end from the committed joint anchors.

    drop_body1=True reproduces composition B v1 (body0 only, i.e. the historical bug).
    drop_body1=False is composition B v2 (T_body0_body1 = X0 * inv(X1)).
    """
    by_name = {a["prim"]: a for a in anchors["joints"]}
    order = [("root_joint", 3), ("joint1", 4), ("joint2", 5), ("joint3", 6),
             ("joint4", 7), ("joint5", 8), ("joint6", 9), ("gripper_end", 12)]
    T = np.eye(4)
    for name, merged_edge in order:
        if name == "gripper_end":
            ft = tree["edges"][merged_edge - 1]
            assert ft["to"] == "gripper_end"
            T = T @ homog(ft["translation_m"], ft["rotation_quat"]["values"])
            continue
        a = by_name[name]
        X0 = homog(a["local_pos0_m"], a["local_rot0_wxyz"])
        if drop_body1:
            local = X0
        else:
            local = X0 @ inv_h(homog(a["local_pos1_m"], a["local_rot1_wxyz"]))
        T = T @ local
    return T


# --------------------------------------------------------------------------- #
# 1. provenance: the anchors are real, complete and still pointed at
# --------------------------------------------------------------------------- #
class TestAnchorProvenance:
    def test_chain_v2_pins_the_anchor_file_by_sha256(self):
        v2 = load(CHAIN_V2)
        ref = v2["joint_anchors"]
        assert ref["sha256"] == sha256_norm(ANCHORS), "mounted_chain_v2 points at stale anchor bytes"
        assert ref["sha256_basis"] == "LF-normalised bytes"
        assert ref["file"].endswith("joint_anchors.json")
        assert len(ref["sha256"]) == 64

    def test_the_source_layer_hash_is_recorded_and_consistent(self):
        anchors = load(ANCHORS)
        v2 = load(CHAIN_V2)
        src = anchors["source"]
        assert src["joint_layer_sha256"] == v2["joint_anchors"]["source_layer_sha256"]
        assert len(src["joint_layer_sha256"]) == 64
        assert src["joint_layer_sha256_matches_remote"] is True
        assert src["joint_layer_relative_path"] == "payloads/Physics/physics.usda"
        # the same layer whose body0 side the P6 frame tree already used
        assert src["arm_asset_sha256"] == (
            "96d4f2ec1126dec6eaf9e70ab664ba6cc30fa81bdec86178a3e50b7a23bdd510")

    def test_every_joint_records_both_body_sides_and_a_source_line(self):
        anchors = load(ANCHORS)
        names = [a["prim"] for a in anchors["joints"]]
        assert names == ["root_joint", "joint1", "joint2", "joint3", "joint4",
                         "joint5", "joint6"]
        for a in anchors["joints"]:
            assert a["usd_type"] in ("PhysicsRevoluteJoint", "PhysicsFixedJoint"), a["prim"]
            assert a["body0"].startswith("/ReBot_Arm_RS"), a
            assert a["body1"].startswith("/ReBot_Arm_RS"), a
            assert a["body1"] != a["body0"], a["prim"]
            for key, n in (("local_pos0_m", 3), ("local_pos1_m", 3),
                           ("local_rot0_wxyz", 4), ("local_rot1_wxyz", 4)):
                assert len(a[key]) == n, (a["prim"], key)
                assert all(isinstance(float(v), float) for v in a[key])
            lines = a["source_lines"]
            for key in ("prim_def", "body0", "body1", "localPos0", "localPos1",
                        "localRot0", "localRot1"):
                assert isinstance(lines[key], int) and lines[key] > 0, (a["prim"], key)
            # the four anchor lines must be real, ordered, distinct lines of one block
            assert lines["localPos0"] < lines["localPos1"] < lines["localRot0"] < lines["localRot1"]
            assert lines["prim_def"] < lines["localPos0"] < lines["prim_end"]

    def test_revolute_joints_declare_their_axis(self):
        anchors = load(ANCHORS)
        for a in anchors["joints"]:
            if a["usd_type"] == "PhysicsRevoluteJoint":
                assert a["physics_axis"] == "Z", a["prim"]

    def test_the_body1_side_is_the_identity_only_where_it_really_is(self):
        """Only joint2 and root_joint have an identity body1; the other five are 180 deg flips."""
        anchors = load(ANCHORS)
        identity = [a["prim"] for a in anchors["joints"] if a["x1_is_identity"]]
        assert identity == ["root_joint", "joint2"]
        assert anchors["counts"]["joints_whose_body1_pose_is_not_the_identity"] == 5
        for a in anchors["joints"]:
            assert a["local_rot1_wxyz"] == pytest.approx(EXPECTED_LOCAL_ROT1_WXYZ[a["prim"]], abs=1e-7), a["prim"]
            # every localPos1 in this arm is the origin; only the rotation carries the flip
            assert a["local_pos1_m"] == pytest.approx([0.0, 0.0, 0.0], abs=1e-12), a["prim"]

    def test_the_p6_edges_really_were_the_body0_side(self):
        """Proves the diagnosis: the number the frame tree stored IS localPos0/localRot0."""
        tree = load(FRAME_TREE)
        anchors = {a["prim"]: a for a in load(ANCHORS)["joints"]}
        for i in range(1, 7):
            edge = next(e for e in tree["edges"] if e.get("joint_name") == "joint%d" % i)
            a = anchors["joint%d" % i]
            assert edge["translation_m"] == pytest.approx(a["local_pos0_m"], abs=1e-12), i
            assert edge["rotation_quat"]["values"] == pytest.approx(a["local_rot0_wxyz"], abs=1e-12), i
            assert edge["rotation_quat"]["order"] == "wxyz"
            # and the frame tree carries NO body1 datum for that edge
            assert "local_pos1_m" not in edge and "local_rot1_wxyz" not in edge

    def test_stored_matrices_agree_with_the_stored_scalars(self):
        """A stale derived matrix next to correct scalars would be a second truth."""
        anchors = load(ANCHORS)
        for a in anchors["joints"]:
            X0 = homog(a["local_pos0_m"], a["local_rot0_wxyz"])
            X1 = homog(a["local_pos1_m"], a["local_rot1_wxyz"])
            assert np.array(a["X0_joint_in_body0"], float) == pytest.approx(X0, abs=1e-12), a["prim"]
            assert np.array(a["X1_joint_in_body1"], float) == pytest.approx(X1, abs=1e-12), a["prim"]
            assert np.array(a["T_body0_body1"], float) == pytest.approx(X0 @ inv_h(X1), abs=1e-12), a["prim"]
            assert a["det_R_T_body0_body1"] == pytest.approx(1.0, abs=1e-9), a["prim"]


# --------------------------------------------------------------------------- #
# 2. THE DRIFT GATE
# --------------------------------------------------------------------------- #
class TestCompositionBReproducesPxr:
    def _reference(self):
        v2 = load(CHAIN_V2)
        base = load(BASE_CHAIN)
        A = np.array(base["composition"]["A_usdgeom_xformops_authoritative"]["T"], float)
        ref = base["composition"]["A_usdgeom_xformops_authoritative"]["vs_pxr_end_pose"]
        p3 = resolve(load(ROOT / ref["p3_gripper_end_world_rest_pose_ref"]["file"]),
                     ref["p3_gripper_end_world_rest_pose_ref"]["json_pointer"])
        return v2, base, A, np.array(p3["world_position"], float)

    def test_the_pxr_reference_is_the_same_point_composition_a_lands_on(self):
        _, _, A, pxr_pos = self._reference()
        assert A[:3, 3] == pytest.approx(pxr_pos, abs=1e-9)

    def test_composition_b_v2_agrees_with_pxr_within_1e_6(self):
        """THE GATE. Drop the body1 side again and this fails by 0.33242 m."""
        anchors = load(ANCHORS)
        tree = load(FRAME_TREE)
        _, _, A, pxr_pos = self._reference()

        T = compose_chain(anchors, tree, None, drop_body1=False)
        err_m = float(np.abs(T[:3, 3] - pxr_pos).max())
        err_rad = angle_rad(T[:3, :3], A[:3, :3])
        assert err_m <= TOL_M, (
            "joint-edge composition drifted %.6e m from the pxr end pose (tolerance %.1e m); "
            "the body1 anchors are missing or wrong" % (err_m, TOL_M))
        assert err_rad <= TOL_RAD, (
            "joint-edge composition drifted %.6e rad from the pxr orientation "
            "(tolerance %.1e rad)" % (err_rad, TOL_RAD))
        # and it is the same point, not merely a nearby one
        assert T[:3, 3] == pytest.approx(pxr_pos, abs=TOL_M)

    def test_the_drift_gate_really_opens_when_the_body1_side_is_dropped(self):
        """Red-green: the same code path WITHOUT body1 fails, by exactly the historical number."""
        anchors = load(ANCHORS)
        tree = load(FRAME_TREE)
        _, _, A, pxr_pos = self._reference()
        T1 = compose_chain(anchors, tree, None, drop_body1=True)
        err1 = float(np.abs(T1[:3, 3] - pxr_pos).max())
        assert err1 > TOL_M * 1e5, (
            "composing with body0 only should reproduce the historical 0.33242 m gap, got %.6e" % err1)
        assert err1 == pytest.approx(HISTORICAL_B_V1_DELTA_M, abs=1e-5)
        assert angle_rad(T1[:3, :3], A[:3, :3]) == pytest.approx(math.pi, abs=1e-5), (
            "the five flipped joints should leave the chain oriented 180 deg away")

    def test_recomputed_b_v2_matches_the_number_stored_in_chain_v2(self):
        anchors = load(ANCHORS)
        tree = load(FRAME_TREE)
        v2 = load(CHAIN_V2)
        T = compose_chain(anchors, tree, None, drop_body1=False)
        stored = np.array(v2["composition_v2"]["B_v2_body0_and_body1"]["T"], float)
        assert T == pytest.approx(stored, abs=1e-12)

    def test_stored_error_numbers_are_the_measured_ones(self):
        v2 = load(CHAIN_V2)
        anchors = load(ANCHORS)
        tree = load(FRAME_TREE)
        _, _, A, pxr_pos = self._reference()
        prog = v2["composition_v2"]["error_progression"]

        Tv2 = compose_chain(anchors, tree, None, drop_body1=False)
        Tv1 = compose_chain(anchors, tree, None, drop_body1=True)
        e2 = float(np.abs(Tv2[:3, 3] - A[:3, 3]).max())
        r2 = angle_rad(Tv2[:3, :3], A[:3, :3])
        e1 = float(np.abs(Tv1[:3, 3] - A[:3, 3]).max())
        r1 = angle_rad(Tv1[:3, :3], A[:3, :3])

        assert prog["after"]["max_abs_delta_m"] == pytest.approx(e2, abs=1e-12)
        assert prog["after"]["geodesic_angle_rad"] == pytest.approx(r2, abs=1e-12)
        # the "before" translation is the rounded value recorded in history; the float64
        # recomputation is carried separately so neither number pretends to be the other
        assert prog["before"]["max_abs_delta_m"] == pytest.approx(HISTORICAL_B_V1_DELTA_M, abs=1e-12)
        assert prog["before"]["recomputed_from_anchors_m"] == pytest.approx(e1, abs=1e-12)
        assert prog["before"]["geodesic_angle_rad"] == pytest.approx(r1, abs=1e-12)
        assert prog["tolerance_m"] == TOL_M and prog["tolerance_rad"] == TOL_RAD
        assert prog["verdict"] == "PASS"
        assert prog["after"]["max_abs_delta_m"] < prog["before"]["max_abs_delta_m"]

    def test_the_before_number_is_a_pointer_not_a_second_copy(self):
        v2 = load(CHAIN_V2)
        before = v2["composition_v2"]["error_progression"]["before"]
        ref = before["ref"]
        assert ref["file"].endswith("mounted_chain.json")
        assert ref["sha256"] == sha256_norm(BASE_CHAIN)
        historical = resolve(load(ROOT / ref["file"]), ref["json_pointer"])
        assert historical == pytest.approx(HISTORICAL_B_V1_DELTA_M, abs=1e-12)
        assert before["max_abs_delta_m"] == pytest.approx(historical, abs=1e-12)
        # the full-precision recomposition agrees with the recorded value to its rounding
        assert before["recomputed_from_anchors_m"] == pytest.approx(historical, abs=1e-9)

    def test_b_v2_is_a_proper_rigid_transform(self):
        v2 = load(CHAIN_V2)
        B = v2["composition_v2"]["B_v2_body0_and_body1"]
        T = np.array(B["T"], float)
        assert float(np.linalg.det(T[:3, :3])) == pytest.approx(1.0, abs=1e-9)
        assert T[:3, :3].T @ T[:3, :3] == pytest.approx(np.eye(3), abs=1e-9)
        assert np.array_equal(T[3], np.array([0.0, 0.0, 0.0, 1.0]))
        assert B["det_R"] == pytest.approx(1.0, abs=1e-9)
        assert B["R_orthonormal_max_error"] <= 1e-9


# --------------------------------------------------------------------------- #
# 3. history is not rewritten
# --------------------------------------------------------------------------- #
class TestHistoryIsNotRewritten:
    def test_mounted_chain_is_byte_identical_to_the_pinned_hash(self):
        v2 = load(CHAIN_V2)
        assert v2["base_chain"]["sha256"] == sha256_norm(BASE_CHAIN)
        assert v2["base_chain"]["sha256_basis"] == "LF-normalised bytes"
        assert v2["provenance"]["base_chain_modified"] is False
        assert v2["additive_only"]

    def test_the_open_issue_is_still_recorded_as_it_was_found(self):
        base = load(BASE_CHAIN)
        issue = base["composition"]["open_issue"]
        assert issue["id"] == "P6-02-OI-1"
        assert issue["status"] == "OPEN", "history must keep the status it was found with"
        assert issue["max_abs_delta_m"] > 0.001
        assert issue["max_abs_delta_m"] == pytest.approx(HISTORICAL_B_V1_DELTA_M, abs=1e-5)
        # ...and the resolution lives NEXT to it, not on top of it
        v2 = load(CHAIN_V2)
        assert v2["issue_resolution"]["id"] == "P6-02-OI-1"
        assert v2["issue_resolution"]["status"] == "RESOLVED"

    def test_the_historical_composition_matrices_are_inherited_verbatim(self):
        base = load(BASE_CHAIN)
        v2 = load(CHAIN_V2)
        for key in ("A_usdgeom_xformops_authoritative", "B_p6_physics_joint_anchors"):
            assert v2["composition"][key] == base["composition"][key], key

    def test_the_five_unknown_edges_are_the_same_five(self):
        base = load(BASE_CHAIN)
        v2 = load(CHAIN_V2)
        assert base["summary"]["unknown_edge_ids"] == UNKNOWN_EDGE_IDS
        assert base["summary"]["unknown"] == 5
        assert base["summary"]["asserted"] == 10
        assert base["summary"]["edges_total"] == 15
        assert v2["unknown_edges_unchanged"]["edge_ids"] == UNKNOWN_EDGE_IDS
        assert v2["unknown_edges_unchanged"]["status"] == "UNKNOWN"

    def test_no_anchor_claim_is_made_for_an_unknown_boundary(self):
        """The anchors describe the ARM only: they may not name a mount, racket or camera node."""
        blob = json.dumps(load(ANCHORS)) + json.dumps(load(CHAIN_V2)["joint_anchor_extension"])
        for forbidden in ("racket_mount", "racket_root", "racket_contact_frame",
                          "chassis_morph_one", "gripper_end\", \"body1"):
            assert forbidden not in blob, forbidden


# --------------------------------------------------------------------------- #
# 4. the UNKNOWN guard
# --------------------------------------------------------------------------- #
class TestUnknownEdgesGainedNoNumber:
    NUMERIC_KEYS = {"translation_m", "rotation_quat", "rotation_matrix", "values",
                    "translation", "quaternion", "rotation", "offset", "position",
                    "local_pos1_m", "local_rot1_wxyz", "T_body0_body1"}

    def test_the_edge_payload_rule_still_holds_in_the_base_chain(self):
        base = load(BASE_CHAIN)
        for e in base["edges"]:
            if e["status"] != "UNKNOWN":
                continue
            assert e["value_ref"] is None
            assert e["missing_datum"] and e["would_come_from"]
            payload = {k: v for k, v in e.items()
                       if k not in ("edge", "from", "to", "edge_kind", "status", "value_ref",
                                    "missing_datum", "would_come_from", "unknown_guard",
                                    "p3_01_anchor")}
            assert payload == {}, "edge %d gained a payload: %s" % (e["edge"], sorted(payload))

    def test_no_unknown_edge_carries_a_numeric_scalar(self):
        base = load(BASE_CHAIN)

        def walk(value, path):
            if isinstance(value, bool):
                return []
            if isinstance(value, (int, float)):
                return [(path, value)]
            if isinstance(value, dict):
                out = []
                for k, v in value.items():
                    out += walk(v, path + "." + k)
                return out
            if isinstance(value, (list, tuple)):
                out = []
                for i, v in enumerate(value):
                    out += walk(v, "%s[%d]" % (path, i))
                return out
            return []

        for e in base["edges"]:
            if e["status"] != "UNKNOWN":
                continue
            for key in ("missing_datum", "would_come_from", "unknown_guard"):
                leaked = walk(e[key], key)
                assert leaked == [], "edge %d.%s smuggles numbers: %s" % (e["edge"], key, leaked)

    def test_the_v2_unknown_block_is_metadata_only(self):
        """Filling in a mount offset would have to appear here; it must not."""
        block = load(CHAIN_V2)["unknown_edges_unchanged"]
        for key in self.NUMERIC_KEYS:
            assert key not in block, "unknown_edges_unchanged carries a numeric key %s" % key
        assert block["numeric_payload"] is None
        assert block["status"] == "UNKNOWN"

        def floats_only(value, path):
            if isinstance(value, bool):
                return []
            if isinstance(value, float):
                return [(path, value)]
            if isinstance(value, dict):
                out = []
                for k, v in value.items():
                    out += floats_only(v, path + "." + k)
                return out
            if isinstance(value, (list, tuple)):
                out = []
                for i, v in enumerate(value):
                    out += floats_only(v, "%s[%d]" % (path, i))
                return out
            return []

        leaked = floats_only(block, "unknown_edges_unchanged")
        assert leaked == [], "unknown_edges_unchanged carries measurements: %s" % leaked
        # edge ids are the only integers allowed, and they must be the same five
        assert list(block["edge_ids"]) == UNKNOWN_EDGE_IDS

    def test_the_still_unknown_list_names_the_real_open_datas(self):
        block = load(CHAIN_V2)["unknown_edges_unchanged"]
        joined = " ".join(block["still_unknown"]).lower()
        for needle in ("mount", "flange", "racket", "d455"):
            assert needle in joined, needle
        assert len(block["still_unknown"]) >= 4
        # the resolution may not claim to have answered them
        assert "resolved" not in json.dumps(block["still_unknown"]).lower()


# --------------------------------------------------------------------------- #
# 5. the text parse itself: re-read the layer when a read-only copy is around
# --------------------------------------------------------------------------- #
class TestLayerReparse:
    """The layer is NOT committed (it lives on the GPU host). When a read-only copy is
    present, the recorded anchors are re-derived from the raw text and must match."""

    def _physics(self):
        cand = [os.environ.get("P6_PHYSICS_USDA"), str(ROOT / "_scratch_p6c" / "physics.usda")]
        for c in cand:
            if c and Path(c).is_file():
                return Path(c)
        return None

    def test_reparsed_layer_reproduces_every_recorded_anchor(self):
        path = self._physics()
        if path is None:
            pytest.skip("no read-only copy of physics.usda locally; anchors are pinned by sha256 instead")
        import re
        text = path.read_text(encoding="utf-8")
        anchors = load(ANCHORS)
        if sha256_norm(path) != anchors["source"]["joint_layer_sha256"]:
            if hashlib.sha256(path.read_bytes()).hexdigest() != anchors["source"]["joint_layer_sha256"]:
                pytest.skip("local copy differs from the recorded layer")
        for a in anchors["joints"]:
            block = re.search(r'def\s+\w+Joint\s+"%s".*?\n\s*\}' % a["prim"], text, re.S)
            assert block, a["prim"]
            body = block.group(0)
            for field, key in (("physics:localPos0", "local_pos0_m"),
                               ("physics:localPos1", "local_pos1_m"),
                               ("physics:localRot0", "local_rot0_wxyz"),
                               ("physics:localRot1", "local_rot1_wxyz")):
                m = re.search(r'%s\s*=\s*\(([^)]*)\)' % re.escape(field), body)
                assert m, (a["prim"], field)
                vals = [float(v) for v in m.group(1).replace(",", " ").split()]
                assert vals == pytest.approx(a[key], abs=1e-12), (a["prim"], field)
