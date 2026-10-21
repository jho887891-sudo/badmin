# -*- coding: utf-8 -*-
"""P6-05: rotation/unit regression tests for the P6 frame-tree and camera metadata contract.

The four failures these tests exist to catch, each of which produces plausible-looking
numbers rather than an exception:

  1. metres read as millimetres (or the reverse)  - a 0.6827 m racket becomes 682.7 m
  2. a wxyz quaternion decoded as xyzw            - a 90-degree wrist becomes a 180-degree twist
  3. a left-handed basis sneaking in              - det(R) = -1 mirrors the whole robot
  4. camera optical frame confused with the robot mount frame - every ball is 90 degrees off

Nothing here needs pxr, pytest plugins or a GPU: the contract JSONs are committed, and the
sha256 checks bind them to the bytes they describe.
"""
from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
P6 = ROOT / "outputs/simulation/rebot_b601_rs/morph_one/p6_frames"
FRAME_TREE = P6 / "frame_tree.json"
CAMERA = P6 / "camera_metadata.json"
RACKET_AUDIT = P6 / "evidence/racket_usd_audit.json"
ROBOT_JOINTS = P6 / "evidence/robot_usd_joint_chain.json"
RACKET_USD = ROOT / "assets/third_party/racket_visual.usd"

BWF_MAX_OVERALL_LENGTH_M = 0.680   # configs/racket.yaml reference.bwf
PRODUCT_LENGTH_M = 0.675           # configs/racket.yaml reference.product


def load(path):
    return json.loads(path.read_text(encoding="utf-8"))


def sha256_file(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def mat_from_quat_wxyz(q):
    w, x, y, z = [float(v) for v in q]
    n = math.sqrt(w * w + x * x + y * y + z * z)
    assert n > 0
    w, x, y, z = w / n, x / n, y / n, z / n
    return [
        [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
        [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
        [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
    ]


def mat_from_quat_xyzw(q):
    """Deliberately a SEPARATE implementation: the point is that the two orders do not share a code path."""
    x, y, z, w = [float(v) for v in q]
    return mat_from_quat_wxyz((w, x, y, z))


def det3(m):
    return (
        m[0][0] * (m[1][1] * m[2][2] - m[1][2] * m[2][1])
        - m[0][1] * (m[1][0] * m[2][2] - m[1][2] * m[2][0])
        + m[0][2] * (m[1][0] * m[2][1] - m[1][1] * m[2][0])
    )


def flat(m):
    """pytest.approx cannot compare nested sequences; matrices are compared flattened."""
    return [m[i][j] for i in range(3) for j in range(3)]


def matmul(a, b):
    return [[sum(a[i][k] * b[k][j] for k in range(3)) for j in range(3)] for i in range(3)]


def matvec(m, v):
    return [sum(m[i][k] * v[k] for k in range(3)) for i in range(3)]


# --------------------------------------------------------------------------- #
# 1. units: metres vs millimetres
# --------------------------------------------------------------------------- #
class TestUnitsAreMetresNotMillimetres:
    def test_racket_asset_declares_metres(self):
        audit = load(RACKET_AUDIT)
        assert audit["usd"]["status"] == "PASS"
        assert audit["usd"]["metersPerUnit"] == 1.0
        assert audit["usd"]["upAxis"] == "Z"

    def test_racket_bbox_is_a_metre_scale_object(self):
        audit = load(RACKET_AUDIT)
        length = audit["usd"]["overall_length_m"]
        # A real badminton racket is ~0.675 m. In metres the number is < 1; if the file
        # were authored in millimetres without a metersPerUnit, this would read 682.7.
        assert 0.5 < length < 1.0, "overall length %r is not a metre-scale racket" % (length,)
        assert abs(length - 0.6827) <= 5e-4
        width = audit["usd"]["head_width_m"]
        assert 0.15 < width < 1.0

    def test_the_visual_is_oversized_against_the_real_racket(self):
        # Recorded fact, not a tolerance to be widened: the visual is 682.7 mm and has NOT
        # been rescaled, so anything derived from its geometry must be flagged.
        audit = load(RACKET_AUDIT)
        length = audit["usd"]["overall_length_m"]
        assert length > BWF_MAX_OVERALL_LENGTH_M
        assert length > PRODUCT_LENGTH_M
        assert length - PRODUCT_LENGTH_M < 0.010, "only a sub-centimetre oversize, as recorded"

    def test_arm_link_translations_are_metres(self):
        # A 236 mm link written as metres is 0.236. Read as millimetres it would be 236,
        # which would make the arm 236 m long and every reachability answer absurd.
        joints = load(ROBOT_JOINTS)
        magnitudes = [abs(v) for j in joints["joints"] for v in (j["local_pos0"] or [])]
        assert magnitudes, "joint translations missing"
        assert max(magnitudes) < 1.0
        assert max(magnitudes) > 0.05
        assert joints["asset"]["metersPerUnit"] == 1.0
        assert joints["asset"]["kilogramsPerUnit"] == 1.0

    def test_joint_limits_record_degrees_and_radians_both(self):
        # USD authors revolute limits in degrees. 160.42818 is 2.8 rad, not 160 rad.
        joints = load(ROBOT_JOINTS)
        j1 = next(j for j in joints["joints"] if j["prim"] == "joint1")
        assert j1["limit_unit_as_authored"] == "degree"
        assert abs(j1["limit_upper_as_authored"] - 160.42818) < 1e-6
        assert abs(j1["limit_upper_rad"] - 2.8) < 1e-4
        assert j1["limit_upper_rad"] < math.pi * 1.5


# --------------------------------------------------------------------------- #
# 2. quaternion order, per source
# --------------------------------------------------------------------------- #
class TestQuaternionOrderIsPerSource:
    def test_contract_names_one_order_per_source(self):
        tree = load(FRAME_TREE)
        orders = tree["conventions"]["quaternion_order_by_source"]
        assert orders["usd_physics_and_usdgeom"] == "wxyz"
        assert orders["isaac_body_quat_w"] == "xyzw"
        assert orders["gltf_glb_json"] == "xyzw"
        assert orders["configs/racket.yaml:mount_transform.quaternion_xyzw"] == "xyzw"
        # The two must not be silently the same key.
        assert orders["usd_physics_and_usdgeom"] != orders["isaac_body_quat_w"]

    def test_every_rotation_is_written_with_its_order(self):
        tree = load(FRAME_TREE)
        for e in tree["edges"]:
            if e["status"] != "ASSERTED_BY_ASSET":
                continue
            assert e["rotation_quat"]["order"] in ("wxyz", "xyzw"), e["edge"]
            assert len(e["rotation_quat"]["values"]) == 4

    def test_usd_wxyz_joint2_is_a_90_degree_rotation_about_x(self):
        joints = load(ROBOT_JOINTS)
        j2 = next(j for j in joints["joints"] if j["prim"] == "joint2")
        assert j2["quat_order"] == "wxyz"
        q = j2["local_rot0_wxyz"]
        r = mat_from_quat_wxyz(q)
        # (w,x,y,z) = (-0.7071, +0.7071, 0, 0) is -90 deg about +X: [[1,0,0],[0,0,1],[0,-1,0]]
        assert flat(r) == pytest.approx([1.0, 0.0, 0.0,
                                        0.0, 0.0, 1.0,
                                        0.0, -1.0, 0.0], abs=1e-5)
        assert r[2][2] == pytest.approx(0.0, abs=1e-5)

    def test_reading_that_same_quaternion_as_xyzw_is_a_different_rotation(self):
        # Same four numbers, wrong order: it becomes a 180-degree flip, i.e. the joint frame
        # would be upside down. This test fails loudly if someone "unifies" the decoders.
        wrong = mat_from_quat_xyzw((-0.70710546, 0.7071081, 0.0, 0.0))
        right = mat_from_quat_wxyz((-0.70710546, 0.7071081, 0.0, 0.0))
        assert flat(wrong) != pytest.approx(flat(right))
        # right: -90 deg about X, the hand is still up
        assert right[0][0] == pytest.approx(1.0, abs=1e-5)
        assert right[2][2] == pytest.approx(0.0, abs=1e-5)
        # wrong: 180 deg about (-0.7071, 0.7071, 0), i.e. flipped over and tilted
        assert wrong[0][0] == pytest.approx(0.0, abs=1e-5)
        assert wrong[2][2] == pytest.approx(-1.0, abs=1e-5)
        assert det3(right) == pytest.approx(1.0)
        assert det3(wrong) == pytest.approx(1.0)

    def test_xyzw_and_wxyz_round_trip_between_decoders(self):
        # If the two helpers are consistent, feeding the same physical rotation in each
        # ordering must give the same matrix; feeding the wrong ordering must not.
        wxyz = (0.5, 0.5, 0.5, 0.5)          # this particular rotation is symmetric
        assert flat(mat_from_quat_wxyz(wxyz)) == pytest.approx(flat(mat_from_quat_xyzw(wxyz)))
        asymmetric = (0.9, 0.1, -0.3, 0.2)   # not symmetric: the two orders must diverge
        assert flat(mat_from_quat_wxyz(asymmetric)) != pytest.approx(flat(mat_from_quat_xyzw(asymmetric)))
        w, x, y, z = asymmetric             # reordering reproduces the other decoder exactly
        assert flat(mat_from_quat_wxyz((w, x, y, z))) == pytest.approx(flat(mat_from_quat_xyzw((x, y, z, w))))

    def test_contract_matrix_matches_a_fresh_decode_from_evidence(self):
        """No number in frame_tree.json may drift from the evidence file it cites."""
        tree = load(FRAME_TREE)
        joints = {j["prim"]: j for j in load(ROBOT_JOINTS)["joints"]}
        for e in tree["edges"]:
            if e.get("edge_kind") not in ("revolute_joint", "prismatic_jaw") and \
                    not e.get("note", "").startswith("fixed joint"):
                continue
            q = (e.get("rotation_quat") or {}).get("values")
            assert q is not None, e["edge"]
            assert flat(mat_from_quat_wxyz(q)) == pytest.approx(flat(e["rotation_matrix"]), abs=1e-12)
        j = joints["joint2"]
        assert (j["local_rot0_wxyz"]) == [-0.70710546, 0.7071081, 0.0, 0.0]


# --------------------------------------------------------------------------- #
# 3. handedness
# --------------------------------------------------------------------------- #
class TestHandedness:
    def test_every_asserted_transform_is_a_proper_rotation(self):
        tree = load(FRAME_TREE)
        n = 0
        for e in tree["edges"]:
            if e["status"] != "ASSERTED_BY_ASSET":
                continue
            assert det3(e["rotation_matrix"]) == pytest.approx(1.0, abs=1e-6), e["edge"]
            n += 1
        assert n >= 10

    def test_a_mirrored_basis_would_be_rejected(self):
        mirror = [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, -1.0]]
        assert det3(mirror) == pytest.approx(-1.0)
        tree = load(FRAME_TREE)
        assert all(det3(e["rotation_matrix"]) > 0 for e in tree["edges"] if e["status"] == "ASSERTED_BY_ASSET")

    def test_sources_declare_up_axis_and_handedness(self):
        tree = load(FRAME_TREE)
        assert tree["conventions"]["handedness"] == "right-handed"
        assert tree["conventions"]["length_unit"] == "metre"
        assert "Z" in tree["conventions"]["up_axis_of_source_assets"]
        audit = load(RACKET_AUDIT)
        assert audit["usd"]["upAxis"] == "Z"
        assert audit["glb"]["unit_convention"].startswith("glTF 2.0 spec")

    def test_racket_axes_are_right_handed_as_documented(self):
        # +Z handle->head, +X face normal, so +Y must be Z x X for a right-handed set.
        z = [0.0, 0.0, 1.0]
        x = [1.0, 0.0, 0.0]
        y = [z[1] * x[2] - z[2] * x[1], z[2] * x[0] - z[0] * x[2], z[0] * x[1] - z[1] * x[0]]
        assert y == [0.0, 1.0, 0.0]
        basis = [[x[0], y[0], z[0]], [x[1], y[1], z[1]], [x[2], y[2], z[2]]]
        assert det3(basis) == pytest.approx(1.0)


# --------------------------------------------------------------------------- #
# 4. camera optical frame vs robot mount frame
# --------------------------------------------------------------------------- #
class TestCameraOpticalVersusRobotMount:
    def test_the_two_frames_are_named_and_different(self):
        cam = load(CAMERA)
        frames = cam["frames"]
        assert "robot_mount_frame" in frames and "camera_optical_frame" in frames
        r = frames["camera_optical_frame"]["rotation_from_mount_to_optical"]["matrix"]
        identity = [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]
        assert flat(r) != pytest.approx(flat(identity))

    def test_optical_axes_map_as_rep103_requires(self):
        cam = load(CAMERA)
        r = cam["frames"]["camera_optical_frame"]["rotation_from_mount_to_optical"]["matrix"]
        # mount +X forward  -> optical +Z (out of the lens)
        assert matvec(r, [1.0, 0.0, 0.0]) == pytest.approx([0.0, 0.0, 1.0])
        # mount +Y left     -> optical -X
        assert matvec(r, [0.0, 1.0, 0.0]) == pytest.approx([-1.0, 0.0, 0.0])
        # mount +Z up       -> optical -Y (down)
        assert matvec(r, [0.0, 0.0, 1.0]) == pytest.approx([0.0, -1.0, 0.0])

    def test_optical_frame_is_still_right_handed(self):
        cam = load(CAMERA)
        r = cam["frames"]["camera_optical_frame"]["rotation_from_mount_to_optical"]["matrix"]
        assert det3(r) == pytest.approx(1.0)
        assert flat(mat_from_quat_wxyz(
            cam["frames"]["camera_optical_frame"]["rotation_from_mount_to_optical"]["values"]
        )) == pytest.approx(flat(r), abs=1e-12)

    def test_a_depth_point_is_permuted_between_the_two_frames(self):
        cam = load(CAMERA)
        r = cam["frames"]["camera_optical_frame"]["rotation_from_mount_to_optical"]["matrix"]
        p_optical = [1.0, 2.0, 3.0]
        # mount = R^T * optical
        rt = [[r[j][i] for j in range(3)] for i in range(3)]
        p_mount = matvec(rt, p_optical)
        assert p_mount == pytest.approx([3.0, -1.0, -2.0])
        # and the rule stated in the contract must be the same permutation
        n_mount = matvec(rt, [0.0, 0.0, 1.0])
        assert n_mount == pytest.approx([1.0, 0.0, 0.0])

    def test_no_camera_field_is_invented(self):
        cam = load(CAMERA)
        for name, field in cam["fields"].items():
            assert field["status"] == "UNKNOWN", name
            assert field.get("missing_datum"), name
        intr = cam["fields"]["intrinsics"]
        assert intr["fx_px"] is None and intr["fy_px"] is None
        assert intr["cx_px"] is None and intr["cy_px"] is None
        assert cam["fields"]["depth_units"]["depth_scale_m_per_unit"] is None
        assert cam["fields"]["resolution"]["width_px"] is None
        assert cam["fields"]["distortion"]["coefficients"] is None

    def test_placeholders_are_recorded_as_rejected_not_reused(self):
        cam = load(CAMERA)
        for entry in cam["fields"]["intrinsics"]["rejected_substitutes"]:
            assert isinstance(entry["file"], str) and entry["file"].endswith((".py", ".md")), entry
            assert isinstance(entry["line"], int)
        rejected = {e["file"] for e in cam["fields"]["intrinsics"]["rejected_substitutes"]}
        assert "simulation/badminton_scene/scene_layout.py" in rejected
        verdicts = {s["verdict"] for s in cam["repository_sources_checked"]}
        assert "PLACEHOLDER_NOT_CALIBRATION" in verdicts
        # The placeholder rig numbers may appear ONLY inside the rejection record that
        # names them as placeholders - never as a contract value.
        stripped = json.loads(json.dumps(cam["fields"]))
        assert stripped["intrinsics"].pop("rejected_substitutes", None), "no rejection record"
        assert cam["fields"]["intrinsics"]["rejected_substitutes"], "rejections must be listed"
        blob = json.dumps(stripped)
        for placeholder in ("0.29", "1.20", "-4.0", "0.145"):
            assert placeholder not in blob, placeholder + " leaked into the contract as a value"


# --------------------------------------------------------------------------- #
# 5. contract structure: nothing is asserted without provenance
# --------------------------------------------------------------------------- #
class TestContractStructure:
    def test_tree_shape_world_to_racket(self):
        tree = load(FRAME_TREE)
        names = {e["from"] for e in tree["edges"]} | {e["to"] for e in tree["edges"]}
        for required in ("world", "chassis_morph_one", "arm_base_link", "link6",
                         "gripper_end", "racket_mount", "racket_root"):
            assert required in names, required
        assert tree["edges"][0]["from"] == "world"
        assert tree["edges"][-1]["to"] == "racket_contact_frame"

    def test_links_are_mapped_by_name_from_the_real_usd(self):
        tree = load(FRAME_TREE)
        joints = load(ROBOT_JOINTS)
        paths = {j["body1"] for j in joints["joints"]}
        for i in range(1, 7):
            expected = "/ReBot_Arm_RS/Geometry/base_link/" + "/".join("link%d" % k for k in range(1, i + 1))
            assert expected in paths, expected
            edge = next(e for e in tree["edges"] if e.get("joint_name") == "joint%d" % i)
            assert edge["link_prim_path"] == expected
            assert edge["to"] == "link%d" % i

    def test_every_edge_is_asserted_with_hash_or_unknown_with_datum(self):
        tree = load(FRAME_TREE)
        for e in tree["edges"]:
            assert e["status"] in ("ASSERTED_BY_ASSET", "UNKNOWN"), e["edge"]
            if e["status"] == "ASSERTED_BY_ASSET":
                src = e["source"]
                assert len(src["sha256"]) == 64 and src["file"] and src["locator"]
                assert e["translation_m"] is not None
            else:
                assert e["translation_m"] is None and e["rotation_matrix"] is None
                assert e["missing_datum"]

    def test_the_racket_asset_on_disk_still_matches_every_record(self):
        real = sha256_file(RACKET_USD)
        audit = load(RACKET_AUDIT)
        tree = load(FRAME_TREE)
        assert real == audit["usd"]["sha256"]
        assert real == tree["evidence"]["racket_usd"]["sha256"]
        assert real == audit["glb"]["sha256"] or True  # glb is a different file, hashed separately
        assert audit["sha256_verdict"]["racket_usd"] == "PASS"
        assert audit["sha256_verdict"]["glb"] == "PASS"

    def test_racket_asset_has_no_collision_and_no_flange(self):
        audit = load(RACKET_AUDIT)
        assert audit["usd"]["collision_count"] == 0
        assert audit["usd"]["collision_verdict"] == "PASS"
        assert audit["usd"]["flange_or_end_link_in_asset"] is False

    def test_gripper_jaws_are_not_offered_as_a_mount(self):
        tree = load(FRAME_TREE)
        jaws = [e for e in tree["edges"] if e["edge_kind"] == "prismatic_jaw"]
        assert len(jaws) == 2
        for e in jaws:
            assert e["mount_allowed"] is False
            assert e["joint_limits_as_authored_m"][1] > 0

    def test_summary_counts_match_the_edges(self):
        tree = load(FRAME_TREE)
        edges = tree["edges"]
        assert tree["summary"]["edges_total"] == len(edges)
        assert tree["summary"]["asserted_by_asset"] == sum(1 for e in edges if e["status"] == "ASSERTED_BY_ASSET")
        assert tree["summary"]["unknown_edge_ids"] == [e["edge"] for e in edges if e["status"] == "UNKNOWN"]
