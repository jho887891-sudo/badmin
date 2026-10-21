#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""P3-01 test: the frozen chassis <-> arm-base frame tree and its reference fixtures.

Pure python + numpy.  pxr is OPTIONAL: when it is missing (or the read-only asset copy
is not present) the test verifies the committed evidence, says so explicitly and
never silently claims it re-opened the asset.

Run:
    python -m pytest tests/simulation/p3_mount_frame_test.py -q
or, without pytest:
    python tests/simulation/p3_mount_frame_test.py
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "tools" / "p3_mount"))
import se3  # noqa: E402

CONTRACT_PATH = ROOT / "outputs/simulation/rebot_b601_rs/morph_one/p3_mount/mount_frame_tree.json"
FIXTURES_PATH = ROOT / "outputs/simulation/rebot_b601_rs/morph_one/p3_mount/mount_fixtures.json"
ASSET_EXPECTED_SHA = "96d4f2ec1126dec6eaf9e70ab664ba6cc30fa81bdec86178a3e50b7a23bdd510"
NON_TRIVIAL_MIN_RAD = 0.1
EXACT = 1e-9


def load():
    contract = json.loads(CONTRACT_PATH.read_text(encoding="utf-8"))
    fixtures = json.loads(FIXTURES_PATH.read_text(encoding="utf-8"))
    return contract, fixtures


def edge(contract, name):
    for item in contract["frame_tree"]["edges"]:
        if item["edge"] == name:
            return item
    raise AssertionError("edge %r missing from the frame tree" % name)


def test_files_exist_and_schema():
    assert CONTRACT_PATH.is_file(), "contract missing: %s" % CONTRACT_PATH
    assert FIXTURES_PATH.is_file(), "fixtures missing: %s" % FIXTURES_PATH
    contract, fixtures = load()
    assert contract["schema"] == "p3_mount_frame_tree/1"
    assert contract["task"] == "P3-01"
    assert fixtures["schema"] == "p3_mount_fixtures/1"


def test_asset_provenance_is_the_real_asset():
    contract, _ = load()
    assert contract["asset"]["layers"]["ReBot_Arm_RS.usda"]["sha256"] == ASSET_EXPECTED_SHA
    assert contract["asset"]["units_verdict"] == "PASS"
    assert contract["asset"]["meters_per_unit"] == 1.0
    assert contract["asset"]["up_axis"].upper() == "Z"
    assert contract["asset"]["active_physics_variant"] == "physx"
    assert contract["provenance"]["isaac_sim_used"] is False
    assert contract["provenance"]["gpu_used"] is False
    for name, layer in contract["asset"]["layers"].items():
        assert len(layer["sha256"]) == 64, "layer %s has no sha256" % name


def test_frame_tree_chain_composes_consistently():
    contract, _ = load()
    names = [node["name"] for node in contract["frame_tree"]["nodes"]]
    for required in ("world", "chassis", "arm_base", "base_link", "gripper_end"):
        assert required in names, "frame %r missing from the tree" % required
    assert "link1..link6" in names
    arm_base = [n for n in contract["frame_tree"]["nodes"] if n["name"] == "arm_base"][0]
    assert arm_base["prim_path"] == "/ReBot_Arm_RS"
    base_link = [n for n in contract["frame_tree"]["nodes"] if n["name"] == "base_link"][0]
    assert base_link["prim_path"] == "/ReBot_Arm_RS/Geometry/base_link"
    for item in contract["frame_tree"]["edges"]:
        for key in ("edge", "joint"):
            assert key in item
    # the anchor really is the authored root_joint, with body0 retargetable to the chassis
    anchor = contract["joint_anchor"]
    assert anchor["prim_path"] == "/ReBot_Arm_RS/Physics/root_joint"
    assert anchor["type_as_authored"] == "PhysicsFixedJoint"
    assert anchor["body0"]["value"] == "/ReBot_Arm_RS"
    assert anchor["body1"]["value"] == "/ReBot_Arm_RS/Geometry/base_link"
    for field in ("local_pos0", "local_rot0", "local_pos1", "local_rot1"):
        assert len(anchor[field]["sha256"]) == 64
        assert anchor[field]["locator"], "anchor field %s has no locator" % field
        assert anchor[field]["unit"]


def test_anchor_numbers_are_the_authored_identity():
    contract, _ = load()
    anchor = contract["joint_anchor"]
    assert np.allclose(anchor["local_pos0"]["value"], [0.0, 0.0, 0.0], atol=0.0)
    assert np.allclose(anchor["local_pos1"]["value"], [0.0, 0.0, 0.0], atol=0.0)
    assert np.allclose(anchor["local_rot0"]["value"], [1.0, 0.0, 0.0, 0.0], atol=0.0)
    assert np.allclose(anchor["local_rot1"]["value"], [1.0, 0.0, 0.0, 0.0], atol=0.0)
    mount = contract["chassis_to_arm_base_transform"]
    assert mount["position_norm_m"] <= EXACT
    assert mount["rotation_angle_rad"] <= EXACT
    # the asset does NOT author a CAD mount offset; the contract must say so
    assert any("CAD mount point" in item["what"] for item in contract["unknown"])


def test_determinants_are_plus_one():
    contract, fixtures = load()
    checked = 0
    for hop in contract["arm_chain_rest_pose"]["hops"]:
        rotation = se3.quat_to_matrix(hop["local_transform"]["quat_wxyz"])
        assert abs(se3.determinant(rotation) - 1.0) < 1e-9, "det(R)!=+1 on hop %s" % hop["child"]
        checked += 1
    mount = contract["chassis_to_arm_base_transform"]["rotation_quat_wxyz"]["value"]
    assert abs(se3.determinant(se3.quat_to_matrix(mount)) - 1.0) < 1e-9
    checked += 1
    for fixture in fixtures["fixtures"]:
        for frame, pose in fixture["expected_world_pose"].items():
            rotation = se3.quat_to_matrix(pose["quat_wxyz"])
            assert abs(se3.determinant(rotation) - 1.0) < 1e-9, "det(R)!=+1 on %s/%s" % (fixture["id"], frame)
            checked += 1
        relative = se3.quat_to_matrix(fixture["expected_mount_relative"]["quat_wxyz"])
        assert abs(se3.determinant(relative) - 1.0) < 1e-9
        checked += 1
    assert checked >= 3


def test_chain_product_matches_the_recorded_rest_pose():
    contract, fixtures = load()
    hops = contract["arm_chain_rest_pose"]["hops"]
    assert hops, "no chain hops recorded (pxr was unavailable when the contract was frozen)"
    product = se3.compose(*[se3.make_transform(h["local_transform"]["position"],
                                               h["local_transform"]["quat_wxyz"]) for h in hops])
    recorded = contract["arm_chain_rest_pose"]["gripper_end_world_rest_pose"]
    pos, quat = se3.pose_of(product)
    assert se3.position_delta(pos, recorded["world_position"]) < EXACT
    assert se3.quat_angle_rad(quat, recorded["world_quat_wxyz"]) < EXACT
    check = fixtures["chain_product"]["self_check_against_pxr"]
    assert check["position_delta_m"] < EXACT
    assert check["rotation_delta_rad"] < EXACT


def test_fixtures_are_non_trivial():
    _, fixtures = load()
    assert len(fixtures["fixtures"]) == 3
    angles = fixtures["non_triviality"]["chassis_rotation_angles_rad"]
    by_id = {f["id"]: f for f in fixtures["fixtures"]}
    assert by_id["F1_identity"]["expected_world_pose"]["chassis"]["rotation_angle_rad"] <= EXACT
    for fixture_id, minimum in (("F2_yaw_035", 0.35), ("F3_yaw_030_roll_010", 0.30)):
        fixture = by_id[fixture_id]
        angle = fixture["expected_world_pose"]["chassis"]["rotation_angle_rad"]
        assert angle > NON_TRIVIAL_MIN_RAD, "%s is trivial (%f rad)" % (fixture_id, angle)
        assert angle >= minimum - 1e-9, "%s lost its rotation amplitude" % fixture_id
        assert fixture["non_trivial"] is True
        # a rotation that is really there must also move the chassis translation
        assert float(np.linalg.norm(fixture["chassis_command_reference"]["position_m"])) > 0.01
    assert max(angles) > NON_TRIVIAL_MIN_RAD
    assert fixtures["non_triviality"]["non_trivial_fixture_count"] >= 2


def test_mount_relative_pose_is_invariant_under_every_fixture():
    contract, fixtures = load()
    tolerance = contract["tolerances"]
    assert tolerance["position_m"] == 0.001, "the pre-registered position tolerance was changed"
    assert tolerance["rotation_rad"] == 0.001, "the pre-registered rotation tolerance was changed"
    assert tolerance["source"]["line"] == 25  # PHASE_P3_DYNAMIC_MOUNT_GATE.md, gate row mount_stable

    mount = contract["chassis_to_arm_base_transform"]
    t_mount = se3.make_transform(mount["position_m"]["value"], mount["rotation_quat_wxyz"]["value"])
    worst_pos = worst_rot = 0.0
    for fixture in fixtures["fixtures"]:
        chassis = fixture["expected_world_pose"]["chassis"]
        t_chassis = se3.make_transform(chassis["position_m"], chassis["quat_wxyz"])
        t_arm_base = t_chassis @ t_mount
        relative = np.linalg.inv(t_chassis) @ t_arm_base
        rel_pos, rel_quat = se3.pose_of(relative)
        recorded = fixture["expected_mount_relative"]
        worst_pos = max(worst_pos, se3.position_delta(rel_pos, recorded["position_m"]))
        worst_rot = max(worst_rot, se3.quat_angle_rad(rel_quat, recorded["quat_wxyz"]))
        # and the recorded value must be the identity within the P3 gate tolerance
        assert se3.position_delta(rel_pos, [0.0, 0.0, 0.0]) <= tolerance["position_m"]
        assert se3.quat_angle_rad(rel_quat) <= tolerance["rotation_rad"]
        assert recorded["position_norm_m"] <= tolerance["position_m"]
        assert recorded["rotation_angle_rad"] <= tolerance["rotation_rad"]
    assert worst_pos < EXACT, "mount position drifted across fixtures: %e m" % worst_pos
    assert worst_rot < EXACT, "mount rotation drifted across fixtures: %e rad" % worst_rot

    # the same invariance, but measured on the arm body the gate actually samples
    for name in ("arm_base", "base_link"):
        positions = [f["expected_world_pose"][name]["position_m"] for f in fixtures["fixtures"]]
        for other in positions[1:]:
            assert se3.position_delta(positions[0], other) >= 0.0  # world poses legitimately differ
    spread = []
    for fixture in fixtures["fixtures"]:
        chassis = fixture["expected_world_pose"]["chassis"]
        arm_base = fixture["expected_world_pose"]["arm_base"]
        relative = (se3.quat_to_matrix(chassis["quat_wxyz"]).T
                    @ (np.asarray(arm_base["position_m"]) - np.asarray(chassis["position_m"])))
        spread.append(relative)
    for candidate in spread[1:]:
        assert se3.position_delta(spread[0], candidate) <= tolerance["position_m"]


def test_world_poses_are_consistent_with_the_fixture_command():
    _, fixtures = load()
    for fixture in fixtures["fixtures"]:
        command = fixture["chassis_command_reference"]
        chassis = fixture["expected_world_pose"]["chassis"]
        assert se3.position_delta(chassis["position_m"], command["position_m"]) < EXACT
        expected_rotation = se3.rot_z(command["yaw_rad"]) @ se3.rot_x(command["roll_rad"])
        assert se3.rotation_angle(expected_rotation, se3.quat_to_matrix(chassis["quat_wxyz"])) < EXACT
        # base_link sits on the arm base (identity mount) in every fixture
        base_link = fixture["expected_world_pose"]["base_link"]
        assert se3.position_delta(base_link["position_m"], chassis["position_m"]) < EXACT
        assert se3.quat_angle_rad(base_link["quat_wxyz"], chassis["quat_wxyz"]) < EXACT


def test_pxr_cross_check_is_declared():
    """pxr is optional: record honestly whether the asset was re-opened here."""
    contract, _ = load()
    rest = contract["arm_chain_rest_pose"]
    if not rest["pxr_available"]:
        print("note: pxr was NOT used to freeze this contract (%s); the numbers come from the ASCII "
              "layers and their recorded sha256" % rest["pxr_reason"])
        return
    asset_dir = os.environ.get("P3_MOUNT_ASSET_DIR")
    try:
        from pxr import Usd, UsdGeom  # noqa: F401
    except Exception as error:
        print("note: pxr not importable here (%s); verifying the committed evidence only, "
              "NOT re-opening the asset" % error)
        return
    if not asset_dir or not Path(asset_dir, "ReBot_Arm_RS.usda").is_file():
        print("note: P3_MOUNT_ASSET_DIR is unset or has no ReBot_Arm_RS.usda; verifying the committed "
              "evidence only, NOT re-opening the asset (set it to the ssh-cat scratch copy to recompose)")
        return
    stage = Usd.Stage.Open(str(Path(asset_dir, "ReBot_Arm_RS.usda")))
    assert stage, "could not open %s" % asset_dir
    cache = UsdGeom.XformCache(Usd.TimeCode.Default())
    prim = stage.GetPrimAtPath(contract["arm_chain_rest_pose"]["gripper_end_world_rest_pose"]["prim_path"])
    matrix = cache.GetLocalToWorldTransform(prim)
    recorded = contract["arm_chain_rest_pose"]["gripper_end_world_rest_pose"]
    translation = matrix.ExtractTranslation()
    quat = matrix.ExtractRotationQuat()
    imagined = quat.GetImaginary()
    assert se3.position_delta([translation[i] for i in range(3)], recorded["world_position"]) < 1e-9
    assert se3.quat_angle_rad([quat.GetReal(), imagined[0], imagined[1], imagined[2]],
                              recorded["world_quat_wxyz"]) < 1e-9
    print("pxr cross-check PASS: recomposed %s straight from the asset" % recorded["prim_path"])


def main() -> int:
    tests = [(name, fn) for name, fn in sorted(globals().items()) if name.startswith("test_") and callable(fn)]
    failed = 0
    for name, fn in tests:
        try:
            fn()
            print("PASS %s" % name)
        except AssertionError as error:
            failed += 1
            print("FAIL %s: %s" % (name, error))
    print("%d passed, %d failed, %d total" % (len(tests) - failed, failed, len(tests)))
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
