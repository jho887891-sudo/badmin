#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""P3-01 reference fixtures: three NON-TRIVIAL chassis poses + the expected frames.

The point of the fixtures is that a "mount is stable" claim computed only at the
identity proves nothing (every rotation is the identity and every relative pose is
trivially invariant).  So fixture 1 is the control and fixtures 2/3 carry a real
non-zero chassis rotation (pure yaw 0.35 rad, combined yaw 0.30 + roll 0.10 rad) plus
a non-zero translation: the chassis world pose changes a lot while the
chassis -> arm_base relative pose must NOT move at all.

Everything is derived from the frozen contract (mount_frame_tree.json), so this file
is reproducible on a machine that has no USD asset at all.

    python tools/p3_mount/gen_mount_fixtures.py \
        --contract outputs/.../p3_mount/mount_frame_tree.json \
        --out outputs/.../p3_mount/mount_fixtures.json
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import se3  # noqa: E402

FIXTURES = [
    {
        "id": "F1_identity",
        "label": "control: chassis at the origin, no rotation",
        "chassis_position_m": [0.0, 0.0, 0.0],
        "yaw_rad": 0.0,
        "roll_rad": 0.0,
        "non_trivial": False,
        "why": "reference baseline; on its own it proves nothing about the mount",
    },
    {
        "id": "F2_yaw_035",
        "label": "pure yaw 0.35 rad (+0.30 m x)",
        "chassis_position_m": [0.30, 0.0, 0.0],
        "yaw_rad": 0.35,
        "roll_rad": 0.0,
        "non_trivial": True,
        "why": "P3-02 yaw profile amplitude; a pure yaw must leave the mount offset untouched",
    },
    {
        "id": "F3_yaw_030_roll_010",
        "label": "combined yaw 0.30 rad + roll 0.10 rad (-0.15 m x, +0.20 m y, +0.10 m z)",
        "chassis_position_m": [-0.15, 0.20, 0.10],
        "yaw_rad": 0.30,
        "roll_rad": 0.10,
        "non_trivial": True,
        "why": "combined yaw+roll couples the mount axes; a mount error hidden by pure yaw shows up here",
    },
]


def main() -> int:
    parser = argparse.ArgumentParser(description="P3-01 reference fixtures")
    parser.add_argument("--contract", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    contract = json.loads(Path(args.contract).read_text(encoding="utf-8"))
    tolerance = contract["tolerances"]
    mount = contract["chassis_to_arm_base_transform"]
    t_mount = se3.make_transform(mount["position_m"]["value"], mount["rotation_quat_wxyz"]["value"])

    arm_base_to_base_link = contract["frame_tree"]["edges"][2]["local_transform"]
    t_base_link = se3.make_transform(arm_base_to_base_link["position"]["value"],
                                     arm_base_to_base_link["quat_wxyz"]["value"])

    hops = contract["arm_chain_rest_pose"]["hops"]
    if not hops:
        raise SystemExit("contract has no arm chain hops (pxr was unavailable); cannot build fixtures")
    # The hop list already runs arm_base -> Geometry -> base_link -> link1.. -> gripper_end,
    # so its product is exactly the frame the contract recorded from pxr.
    t_chain = se3.compose(*[se3.make_transform(h["local_transform"]["position"], h["local_transform"]["quat_wxyz"])
                            for h in hops])
    chain_pos, chain_quat = se3.pose_of(t_chain)

    recorded = contract["arm_chain_rest_pose"]["gripper_end_world_rest_pose"]
    chain_self_check = {
        "recorded_by_pxr": {"position": recorded["world_position"], "quat_wxyz": recorded["world_quat_wxyz"]},
        "recomposed_here": {"position": chain_pos, "quat_wxyz": chain_quat},
        "position_delta_m": round(se3.position_delta(chain_pos, recorded["world_position"]), 12),
        "rotation_delta_rad": round(se3.quat_angle_rad(chain_quat, recorded["world_quat_wxyz"]), 12),
    }

    fixtures = []
    for spec in FIXTURES:
        rotation = se3.rot_z(spec["yaw_rad"]) @ se3.rot_x(spec["roll_rad"])
        t_chassis = se3.transform_from_matrix(rotation, spec["chassis_position_m"])
        t_arm_base = se3.compose(t_chassis, t_mount)
        t_base = se3.compose(t_arm_base, t_base_link)
        t_gripper = se3.compose(t_arm_base, t_chain)  # arm_base -> .. -> gripper_end

        chassis_pos, chassis_quat = se3.pose_of(t_chassis)
        arm_base_pos, arm_base_quat = se3.pose_of(t_arm_base)
        base_pos, base_quat = se3.pose_of(t_base)
        gripper_pos, gripper_quat = se3.pose_of(t_gripper)

        # the quantity the P3 mount_stable gate measures: chassis -> arm_base, relative
        t_relative = np.linalg.inv(t_chassis) @ t_arm_base
        rel_pos, rel_quat = se3.pose_of(t_relative)

        fixtures.append({
            "id": spec["id"],
            "label": spec["label"],
            "why": spec["why"],
            "non_trivial": spec["non_trivial"],
            "chassis_command_reference": {"position_m": spec["chassis_position_m"],
                                          "yaw_rad": spec["yaw_rad"], "roll_rad": spec["roll_rad"],
                                          "rotation_order": "R = Rz(yaw) @ Rx(roll)"},
            "expected_world_pose": {
                "chassis": {"position_m": chassis_pos, "quat_wxyz": chassis_quat,
                            "rotation_angle_rad": round(se3.rotation_angle(rotation), 12)},
                "arm_base": {"position_m": arm_base_pos, "quat_wxyz": arm_base_quat,
                             "rotation_angle_rad": round(se3.quat_angle_rad(arm_base_quat), 12)},
                "base_link": {"position_m": base_pos, "quat_wxyz": base_quat,
                              "rotation_angle_rad": round(se3.quat_angle_rad(base_quat), 12)},
                "gripper_end": {"position_m": gripper_pos, "quat_wxyz": gripper_quat,
                                "rotation_angle_rad": round(se3.quat_angle_rad(gripper_quat), 12)},
            },
            "expected_mount_relative": {
                "frame_pair": "chassis -> arm_base",
                "position_m": rel_pos,
                "quat_wxyz": rel_quat,
                "position_norm_m": round(float(np.linalg.norm(rel_pos)), 12),
                "rotation_angle_rad": round(se3.quat_angle_rad(rel_quat), 12),
                "invariance": "must equal the identity within tolerance in EVERY fixture",
            },
            "determinants": {
                "chassis_R": round(se3.determinant(rotation), 12),
                "mount_rotation_R": round(se3.determinant(se3.quat_to_matrix(rel_quat)), 12),
                "chain_R": round(se3.determinant(t_chain[:3, :3]), 12),
            },
        })

    world_angles = [f["expected_world_pose"]["chassis"]["rotation_angle_rad"] for f in fixtures]
    out = {
        "schema": "p3_mount_fixtures/1",
        "task": "P3-01",
        "generated_by": "tools/p3_mount/gen_mount_fixtures.py",
        "derived_from_contract": str(args.contract).replace("\\", "/"),
        "asset_sha256_root_layer": contract["asset"]["layers"]["ReBot_Arm_RS.usda"]["sha256"],
        "quaternion_convention": "wxyz (real first)",
        "tolerance_used": {"position_m": tolerance["position_m"], "rotation_rad": tolerance["rotation_rad"],
                           "source": tolerance["source"]},
        "chain_product": {
            "frame": "arm_base -> gripper_end (authored rest pose, joints at their USD rest transforms)",
            "position_m": chain_pos,
            "quat_wxyz": chain_quat,
            "self_check_against_pxr": chain_self_check,
        },
        "non_triviality": {
            "rule": "fixtures 2 and 3 must have |rotation angle| > 0.1 rad on the chassis",
            "chassis_rotation_angles_rad": world_angles,
            "non_trivial_fixture_count": sum(1 for f in fixtures if f["non_trivial"]),
            "note": ("F1_identity is the deliberate control with angle 0; the requirement 'non-trivial' "
                     "applies to the yaw/roll references, which is exactly what F2/F3 carry."),
        },
        "fixtures": fixtures,
    }

    path = Path(args.out)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(out, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print("wrote %s" % path)
    print("chain self-check: dpos=%.3e m drot=%.3e rad"
          % (chain_self_check["position_delta_m"], chain_self_check["rotation_delta_rad"]))
    for fixture in fixtures:
        rel = fixture["expected_mount_relative"]
        print("%-24s chassis_rot=%.6f rad  mount_relative pos=%.6f m rot=%.6f rad"
              % (fixture["id"], fixture["expected_world_pose"]["chassis"]["rotation_angle_rad"],
                 rel["position_norm_m"], rel["rotation_angle_rad"]))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
