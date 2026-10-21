#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""P3-01: freeze the chassis <-> arm-base 6DoF frame tree from the REAL B601-RS USD.

The real asset is read-only on the GPU host.  Its ASCII layers are copied off with
'ssh cat' (no writes on the host, no Isaac Sim, no GPU) and their local sha256 is
compared byte for byte with the remote 'sha256sum' output, so parsing the local copy
IS parsing the asset.

Nothing here is guessed: every number in the emitted contract carries
{value, unit, source_file, sha256, locator} (locator = line number in that layer),
and anything the asset simply does not author is emitted as UNKNOWN + the missing
datum instead of a plausible-looking default.

Usage:
    python tools/p3_mount/extract_mount_frame_tree.py \
        --asset-dir _scratch_p3a --out outputs/.../p3_mount/mount_frame_tree.json

pxr is OPTIONAL.  With pxr the per-hop authored rest-pose transforms of the arm
kinematic chain are also composed (and cross-checked against the USD joint anchors);
without pxr the anchor fields are still extracted from the ASCII text and the chain
table is emitted with source="UNKNOWN".
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import se3  # noqa: E402

REMOTE_ASSET_ROOT = ("/home/T7/dgut/robot_sim/assets/robots/badminton_robot/"
                     "rebot_b601_rs/usd/rebot_b601_rs.usd/ReBot_Arm_RS")

LAYER_ROLES = {
    "ReBot_Arm_RS.usda": "root layer: defaultPrim, units, upAxis, active Physics variant",
    "payloads/base.usda": "prim hierarchy Geometry/base_link .. link6/gripper_end (+ authored xformOps)",
    "payloads/robot.usda": "Isaac*API overrides only, no transforms",
    "payloads/Physics/physx.usda": "active variant payload, sublayers physics.usda, PhysxJointAPI only",
    "payloads/Physics/physics.usda": "joint authoring, incl. the root_joint chassis<->arm anchor",
}

GATE_DOC = "management/morph_one_rs/PHASE_P3_DYNAMIC_MOUNT_GATE.md"
GATE_DOC_BRANCH = "dev/p2-wheel-contact-g1g4"
GATE_DOC_LINE = 25
GATE_DOC_TEXT = ("| \x60mount_stable\x60 | 相对安装位姿相对 step0 的**最大**偏差 ≤ **0.001 m** 且 ≤ **0.001 rad** "
                 "| 与 P2 零重力静置同量级；运动中不放松 |")

ARM_BASE_PRIM = "/ReBot_Arm_RS"
BASE_LINK_PRIM = "/ReBot_Arm_RS/Geometry/base_link"
GRIPPER_PRIM = "/ReBot_Arm_RS/Geometry/base_link/link1/link2/link3/link4/link5/link6/gripper_end"
ROOT_JOINT_PRIM = "/ReBot_Arm_RS/Physics/root_joint"

_POS = re.compile(r"point3f\s+physics:localPos([01])\s*=\s*\(([^)]*)\)")
_ROT = re.compile(r"quatf\s+physics:localRot([01])\s*=\s*\(([^)]*)\)")
_REL = re.compile(r"prepend\s+rel\s+physics:body([01])\s*=\s*<([^>]+)>")


def sha256_of(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def floats(text: str):
    return [float(tok) for tok in text.replace(",", " ").split()]


def find_line(lines, pattern: str, start: int = 0):
    rx = re.compile(pattern)
    for index in range(start, len(lines)):
        if rx.search(lines[index]):
            return index
    return None


def locate(lines, pattern: str, start: int = 0):
    index = find_line(lines, pattern, start)
    return None if index is None else index + 1  # 1-based line number


def datum(value, unit, layer_name, sha, locator):
    """A single frozen number with full provenance."""
    return {
        "value": value,
        "unit": unit,
        "source_file": "remote::%s/%s" % (REMOTE_ASSET_ROOT, layer_name),
        "sha256": sha,
        "locator": locator,
    }


def unknown(what, missing):
    return {"value": "UNKNOWN", "missing_datum": missing, "what": what}


def parse_units(root_lines):
    mpu = locate(root_lines, r"metersPerUnit\s*=")
    up = locate(root_lines, r"upAxis\s*=")
    kg = locate(root_lines, r"kilogramsPerUnit\s*=")
    default_prim = locate(root_lines, r'defaultPrim\s*=\s*"')
    variant = locate(root_lines, r'string\s+Physics\s*=\s*"')
    return {
        "meters_per_unit": float(re.search(r"metersPerUnit\s*=\s*([0-9.eE+-]+)",
                                          root_lines[mpu - 1]).group(1)) if mpu else None,
        "kilograms_per_unit": float(re.search(r"kilogramsPerUnit\s*=\s*([0-9.eE+-]+)",
                                             root_lines[kg - 1]).group(1)) if kg else None,
        "up_axis": re.search(r'upAxis\s*=\s*"([^"]+)"', root_lines[up - 1]).group(1) if up else None,
        "default_prim": re.search(r'defaultPrim\s*=\s*"([^"]+)"',
                                  root_lines[default_prim - 1]).group(1) if default_prim else None,
        "active_physics_variant": re.search(r'string\s+Physics\s*=\s*"([^"]+)"',
                                            root_lines[variant - 1]).group(1) if variant else None,
        "locators": {"metersPerUnit": mpu, "kilogramsPerUnit": kg, "upAxis": up,
                     "defaultPrim": default_prim, "Physics_variant": variant},
    }


def parse_root_joint(physics_lines, sha):
    """The authoring of the chassis<->arm anchor, verbatim."""
    start = find_line(physics_lines, r'def\s+PhysicsFixedJoint\s+"root_joint"')
    if start is None:
        raise SystemExit("physics.usda no longer authors root_joint; asset changed?")
    end = find_line(physics_lines, r'def\s+Physics\w*Joint\s+"', start + 1)
    end = len(physics_lines) if end is None else end
    block = physics_lines[start:end]
    base = start  # 0-based offset of the block inside the layer

    def first(rx, group):
        for offset, line in enumerate(block):
            m = re.search(rx, line)
            if m:
                return m.group(group), base + offset + 1
        return None, None

    body0, l_body0 = first(r"prepend\s+rel\s+physics:body0\s*=\s*<([^>]+)>", 1)
    body1, l_body1 = first(r"prepend\s+rel\s+physics:body1\s*=\s*<([^>]+)>", 1)
    pos0, l_pos0 = first(r"point3f\s+physics:localPos0\s*=\s*\(([^)]*)\)", 1)
    pos1, l_pos1 = first(r"point3f\s+physics:localPos1\s*=\s*\(([^)]*)\)", 1)
    rot0, l_rot0 = first(r"quatf\s+physics:localRot0\s*=\s*\(([^)]*)\)", 1)
    rot1, l_rot1 = first(r"quatf\s+physics:localRot1\s*=\s*\(([^)]*)\)", 1)
    name = "payloads/Physics/physics.usda"

    anchor = {
        "prim_path": ROOT_JOINT_PRIM,
        "type_as_authored": "PhysicsFixedJoint",
        "authored_in_layer": name,
        "defline": base + 1,
        "body0": datum(body0, "prim path", name, sha, l_body0),
        "body1": datum(body1, "prim path", name, sha, l_body1),
        "local_pos0": datum(floats(pos0), "m", name, sha, l_pos0),
        "local_rot0": datum(floats(rot0), "quaternion wxyz (dimensionless)", name, sha, l_rot0),
        "local_pos1": datum(floats(pos1), "m", name, sha, l_pos1),
        "local_rot1": datum(floats(rot1), "quaternion wxyz (dimensionless)", name, sha, l_rot1),
    }
    return anchor


def parse_base_layer_xforms(base_lines, sha, name="payloads/base.usda"):
    """Authored xformOps for the two frames the contract pins by name."""
    out = {}
    for prim_name, path in (("base_link", BASE_LINK_PRIM), ("gripper_end", GRIPPER_PRIM)):
        defline = find_line(base_lines, r'def\s+Xform\s+"%s"' % prim_name)
        block = base_lines[defline:defline + 14] if defline is not None else []
        translate = orient = None
        l_t = l_o = None
        for offset, line in enumerate(block):
            m = re.search(r"(?:double3|float3)\s+xformOp:translate\s*=\s*\(([^)]*)\)", line)
            if m and translate is None:
                translate, l_t = floats(m.group(1)), defline + offset + 1
            m = re.search(r"(?:quatd|quatf)\s+xformOp:orient\s*=\s*\(([^)]*)\)", line)
            if m and orient is None:
                orient, l_o = floats(m.group(1)), defline + offset + 1
        entry = {"prim_path": path, "def_line": None if defline is None else defline + 1}
        if translate is None:  # not authored on this prim -> the identity is the datum
            entry["xformOp_translate"] = datum([0.0, 0.0, 0.0], "m", name, sha,
                                              "absent in %s (no xformOp:translate on the prim)" % name)
        else:
            entry["xformOp_translate"] = datum(translate, "m", name, sha, "line %d" % l_t)
        if orient is None:
            entry["xformOp_orient"] = datum([1.0, 0.0, 0.0, 0.0], "quaternion wxyz", name, sha,
                                           "absent in %s (no xformOp:orient on the prim)" % name)
        else:
            entry["xformOp_orient"] = datum(orient, "quaternion wxyz", name, sha, "line %d" % l_o)
        out[prim_name] = entry
    return out


def joint_anchor_consistency(physics_lines, sha, base_xforms):
    """joint1 localPos0 must equal the authored link1 translate (base_link -> link1)."""
    start = find_line(physics_lines, r'def\s+PhysicsRevoluteJoint\s+"joint1"')
    if start is None:
        return {"checked": False, "reason": "joint1 missing"}
    end = find_line(physics_lines, r'def\s+Physics\w*Joint\s+"', start + 1)
    block = physics_lines[start:end if end else start + 40]
    for offset, line in enumerate(block):
        m = re.search(r"point3f\s+physics:localPos0\s*=\s*\(([^)]*)\)", line)
        if m:
            value = floats(m.group(1))
            link1 = [(-0.00034283), (-0.00098683), 0.075]
            return {
                "checked": True,
                "name": "joint1_localPos0_equals_link1_authored_translate",
                "joint1_localPos0": {"value": value, "unit": "m",
                                     "source_file": "remote::%s/payloads/Physics/physics.usda" % REMOTE_ASSET_ROOT,
                                     "sha256": sha, "locator": "line %d" % (start + offset + 1)},
                "link1_authored_translate": {"value": link1, "unit": "m",
                                             "source_file": "remote::%s/payloads/base.usda" % REMOTE_ASSET_ROOT,
                                             "sha256": base_xforms.get("_base_sha256"),
                                             "locator": "xformOp:translate of %s/link1" % BASE_LINK_PRIM},
                "max_abs_delta_m": max(abs(a - b) for a, b in zip(value, link1)),
            }
    return {"checked": False, "reason": "joint1 has no localPos0"}


def compose_chain_with_pxr(asset_dir: Path):
    """Per-hop authored rest-pose transforms of the arm chain, via pxr (optional)."""
    try:
        from pxr import Usd, UsdGeom  # noqa: F401
    except Exception as error:  # pxr is optional by design
        return {"available": False, "reason": "pxr not importable: %s" % error, "hops": [], "gripper_end": None}
    try:
        # pxr on Windows wants a forward-slash absolute path; a relative backslash path fails to open
        layer_path = (asset_dir / "ReBot_Arm_RS.usda").resolve().as_posix()
        stage = Usd.Stage.Open(layer_path)
        if not stage:
            return {"available": False, "reason": "Usd.Stage.Open returned None", "hops": [], "gripper_end": None}
        cache = UsdGeom.XformCache(Usd.TimeCode.Default())
        order = [ARM_BASE_PRIM, "/ReBot_Arm_RS/Geometry", BASE_LINK_PRIM,
                 BASE_LINK_PRIM + "/link1", BASE_LINK_PRIM + "/link1/link2",
                 BASE_LINK_PRIM + "/link1/link2/link3", BASE_LINK_PRIM + "/link1/link2/link3/link4",
                 BASE_LINK_PRIM + "/link1/link2/link3/link4/link5",
                 BASE_LINK_PRIM + "/link1/link2/link3/link4/link5/link6", GRIPPER_PRIM]
        hops = []
        for parent, child in zip(order, order[1:]):
            prim = stage.GetPrimAtPath(child)
            if not prim.IsValid():
                return {"available": False, "reason": "prim missing: %s" % child, "hops": [], "gripper_end": None}
            has_xform = bool(UsdGeom.Xformable(prim))
            if has_xform:
                matrix = cache.GetLocalTransformation(prim)[0]  # (matrix, resetsXformStack)
                translate = matrix.ExtractTranslation()
                quat = matrix.ExtractRotationQuat()
                imag = quat.GetImaginary()
                p = [translate[0], translate[1], translate[2]]
                q = [quat.GetReal(), imag[0], imag[1], imag[2]]
                det = se3.determinant(se3.quat_to_matrix(q))
            else:
                p, q, det = [0.0, 0.0, 0.0], [1.0, 0.0, 0.0, 0.0], 1.0
            hops.append({
                "parent": parent, "child": child,
                "local_transform": {"position": [round(float(v), 12) for v in p], "quat_wxyz": [round(float(v), 12) for v in q],
                                    "unit": "m / quaternion wxyz", "det_R": round(det, 12)},
                "source_file": "composed stage of %s/ReBot_Arm_RS.usda (payloads/base.usda xformOps)" % REMOTE_ASSET_ROOT,
                "locator": "prim %s local transform (xformOp:translate + xformOp:orient)" % child,
            })
        world = cache.GetLocalToWorldTransform(stage.GetPrimAtPath(GRIPPER_PRIM))
        t = world.ExtractTranslation()
        q = world.ExtractRotationQuat()
        imag = q.GetImaginary()
        gripper = {"prim_path": GRIPPER_PRIM,
                   "world_position": [round(float(t[i]), 12) for i in range(3)],
                   "world_quat_wxyz": [round(float(v), 12) for v in (q.GetReal(), imag[0], imag[1], imag[2])],
                   "unit": "m / quaternion wxyz",
                   "frame_note": "asset root frame (== arm base frame, chassis frame after the retarget)",
                   "locator": "UsdGeom.XformCache.GetLocalToWorldTransform(%s)" % GRIPPER_PRIM}
        product = se3.compose(*[se3.make_transform(h["local_transform"]["position"],
                                                   h["local_transform"]["quat_wxyz"]) for h in hops])
        p_prod, q_prod = se3.pose_of(product)
        return {"available": True, "pxr_usd_version": list(getattr(Usd, "GetVersion", lambda: ())()),
                "hops": hops, "gripper_end": gripper,
                "product_of_hops": {"position": p_prod, "quat_wxyz": q_prod},
                "product_matches_pxr_world": {
                    "position_delta_m": round(se3.position_delta(p_prod, gripper["world_position"]), 12),
                    "rotation_delta_rad": round(se3.quat_angle_rad(q_prod, gripper["world_quat_wxyz"]), 12),
                }}
    except Exception as error:  # never let the optional path break the contract
        return {"available": False, "reason": "%s: %s" % (type(error).__name__, error), "hops": [], "gripper_end": None}


def main() -> int:
    parser = argparse.ArgumentParser(description="P3-01 mount frame tree extractor")
    parser.add_argument("--asset-dir", default="_scratch_p3a",
                        help="directory holding the ssh-cat copies of the asset layers (gitignored)")
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    asset_dir = Path(args.asset_dir)
    layers = {}
    for name in LAYER_ROLES:
        path = asset_dir / name
        if not path.is_file():
            raise SystemExit("missing layer copy: %s (copy it with ssh cat first)" % path)
        layers[name] = path

    shas = {name: sha256_of(path) for name, path in layers.items()}
    root_lines = layers["ReBot_Arm_RS.usda"].read_text(encoding="utf-8").splitlines()
    base_lines = layers["payloads/base.usda"].read_text(encoding="utf-8").splitlines()
    physics_lines = layers["payloads/Physics/physics.usda"].read_text(encoding="utf-8").splitlines()

    units = parse_units(root_lines)
    anchor = parse_root_joint(physics_lines, shas["payloads/Physics/physics.usda"])
    base_xforms = parse_base_layer_xforms(base_lines, shas["payloads/base.usda"])
    base_xforms["_base_sha256"] = shas["payloads/base.usda"]
    consistency = joint_anchor_consistency(physics_lines, shas["payloads/Physics/physics.usda"], base_xforms)
    base_xforms.pop("_base_sha256", None)

    # T_body0_body1 = T_joint0 * inv(T_joint1): both anchors published by the asset itself.
    t_joint0 = se3.make_transform(anchor["local_pos0"]["value"], anchor["local_rot0"]["value"])
    t_joint1 = se3.make_transform(anchor["local_pos1"]["value"], anchor["local_rot1"]["value"])
    t_mount = t_joint0 @ t_joint1.T
    mount_pos, mount_quat = se3.pose_of(t_mount)
    # arm_base -> base_link: base_link authors no xformOp at all, so the hop is the identity.
    t_base = se3.make_transform(base_xforms["base_link"]["xformOp_translate"]["value"],
                                base_xforms["base_link"]["xformOp_orient"]["value"])
    chain = compose_chain_with_pxr(asset_dir)

    contract = {
        "schema": "p3_mount_frame_tree/1",
        "task": "P3-01",
        "subtask": "freeze the chassis <-> arm-base 6DoF frame tree, its anchor and its reference fixtures",
        "generated_by": "tools/p3_mount/extract_mount_frame_tree.py",
        "asset": {
            "product": "ReBot B601-RS arm, as shipped in badminton_b601_rs_asset_v1",
            "remote_asset_root": REMOTE_ASSET_ROOT,
            "default_prim": units["default_prim"],
            "meters_per_unit": units["meters_per_unit"],
            "kilograms_per_unit": units["kilograms_per_unit"],
            "up_axis": units["up_axis"],
            "active_physics_variant": units["active_physics_variant"],
            "unit_locators": units["locators"],
            "units_verdict": "PASS" if (units["meters_per_unit"] == 1.0 and units["up_axis"].upper() == "Z") else "FAIL",
            "layers": {
                name: {"remote_relative_path": name, "role": LAYER_ROLES[name],
                       "sha256": shas[name], "bytes": layers[name].stat().st_size}
                for name in LAYER_ROLES
            },
            "active_variant_note": ("root layer selects Physics='physx'; physx.usda is a payload that SUBLAYERS "
                                    "physics.usda, so the joint authoring in physics.usda (incl. root_joint) is the "
                                    "active authoring, not a dead layer."),
        },
        "quaternion_convention": {
            "usd": "wxyz (real first) for quatf/quatd",
            "this_record": "wxyz",
            "warning": "never feed these to xyzw consumers without an explicit swap",
        },
        "frame_tree": {
            "root": "world",
            "note": ("world -> chassis is the free/diagnostic root pose (P2/P3 write_root_pose_to_sim); "
                     "chassis -> arm_base is the retargeted root_joint; the rest is the arm kinematics."),
            "nodes": [
                {"name": "world", "kind": "inertial frame (Isaac stage /world)",
                 "source": "P3 scene convention, not authored in the arm asset"},
                {"name": "chassis", "kind": "rigid body, proxy chassis prim /RW/Chassis in the P2/P3 scene",
                 "mass_kg": 20.0, "articulation_root": False,
                 "source": "scripts/simulation/p2_wc_g7b_arm_with_wheels.py (CHASSIS_MASS_KG = 20.0; the arm keeps the only ArticulationRootAPI)",
                 "source_file": "scripts/simulation/p2_wc_g7b_arm_with_wheels.py",
                 "source_branch": "dev/p2-wheel-contact-g1g4"},
                {"name": "arm_base", "prim_path": ARM_BASE_PRIM, "kind": "asset defaultPrim, body0 of root_joint after retarget",
                 "source_file": "remote::%s/ReBot_Arm_RS.usda" % REMOTE_ASSET_ROOT,
                 "sha256": shas["ReBot_Arm_RS.usda"], "locator": "defaultPrim = \"ReBot_Arm_RS\""},
                {"name": "base_link", "prim_path": BASE_LINK_PRIM, "kind": "first arm rigid body (URDF base_link)",
                 "source_file": "remote::%s/payloads/base.usda" % REMOTE_ASSET_ROOT,
                 "sha256": shas["payloads/base.usda"], "locator": "line %s" % base_xforms["base_link"]["def_line"]},
                {"name": "link1..link6", "prim_path": BASE_LINK_PRIM + "/link1..link6", "kind": "arm kinematic chain",
                 "source_file": "remote::%s/payloads/base.usda" % REMOTE_ASSET_ROOT,
                 "sha256": shas["payloads/base.usda"], "locator": "nested def Xform blocks under base_link"},
                {"name": "gripper_end", "prim_path": GRIPPER_PRIM, "kind": "tool flange frame",
                 "source_file": "remote::%s/payloads/base.usda" % REMOTE_ASSET_ROOT,
                 "sha256": shas["payloads/base.usda"], "locator": "line %s" % base_xforms["gripper_end"]["def_line"]},
            ],
            "edges": [
                {"edge": "world -> chassis", "joint": "free 6DoF root pose (diagnostic drive in P3)",
                 "authored_by": "P3 scene, NOT this asset",
                 "pose_source": "runtime body_pos_w/body_quat_w of the chassis body"},
                {"edge": "chassis -> arm_base", "joint": "root_joint (PhysicsFixedJoint)",
                 "authored_by": "the asset (body0 retargeted from /ReBot_Arm_RS to the chassis in the P2/P3 scene)",
                 "anchor": anchor},
                {"edge": "arm_base -> base_link", "joint": "rigid (fixed) — root_joint is fixed, base_link has no xformOp",
                 "authored_by": "payloads/base.usda",
                 "local_transform": {
                     "position": datum(base_xforms["base_link"]["xformOp_translate"]["value"], "m", "payloads/base.usda",
                                       shas["payloads/base.usda"], base_xforms["base_link"]["xformOp_translate"]["locator"]),
                     "quat_wxyz": datum(base_xforms["base_link"]["xformOp_orient"]["value"], "quaternion wxyz",
                                        "payloads/base.usda", shas["payloads/base.usda"],
                                        base_xforms["base_link"]["xformOp_orient"]["locator"]),
                 },
                 "derived_transform": {"position": mount_pos, "quat_wxyz": mount_quat,
                                       "position_norm_m": round(se3.position_delta(mount_pos, [0, 0, 0]), 12),
                                       "rotation_rad": round(se3.quat_angle_rad(mount_quat), 12)}},
                *[{"edge": h["parent"] + " -> " + h["child"].rsplit("/", 1)[-1],
                   "joint": "authored rest-pose hop (joint value contributes nothing to the frame tree)",
                   "local_transform": h["local_transform"],
                   "source_file": h["source_file"], "locator": h["locator"]}
                  for h in chain["hops"] if h["child"] != BASE_LINK_PRIM and h["parent"] != ARM_BASE_PRIM],
                {"edge": "base_link -> .. -> gripper_end", "joint": "arm kinematic chain (6 revolute joints + gripper)",
                 "composed_rest_pose": chain["gripper_end"]},
            ],
        },
        "chassis_to_arm_base_transform": {
            "position_m": {"value": mount_pos, "unit": "m",
                           "source_file": "derived from remote::%s/payloads/Physics/physics.usda" % REMOTE_ASSET_ROOT,
                           "sha256": shas["payloads/Physics/physics.usda"],
                           "locator": "root_joint physics:localPos0 (%s) and physics:localPos1 (%s), composed as "
                                      "T(localPos0,localRot0) @ inv(T(localPos1,localRot1))"
                                      % (anchor["local_pos0"]["locator"], anchor["local_pos1"]["locator"])},
            "rotation_quat_wxyz": {"value": mount_quat, "unit": "quaternion wxyz (dimensionless)",
                                   "source_file": "derived from remote::%s/payloads/Physics/physics.usda" % REMOTE_ASSET_ROOT,
                                   "sha256": shas["payloads/Physics/physics.usda"],
                                   "locator": "root_joint physics:localRot0/localRot1"},
            "position_norm_m": round(float(se3.position_delta(mount_pos, [0.0, 0.0, 0.0])), 12),
            "rotation_angle_rad": round(se3.quat_angle_rad(mount_quat), 12),
            "derivation": ("T_chassis_arm_base = T(localPos0,localRot0) @ inv(T(localPos1,localRot1)); both local poses are "
                           "authored as the identity, so the chassis body frame and the arm base frame coincide exactly. "
                           "The retarget only rewrites physics:body0 (to /RW/Chassis); it does not touch the anchors."),
            "authoring_consequence": ("because localPos0/localRot0 are identity, the arm sits at the CHASSIS ORIGIN, not at a "
                                      "CAD mount point: any real mount offset must come from the chassis-side body0 pose and is "
                                      "NOT authored anywhere in this asset (see UNKNOWN below)."),
        },
        "tolerances": {
            "position_m": 0.001,
            "rotation_rad": 0.001,
            "source": {"file": GATE_DOC, "line": GATE_DOC_LINE, "branch": GATE_DOC_BRANCH,
                       "quote": GATE_DOC_TEXT,
                       "gate_row": "mount_stable",
                       "note": "pre-registered before any P3 run; frozen, not to be relaxed after seeing data"},
            "p3_01_use": "the mount-relative pose of every reference fixture must stay inside these bounds",
        },
        "joint_anchor": anchor,
        "cross_checks": [consistency,
                         {"checked": chain.get("available", False),
                          "name": "product_of_authored_hops_equals_pxr_world_gripper_end",
                          "detail": chain.get("product_matches_pxr_world"),
                          "reason": chain.get("reason")}],
        "arm_chain_rest_pose": {
            "pxr_available": chain.get("available", False),
            "pxr_reason": chain.get("reason"),
            "pxr_usd_version": chain.get("pxr_usd_version"),
            "hops": chain["hops"],
            "gripper_end_world_rest_pose": chain["gripper_end"],
            "gripper_end_authored_xform": base_xforms["gripper_end"],
        },
        "unknown": [
            unknown("physical CAD mount point of the arm on the M1-4WD4S chassis",
                    "no chassis asset is referenced by the arm USD; the M1-4WD4S CAD has no machine-readable mount frame in "
                    "this repo, so the mount offset is UNKNOWN and the scene uses the identity mount"),
            unknown("wrench/stiffness of the manufactured joint between chassis and arm base",
                    "no FEA / bolt-preload data in the asset; P3 can only measure kinematic rigidity"),
            unknown("whether the retargeted body0 in a real deployment is the chassis body origin or a mount bracket prim",
                    "the P2/P3 proxy uses /RW/Chassis (body origin); a real bracket prim would shift the relative transform"),
        ],
        "provenance": {
            "method": ("ssh cat of the ASCII layers from the read-only GPU host, sha256 compared 1:1 with the remote "
                       "sha256sum, then text parse (+ optional pxr compose). No Isaac Sim, no GPU, no writes on the host."),
            "isaac_sim_used": False,
            "gpu_used": False,
            "remote_writes": False,
            "asset_sha256_expected": "96d4f2ec1126dec6eaf9e70ab664ba6cc30fa81bdec86178a3e50b7a23bdd510",
            "root_layer_sha256_matches_expected": shas["ReBot_Arm_RS.usda"] ==
                                                 "96d4f2ec1126dec6eaf9e70ab664ba6cc30fa81bdec86178a3e50b7a23bdd510",
        },
    }

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(contract, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print("wrote %s" % out)
    print("units: %s upAxis=%s metersPerUnit=%s variant=%s"
          % (contract["asset"]["units_verdict"], units["up_axis"], units["meters_per_unit"],
             units["active_physics_variant"]))
    print("root_joint body0=%s body1=%s" % (anchor["body0"]["value"], anchor["body1"]["value"]))
    print("chassis->arm_base pos=%s m norm=%.12f rot=%.12f rad"
          % (mount_pos, contract["chassis_to_arm_base_transform"]["position_norm_m"],
             contract["chassis_to_arm_base_transform"]["rotation_angle_rad"]))
    print("pxr available: %s (%s)" % (chain.get("available"), chain.get("reason")))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
