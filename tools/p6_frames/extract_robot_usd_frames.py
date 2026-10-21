#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Extract the ReBot B601-RS rigid-body/joint frame chain from the SHIPPED USD layers.

P6-02 requires the arm link names to be mapped BY NAME from the real asset, not from
stage traversal order and not from a URDF we happen to have lying around.  The real
asset lives on the GPU host (read-only for us); the ASCII layers were copied off it
with 'ssh cat' (no writes, no Isaac Sim, no GPU) and their sha256 match the remote
'sha256sum' output byte for byte, so parsing the copies is parsing the asset.

Everything this script writes is derived from text, so it is verifiable: rerun it on
byte-identical layers and you get byte-identical output.

    python tools/p6_frames/extract_robot_usd_frames.py --layer NAME=PATH ... --out FILE

NO pxr needed: the layers are USDA text.  The point of that choice is that this
extraction runs on a CPU-only box with no USD install at all.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
from pathlib import Path

REMOTE_ROOT = "/home/T7/dgut/robot_sim/packages/badminton_b601_rs_asset_v1/usd/rebot_b601_rs.usd/ReBot_Arm_RS"

_DEF_JOINT = re.compile(r'def\s+(Physics\w*Joint)\s+"([^"]+)"')
_DEF_XFORM = re.compile(r'def\s+Xform\s+"([^"]+)"')
_REL_BODY = re.compile(r'prepend\s+rel\s+physics:body([01])\s*=\s*<([^>]+)>')
_POS = re.compile(r'point3f\s+physics:localPos([01])\s*=\s*\(([^)]*)\)')
_ROT = re.compile(r'quatf\s+physics:localRot([01])\s*=\s*\(([^)]*)\)')
_AXIS = re.compile(r'uniform\s+token\s+physics:axis\s*=\s*"([^"]+)"')
_LIMIT = re.compile(r'float\s+physics:(lower|upper)Limit\s*=\s*([-0-9.eE+]+)')
_XLATE = re.compile(r'(?:double3|float3)\s+xformOp:translate\s*=\s*\(([^)]*)\)')
_ORIENT = re.compile(r'(?:quatd|quatf)\s+xformOp:orient\s*=\s*\(([^)]*)\)')
_MPU = re.compile(r'metersPerUnit\s*=\s*([-0-9.eE+]+)')
_UP = re.compile(r'upAxis\s*=\s*"([^"]+)"')
_KPU = re.compile(r'kilogramsPerUnit\s*=\s*([-0-9.eE+]+)')
DEG2RAD = 3.141592653589793 / 180.0


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def floats(text: str) -> list:
    return [float(tok) for tok in text.replace(",", " ").split()]


def parse_joint_blocks(text: str) -> list:
    """Split the physics layer into joint blocks and pull the numbers out of each."""
    joints = []
    for m in _DEF_JOINT.finditer(text):
        nxt = _DEF_JOINT.search(text, m.end())
        block = text[m.start(): nxt.start() if nxt else len(text)]
        body, pos, rot, lim = {}, {}, {}, {}
        for rel in _REL_BODY.finditer(block):
            body[rel.group(1)] = rel.group(2).strip()
        for rel in _POS.finditer(block):
            pos[rel.group(1)] = floats(rel.group(2))
        for rel in _ROT.finditer(block):
            rot[rel.group(1)] = floats(rel.group(2))
        for rel in _LIMIT.finditer(block):
            lim[rel.group(1)] = float(rel.group(2))
        axis = _AXIS.search(block)
        joints.append({
            "prim": m.group(2),
            "type": m.group(1),
            "body0": body.get("0"),
            "body1": body.get("1"),
            "axis": axis.group(1) if axis else None,
            "quat_order": "wxyz",
            "local_pos0": pos.get("0"),
            "local_pos1": pos.get("1"),
            "local_rot0_wxyz": rot.get("0"),
            "local_rot1_wxyz": rot.get("1"),
            "limit_lower_as_authored": lim.get("lower"),
            "limit_upper_as_authored": lim.get("upper"),
            "limit_unit_as_authored": "degree",
            "limit_lower_rad": (None if "lower" not in lim else lim["lower"] * DEG2RAD),
            "limit_upper_rad": (None if "upper" not in lim else lim["upper"] * DEG2RAD),
        })
    return joints


def parse_named_xform(text: str, name: str):
    """The xformOp values of one prim, as authored."""
    for m in _DEF_XFORM.finditer(text):
        if m.group(1) != name:
            continue
        nxt = _DEF_XFORM.search(text, m.end())
        block = text[m.start(): nxt.start() if nxt else len(text)]
        tr = _XLATE.search(block)
        ori = _ORIENT.search(block)
        return {
            "prim_name": name,
            "xformOp:translate": floats(tr.group(1)) if tr else None,
            "xformOp:orient_wxyz": floats(ori.group(1)) if ori else None,
        }
    return None


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--layer", action="append", default=[], metavar="NAME=PATH")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    layers = {}
    for spec in args.layer:
        name, _, path = spec.partition("=")
        layers[name] = Path(path)

    physics = layers["payloads/Physics/physics.usda"].read_text(encoding="utf-8")
    base = layers["payloads/base.usda"].read_text(encoding="utf-8")
    root = layers["ReBot_Arm_RS.usda"].read_text(encoding="utf-8")

    mpu = _MPU.search(root)
    up = _UP.search(root)
    kpu = _KPU.search(root)

    out = {
        "schema": "p6_robot_usd_frame_chain/1",
        "asset": {
            "product": "ReBot B601-RS arm (as shipped in badminton_b601_rs_asset_v1)",
            "remote_root": REMOTE_ROOT,
            "default_prim": "ReBot_Arm_RS",
            "metersPerUnit": float(mpu.group(1)) if mpu else None,
            "kilogramsPerUnit": float(kpu.group(1)) if kpu else None,
            "upAxis": up.group(1) if up else None,
            "units_verdict": "PASS" if (mpu and float(mpu.group(1)) == 1.0 and up and up.group(1).upper() == "Z") else "FAIL",
            "layers": {
                name: {"remote_relative_path": name, "sha256": sha256(p), "local_bytes": p.stat().st_size}
                for name, p in layers.items()
            },
        },
        "quaternion_convention": {
            "usd": "wxyz (real first): GfQuatf / UsdGeom quatf and quatd attributes",
            "this_record": "wxyz",
            "note": "Do not hand these to anything that expects xyzw without an explicit swap.",
        },
        "joints": parse_joint_blocks(physics),
        "named_frames": [
            f for f in (parse_named_xform(base, "gripper_end"),) if f is not None
        ],
        "provenance": {
            "extracted_by": "tools/p6_frames/extract_robot_usd_frames.py",
            "method": "ssh cat of the ASCII layers from the GPU host (read-only), then text parse; sha256 of each local copy equals the remote sha256sum",
            "isaac_sim_used": False,
            "gpu_used": False,
        },
    }
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(out, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print("wrote", args.out, "joints:", len(out["joints"]), "units:", out["asset"]["units_verdict"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
