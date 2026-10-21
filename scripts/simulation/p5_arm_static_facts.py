# -*- coding: utf-8 -*-
"""Statically export the B601-RS arm joint facts from the converted USD (pxr only, no simulator).

WHY
    The P5 static consistency check needs the joints exactly as the asset authors them, independent of
    any simulator run. This reads the composed stage (top-level .usda + physx variant payload) with pxr
    only: no Isaac Sim, no torch, no cuda.

WHAT IS EXPORTED (per joint, verbatim from the layer)
    path, type, body0/body1 targets, physics:axis, physics:lowerLimit / physics:upperLimit
    (UsdPhysics authors REVOLUTE limits in DEGREES; both degrees and radians are recorded),
    every authored drive:* attribute (stiffness / damping / maxForce / targetPosition / ...),
    plus the rigid-body list and the prim carrying ArticulationRootAPI.

READ-ONLY
    The stage is opened for reading only and the asset sha256 is computed before and after the walk,
    so a report can show that the file was not modified.

USAGE
    /home/T7/ojh/robot_sim/env_isaaclab/bin/python scripts/simulation/p5_arm_static_facts.py \
        --usda <...>/Rebot_Arm_RS.usda --out outputs/.../arm_joint_facts.json
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import sys

from pxr import Sdf, Usd, UsdPhysics  # noqa: F401

JOINT_TYPE_NAMES = (
    "PhysicsFixedJoint",
    "PhysicsRevoluteJoint",
    "PhysicsPrismaticJoint",
    "PhysicsSphericalJoint",
    "PhysicsDistanceJoint",
)

DEG_PER_RAD = 180.0 / math.pi


def sha256_of(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def targets(prim, name: str):
    rel = prim.GetRelationship(name)
    if not rel:
        return None
    out = [str(p) for p in rel.GetTargets()]
    return out if out else None


def authored_attrs(prim, prefixes):
    """Every authored attribute on prim whose name starts with one of prefixes, as plain values."""
    out = {}
    for attr in prim.GetAttributes():
        name = attr.GetName()
        if not any(name.startswith(p) for p in prefixes):
            continue
        val = attr.Get()
        if val is None and attr.GetTypeName() == Sdf.ValueTypeNames.Double:
            val = None
        out[name] = _plain(val)
    return out


def _plain(val):
    if isinstance(val, (bool, int, str)):
        return val
    if isinstance(val, float):
        # Strict JSON has no Infinity/NaN; USD authors physics:breakForce = inf to mean "never break".
        if not math.isfinite(val):
            return "Infinity" if val > 0 else ("-Infinity" if val < 0 else "NaN")
        return val
    if isinstance(val, (list, tuple)):
        return [_plain(v) for v in val]
    if hasattr(val, "__len__") and not isinstance(val, str):
        try:
            return [_plain(v) for v in val]
        except TypeError:
            pass
    return str(val)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--usda", required=True, help="top-level ReBot_Arm_RS.usda (read only)")
    ap.add_argument("--out", required=True, help="where to write the JSON facts")
    ap.add_argument("--rigid-body-note", default="", help="free-form note stored in the payload")
    args = ap.parse_args(argv)

    usda = os.path.abspath(args.usda)
    if not os.path.isfile(usda):
        print("MISSING: " + usda, file=sys.stderr)
        return 2

    sha_before = sha256_of(usda)
    stage = Usd.Stage.Open(usda)
    if not stage:
        print("OPEN_FAILED: " + usda, file=sys.stderr)
        return 3

    root_layer = stage.GetRootLayer()
    variant_selection = {}
    for prim in stage.Traverse():
        vs = prim.GetVariantSets()
        for name in vs.GetNames():
            variant_selection[str(prim.GetPath()) + "|" + name] = vs.GetVariantSet(name).GetVariantSelection()

    joints, rigid_bodies, articulation_roots = [], [], []
    for prim in stage.Traverse():
        path = str(prim.GetPath())
        if prim.HasAPI(UsdPhysics.ArticulationRootAPI):
            articulation_roots.append(path)
        if prim.HasAPI(UsdPhysics.RigidBodyAPI):
            rb = {"path": path, "typeName": prim.GetTypeName(), "physics": {}}
            mass_api = UsdPhysics.MassAPI(prim)
            if mass_api and mass_api.GetMassAttr():
                rb["mass_kg"] = _plain(mass_api.GetMassAttr().Get())
            rb["physics"] = authored_attrs(prim, ("physics:",))
            rigid_bodies.append(rb)
        type_name = prim.GetTypeName()
        if type_name not in JOINT_TYPE_NAMES:
            continue
        joint = {
            "path": path,
            "name": prim.GetName(),
            "typeName": type_name,
            "body0": targets(prim, "physics:body0"),
            "body1": targets(prim, "physics:body1"),
            "drives": {},
        }
        joint["authored_attrs"] = authored_attrs(prim, ("physics:", "drive:", "limit:"))
        axis_attr = prim.GetAttribute("physics:axis")
        joint["axis"] = _plain(axis_attr.Get()) if axis_attr else None
        # The converted asset applies the drive as PhysicsDriveAPI:angular / PhysicsDriveAPI:linear
        # (never one per axis), so the instance name is the drive kind, not X/Y/Z.
        for dname in ("angular", "linear"):
            api = UsdPhysics.DriveAPI.Get(prim, dname)
            if not api:
                continue
            ent = {}
            for label, attr in (
                ("stiffness", api.GetStiffnessAttr()),
                ("damping", api.GetDampingAttr()),
                ("maxForce", api.GetMaxForceAttr()),
            ):
                ent[label] = _plain(attr.Get()) if attr else None
            for extra in ("targetPosition", "targetVelocity", "type"):
                a = prim.GetAttribute("drive:" + dname + ":physics:" + extra)
                if a:
                    ent[extra] = _plain(a.Get())
            joint["drives"][dname] = ent
        lower = prim.GetAttribute("physics:lowerLimit")
        upper = prim.GetAttribute("physics:upperLimit")
        lo = _plain(lower.Get()) if lower else None
        hi = _plain(upper.Get()) if upper else None
        joint["limits"] = {"lower": lo, "upper": hi}
        if type_name == "PhysicsRevoluteJoint":
            joint["limits"]["unit"] = "degree"
            joint["limits"]["lower_deg"] = lo
            joint["limits"]["upper_deg"] = hi
            joint["limits"]["lower_rad"] = (lo / DEG_PER_RAD) if isinstance(lo, (int, float)) else None
            joint["limits"]["upper_rad"] = (hi / DEG_PER_RAD) if isinstance(hi, (int, float)) else None
        elif type_name == "PhysicsPrismaticJoint":
            joint["limits"]["unit"] = "meter"
            joint["limits"]["lower_m"] = lo
            joint["limits"]["upper_m"] = hi
        joints.append(joint)

    traversal_order = [j["name"] for j in joints]
    joints.sort(key=lambda j: j["path"])
    rigid_bodies.sort(key=lambda r: r["path"])

    counts = {}
    for j in joints:
        counts[j["typeName"]] = counts.get(j["typeName"], 0) + 1

    payload = {
        "schema": "p5.arm_joint_facts.v1",
        "generated_by": "scripts/simulation/p5_arm_static_facts.py",
        "tool": {
            "python": sys.version.split()[0],
            "pxr_dir": os.path.dirname(__import__("pxr").__file__),
            "usd_version": ".".join(str(v) for v in Usd.GetVersion()),
            "gpu_used": False,
            "note": "pxr static read only: no Isaac Sim, no torch, no cuda",
            "non_finite_encoding": "non-finite floats are stored as the strings 'Infinity'/'-Infinity'/'NaN' so that the file stays strict JSON",
        },
        "asset": {
            "usda_path": usda,
            "usda_sha256_before": sha_before,
            "usda_sha256_after": None,  # filled in below
            "root_layer_identifier": root_layer.identifier,
            "defaultPrim": stage.GetDefaultPrim().GetName() if stage.GetDefaultPrim() else None,
            "upAxis": str(stage.GetMetadata("upAxis")),
            "metersPerUnit": stage.GetMetadata("metersPerUnit"),
            "kilogramsPerUnit": stage.GetMetadata("kilogramsPerUnit"),
            "variant_selection": variant_selection,
            "note": args.rigid_body_note,
        },
        "counts": {
            "joints": len(joints),
            "by_type": counts,
            "rigid_bodies": len(rigid_bodies),
            "articulation_roots": len(articulation_roots),
        },
        "articulation_root_prims": articulation_roots,
        # Usd.Traverse() order is NOT the PhysX dof order: it is the stage's child order, which is
        # alphabetical here (gripper_joint1..root_joint). Recorded so no reader mistakes it for one.
        "stage_traversal_order": traversal_order,
        "joints": joints,
        "rigid_bodies": rigid_bodies,
    }
    payload["asset"]["usda_sha256_after"] = sha256_of(usda)
    payload["asset"]["unmodified"] = payload["asset"]["usda_sha256_before"] == payload["asset"]["usda_sha256_after"]

    out = os.path.abspath(args.out)
    os.makedirs(os.path.dirname(out), exist_ok=True)
    with open(out, "w", encoding="utf-8", newline="\n") as fh:
        json.dump(payload, fh, indent=2, sort_keys=False, ensure_ascii=False)
        fh.write("\n")

    print("joints=%d rigid_bodies=%d articulation_roots=%d unmodified=%s"
          % (len(joints), len(rigid_bodies), len(articulation_roots), payload["asset"]["unmodified"]))
    for j in joints:
        print("  %-52s %-24s axis=%s limits=%s" % (j["path"], j["typeName"], j["axis"], j["limits"]))
    print("stage traversal order: " + ", ".join(traversal_order))
    print("wrote " + out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
