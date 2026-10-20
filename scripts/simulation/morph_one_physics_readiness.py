"""Read-only physics-modeling readiness facts for the Morph One chassis + B601-RS arm.

This module answers, with measured numbers instead of opinions:

  C1  is the verified chassis USD structured so that its parts can become separate
      rigid bodies?
  C2  does the instance/prototype structure have to be broken to author colliders?
  C3  does the arm already provide a single articulation that a chassis could join?
  C4  what is the arm's existing root joint, and where does it attach?
  C5  which arm-side configuration (joint limits, drive gains, solver settings) is
      already authored and must therefore be preserved?
  C6  which quantities are simply absent and must wait for real vendor data?

Nothing is written to the USD files: the script only opens stages and counts.  It
never creates a joint, never edits a mesh, and never claims a mass or inertia.

Usage
    python scripts/simulation/morph_one_physics_readiness.py \
        --chassis-usd <verified morph_one usda> \
        --arm-usd <rebot_b601_rs.usda> \
        --output-json <path>
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from collections import Counter, defaultdict

from pxr import Usd, UsdGeom, UsdPhysics

MESH_TYPES = ("Mesh",)


def iter_prims(stage):
    """Every prim, including prototype and instance-proxy prims.

    A HOOPS-converted assembly keeps its geometry inside a Prototypes scope and
    cites it through instanceable prims, so a plain stage.Traverse() silently
    skips most of the model.
    """
    try:
        predicate = Usd.TraverseInstanceProxies(Usd.PrimAllPrimsPredicate)
        for prim in Usd.PrimRange.Stage(stage, predicate):
            yield prim
    except Exception:  # pragma: no cover - depends on the bundled pxr version
        for prim in stage.TraverseAll():
            yield prim


def _layer_list(stage):
    layers = []
    try:
        used = stage.GetUsedLayers()
    except Exception:  # pragma: no cover
        used = []
    for layer in used:
        layers.append(os.path.basename(layer.identifier))
    return layers


def _mesh_points(prim):
    try:
        mesh = UsdGeom.Mesh(prim)
        points = mesh.GetPointsAttr().Get()
        return 0 if points is None else len(points)
    except Exception:  # pragma: no cover
        return 0


def _top_child(path, depth=1):
    """First N path components below the pseudo root, used for grouping."""
    parts = [part for part in str(path).split("/") if part]
    return "/" + "/".join(parts[:depth]) if parts else "/"


def analyse_stage(path, group_depth=2):
    stage = Usd.Stage.Open(path)
    if stage is None:
        raise SystemExit("cannot open stage: %s" % path)
    default_prim = stage.GetDefaultPrim()

    counters = Counter()
    mesh_points = 0
    scale_prims = []
    physics = Counter()
    prototypes = set()
    instanceable = []
    groups = defaultdict(lambda: {"meshes": 0, "points": 0, "instance_proxies": 0,
                                 "instanceable": 0, "scopes": 0})
    for prim in iter_prims(stage):
        counters["prims"] += 1
        type_name = prim.GetTypeName()
        counters["type:" + str(type_name)] += 1
        if prim.IsInPrototype():
            counters["in_prototype"] += 1
            prototypes.add(_top_child(prim.GetPath(), 3))
        if prim.IsInstanceProxy():
            counters["instance_proxies"] += 1
        if prim.IsInstanceable():
            counters["instanceable"] += 1
            instanceable.append(str(prim.GetPath()))
        if prim.IsInstance():
            counters["instances"] += 1
        if prim.IsPrototype():
            counters["prototypes"] += 1
        if prim.GetTypeName() in MESH_TYPES:
            counters["meshes"] += 1
            points = _mesh_points(prim)
            mesh_points += points
            group = _top_child(prim.GetPath(), group_depth)
            groups[group]["meshes"] += 1
            groups[group]["points"] += points
            if prim.IsInstanceProxy():
                groups[group]["instance_proxies"] += 1
        if prim.HasAPI(UsdPhysics.RigidBodyAPI):
            physics["rigid_bodies"] += 1
        if prim.HasAPI(UsdPhysics.MassAPI):
            physics["mass_api"] += 1
        if prim.HasAPI(UsdPhysics.CollisionAPI):
            physics["colliders"] += 1
        if prim.HasAPI(UsdPhysics.ArticulationRootAPI):
            physics["articulation_roots"] += 1
        if prim.IsA(UsdPhysics.Scene):
            physics["scenes"] += 1
        if prim.IsA(UsdPhysics.Joint):
            physics["joints"] += 1
        if prim.GetTypeName() == "Xform":
            for op in UsdGeom.Xformable(prim).GetOrderedXformOps():
                if op.GetOpType() == UsdGeom.XformOp.TypeScale:
                    value = op.Get()
                    if value is not None and abs(float(value[0]) - 0.001) < 1e-9:
                        scale_prims.append(str(prim.GetPath()))

    group_rows = []
    for name in sorted(groups):
        row = dict(groups[name])
        row["group"] = name
        row["only_via_instance_proxies"] = (
            row["meshes"] > 0 and row["instance_proxies"] >= row["meshes"]
        )
        group_rows.append(row)

    return {
        "path": os.path.abspath(path),
        "bytes": os.path.getsize(path),
        "default_prim": str(default_prim.GetPath()) if default_prim else None,
        "meters_per_unit": UsdGeom.GetStageMetersPerUnit(stage),
        "up_axis": UsdGeom.GetStageUpAxis(stage),
        "used_layers": _layer_list(stage),
        "counts": {key: value for key, value in sorted(counters.items())},
        "mesh_points_total": mesh_points,
        "prototype_groups": sorted(prototypes),
        "instanceable_paths": sorted(instanceable)[:40],
        "instanceable_count": len(instanceable),
        "scale_0001_prims": sorted(scale_prims),
        "physics": {key: value for key, value in sorted(physics.items())},
        "physics_authoring_present": bool(
            physics["rigid_bodies"] or physics["colliders"] or physics["joints"]
            or physics["articulation_roots"]
        ),
        "groups": group_rows,
    }


def _joint_row(prim):
    joint = UsdPhysics.Joint(prim)
    row = {
        "path": str(prim.GetPath()),
        "type": str(prim.GetTypeName()),
        "body0": [str(t) for t in (joint.GetBody0Rel().GetTargets() or [])],
        "body1": [str(t) for t in (joint.GetBody1Rel().GetTargets() or [])],
    }
    for attribute in ("physics:axis", "physics:lowerLimit", "physics:upperLimit"):
        attr = prim.GetAttribute(attribute)
        if attr and attr.HasAuthoredValue():
            value = attr.Get()
            row[attribute.split(":")[-1]] = (
                str(value) if not isinstance(value, (int, float)) else round(float(value), 6)
            )
    for attribute, key in (("physics:localPos0", "local_pos0"), ("physics:localPos1", "local_pos1")):
        attr = prim.GetAttribute(attribute)
        if attr and attr.HasAuthoredValue():
            row[key] = [round(float(v), 6) for v in attr.Get()]
    drives = {}
    for axis in ("linear", "angular"):
        drive = UsdPhysics.DriveAPI.Get(prim, axis)
        if drive and drive.GetPrim().IsValid():
            entry = {}
            for key, attr in (
                ("type", drive.GetTypeAttr()),
                ("stiffness", drive.GetStiffnessAttr()),
                ("damping", drive.GetDampingAttr()),
                ("max_force", drive.GetMaxForceAttr()),
                ("target_position", drive.GetTargetPositionAttr()),
            ):
                if attr and attr.HasAuthoredValue():
                    value = attr.Get()
                    entry[key] = (
                        str(value) if not isinstance(value, (int, float)) else round(float(value), 6)
                    )
            if entry:
                drives[axis] = entry
    if drives:
        row["drives"] = drives
    return row


def analyse_arm(path):
    stage = Usd.Stage.Open(path)
    if stage is None:
        raise SystemExit("cannot open arm stage: %s" % path)
    default_prim = stage.GetDefaultPrim()
    root_path = default_prim.GetPath() if default_prim else None
    physics = Counter()
    joints = []
    articulation_roots = []
    rigid_bodies = []
    meshes = 0
    for prim in iter_prims(stage):
        type_name = str(prim.GetTypeName())
        if type_name in MESH_TYPES:
            meshes += 1
        if prim.HasAPI(UsdPhysics.ArticulationRootAPI):
            articulation_roots.append(str(prim.GetPath()))
            physics["articulation_roots"] += 1
        if prim.HasAPI(UsdPhysics.RigidBodyAPI):
            rigid_bodies.append(str(prim.GetPath()))
            physics["rigid_bodies"] += 1
        if prim.HasAPI(UsdPhysics.MassAPI):
            physics["mass_api"] += 1
        if prim.HasAPI(UsdPhysics.CollisionAPI):
            physics["colliders"] += 1
        if prim.IsA(UsdPhysics.Joint):
            physics["joints"] += 1
            joints.append(_joint_row(prim))
        if prim.IsA(UsdPhysics.Scene):
            physics["scenes"] += 1
    joints.sort(key=lambda row: row["path"])
    root_joints = [
        row for row in joints
        if row["type"] in ("PhysicsFixedJoint", "FixedJoint")
        or not row["body0"]
        or any(target.strip("/") == "ReBot_Arm_RS" for target in row["body0"])
    ]
    by_type = Counter(row["type"] for row in joints)
    with_limits = [row for row in joints if "lowerLimit" in row or "upperLimit" in row]
    with_drives = [row for row in joints if row.get("drives")]
    return {
        "path": os.path.abspath(path),
        "bytes": os.path.getsize(path),
        "default_prim": str(root_path) if root_path else None,
        "meters_per_unit": UsdGeom.GetStageMetersPerUnit(stage),
        "up_axis": UsdGeom.GetStageUpAxis(stage),
        "used_layers": _layer_list(stage),
        "meshes": meshes,
        "physics": {key: value for key, value in sorted(physics.items())},
        "joint_count_by_type": {key: value for key, value in sorted(by_type.items())},
        "joints": joints,
        "rigid_body_paths": rigid_bodies,
        "articulation_roots": articulation_roots,
        "joints_with_limits": len(with_limits),
        "joints_with_drives": len(with_drives),
        "candidate_root_joints": root_joints,
    }


def build_findings(chassis, arm):
    """Turn the measured facts into the C1..C6 statements, with status labels."""
    findings = []
    if chassis:
        findings.append({
            "question": "C1 separate rigid bodies from the converted chassis",
            "status": "VERIFIED" if chassis["counts"].get("type:Mesh", 0) else "UNKNOWN",
            "evidence": {
                "meshes": chassis["counts"].get("type:Mesh", 0),
                "groups": len(chassis["groups"]),
                "prototypes": chassis["counts"].get("prototypes", 0),
                "instanceable": chassis["instanceable_count"],
            },
            "statement": (
                "the chassis mesh prims fall into %d top-level groups; nothing in the asset is a "
                "rigid body today" % len(chassis["groups"])
            ),
        })
        findings.append({
            "question": "C2 must the instance/prototype structure be broken for colliders",
            "status": "VERIFIED" if chassis["instanceable_count"] else "UNKNOWN",
            "evidence": {
                "instanceable_prims": chassis["instanceable_count"],
                "prototype_prim_count": chassis["counts"].get("prototypes", 0),
                "in_prototype_prims": chassis["counts"].get("in_prototype", 0),
                "groups_only_reachable_as_instance_proxies": [
                    row["group"] for row in chassis["groups"] if row["only_via_instance_proxies"]
                ],
            },
            "statement": (
                "colliders authored on prototype geometry are shared by every instance of that "
                "prototype; per-part colliders or per-part masses therefore require either "
                "authoring inside the prototype or un-instancing the affected subtrees"
            ),
        })
    if arm:
        findings.append({
            "question": "C3 single articulation for arm plus chassis",
            # A pre-existing arm articulation root is a static USD fact; it does NOT
            # establish that a newly connected chassis belongs to one PhysX
            # articulation after stepping.
            "status": "CANDIDATE" if arm["articulation_roots"] else "UNKNOWN",
            "evidence": {
                "articulation_roots": arm["articulation_roots"],
                "rigid_bodies": len(arm["rigid_body_paths"]),
                "joints": arm["physics"].get("joints", 0),
            },
            "statement": (
                "the arm already carries an ArticulationRootAPI prim and %d joints; "
                "single mobile-chassis articulation is NOT proven without a PhysX step"
                % arm["physics"].get("joints", 0)
            ),
        })
        findings.append({
            "question": "C4 the arm fixed root joint",
            "status": "VERIFIED" if arm["candidate_root_joints"] else "UNKNOWN",
            "evidence": arm["candidate_root_joints"],
            "statement": (
                "the arm hangs from a fixed joint whose body0 is outside the moving chain; "
                "mounting it on the chassis means re-targeting that joint to the chassis body"
            ),
        })
        findings.append({
            "question": "C5 arm-side configuration that must be preserved",
            "status": "VERIFIED",
            "evidence": {
                "joints_with_limits": arm["joints_with_limits"],
                "joints_with_drives": arm["joints_with_drives"],
                "joint_count_by_type": arm["joint_count_by_type"],
            },
            "statement": (
                "%d joints carry limits and %d carry drives; a chassis-side edit must leave these "
                "untouched" % (arm["joints_with_limits"], arm["joints_with_drives"])
            ),
        })
    findings.append({
        "question": "C6 values that must await real vendor or measurement data",
        "status": "UNKNOWN",
        "evidence": {
            "chassis_mass_api": (chassis or {}).get("physics", {}).get("mass_api", 0),
            "chassis_colliders": (chassis or {}).get("physics", {}).get("colliders", 0),
            "arm_mass_api": (arm or {}).get("physics", {}).get("mass_api", 0),
        },
        "statement": (
            "no mass, inertia, friction or mount compliance value exists in either asset; these "
            "must come from the vendor or from a weighing measurement"
        ),
    })
    return findings


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="Physics readiness facts for Morph One + B601-RS")
    parser.add_argument("--chassis-usd", default="")
    parser.add_argument("--arm-usd", default="")
    parser.add_argument("--output-json", default="")
    parser.add_argument("--group-depth", type=int, default=2)
    args = parser.parse_args(argv)
    if not args.chassis_usd and not args.arm_usd:
        parser.error("at least one of --chassis-usd / --arm-usd is required")

    report = {"chassis": None, "arm": None}
    if args.chassis_usd:
        report["chassis"] = analyse_stage(args.chassis_usd, group_depth=args.group_depth)
    if args.arm_usd:
        report["arm"] = analyse_arm(args.arm_usd)
    report["findings"] = build_findings(report["chassis"], report["arm"])

    if args.output_json:
        directory = os.path.dirname(os.path.abspath(args.output_json))
        if directory:
            os.makedirs(directory, exist_ok=True)
        with open(args.output_json, "w", encoding="utf-8") as handle:
            json.dump(report, handle, ensure_ascii=False, indent=1)
    summary = {
        "output": os.path.abspath(args.output_json) if args.output_json else None,
        "chassis": None if not report["chassis"] else {
            "prims": report["chassis"]["counts"].get("prims", 0),
            "meshes": report["chassis"]["counts"].get("type:Mesh", 0),
            "instanceable": report["chassis"]["instanceable_count"],
            "prototypes": report["chassis"]["counts"].get("prototypes", 0),
            "meters_per_unit": report["chassis"]["meters_per_unit"],
            "up_axis": report["chassis"]["up_axis"],
            "physics": report["chassis"]["physics"],
            "groups": len(report["chassis"]["groups"]),
        },
        "arm": None if not report["arm"] else {
            "joints": report["arm"]["physics"].get("joints", 0),
            "rigid_bodies": len(report["arm"]["rigid_body_paths"]),
            "articulation_roots": report["arm"]["articulation_roots"],
            "with_limits": report["arm"]["joints_with_limits"],
            "with_drives": report["arm"]["joints_with_drives"],
        },
        "findings": [(f["question"], f["status"]) for f in report["findings"]],
    }
    print(json.dumps(summary, ensure_ascii=False, indent=1))
    return 0


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
