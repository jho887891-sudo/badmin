"""Mount-scheme sandbox for a single articulated Morph One chassis + B601-RS arm.

Scope, stated plainly
---------------------
This tool builds two *test proxies* and checks that a USD stage can express the
mounting scheme:

  A) a proxy chain (boxes, revolute joints, one fixed mount) that a PhysX
     articulation would drive - simple geometry, definitely NOT the real robot;
  B) an override of the real arm's fixed root joint so that its body0 points at a
     chassis root body, proving the composition resolves - again without claiming
     that PhysX agrees, because nothing here is stepped in a physics engine.

It therefore provides evidence about *authoring and composition*, not about
dynamic behaviour.  The real arm and chassis assets are opened read-only; their
sha256 is recorded before and after so the evidence shows nothing was written.

Usage
    python scripts/simulation/morph_one_mount_sandbox.py \
        --chassis-usd <verified morph_one usda> --arm-usd <rebot_b601_rs.usda> \
        --output-usda <proxy.usda> --output-json <report.json> [--self-check]
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys

from pxr import Gf, Sdf, Usd, UsdGeom, UsdPhysics

PROXY_DOF = 2
ARM_REVOLUTE = 6
ARM_PRISMATIC = 2
PROXY_NOTE = "TEST_PROXY: simple boxes, not the Morph One mechanical assembly"


def sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _box(stage, path, size, translation=(0.0, 0.0, 0.0), mass=1.0, name=None):
    """Author one rigid box body with a cube collider."""
    path = Sdf.Path(path) if not isinstance(path, Sdf.Path) else path
    xform = UsdGeom.Xform.Define(stage, path)
    if name:
        xform.GetPrim().SetCustomDataByKey("label", name)
    cube = UsdGeom.Cube.Define(stage, path.AppendChild("geometry"))
    cube.CreateSizeAttr(1.0)
    cube.CreateExtentAttr([Gf.Vec3f(-0.5), Gf.Vec3f(0.5)])
    scale = UsdGeom.Xformable(cube.GetPrim()).AddScaleOp()
    scale.Set(Gf.Vec3f(*size))
    translate = UsdGeom.Xformable(cube.GetPrim()).AddTranslateOp()
    translate.Set(Gf.Vec3f(*translation))
    UsdPhysics.RigidBodyAPI.Apply(xform.GetPrim())
    UsdPhysics.CollisionAPI.Apply(cube.GetPrim())
    mass_api = UsdPhysics.MassAPI.Apply(xform.GetPrim())
    mass_api.CreateMassAttr(mass)
    mass_api.CreateCenterOfMassAttr(Gf.Vec3f(0.0))
    return xform.GetPrim()


def _revolute(stage, path, body0, body1, axis="Y", lower=-180.0, upper=180.0,
              stiffness=100.0, damping=2.0, max_force=50.0):
    joint = UsdPhysics.RevoluteJoint.Define(stage, path)
    joint.CreateBody0Rel().SetTargets([body0])
    joint.CreateBody1Rel().SetTargets([body1])
    joint.CreateAxisAttr(axis)
    joint.CreateLowerLimitAttr(lower)
    joint.CreateUpperLimitAttr(upper)
    drive = UsdPhysics.DriveAPI.Apply(joint.GetPrim(), "angular")
    drive.CreateTypeAttr("force")
    drive.CreateStiffnessAttr(stiffness)
    drive.CreateDampingAttr(damping)
    drive.CreateMaxForceAttr(max_force)
    return joint.GetPrim()


def _fixed(stage, path, body0, body1):
    joint = UsdPhysics.FixedJoint.Define(stage, path)
    joint.CreateBody0Rel().SetTargets([body0])
    joint.CreateBody1Rel().SetTargets([body1])
    return joint.GetPrim()


def build_proxy_stage():
    """The proxy chain: a base, two links, two revolute joints, one fixed mount."""
    stage = Usd.Stage.CreateInMemory()
    UsdGeom.SetStageMetersPerUnit(stage, 1.0)
    UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)
    world = UsdGeom.Xform.Define(stage, "/World")
    stage.SetDefaultPrim(world.GetPrim())
    world.GetPrim().SetCustomDataByKey("label", PROXY_NOTE)

    chassis = UsdGeom.Xform.Define(stage, "/World/ChassisProxy")
    chassis.GetPrim().SetCustomDataByKey("label", "TEST_PROXY chassis plate")
    UsdPhysics.RigidBodyAPI.Apply(chassis.GetPrim())
    plate = UsdGeom.Cube.Define(stage, "/World/ChassisProxy/geometry")
    plate.CreateSizeAttr(1.0)
    plate_scale = UsdGeom.Xformable(plate.GetPrim()).AddScaleOp()
    plate_scale.Set(Gf.Vec3f(0.6, 0.4, 0.05))
    UsdPhysics.CollisionAPI.Apply(plate.GetPrim())
    chassis_mass = UsdPhysics.MassAPI.Apply(chassis.GetPrim())
    chassis_mass.CreateMassAttr(15.0)

    arm = UsdGeom.Xform.Define(stage, "/World/ArmProxy")
    arm.GetPrim().SetCustomDataByKey("label", "TEST_PROXY two-link arm, NOT the B601-RS")
    UsdPhysics.ArticulationRootAPI.Apply(arm.GetPrim())

    base = _box(stage, "/World/ArmProxy/base", (0.1, 0.1, 0.1),
                (0.0, 0.0, 0.08), 2.0, "proxy base")
    upper = _box(stage, "/World/ArmProxy/link1", (0.3, 0.05, 0.05),
                 (0.15, 0.0, 0.08), 1.0, "proxy upper link")
    fore = _box(stage, "/World/ArmProxy/link2", (0.25, 0.04, 0.04),
                (0.15, 0.0, 0.08), 0.5, "proxy forearm")
    arm_joint = _revolute(stage, "/World/ArmProxy/joint1", base.GetPath(), upper.GetPath(),
                          "Y", -90.0, 90.0)
    fore_joint = _revolute(stage, "/World/ArmProxy/joint2", upper.GetPath(), fore.GetPath(),
                           "Y", -120.0, 120.0)
    mount = _fixed(stage, "/World/mount_fixed", chassis.GetPath(), base.GetPath())
    mount.GetPrim().SetCustomDataByKey("label", "fixed mount: chassis body -> arm root body")
    return stage, {
        "arm_joints": [str(arm_joint.GetPath()), str(fore_joint.GetPath())],
        "mount_joint": str(mount.GetPath()),
        "proxy_dof": PROXY_DOF,
    }


def validate_proxy_stage(path):
    """Reopen the written proxy and read the facts back from the file."""
    stage = Usd.Stage.Open(path)
    if stage is None:
        raise SystemExit("cannot reopen proxy stage: %s" % path)
    joints = []
    roots = []
    bodies = 0
    for prim in stage.Traverse():
        if prim.HasAPI(UsdPhysics.ArticulationRootAPI):
            roots.append(str(prim.GetPath()))
        if prim.HasAPI(UsdPhysics.RigidBodyAPI):
            bodies += 1
        if prim.IsA(UsdPhysics.Joint):
            joint = UsdPhysics.Joint(prim)
            joints.append({
                "path": str(prim.GetPath()),
                "type": str(prim.GetTypeName()),
                "body0": [str(t) for t in (joint.GetBody0Rel().GetTargets() or [])],
                "body1": [str(t) for t in (joint.GetBody1Rel().GetTargets() or [])],
                "resolved": all(
                    stage.GetPrimAtPath(target).IsValid()
                    for target in list(joint.GetBody0Rel().GetTargets() or [])
                    + list(joint.GetBody1Rel().GetTargets() or [])
                ),
            })
    dof = sum(1 for row in joints if row["type"] in ("PhysicsRevoluteJoint", "PhysicsPrismaticJoint"))
    return {
        "path": os.path.abspath(path),
        "articulation_roots": roots,
        "rigid_bodies": bodies,
        "joints": joints,
        "dof_from_joint_types": dof,
        "all_joint_targets_resolve": all(row["resolved"] for row in joints),
        "meters_per_unit": UsdGeom.GetStageMetersPerUnit(stage),
        "up_axis": UsdGeom.GetStageUpAxis(stage),
    }


def _stage_mpu(path):
    stage = Usd.Stage.Open(path)
    if stage is None:
        raise SystemExit("cannot open %s" % path)
    return UsdGeom.GetStageMetersPerUnit(stage)


def _apply_unit_scale(prim, source_mpu, target_mpu, collected):
    """USD unit metadata is not converted by referencing, so make the scale explicit."""
    if source_mpu is None or target_mpu is None or source_mpu == target_mpu:
        collected.append({"prim": str(prim.GetPath()), "scale": 1.0,
                          "reason": "same metersPerUnit, no scale needed"})
        return 1.0
    factor = float(source_mpu) / float(target_mpu)
    scale = UsdGeom.Xformable(prim).AddScaleOp()
    scale.Set(Gf.Vec3f(factor))
    collected.append({
        "prim": str(prim.GetPath()),
        "scale": factor,
        "reason": "source metersPerUnit %s -> stage metersPerUnit %s" % (source_mpu, target_mpu),
    })
    return factor


def build_mount_override(chassis_usd, arm_usd, override_path):
    """Author a mount override for the real arm's root joint, without editing assets."""
    layer = Sdf.Layer.CreateNew(override_path)
    stage = Usd.Stage.Open(layer)
    UsdGeom.SetStageMetersPerUnit(stage, 1.0)
    UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)

    root = UsdGeom.Xform.Define(stage, "/MorphOneMount")
    stage.SetDefaultPrim(root.GetPrim())
    root.GetPrim().SetCustomDataByKey(
        "label",
        "TEST_PROXY composition: real chassis and arm referenced read-only; the mount "
        "override below is the only authored physics statement",
    )

    chassis = UsdGeom.Xform.Define(stage, "/MorphOneMount/Chassis")
    chassis.GetPrim().GetReferences().AddReference(os.path.abspath(chassis_usd))
    chassis.GetPrim().SetCustomDataByKey("source", os.path.abspath(chassis_usd))

    arm = UsdGeom.Xform.Define(stage, "/MorphOneMount/Arm")
    arm.GetPrim().GetReferences().AddReference(os.path.abspath(arm_usd))
    arm.GetPrim().SetCustomDataByKey("source", os.path.abspath(arm_usd))

    unit_scales = []
    _apply_unit_scale(chassis.GetPrim(), _stage_mpu(chassis_usd), 1.0, unit_scales)
    _apply_unit_scale(arm.GetPrim(), _stage_mpu(arm_usd), 1.0, unit_scales)

    # The chassis asset has no rigid body, so the mount body0 is authored on the
    # chassis root prim itself; that is exactly the decision the report asks for.
    UsdPhysics.RigidBodyAPI.Apply(chassis.GetPrim())

    arm_root = stage.GetPrimAtPath("/MorphOneMount/Arm")
    joint_path = None
    for prim in Usd.PrimRange(arm_root):
        if prim.IsA(UsdPhysics.Joint) and prim.GetName() == "root_joint":
            joint_path = prim.GetPath()
            break
    if joint_path is None:
        raise SystemExit("the arm asset has no root_joint prim to override")
    over = stage.OverridePrim(joint_path)
    joint = UsdPhysics.FixedJoint(over)
    joint.CreateBody0Rel().SetTargets(["/MorphOneMount/Chassis"])
    stage.GetRootLayer().Save()
    return {
        "override_path": os.path.abspath(override_path),
        "overridden_joint": str(joint_path),
        "new_body0": "/MorphOneMount/Chassis",
        "unit_scales": unit_scales,
    }


def validate_mount_override(override_path):
    stage = Usd.Stage.Open(override_path)
    if stage is None:
        raise SystemExit("cannot open override stage: %s" % override_path)
    joint_paths = []
    roots = []
    for prim in Usd.PrimRange.Stage(
        stage, Usd.TraverseInstanceProxies(Usd.PrimAllPrimsPredicate)
    ):
        if prim.HasAPI(UsdPhysics.ArticulationRootAPI):
            roots.append(str(prim.GetPath()))
        if prim.IsA(UsdPhysics.Joint):
            joint_paths.append(prim)
    root_joint = None
    for prim in joint_paths:
        if prim.GetName() == "root_joint":
            joint = UsdPhysics.Joint(prim)
            targets0 = [str(t) for t in (joint.GetBody0Rel().GetTargets() or [])]
            targets1 = [str(t) for t in (joint.GetBody1Rel().GetTargets() or [])]
            root_joint = {
                "path": str(prim.GetPath()),
                "body0": targets0,
                "body1": targets1,
                "body0_resolves": all(stage.GetPrimAtPath(t).IsValid() for t in targets0),
                "body1_resolves": all(stage.GetPrimAtPath(t).IsValid() for t in targets1),
                "overridden": targets0 == ["/MorphOneMount/Chassis"],
            }
            break
    return {
        "path": os.path.abspath(override_path),
        "joint_count": len(joint_paths),
        "articulation_roots": roots,
        "single_articulation_root": len(roots) == 1,
        "root_joint": root_joint,
    }


def self_check(proxy_report, override_report):
    """Assertions that make the evidence falsifiable instead of descriptive."""
    checks = []

    def check(name, condition, detail=""):
        checks.append({"check": name, "pass": bool(condition), "detail": detail})

    check("proxy has exactly one articulation root", len(proxy_report["articulation_roots"]) == 1,
          proxy_report["articulation_roots"])
    check("proxy DOF equals the expected %d" % PROXY_DOF,
          proxy_report["dof_from_joint_types"] == PROXY_DOF,
          proxy_report["dof_from_joint_types"])
    check("every proxy joint target resolves", proxy_report["all_joint_targets_resolve"])
    check("proxy includes the fixed mount joint",
          any(row["type"] == "PhysicsFixedJoint" for row in proxy_report["joints"]))
    check("override retargets body0 at the chassis body",
          bool(override_report["root_joint"] and override_report["root_joint"]["overridden"]),
          override_report["root_joint"])
    check("override targets resolve in the composed stage",
          bool(override_report["root_joint"]
               and override_report["root_joint"]["body0_resolves"]
               and override_report["root_joint"]["body1_resolves"]))
    check("one articulation root after mounting", override_report["single_articulation_root"],
          override_report["articulation_roots"])
    return checks


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="Mount-scheme sandbox (proxy geometry only)")
    parser.add_argument("--chassis-usd", default="")
    parser.add_argument("--arm-usd", default="")
    parser.add_argument("--output-usda", default="mount_proxy.usda")
    parser.add_argument("--override-usda", default="")
    parser.add_argument("--output-json", default="")
    parser.add_argument("--self-check", action="store_true")
    args = parser.parse_args(argv)

    proxy_stage, description = build_proxy_stage()
    proxy_stage.GetRootLayer().Export(args.output_usda)
    proxy = validate_proxy_stage(args.output_usda)

    override = None
    chassis_hash_before = arm_hash_before = None
    if args.chassis_usd and args.arm_usd:
        chassis_hash_before = sha256(args.chassis_usd)
        arm_hash_before = sha256(args.arm_usd)
        override_path = args.override_usda or args.output_usda.replace(".usda", "_mount.usda")
        authored = build_mount_override(args.chassis_usd, args.arm_usd, override_path)
        override = validate_mount_override(override_path)
        override["unit_scales"] = authored["unit_scales"]
        override["chassis_sha256_before"] = chassis_hash_before
        override["chassis_sha256_after"] = sha256(args.chassis_usd)
        override["arm_sha256_before"] = arm_hash_before
        override["arm_sha256_after"] = sha256(args.arm_usd)
        override["assets_unchanged"] = (
            override["chassis_sha256_before"] == override["chassis_sha256_after"]
            and override["arm_sha256_before"] == override["arm_sha256_after"]
        )

    report = {
        "labels": {
            "proxy": PROXY_NOTE,
            "physics_stepped": False,
            "claim": "authoring and composition evidence only; no PhysX evaluation",
        },
        "proxy": proxy,
        "proxy_description": description,
        "mount_override": override,
        "arm_dof_expected": ARM_REVOLUTE + ARM_PRISMATIC,
    }
    checks = self_check(proxy, override) if override else []
    report["self_checks"] = checks
    if args.output_json:
        directory = os.path.dirname(os.path.abspath(args.output_json))
        if directory:
            os.makedirs(directory, exist_ok=True)
        with open(args.output_json, "w", encoding="utf-8") as handle:
            json.dump(report, handle, ensure_ascii=False, indent=1)
    summary = {
        "proxy_usda": os.path.abspath(args.output_usda),
        "override_usda": override["path"] if override else None,
        "proxy_dof": proxy["dof_from_joint_types"],
        "proxy_articulation_roots": proxy["articulation_roots"],
        "override_root_joint": (override or {}).get("root_joint"),
        "assets_unchanged": (override or {}).get("assets_unchanged"),
        "checks_passed": sum(1 for c in checks if c["pass"]),
        "checks_total": len(checks),
        "output_json": os.path.abspath(args.output_json) if args.output_json else None,
    }
    print(json.dumps(summary, ensure_ascii=False, indent=1))
    if args.self_check and checks and not all(c["pass"] for c in checks):
        return 1
    return 0


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
