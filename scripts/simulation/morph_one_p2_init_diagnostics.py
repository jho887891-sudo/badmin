"""P2 diagnostics: is the t=0 mount mismatch an authoring, spawn, or read-convention problem?

The roadmap (issue #13) lists one blocking symptom: after sim.reset() the proxy chassis
already sits at ~179 deg yaw and the mount offset appears with flipped x/y signs, which
cannot be reconciled with the authored USD.  This script does not guess.  It records the
same proxy at four moments and compares them explicitly:

  A. authored   - world transform computed from the authored USD layer (read back)
  B. spawned    - composed world transform of the same prims on the live stage, after
                  referencing but before any physics step
  C. reset0     - first Articulation read after sim.reset(), before a single step
  D. reset1/settle - subsequent steps

Every comparison is evaluated under BOTH quaternion conventions (wxyz and xyzw), because a
component-order mistake and a genuine initial offset produce identical-looking yaw reports.
Whichever convention keeps the rigid mount satisfied is the real one; if neither does, the
initialisation is genuinely inconsistent and the root cause stays UNKNOWN.

TEST_PROXY geometry only.  The real B601-RS asset is never opened by this script.

Usage:
    python morph_one_p2_init_diagnostics.py --variant joint-local --out p2_joint.json
"""

from __future__ import annotations

import argparse
import json
import math
import os

import numpy as np

from isaaclab.app import AppLauncher

MOUNT_OFFSET = (0.12, -0.05, 0.20)
CHASSIS_HALF = (0.25, 0.175, 0.04)
CHASSIS_CUBE_CENTRE = (0.0, 0.0, 0.06)
ARM_BASE_HALF = (0.04, 0.04, 0.04)
LINK1_HALF = (0.14, 0.025, 0.025)
LINK2_HALF = (0.11, 0.02, 0.02)
# small clearance between the arm base cube and the first link cube
JOINT1_GAP = 0.045
SETTLE_STEPS = 60
DT = 1.0 / 120.0
SPAWN_PATH = "/World/P2"
BODY_PATHS = ("Chassis", "Arm/base_link", "Arm/link1", "Arm/link2")

PARSER = argparse.ArgumentParser(description="P2 initialisation diagnostics (TEST_PROXY)")
PARSER.add_argument("--variant", choices=["joint-local", "body-only"], default="joint-local")
PARSER.add_argument("--out", required=True)
PARSER.add_argument("--usd-out", default="")
PARSER.add_argument("--settle-steps", type=int, default=SETTLE_STEPS)
PARSER.add_argument("--pos-tol", type=float, default=1e-3)
PARSER.add_argument("--rot-tol", type=float, default=1e-3)
AppLauncher.add_app_launcher_args(PARSER)
ARGS = PARSER.parse_args()
LAUNCHER = AppLauncher(ARGS)
SIMULATION_APP = LAUNCHER.app

import torch  # noqa: E402
import isaaclab.sim as sim_utils  # noqa: E402
from isaaclab.assets import Articulation, ArticulationCfg  # noqa: E402
from isaaclab.actuators import ImplicitActuatorCfg  # noqa: E402
from isaaclab_physx.physics import PhysxCfg  # noqa: E402
from pxr import Gf, Sdf, Usd, UsdGeom, UsdPhysics  # noqa: E402


# ----------------------------------------------------------------- authoring
def _cube(stage, path, half, centre, mass, label):
    path = path if isinstance(path, Sdf.Path) else Sdf.Path(path)
    body = UsdGeom.Xform.Define(stage, path)
    body.GetPrim().SetCustomDataByKey("label", label)
    cube = UsdGeom.Cube.Define(stage, path.AppendChild("geometry"))
    cube.CreateSizeAttr(1.0)
    UsdGeom.Xformable(cube.GetPrim()).AddScaleOp().Set(
        Gf.Vec3f(2.0 * half[0], 2.0 * half[1], 2.0 * half[2]))
    UsdGeom.Xformable(cube.GetPrim()).AddTranslateOp().Set(Gf.Vec3f(*centre))
    UsdPhysics.RigidBodyAPI.Apply(body.GetPrim())
    UsdPhysics.CollisionAPI.Apply(cube.GetPrim())
    mass_api = UsdPhysics.MassAPI.Apply(body.GetPrim())
    mass_api.CreateMassAttr(mass)
    mass_api.CreateCenterOfMassAttr(Gf.Vec3f(0.0))
    return body.GetPrim()


def _revolute(stage, path, body0, body1, lower, upper, local_pos0):
    joint = UsdPhysics.RevoluteJoint.Define(stage, path)
    joint.CreateBody0Rel().SetTargets([body0])
    joint.CreateBody1Rel().SetTargets([body1])
    joint.CreateAxisAttr("Y")
    joint.CreateLowerLimitAttr(lower)
    joint.CreateUpperLimitAttr(upper)
    joint.CreateLocalPos0Attr().Set(Gf.Vec3f(*local_pos0))
    joint.CreateLocalPos1Attr().Set(Gf.Vec3f(0.0, 0.0, 0.0))
    joint.CreateLocalRot0Attr().Set(Gf.Quatf(1.0, 0.0, 0.0, 0.0))
    joint.CreateLocalRot1Attr().Set(Gf.Quatf(1.0, 0.0, 0.0, 0.0))
    drive = UsdPhysics.DriveAPI.Apply(joint.GetPrim(), "angular")
    drive.CreateTypeAttr("force")
    drive.CreateStiffnessAttr(50.0)
    drive.CreateDampingAttr(5.0)
    drive.CreateMaxForceAttr(100.0)
    return joint.GetPrim()


def author(path, variant):
    stage = Usd.Stage.CreateNew(path)
    UsdGeom.SetStageMetersPerUnit(stage, 1.0)
    UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)
    root = UsdGeom.Xform.Define(stage, "/Sandbox")
    stage.SetDefaultPrim(root.GetPrim())
    root.GetPrim().SetCustomDataByKey("label", "TEST_PROXY P2 diagnostics")
    UsdPhysics.ArticulationRootAPI.Apply(root.GetPrim())

    chassis = _cube(stage, "/Sandbox/Chassis", CHASSIS_HALF, CHASSIS_CUBE_CENTRE, 15.0,
                    "TEST_PROXY chassis body")
    base = _cube(stage, "/Sandbox/Arm/base_link", ARM_BASE_HALF, (0.0, 0.0, 0.0), 2.0,
                 "TEST_PROXY arm base body")
    UsdGeom.Xformable(base).AddTranslateOp().Set(Gf.Vec3f(*MOUNT_OFFSET))
    # Links are laid out so that NO two proxy bodies overlap: the previous layout put a
    # 0.28 m upper link and a 0.22 m forearm through each other, and the solver's
    # depenetration pushed the free-floating base 0.36 m during settle.  Each link body
    # sits at its joint anchor and its collision cube extends outward from there.
    link1 = _cube(stage, "/Sandbox/Arm/link1", LINK1_HALF, (LINK1_HALF[0], 0.0, 0.0), 1.0,
                  "TEST_PROXY upper link")
    UsdGeom.Xformable(link1).AddTranslateOp().Set(
        Gf.Vec3f(MOUNT_OFFSET[0] + JOINT1_GAP, MOUNT_OFFSET[1], MOUNT_OFFSET[2]))
    link2 = _cube(stage, "/Sandbox/Arm/link2", LINK2_HALF, (LINK2_HALF[0], 0.0, 0.0), 0.5,
                  "TEST_PROXY forearm")
    UsdGeom.Xformable(link2).AddTranslateOp().Set(
        Gf.Vec3f(MOUNT_OFFSET[0] + JOINT1_GAP + 2.0 * LINK1_HALF[0],
                 MOUNT_OFFSET[1], MOUNT_OFFSET[2]))

    _revolute(stage, "/Sandbox/Arm/joint1", base.GetPath(), link1.GetPath(), -90.0, 90.0,
              (JOINT1_GAP, 0.0, 0.0))
    _revolute(stage, "/Sandbox/Arm/joint2", link1.GetPath(), link2.GetPath(), -120.0, 120.0,
              (2.0 * LINK1_HALF[0], 0.0, 0.0))
    mount = UsdPhysics.FixedJoint.Define(stage, "/Sandbox/mount_fixed")
    mount.CreateBody0Rel().SetTargets([chassis.GetPath()])
    mount.CreateBody1Rel().SetTargets([base.GetPath()])
    if variant == "joint-local":
        mount.CreateLocalPos0Attr().Set(Gf.Vec3f(*MOUNT_OFFSET))
        mount.CreateLocalRot0Attr().Set(Gf.Quatf(1.0, 0.0, 0.0, 0.0))
        mount.CreateLocalPos1Attr().Set(Gf.Vec3f(0.0, 0.0, 0.0))
        mount.CreateLocalRot1Attr().Set(Gf.Quatf(1.0, 0.0, 0.0, 0.0))
    stage.GetRootLayer().Save()
    return {"usd": os.path.abspath(path), "variant": variant,
            "mount_offset_m": list(MOUNT_OFFSET)}


def read_world_poses(stage, prefix):
    """Composed world pos/quat of the named bodies, read from a live USD stage."""
    xform_cache = UsdGeom.XformCache(Usd.TimeCode.Default())
    out = {}
    for name in BODY_PATHS:
        prim = stage.GetPrimAtPath("%s/%s" % (prefix, name))
        if not prim or not prim.IsValid():
            out[name] = {"found": False}
            continue
        matrix = xform_cache.GetLocalToWorldTransform(prim)
        translation = matrix.ExtractTranslation()
        rotation = matrix.ExtractRotationQuat()
        imaginary = rotation.GetImaginary()
        out[name] = {
            "found": True,
            "pos": [round(float(v), 9) for v in translation],
            "quat_wxyz": [round(float(rotation.GetReal()), 9),
                          round(float(imaginary[0]), 9),
                          round(float(imaginary[1]), 9),
                          round(float(imaginary[2]), 9)],
        }
    return out


def read_joint_anchors(stage, prefix):
    prim = stage.GetPrimAtPath("%s/mount_fixed" % prefix)
    if not prim or not prim.IsValid():
        return {"found": False}
    joint = UsdPhysics.FixedJoint(prim)
    def target(rel):
        return [str(t) for t in (rel.GetTargets() or [])]
    def value(attr):
        a = prim.GetAttribute(attr)
        if a and a.HasAuthoredValue():
            v = a.Get()
            try:
                return [round(float(x), 9) for x in v]
            except TypeError:
                return str(v)
        return None
    return {
        "found": True,
        "body0": target(joint.GetBody0Rel()),
        "body1": target(joint.GetBody1Rel()),
        "localPos0": value("physics:localPos0"),
        "localPos1": value("physics:localPos1"),
        "localRot0": value("physics:localRot0"),
        "localRot1": value("physics:localRot1"),
    }


# --------------------------------------------------------------- quaternion math
def q_wxyz_to_matrix(quat):
    w, x, y, z = unit(quat)
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
        [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
        [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)],
    ])


def unit(quat):
    values = np.asarray(quat, dtype=float)
    norm = float(np.linalg.norm(values))
    if not np.isfinite(norm) or norm <= 0.0:
        raise ValueError("degenerate quaternion %r" % (values,))
    return values / norm


def as_wxyz(quat, convention):
    """Reinterpret a 4-tuple under the requested component order."""
    values = np.asarray(quat, dtype=float)
    if convention == "wxyz":
        return unit(values)
    return unit(np.array([values[3], values[0], values[1], values[2]]))


def relative_position_error(chassis_pos, chassis_quat, base_pos, base_quat, convention,
                            expected_offset):
    rotation = q_wxyz_to_matrix(as_wxyz(chassis_quat, convention))
    measured = rotation.T @ (np.asarray(base_pos, dtype=float)
                             - np.asarray(chassis_pos, dtype=float))
    return float(np.linalg.norm(measured - np.asarray(expected_offset, dtype=float))), \
        [round(float(v), 9) for v in measured]


def relative_rotation_angle(chassis_quat, base_quat, convention):
    chassis = q_wxyz_to_matrix(as_wxyz(chassis_quat, convention))
    base = q_wxyz_to_matrix(as_wxyz(base_quat, convention))
    delta = chassis.T @ base
    cosine = max(-1.0, min(1.0, (float(np.trace(delta)) - 1.0) / 2.0))
    return float(math.acos(cosine))


def evaluate(poses, convention, expected_offset):
    """Mount residual and chassis-vs-authored deviation under one convention."""
    chassis = poses["Chassis"]
    base = poses["Arm/base_link"]
    position_error, relative = relative_position_error(
        chassis["pos"], chassis["quat_wxyz"], base["pos"], base["quat_wxyz"],
        convention, expected_offset)
    rotation_error = relative_rotation_angle(
        chassis["quat_wxyz"], base["quat_wxyz"], convention)
    return {"convention": convention, "mount_position_error_m": round(position_error, 9),
            "mount_relative_position_m": relative,
            "mount_rotation_error_rad": round(rotation_error, 9)}


def flat(values):
    if hasattr(values, "detach"):
        values = values.detach().cpu().numpy()
    return np.asarray(values).reshape(-1).astype(float).tolist()

def build_sim():
    physics = PhysxCfg(solver_type=1, enable_external_forces_every_iteration=True)
    sim = sim_utils.SimulationContext(
        sim_utils.SimulationCfg(dt=DT, device="cuda:0", gravity=(0.0, 0.0, 0.0),
                                physics=physics))
    ground = sim_utils.GroundPlaneCfg()
    ground.func("/World/ground", ground)
    return sim


def live_stage_poses(prefix):
    import omni.usd

    stage = omni.usd.get_context().get_stage()
    return read_world_poses(stage, prefix), read_joint_anchors(stage, prefix)


def main():
    usd_out = ARGS.usd_out or ARGS.out.replace(".json", ".usda")
    authored_layer = author(usd_out, ARGS.variant)

    # A. authored: read the file back exactly as composed on disk
    stage_authored = Usd.Stage.Open(authored_layer["usd"])
    authored = read_world_poses(stage_authored, "/Sandbox")
    authored_anchors = read_joint_anchors(stage_authored, "/Sandbox")

    sim = build_sim()
    spawn = sim_utils.UsdFileCfg(usd_path=authored_layer["usd"])
    spawn.func(SPAWN_PATH, spawn)

    # B. spawned: composed on the live stage, before any physics step
    spawned, live_anchors = live_stage_poses(SPAWN_PATH)

    actuators = {"all": ImplicitActuatorCfg(joint_names_expr=[".*"], stiffness=300.0,
                                            damping=20.0)}
    robot = Articulation(ArticulationCfg(prim_path=SPAWN_PATH, spawn=None,
                                         actuators=actuators))
    sim.reset()

    body_names = list(robot.body_names)
    joint_names = list(robot.joint_names)
    index_of = {name: position for position, name in enumerate(body_names)}

    def live_physics_poses():
        out = {}
        non_finite = 0
        for name in BODY_PATHS:
            link = name.split("/")[-1]
            if link not in index_of:
                out[name] = {"found": False}
                continue
            index = index_of[link]
            pos = flat(robot.data.body_pos_w[0, index])
            quat = flat(robot.data.body_quat_w[0, index])
            if not all(math.isfinite(v) for v in pos + quat):
                non_finite += 1
            out[name] = {"found": True, "pos": [round(v, 9) for v in pos],
                         "quat_wxyz": [round(v, 9) for v in quat]}
        return out, non_finite

    # C. reset0: first Articulation read after sim.reset(), zero steps taken
    reset0, non_finite_reset = live_physics_poses()
    root_reset0 = {"pos": [round(v, 9) for v in flat(robot.data.root_pos_w[0])],
                   "quat_wxyz": [round(v, 9) for v in flat(robot.data.root_quat_w[0])]}

    def step(count):
        for _ in range(count):
            robot.write_data_to_sim()
            sim.step()
            robot.update(sim.get_physics_dt())

    step(1)
    reset1, non_finite_1 = live_physics_poses()
    settle = []
    for position in range(ARGS.settle_steps):
        step(1)
        poses, bad = live_physics_poses()
        settle.append({"step": position + 2, "poses": poses, "non_finite": bad})

    offset = np.asarray(MOUNT_OFFSET, dtype=float)
    comparisons = {
        "authored_vs_spawned": {conv: evaluate(spawned, conv, offset) for conv in ("wxyz", "xyzw")},
        "authored_vs_reset0": {conv: evaluate(reset0, conv, offset) for conv in ("wxyz", "xyzw")},
        "authored_vs_reset1": {conv: evaluate(reset1, conv, offset) for conv in ("wxyz", "xyzw")},
    }

    def chassis_deviation(measured, convention):
        reference = authored["Chassis"]
        delta = np.asarray(measured["Chassis"]["pos"], dtype=float) \
            - np.asarray(reference["pos"], dtype=float)
        rotation = q_wxyz_to_matrix(as_wxyz(reference["quat_wxyz"], convention))
        measured_rotation = q_wxyz_to_matrix(as_wxyz(measured["Chassis"]["quat_wxyz"], convention))
        delta_rotation = rotation.T @ measured_rotation
        cosine = max(-1.0, min(1.0, (float(np.trace(delta_rotation)) - 1.0) / 2.0))
        return {"position_delta_m": [round(float(v), 9) for v in delta],
                "position_norm_m": round(float(np.linalg.norm(delta)), 9),
                "rotation_angle_rad": round(float(math.acos(cosine)), 9)}

    chassis_checks = {}
    for label, measured in (("spawned", spawned), ("reset0", reset0), ("reset1", reset1)):
        chassis_checks[label] = {conv: chassis_deviation(measured, conv)
                                 for conv in ("wxyz", "xyzw")}

    settle_drift = round(float(np.linalg.norm(
        np.asarray(settle[-1]["poses"]["Chassis"]["pos"], dtype=float)
        - np.asarray(reset0["Chassis"]["pos"], dtype=float))), 9) if settle else None
    settle_mount = {conv: evaluate(settle[-1]["poses"], conv, offset) for conv in ("wxyz", "xyzw")} \
        if settle else {}
    non_finite_total = non_finite_reset + non_finite_1 + sum(row["non_finite"] for row in settle)

    # the convention that keeps the rigid mount satisfied at reset0 is the real one
    best = min(("wxyz", "xyzw"),
               key=lambda c: comparisons["authored_vs_reset0"][c]["mount_position_error_m"]
               + comparisons["authored_vs_reset0"][c]["mount_rotation_error_rad"])
    t0 = comparisons["authored_vs_reset0"][best]
    gates = {
        "t0_mount_position_within_tol": t0["mount_position_error_m"] <= ARGS.pos_tol,
        "t0_mount_rotation_within_tol": t0["mount_rotation_error_rad"] <= ARGS.rot_tol,
        "settle_drift_within_tol": (settle_drift is not None and settle_drift <= ARGS.pos_tol),
        "no_non_finite": non_finite_total == 0,
    }
    report = {
        "label": "TEST_PROXY P2 initialisation diagnostics",
        "variant": ARGS.variant,
        "physics_dt": DT,
        "gravity": [0.0, 0.0, 0.0],
        "solver": "TGS + external forces every iteration",
        "thresholds": {"position_m": ARGS.pos_tol, "rotation_rad": ARGS.rot_tol},
        "authored_usd": authored_layer,
        "body_names_physx": body_names,
        "joint_names_physx": joint_names,
        "dofs_physx": int(robot.data.joint_pos.shape[1]),
        "authored_poses": authored,
        "authored_joint_anchors": authored_anchors,
        "spawned_poses": spawned,
        "live_joint_anchors": live_anchors,
        "reset0_poses": reset0,
        "root_reset0": root_reset0,
        "reset1_poses": reset1,
        "comparisons": comparisons,
        "chassis_checks": chassis_checks,
        "settle_drift_m": settle_drift,
        "settle_mount_residual": settle_mount,
        "non_finite_total": non_finite_total,
        "self_consistent_convention": best,
        "gates": gates,
        "gate_result": "PASS" if all(gates.values()) else "FAIL",
        "settle_samples": [{"step": row["step"],
                            "chassis_pos": row["poses"]["Chassis"]["pos"],
                            "mount_wxyz": evaluate(row["poses"], "wxyz", offset),
                            "mount_xyzw": evaluate(row["poses"], "xyzw", offset)}
                           for row in settle[:: max(1, len(settle) // 6)]],
    }
    directory = os.path.dirname(os.path.abspath(ARGS.out))
    if directory:
        os.makedirs(directory, exist_ok=True)
    with open(ARGS.out, "w", encoding="utf-8") as handle:
        json.dump(report, handle, ensure_ascii=False, indent=1)
    brief = {
        "variant": ARGS.variant,
        "self_consistent_convention": best,
        "authored_vs_reset0": comparisons["authored_vs_reset0"],
        "chassis_reset0": chassis_checks["reset0"],
        "settle_drift_m": settle_drift,
        "gates": gates,
        "gate_result": report["gate_result"],
        "out": os.path.abspath(ARGS.out),
    }
    print(json.dumps(brief, ensure_ascii=False, indent=1))
    code = 0 if report["gate_result"] == "PASS" else 1
    print("PROCESS_EXIT_CODE=%d" % code, flush=True)
    # Kit's close() tears the application down and can replace the process status with 0,
    # which is exactly the "gate FAIL but EXIT=0" contract bug reported in issue #13.
    # Close first, then leave with the gate status via os._exit so nothing can override it.
    SIMULATION_APP.close()
    os._exit(code)


if __name__ == "__main__":
    main()
