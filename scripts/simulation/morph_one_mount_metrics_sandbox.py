"""Mount-metric sandbox: non-zero rigid mount offset, no initial geometry overlap.

Why this exists
---------------
The first PhysX round put a proxy chassis box on top of the real arm's base geometry.
Depenetration then moved the whole assembly and made the "vertical drift" number
uninterpretable.  This sandbox rebuilds the test with

  * a proxy chassis body whose collision cube is BELOW the mount plane, and
  * a proxy arm whose base body sits at an explicit, NON-ZERO mount offset,

so the rigid mount can be measured without an initial penetration artefact.

It also separates the phases the audit asked for: a settle phase with no commanded
motion, then a drive phase where a base twist is injected once.  Mount errors are
reported in the CHASSIS body frame (translation and relative rotation), not in world
coordinates.

This is a TEST_PROXY study.  Injecting a root velocity is NOT four-wheel-drive or
four-wheel-steer control: no wheel contact, no tyre friction and no actuator model is
involved, so nothing here can be read as a real drivetrain success rate.

Usage:
    python morph_one_mount_metrics_sandbox.py --mode translate --out t.json
    python morph_one_mount_metrics_sandbox.py --mode rotate    --out r.json
    python morph_one_mount_metrics_sandbox.py --mode combined  --out c.json
"""

from __future__ import annotations

import argparse
import json
import math
import os

import numpy as np

from isaaclab.app import AppLauncher

MOUNT_OFFSET = (0.12, -0.05, 0.20)   # chassis body origin -> arm base body origin
CHASSIS_HALF = (0.25, 0.175, 0.04)
# The collision cube sits BELOW the mount plane but ABOVE the ground plane: putting it
# below z=0 (the first attempt) made the static ground push the whole assembly away
# during settle, which is a second artefact that has nothing to do with the mount.
CHASSIS_CUBE_CENTRE = (0.0, 0.0, 0.06)
ARM_BASE_HALF = (0.04, 0.04, 0.04)
LINK1_HALF = (0.14, 0.025, 0.025)
LINK2_HALF = (0.11, 0.02, 0.02)
SETTLE_STEPS = 60
DRIVE_STEPS = 180
DT = 1.0 / 120.0
MODES = {
    "translate": (0.4, 0.0, 0.0),
    "rotate": (0.0, 0.0, 0.3),
    "combined": (0.4, 0.0, 0.3),
}

PARSER = argparse.ArgumentParser(description="TEST_PROXY mount-metric sandbox")
PARSER.add_argument("--mode", choices=sorted(MODES), default="translate")
PARSER.add_argument("--out", required=True)
PARSER.add_argument("--usd-out", default="")
PARSER.add_argument("--gains", choices=["none", "verified"], default="verified")
PARSER.add_argument("--gravity-on", action="store_true")
PARSER.add_argument("--mount-authoring", choices=["joint-local", "body-only"],
                    default="joint-local",
                    help="'joint-local' authors physics:localPos0 = mount offset; 'body-only' "
                         "leaves the fixed joint at default anchors and relies on body transforms")
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


# ----------------------------------------------------------------- rigid bodies
def _cube(stage, path, half, centre, mass, label):
    path = path if isinstance(path, Sdf.Path) else Sdf.Path(path)
    body = UsdGeom.Xform.Define(stage, path)
    prim = body.GetPrim()
    prim.SetCustomDataByKey("label", label)
    cube = UsdGeom.Cube.Define(stage, path.AppendChild("geometry"))
    cube.CreateSizeAttr(1.0)
    UsdGeom.Xformable(cube.GetPrim()).AddScaleOp().Set(
        Gf.Vec3f(2.0 * half[0], 2.0 * half[1], 2.0 * half[2]))
    UsdGeom.Xformable(cube.GetPrim()).AddTranslateOp().Set(Gf.Vec3f(*centre))
    UsdPhysics.RigidBodyAPI.Apply(prim)
    UsdPhysics.CollisionAPI.Apply(cube.GetPrim())
    mass_api = UsdPhysics.MassAPI.Apply(prim)
    mass_api.CreateMassAttr(mass)
    mass_api.CreateCenterOfMassAttr(Gf.Vec3f(0.0))
    return prim


def _revolute(stage, path, body0, body1, lower, upper, local_pos0):
    joint = UsdPhysics.RevoluteJoint.Define(stage, path)
    joint.CreateBody0Rel().SetTargets([body0])
    joint.CreateBody1Rel().SetTargets([body1])
    joint.CreateAxisAttr("Y")
    joint.CreateLowerLimitAttr(lower)
    joint.CreateUpperLimitAttr(upper)
    # anchor the joint where the two bodies already are, so setup adds no impulse
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


def author(path):
    """One proxy: chassis body at the origin, arm base body at MOUNT_OFFSET."""
    stage = Usd.Stage.CreateNew(path)
    UsdGeom.SetStageMetersPerUnit(stage, 1.0)
    UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)
    root = UsdGeom.Xform.Define(stage, "/Sandbox")
    stage.SetDefaultPrim(root.GetPrim())
    root.GetPrim().SetCustomDataByKey("label", "TEST_PROXY mount-metric sandbox")
    UsdPhysics.ArticulationRootAPI.Apply(root.GetPrim())

    chassis = _cube(stage, "/Sandbox/Chassis", CHASSIS_HALF, CHASSIS_CUBE_CENTRE, 15.0,
                    "TEST_PROXY chassis body (collision cube below the mount plane)")

    base_path = "/Sandbox/Arm/base_link"
    base = _cube(stage, base_path, ARM_BASE_HALF, (0.0, 0.0, 0.0), 2.0,
                 "TEST_PROXY arm base body at the mount offset")
    UsdGeom.Xformable(base).AddTranslateOp().Set(Gf.Vec3f(*MOUNT_OFFSET))

    link1_offset = (MOUNT_OFFSET[0] + 0.14, MOUNT_OFFSET[1], MOUNT_OFFSET[2])
    link1 = _cube(stage, "/Sandbox/Arm/link1", LINK1_HALF,
                  (0.0, 0.0, 0.0), 1.0, "TEST_PROXY upper link")
    UsdGeom.Xformable(link1).AddTranslateOp().Set(Gf.Vec3f(*link1_offset))

    link2_offset = (MOUNT_OFFSET[0] + 0.25, MOUNT_OFFSET[1], MOUNT_OFFSET[2])
    link2 = _cube(stage, "/Sandbox/Arm/link2", LINK2_HALF,
                  (0.0, 0.0, 0.0), 0.5, "TEST_PROXY forearm")
    UsdGeom.Xformable(link2).AddTranslateOp().Set(Gf.Vec3f(*link2_offset))

    _revolute(stage, "/Sandbox/Arm/joint1", base.GetPath(), link1.GetPath(), -90.0, 90.0,
              (0.14, 0.0, 0.0))
    _revolute(stage, "/Sandbox/Arm/joint2", link1.GetPath(), link2.GetPath(), -120.0, 120.0,
              (0.11, 0.0, 0.0))

    mount = UsdPhysics.FixedJoint.Define(stage, "/Sandbox/mount_fixed")
    mount.CreateBody0Rel().SetTargets([chassis.GetPath()])
    mount.CreateBody1Rel().SetTargets([base.GetPath()])
    # the anchor in the chassis frame IS the mount offset, so setup is consistent
    if ARGS.mount_authoring == "joint-local":
        mount.CreateLocalPos0Attr().Set(Gf.Vec3f(*MOUNT_OFFSET))
        mount.CreateLocalRot0Attr().Set(Gf.Quatf(1.0, 0.0, 0.0, 0.0))
        mount.CreateLocalPos1Attr().Set(Gf.Vec3f(0.0, 0.0, 0.0))
        mount.CreateLocalRot1Attr().Set(Gf.Quatf(1.0, 0.0, 0.0, 0.0))
    mount.GetPrim().SetCustomDataByKey("label", "rigid mount at a NON-ZERO offset")
    stage.GetRootLayer().Save()
    return {"usd": os.path.abspath(path), "mount_offset_m": list(MOUNT_OFFSET),
            "mount_authoring": ARGS.mount_authoring,
            "articulation_root": "/Sandbox"}


# --------------------------------------------------------------- quaternion math
def unit(quat):
    values = np.asarray(quat, dtype=float)
    norm = float(np.linalg.norm(values))
    if not np.isfinite(norm) or norm <= 0.0:
        raise ValueError("degenerate quaternion")
    return values / norm


def qmul(a, b):
    w, x, y, z = a
    p, u, v, t = b
    return np.array([w * p - x * u - y * v - z * t,
                     w * u + x * p + y * t - z * v,
                     w * v - x * t + y * p + z * u,
                     w * t + x * v - y * u + z * p])


def qinv(q):
    q = unit(q)
    return np.array([q[0], -q[1], -q[2], -q[3]])


def rotate(q, vec):
    q = unit(q)
    v = np.asarray(vec, dtype=float)
    t = 2.0 * np.cross(q[1:], v)
    return v + q[0] * t + np.cross(q[1:], t)


def relative_pose(p_chassis, q_chassis, p_body, q_body):
    inverse = qinv(q_chassis)
    return rotate(inverse, np.asarray(p_body, dtype=float) - np.asarray(p_chassis, dtype=float)), \
        unit(qmul(inverse, q_body))


def rotation_angle(a, b):
    dot = abs(float(np.dot(unit(a), unit(b))))
    return 2.0 * math.acos(max(-1.0, min(1.0, dot)))


# --------------------------------------------------------------------- run
def flatten(values):
    if hasattr(values, "detach"):
        values = values.detach().cpu().numpy()
    return np.asarray(values).reshape(-1).astype(float).tolist()


def read_gains(robot, joint_names):
    """Whatever the simulator will actually use; UNKNOWN when it cannot be read."""
    attempts = []
    for attribute in ("joint_stiffness", "joint_damping"):
        try:
            value = getattr(robot.data, attribute)
            attempts.append((attribute, [round(float(v), 6) for v in flatten(value)]))
        except Exception as exc:
            attempts.append((attribute, "unavailable: %s" % type(exc).__name__))
    stiffness = damping = None
    for name, value in attempts:
        if name == "joint_stiffness" and isinstance(value, list):
            stiffness = value
        if name == "joint_damping" and isinstance(value, list):
            damping = value
    if stiffness and damping:
        return {"status": "READ_FROM_ARTICULATION_DATA", "source": "robot.data.joint_*",
                "joint_names": joint_names, "stiffness": stiffness, "damping": damping}
    actuators = getattr(robot, "actuators", {}) or {}
    collected = {}
    for name, actuator in actuators.items():
        try:
            collected[name] = {
                "stiffness": [round(float(v), 6) for v in flatten(actuator.stiffness)],
                "damping": [round(float(v), 6) for v in flatten(actuator.damping)],
            }
        except Exception:
            continue
    if collected:
        return {"status": "READ_FROM_ACTUATOR_MODELS", "source": "robot.actuators",
                "joint_names": joint_names, "actuators": collected}
    return {"status": "UNKNOWN", "joint_names": joint_names, "attempts": attempts,
            "note": "neither ArticulationData nor actuator models exposed the gains"}


def build_sim():
    physics = PhysxCfg(solver_type=1, enable_external_forces_every_iteration=True)
    kwargs = {"dt": DT, "device": "cuda:0", "physics": physics}
    if not ARGS.gravity_on:
        kwargs["gravity"] = (0.0, 0.0, 0.0)
    sim = sim_utils.SimulationContext(sim_utils.SimulationCfg(**kwargs))
    ground = sim_utils.GroundPlaneCfg()
    ground.func("/World/ground", ground)
    return sim


def main():
    usd_out = ARGS.usd_out or ARGS.out.replace(".json", ".usda")
    authored = author(usd_out)
    sim = build_sim()
    spawn = sim_utils.UsdFileCfg(usd_path=authored["usd"])
    spawn.func("/World/Sandbox", spawn)
    actuators = {}
    if ARGS.gains == "verified":
        actuators = {"all": ImplicitActuatorCfg(joint_names_expr=[".*"], stiffness=300.0,
                                                damping=20.0)}
    robot = Articulation(ArticulationCfg(prim_path="/World/Sandbox", spawn=None,
                                         actuators=actuators))
    sim.reset()

    body_names = list(robot.body_names)
    joint_names = list(robot.joint_names)
    chassis_index = next(i for i, n in enumerate(body_names) if "Chassis" in n)
    base_index = next(i for i, n in enumerate(body_names) if "base_link" in n)

    def state():
        return (flatten(robot.data.body_pos_w[0, chassis_index]),
                flatten(robot.data.body_quat_w[0, chassis_index]),
                flatten(robot.data.body_pos_w[0, base_index]),
                flatten(robot.data.body_quat_w[0, base_index]))

    def yaw_of(quat):
        w, x, y, z = unit(quat)
        return math.atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))

    def step(count):
        for _ in range(count):
            robot.write_data_to_sim()
            sim.step()
            robot.update(sim.get_physics_dt())

    initial = state()
    initial_relative = relative_pose(*initial)
    initial_diagnostics = {
        "chassis_pos_w": initial[0],
        "chassis_quat_wxyz": initial[1],
        "arm_base_pos_w": initial[2],
        "arm_base_quat_wxyz": initial[3],
        "relative_position_m": [round(float(v), 9) for v in initial_relative[0]],
        "relative_quaternion_wxyz": [round(float(v), 9) for v in initial_relative[1]],
        "authored_mount_offset_m": list(MOUNT_OFFSET),
        "ground_clearance_m": round(float(min(initial[0][2], initial[2][2])), 6),
    }
    setup_error_m = float(np.linalg.norm(
        np.asarray(initial_relative[0]) - np.asarray(MOUNT_OFFSET)))
    setup_error_rad = rotation_angle(initial_relative[1], [1.0, 0.0, 0.0, 0.0])

    # ---- settle: no commanded motion
    settle_rows = []
    for index in range(SETTLE_STEPS):
        step(1)
        chassis_p, chassis_q, base_p, base_q = state()
        rel_p, rel_q = relative_pose(chassis_p, chassis_q, base_p, base_q)
        settle_rows.append({
            "step": index + 1,
            "mount_pos_error_m": float(np.linalg.norm(np.asarray(rel_p) - initial_relative[0])),
            "mount_rot_error_rad": rotation_angle(rel_q, initial_relative[1]),
            "chassis_pos": chassis_p,
        })

    drive_start = state()
    drive_start_relative = relative_pose(*drive_start)
    linear, angular, yaw = MODES[ARGS.mode]
    twist = torch.tensor([[linear, 0.0, 0.0, 0.0, 0.0, angular]], dtype=torch.float32,
                        device=robot.device)
    twist_written = True
    twist_error = None
    try:
        robot.write_root_velocity_to_sim(twist)
    except Exception as exc:
        twist_written = False
        twist_error = "%s: %s" % (type(exc).__name__, exc)

    drive_rows = []
    non_finite = 0
    for index in range(DRIVE_STEPS):
        step(1)
        chassis_p, chassis_q, base_p, base_q = state()
        rel_p, rel_q = relative_pose(chassis_p, chassis_q, base_p, base_q)
        row = {
            "step": index + 1,
            "chassis_pos": chassis_p,
            "chassis_yaw_rad": yaw_of(chassis_q),
            "root_yaw_rad": yaw_of(flatten(robot.data.root_quat_w[0])),
            "root_z": flatten(robot.data.root_pos_w[0])[2],
            "mount_pos_error_m": float(np.linalg.norm(np.asarray(rel_p) - drive_start_relative[0])),
            "mount_rot_error_rad": rotation_angle(rel_q, drive_start_relative[1]),
            "joint_pos": flatten(robot.data.joint_pos[0]),
            "joint_vel": flatten(robot.data.joint_vel[0]),
        }
        for key in ("chassis_pos", "joint_pos", "joint_vel"):
            if not all(math.isfinite(v) for v in row[key]):
                non_finite += 1
        drive_rows.append(row)

    start_chassis_p = np.asarray(drive_start[0], dtype=float)
    end_chassis_p = np.asarray(drive_rows[-1]["chassis_pos"], dtype=float)
    settle_drift = float(np.linalg.norm(
        np.asarray(settle_rows[-1]["chassis_pos"]) - np.asarray(initial[0])))
    vertical_deltas = [row["chassis_pos"][2] for row in drive_rows]

    report = {
        "label": "TEST_PROXY mount-metric sandbox; not Morph One hardware",
        "mode": ARGS.mode,
        "twist_command": {"linear_x_m_s": linear, "angular_z_rad_s": angular},
        "gravity_on": bool(ARGS.gravity_on),
        "solver": "TGS + external forces every iteration",
        "physics_dt": DT,
        "settle_steps": SETTLE_STEPS,
        "drive_steps": DRIVE_STEPS,
        "authored": authored,
        "body_names": body_names,
        "joint_names": joint_names,
        "num_bodies_physx": len(body_names),
        "num_joints_physx": len(joint_names),
        "dofs_physx": int(robot.data.joint_pos.shape[1]),
        "chassis_in_articulation": any("Chassis" in name for name in body_names),
        "initial_state": initial_diagnostics,
        "setup_mount_offset_error_m": round(setup_error_m, 9),
        "setup_mount_rotation_error_rad": round(setup_error_rad, 9),
        "twist_written": twist_written,
        "twist_write_error": twist_error,
        "settle": {
            "chassis_world_drift_m": round(settle_drift, 9),
            "max_mount_pos_error_m": round(max(r["mount_pos_error_m"] for r in settle_rows), 9),
            "max_mount_rot_error_rad": round(max(r["mount_rot_error_rad"] for r in settle_rows), 9),
        },
        "drive": {
            "chassis_translation_m": round(float(np.linalg.norm(end_chassis_p - start_chassis_p)), 6),
            "chassis_horizontal_m": round(float(np.linalg.norm(
                end_chassis_p[:2] - start_chassis_p[:2])), 6),
            "net_vertical_displacement_m": round(float(end_chassis_p[2] - start_chassis_p[2]), 9),
            "vertical_span_m": round(float(max(vertical_deltas) - min(vertical_deltas)), 9),
            "max_mount_pos_error_m": round(max(r["mount_pos_error_m"] for r in drive_rows), 9),
            "max_mount_rot_error_rad": round(max(r["mount_rot_error_rad"] for r in drive_rows), 9),
            "max_abs_joint_vel_rad_s": round(max(
                abs(v) for row in drive_rows for v in row["joint_vel"]), 6),
            "chassis_yaw_change_rad": round(
                float(drive_rows[-1]["chassis_yaw_rad"] - drive_rows[0]["chassis_yaw_rad"]), 6),
            "root_yaw_change_rad": round(
                float(drive_rows[-1]["root_yaw_rad"] - drive_rows[0]["root_yaw_rad"]), 6),
            "non_finite_samples": non_finite,
        },
        "effective_joint_gains": read_gains(robot, joint_names),
        "interpretation_limits": [
            "root-velocity injection is not wheel drive: no tyre contact, no friction",
            "nothing here is a four-wheel-drive or four-wheel-steer success rate",
            "proxy masses and inertias are arbitrary test values",
        ],
        "samples": drive_rows[:: max(1, DRIVE_STEPS // 10)],
        "settle_samples": settle_rows[:: max(1, SETTLE_STEPS // 6)],
    }
    report["acceptance"] = {
        "setup_offset_matches_authored_m": setup_error_m < 1e-6,
        "no_initial_penetration_drift": settle_drift < 1e-3,
        "rigid_mount_position_holds_m": report["drive"]["max_mount_pos_error_m"] < 1e-3,
        "rigid_mount_orientation_holds_rad": report["drive"]["max_mount_rot_error_rad"] < 1e-3,
        "no_non_finite": non_finite == 0,
    }
    directory = os.path.dirname(os.path.abspath(ARGS.out))
    if directory:
        os.makedirs(directory, exist_ok=True)
    with open(ARGS.out, "w", encoding="utf-8") as handle:
        json.dump(report, handle, ensure_ascii=False, indent=1)
    brief = {key: report[key] for key in (
        "mode", "num_bodies_physx", "dofs_physx", "chassis_in_articulation",
        "setup_mount_offset_error_m", "settle", "drive", "effective_joint_gains",
        "acceptance")}
    print(json.dumps(brief, ensure_ascii=False, indent=1))
    print("OUT", ARGS.out)
    SIMULATION_APP.close()
    return 0 if all(report["acceptance"].values()) else 1


if __name__ == "__main__":
    raise SystemExit(main())
