"""B1/B2: real PhysX stepping of a proxy mobile-base + mount + arm articulation.

This is a TEST PROXY study, not the Morph One robot.  It answers two questions with
actual PhysX stepping rather than USD inspection:

  1. With the ArticulationRootAPI left where the real B601-RS puts it (on an Xform
     that only contains the arm bodies), is a chassis body that is joined to the
     arm base by a fixed joint part of the SAME articulation, or does PhysX keep it
     outside?
  2. Does the same proxy become one articulation (chassis + arm) when the root API
     is placed above both?

Measured per variant: PhysX DOF/body counts, whether the base can be driven, the
fixed-mount error, joint states, and any non-finite/divergent step.

Usage:
    python morph_one_proxy_physx_probe.py --headless --variant arm_rooted --out a.json
    python morph_one_proxy_physx_probe.py --headless --variant whole_rooted --out b.json
    python morph_one_proxy_physx_probe.py --headless --variant real_arm --out c.json
"""

from __future__ import annotations

import argparse
import json
import math
import os

import numpy as np

from isaaclab.app import AppLauncher

PARSER = argparse.ArgumentParser(description="TEST_PROXY PhysX articulation probe")
PARSER.add_argument("--variant", required=True,
                    choices=["arm_rooted", "whole_rooted", "chassis_rooted",
                             "real_arm", "real_arm_mobile_root"])
PARSER.add_argument("--out", required=True)
PARSER.add_argument("--usd-out", default="")
PARSER.add_argument("--gravity-off", action="store_true",
                    help="zero gravity: isolates base drive / mount tracking from free fall")
PARSER.add_argument("--joint-step", action="store_true",
                    help="after the base drive, command joint1 by +0.1 rad and measure tracking")
PARSER.add_argument("--gains", choices=["none", "verified"], default="none",
                    help="'verified' applies the documented B601-RS runtime gains (300/20) "
                         "instead of letting Isaac Lab pick implicit defaults")
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

DT = 1.0 / 120.0
SETTLE_STEPS = 60
DRIVE_STEPS = 180
PROXY_NOTE = "TEST_PROXY: boxes, not the Morph One structure"
REAL_ARM = ("/home/T7/dgut/robot_sim/assets/robots/badminton_robot/rebot_b601_rs"
            "/usd/rebot_b601_rs.usd/ReBot_Arm_RS/ReBot_Arm_RS.usda")

# ------------------------------------------------------------------ authoring
def _cube(stage, path, size, translation, mass, label=None):
    path = path if isinstance(path, Sdf.Path) else Sdf.Path(path)
    xform = UsdGeom.Xform.Define(stage, path)
    prim = xform.GetPrim()
    if label:
        prim.SetCustomDataByKey("label", label)
    cube = UsdGeom.Cube.Define(stage, path.AppendChild("geometry"))
    cube.CreateSizeAttr(1.0)
    scale = UsdGeom.Xformable(cube.GetPrim()).AddScaleOp()
    scale.Set(Gf.Vec3f(*size))
    translate = UsdGeom.Xformable(cube.GetPrim()).AddTranslateOp()
    translate.Set(Gf.Vec3f(*translation))
    UsdPhysics.RigidBodyAPI.Apply(prim)
    UsdPhysics.CollisionAPI.Apply(cube.GetPrim())
    mass_api = UsdPhysics.MassAPI.Apply(prim)
    mass_api.CreateMassAttr(mass)
    mass_api.CreateCenterOfMassAttr(Gf.Vec3f(0.0))
    return prim


def _revolute(stage, path, body0, body1, lower, upper, axis="Y"):
    joint = UsdPhysics.RevoluteJoint.Define(stage, path)
    joint.CreateBody0Rel().SetTargets([body0])
    joint.CreateBody1Rel().SetTargets([body1])
    joint.CreateAxisAttr(axis)
    joint.CreateLowerLimitAttr(lower)
    joint.CreateUpperLimitAttr(upper)
    drive = UsdPhysics.DriveAPI.Apply(joint.GetPrim(), "angular")
    drive.CreateTypeAttr("force")
    drive.CreateStiffnessAttr(50.0)
    drive.CreateDampingAttr(5.0)
    drive.CreateMaxForceAttr(100.0)
    return joint.GetPrim()


def _fixed(stage, path, body0, body1):
    joint = UsdPhysics.FixedJoint.Define(stage, path)
    joint.CreateBody0Rel().SetTargets([body0])
    joint.CreateBody1Rel().SetTargets([body1])
    return joint.GetPrim()


def author_proxy(path, root_location):
    """Write one proxy assembly; only the articulation-root placement changes."""
    stage = Usd.Stage.CreateNew(path)
    UsdGeom.SetStageMetersPerUnit(stage, 1.0)
    UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)
    root = UsdGeom.Xform.Define(stage, "/Proxy")
    stage.SetDefaultPrim(root.GetPrim())
    root.GetPrim().SetCustomDataByKey("label", PROXY_NOTE)

    chassis = _cube(stage, "/Proxy/Chassis", (0.5, 0.35, 0.06), (0.0, 0.0, 0.10), 12.0,
                    "TEST_PROXY chassis body")
    base = _cube(stage, "/Proxy/Arm/base_link", (0.08, 0.08, 0.08), (0.0, 0.0, 0.17), 2.0,
                 "TEST_PROXY arm base")
    link1 = _cube(stage, "/Proxy/Arm/link1", (0.28, 0.05, 0.05), (0.14, 0.0, 0.17), 1.0,
                  "TEST_PROXY upper link")
    link2 = _cube(stage, "/Proxy/Arm/link2", (0.22, 0.04, 0.04), (0.14, 0.0, 0.17), 0.5,
                  "TEST_PROXY forearm")

    _revolute(stage, "/Proxy/Arm/joint1", base.GetPath(), link1.GetPath(), -90.0, 90.0)
    _revolute(stage, "/Proxy/Arm/joint2", link1.GetPath(), link2.GetPath(), -120.0, 120.0)
    mount = _fixed(stage, "/Proxy/mount_fixed", chassis.GetPath(), base.GetPath())
    mount.GetPrim().SetCustomDataByKey("label", "fixed mount chassis -> arm base")

    if root_location == "arm":
        target = "/Proxy/Arm"
    elif root_location == "whole":
        target = "/Proxy"
    elif root_location == "chassis":
        target = "/Proxy/Chassis"
    else:
        raise SystemExit("unknown root location: %s" % root_location)
    UsdPhysics.ArticulationRootAPI.Apply(stage.GetPrimAtPath(target))
    stage.GetRootLayer().Save()
    return {"usd": os.path.abspath(path), "articulation_root_prim": target}


def author_real_arm_mount(path, chassis_root_location):
    """Reference the real arm read-only and mount it on a proxy chassis body."""
    stage = Usd.Stage.CreateNew(path)
    UsdGeom.SetStageMetersPerUnit(stage, 1.0)
    UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)
    root = UsdGeom.Xform.Define(stage, "/Mobile")
    stage.SetDefaultPrim(root.GetPrim())
    root.GetPrim().SetCustomDataByKey(
        "label", "TEST_PROXY mobile base carrying the REAL B601-RS asset (read-only reference)")

    chassis = _cube(stage, "/Mobile/Chassis", (0.6, 0.4, 0.1), (0.0, 0.0, 0.25), 25.0,
                    "TEST_PROXY chassis body")
    arm = UsdGeom.Xform.Define(stage, "/Mobile/Arm")
    arm.GetPrim().GetReferences().AddReference(REAL_ARM)

    # retarget the arm's existing fixed root joint at the chassis body (override only)
    joint_path = None
    for prim in Usd.PrimRange(stage.GetPrimAtPath("/Mobile/Arm")):
        if prim.IsA(UsdPhysics.Joint) and prim.GetName() == "root_joint":
            joint_path = prim.GetPath()
            break
    if joint_path is None:
        raise SystemExit("real arm has no root_joint")
    over = stage.OverridePrim(joint_path)
    # retarget only body0; the arm keeps its own body1 (base_link) and its local poses
    UsdPhysics.FixedJoint(over).CreateBody0Rel().SetTargets([chassis.GetPath()])

    if chassis_root_location == "arm_geometry":
        pass  # leave the asset's own ArticulationRootAPI where it is
    elif chassis_root_location == "mobile":
        UsdPhysics.ArticulationRootAPI.Apply(root.GetPrim())
    elif chassis_root_location == "whole":
        UsdPhysics.ArticulationRootAPI.Apply(root.GetPrim())
    else:
        raise SystemExit("unknown root location: %s" % chassis_root_location)
    stage.GetRootLayer().Save()
    return {"usd": os.path.abspath(path),
            "overridden_joint": str(joint_path),
            "root_location": chassis_root_location}


# ------------------------------------------------------------------ measurement
def flatten(values):
    if hasattr(values, "detach"):
        values = values.detach().cpu().numpy()
    return np.asarray(values).reshape(-1).astype(float).tolist()


def _unit_quat_wxyz(quat):
    """Normalized world-frame quaternion; allow q and -q as same rotation."""
    values = np.asarray(quat, dtype=float)
    norm = float(np.linalg.norm(values))
    if not np.isfinite(norm) or norm <= 0.0:
        raise ValueError("non-finite or degenerate body quaternion")
    return values / norm


def _quat_mul_wxyz(a, b):
    w, x, y, z = a
    p, u, v, t = b
    return np.array([
        w*p-x*u-y*v-z*t,
        w*u+x*p+y*t-z*v,
        w*v-x*t+y*p+z*u,
        w*t+x*v-y*u+z*p,
    ], dtype=float)


def _relative_mount_pose(chassis_p, chassis_q, arm_p, arm_q):
    """Return arm-body pose in the chassis-body frame (position, quaternion)."""
    q = _unit_quat_wxyz(chassis_q)
    inverse_q = q * np.array([1.0, -1.0, -1.0, -1.0])
    relative_quat = _quat_mul_wxyz(inverse_q, _unit_quat_wxyz(arm_q))
    displacement = np.asarray(arm_p, dtype=float) - np.asarray(chassis_p, dtype=float)
    # Rotate displacement into chassis local coordinates.
    v = inverse_q[1:]
    t = 2.0 * np.cross(v, displacement)
    relative_pos = displacement + inverse_q[0] * t + np.cross(v, t)
    return relative_pos, _unit_quat_wxyz(relative_quat)


def _quat_difference_angle(a, b):
    """Minimal SO(3) rotation angle, accounting for q == -q equivalence."""
    dot = abs(float(np.dot(_unit_quat_wxyz(a), _unit_quat_wxyz(b))))
    return 2.0 * math.acos(max(-1.0, min(1.0, dot)))


def build_sim(gravity_off=False):
    physics = PhysxCfg(solver_type=1, enable_external_forces_every_iteration=True)
    kwargs = {"dt": DT, "device": "cuda:0", "physics": physics}
    if gravity_off:
        kwargs["gravity"] = (0.0, 0.0, 0.0)
    sim = sim_utils.SimulationContext(sim_utils.SimulationCfg(**kwargs))
    ground = sim_utils.GroundPlaneCfg()
    ground.func("/World/ground", ground)
    return sim


def run(usd_path, prim_path, out_path, gravity_off=False):
    sim = build_sim(gravity_off)
    spawn = sim_utils.UsdFileCfg(usd_path=usd_path)
    spawn.func(prim_path, spawn)
    if ARGS.gains == "verified":
        # Mirror of the repository values (simulation/badminton_scene/rs_physics.py +
        # scene_layout.REBOT_RS): stiffness 300, damping 20, TGS with external forces
        # every iteration.  Nothing else about the arm baseline is changed.
        actuators = {"all": ImplicitActuatorCfg(joint_names_expr=[".*"], stiffness=300.0,
                                                damping=20.0)}
    else:
        actuators = {}
    robot = Articulation(ArticulationCfg(prim_path=prim_path, spawn=None,
                                         actuators=actuators))
    sim.reset()

    body_names = list(robot.body_names)
    joint_names = list(robot.joint_names)
    view = robot.root_physx_view
    dof_count = int(robot.data.joint_pos.shape[1])
    try:
        applied_gains = {
            "stiffness": flatten(view.get_joint_stiffness()),
            "damping": flatten(view.get_joint_damping()),
        }
    except Exception as exc:  # pragma: no cover
        applied_gains = {"error": "%s: %s" % (type(exc).__name__, exc)}

    report = {
        "usd": usd_path,
        "prim_path": prim_path,
        "physics_dt": DT,
        "gravity_off": bool(gravity_off),
        "actuator_gains": ARGS.gains,
        "solver": "TGS + external forces every iteration",
        "body_names": body_names,
        "joint_names": joint_names,
        "num_bodies_physx": int(len(body_names)),
        "num_joints_physx": int(len(joint_names)),
        "dofs_physx": dof_count,
        "applied_joint_gains_from_physx": applied_gains,
        "expected_proxy_bodies": 4,
        "expected_proxy_dofs": 2,
        "chassis_in_articulation": any("Chassis" in name for name in body_names),
    }

    def body_index(fragment):
        for index, name in enumerate(body_names):
            if fragment in name:
                return index
        return None

    chassis_index = body_index("Chassis")
    base_index = body_index("base_link")

    def step(count):
        for _ in range(count):
            robot.write_data_to_sim()
            sim.step()
            robot.update(sim.get_physics_dt())

    def root_state():
        position = flatten(robot.data.root_pos_w[0])
        quat = flatten(robot.data.root_quat_w[0])
        w, x, y, z = quat
        yaw = math.atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))
        return position, quat, yaw

    step(SETTLE_STEPS)
    start_chassis = None if chassis_index is None else flatten(
        robot.data.body_pos_w[0, chassis_index])
    start_base = None if base_index is None else flatten(robot.data.body_pos_w[0, base_index])
    start_chassis_quat = None if chassis_index is None else flatten(
        robot.data.body_quat_w[0, chassis_index])
    start_base_quat = None if base_index is None else flatten(
        robot.data.body_quat_w[0, base_index])
    initial_mount_pose = (
        _relative_mount_pose(start_chassis, start_chassis_quat, start_base, start_base_quat)
        if start_chassis is not None and start_base is not None else None
    )
    start_pos, start_quat, start_yaw = root_state()

    # command a base twist: only meaningful when the base body is part of the articulation
    twist = torch.tensor([[0.4, 0.0, 0.0, 0.0, 0.0, 0.3]], dtype=torch.float32,
                        device=robot.device)
    try:
        robot.write_root_velocity_to_sim(twist)
        twist_applied = True
        twist_error = None
    except Exception as exc:  # pragma: no cover - depends on API version
        twist_applied = False
        twist_error = "%s: %s" % (type(exc).__name__, exc)

    samples = []
    non_finite = 0
    for index in range(DRIVE_STEPS):
        step(1)
        position, _quat, yaw = root_state()
        row = {
            "step": index + 1,
            "root_pos": position,
            "root_yaw": yaw,
            "root_lin_vel": flatten(robot.data.root_lin_vel_w[0]),
            "joint_pos": flatten(robot.data.joint_pos[0]),
            "joint_vel": flatten(robot.data.joint_vel[0]),
        }
        if chassis_index is not None:
            row["chassis_pos"] = flatten(robot.data.body_pos_w[0, chassis_index])
        if base_index is not None:
            row["arm_base_pos"] = flatten(robot.data.body_pos_w[0, base_index])
        if chassis_index is not None:
            row["chassis_quat_wxyz"] = flatten(robot.data.body_quat_w[0, chassis_index])
        if base_index is not None:
            row["arm_base_quat_wxyz"] = flatten(robot.data.body_quat_w[0, base_index])
        for key in ("root_pos", "root_lin_vel", "joint_pos", "joint_vel"):
            if not all(math.isfinite(v) for v in row[key]):
                non_finite += 1
        samples.append(row)

    joint_step = None
    if ARGS.joint_step and len(joint_names) > 0:
        dof = int(robot.data.joint_pos.shape[1])
        target = torch.zeros((1, dof), dtype=torch.float32, device=robot.device)
        target[0, 0] = 0.1
        track = []
        for _ in range(60):
            robot.set_joint_position_target(target)
            step(1)
            track.append({
                "q0": flatten(robot.data.joint_pos[0])[0],
                "qd0": flatten(robot.data.joint_vel[0])[0],
            })
        joint_step = {
            "joint_name": joint_names[0],
            "command_rad": 0.1,
            "final_joint_rad": round(float(track[-1]["q0"]), 6),
            "steady_error_rad": round(abs(0.1 - float(track[-1]["q0"])), 6),
            "max_abs_vel_rad_s": round(max(abs(row["qd0"]) for row in track), 6),
            "non_finite": sum(1 for row in track if not math.isfinite(row["q0"])),
        }

    end_root = samples[-1]["root_pos"]
    mount_translation_errors = []
    mount_rotation_errors = []
    if initial_mount_pose is not None:
        original_relative_pos, original_relative_quat = initial_mount_pose
        for row in samples:
            measured_pos, measured_quat = _relative_mount_pose(
                row["chassis_pos"], row["chassis_quat_wxyz"],
                row["arm_base_pos"], row["arm_base_quat_wxyz"],
            )
            mount_translation_errors.append(
                float(np.linalg.norm(measured_pos - original_relative_pos))
            )
            mount_rotation_errors.append(
                _quat_difference_angle(measured_quat, original_relative_quat)
            )
    report.update({
        "twist_command_xy_yaw": [0.4, 0.0, 0.3],
        "twist_written": twist_applied,
        "twist_write_error": twist_error,
        "root_start_pos": start_pos,
        "root_start_quat_wxyz": start_quat,
        "root_end_pos": end_root,
        "root_translation_m": round(float(np.linalg.norm(
            np.array(end_root[:3]) - np.array(start_pos[:3]))), 6),
        "root_horizontal_travel_m": round(float(np.linalg.norm(
            np.array(end_root[:2]) - np.array(start_pos[:2]))), 6),
        "root_end_z_m": round(float(end_root[2]), 6),
        # Absolute root z is NOT a vertical drift. Compare with the measured
        # post-settling start, and expose the transient range separately.
        "root_vertical_delta_after_settle_m": round(float(end_root[2] - start_pos[2]), 6),
        "root_vertical_span_during_drive_m": round(float(
            max(row["root_pos"][2] for row in samples)
            - min(row["root_pos"][2] for row in samples)), 6),
        "root_min_z_m": round(float(min(row["root_pos"][2] for row in samples)), 6),
        "max_fixed_mount_local_translation_error_m": (
            round(max(mount_translation_errors), 6) if mount_translation_errors else None
        ),
        "max_fixed_mount_relative_rotation_error_rad": (
            round(max(mount_rotation_errors), 6) if mount_rotation_errors else None
        ),
        # Backward-compatible old metric: world-frame offset difference only.
        # It cannot certify a nonzero-offset rigid mount under chassis rotation.
        "max_fixed_mount_error_m": round(float(max(
            np.linalg.norm(
                (np.array(row["arm_base_pos"]) - np.array(row["chassis_pos"]))
                - (np.array(start_base) - np.array(start_chassis)))
            for row in samples
            if "arm_base_pos" in row and "chassis_pos" in row
        )), 6) if (chassis_index is not None and base_index is not None) else None,
        "root_yaw_change_rad": round(float(samples[-1]["root_yaw"] - start_yaw), 6),
        "chassis_moved_m": (
            None if (chassis_index is None or start_chassis is None)
            else round(float(np.linalg.norm(
                np.array(samples[-1]["chassis_pos"][:3]) - np.array(start_chassis[:3]))), 6)
        ),
        "arm_base_moved_m": (
            None if (base_index is None or start_base is None)
            else round(float(np.linalg.norm(
                np.array(samples[-1]["arm_base_pos"][:3]) - np.array(start_base[:3]))), 6)
        ),
        "fixed_mount_relative_error_m": (
            None if (chassis_index is None or base_index is None)
            else round(float(np.linalg.norm(
                (np.array(samples[-1]["arm_base_pos"]) - np.array(samples[-1]["chassis_pos"]))
                - (np.array(start_base) - np.array(start_chassis)))), 6)
        ),
        "max_abs_joint_pos": (
            round(max(abs(v) for row in samples for v in row["joint_pos"]), 6)
            if samples and samples[0]["joint_pos"] else None
        ),
        "max_abs_joint_vel": (
            round(max(abs(v) for row in samples for v in row["joint_vel"]), 6)
            if samples and samples[0]["joint_vel"] else None
        ),
        "joint_step_result": joint_step,
        "non_finite_samples": non_finite,
        "diverged": bool(non_finite > 0),
        "sample_stride": max(1, DRIVE_STEPS // 12),
        "samples": samples[:: max(1, DRIVE_STEPS // 12)],
    })
    with open(out_path, "w", encoding="utf-8") as handle:
        json.dump(report, handle, ensure_ascii=False, indent=1)
    brief = {key: report[key] for key in (
        "num_bodies_physx", "num_joints_physx", "dofs_physx", "chassis_in_articulation",
        "body_names", "twist_written", "root_translation_m", "chassis_moved_m",
        "arm_base_moved_m", "fixed_mount_relative_error_m",
        "max_fixed_mount_local_translation_error_m",
        "max_fixed_mount_relative_rotation_error_rad",
        "root_vertical_delta_after_settle_m", "max_abs_joint_vel",
        "non_finite_samples", "diverged")}
    print(json.dumps(brief, ensure_ascii=False, indent=1))
    print("OUT", out_path)
    return 0


if __name__ == "__main__":
    usd_out = ARGS.usd_out or ARGS.out.replace(".json", ".usda")
    if ARGS.variant.startswith("real_arm"):
        root_location = "arm_geometry" if ARGS.variant == "real_arm" else "mobile"
        authoring = author_real_arm_mount(usd_out, root_location)
        prim_path = "/World/Mobile"
    else:
        root_location = {"arm_rooted": "arm", "whole_rooted": "whole",
                         "chassis_rooted": "chassis"}[ARGS.variant]
        authoring = author_proxy(usd_out, root_location)
        prim_path = "/World/Proxy"
    print("AUTHORING", json.dumps(authoring, ensure_ascii=False))
    code = run(authoring["usd"], prim_path, ARGS.out, gravity_off=ARGS.gravity_off)
    SIMULATION_APP.close()
    raise SystemExit(code)
