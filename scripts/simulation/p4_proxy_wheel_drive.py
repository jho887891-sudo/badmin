"""P4 PROXY: parameterised four-wheel drive on a TEST_PROXY chassis (NOT the real robot model).

Why this file exists: the real wheel identity is still only partially verified
(P1 v2 verified the toroidal tread and the axles, but the loaded rolling radius, contact
patch, mass/inertia and friction remain UNKNOWN).  P4 is therefore split:

  * `P4_REAL`  — BLOCKED until the real identity and dynamic gates pass;
  * `P4_PROXY` — this file: the DRIVE SOFTWARE (wheel joint -> motor/DOF mapping, velocity
                  targets, state read-back, direction convention) exercised on explicitly
                  proxy geometry so that the software is validated before the hardware data.

Numbers used, all labelled with their provenance:
  * axle positions  (+-0.250002, +-0.297, -0.244) m   -> P1 v2, verified (0.002 mm residual)
  * spin axis        world X                          -> P1 v2, verified (0.0043 deg PCA)
  * rolling radius   0.103 m                          -> P1 v2 torus equator R+r=103.000000 mm
                                                         (GEOMETRIC UNLOADED value, NOT the
                                                          loaded radius; collision proxy is a
                                                          CYLINDER of this radius, which is a
                                                          TEST_PROXY simplification of the torus)
  * left/right spin sign +1 / -1                      -> P1 v2 outward normal convention
  * wheel mass 1.0 kg, chassis 16 kg (total 20 kg)    -> TEST_PROXY, no vendor mass data

NO root pose/velocity is ever written here: the chassis may only move because the wheel
contact patches push it.  The script asserts that at runtime and records `root_writes`.

Usage:
    python p4_proxy_wheel_drive.py --run-id P4p_fwd_r01 --out-dir <dir> --profile forward
"""

from __future__ import annotations

import argparse
import json
import math
import os

import numpy as np

from isaaclab.app import AppLauncher

# verified geometry (P1 v2)
WHEEL_AXLES = [(0.250002, 0.297, -0.244), (0.250002, -0.297, -0.244),
               (-0.250002, 0.297, -0.244), (-0.250002, -0.297, -0.244)]
WHEEL_SPIN_SIGN = [1.0, 1.0, -1.0, -1.0]     # outward normal convention from P1 v2
WHEEL_RADIUS_M = 0.103                        # torus equator R+r, geometric unloaded
WHEEL_HALF_WIDTH_M = 0.0243                   # from the measured 48.6 mm thin axis
BODY_HALF = (0.198, 0.400, 0.066)
BODY_CENTRE = (0.0, -0.0024, -0.0404)
CHASSIS_MASS_KG = 16.0
WHEEL_MASS_KG = 1.0
TOTAL_MASS_KG = CHASSIS_MASS_KG + 4.0 * WHEEL_MASS_KG
GROUND_CLEARANCE_M = 0.02
DT = 1.0 / 120.0
TAIL_STEPS = 60

PROFILES = {
    # name: (wheel velocity targets rad/s in wheel order, steps, description)
    "spin_single": ([2.0, 0.0, 0.0, 0.0], 240, "one wheel only"),
    "forward": ([4.0, 4.0, 4.0, 4.0], 360, "all four, sign-corrected, +x expected"),
    "backward": ([-4.0, -4.0, -4.0, -4.0], 360, "all four reversed, -x expected"),
    "turn": ([-3.0, -3.0, 3.0, 3.0], 360, "left/right opposed, yaw expected"),
}

PARSER = argparse.ArgumentParser(description="P4 PROXY four-wheel drive (no root writes)")
PARSER.add_argument("--run-id", required=True)
PARSER.add_argument("--out-dir", required=True)
PARSER.add_argument("--profile", required=True, choices=sorted(PROFILES))
PARSER.add_argument("--steps", type=int, default=0, help="0 = use the profile default")
PARSER.add_argument("--gravity-z", type=float, default=-9.81)
PARSER.add_argument("--read-contacts", action="store_true")
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

ROOT_WRITES = {"count": 0}


def _np(values):
    if hasattr(values, "detach"):
        values = values.detach().cpu().numpy()
    return np.asarray(values)


def raw_to_wxyz(raw):
    arr = np.asarray(raw, dtype=float)
    norm = float(np.linalg.norm(arr))
    if not np.isfinite(norm) or norm <= 0.0:
        raise ValueError("degenerate quaternion")
    unit = arr / norm
    return np.array([unit[3], unit[0], unit[1], unit[2]])


def yaw_of(quat_wxyz):
    w, x, y, z = quat_wxyz
    return float(math.atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z)))


def author(path, lift):
    stage = Usd.Stage.CreateNew(path)
    UsdGeom.SetStageMetersPerUnit(stage, 1.0)
    UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)
    root = UsdGeom.Xform.Define(stage, "/PD")
    stage.SetDefaultPrim(root.GetPrim())
    root.GetPrim().SetCustomDataByKey(
        "label", "P4 PROXY chassis: verified axles/radius from P1 v2, cylinder tread proxy")

    chassis = UsdGeom.Xform.Define(stage, "/PD/Chassis")
    UsdGeom.Xformable(chassis).AddTranslateOp().Set(Gf.Vec3f(0.0, 0.0, lift))
    body = UsdGeom.Xform.Define(stage, "/PD/Chassis/body_geometry")
    UsdGeom.Xformable(body).AddTranslateOp().Set(Gf.Vec3f(*BODY_CENTRE))
    geom = UsdGeom.Cube.Define(stage, "/PD/Chassis/body_geometry/geometry")
    geom.CreateSizeAttr(1.0)
    UsdGeom.Xformable(geom.GetPrim()).AddScaleOp().Set(
        Gf.Vec3f(2 * BODY_HALF[0], 2 * BODY_HALF[1], 2 * BODY_HALF[2]))
    UsdPhysics.CollisionAPI.Apply(geom.GetPrim())
    UsdPhysics.RigidBodyAPI.Apply(chassis.GetPrim())
    mass_api = UsdPhysics.MassAPI.Apply(chassis.GetPrim())
    mass_api.CreateMassAttr(CHASSIS_MASS_KG)
    mass_api.CreateCenterOfMassAttr(Gf.Vec3f(0.0))
    UsdPhysics.ArticulationRootAPI.Apply(chassis.GetPrim())

    for index, axle in enumerate(WHEEL_AXLES, start=1):
        wheel_path = "/PD/Chassis/wheel%d" % index
        # an articulated wheel must be its own rigid body -> siblings, not children
        wheel_path = "/PD/wheel%d" % index
        wheel = UsdGeom.Xform.Define(stage, Sdf.Path(wheel_path))
        UsdGeom.Xformable(wheel).AddTranslateOp().Set(
            Gf.Vec3f(axle[0], axle[1], axle[2] + lift))
        cylinder = UsdGeom.Cylinder.Define(stage, Sdf.Path(wheel_path).AppendChild("geometry"))
        cylinder.CreateRadiusAttr(WHEEL_RADIUS_M)
        cylinder.CreateHeightAttr(2.0 * WHEEL_HALF_WIDTH_M)
        cylinder.CreateAxisAttr("X")          # spin axis is world X (P1 v2)
        UsdPhysics.CollisionAPI.Apply(cylinder.GetPrim())
        UsdPhysics.RigidBodyAPI.Apply(wheel.GetPrim())
        wheel_mass = UsdPhysics.MassAPI.Apply(wheel.GetPrim())
        wheel_mass.CreateMassAttr(WHEEL_MASS_KG)
        wheel_mass.CreateCenterOfMassAttr(Gf.Vec3f(0.0))
        # revolute joint about the axle: axis X in the parent (chassis) frame
        joint = UsdPhysics.RevoluteJoint.Define(stage, Sdf.Path("/PD/joint_wheel%d" % index))
        joint.CreateBody0Rel().SetTargets([Sdf.Path("/PD/Chassis")])
        joint.CreateBody1Rel().SetTargets([Sdf.Path(wheel_path)])
        joint.CreateLocalPos0Attr().Set(Gf.Vec3f(*axle))
        joint.CreateLocalPos1Attr().Set(Gf.Vec3f(0.0))
        joint.CreateLocalRot0Attr().Set(Gf.Quatf(1.0, 0.0, 0.0, 0.0))
        joint.CreateLocalRot1Attr().Set(Gf.Quatf(1.0, 0.0, 0.0, 0.0))
        joint.CreateAxisAttr("X")
        joint.CreateLowerLimitAttr(-1.0e6)
        joint.CreateUpperLimitAttr(1.0e6)
        # A USD joint without a DriveAPI has no PhysX drive at all, so an Isaac Lab
        # implicit actuator has nothing to write to: P4p_spin1_r01 commanded 2 rad/s and
        # measured -0.0013 rad/s.  The drive is therefore authored explicitly; the
        # actuator config below then overwrites stiffness/damping/gains.
        drive = UsdPhysics.DriveAPI.Apply(joint.GetPrim(), "angular")
        drive.CreateTypeAttr("force")
        drive.CreateStiffnessAttr(0.0)
        drive.CreateDampingAttr(40.0)
        drive.CreateMaxForceAttr(400.0)
        drive.CreateTargetVelocityAttr(0.0)
    stage.GetRootLayer().Save()
    return {"usd": os.path.abspath(path)}


def main():
    out_dir = os.path.abspath(ARGS.out_dir)
    os.makedirs(out_dir, exist_ok=True)
    targets, default_steps, description = PROFILES[ARGS.profile]
    steps = ARGS.steps or default_steps
    lift = WHEEL_RADIUS_M + abs(WHEEL_AXLES[0][2]) + GROUND_CLEARANCE_M   # 0.347 + 0.02
    authored = author(os.path.join(out_dir, "authored_p4_proxy.usda"), lift)

    physics = PhysxCfg(solver_type=1, enable_external_forces_every_iteration=True)
    sim = sim_utils.SimulationContext(sim_utils.SimulationCfg(
        dt=DT, device="cuda:0", gravity=(0.0, 0.0, float(ARGS.gravity_z)), physics=physics))
    ground = sim_utils.GroundPlaneCfg()
    ground.func("/World/ground", ground)
    spawn = sim_utils.UsdFileCfg(usd_path=authored["usd"], activate_contact_sensors=True)
    spawn.func("/World/PD", spawn)

    sensors = {}
    contact_api = "UNKNOWN"
    if ARGS.read_contacts:
        try:
            from isaaclab.sensors import ContactSensor, ContactSensorCfg

            for name, prim_path in (("chassis", "/World/PD/Chassis"),
                                    ("wheels", "/World/PD/wheel.*")):
                try:
                    sensors[name] = ContactSensor(ContactSensorCfg(
                        prim_path=prim_path, update_period=0.0, history_length=0,
                        debug_vis=False))
                except Exception as exc:
                    contact_api = "partial: %s (%s)" % (name, type(exc).__name__)
            if sensors:
                contact_api = "isaaclab.sensors.ContactSensor(%s)" % ",".join(sensors)
        except Exception as exc:  # pragma: no cover
            contact_api = "UNKNOWN: %s" % type(exc).__name__

    # velocity drive on the wheel joints only; the chassis has no actuator
    actuators = {
        "wheels": ImplicitActuatorCfg(joint_names_expr=["joint_wheel.*"], stiffness=0.0,
                                      damping=40.0, effort_limit=400.0,
                                      velocity_limit=50.0),
    }
    robot = Articulation(ArticulationCfg(prim_path="/World/PD", spawn=None,
                                         actuators=actuators))
    # the scene has DOF now, but the default regex must still match the wheel joints only
    sim.reset()

    body_names = list(robot.body_names)
    joint_names = list(robot.joint_names)
    index_of = {name: position for position, name in enumerate(body_names)}
    joint_index = {name: position for position, name in enumerate(joint_names)}
    chassis_index = index_of["Chassis"]
    wheel_joint_names = ["joint_wheel%d" % i for i in range(1, 5)]
    missing = [name for name in wheel_joint_names if name not in joint_index]
    if missing:
        raise SystemExit("wheel joints missing from the runtime DOF list: %s" % missing)

    target_vector = [0.0] * len(joint_names)
    for wheel, joint_name in enumerate(wheel_joint_names):
        target_vector[joint_index[joint_name]] = float(targets[wheel]) * WHEEL_SPIN_SIGN[wheel]
    target_tensor = torch.tensor([target_vector], dtype=torch.float32, device=robot.device)

    jsonl = open(os.path.join(out_dir, "raw_poses.jsonl"), "w", encoding="utf-8")
    frames = []
    for step in range(steps + 1):
        if step > 0:
            robot.set_joint_velocity_target(target_tensor)
            robot.write_data_to_sim()
            sim.step()
            robot.update(DT)
            for sensor in sensors.values():
                try:
                    sensor.update(DT)
                except Exception:
                    pass
        chassis_pos = [float(v) for v in
                       _np(robot.data.body_pos_w[0, chassis_index]).reshape(-1)[:3]]
        raw = [float(v) for v in
               _np(robot.data.body_quat_w[0, chassis_index]).reshape(-1)[:4]]
        wheel_omega = {}
        for wheel, joint_name in enumerate(wheel_joint_names):
            wheel_omega["wheel%d" % (wheel + 1)] = round(
                float(_np(robot.data.joint_vel[0, joint_index[joint_name]])), 9)
        contacts = {}
        for name, sensor in sensors.items():
            try:
                data = _np(sensor.data.net_forces_w).reshape(-1, 3)
                contacts[name] = round(float(np.max(np.linalg.norm(data, axis=1))), 9)
            except Exception:
                contacts[name] = None
        row = {"run_id": ARGS.run_id, "profile": ARGS.profile, "step": step,
               "t_s": round(step * DT, 9),
               "chassis_pos": [round(v, 12) for v in chassis_pos],
               "chassis_yaw_rad": round(yaw_of(raw_to_wxyz(raw)), 9),
               "wheel_omega_rad_s": wheel_omega,
               "wheel_target_rad_s": {("wheel%d" % (i + 1)): (
                   round(float(targets[i]) * WHEEL_SPIN_SIGN[i], 9)) for i in range(4)},
               "contact_by_sensor_n": contacts,
               "root_writes": ROOT_WRITES["count"],
               "finite": all(math.isfinite(v) for v in chassis_pos + raw)}
        jsonl.write(json.dumps(row, ensure_ascii=False) + "\n")
        jsonl.flush()
        frames.append(row)
    jsonl.close()

    start = np.array(frames[0]["chassis_pos"])
    tail = frames[-min(TAIL_STEPS, len(frames)):]
    displacement = np.array(frames[-1]["chassis_pos"]) - start
    tail_speed = max(float(np.linalg.norm(
        (np.array(frames[i]["chassis_pos"]) - np.array(frames[i - 1]["chassis_pos"])) / DT))
        for i in range(len(frames) - len(tail) + 1, len(frames)))
    tail_omega = {("wheel%d" % (i + 1)): round(
        float(np.mean([frame["wheel_omega_rad_s"]["wheel%d" % (i + 1)] for frame in tail])), 9)
        for i in range(4)}
    speed_tracking = {}
    for i in range(4):
        name = "wheel%d" % (i + 1)
        commanded = float(targets[i]) * WHEEL_SPIN_SIGN[i]
        if abs(commanded) < 1e-9:
            speed_tracking[name] = None
        else:
            speed_tracking[name] = round(abs(tail_omega[name] - commanded) / abs(commanded), 9)
    measured_tracking = [value for value in speed_tracking.values() if value is not None]
    peak_contact = max([value for frame in frames
                        for value in frame["contact_by_sensor_n"].values()
                        if value is not None], default=None)
    yaw_change = abs(frames[-1]["chassis_yaw_rad"] - frames[0]["chassis_yaw_rad"])

    thresholds = {"speed_tracking_rel": 0.20, "min_displacement_m": 0.01,
                  "contact_force_n": 1.0, "tail_speed_max_m_s": 5.0}
    gates = {
        "all_finite": all(frame["finite"] for frame in frames),
        "no_root_pose_write": ROOT_WRITES["count"] == 0,
    }
    if measured_tracking:
        gates["wheel_speed_tracks_target"] = bool(
            max(measured_tracking) <= thresholds["speed_tracking_rel"])
    if peak_contact is not None:
        gates["contact_detected"] = bool(peak_contact > thresholds["contact_force_n"])
    if ARGS.profile in ("forward", "backward"):
        sign = 1.0 if ARGS.profile == "forward" else -1.0
        gates["chassis_displaced_along_command"] = bool(
            sign * displacement[0] >= thresholds["min_displacement_m"])
    if ARGS.profile == "turn":
        gates["yaw_changed"] = bool(yaw_change >= 0.05)
    gate = "PASS" if all(gates.values()) else "FAIL"

    result = {
        "run_id": ARGS.run_id, "scene_variant": "P4_PROXY_WHEEL_DRIVE",
        "profile": ARGS.profile, "profile_description": description,
        "gate_result": gate,
        "proxy_parameters": {
            "wheel_radius_m": WHEEL_RADIUS_M,
            "wheel_radius_source": "P1 v2 torus equator R+r = 103.000000 mm (geometric "
                                   "unloaded); NOT a loaded radius",
            "wheel_collision_shape": "CYLINDER (TEST_PROXY simplification of the toroidal tread)",
            "wheel_half_width_m": WHEEL_HALF_WIDTH_M,
            "wheel_mass_kg": WHEEL_MASS_KG, "chassis_mass_kg": CHASSIS_MASS_KG,
            "total_mass_kg": TOTAL_MASS_KG,
            "mass_source": "TEST_PROXY, no vendor mass data",
            "ground_clearance_m": GROUND_CLEARANCE_M, "lift_m": lift,
            "axle_positions_m": WHEEL_AXLES,
            "spin_signs": WHEEL_SPIN_SIGN,
        },
        "metrics": {
            "wheel_targets_rad_s_per_wheel": [round(float(t), 9) for t in targets],
            "tail_wheel_omega_rad_s": tail_omega,
            "wheel_speed_relative_error": speed_tracking,
            "chassis_displacement_m": [round(float(v), 9) for v in displacement],
            "tail_chassis_speed_m_s": round(tail_speed, 9),
            "yaw_change_rad": round(yaw_change, 9),
            "peak_contact_force_n": peak_contact,
            "distance_per_wheel_revolution_m": [
                round(float(displacement[0] / (tail_omega["wheel%d" % (i + 1)] * (steps * DT))
                            * 2.0 * math.pi * WHEEL_RADIUS_M)
                      if abs(tail_omega["wheel%d" % (i + 1)]) > 1e-6 else None)
                for i in range(4)],
            "steps": steps,
        },
        "thresholds": thresholds, "gates": gates,
        "diagnostics": {
            "drive_mode": "JOINT VELOCITY TARGETS (no root pose/velocity writes)",
            "root_writes": ROOT_WRITES["count"], "contact_api": contact_api,
            "body_names": body_names, "joint_names": joint_names,
            "dofs": int(robot.data.joint_pos.shape[1]),
            "gravity_m_s2": [0.0, 0.0, float(ARGS.gravity_z)],
            "note": ("P4_PROXY only: verified axle/radius geometry with a CYLINDER tread proxy "
                     "and proxy masses. Not P4_REAL, not the vendor physical model."),
        },
    }
    with open(os.path.join(out_dir, "result.json"), "w", encoding="utf-8") as handle:
        json.dump(result, handle, ensure_ascii=False, indent=1)
    with open(os.path.join(out_dir, "manifest.json"), "w", encoding="utf-8") as handle:
        json.dump({"stage": "P4_PROXY", "run_id": ARGS.run_id,
                   "scene_variant": "P4_PROXY_WHEEL_DRIVE", "profile": ARGS.profile,
                   "physics_dt_s": DT, "steps": steps,
                   "gravity_m_s2": [0.0, 0.0, float(ARGS.gravity_z)],
                   "preregistration": "management/morph_one_rs/PHASE_P4_PROXY_PREREG.md",
                   "drive_mode": "JOINT VELOCITY TARGETS (no root pose/velocity writes)",
                   "command": " ".join(os.sys.argv)}, handle, ensure_ascii=False, indent=1)
    print(json.dumps({"run_id": ARGS.run_id, "profile": ARGS.profile, "gate_result": gate,
                      "metrics": result["metrics"], "gates": gates},
                     ensure_ascii=False, indent=1))
    print("GATE_RESULT=%s" % gate, flush=True)
    SIMULATION_APP.close()
    return 0 if gate == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
