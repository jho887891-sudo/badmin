"""P2-07/P2-08: REAL B601-RS arm carried by the CORRECTED four-wheel compound chassis.

The previous G7 run (G7_r01) proved only that the real arm mounts cleanly on a single-box
proxy chassis at g=0.  P2-07 asks a different question: does the real arm survive in the
SAME PhysX scene as the corrected four-wheel compound collider, without self collision,
without a double-counted mass and without wrong rigid-body nesting - and does that whole
assembly support its own weight under gravity?

Two gate modes, never mixed up:
  * --gate-mode static   (g=0)  t0 world pose, mount residual, MAX drift over all steps
  * --gate-mode landing  (g<0)  pre-registered landing/load-bearing gates + mount stability
                                under load + arm-must-not-touch-ground

The real arm asset is referenced READ-ONLY and its sha256 is recorded before and after.

Usage:
    python p2_wc_g7b_arm_with_wheels.py --run-id G7B_static_r01 --out-dir <dir> \\
        --lift 0.40 --steps 120 --gate-mode static --gravity-z 0 --read-contacts
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os

import numpy as np

from isaaclab.app import AppLauncher

REAL_ARM = ("/home/T7/dgut/robot_sim/assets/robots/badminton_robot/rebot_b601_rs"
            "/usd/rebot_b601_rs.usd/ReBot_Arm_RS/ReBot_Arm_RS.usda")
WHEEL_CENTRES = [(0.250002, -0.297, -0.244), (0.250002, 0.297, -0.244),
                 (-0.250002, -0.297, -0.244), (-0.250002, 0.297, -0.244)]
WHEEL_HALF = (0.0243, 0.0789, 0.0900)
BODY_HALF = (0.198, 0.400, 0.066)
BODY_CENTRE = (0.0, -0.0024, -0.0404)
CHASSIS_MASS_KG = 20.0
DT = 1.0 / 120.0
TAIL_STEPS = 30

PARSER = argparse.ArgumentParser(description="P2-07 real arm + corrected four-wheel chassis")
PARSER.add_argument("--run-id", required=True)
PARSER.add_argument("--out-dir", required=True)
PARSER.add_argument("--lift", type=float, default=0.40)
PARSER.add_argument("--steps", type=int, default=120)
PARSER.add_argument("--gravity-z", type=float, default=0.0)
PARSER.add_argument("--gate-mode", choices=("static", "landing"), default="static")
PARSER.add_argument("--read-contacts", action="store_true")
AppLauncher.add_app_launcher_args(PARSER)
ARGS = PARSER.parse_args()
LAUNCHER = AppLauncher(ARGS)
SIMULATION_APP = LAUNCHER.app

import isaaclab.sim as sim_utils  # noqa: E402
from isaaclab.assets import Articulation, ArticulationCfg  # noqa: E402
from isaaclab.actuators import ImplicitActuatorCfg  # noqa: E402
from isaaclab_physx.physics import PhysxCfg  # noqa: E402
from pxr import Gf, Sdf, Usd, UsdGeom, UsdPhysics  # noqa: E402
from p2_landing_gates import compute_landing_gates  # noqa: E402


def _np(values):
    if hasattr(values, "detach"):
        values = values.detach().cpu().numpy()
    return np.asarray(values)


def sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def unit(values):
    arr = np.asarray(values, dtype=float)
    norm = float(np.linalg.norm(arr))
    if not np.isfinite(norm) or norm <= 0.0:
        raise ValueError("degenerate quaternion")
    return arr / norm


def raw_to_wxyz(raw):
    """Isaac Lab body_quat_w in THIS build is raw (x, y, z, w); USD Gf.Quat is wxyz."""
    arr = np.asarray(raw, dtype=float)
    return unit(np.array([arr[3], arr[0], arr[1], arr[2]]))


def rotation_matrix(quat_wxyz):
    w, x, y, z = unit(quat_wxyz)
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
        [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
        [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)],
    ])


def angle_between(a, b):
    delta = rotation_matrix(a).T @ rotation_matrix(b)
    cosine = max(-1.0, min(1.0, (float(np.trace(delta)) - 1.0) / 2.0))
    return float(math.acos(cosine))


def author(path, lift):
    stage = Usd.Stage.CreateNew(path)
    UsdGeom.SetStageMetersPerUnit(stage, 1.0)
    UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)
    root = UsdGeom.Xform.Define(stage, "/RW")
    stage.SetDefaultPrim(root.GetPrim())
    root.GetPrim().SetCustomDataByKey(
        "label", "P2-07 CORRECTED four-wheel compound chassis + REAL B601-RS (read-only reference)")

    chassis = UsdGeom.Xform.Define(stage, "/RW/Chassis")
    UsdGeom.Xformable(chassis).AddTranslateOp().Set(Gf.Vec3f(0.0, 0.0, lift))

    # main body geometry sits at the CAD bbox offset inside the chassis frame
    mesh_body = UsdGeom.Xform.Define(stage, "/RW/Chassis/body_geometry")
    UsdGeom.Xformable(mesh_body).AddTranslateOp().Set(Gf.Vec3f(*BODY_CENTRE))
    geom = UsdGeom.Cube.Define(stage, "/RW/Chassis/body_geometry/geometry")
    geom.CreateSizeAttr(1.0)
    UsdGeom.Xformable(geom.GetPrim()).AddScaleOp().Set(
        Gf.Vec3f(2 * BODY_HALF[0], 2 * BODY_HALF[1], 2 * BODY_HALF[2]))
    UsdPhysics.CollisionAPI.Apply(geom.GetPrim())

    UsdPhysics.RigidBodyAPI.Apply(chassis.GetPrim())
    mass_api = UsdPhysics.MassAPI.Apply(chassis.GetPrim())
    mass_api.CreateMassAttr(CHASSIS_MASS_KG)
    mass_api.CreateCenterOfMassAttr(Gf.Vec3f(0.0))
    # Do NOT add a second ArticulationRootAPI here: the real B601-RS asset already carries
    # one at Arm/Geometry, and PhysX rejects two roots under the same articulation
    # ("Expected exactly one ArticulationRootAPI prim ... found 2", G7B_static_r03).
    # The chassis joins the existing articulation through the retargeted root_joint,
    # exactly like the proven G7_r01 mount.

    for index, centre in enumerate(WHEEL_CENTRES, start=1):
        wheel = UsdGeom.Xform.Define(stage, Sdf.Path("/RW/Chassis/wheel%d" % index))
        UsdGeom.Xformable(wheel).AddTranslateOp().Set(Gf.Vec3f(*centre))
        wheel_geom = UsdGeom.Cube.Define(
            stage, Sdf.Path("/RW/Chassis/wheel%d/geometry" % index))
        wheel_geom.CreateSizeAttr(1.0)
        UsdGeom.Xformable(wheel_geom.GetPrim()).AddScaleOp().Set(
            Gf.Vec3f(2 * WHEEL_HALF[0], 2 * WHEEL_HALF[1], 2 * WHEEL_HALF[2]))
        UsdPhysics.CollisionAPI.Apply(wheel_geom.GetPrim())

    cache = UsdGeom.XformCache(Usd.TimeCode.Default())
    bottoms = []
    for index in range(1, 5):
        prim = stage.GetPrimAtPath("/RW/Chassis/wheel%d" % index)
        world_z = float(cache.GetLocalToWorldTransform(prim).ExtractTranslation()[2])
        bottoms.append(world_z - WHEEL_HALF[2])
    expected = float(lift) + WHEEL_CENTRES[0][2] - WHEEL_HALF[2]
    if abs(min(bottoms) - expected) > 1e-5:
        raise ValueError("authored wheel world bottom %.9f != expected %.9f"
                         % (min(bottoms), expected))

    arm = UsdGeom.Xform.Define(stage, "/RW/Arm")
    UsdGeom.Xformable(arm).AddTranslateOp().Set(Gf.Vec3f(0.0, 0.0, lift))
    arm.GetPrim().GetReferences().AddReference(REAL_ARM)

    joint_path = None
    for prim in Usd.PrimRange(stage.GetPrimAtPath("/RW/Arm")):
        if prim.IsA(UsdPhysics.Joint) and prim.GetName() == "root_joint":
            joint_path = prim.GetPath()
            break
    if joint_path is None:
        raise SystemExit("real arm has no root_joint to retarget")
    over = stage.OverridePrim(joint_path)
    UsdPhysics.FixedJoint(over).CreateBody0Rel().SetTargets([chassis.GetPath()])
    stage.GetRootLayer().Save()
    return {"usd": os.path.abspath(path), "lift_m": float(lift),
            "per_wheel_world_bottom_z_m": bottoms,
            "wheel_ground_clearance_m": min(bottoms),
            "overridden_joint": str(joint_path)}


def main():
    out_dir = os.path.abspath(ARGS.out_dir)
    os.makedirs(out_dir, exist_ok=True)
    arm_hash_before = sha256(REAL_ARM)
    authored = author(os.path.join(out_dir, "authored_arm_with_wheels.usda"), ARGS.lift)

    physics = PhysxCfg(solver_type=1, enable_external_forces_every_iteration=True)
    sim = sim_utils.SimulationContext(sim_utils.SimulationCfg(
        dt=DT, device="cuda:0", gravity=(0.0, 0.0, float(ARGS.gravity_z)), physics=physics))
    ground = sim_utils.GroundPlaneCfg()
    ground.func("/World/ground", ground)
    spawn = sim_utils.UsdFileCfg(usd_path=authored["usd"], activate_contact_sensors=True)
    spawn.func("/World/RW", spawn)

    def has_contact_reporter(path):
        try:
            import omni.usd

            stage = omni.usd.get_context().get_stage()
            prim = stage.GetPrimAtPath(path)
            return bool(prim and prim.IsValid()
                        and "PhysxContactReportAPI" in prim.GetAppliedSchemas())
        except Exception:
            return False

    sensors = {}
    reporter_status = {}
    contact_api = "UNKNOWN"
    if ARGS.read_contacts:
        try:
            from isaaclab.sensors import ContactSensor, ContactSensorCfg

            for name, prim_path in (("chassis", "/World/RW/Chassis"),
                                    ("arm", "/World/RW/Arm/.*")):
                if not has_contact_reporter("/World/RW/Chassis") and name == "chassis":
                    reporter_status[name] = False
                    continue
                try:
                    sensors[name] = ContactSensor(ContactSensorCfg(
                        prim_path=prim_path, update_period=0.0, history_length=0,
                        debug_vis=False))
                    reporter_status[name] = True
                except Exception as exc:
                    reporter_status[name] = "FAILED: %s" % type(exc).__name__
            contact_api = "isaaclab.sensors.ContactSensor(%s)" % ",".join(sensors) or "none"
        except Exception as exc:  # pragma: no cover
            contact_api = "UNKNOWN: %s" % type(exc).__name__

    actuators = {"all": ImplicitActuatorCfg(joint_names_expr=[".*"], stiffness=300.0,
                                            damping=20.0)}
    robot = Articulation(ArticulationCfg(prim_path="/World/RW", spawn=None,
                                         actuators=actuators))
    sim.reset()

    body_names = list(robot.body_names)
    joint_names = list(robot.joint_names)
    index_of = {name: position for position, name in enumerate(body_names)}
    chassis_index = index_of.get("Chassis")
    base_index = index_of.get("base_link")
    masses = None
    try:
        masses = [round(float(value), 9) for value in _np(robot.root_physx_view.get_masses()[0])]
    except Exception:
        masses = None

    jsonl = open(os.path.join(out_dir, "raw_poses.jsonl"), "w", encoding="utf-8")
    frames = []

    def sensor_forces():
        values = {}
        for name, sensor in sensors.items():
            try:
                data = _np(sensor.data.net_forces_w).reshape(-1, 3)
                values[name] = round(float(np.max(np.linalg.norm(data, axis=1))), 9)
            except Exception:
                values[name] = None
        return values

    def capture(tag, step):
        row = {"run_id": ARGS.run_id, "tag": tag, "step": step, "t_s": round(step * DT, 9),
               "bodies": {}}
        for name in body_names:
            index = index_of[name]
            raw = [float(v) for v in _np(robot.data.body_quat_w[0, index]).reshape(-1)[:4]]
            row["bodies"][name] = {
                "pos": [round(float(v), 12) for v in
                        _np(robot.data.body_pos_w[0, index]).reshape(-1)[:3]],
                "raw_quat": [round(v, 12) for v in raw],
                "normalized_quat_wxyz": [round(v, 12) for v in raw_to_wxyz(raw)],
                "quat_source": "isaaclab_body_quat_raw(xyzw)",
                "lin_vel": [round(float(v), 12) for v in
                            _np(robot.data.body_lin_vel_w[0, index]).reshape(-1)[:3]],
                "ang_vel": [round(float(v), 12) for v in
                            _np(robot.data.body_ang_vel_w[0, index]).reshape(-1)[:3]],
            }
        chassis = row["bodies"].get("Chassis")
        if chassis is not None:
            row["pos"] = chassis["pos"]
            row["lin_vel"] = chassis["lin_vel"]
            row["tilt_rad"] = round(float(math.acos(max(-1.0, min(
                1.0, rotation_matrix(chassis["normalized_quat_wxyz"])[2, 2])))), 12)
        row["joint_pos"] = [round(float(v), 9) for v in _np(robot.data.joint_pos[0]).reshape(-1)]
        row["joint_vel"] = [round(float(v), 9) for v in _np(robot.data.joint_vel[0]).reshape(-1)]
        forces = sensor_forces()
        row["contact_by_sensor_n"] = forces
        known = [value for value in forces.values() if value is not None]
        row["max_contact_force_n"] = round(max(known), 9) if known else None
        row["chassis_contact_n"] = forces.get("chassis")
        row["arm_contact_n"] = forces.get("arm")
        row["finite"] = all(
            math.isfinite(v) for entry in row["bodies"].values()
            for key in ("pos", "raw_quat", "lin_vel", "ang_vel") for v in entry[key])
        jsonl.write(json.dumps(row, ensure_ascii=False) + "\n")
        jsonl.flush()
        frames.append(row)

    def update_sensors():
        for sensor in sensors.values():
            try:
                sensor.update(DT)
            except Exception:
                pass

    update_sensors()
    capture("after_reset_step0", 0)
    for step in range(1, ARGS.steps + 1):
        robot.write_data_to_sim()
        sim.step()
        robot.update(sim.get_physics_dt())
        # A ContactSensor returns its initial zeros until update() is called; G7B_grav_r07
        # reported 0 N for that reason alone while the chassis still settled at the wheel
        # contact height (z=0.333999723 after a 64.6 mm fall).  Update every step.
        update_sensors()
        capture("settle", step)
    jsonl.close()

    reset0 = frames[0]
    chassis0 = reset0["bodies"]["Chassis"]
    authored_pos = [0.0, 0.0, float(ARGS.lift)]
    world_pos_0 = float(np.linalg.norm(np.asarray(chassis0["pos"]) - np.asarray(authored_pos)))
    world_rot_0 = angle_between(chassis0["normalized_quat_wxyz"], [1.0, 0.0, 0.0, 0.0])
    drift = max(float(np.linalg.norm(np.asarray(frame["bodies"]["Chassis"]["pos"]) -
                                     np.asarray(chassis0["pos"]))) for frame in frames[1:])

    mount_pos = mount_rot = None
    mount_pos_max = mount_rot_max = None
    if base_index is not None and chassis_index is not None:
        base0 = reset0["bodies"]["base_link"]
        rotation0 = rotation_matrix(chassis0["normalized_quat_wxyz"])
        relative0 = rotation0.T @ (np.asarray(base0["pos"]) - np.asarray(chassis0["pos"]))
        mount_pos = float(np.linalg.norm(relative0))
        mount_rot = angle_between(chassis0["normalized_quat_wxyz"],
                                  base0["normalized_quat_wxyz"])
        positions, rotations = [], []
        for frame in frames:
            chassis = frame["bodies"]["Chassis"]
            base = frame["bodies"]["base_link"]
            rotation = rotation_matrix(chassis["normalized_quat_wxyz"])
            relative = rotation.T @ (np.asarray(base["pos"]) - np.asarray(chassis["pos"]))
            positions.append(float(np.linalg.norm(relative - relative0)))
            rotations.append(angle_between(chassis["normalized_quat_wxyz"],
                                           base["normalized_quat_wxyz"]))
        mount_pos_max = max(positions)
        mount_rot_max = max(rotations)

    wheel_bottoms = []
    for frame in frames:
        rotation = rotation_matrix(frame["bodies"]["Chassis"]["normalized_quat_wxyz"])
        wheel_bottoms.append([frame["pos"][2] + float(rotation[2, 0] * centre[0]
                                                      + rotation[2, 1] * centre[1]
                                                      + rotation[2, 2] * centre[2])
                              - WHEEL_HALF[2] for centre in WHEEL_CENTRES])

    arm_forces = [frame.get("arm_contact_n") for frame in frames
                  if frame.get("arm_contact_n") is not None]
    arm_peak = max(arm_forces) if arm_forces else None
    static_thresholds = {"position_m": 0.001, "rotation_rad": 0.001, "drift_m": 0.001,
                         "mount_position_m": 0.001, "mount_rotation_rad": 0.001}
    if ARGS.gate_mode == "static":
        gates = {
            "t0_world_position_within_tol": world_pos_0 <= static_thresholds["position_m"],
            "t0_world_rotation_within_tol": world_rot_0 <= static_thresholds["rotation_rad"],
            "t0_mount_position_within_tol": (mount_pos is not None and
                                             mount_pos <= static_thresholds["mount_position_m"]),
            "t0_mount_rotation_within_tol": (mount_rot is not None and
                                             mount_rot <= static_thresholds["mount_rotation_rad"]),
            "max_drift_within_tol": drift <= static_thresholds["drift_m"],
            "all_finite": all(frame["finite"] for frame in frames),
        }
        metrics = {
            "t0_world_position_error_m": round(world_pos_0, 12),
            "t0_world_rotation_error_rad": round(world_rot_0, 12),
            "t0_mount_position_error_m": None if mount_pos is None else round(mount_pos, 12),
            "t0_mount_rotation_error_rad": None if mount_rot is None else round(mount_rot, 12),
            "max_chassis_drift_m": round(drift, 12),
            "arm_peak_contact_n": arm_peak,
            "steps": ARGS.steps,
        }
        thresholds = static_thresholds
    else:
        gates, metrics = compute_landing_gates(
            frames, wheel_bottoms, gravity_z=float(ARGS.gravity_z), mass_kg=None)
        gates["mount_stable_under_load"] = bool(
            mount_pos_max is not None and mount_pos_max <= static_thresholds["mount_position_m"]
            and mount_rot_max <= static_thresholds["mount_rotation_rad"])
        if arm_peak is not None:
            gates["arm_does_not_touch_ground"] = bool(arm_peak <= 1.0)
        metrics.update({
            "mount_position_drift_max_m": (None if mount_pos_max is None
                                           else round(mount_pos_max, 12)),
            "mount_rotation_drift_max_rad": (None if mount_rot_max is None
                                             else round(mount_rot_max, 12)),
            "t0_mount_position_error_m": None if mount_pos is None else round(mount_pos, 12),
            "t0_mount_rotation_error_rad": None if mount_rot is None else round(mount_rot, 12),
            "arm_peak_contact_n": arm_peak,
            "t0_world_position_error_m": round(world_pos_0, 12),
            "max_chassis_drift_m": round(drift, 12),
            "steps": ARGS.steps,
        })
        thresholds = {"landing": "PHASE_P2_G5_LANDING_GATE.md",
                      "mount_position_m": static_thresholds["mount_position_m"],
                      "mount_rotation_rad": static_thresholds["mount_rotation_rad"],
                      "arm_ground_contact_n": 1.0}

    gate = "PASS" if all(gates.values()) else "FAIL"
    arm_hash_after = sha256(REAL_ARM)
    result = {
        "run_id": ARGS.run_id, "scene_variant": "REAL_ARM_ON_CORRECTED_WHEELS",
        "gate_mode": ARGS.gate_mode, "gate_result": gate,
        "metrics": metrics, "thresholds": thresholds, "gates": gates,
        "diagnostics": {
            "lift_m": float(ARGS.lift), "gravity_m_s2": [0.0, 0.0, float(ARGS.gravity_z)],
            "wheel_ground_clearance_m": round(authored["wheel_ground_clearance_m"], 9),
            "per_wheel_world_bottom_z_m": authored["per_wheel_world_bottom_z_m"],
            "contact_api": contact_api, "contact_reporter_present": reporter_status,
            "body_names": body_names, "joint_names": joint_names,
            "num_bodies": len(body_names), "dofs": int(robot.data.joint_pos.shape[1]),
            "body_masses_kg": masses,
            "total_mass_kg": (round(float(sum(masses)), 9) if masses else None),
            "chassis_mass_kg": CHASSIS_MASS_KG,
            "chassis_in_articulation": chassis_index is not None,
            "arm_asset": REAL_ARM,
            "arm_sha256_before": arm_hash_before, "arm_sha256_after": arm_hash_after,
            "arm_asset_unchanged": arm_hash_before == arm_hash_after,
            "wheel_bottom_model": "chassis body origin + R*(wheel centre) - 0.090 (TEST_PROXY)",
        },
    }
    with open(os.path.join(out_dir, "result.json"), "w", encoding="utf-8") as handle:
        json.dump(result, handle, ensure_ascii=False, indent=1)
    with open(os.path.join(out_dir, "manifest.json"), "w", encoding="utf-8") as handle:
        json.dump({"stage": "P2", "run_id": ARGS.run_id,
                   "scene_variant": "REAL_ARM_ON_CORRECTED_WHEELS",
                   "gate_mode": ARGS.gate_mode,
                   "preregistration": "management/morph_one_rs/PHASE_P2_G5_LANDING_GATE.md",
                   "physics_dt_s": DT, "steps": ARGS.steps,
                   "gravity_m_s2": [0.0, 0.0, float(ARGS.gravity_z)],
                   "include_ground": True, "lift_m": float(ARGS.lift),
                   "arm_asset": REAL_ARM, "arm_sha256": arm_hash_after,
                   "arm_asset_unchanged": arm_hash_before == arm_hash_after,
                   "chassis_geometry": "measured CAD body module + 4 wheel envelope cubes",
                   "drive_mode": "implicit 300/20, targets untouched",
                   "command": " ".join(os.sys.argv)}, handle, ensure_ascii=False, indent=1)
    print(json.dumps({"run_id": ARGS.run_id, "gate_result": gate, "metrics": metrics,
                      "gates": gates, "bodies": len(body_names),
                      "dofs": result["diagnostics"]["dofs"],
                      "total_mass_kg": result["diagnostics"]["total_mass_kg"]},
                     ensure_ascii=False, indent=1))
    print("GATE_RESULT=%s" % gate, flush=True)
    SIMULATION_APP.close()
    return 0 if gate == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
