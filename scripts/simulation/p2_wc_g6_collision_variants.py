"""P2-WC G6: change ONLY the collision implementation, keep the scene identical.

G5 (PR #20) proved that the corrected four-wheel compound collider reacts to gravity
(peak 2693 N).  G6 re-asks the question with a single changed factor per run so the
reliable collision structure can be named instead of assumed:

    box_child_xform   (A) exactly the G5 authoring: one rigid body (Chassis) with four
                         child Xform cube colliders, inherited transform, no joints.
    box_child_rigid   (B) same cube envelope, but every wheel is an EXPLICIT rigid body
                         joined to the chassis with a UsdPhysics.FixedJoint.  Total mass
                         is held at 20 kg (16 + 4 x 1 kg) so only the collision/hierarchy
                         implementation changes; the mass split is unavoidable and is
                         recorded in the manifest.
    mesh_convex_hull  (C) same envelope as the cube, but each wheel collider is a
                         generated elliptical-cylinder MESH cooked with
                         UsdPhysics.MeshCollisionAPI(convexHull).  Lowest point is
                         identical by construction (z = centre - 0.090 m).

Metrics, thresholds and the run plan are pre-registered in
management/morph_one_rs/PHASE_P2_WC_G6_PREREGISTRATION.md and must not be relaxed
after seeing data.

Usage:
    python p2_wc_g6_collision_variants.py --run-id G6A_r01 --out-dir /home/dgut/runs/g/G6A_r01 \
        --impl box_child_xform --lift 0.400 --gravity-z -9.81 --steps 240 --read-contacts
"""

from __future__ import annotations

import argparse
import json
import math
import os

import numpy as np

from isaaclab.app import AppLauncher

WHEEL_CENTRES = [(0.250002, -0.297, -0.244), (0.250002, 0.297, -0.244),
                 (-0.250002, -0.297, -0.244), (-0.250002, 0.297, -0.244)]
WHEEL_HALF = (0.0243, 0.0789, 0.0900)
BODY_HALF = (0.198, 0.400, 0.066)
BODY_CENTRE = (0.0, -0.0024, -0.0404)
MEASURED_WHEEL_BOTTOM_Z_M = -0.334
TOTAL_MASS_KG = 20.0
WHEEL_MASS_KG = 1.0
DT = 1.0 / 120.0
TAIL_STEPS = 30          # pre-registered settle window: last 0.25 s of a landing run
MESH_SEGMENTS = 32
IMPLS = ("box_child_xform", "box_child_rigid", "mesh_convex_hull")

PARSER = argparse.ArgumentParser(description="P2-WC G6 collision implementation A/B/C")
PARSER.add_argument("--run-id", required=True)
PARSER.add_argument("--out-dir", required=True)
PARSER.add_argument("--impl", required=True, choices=IMPLS)
PARSER.add_argument("--lift", type=float, default=0.400)
PARSER.add_argument("--steps", type=int, default=360)
PARSER.add_argument("--no-ground", dest="include_ground", action="store_false", default=True)
PARSER.add_argument("--read-contacts", action="store_true")
PARSER.add_argument("--gravity-z", type=float, default=-9.81)
PARSER.add_argument("--gate-mode", choices=("landing", "equivalence"), default="landing",
                    help="'equivalence' is the 60-step harness check against G5: it only "
                         "asserts that contact is reported at all.")
AppLauncher.add_app_launcher_args(PARSER)
ARGS = PARSER.parse_args()
LAUNCHER = AppLauncher(ARGS)
SIMULATION_APP = LAUNCHER.app

import isaaclab.sim as sim_utils  # noqa: E402
from isaaclab.assets import Articulation, ArticulationCfg  # noqa: E402
from isaaclab_physx.physics import PhysxCfg  # noqa: E402
from pxr import Gf, Sdf, Usd, UsdGeom, UsdPhysics  # noqa: E402
from p2_chassis_frame_math import wheel_bottom_world_z  # noqa: E402


def _np(values):
    if hasattr(values, "detach"):
        values = values.detach().cpu().numpy()
    return np.asarray(values)


def unit(values):
    arr = np.asarray(values, dtype=float)
    norm = float(np.linalg.norm(arr))
    if not np.isfinite(norm) or norm <= 0.0:
        raise ValueError("degenerate quaternion")
    return arr / norm


def raw_to_wxyz(raw):
    """Isaac Lab body_quat_w in THIS build is raw (x, y, z, w)."""
    arr = np.asarray(raw, dtype=float)
    return unit(np.array([arr[3], arr[0], arr[1], arr[2]]))


def rotation_matrix(quat_wxyz):
    w, x, y, z = unit(quat_wxyz)
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
        [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
        [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)],
    ])


def tilt_from_upright(quat_wxyz):
    """Angle between the body's local +Z and world +Z."""
    rot = rotation_matrix(quat_wxyz)
    cosine = max(-1.0, min(1.0, float(rot[2, 2])))
    return float(math.acos(cosine))


def elliptical_cylinder_mesh(half_x, half_y, half_z, segments=MESH_SEGMENTS):
    """Convex mesh inscribed in the cube envelope, lowest point exactly -half_z."""
    points = []
    for x in (-half_x, half_x):
        for index in range(segments):
            angle = -math.pi / 2.0 + 2.0 * math.pi * index / segments
            points.append((x, half_y * math.cos(angle), half_z * math.sin(angle)))
    count = segments
    faces = []
    flat = []
    for index in range(segments):
        nxt = (index + 1) % segments
        faces.extend([3, index, nxt, count + nxt])
        faces.extend([3, index, count + nxt, count + index])
        flat.append((index, nxt, count + nxt))
    for index in range(1, segments - 1):
        faces.extend([3, 0, index + 1, index])
        faces.extend([3, count, count + index, count + index + 1])
    return points, faces


def author(path, lift, impl):
    stage = Usd.Stage.CreateNew(path)
    UsdGeom.SetStageMetersPerUnit(stage, 1.0)
    UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)
    root = UsdGeom.Xform.Define(stage, "/RC")
    stage.SetDefaultPrim(root.GetPrim())
    root.GetPrim().SetCustomDataByKey(
        "label", "P2-WC G6 collision variant %s (TEST_PROXY geometry)" % impl)

    chassis_mass = TOTAL_MASS_KG if impl != "box_child_rigid" else (
        TOTAL_MASS_KG - 4.0 * WHEEL_MASS_KG)

    def cube_geometry(parent_path, half, local_offset=None):
        cube = UsdGeom.Cube.Define(stage, Sdf.Path(parent_path).AppendChild("geometry"))
        cube.CreateSizeAttr(1.0)
        if local_offset is not None:
            UsdGeom.Xformable(cube.GetPrim()).AddTranslateOp().Set(Gf.Vec3f(*local_offset))
        UsdGeom.Xformable(cube.GetPrim()).AddScaleOp().Set(
            Gf.Vec3f(2 * half[0], 2 * half[1], 2 * half[2]))
        UsdPhysics.CollisionAPI.Apply(cube.GetPrim())
        return cube.GetPrim()

    # /RC/Chassis: the only translated frame.  Children stay in chassis-local CAD
    # coordinates, so the lift is applied exactly once (historical double-lift bug).
    chassis = UsdGeom.Xform.Define(stage, "/RC/Chassis")
    UsdGeom.Xformable(chassis).AddTranslateOp().Set(Gf.Vec3f(0.0, 0.0, lift))
    cube_geometry("/RC/Chassis", BODY_HALF, BODY_CENTRE)
    UsdPhysics.RigidBodyAPI.Apply(chassis.GetPrim())
    mass_api = UsdPhysics.MassAPI.Apply(chassis.GetPrim())
    mass_api.CreateMassAttr(chassis_mass)
    # COM at the rigid-body origin for every variant, exactly as in G5, so that the
    # only difference between A/B/C really is the collision implementation.  The CAD
    # body-bbox offset is left unmodelled and is listed as a known simplification.
    mass_api.CreateCenterOfMassAttr(Gf.Vec3f(0.0))
    UsdPhysics.ArticulationRootAPI.Apply(chassis.GetPrim())

    wheel_paths = []
    for index, centre in enumerate(WHEEL_CENTRES, start=1):
        # Variant B gives every wheel an explicit rigid body.  USD/PhysX forbid NESTED
        # rigid bodies, so those wheel prims become SIBLINGS of the chassis and must
        # therefore carry the lift themselves (they are not children of the lifted
        # chassis).  The authored world-bottom assertion below still guards this.
        if impl == "box_child_rigid":
            wheel_path = "/RC/wheel%d" % index
            wheel = UsdGeom.Xform.Define(stage, Sdf.Path(wheel_path))
            UsdGeom.Xformable(wheel).AddTranslateOp().Set(
                Gf.Vec3f(centre[0], centre[1], centre[2] + lift))
        else:
            wheel_path = "/RC/Chassis/wheel%d" % index
            wheel = UsdGeom.Xform.Define(stage, Sdf.Path(wheel_path))
            UsdGeom.Xformable(wheel).AddTranslateOp().Set(Gf.Vec3f(*centre))
        if impl == "mesh_convex_hull":
            points, faces = elliptical_cylinder_mesh(*WHEEL_HALF)
            mesh = UsdGeom.Mesh.Define(stage, Sdf.Path(wheel_path).AppendChild("geometry"))
            mesh.CreatePointsAttr([Gf.Vec3f(*p) for p in points])
            mesh.CreateFaceVertexCountsAttr([3] * (len(faces) // 4))
            mesh.CreateFaceVertexIndicesAttr(faces)
            mesh.CreateSubdivisionSchemeAttr(UsdGeom.Tokens.none)
            mesh.CreateOrientationAttr(UsdGeom.Tokens.rightHanded)
            UsdPhysics.CollisionAPI.Apply(mesh.GetPrim())
            UsdPhysics.MeshCollisionAPI.Apply(mesh.GetPrim()).CreateApproximationAttr("convexHull")
        else:
            cube_geometry(wheel_path, WHEEL_HALF)
        if impl == "box_child_rigid":
            prim = wheel.GetPrim()
            UsdPhysics.RigidBodyAPI.Apply(prim)
            wheel_mass = UsdPhysics.MassAPI.Apply(prim)
            wheel_mass.CreateMassAttr(WHEEL_MASS_KG)
            wheel_mass.CreateCenterOfMassAttr(Gf.Vec3f(0.0))
            joint = UsdPhysics.FixedJoint.Define(stage, Sdf.Path("/RC/joint_wheel%d" % index))
            joint.CreateBody0Rel().SetTargets([Sdf.Path("/RC/Chassis")])
            joint.CreateBody1Rel().SetTargets([Sdf.Path(wheel_path)])
            joint.CreateLocalPos0Attr().Set(Gf.Vec3f(*centre))
            joint.CreateLocalPos1Attr().Set(Gf.Vec3f(0.0))
            joint.CreateLocalRot0Attr().Set(Gf.Quatf(1.0, 0.0, 0.0, 0.0))
            joint.CreateLocalRot1Attr().Set(Gf.Quatf(1.0, 0.0, 0.0, 0.0))
        wheel_paths.append(wheel_path)

    cache = UsdGeom.XformCache(Usd.TimeCode.Default())
    bottoms = []
    for wheel_path in wheel_paths:
        prim = stage.GetPrimAtPath(wheel_path)
        if not prim or not prim.IsValid():
            raise ValueError("missing authored wheel: %s" % wheel_path)
        world_z = float(cache.GetLocalToWorldTransform(prim).ExtractTranslation()[2])
        bottoms.append(world_z - WHEEL_HALF[2])
    actual_bottom = min(bottoms)
    expected_bottom = wheel_bottom_world_z(float(lift), WHEEL_CENTRES[0][2], WHEEL_HALF[2])
    if abs(actual_bottom - expected_bottom) > 1e-5:
        raise ValueError("authored wheel frame mismatch: %.9f vs %.9f" %
                         (actual_bottom, expected_bottom))
    stage.GetRootLayer().Save()
    return {"usd": os.path.abspath(path), "lift_m": float(lift),
            "wheel_world_bottom_z_m": actual_bottom,
            "per_wheel_world_bottom_z_m": bottoms,
            "chassis_mass_kg": chassis_mass,
            "wheel_mass_kg": WHEEL_MASS_KG if impl == "box_child_rigid" else None,
            "total_mass_kg": TOTAL_MASS_KG,
            "wheel_paths": wheel_paths,
            "root_api_target": str(chassis.GetPath())}


def main():
    out_dir = os.path.abspath(ARGS.out_dir)
    os.makedirs(out_dir, exist_ok=True)
    authored = author(os.path.join(out_dir, "authored_chassis.usda"), ARGS.lift, ARGS.impl)

    gravity = (0.0, 0.0, float(ARGS.gravity_z))
    physics = PhysxCfg(solver_type=1, enable_external_forces_every_iteration=True)
    sim = sim_utils.SimulationContext(sim_utils.SimulationCfg(
        dt=DT, device="cuda:0", gravity=gravity, physics=physics))
    if ARGS.include_ground:
        ground = sim_utils.GroundPlaneCfg()
        ground.func("/World/ground", ground)
    spawn = sim_utils.UsdFileCfg(usd_path=authored["usd"], activate_contact_sensors=True)
    spawn.func("/World/RC", spawn)

    def has_contact_reporter(path):
        """Isaac Lab only marks rigid-body prims and stops at the first nested body."""
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

            targets = [("chassis", "/World/RC/Chassis")]
            if ARGS.impl == "box_child_rigid":
                # Only an explicit rigid body can host a ContactSensor, so per-wheel
                # contact is measurable for variant B alone; A/C stay chassis-aggregate.
                targets += [("wheel%d" % i, "/World/RC/wheel%d" % i) for i in range(1, 5)]
            skipped = []
            for name, prim_path in targets:
                reporter_status[name] = has_contact_reporter(prim_path)
                if not reporter_status[name]:
                    # Never let a missing reporter abort the physics run: record it.
                    skipped.append(prim_path)
                    continue
                try:
                    sensors[name] = ContactSensor(ContactSensorCfg(
                        prim_path=prim_path, update_period=0.0, history_length=0,
                        debug_vis=False))
                except Exception as exc:
                    reporter_status[name] = False
                    skipped.append("%s(%s)" % (prim_path, type(exc).__name__))
            contact_api = "isaaclab.sensors.ContactSensor(%d prims)" % len(sensors)
            if skipped:
                contact_api += "; no_contact_reporter=%s" % ",".join(skipped)
        except Exception as exc:  # pragma: no cover
            sensors = {}
            contact_api = "UNKNOWN: %s" % type(exc).__name__

    robot = Articulation(ArticulationCfg(prim_path="/World/RC", spawn=None, actuators={}))
    # Pitfall 3: this scene exposes no DOF (four FixedJoints are not DOF joints), so the
    # default init-state regex ".*" has nothing to match and reset() raises.  Clear both
    # dictionaries for EVERY variant, not just the jointless ones.
    for field in ("joint_pos", "joint_vel"):
        try:
            setattr(robot.cfg.init_state, field, {})
        except Exception:
            pass
    sim.reset()

    def update_sensors():
        for sensor in sensors.values():
            try:
                sensor.update(DT)
            except Exception:
                pass

    update_sensors()

    body_names = list(robot.body_names)
    jsonl = open(os.path.join(out_dir, "raw_poses.jsonl"), "w", encoding="utf-8")
    contacts_jsonl = open(os.path.join(out_dir, "contacts.jsonl"), "w", encoding="utf-8")
    frames = []

    def sensor_forces():
        forces = {}
        for name, sensor in sensors.items():
            try:
                values = _np(sensor.data.net_forces_w).reshape(-1, 3)
                forces[name] = round(float(np.max(np.linalg.norm(values, axis=1))), 9)
            except Exception:
                forces[name] = None
        return forces

    def capture(tag, step):
        raw = [float(v) for v in _np(robot.data.body_quat_w[0, 0]).reshape(-1)[:4]]
        quat = raw_to_wxyz(raw)
        pos = _np(robot.data.body_pos_w[0, 0]).reshape(-1)[:3]
        bodies = {}
        for index, name in enumerate(body_names):
            bodies[name] = {
                "pos": [round(float(v), 9) for v in
                        _np(robot.data.body_pos_w[0, index]).reshape(-1)[:3]],
                "lin_vel": [round(float(v), 9) for v in
                            _np(robot.data.body_lin_vel_w[0, index]).reshape(-1)[:3]],
            }
        contact = sensor_forces()
        aggregate = [v for v in contact.values() if v is not None]
        lin_vel = [float(v) for v in _np(robot.data.body_lin_vel_w[0, 0]).reshape(-1)[:3]]
        ang_vel = [float(v) for v in _np(robot.data.body_ang_vel_w[0, 0]).reshape(-1)[:3]]
        row = {"run_id": ARGS.run_id, "impl": ARGS.impl, "tag": tag, "step": step,
               "t_s": round(step * DT, 9),
               "pos": [round(float(v), 12) for v in pos],
               "raw_quat": [round(v, 12) for v in raw],
               "normalized_quat_wxyz": [round(v, 12) for v in quat],
               "lin_vel": [round(v, 12) for v in lin_vel],
               "ang_vel": [round(v, 12) for v in ang_vel],
               "tilt_rad": round(tilt_from_upright(quat), 12),
               "bodies": bodies,
               "contact_by_sensor_n": contact,
               "max_contact_force_n": (round(max(aggregate), 9) if aggregate else None)}
        row["finite"] = all(math.isfinite(v) for v in
                            row["pos"] + row["raw_quat"] + row["lin_vel"] + row["ang_vel"])
        jsonl.write(json.dumps(row, ensure_ascii=False) + "\n")
        jsonl.flush()
        contacts_jsonl.write(json.dumps(
            {"run_id": ARGS.run_id, "impl": ARGS.impl, "step": step, "t_s": round(step * DT, 9),
             "contact_api": contact_api, "contact_by_sensor_n": contact,
             "max_contact_force_n": row["max_contact_force_n"]}, ensure_ascii=False) + "\n")
        contacts_jsonl.flush()
        frames.append(row)

    capture("after_reset_step0", 0)
    for step in range(1, ARGS.steps + 1):
        sim.step()
        robot.update(sim.get_physics_dt())
        update_sensors()
        capture("settle", step)
    jsonl.close()
    contacts_jsonl.close()

    rot0 = rotation_matrix(frames[0]["normalized_quat_wxyz"])
    wheel_local = np.array([centre for centre in WHEEL_CENTRES])
    chassis_positions = np.array([frame["pos"] for frame in frames])
    # world_z of a local point p is the third ROW of R dotted with p
    wheel_world_z = chassis_positions[:, None, 2] + (rot0[None, 2, :] @ wheel_local.T)
    wheel_bottom = wheel_world_z - WHEEL_HALF[2]
    force_history = [frame["max_contact_force_n"] for frame in frames
                     if frame["max_contact_force_n"] is not None]
    peak_force = max(force_history) if force_history else None
    first_contact_step = next((frame["step"] for frame in frames
                               if frame["max_contact_force_n"] is not None
                               and frame["max_contact_force_n"] > 1e-6), None)
    per_wheel_peak = {}
    for name in ("wheel1", "wheel2", "wheel3", "wheel4"):
        values = [frame["contact_by_sensor_n"].get(name) for frame in frames]
        values = [v for v in values if v is not None]
        per_wheel_peak[name] = max(values) if values else None

    tilt_max = max(frame["tilt_rad"] for frame in frames)
    tail = frames[-min(TAIL_STEPS, len(frames)):]
    tail_speed = max(float(np.linalg.norm(frame["lin_vel"])) for frame in tail)
    tail_z_span = (max(frame["pos"][2] for frame in tail)
                   - min(frame["pos"][2] for frame in tail))
    tail_penetration = max(0.0, float(-np.min(wheel_bottom[-len(tail):])))
    drift = max(float(np.linalg.norm(np.asarray(frame["pos"]) -
                                     np.asarray(frames[0]["pos"]))) for frame in frames[1:])
    static_t0_pos = float(np.linalg.norm(np.asarray(frames[0]["pos"]) -
                                         np.asarray([0.0, 0.0, ARGS.lift])))
    t0_rot = tilt_from_upright(frames[0]["normalized_quat_wxyz"])

    gates = {"all_finite": all(frame.get("finite", True) for frame in frames),
             "contact_detected": bool(peak_force is not None and peak_force > 1.0)}
    if ARGS.gate_mode == "landing":
        gates.update({
            "no_tip_over": tilt_max < 0.05,
            "settles": tail_speed < 0.02 and tail_z_span < 0.002,
            "no_long_term_penetration": tail_penetration < 0.002,
        })
        if per_wheel_peak and all(v is not None for v in per_wheel_peak.values()):
            gates["per_wheel_contact_all_four"] = all(
                v > 1.0 for v in per_wheel_peak.values())
    gate = "PASS" if all(gates.values()) else "FAIL"

    result = {
        "run_id": ARGS.run_id, "impl": ARGS.impl,
        "gate_mode": ARGS.gate_mode, "gate_result": gate,
        "metrics": {
            "peak_contact_force_n": peak_force,
            "first_contact_step": first_contact_step,
            "final_chassis_z_m": frames[-1]["pos"][2],
            "final_speed_m_s": round(float(np.linalg.norm(frames[-1]["lin_vel"])), 9),
            "tail_speed_max_m_s": round(tail_speed, 9),
            "tail_z_span_m": round(tail_z_span, 9),
            "tail_max_penetration_m": round(tail_penetration, 9),
            "max_tilt_rad": round(tilt_max, 9),
            "chassis_drift_m": round(drift, 9),
            "t0_static_position_error_m": round(static_t0_pos, 9),
            "t0_tilt_rad": round(t0_rot, 12),
            "steps": ARGS.steps,
        },
        "per_wheel_peak_contact_n": per_wheel_peak,
        "gates": gates,
        "diagnostics": {
            "lift_m": ARGS.lift, "gravity_m_s2": [0.0, 0.0, float(ARGS.gravity_z)],
            "include_ground": bool(ARGS.include_ground),
            "clearance_m": round(authored["wheel_world_bottom_z_m"], 9),
            "usd_authoring_wheel_world_bottom_z_m": authored["wheel_world_bottom_z_m"],
            "per_wheel_world_bottom_z_m": authored["per_wheel_world_bottom_z_m"],
            "chassis_mass_kg": authored["chassis_mass_kg"],
            "wheel_mass_kg": authored["wheel_mass_kg"],
            "total_mass_kg": authored["total_mass_kg"],
            "contact_api": contact_api,
            "contact_sensor_prims": list(sensors.keys()),
            "contact_reporter_present": reporter_status,
            "bodies": body_names,
            "wheel_bottom_model": "chassis body origin + R*(wheel centre local) - 0.090",
            "cooked_shape_query": "UNKNOWN (no exposed PhysX cooked-geometry API in this build)",
        },
    }
    with open(os.path.join(out_dir, "result.json"), "w", encoding="utf-8") as handle:
        json.dump(result, handle, ensure_ascii=False, indent=1)
    with open(os.path.join(out_dir, "manifest.json"), "w", encoding="utf-8") as handle:
        json.dump({"stage": "P2-WC", "run_id": ARGS.run_id, "scene_variant": "G6_COLLISION_VARIANT",
                   "impl": ARGS.impl, "impl_definition": {
                       "box_child_xform": "chassis rigid body + 4 child Xform cube colliders",
                       "box_child_rigid": "4 explicit wheel rigid bodies + FixedJoint, total mass 20 kg",
                       "mesh_convex_hull": "4 child Xform mesh colliders (convexHull), same envelope",
                   }[ARGS.impl],
                   "preregistration": "management/morph_one_rs/PHASE_P2_WC_G6_PREREGISTRATION.md",
                   "physics_dt_s": DT, "steps": ARGS.steps, "gate_mode": ARGS.gate_mode,
                   "gravity_m_s2": [0.0, 0.0, float(ARGS.gravity_z)],
                   "include_ground": bool(ARGS.include_ground),
                   "lift_m": ARGS.lift,
                   "geometry_source": ("measured CAD wheel envelope 48.6 x 157.8 x 180.0 mm "
                                       "(TEST_PROXY simplification, not the real tyre)"),
                   "solver": "PhysxCfg(solver_type=1=TGS, enable_external_forces_every_iteration=True)",
                   "command": " ".join(os.sys.argv)},
                  handle, ensure_ascii=False, indent=1)
    print(json.dumps({k: result[k] for k in ("run_id", "impl", "gate_result", "metrics")},
                     ensure_ascii=False, indent=1))
    print("GATE_RESULT=%s" % gate, flush=True)
    SIMULATION_APP.close()
    return 0 if gate == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
