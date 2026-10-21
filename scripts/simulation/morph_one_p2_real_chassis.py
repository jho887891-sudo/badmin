"""P2 -> P4 bridge: simplified chassis collision built from MEASURED Morph One geometry.

The wheel-module and body boxes below are derived from recorded CAD measurements, not guessed:

  * four wheel solids (STEP eid 435700 / 759491 / 773650 / 871375)
      centre (CAD mm) (+-250.002, -244.000, +-297.000), bbox 48.653 x 180.782 x 157.783 mm
      after 0.001 scale and the +90 deg X rotation: world (+-0.250002, -+0.297, -0.244) m
  * main body module bbox 395.656 x 132.837 x 800.852 mm, centre (0, -40.419, 2.426) mm
      after the same transform: half extents (0.198, 0.400, 0.066) m at (0, -0.0024, -0.0404) m

The wheel box half-height is 0.090 m, so the lowest simulated point sits at z = -0.334 m; the
robot therefore has to spawn with lift ~= 0.334 m for the wheels to just touch z = 0.  This
script sweeps that number and, when a ContactSensor can be created, records contact forces so
the penetration claim stops being an inference.

Usage:
    python morph_one_p2_real_chassis.py --run-id RC_lift0334_r01 --out-dir <dir> --lift 0.334
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
STEPS = 60
DT = 1.0 / 120.0

PARSER = argparse.ArgumentParser(description="P2 simplified chassis from measured geometry")
PARSER.add_argument("--run-id", required=True)
PARSER.add_argument("--out-dir", required=True)
PARSER.add_argument("--lift", type=float, default=0.334)
PARSER.add_argument("--steps", type=int, default=STEPS)
PARSER.add_argument("--no-ground", dest="include_ground", action="store_false", default=True)
PARSER.add_argument("--read-contacts", action="store_true")
AppLauncher.add_app_launcher_args(PARSER)
ARGS = PARSER.parse_args()
LAUNCHER = AppLauncher(ARGS)
SIMULATION_APP = LAUNCHER.app

import isaaclab.sim as sim_utils  # noqa: E402
from isaaclab.assets import Articulation, ArticulationCfg  # noqa: E402
from isaaclab_physx.physics import PhysxCfg  # noqa: E402
from pxr import Gf, Sdf, Usd, UsdGeom, UsdPhysics  # noqa: E402


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
    root = UsdGeom.Xform.Define(stage, "/RC")
    stage.SetDefaultPrim(root.GetPrim())
    root.GetPrim().SetCustomDataByKey(
        "label", "TEST_PROXY chassis collision from measured Morph One geometry")

    def collider(path, half, centre, mass=None):
        body = UsdGeom.Xform.Define(stage, Sdf.Path(path))
        UsdGeom.Xformable(body).AddTranslateOp().Set(
            Gf.Vec3f(centre[0], centre[1], centre[2] + lift))
        cube = UsdGeom.Cube.Define(stage, Sdf.Path(path).AppendChild("geometry"))
        cube.CreateSizeAttr(1.0)
        UsdGeom.Xformable(cube.GetPrim()).AddScaleOp().Set(
            Gf.Vec3f(2 * half[0], 2 * half[1], 2 * half[2]))
        UsdPhysics.CollisionAPI.Apply(cube.GetPrim())
        prim = body.GetPrim()
        if mass is not None:
            UsdPhysics.RigidBodyAPI.Apply(prim)
            mass_api = UsdPhysics.MassAPI.Apply(prim)
            mass_api.CreateMassAttr(mass)
            mass_api.CreateCenterOfMassAttr(Gf.Vec3f(0.0))
        return prim

    chassis = collider("/RC/Chassis", BODY_HALF, BODY_CENTRE, mass=20.0)
    UsdPhysics.ArticulationRootAPI.Apply(chassis)
    for index, centre in enumerate(WHEEL_CENTRES, start=1):
        # children inherit the chassis transform, so the wheel offsets must be RELATIVE to the
        # chassis body origin; passing absolute centres double-counted the offset and the lift,
        # which is why the first version produced 0 N of contact even when it claimed penetration.
        relative = (centre[0] - BODY_CENTRE[0],
                    centre[1] - BODY_CENTRE[1],
                    centre[2] - BODY_CENTRE[2])
        collider("/RC/Chassis/wheel%d" % index, WHEEL_HALF, relative, mass=None)
    stage.GetRootLayer().Save()
    return {"usd": os.path.abspath(path), "lift_m": float(lift),
            "measured_wheel_bottom_z_m": MEASURED_WHEEL_BOTTOM_Z_M,
            "root_api_target": str(chassis.GetPath())}

def main():
    out_dir = os.path.abspath(ARGS.out_dir)
    os.makedirs(out_dir, exist_ok=True)
    authored = author(os.path.join(out_dir, "authored_chassis.usda"), ARGS.lift)

    physics = PhysxCfg(solver_type=1, enable_external_forces_every_iteration=True)
    sim = sim_utils.SimulationContext(
        sim_utils.SimulationCfg(dt=DT, device="cuda:0", gravity=(0.0, 0.0, 0.0),
                                physics=physics))
    if ARGS.include_ground:
        ground = sim_utils.GroundPlaneCfg()
        ground.func("/World/ground", ground)
    # ContactSensor only reports on prims whose asset enabled contact reporters.
    spawn = sim_utils.UsdFileCfg(usd_path=authored["usd"], activate_contact_sensors=True)
    spawn.func("/World/RC", spawn)

    sensor = None
    contact_api = "UNKNOWN"
    if ARGS.read_contacts:
        try:
            from isaaclab.sensors import ContactSensor, ContactSensorCfg

            sensor = ContactSensor(ContactSensorCfg(
                prim_path="/World/RC/Chassis", update_period=0.0, history_length=0,
                debug_vis=False))
            contact_api = "isaaclab.sensors.ContactSensor"
        except Exception as exc:  # pragma: no cover
            contact_api = "UNKNOWN: %s" % type(exc).__name__

    robot = Articulation(ArticulationCfg(prim_path="/World/RC", spawn=None, actuators={}))
    for field in ("joint_pos", "joint_vel"):
        try:
            setattr(robot.cfg.init_state, field, {})
        except Exception:
            pass
    sim.reset()
    if sensor is not None:
        try:
            sensor.update(DT)
        except Exception as exc:  # pragma: no cover
            contact_api = "UNKNOWN: sensor.update failed %s" % type(exc).__name__

    body_names = list(robot.body_names)
    jsonl = open(os.path.join(out_dir, "raw_poses.jsonl"), "w", encoding="utf-8")
    frames = []

    def contact_forces():
        if sensor is None:
            return None
        try:
            forces = _np(sensor.data.net_forces_w)
            return round(float(np.max(np.linalg.norm(forces.reshape(-1, 3), axis=1))), 9)
        except Exception:
            return None

    def capture(tag, step):
        raw = [float(v) for v in _np(robot.data.body_quat_w[0, 0]).reshape(-1)[:4]]
        row = {"run_id": ARGS.run_id, "tag": tag, "step": step, "t_s": round(step * DT, 9),
               "body": body_names[0],
               "pos": [round(float(v), 12) for v in _np(robot.data.body_pos_w[0, 0]).reshape(-1)[:3]],
               "raw_quat": [round(v, 12) for v in raw],
               "normalized_quat_wxyz": [round(v, 12) for v in raw_to_wxyz(raw)],
               "lin_vel": [round(float(v), 12) for v in
                           _np(robot.data.body_lin_vel_w[0, 0]).reshape(-1)[:3]],
               "ang_vel": [round(float(v), 12) for v in
                           _np(robot.data.body_ang_vel_w[0, 0]).reshape(-1)[:3]],
               "max_contact_force_n": contact_forces()}
        row["finite"] = all(math.isfinite(v) for v in
                            row["pos"] + row["raw_quat"] + row["lin_vel"] + row["ang_vel"])
        jsonl.write(json.dumps(row, ensure_ascii=False) + "\n")
        jsonl.flush()
        frames.append(row)

    capture("after_reset_step0", 0)
    for step in range(1, ARGS.steps + 1):
        sim.step()
        robot.update(sim.get_physics_dt())
        if sensor is not None:
            try:
                sensor.update(DT)
            except Exception:
                pass
        capture("settle", step)
    jsonl.close()

    reset0 = frames[0]
    settle = frames[1:]
    # the chassis body origin sits at BODY_CENTRE + lift, not at (0, 0, lift)
    authored_pos = [BODY_CENTRE[0], BODY_CENTRE[1], BODY_CENTRE[2] + ARGS.lift]
    world_pos_0 = float(np.linalg.norm(np.asarray(reset0["pos"]) - np.asarray(authored_pos)))
    world_rot_0 = angle_between(reset0["normalized_quat_wxyz"], [1.0, 0.0, 0.0, 0.0])
    drift = float(np.linalg.norm(np.asarray(settle[-1]["pos"]) - np.asarray(reset0["pos"])))
    v0 = float(np.linalg.norm(np.asarray(reset0["lin_vel"])))
    thresholds = {"position_m": 0.001, "rotation_rad": 0.001, "settle_drift_m": 0.001}
    verdict = {
        "t0_world_position_within_tol": world_pos_0 <= thresholds["position_m"],
        "t0_world_rotation_within_tol": world_rot_0 <= thresholds["rotation_rad"],
        "settle_drift_within_tol": drift <= thresholds["settle_drift_m"],
        "all_finite": all(row["finite"] for row in frames),
    }
    gate = "PASS" if all(verdict.values()) else "FAIL"
    result = {
        "run_id": ARGS.run_id, "scene_variant": "REAL_CHASSIS_SIMPLIFIED",
        "gate_result": gate,
        "metrics": {"t0_world_position_error_m": round(world_pos_0, 12),
                    "t0_world_rotation_error_rad": round(world_rot_0, 12),
                    "settle_60_step_chassis_drift_m": round(drift, 12)},
        "thresholds": thresholds, "gates": verdict,
        "diagnostics": {"lift_m": float(ARGS.lift),
                        "measured_wheel_bottom_z_m": MEASURED_WHEEL_BOTTOM_Z_M,
                        "clearance_m": round(float(ARGS.lift) + MEASURED_WHEEL_BOTTOM_Z_M, 9),
                        "root_lin_speed_at_step0_m_s": round(v0, 12),
                        "contact_api": contact_api,
                        "max_contact_force_step0_n": reset0.get("max_contact_force_n"),
                        "max_contact_force_settle_n": (
                            max((row.get("max_contact_force_n") or 0.0) for row in settle)
                            if sensor is not None else None),
                        "bodies": body_names},
    }
    with open(os.path.join(out_dir, "result.json"), "w", encoding="utf-8") as handle:
        json.dump(result, handle, ensure_ascii=False, indent=1)
    with open(os.path.join(out_dir, "manifest.json"), "w", encoding="utf-8") as handle:
        json.dump({"stage": "P2", "run_id": ARGS.run_id, "scene_variant": "REAL_CHASSIS_SIMPLIFIED",
                   "geometry_source": "measured CAD (wheel solids 435700/759491/773650/871375 + body module)",
                   "physics_dt_s": DT, "gravity_m_s2": [0, 0, 0],
                   "include_ground": bool(ARGS.include_ground), "lift_m": float(ARGS.lift),
                   "command": " ".join(os.sys.argv)}, handle, ensure_ascii=False, indent=1)
    print(json.dumps({k: result[k] for k in ("run_id", "gate_result", "metrics")},
                     ensure_ascii=False, indent=1))
    print("GATE_RESULT=%s" % gate, flush=True)
    SIMULATION_APP.close()
    return 0 if gate == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
