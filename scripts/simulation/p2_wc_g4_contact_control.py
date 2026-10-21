"""G4 positive control: a standard cube falls onto the ground plane.

Purpose: prove that the PhysX + ContactSensor pipeline can report a real contact at all, and
give a measured reference force for a known mass.  This is NOT evidence about the wheel
chassis; it only calibrates the sensor path.

Usage:
    python p2_wc_g4_contact_control.py --run-id G4_r01 --out-dir <dir>
"""

from __future__ import annotations

import argparse
import json
import math
import os

import numpy as np

from isaaclab.app import AppLauncher

DT = 1.0 / 120.0
HALF = 0.05
MASS = 1.0
DROP_Z = 0.20

PARSER = argparse.ArgumentParser(description="G4 contact positive control")
PARSER.add_argument("--run-id", required=True)
PARSER.add_argument("--out-dir", required=True)
PARSER.add_argument("--steps", type=int, default=120)
AppLauncher.add_app_launcher_args(PARSER)
ARGS = PARSER.parse_args()
LAUNCHER = AppLauncher(ARGS)
SIMULATION_APP = LAUNCHER.app

import isaaclab.sim as sim_utils  # noqa: E402
from isaaclab.assets import RigidObject, RigidObjectCfg  # noqa: E402
from isaaclab.sensors import ContactSensor, ContactSensorCfg  # noqa: E402
from isaaclab_physx.physics import PhysxCfg  # noqa: E402
from pxr import Gf, Usd, UsdGeom, UsdPhysics  # noqa: E402


def _np(values):
    if hasattr(values, "detach"):
        values = values.detach().cpu().numpy()
    return np.asarray(values)


def author(path):
    stage = Usd.Stage.CreateNew(path)
    UsdGeom.SetStageMetersPerUnit(stage, 1.0)
    UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)
    root = UsdGeom.Xform.Define(stage, "/G4")
    stage.SetDefaultPrim(root.GetPrim())
    body = UsdGeom.Xform.Define(stage, "/G4/Cube")
    UsdGeom.Xformable(body).AddTranslateOp().Set(Gf.Vec3f(0.0, 0.0, DROP_Z))
    cube = UsdGeom.Cube.Define(stage, "/G4/Cube/geometry")
    cube.CreateSizeAttr(1.0)
    UsdGeom.Xformable(cube.GetPrim()).AddScaleOp().Set(
        Gf.Vec3f(2 * HALF, 2 * HALF, 2 * HALF))
    UsdPhysics.RigidBodyAPI.Apply(body.GetPrim())
    UsdPhysics.CollisionAPI.Apply(cube.GetPrim())
    mass_api = UsdPhysics.MassAPI.Apply(body.GetPrim())
    mass_api.CreateMassAttr(MASS)
    mass_api.CreateCenterOfMassAttr(Gf.Vec3f(0.0))
    stage.GetRootLayer().Save()
    return {"usd": os.path.abspath(path), "drop_z_m": DROP_Z, "mass_kg": MASS,
            "half_m": HALF, "expected_contact_n": round(MASS * 9.81, 6)}


def main():
    out_dir = os.path.abspath(ARGS.out_dir)
    os.makedirs(out_dir, exist_ok=True)
    authored = author(os.path.join(out_dir, "authored_control.usda"))

    physics = PhysxCfg(solver_type=1, enable_external_forces_every_iteration=True)
    sim = sim_utils.SimulationContext(
        sim_utils.SimulationCfg(dt=DT, device="cuda:0", gravity=(0.0, 0.0, -9.81),
                                physics=physics))
    ground = sim_utils.GroundPlaneCfg()
    ground.func("/World/ground", ground)
    spawn = sim_utils.UsdFileCfg(usd_path=authored["usd"], activate_contact_sensors=True)
    spawn.func("/World/G4", spawn)
    body = RigidObject(RigidObjectCfg(prim_path="/World/G4/Cube", spawn=None))
    sensor = ContactSensor(ContactSensorCfg(prim_path="/World/G4/Cube",
                                            update_period=0.0, history_length=0,
                                            debug_vis=False))
    sim.reset()

    jsonl = open(os.path.join(out_dir, "raw_poses.jsonl"), "w", encoding="utf-8")
    frames = []
    max_force = 0.0
    first_contact_step = None

    for step in range(ARGS.steps + 1):
        if step > 0:
            sim.step()
            body.update(DT)
            sensor.update(DT)
        position = [round(float(v), 9) for v in _np(body.data.root_pos_w[0]).reshape(-1)[:3]]
        try:
            forces = _np(sensor.data.net_forces_w).reshape(-1, 3)
            force = round(float(np.max(np.linalg.norm(forces, axis=1))), 9)
        except Exception:
            force = None
        if force:
            max_force = max(max_force, force)
            if first_contact_step is None and force > 1e-6:
                first_contact_step = step
        row = {"run_id": ARGS.run_id, "step": step, "t_s": round(step * DT, 9),
               "pos": position, "contact_force_n": force,
               "finite": all(math.isfinite(v) for v in position)}
        jsonl.write(json.dumps(row, ensure_ascii=False) + "\n")
        jsonl.flush()
        frames.append(row)
    jsonl.close()

    final = frames[-1]
    grounded = first_contact_step is not None
    gate = "PASS" if (grounded and max_force and max_force > 1.0
                      and final["pos"][2] < DROP_Z - 0.05) else "FAIL"
    result = {
        "run_id": ARGS.run_id, "scene_variant": "G4_CONTACT_CONTROL", "gate_result": gate,
        "metrics": {"max_contact_force_n": round(max_force, 6),
                    "first_contact_step": first_contact_step,
                    "final_z_m": final["pos"][2],
                    "drop_height_m": DROP_Z},
        "expected_contact_n": authored["expected_contact_n"],
        "gates": {"contact_reported": bool(grounded),
                  "force_above_1n": bool(max_force and max_force > 1.0),
                  "cube_fell": bool(final["pos"][2] < DROP_Z - 0.05),
                  "all_finite": all(row["finite"] for row in frames)},
    }
    result["gate_result"] = "PASS" if all(result["gates"].values()) else "FAIL"
    with open(os.path.join(out_dir, "result.json"), "w", encoding="utf-8") as handle:
        json.dump(result, handle, ensure_ascii=False, indent=1)
    with open(os.path.join(out_dir, "manifest.json"), "w", encoding="utf-8") as handle:
        json.dump({"stage": "P2-WC", "run_id": ARGS.run_id, "scene_variant": "G4_CONTACT_CONTROL",
                   "gravity_m_s2": [0, 0, -9.81], "physics_dt_s": DT,
                   "steps": ARGS.steps, "command": " ".join(os.sys.argv)},
                  handle, ensure_ascii=False, indent=1)
    print(json.dumps(result, ensure_ascii=False, indent=1))
    print("GATE_RESULT=%s" % result["gate_result"], flush=True)
    SIMULATION_APP.close()
    return 0 if result["gate_result"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
