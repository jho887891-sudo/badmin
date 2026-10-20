"""P2 on the REAL B601-RS asset: does the ground-clearance rule hold with the official arm?

The proxy study found that an initial overlap between proxy collision geometry and the static
ground plane is what injected 0.46-0.53 m/s before step 0.  This script repeats the same
measurement with the official B601-RS arm referenced READ-ONLY and a simplified proxy chassis,
so the rule is validated on the asset that P4/P6 will actually use.

The arm asset is never written: only a new root layer references it and retargets the existing
root_joint body0 at the proxy chassis body.  Its sha256 is recorded before and after.

Usage:
    python morph_one_p2_real_arm_mount.py --run-id RM_lift_r01 --out-dir <dir> --lift 0.40
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
CHASSIS_HALF = (0.25, 0.175, 0.05)
STEPS = 60
DT = 1.0 / 120.0

PARSER = argparse.ArgumentParser(description="P2 real-arm mount with ground clearance")
PARSER.add_argument("--run-id", required=True)
PARSER.add_argument("--out-dir", required=True)
PARSER.add_argument("--lift", type=float, default=0.0)
PARSER.add_argument("--steps", type=int, default=STEPS)
PARSER.add_argument("--include-ground", action="store_true", default=True)
PARSER.add_argument("--no-ground", dest="include_ground", action="store_false")
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
    root = UsdGeom.Xform.Define(stage, "/RM")
    stage.SetDefaultPrim(root.GetPrim())
    root.GetPrim().SetCustomDataByKey(
        "label", "TEST_PROXY chassis carrying the REAL B601-RS asset (read-only reference)")

    chassis = UsdGeom.Xform.Define(stage, "/RM/Chassis")
    UsdGeom.Xformable(chassis).AddTranslateOp().Set(Gf.Vec3f(0.0, 0.0, lift))
    cube = UsdGeom.Cube.Define(stage, "/RM/Chassis/geometry")
    cube.CreateSizeAttr(1.0)
    UsdGeom.Xformable(cube.GetPrim()).AddScaleOp().Set(
        Gf.Vec3f(2 * CHASSIS_HALF[0], 2 * CHASSIS_HALF[1], 2 * CHASSIS_HALF[2]))
    UsdPhysics.RigidBodyAPI.Apply(chassis.GetPrim())
    UsdPhysics.CollisionAPI.Apply(cube.GetPrim())
    mass_api = UsdPhysics.MassAPI.Apply(chassis.GetPrim())
    mass_api.CreateMassAttr(25.0)
    mass_api.CreateCenterOfMassAttr(Gf.Vec3f(0.0))

    arm = UsdGeom.Xform.Define(stage, "/RM/Arm")
    UsdGeom.Xformable(arm).AddTranslateOp().Set(Gf.Vec3f(0.0, 0.0, lift))
    arm.GetPrim().GetReferences().AddReference(REAL_ARM)

    joint_path = None
    for prim in Usd.PrimRange(stage.GetPrimAtPath("/RM/Arm")):
        if prim.IsA(UsdPhysics.Joint) and prim.GetName() == "root_joint":
            joint_path = prim.GetPath()
            break
    if joint_path is None:
        raise SystemExit("real arm has no root_joint to retarget")
    over = stage.OverridePrim(joint_path)
    UsdPhysics.FixedJoint(over).CreateBody0Rel().SetTargets([chassis.GetPath()])
    stage.GetRootLayer().Save()
    return {"usd": os.path.abspath(path), "overridden_joint": str(joint_path),
            "lift_m": float(lift), "arm_reference": REAL_ARM}

def main():
    out_dir = os.path.abspath(ARGS.out_dir)
    os.makedirs(out_dir, exist_ok=True)
    arm_hash_before = sha256(REAL_ARM)
    authored_layer = author(os.path.join(out_dir, "authored_mount.usda"), ARGS.lift)

    physics = PhysxCfg(solver_type=1, enable_external_forces_every_iteration=True)
    sim = sim_utils.SimulationContext(
        sim_utils.SimulationCfg(dt=DT, device="cuda:0", gravity=(0.0, 0.0, 0.0),
                                physics=physics))
    if ARGS.include_ground:
        ground = sim_utils.GroundPlaneCfg()
        ground.func("/World/ground", ground)
    spawn = sim_utils.UsdFileCfg(usd_path=authored_layer["usd"])
    spawn.func("/World/RM", spawn)
    actuators = {"all": ImplicitActuatorCfg(joint_names_expr=[".*"], stiffness=300.0,
                                            damping=20.0)}
    robot = Articulation(ArticulationCfg(prim_path="/World/RM", spawn=None,
                                         actuators=actuators))
    sim.reset()

    body_names = list(robot.body_names)
    joint_names = list(robot.joint_names)
    index_of = {name: position for position, name in enumerate(body_names)}
    chassis_index = index_of.get("Chassis")
    base_index = index_of.get("base_link")

    jsonl = open(os.path.join(out_dir, "raw_poses.jsonl"), "w", encoding="utf-8")
    frames = []

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
        row["root"] = {
            "pos": [round(float(v), 12) for v in _np(robot.data.root_pos_w[0]).reshape(-1)[:3]],
            "lin_vel": [round(float(v), 12) for v in
                        _np(robot.data.root_lin_vel_w[0]).reshape(-1)[:3]],
        }
        row["finite"] = all(
            math.isfinite(v)
            for entry in row["bodies"].values()
            for key in ("pos", "raw_quat", "lin_vel", "ang_vel")
            for v in entry[key]
        )
        jsonl.write(json.dumps(row, ensure_ascii=False) + "\n")
        jsonl.flush()
        frames.append(row)

    capture("after_reset_step0", 0)
    for step in range(1, ARGS.steps + 1):
        robot.write_data_to_sim()
        sim.step()
        robot.update(sim.get_physics_dt())
        capture("settle", step)
    jsonl.close()

    reset0 = frames[0]
    settle = frames[1:]
    chassis0 = reset0["bodies"]["Chassis"]
    authored_chassis_pos = [0.0, 0.0, float(ARGS.lift)]
    world_pos_0 = float(np.linalg.norm(
        np.asarray(chassis0["pos"]) - np.asarray(authored_chassis_pos)))
    world_rot_0 = angle_between(chassis0["normalized_quat_wxyz"], [1.0, 0.0, 0.0, 0.0])

    mount_pos = mount_rot = None
    if base_index is not None and chassis_index is not None:
        base0 = reset0["bodies"]["base_link"]
        rotation = rotation_matrix(chassis0["normalized_quat_wxyz"])
        relative = rotation.T @ (np.asarray(base0["pos"]) - np.asarray(chassis0["pos"]))
        mount_pos = float(np.linalg.norm(relative))
        mount_rot = angle_between(chassis0["normalized_quat_wxyz"],
                                  base0["normalized_quat_wxyz"])

    drift = float(np.linalg.norm(
        np.asarray(settle[-1]["bodies"]["Chassis"]["pos"])
        - np.asarray(chassis0["pos"])))
    v0 = float(np.linalg.norm(np.asarray(reset0["root"]["lin_vel"])))
    thresholds = {"position_m": 0.001, "rotation_rad": 0.001, "settle_drift_m": 0.001}
    verdict = {
        "t0_world_position_within_tol": world_pos_0 <= thresholds["position_m"],
        "t0_world_rotation_within_tol": world_rot_0 <= thresholds["rotation_rad"],
        "t0_mount_position_within_tol": (mount_pos is not None
                                         and mount_pos <= thresholds["position_m"]),
        "t0_mount_rotation_within_tol": (mount_rot is not None
                                         and mount_rot <= thresholds["rotation_rad"]),
        "settle_drift_within_tol": drift <= thresholds["settle_drift_m"],
        "all_finite": all(row["finite"] for row in frames),
    }
    gate_result = "PASS" if all(verdict.values()) else "FAIL"
    arm_hash_after = sha256(REAL_ARM)
    manifest = {
        "stage": "P2", "run_id": ARGS.run_id, "scene_variant": "REAL_ARM_MOUNT",
        "arm_asset": REAL_ARM, "arm_sha256": arm_hash_after,
        "arm_asset_unchanged": arm_hash_before == arm_hash_after,
        "physics_dt_s": DT, "gravity_m_s2": [0.0, 0.0, 0.0],
        "include_ground": bool(ARGS.include_ground), "lift_m": float(ARGS.lift),
        "drive_mode": "implicit 300/20, targets untouched",
        "write_data_to_sim": True, "steps": ARGS.steps,
        "command": " ".join(os.sys.argv),
    }
    result = {
        "run_id": ARGS.run_id, "scene_variant": "REAL_ARM_MOUNT", "gate_result": gate_result,
        "metrics": {"t0_world_position_error_m": round(world_pos_0, 12),
                    "t0_world_rotation_error_rad": round(world_rot_0, 12),
                    "t0_mount_position_error_m": None if mount_pos is None else round(mount_pos, 12),
                    "t0_mount_rotation_error_rad": None if mount_rot is None else round(mount_rot, 12),
                    "settle_60_step_chassis_drift_m": round(drift, 12)},
        "thresholds": thresholds, "gates": verdict,
        "diagnostics": {"body_names": body_names, "joint_names": joint_names,
                        "dofs": int(robot.data.joint_pos.shape[1]),
                        "root_lin_speed_at_step0_m_s": round(v0, 12),
                        "chassis_in_articulation": chassis_index is not None,
                        "arm_asset_unchanged": arm_hash_before == arm_hash_after},
    }
    with open(os.path.join(out_dir, "manifest.json"), "w", encoding="utf-8") as handle:
        json.dump(manifest, handle, ensure_ascii=False, indent=1)
    with open(os.path.join(out_dir, "result.json"), "w", encoding="utf-8") as handle:
        json.dump(result, handle, ensure_ascii=False, indent=1)
    print(json.dumps({"run_id": ARGS.run_id, "gate_result": gate_result,
                      "metrics": result["metrics"], "gates": verdict,
                      "dofs": result["diagnostics"]["dofs"],
                      "bodies": len(body_names)}, ensure_ascii=False, indent=1))
    print("GATE_RESULT=%s" % gate_result, flush=True)
    SIMULATION_APP.close()
    return 0 if gate_result == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
