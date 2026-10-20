"""P2 / C0 - minimal baseline: two rigid bodies + one FixedJoint.

SPEC (EXECUTION_SPEC_P0_P8.md section 4.3) asks for C0 first: two rigid bodies joined by a
FixedJoint, zero gravity, no ground, no drive, no write_data_to_sim.  If C0 already drifts,
the root cause lives in the mount anchor / body inertia / root API / unexpected velocity or
constraint projection, and the joint chain is innocent.

Every step is recorded, not sampled: world position, the RAW quaternion exactly as the
runtime returns it, its normalised wxyz form, linear and angular velocity, and the root
state.  Each quaternion carries its SOURCE so a USD (wxyz) value can never be compared with
a runtime (xyzw) value by accident.

TEST_PROXY geometry only.  No real B601-RS asset is opened.

Usage:
    python morph_one_p2_c0_minimal.py --run-id C0_r01 --out-dir <dir>
"""

from __future__ import annotations

import argparse
import json
import math
import os
import platform

import numpy as np

from isaaclab.app import AppLauncher

MOUNT_OFFSET = (0.12, -0.05, 0.20)
CHASSIS_HALF = (0.20, 0.15, 0.04)
BODY_HALF = (0.05, 0.05, 0.05)
STEPS = 60
DT = 1.0 / 120.0

PARSER = argparse.ArgumentParser(description="P2 C0 minimal FixedJoint baseline")
PARSER.add_argument("--run-id", required=True)
PARSER.add_argument("--out-dir", required=True)
PARSER.add_argument("--steps", type=int, default=STEPS)
PARSER.add_argument("--include-ground", action="store_true")
PARSER.add_argument("--root-api", choices=["chassis", "xform", "none"], default="chassis",
                    help="SPEC C3: where ArticulationRootAPI is applied")
PARSER.add_argument("--joints", type=int, choices=[0, 1, 2], default=0,
                    help="SPEC C4/C5: number of revolute joints in the chain")
PARSER.add_argument("--variant", default="C0")
PARSER.add_argument("--write-data", action="store_true",
                    help="call write_data_to_sim every step (SPEC C6/C7 factor)")
AppLauncher.add_app_launcher_args(PARSER)
ARGS = PARSER.parse_args()
LAUNCHER = AppLauncher(ARGS)
SIMULATION_APP = LAUNCHER.app

import isaaclab.sim as sim_utils  # noqa: E402
from isaaclab.assets import Articulation, ArticulationCfg  # noqa: E402
from isaaclab_physx.physics import PhysxCfg  # noqa: E402
from pxr import Gf, Sdf, Usd, UsdGeom, UsdPhysics  # noqa: E402

def _np(values):
    """CUDA tensors must be moved to CPU before numpy; np.asarray raises otherwise."""
    if hasattr(values, "detach"):
        values = values.detach().cpu().numpy()
    return np.asarray(values)


USD_SOURCE = "usd_gf_quat_wxyz"
RUNTIME_SOURCE = "isaaclab_body_quat_raw"


def author(path, root_api="chassis", joints=0):
    stage = Usd.Stage.CreateNew(path)
    UsdGeom.SetStageMetersPerUnit(stage, 1.0)
    UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)
    root = UsdGeom.Xform.Define(stage, "/C0")
    stage.SetDefaultPrim(root.GetPrim())
    root.GetPrim().SetCustomDataByKey("label", "TEST_PROXY C0 two bodies + one FixedJoint")

    def box(path, half, translate, mass):
        body = UsdGeom.Xform.Define(stage, Sdf.Path(path))
        if translate is not None:
            UsdGeom.Xformable(body).AddTranslateOp().Set(Gf.Vec3f(*translate))
        cube = UsdGeom.Cube.Define(stage, Sdf.Path(path).AppendChild("geometry"))
        cube.CreateSizeAttr(1.0)
        UsdGeom.Xformable(cube.GetPrim()).AddScaleOp().Set(
            Gf.Vec3f(2 * half[0], 2 * half[1], 2 * half[2]))
        UsdPhysics.RigidBodyAPI.Apply(body.GetPrim())
        UsdPhysics.CollisionAPI.Apply(cube.GetPrim())
        mass_api = UsdPhysics.MassAPI.Apply(body.GetPrim())
        mass_api.CreateMassAttr(mass)
        mass_api.CreateCenterOfMassAttr(Gf.Vec3f(0.0))
        return body.GetPrim()

    chassis = box("/C0/Chassis", CHASSIS_HALF, None, 12.0)
    payload = box("/C0/Payload", BODY_HALF, MOUNT_OFFSET, 2.0)

    joint = UsdPhysics.FixedJoint.Define(stage, "/C0/mount_fixed")
    joint.CreateBody0Rel().SetTargets([chassis.GetPath()])
    joint.CreateBody1Rel().SetTargets([payload.GetPath()])
    joint.CreateLocalPos0Attr().Set(Gf.Vec3f(*MOUNT_OFFSET))
    joint.CreateLocalRot0Attr().Set(Gf.Quatf(1.0, 0.0, 0.0, 0.0))
    joint.CreateLocalPos1Attr().Set(Gf.Vec3f(0.0, 0.0, 0.0))
    joint.CreateLocalRot1Attr().Set(Gf.Quatf(1.0, 0.0, 0.0, 0.0))
    targets = []
    if joints >= 1:
        link1 = box("/C0/Link1", (0.12, 0.02, 0.02),
                    (MOUNT_OFFSET[0] + 0.13, MOUNT_OFFSET[1], MOUNT_OFFSET[2]), 1.0)
        targets.append((payload, link1, (0.08, 0.0, 0.0)))
    if joints >= 2:
        link2 = box("/C0/Link2", (0.09, 0.018, 0.018),
                    (MOUNT_OFFSET[0] + 0.36, MOUNT_OFFSET[1], MOUNT_OFFSET[2]), 0.5)
        targets.append((link1, link2, (0.24, 0.0, 0.0)))
    for index, (parent, child, anchor) in enumerate(targets, start=1):
        joint = UsdPhysics.RevoluteJoint.Define(stage, "/C0/joint%d" % index)
        joint.CreateBody0Rel().SetTargets([parent.GetPath()])
        joint.CreateBody1Rel().SetTargets([child.GetPath()])
        joint.CreateAxisAttr("Y")
        joint.CreateLowerLimitAttr(-90.0)
        joint.CreateUpperLimitAttr(90.0)
        joint.CreateLocalPos0Attr().Set(Gf.Vec3f(*anchor))
        joint.CreateLocalPos1Attr().Set(Gf.Vec3f(0.0, 0.0, 0.0))
        joint.CreateLocalRot0Attr().Set(Gf.Quatf(1.0, 0.0, 0.0, 0.0))
        joint.CreateLocalRot1Attr().Set(Gf.Quatf(1.0, 0.0, 0.0, 0.0))
        drive = UsdPhysics.DriveAPI.Apply(joint.GetPrim(), "angular")
        drive.CreateTypeAttr("force")
        drive.CreateStiffnessAttr(50.0)
        drive.CreateDampingAttr(5.0)
        drive.CreateMaxForceAttr(100.0)

    if root_api == "chassis":
        UsdPhysics.ArticulationRootAPI.Apply(chassis.GetPrim())
        root_target = str(chassis.GetPath())
    elif root_api == "xform":
        UsdPhysics.ArticulationRootAPI.Apply(root.GetPrim())
        root_target = str(root.GetPath())
    else:
        root_target = None
    if root_api == "none":
        root_target = None
    stage.GetRootLayer().Save()
    return {"usd": os.path.abspath(path), "root_api_target": root_target,
            "mount_offset_m": list(MOUNT_OFFSET), "joints": joints,
            "root_api_mode": root_api}


def read_authored_world(usd_path):
    """Authored world poses, source-labelled as USD (wxyz)."""
    stage = Usd.Stage.Open(usd_path)
    cache = UsdGeom.XformCache(Usd.TimeCode.Default())
    out = {}
    for name in ("Chassis", "Payload"):
        prim = stage.GetPrimAtPath("/C0/" + name)
        matrix = cache.GetLocalToWorldTransform(prim)
        translation = matrix.ExtractTranslation()
        rotation = matrix.ExtractRotationQuat()
        imaginary = rotation.GetImaginary()
        out[name] = {
            "source": USD_SOURCE,
            "pos": [round(float(v), 12) for v in translation],
            "quat_wxyz": [round(float(rotation.GetReal()), 12),
                          round(float(imaginary[0]), 12),
                          round(float(imaginary[1]), 12),
                          round(float(imaginary[2]), 12)],
        }
    return out


def unit(values):
    arr = np.asarray(values, dtype=float)
    norm = float(np.linalg.norm(arr))
    if not np.isfinite(norm) or norm <= 0.0:
        raise ValueError("degenerate quaternion")
    return arr / norm


def raw_to_wxyz(raw):
    """Runtime raw array for this build reads as (x, y, z, w); normalise to wxyz."""
    arr = np.asarray(raw, dtype=float)
    return unit(np.array([arr[3], arr[0], arr[1], arr[2]]))


def rotation_matrix(quat_wxyz):
    w, x, y, z = unit(quat_wxyz)
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
        [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
        [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)],
    ])


def angle_between(a_wxyz, b_wxyz):
    delta = rotation_matrix(a_wxyz).T @ rotation_matrix(b_wxyz)
    cosine = max(-1.0, min(1.0, (float(np.trace(delta)) - 1.0) / 2.0))
    return float(math.acos(cosine))

def main():
    out_dir = os.path.abspath(ARGS.out_dir)
    os.makedirs(out_dir, exist_ok=True)
    authored_layer = author(os.path.join(out_dir, "authored_proxy.usda"),
                            root_api=ARGS.root_api, joints=ARGS.joints)
    authored = read_authored_world(authored_layer["usd"])

    physics = PhysxCfg(solver_type=1, enable_external_forces_every_iteration=True)
    sim = sim_utils.SimulationContext(
        sim_utils.SimulationCfg(dt=DT, device="cuda:0", gravity=(0.0, 0.0, 0.0),
                                physics=physics))
    if ARGS.include_ground:
        ground = sim_utils.GroundPlaneCfg()
        ground.func("/World/ground", ground)
    spawn = sim_utils.UsdFileCfg(usd_path=authored_layer["usd"])
    spawn.func("/World/C0", spawn)
    # C0 has NO joints, so the default ".*": 0.0 init-state regex cannot match anything and
    # Isaac Lab raises before the first step.  Empty joint dicts are the correct authoring for
    # a two-body, one-FixedJoint articulation.
    cfg = ArticulationCfg(prim_path="/World/C0", spawn=None, actuators={})
    # Version-agnostic: clear the default {".*": 0.0} init-state patterns, which cannot match
    # an articulation that has no joints at all.
    for field in ("joint_pos", "joint_vel"):
        try:
            setattr(cfg.init_state, field, {})
        except Exception as exc:  # pragma: no cover
            print("init_state.%s override failed: %s" % (field, exc), flush=True)
    robot = Articulation(cfg)
    sim.reset()

    body_names = list(robot.body_names)
    index_of = {name: position for position, name in enumerate(body_names)}

    jsonl = open(os.path.join(out_dir, "raw_poses.jsonl"), "w", encoding="utf-8")
    frames = []

    def capture(tag, step):
        row = {"run_id": ARGS.run_id, "tag": tag, "step": step, "t_s": round(step * DT, 9),
               "bodies": {}}
        root_pos = [float(v) for v in _np(robot.data.root_pos_w[0]).reshape(-1)[:3]]
        root_raw = [float(v) for v in _np(robot.data.root_quat_w[0]).reshape(-1)[:4]]
        root_lin = [float(v) for v in _np(robot.data.root_lin_vel_w[0]).reshape(-1)[:3]]
        root_ang = [float(v) for v in _np(robot.data.root_ang_vel_w[0]).reshape(-1)[:3]]
        row["root"] = {
            "pos": [round(v, 12) for v in root_pos],
            "raw_quat": [round(v, 12) for v in root_raw],
            "normalized_quat_wxyz": [round(v, 12) for v in raw_to_wxyz(root_raw)],
            "quat_source": RUNTIME_SOURCE,
            "lin_vel": [round(v, 12) for v in root_lin],
            "ang_vel": [round(v, 12) for v in root_ang],
        }
        for name in body_names:
            if name not in index_of:
                row["bodies"][name] = {"found": False}
                continue
            index = index_of[name]
            raw = [float(v) for v in _np(robot.data.body_quat_w[0, index]).reshape(-1)[:4]]
            row["bodies"][name] = {
                "pos": [round(float(v), 12) for v in
                        _np(robot.data.body_pos_w[0, index]).reshape(-1)[:3]],
                "raw_quat": [round(v, 12) for v in raw],
                "normalized_quat_wxyz": [round(v, 12) for v in raw_to_wxyz(raw)],
                "quat_source": RUNTIME_SOURCE,
                "lin_vel": [round(float(v), 12) for v in
                            _np(robot.data.body_lin_vel_w[0, index]).reshape(-1)[:3]],
                "ang_vel": [round(float(v), 12) for v in
                            _np(robot.data.body_ang_vel_w[0, index]).reshape(-1)[:3]],
            }
        row["finite"] = all(
            math.isfinite(v)
            for entry in row["bodies"].values()
            for key in ("pos", "raw_quat", "lin_vel", "ang_vel")
            for v in entry.get(key, [])
        ) and all(math.isfinite(v) for v in row["root"]["pos"] + row["root"]["raw_quat"]
                  + row["root"]["lin_vel"] + row["root"]["ang_vel"])
        jsonl.write(json.dumps(row, ensure_ascii=False) + "\n")
        jsonl.flush()
        frames.append(row)
        return row

    capture("after_reset_step0", 0)
    for step in range(1, ARGS.steps + 1):
        if ARGS.write_data:
            robot.write_data_to_sim()
        sim.step()
        robot.update(sim.get_physics_dt())
        capture("settle", step)
    jsonl.close()

    def pose(frames_list, tag):
        for row in frames_list:
            if row["tag"] == tag:
                return row
        return None

    reset0 = pose(frames, "after_reset_step0")
    settle = [row for row in frames if row["tag"] == "settle"]

    def mount_error(row):
        chassis = row["bodies"]["Chassis"]
        payload = row["bodies"]["Payload"]
        rotation = rotation_matrix(chassis["normalized_quat_wxyz"])
        relative = rotation.T @ (np.asarray(payload["pos"]) - np.asarray(chassis["pos"]))
        return (float(np.linalg.norm(relative - np.asarray(MOUNT_OFFSET))),
                angle_between(chassis["normalized_quat_wxyz"], payload["normalized_quat_wxyz"]))

    mount_pos_0, mount_rot_0 = mount_error(reset0)
    world_pos_0 = float(np.linalg.norm(
        np.asarray(reset0["bodies"]["Chassis"]["pos"]) - np.asarray(authored["Chassis"]["pos"])))
    world_rot_0 = angle_between(reset0["bodies"]["Chassis"]["normalized_quat_wxyz"],
                                authored["Chassis"]["quat_wxyz"])
    drift = float(np.linalg.norm(
        np.asarray(settle[-1]["bodies"]["Chassis"]["pos"])
        - np.asarray(reset0["bodies"]["Chassis"]["pos"])))
    root_speed_0 = float(np.linalg.norm(reset0["root"]["lin_vel"]))
    root_speed_1 = float(np.linalg.norm(settle[0]["root"]["lin_vel"]))
    velocities = [float(np.linalg.norm(row["root"]["lin_vel"])) for row in settle]

    gates = {
        "t0_world_position_error_m": world_pos_0,
        "t0_world_rotation_error_rad": world_rot_0,
        "t0_mount_position_error_m": mount_pos_0,
        "t0_mount_rotation_error_rad": mount_rot_0,
        "settle_60_step_chassis_drift_m": drift,
    }
    thresholds = {"position_m": 0.001, "rotation_rad": 0.001, "settle_drift_m": 0.001}
    verdict = {
        "t0_world_position_within_tol": world_pos_0 <= thresholds["position_m"],
        "t0_world_rotation_within_tol": world_rot_0 <= thresholds["rotation_rad"],
        "t0_mount_position_within_tol": mount_pos_0 <= thresholds["position_m"],
        "t0_mount_rotation_within_tol": mount_rot_0 <= thresholds["rotation_rad"],
        "settle_drift_within_tol": drift <= thresholds["settle_drift_m"],
        "all_finite": all(row["finite"] for row in frames),
    }
    gate_result = "PASS" if all(verdict.values()) else "FAIL"
    manifest = {
        "stage": "P2", "run_id": ARGS.run_id, "scene_variant": "C0",
        "commit": os.environ.get("GIT_COMMIT", "UNKNOWN"),
        "base_commit": os.environ.get("GIT_BASE", "UNKNOWN"),
        "asset_hashes": {"authored_proxy.usda": "TEST_PROXY"},
        "python_version": platform.python_version(),
        "isaac_sim_version": os.environ.get("ISAAC_SIM_VERSION", "UNKNOWN"),
        "physics_dt_s": DT, "gravity_m_s2": [0.0, 0.0, 0.0],
        "root_api_target": authored_layer["root_api_target"],
        "include_ground": bool(ARGS.include_ground),
        "collision_mode": "default",
        "drive_mode": "none",
        "write_data_to_sim": bool(ARGS.write_data),
        "steps": ARGS.steps,
        "command": " ".join(os.sys.argv),
        "seed": "n/a (no randomness)",
    }
    result = {
        "run_id": ARGS.run_id, "scene_variant": "C0", "gate_result": gate_result,
        "metrics": {k: round(float(v), 12) for k, v in gates.items()},
        "thresholds": thresholds, "gates": verdict,
        "diagnostics": {
            "root_lin_speed_at_step0_m_s": round(root_speed_0, 12),
            "root_lin_speed_after_step1_m_s": round(root_speed_1, 12),
            "root_lin_speed_series_mean_m_s": round(float(np.mean(velocities)), 12),
            "root_lin_speed_series_max_m_s": round(float(np.max(velocities)), 12),
            "speed_series": [round(v, 6) for v in velocities],
            "body_names": body_names,
            "body_quat_raw_order": "xyzw (runtime), normalised to wxyz in this file",
            "contact_signals": "UNKNOWN - not queried in C0",
        },
    }
    with open(os.path.join(out_dir, "manifest.json"), "w", encoding="utf-8") as handle:
        json.dump(manifest, handle, ensure_ascii=False, indent=1)
    with open(os.path.join(out_dir, "result.json"), "w", encoding="utf-8") as handle:
        json.dump(result, handle, ensure_ascii=False, indent=1)
    with open(os.path.join(out_dir, "notes.md"), "w", encoding="utf-8") as handle:
        handle.write("# %s\n\nTEST_PROXY C0 (two bodies + one FixedJoint, zero gravity).\n"
                     "gate_result = %s\n" % (ARGS.run_id, gate_result))
    print(json.dumps({k: result[k] for k in ("run_id", "gate_result", "metrics", "gates")},
                     ensure_ascii=False, indent=1))
    print("GATE_RESULT=%s" % gate_result, flush=True)
    SIMULATION_APP.close()
    return 0 if gate_result == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
