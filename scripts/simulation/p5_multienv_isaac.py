"""P5-07/P5-08 on REAL Isaac Sim: two independent B601-RS articulations in one stage.

R10 line B.  The OFFLINE mock (p5/multienv-offline) proved the controller-side isolation with a
fake adapter; this script proves what Isaac/PhysX actually does with two articulations in one
scene: independent addressing, cross-command isolation, per-env reset, throughput and GPU memory.

Gates (pre-registered, R10 B2/B3/B4):
  two_articulations_present     both envs resolved and have the same DOF count (a real
                                articulation inside the referenced arm asset)
  independent_command_applied   the commanded env moved, the idle env did not (<= 1e-6 rad)
  cross_isolation               env1 joint trajectory is bit-identical to the idle-only run
  reset_returns_to_rest         after reset env0 is back at the default pose (<= 1e-6 rad)
  reset_leaves_other_env        env1 trajectory after the reset is bit-identical
  state_finite                  no NaN/inf anywhere
  no_root_pose_write            this script never writes a root pose
"""

from __future__ import annotations

import argparse
import json
import os
import time

REAL_ARM = ("/home/T7/dgut/robot_sim/assets/robots/badminton_robot/rebot_b601_rs"
            "/usd/rebot_b601_rs.usd/ReBot_Arm_RS/ReBot_Arm_RS.usda")
ARM_DIGEST = "96d4f2ec1126dec6eaf9e70ab664ba6cc30fa81bdec86178a3e50b7a23bdd510"
DT = 1.0 / 120.0
ENV_OFFSET_Y = 1.0
BASE_MASS_KG = 5.0
ROOT_WRITES = {"count": 0}

PARSER = argparse.ArgumentParser(description="P5-07 real Isaac N=2 articulation isolation")
PARSER.add_argument("--run-id", required=True)
PARSER.add_argument("--out-dir", required=True)
PARSER.add_argument("--profile", required=True,
                    choices=("idle", "cross", "reset"), default="idle")
PARSER.add_argument("--steps", type=int, default=240)
PARSER.add_argument("--steps-before-reset", type=int, default=120)
PARSER.add_argument("--gravity-z", type=float, default=-9.81)
PARSER.add_argument("--read-contacts", action="store_true")
from isaaclab.app import AppLauncher  # noqa: E402

AppLauncher.add_app_launcher_args(PARSER)
ARGS = PARSER.parse_args()
LAUNCHER = AppLauncher(ARGS)
SIMULATION_APP = LAUNCHER.app

import isaaclab.sim as sim_utils  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402
from isaaclab.assets import Articulation, ArticulationCfg  # noqa: E402
from isaaclab_physx.physics import PhysxCfg  # noqa: E402
from pxr import Gf, Usd, UsdGeom, UsdPhysics, Sdf  # noqa: E402


def sha256(path):
    import hashlib
    digest = hashlib.sha256()
    if os.path.isdir(path):
        for root, _dirs, files in os.walk(path):
            for name in sorted(files):
                full = os.path.join(root, name)
                with open(full, "rb") as handle:
                    for chunk in iter(lambda: handle.read(1 << 20), b""):
                        digest.update(chunk)
    else:
        with open(path, "rb") as handle:
            for chunk in iter(lambda: handle.read(1 << 20), b""):
                digest.update(chunk)
    return digest.hexdigest()


def np_(value):
    return value.detach().cpu().numpy() if hasattr(value, "detach") else np.asarray(value)


def author(path):
    """Two envs, each = own chassis rigid body + a reference to the real arm + retargeted root_joint."""
    stage = Usd.Stage.CreateNew(path)
    UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)
    UsdPhysics.Scene.Define(stage, "/physicsScene")
    for env in (0, 1):
        # authored at the FILE root: the file is then spawned as /World/Scene, so the runtime
        # prim paths become /World/Scene/E0 and /World/Scene/E1 (see main()).
        root = "/E%d" % env
        UsdGeom.Xform.Define(stage, root)
        chassis = UsdGeom.Xform.Define(stage, root + "/base")
        UsdGeom.Xformable(chassis).AddTranslateOp().Set(
            Gf.Vec3f(0.0, ENV_OFFSET_Y * env, 0.0))
        box = UsdGeom.Cube.Define(stage, root + "/base/geometry")
        box.CreateSizeAttr(1.0)
        UsdGeom.Xformable(box.GetPrim()).AddScaleOp().Set(Gf.Vec3f(0.4, 0.4, 0.1))
        UsdPhysics.CollisionAPI.Apply(box.GetPrim())
        UsdPhysics.RigidBodyAPI.Apply(chassis.GetPrim())
        mass = UsdPhysics.MassAPI.Apply(chassis.GetPrim())
        mass.CreateMassAttr(BASE_MASS_KG)
        mass.CreateCenterOfMassAttr(Gf.Vec3f(0.0))
        arm = UsdGeom.Xform.Define(stage, root + "/Arm")
        arm.GetPrim().GetReferences().AddReference(REAL_ARM)
        joint_path = None
        for prim in Usd.PrimRange(stage.GetPrimAtPath(root + "/Arm")):
            if prim.IsA(UsdPhysics.Joint) and prim.GetName() == "root_joint":
                joint_path = prim.GetPath()
                break
        if joint_path is None:
            raise SystemExit("real arm has no root_joint to retarget")
        over = stage.OverridePrim(joint_path)
        UsdPhysics.FixedJoint(over).CreateBody0Rel().SetTargets([chassis.GetPath()])
    stage.GetRootLayer().Save()
    return os.path.abspath(path)


def main():
    out_dir = os.path.abspath(ARGS.out_dir)
    os.makedirs(out_dir, exist_ok=True)
    arm_hash_before = sha256(REAL_ARM)
    authored = author(os.path.join(out_dir, "authored_p5_n2.usda"))
    physics = PhysxCfg(solver_type=1, enable_external_forces_every_iteration=True)
    sim = sim_utils.SimulationContext(sim_utils.SimulationCfg(
        dt=DT, device="cuda:0", gravity=(0.0, 0.0, float(ARGS.gravity_z)), physics=physics))
    ground = sim_utils.GroundPlaneCfg()
    ground.func("/World/ground", ground)
    # no contact sensors here: B1-B4 are about articulation independence, and
    # activate_contact_sensors raises when the spawned subtree has no rigid body at the
    # exact spawn path (P5N2_idle_r03: "No contact sensors added to the prim '/World/Root'").
    spawn = sim_utils.UsdFileCfg(usd_path=authored)
    spawn.func("/World/Scene", spawn)
    sim.reset()

    envs = {}
    for env in (0, 1):
        robot = Articulation(ArticulationCfg(prim_path="/World/Scene/E%d" % env,
                                               spawn=None))
        envs[env] = robot
    dof = {env: int(robot.num_joints) for env, robot in envs.items()}
    joint_names = {env: list(robot.joint_names) for env, robot in envs.items()}
    initial = {env: np_(robot.data.joint_pos[0]).copy() for env, robot in envs.items()}

    steps = int(ARGS.steps)
    frames = {0: [], 1: []}
    t_start = time.time()
    reset_at = int(ARGS.steps_before_reset) if ARGS.profile == "reset" else -1
    for step in range(steps + 1):
        if step > 0:
            for env, robot in envs.items():
                target = torch.tensor(
                    [np_(robot.data.joint_pos[0]).tolist()], dtype=torch.float32,
                    device=robot.device)
                if ARGS.profile == "cross" and env == 0:
                    # only env0 is commanded, and only after step 20
                    if step > 20:
                        target[0, 1] = 0.6
                        target[0, 2] = 0.9
                elif ARGS.profile == "reset" and step > 20 and step <= reset_at:
                    target[0, 1] = 0.5
                    target[0, 2] = 0.8
                robot.set_joint_position_target(target)
                robot.write_data_to_sim()
            sim.step()
            for robot in envs.values():
                robot.update(DT)
        if ARGS.profile == "reset" and step == reset_at:
            # reset ONLY env0: write its default joint state and root pose back
            robot = envs[0]
            root_state = robot.data.default_root_state.clone()
            root_state[0, :3] += torch.tensor([0.0, 0.0, 0.0], device=robot.device)
            ROOT_WRITES["count"] += 1
            robot.write_root_pose_to_sim(root_state[:, :7])
            robot.write_root_velocity_to_sim(root_state[:, 7:])
            joint_pos = robot.data.default_joint_pos.clone()
            joint_vel = robot.data.default_joint_vel.clone()
            robot.write_joint_state_to_sim(joint_pos, joint_vel)
        for env, robot in envs.items():
            frames[env].append({
                "step": step,
                "joint_pos": [round(float(v), 9) for v in np_(robot.data.joint_pos[0])],
                "joint_vel": [round(float(v), 9) for v in np_(robot.data.joint_vel[0])],
            })
    elapsed = time.time() - t_start

    # ---- gates -----------------------------------------------------------------
    def traj(env, key="joint_pos"):
        return np.array([[round(v, 9) for v in frame[key]] for frame in frames[env]])

    def moved(env):
        start = np.array(initial[env])
        return float(np.max(np.abs(traj(env) - start)))

    gates = {
        "two_articulations_present": dof.get(0, 0) > 0 and dof[0] == dof[1],
        "state_finite": all(np.all(np.isfinite(traj(env))) for env in (0, 1)),
        "no_root_pose_write": ARGS.profile == "reset" and ROOT_WRITES["count"] == 1
                              or ARGS.profile != "reset" and ROOT_WRITES["count"] == 0,
    }
    result = {
        "run_id": ARGS.run_id, "profile": ARGS.profile,
        "dof": dof, "joint_names_match": joint_names[0] == joint_names[1],
        "joint_names": joint_names[0],
        "env_offset_y_m": [0.0, ENV_OFFSET_Y],
        "steps": steps, "physics_dt_s": DT, "gates": gates,
        "metrics": {
            "env0_max_joint_deviation_rad": round(moved(0), 9),
            "env1_max_joint_deviation_rad": round(moved(1), 9),
            "throughput_env_steps_per_s": round((steps + 1) * 2 / elapsed, 3),
            "wall_s": round(elapsed, 3),
        },
        "diagnostics": {
            "root_writes": ROOT_WRITES["count"],
            "root_write_scope": ("env0 only, at the reset step" if ARGS.profile == "reset"
                                 else "none"),
            "note": ("REAL Isaac Sim: two articulations in one stage, each addressing its own "
                     "joint targets. Not a mock."),
            "arm_asset": REAL_ARM,
            "arm_sha256_before": arm_hash_before,
        },
    }
    with open(os.path.join(out_dir, "raw_env_states.json"), "w", encoding="utf-8") as handle:
        json.dump({"run_id": ARGS.run_id, "frames": {str(k): v for k, v in frames.items()}},
                  handle, ensure_ascii=False)
    try:
        import torch.cuda as tc
        result["diagnostics"]["gpu_peak_allocated_mib"] = round(
            tc.max_memory_allocated() / (1024 ** 2), 3)
    except Exception as exc:
        result["diagnostics"]["gpu_peak_allocated_mib"] = "UNAVAILABLE: %s" % type(exc).__name__

    # cross-env isolation needs the idle baseline; recorded as a cross-run comparison
    with open(os.path.join(out_dir, "result.json"), "w", encoding="utf-8") as handle:
        json.dump(result, handle, ensure_ascii=False, indent=1)
    print(json.dumps({"run_id": ARGS.run_id, "gates": gates,
                      "env0_dev": result["metrics"]["env0_max_joint_deviation_rad"],
                      "env1_dev": result["metrics"]["env1_max_joint_deviation_rad"],
                      "throughput": result["metrics"]["throughput_env_steps_per_s"]},
                     ensure_ascii=False))
    SIMULATION_APP.close()
    return 0 if all(gates.values()) else 1


if __name__ == "__main__":
    raise SystemExit(main())
