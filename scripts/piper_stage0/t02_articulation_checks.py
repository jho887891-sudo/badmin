# -*- coding: utf-8 -*-
"""
t02_articulation_checks.py - PiPER Stage-0 acceptance battery in Isaac Lab.

Phases:
  import   - USD import + Articulation creation, joint names/order/limits vs URDF
  joints   - dump per-joint limits/velocity/effort, coordinate frames, actuator map
  stability- gravity + held rest pose: no NaN, static stability (drift check)
  single   - single-joint position control (each of the 6 joints, one at a time)
  allaxis  - full 6-axis simultaneous position control

Run (after sourcing robot_sim env.sh):
    ./isaaclab.sh -p /home/T7/dgut/robot_sim/projects/piper_stage0/scripts/t02_articulation_checks.py --headless --phases all
"""
import argparse
import json
import sys
import time
from pathlib import Path

import torch

# app must be started before importing most isaaclab modules
from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="PiPER Stage-0 articulation checks")
parser.add_argument("--headless", action="store_true", default=True, help="Headless mode")
parser.add_argument("--num_envs", type=int, default=1)
parser.add_argument("--phases", type=str, default="all",
                    help="comma list: import,joints,stability,single,allaxis (default all)")
parser.add_argument("--out_dir", type=str, default="")
args = parser.parse_args()

app_launcher = AppLauncher(headless=args.headless)
simulation_app = app_launcher.app

sys.path.insert(0, str(Path(__file__).resolve().parent))
from piper_cfg import PIPER_REST_JOINT_POS  # noqa: E402
from piper_scene import SIM_DT, create_piper_scene  # noqa: E402

# Reference from the AgileX URDF (piper_no_gripper_description.urdf)
URDF_LIMITS = {
    "joint1": (-2.618, 2.168),
    "joint2": (0.0, 3.14),
    "joint3": (-2.967, 0.0),
    "joint4": (-1.745, 1.745),
    "joint5": (-1.22, 1.22),
    "joint6": (-2.0944, 2.0944),
}
URDF_EFFORT = 100.0
URDF_VEL = 5.0

# Per-joint step targets for the single-joint phase (rest pose + offset, inside limits)
SINGLE_STEP = {
    "joint1": 1.0,
    "joint2": 0.8,
    "joint3": -0.8,
    "joint4": 0.6,
    "joint5": -0.5,
    "joint6": 1.2,
}
FULL_POSE = [0.6, 2.2, -2.2, 0.8, -0.6, 1.2]

RESULTS = {}


def norm_2d(t):
    """Strip a leading env axis if present (data arrays may be (N,dof,2) or (dof,2))."""
    tt = torch.tensor(t).cpu()
    return tt[0] if tt.dim() == 3 else tt


def has_nan(*tensors):
    return any(bool(torch.isnan(t).any().item()) for t in tensors if t is not None)


def step_for(sim, scene, robot, seconds, targets=None):
    n = int(round(seconds / SIM_DT))
    if targets is not None:
        t = targets
    else:
        t = robot.data.default_joint_pos.clone()
    for _ in range(n):
        robot.set_joint_position_target(t)
        scene.write_data_to_sim()
        sim.step()
        scene.update(SIM_DT)
    return robot.data.joint_pos.clone(), robot.data.joint_vel.clone()


def phase_import_joints(robot, out_dir):
    res = {}
    names = list(robot.joint_names)
    res["num_joints"] = robot.num_joints
    res["joint_names_order"] = names
    limits = norm_2d(robot.data.joint_pos_limits)  # (n,2)
    vlim = torch.tensor(robot.data.joint_vel_limits).cpu().reshape(-1)
    elim = torch.tensor(robot.data.joint_effort_limits).cpu().reshape(-1)
    res["bodies"] = list(robot.body_names)
    rows = []
    ok_limits = True
    for i, name in enumerate(names):
        lo, hi = float(limits[i, 0]), float(limits[i, 1])
        rlo, rhi = URDF_LIMITS.get(name, (float("nan"), float("nan")))
        lim_ok = abs(lo - rlo) < 1e-2 and abs(hi - rhi) < 1e-2
        ok_limits &= lim_ok
        rows.append({
            "index": i, "name": name, "usd_lower": round(lo, 4), "usd_upper": round(hi, 4),
            "urdf_lower": rlo, "urdf_upper": rhi, "limits_match": bool(lim_ok),
            "vel_limit": round(float(vlim[i]), 3), "effort_limit": round(float(elim[i]), 3),
        })
        print(f"[joints] idx={i} name={name} usd=[{lo:.4f},{hi:.4f}] urdf=[{rlo},{rhi}] match={lim_ok}", flush=True)
    res["limits_match"] = bool(ok_limits)
    with open(out_dir / "joints.csv", "w", encoding="utf-8") as f:
        f.write("index,name,usd_lower,usd_upper,urdf_lower,urdf_upper,limits_match,vel_limit,effort_limit\n")
        for r in rows:
            f.write(f"{r['index']},{r['name']},{r['usd_lower']},{r['usd_upper']},{r['urdf_lower']},"
                    f"{r['urdf_upper']},{r['limits_match']},{r['vel_limit']},{r['effort_limit']}\n")
    return res


def phase_stability(sim, scene, robot, out_dir):
    # settle from rest pose with gravity; hold targets at rest pose
    p0 = robot.data.default_joint_pos.clone()
    vel0 = robot.data.default_joint_vel.clone()
    step_for(sim, scene, robot, 1.0)  # settle
    step_for(sim, scene, robot, 2.0)  # hold
    p, v = robot.data.joint_pos.clone(), robot.data.joint_vel.clone()
    nan = has_nan(p, v)
    drift = float((p - p0).abs().max().item())
    vmax = float(v.abs().max().item())
    stable = (not nan) and drift < 0.05 and vmax < 0.5
    print(f"[stability] max|pos-default|={drift:.5f} rad max|vel|={vmax:.5f} rad/s nan={nan} stable={stable}", flush=True)
    return {"nan": nan, "max_drift_rad": drift, "max_vel_rad_s": vmax, "stable": bool(stable)}


def phase_single(sim, scene, robot, out_dir):
    names = list(robot.joint_names)
    results = []
    all_ok = True
    for i, name in enumerate(names):
        # reset to rest pose
        sim.reset()
        scene.reset()
        p_default = robot.data.default_joint_pos.clone()
        step_for(sim, scene, robot, 0.4, targets=p_default)
        tgt = p_default.clone()
        tgt[0, i] = float(PIPER_REST_JOINT_POS[name]) + SINGLE_STEP[name]
        step_for(sim, scene, robot, 2.5, targets=tgt)
        p, v = robot.data.joint_pos.clone(), robot.data.joint_vel.clone()
        err = float((p[0, i] - tgt[0, i]).abs().item())
        nan = has_nan(p, v)
        ok = (not nan) and err < 0.05 and float(v.abs().max().item()) < 0.5
        all_ok &= ok
        results.append({"joint": name, "cmd": float(tgt[0, i].item()), "achieved": float(p[0, i].item()),
                        "err_rad": err, "nan": nan, "ok": bool(ok)})
        print(f"[single] {name}: cmd={tgt[0,i].item():.4f} achieved={p[0,i].item():.4f} err={err:.5f} nan={nan} ok={ok}", flush=True)
    return {"results": results, "all_ok": bool(all_ok)}


def phase_allaxis(sim, scene, robot, out_dir):
    names = list(robot.joint_names)
    sim.reset()
    scene.reset()
    p_default = robot.data.default_joint_pos.clone()
    step_for(sim, scene, robot, 0.4, targets=p_default)
    tgt = p_default.clone()
    for i in range(len(names)):
        tgt[0, i] = FULL_POSE[i]
    step_for(sim, scene, robot, 3.0, targets=tgt)
    p, v = robot.data.joint_pos.clone(), robot.data.joint_vel.clone()
    errs = [(float((p[0, i] - tgt[0, i]).abs().item())) for i in range(len(names))]
    nan = has_nan(p, v)
    ok = (not nan) and max(errs) < 0.05 and float(v.abs().max().item()) < 0.8
    print(f"[allaxis] cmd={FULL_POSE}")
    print(f"[allaxis] errs={[round(e,5) for e in errs]} nan={nan} ok={ok}", flush=True)
    return {"errs_rad": errs, "nan": nan, "ok": bool(ok), "final_pos": [float(x) for x in p[0].cpu()],
            "final_vel": [float(x) for x in v[0].cpu()]}


def main():
    out_root = Path(args.out_dir) if args.out_dir else (
        Path(__file__).resolve().parents[1] / "outputs" / time.strftime("%Y%m%d_%H%M%S"))
    out_root.mkdir(parents=True, exist_ok=True)
    sim, scene, robot = create_piper_scene(num_envs=args.num_envs)
    print(f"[import] created scene; num_envs={args.num_envs}", flush=True)

    phases = args.phases.split(",") if args.phases != "all" else ["import", "joints", "stability", "single", "allaxis"]
    t0 = time.time()
    if "import" in phases or "joints" in phases:
        RESULTS["articulation"] = phase_import_joints(robot, out_root)
    if "stability" in phases:
        RESULTS["stability"] = phase_stability(sim, scene, robot, out_root)
    if "single" in phases:
        RESULTS["single_joint"] = phase_single(sim, scene, robot, out_root)
    if "allaxis" in phases:
        RESULTS["all_axis"] = phase_allaxis(sim, scene, robot, out_root)
    # final joint state dump
    p, v = robot.data.joint_pos.clone(), robot.data.joint_vel.clone()
    with open(out_root / "joint_state_final.csv", "w", encoding="utf-8") as f:
        f.write("name,position_rad,velocity_rad_s\n")
        for i, name in enumerate(list(robot.joint_names)):
            f.write(f"{name},{float(p[0,i].item()):.6f},{float(v[0,i].item()):.6f}\n")
    RESULTS["wall_seconds"] = round(time.time() - t0, 1)
    (out_root / "results.json").write_text(json.dumps(RESULTS, indent=2, default=str), encoding="utf-8")
    print("[t02] RESULTS_JSON:\n" + json.dumps(RESULTS, indent=2, default=str), flush=True)
    print(f"[t02] outputs -> {out_root}", flush=True)
    simulation_app.close()


if __name__ == "__main__":
    main()