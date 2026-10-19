# -*- coding: utf-8 -*-
"""F1 acceptance harness: hold the B601-RS rest pose for 5 s, in the bare arm and in the scene.

Run on the GPU host (Isaac Sim + Isaac Lab). Writes one JSON per run plus a 60 Hz curve CSV; a merge
step combines the runs into outputs/simulation/rebot_b601_rs/HOLD_rest_pose.json, which
tests/simulation/rebot_b601_rs/test_rest_pose_hold.py turns into a regression gate.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "simulation"))

from isaaclab.app import AppLauncher

p = argparse.ArgumentParser()
p.add_argument("--mode", choices=["standalone", "scene"], required=True)
p.add_argument("--out", required=True)
p.add_argument("--curve", default="")
AppLauncher.add_app_launcher_args(p)
args = p.parse_args()
launcher = AppLauncher(args)
simulation_app = launcher.app

import numpy as np
import torch
from badminton_scene.robot_cfg import REBOT_RS_PRIM_PATH, rebot_rs_spawn_cfg, rebot_rs_actuators
from badminton_scene.scene_layout import REBOT_RS

DT = 1.0 / 240.0
STEPS = 1200          # 5 s at 240 Hz
CHECKPOINTS = (0, 1, 5, 30, 60, 240, 600, 1200)
REST = dict(REBOT_RS["rest_joint_pos"])
MOUNT = tuple(REBOT_RS["mount_world"])

report: dict = {
    "name": args.mode + "_hold_rest",
    "mode": args.mode,
    "physics_dt": DT,
    "steps_total": STEPS,
    "rest_pose": REST,
    "mount_world": list(MOUNT),
    "usd_path": str(REBOT_RS["usd"]),
    "unsupported_api": {},
}

if args.mode == "scene":
    from badminton_scene.scene_cfg import create_scene, step_env

    sim, scene, robot, shuttle, contact, extra = create_scene(
        num_envs=1, include_robot=True, include_shuttle=False, include_ground=True)
    report["prim_path"] = str(robot.cfg.prim_path)
    report["direct_spawn"] = bool(getattr(scene, "direct_articulations", None))
else:
    import isaaclab.sim as sim_utils
    from isaaclab.assets import Articulation, ArticulationCfg

    sim = sim_utils.SimulationContext(sim_utils.SimulationCfg(dt=DT, device="cuda:0"))
    ground = sim_utils.GroundPlaneCfg()
    ground.func("/World/ground", ground)
    spawn = rebot_rs_spawn_cfg()
    spawn.func(REBOT_RS_PRIM_PATH, spawn, translation=MOUNT)  # mount at spawn: see F1 note
    robot = Articulation(ArticulationCfg(
        prim_path=REBOT_RS_PRIM_PATH, spawn=None, actuators=rebot_rs_actuators()))
    sim.reset()
    _jp0 = torch.tensor([[REST.get(nm, 0.0) for nm in robot.joint_names]],
                        dtype=torch.float32, device=robot.device)
    robot.write_joint_state_to_sim(_jp0, torch.zeros_like(_jp0))
    robot.set_joint_position_target(_jp0)
    robot.write_data_to_sim()
    report["prim_path"] = REBOT_RS_PRIM_PATH
    report["direct_spawn"] = True

names = list(robot.joint_names)
report["joint_names"] = names
report["joint_indices"] = {nm: i for i, nm in enumerate(names)}
n = robot.num_joints


def step_once():
    if args.mode == "scene":
        step_env(scene, sim)
    else:
        robot.write_data_to_sim()
        sim.step()
        robot.update(sim.get_physics_dt())


def flat(v):
    if hasattr(v, "detach"):          # torch tensor (possibly CUDA)
        v = v.detach().cpu().numpy()
    elif hasattr(v, "numpy"):         # warp / numpy-like
        v = v.numpy()
    return [round(float(x), 6) for x in np.asarray(v).ravel()]


report["default_joint_pos"] = dict(zip(names, flat(robot.data.default_joint_pos[0])))
report["joint_limits"] = {nm: [round(float(x), 6) for x in lim]
                          for nm, lim in zip(names, robot.data.joint_limits[0].cpu().tolist())}
for label, getter in (("physx_dof_stiffnesses", lambda: robot.root_physx_view.get_dof_stiffnesses()),
                      ("physx_dof_dampings", lambda: robot.root_physx_view.get_dof_dampings()),
                      ("physx_dof_max_forces", lambda: robot.root_physx_view.get_dof_max_forces())):
    try:
        report[label] = flat(getter())
    except Exception as exc:  # noqa: BLE001
        report["unsupported_api"][label] = type(exc).__name__ + ": " + str(exc)[:100]
for label, getter in (("physx_dof_positions", lambda: robot.root_physx_view.get_dof_positions()),
                      ("physx_dof_position_targets", lambda: robot.root_physx_view.get_dof_position_targets()),
                      ("applied_joint_efforts", lambda: robot.root_physx_view.get_applied_joint_efforts())):
    try:
        report[label] = flat(getter()[0])
    except Exception as exc:  # noqa: BLE001
        report["unsupported_api"][label] = type(exc).__name__ + ": " + str(exc)[:100]

try:
    report["actuators"] = [
        {"type": type(act).__name__,
         "joint_names": list(act.joint_names),
         "stiffness": act.stiffness,
         "damping": act.damping,
         "effort_limit_sim": act.effort_limit_sim,
         "velocity_limit_sim": act.velocity_limit_sim}
        for act in robot.actuators.values()
    ]
except Exception as exc:  # noqa: BLE001
    report["unsupported_api"]["actuators"] = type(exc).__name__ + ": " + str(exc)[:100]

# start from the rest pose and command it, so "hold" means hold, not "arrive eventually"
target = torch.tensor([[REST.get(nm, 0.0) for nm in names]], dtype=torch.float32, device=robot.device)
robot.write_joint_state_to_sim(target, torch.zeros_like(target))
robot.set_joint_position_target(target)
robot.write_data_to_sim()

ee_idx = list(robot.body_names).index("link6")
mount_z = MOUNT[2]


def snapshot(step: int) -> dict:
    q = [float(x) for x in robot.data.joint_pos[0].cpu().tolist()]
    qd = [float(x) for x in robot.data.joint_vel[0].cpu().tolist()]
    qt = [float(x) for x in robot.data.joint_pos_target[0].cpu().tolist()]
    ee = [float(x) for x in robot.data.body_pos_w[0, ee_idx].cpu().tolist()]
    body_z = [float(x) for x in robot.data.body_pos_w[0, :, 2].cpu().tolist()]
    sample = {
        "step": step,
        "time_s": round(step * DT, 4),
        "joint_pos": [round(v, 6) for v in q],
        "joint_vel": [round(v, 6) for v in qd],
        "joint_pos_target": [round(v, 6) for v in qt],
        "hold_error_rad": [round(t - v, 6) for t, v in zip(qt, q)],
        "ee_world": [round(v, 6) for v in ee],
        "ee_rel_mount": [round(ee[0], 6), round(ee[1], 6), round(ee[2] - mount_z, 6)],
        "min_body_z": round(min(body_z), 6),
    }
    for label, getter in (("applied_joint_efforts", lambda: robot.root_physx_view.get_applied_joint_efforts()),
                          ("physx_dof_position_targets", lambda: robot.root_physx_view.get_dof_position_targets())):
        try:
            sample[label] = flat(getter()[0])
        except Exception as exc:  # noqa: BLE001
            sample[label] = "UNAVAILABLE:" + type(exc).__name__
    return sample


samples = [snapshot(0)]
curve_step = [0, 0.0, list(robot.data.joint_pos[0].tolist())]
curve = {"step": [0], "time_s": [0.0], "joint_pos": [[float(v) for v in robot.data.joint_pos[0].tolist()]]}
for step in range(1, STEPS + 1):
    step_once()
    if step in CHECKPOINTS:
        samples.append(snapshot(step))
    if step % 4 == 0:
        curve["step"].append(step)
        curve["time_s"].append(round(step * DT, 4))
        curve["joint_pos"].append([float(v) for v in robot.data.joint_pos[0].tolist()])

report["samples"] = samples
last = samples[-1]
arm = [i for i, nm in enumerate(names) if nm.startswith("joint")]
report["verdict"] = {
    "held_5s": max(abs(last["hold_error_rad"][i]) for i in arm) <= 0.05,
    "worst_hold_error_rad": round(max(abs(last["hold_error_rad"][i]) for i in arm), 6),
    "joint2_joint3_inside_limits": all(
        report["joint_limits"][nm][0] - 1e-3 <= last["joint_pos"][names.index(nm)] <=
        report["joint_limits"][nm][1] + 1e-3 for nm in ("joint2", "joint3")),
    "ee_final_rel_mount_z": last["ee_rel_mount"][2],
    "ee_drop_m": round(samples[0]["ee_rel_mount"][2] - min(s["ee_rel_mount"][2] for s in samples), 6),
    "worst_body_below_ground": round(min(s["min_body_z"] for s in samples), 6),
}

out = Path(args.out)
out.parent.mkdir(parents=True, exist_ok=True)
out.write_text(json.dumps(report, indent=2, ensure_ascii=False, default=str), encoding="utf-8")
if args.curve:
    c = Path(args.curve)
    c.parent.mkdir(parents=True, exist_ok=True)
    with c.open("w", encoding="utf-8") as fh:
        fh.write("step,time_s," + ",".join(names) + "\n")
        for st, t, q in zip(curve["step"], curve["time_s"], curve["joint_pos"]):
            fh.write(str(st) + "," + str(t) + "," + ",".join(repr(round(v, 6)) for v in q) + "\n")

print("HOLD_JSON_START")
print(json.dumps({"name": report["name"], "mode": args.mode, "direct_spawn": report["direct_spawn"],
                  "verdict": report["verdict"],
                  "unsupported": sorted(report["unsupported_api"])}, ensure_ascii=False))
print("HOLD_JSON_END")
simulation_app.close()
