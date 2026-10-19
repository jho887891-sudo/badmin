# -*- coding: utf-8 -*-
"""Multi-env acceptance harness: N independently targeted B601-RS arms holding their own poses.

Records, per environment: target, final joint positions, hold error, limit status and wrist height;
plus an independent-reset probe (reset env 0, measure the others' drift) and performance numbers
(wall time, steps/s, GPU peak memory). Writes one JSON per env count.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

REPO = Path("/home/T7/dgut/robot_sim/migration_check")
sys.path.insert(0, str(REPO / "simulation"))

from isaaclab.app import AppLauncher

p = argparse.ArgumentParser()
p.add_argument("--num-envs", type=int, required=True)
p.add_argument("--hold-seconds", type=float, default=2.0)
p.add_argument("--out", required=True)
p.add_argument("--curve", default="")
p.add_argument("--stiffness", type=float, default=None)
p.add_argument("--damping", type=float, default=None)
AppLauncher.add_app_launcher_args(p)
args = p.parse_args()
launcher = AppLauncher(args)
simulation_app = launcher.app

import torch
from badminton_scene.multienv_arm import MultiEnvArm
from badminton_scene.scene_layout import REBOT_RS, SIMULATION

DT = 1.0 / 240.0
N = args.num_envs
STEPS = int(round(args.hold_seconds / DT))
ARM = ("joint1", "joint2", "joint3", "joint4", "joint5", "joint6")

arm = MultiEnvArm(
    num_envs=N,
    env_spacing=SIMULATION["env_spacing"],
    mount_world=tuple(REBOT_RS["mount_world"]),
    rest_joint_pos=dict(REBOT_RS["rest_joint_pos"]),
    stiffness=float(REBOT_RS["stiffness"] if args.stiffness is None else args.stiffness),
    damping=float(REBOT_RS["damping"] if args.damping is None else args.damping),
)
_gpu_before = None
try:
    import subprocess
    _gpu_before = int(subprocess.check_output(
        ["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"]
    ).decode().split()[0])
except Exception:
    _gpu_before = None
arm.spawn()
names = list(arm.joint_names)
limits = arm.joint_limits()

rest_vec = [REBOT_RS["rest_joint_pos"].get(nm, 0.0) for nm in names]
targets = torch.tensor([[rest_vec[j] for j in range(len(names))] for _ in range(N)],
                       dtype=torch.float32, device=arm.device)
i1, i3 = names.index("joint1"), names.index("joint3")
for env_id in range(N):
    targets[env_id, i1] = 0.10 * ((env_id % 3) - 1)      # -0.10 / 0.00 / +0.10
    targets[env_id, i3] = 1.0 + 0.15 * ((env_id % 4) - 1.5)  # 0.775 .. 1.225, inside [0, 3.14]

arm.set_targets(targets)
for _ in range(120):        # let the initial transient settle before the measured hold
    arm.step()

t0 = time.perf_counter()
curve = {"step": [], "time_s": [], "joint_pos": []}
for step in range(1, STEPS + 1):
    arm.step()
    if step % 4 == 0:
        curve["step"].append(step)
        curve["time_s"].append(round(step * DT, 4))
        curve["joint_pos"].append([[round(float(v), 6) for v in row] for row in arm.joint_pos().cpu().tolist()])
wall = time.perf_counter() - t0
try:
    import subprocess
    _gpu_after = int(subprocess.check_output(
        ["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"]
    ).decode().split()[0])
except Exception:
    _gpu_after = None

pos = arm.joint_pos().cpu()
vel = arm.joint_vel().cpu()
ee_z = arm.ee_world_z()

envs = []
for env_id in range(N):
    err = (targets.cpu()[env_id] - pos[env_id]).abs()
    arm_err = max(float(err[names.index(nm)]) for nm in ARM)
    q2, q3 = float(pos[env_id, names.index("joint2")]), float(pos[env_id, names.index("joint3")])
    inside = (limits[names.index("joint2")][0] - 1e-3 <= q2 <= limits[names.index("joint2")][1] + 1e-3
              and limits[names.index("joint3")][0] - 1e-3 <= q3 <= limits[names.index("joint3")][1] + 1e-3)
    envs.append({
        "env_id": env_id,
        "target": [round(float(v), 6) for v in targets.cpu()[env_id].tolist()],
        "final_joint_pos": [round(float(v), 6) for v in pos[env_id].tolist()],
        "final_joint_vel": [round(float(v), 6) for v in vel[env_id].tolist()],
        "hold_error_rad": round(arm_err, 6),
        "joint2_joint3_inside_limits": bool(inside),
        "ee_world_z": round(ee_z[env_id], 6),
        "ee_above_mount": round(ee_z[env_id] - REBOT_RS["mount_world"][2], 6),
    })

# independent reset: only env 0 goes back to rest, the others must not move
before = arm.joint_pos().cpu().clone()
arm.reset([0])
for _ in range(30):
    arm.step()
after = arm.joint_pos().cpu()
drift = {str(e): round(float((after[e] - before[e]).abs().max()), 6) for e in range(1, N)}
reset_info = {
    "reset_env": 0,
    "env0_after_reset_error": round(float(
        (targets.cpu()[0] - after[0]).abs().max()), 6),
    "other_env_drift_rad": drift,
}

report = {
    "num_envs": N,
    "hold_seconds": args.hold_seconds,
    "steps_measured": STEPS,
    "physics_dt": DT,
    "assembly": "direct Articulation per env (MultiEnvArm)",
    "gains": {"stiffness": arm.stiffness, "damping": arm.damping},
    "joint_names": names,
    "joint_limits": {nm: limits[i] for i, nm in enumerate(names)},
    "envs": envs,
    "independent_reset": reset_info,
    "performance": {
        "wall_seconds": round(wall, 3),
        "steps_per_second": round(STEPS / wall, 1),
        "sim_seconds_per_wall_second": round((STEPS * DT) / wall, 2),
        "gpu_memory_before_mib": _gpu_before,
        "gpu_memory_after_mib": _gpu_after,
        "gpu_memory_delta_mib": (None if (_gpu_before is None or _gpu_after is None)
                                 else _gpu_after - _gpu_before),
        "gpu_memory_used_mib": _gpu_after if _gpu_after is not None else 0,
        "num_envs": N,
    },
    "verdict": {
        "all_envs_hold": all(e["hold_error_rad"] <= 0.05 for e in envs),
        "max_final_vel_rad_s": round(max(
            max(abs(v) for v in e["final_joint_vel"][:6]) for e in envs), 6),
        "worst_hold_error_rad": max(e["hold_error_rad"] for e in envs),
        "all_within_limits": all(e["joint2_joint3_inside_limits"] for e in envs),
        "min_ee_above_mount": round(min(e["ee_above_mount"] for e in envs), 6),
        "independent_reset_ok": all(v <= 1e-3 for v in drift.values()) if drift else True,
    },
}
out = Path(args.out)
out.parent.mkdir(parents=True, exist_ok=True)
out.write_text(json.dumps(report, indent=2, ensure_ascii=False, default=str), encoding="utf-8")
if args.curve:
    c = Path(args.curve)
    c.parent.mkdir(parents=True, exist_ok=True)
    with c.open("w", encoding="utf-8") as fh:
        fh.write("step,time_s,env," + ",".join(names) + "\n")
        for st, t, rows in zip(curve["step"], curve["time_s"], curve["joint_pos"]):
            for env_id, row in enumerate(rows):
                fh.write("%d,%s,%d,%s\n" % (st, t, env_id, ",".join(repr(v) for v in row)))

print("MULTIENV_JSON_START")
print(json.dumps({"num_envs": N, "verdict": report["verdict"],
                  "performance": report["performance"]}, ensure_ascii=False))
print("MULTIENV_JSON_END")
simulation_app.close()
