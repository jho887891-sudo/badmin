# -*- coding: utf-8 -*-
"""Multi-env acceptance harness: N independently targeted B601-RS arms holding their own poses.

Records, per environment: target, final joint positions, hold error, and the running extrema of ALL
eight DOFs over every step of the run (not just joint2/joint3 at the end). Also records a 240 Hz raw
position+velocity series for a probe env, an independent-reset probe that checks the reset env returns
to the rest pose AND that the other envs keep their targets and states, and performance numbers.

Writes one JSON per env count; tools/merge_multienv_evidence.py merges them.
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

REPO = Path("/home/T7/dgut/robot_sim/migration_check")
sys.path.insert(0, str(REPO / "simulation"))

from isaaclab.app import AppLauncher

p = argparse.ArgumentParser()
p.add_argument("--num-envs", type=int, required=True)
p.add_argument("--hold-seconds", type=float, default=2.0)
p.add_argument("--probe-env", type=int, default=0)
p.add_argument("--out", required=True)
p.add_argument("--curve", default="")
p.add_argument("--stiffness", type=float, default=None)
p.add_argument("--damping", type=float, default=None)
AppLauncher.add_app_launcher_args(p)
args = p.parse_args()
launcher = AppLauncher(args)
simulation_app = launcher.app

import numpy as np
import torch
from badminton_scene.multienv_arm import MultiEnvArm
from badminton_scene.scene_layout import REBOT_RS, SIMULATION

DT = 1.0 / 240.0
N = args.num_envs
STEPS = int(round(args.hold_seconds / DT))
ARM = ("joint1", "joint2", "joint3", "joint4", "joint5", "joint6")
PROBE = args.probe_env


def gpu_mib():
    try:
        return int(subprocess.check_output(
            ["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"]
        ).decode().split()[0])
    except Exception:
        return None


gpu_before = gpu_mib()
arm = MultiEnvArm(
    num_envs=N,
    env_spacing=SIMULATION["env_spacing"],
    mount_world=tuple(REBOT_RS["mount_world"]),
    rest_joint_pos=dict(REBOT_RS["rest_joint_pos"]),
    stiffness=float(REBOT_RS["stiffness"] if args.stiffness is None else args.stiffness),
    damping=float(REBOT_RS["damping"] if args.damping is None else args.damping),
)
arm.spawn()
names = list(arm.joint_names)
limits = arm.joint_limits()

# ---- targets: different per env, and each must be inside the official range by construction
rest_vec = [REBOT_RS["rest_joint_pos"].get(nm, 0.0) for nm in names]
targets = torch.tensor([[rest_vec[j] for j in range(len(names))] for _ in range(N)],
                       dtype=torch.float32, device=arm.device)
i1, i3 = names.index("joint1"), names.index("joint3")
for env_id in range(N):
    targets[env_id, i1] = 0.10 * ((env_id % 3) - 1)
    targets[env_id, i3] = 1.0 + 0.15 * ((env_id % 4) - 1.5)
for env_id in range(N):
    for j, nm in enumerate(names):
        lo, hi = limits[j]
        if not (lo - 1e-9 <= float(targets[env_id, j]) <= hi + 1e-9):
            raise SystemExit("target out of range at env %d joint %s: %r not in [%r, %r]"
                             % (env_id, nm, float(targets[env_id, j]), lo, hi))

arm.set_targets(targets)
for _ in range(120):        # settle before the measured window
    arm.step()

# ---- measured window: every step, every env, all 8 DOFs
lo_trace = {nm: float("inf") for nm in names}
hi_trace = {nm: float("-inf") for nm in names}
# Evaluate the complete hold window, not only the last position sample.
targets_cpu = targets.detach().cpu()
arm_indices = [names.index(nm) for nm in ARM]
worst_window_error = torch.zeros(N, dtype=torch.float64)
first_violation = {}
q_start = arm.joint_pos().cpu().clone()
series = {"step": [], "time_s": [], "q": [], "qd": []}
curve = {"step": [], "time_s": [], "joint_pos": []}

t0 = time.perf_counter()
for step in range(1, STEPS + 1):
    arm.step()
    q = arm.joint_pos().cpu()
    qd = arm.joint_vel().cpu()
    # NaN comparisons can silently pass a limit check; fail at the source instead.
    if not torch.isfinite(q).all() or not torch.isfinite(qd).all():
        raise RuntimeError("non-finite joint position/velocity at measured step %d" % step)
    sample_error = (targets_cpu[:, arm_indices] - q[:, arm_indices]).abs().amax(dim=1)
    worst_window_error = torch.maximum(worst_window_error, sample_error.to(torch.float64))
    for j, nm in enumerate(names):
        col = q[:, j]
        lo_trace[nm] = min(lo_trace[nm], float(col.min()))
        hi_trace[nm] = max(hi_trace[nm], float(col.max()))
        if nm not in first_violation:
            lo, hi = limits[j]
            bad = ((col < lo - 1e-6) | (col > hi + 1e-6)).nonzero()
            if len(bad):
                first_violation[nm] = {"step": step, "env": int(bad[0, 0]),
                                       "value": round(float(col[bad[0, 0]]), 6)}
    series["step"].append(step)
    series["time_s"].append(round(step * DT, 4))
    series["q"].append([round(float(v), 6) for v in q[PROBE].tolist()])
    series["qd"].append([round(float(v), 6) for v in qd[PROBE].tolist()])
    if step % 4 == 0:
        curve["step"].append(step)
        curve["time_s"].append(round(step * DT, 4))
        curve["joint_pos"].append([[round(float(v), 6) for v in row] for row in q.tolist()])
wall = time.perf_counter() - t0
gpu_after = gpu_mib()

q_end = arm.joint_pos().cpu()
qd_end = arm.joint_vel().cpu()
ee_z = arm.ee_world_z()

envs = []
for env_id in range(N):
    err = (targets.cpu()[env_id] - q_end[env_id]).abs()
    envs.append({
        "env_id": env_id,
        "target": [round(float(v), 6) for v in targets.cpu()[env_id].tolist()],
        "final_joint_pos": [round(float(v), 6) for v in q_end[env_id].tolist()],
        "final_joint_vel": [round(float(v), 6) for v in qd_end[env_id].tolist()],
        "hold_error_rad": round(max(float(err[names.index(nm)]) for nm in ARM), 6),
        "worst_window_hold_error_rad": round(float(worst_window_error[env_id]), 6),
        "ee_world_z": round(ee_z[env_id], 6),
        "ee_above_mount": round(ee_z[env_id] - REBOT_RS["mount_world"][2], 6),
    })

# all-8-DOF limit trace over the whole run
limit_trace = {}
for j, nm in enumerate(names):
    lo, hi = limits[j]
    limit_trace[nm] = {
        "limit": [lo, hi],
        "observed_min": round(lo_trace[nm], 6),
        "observed_max": round(hi_trace[nm], 6),
        "inside": bool(lo_trace[nm] >= lo - 1e-6 and hi_trace[nm] <= hi + 1e-6),
    }

# velocity vs position consistency on the probe env (signed integration must match the motion)
consistency = {}
for j, nm in enumerate(names):
    qd_col = [row[j] for row in series["qd"]]
    integrated = float(sum(qd_col) * DT)
    measured = float(q_end[PROBE, j] - q_start[PROBE, j])
    consistency[nm] = {
        "measured_delta_rad": round(measured, 6),
        "integrated_vel_delta_rad": round(integrated, 6),
        "mismatch_rad": round(integrated - measured, 6),
        "max_abs_vel_rad_s": round(max(abs(v) for v in qd_col), 6),
        "start": round(float(q_start[PROBE, j]), 6),
        "end": round(float(q_end[PROBE, j]), 6),
    }

# independent reset: env PROBE goes back to rest; nobody else may move or be retargeted
before_pos = arm.joint_pos().cpu().clone()
before_targets = torch.cat([r.data.joint_pos_target for r in arm.robots], dim=0).cpu().clone()
arm.reset([PROBE])
immediate_pos = arm.joint_pos().cpu().clone()
immediate_targets = torch.cat([r.data.joint_pos_target for r in arm.robots], dim=0).cpu().clone()
for _ in range(60):
    arm.step()
after_pos = arm.joint_pos().cpu()
after_targets = torch.cat([r.data.joint_pos_target for r in arm.robots], dim=0).cpu().clone()

rest_ref = torch.tensor([rest_vec], dtype=torch.float32)
others = [e for e in range(N) if e != PROBE]
reset_info = {
    "reset_env": PROBE,
    "env_immediate_error_vs_rest_rad": round(
        float((immediate_pos[PROBE].cpu() - rest_ref[0]).abs().max()), 6),
    "env_after_60_steps_error_vs_rest_rad": round(
        float((after_pos[PROBE].cpu() - rest_ref[0]).abs().max()), 6),
    "env_target_after_reset": [round(float(v), 6) for v in immediate_targets[PROBE].tolist()],
    "env_target_matches_rest": bool(torch.allclose(
        immediate_targets[PROBE], rest_ref[0], rtol=0, atol=1e-9)),
    "env_target_after_60_steps_matches_rest": bool(torch.allclose(
        after_targets[PROBE], rest_ref[0], rtol=0, atol=1e-9)),
    "others_target_unchanged": bool(torch.allclose(
        immediate_targets[others], before_targets[others], rtol=0, atol=1e-9)) if others else True,
    "others_target_unchanged_after_steps": bool(torch.allclose(
        after_targets[others], before_targets[others], rtol=0, atol=1e-9)) if others else True,
    "other_env_drift_rad": {
        str(e): round(float((after_pos[e] - before_pos[e]).abs().max()), 6) for e in others},
}

report = {
    "num_envs": N,
    "hold_seconds": STEPS * DT,
    "hold_seconds_requested": args.hold_seconds,
    "steps_measured": STEPS,
    "physics_dt": DT,
    "assembly": "direct Articulation per env (MultiEnvArm)",
    "gains": {"stiffness": arm.stiffness, "damping": arm.damping},
    "joint_names": names,
    "joint_limits": {nm: limits[i] for i, nm in enumerate(names)},
    "envs": envs,
    "limit_trace": limit_trace,
    "first_limit_violation": first_violation,
    "probe_env": PROBE,
    "probe_series_rows": len(series["step"]),
    "velocity_position_consistency": consistency,
    "independent_reset": reset_info,
    "performance": {
        "wall_seconds": round(wall, 3),
        "steps_per_second": round(STEPS / wall, 1),
        "sim_seconds_per_wall_second": round((STEPS * DT) / wall, 2),
        "gpu_memory_before_mib": gpu_before,
        "gpu_memory_after_mib": gpu_after,
        "gpu_memory_delta_mib": (None if (gpu_before is None or gpu_after is None)
                                 else gpu_after - gpu_before),
        "num_envs": N,
    },
    "verdict": {
        "all_envs_hold": all(e["worst_window_hold_error_rad"] <= 0.05 for e in envs),
        "all_envs_finite": True,  # loop raises immediately upon a non-finite state
        "worst_hold_error_rad": max(e["hold_error_rad"] for e in envs),
        "max_window_hold_error_rad": max(e["worst_window_hold_error_rad"] for e in envs),
        "all_dofs_within_limits": all(v["inside"] for v in limit_trace.values()),
        "max_final_vel_rad_s": round(max(
            max(abs(v) for v in e["final_joint_vel"][:6]) for e in envs), 6),
        "min_ee_above_mount": round(min(e["ee_above_mount"] for e in envs), 6),
        "independent_reset_ok": (
            reset_info["env_immediate_error_vs_rest_rad"] <= 1e-4
            and reset_info["env_after_60_steps_error_vs_rest_rad"] <= 0.05
            and reset_info["env_target_matches_rest"]
            and reset_info["env_target_after_60_steps_matches_rest"]
            and reset_info["others_target_unchanged"]
            and reset_info["others_target_unchanged_after_steps"]
            and all(v <= 1e-3 for v in reset_info["other_env_drift_rad"].values())
        ),
        "velocity_position_consistent": all(
            abs(c["mismatch_rad"]) <= 0.02 for c in consistency.values()),
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
    s = Path(str(c).replace(".csv", "_probe_240hz.csv"))
    with s.open("w", encoding="utf-8") as fh:
        fh.write("step,time_s," + ",".join("q_" + nm for nm in names) + ","
                 + ",".join("qd_" + nm for nm in names) + "\n")
        for i, st in enumerate(series["step"]):
            fh.write("%d,%s,%s,%s\n" % (st, series["time_s"][i],
                                        ",".join(repr(v) for v in series["q"][i]),
                                        ",".join(repr(v) for v in series["qd"][i])))

print("MULTIENV_JSON_START")
print(json.dumps({"num_envs": N, "verdict": report["verdict"],
                  "performance": report["performance"]}, ensure_ascii=False))
print("MULTIENV_JSON_END")
simulation_app.close()
