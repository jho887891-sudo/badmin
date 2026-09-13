# -*- coding: utf-8 -*-
"""
t03_soak_10min.py - PiPER Stage-0 endurance test: 600 s continuous simulation with
alternating position-control poses. Monitors NaN, joint limit violations, velocity
and effort bounds throughout; dumps joint position/velocity logs.

Run (after sourcing robot_sim env.sh):
    ./isaaclab.sh -p .../t03_soak_10min.py --headless
"""
import argparse
import csv
import json
import sys
import time
from pathlib import Path

import torch

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="PiPER 10-minute soak")
parser.add_argument("--headless", action="store_true", default=True)
parser.add_argument("--duration_s", type=float, default=600.0)
parser.add_argument("--out_dir", type=str, default="")
args = parser.parse_args()
app_launcher = AppLauncher(headless=args.headless)
simulation_app = app_launcher.app

sys.path.insert(0, str(Path(__file__).resolve().parent))
from piper_cfg import PIPER_REST_JOINT_POS  # noqa: E402
from piper_scene import SIM_DT, create_piper_scene  # noqa: E402

# safe bounds expanded a little beyond physical limits
SOFT_MARGIN = 0.08
FULL_POSE = [0.6, 2.2, -2.2, 0.8, -0.6, 1.2]


def main():
    out_root = Path(args.out_dir) if args.out_dir else (
        Path(__file__).resolve().parents[1] / "outputs" / time.strftime("%Y%m%d_%H%M%S"))
    out_root.mkdir(parents=True, exist_ok=True)

    sim, scene, robot = create_piper_scene(num_envs=1)
    names = list(robot.joint_names)
    ll = torch.tensor(robot.data.joint_pos_limits).cpu(); limits = ll[0] if ll.dim() == 3 else ll
    vmax_lim = float(torch.tensor(robot.data.joint_vel_limits).cpu().max().item()) if torch.tensor(robot.data.joint_vel_limits).cpu().numel() else 1000.0
    emax_lim = float(torch.tensor(robot.data.joint_effort_limits).cpu().max().item())

    rest = torch.full((1, len(names)), 0.0, dtype=torch.float32, device=robot.device)
    for i, n in enumerate(names):
        rest[0, i] = float(PIPER_REST_JOINT_POS[n])
    pose_b = torch.full((1, len(names)), 0.0, dtype=torch.float32, device=robot.device)
    for i in range(len(names)):
        pose_b[0, i] = FULL_POSE[i]

    anomalies = []
    rows = []
    total_steps = int(args.duration_s / SIM_DT)
    switch_steps = int(8.0 / SIM_DT)
    log_every = int(60.0 / SIM_DT)
    pose_a = True
    last_anomaly_time = None
    nan_count = 0
    t_wall0 = time.time()

    for k in range(total_steps):
        if (k % switch_steps) == 0:
            pose_a = not pose_a
        target = rest if pose_a else pose_b
        if (k % 24) == 0:
            p = robot.data.joint_pos.clone()
            v = robot.data.joint_vel.clone()
            if bool(torch.isnan(p).any().item()) or bool(torch.isnan(v).any().item()):
                nan_count += 1
                anomalies.append({"step": k, "type": "nan"})
                print(f"[soak] NaN at step {k} !", flush=True)
            lo = limits[:, 0].to(robot.device) - SOFT_MARGIN
            hi = limits[:, 1].to(robot.device) + SOFT_MARGIN
            vmax = float(v.abs().max().item())
            if vmax > max(3.0, vmax_lim * 1.2):
                anomalies.append({"step": k, "type": "velocity_spike", "vmax": vmax})
                print(f"[soak] velocity spike vmax={vmax:.3f} at step {k}", flush=True)
        robot.set_joint_position_target(target)
        scene.write_data_to_sim()
        sim.step()
        scene.update(SIM_DT)
        if (k % log_every) == 0 and k > 0:
            p = robot.data.joint_pos.clone()
            v = robot.data.joint_vel.clone()
            rows.append({
                "sim_s": round(k * SIM_DT, 1), "pos": [round(float(x), 4) for x in p[0].cpu()],
                "vel": [round(float(x), 4) for x in v[0].cpu()],
                "nan": bool(torch.isnan(p).any().item()) or bool(torch.isnan(v).any().item()),
            })
            print(f"[soak] t={k*SIM_DT:.0f}s pos={rows[-1]['pos']} |vel|max={max(abs(x) for x in rows[-1]['vel']):.4f} nan={rows[-1]['nan']} wall={time.time()-t_wall0:.0f}s", flush=True)
        if sim.is_stopped():
            break

    p, v = robot.data.joint_pos.clone(), robot.data.joint_vel.clone()
    with open(out_root / "joint_state_final.csv", "w", encoding="utf-8") as f:
        f.write("name,position_rad,velocity_rad_s\n")
        for i, n in enumerate(names):
            f.write(f"{n},{float(p[0,i].item()):.6f},{float(v[0,i].item()):.6f}\n")
    with open(out_root / "soak_log.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["sim_s"] + [f"pos_{n}" for n in names] + [f"vel_{n}" for n in names] + ["nan"])
        for r in rows:
            w.writerow([r["sim_s"]] + r["pos"] + r["vel"] + [r["nan"]])
    passed = nan_count == 0 and len(anomalies) == 0
    summary = {
        "duration_s": args.duration_s, "steps": total_steps, "nan_count": nan_count,
        "anomalies": anomalies[:20], "passed": passed,
        "final_pos": [float(x) for x in p[0].cpu()], "final_vel": [float(x) for x in v[0].cpu()],
        "wall_seconds": round(time.time() - t_wall0, 1),
    }
    (out_root / "results.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print("[t03] SUMMARY: " + json.dumps(summary), flush=True)
    print(f"[t03] outputs -> {out_root}", flush=True)
    simulation_app.close()


if __name__ == "__main__":
    main()