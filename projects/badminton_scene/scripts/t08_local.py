# -*- coding: utf-8 -*-
"""t08 - 128-env soak >=600 s simulated time; anomalies must be 0."""
import argparse, json, subprocess, sys, time
from pathlib import Path
from isaaclab.app import AppLauncher
p = argparse.ArgumentParser()
p.add_argument("--headless", action="store_true", default=True)
p.add_argument("--num_envs", type=int, default=128)
p.add_argument("--sim_s", type=float, default=600.0)
p.add_argument("--out_dir", type=str, default="")
args = p.parse_args()
app_launcher = AppLauncher(headless=args.headless)
simulation_app = app_launcher.app
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import torch
from badminton_scene.scene_cfg import create_scene, SIM_DT

def main():
    out = Path(args.out_dir) if args.out_dir else (Path(__file__).resolve().parents[1] / "outputs" / time.strftime("t08_%Y%m%d_%H%M%S"))
    out.mkdir(parents=True, exist_ok=True)
    n = args.num_envs
    total_steps = int(round(args.sim_s / SIM_DT))
    sim, scene, robot, shuttle, _cs = create_scene(num_envs=n)
    scene.write_data_to_sim(); sim.step(); scene.update(SIM_DT)
    jlim = robot.data.joint_pos_limits.cpu()
    nan_c = 0; inf_c = 0; vel_exp = 0; joint_bad = 0
    t_wall0 = time.time(); beats = 0
    logf = (out / "run.log").open("w")
    gpu_log = (out / "gpu_stats.txt").open("w")
    def log(msg):
        logf.write(msg + "\n"); logf.flush()
    for i in range(total_steps):
        scene.write_data_to_sim(); sim.step(); scene.update(SIM_DT)
        if i % 1200 != 0:
            continue
        pos = torch.from_numpy(shuttle.root_physx_view.get_transforms().numpy()).float()[:, :3]
        jp = robot.data.joint_pos.cpu()
        jv = robot.data.joint_vel.cpu()
        nan_c += int(torch.isnan(pos).any().item()) + int(torch.isnan(jp).any().item()) + int(torch.isinf(jp).any().item())
        inf_c += int(torch.isinf(pos).any().item())
        vnorm = jv.abs().max().item() if jv.numel() else 0.0
        if vnorm > 200.0: vel_exp += 1
        below = (jp < (jlim[:, 0] - 0.2)).any().item()
        above = (jp > (jlim[:, 1] + 0.2)).any().item()
        if below or above: joint_bad += 1
        beats += 1
        if beats % 10 == 0:
            log("t=%ds step=%d wall=%.0fs nan=%d inf=%d velExp=%d jointBad=%d" % (int(i * SIM_DT), i, time.time() - t_wall0, nan_c, inf_c, vel_exp, joint_bad))
        try:
            g = subprocess.run(["nvidia-smi", "--query-gpu=utilization.gpu,memory.used,memory.total", "--format=csv,noheader"], capture_output=True, text=True, timeout=10)
            if g.returncode == 0:
                gpu_log.write("t=%ds %s\n" % (int(i * SIM_DT), g.stdout.strip())); gpu_log.flush()
        except Exception:
            pass
    wall = time.time() - t_wall0
    sim_time = total_steps * SIM_DT
    res = dict(num_envs=n, sim_time_s=sim_time, steps=total_steps, wall_s=wall,
               nan_events=nan_c, inf_events=inf_c, velocity_explosions=vel_exp, joint_violations=joint_bad, anomalies=nan_c + inf_c + vel_exp + joint_bad,
               passed=(nan_c == 0 and inf_c == 0 and vel_exp == 0 and joint_bad == 0))
    (out / "results.json").write_text(json.dumps(res, indent=2))
    log(json.dumps(res))
    logf.close(); gpu_log.close()
    print("[T08]", "PASS" if res["passed"] else "FAIL", json.dumps(dict(sim_s=sim_time, wall_s=round(wall, 1), anomalies=res["anomalies"])))
    simulation_app.close()

if __name__ == "__main__":
    main()
