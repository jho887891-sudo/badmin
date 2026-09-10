# -*- coding: utf-8 -*-
"""t08 - 128-env soak >=600 s sim with spawn/flight/contact/reset cycles."""
import argparse, csv, json, subprocess, sys, time
from pathlib import Path
from isaaclab.app import AppLauncher
p = argparse.ArgumentParser()
p.add_argument("--headless", action="store_true", default=True)
p.add_argument("--num_envs", type=int, default=128)
p.add_argument("--sim_s", type=float, default=600.0)
p.add_argument("--reset_every", type=int, default=1200)
p.add_argument("--reset_batch", type=int, default=16)
args = p.parse_args()
app_launcher = AppLauncher(headless=args.headless)
simulation_app = app_launcher.app
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parents[0]))
import torch
from isaaclab.sensors import ContactSensorCfg
from badminton_scene.scene_cfg import create_scene, SIM_DT
from badminton_scene.reset import reset_idx
from badminton_scene.scene_layout import SPAWN_REGION

def gpu_snapshot():
    try:
        g = subprocess.run(["nvidia-smi", "--query-gpu=utilization.gpu,memory.used,memory.total",
                            "--format=csv,noheader"], capture_output=True, text=True, timeout=10)
        return g.stdout.strip()
    except Exception as e:
        return "NA: " + str(e)

def main():
    out = Path(__file__).resolve().parents[1] / "outputs" / time.strftime("t08_%Y%m%d_%H%M%S")
    out.mkdir(parents=True, exist_ok=True)
    n = args.num_envs
    total_steps = int(round(args.sim_s / SIM_DT))
    cfg = ContactSensorCfg(prim_path="/World/envs/env_.*/Shuttle", history_length=0, update_period=0.0)
    sim, scene, robot, shuttle, sensor, _ex = create_scene(num_envs=n, contact_cfg=cfg)
    dev = shuttle.device
    eo = scene.env_origins
    logf = (out / "run.log").open("w")
    csvf = (out / "soak.csv").open("w", newline="")
    gpu_log = (out / "gpu_stats.txt").open("w")
    w = csv.writer(csvf)
    w.writerow(["sim_t", "step", "wall_s", "nan", "inf", "reset_ok", "reset_fail",
                "cross_env", "contact_events", "gpu_util", "gpu_mem_used"])
    def log(m):
        logf.write(m + "\n"); logf.flush()
    nan_c = inf_c = reset_fail = cross_env_c = cuda_fatal = physx_fatal = contact_events = 0
    reset_ok = 0
    t0 = time.time()
    rng = torch.Generator(device="cpu"); rng.manual_seed(7)
    def spos():
        return torch.from_numpy(shuttle.root_physx_view.get_transforms().numpy()).float()[:, :3]
    try:
        for i in range(total_steps):
            # periodic spawn/flight/reset cycle
            if i > 0 and i % args.reset_every == 0:
                k = args.reset_batch
                ids = torch.randperm(n, generator=rng)[:k].to(dev)
                sx = (SPAWN_REGION["world_x"][0] + (SPAWN_REGION["world_x"][1] - SPAWN_REGION["world_x"][0]) * torch.rand(k, generator=rng)).unsqueeze(1)
                sy = (SPAWN_REGION["world_y"][0] + (SPAWN_REGION["world_y"][1] - SPAWN_REGION["world_y"][0]) * torch.rand(k, generator=rng)).unsqueeze(1)
                sz = (SPAWN_REGION["z"][0] + (SPAWN_REGION["z"][1] - SPAWN_REGION["z"][0]) * torch.rand(k, generator=rng)).unsqueeze(1)
                local = torch.cat([sx, sy, sz], dim=1).to(dev)
                quat = torch.tensor([[1.0, 0.0, 0.0, 0.0]], device=dev).repeat(k, 1)
                vel = torch.tensor([[-3.0, 0.0, 1.5]], device=dev).repeat(k, 1)
                vel[:, 0] += 0.4 * (torch.rand(k, device=dev) - 0.5)
                vel[:, 1] += 0.6 * (torch.rand(k, device=dev) - 0.5)
                ang = torch.zeros((k, 3), device=dev)
                reset_idx(ids, robot, shuttle, env_origins=eo, states=(local, quat, vel, ang))
                scene.write_data_to_sim(); sim.step(); scene.update(SIM_DT)
                exp_world = local + eo[ids]
                err = (spos()[ids] - exp_world).norm(dim=1).max().item()
                if err < 0.25:
                    reset_ok += 1
                else:
                    reset_fail += 1
                continue
            scene.write_data_to_sim(); sim.step(); scene.update(SIM_DT)
            if i % 1200 == 0:
                sensor.update(SIM_DT)
                sp = spos()
                jp = robot.data.joint_pos.cpu()
                nan_c += int(torch.isnan(sp).any().item()) + int(torch.isnan(jp).any().item())
                inf_c += int(torch.isinf(sp).any().item()) + int(torch.isinf(jp).any().item())
                lx = sp - eo.cpu().float()
                cell_viol = int(((lx[:, 0].abs() > 7.5) | (lx[:, 1].abs() > 7.5)).sum().item())
                rb = robot.data.body_pos_w[:, 0, :].cpu()
                dmin = 1e9
                for e in range(0, n, 16):
                    d = (sp - rb[e]).norm(dim=1); d[e] = 1e9
                    dmin = min(dmin, float(d.min().item()))
                cross_env_c += cell_viol + (1 if dmin < 0.05 else 0)
                f = sensor.data.net_forces_w.torch
                contact_events += int((torch.linalg.norm(f, dim=-1) > 1e-4).sum().item())
                g = gpu_snapshot()
                gpu_log.write("t=%ds %s\n" % (int(i * SIM_DT), g)); gpu_log.flush()
                w.writerow([round(i * SIM_DT, 2), i, round(time.time() - t0, 1), nan_c, inf_c, reset_ok,
                            reset_fail, cross_env_c, contact_events, g.split(",")[0].strip() if g else "NA",
                            g.split(",")[1].strip() if "," in g else "NA"])
                csvf.flush()
                if (i // 1200) % 10 == 0:
                    log("t=%ds step=%d wall=%.0fs nan=%d inf=%d reset_ok=%d reset_fail=%d cross=%d contacts=%d"
                        % (int(i * SIM_DT), i, time.time() - t0, nan_c, inf_c, reset_ok, reset_fail,
                           cross_env_c, contact_events))
    except Exception as e:
        msg = str(e)
        if "CUDA" in msg.upper():
            cuda_fatal += 1
        if "PhysX" in msg or "physx" in msg:
            physx_fatal += 1
        log("FATAL at step %d: %s" % (i, msg))
    wall = time.time() - t0
    sim_time = total_steps * SIM_DT
    anomalies = nan_c + inf_c + reset_fail + cross_env_c + cuda_fatal + physx_fatal
    res = dict(num_envs=n, sim_time_s=round(sim_time, 2), steps=total_steps, wall_s=round(wall, 1),
               nan=nan_c, inf=inf_c, reset_ok=reset_ok, reset_fail=reset_fail,
               cross_env_collision=cross_env_c, contact_events=contact_events,
               cuda_fatal=cuda_fatal, physx_fatal=physx_fatal, anomalies=anomalies,
               passed=bool(anomalies == 0 and sim_time >= 600.0 and reset_ok > 0 and contact_events > 0),
               gpu_final=gpu_snapshot())
    (out / "results.json").write_text(json.dumps(res, indent=2))
    log(json.dumps(res))
    logf.close(); csvf.close(); gpu_log.close()
    print("[T08]", "PASS" if res["passed"] else "FAIL", json.dumps(dict(sim_s=res["sim_time_s"], wall_s=res["wall_s"], anomalies=anomalies, reset_ok=reset_ok, reset_fail=reset_fail, contacts=contact_events)))
    simulation_app.close()

if __name__ == "__main__":
    main()
