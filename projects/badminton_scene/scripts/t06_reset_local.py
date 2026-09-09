# -*- coding: utf-8 -*-
"""t06 - 1000 per-env resets; isolation measured per-event (pre vs post)."""
import argparse, csv, json, sys, time, random
from pathlib import Path
from isaaclab.app import AppLauncher
p = argparse.ArgumentParser()
p.add_argument("--headless", action="store_true", default=True)
p.add_argument("--num_envs", type=int, default=8)
p.add_argument("--resets", type=int, default=1000)
args = p.parse_args()
app_launcher = AppLauncher(headless=args.headless)
simulation_app = app_launcher.app
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parents[0]))
import torch
from badminton_scene.scene_cfg import create_scene, SIM_DT
from badminton_scene.reset import reset_idx

def main():
    out = Path(__file__).resolve().parents[1] / "outputs" / time.strftime("t06_%Y%m%d_%H%M%S")
    out.mkdir(parents=True, exist_ok=True)
    sim, scene, robot, shuttle, _cs = create_scene(num_envs=args.num_envs)
    N = args.num_envs; dev = shuttle.device
    eo = scene.env_origins
    all_ids = torch.arange(N, device=dev)
    rows = []; leak_ev = 0; nan_ev = 0; bad = 0
    rng = random.Random(7)
    for it in range(args.resets):
        pre_pos = shuttle.data.root_pos_w.clone()
        pre_vel = shuttle.data.root_lin_vel_w.clone()
        k = rng.randint(1, N)
        ids = torch.tensor(rng.sample(range(N), k), device=dev)
        sx = (0.80 + 0.6 * torch.rand(k, device=dev)).unsqueeze(1)
        sy = (-0.8 + 1.6 * torch.rand(k, device=dev)).unsqueeze(1)
        sz = (1.60 + 0.8 * torch.rand(k, device=dev)).unsqueeze(1)
        local = torch.cat([sx, sy, sz], dim=1)
        vel = torch.zeros((k, 3), device=dev); ang = torch.zeros((k, 3), device=dev)
        quat = torch.tensor([[1.0, 0.0, 0.0, 0.0]], device=dev).repeat(k, 1)
        reset_idx(ids, robot, shuttle, env_origins=eo, states=(local, quat, vel, ang))
        scene.write_data_to_sim(); sim.step(); scene.update(SIM_DT)
        exp_world = local + eo[ids]
        ok_sel = bool(torch.all(torch.abs(shuttle.data.root_pos_w[ids] - exp_world) < 0.15).item())
        un = [i for i in range(N) if i not in ids.tolist()]
        leak = False
        if un:
            ut = torch.tensor(un, device=dev)
            delta = (shuttle.data.root_pos_w[ut] - pre_pos[ut]).norm(dim=1)
            # allowed natural motion in one step from prior velocity + contact rest eps
            allowed = pre_vel[ut].norm(dim=1) * SIM_DT * 1.6 + 0.02
            leak = bool(torch.any(delta > allowed).item())
        if leak: leak_ev += 1
        nan = bool(torch.isnan(shuttle.data.root_pos_w).any().item()) or bool(torch.isnan(robot.data.joint_pos).any().item())
        if nan: nan_ev += 1
        if not ok_sel: bad += 1
        rows.append([it, len(ids.tolist()), ok_sel, leak, nan])
    hdr = ["iter", "n_reset", "selected_ok", "leakage", "nan"]
    with (out / "reset_test.csv").open("w", newline="") as f:
        w = csv.writer(f); w.writerow(hdr); w.writerows(rows)
    res = dict(iterations=args.resets, num_envs=N, leak_events=leak_ev, nan_events=nan_ev, bad_sel=bad, passed=(leak_ev == 0 and nan_ev == 0 and bad == 0))
    (out / "results.json").write_text(json.dumps(res, indent=2))
    print("[T06]", "PASS" if res["passed"] else "FAIL", json.dumps(dict(leaks=leak_ev, nans=nan_ev, bad=bad)))
    simulation_app.close()

if __name__ == "__main__":
    main()
