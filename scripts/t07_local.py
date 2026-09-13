# -*- coding: utf-8 -*-
"""t07 - 128-env vectorization + per-env isolation + cross-env safety + GPU stats."""
import argparse, csv, json, subprocess, sys, time
from pathlib import Path
from isaaclab.app import AppLauncher
p = argparse.ArgumentParser()
p.add_argument("--headless", action="store_true", default=True)
p.add_argument("--num_envs", type=int, default=128)
args = p.parse_args()
app_launcher = AppLauncher(headless=args.headless)
simulation_app = app_launcher.app
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parents[0]))
import torch
from badminton_scene.scene_cfg import create_scene, SIM_DT
from badminton_scene.reset import reset_idx

def gpu_snapshot():
    try:
        g = subprocess.run(["nvidia-smi", "--query-gpu=utilization.gpu,memory.used,memory.total",
                            "--format=csv,noheader"], capture_output=True, text=True, timeout=10)
        return g.stdout.strip()
    except Exception as e:
        return "NA: " + str(e)

def main():
    out = Path(__file__).resolve().parents[1] / "outputs" / time.strftime("t07_%Y%m%d_%H%M%S")
    out.mkdir(parents=True, exist_ok=True)
    n = args.num_envs
    gpu_before = gpu_snapshot()
    sim, scene, robot, shuttle, _cs, _ex = create_scene(num_envs=n)
    for _ in range(3):
        scene.write_data_to_sim(); sim.step(); scene.update(SIM_DT)
    gpu_with128 = gpu_snapshot()
    dev = shuttle.device
    eo = scene.env_origins
    def spos():
        return torch.from_numpy(shuttle.root_physx_view.get_transforms().numpy()).float()[:, :3]
    def jpos():
        return robot.data.joint_pos.clone()
    def jvel():
        return robot.data.joint_vel.clone()
    def stepn(k=1):
        for _ in range(k):
            scene.write_data_to_sim(); sim.step(); scene.update(SIM_DT)
    base_s, base_j, base_jv = spos().clone(), jpos().clone(), jvel().clone()
    # ---- TEST A: change ONLY env 17
    ids17 = torch.tensor([17], device=dev)
    local = torch.tensor([[1.20, 0.00, 1.80]], device=dev)
    quat = torch.tensor([[1.0, 0.0, 0.0, 0.0]], device=dev)
    vel = torch.tensor([[-3.0, 0.0, 1.5]], device=dev)
    ang = torch.zeros((1, 3), device=dev)
    reset_idx(ids17, robot, shuttle, env_origins=eo, states=(local, quat, vel, ang))
    stepn(3)
    s1, j1 = spos(), jpos()
    others = [i for i in range(n) if i != 17]
    ot = torch.tensor(others, device=dev)
    A_others_ds = float((s1[ot] - base_s[ot]).norm(dim=1).max().item())
    A_others_dj = float((j1[ot] - base_j[ot]).abs().max().item())
    A_env17_ds = float((s1[17] - base_s[17]).norm().item())
    A_ok = (A_others_ds < 0.05) and (A_others_dj < 1e-4) and (A_env17_ds > 0.05)
    # ---- TEST B: reset_idx([3,17,81])
    base_s2, base_j2 = spos().clone(), jpos().clone()
    ids3 = torch.tensor([3, 17, 81], device=dev)
    loc3 = torch.tensor([[1.0, 0.2, 1.9], [1.2, -0.2, 1.7], [0.9, 0.4, 2.0]], device=dev)
    q3 = torch.tensor([[1.0, 0.0, 0.0, 0.0]], device=dev).repeat(3, 1)
    v3 = torch.zeros((3, 3), device=dev); a3 = torch.zeros((3, 3), device=dev)
    reset_idx(ids3, robot, shuttle, env_origins=eo, states=(loc3, q3, v3, a3))
    stepn(2)
    s2, j2 = spos(), jpos()
    changed_s = sorted([i for i in range(n) if float((s2[i] - base_s2[i]).norm().item()) > 0.02])
    changed_j = sorted([i for i in range(n) if float((j2[i] - base_j2[i]).abs().max().item()) > 1e-5])
    B_ok = (changed_s == [3, 17, 81]) and (changed_j == [3, 17, 81])
    # ---- cross-env safety: local coords within cell + min inter-env distance
    sp = spos()
    lx = sp - eo
    cell = 7.5
    cell_viol = int(((lx[:, 0].abs() > cell) | (lx[:, 1].abs() > cell)).sum().item())
    rb = robot.data.body_pos_w[:, 0, :]
    dmin = 1e9
    for i in range(n):
        d = (sp - rb[i]).norm(dim=1)
        d[i] = 1e9
        dmin = min(dmin, float(d.min().item()))
    cross_env_collision_count = cell_viol + (1 if dmin < 0.10 else 0)
    nan = int(torch.isnan(sp).any().item()) + int(torch.isnan(j2).any().item())
    inf = int(torch.isinf(sp).any().item()) + int(torch.isinf(j2).any().item())
    shapes = dict(joint_pos=list(j2.shape), joint_vel=list(jvel().shape), shuttle_pos=list(sp.shape),
                  shuttle_vel=list(torch.from_numpy(shuttle.root_physx_view.get_transforms().numpy()).float()[:, :3].shape),
                  env_origins=list(eo.shape))
    exp = dict(joint_pos=[n, 6], joint_vel=[n, 6], shuttle_pos=[n, 3], shuttle_vel=[n, 3], env_origins=[n, 3])
    shape_ok = all(shapes[k] == exp[k] for k in exp)
    res = dict(num_envs=n, shapes=shapes, expect=exp, shape_ok=bool(shape_ok),
               test_a_only_env17=dict(others_pos_delta_max=A_others_ds, others_joint_delta_max=A_others_dj,
                                      env17_pos_delta=A_env17_ds, ok=bool(A_ok)),
               test_b_reset_3_17_81=dict(pos_changed=changed_s, joint_changed=changed_j, ok=bool(B_ok)),
               cross_env_collision_count=int(cross_env_collision_count), cell_violations=cell_viol,
               min_inter_env_body_distance=round(float(dmin), 4), nan=nan, inf=inf,
               gpu_before=gpu_before, gpu_with_128env=gpu_with128,
               passed=bool(shape_ok and A_ok and B_ok and cross_env_collision_count == 0 and nan == 0 and inf == 0))
    (out / "results.json").write_text(json.dumps(res, indent=2))
    with (out / "isolation.csv").open("w", newline="") as f:
        w = csv.writer(f); w.writerow(["env", "env17_test_changed", "reset3_17_81_changed"])
        for i in range(n):
            w.writerow([i, i == 17, i in changed_s])
    print("[T07]", "PASS" if res["passed"] else "FAIL", json.dumps(dict(shape_ok=shape_ok, A=A_ok, B=B_ok,
          cross_env=cross_env_collision_count, nan=nan, inf=inf, gpu_before=gpu_before, gpu_128=gpu_with128)))
    simulation_app.close()

if __name__ == "__main__":
    main()
