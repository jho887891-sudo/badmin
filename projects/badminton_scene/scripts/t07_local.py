# -*- coding: utf-8 -*-
"""t07 - 128-env vectorization: tensor shapes, per-env geometry, cross-env sanity."""
import argparse, json, sys, time
from pathlib import Path
from isaaclab.app import AppLauncher
p = argparse.ArgumentParser()
p.add_argument("--headless", action="store_true", default=True)
p.add_argument("--num_envs", type=int, default=128)
args = p.parse_args()
app_launcher = AppLauncher(headless=args.headless)
simulation_app = app_launcher.app
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import torch, omni.usd
from pxr import UsdGeom
from badminton_scene.scene_cfg import create_scene, SIM_DT

def main():
    out = Path(__file__).resolve().parents[1] / "outputs" / time.strftime("t07_%Y%m%d_%H%M%S")
    out.mkdir(parents=True, exist_ok=True)
    n = args.num_envs
    sim, scene, robot, shuttle, _cs = create_scene(num_envs=n)
    scene.write_data_to_sim(); sim.step(); scene.update(SIM_DT)
    vw = shuttle.root_physx_view
    t0 = torch.from_numpy(vw.get_transforms().numpy()).float()
    scene.write_data_to_sim(); sim.step(); scene.update(SIM_DT)
    t1 = torch.from_numpy(vw.get_transforms().numpy()).float()
    pos_w = t1[:, :3]
    vel_w = (t1[:, :3] - t0[:, :3]) / SIM_DT
    eo = scene.env_origins.cpu().float()
    jp = robot.data.joint_pos.cpu()
    jv = robot.data.joint_vel.cpu()
    shapes = dict(
        joint_pos=list(jp.shape),
        joint_vel=list(jv.shape),
        shuttle_pos=list(pos_w.shape),
        shuttle_vel=list(vel_w.shape),
        env_origins=list(eo.shape),
    )
    nan = bool(torch.isnan(jp).any().item()) or bool(torch.isnan(pos_w).any().item())
    local = pos_w - eo
    within = bool((torch.abs(local[:, 0]) < 7.0).all().item()) and bool((torch.abs(local[:, 1]) < 3.5).all().item())
    st = omni.usd.get_context().get_stage()
    nets = sum(1 for i in range(n) if st.GetPrimAtPath("/World/envs/env_%d/Court/Net/NetCollision" % i))
    expect = dict(joint_pos=(n, 6), joint_vel=(n, 6), shuttle_pos=(n, 3), shuttle_vel=(n, 3), env_origins=(n, 3))
    shape_ok = all(tuple(shapes[k]) == expect[k] for k in expect)
    result = dict(num_envs=n, shapes=shapes, expect=expect, shape_ok=bool(shape_ok), nan=nan,
                  within_cell=bool(within), net_collision_prims=nets, env_spacing=15.0,
                  vel_note="shuttle velocity derived from physx get_transforms diff/dt")
    (out / "results.json").write_text(json.dumps(result, indent=2, default=str))
    ok = bool(shape_ok) and not nan and bool(within) and nets == n
    print("[T07]", "PASS" if ok else "FAIL", json.dumps(dict(shapes=shapes, nan=nan, within=bool(within), nets=nets)))
    simulation_app.close()

if __name__ == "__main__":
    main()
