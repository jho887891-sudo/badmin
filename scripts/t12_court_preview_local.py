# -*- coding: utf-8 -*-
"""t12 - two previews: (1) court asset only, (2) full scene with asset."""
import argparse, sys, math, time
from pathlib import Path
from isaaclab.app import AppLauncher
p = argparse.ArgumentParser()
AppLauncher.add_app_launcher_args(p)
args = p.parse_args()
args.headless = True
args.enable_cameras = True
app_launcher = AppLauncher(args)
simulation_app = app_launcher.app
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "simulation"))
import numpy as np
import torch, omni.usd
from pxr import UsdGeom, Gf
from PIL import Image
from isaaclab.sensors import Camera, CameraCfg
import isaaclab.sim as sim_utils
from badminton_scene.scene_cfg import create_scene, SIM_DT

W, H = 1600, 900
EVID = Path("/home/T7/ojh/robot_sim/outputs/evidence")
EVID.mkdir(parents=True, exist_ok=True)


def look_at_quat(eye, target, up=(0.0, 0.0, 1.0)):
    f = np.array(target, float) - np.array(eye, float); f /= (np.linalg.norm(f) + 1e-9)
    u = np.array(up, float)
    if abs(float(np.dot(f, u))) > 0.999: u = np.array([0.0, 1.0, 0.0])
    r = np.cross(f, u); r /= (np.linalg.norm(r) + 1e-9)
    u2 = np.cross(r, f)
    m = np.stack([r, u2, -f], axis=1)
    tr = m[0,0] + m[1,1] + m[2,2]
    if tr > 0:
        s = math.sqrt(tr + 1.0) * 2.0; w = 0.25*s; x = (m[2,1]-m[1,2])/s; y = (m[0,2]-m[2,0])/s; z = (m[1,0]-m[0,1])/s
    elif m[0,0] > m[1,1] and m[0,0] > m[2,2]:
        s = math.sqrt(1.0+m[0,0]-m[1,1]-m[2,2])*2.0; w = (m[2,1]-m[1,2])/s; x = 0.25*s; y = (m[0,1]+m[1,0])/s; z = (m[0,2]+m[2,0])/s
    elif m[1,1] > m[2,2]:
        s = math.sqrt(1.0+m[1,1]-m[0,0]-m[2,2])*2.0; w = (m[0,2]-m[2,0])/s; x = (m[0,1]+m[1,0])/s; y = 0.25*s; z = (m[1,2]+m[2,1])/s
    else:
        s = math.sqrt(1.0+m[2,2]-m[0,0]-m[1,1])*2.0; w = (m[1,0]-m[0,1])/s; x = (m[0,2]+m[2,0])/s; y = (m[1,2]+m[2,1])/s; z = 0.25*s
    return [w, x, y, z]


def set_cam(path, eye, target):
    q = look_at_quat(eye, target)
    st = omni.usd.get_context().get_stage()
    xf = UsdGeom.Xformable(st.GetPrimAtPath(path))
    ops = {o.GetOpType(): o for o in xf.GetOrderedXformOps()}
    tr = ops.get(UsdGeom.XformOp.TypeTranslate) or xf.AddTranslateOp()
    tr.Set(Gf.Vec3d(float(eye[0]), float(eye[1]), float(eye[2])))
    orr = ops.get(UsdGeom.XformOp.TypeOrient) or xf.AddOrientOp()
    orr.Set(Gf.Quatd(float(q[0]), Gf.Vec3d(float(q[1]), float(q[2]), float(q[3]))))


def shoot(cam, name, eye, target, W, H):
    set_cam("/World/shot_cam", eye, target)
    for _ in range(5):
        sim.render(); cam.update(SIM_DT)
    rgb = cam.data.output["rgb"]
    arr = rgb.detach().cpu().numpy() if hasattr(rgb, "detach") else np.asarray(rgb)
    img = arr.reshape(H, W, -1)[:, :, :3].astype("uint8")
    path = EVID / name
    Image.fromarray(img).save(path)
    print("[T12] shot", path, img.shape[1], "x", img.shape[0], "mean", round(float(img.mean()), 1), flush=True)


def main():
    cam_cfg = CameraCfg(prim_path="/World/shot_cam",
                        spawn=sim_utils.PinholeCameraCfg(focal_length=24.0, horizontal_aperture=20.955,
                                                          clipping_range=(0.05, 1.0e5)),
                        data_types=["rgb"], width=W, height=H)
    global sim
    sim, scene, robot, shuttle, _cs, _extra = create_scene(
        num_envs=1, court_source="asset", extra_sensors=[("shot_cam", cam_cfg, Camera)])
    cam = _extra["shot_cam"]
    for _ in range(5):
        scene.write_data_to_sim(); sim.step(); scene.update(SIM_DT)
    # ---- image 2: full scene (asset + robot + shuttle + ROIs) ----
    shoot(cam, "preview_full_scene.png", (-9.5, -8.0, 5.0), (-0.5, 0.0, 0.8), W, H)
    shoot(cam, "preview_full_scene_top.png", (0.0, 0.0, 16.0), (0.0, 0.001, 0.0), W, H)
    # ---- image 1: only the court asset ----
    st = omni.usd.get_context().get_stage()
    hide = ["/World/envs/env_0/Robot", "/World/envs/env_0/Shuttle", "/World/Ground"]
    for name in ("PerceptionROI", "CandidateStrikeROI", "SpawnRegion", "TargetZone"):
        hide.append("/World/envs/env_0/Court/" + name)
    for hp in hide:
        pr = st.GetPrimAtPath(hp)
        if pr:
            UsdGeom.Imageable(pr).MakeInvisible()
    shoot(cam, "preview_court_only.png", (7.5, -6.5, 4.6), (0.0, 0.0, 0.8), W, H)
    shoot(cam, "preview_court_only_top.png", (0.0, 0.0, 15.0), (0.0, 0.001, 0.0), W, H)
    print("[T12] DONE", flush=True)
    simulation_app.close()


if __name__ == "__main__":
    main()
