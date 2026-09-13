# -*- coding: utf-8 -*-
"""Render standalone previews of the shuttlecock asset (visual check)."""
import argparse, math, sys
from pathlib import Path
import numpy as np
from isaaclab.app import AppLauncher
p = argparse.ArgumentParser(); AppLauncher.add_app_launcher_args(p)
args = p.parse_args(); args.headless = True; args.enable_cameras = True
app_launcher = AppLauncher(args); simulation_app = app_launcher.app
import torch, omni.usd
from pxr import Usd, UsdGeom, Gf
from PIL import Image
from isaaclab.sensors import Camera, CameraCfg
import isaaclab.sim as sim_utils

W, H = 1400, 1000
OUT = Path('/home/T7/ojh/robot_sim/outputs/evidence')
OUT.mkdir(parents=True, exist_ok=True)
ASSET = '/home/T7/ojh/robot_sim/assets/shuttle/shuttlecock.usd'

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

sim = sim_utils.SimulationContext(sim_utils.SimulationCfg(dt=1.0/60.0, device='cuda:0'))
st = omni.usd.get_context().get_stage()
holder = UsdGeom.Xform.Define(st, '/World/Shuttle'); holder.GetPrim().GetReferences().AddReference(ASSET)
sim_utils.DistantLightCfg(intensity=1000.0, color=(0.9,0.9,0.9)).func('/World/Light', sim_utils.DistantLightCfg(intensity=1000.0), translation=(0,0,3))
sim_utils.DomeLightCfg(intensity=500.0).func('/World/DomeLight', sim_utils.DomeLightCfg(intensity=500.0))
cam_cfg = CameraCfg(prim_path='/World/cam', spawn=sim_utils.PinholeCameraCfg(focal_length=50.0, horizontal_aperture=20.955, clipping_range=(0.01, 100.0)), data_types=['rgb'], width=W, height=H)
cam = Camera(cam_cfg)
sim.reset()

def shoot(name, eye, target, focal=50.0):
    q = look_at_quat(eye, target)
    xf = UsdGeom.Xformable(st.GetPrimAtPath('/World/cam'))
    ops = {o.GetOpType(): o for o in xf.GetOrderedXformOps()}
    (ops.get(UsdGeom.XformOp.TypeTranslate) or xf.AddTranslateOp()).Set(Gf.Vec3d(*[float(v) for v in eye]))
    (ops.get(UsdGeom.XformOp.TypeOrient) or xf.AddOrientOp()).Set(Gf.Quatd(float(q[0]), Gf.Vec3d(*[float(v) for v in q[1:]])))
    for _ in range(6):
        sim.render(); cam.update(1.0/60.0)
    rgb = cam.data.output['rgb']
    arr = rgb.detach().cpu().numpy() if hasattr(rgb,'detach') else np.asarray(rgb)
    img = arr.reshape(H, W, -1)[:,:,:3].astype('uint8')
    Image.fromarray(img).save(OUT / name)
    print('[SHUTTLE_PREVIEW]', name, img.shape[1], 'x', img.shape[0], 'mean', round(float(img.mean()),1), flush=True)

shoot('shuttle_preview_side.png', (0.22, -0.20, 0.06), (0.0, 0.0, 0.040))
shoot('shuttle_preview_top.png', (0.02, 0.14, 0.055), (0.0, 0.0, 0.045))
print('[SHUTTLE_PREVIEW] DONE', flush=True)
simulation_app.close()
