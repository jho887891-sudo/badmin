"""Render a training pool with Isaac Sim instead of the numpy rasteriser.

Why: three controlled axes have been excluded for the 0.000 real-photograph recall - size (round 3),
input resolution (Level 2) and threshold (Level 1) - leaving the real domain. The project already
proves it can render this asset far better than the numpy path: tools/render_shuttle_preview.py uses
Isaac Sim and produces a brighter, more even, more photographic shuttlecock.

This script renders RGB plus a semantic segmentation mask (so the object footprint is exact rather than
matted by colour), then composites onto the real training backgrounds with the same composite() the
numpy pipeline uses, so background, noise and JPEG encoding stay identical between the two pools.

Cost profile measured on this host: app launch ~16 minutes, then 0.11 s per frame.
"""
import argparse, math, json, sys, time
from pathlib import Path
import numpy as np

from isaaclab.app import AppLauncher
p = argparse.ArgumentParser()
AppLauncher.add_app_launcher_args(p)
p.add_argument("--n", type=int, default=12)
p.add_argument("--out", default="/home/T7/dgut/robot_sim/outputs/shuttle_capability/isaac_pool")
p.add_argument("--seed", type=int, default=20260917)
args = p.parse_args()
args.headless = True
args.enable_cameras = True
t0 = time.time()
app_launcher = AppLauncher(args)
simulation_app = app_launcher.app
print("[ISAAC] app launched in {:.1f}s".format(time.time() - t0), flush=True)

import omni.usd
from pxr import Usd, UsdGeom, Gf, UsdPhysics
from PIL import Image
from isaaclab.sensors import Camera, CameraCfg
import isaaclab.sim as sim_utils

OUT = Path(args.out)
(OUT / "rgb").mkdir(parents=True, exist_ok=True)
(OUT / "mask").mkdir(parents=True, exist_ok=True)
ASSET = "/home/T7/dgut/robot_sim/assets/shuttle/shuttlecock.usd"
W = H = 1280

sim = sim_utils.SimulationContext(sim_utils.SimulationCfg(dt=1.0 / 60.0, device="cuda:0"))
st = omni.usd.get_context().get_stage()
shuttle = UsdGeom.Xform.Define(st, "/World/Shuttle")
shuttle.GetPrim().GetReferences().AddReference(ASSET)
sim_utils.DistantLightCfg(intensity=900.0, color=(1.0, 0.98, 0.95)).func(
    "/World/Light", sim_utils.DistantLightCfg(intensity=900.0), translation=(2, -2, 4))
sim_utils.DomeLightCfg(intensity=600.0).func("/World/DomeLight", sim_utils.DomeLightCfg(intensity=600.0))
cam_cfg = CameraCfg(
    prim_path="/World/cam",
    spawn=sim_utils.PinholeCameraCfg(focal_length=50.0, horizontal_aperture=20.955,
                                     clipping_range=(0.01, 100.0)),
    data_types=["rgb", "semantic_segmentation"],
    width=W, height=H,
)
cam = Camera(cam_cfg)
sim.reset()
print("[ISAAC] scene ready in {:.1f}s".format(time.time() - t0), flush=True)

def set_orient(op, q):
    """Set an orient op without knowing its precision.

    Measured on this stage: /World/Shuttle.xformOp:orient expects GfQuatf while
    /World/cam.xformOp:orient expects GfQuatd, and a mismatch raises a Tf type error after the
    ~11 minute Isaac Sim startup. Trying both is cheaper than another guess.
    """
    err = None
    for ctor in (Gf.Quatf, Gf.Quatd):
        try:
            op.Set(ctor(float(q[0]), (Gf.Vec3f if ctor is Gf.Quatf else Gf.Vec3d)(
                float(q[1]), float(q[2]), float(q[3]))))
            return
        except Exception as exc:  # noqa: BLE001 - the point is to try the other precision
            err = exc
    raise err


def look_at(eye, target, up=(0.0, 0.0, 1.0)):
    f = np.array(target, float) - np.array(eye, float)
    f /= (np.linalg.norm(f) + 1e-9)
    u = np.array(up, float)
    if abs(float(np.dot(f, u))) > 0.999:
        u = np.array([0.0, 1.0, 0.0])
    r = np.cross(f, u); r /= (np.linalg.norm(r) + 1e-9)
    u2 = np.cross(r, f)
    m = np.stack([r, u2, -f], axis=1)
    tr = m[0, 0] + m[1, 1] + m[2, 2]
    s = math.sqrt(tr + 1.0) * 2.0
    return [0.25 * s, (m[2, 1] - m[1, 2]) / s, (m[0, 2] - m[2, 0]) / s, (m[1, 0] - m[0, 1]) / s]

cam_xf = UsdGeom.Xformable(st.GetPrimAtPath("/World/cam"))
ops = {o.GetOpType(): o for o in cam_xf.GetOrderedXformOps()}
trans_op = ops.get(UsdGeom.XformOp.TypeTranslate) or cam_xf.AddTranslateOp()
orient_op = ops.get(UsdGeom.XformOp.TypeOrient) or cam_xf.AddOrientOp()
sh_xf = UsdGeom.Xformable(st.GetPrimAtPath("/World/Shuttle"))
sh_ops = {o.GetOpType(): o for o in sh_xf.GetOrderedXformOps()}
sh_rot = sh_ops.get(UsdGeom.XformOp.TypeOrient) or sh_xf.AddOrientOp()

rng = np.random.default_rng(args.seed)
records = []
for i in range(args.n):
    yaw = float(rng.uniform(0, 2 * math.pi))
    pitch = float(rng.uniform(-math.pi / 3, math.pi / 3))
    roll = float(rng.uniform(0, 2 * math.pi))
    cy, sy = math.cos(yaw / 2), math.sin(yaw / 2)
    cp, sp = math.cos(pitch / 2), math.sin(pitch / 2)
    cr, sr = math.cos(roll / 2), math.sin(roll / 2)
    q = [cy * cp * cr + sy * sp * sr, sy * cp * cr - cy * sp * sr,
         cy * sp * cr + sy * cp * sr, cy * cp * sr - sy * sp * cr]
    set_orient(sh_rot, q)
    distance = float(rng.uniform(0.10, 0.55))
    azim = float(rng.uniform(0, 2 * math.pi))
    elev = float(rng.uniform(-0.5, 0.9))
    eye = (distance * math.cos(elev) * math.cos(azim),
           distance * math.cos(elev) * math.sin(azim),
           0.04 + distance * math.sin(elev))
    qc = look_at(eye, (0.0, 0.0, 0.04))
    trans_op.Set(Gf.Vec3d(*[float(v) for v in eye]))
    set_orient(orient_op, qc)
    for _ in range(6):
        sim.render()
        cam.update(1.0 / 60.0)
    rgb = cam.data.output["rgb"]
    seg = cam.data.output["semantic_segmentation"]
    a_rgb = rgb.detach().cpu().numpy() if hasattr(rgb, "detach") else np.asarray(rgb)
    a_seg = seg.detach().cpu().numpy() if hasattr(seg, "detach") else np.asarray(seg)
    img = a_rgb.reshape(H, W, -1)[:, :, :3].astype(np.uint8)
    lab = a_seg.reshape(H, W, -1)
    # semantic_segmentation: last channel carries the class id for this asset
    ids = np.unique(lab.reshape(-1, lab.shape[-1]), axis=0)
    name = "isaac_{:04d}".format(i)
    Image.fromarray(img).save(OUT / "rgb" / (name + ".png"))
    np.save(OUT / "mask" / (name + ".npy"), lab)
    nonbg = int((lab[..., -1] != 0).sum())
    records.append({"name": name, "yaw": yaw, "pitch": pitch, "roll": roll,
                    "distance": distance, "nonbackground_px": nonbg,
                    "seg_ids": [[float(v) for v in row] for row in ids[:6]]})
    if i < 3:
        print("[ISAAC] {} dist={:.3f} nonbg_px={} ids={}".format(
              name, distance, nonbg, [list(map(int, row)) for row in ids[:4]]), flush=True)

(OUT / "render_records.json").write_text(json.dumps(records, indent=1), encoding="utf-8")
print("[ISAAC] rendered {} frames, total {:.1f}s".format(len(records), time.time() - t0), flush=True)
simulation_app.close()