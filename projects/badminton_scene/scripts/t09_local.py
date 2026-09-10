# -*- coding: utf-8 -*-
"""t09 - real Isaac renders (IsaacLab Camera/RTX) + projection visibility evidence."""
import argparse, json, math, sys
from pathlib import Path
import numpy as np
from isaaclab.app import AppLauncher
p = argparse.ArgumentParser()
p.add_argument("--num_envs", type=int, default=1)
p.add_argument("--shots", type=str, default="world,persp,top,side,racket")
p.add_argument("--width", type=int, default=1280)
p.add_argument("--height", type=int, default=720)
AppLauncher.add_app_launcher_args(p)
args = p.parse_args()
args.headless = True
args.enable_cameras = True
app_launcher = AppLauncher(args)
simulation_app = app_launcher.app
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import torch, omni.usd
from pxr import UsdGeom, Gf
from PIL import Image
from isaaclab.sensors import Camera, CameraCfg
import isaaclab.sim as sim_utils
from badminton_scene.scene_cfg import create_scene, SIM_DT, _box
from badminton_scene.scene_layout import (PERCEPTION_ROI, CANDIDATE_STRIKE_ROI, SPAWN_REGION, TARGET_ZONE, NET, CAMERA, PIPER)

EVID = Path(__file__).resolve().parents[1] / "evidence"
EVID.mkdir(parents=True, exist_ok=True)


def look_at_quat(eye, target, up=(0.0, 0.0, 1.0)):
    f = np.array(target, float) - np.array(eye, float)
    f = f / (np.linalg.norm(f) + 1e-9)
    u = np.array(up, float)
    if abs(float(np.dot(f, u))) > 0.999:
        u = np.array([0.0, 1.0, 0.0])
    r = np.cross(f, u); r = r / (np.linalg.norm(r) + 1e-9)
    u2 = np.cross(r, f)
    m = np.stack([r, u2, -f], axis=1)
    tr = m[0, 0] + m[1, 1] + m[2, 2]
    if tr > 0:
        s = math.sqrt(tr + 1.0) * 2.0
        w = 0.25 * s; x = (m[2, 1] - m[1, 2]) / s; y = (m[0, 2] - m[2, 0]) / s; z = (m[1, 0] - m[0, 1]) / s
    elif m[0, 0] > m[1, 1] and m[0, 0] > m[2, 2]:
        s = math.sqrt(1.0 + m[0, 0] - m[1, 1] - m[2, 2]) * 2.0
        w = (m[2, 1] - m[1, 2]) / s; x = 0.25 * s; y = (m[0, 1] + m[1, 0]) / s; z = (m[0, 2] + m[2, 0]) / s
    elif m[1, 1] > m[2, 2]:
        s = math.sqrt(1.0 + m[1, 1] - m[0, 0] - m[2, 2]) * 2.0
        w = (m[0, 2] - m[2, 0]) / s; x = (m[0, 1] + m[1, 0]) / s; y = 0.25 * s; z = (m[1, 2] + m[2, 1]) / s
    else:
        s = math.sqrt(1.0 + m[2, 2] - m[0, 0] - m[1, 1]) * 2.0
        w = (m[1, 0] - m[0, 1]) / s; x = (m[0, 2] + m[2, 0]) / s; y = (m[1, 2] + m[2, 1]) / s; z = 0.25 * s
    return [w, x, y, z]


def project(eye, quat_wxyz, target_world, W, H, focal, aperture):
    w, x, y, z = [float(v) for v in quat_wxyz]
    qi = (w, -x, -y, -z)
    def qmul(a, b):
        aw, ax, ay, az = a; bw, bx, by, bz = b
        return (aw*bw - ax*bx - ay*by - az*bz, aw*bx + ax*bw + ay*bz - az*by,
                aw*by - ax*bz + ay*bw + az*bx, aw*bz + ax*by - ay*bx + az*bw)
    d = [float(target_world[i]) - float(eye[i]) for i in range(3)]
    qd = qmul(qi, (0.0, d[0], d[1], d[2]))
    cx, cy, cz = qd[1], qd[2], qd[3]
    depth = -cz
    fx = W * float(focal) / float(aperture)
    u = W / 2.0 + fx * (cx / depth if depth != 0 else 0.0)
    v = H / 2.0 - fx * (cy / depth if depth != 0 else 0.0)
    return [round(u, 1), round(v, 1), round(depth, 3), bool(depth > 0 and 0 <= u < W and 0 <= v < H)]


def add_axes(prefix, pos, quat, length=0.35, thick=0.02):
    st = omni.usd.get_context().get_stage()
    xf = UsdGeom.Xform.Define(st, prefix)
    xf.AddTranslateOp().Set(Gf.Vec3d(*[float(v) for v in pos]))
    xf.AddOrientOp().Set(Gf.Quatf(float(quat[0]), Gf.Vec3f(float(quat[1]), float(quat[2]), float(quat[3]))))
    _box(prefix + "/X", (length, thick, thick), (length / 2.0, 0.0, 0.0), (1.0, 0.1, 0.1))
    _box(prefix + "/Y", (thick, length, thick), (0.0, length / 2.0, 0.0), (0.1, 1.0, 0.1))
    _box(prefix + "/Z", (thick, thick, length), (0.0, 0.0, length / 2.0), (0.2, 0.4, 1.0))


def main():
    shots = [s.strip() for s in args.shots.split(",") if s.strip()]
    W, H = args.width, args.height
    APERTURE = 20.955
    cam_cfg = CameraCfg(
        prim_path="/World/shot_cam",
        spawn=sim_utils.PinholeCameraCfg(focal_length=24.0, horizontal_aperture=APERTURE, clipping_range=(0.05, 1.0e5)),
        data_types=["rgb"], width=W, height=H,
    )
    sim, scene, robot, shuttle, _cs, extra = create_scene(num_envs=args.num_envs, extra_sensors=[("shot_cam", cam_cfg, Camera)])
    cam = extra["shot_cam"]
    for _ in range(5):
        scene.write_data_to_sim(); sim.step(); scene.update(SIM_DT)
    dev = cam.device
    eo = scene.env_origins[0].cpu().numpy()
    names = list(robot.body_names)
    rb = names.index("RacketBody"); l6 = names.index("link6")
    p6 = robot.data.body_pos_w[0, l6].cpu().numpy()
    q6 = robot.data.body_quat_w[0, l6].cpu().numpy().tolist()
    pr = robot.data.body_pos_w[0, rb].cpu().numpy()
    sp = torch.from_numpy(shuttle.root_physx_view.get_transforms().numpy()).float()[0, :3].numpy()
    def ctr(d):
        return [float((d["world_x"][0] + d["world_x"][1]) / 2.0 + eo[0]),
                float((d["world_y"][0] + d["world_y"][1]) / 2.0 + eo[1]),
                float((d["z"][0] + d["z"][1]) / 2.0)]
    targets = {
        "world_origin": [float(eo[0]), float(eo[1]), 0.0],
        "robot_base": [float(eo[0] - 1.6), float(eo[1]), 0.3],
        "piper_mount": [float(eo[0] + PIPER["mount_world"][0] + eo[0] * 0.0), float(eo[1] + PIPER["mount_world"][1]), float(PIPER["mount_world"][2])],
        "racket_body": [float(pr[0]), float(pr[1]), float(pr[2])],
        "racket_contact": [float(pr[0]), float(pr[1]), float(pr[2])],
        "net_center_top": [float(eo[0]), float(eo[1]), float(NET["center_top_z"])],
        "shuttle": [float(sp[0]), float(sp[1]), float(sp[2])],
        "spawn_region": ctr(SPAWN_REGION),
        "target_zone": ctr(TARGET_ZONE),
        "perception_roi": ctr(PERCEPTION_ROI),
        "candidate_strike_roi": ctr(CANDIDATE_STRIKE_ROI),
        "camera_rig": [float(eo[0] + CAMERA["rig_world"][0]), float(eo[1]), float(CAMERA["rig_world"][2])],
    }
    info = []

    def shoot(name, eye, target, focal=24.0):
        q = look_at_quat(eye, target)
        cam.set_world_poses(torch.tensor([eye], dtype=torch.float32, device=dev),
                            torch.tensor([q], dtype=torch.float32, device=dev))
        for _ in range(4):
            sim.render(); cam.update(SIM_DT)
        rgb = cam.data.output["rgb"]
        arr = rgb.detach().cpu().numpy() if hasattr(rgb, "detach") else np.asarray(rgb)
        img = arr.reshape(H, W, -1)[:, :, :3].astype("uint8")
        path = EVID / (name + ".png")
        Image.fromarray(img).save(path)
        proj = {k: project(eye, q, v, W, H, focal, APERTURE) for k, v in targets.items()}
        vis = [k for k, v in proj.items() if v[3]]
        print("[T09] SHOT", name, str(path), img.shape[1], "x", img.shape[0], "visible:", ",".join(vis))
        info.append(dict(name=name, path=str(path), width=int(img.shape[1]), height=int(img.shape[0]),
                         eye=[float(v) for v in eye], target=[float(v) for v in target], focal=focal,
                         projections=proj, visible_objects=vis))

    add_axes("/World/Axes/world", (float(eo[0]), float(eo[1]), 0.0), (1.0, 0.0, 0.0, 0.0), 0.9, 0.03)
    if "world" in shots:
        shoot("world_frame", (3.2, -3.2, 2.4), (0.0, 0.0, 0.6))
    if "persp" in shots:
        shoot("scene_perspective", (-5.2, -4.6, 3.2), (0.4, 0.0, 0.9))
    if "top" in shots:
        shoot("scene_top", (0.0, 0.0, 15.0), (0.0, 0.001, 0.0), focal=18.0)
    if "side" in shots:
        shoot("scene_side", (0.0, -9.5, 1.6), (-0.4, 0.0, 1.0), focal=28.0)
    if "racket" in shots:
        add_axes("/World/Axes/tcp", p6.tolist(), q6, 0.28, 0.012)
        add_axes("/World/Axes/contact", [float(p6[0]), float(p6[1]), float(p6[2] + 0.5)], q6, 0.22, 0.012)
        shoot("racket_frames", (float(pr[0] - 1.0), float(pr[1] - 0.9), float(pr[2] + 0.7)), tuple(float(v) for v in pr), focal=35.0)
    if "wide" in shots:
        eno = scene.env_origins.cpu().numpy()
        cx = float(eno[:, 0].mean()); cy = float(eno[:, 1].mean())
        tgt = dict(targets); tgt = {"world_origin": [cx, cy, 0.0], "shuttle": [float(sp[0] + cx), float(sp[1] + cy), float(sp[2])]}
        shoot("scene_128env", (cx, cy - 60.0, 130.0), (cx, cy, 0.0), focal=20.0)
    (EVID / "screenshots.json").write_text(json.dumps(dict(num_envs=args.num_envs, shots=info), indent=2))
    print("[T09] DONE", len(info))
    simulation_app.close()

if __name__ == "__main__":
    main()
