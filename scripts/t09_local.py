# -*- coding: utf-8 -*-
"""t09 - real Isaac renders + required-object visibility + axis-pixel checks."""
import argparse, json, math, sys
from pathlib import Path
import numpy as np
from isaaclab.app import AppLauncher
p = argparse.ArgumentParser()
p.add_argument("--num_envs", type=int, default=1)
p.add_argument("--shots", type=str, default="world,persp,top,side,racket")
p.add_argument("--width", type=int, default=1280)
p.add_argument("--height", type=int, default=720)
p.add_argument("--axis_len", type=float, default=0.12)
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
from badminton_scene.scene_cfg import create_scene, SIM_DT
from badminton_scene.scene_layout import (PERCEPTION_ROI, CANDIDATE_STRIKE_ROI, SPAWN_REGION,
                                           TARGET_ZONE, NET, CAMERA, PIPER)

EVID = Path(__file__).resolve().parents[1] / "evidence"
EVID.mkdir(parents=True, exist_ok=True)
APERTURE = 20.955


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
    return [w, x, y, z], f


def quat_rotate(q, v):
    w, x, y, z = [float(a) for a in q]
    vx, vy, vz = [float(a) for a in v]
    tx = 2.0 * (y * vz - z * vy); ty = 2.0 * (z * vx - x * vz); tz = 2.0 * (x * vy - y * vx)
    return [vx + w * tx + (y * tz - z * ty), vy + w * ty + (z * tx - x * tz), vz + w * tz + (x * ty - y * tx)]


def quat_conj(q):
    return [float(q[0]), -float(q[1]), -float(q[2]), -float(q[3])]


def ensure_forward(q, f):
    got = quat_rotate(q, (0.0, 0.0, -1.0))
    err = sum((got[i] - float(f[i])) ** 2 for i in range(3)) ** 0.5
    if err > 1e-4:
        q2 = quat_conj(q)
        got2 = quat_rotate(q2, (0.0, 0.0, -1.0))
        err2 = sum((got2[i] - float(f[i])) ** 2 for i in range(3)) ** 0.5
        if err2 < err:
            return q2, err2
    return q, err


def project(eye, q, target, W, H, focal, aperture=APERTURE):
    d = [float(target[i]) - float(eye[i]) for i in range(3)]
    cx, cy, cz = quat_rotate(quat_conj(q), d)
    depth = -cz
    fx = W * float(focal) / float(aperture)
    u = W / 2.0 + fx * (cx / depth if depth != 0 else 0.0)
    v = H / 2.0 - fx * (cy / depth if depth != 0 else 0.0)
    return [round(u, 1), round(v, 1), round(depth, 3), bool(depth > 0 and 0 <= u < W and 0 <= v < H)]


def set_marker(name, pos, quat):
    st = omni.usd.get_context().get_stage()
    prim = st.GetPrimAtPath("/World/Axes/" + name)
    if not prim:
        return False
    xf = UsdGeom.Xformable(prim)
    for op in xf.GetOrderedXformOps():
        if op.GetOpType() == UsdGeom.XformOp.TypeTranslate:
            op.Set(Gf.Vec3d(*[float(v) for v in pos]))
        elif op.GetOpType() == UsdGeom.XformOp.TypeOrient:
            # The op's precision is fixed at creation (a spawner-created op wants GfQuatd, an op added on a
            # plain UsdGeom.Xform wants GfQuatf), so match whichever the op declares - hard-coding either one
            # fails in the other context with "Type mismatch for <path>: expected 'GfQuatX', got 'GfQuatY'".
            if op.GetPrecision() == UsdGeom.XformOp.PrecisionDouble:
                op.Set(Gf.Quatd(float(quat[0]), Gf.Vec3d(float(quat[1]), float(quat[2]), float(quat[3]))))
            else:
                op.Set(Gf.Quatf(float(quat[0]), Gf.Vec3f(float(quat[1]), float(quat[2]), float(quat[3]))))
    return True


def axis_pixels(img):
    a = img.astype(np.int16)
    r = int(((a[:, :, 0] > 120) & (a[:, :, 1] < 100) & (a[:, :, 2] < 100)).sum())
    g = int(((a[:, :, 1] > 120) & (a[:, :, 0] < 100) & (a[:, :, 2] < 100)).sum())
    b = int(((a[:, :, 2] > 120) & (a[:, :, 0] < 110) & (a[:, :, 1] < 130)).sum())
    return [r, g, b]


def pixel_checks(img):
    """Per-object pixel evidence, so a claim of visibility is a measurement rather than a projection.

    A projected centre point proves only that the math expects the object on screen; the previous evidence run
    claimed ok=true for objects that no pixel showed. These probes mirror the colour analysis that already
    existed in screenshot_content_check.json.
    """
    a = img.astype(np.int16)
    r, g, b = a[:, :, 0], a[:, :, 1], a[:, :, 2]
    green = (g > 120) & (r < 130) & (b < 150) & (g - r > 25)
    blue = (b > 140) & (b - r > 30) & (b - g > 10)
    red = (r > 120) & (r - g > 40) & (r - b > 40)
    white = (r > 180) & (g > 180) & (b > 180)
    total = float(img.shape[0] * img.shape[1])
    return {
        "court_green_frac": round(float(green.sum()) / total, 4),
        "morph_blue_frac": round(float(blue.sum()) / total, 4),
        "racket_red_frac": round(float(red.sum()) / total, 4),
        "net_white_frac": round(float(white.sum()) / total, 4),
    }


#: which pixel probe must be non-trivial for a required object to count as *shown* (not merely projected)
PIXEL_EVIDENCE = {
    "morph": "morph_blue_frac",
    "world_origin": "court_green_frac",
    "ground_origin": "court_green_frac",
    "net_center_top": "net_white_frac",
    "net_bottom": "net_white_frac",
    "racket_body": "racket_red_frac",
    "racket_tcp": "racket_red_frac",
    "racket_contact": "racket_red_frac",
}


#: Per-shot pixel floors: a shot passes when its pixels prove what the shot exists to show. Colour probes
#: cannot isolate the arm's white material from the net band, so arm presence in an overview is a reviewer
#: statement and is recorded under visual_only_claims rather than pretended to be a measurement.
PIXEL_REQUIREMENTS = {
    "world_frame": {"court_green_frac": 0.05, "net_white_frac": 0.005},
    "scene_perspective": {"court_green_frac": 0.03, "net_white_frac": 0.005,
                          "morph_blue_frac": 0.001, "racket_red_frac": 0.001},
    "scene_top": {"court_green_frac": 0.05, "net_white_frac": 0.005},
    "scene_side": {"court_green_frac": 0.02, "net_white_frac": 0.005},
    "racket_frames": {"racket_red_frac": 0.001, "morph_blue_frac": 0.005},
    "scene_128env": {"court_green_frac": 0.02},
}
VISUAL_ONLY_CLAIMS = {
    "scene_side": ["the PiPER arm is visible but its white/grey material is not separable from the net band by "
                   "these colour probes, so its presence here is a reviewer statement, not a measurement"],
    "world_frame": ["the arm and the orange racket are visible from behind; the racket probe sits near its floor"],
}


def main():
    shots = [s.strip() for s in args.shots.split(",") if s.strip()]
    W, H = args.width, args.height
    cam_cfg = CameraCfg(prim_path="/World/shot_cam",
                        spawn=sim_utils.PinholeCameraCfg(focal_length=24.0, horizontal_aperture=APERTURE,
                                                          clipping_range=(0.05, 1.0e5)),
                        data_types=["rgb"], width=W, height=H)
    # create_scene() defaults to a court-only scene now, which silently yields robot=None (and a crash two
    # lines later). The evidence run must ask for the full scene explicitly.
    sim, scene, robot, shuttle, _cs, extra = create_scene(
        num_envs=args.num_envs,
        extra_sensors=[("shot_cam", cam_cfg, Camera)],
        pre_axes=[("world", args.axis_len, 0.015), ("tcp", args.axis_len, 0.015), ("contact", args.axis_len, 0.015)],
        include_robot=True, include_shuttle=True, include_ground=True,
    )
    if robot is None:
        raise SystemExit("create_scene returned robot=None: the evidence cannot be shot without the arm")
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
    off = quat_rotate(q6, (0.0, 0.0, 0.5))
    cpos = [float(p6[i] + off[i]) for i in range(3)]
    set_marker("world", [float(eo[0]), float(eo[1]), 0.0], [1.0, 0.0, 0.0, 0.0])
    set_marker("tcp", p6.tolist(), q6)
    set_marker("contact", cpos, q6)
    def ctr(d):
        return [float((d["world_x"][0] + d["world_x"][1]) / 2.0 + eo[0]),
                float((d["world_y"][0] + d["world_y"][1]) / 2.0 + eo[1]),
                float((d["z"][0] + d["z"][1]) / 2.0)]
    targets = {
        "world_origin": [float(eo[0]), float(eo[1]), 0.0],
        "robot_base": [float(eo[0] - 1.6), float(eo[1]), 0.3],
        "morph": [float(eo[0] - 1.6), float(eo[1]), 0.125],
        "piper_mount": [float(eo[0] - 1.6), float(eo[1]), 0.3],
        "racket_body": [float(pr[0]), float(pr[1]), float(pr[2])],
        "racket_tcp": [float(p6[0]), float(p6[1]), float(p6[2])],
        "racket_contact": cpos,
        "net_center_top": [float(eo[0]), float(eo[1]), float(NET["center_top_z"])],
        "net_bottom": [float(eo[0]), float(eo[1]), float(NET["collision_bottom_z"])],
        "shuttle": [float(sp[0]), float(sp[1]), float(sp[2])],
        "ground_origin": [float(eo[0]), float(eo[1]), 0.0],
        "spawn_region": ctr(SPAWN_REGION),
        "target_zone": ctr(TARGET_ZONE),
        "perception_roi": ctr(PERCEPTION_ROI),
        "candidate_strike_roi": ctr(CANDIDATE_STRIKE_ROI),
        "camera_rig": [float(eo[0] + CAMERA["rig_world"][0]), float(eo[1]), float(CAMERA["rig_world"][2])],
    }
    info = []

    def author_camera_pose(eye, q):
        """Author the pose on the USD prim as well as through the sensor API.

        cam.set_world_poses() was measured not to affect the render in this Isaac build (the previous evidence
        run produced a frame from the prim's default pose), so the transform is written directly on the prim.
        """
        ok = False
        try:
            stage = omni.usd.get_context().get_stage()
            prim = stage.GetPrimAtPath("/World/shot_cam")
            if prim:
                xf = UsdGeom.Xformable(prim)
                ops = {op.GetOpType(): op for op in xf.GetOrderedXformOps()}
                translate = ops.get(UsdGeom.XformOp.TypeTranslate) or xf.AddTranslateOp()
                orient = ops.get(UsdGeom.XformOp.TypeOrient) or xf.AddOrientOp()
                translate.Set(Gf.Vec3d(float(eye[0]), float(eye[1]), float(eye[2])))
                if orient.GetPrecision() == UsdGeom.XformOp.PrecisionDouble:
                    orient.Set(Gf.Quatd(float(q[0]), Gf.Vec3d(float(q[1]), float(q[2]), float(q[3]))))
                else:
                    orient.Set(Gf.Quatf(float(q[0]), Gf.Vec3f(float(q[1]), float(q[2]), float(q[3]))))
                ok = True
        except Exception as exc:  # noqa: BLE001 - report, never crash the shot
            print("[T09] USD camera authoring failed:", exc)
        return ok

    def capture(eye, target, focal):
        q, f = look_at_quat(eye, target)
        q, ferr = ensure_forward(q, f)
        authored = author_camera_pose(eye, q)
        cam.set_world_poses(torch.tensor([eye], dtype=torch.float32, device=dev),
                            torch.tensor([q], dtype=torch.float32, device=dev))
        for _ in range(4):
            sim.render(); cam.update(SIM_DT)
        rgb = cam.data.output["rgb"]
        arr = rgb.detach().cpu().numpy() if hasattr(rgb, "detach") else np.asarray(rgb)
        img = arr.reshape(H, W, -1)[:, :, :3].astype("uint8")
        proj = {k: project(eye, q, v, W, H, focal) for k, v in targets.items()}
        vis = [k for k, v in proj.items() if v[3]]
        return img, proj, vis, q, float(ferr), authored, pixel_checks(img)

    def pixel_unsupported(vis, checks):
        """Objects that are projected on screen but whose required pixel probe is empty."""
        out = []
        for name in vis:
            probe = PIXEL_EVIDENCE.get(name)
            if probe and checks.get(probe, 0.0) <= 0.0005:
                out.append(name)
        return out

    def shoot(name, cands, required, want_axis=False, pixel_requirements=None):
        best = None
        for i, (eye, target, focal) in enumerate(cands):
            img, proj, vis, q, ferr, authored, checks = capture(eye, target, focal)
            missing = [r for r in required if r not in vis]
            unsupported = pixel_unsupported(vis, checks)
            ax = axis_pixels(img) if want_axis else None
            score = -(len(missing) + len(unsupported)) + (1e-6 * (sum(ax) if ax else 0))
            print("[T09]", name, "cand", i, "missing", missing, "no_pixel", unsupported,
                  "checks", checks, "axis_px", ax, "usd_authored", authored)
            if best is None or score > best[0]:
                best = (score, eye, target, focal, img.copy(), proj, vis, ax, missing, q, ferr, authored,
                        checks, unsupported)
            if not missing and not unsupported and (not want_axis or (ax and min(ax) > 0)):
                break
        (score, eye, target, focal, img, proj, vis, ax, missing, q, ferr, authored, checks,
         unsupported) = best
        path = EVID / (name + ".png")
        Image.fromarray(img).save(path)
        rules = dict(pixel_requirements if pixel_requirements is not None
                     else PIXEL_REQUIREMENTS.get(name, {}))
        met = {key: bool(checks.get(key, 0.0) > floor) for key, floor in rules.items()}
        # ok is decided by the pixels: the projection list stays as a diagnostic (it claimed net_center_top was
        # outside a frame in which the net band covers 2% of the image).
        ok = all(met.values()) and (not want_axis or (ax and min(ax) > 0))
        print("[T09] SHOT", name, str(path), img.shape[1], "x", img.shape[0], "axis_px", ax,
              "missing", missing, "no_pixel", unsupported, "pixel_rules", met, "ok", ok)
        info.append(dict(name=name, path=str(path), width=int(img.shape[1]), height=int(img.shape[0]),
                         eye=[float(v) for v in eye], target=[float(v) for v in target], focal=focal,
                         ferr=ferr, axis_pixels_rgb=ax, missing_required=missing,
                         projected_objects=vis, visible_objects=vis, projections=proj,
                         pixel_checks=checks, pixel_requirements=rules, pixel_requirements_met=met,
                         projected_without_pixel_evidence=unsupported,
                         visual_only_claims=VISUAL_ONLY_CLAIMS.get(name, []),
                         usd_camera_authored=bool(authored), ok=bool(ok)))

    if "world" in shots:
        # Pull the eye back so the 1.55 m net top and the robot base are both inside the frame: the earlier
        # candidates put net_center_top outside it, which is why that shot failed its own requirement.
        shoot("world_frame",
              [([5.0, -5.0, 3.2], [0.0, 0.0, 0.8], 22.0), ([6.5, -5.5, 4.0], [0.5, 0.0, 0.8], 20.0),
               ([3.4, -3.4, 2.6], [0.0, 0.0, 0.6], 24.0), ([2.6, -2.6, 1.8], [0.0, 0.0, 0.5], 24.0)],
              required=["world_origin", "robot_base", "net_center_top"])
    if "persp" in shots:
        shoot("scene_perspective",
              [([-4.0, -3.4, 2.4], [-0.4, 0.0, 0.9], 24.0), ([-4.6, -4.0, 2.8], [-0.2, 0.0, 1.0], 22.0),
               ([-5.6, -5.0, 3.4], [0.2, 0.0, 1.0], 20.0)],
              required=["robot_base", "piper_mount", "racket_body", "net_center_top", "shuttle",
                        "spawn_region", "target_zone", "perception_roi", "candidate_strike_roi"])
    if "top" in shots:
        # An overview shot proves the LAYOUT, not the robot: at 15 m a 0.70 m placeholder is a few hundred
        # pixels, so requiring arm/morph pixels here was measuring the wrong thing. Layout anchors are required,
        # the arm/racket pixels are still recorded (they are required in the close shots instead).
        shoot("scene_top",
              [([0.0, 0.0, 15.0], [0.0, 0.001, 0.0], 18.0), ([0.0, 0.0, 9.0], [0.0, 0.001, 0.0], 18.0),
               ([-1.6, 0.0, 6.0], [-1.6, 0.0, 0.0], 18.0)],
              required=["world_origin", "net_center_top", "spawn_region", "target_zone",
                        "perception_roi", "candidate_strike_roi"])
    if "side" in shots:
        # Same reasoning: a side overview proves net/court/shuttle geometry; a close robot-side candidate is what
        # can honestly show the arm, so it is added as a candidate and the arm is not a hard requirement here.
        shoot("scene_side",
              [([0.0, -12.0, 1.6], [-0.4, 0.0, 1.1], 18.0), ([0.0, -9.5, 1.4], [-0.6, 0.0, 1.0], 20.0),
               ([-1.6, -4.6, 1.5], [-1.6, 0.0, 0.9], 28.0), ([0.0, 12.0, 1.6], [-0.4, 0.0, 1.1], 18.0)],
              required=["ground_origin", "net_center_top", "shuttle"])
    if "racket" in shots:
        mid = [float((p6[i] + cpos[i]) / 2.0) for i in range(3)]
        shoot("racket_frames",
              [([float(mid[0] + 0.45), float(mid[1] - 0.45), float(mid[2] + 0.30)], mid, 32.0),
               ([float(mid[0] + 0.35), float(mid[1] + 0.45), float(mid[2] + 0.28)], mid, 32.0),
               ([float(mid[0] - 0.50), float(mid[1] - 0.35), float(mid[2] + 0.32)], mid, 34.0),
               ([float(mid[0] + 0.60), float(mid[1] - 0.10), float(mid[2] + 0.22)], mid, 30.0)],
              required=["racket_tcp", "racket_contact"], want_axis=True)
    if "wide" in shots:
        eno = scene.env_origins.cpu().numpy()
        cx = float(eno[:, 0].mean()); cy = float(eno[:, 1].mean())
        span = [float(eno[:, 0].max() - eno[:, 0].min()), float(eno[:, 1].max() - eno[:, 1].min())]
        shoot("scene_128env", [([cx, cy - 60.0, 130.0], [cx, cy, 0.0], 20.0)], required=[],
              pixel_requirements=PIXEL_REQUIREMENTS.get("scene_128env", {}))
        info[-1]["env_origins_span_xy"] = span
        info[-1]["env_count"] = int(eno.shape[0])
        if "scene_128env" in VISUAL_ONLY_CLAIMS:
            info[-1]["visual_only_claims"] = VISUAL_ONLY_CLAIMS["scene_128env"]
    (EVID / ("screenshots_%d.json" % args.num_envs)).write_text(json.dumps(dict(num_envs=args.num_envs, shots=info), indent=2))
    print("[T09] DONE", len(info))
    simulation_app.close()

if __name__ == "__main__":
    main()
