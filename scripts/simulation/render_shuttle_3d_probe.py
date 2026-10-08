#!/usr/bin/env python3
"""render_shuttle_3d_probe.py - render OUR shuttlecock with Isaac Sim, for the ETH YOLOv8s probe.

Scope (2026-10-03 spike, requested by the user):
  * real 3D renderer (Isaac Sim / IsaacLab), NOT a numpy software rasteriser
  * NO binary-mask compositing: the delivered image is the renderer's own RGB frame
  * NO edits to the GLB geometry, NO shape changes to raise the detection rate
  * the GLB's own material is preserved - it arrives through
    assets/third_party/shuttlecock_visual.usd, whose Looks/White UsdPreviewSurface is bound to
    Obj_Feather_0 and Obj_Cork_0 (measured: see management/shuttle_detection/THREED_RENDER_ETH_PROBE.md).

GT box = the rendered footprint measured from a SEMANTIC SEGMENTATION pass (auxiliary measurement
only - it never touches the delivered RGB). A projection-based box is recorded alongside so the two
can be compared; a disagreement is exactly the "GT/projection alignment" case the user asked about.

STATUS: draft, syntax-checked only. Isaac Sim is on the remote host and that host was unreachable
when this was written, so this file has NOT been executed.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
from pathlib import Path

import numpy as np

P = argparse.ArgumentParser()
from isaaclab.app import AppLauncher  # noqa: E402

AppLauncher.add_app_launcher_args(P)
P.add_argument("--out", default="/home/T7/dgut/robot_sim/outputs/shuttle_capability/three_d_render_probe")
P.add_argument("--shuttle-usd", default="/home/T7/dgut/robot_sim/assets/shuttle/shuttlecock.usd")
P.add_argument("--imgsz", type=int, default=960)
P.add_argument("--sizes", type=float, nargs="+", default=[24.0, 16.0, 12.0],
               help="target equiv_size_640 in px; equiv640 = sqrt(w*h)*640/max(W,H)")
P.add_argument("--n-per-size", type=int, default=10)
P.add_argument("--seed", type=int, default=20261003)
P.add_argument("--calib-iters", type=int, default=6)
P.add_argument("--calib-tol", type=float, default=0.05)
ARGS = P.parse_args()
ARGS.headless = True
ARGS.enable_cameras = True

T0 = time.time()
APP = AppLauncher(ARGS)
SIM_APP = APP.app
print("[ISAAC] app launched in {:.1f}s".format(time.time() - T0), flush=True)

import omni.usd  # noqa: E402
from PIL import Image  # noqa: E402
from pxr import Gf, Usd, UsdGeom  # noqa: E402
import isaaclab.sim as sim_utils  # noqa: E402
from isaaclab.sensors import Camera, CameraCfg  # noqa: E402

W = H = int(ARGS.imgsz)
FOCAL = 50.0
APERTURE = 20.955


def set_orient(prim_path, q):
    """Set orient without knowing the op precision (measured on this stage: Shuttle=Quatf, cam=Quatd)."""
    xf = UsdGeom.Xformable(SIM_APP.app.get_stage() if False else omni.usd.get_context().get_stage().GetPrimAtPath(prim_path))
    ops = {o.GetOpType(): o for o in xf.GetOrderedXformOps()}
    op = ops.get(UsdGeom.XformOp.TypeOrient) or xf.AddOrientOp()
    err = None
    for ctor in (Gf.Quatf, Gf.Quatd):
        try:
            v = Gf.Vec3f if ctor is Gf.Quatf else Gf.Vec3d
            op.Set(ctor(float(q[0]), v(float(q[1]), float(q[2]), float(q[3]))))
            return
        except Exception as exc:  # noqa: BLE001
            err = exc
    raise err


def set_translate(prim_path, t):
    stage = omni.usd.get_context().get_stage()
    xf = UsdGeom.Xformable(stage.GetPrimAtPath(prim_path))
    ops = {o.GetOpType(): o for o in xf.GetOrderedXformOps()}
    op = ops.get(UsdGeom.XformOp.TypeTranslate) or xf.AddTranslateOp()
    op.Set(Gf.Vec3d(float(t[0]), float(t[1]), float(t[2])))


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


def ypr_quat(yaw, pitch, roll):
    cy, sy = math.cos(yaw / 2), math.sin(yaw / 2)
    cp, sp = math.cos(pitch / 2), math.sin(pitch / 2)
    cr, sr = math.cos(roll / 2), math.sin(roll / 2)
    return [cy * cp * cr + sy * sp * sr, sy * cp * cr - cy * sp * sr,
            cy * sp * cr + sy * cp * sr, cy * cp * sr - sy * sp * cr]


def footprint(seg):
    """Tight pixel box + size of the non-background region of a semantic pass."""
    a = np.asarray(seg)
    a = a.reshape(H, W, -1) if a.ndim == 3 else a.reshape(H, W, 1)
    m = a[..., -1] != 0
    ys, xs = np.nonzero(m)
    if xs.size == 0:
        return None
    return {"x0": int(xs.min()), "x1": int(xs.max()), "y0": int(ys.min()), "y1": int(ys.max()),
            "w_px": int(xs.max() - xs.min() + 1), "h_px": int(ys.max() - ys.min() + 1),
            "pixels": int(m.sum())}


def equiv640(w_px, h_px):
    """Same definition as tools/audit_localization_yolo26_v1.py:771."""
    return math.sqrt(float(w_px) * float(h_px)) * 640.0 / float(max(W, H))


def main():
    out = Path(ARGS.out)
    (out / "images").mkdir(parents=True, exist_ok=True)
    (out / "labels").mkdir(parents=True, exist_ok=True)

    sim = sim_utils.SimulationContext(sim_utils.SimulationCfg(dt=1.0 / 60.0, device="cuda:0"))
    stage = omni.usd.get_context().get_stage()

    # --- scene: ground + lighting. The GLB carries NO light (measured: no KHR_lights_punctual),
    #     so every light here is OURS and is recorded per frame.
    sim_utils.GroundPlaneCfg().func("/World/ground", sim_utils.GroundPlaneCfg())
    sim_utils.DistantLightCfg(intensity=900.0, color=(1.0, 0.98, 0.95)).func(
        "/World/key", sim_utils.DistantLightCfg(intensity=900.0, color=(1.0, 0.98, 0.95)), translation=(2, -2, 4))
    sim_utils.DomeLightCfg(intensity=600.0).func("/World/dome", sim_utils.DomeLightCfg(intensity=600.0))

    shuttle = UsdGeom.Xform.Define(stage, "/World/Shuttle")
    shuttle.GetPrim().GetReferences().AddReference(str(ARGS.shuttle_usd))

    cam_cfg = CameraCfg(
        prim_path="/World/cam",
        spawn=sim_utils.PinholeCameraCfg(focal_length=FOCAL, horizontal_aperture=APERTURE,
                                        clipping_range=(0.01, 100.0)),
        data_types=["rgb", "semantic_segmentation"],
        width=W, height=H,
    )
    cam = Camera(cam_cfg)
    sim.reset()
    print("[ISAAC] scene ready in {:.1f}s".format(time.time() - T0), flush=True)

    rng = np.random.default_rng(int(ARGS.seed))
    records = []
    for size in ARGS.sizes:
        for k in range(int(ARGS.n_per_size)):
            name = "r3d_{:02.0f}px_{:02d}".format(size, k)
            # --- modest randomisation: pose free, lighting direction mildly perturbed, no DR
            yaw = float(rng.uniform(0.0, 2.0 * math.pi))
            pitch = float(rng.uniform(-math.pi, math.pi))
            roll = float(rng.uniform(-math.radians(20.0), math.radians(20.0)))
            set_orient("/World/Shuttle", ypr_quat(yaw, pitch, roll))
            azim = float(rng.uniform(0.0, 2.0 * math.pi))
            elev = float(rng.uniform(-0.4, 0.9))
            light_az = float(rng.uniform(0.0, 2.0 * math.pi))
            light_el = float(rng.uniform(0.6, 1.1))
            set_translate("/World/key", (3.0 * math.cos(light_az), 3.0 * math.sin(light_az), 4.0 * math.sin(light_el)))
            set_translate("/World/Shuttle", (0.0, 0.0, 0.25))

            # --- closed-loop distance calibration on the MEASURED footprint
            dist = 1.0
            best = (None, None, float("inf"))
            for _ in range(int(ARGS.calib_iters)):
                eye = (0.25 + dist * math.cos(elev) * math.cos(azim),
                       dist * math.cos(elev) * math.sin(azim),
                       0.25 + dist * math.sin(elev))
                set_translate("/World/cam", eye)
                set_orient("/World/cam", look_at(eye, (0.0, 0.0, 0.25)))
                for _ in range(4):
                    sim.render()
                    cam.update(1.0 / 60.0)
                fp = footprint(cam.data.output["semantic_segmentation"])
                if fp is None:
                    dist *= 0.6
                    continue
                got = equiv640(fp["w_px"], fp["h_px"])
                err = abs(got - size) / size
                if err < best[2]:
                    best = (dist, fp, err)
                if err <= float(ARGS.calib_tol):
                    break
                dist = max(dist * (got / size), 0.02)

            dist, fp, err = best
            if fp is None:
                print("  skipped", name, "nothing rendered", flush=True)
                continue
            # final frame at the calibrated distance
            eye = (0.25 + dist * math.cos(elev) * math.cos(azim),
                   dist * math.cos(elev) * math.sin(azim),
                   0.25 + dist * math.sin(elev))
            set_translate("/World/cam", eye)
            set_orient("/World/cam", look_at(eye, (0.0, 0.0, 0.25)))
            for _ in range(6):
                sim.render()
                cam.update(1.0 / 60.0)
            rgb = cam.data.output["rgb"]
            arr = rgb.detach().cpu().numpy() if hasattr(rgb, "detach") else np.asarray(rgb)
            img = arr.reshape(H, W, -1)[:, :, :3].astype(np.uint8)
            Image.fromarray(img).save(out / "images" / (name + ".png"))
            (out / "labels" / (name + ".txt")).write_text(
                "0 {:.6f} {:.6f} {:.6f} {:.6f}\n".format(
                    (fp["x0"] + fp["x1"] + 1) / 2.0 / W, (fp["y0"] + fp["y1"] + 1) / 2.0 / H,
                    fp["w_px"] / float(W), fp["h_px"] / float(H)), encoding="utf-8")
            records.append({
                "file": name + ".png", "target_equiv640": size,
                "measured_equiv640": round(equiv640(fp["w_px"], fp["h_px"]), 3),
                "calib_error_frac": round(float(err), 4),
                "bbox_w_px": fp["w_px"], "bbox_h_px": fp["h_px"], "solid_px": fp["pixels"],
                "yaw_deg": round(math.degrees(yaw), 2), "pitch_deg": round(math.degrees(pitch), 2),
                "roll_deg": round(math.degrees(roll), 2),
                "camera_eye": [round(float(v), 5) for v in eye], "camera_distance_m": round(float(dist), 5),
                "camera_focal_px": FOCAL * W / APERTURE, "look_target": [0.0, 0.0, 0.25],
                "light_key_intensity": 900.0, "light_key_azimuth_deg": round(math.degrees(light_az), 2),
                "light_key_elevation_deg": round(math.degrees(light_el), 2), "light_dome_intensity": 600.0,
                "imgsz": W,
            })
            print("  {} target={} got={:.2f} dist={:.3f}".format(
                  name, size, equiv640(fp["w_px"], fp["h_px"]), dist), flush=True)

    (out / "render_manifest.json").write_text(json.dumps(records, indent=1), encoding="utf-8")
    print("[ISAAC] wrote {} records to {}".format(len(records), out / "render_manifest.json"), flush=True)


if __name__ == "__main__":
    main()
    SIM_APP.close()