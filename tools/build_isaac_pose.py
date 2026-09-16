"""Build an appearance-aligned POSE sweep from the holdout Isaac frames.

The controlled matrix pose curve was numpy-rendered: side view scored 0.150 while flight_rotation scored
0.600. That may be a property of the object or a property of the renderer, and the size curve showed how
much renderer alignment matters. This rebuilds the pose curve in the appearance the release is aimed at,
using the HOLDOUT frames so nothing was seen in training.
"""
import argparse, csv, json, math, sys
from pathlib import Path
import numpy as np, cv2
REPO = Path("/home/T7/dgut/robot_sim")
sys.path.insert(0, str(REPO / "tools"))
from shuttle_render import imwrite_unicode

POOL = REPO / "outputs/shuttle_capability/isaac_pool_holdout"
CORE = REPO / "outputs/shuttle_capability/real_images/backgrounds"
OUT = REPO / "outputs/shuttle_capability/isaac_pose"
(OUT / "images").mkdir(parents=True, exist_ok=True)
(OUT / "labels").mkdir(parents=True, exist_ok=True)
IMGSZ = 1280
TARGET_PX = 48.0   # fixed apparent size, so pose is the only variable that matters
REPEATS = 20

recs = json.loads((POOL / "render_records.json").read_text(encoding="utf-8"))
print("frames with recorded pose:", len(recs))

def family(yaw, pitch):
    """Classify by the angle between the shuttle length axis and the camera view."""
    p = abs(math.degrees(pitch))
    y = abs(math.degrees(yaw)) % 180.0
    if p < 25.0:
        return "cork_end_on" if y < 90.0 else "feather_end_on"
    if p > 65.0:
        return "side"
    return "oblique"

groups = {}
for r in recs:
    groups.setdefault(family(r["yaw"], r["pitch"]), []).append(r)
for k in sorted(groups):
    print("  {:<16} {}".format(k, len(groups[k])))

bgs = sorted(CORE.glob("*.jpg")) + sorted(CORE.glob("*.png"))
rng = np.random.default_rng(20260920)

def load_obj(frame):
    img = cv2.imread(str(POOL / "rgb" / (frame + ".png")))
    lab = np.load(POOL / "mask" / (frame + ".npy"))
    m = (lab[..., -1] != 0).astype(np.uint8)
    ys, xs = np.nonzero(m)
    if xs.size < 50:
        return None
    x0, x1, y0, y1 = int(xs.min()), int(xs.max()) + 1, int(ys.min()), int(ys.max()) + 1
    return cv2.cvtColor(img[y0:y1, x0:x1], cv2.COLOR_BGR2RGB).astype(np.float32), m[y0:y1, x0:x1].astype(np.float32)

rows = []
for fam in sorted(groups):
    picks = groups[fam][:REPEATS * 3]
    used = 0
    for r in picks:
        if used >= REPEATS:
            break
        o = load_obj(r["name"])
        if o is None:
            continue
        obj, alpha = o
        oh, ow = alpha.shape
        scale = TARGET_PX / math.sqrt(oh * ow)
        nw, nh = max(2, int(round(ow * scale))), max(2, int(round(oh * scale)))
        obj_s = cv2.resize(obj, (nw, nh), interpolation=cv2.INTER_AREA)
        a_s = cv2.resize(alpha, (nw, nh), interpolation=cv2.INTER_AREA)
        bgp = bgs[(used * 5 + len(rows)) % len(bgs)]
        bg = cv2.cvtColor(cv2.imread(str(bgp)), cv2.COLOR_BGR2RGB)
        bh, bw = bg.shape[:2]
        if bh < IMGSZ or bw < IMGSZ:
            bg = cv2.resize(bg, (max(IMGSZ, bw), max(IMGSZ, bh)))
            bh, bw = bg.shape[:2]
        px = IMGSZ // 2 + int(rng.integers(-IMGSZ // 8, IMGSZ // 8))
        py = IMGSZ // 2 + int(rng.integers(-IMGSZ // 8, IMGSZ // 8))
        top = max(0, min(IMGSZ - nh, py - nh // 2))
        left = max(0, min(IMGSZ - nw, px - nw // 2))
        canvas = bg[0:IMGSZ, 0:IMGSZ].astype(np.float32).copy()
        a3 = a_s[..., None]
        canvas[top:top + nh, left:left + nw] = (canvas[top:top + nh, left:left + nw] * (1 - a3) + obj_s * a3)
        canvas = np.clip(canvas, 0, 255).astype(np.uint8)
        name = "ip_{}_{:02d}".format(fam, used)
        imwrite_unicode(OUT / "images" / (name + ".jpg"), cv2.cvtColor(canvas, cv2.COLOR_RGB2BGR), quality=92)
        nx1, ny1 = left / IMGSZ, top / IMGSZ
        nx2, ny2 = (left + nw) / IMGSZ, (top + nh) / IMGSZ
        (OUT / "labels" / (name + ".txt")).write_text(
            "0 {:.6f} {:.6f} {:.6f} {:.6f}\n".format((nx1 + nx2) / 2, (ny1 + ny2) / 2, nx2 - nx1, ny2 - ny1),
            encoding="utf-8")
        # The centre MUST be recorded: the evaluator reconstructs the ground-truth box from the recorded
        # centre and extent, and without these two columns it reports "no ground truth in this group".
        rows.append({"file": "images/" + name + ".jpg", "split": "fixed_core_test", "source_type": "SYNTHETIC_3D",
                     "bbox_w_px": nw, "bbox_h_px": nh,
                     "pos_x_px": int(round(left + nw / 2.0)), "pos_y_px": int(round(top + nh / 2.0)),
                     "equiv_size_px": round(math.sqrt(nw * nh), 3),
                     "pose_bucket": fam, "target_px": TARGET_PX, "background": bgp.name, "imgsz": IMGSZ,
                     "is_negative": "False", "camera_id": "isaac_pose", "source_frame": r["name"],
                     "yaw_deg": round(math.degrees(r["yaw"]), 2), "pitch_deg": round(math.degrees(r["pitch"]), 2)})
        used += 1
    print("  {:<16} -> {} rows".format(fam, used))
cols = list(rows[0].keys())
with (OUT / "manifest.csv").open("w", newline="", encoding="utf-8") as fh:
    w = csv.DictWriter(fh, fieldnames=cols)
    w.writeheader(); w.writerows(rows)
print("wrote", len(rows), "rows to", OUT / "manifest.csv")