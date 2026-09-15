"""Build an appearance-aligned size ladder: Isaac-rendered shuttles at controlled apparent sizes.

Why this exists: the frozen controlled matrix and the challenge set are both rendered by the numpy
rasteriser, so the release that was trained toward photographic appearance scores WORSE on them than the
numpy-trained model did. That makes their numbers hard to interpret and impossible to quote as
deployment capability. This set renders the same asset with Isaac Sim, composites it at a ladder of
controlled sizes onto the FROZEN test backgrounds, and measures the capability curve in the appearance
the model is actually aimed at.

It is an ADDITIONAL set. The frozen controlled matrix is left untouched so cross-version comparisons
remain valid.
"""
import argparse, csv, math, sys
from pathlib import Path
import numpy as np, cv2
REPO = Path("/home/T7/dgut/robot_sim")
sys.path.insert(0, str(REPO / "tools"))
from shuttle_render import imwrite_unicode

ap = argparse.ArgumentParser()
ap.add_argument("--sizes", default="4,6,8,12,16,24,32,64,128,256,512,1024")
ap.add_argument("--repeats", type=int, default=4)
ap.add_argument("--imgsz", type=int, default=1280)
ap.add_argument("--seed", type=int, default=20260918)
ap.add_argument("--out", default="/home/T7/dgut/robot_sim/outputs/shuttle_capability/isaac_ladder")
# The source pool matters for validity: a ladder built from the frames that produced the TRAINING pool
# measures renders the model has already seen. The holdout pool is rendered with a different seed and
# never touched by training, so a ladder built from it is a generalisation measurement.
ap.add_argument("--pool", default="/home/T7/dgut/robot_sim/outputs/shuttle_capability/isaac_pool")
args = ap.parse_args()

POOL = Path(args.pool)
CORE = REPO / "outputs/shuttle_capability/real_images/backgrounds"
OUT = Path(args.out)
(OUT / "images").mkdir(parents=True, exist_ok=True)
(OUT / "labels").mkdir(parents=True, exist_ok=True)

frames = sorted((POOL / "rgb").glob("*.png"))
bgs = sorted(CORE.glob("*.jpg")) + sorted(CORE.glob("*.png"))
sizes = [float(x) for x in args.sizes.split(",")]
print("frames {} | frozen backgrounds {} | sizes {}".format(len(frames), len(bgs), sizes))
rng = np.random.default_rng(args.seed)

def load_obj(frame):
    img = cv2.imread(str(frame))
    lab = np.load(POOL / "mask" / (frame.stem + ".npy"))
    mask = (lab[..., -1] != 0).astype(np.uint8)
    ys, xs = np.nonzero(mask)
    if xs.size < 50:
        return None
    x0, x1, y0, y1 = int(xs.min()), int(xs.max()) + 1, int(ys.min()), int(ys.max()) + 1
    return cv2.cvtColor(img[y0:y1, x0:x1], cv2.COLOR_BGR2RGB).astype(np.float32), mask[y0:y1, x0:x1].astype(np.float32)

usable = []
for f in frames:
    o = load_obj(f)
    if o is not None:
        usable.append((f, o[0], o[1]))
print("usable source frames:", len(usable))

rows = []
idx = 0
for size in sizes:
    n = 0
    for rep in range(args.repeats):
        src_frame, obj, alpha = usable[idx % len(usable)]
        idx += 1
        oh, ow = alpha.shape
        scale = size / math.sqrt(oh * ow)
        nw, nh = max(2, int(round(ow * scale))), max(2, int(round(oh * scale)))
        if nw >= args.imgsz - 8 or nh >= args.imgsz - 8:
            continue
        obj_s = cv2.resize(obj, (nw, nh), interpolation=cv2.INTER_AREA)
        a_s = cv2.resize(alpha, (nw, nh), interpolation=cv2.INTER_AREA)
        bgp = bgs[(rep * 7 + int(size)) % len(bgs)]
        bg = cv2.cvtColor(cv2.imread(str(bgp)), cv2.COLOR_BGR2RGB)
        bh, bw = bg.shape[:2]
        if bh < args.imgsz or bw < args.imgsz:
            bg = cv2.resize(bg, (max(args.imgsz, bw), max(args.imgsz, bh)))
            bh, bw = bg.shape[:2]
        # centre position with a small jitter, matching the controlled set convention
        px = args.imgsz // 2 + int(rng.integers(-args.imgsz // 8, args.imgsz // 8))
        py = args.imgsz // 2 + int(rng.integers(-args.imgsz // 8, args.imgsz // 8))
        # Clamp the placement: with a jittered centre a large object can start at a negative offset,
        # and a negative slice bound silently wraps in Python rather than raising.
        top = max(0, min(args.imgsz - nh, py - nh // 2))
        left = max(0, min(args.imgsz - nw, px - nw // 2))
        canvas = bg[0:args.imgsz, 0:args.imgsz].astype(np.float32).copy()
        a3 = a_s[..., None]
        canvas[top:top + nh, left:left + nw] = (canvas[top:top + nh, left:left + nw] * (1 - a3) + obj_s * a3)
        canvas = np.clip(canvas, 0, 255).astype(np.uint8)
        name = "il_{:04.0f}_{:02d}".format(size, rep)
        imwrite_unicode(OUT / "images" / (name + ".jpg"), cv2.cvtColor(canvas, cv2.COLOR_RGB2BGR), quality=92)
        nx1, ny1 = left / args.imgsz, top / args.imgsz
        nx2, ny2 = (left + nw) / args.imgsz, (top + nh) / args.imgsz
        (OUT / "labels" / (name + ".txt")).write_text(
            "0 {:.6f} {:.6f} {:.6f} {:.6f}\n".format((nx1 + nx2) / 2, (ny1 + ny2) / 2, nx2 - nx1, ny2 - ny1),
            encoding="utf-8")
        rows.append({"file": "images/" + name + ".jpg", "split": "fixed_core_test", "source_type": "SYNTHETIC_3D",
                     "bbox_w_px": nw, "bbox_h_px": nh, "pos_x_px": px, "pos_y_px": py,
                     "equiv_size_px": round(math.sqrt(nw * nh), 3), "target_px": size,
                     "background": bgp.name, "imgsz": args.imgsz, "is_negative": "False",
                     "camera_id": "isaac_ladder", "source_frame": src_frame.stem})
        n += 1
    print("  {:>6.0f} px -> {} rows".format(size, n))
cols = list(rows[0].keys())
with (OUT / "manifest.csv").open("w", newline="", encoding="utf-8") as fh:
    w = csv.DictWriter(fh, fieldnames=cols)
    w.writeheader()
    w.writerows(rows)
print("wrote", len(rows), "rows to", OUT / "manifest.csv")