"""Composite the Isaac Sim pool onto the real training backgrounds.

The Isaac frames are rendered on a flat grey background with an exact semantic mask. This cuts the
object out, rescales it to a target apparent size drawn from the SAME log-uniform range the numpy pool
uses, and composites it with the SAME composite() the numpy pipeline uses - so background, shadow
behaviour, noise and JPEG encoding are identical between the two pools and the only difference is the
RENDERER. That is what makes the comparison interpretable.
"""
import argparse, csv, math, sys
from pathlib import Path
import numpy as np, cv2
REPO = Path("/home/T7/dgut/robot_sim")
sys.path.insert(0, str(REPO / "tools"))
from shuttle_render import composite, imread_unicode, imwrite_unicode

ap = argparse.ArgumentParser()
ap.add_argument("--n", type=int, default=400)
ap.add_argument("--min-px", type=float, default=3.0)
ap.add_argument("--max-px", type=float, default=800.0)
ap.add_argument("--imgsz", type=int, default=960)
ap.add_argument("--seed", type=int, default=20260917)
ap.add_argument("--pool", default=None)
ap.add_argument("--name-prefix", default="isaac_train")
args = ap.parse_args()

# Parameterised so a second pool rendered with a wider pitch range can be composited alongside the first,
# with distinct filenames so neither overwrites the other.
POOL = Path(args.pool) if getattr(args, "pool", None) else REPO / "outputs/shuttle_capability/isaac_pool"
NAME_PREFIX = getattr(args, "name_prefix", "isaac_train")
BASE = REPO / "outputs/shuttle_capability/train_data"
OUTI = BASE / "train" / "images"
OUTL = BASE / "train" / "labels"
OUTI.mkdir(parents=True, exist_ok=True)
OUTL.mkdir(parents=True, exist_ok=True)
frames = sorted(p for p in (POOL / "rgb").glob("*.png"))
bgs = sorted(p for p in (BASE / "bg_train").glob("*") if p.suffix.lower() in (".jpg", ".jpeg", ".png"))
print("isaac frames:", len(frames), "| training backgrounds:", len(bgs))
rng = np.random.default_rng(args.seed)
rows = []
for i in range(args.n):
    fp = frames[i % len(frames)]
    img = imread_unicode(fp)
    lab = np.load(POOL / "mask" / (fp.stem + ".npy"))
    mask = (lab[..., -1] != 0).astype(np.uint8)
    ys, xs = np.nonzero(mask)
    if xs.size < 20:
        continue
    x0, x1 = int(xs.min()), int(xs.max()) + 1
    y0, y1 = int(ys.min()), int(ys.max()) + 1
    obj = img[y0:y1, x0:x1].astype(np.float32)
    alpha = (mask[y0:y1, x0:x1].astype(np.float32))
    oh, ow = alpha.shape
    target = float(np.exp(rng.uniform(math.log(args.min_px), math.log(args.max_px))))
    scale = target / math.sqrt(oh * ow)
    nw, nh = max(2, int(round(ow * scale))), max(2, int(round(oh * scale)))
    if nw >= args.imgsz or nh >= args.imgsz:
        continue
    obj_s = cv2.resize(obj, (nw, nh), interpolation=cv2.INTER_AREA)
    alpha_s = cv2.resize(alpha, (nw, nh), interpolation=cv2.INTER_AREA)
    bgp = bgs[i % len(bgs)]
    bg = imread_unicode(bgp)
    h, w = bg.shape[:2]
    if h < args.imgsz or w < args.imgsz:
        bg = cv2.resize(bg, (max(args.imgsz, w), max(args.imgsz, h)))
        h, w = bg.shape[:2]
    bx = int(rng.integers(0, w - args.imgsz + 1))
    by = int(rng.integers(0, h - args.imgsz + 1))
    canvas = bg[by:by + args.imgsz, bx:bx + args.imgsz].copy()
    px = int(rng.integers(nw // 2 + 2, args.imgsz - nw // 2 - 2))
    py = int(rng.integers(nh // 2 + 2, args.imgsz - nh // 2 - 2))
    canvas_f = canvas.astype(np.float32)
    top, left = py - nh // 2, px - nw // 2
    alpha3 = alpha_s[..., None]
    canvas_f[top:top + nh, left:left + nw] = (canvas_f[top:top + nh, left:left + nw] * (1 - alpha3)
                                              + obj_s * alpha3)
    canvas = np.clip(canvas_f, 0, 255).astype(np.uint8)
    name = "{}_{:05d}".format(NAME_PREFIX, i)
    imwrite_unicode(OUTI / (name + ".jpg"), canvas, quality=92)
    nx1, ny1 = left / args.imgsz, top / args.imgsz
    nx2, ny2 = (left + nw) / args.imgsz, (top + nh) / args.imgsz
    (OUTL / (name + ".txt")).write_text(
        "0 {:.6f} {:.6f} {:.6f} {:.6f}\n".format((nx1 + nx2) / 2, (ny1 + ny2) / 2, nx2 - nx1, ny2 - ny1),
        encoding="utf-8")
    rows.append({"name": name, "equiv": round(math.sqrt(nw * nh), 2), "background": bgp.name,
                 "source_frame": fp.stem})
with (POOL / "composite_manifest.csv").open("w", newline="", encoding="utf-8") as fh:
    w = csv.DictWriter(fh, fieldnames=["name", "equiv", "background", "source_frame"])
    w.writeheader()
    w.writerows(rows)
sizes = [r["equiv"] for r in rows]
if sizes:
    print("composited {} samples; size min {:.1f} median {:.1f} max {:.1f} px".format(
          len(rows), min(sizes), float(np.median(sizes)), max(sizes)))
print("wrote", POOL / "composite_manifest.csv")