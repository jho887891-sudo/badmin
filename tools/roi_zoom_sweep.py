"""Level-2 section 3.3, completed: what is the best achievable zoom, size by size?

The 2x ROI result showed a win at the small end and a loss at the large end, which means an OPTIMUM rather
than "more is better". This sweeps five zoom levels across eight target sizes to find the per-size optimum
and the ceiling it implies. The ROI still comes from the ground truth, so these remain upper bounds.
"""
import csv, sys
from pathlib import Path
import cv2
ROOT = Path("/home/T7/dgut/robot_sim")
sys.path.insert(0, str(ROOT / "third_party/ultralytics"))
from ultralytics import YOLO

L = ROOT / "outputs/shuttle_capability/isaac_ladder_small"
man = list(csv.DictReader((L / "manifest.csv").open(encoding="utf-8")))
# NOTE: this parameter is a factor on a 640 crop, and the full 1280 frame is letterboxed to 640 before the
# network sees it. So "1.0" here is already 2x magnified relative to the full frame, and 0.5 here IS the full
# frame. The first version of this sweep omitted 0.5 and therefore had no true baseline - caught because the
# 2.0 and 4.0 columns matched the separate ROI experiment exactly, which pinned the mapping down.
ZOOMS = [0.5, 1.0, 1.5, 2.0, 3.0, 4.0]
m = YOLO(str(ROOT / "outputs/shuttle_detection/training/baseline_os/weights/best.pt"))

def iou(a, b):
    ix = max(0.0, min(a[2], b[2]) - max(a[0], b[0])) * max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    ar = lambda z: max(0.0, z[2] - z[0]) * max(0.0, z[3] - z[1])
    u = ar(a) + ar(b) - ix
    return ix / u if u > 0 else 0.0

by_size_zoom = {}
for z in ZOOMS:
    CROP = int(640 / z)
    for r in man:
        img = cv2.imread(str(L / r["file"]))
        h, w = img.shape[:2]
        cx, cy = float(r["pos_x_px"]), float(r["pos_y_px"])
        bw, bh = float(r["bbox_w_px"]), float(r["bbox_h_px"])
        gt = (cx - bw / 2, cy - bh / 2, cx + bw / 2, cy + bh / 2)
        x0 = max(0, min(w - CROP, int(cx - CROP / 2)))
        y0 = max(0, min(h - CROP, int(cy - CROP / 2)))
        patch = img[y0:y0 + CROP, x0:x0 + CROP]
        b = m.predict(patch, imgsz=640, conf=0.05, verbose=False, device=0)[0].boxes
        hit = False
        if b is not None and len(b):
            for x1, y1, x2, y2 in b.xyxy.cpu().numpy():
                if iou((x0 + x1, y0 + y1, x0 + x2, y0 + y2), gt) >= 0.5:
                    hit = True
                    break
        size = float(r["target_px"])
        by_size_zoom.setdefault((size, z), []).append(hit)

print("  size     " + "".join("{:>9}".format("{:g}x".format(z)) for z in ZOOMS) + "   best")
best_of = {}
for s in sorted({k[0] for k in by_size_zoom}):
    row = "  {:>5.0f}   ".format(s)
    vals = []
    for z in ZOOMS:
        g = by_size_zoom[(s, z)]
        v = sum(g) / len(g)
        vals.append((v, z))
        row += "{:>9.3f}".format(v)
    bv, bz = max(vals)
    best_of[s] = (bv, bz, vals[0][0])
    row += "   {:g}x {:.3f} ({:+.3f})".format(bz, bv, bv - vals[0][0])
    print(row)
print()
print("  ceiling if the best zoom were always chosen:")
flat = sum(v[2] for v in best_of.values()) / len(best_of)
best = sum(v[0] for v in best_of.values()) / len(best_of)
print("    flat 1x   {:.3f}".format(flat))
print("    adaptive  {:.3f}   delta {:+.3f}".format(best, best - flat))