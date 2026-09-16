"""Level-2 section 3.3: input ROI / crop, measured as an UPPER BOUND.

The small-target limit is a recognition failure at 8-32 px and it sets the far bound of the working range.
Spec 07 section 3.3 asks whether changing the INPUT - here, cropping to a region of interest so the target
occupies more of the network input - moves it. In deployment a ROI would come from a coarse pass or a motion
cue; here it is taken from the ground truth, which makes this an ORACLE upper bound. If even the oracle does
not help, no ROI scheme will, and the item is closed.
"""
import csv, sys
from pathlib import Path
import numpy as np, cv2
ROOT = Path("/home/T7/dgut/robot_sim")
sys.path.insert(0, str(ROOT / "third_party/ultralytics"))
from ultralytics import YOLO

L = ROOT / "outputs/shuttle_capability/isaac_ladder_small"
man = list(csv.DictReader((L / "manifest.csv").open(encoding="utf-8")))
ZOOM = 4.0          # crop side = 640 / ZOOM, so the target is magnified this much before the network sees it
CROP = int(640 / ZOOM)
m = YOLO(str(ROOT / "outputs/shuttle_detection/training/baseline_os/weights/best.pt"))

def load(p):
    return cv2.imread(str(p))

def iou(a, b):
    ix = max(0.0, min(a[2], b[2]) - max(a[0], b[0])) * max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    ar = lambda z: max(0.0, z[2] - z[0]) * max(0.0, z[3] - z[1])
    u = ar(a) + ar(b) - ix
    return ix / u if u > 0 else 0.0

results = {}
for mode in ("full frame", "oracle ROI"):
    by = {}
    for r in man:
        img = load(L / r["file"])
        h, w = img.shape[:2]
        cx, cy = float(r["pos_x_px"]), float(r["pos_y_px"])
        bw, bh = float(r["bbox_w_px"]), float(r["bbox_h_px"])
        gt = (cx - bw / 2, cy - bh / 2, cx + bw / 2, cy + bh / 2)
        if mode == "full frame":
            patch, ox, oy = img, 0, 0
        else:
            x0 = max(0, min(w - CROP, int(cx - CROP / 2)))
            y0 = max(0, min(h - CROP, int(cy - CROP / 2)))
            patch = img[y0:y0 + CROP, x0:x0 + CROP]
            ox, oy = x0, y0
        b = m.predict(patch, imgsz=640, conf=0.05, verbose=False, device=0)[0].boxes
        hit = False
        if b is not None and len(b):
            xy = b.xyxy.cpu().numpy()
            for x1, y1, x2, y2 in xy:
                # Ultralytics already returns boxes in the coordinates of the array it was GIVEN, not in
                # the 640 network input. Dividing by a 640/patch_width factor would scale them a second
                # time, which silently zeroed every result on the first run of this script.
                fb = (ox + x1, oy + y1, ox + x2, oy + y2)
                if iou(fb, gt) >= 0.5:
                    hit = True
                    break
        size = float(r["target_px"])
        by.setdefault(size, []).append(hit)
    results[mode] = by

print("  size    n    full frame   oracle ROI ({}x)   delta".format(ZOOM))
for s in sorted(results["full frame"]):
    a = results["full frame"][s]
    c = results["oracle ROI"][s]
    print("  {:>5.0f} {:>4}      {:.3f}         {:.3f}        {:+.3f}".format(
        s, len(a), sum(a) / len(a), sum(c) / len(c), sum(c) / len(c) - sum(a) / len(a)))
print()
fa = [v for s in results["full frame"] for v in results["full frame"][s]]
fb = [v for s in results["oracle ROI"] for v in results["oracle ROI"][s]]
print("  overall: full frame {:.3f}  vs  oracle ROI {:.3f}   delta {:+.3f}".format(
    sum(fa) / len(fa), sum(fb) / len(fb), sum(fb) / len(fb) - sum(fa) / len(fa)))