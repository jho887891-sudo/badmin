"""Level-1 threshold operating point scan (spec 07 section 2.2: inference confidence threshold).

Every measurement so far used conf >= 0.05, which spec 07 requires be scanned before Level 2 can be
entered. For each threshold this reports, on the SAME images:
  - recall on the six DISTINCT verified real positives (the set is 10 rows but 4 are byte-duplicates)
  - false positives on the 30 shuttle-free real scenes, which is the cleanest precision protocol
  - recall on the controlled S8 size curve, to see what the small-target cost of a high threshold is
"""
import csv, sys
from pathlib import Path
from collections import defaultdict
ROOT = Path("/home/T7/dgut/robot_sim")
sys.path.insert(0, str(ROOT / "third_party/ultralytics"))
from ultralytics import YOLO

BG = ROOT / "outputs/shuttle_capability/real_images/backgrounds"
RAW = ROOT / "outputs/shuttle_capability/real_images/raw"
MAN = ROOT / "outputs/shuttle_capability/metrics/real_image_verified_manifest.csv"
CC = ROOT / "outputs/shuttle_capability/controlled_capability"
W = ROOT / "outputs/shuttle_detection/training/baseline_round3/weights/best.pt"
THRESHOLDS = [0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.40, 0.50, 0.60, 0.70]

bg_files = sorted(BG.glob("*.jpg")) + sorted(BG.glob("*.png"))
rows = list(csv.DictReader(MAN.open(encoding="utf-8")))
pos = [r for r in rows if r.get("verified_kind") == "positive"]
seen = set()
distinct = []
for r in pos:
    if r["file"] not in seen:
        seen.add(r["file"])
        distinct.append(r)
print("distinct real positives:", len(distinct), "| shuttle-free scenes:", len(bg_files))

sweep = {}
for r in csv.DictReader((CC / "manifest.csv").open(encoding="utf-8")):
    sweep[r["file"].split("/")[-1]] = r.get("sweep", "?")
s8 = [k for k, v in sweep.items() if v == "S8"]
print("S8 rows:", len(s8))
print()

m = YOLO(str(W))
# one inference pass per image at the lowest threshold, then post-hoc thresholding
bg_dets = []
for f in bg_files:
    b = m.predict(str(f), imgsz=640, conf=0.05, verbose=False, device=0)[0].boxes
    bg_dets.append([] if b is None else list(b.conf.cpu().numpy()))

real_dets = []
for r in distinct:
    p = RAW / r["file"]
    b = m.predict(str(p), imgsz=640, conf=0.05, verbose=False, device=0)[0].boxes
    confs = [] if b is None else list(b.conf.cpu().numpy())
    boxes = None if b is None else b.xyxy.cpu().numpy()
    gt = None
    try:
        w_, h_ = float(r["bbox_w_px"]), float(r["bbox_h_px"])
        cx, cy = float(r["pos_x_px"]), float(r["pos_y_px"])
        gt = (cx - w_ / 2, cy - h_ / 2, cx + w_ / 2, cy + h_ / 2)
    except (TypeError, ValueError):
        pass
    real_dets.append((confs, boxes, gt))

s8_dets = []
for name in s8:
    p = CC / "images" / name
    res = m.predict(str(p), imgsz=640, conf=0.05, verbose=False, device=0)[0]
    b = res.boxes
    confs = [] if b is None else list(b.conf.cpu().numpy())
    boxes = None if b is None else b.xyxy.cpu().numpy()
    gt = None
    lab = CC / "labels" / (name.rsplit(".", 1)[0] + ".txt")
    if lab.is_file():
        parts = lab.read_text(encoding="utf-8").split()
        if len(parts) >= 5:
            cx, cy, bw, bh = (float(v) for v in parts[1:5])
            gt = ((cx - bw / 2) * 960, (cy - bh / 2) * 960, (cx + bw / 2) * 960, (cy + bh / 2) * 960)
    s8_dets.append((confs, boxes, gt))

def iou(a, b):
    ix = max(0.0, min(a[2], b[2]) - max(a[0], b[0])) * max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    ar = lambda z: max(0.0, z[2] - z[0]) * max(0.0, z[3] - z[1])
    u = ar(a) + ar(b) - ix
    return ix / u if u > 0 else 0.0

print("{:>6} {:>14} {:>12} {:>14} {:>12}".format("conf", "real recall", "real FP", "clean scenes", "S8 recall"))
for t in THRESHOLDS:
    hits = 0
    fp_real = 0
    for confs, boxes, gt in real_dets:
        keep = [boxes[i] for i in range(len(confs)) if confs[i] >= t] if boxes is not None else []
        matched = False
        for bx in keep:
            if gt is not None and iou(bx, gt) >= 0.5:
                matched = True
            elif gt is None:
                fp_real += 1
        if matched:
            hits += 1
    fp_bg = sum(sum(1 for c in confs if c >= t) for confs in bg_dets)
    clean = sum(1 for confs in bg_dets if not any(c >= t for c in confs))
    s8_hits = s8_ev = 0
    for confs, boxes, gt in s8_dets:
        if gt is None:
            continue
        s8_ev += 1
        keep = [boxes[i] for i in range(len(confs)) if confs[i] >= t] if boxes is not None else []
        if any(iou(bx, gt) >= 0.5 for bx in keep):
            s8_hits += 1
    print("{:>6.2f} {:>14} {:>12} {:>14} {:>12}".format(
        t, "{}/{}".format(hits, len(real_dets)), fp_bg, "{}/{}".format(clean, len(bg_files)),
        "{:.3f}".format(s8_hits / s8_ev if s8_ev else 0.0)))