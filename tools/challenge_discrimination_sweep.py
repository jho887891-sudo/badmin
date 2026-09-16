"""Threshold sweep on the paired C1/C1n design: is there an operating point WITH discrimination?

The release fires about once per scene on 33 cluttered scenes whether or not a shuttle is present, so a
detection there carries little information. The threshold scan so far was done on shuttle-free scenes only.
This one asks the question that matters: does any threshold make the empty-scene rate fall faster than the
true-positive rate, which would mean an operating point with real discrimination?
"""
import csv, sys
from pathlib import Path
ROOT = Path("/home/T7/dgut/robot_sim")
sys.path.insert(0, str(ROOT / "third_party/ultralytics"))
from ultralytics import YOLO

CH = ROOT / "outputs/shuttle_capability/challenge_test"
MAN = CH / "manifest.csv"
W = ROOT / "outputs/shuttle_detection/training/baseline_isaac2/weights/best.pt"
THRESHOLDS = [0.05, 0.10, 0.20, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80]

rows = list(csv.DictReader(MAN.open(encoding="utf-8")))
c1 = [r for r in rows if r.get("sweep") == "C1"]
c1n = [r for r in rows if r.get("sweep") == "C1n"]
print("C1 rows:", len(c1), "| C1n rows:", len(c1n))

m = YOLO(str(W))
c1_pred = []
for r in c1:
    res = m.predict(str(CH / r["file"]), imgsz=640, conf=0.05, verbose=False, device=0)[0]
    b = res.boxes
    confs = [] if b is None else list(b.conf.cpu().numpy())
    boxes = None if b is None else b.xyxy.cpu().numpy()
    try:
        w_, h_ = float(r["bbox_w_px"]), float(r["bbox_h_px"])
        cx, cy = float(r["pos_x_px"]), float(r["pos_y_px"])
        gt = (cx - w_ / 2, cy - h_ / 2, cx + w_ / 2, cy + h_ / 2)
    except (TypeError, ValueError):
        gt = None
    c1_pred.append((confs, boxes, gt))
c1n_counts = []
for r in c1n:
    res = m.predict(str(CH / r["file"]), imgsz=640, conf=0.05, verbose=False, device=0)[0]
    b = res.boxes
    c1n_counts.append([] if b is None else list(b.conf.cpu().numpy()))

def iou(a, b):
    ix = max(0.0, min(a[2], b[2]) - max(a[0], b[0])) * max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    ar = lambda z: max(0.0, z[2] - z[0]) * max(0.0, z[3] - z[1])
    u = ar(a) + ar(b) - ix
    return ix / u if u > 0 else 0.0

print()
print("{:>6} {:>16} {:>22} {:>14}".format("conf", "C1 recall", "C1n boxes/scene", "discrimination"))
for t in THRESHOLDS:
    hits = 0
    for confs, boxes, gt in c1_pred:
        keep = [boxes[i] for i in range(len(confs)) if confs[i] >= t] if boxes is not None else []
        if gt is not None and any(iou(bx, gt) >= 0.5 for bx in keep):
            hits += 1
    rec = hits / max(1, len(c1_pred))
    fp = sum(sum(1 for c in confs if c >= t) for confs in c1n_counts) / max(1, len(c1n_counts))
    # discrimination ratio: a true positive per empty-scene box. >1 means a detection is informative.
    ratio = (rec / fp) if fp > 0 else float("inf")
    print("{:>6.2f} {:>16} {:>22} {:>14}".format(
        t, "{}/{} = {:.3f}".format(hits, len(c1_pred), rec), "{:.2f}".format(fp),
        "{:.2f}".format(ratio) if ratio != float("inf") else "inf"))
print()
print("A ratio at or below 1.0 means a detection on these scenes is no more likely to be the target than")
print("to be a false positive; the release needs a ratio clearly above 1 for these scenes to be usable.")