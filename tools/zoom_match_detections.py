"""Zoom into every detection on real match frames: is the model looking at an actual shuttle?"""
import csv, sys
from pathlib import Path
import numpy as np, cv2
ROOT = Path("/home/T7/dgut/robot_sim")
sys.path.insert(0, str(ROOT / "third_party/ultralytics"))
from ultralytics import YOLO

L = ROOT / "outputs/shuttle_capability/real_match_frames"
rows = list(csv.DictReader((L / "manifest.csv").open(encoding="utf-8")))
m = YOLO(str(ROOT / "outputs/shuttle_detection/training/baseline_os/weights/best.pt"))

dets = []
for r in rows:
    p = L / r["file"]
    res = m.predict(str(p), imgsz=640, conf=0.05, verbose=False, device=0)[0]
    b = res.boxes
    if b is None or not len(b):
        continue
    for (x1, y1, x2, y2), cf in zip(b.xyxy.cpu().numpy(), b.conf.cpu().numpy()):
        dets.append((float(cf), str(p), (x1, y1, x2, y2)))
dets.sort(key=lambda d: -d[0])
print("total detections:", len(dets))
print("top 12 confidences:", ["%.2f" % d[0] for d in dets[:12]])

# crop a fixed window around each of the strongest detections and magnify it
WIN = 110
COLS = 6
sel = dets[:24]
TILE = 320
rows_n = (len(sel) + COLS - 1) // COLS
sheet = np.full((rows_n * TILE, COLS * TILE, 3), 18, np.uint8)
for i, (cf, path, (x1, y1, x2, y2)) in enumerate(sel):
    img = cv2.imread(path)
    h, w = img.shape[:2]
    cx, cy = int((x1 + x2) / 2), int((y1 + y2) / 2)
    x0, y0 = max(0, cx - WIN), max(0, cy - WIN)
    crop = img[y0:y0 + 2 * WIN, x0:x0 + 2 * WIN].copy()
    # draw the box in crop coordinates, then magnify
    cv2.rectangle(crop, (int(x1) - x0, int(y1) - y0), (int(x2) - x0, int(y2) - y0), (0, 0, 255), 2)
    t = cv2.resize(crop, (TILE, TILE), interpolation=cv2.INTER_NEAREST)
    rr, cc = i // COLS, i % COLS
    sheet[rr * TILE:(rr + 1) * TILE, cc * TILE:(cc + 1) * TILE] = t
    cv2.putText(sheet, "%.2f" % cf, (cc * TILE + 6, rr * TILE + 24), cv2.FONT_HERSHEY_SIMPLEX, 0.8,
                (0, 255, 255), 2)
    cv2.rectangle(sheet, (cc * TILE, rr * TILE), (cc * TILE + TILE - 1, rr * TILE + TILE - 1), (70, 70, 70), 1)
out = ROOT / "outputs/evidence/match_detections_zoom.jpg"
cv2.imwrite(str(out), sheet, [int(cv2.IMWRITE_JPEG_QUALITY), 94])
print("wrote", out, sheet.shape)