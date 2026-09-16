"""Is the real-match false positive a STATIC object? Test it geometrically rather than by eye."""
import csv, sys
from pathlib import Path
import numpy as np
ROOT = Path("/home/T7/dgut/robot_sim")
sys.path.insert(0, str(ROOT / "third_party/ultralytics"))
from ultralytics import YOLO

L = ROOT / "outputs/shuttle_capability/real_match_frames"
rows = list(csv.DictReader((L / "manifest.csv").open(encoding="utf-8")))
m = YOLO(str(ROOT / "outputs/shuttle_detection/training/baseline_os/weights/best.pt"))

per_video = {}
for r in rows:
    res = m.predict(str(L / r["file"]), imgsz=640, conf=0.05, verbose=False, device=0)[0]
    b = res.boxes
    if b is None or not len(b):
        continue
    key = r["video"]
    for (x1, y1, x2, y2), cf in zip(b.xyxy.cpu().numpy(), b.conf.cpu().numpy()):
        cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
        per_video.setdefault(key, []).append((float(cf), float(cx), float(cy)))

print("{:<26} {:>5} {:>28} {:>26}".format("clip", "boxes", "centre spread (px)", "confidence range"))
static_clips = 0
for key in sorted(per_video):
    v = per_video[key]
    if len(v) < 2:
        print("  {:<24} {:>5}   (single box, spread undefined)".format(key[:24], len(v)))
        continue
    xs = np.array([c[1] for c in v]); ys = np.array([c[2] for c in v])
    sx, sy = float(xs.std()), float(ys.std())
    cf = [c[0] for c in v]
    flag = ""
    if sx < 12 and sy < 12:
        static_clips += 1
        flag = "  <-- STATIC"
    print("  {:<24} {:>5}   {:>10.1f}, {:>10.1f}   {:>10.2f} - {:<6.2f}{}".format(
        key[:24], len(v), sx, sy, min(cf), max(cf), flag))
print()
print("clips whose boxes barely move (< 12 px spread):", static_clips, "of", len(per_video))
print("A shuttle in flight moves; a wall fixture does not. This distinguishes them without a ground truth.")