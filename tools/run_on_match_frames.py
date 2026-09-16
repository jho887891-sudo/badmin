"""Run the frozen v2 model on real match frames and build a contact sheet for visual judgement."""
import csv, sys
from pathlib import Path
import numpy as np, cv2
ROOT = Path("/home/T7/dgut/robot_sim")
sys.path.insert(0, str(ROOT / "third_party/ultralytics"))
from ultralytics import YOLO

L = ROOT / "outputs/shuttle_capability/real_match_frames"
rows = list(csv.DictReader((L / "manifest.csv").open(encoding="utf-8")))
m = YOLO(str(ROOT / "outputs/shuttle_detection/training/baseline_os/weights/best.pt"))
print("frames:", len(rows))

out = []
for r in rows:
    p = L / r["file"]
    res = m.predict(str(p), imgsz=640, conf=0.05, verbose=False, device=0)[0]
    b = res.boxes
    confs = [] if b is None else [float(c) for c in b.conf.cpu().numpy()]
    n = len(confs)
    out.append({"file": r["file"].split("/")[-1], "match": r["match"], "video": r["video"],
                "n": n, "worst": max(confs) if confs else 0.0,
                "n_hi": sum(1 for c in confs if c >= 0.5)})

fired = sum(1 for o in out if o["n"] > 0)
hi = sum(1 for o in out if o["n_hi"] > 0)
print("  frames with any box      : {} of {} ({:.1%})".format(fired, len(out), fired / len(out)))
print("  frames with a box >= 0.5 : {} of {}".format(hi, len(out)))
print("  total boxes              : {}".format(sum(o["n"] for o in out)))
print("  highest confidence seen  : {:.3f}".format(max(o["worst"] for o in out)))
print()
from collections import Counter
c = Counter(o["match"] for o in out if o["n_hi"] > 0)
tot = Counter(o["match"] for o in out)
print("  by match (frames with a >=0.5 box / frames):")
for k in sorted(tot):
    print("    {:<18} {:>3} / {:>3}".format(k, c.get(k, 0), tot[k]))

with (L / "detections.csv").open("w", newline="", encoding="utf-8") as fh:
    w = csv.DictWriter(fh, fieldnames=list(out[0].keys()))
    w.writeheader(); w.writerows(out)

# contact sheet: the frames with the strongest detections, boxes drawn
ranked = sorted(out, key=lambda o: -o["worst"])[:24]
TILE, COLS = 320, 6
sheet = np.full(((len(ranked) // COLS + 1) * TILE, COLS * TILE, 3), 18, np.uint8)
for i, o in enumerate(ranked):
    img = cv2.imread(str(L / "images" / o["file"]))
    b = m.predict(str(L / "images" / o["file"]), imgsz=640, conf=0.05, verbose=False, device=0)[0].boxes
    if b is not None and len(b):
        for (x1, y1, x2, y2), cf in zip(b.xyxy.cpu().numpy(), b.conf.cpu().numpy()):
            col = (0, 0, 255) if cf >= 0.5 else (0, 165, 255)
            cv2.rectangle(img, (int(x1), int(y1)), (int(x2), int(y2)), col, 2)
            cv2.putText(img, "%.2f" % cf, (int(x1), max(14, int(y1) - 4)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, col, 1)
    h, w = img.shape[:2]
    s = TILE / max(h, w)
    t = cv2.resize(img, (max(1, int(w * s)), max(1, int(h * s))), interpolation=cv2.INTER_AREA)
    rr, cc = i // COLS, i % COLS
    sheet[rr * TILE:rr * TILE + t.shape[0], cc * TILE:cc * TILE + t.shape[1]] = t
    cv2.putText(sheet, o["match"][:12], (cc * TILE + 4, rr * TILE + 16), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 255), 1)
OUTP = ROOT / "outputs/evidence/real_match_detections.jpg"
cv2.imwrite(str(OUTP), sheet, [int(cv2.IMWRITE_JPEG_QUALITY), 92])
print()
print("wrote", OUTP, sheet.shape)