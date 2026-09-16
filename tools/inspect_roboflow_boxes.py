"""Are these real bounding boxes or fixed-size point markers? Look at them."""
import sys, io, random
from pathlib import Path
import cv2, numpy as np
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
REPO = Path(r"E:\具身智能\badmin_project")
D = REPO / "_scratch_rf" / "extracted" / "train"
from collections import Counter

# first: the exact set of (w,h) pairs, to see how many distinct box shapes exist
shapes = Counter()
for lab in sorted((D / "labels").glob("*.txt")):
    for line in lab.read_text(encoding="utf-8", errors="replace").splitlines():
        p = line.split()
        if len(p) >= 5:
            shapes[(round(float(p[3]), 5), round(float(p[4]), 5))] += 1
print("distinct (w,h) shapes in train:", len(shapes))
for sh, c in shapes.most_common(12):
    print("   w={:<9} h={:<9} n={}".format(sh[0], sh[1], c))
print()

rng = random.Random(7)
imgs = sorted((D / "images").glob("*.jpg"))
sel = rng.sample(imgs, 12)
TILE, COLS = 320, 4
rows = (len(sel) + COLS - 1) // COLS
sheet = np.full((rows * TILE, COLS * TILE, 3), 18, np.uint8)
for i, ip in enumerate(sel):
    # cv2.imread fails on a non-ASCII path; decode from bytes instead (same trap as before)
    img = cv2.imdecode(np.fromfile(str(ip), dtype=np.uint8), cv2.IMREAD_COLOR)
    if img is None:
        continue
    h, w = img.shape[:2]
    lab = D / "labels" / (ip.stem + ".txt")
    for line in (lab.read_text(encoding="utf-8", errors="replace").splitlines() if lab.is_file() else []):
        p = line.split()
        if len(p) < 5:
            continue
        cx, cy, bw, bh = (float(x) for x in p[1:5])
        x1, y1 = int((cx - bw / 2) * w), int((cy - bh / 2) * h)
        x2, y2 = int((cx + bw / 2) * w), int((cy + bh / 2) * h)
        cv2.rectangle(img, (x1 - 8, y1 - 8), (x2 + 8, y2 + 8), (0, 0, 255), 3)
    s = TILE / max(h, w)
    t = cv2.resize(img, (max(1, int(w * s)), max(1, int(h * s))), interpolation=cv2.INTER_AREA)
    rr, cc = i // COLS, i % COLS
    sheet[rr * TILE:rr * TILE + t.shape[0], cc * TILE:cc * TILE + t.shape[1]] = t
    cv2.putText(sheet, "{}x{}".format(w, h), (cc * TILE + 4, rr * TILE + 18),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
out = REPO / "outputs" / "evidence" / "roboflow_samples.jpg"
cv2.imencode(".jpg", sheet, [int(cv2.IMWRITE_JPEG_QUALITY), 92])[1].tofile(str(out))
print("wrote", out, sheet.shape)