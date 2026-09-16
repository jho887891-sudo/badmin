"""Unpack the Shuttlecock dataset and characterise its boxes."""
import sys, io, zipfile, statistics
from pathlib import Path
from collections import Counter
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
REPO = Path(r"E:\具身智能\badmin_project")
Z = REPO / "_scratch_rf" / "shuttlecock_yolo26_v1.zip"
DST = REPO / "_scratch_rf" / "extracted"
DST.mkdir(parents=True, exist_ok=True)
with zipfile.ZipFile(Z) as z:
    z.extractall(DST)
print("extracted to", DST)

import cv2
rows = []
for split in ("train", "valid", "test"):
    imgs = sorted((DST / split / "images").glob("*.jpg"))
    labs = sorted((DST / split / "labels").glob("*.txt"))
    n_boxes = 0
    n_empty = 0
    sizes = []
    for lab in labs:
        txt = lab.read_text(encoding="utf-8", errors="replace").strip()
        if not txt:
            n_empty += 1
            continue
        for line in txt.splitlines():
            parts = line.split()
            if len(parts) < 5:
                continue
            try:
                cx, cy, w, h = (float(x) for x in parts[1:5])
            except ValueError:
                continue
            n_boxes += 1
            sizes.append((w, h))
    rows.append((split, len(imgs), len(labs), n_boxes, n_empty, sizes))
    print("  {:<6} images={:<6} labels={:<6} boxes={:<6} empty_labels={}".format(split, len(imgs), len(labs), n_boxes, n_empty))

print()
# convert normalised box size to pixels assuming the common 640-wide frame; report both raw and scaled
allw = [w for _, _, _, _, _, s in rows for w, h in s]
allh = [h for _, _, _, _, _, s in rows for w, h in s]
print("normalised box width  : min={:.5f} median={:.5f} max={:.5f}".format(min(allw), statistics.median(allw), max(allw)))
print("normalised box height : min={:.5f} median={:.5f} max={:.5f}".format(min(allh), statistics.median(allh), max(allh)))
print()
px = sorted((w * 640) ** 0.5 * (h * 640) ** 0.5 for w, h in zip(allw, allh))
tot = len(px)
BUCKETS = [("<4", 0, 4), ("4-6", 4, 6), ("6-8", 6, 8), ("8-12", 8, 12), ("12-16", 12, 16),
           ("16-24", 16, 24), ("24-32", 24, 32), ("32-64", 32, 64), (">64", 64, 1e9)]
print("equivalent size at 640 px (sqrt(w*h)):")
for name, lo, hi in BUCKETS:
    c = sum(1 for v in px if lo <= v < hi)
    print("   {:<7} {:>6}  {:>6.2%}".format(name, c, c / tot))
print()
print("median equivalent size at 640 px: {:.1f} px".format(statistics.median(px)))