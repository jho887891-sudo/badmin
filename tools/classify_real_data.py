"""Classify the acquired real data: by source video, by annotation geometry, by image properties."""
import sys, io, re, statistics, collections
from pathlib import Path
import numpy as np, cv2
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
REPO = Path(r"E:\具身智能\badmin_project")
RF = REPO / "_scratch_rf" / "extracted"

def read(p):
    return cv2.imdecode(np.fromfile(str(p), dtype=np.uint8), cv2.IMREAD_COLOR)

rec = []
for split in ("train", "valid", "test"):
    for lab in sorted((RF / split / "labels").glob("*.txt")):
        name = lab.stem
        m = re.match(r"video_label_(\d+)_(\d+)", name)
        grp = "video_label_" + m.group(1) if m else "other"
        frame = int(m.group(2)) if m else -1
        boxes = []
        for line in lab.read_text(encoding="utf-8", errors="replace").splitlines():
            p = line.split()
            if len(p) >= 5:
                boxes.append(tuple(round(float(v), 5) for v in p[1:5]))
        rec.append({"split": split, "name": name, "grp": grp, "frame": frame, "boxes": boxes})

print("total labelled images:", len(rec))
groups = collections.Counter(r["grp"] for r in rec)
print("distinct source-video groups:", len(groups))
print()
print("=== classification by source video ===")
print("{:<18} {:>6} {:>7} {:>9} {:>22} {:>10}".format("group", "images", "empty", "boxes/img", "box (w,h) normalized", "frame range"))
for g, n in sorted(groups.items(), key=lambda kv: -kv[1]):
    sub = [r for r in rec if r["grp"] == g]
    nb = sum(len(r["boxes"]) for r in sub)
    empty = sum(1 for r in sub if not r["boxes"])
    # boxes are stored as (cx, cy, w, h); the SIZE is b[2:4], not b[:2] - the first version took the
    # centre by mistake and reported 0.32x0.48 "box sizes" that were really normalised coordinates.
    shapes = collections.Counter(b[2:4] for r in sub if r["boxes"] for b in r["boxes"])
    top = shapes.most_common(1)[0] if shapes else ((0, 0), 0)
    fr = [r["frame"] for r in sub if r["frame"] >= 0]
    print("{:<18} {:>6} {:>7} {:>9.2f} {:>22} {:>10}".format(
        g, len(sub), empty, nb / len(sub),
        "{}x{}".format(top[0][0], top[0][1]),
        "{}-{}".format(min(fr), max(fr)) if fr else "-"))
# image properties and the per-group size distribution in PIXELS
print()
print("=== classification by image property ===")
import random
rng = random.Random(11)
for g, n in sorted(groups.items(), key=lambda kv: -kv[1]):
    sub = [r for r in rec if r["grp"] == g]
    samp = rng.sample(sub, min(120, len(sub)))
    dims = collections.Counter()
    means = []
    for r in samp:
        p = None
        for split in ("train", "valid", "test"):
            cand = RF / split / "images" / (r["name"] + ".jpg")
            if cand.is_file():
                p = cand
                break
        if p is None:
            continue
        im = read(p)
        if im is None:
            continue
        h, w = im.shape[:2]
        dims["{}x{}".format(w, h)] += 1
        means.append(float(im.mean()))
    sizes_px = []
    for r in sub:
        for b in r["boxes"]:
            sizes_px.append((b[2] * 640 * b[3] * 640) ** 0.5)
    print("  {:<18} dims={:<22} mean_brightness={:.0f}  box_px@640: min={:.1f} med={:.1f} max={:.1f}".format(
        g, str(dims.most_common(2)), (sum(means) / len(means)) if means else 0,
        min(sizes_px) if sizes_px else 0, statistics.median(sizes_px) if sizes_px else 0,
        max(sizes_px) if sizes_px else 0))

print()
print("=== negatives (empty labels) by group and split ===")
for g in sorted(groups):
    sub = [r for r in rec if r["grp"] == g and not r["boxes"]]
    if sub:
        c = collections.Counter(r["split"] for r in sub)
        print("  {:<18} {}  {}".format(g, len(sub), dict(c)))