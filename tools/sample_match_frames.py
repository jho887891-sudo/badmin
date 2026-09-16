"""Sample frames from real match footage and run the frozen model on them.

This is the real-domain measurement the project never had: 30 rally clips of actual matches, professional and
amateur, rather than the 150 low-quality frames used earlier whose suitability was itself in doubt.
"""
import csv, sys, io
from pathlib import Path
import cv2, numpy as np
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
REPO = Path(r"E:\具身智能\badmin_project")
SRC = REPO / "_scratch_matches"
OUT = REPO / "outputs" / "shuttle_capability" / "real_match_frames"
(OUT / "images").mkdir(parents=True, exist_ok=True)
PER_CLIP = 8
rows = []
for vid in sorted(SRC.rglob("*.mp4")):
    cap = cv2.VideoCapture(str(vid))
    n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    if n < PER_CLIP * 2:
        cap.release(); continue
    idxs = np.linspace(0, n - 1, PER_CLIP).astype(int)
    got = 0
    for k, fi in enumerate(idxs):
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(fi))
        ok, frame = cap.read()
        if not ok:
            continue
        h, w = frame.shape[:2]
        name = "{}_{}_{:03d}.jpg".format(vid.parent.parent.name, vid.stem, k)
        # cv2.imwrite SILENTLY FAILS on a non-ASCII path, which is what this repository lives on:
        # the first run of this script recorded 235 manifest rows and wrote 0 images. Use the
        # project helper, which encodes to a buffer and writes the bytes itself.
        sys.path.insert(0, str(REPO / "tools"))
        from shuttle_render import imwrite_unicode
        imwrite_unicode(OUT / "images" / name, frame, quality=92)
        rows.append({"file": "images/" + name, "split": "fixed_core_test", "source_type": "REAL_VIDEO",
                     "video": vid.name, "match": vid.parent.parent.name, "frame_index": int(fi),
                     "width": w, "height": h, "is_negative": "False"})
        got += 1
    cap.release()
    print("  {:<34} {} frames of {}".format(vid.parent.parent.name + "/" + vid.name, got, n))
cols = ["file", "split", "source_type", "video", "match", "frame_index", "width", "height", "is_negative"]
with (OUT / "manifest.csv").open("w", newline="", encoding="utf-8") as fh:
    w = csv.DictWriter(fh, fieldnames=cols)
    w.writeheader(); w.writerows(rows)
print()
print("sampled", len(rows), "frames to", OUT)