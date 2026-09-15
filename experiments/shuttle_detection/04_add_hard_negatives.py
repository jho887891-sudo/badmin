#!/usr/bin/env python3
"""04_add_hard_negatives.py - merge the HARD_NEGATIVE_POOL into a training manifest.

Spec 06 section 6: false positives -> classify the source -> collect NEW images of the same kind ->
retrain. The pool came from Wikimedia Commons and was selected by MEASURED model response: 31 of the 36
candidates already fire on baseline_nearfield, producing 70 boxes with 6 above 0.5 confidence.

Two defects the dataset audit caught in the first version of this script, both now fixed:
  1. LABEL_MISSING for all 36 - the existing negatives each carry a ZERO-BYTE .txt label, and these had
     none. Training on a negative with no label is not the same as training on an empty label.
  2. IMAGE_SIZE_MISMATCH for all 36 - imgsz was recorded as 1280 while the downloads are 1280 wide and
     of varying height. They are now centre-cropped to a square and resized to the same 960x960 the
     other training images use, so the pool is homogeneous and the recorded size is the real size.
"""
from __future__ import annotations

import argparse
import csv
import shutil
import sys
from pathlib import Path

import cv2

REPO = Path(__file__).resolve().parents[2]
PREFIX = "hardneg_train"
SIZE = 960


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", default=str(REPO))
    ap.add_argument("--base-manifest", required=True)
    ap.add_argument("--out-manifest", required=True)
    args = ap.parse_args()

    repo = Path(args.repo).resolve()
    sys.path.insert(0, str(repo / "tools"))
    from shuttle_render import imread_unicode, imwrite_unicode

    base_dir = repo / "outputs" / "shuttle_capability" / "train_data"
    hn_dir = repo / "outputs" / "shuttle_capability" / "hard_negatives"
    core_dir = repo / "outputs" / "shuttle_capability" / "real_images" / "backgrounds"

    def signature(path, n=64):
        im = imread_unicode(path)
        if im is None:
            return None
        g = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
        g = cv2.resize(g, (n, n), interpolation=cv2.INTER_AREA).astype("float64")
        g -= g.mean()
        s = g.std()
        return g / s if s > 1e-9 else g

    # LEAK GUARD. The first version of this script had none, and three of the 36 downloaded candidates
    # turned out to be FROZEN CORE test backgrounds re-downloaded from the same source: hn_001 == bg_026,
    # hn_007 == bg_025 and hn_008 == bg_028, all at r = 1.0000 on this signature. Training on those puts
    # test scenes into the training pool, which the project's own leakage rule (r >= 0.95) forbids.
    core = {p.name: signature(p) for p in sorted(core_dir.glob("*.jpg"))}
    core = {k: v for k, v in core.items() if v is not None}
    rows = list(csv.DictReader((base_dir / args.base_manifest).open(encoding="utf-8")))
    columns = list(rows[0].keys())
    meta = list(csv.DictReader((hn_dir / "hard_negative_metadata.csv").open(encoding="utf-8")))

    added = 0
    for i, m in enumerate(meta):
        src = hn_dir / "raw" / m["file"]
        if not src.is_file():
            print("  missing", src.name)
            continue
        sig = signature(src)
        if sig is not None and core:
            worst = max(((float((sig * c).mean()), n) for n, c in core.items()), key=lambda t: t[0])
            if worst[0] >= 0.95:
                print("  EXCLUDED {}: content matches frozen core {} at r={:.4f}".format(
                    src.name, worst[1], worst[0]))
                continue
        img = imread_unicode(src)
        if img is None:
            print("  unreadable", src.name)
            continue
        h, w = img.shape[:2]
        side = min(h, w)
        y0 = (h - side) // 2
        x0 = (w - side) // 2
        square = img[y0:y0 + side, x0:x0 + side]
        resized = cv2.resize(square, (SIZE, SIZE), interpolation=cv2.INTER_AREA)
        name = "{}_{:05d}.jpg".format(PREFIX, i)
        imwrite_unicode(base_dir / "train" / "images" / name, resized, quality=92)
        # A negative carries an EMPTY label file, exactly like the existing neg_train_* negatives.
        (base_dir / "train" / "labels" / ("{}_{:05d}.txt".format(PREFIX, i))).write_text("", encoding="utf-8")
        row = {c: "" for c in columns}
        row.update({
            "file": name,
            "split": "train",
            "source_type": "NEGATIVE",
            "background": m["file"],
            "imgsz": str(SIZE),
            "is_negative": "True",
        })
        rows.append(row)
        added += 1

    out = base_dir / args.out_manifest
    with out.open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=columns)
        w.writeheader()
        for r in rows:
            w.writerow({c: r.get(c, "") for c in columns})
    pos = sum(1 for r in rows if r.get("is_negative") == "False")
    neg = sum(1 for r in rows if r.get("is_negative") == "True")
    print("added {} hard negatives from {} candidates".format(added, len(meta)))
    print("  each centre-cropped to a square and resized to {}x{}, with an empty label file".format(SIZE, SIZE))
    print("  manifest now {} rows: {} positives, {} negatives".format(len(rows), pos, neg))
    print("  negatives are {:.0%} of the pool".format(neg / len(rows)))
    print("wrote", out.name)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())