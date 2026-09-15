#!/usr/bin/env python3
"""01_build_negatives.py - add ordinary (shuttle-free) negatives and build the training
manifest pair required by Spec 02 section 6.

Why this file exists: the 400 train / 120 val samples delivered by the P4-A renderer are 100%
positive - every image carries exactly one composited shuttlecock. Spec 02 section 6 requires
ordinary negatives in every training run; without them precision and the false-positive rate
are meaningless. This script crops the split background pool with no shuttle composited,
writes an EMPTY YOLO label for each, and emits manifest_<split>_synthetic.csv combining the
positive rows with the negative rows, with an explicit is_negative column.

It writes new manifests rather than editing the audited ones: the originals stay the record of
the positive pool, and the plan calls for the training manifests to be built as their own step.

Deterministic: the crops are drawn from a seeded generator, so the same command reproduces
byte-identical negatives.
"""
from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import cv2
import numpy as np

REPO = Path(__file__).resolve().parents[2]

POSITIVE_COLUMNS = None  # taken from the source manifest


def vary(crop: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Make each negative distinct even when its source is smaller than the crop window.

    prepare_background cannot crop a source that is at most the requested size - it upscales
    and returns the whole frame - so a small background yields N identical negatives. An
    audit with the duplicate check found exactly that: two groups of four byte-identical
    negatives from tbg_026.jpg (800x465) and tbg_029.jpg (640x481), 8 wasted samples in 100.

    Flips and a mild random zoom keep the frame shuttle-free (the property that matters) while
    guaranteeing variation. Driven by the caller's seeded generator, so still reproducible.
    """
    h, w = crop.shape[:2]
    out = crop
    if rng.random() < 0.5:
        out = out[:, ::-1]
    scale = float(rng.uniform(0.75, 1.0))
    if scale < 0.999:
        ch, cw = int(round(h * scale)), int(round(w * scale))
        y0 = int(rng.integers(0, max(1, h - ch + 1)))
        x0 = int(rng.integers(0, max(1, w - cw + 1)))
        out = cv2.resize(out[y0:y0 + ch, x0:x0 + cw], (w, h), interpolation=cv2.INTER_LINEAR)
    return np.ascontiguousarray(out)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", default=str(REPO))
    ap.add_argument("--per-background", type=int, default=4,
                    help="negative crops taken from each background image")
    ap.add_argument("--imgsz", type=int, default=960)
    ap.add_argument("--seed", type=int, default=20260915)
    args = ap.parse_args()

    repo = Path(args.repo).resolve()
    sys.path.insert(0, str(repo / "tools"))
    from shuttle_render import imread_unicode, imwrite_unicode, prepare_background

    base = repo / "outputs" / "shuttle_capability" / "train_data"
    total_neg = 0
    for split in ("train", "val"):
        src_manifest = base / ("manifest_" + split + ".csv")
        rows = list(csv.DictReader(src_manifest.open(encoding="utf-8")))
        columns = list(rows[0].keys())
        if "is_negative" not in columns:
            columns.append("is_negative")
        for r in rows:
            r["is_negative"] = "False"

        img_dir = base / split / "images"
        lab_dir = base / split / "labels"
        bgs = sorted(p for p in (base / ("bg_" + split)).glob("*")
                     if p.suffix.lower() in (".jpg", ".jpeg", ".png"))
        rng = np.random.default_rng(args.seed + (0 if split == "train" else 1))
        made = 0
        for bg_path in bgs:
            bg = imread_unicode(bg_path)
            if bg is None:
                print("  unreadable background", bg_path.name)
                continue
            for k in range(args.per_background):
                crop = vary(prepare_background(bg, args.imgsz, args.imgsz, rng), rng)
                name = "neg_{}_{:05d}".format(split, made)
                imwrite_unicode(img_dir / (name + ".jpg"), crop, quality=92)
                (lab_dir / (name + ".txt")).write_text("", encoding="utf-8")
                row = {c: "" for c in columns}
                row.update({
                    "file": name + ".jpg",
                    "split": split,
                    "source_type": "NEGATIVE",
                    "background": bg_path.name,
                    "imgsz": args.imgsz,
                    "is_negative": "True",
                })
                rows.append(row)
                made += 1
        out = base / ("manifest_" + split + "_synthetic.csv")
        with out.open("w", newline="", encoding="utf-8") as fh:
            w = csv.DictWriter(fh, fieldnames=columns)
            w.writeheader()
            for r in rows:
                w.writerow({c: r.get(c, "") for c in columns})
        pos = len(rows) - made
        total_neg += made
        print("{}: {} positives + {} negatives = {} rows  (ratio {:.1f}:1) -> {}".format(
            split, pos, made, len(rows), pos / max(made, 1), out.name))
    print("total negatives:", total_neg)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
