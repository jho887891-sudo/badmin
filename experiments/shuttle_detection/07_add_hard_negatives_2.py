#!/usr/bin/env python3
"""Add the SECOND hard-negative batch: measure the model, then merge with empty labels."""
from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import cv2

_HERE = Path(__file__).resolve()
REPO = _HERE.parents[2] if len(_HERE.parents) > 2 else Path("/home/T7/dgut/robot_sim")
PREFIX = "hardneg2_train"
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
    hn_dir = repo / "outputs" / "shuttle_capability" / "hard_negatives2"
    excluded = set()
    ex_file = hn_dir / "excluded.txt"
    if ex_file.is_file():
        excluded = {ln.strip() for ln in ex_file.read_text(encoding="utf-8").splitlines() if ln.strip()}

    rows = list(csv.DictReader((base_dir / args.base_manifest).open(encoding="utf-8")))
    columns = list(rows[0].keys())
    meta = [m for m in csv.DictReader((hn_dir / "hard_negative_metadata.csv").open(encoding="utf-8"))
            if m["file"] not in excluded]
    print("candidates after the leak guard: {} of {} (excluded {})".format(
        len(meta), len(meta) + len(excluded), len(excluded)))

    added = 0
    for i, m in enumerate(meta):
        src = hn_dir / "raw" / m["file"]
        if not src.is_file():
            continue
        img = imread_unicode(src)
        if img is None:
            continue
        h, w = img.shape[:2]
        side = min(h, w)
        y0, x0 = (h - side) // 2, (w - side) // 2
        sq = img[y0:y0 + side, x0:x0 + side]
        small = cv2.resize(sq, (SIZE, SIZE), interpolation=cv2.INTER_AREA)
        name = "{}_{:05d}.jpg".format(PREFIX, i)
        imwrite_unicode(base_dir / "train" / "images" / name, small, quality=92)
        (base_dir / "train" / "labels" / ("{}_{:05d}.txt".format(PREFIX, i))).write_text("", encoding="utf-8")
        row = {c: "" for c in columns}
        row.update({"file": name, "split": "train", "source_type": "NEGATIVE",
                    "background": m["file"], "imgsz": str(SIZE), "is_negative": "True"})
        rows.append(row)
        added += 1

    out = base_dir / args.out_manifest
    with out.open("w", newline="", encoding="utf-8") as fh:
        wtr = csv.DictWriter(fh, fieldnames=columns)
        wtr.writeheader()
        for r in rows:
            wtr.writerow({c: r.get(c, "") for c in columns})
    pos = sum(1 for r in rows if r.get("is_negative") == "False")
    neg = sum(1 for r in rows if r.get("is_negative") == "True")
    print("added {} hard negatives from batch 2".format(added))
    print("  manifest now {} rows: {} positives, {} negatives ({:.1%})".format(
        len(rows), pos, neg, neg / len(rows)))
    print("wrote", out.name)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())