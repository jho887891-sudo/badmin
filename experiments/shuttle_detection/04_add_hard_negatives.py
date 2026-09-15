#!/usr/bin/env python3
"""04_add_hard_negatives.py - merge the HARD_NEGATIVE_POOL into a training manifest.

Spec 06 section 6: false positives -> classify the source -> collect NEW images of the same kind ->
retrain. The pool was acquired from Wikimedia Commons and selected by MEASURED model response: 31 of
the 36 candidates already fire on baseline_nearfield, producing 70 boxes with 6 above 0.5 confidence.
They are all shuttle-free, and they are NEW images - neither the frozen test pool nor the training
backgrounds were reused.

Images are copied under a distinct name prefix so no earlier round is overwritten, and the base
manifest is left untouched.
"""
from __future__ import annotations

import argparse
import csv
import shutil
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
PREFIX = "hardneg_train"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", default=str(REPO))
    ap.add_argument("--base-manifest", required=True)
    ap.add_argument("--out-manifest", required=True)
    args = ap.parse_args()

    repo = Path(args.repo).resolve()
    base_dir = repo / "outputs" / "shuttle_capability" / "train_data"
    hn_dir = repo / "outputs" / "shuttle_capability" / "hard_negatives"
    rows = list(csv.DictReader((base_dir / args.base_manifest).open(encoding="utf-8")))
    columns = list(rows[0].keys())
    meta = list(csv.DictReader((hn_dir / "hard_negative_metadata.csv").open(encoding="utf-8")))

    added = 0
    for i, m in enumerate(meta):
        src = hn_dir / "raw" / m["file"]
        if not src.is_file():
            print("  missing", src.name)
            continue
        name = "{}_{:05d}.jpg".format(PREFIX, i)
        shutil.copyfile(src, base_dir / "train" / "images" / name)
        row = {c: "" for c in columns}
        row.update({
            "file": name,
            "split": "train",
            "source_type": "NEGATIVE",
            "background": m["file"],
            "imgsz": "1280",
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
    print("  manifest now {} rows: {} positives, {} negatives".format(len(rows), pos, neg))
    print("  negatives are {:.0%} of the pool".format(neg / len(rows)))
    print("wrote", out.name)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())