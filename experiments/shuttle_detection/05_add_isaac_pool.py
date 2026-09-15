#!/usr/bin/env python3
"""Append the Isaac-rendered pool to a training manifest.

These samples use the SAME background crops, the same composite(), the same noise and the same JPEG
quality as the numpy pool; only the renderer differs. They are ADDED rather than substituted, so the
model sees both appearances and the retrain tests whether renderer fidelity moves the real-photograph
recall - the one axis still unexplained.
"""
from __future__ import annotations

import argparse
import csv
import math
import sys
from pathlib import Path

# Run from anywhere: the script may be copied to /tmp on the training host, where a fixed
# parents[2] walk would raise IndexError before argparse ever runs.
_HERE = Path(__file__).resolve()
REPO = _HERE.parents[2] if len(_HERE.parents) > 2 else Path("/home/T7/dgut/robot_sim")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", default=str(REPO))
    ap.add_argument("--base-manifest", required=True)
    ap.add_argument("--out-manifest", required=True)
    args = ap.parse_args()

    repo = Path(args.repo).resolve()
    base_dir = repo / "outputs" / "shuttle_capability" / "train_data"
    pool = repo / "outputs" / "shuttle_capability" / "isaac_pool"
    rows = list(csv.DictReader((base_dir / args.base_manifest).open(encoding="utf-8")))
    columns = list(rows[0].keys())
    src = list(csv.DictReader((pool / "composite_manifest.csv").open(encoding="utf-8")))

    added = 0
    for r in src:
        name = r["name"] + ".jpg"
        if not (base_dir / "train" / "images" / name).is_file():
            continue
        if not (base_dir / "train" / "labels" / (r["name"] + ".txt")).is_file():
            continue
        row = {c: "" for c in columns}
        row.update({
            "file": name,
            "split": "train",
            "source_type": "SYNTHETIC_3D",
            "background": r["background"],
            "target_px": r["equiv"],
            "equiv_size_px": r["equiv"],
            "imgsz": "960",
            "is_negative": "False",
        })
        rows.append(row)
        added += 1

    out = base_dir / args.out_manifest
    with out.open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=columns)
        w.writeheader()
        for r in rows:
            w.writerow({c: r.get(c, "") for c in columns})
    pos = [r for r in rows if r.get("is_negative") == "False"]
    neg = [r for r in rows if r.get("is_negative") == "True"]
    isaac = [r for r in pos if str(r.get("file", "")).startswith("isaac_train_")]
    sizes = [float(r["equiv_size_px"]) for r in pos if r.get("equiv_size_px")]
    print("added {} Isaac-rendered positives".format(added))
    print("  manifest now {} rows: {} positives ({} of them Isaac), {} negatives".format(
          len(rows), len(pos), len(isaac), len(neg)))
    if sizes:
        print("  positive size range {:.2f} .. {:.2f} px".format(min(sizes), max(sizes)))
    print("wrote", out.name)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())