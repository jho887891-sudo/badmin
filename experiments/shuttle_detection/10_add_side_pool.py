#!/usr/bin/env python3
"""Append the side-view Isaac pool to a training manifest."""
from __future__ import annotations
import argparse, csv, sys
from pathlib import Path

_H = Path(__file__).resolve()
REPO = _H.parents[2] if len(_H.parents) > 2 else Path("/home/T7/dgut/robot_sim")

def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", default=str(REPO))
    ap.add_argument("--base-manifest", required=True)
    ap.add_argument("--out-manifest", required=True)
    ap.add_argument("--pool", default="isaac_pool_side")
    args = ap.parse_args()
    repo = Path(args.repo).resolve()
    base_dir = repo / "outputs" / "shuttle_capability" / "train_data"
    pool = repo / "outputs" / "shuttle_capability" / args.pool
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
        row.update({"file": name, "split": "train", "source_type": "SYNTHETIC_3D",
                    "background": r["background"], "target_px": r["equiv"], "equiv_size_px": r["equiv"],
                    "imgsz": "960", "is_negative": "False"})
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
    print("added {} side-view-inclusive positives".format(added))
    print("  manifest now {} rows: {} positives, {} negatives".format(len(rows), pos, neg))
    print("wrote", out.name)
    return 0

if __name__ == "__main__":
    raise SystemExit(main())