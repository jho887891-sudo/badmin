#!/usr/bin/env python3
"""Level-2 section 3.4: difficult-example oversampling.

Spec 07 section 3.4 lists the training-organisation comparisons Level 2 requires, and this is the one that
does not need real data: repeat the hard negatives in the training list so their weight in the loss matches
their importance to the remaining defect, which is false positives on cluttered scenes.

The instances are duplicated rather than the loss being reweighted, because Ultralytics takes an image
list and duplicating a row is the mechanism its sampler actually honours.
"""
from __future__ import annotations
import argparse, csv
from pathlib import Path

_H = Path(__file__).resolve()
REPO = _H.parents[2] if len(_H.parents) > 2 else Path("/home/T7/dgut/robot_sim")

def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", default=str(REPO))
    ap.add_argument("--base-manifest", required=True)
    ap.add_argument("--out-manifest", required=True)
    ap.add_argument("--factor", type=int, default=3)
    args = ap.parse_args()
    repo = Path(args.repo).resolve()
    base_dir = repo / "outputs" / "shuttle_capability" / "train_data"
    rows = list(csv.DictReader((base_dir / args.base_manifest).open(encoding="utf-8")))
    columns = list(rows[0].keys())
    out_rows = []
    dup = 0
    for r in rows:
        out_rows.append(r)
        f = str(r.get("file", ""))
        if r.get("is_negative") == "True" and f.startswith(("hardneg_train_", "hardneg2_train_")):
            for _ in range(args.factor - 1):
                out_rows.append(dict(r))
                dup += 1
    out = base_dir / args.out_manifest
    with out.open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=columns)
        w.writeheader()
        for r in out_rows:
            w.writerow({c: r.get(c, "") for c in columns})
    neg = sum(1 for r in out_rows if r.get("is_negative") == "True")
    pos = sum(1 for r in out_rows if r.get("is_negative") == "False")
    print("oversampled hard negatives x{}: {} duplicated rows".format(args.factor, dup))
    print("  list now {} entries: {} positives, {} negative entries ({:.1%} negatives)".format(
        len(out_rows), pos, neg, neg / len(out_rows)))
    print("  (the images themselves are unchanged; only their frequency in the training list differs)")
    print("wrote", out.name)
    return 0

if __name__ == "__main__":
    raise SystemExit(main())