#!/usr/bin/env python3
"""Set a declared repeat count on the hard negatives, now that the pipeline honours one.

The first attempt duplicated manifest ROWS and had no effect, because _lines_from_manifest now deduplicates
at row level to stop an image being silently trained twice. The repeat COLUMN is the explicit mechanism, and
this writes it. It is spec 07 section 3.4 difficult-example oversampling, aimed at the remaining defect:
false positives on cluttered scenes.
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
    if "repeat" not in columns:
        columns.append("repeat")
    target = 0
    out_rows = []
    for r in rows:
        row = dict(r)
        f = str(r.get("file", ""))
        if r.get("is_negative") == "True" and f.startswith(("hardneg_train_", "hardneg2_train_")):
            row["repeat"] = str(args.factor)
            target += 1
        else:
            row.setdefault("repeat", "")
        out_rows.append(row)
    out = base_dir / args.out_manifest
    with out.open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=columns)
        w.writeheader()
        for r in out_rows:
            w.writerow({c: r.get(c, "") for c in columns})
    total = sum(int(r["repeat"] or 1) for r in out_rows)
    neg_entries = sum(int(r["repeat"] or 1) for r in out_rows if r.get("is_negative") == "True")
    print("declared repeat={} on {} hard negatives".format(args.factor, target))
    print("  manifest rows {} but the training list will hold {} entries ({} negative, {:.1%})".format(
        len(out_rows), total, neg_entries, neg_entries / total))
    print("wrote", out.name)
    return 0

if __name__ == "__main__":
    raise SystemExit(main())