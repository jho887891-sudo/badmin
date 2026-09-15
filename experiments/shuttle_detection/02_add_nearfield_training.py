#!/usr/bin/env python3
"""02_add_nearfield_training.py - FAILURE_RETURN_POOL for the large-target gap.

Why: the near-field probe (NEAR_FIELD_GAP.md) measured recall 0.000 for synthetic shuttles at 64,
128, 256, 512 and 800 px rendered in the model's OWN appearance. So the real-image failure at
838-1578 px is a SIZE-COVERAGE gap, not an appearance gap: the pool tops out at 32.86 px and the
model emits no box, or a tiny 6-25 px box at IoU ~0.000, when the target is large.

Spec 06 section 5 forbids training on frozen test samples, so this generates NEW samples into
FAILURE_RETURN_POOL on the TRAINING backgrounds (bg_train), which are disjoint from the frozen test
backgrounds. It writes a NEW manifest and leaves manifest_train_synthetic.csv intact so the v1
record of what the baseline was trained on stays auditable. Seeded, so it reproduces.
"""
from __future__ import annotations

import argparse
import csv
import math
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", default=str(REPO))
    ap.add_argument("--n", type=int, default=200)
    ap.add_argument("--min-px", type=float, default=40.0)
    # 800 px was the first choice and was abandoned on measured cost: one 800 px render takes ~52 s
    # and calibration multiplies that (29 s at fx=700, 99 s at fx=5600 - a longer focal length does
    # NOT help, because the cost is the rasteriser itself, not near-plane perspective). At 512 px the
    # cost is still bounded while the trained range still extends ~16x beyond the old 32.86 px ceiling.
    # Residual, recorded rather than hidden: the real verified positives run to 1578 px.
    ap.add_argument("--max-px", type=float, default=512.0)
    ap.add_argument("--imgsz", type=int, default=960)
    ap.add_argument("--seed", type=int, default=20260915)
    # Round 2 writes its own manifest so round 1's record stays intact and each retrain can name the
    # exact pool it used.
    ap.add_argument("--base-manifest", default="manifest_train_nearfield.csv")
    ap.add_argument("--out-manifest", default="manifest_train_nearfield2.csv")
    # Image names must not collide across rounds: round 1 wrote nf_train_00000..., and running round 2
    # without a distinct prefix OVERWROTE round 1's first frames while round 1's manifest still
    # referenced them - silent data corruption. Each round now names its own frames.
    ap.add_argument("--name-prefix", default="nf_train")
    args = ap.parse_args()

    repo = Path(args.repo).resolve()
    sys.path.insert(0, str(repo / "tools"))
    from shuttle_render import imread_unicode, imwrite_unicode, load_shuttle_parts, render_sample

    base = repo / "outputs" / "shuttle_capability" / "train_data"
    src = base / args.base_manifest
    rows = list(csv.DictReader(src.open(encoding="utf-8")))
    columns = list(rows[0].keys())
    bgs = sorted(p for p in (base / "bg_train").glob("*") if p.suffix.lower() in (".jpg", ".jpeg", ".png"))
    if not bgs:
        print("no bg_train backgrounds")
        return 2

    parts = load_shuttle_parts()
    rng = np.random.default_rng(args.seed)
    targets = np.exp(rng.uniform(math.log(args.min_px), math.log(args.max_px), args.n))
    made = 0
    added_rows = []
    for i, target in enumerate(targets):
        bg_path = bgs[i % len(bgs)]
        bg = imread_unicode(bg_path)
        if bg is None:
            continue
        # Anti-aliasing matters at 2-32 px, not at 128-800 px, and supersampling dominates the cost:
        # ss=3 allocates a 2880x2880 canvas per render. One pixel of edge on an 800 px object is
        # 0.1% of its size, so ss=1 is used above 128 px and the choice is recorded per row.
        ss = 3 if target < 128.0 else 1
        name = "{}_{:05d}".format(args.name_prefix, i)
        try:
            sample = render_sample(parts, bg, target_px=float(target), seed=args.seed * 7919 + i,
                                   split="train", name=name, width=args.imgsz, height=args.imgsz,
                                   supersample=ss, background_name=bg_path.name)
        except ValueError as exc:
            print("  skipped", name, exc)
            continue
        imwrite_unicode(base / "train" / "images" / (name + ".jpg"), sample.image, quality=92)
        (base / "train" / "labels" / (name + ".txt")).write_text(sample.label, encoding="utf-8")
        row = {c: "" for c in columns}
        row.update(sample.record)
        row["is_negative"] = "False"
        rows.append(row)
        added_rows.append(row)
        made += 1

    out = base / args.out_manifest
    with out.open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=columns)
        w.writeheader()
        for r in rows:
            w.writerow({c: r.get(c, "") for c in columns})
    pos = [float(r["equiv_size_px"]) for r in rows if r.get("is_negative") == "False" and r.get("equiv_size_px")]
    added = [float(r["equiv_size_px"]) for r in added_rows if r.get("equiv_size_px")]
    print("added {} near-field samples on {} training backgrounds".format(made, len(bgs)))
    if added:
        print("  added sizes: min {:.1f}  median {:.1f}  max {:.1f} px".format(min(added), float(np.median(added)), max(added)))
    print("  positives now {} | size range {:.2f} .. {:.2f} px".format(len(pos), min(pos), max(pos)))
    print("wrote", out.name)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())