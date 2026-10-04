#!/usr/bin/env python3
"""Render the top-K highest-confidence false-positive images from hard_negative_predictions.csv.

Only images that actually contain at least one FP at or above the lowest evaluated threshold are
ranked, so the saved set is real evidence (not images with zero detections).
"""
from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from pathlib import Path


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", default=None)
    ap.add_argument("--predictions", default="outputs/shuttle_capability/metrics/hard_negative_predictions.csv")
    ap.add_argument("--out-dir", default="outputs/shuttle_capability/hard_negative_eval")
    ap.add_argument("--min-confidence", type=float, default=0.25)
    ap.add_argument("--topk", type=int, default=20)
    args = ap.parse_args()
    repo = Path(args.repo).resolve() if args.repo else Path(__file__).resolve().parents[1]
    rows = list(csv.DictReader(open(repo / args.predictions, encoding="utf-8")))
    groups = defaultdict(list)
    for r in rows:
        if float(r["confidence"]) < args.min_confidence:
            continue
        groups[(r["checkpoint"], r["set"], r["image_path"])].append(r)
    from PIL import Image, ImageDraw
    written = 0
    per_set = defaultdict(int)
    # rank inside each (checkpoint, set) by the image max confidence, then take the global top-K per checkpoint
    by_ckpt = defaultdict(list)
    for (ck, st, path), rs in groups.items():
        mx = max(float(x["confidence"]) for x in rs)
        by_ckpt[ck].append((mx, st, path, rs))
    for ck, items in by_ckpt.items():
        items.sort(key=lambda t: -t[0])
        for rank, (mx, st, path, rs) in enumerate(items[: args.topk], 1):
            im = Image.open(path).convert("RGB")
            d = ImageDraw.Draw(im)
            for x in rs:
                d.rectangle([float(x["x1"]), float(x["y1"]), float(x["x2"]), float(x["y2"])], outline=(255, 0, 0), width=3)
                d.text((float(x["x1"]) + 4, max(0.0, float(x["y1"]) - 14)), x["confidence"], fill=(255, 0, 0))
            d.text((4, 4), "%s | %s | FP#%d max_conf=%s" % (ck, st, rank, mx), fill=(255, 255, 0))
            dst = repo / args.out_dir / ck / "top20_fp" / ("%s_%s_conf%.3f_%s.jpg" % (ck, st, mx, Path(path).stem))
            dst.parent.mkdir(parents=True, exist_ok=True)
            im.save(dst)
            written += 1
            per_set[st] += 1
    print("written:", written)
    for k, v in sorted(per_set.items()):
        print("   ", k, v)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())