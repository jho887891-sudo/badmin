#!/usr/bin/env python3
"""Audit the ETH RSL one-shot shuttle detection dataset (arXiv:2603.06691).

Reads the downloaded YOLO-format dataset and reports per-location / per-difficulty image counts,
image resolutions, label statistics and shuttlecock bbox size distributions both in original pixels
and at a 640x640 letterboxed network input. Writes CSV/JSON only (no images).
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import os
from collections import Counter, defaultdict

from PIL import Image

BUCKETS = [(0, 4), (4, 6), (6, 8), (8, 12), (12, 16), (16, 24), (24, 32), (32, 64), (64, 10 ** 9)]
BUCKET_LABELS = ["<4", "4-6", "6-8", "8-12", "12-16", "16-24", "24-32", "32-64", ">64"]
EXCLUDE = ["coco_train_easy", "coco_val_easy", "old"]
PAPER = {"cab_1": 2296, "cab_2": 3407, "glc_1": 2501, "glc_2": 1648, "ml_3": 1148, "ml_4": 1366,
         "ml_6": 1238, "ticino_1": 2572, "ticino_2": 2382, "uetlibergstrasse_1": 642,
         "uetlibergstrasse_2": 1310}


def bucket(v):
    for (lo, hi), lab in zip(BUCKETS, BUCKET_LABELS):
        if lo <= v < hi:
            return lab
    return BUCKET_LABELS[-1]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    root, out = args.root, args.out
    os.makedirs(out, exist_ok=True)

    per_loc_diff = Counter()
    per_loc_split = Counter()
    res_hist = Counter()
    size_buckets_640 = Counter()
    size_buckets_orig = Counter()
    class_ids = Counter()
    boxes_per_image = Counter()
    missing_labels = []
    empty_labels = 0
    n_images = 0
    n_boxes = 0
    eq640_values = []
    loc_frames_raw = defaultdict(int)

    for dirpath, dirnames, filenames in os.walk(root):
        rel = os.path.relpath(dirpath, root)
        parts = rel.split(os.sep)
        if len(parts) != 3 or parts[1] != "images":
            continue
        locdiff, underscore, split = parts
        if locdiff in EXCLUDE:
            continue
        loc = locdiff.rsplit("_", 1)[0]
        label_dir = os.path.join(root, locdiff, "labels", split)
        for fn in filenames:
            if not fn.lower().endswith((".jpg", ".jpeg", ".png")):
                continue
            n_images += 1
            per_loc_diff[locdiff] += 1
            per_loc_split[(loc, split)] += 1
            loc_frames_raw[loc] += 1
            try:
                with Image.open(os.path.join(dirpath, fn)) as im:
                    w, h = im.size
            except Exception:
                res_hist["UNREADABLE"] += 1
                continue
            res_hist["%dx%d" % (w, h)] += 1
            scale = 640.0 / max(w, h)
            lp = os.path.join(label_dir, os.path.splitext(fn)[0] + ".txt")
            if not os.path.exists(lp):
                missing_labels.append(os.path.join(rel, fn))
                boxes_per_image[0] += 1
                continue
            with open(lp, "r", encoding="utf-8", errors="replace") as f:
                rows = [ln.split() for ln in f.read().splitlines() if ln.strip()]
            if not rows:
                empty_labels += 1
            boxes_per_image[len(rows)] += 1
            for r in rows:
                if len(r) < 5:
                    continue
                class_ids[r[0]] += 1
                try:
                    bw = float(r[3]) * w
                    bh = float(r[4]) * h
                except ValueError:
                    continue
                eq = math.sqrt(max(bw, 0.0) * max(bh, 0.0))
                n_boxes += 1
                size_buckets_orig[bucket(eq)] += 1
                eq640 = eq * scale
                size_buckets_640[bucket(eq640)] += 1
                eq640_values.append(eq640)

    eq640_values.sort()
    quant = {}
    if eq640_values:
        for q in (1, 5, 25, 50, 75, 95, 99):
            quant["p%d" % q] = round(eq640_values[min(len(eq640_values) - 1, int(len(eq640_values) * q / 100))], 2)
        quant["mean"] = round(sum(eq640_values) / len(eq640_values), 2)
        quant["min"] = round(eq640_values[0], 2)
        quant["max"] = round(eq640_values[-1], 2)

    def pct(c, total):
        return {k: round(100.0 * c.get(k, 0) / total, 2) for k in BUCKET_LABELS} if total else {}

    summary = {
        "root": root,
        "images_total": n_images,
        "boxes_total": n_boxes,
        "empty_label_files": empty_labels,
        "images_without_label_file": len(missing_labels),
        "boxes_per_image": dict(sorted(boxes_per_image.items())),
        "class_ids": dict(class_ids),
        "resolutions": dict(res_hist.most_common()),
        "size_buckets_original_px": dict(size_buckets_orig),
        "size_buckets_at_640_input": dict(size_buckets_640),
        "size_pct_at_640_input": pct(size_buckets_640, n_boxes),
        "equivalent_size_at_640_px_quantiles": quant,
        "per_location_frames": dict(sorted(loc_frames_raw.items())),
        "paper_per_location_frames": PAPER,
        "per_location_delta_vs_paper": {k: loc_frames_raw.get(k, 0) - v for k, v in PAPER.items()},
        "paper_total_11_locations": sum(PAPER.values()),
        "measured_total_11_locations": sum(loc_frames_raw.get(k, 0) for k in PAPER),
    }
    with open(os.path.join(out, "eth_dataset_audit.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=1)

    with open(os.path.join(out, "eth_dataset_counts.csv"), "w", newline="", encoding="utf-8") as f:
        wr = csv.writer(f)
        wr.writerow(["group", "images"])
        for k in sorted(per_loc_diff):
            wr.writerow([k, per_loc_diff[k]])
        for key in sorted(per_loc_split):
            wr.writerow(["%s/%s" % (key[0], key[1]), per_loc_split[key]])

    with open(os.path.join(out, "eth_dataset_size_buckets_640.csv"), "w", newline="", encoding="utf-8") as f:
        wr = csv.writer(f)
        wr.writerow(["bucket_px", "boxes", "percent"])
        for k in BUCKET_LABELS:
            c = size_buckets_640.get(k, 0)
            wr.writerow([k, c, round(100.0 * c / n_boxes, 2) if n_boxes else 0.0])

    with open(os.path.join(out, "eth_dataset_resolution.csv"), "w", newline="", encoding="utf-8") as f:
        wr = csv.writer(f)
        wr.writerow(["resolution", "images"])
        for k, v in res_hist.most_common():
            wr.writerow([k, v])

    if missing_labels:
        with open(os.path.join(out, "eth_dataset_missing_labels.txt"), "w", encoding="utf-8") as f:
            f.write(chr(10).join(missing_labels[:5000]))

    keys = ["images_total", "boxes_total", "empty_label_files", "images_without_label_file",
            "resolutions", "size_pct_at_640_input", "per_location_frames", "per_location_delta_vs_paper"]
    print(json.dumps({k: summary[k] for k in keys}, ensure_ascii=False, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())