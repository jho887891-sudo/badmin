#!/usr/bin/env python3
"""FP confidence distribution + train-membership audit for the hard-negative sets."""
from __future__ import annotations

import argparse
import csv
from collections import Counter, defaultdict
from pathlib import Path

BINS = [(0.01, 0.05), (0.05, 0.10), (0.10, 0.25), (0.25, 0.50), (0.50, 0.75), (0.75, 1.0001)]


def bin_of(c: float):
    for lo, hi in BINS:
        if lo <= c < hi:
            return "%.2f-%.2f" % (lo, hi)
    return "other"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", default=None)
    args = ap.parse_args()
    repo = Path(args.repo).resolve() if args.repo else Path(__file__).resolve().parents[1]
    pred = repo / "outputs/shuttle_capability/metrics/hard_negative_predictions.csv"
    rows = list(csv.DictReader(pred.open(encoding="utf-8")))
    hist = defaultdict(Counter)
    seen = defaultdict(int)
    for r in rows:
        key = (r["checkpoint"], r["set"])
        hist[key][bin_of(float(r["confidence"]))] += 1
        seen[key] += 1
    order = [b for b, _ in [("0.01-0.05", 0), ("0.05-0.10", 0), ("0.10-0.25", 0), ("0.25-0.50", 0), ("0.50-0.75", 0), ("0.75-1.00", 0)]]
    outp = repo / "outputs/shuttle_capability/metrics/hard_negative_fp_confidence_hist.csv"
    with outp.open("w", encoding="utf-8", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["checkpoint", "set", "total_FP_conf_ge_0.01"] + order + ["max_conf"])
        for key in sorted(hist):
            h = hist[key]
            mx = max(float(r["confidence"]) for r in rows if (r["checkpoint"], r["set"]) == key)
            w.writerow([key[0], key[1], seen[key]] + [h.get(b, 0) for b in order] + [round(mx, 6)])
    print("wrote", outp)
    for key in sorted(hist):
        h = hist[key]
        print(key, "n=%d" % seen[key], {b: h.get(b, 0) for b in order})

    # --- train membership audit ---
    man = repo / "outputs/shuttle_capability/v1_dataset/v1_dataset_manifest.csv"
    mrows = list(csv.DictReader(man.open(encoding="utf-8")))
    print("manifest rows:", len(mrows), "columns:", list(mrows[0].keys())[:12])
    srcs = Counter(r["source"] for r in mrows)
    for s in sorted(srcs):
        if "neg" in s or "hard" in s:
            splits = Counter(r["split"] for r in mrows if r["source"] == s)
            print("  source=%s total=%d splits=%s" % (s, srcs[s], dict(splits)))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())