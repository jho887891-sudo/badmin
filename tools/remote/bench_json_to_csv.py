#!/usr/bin/env python3
"""Flatten a remote_bench24.py JSON into the committed gpu_bench_a6000_24g.csv schema.

Verified by round-trip: converting the older gpu_bench_a6000_24g.json must reproduce the committed
gpu_bench_a6000_24g.csv (column order and values), which proves the mapping used here is the one that
produced the published table.
"""
from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

HEADER = ["section", "model", "point", "imgsz", "batch", "median_ms", "min_ms", "p90_ms", "fps_median",
          "images_per_s", "vram_peak_MB", "oom"]


def rows_from(data: dict):
    rows = []
    for tag, points in (data.get("res_sweep") or {}).items():
        for res in sorted(points, key=lambda s: int(s)):
            p = points[res]
            rows.append(["resolution", tag, res, res, 1, p.get("median_ms"), p.get("min_ms"), p.get("p90_ms"),
                         p.get("fps_median"), "", p.get("vram_peak_MB"), bool(p.get("oom", False))])
    for tag, points in (data.get("batch_sweep") or {}).items():
        for b in sorted(points, key=lambda s: int(s)):
            p = points[b]
            ms = p.get("ms_per_iter_median")
            rows.append(["batch", tag, b, data.get("batch_imgsz", 1024), b, ms,
                         p.get("ms_per_iter_min"), "", "",   # batch rows leave p90/fps empty (fps is per iteration)
                         p.get("images_per_s"), p.get("vram_peak_MB"), bool(p.get("oom", False))])
    return rows


def main(argv):
    src, dst = Path(argv[1]), Path(argv[2])
    data = json.loads(src.read_text(encoding="utf-8"))
    rows = rows_from(data)
    with dst.open("w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(HEADER)
        for r in rows:
            w.writerow(["" if v is None else v for v in r])
    print("wrote %s (%d rows)" % (dst, len(rows)))
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
