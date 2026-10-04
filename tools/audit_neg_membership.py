#!/usr/bin/env python3
"""Evidence: full source/split table of the V1 manifest + excluded hard-negative membership check."""
from __future__ import annotations
import csv, re
from collections import Counter, defaultdict
from pathlib import Path

repo = Path(__file__).resolve().parents[1]
rows = list(csv.DictReader((repo / "outputs/shuttle_capability/v1_dataset/v1_dataset_manifest.csv").open(encoding="utf-8")))
tbl = defaultdict(Counter)
for r in rows:
    tbl[r["source"]][r["split"]] += 1
print("source                          train    val   total")
for s in sorted(tbl):
    t, v = tbl[s]["train"], tbl[s]["val"]
    print("%-30s %6d %6d %7d" % (s, t, v, t + v))
ex = repo / "outputs/shuttle_capability/hard_negatives2/excluded.txt"
names = [l.strip() for l in ex.read_text(encoding="utf-8").splitlines() if l.strip()] if ex.exists() else []
print("excluded.txt entries:", len(names))
trained = 0
for n in names:
    stem = Path(n).name
    hit = [r for r in rows if Path(r["image"]).name == stem]
    for h in hit:
        print("  ", stem, "-> manifest source=%s split=%s location=%s" % (h["source"], h["split"], h["location"]))
        trained += 1
    if not hit:
        print("  ", stem, "-> NOT in manifest")
print("excluded-but-trained:", trained)