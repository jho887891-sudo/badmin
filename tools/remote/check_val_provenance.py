import csv
from collections import Counter
rows = list(csv.DictReader(open("data/eth_only_v1_val_manifest.csv", encoding="utf-8")))
print("val rows", len(rows))
print("split subdir", Counter(r["split"] for r in rows))
print("set_dir", Counter(r["set_dir"] for r in rows))
print("role", Counter(r["role"] for r in rows))
print("in_official_training", Counter(r["in_official_training"] for r in rows))
print("sample path", rows[0]["image"])