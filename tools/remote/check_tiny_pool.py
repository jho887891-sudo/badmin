import csv
from collections import Counter
from pathlib import Path
rows = list(csv.DictReader(open("data/eth_real_hardneg_v2_train_manifest.csv", encoding="utf-8")))
pos = [r for r in rows if str(r["is_negative"]).lower() in ("false", "0")]
tiny = [r for r in pos if r["equiv_size_640"] not in ("", None) and float(r["equiv_size_640"]) < 8.0]
print("V2 train rows", len(rows), "| positives", len(pos), "| tiny(<8) unique positives", len(tiny))
print("unique sha of tiny", len(set(r["sha256"] for r in tiny)))
def bucket(v):
    v = float(v)
    return "<4" if v < 4 else ("4-6" if v < 6 else ("6-8" if v < 8 else ">=8"))
print("tiny by bucket", dict(Counter(bucket(r["equiv_size_640"]) for r in tiny)))
print("all positives by bucket", dict(Counter(bucket(r["equiv_size_640"]) for r in pos)))
print("gate: eligible >= 712 ?", len(tiny) >= 712, "| extra per unique image <= 1 possible ?", len(set(r["sha256"] for r in tiny)) >= 712)
prohibited = set()
for r in tiny:
    s = (r.get("source") or "").lower()
    if any(k in s for k in ("synthetic", "iphone", "d455", "roboflow")):
        prohibited.add(r.get("source"))
print("prohibited sources inside tiny", prohibited or "none")