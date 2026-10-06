import csv
from collections import Counter
from pathlib import Path
man = {r["image"]: r for r in csv.DictReader(open("outputs/shuttle_capability/v1_dataset/v1_dataset_manifest.csv", encoding="utf-8"))}
rows = [r for r in csv.DictReader(open("outputs/shuttle_capability/metrics/tiny_representability_v1.csv", encoding="utf-8"))
        if float(r["equiv_size_640"]) < 8.0]
cross = Counter()
for r in rows:
    s = (man.get(r["image"]) or {}).get("source", "?")
    cross[(s, r["bucket"])] += 1
print("tiny GT by (source, bucket):", dict(sorted(cross.items())))
images = {}
for r in rows:
    m = man.get(r["image"]) or {}
    images.setdefault(m.get("source", "?"), set()).add(r["image"])
for src, imgs in images.items():
    total = 0
    missing = 0
    for p in imgs:
        fp = Path(p)
        if fp.is_file():
            total += fp.stat().st_size
        else:
            missing += 1
    print("%-10s images=%-4d local_MB=%.1f missing=%d" % (src, len(imgs), total/1e6, missing))
sub = [r for r in rows if float(r["net_px_1024"]) < 8.0]
print("strict sub-stride images by source:", dict(Counter((man.get(r["image"]) or {}).get("source","?") for r in sub)))
print("sample synthetic path:", next(iter(images.get("synthetic", [])))[:120])