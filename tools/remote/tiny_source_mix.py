import csv
from collections import Counter
man = {r["image"]: r for r in csv.DictReader(open("outputs/shuttle_capability/v1_dataset/v1_dataset_manifest.csv", encoding="utf-8"))}
rows = [r for r in csv.DictReader(open("outputs/shuttle_capability/metrics/tiny_representability_v1.csv", encoding="utf-8")) if float(r["equiv_size_640"]) < 8.0]
c = Counter()
paths = Counter()
for r in rows:
    m = man.get(r["image"]) or {}
    c[m.get("source", "?")] += 1
    paths[m.get("source", "?")+"/"+str(m.get("location", "?"))] += 1
print("105 tiny GT by source:", dict(c))
print("by source/location:", dict(paths))