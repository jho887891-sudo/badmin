import csv
from collections import Counter
man = {r["image"]: r for r in csv.DictReader(open("outputs/shuttle_capability/v1_dataset/v1_dataset_manifest.csv", encoding="utf-8"))}
rows = [r for r in csv.DictReader(open("outputs/shuttle_capability/metrics/tiny_representability_v1.csv", encoding="utf-8"))
        if float(r["equiv_size_640"]) < 8.0]
sub = [r for r in rows if float(r["net_px_1024"]) < 8.0]
def mix(sel, label):
    c = Counter()
    hits = Counter()
    for r in sel:
        s = (man.get(r["image"]) or {}).get("source", "?")
        c[s] += 1
        if r["matched_op"] == "True":
            hits[s] += 1
    print("%s: n=%d by_source=%s hits=%s" % (label, len(sel), dict(c), dict(hits)))
mix(rows, "all tiny GT")
mix(sub, "strict sub-stride GT")