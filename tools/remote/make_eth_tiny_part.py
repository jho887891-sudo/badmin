import csv
from pathlib import Path
man = {r["image"]: r for r in csv.DictReader(open("outputs/shuttle_capability/v1_dataset/v1_dataset_manifest.csv", encoding="utf-8"))}
rows = []
KEY = "eth_shuttle_detection"
for r in csv.DictReader(open("outputs/shuttle_capability/metrics/tiny_representability_v1.csv", encoding="utf-8")):
    if float(r["equiv_size_640"]) >= 8.0:
        continue
    m = man.get(r["image"]) or {}
    if m.get("source") != "eth_main":
        continue
    p = r["image"].replace("\\", "/")
    rel = p.split(KEY + "/", 1)[1] if KEY + "/" in p else ""
    rows.append({
        "basename": Path(p).name,
        "rel_path": rel,
        "location": m.get("location", ""), "difficulty": m.get("difficulty", ""),
        "bucket": r["bucket"], "equiv_size_640": r["equiv_size_640"], "net_px_1024": r["net_px_1024"],
        "v2_matched_op_1024": r["matched_op"], "v2_best_iou_weak_1024": r["best_iou_weak"],
        "v2_best_conf_weak_1024": r["best_conf_weak"],
    })
assert rows and all(x["rel_path"] for x in rows), "rel_path resolution failed"
rows.sort(key=lambda d: (d["bucket"], d["basename"]))
out = Path("tools/remote/eth_tiny_part.csv")
with out.open("w", newline="", encoding="utf-8") as fh:
    w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
    w.writeheader()
    for r in rows:
        w.writerow(r)
print("rows", len(rows), "->", out, out.stat().st_size, "B")
from collections import Counter
print("by rel_dir", dict(Counter(x["rel_path"].split("/images/")[0] for x in rows)))
print("hits@1024", sum(1 for x in rows if x["v2_matched_op_1024"] == "True"), "/", len(rows))
print("near-miss conf>=0.2 & iou>=0.5:",
      sum(1 for x in rows if x["v2_best_conf_weak_1024"] and float(x["v2_best_conf_weak_1024"]) >= 0.2 and float(x["v2_best_iou_weak_1024"]) >= 0.5))
