import csv, sys, time
from pathlib import Path
sys.path.insert(0, "tools")
import eval_yolo26_v1 as ev
t0=time.time()
rows=list(csv.DictReader(open("outputs/shuttle_capability/v1_dataset/v1_dataset_manifest.csv", encoding="utf-8")))
print("manifest rows", len(rows), "read_s %.2f" % (time.time()-t0), "cols", list(rows[0].keys())[:10])
val=[r for r in rows if str(r.get("split"))=="val"]
print("val rows", len(val), "with label", sum(1 for r in val if (r.get("label") or "").strip()))
print("sample", {k: val[0].get(k) for k in ("image","label","split")})
t0=time.time()
n=0
for r in val[:20]:
    lab=(r.get("label") or "").strip()
    if lab:
        ev.load_gt_boxes(r["image"], lab); n+=1
print("20 label decodes: n=%d s=%.2f" % (n, time.time()-t0))
import json
d=json.load(open("_scratch_localization_audit/loc_eth_only_v1_best_val.json", encoding="utf-8"))
dets=d["dets"]
print("dets", len(dets), "first key", list(dets)[0])
print("first manifest image", val[0]["image"])
print("key match", val[0]["image"] in dets)
