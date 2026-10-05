import csv, json, sys
from pathlib import Path
repo = Path(".")
v1neg = repo / "outputs/shuttle_capability/metrics/eth_only_v1_frozen_test_negatives.csv"
v1tbl = repo / "outputs/shuttle_capability/metrics/eth_vs_ours_eth_only_v1.csv"
out = repo / "_scratch_eth_only_v1_eval/task5_dryrun"
out.mkdir(parents=True, exist_ok=True)
rows = list(csv.DictReader(v1neg.open(encoding="utf-8")))
# stub V2: remove one FP from backgrounds and one from raw at the operating threshold (6 -> 4 pooled)
drop = {"real_images/backgrounds": 1, "real_images/raw": 1}
stub_rows = []
for r in rows:
    r = dict(r)
    if r["threshold"] == "0.25" and r["set"] in drop:
        r["total_FP"] = str(int(float(r["total_FP"])) - drop[r["set"]])
        n = int(float(r["images"]))
        r["FP_per_image"] = str(float(r["total_FP"]) / n)
        r["checkpoint"] = "stub_v2"
    stub_rows.append(r)
with (out / "stub_v2_negatives.csv").open("w", newline="", encoding="utf-8") as fh:
    w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
    w.writeheader()
    for r in stub_rows:
        w.writerow(r)
# stub V2 accuracy table: V1 val|eth_unseen rows renamed, recall -0.01, mAP50-95 -0.005
tbl = list(csv.DictReader(v1tbl.open(encoding="utf-8")))
keep = [r for r in tbl if r["model"] == "eth_only_v1_best"]
stub_tbl = []
for r in keep:
    r = dict(r)
    r["model"] = "eth_real_hardneg_v2_best"
    for col, d in (("Recall", -0.01), ("mAP50-95", -0.005), ("AP50", -0.005)):
        try:
            r[col] = "%.6f" % (float(r[col]) + d)
        except (TypeError, ValueError):
            pass
    stub_tbl.append(r)
with (out / "stub_v2_table.csv").open("w", newline="", encoding="utf-8") as fh:
    w = csv.DictWriter(fh, fieldnames=list(tbl[0].keys()))
    w.writeheader()
    for r in stub_tbl:
        w.writerow(r)
print("stub inputs written:", len(stub_rows), "neg rows,", len(stub_tbl), "table rows")