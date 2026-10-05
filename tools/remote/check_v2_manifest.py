import csv, json, sys
from pathlib import Path
M = Path(sys.argv[1]); R = Path(sys.argv[2])
m = json.loads(M.read_text(encoding="utf-8"))
rows = list(csv.DictReader(R.open(encoding="utf-8")))
col = m["selection"]["column"]
def g(r, k):
    for f in r:
        if f.strip() == k:
            return float(r[f])
    raise KeyError(k)
best_map = max(rows, key=lambda r: g(r, col))
best_fit = max(rows, key=lambda r: 0.1 * g(r, "metrics/mAP50(B)") + 0.9 * g(r, col))
print("epochs_recorded", len(rows), "| selection", json.dumps(m["selection"], ensure_ascii=False))
print("argmax mAP50-95 -> epoch %d value %.5f" % (int(float(best_map["epoch"])), g(best_map, col)))
print("argmax fitness  -> epoch %d value %.5f" % (int(float(best_fit["epoch"])), g(best_fit, col)))
print("fitness_argmax_equals_map_argmax", int(float(best_map["epoch"])) == int(float(best_fit["epoch"])))
print("batch", m["batch"], "| tried", m["batch_attempts_tried"], "| gpu_cap", m["gpu_cap"]["fraction"])
print("diagnostic_only", m["diagnostic_only"], "| eligible", m["eligible_for_final_report"], "| reasons", m["diagnostic_reasons"])
print("best_checkpoint", m["best_checkpoint"]["bytes"], m["best_checkpoint"]["sha256"])
print("val curve every 5:", [(int(float(r["epoch"])), round(g(r, col), 5)) for r in rows if int(float(r["epoch"])) % 5 == 0])
print("last 3:", [(int(float(r["epoch"])), round(g(r, col), 5)) for r in rows[-3:]])