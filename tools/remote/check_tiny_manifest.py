import csv, json, sys
from pathlib import Path
M, R = Path(sys.argv[1]), Path(sys.argv[2])
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
print("epochs", len(rows), "| selection", json.dumps(m["selection"], ensure_ascii=False))
print("argmax mAP50-95 epoch %d value %.5f | argmax fitness epoch %d value %.5f | equal %s"
      % (int(float(best_map["epoch"])), g(best_map, col), int(float(best_fit["epoch"])), g(best_fit, col),
         int(float(best_map["epoch"])) == int(float(best_fit["epoch"]))))
print("batch", m["batch"], m["batch_attempts_tried"], "| gpu_cap", round(m["gpu_cap"]["fraction"], 4))
print("diagnostic_only", m["diagnostic_only"], "eligible", m["eligible_for_final_report"], m["diagnostic_reasons"])
print("best", m["best_checkpoint"]["bytes"], m["best_checkpoint"]["sha256"])
print("curve every 5:", [(int(float(r["epoch"])), round(g(r, col), 5)) for r in rows if int(float(r["epoch"])) % 5 == 0])