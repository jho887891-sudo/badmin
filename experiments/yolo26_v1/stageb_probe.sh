#!/usr/bin/env bash
RU=/home/T7/ojh/robot_sim/runs/shuttle_yolo26_v1/gate6B_lr0001_musgd_b16_w8_20260927-235730
cd /home/T7/ojh/robot_sim || exit 1
env_isaaclab/bin/python - "$RU" <<PY
import csv, json, os, subprocess, sys
ru = sys.argv[1]
rows = list(csv.DictReader(open(os.path.join(ru, "results.csv"), encoding="utf-8")))
best = None
for r in rows:
    try:
        e = int(r["epoch"]); m50 = float(r["metrics/mAP50(B)"]); m95 = float(r["metrics/mAP50-95(B)"])
    except Exception:
        continue
    f = 0.1 * m50 + 0.9 * m95
    if best is None or f > best["fitness"]:
        best = {"epoch": e, "map50": m50, "map5095": m95, "fitness": round(f, 5),
                "recall": float(r["metrics/recall(B)"]), "box": float(r["train/box_loss"])}
last = None
if rows:
    r = rows[-1]
    last = {"epoch": int(r["epoch"]), "box": float(r["train/box_loss"]),
            "map50": float(r["metrics/mAP50(B)"]), "map5095": float(r["metrics/mAP50-95(B)"]),
            "recall": float(r["metrics/recall(B)"])}
def tail(path, n=1):
    try:
        with open(path, encoding="utf-8") as f:
            return f.readlines()[-n:]
    except Exception:
        return []
mem = {}
for line in tail(os.path.join(ru, "mem_probe.jsonl")):
    try: mem = json.loads(line)
    except Exception: pass
lrs = []
for line in tail(os.path.join(ru, "lr_probe.jsonl")):
    try:
        d = json.loads(line)
        lrs = [g.get("lr") for g in d.get("groups", [])]
    except Exception: pass
alive = subprocess.run(["pgrep", "-f", "train_yolo26_v1.py"], capture_output=True, text=True).stdout.strip()
out = {"alive": bool(alive), "epochs": len(rows), "best": best, "last": last,
       "mem": {k: mem.get(k) for k in ("MemAvailable_GB", "SwapFree_GB", "swap_in_KBps", "swap_out_KBps", "cgroup_frac", "trainer_rss_GB")},
       "lr_groups": lrs, "best_pt_mtime": os.path.getmtime(os.path.join(ru, "weights", "best.pt")) if os.path.exists(os.path.join(ru, "weights", "best.pt")) else None}
print(json.dumps(out))
PY