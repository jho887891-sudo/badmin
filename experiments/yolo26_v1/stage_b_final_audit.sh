#!/usr/bin/env bash
RU=/home/T7/ojh/robot_sim/runs/shuttle_yolo26_v1/gate6B_lr0001_musgd_b16_w8_20260927-235730
PY=/home/T7/ojh/robot_sim/env_isaaclab/bin/python
echo "== run dir =="
ls -l --time-style=+%Y-%m-%dT%H:%M:%S $RU
echo "== weights =="
ls -l --time-style=+%Y-%m-%dT%H:%M:%S $RU/weights
sha256sum $RU/weights/best.pt $RU/weights/last.pt
echo "== results rows =="
wc -l $RU/results.csv
echo "== results scan =="
awk -F, 'NR==1{next} {if (min50==""||$8+0<min50){min50=$8+0;mine=$1} if (min95==""||$9+0<min95){min95=$9+0} if ($6+0>maxp)maxp=$6+0; if ($7+0>maxr){maxr=$7+0;maxre=$1} if ($8+0>max50){max50=$8+0;max50e=$1} if ($9+0>max95){max95=$9+0;max95e=$1} if ($4+0!=$4+0) nan++} END{printf "epochs=%d min_mAP50=%.5f(e%s) min_mAP50-95=%.5f max_mAP50=%.5f(e%s) max_mAP50-95=%.5f(e%s) max_P=%.5f max_R=%.5f(e%s) nan_rows=%d\n", NR-1, min50, mine, min95, max50, max50e, max95, max95e, maxp, maxr, maxre, nan+0}' $RU/results.csv
echo "== epoch 1 / 10 / 69 / 70 =="
awk -F, 'NR==1 || NR==2 || NR==11 || NR==70 || NR==71' $RU/results.csv | cut -d, -f1-10
echo "== probe scan =="
$PY - "$RU" <<PY
import json, sys, os
ru = sys.argv[1]
def scan(name, keys):
    p = os.path.join(ru, name)
    if not os.path.exists(p):
        print(name, "MISSING"); return
    n = 0; agg = {k: None for k in keys}
    for line in open(p, encoding="utf-8"):
        line = line.strip()
        if not line: continue
        try: d = json.loads(line)
        except Exception: continue
        n += 1
        for k in keys:
            v = d.get(k)
            if v is None: continue
            if agg[k] is None: agg[k] = v
            elif k.startswith("min"): agg[k] = min(agg[k], v)
            else: agg[k] = max(agg[k], v)
    print("%s lines=%d %s" % (name, n, {k: (round(v, 5) if isinstance(v, float) else v) for k, v in agg.items()}))
scan("mem_probe.jsonl", ["MemAvailable_GB", "SwapFree_GB", "swap_in_KBps", "swap_out_KBps", "trainer_rss_GB"])
p = os.path.join(ru, "lr_probe.jsonl")
if os.path.exists(p):
    n = 0; mx = 0.0; mn = None; gmax = None
    for line in open(p, encoding="utf-8"):
        line = line.strip()
        if not line: continue
        try: d = json.loads(line)
        except Exception: continue
        n += 1
        lrs = [g.get("lr") for g in d.get("groups", []) if g.get("lr") is not None]
        if lrs:
            m = max(lrs)
            if m > mx: mx = m; gmax = lrs
            mn = min(lrs) if mn is None else min(mn, min(lrs))
    print("lr_probe.jsonl lines=%d max_any_group_lr=%.8f min_any_group_lr=%.10f" % (n, mx, mn or 0))
print("lr_config.json:", json.dumps(json.load(open(os.path.join(ru, "lr_config.json"))))[:600] if os.path.exists(os.path.join(ru, "lr_config.json")) else "MISSING")
PY
echo "== mem now =="
grep -E "MemAvailable|SwapFree" /proc/meminfo
pgrep -f "[t]rain_yolo26_v1" | head -n 3; echo "pgrep_done"