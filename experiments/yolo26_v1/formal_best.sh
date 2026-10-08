#!/usr/bin/env bash
RUN=/home/T7/ojh/robot_sim/runs/shuttle_yolo26_v1/gate6B_lr0001_musgd_b16_w8_20260927-235730
echo "=== last 5 epochs ==="
tail -5 "$RUN/results.csv"
echo "=== best so far (by mAP50-95) ==="
awk -F, "NR>1 {if ($9+0 > m) {m=$9+0; line=$0}} END {print line}" "$RUN/results.csv"
echo "=== best so far (by mAP50) ==="
awk -F, "NR>1 {if ($8+0 > m) {m=$8+0; line=$0}} END {print line}" "$RUN/results.csv"
echo "=== rows ==="
wc -l < "$RUN/results.csv"
echo "=== weights ==="
ls -l "$RUN/weights" 2>/dev/null
echo "=== progress ==="
grep -a "1024:" "$RUN.log" | tail -1 | tr -d "\r" | cut -c1-140
echo "=== mem ==="
tail -1 "$RUN/mem_probe.jsonl"