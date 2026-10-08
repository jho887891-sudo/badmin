#!/usr/bin/env bash
RUN=/home/T7/ojh/robot_sim/runs/shuttle_yolo26_v1/gate6B_lr0001_musgd_b16_w8_20260927-235730
echo "=== process ==="
pgrep -fa "[t]rain_yolo26_v1" | head -1 | cut -c1-100 || echo NONE
echo "=== progress (current epoch) ==="
grep -a "1024:" "$RUN.log" | tail -1 | tr -d "\r" | cut -c1-140
echo "=== last 3 epochs ==="
tail -3 "$RUN/results.csv"
echo "=== best by fitness (0.1*mAP50 + 0.9*mAP50-95) ==="
awk -F, "NR>1 {f=0.1*\$8+0.9*\$9; if (f>m) {m=f; e=\$1; r=\$8; p=\$9}} END {printf \"epoch=%s mAP50=%s mAP50-95=%s fitness=%.5f\n\", e, r, p, m}" "$RUN/results.csv"
echo "=== rows / weights ==="
wc -l < "$RUN/results.csv"
ls -l "$RUN/weights" | tail -2
echo "=== host mem ==="
free -g | head -2
tail -1 "$RUN/mem_probe.jsonl"
echo "=== gpu ==="
nvidia-smi --query-gpu=memory.free,utilization.gpu --format=csv,noheader