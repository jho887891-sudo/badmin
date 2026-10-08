#!/usr/bin/env bash
RUN=/home/T7/ojh/robot_sim/runs/shuttle_yolo26_v1/gate6B_smoke_musgd_lr0001_20260927-211053
echo "=== results ==="
cat "$RUN/results.csv" 2>/dev/null || echo "(none yet)"
echo "=== progress ==="
grep -a "1024:" "$RUN.log" 2>/dev/null | tail -1 | tr -d "\r" | cut -c1-160
echo "=== mem tail ==="
tail -2 "$RUN/mem_probe.jsonl" 2>/dev/null
echo "=== gpu ==="
nvidia-smi --query-gpu=memory.free,utilization.gpu --format=csv,noheader
echo "=== proc ==="
pgrep -f "[t]rain_yolo26_v1" | head -1 || echo NONE