#!/usr/bin/env bash
echo "=== B proc ==="
pgrep -fa "[t]rain_yolo26_v1" | head -1 || echo NONE
RUN=$(ls -dt /home/T7/ojh/robot_sim/runs/shuttle_yolo26_v1/gate6B_*/ 2>/dev/null | head -1)
RUN=${RUN%/}
echo "RUN=$RUN"
echo "=== progress ==="
grep -a "1024:" "$RUN.log" 2>/dev/null | tail -1 | tr -d "\r" | cut -c1-170
echo "=== results ==="
tail -2 "$RUN/results.csv" 2>/dev/null || echo "(no results yet)"
echo "=== val list size ==="
wc -l "$RUN/val_resolved.txt" 2>/dev/null
echo "=== gpu ==="
nvidia-smi --query-gpu=memory.free,utilization.gpu --format=csv,noheader