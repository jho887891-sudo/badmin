#!/usr/bin/env bash
RUN=$(ls -dt /home/T7/ojh/robot_sim/runs/shuttle_yolo26_v1/gate6A_*/ 2>/dev/null | head -1)
RUN=${RUN%/}
echo "RUN=$RUN"
echo "=== training proc ==="
pgrep -fa "[t]rain_yolo26_v1" | head -2 || echo NONE
echo "=== results.csv (all) ==="
cat "$RUN/results.csv" 2>/dev/null
echo "=== weights ==="
ls -l "$RUN/weights" 2>/dev/null || echo "(none)"
echo "=== log tail ==="
tail -c 400 "$RUN.log" 2>/dev/null | tr -d "\r" | tail -3
echo "=== metrics.json ==="
cat "$RUN/metrics.json" 2>/dev/null || echo "(none yet)"
echo "=== gpu ==="
nvidia-smi --query-gpu=memory.free,utilization.gpu --format=csv,noheader