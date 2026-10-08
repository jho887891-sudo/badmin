#!/usr/bin/env bash
RUN=$(ls -dt /home/T7/ojh/robot_sim/runs/shuttle_yolo26_v1/gate6A_*w8*/ 2>/dev/null | head -1)
RUN=${RUN%/}
echo "RUN=$RUN"
echo "=== progress ==="
grep -a "1024:" "$RUN.log" 2>/dev/null | tail -2 | tr -d "\r" | cut -c1-190
echo "=== results.csv ==="
tail -2 "$RUN/results.csv" 2>/dev/null || echo "(no results yet)"
echo "=== host mem ==="
free -g | head -3
echo "=== gpu ==="
nvidia-smi --query-gpu=memory.free,utilization.gpu --format=csv,noheader