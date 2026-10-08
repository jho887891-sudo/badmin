#!/usr/bin/env bash
RUN=$(ls -dt /home/T7/ojh/robot_sim/runs/shuttle_yolo26_v1/gate6A_*/ 2>/dev/null | head -1)
RUN=${RUN%/}
echo "RUN=$RUN"
echo "=== progress (last lines) ==="
tail -c 600 "$RUN.log" 2>/dev/null | tr -d "\r" | tail -4
echo "=== results.csv ==="
tail -2 "$RUN/results.csv" 2>/dev/null || echo "(no results.csv yet)"
echo "=== artifacts ==="
ls "$RUN" 2>/dev/null | tr "\n" " "; echo
echo "=== gpu ==="
nvidia-smi --query-gpu=memory.free,utilization.gpu --format=csv,noheader