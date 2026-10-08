#!/usr/bin/env bash
RUN=$(ls -dt /home/T7/ojh/robot_sim/runs/shuttle_yolo26_v1/gate6B_*/ 2>/dev/null | head -1)
RUN=${RUN%/}
echo "RUN=$RUN"
echo "=== resolved_config.yaml (what was actually passed) ==="
cat "$RUN/resolved_config.yaml" 2>/dev/null | head -30
echo "=== progress ==="
grep -a "1024:" "$RUN.log" 2>/dev/null | tail -1 | tr -d "\r" | cut -c1-170
echo "=== results ==="
tail -2 "$RUN/results.csv" 2>/dev/null || echo "(no results yet)"