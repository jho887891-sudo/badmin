#!/usr/bin/env bash
RUN=$(ls -dt /home/T7/ojh/robot_sim/runs/shuttle_yolo26_v1/gate6B_fromA_*/ 2>/dev/null | head -1)
RUN=${RUN%/}
echo "=== results.csv ==="
cat "$RUN/results.csv" 2>/dev/null || echo "(none)"
echo "=== current epoch ==="
grep -a "1024:" "$RUN.log" 2>/dev/null | tail -1 | tr -d "\r" | cut -c1-160
echo "=== host mem ==="
free -g | head -2