#!/usr/bin/env bash
set -u
RUN=/home/T7/ojh/robot_sim/runs/shuttle_yolo26_v1/gate6A_imgsz1024_b16_20260922-132656
echo "now: $(date)"
echo "uptime: $(uptime)"
echo "log mtime: $(stat -c %y $RUN.log)"
echo "=== errors in log ==="
grep -a -iE "error|traceback|out of memory|cuda" "$RUN.log" | tail -5 || echo "(none)"
echo "=== last 2 progress lines ==="
grep -a "1024:" "$RUN.log" | tail -2 | tr -d "\r" | cut -c1-200
echo "=== results.csv ==="
cat "$RUN/results.csv" 2>/dev/null || echo "(none)"
echo "=== gpu apps now ==="
nvidia-smi --query-compute-apps=pid,used_memory --format=csv,noheader
echo "=== train procs ==="
pgrep -fa "[t]rain_yolo26_v1" || echo NONE
echo "=== dmesg oom ==="
(dmesg 2>/dev/null | tail -30 | grep -i -E "oom|killed" || echo "(dmesg not readable or no oom)")