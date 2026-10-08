#!/usr/bin/env bash
set -u
echo "=== my python procs (top by RSS) ==="
ps -u dgut -o pid,ppid,etime,rss,args --sort=-rss | head -8
echo "=== any orphaned trainer workers ==="
pgrep -fa "[t]rain_yolo26_v1" || echo NONE
echo "=== gpu ==="
nvidia-smi --query-gpu=memory.free,utilization.gpu --format=csv,noheader