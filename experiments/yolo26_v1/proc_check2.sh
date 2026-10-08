#!/usr/bin/env bash
echo "=== MY v1 processes ==="
pgrep -fa "[t]rain_yolo26_v1" || echo NONE
pgrep -fa "[y]olo26_v1" || echo NO_V1_SCRIPTS
echo "=== my account processes (top RSS) ==="
ps -u dgut -o pid,etime,rss,args --sort=-rss | head -8
echo "=== all python on the box ==="
ps -eo pid,user,etime,rss,args | grep "[p]ython" | head -8
echo "=== gpu ==="
nvidia-smi --query-gpu=memory.free,utilization.gpu --format=csv,noheader
nvidia-smi --query-compute-apps=pid,used_memory --format=csv,noheader