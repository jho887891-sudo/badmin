#!/bin/bash
B=/home/dgut/.dsh-bench
sleep 420
echo "=== $(date -Is)"
echo "--- log tail:"; tail -20 "$B/cont_v1/smoke_xfs.log"
echo "--- trainer procs: $(pgrep -fc 'train_eth_only_v1[.]py')"
P=$(pgrep -f "python3 .*train_eth_only_v1[.]py" | head -1)
echo "--- pid=$P threads=$(ls /proc/$P/task 2>/dev/null | wc -l) children=$(pgrep -Pc $P 2>/dev/null)"
ps -o pid,etime,time,pcpu,stat,wchan:20 -p "$P" 2>&1 | tail -2
echo "--- io:"; head -3 /proc/$P/io 2>/dev/null
echo "--- run dir:"; find "$B/runs_cont_v1" "$B/runs/detect/runs_cont_v1" -maxdepth 2 2>/dev/null | head -12
echo "--- label caches:"; ls -l "$B/data/yolo26s_v2_full_eth_cont_v1/"*.cache 2>/dev/null || echo "no cache yet"
echo "--- gpu:"; nvidia-smi --query-gpu=utilization.gpu,memory.used --format=csv,noheader
