#!/bin/bash
L=/home/dgut/.dsh-bench/cont_v1
P=$(pgrep -f "python3 .*train_eth_only_v1[.]py" | head -1)
echo "pid=$P elapsed=$(ps -o etime= -p $P 2>/dev/null) cpu=$(ps -o time= -p $P 2>/dev/null) state=$(ps -o stat= -p $P 2>/dev/null) wchan=$(cat /proc/$P/wchan 2>/dev/null)"
echo "threads=$(ls /proc/$P/task 2>/dev/null | wc -l) children=$(pgrep -P $P 2>/dev/null | wc -l)"
echo "--- io:"; head -3 /proc/$P/io 2>/dev/null
echo "--- log tail (translated):"
tr '\r' '\n' < "$L/smoke_xfs2.log" | tail -8
echo "--- log size: $(stat -c%s $L/smoke_xfs2.log)"
echo "--- cache files:"; ls -l /home/dgut/.dsh-bench/data/yolo26s_v2_full_eth_cont_v1/ 2>/dev/null
echo "--- gpu:"; nvidia-smi --query-gpu=utilization.gpu,memory.used --format=csv,noheader
