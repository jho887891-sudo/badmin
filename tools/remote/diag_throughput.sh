#!/bin/bash
echo "=== nproc=$(nproc) load=$(cut -d' ' -f1-3 /proc/loadavg)"
echo "=== free -g:"; free -g | head -2
echo "=== gpu util over 8s:"
nvidia-smi --query-gpu=utilization.gpu,utilization.memory,memory.used --format=csv,noheader -l 2 -c 4
echo "=== top 6 by cpu:"
ps -eo pcpu,pmem,etime,cmd --sort=-pcpu | head -7
echo "=== dataloader worker count (children of the trainer):"
P=$(pgrep -f "python3 .*train_eth_only_v1[.]py" | head -1)
echo "pid=$P threads=$(ls /proc/$P/task 2>/dev/null | wc -l) children=$(pgrep -Pc $P)"
echo "=== eth pool read speed (10 x 1 MB random-ish files):"
E=/home/T7/dgut/robot_sim/eth_shuttle_detection
time (for f in $(ls $E/ml_3_medium/images/train/*.jpg | head -10); do cat "$f" > /dev/null; done)
