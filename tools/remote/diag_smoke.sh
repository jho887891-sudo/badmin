#!/bin/bash
B=/home/dgut/.dsh-bench
P=$(pgrep -f "miniconda3/bin/python .*train_eth_only_v1" | head -1)
echo "pid=$P"
ps -o pid,etime,time,pcpu,stat,wchan:24 -p "$P" 2>&1 | tail -2
echo "--- io:"; head -4 /proc/$P/io 2>/dev/null
echo "--- threads: $(ls /proc/$P/task 2>/dev/null | wc -l)"
echo "--- open files (last 6):"; ls -l /proc/$P/fd 2>/dev/null | tail -6
echo "--- cwd: $(readlink /proc/$P/cwd 2>/dev/null)"
echo "--- dataset dir:"; ls -la "$B/data/yolo26s_v2_full_eth_cont_v1/"
echo "--- runs dir:"; ls -la "$B/runs_cont_v1/" 2>/dev/null; ls -la "$B/runs/detect/runs_cont_v1/" 2>/dev/null | head
echo "--- recent files touched under runs (last 3):"; find "$B/runs_cont_v1" "$B/runs/detect/runs_cont_v1" -newermt "-20 minutes" 2>/dev/null | head -5
echo "--- top cpu:"; ps -eo pcpu,etime,cmd --sort=-pcpu | head -4
