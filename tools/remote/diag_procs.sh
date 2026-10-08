#!/bin/bash
echo "=== all trainer-related processes:"
ps -eo pid,ppid,etime,pcpu,stat,cmd | grep -E "train_eth_only_v1[.]py" | grep -v grep | awk '{print $1, $2, $3, $4, $5, substr($0, index($0,$6), 60)}'
echo "=== count of MAIN trainer processes (cmd contains --name and not forked worker marker):"
pgrep -f "train_eth_only_v1[.]py" | wc -l
echo "=== distinct --name values:"
ps -eo cmd | grep -oE "\-\-name [A-Za-z0-9_]+" | sort | uniq -c
echo "=== parent pids:"
for p in $(pgrep -f "train_eth_only_v1[.]py"); do echo "$p ppid=$(ps -o ppid= -p $p) state=$(ps -o stat= -p $p)"; done | sort -t= -k2 | head -20
echo "=== run dirs:"
find /home/dgut/.dsh-bench -maxdepth 3 -name "runs_cont_v1" -o -maxdepth 4 -name "smoke_e2" 2>/dev/null | head
echo "=== nvidia-smi processes:"
nvidia-smi --query-compute-apps=pid,used_memory --format=csv,noheader 2>/dev/null | head
