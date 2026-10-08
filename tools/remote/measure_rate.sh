#!/bin/bash
L=/home/dgut/.dsh-bench/cont_v1
get() { tr '\r' '\n' < "$L/smoke_xfs2.log" | grep -E "1024:" | tail -1; }
echo "=== $(date -Is)"
A=$(get); echo "A: $A"
IA=$(echo "$A" | grep -oE "[0-9]+/772" | head -1 | cut -d/ -f1)
sleep 90
B=$(get); echo "B: $B"
IB=$(echo "$B" | grep -oE "[0-9]+/772" | head -1 | cut -d/ -f1)
echo "iters: $IA -> $IB in 90 s  => $(echo "scale=2; ($IB-$IA)/90" | bc) it/s"
echo "gpu: $(nvidia-smi --query-gpu=utilization.gpu,memory.used --format=csv,noheader)"
echo "trainer workers: $(pgrep -P $(pgrep -f 'python3 .*train_eth_only_v1[.]py' | head -1) | wc -l)"
