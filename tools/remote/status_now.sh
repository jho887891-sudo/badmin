#!/bin/bash
B=/home/dgut/.dsh-bench
L=$B/cont_v1
echo "=== $(date -Is)"
echo "--- auto launcher log:"; cat "$L/auto_launcher.log" 2>/dev/null
echo "--- trainer alive: $(pgrep -fc 'train_eth_only_v1[.]py')"
echo "--- smoke results:"; cut -d, -f1-8 /home/dgut/runs/detect/runs_cont_v1/smoke_b16/results.csv 2>/dev/null
echo "--- smoke manifest: $([ -f $L/smoke_b16_manifest.json ] && echo yes || echo no)  full manifest: $([ -f $L/full_cont_e20_manifest.json ] && echo yes || echo no)"
echo "--- full run log size: $(stat -c%s $L/full_cont_e20.log 2>/dev/null || echo 0) B"
tr '\r' '\n' < "$L/full_cont_e20.log" 2>/dev/null | grep -E "1024:|Starting training|Epoch" | tail -3
echo "--- full results.csv:"; cut -d, -f1-8 "$B/runs_cont_v1/full_cont_e20/results.csv" 2>/dev/null | tail -4
echo "--- weights:"; ls -l "$B/runs_cont_v1/full_cont_e20/weights/" 2>/dev/null | tail -6
echo "--- gpu: $(nvidia-smi --query-gpu=utilization.gpu,memory.used --format=csv,noheader)"
