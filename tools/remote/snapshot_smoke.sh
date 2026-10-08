#!/bin/bash
B=/home/dgut/.dsh-bench
echo "=== $(date -Is) alive=$(pgrep -fc 'train_eth_only_v1[.]py')"
echo "--- last log lines:"
tr '\r' '\n' < "$B/cont_v1/smoke_xfs.log" 2>/dev/null | grep -vE "Scanning" | tail -12
echo "--- epoch progress (results.csv):"
CSV=$(find "$B/runs_cont_v1" "$B/runs/detect/runs_cont_v1" -name results.csv 2>/dev/null | head -1)
echo "file=$CSV rows=$([ -n "$CSV" ] && wc -l < "$CSV")"
[ -n "$CSV" ] && head -1 "$CSV" | cut -d, -f1-8 && tail -2 "$CSV" | cut -d, -f1-8
echo "--- weights: $(find "$B/runs_cont_v1" "$B/runs/detect/runs_cont_v1" -name '*.pt' 2>/dev/null | wc -l)"
find "$B/runs_cont_v1" "$B/runs/detect/runs_cont_v1" -name "*.pt" 2>/dev/null | head -8
echo "--- gpu: $(nvidia-smi --query-gpu=utilization.gpu,memory.used --format=csv,noheader)"
