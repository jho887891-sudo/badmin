#!/bin/bash
# A/B test: batch 32 sat at 98.6% of the 24 GiB per-process cap (24.2 GiB allocated) with the GPU idle ~99% of the
# time; take the spec's documented fallback ladder (32 -> 24 -> 16) and measure. workers back to 8 (proven to run).
set -u
B=/home/dgut/.dsh-bench
L=$B/cont_v1
pkill -9 -f "train_eth_only_v1[.]py" 2>/dev/null
sleep 5
echo "alive=$(pgrep -fc 'train_eth_only_v1[.]py')"
rm -rf "$L/runs" "$B/runs_cont_v1" "$L/smoke_e2_manifest.json"
echo "=== launching smoke batch 16, workers 8"
setsid nohup /usr/bin/python3 "$B/train_eth_only_v1.py" --contract "$L/yolo26s_v2_full_eth_cont_v1.yaml" \
  --recipe-json "$B/eth_only_v1_official_recipe.json" \
  --data-yaml "$B/data/yolo26s_v2_full_eth_cont_v1/dataset.yaml" \
  --weights "$B/runs/detect/runs_eth_hardneg_v2/full_v2_e50/weights/best.pt" \
  --project runs_cont_v1 --name smoke_b16 --out-manifest "$L/smoke_b16_manifest.json" \
  --smoke --smoke-epochs 2 --batch 16 --mem-cap-gib 24 --workers 8 --device 0 \
  > "$L/smoke_b16.log" 2>&1 < /dev/null &
echo launched
sleep 420
echo "=== $(date -Is)"
tr '\r' '\n' < "$L/smoke_b16.log" | grep -E "1024:" | tail -2
echo "gpu: $(nvidia-smi --query-gpu=utilization.gpu,memory.used --format=csv,noheader)"
