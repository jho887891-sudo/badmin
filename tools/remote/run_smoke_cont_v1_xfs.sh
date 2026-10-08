#!/bin/bash
# spec 12 smoke, 2 epochs, xfs runtime
# Runtime is deliberately the xfs python (/usr/bin/python3) with ultralytics copied to xfs: the fuseblk mount has
# ~200 ms per-request latency, so importing torch from /home/T7/public/miniconda3 exceeds 120 s there.
set -u
B=/home/dgut/.dsh-bench
L=$B/cont_v1
PY=/usr/bin/python3
export PYTHONPATH=$B/wheel_xfs
export CUDA_VISIBLE_DEVICES=0
export WANDB_MODE=disabled
NAME=${1:-smoke_e2}
MAN="$L/${NAME}_manifest.json"
[ -e "$MAN" ] && { echo "refusing: $MAN exists"; exit 3; }
echo "=== spec 12 smoke, 2 epochs, xfs runtime $(date -Is)"
"$PY" "$B/train_eth_only_v1.py" --contract "$L/yolo26s_v2_full_eth_cont_v1.yaml" \
  --recipe-json "$B/eth_only_v1_official_recipe.json" \
  --data-yaml "$B/data/yolo26s_v2_full_eth_cont_v1/dataset.yaml" \
  --weights "$B/runs/detect/runs_eth_hardneg_v2/full_v2_e50/weights/best.pt" \
  --project runs_cont_v1 --name "$NAME" --out-manifest "$MAN" \
  --smoke --smoke-epochs 2 --mem-cap-gib 24 --workers 8 --device 0
echo "rc=$?"
echo "=== end $(date -Is)"
