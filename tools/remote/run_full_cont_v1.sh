#!/bin/bash
# Spec sections 9-10 full run: 20 epochs, imgsz 1024, AdamW, lr0 3e-5, nbs 64, physical batch 32 -> 24 -> 16.
set -u
B=/home/dgut/.dsh-bench
L=$B/cont_v1
PY=/home/T7/public/miniconda3/bin/python
export PYTHONPATH=/home/T7/ojh/robot_sim/experiments/yolo26_p2_ab/_wheel_extract
export CUDA_VISIBLE_DEVICES=0
export WANDB_MODE=disabled
NAME=${1:-full_cont_e20}
MAN="$L/${NAME}_manifest.json"
[ -e "$MAN" ] && { echo "refusing: $MAN exists"; exit 3; }
echo "=== full run $(date -Is)"
"$PY" "$B/train_eth_only_v1.py" --contract "$L/yolo26s_v2_full_eth_cont_v1.yaml" \
  --recipe-json "$B/eth_only_v1_official_recipe.json" \
  --data-yaml "$B/data/yolo26s_v2_full_eth_cont_v1/dataset.yaml" \
  --weights "$B/runs/detect/runs_eth_hardneg_v2/full_v2_e50/weights/best.pt" \
  --project runs_cont_v1 --name "$NAME" --out-manifest "$MAN" \
  --mem-cap-gib 24 --workers 8 --device 0
echo "rc=$?"
echo "=== end $(date -Is)"
