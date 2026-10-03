#!/usr/bin/env bash
# Task 3 integration check: 1 epoch through the FULL launcher path (wandb off, manifest + selection).
# The plan-required 3-epoch smoke already ran (runs_eth_only_v1/smoke_e3b, 0.761 h, best.pt mAP50-95 0.616).
set -euo pipefail
cd "$HOME/.dsh-bench"; B="$HOME/.dsh-bench"
PY=/home/T7/public/miniconda3/bin/python
export PYTHONPATH=/home/T7/ojh/robot_sim/experiments/yolo26_p2_ab/_wheel_extract
export CUDA_VISIBLE_DEVICES=0
export WANDB_MODE=disabled
NAME="${1:-integration_e1}"
MAN="$B/${NAME}_manifest.json"
if [ -e "$MAN" ]; then echo "refusing: $MAN exists"; exit 3; fi
mkdir -p "$B/data/eth_only_v1"
cp -f "$B/eth_only_v1_train_train.txt" "$B/eth_only_v1_train_val.txt" "$B/data/eth_only_v1/"
echo "=== $(date -Is) $NAME (1 epoch, integration) ==="
"$PY" train_eth_only_v1.py --contract eth_only_yolo26s_1024_v1.yaml \
  --recipe-json eth_only_v1_official_recipe.json --data-yaml eth_only_v1_train_remote.yaml \
  --weights /home/T7/dgut/robot_sim/yolo26s.pt --project runs_eth_only_v1 --name "$NAME" \
  --out-manifest "$MAN" --epochs 1 --mem-cap-gib 24 --workers 8 --device 0
echo "=== done $(date -Is) ==="
