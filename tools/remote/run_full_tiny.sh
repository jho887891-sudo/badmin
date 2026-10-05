#!/usr/bin/env bash
# Tiny recovery V1: full 50-epoch run, internal-validation-only checkpoint selection.
set -euo pipefail
cd "$HOME/.dsh-bench"; B="$HOME/.dsh-bench"
PY=/home/T7/public/miniconda3/bin/python
export PYTHONPATH=/home/T7/ojh/robot_sim/experiments/yolo26_p2_ab/_wheel_extract
export CUDA_VISIBLE_DEVICES=0
export WANDB_MODE=disabled
NAME="${1:-tiny_full_e50}"
MAN="$B/${NAME}_manifest.json"
[ -e "$MAN" ] && { echo "refusing: $MAN exists"; exit 3; }
echo "=== full run started $(date -Is) ==="
"$PY" train_eth_only_v1.py --contract tiny_recovery_yolo26s_v1.yaml \
  --recipe-json eth_only_v1_official_recipe.json --data-yaml tiny_recovery_v1_remote.yaml \
  --weights /home/T7/dgut/robot_sim/yolo26s.pt --project runs_tiny_recovery_v1 --name "$NAME" \
  --out-manifest "$MAN" --mem-cap-gib 24 --workers 8 --device 0
echo "=== full run done $(date -Is) ==="
