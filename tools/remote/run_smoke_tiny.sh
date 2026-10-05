#!/usr/bin/env bash
# Tiny recovery V1: 3-epoch smoke with the canonical launcher (diagnostic only, never a final result).
set -euo pipefail
cd "$HOME/.dsh-bench"; B="$HOME/.dsh-bench"
PY=/home/T7/public/miniconda3/bin/python
export PYTHONPATH=/home/T7/ojh/robot_sim/experiments/yolo26_p2_ab/_wheel_extract
export CUDA_VISIBLE_DEVICES=0
export WANDB_MODE=disabled
NAME="${1:-tiny_smoke_e3}"
MAN="$B/${NAME}_manifest.json"
[ -e "$MAN" ] && { echo "refusing: $MAN exists"; exit 3; }
mkdir -p "$B/data/tiny_recovery_v1"
cp -f "$B/tiny_recovery_v1_train_train.txt" "$B/tiny_recovery_v1_val.txt" "$B/data/tiny_recovery_v1/"
cat > "$B/tiny_recovery_v1_remote.yaml" <<EOF
path: /home/dgut/.dsh-bench/data/tiny_recovery_v1
train: tiny_recovery_v1_train_train.txt
val: tiny_recovery_v1_val.txt
nc: 1
names:
  0: shuttlecock
EOF
echo "=== smoke started $(date -Is) ==="
"$PY" train_eth_only_v1.py --contract tiny_recovery_yolo26s_v1.yaml \
  --recipe-json eth_only_v1_official_recipe.json --data-yaml tiny_recovery_v1_remote.yaml \
  --weights /home/T7/dgut/robot_sim/yolo26s.pt --project runs_tiny_recovery_v1 --name "$NAME" \
  --out-manifest "$MAN" --smoke --mem-cap-gib 24 --workers 8 --device 0
echo "=== smoke done $(date -Is) ==="
