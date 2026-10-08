#!/usr/bin/env bash
# Task 4 step 3: 3-epoch V2 smoke with the canonical V1 launcher (diagnostic only, never a final result).
set -euo pipefail
cd "$HOME/.dsh-bench"; B="$HOME/.dsh-bench"
PY=/home/T7/public/miniconda3/bin/python
export PYTHONPATH=/home/T7/ojh/robot_sim/experiments/yolo26_p2_ab/_wheel_extract
export CUDA_VISIBLE_DEVICES=0
export WANDB_MODE=disabled
NAME="${1:-smoke_e3}"
MAN="$B/${NAME}_manifest.json"
[ -e "$MAN" ] && { echo "refusing: $MAN exists"; exit 3; }
mkdir -p "$B/data/eth_real_hardneg_v2"
cp -f "$B/eth_real_hardneg_v2_train_train.txt" "$B/eth_real_hardneg_v2_val.txt" "$B/data/eth_real_hardneg_v2/"
cat > "$B/eth_real_hardneg_v2_remote.yaml" <<EOF
path: /home/dgut/.dsh-bench/data/eth_real_hardneg_v2
train: eth_real_hardneg_v2_train_train.txt
val: eth_real_hardneg_v2_val.txt
nc: 1
names:
  0: shuttlecock
EOF
echo "=== started $(date -Is) ==="
"$PY" train_eth_only_v1.py --contract eth_real_hardneg_yolo26s_v2.yaml \
  --recipe-json eth_only_v1_official_recipe.json --data-yaml eth_real_hardneg_v2_remote.yaml \
  --weights /home/T7/dgut/robot_sim/yolo26s.pt --project runs_eth_hardneg_v2 --name "$NAME" \
  --out-manifest "$MAN" --smoke --mem-cap-gib 24 --workers 8 --device 0
echo "=== done $(Date_IS=$(date -Is); echo $Date_IS) ==="
