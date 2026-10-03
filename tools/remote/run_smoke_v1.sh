#!/usr/bin/env bash
# Task 3 smoke: 3-epoch diagnostic run of the frozen ETH-only YOLO26s 1024 recipe (never a final result).
set -euo pipefail
B="$HOME/.dsh-bench"; cd "$B"
PY=/home/T7/public/miniconda3/bin/python
export PYTHONPATH=/home/T7/ojh/robot_sim/experiments/yolo26_p2_ab/_wheel_extract
export CUDA_VISIBLE_DEVICES=0
NAME="${1:-smoke_e3}"
MAN="$B/${NAME}_manifest.json"
DRY="$B/${NAME}_dryrun_manifest.json"
if [ -e "$MAN" ] || [ -e "$DRY" ]; then echo "refusing: $MAN or $DRY already exists"; exit 3; fi
# dataset layout implied by the frozen yaml (path: data/eth_only_v1 + relative list names)
mkdir -p "$B/data/eth_only_v1"
cp -f "$B/eth_only_v1_train_train.txt" "$B/eth_only_v1_train_val.txt" "$B/data/eth_only_v1/"
cat > "$B/eth_only_v1_train_remote.yaml" <<EOF
path: /home/dgut/.dsh-bench/data/eth_only_v1
train: eth_only_v1_train_train.txt
val: eth_only_v1_train_val.txt
nc: 1
names:
  0: shuttlecock
EOF
echo "=== $(date -Is) dry-run gate ==="
"$PY" train_eth_only_v1.py --contract eth_only_yolo26s_1024_v1.yaml \
  --recipe-json eth_only_v1_official_recipe.json --data-yaml eth_only_v1_train_remote.yaml \
  --weights /home/T7/dgut/robot_sim/yolo26s.pt --project runs_eth_only_v1 --name "${NAME}_dry" \
  --out-manifest "$DRY" --smoke --dry-run --mem-cap-gib 24 --device 0
echo "=== $(date -Is) 3-epoch smoke ==="
"$PY" train_eth_only_v1.py --contract eth_only_yolo26s_1024_v1.yaml \
  --recipe-json eth_only_v1_official_recipe.json --data-yaml eth_only_v1_train_remote.yaml \
  --weights /home/T7/dgut/robot_sim/yolo26s.pt --project runs_eth_only_v1 --name "$NAME" \
  --out-manifest "$MAN" --smoke --mem-cap-gib 24 --workers 8 --device 0
echo "=== done $(date -Is) ==="
cat "$MAN"
