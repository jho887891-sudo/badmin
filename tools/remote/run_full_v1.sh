#!/usr/bin/env bash
# Task 4: full 50-epoch ETH-only YOLO26s 1024 run, internal-validation-only checkpoint selection.
set -euo pipefail
cd "$HOME/.dsh-bench"; B="$HOME/.dsh-bench"
PY=/home/T7/public/miniconda3/bin/python
export PYTHONPATH=/home/T7/ojh/robot_sim/experiments/yolo26_p2_ab/_wheel_extract
export CUDA_VISIBLE_DEVICES=0
NAME="${1:-full_e50}"
MAN="$B/${NAME}_manifest.json"
DRY="$B/${NAME}_dryrun_manifest.json"
if [ -e "$MAN" ]; then echo "refusing: $MAN exists"; exit 3; fi
mkdir -p "$B/data/eth_only_v1"
cp -f "$B/eth_only_v1_train_train.txt" "$B/eth_only_v1_train_val.txt" "$B/data/eth_only_v1/"
echo "=== $(date -Is) full run $NAME (50 epochs) ==="
"$PY" train_eth_only_v1.py --contract eth_only_yolo26s_1024_v1.yaml \
  --recipe-json eth_only_v1_official_recipe.json --data-yaml eth_only_v1_train_remote.yaml \
  --weights /home/T7/dgut/robot_sim/yolo26s.pt --project runs_eth_only_v1 --name "$NAME" \
  --out-manifest "$MAN" --mem-cap-gib 24 --workers 8 --device 0
echo "=== done $(date -Is) ==="
"$PY" -c "import json,sys; m=json.load(open(sys.argv[1])); print(json.dumps({k:m.get(k) for k in ('batch','gpu_cap','selection','best_checkpoint','started','finished')}, indent=1))" "$MAN"
