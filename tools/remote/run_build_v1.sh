#!/usr/bin/env bash
# ETH-only V1 dataset build (Task 2), corrected: box-level equiv_size_640 + val_locations from real frames.
set -euo pipefail
cd "$HOME/.dsh-bench"
B="$HOME/.dsh-bench"
PY=/home/T7/public/miniconda3/bin/python
mkdir -p "$B/v1_buggy_backup"
for f in eth_only_v1_train_manifest.csv eth_only_v1_val_manifest.csv eth_only_v1_train.yaml eth_only_v1_train_train.txt eth_only_v1_train_val.txt eth_only_v1_data_audit.json eth_only_v1_size_distribution.csv; do
  [ -f "$B/$f" ] && cp -p "$B/$f" "$B/v1_buggy_backup/$f" || true
done
echo "=== started $(date -Is) ==="
echo "cmd: $PY build_eth_only_v1_dataset.py --eth-data /home/T7/dgut/robot_sim/eth_shuttle_detection --recipe eth_only_v1_official_recipe.json --exclude-eval eval_exclusion.csv --train-out eth_only_v1_train_manifest.csv --val-out eth_only_v1_val_manifest.csv --dataset-yaml eth_only_v1_train.yaml --audit-out eth_only_v1_data_audit.json --size-out eth_only_v1_size_distribution.csv"
"$PY" build_eth_only_v1_dataset.py \
  --eth-data /home/T7/dgut/robot_sim/eth_shuttle_detection \
  --recipe eth_only_v1_official_recipe.json \
  --exclude-eval eval_exclusion.csv \
  --train-out eth_only_v1_train_manifest.csv \
  --val-out eth_only_v1_val_manifest.csv \
  --dataset-yaml eth_only_v1_train.yaml \
  --audit-out eth_only_v1_data_audit.json \
  --size-out eth_only_v1_size_distribution.csv
echo "=== done $(date -Is) rc=$? ==="
