#!/usr/bin/env bash
# ETH-only V1 dataset build (Task 2), rerun: recipe-derived negative fraction + explicit 0.1.
set -euo pipefail
cd "$HOME/.dsh-bench"; B="$HOME/.dsh-bench"
PY=/home/T7/public/miniconda3/bin/python
mkdir -p "$B/v1_badneg_backup"
for f in eth_only_v1_train_manifest.csv eth_only_v1_val_manifest.csv eth_only_v1_train.yaml eth_only_v1_train_train.txt eth_only_v1_train_val.txt eth_only_v1_data_audit.json eth_only_v1_size_distribution.csv; do
  [ -f "$B/$f" ] && cp -p "$B/$f" "$B/v1_badneg_backup/$f" || true
done
echo "=== started $(date -Is) ==="
"$PY" build_eth_only_v1_dataset.py \
  --eth-data /home/T7/dgut/robot_sim/eth_shuttle_detection \
  --recipe eth_only_v1_official_recipe.json \
  --exclude-eval eval_exclusion.csv \
  --negative-fraction 0.1 \
  --train-out eth_only_v1_train_manifest.csv \
  --val-out eth_only_v1_val_manifest.csv \
  --dataset-yaml eth_only_v1_train.yaml \
  --audit-out eth_only_v1_data_audit.json \
  --size-out eth_only_v1_size_distribution.csv
echo "=== self-check $(date -Is) ==="
"$PY" - <<'PYEOF'
import json, csv
a = json.load(open("eth_only_v1_data_audit.json"))
tr = list(csv.DictReader(open("eth_only_v1_train_manifest.csv")))
va = list(csv.DictReader(open("eth_only_v1_val_manifest.csv")))
coco = sum(1 for r in tr if r["location"] == "coco_train")
print("train", len(tr), "val", len(va), "coco", coco, "frac", a["val_positive_fraction"],
      "src", a.get("negative_fraction_source"), "val_locs", a["val_locations"])
assert len(tr) == 14543, len(tr)
assert len(va) == 2920, len(va)
assert coco == 550, coco
assert a["negative_fraction"] == 0.1 and a["negative_fraction_source"] == "cli"
assert a["val_locations"] == ["glc_2", "uetlibergstrasse_2"]
assert a["status"] == "PASS" and not a["problems"]
print("SELF-CHECK OK")
PYEOF
echo "=== done $(date -Is) ==="
