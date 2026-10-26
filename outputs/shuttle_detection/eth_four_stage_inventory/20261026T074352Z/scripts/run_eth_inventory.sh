#!/usr/bin/env bash
# ETH all-box inventory: real V2 TRAIN manifest + label root + two frozen evaluation manifests.
# Everything is real mounted ETH data; nothing is synthesized. Output dir must be new.
set -euo pipefail

REPO=/home/dgut/eth_audit/repo
PY=/home/dgut/eth_audit/venv/bin/python
DATA=/home/T7/dgut/robot_sim/eth_shuttle_detection

ETH_V2_MANIFEST="$REPO/data/eth_real_hardneg_v2_train_manifest.csv"
ETH_LABEL_ROOT="$DATA"
FROZEN_EVAL_1="$REPO/data/clean_outdoor_cp_v1/frozen_eval_index.csv"
FROZEN_EVAL_2="$REPO/data/eth_real_hardneg_v2_val_manifest.csv"

OUT_ROOT=/home/dgut/eth_audit/out/inventory_$(date -u +%Y%m%dT%H%M%SZ)
if [ -e "$OUT_ROOT" ]; then
  echo "REFUSING_EXISTING_OUT_ROOT=$OUT_ROOT" >&2
  exit 3
fi
mkdir -p "$OUT_ROOT"

echo "OUT_ROOT=$OUT_ROOT"
echo "MANIFEST=$ETH_V2_MANIFEST"
sha256sum "$ETH_V2_MANIFEST"
echo "LABEL_ROOT=$ETH_LABEL_ROOT"
echo "FROZEN=$FROZEN_EVAL_1"
sha256sum "$FROZEN_EVAL_1"
echo "FROZEN=$FROZEN_EVAL_2"
sha256sum "$FROZEN_EVAL_2"

cd "$REPO"
set +e
"$PY" tools/eth_four_stage_inventory.py \
  --manifest "$ETH_V2_MANIFEST" \
  --label-root "$ETH_LABEL_ROOT" \
  --frozen-eval-manifest "$FROZEN_EVAL_1" \
  --frozen-eval-manifest "$FROZEN_EVAL_2" \
  --out-inventory "$OUT_ROOT/instances.csv" \
  --out-audit "$OUT_ROOT/audit.json" \
  > "$OUT_ROOT/stdout.json" 2> "$OUT_ROOT/stderr.log"
INV_RC=$?
set -e

echo "INVENTORY_RC=$INV_RC"
echo "ETH_INVENTORY_OUTPUT=$OUT_ROOT"
echo "--- stdout ---"
cat "$OUT_ROOT/stdout.json"
echo "--- stderr tail ---"
tail -20 "$OUT_ROOT/stderr.log"
exit "$INV_RC"
