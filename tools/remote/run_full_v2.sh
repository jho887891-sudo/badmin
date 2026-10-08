#!/usr/bin/env bash
# Task 4 step 4: full 50-epoch V2 run, internal-validation-only checkpoint selection.
set -euo pipefail
cd "$HOME/.dsh-bench"; B="$HOME/.dsh-bench"
PY=/home/T7/public/miniconda3/bin/python
export PYTHONPATH=/home/T7/ojh/robot_sim/experiments/yolo26_p2_ab/_wheel_extract
export CUDA_VISIBLE_DEVICES=0
export WANDB_MODE=disabled
NAME="${1:-full_e50}"
MAN="$B/${NAME}_manifest.json"
[ -e "$MAN" ] && { echo "refusing: $MAN exists"; exit 3; }
echo "=== started $(date -Is) V2 full 50 epochs ==="
"$PY" train_eth_only_v1.py --contract eth_real_hardneg_yolo26s_v2.yaml \
  --recipe-json eth_only_v1_official_recipe.json --data-yaml eth_real_hardneg_v2_remote.yaml \
  --weights /home/T7/dgut/robot_sim/yolo26s.pt --project runs_eth_hardneg_v2 --name "$NAME" \
  --out-manifest "$MAN" --mem-cap-gib 24 --workers 8 --device 0
echo "=== done $(date -Is) ==="
"$PY" -c "import json,sys; m=json.load(open(sys.argv[1])); print(json.dumps({k:m.get(k) for k in ('batch','gpu_cap','selection','best_checkpoint','started','finished')}, indent=1))" "$MAN"
