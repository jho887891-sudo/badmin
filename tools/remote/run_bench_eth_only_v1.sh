#!/usr/bin/env bash
# Task 6: same forward protocol as the earlier A6000 24 GB envelope run, now including eth_only_v1_best.
set -euo pipefail
cd "$HOME/.dsh-bench"
export PYTHONPATH=/home/T7/ojh/robot_sim/experiments/yolo26_p2_ab/_wheel_extract
export CUDA_VISIBLE_DEVICES=0
export WANDB_MODE=disabled
cp -f "$HOME/.dsh-bench/runs/detect/runs_eth_only_v1/full_e50/weights/best.pt" "$HOME/.dsh-bench/eth_only_v1_best.pt"
echo "=== started $(date -Is) ==="
/home/T7/public/miniconda3/bin/python remote_bench24.py \
  --eth /home/T7/dgut/robot_sim/eth_official_code/shuttle_detection/runs/final-model/best.pt \
  --ours stageA_best.pt --ours stageB_best_e10_stripped.pt --ours eth_only_v1_best.pt \
  --mem-gb 24 --out gpu_bench_a6000_24g_eth_only_v1.json
echo "=== done $(date -Is) ==="
