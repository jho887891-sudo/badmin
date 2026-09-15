#!/bin/bash
# 05_retrain_and_retest.sh - one failure-driven retrain, then re-measure every frozen protocol.
#
# Why this exists as one script: the retrain must be followed by the SAME three protocols every time,
# on frozen sets that never enter training. Running them by hand is how a protocol silently drifts.
#
# Inputs (all on jxxy under /home/T7/dgut/robot_sim):
#   train_data/manifest_train_nearfield2.csv   round-2 large-target pool, already merged with the
#                                              round-1 positives and the original negatives
#   train_data/manifest_val_synthetic.csv      the unchanged validation split
# Usage: bash 05_retrain_and_retest.sh <run-name>
set -e
RUN=${1:?usage: 05_retrain_and_retest.sh <run-name>}
TGT=/home/T7/dgut/robot_sim/third_party/ultralytics
PY=/home/T7/ojh/robot_sim/env_isaaclab/bin/python
cd /home/T7/dgut/robot_sim
export PYTHONPATH=$TGT WANDB_MODE=disabled
TRAIN=outputs/shuttle_capability/train_data
CC=outputs/shuttle_capability/controlled_capability
OUT=outputs/shuttle_detection/training/$RUN

echo "################ 1. TRAIN ################"
$PY scripts/shuttle_detection/train_baseline.py \
  --config configs/shuttle_detection/baseline.yaml \
  --train-manifest $TRAIN/manifest_train_full.csv \
  --val-manifest   $TRAIN/manifest_val_synthetic.csv \
  --data-root /home/T7/dgut/robot_sim \
  --run-name $RUN 2>&1 | tail -4
W=$OUT/weights/best.pt

echo "################ 2. PROTOCOL A: controlled matrix (2070 rows) ################"
$PY scripts/shuttle_detection/evaluate_controlled.py --weights $W --manifest $CC/manifest.csv \
  --images $CC --out outputs/shuttle_detection/capability/rt_$RUN/controlled \
  --imgsz 640 --conf 0.05 --expected-class 0 --top-k 5 --device 0 2>&1 | grep overall

echo "################ 3. PROTOCOL B: frozen P3 (160 real-background images) ################"
$PY scripts/shuttle_detection/evaluate_controlled.py --weights $W \
  --manifest outputs/shuttle_capability/metrics/p3_real_bg_manifest.csv \
  --images outputs/shuttle_capability/synthetic_on_real_bg/images \
  --out outputs/shuttle_detection/capability/rt_$RUN/p3 \
  --imgsz 640 --conf 0.05 --expected-class 0 --top-k 5 --device 0 2>&1 | grep overall

echo "################ 4. PROTOCOL C: 16 verified real photographs ################"
$PY scripts/shuttle_detection/evaluate_controlled.py --weights $W \
  --manifest outputs/shuttle_capability/metrics/real_image_verified_manifest.csv \
  --images outputs/shuttle_capability/real_images/raw \
  --out outputs/shuttle_detection/capability/rt_$RUN/real \
  --imgsz 640 --conf 0.05 --expected-class 0 --top-k 5 --device 0 2>&1 | grep overall

echo "################ 5. PROTOCOL D: 30 shuttle-free scenes (false positives) ################"
$PY /tmp/bg_fp_cmp.py 2>&1 | tail -4

echo "################ DONE: $RUN ################"
