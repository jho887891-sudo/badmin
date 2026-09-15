#!/bin/bash
# 05_retrain_and_retest.sh - one failure-driven retrain, then re-measure every frozen protocol.
#
# Why this exists as one script: the retrain must be followed by the SAME protocols every time, on
# frozen sets that never enter training. Running them by hand is how a protocol silently drifts.
#
# Why the training step is monitored rather than piped through tail: the first version piped the
# trainer output through `tail -4`, which swallows everything until the command EXITS. On a degraded
# host where startup alone takes 35 minutes, that produced a job with no log, no GPU use and no CPU
# burn for half an hour - indistinguishable from a dead process, and it was killed six times before
# anyone measured the environment. The trainer now writes unbuffered to its own log and this script
# echoes progress every 30 s, so the job is observable while it runs and its exit status is preserved.
#
# Usage: bash 05_retrain_and_retest.sh <run-name>
set -e
RUN=${1:?usage: 05_retrain_and_retest.sh <run-name> [config-yaml]}
# Defaults to the declared baseline config. Pass baseline_cached.yaml when the host storage is too
# slow to decode 676 images per epoch; the two files differ ONLY in cache, which resolved_config.json
# records.
CONFIG=${2:-configs/shuttle_detection/baseline.yaml}
TGT=/home/T7/dgut/robot_sim/third_party/ultralytics
PY=/home/T7/ojh/robot_sim/env_isaaclab/bin/python
cd /home/T7/dgut/robot_sim
export PYTHONPATH=$TGT WANDB_MODE=disabled
TRAIN=outputs/shuttle_capability/train_data
CC=outputs/shuttle_capability/controlled_capability
OUT=outputs/shuttle_detection/training/$RUN
TRAINLOG=/tmp/${RUN}_train.log

echo "################ 1. TRAIN ################"
echo "  config: $CONFIG"
echo "  log: $TRAINLOG   (progress echoed here every 30 s)"
$PY -u scripts/shuttle_detection/train_baseline.py \
  --config $CONFIG \
  --train-manifest $TRAIN/manifest_train_full.csv \
  --val-manifest   $TRAIN/manifest_val_synthetic.csv \
  --data-root /home/T7/dgut/robot_sim \
  --run-name $RUN > "$TRAINLOG" 2>&1 &
TRAINPID=$!
while kill -0 $TRAINPID 2>/dev/null; do
  sleep 30
  line=$(tr "\r" "\n" < "$TRAINLOG" | grep -E "[0-9]+/100" | tail -1)
  ep=$(wc -l < "$OUT/results.csv" 2>/dev/null || echo 1)
  echo "  [train] epoch $((ep-1))/100   ${line:0:90}"
done
wait $TRAINPID
trainrc=$?
if [ $trainrc -ne 0 ]; then
  echo "  TRAINING FAILED (exit $trainrc); last 20 log lines:"
  tail -20 "$TRAINLOG"
  exit $trainrc
fi
echo "  training finished; last validation line:"
tr "\r" "\n" < "$TRAINLOG" | grep -E "^\s+all" | tail -1
W=$OUT/weights/best.pt

echo "################ 2. PROTOCOL A: controlled matrix (2070 rows) ################"
$PY -u scripts/shuttle_detection/evaluate_controlled.py --weights $W --manifest $CC/manifest.csv \
  --images $CC --out outputs/shuttle_detection/capability/rt_$RUN/controlled \
  --imgsz 640 --conf 0.05 --expected-class 0 --top-k 5 --device 0 2>&1 | grep -E "overall|Error"

echo "################ 3. PROTOCOL B: frozen P3 (160 real-background images) ################"
$PY -u scripts/shuttle_detection/evaluate_controlled.py --weights $W \
  --manifest outputs/shuttle_capability/metrics/p3_real_bg_manifest.csv \
  --images outputs/shuttle_capability/synthetic_on_real_bg/images \
  --out outputs/shuttle_detection/capability/rt_$RUN/p3 \
  --imgsz 640 --conf 0.05 --expected-class 0 --top-k 5 --device 0 2>&1 | grep -E "overall|Error"

echo "################ 4. PROTOCOL C: 16 verified real photographs ################"
$PY -u scripts/shuttle_detection/evaluate_controlled.py --weights $W \
  --manifest outputs/shuttle_capability/metrics/real_image_verified_manifest.csv \
  --images outputs/shuttle_capability/real_images/raw \
  --out outputs/shuttle_detection/capability/rt_$RUN/real \
  --imgsz 640 --conf 0.05 --expected-class 0 --top-k 5 --device 0 2>&1 | grep -E "overall|Error"

echo "################ 5. PROTOCOL D: 30 shuttle-free scenes (false positives) ################"
$PY -u /tmp/bg_fp_cmp.py 2>&1 | tail -4

echo "################ DONE: $RUN ################"
