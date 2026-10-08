#!/bin/bash
E=/home/T7/dgut/robot_sim/eth_shuttle_detection
FILES=$(ls $E/ml_3_medium/images/train/*.jpg | head -16)
echo "=== 16 sequential single-file reads:"
time (for f in $FILES; do cat "$f" > /dev/null; done) 2>&1 | tail -3
echo "=== same 16 files with 16 parallel readers:"
time (echo "$FILES" | xargs -P 16 -I{} cat {} > /dev/null) 2>&1 | tail -3
echo "=== 16 label files sequential:"
LAB=$(ls $E/ml_3_medium/labels/train/*.txt | head -16)
time (for f in $LAB; do cat "$f" > /dev/null; done) 2>&1 | tail -3
echo "=== gpu util samples:"
for i in 1 2 3 4; do nvidia-smi --query-gpu=utilization.gpu --format=csv,noheader; sleep 1; done
echo "=== current iter rate from the log:"
tr '\r' '\n' < /home/dgut/.dsh-bench/cont_v1/smoke_xfs.log | tail -1
