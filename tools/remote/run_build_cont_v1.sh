#!/bin/bash
# Full ETH continuation pool build (spec sections 5-7). Runnable twice: it resumes from the rows already written.
set -u
B=/home/dgut/.dsh-bench
L=$B/cont_v1
R=/home/T7/dgut/robot_sim/eth_shuttle_detection
echo "=== start $(date -Is)"
/usr/bin/python3 "$L/build_cont_v1_pool.py" --root "$R" \
  --out-manifest "$B/yolo26s_v2_full_eth_cont_v1_train_manifest.csv" \
  --out-audit "$B/yolo26s_v2_full_eth_cont_v1_data_audit.json" \
  --lists-dir "$B/data/yolo26s_v2_full_eth_cont_v1" \
  --val-stride 50 --workers 12
echo "rc=$?"
echo "=== end $(date -Is)"
