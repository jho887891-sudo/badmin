#!/usr/bin/env bash
# Remote launcher for the YOLO26 V1 trainer: fixes PYTHONPATH/YOLO_CONFIG_DIR and rewrites the
# manifest paths (built on Windows) to this host, then execs the trainer.
set -u
REPO=/home/T7/ojh/robot_sim
EX=$REPO/experiments/yolo26_p2_ab/_wheel_extract
DEPS=$REPO/experiments/yolo26_p2_ab/_deps
PY=$REPO/env_isaaclab/bin/python
export PYTHONPATH="$EX:$DEPS"
export YOLO_CONFIG_DIR=$REPO/.yolo_cfg
mkdir -p "$YOLO_CONFIG_DIR"
cd "$REPO" || exit 1
echo "RUNNER repo=$REPO python=$PY"
MAP1="D:\_eth_data\eth_shuttle_detection=/home/T7/dgut/robot_sim/eth_shuttle_detection"
MAP2="E:\具身智能\badmin_project=/home/T7/ojh/robot_sim"
exec "$PY" tools/train_yolo26_v1.py --map "$MAP1" --map "$MAP2" "$@"