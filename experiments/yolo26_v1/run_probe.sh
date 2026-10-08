#!/usr/bin/env bash
set -u
REPO=/home/T7/ojh/robot_sim
EX=$REPO/experiments/yolo26_p2_ab/_wheel_extract
DEPS=$REPO/experiments/yolo26_p2_ab/_deps
PY=$REPO/env_isaaclab/bin/python
ls -d "$EX" "$DEPS" "$PY"
cd "$REPO" || exit 1
export PYTHONPATH="$EX:$DEPS"
export YOLO_CONFIG_DIR=/tmp/yolo26v1_cfg
mkdir -p "$YOLO_CONFIG_DIR"
"$PY" experiments/yolo26_v1/probe_env.py