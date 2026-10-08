#!/usr/bin/env bash
set -u
REPO=/home/T7/ojh/robot_sim
TMP=/tmp/yolo26v1
PY=$REPO/env_isaaclab/bin/python
LOG=/tmp/gates13.log
mkdir -p "$TMP/cfg"
[ -d "$TMP/_wheel_extract" ] || cp -r "$REPO/experiments/yolo26_p2_ab/_wheel_extract" "$TMP/"
[ -d "$TMP/_deps" ] || cp -r "$REPO/experiments/yolo26_p2_ab/_deps" "$TMP/"
[ -f "$TMP/yolo26s.pt" ] || cp "$REPO/assets/external/_staging/F_yolo/weights/yolo26s.pt" "$TMP/"
cd "$REPO" || exit 1
export PYTHONPATH="$TMP/_wheel_extract:$TMP/_deps"
export YOLO_CONFIG_DIR="$TMP/cfg"
{
  "$PY" "$TMP/gates_1_3_audit.py" --repo "$REPO" --scale s --imgsz 640 --weights "$TMP/yolo26s.pt"
  echo "PYRC=$?"
} > "$LOG" 2>&1
echo "--- tail of $LOG ---"
tail -c 3500 "$LOG"