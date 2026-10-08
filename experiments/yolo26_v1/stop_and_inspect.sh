#!/usr/bin/env bash
P=$(pgrep -f "[t]rain_yolo26_v1" | tr "\n" " ")
echo "stopping: $P"
[ -n "$P" ] && kill -INT $P && sleep 20
P2=$(pgrep -f "[t]rain_yolo26_v1" | tr "\n" " ")
[ -n "$P2" ] && kill -TERM $P2 && sleep 10
P3=$(pgrep -f "[t]rain_yolo26_v1" | tr "\n" " ")
[ -n "$P3" ] && kill -KILL $P3
echo "remaining=$(pgrep -f "[t]rain_yolo26_v1" | wc -l)"
B=/home/T7/ojh/robot_sim/runs/shuttle_yolo26_v1/gate6B_fromA_imgsz1024_b16_w8_20260927-130317
echo "=== FAILED RUN preserved ==="
ls -l "$B" | head -12
ls -l "$B/weights" 2>/dev/null
echo "=== FAILED RUN results.csv ==="
cat "$B/results.csv" 2>/dev/null
echo "=== Stage A checkpoint metadata ==="
cd /home/T7/ojh/robot_sim && PYTHONPATH=$PWD/experiments/yolo26_p2_ab/_wheel_extract:$PWD/experiments/yolo26_p2_ab/_deps env_isaaclab/bin/python experiments/yolo26_v1/inspect_checkpoint.py 2>&1 | tail -25