#!/usr/bin/env bash
cd /home/T7/ojh/robot_sim || exit 1
env_isaaclab/bin/python tools/check_stage_b_gate.py \
  --run /home/T7/ojh/robot_sim/runs/shuttle_yolo26_v1/gate6B_lr0001_musgd_b16_w8_20260927-235730 \
  --label gate5 --required-epochs 5 > /home/T7/ojh/robot_sim/gate5.json 2>/home/T7/ojh/robot_sim/gate5.err
echo "rc=$?"
head -c 600 /home/T7/ojh/robot_sim/gate5.err 2>/dev/null