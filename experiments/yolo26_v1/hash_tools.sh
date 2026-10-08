#!/usr/bin/env bash
cd /home/T7/ojh/robot_sim/tools || exit 1
sha256sum eval_yolo26_v1.py eval_hard_negative.py compare_hard_negative_eval.py hard_negative_confidence_hist.py render_top_fp.py
