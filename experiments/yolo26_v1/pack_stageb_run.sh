#!/usr/bin/env bash
RU=/home/T7/ojh/robot_sim/runs/shuttle_yolo26_v1/gate6B_lr0001_musgd_b16_w8_20260927-235730
cd "$RU" || exit 1
time tar czf /tmp/sbrun.tgz results.csv lr_probe.jsonl mem_probe.jsonl lr_config.json resolved_config.yaml metrics.json args.yaml
ls -l /tmp/sbrun.tgz