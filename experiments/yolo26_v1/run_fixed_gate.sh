#!/usr/bin/env bash
cd /home/T7/ojh/robot_sim || exit 1
RUN=/home/T7/ojh/robot_sim/runs/shuttle_yolo26_v1/gate6B_lr0001_musgd_b16_w8_20260927-235730
env_isaaclab/bin/python tools/check_stage_b_gate.py \
  --run "$RUN" \
  --label formal_stage_b_epoch5_fixed_gate \
  --required-epochs 5 > /home/T7/ojh/robot_sim/gate5_fixed.json 2>/home/T7/ojh/robot_sim/gate5_fixed.err
echo "rc=$?"
echo "=== verdict ==="
grep -E "\"verdict\"|\"epochs_finished\"|\"collapse\"|catastrophic_metric_collapse|\"trend_collapse\"|\"memory_fail\"|\"hard_fail\"|\"recall_min\"|\"map50_min\"" /home/T7/ojh/robot_sim/gate5_fixed.json | head -20
echo "=== lr_limits ==="
env_isaaclab/bin/python -c "import json;d=json.load(open('/home/T7/ojh/robot_sim/gate5_fixed.json'));print(json.dumps(d['lr_limits'],indent=1));print('violations:',d['lr_violations']);print('mem:',json.dumps(d['memory']));print('mem_reasons:',d['memory_reasons'])"
tail -3 /home/T7/ojh/robot_sim/gate5_fixed.err