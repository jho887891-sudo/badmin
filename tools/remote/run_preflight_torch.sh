#!/bin/bash
# Torch-only preflight for YOLO26S_V2_FULL_ETH_CONTINUATION_V1 (gates 1-3).
set -u
L=/home/dgut/.dsh-bench/cont_v1
W=/home/dgut/.dsh-bench/runs/detect/runs_eth_hardneg_v2/full_v2_e50/weights/best.pt
echo "=== start $(date -Is)"
echo "--- ultralytics runtime (wheel_extract dist-info):"
grep -h "^Version" /home/T7/ojh/robot_sim/experiments/yolo26_p2_ab/_wheel_extract/ultralytics-8.4.150.dist-info/METADATA 2>/dev/null | head -1
echo "--- torch-only preflight:"
timeout 300 /usr/bin/python3 "$L/preflight_cont_v1_torch.py" --weights "$W" --out "$L/yolo26s_v2_full_eth_cont_v1_environment.json"
echo "rc=$?"
ls -l "$L"/yolo26s_v2_full_eth_cont_v1_environment.json 2>/dev/null
echo "=== end $(date -Is)"
