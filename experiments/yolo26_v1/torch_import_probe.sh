#!/usr/bin/env bash
cd /home/T7/ojh/robot_sim || exit 1
echo "loadavg: $(cat /proc/loadavg)"
df -h /home/T7 | tail -n 1
s=$(date +%s)
timeout 900 env_isaaclab/bin/python -c "import torch; print(torch.__version__)" 2>&1 | tail -n 2
echo "torch_import_rc=${PIPESTATUS[0]} elapsed=$(( $(date +%s) - s ))s"
s=$(date +%s)
timeout 600 env_isaaclab/bin/python -c "import ultralytics; print(ultralytics.__version__)" 2>&1 | tail -n 2
echo "ultra_import_rc=${PIPESTATUS[0]} elapsed=$(( $(date +%s) - s ))s"
echo "loadavg_end: $(cat /proc/loadavg)"