#!/bin/bash
B=/home/dgut/.dsh-bench
E=/home/T7/dgut/robot_sim/eth_shuttle_detection
echo "=== FUSE read throughput (200 MB from the ETH pool):"
timeout 120 bash -c "time cat $E/cab_1_easy/images/train/cab_1_0000*.jpg > /dev/null" 2>&1 | tail -4
echo "=== miniconda import torch (timeout 120s):"
time timeout 120 /home/T7/public/miniconda3/bin/python -c "import torch; print('torch', torch.__version__)" 2>&1 | tail -3
echo "=== system python3 import torch (control):"
time timeout 120 /usr/bin/python3 -c "import torch; print('torch', torch.__version__)" 2>&1 | tail -3
echo "=== wheel_extract size:"
du -sh /home/T7/ojh/robot_sim/experiments/yolo26_p2_ab/_wheel_extract 2>/dev/null | tail -1
echo "=== xfs free:"
df -h /home/dgut | tail -1
echo "=== load:"
uptime
