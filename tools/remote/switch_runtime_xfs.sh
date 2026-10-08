#!/bin/bash
# Switch the training runtime off the fuseblk mount: copy ultralytics to xfs and use the xfs python.
# Reason (measured 2026-10-07): FUSE per-request latency ~200 ms, bulk throughput fine (170 MB/s); importing
# torch from /home/T7/public/miniconda3 needs thousands of small reads and exceeds 120 s, while /usr/bin/python3
# + system torch imports in 11.7 s.
set -u
B=/home/dgut/.dsh-bench
echo "=== stopping the stuck smoke"
pkill -9 -f "train_eth_only_v1[.]py" 2>/dev/null
sleep 2
echo "alive=$(pgrep -fc 'train_eth_only_v1[.]py')"
echo "=== copying ultralytics (12 MB) to xfs"
rm -rf "$B/wheel_xfs"; mkdir -p "$B/wheel_xfs"
time cp -r /home/T7/ojh/robot_sim/experiments/yolo26_p2_ab/_wheel_extract/. "$B/wheel_xfs/"
echo "=== verifying the xfs runtime"
time env PYTHONPATH="$B/wheel_xfs" /usr/bin/python3 -c "import torch, ultralytics; print('torch', torch.__version__, 'cuda', torch.cuda.is_available(), torch.cuda.get_device_name(0)); print('ultralytics', ultralytics.__version__)" 2>&1 | tail -3
echo "=== cleaning stale run dirs"
rm -rf "$B/runs_cont_v1/smoke_e2" "$B/runs/detect/runs_cont_v1"
ls -d "$B/runs_cont_v1" 2>/dev/null || echo "no runs_cont_v1 yet"
