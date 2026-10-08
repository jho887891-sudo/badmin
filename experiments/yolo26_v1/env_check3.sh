#!/usr/bin/env bash
echo "== venv =="
timeout 120 /home/T7/ojh/robot_sim/env_isaaclab/bin/python -c "import torch, ultralytics; print(torch.__version__, ultralytics.__version__, torch.cuda.is_available())" 2>&1 | tail -n 3
echo "== dataset =="; ls -d /home/T7/dgut/robot_sim/eth_shuttle_detection 2>&1
echo "== runs =="; ls /home/T7/ojh/robot_sim/runs/shuttle_yolo26_v1/ 2>&1 | tr "\n" " "; echo
echo "== gpu procs =="; nvidia-smi --query-compute-apps=pid,used_memory --format=csv,noheader 2>&1
echo "== metrics dir =="; ls /home/T7/ojh/robot_sim/outputs/shuttle_capability/metrics/ 2>&1 | head -n 20 | tr "\n" " "; echo