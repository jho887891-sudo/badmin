#!/usr/bin/env bash
echo "== gpu =="; nvidia-smi --query-gpu=utilization.gpu,memory.used --format=csv,noheader
echo "== tools =="; ls /home/T7/ojh/robot_sim/tools/ | tr "\n" " "; echo
echo "== outputs =="; ls /home/T7/ojh/robot_sim/outputs/shuttle_capability/ | tr "\n" " "; echo
echo "== venv =="; /home/T7/ojh/robot_sim/env_isaaclab/bin/python -c "import torch, ultralytics; print(torch.__version__, ultralytics.__version__, torch.cuda.is_available())"
echo "== dataset =="; ls -d /home/T7/dgut/robot_sim/eth_shuttle_detection >/dev/null && echo eth_ok
echo "== runs =="; ls /home/T7/ojh/robot_sim/runs/shuttle_yolo26_v1/ | tr "\n" " "; echo