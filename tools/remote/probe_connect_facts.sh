#!/usr/bin/env bash
echo "hostname=$(hostname)"; echo "tailnet_ip=$(tailscale ip -4 2>/dev/null | head -1)"
echo "os=$( . /etc/os-release; echo $PRETTY_NAME )  kernel=$(uname -r)"
echo "whoami=$(whoami) sudo_nopass=$(sudo -n true 2>/dev/null && echo yes || echo no)"
echo "--- pythons"
for p in /home/T7/public/miniconda3/bin/python /home/T7/ojh/robot_sim/env_isaaclab/bin/python /home/T7/ojh/venv/bin/python /usr/bin/python3; do
  if [ -x "$p" ]; then echo "$p -> $( "$p" -c "import sys;print(sys.version.split()[0])" 2>&1 | head -1 )"; fi
done
echo "--- ultralytics import (miniconda + wheel extract)"
PYTHONPATH=/home/T7/ojh/robot_sim/experiments/yolo26_p2_ab/_wheel_extract /home/T7/public/miniconda3/bin/python -c "import ultralytics,torch;print('ultralytics',ultralytics.__version__,'torch',torch.__version__)" 2>&1 | tail -1
echo "--- key paths"
for d in /home/T7/dgut/robot_sim /home/dgut/.dsh-bench /home/T7/public/miniconda3; do [ -e "$d" ] && echo "OK $d"; done
echo "--- disk"
df -h / /home/T7 2>/dev/null | tail -2
echo "--- gpu"
nvidia-smi --query-gpu=name,memory.used,memory.total,utilization.gpu --format=csv,noheader
