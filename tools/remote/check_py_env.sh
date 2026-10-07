#!/bin/bash
for PY in /usr/bin/python3 /home/T7/public/miniconda3/bin/python; do
  echo "== $PY"
  timeout 90 "$PY" -c "import sys, PIL; from PIL import Image; print('py', sys.version.split()[0], 'PIL', PIL.__version__)" 2>&1 | tail -2
done
echo "--- ETH root dirs:"
ls -d /home/T7/dgut/robot_sim/eth_shuttle_detection/*/ | wc -l
ls -d /home/T7/dgut/robot_sim/eth_shuttle_detection/*/ | head -40
