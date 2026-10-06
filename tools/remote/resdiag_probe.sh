#!/bin/bash
set -u
D=/home/dgut/.dsh-bench/resdiag
PY=/home/T7/public/miniconda3/bin/python
export PYTHONPATH=/home/T7/ojh/robot_sim/experiments/yolo26_p2_ab/_wheel_extract
cd "$D" || exit 3
echo "=== start $(date -Is)"
python3 - <<'EOF'
import sys, torch, ultralytics, PIL
print("py", sys.version.split()[0])
print("torch", torch.__version__, "cuda", torch.cuda.is_available(), torch.cuda.get_device_name(0))
print("ultralytics", ultralytics.__version__, "PIL", PIL.__version__)
EOF
exec "$PY" remote_tiny_resolution.py \
  --part-csv "$D/eth_tiny_part.csv" \
  --root /home/T7/dgut/robot_sim/eth_shuttle_detection \
  --weights /home/dgut/.dsh-bench/runs/detect/runs_eth_hardneg_v2/full_v2_e50/weights/best.pt \
  --resolutions 1024 1280 1536 \
  --conf-floor 0.01 --conf-op 0.25 --iou 0.7 --max-det 300 \
  --latency-reps 30 --device 0 \
  --out "$D/tiny_resolution.json"
