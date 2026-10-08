#!/usr/bin/env bash
echo "=== B proc ==="
pgrep -fa "[t]rain_yolo26_v1" | head -1 || echo NONE
RUN=$(ls -dt /home/T7/ojh/robot_sim/runs/shuttle_yolo26_v1/gate6B_fromA_*/ 2>/dev/null | head -1)
RUN=${RUN%/}
echo "RUN=$RUN"
echo "=== init weights actually used ==="
grep -a "pretrained" "$RUN/resolved_config.yaml" 2>/dev/null
echo "=== progress ==="
grep -a "1024:" "$RUN.log" 2>/dev/null | tail -1 | tr -d "\r" | cut -c1-170
echo "=== gpu ==="
nvidia-smi --query-gpu=memory.free,utilization.gpu --format=csv,noheader