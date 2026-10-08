#!/usr/bin/env bash
H="$HOME"
echo "=== 训练/GPU ==="
pgrep -af "[t]rain_yolo26_v1" || echo "  无训练进程"
nvidia-smi --query-gpu=utilization.gpu,memory.used --format=csv,noheader
echo "=== 远端 Harness 服务 ==="
systemctl --user is-active dsh-web; systemctl --user is-enabled dsh-web
"$H/.dsh-web/url.sh" 2>/dev/null | head -2
echo "=== 时间 ==="; date -u