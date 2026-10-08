#!/usr/bin/env bash
echo "== 1) 是否还有训练进程 =="
pgrep -af "[t]rain_yolo26_v1" || echo "  没有训练进程在跑"
echo "== 2) GPU 现状 =="
nvidia-smi --query-gpu=utilization.gpu,memory.used,memory.total --format=csv,noheader
nvidia-smi --query-compute-apps=pid,used_memory --format=csv,noheader | head -6
echo "== 3) runs 目录（按时间，最近 6 个） =="
ls -lt /home/T7/ojh/robot_sim/runs/shuttle_yolo26_v1/ 2>/dev/null | head -10
echo "== 4) Stage B 收官状态 =="
RU=/home/T7/ojh/robot_sim/runs/shuttle_yolo26_v1/gate6B_lr0001_musgd_b16_w8_20260927-235730
echo "results.csv 行数: $(wc -l < $RU/results.csv)（含表头，即 epoch 数 = 行数-1）"
tail -n 2 $RU/results.csv | cut -d, -f1,6,7,8,9
ls -l --time-style=+%m-%d_%H:%M $RU/weights/ 2>/dev/null
echo "== 5) 是否出现过新的 run（比 Stage B 更新） =="
find /home/T7/ojh/robot_sim/runs/shuttle_yolo26_v1 -maxdepth 1 -newer $RU -type d 2>/dev/null | head -5 || echo "  无更新目录"
echo "== 6) 磁盘 =="; df -h /home/T7 | tail -1