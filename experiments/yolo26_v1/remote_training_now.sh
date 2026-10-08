#!/usr/bin/env bash
echo "== 1) 训练进程 =="
pgrep -af "[t]rain_yolo26_v1" || echo "  没有训练进程在跑"
echo "== 2) GPU =="
nvidia-smi --query-gpu=utilization.gpu,memory.used,memory.total --format=csv,noheader
nvidia-smi --query-compute-apps=pid,used_memory --format=csv,noheader | head -5
echo "== 3) runs 目录最近 5 条（只看顶层，不递归） =="
ls -lt /home/T7/ojh/robot_sim/runs/shuttle_yolo26_v1/ 2>/dev/null | head -6
echo "== 4) Stage B 结束状态（应仍是 71 行 = 70 epoch） =="
RU=/home/T7/ojh/robot_sim/runs/shuttle_yolo26_v1/gate6B_lr0001_musgd_b16_w8_20260927-235730
wc -l < $RU/results.csv; tail -n 1 $RU/results.csv | cut -d, -f1,6,7,8,9
echo "== 5) 是否出现比 Stage B 更新的 run =="
find /home/T7/ojh/robot_sim/runs/shuttle_yolo26_v1 -maxdepth 1 -newer $RU/results.csv 2>/dev/null | head -3 || true
echo "== 6) 顺带：远端 Harness 服务 =="
systemctl --user is-active dsh-web 2>/dev/null; systemctl --user is-enabled dsh-web 2>/dev/null
echo "== 7) 时间 =="; date -u