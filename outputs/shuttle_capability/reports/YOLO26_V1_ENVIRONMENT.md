# YOLO26 V1 环境记录（Spec §14）

日期：2026-09-21

## 1. 本次 Gate 1–3 审计执行环境（本地机）

| 项 | 值 |
|---|---|
| Python | 3.11.9（venv: `D:\_yolo26v1\venv`） |
| PyTorch | **2.14.0+cu126**（torchvision 0.29.0+cu126） |
| CUDA build | 12.6 |
| GPU | NVIDIA GeForce RTX 4060 Laptop GPU，8,188 MiB（审计时空闲 ~3,198 MiB） |
| 驱动 | 595.97（驱动报告 CUDA 13.2） |
| Ultralytics | **8.4.150**，官方 wheel 解包至 `D:\_yolo26v1\_wheel_extract`（未安装） |
| 额外依赖 | ultralytics-thop → `D:\_yolo26v1\_deps`；opencv-python / matplotlib / pandas / seaborn / psutil / py-cpuinfo / pyyaml / requests / scipy / tqdm 安装在 venv |
| 运行方式 | `PYTHONPATH=_wheel_extract;_deps`，`YOLO_CONFIG_DIR=D:\_yolo26v1\cfg` |

## 2. 远端 GPU 机（训练目标机）实测状态

| 项 | 值 |
|---|---|
| 主机 | `jxxy`（dgut@172.31.68.251） |
| GPU | NVIDIA RTX A6000，49,140 MiB；**审计时 40,207 MiB 已被占用、利用率 99%** |
| 项目 venv | `/home/T7/ojh/robot_sim/env_isaaclab/bin/python`（torch 2.10.0+cu128） |
| Ultralytics | 远端未安装；使用解包 wheel `/home/T7/ojh/robot_sim/experiments/yolo26_p2_ab/_wheel_extract`（8.4.150） |
| 存储 | `/home/T7` = `/dev/vdb2` NTFS(fuseblk) 7.1T，剩 1.3T；`/`（含 /tmp）仅剩 4.0 GB |
| 磁盘实测 | 审计期间 `vdb` 利用率 98.9%、await 364 ms（另一用户的 4B 模型训练正在读写同一卷） |

## 3. 已确认的环境风险

1. **远端共享资源竞争**：另一用户正在跑 4B 参数训练（`4B修复版.py`，99% GPU、40 GB 显存），`/home/T7` NTFS 被其 IO 打满。我的审计进程被内核阻塞在 FUSE `request_wait_answer`，3 分钟只用了 9 秒 CPU → 因此 Gate 1–3 改在**本地**执行（结论与远端一致，因为 ultralytics 版本、权重 sha256 相同）。
2. **不得干扰他人任务**：全程只用 `nvidia-smi` 只读查询；未 kill 任何非本进程。
3. **`/` 仅剩 4.0 GB**：训练产物（best.pt / last.pt / 数据集）必须写 `/home/T7`，不得写 `/tmp`。