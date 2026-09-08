# SETUP — 仿真环境安装记录（Isaac Sim 6.0.1 + Isaac Lab v3.0.0-beta2.patch1）

> 状态：**安装进行中**（2026-09-08）。此文件是安装的唯一事实记录（本地 SSOT），安装完成后补全"测试结果"。

## 目标版本（用户指定）
- Isaac Sim: 6.0.1
- Isaac Lab: v3.0.0-beta2.patch1（git tag，已确认存在于 isaac-sim/IsaacLab）
- Python: 3.12（IsaacLab pyproject 要求 >=3.12,<3.13 → 采用 3.12.13）
- torch/torchvision: 2.11.0 / 0.26.0（cu128，download.pytorch.org/whl/cu128）

## 机器配置（2026-09-08 实测）
- hostname: jxxy（dgut@172.31.68.251）
- Ubuntu: 22.04.5 LTS x86_64
- GPU: NVIDIA RTX A6000（VRAM 49140 MiB；实测空闲约 23.5GB，vLLM PID 224065 占用其余）
- Driver: 550.163.01（CUDA 12.4 上限）
- RAM: 58GiB（可用约 32GiB，swap 17G 已用满）
- 磁盘: / 根分区 280G 100%满(inode 也满) → 安装目标选 /home/T7（NTFS/FUSE 7.1T，可用 1.6T，dgut 可写）

## 安装目录（全部在 /home/T7/dgut/robot_sim/）
| 路径 | 用途 |
|---|---|
| env_isaaclab/ | uv venv Python 3.12.13（--seed） |
| IsaacLab/ | git clone v3.0.0-beta2.patch1（真实 git 仓库，3364 文件） |
| cache/uv-cache, uv-python, xdg | uv/python 缓存（重定向，避开满盘 /home） |
| projects/ logs/ checkpoints/ | 项目源码/日志/checkpoint |
- 选 /home/T7 原因：/home 所在 LVM 根分区 100% 满且 inode 满；/home/T7 是唯一 ≥150GB 可写空间（1.6TB）；NTFS/FUSE 性能折衷已接受

## 安装方法（进行中）
1. uv venv（uv 0.11.9，CPython 3.12.13）✅
2. git clone IsaacLab @ v3.0.0-beta2.patch1 ✅（需 `-c http.proxy=` 绕过失效代理，见 ISSUE-001）
3. 网络通道核验 ✅：uv 经 --extra-index-url https://pypi.nvidia.com 可解析 isaacsim==6.0.1.0（dry-run 166 包，见 ISSUE-002）；download.pytorch.org cu128 可达；pypi.org/TUNA 可达
4. torch 2.11.0+cu128 / torchvision 0.26.0+cu128 安装中…
5. isaacsim[all,extscache]==6.0.1.0 待装
6. ./isaaclab.sh -i 待执行
7. 最小验证（Cartpole 等）待执行

## 待解决问题
- [ ] Isaac Sim 大包下载耗时与磁盘占用监控（预计 40-60GB）
- [ ] torch 与 isaacsim 依赖的 torch 版本一致性（dry-run 显示 isaacsim 要求 torch==2.11.0）
- [ ] 根分区 inode 满：启动时需 export UV_CACHE_DIR/XDG_CACHE_HOME 到 /home/T7
