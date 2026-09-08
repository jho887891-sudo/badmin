# 每日开发记录（DAILY_LOG）

> 每天结束前更新，新条目加在顶部。请勿只更新这里而不更新 PROJECT_STATUS / TODO。

---

# 2026-09-08

**今日目标：** 检查项目目录与当前状态，建立项目管理结构（本轮不写算法代码）

**完成：**
- 检查工作区现状：目录骨架（docs/management/experiments/runs、configs、scripts、src、tests、assets、tools）已存在但全空；Git 仓库已初始化但 0 commit（分支 master）
- 建立根状态文件：README / PROJECT_STATUS / TODO / CHANGELOG / ROADMAP
- 建立 docs/ 8 个设计文档种子 + management/ 5 个管理文件种子 + experiments/EXPERIMENT_INDEX.md
- 初始化 .gitignore；git 分支 master → main

**修改文件：** README.md、PROJECT_STATUS.md、TODO.md、CHANGELOG.md、ROADMAP.md、.gitignore、docs/×8、management/×5、experiments/EXPERIMENT_INDEX.md、各空目录 .gitkeep

**实验：** 无

**发现问题：** 工作区已有空目录骨架与空 git 仓库，但无任何文件/commit，也无环境与硬件记录

**解决问题：** 补齐知识库种子文件，作为后续一切工作的落盘基线

**未完成：** 环境盘点（GPU/CUDA/Isaac 版本、远端主机）、硬件盘点、Observation/Action v0.1

**明日第一件事：** 完成环境与硬件事实盘点（本地跑检查命令，登记到 docs/deployment.md 与 hardware_interface.md）

**Git commit：** a8bf4d1（chore: init project knowledge base structure，分支 main）

**远端运行任务：** 无

**Checkpoint：** 无

**（同日补充，约 18:26）**
- SSH 到远端开发机验证通过：dgut@172.31.68.251（hostname jxxy）；GPU = RTX A6000 49GB、驱动 550.163.01；Python 3.10.12；docker/nvidia-docker 可用；CUDA toolkit / Isaac Sim / Isaac Lab / conda 均未安装 → 事实已登记 docs/deployment.md
- 远端候选目录 /home/T7/ojh 已查看（含 YOLO 权重与实验目录），**尚未定为项目工作区**
- 关联 GitHub origin = https://github.com/jho887891-sudo/badmin（远程空仓库，用于备份/协同）
- 提交身份默认：jho887891-sudo <jho887891-sudo@users.noreply.github.com>（可改为真实姓名/邮箱）
- 补充 commit：（见 git log）

**（安装推进补充 2026-09-08）**
- 决定安装目标：/home/T7/dgut/robot_sim（根分区 100% 满+inode 满，唯一大空间=NTFS/FUSE 的 T7，1.6T 可用）
- 完成：uv venv Python 3.12.13；git clone IsaacLab @ v3.0.0-beta2.patch1（需 -c http.proxy= 绕过失效代理，ISSUE-001）
- 通道核验：uv 可直连 pypi.nvidia.com 解析 isaacsim==6.0.1.0（curl 301→.cn 为误导，ISSUE-002）；IsaacLab 官方兼容表 v3.0.0* ↔ Isaac Sim 6.0.1 ✓
- torch 版本裁决：IsaacLab 官方 pin torch==2.10.0/0.25.0 cu128（DEC-005）；用户消息 2.11.0/0.26.0 为偏差，已记录
- master 脱离脚本 11:25 启动（PID 2776157）：PHASE1 isaacsim 下载完成（158 包，~1h，CDN 峰值 3.8MB/s）；NTFS/FUSE 解包极慢（uv 线程 FUSE request_wait；ntfs-3g 长时间 I/O）；PHASE2/3 待跑
- 后台编排任务 pwsh-4：轮询至 MASTER DONE 后自动执行验证脚本（torch/cuda/isaaclab import/Cartpole 源检查）
