# SETUP — 仿真环境安装记录（Isaac Sim 6.0.1 + Isaac Lab v3.0.0-beta2.patch1）

> 状态：**✅ 安装完成并通过最小验证**（2026-09-08 UTC）。此文件是安装的唯一事实记录（本地 SSOT）。
> 本版由最终验证会话重写，消除此前草稿中的矛盾条目；所有版本号均为远端实测（pip list / import / nvidia-smi）。

## 目标版本 vs 实际安装
| 项 | 目标（用户指定） | 实际安装（实测） |
|---|---|---|
| Isaac Sim | 6.0.1 | ✅ `isaacsim==6.0.1.0`（isaacsim-core 6.0.1.0，pip 包） |
| Isaac Lab | v3.0.0-beta2.patch1 | ✅ git tag `v3.0.0-beta2.patch1`（detached HEAD 已确认）；包版本 isaaclab **6.1.14**（editable，`/home/T7/dgut/robot_sim/IsaacLab/source/isaaclab`） |
| Python | 3.12 | ✅ **3.12.13**（uv venv `env_isaaclab`，--seed） |
| PyTorch | 用户消息 2.11.0 / 0.26.0（cu128） | ✅ **torch==2.10.0+cu128 / torchvision==0.25.0+cu128 / torchaudio==2.10.0+cu128**（官方 pin，见 DEC-005；**本会话用户已确认走官方 2.10.0/0.25.0**） |
| uv | — | ✅ uv 0.11.9（`/home/dgut/.local/bin/uv`） |

## 机器配置（2026-09-08 实测，全部只读检查）
- **hostname**：jxxy（`dgut@172.31.68.251:22`，SSH key 认证可用）
- **Ubuntu**：22.04.5 LTS，x86_64（uname -m=x86_64 ✅）
- **GPU**：NVIDIA RTX A6000，**VRAM 49140 MiB**（compute 8.6）；实测被 vLLM 进程 PID 224065 常驻占用 ~24.8GB → 训练可用显存约 **24GB**
- **Driver**：550.163.01（nvidia-smi 显示 CUDA 12.4 上限；cu128 wheels 实际可跑，见测试结果）
- **RAM**：58GiB 总，可用 ~33GiB；**swap 17GiB 已用满**（vLLM 等占用，内存压力高）
- **磁盘**：
  - 根分区 `/`（LVM 280G）：**容量 100% 满（仅 356K）+ inode 100% 满（剩 873）** ← 一切运行期写入必须避开
  - `/home/T7`（/dev/vdb2，NTFS/FUSE fuseblk 7.1T）：安装前可用 1.6T；安装后 1.5T（robot_sim 已占 ~100GB+）
  - 门槛检查：Ubuntu x86_64 ✅ / GPU 驱动正常 ✅ / RAM≥32GB ✅（58G）/ 目标盘空闲≥150GB ✅（T7 1.5-1.6T，唯一可用大盘）
- **python3（系统）**：3.10.12（/usr/bin/python3，未改动）；conda 不存在；docker 28.3.1 存在未使用

## 安装目录（全部在 `/home/T7/dgut/robot_sim/`）
| 路径 | 用途 |
|---|---|
| `env_isaaclab/` | uv venv，Python 3.12.13（--seed） |
| `IsaacLab/` | git clone @ `v3.0.0-beta2.patch1`（真实 git 仓库，detached HEAD） |
| `cache/uv-cache` `cache/uv-python` `cache/xdg` `cache/tmp` `cache/pip` | uv/python/pip 缓存与临时目录（重定向，避开满盘根分区） |
| `home/` | **运行期 $HOME 重定向**（Kit/Omniverse 日志缓存不写满盘根分区） |
| `projects/` `logs/` `checkpoints/` | 项目源码/日志/checkpoint |
| `env.sh` | 复用环境脚本（source 后可直接跑 isaaclab；含全部重定向与 git 代理绕过） |

选 `/home/T7` 原因：根分区（含 /home/dgut）100% 满且 inode 满，`/home/T7` 是唯一 ≥150GB 可写空间（NTFS/FUSE 性能折衷已接受，见"问题"）。

## 安装方法（时间线摘要）
1. **uv venv**：`uv venv --python 3.12 --seed env_isaaclab` ✅
2. **git clone IsaacLab + checkout tag**：`git clone https://github.com/isaac-sim/IsaacLab.git` → `git checkout v3.0.0-beta2.patch1`（需 `-c http.proxy= -c https.proxy=` 绕过失效代理，ISSUE-001）；`git describe --tags` 与 `git status` 确认 = `v3.0.0-beta2.patch1` detached ✅
3. **Isaac Sim**：`uv pip install "isaacsim[all,extscache]==6.0.1.0" --extra-index-url https://pypi.nvidia.com --index-strategy unsafe-best-match --prerelease=allow` → **rc=0**（158+ 包，11:25→14:02，约 1h 下载 + NTFS 慢速解包）✅
4. **PyTorch 版本裁决**：IsaacLab tag 官方 pyproject.toml + pip_installation.rst 固定 `torch==2.10.0/torchvision==0.25.0/torchaudio==2.10.0`（cu128）→ DEC-005 采用官方 pin；**本会话用户裁决：官方 2.10.0/0.25.0** ✅
5. **torch 安装**：官方源 cu128 大包下载间歇失败（ISSUE-003，S3 AccessDenied/窗口性）；`retry_torch_run.sh` 长时重试最终 `RC_1=0`：torch 2.10.0+cu128 / torchvision 0.25.0+cu128 / torchaudio 2.10.0+cu128；验证 `TORCHVER 2.10.0+cu128 cuda 12.8 avail True` ✅
6. **`./isaaclab.sh -i`**：多轮失败后成功路径：
   - imgui 编译 `/tmp` ENOSPC → `TMPDIR/TMP/TEMP` 指向 T7（ISSUE-004 解决，isaaclab_teleop 构建通过）
   - robomimic@git clone 失败 → `GIT_CONFIG_COUNT` 空代理绕过（ISSUE-001 pip 内 git 版）→ robomimic 0.4.0 + isaaclab_mimic 装成 ✅
   - run5 完成 isaaclab_rl / isaaclab_tasks / isaaclab_tasks_experimental / isaaclab_visualizers / isaaclab_teleop ✅（过程多次被并发编排进程 SIGTERM，见"问题"）
   - 最终子模块就绪（editable，import 验证）：isaaclab/ppisp/assets/contrib/experimental/newton/ov/ovphysx/physx/rl/tasks/tasks_experimental/visualizers/teleop/mimic
   - **未装**：RL 训练框架 extras（rsl-rl / rl_games / skrl），extras 阶段未执行到；`import rsl_rl/rl_games/skrl` ModuleNotFoundError

## 测试结果（最终验证，2026-09-08 UTC 通过）
| 项 | 命令/依据 | 结果 |
|---|---|---|
| Python 启动 | `env_isaaclab/bin/python --version` | ✅ 3.12.13 |
| PyTorch+CUDA | `import torch` | ✅ torch **2.10.0+cu128**，cuda_build 12.8，`torch.cuda.is_available()=True`，GPU=`NVIDIA RTX A6000` |
| torchvision/torchaudio | import | ✅ 0.25.0+cu128 / 2.10.0+cu128 |
| import isaaclab | `import isaaclab` | ✅ `isaaclab OK 6.1.14` |
| Isaac Sim 引擎启动 | `isaaclab.sh -p ...`（AppLauncher） | ✅ Kit 启动完成（EULA 首次接受后；headless） |
| **官方最小测试** | `./isaaclab.sh -p scripts/environments/zero_agent.py --task Isaac-Cartpole-Direct-v0 --num_envs 128 --viz none`（本 tag 注册名带 **-v0**；headless 需 `--viz none`） | ✅ `[INFO] Completed setting up the environment...`；`Gym observation space: Box(-inf,inf,(128,4))`；`Gym action space: Box(-inf,inf,(128,1))`；GPU 显存 +~2.4GB、利用率 2-16% 持续 ~12min 无错误（zero-agent 无限循环由超时结束，RC=124 属预期） |
| GPU 无异常 | nvidia-smi 全程采样 | ✅ 无 CUDA/driver 报错（cu128 wheels 在 driver 550 上实测可运行） |

## 使用方法（重要）
```bash
source /home/T7/dgut/robot_sim/env.sh     # 一键设置：HOME/TMPDIR/XDG/UV/PIP 重定向 + git 代理绕过 + PATH/VIRTUAL_ENV
cd /home/T7/dgut/robot_sim/IsaacLab
# 例：128 环境 Cartpole Direct（headless）
./isaaclab.sh -p scripts/environments/zero_agent.py --task Isaac-Cartpole-Direct-v0 --num_envs 128 --viz none
# 例：未来训练（rsl-rl 装好后）
./isaaclab.sh -p scripts/reinforcement_learning/train.py --task Isaac-Cartpole-Direct-v0 --headless --num_envs 256
```
首次使用后 EULA 已接受（记录在 `robot_sim/home` 下）。**任务名必须带 `-v0`**（本 tag 无无后缀别名，实测 `Isaac-Cartpole-Direct` 找不到配置）。

## 问题与注意（重要）
1. **根分区 / 100% 满 + inode 100% 满**：Kit/pip/uv 任何写根分区路径都会失败 → 一律经 `env.sh`（HOME/TMPDIR/XDG_CACHE_HOME/UV_CACHE_DIR/PIP_CACHE_DIR 全部指向 /home/T7）。建议尽快清理根分区（~/.cache 17G、/snap 24G、docker 58G 等候选，需用户批准）。
2. **RL 训练框架未装**（rsl-rl / rl_games / skrl 缺失）：`./isaaclab.sh -i` 的 extra 阶段未执行到（多轮被并发 SIGTERM + github 间歇超时）。网络稳定后：`source env.sh && cd IsaacLab && bash isaaclab.sh -i`（幂等重跑补齐）；或 `uv pip install rsl-rl skrl`（PyPI）。
3. **github 间歇不可达**：`~/.gitconfig` 全局代理 127.0.0.1:10080 已死（ISSUE-001，未改用户配置）→ env.sh 内 `GIT_CONFIG_COUNT` 空代理绕过；大克隆仍可能瞬时超时，重试即可。
4. **并发编排进程互相 SIGTERM 对方的 `isaaclab.sh -i`**：本次安装因此多轮重跑（run3/run4/run5 日志在 `logs/`）。如再现，先停旧编排再跑 `-i`。
5. **显存**：vLLM（PID 224065，其他用户进程）常驻 ~24.8GB → 实际可用 ~24GB；大规模 RL 训练需协调。
6. **NTFS/FUSE 慢**：Kit 启动 ~4-6 分钟、pip/解包慢；属接受折衷（唯一大空间在 T7）。robot_sim 占用约 ~100GB（T7 可用 1.6T→1.5T）。
7. **未做/未改**：系统 CUDA toolkit（nvcc）未装未动；驱动未升级（550.163.01 实测可跑 cu128 wheels）；未删除任何既有软件。

## 相关记录
- 问题与解决：`management/ISSUES.md`（ISSUE-001~004）
- 版本裁决：`management/DECISIONS.md`（DEC-005，本会话用户确认官方 pin）
- 远端环境快照：`docs/deployment.md`、`management/DAILY_LOG.md`（2026-09-08）
- 远端运行脚本与日志：`/home/T7/dgut/robot_sim/logs/`（master/retry_torch_run/run3/run4/run5/verify/cartpole_128*.log 等）
