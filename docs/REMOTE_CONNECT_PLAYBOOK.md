# 远端（jxxy）连接与文件传输手册 —— 给另一个窗口/新会话用

> 实测时间：2026-10-03 07:0x–07:2x UTC（北京 15:0x–15:2x）｜记录人：算法窗口 A（ETH-only V1 训练）
> 全部命令都是**本机实测可用**的；标"未验证"的我都注明了。远端真源目录：`/home/T7/dgut/robot_sim/`。
> 本仓库路径：`docs/REMOTE_CONNECT_PLAYBOOK.md`（另一个窗口直接读这个文件 + 第 8 节的可粘贴块即可接入）。

---

## 0. 一句话

```bash
ssh dgut@jxxy.taildd42cc.ts.net          # Tailscale MagicDNS，当前唯一可用入口
ssh jxxy.taildd42cc.ts.net               # 等价（本机 ~/.ssh/config 已有 Host 别名，User dgut）
```

不要再用校园网 IP `172.31.68.251`（离开校园网即不可达，之前"TCP 22 超时"就是这个原因，与磁盘/显存无关）。

## 1. 入口与现状（实测）

| 项 | 值 |
|---|---|
| 入口 A | `dgut@jxxy.taildd42cc.ts.net`（MagicDNS） |
| 入口 B | `dgut@100.88.178.19`（tailnet IPv4） |
| 本机 ssh | OpenSSH_for_Windows_9.5p2，`~/.ssh/id_rsa` 已存在 → **免密**（无密码提示） |
| 本机 ssh config | **已存在** `Host jxxy.taildd42cc.ts.net` + `User dgut`（可省 `dgut@`） |
| 远端主机 | hostname `jxxy`，Ubuntu 22.04.5 LTS，kernel 5.15.0-191-generic |
| 远端身份 | `dgut`（uid 1000，**在 sudo 组但 `sudo -n` 不可用** → 全程不能 sudo/apt） |
| 单次调用耗时 | `ssh -o ConnectTimeout=15 ... date -Is` 完整往返 **11.7 s**（含握手+登录 shell）；后续查询类调用数秒级 |
| 连接质量 | Tailscale `active`（本机 100.79.176.111）；手机 `x70` offline 18h |

## 2. 连通性自检（30 秒，照抄）

```powershell
ssh -o ConnectTimeout=15 dgut@jxxy.taildd42cc.ts.net "date -Is; hostname; uptime | tr -s ' '"
# 期望：远端 ISO 时间 + jxxy + up X days, load average ...
ssh dgut@jxxy.taildd42cc.ts.net "nvidia-smi --query-gpu=name,memory.used,memory.total,utilization.gpu --format=csv,noheader"
# 期望：NVIDIA RTX A6000, <used> MiB, 49140 MiB, <util> %
scp dgut@jxxy.taildd42cc.ts.net:/home/dgut/.dsh-bench/build_v1b_run.log .
# 期望：下载成功（证明 scp 双向可用）
```

## 3. 传文件：只有一条可靠路径 = **write(LF) → scp → ssh "bash 脚本"**

### 3.1 正确套路（本轮用了 8 次，全部成功）

```text
1) 用 write 工具在本地写脚本（write 产出 LF 换行，这点很关键）
2) scp 本地文件 → 远端目录
3) ssh "bash /远端/路径.sh" 执行；参数直接写进脚本，不要在 ssh 命令行里拼
```

```powershell
# 例：上传脚本并执行
scp tools/remote/probe_space.sh dgut@jxxy.taildd42cc.ts.net:/home/dgut/.dsh-bench/
ssh dgut@jxxy.taildd42cc.ts.net "bash /home/dgut/.dsh-bench/probe_space.sh"

# 例：拉回产物（先算 sha256 再比对，别只看文件在不在）
scp dgut@jxxy.taildd42cc.ts.net:/home/dgut/.dsh-bench/eth_only_v1_data_audit.json outputs/shuttle_capability/metrics/
```

### 3.2 四种会炸的写法（都真实发生过）

| 写法 | 真实报错 | 原因 | 正确做法 |
|---|---|---|---|
| 本地脚本按 CRLF 落盘后管给远端 `bash -s` | `unexpected end of file`（行号常指向 `done`/`fi`） | bash 把 `done\r` 当成命令名 | 脚本必须 LF；用 write 工具生成，不要用 Windows 记事本/Out-File |
| `[Convert]::ToBase64String(...)` 结果过 stdout 再 `base64 -d` | `base64: invalid input` | 超长单行被终端/采集按宽度**折行**，解码器看到换行 | **永远不要把 base64 走控制台**；要传文件就用 scp |
| `ssh host "多行命令"` | 命令错乱、`$?` 变成 `True`、路径被截断 | PowerShell **先在本地**展开双引号里的 `$变量`、`$(...)`、`$?` | 写脚本文件再执行；必须内联时用单引号或转义 `\$` |
| `for ... do ... done` 直接塞进 ssh 一行 | `unexpected end of file` / 循环体被吞 | 同一行的引号嵌套 + CRLF 叠加 | 同上，落成 .sh |

### 3.3 如果 scp 真的不可用

本机是 Windows PowerShell，**不支持 `<` 输入重定向**，且 `Get-Content | 原生程序` 在 PS 5.1 下按字符串管道会破坏二进制内容 —— 所以不要试图"用 stdin 灌文件"。替代顺序：① 先查 `scp` 是否可用（`where.exe scp`）；② 用 git-bash/WSL 的 `ssh host "cat > /path" < file`（**本轮未验证**）；③ 任何方法传完都必须 `sha256sum` 两端比对。

## 4. 跑长任务（这是之前"日志 0 字节、进程消失"的根因）

```bash
# 正确：脱离父会话（setsid）+ 关闭 stdin + 重定向日志，全部在一行
cd /home/dgut/.dsh-bench && setsid nohup bash run_full_v1.sh full_e50 > full_e50_run.log 2>&1 < /dev/null &

# 立刻验证存活 + 有输出（只看 pgrep 不够，要看到日志在长）
sleep 30; pgrep -af train_eth_only_v1.py | head -3; tail -c 300 full_e50_run.log | tr "\r" "\n" | tail -2
```

要点：
- 不要用裸 `nohup cmd &`：父 ssh 断开后被连带杀掉，表现为 **日志 0 字节 + 进程消失**。
- 训练类日志充满 `\r`（进度条），`tail` 前先 `tr "\r" "\n"`，否则只能看到一坨。
- `pgrep -f` 的匹配串会命中你自己的 `bash -c` 包装行，**别把"匹配到自己"当成"任务还活着"**，要配合日志时间戳/文件 mtime 判断。

## 5. 远端环境速查（实测，别猜）

| 用途 | 解释器 / 值 |
|---|---|
| 通用（**有 pandas**，推荐） | `/home/T7/public/miniconda3/bin/python` → Python **3.13.2** |
| Isaac Lab | `/home/T7/ojh/robot_sim/env_isaaclab/bin/python` → 3.12.13 |
| 另一套 | `/home/T7/ojh/venv/bin/python` → 3.10.12（**无 pandas**，别用它跑数据脚本） |
| 系统 | `/usr/bin/python3` → 3.10.12 |
| ultralytics | `PYTHONPATH=/home/T7/ojh/robot_sim/experiments/yolo26_p2_ab/_wheel_extract` + miniconda python → **ultralytics 8.4.150 / torch 2.10.0+cu128** |
| GPU | RTX A6000，49140 MiB 总量；本窗口训练进程 6.0–6.7 GB（自设 24 GiB 单进程上限） |
| 磁盘 | `/`（xfs 280 G）**仅剩 ~9 GB（97%）**；`/home/T7`（NTFS 7.1 T）剩 **1.2 T** |
| 大资产真源 | `/home/T7/dgut/robot_sim/`（图像池、数据集、`*.pt`、ETH 官方仓库） |
| 临时工作区 | `~/.dsh-bench/`（脚本、checkpoint、日志；在 **`/` 分区**，注意 9 GB 余量） |
| 权限 | 无 sudo/apt；pip 装包需 `--user` 或直接放 `PYTHONPATH` |
| 第三方上报 | 训练必须 `WANDB_MODE=disabled`（远端装了 `wandb`，ultralytics 会自动上传产物） |

**磁盘铁律**：渲染/训练/中间产物**写 `/home/T7`**（1.2 T），不要写 `/` 或 `~`（9 GB）。`/home/dgut` 就在 `/` 分区上。

## 6. 只读勘察清单（给 Isaac 渲染准备用，全都安全）

```bash
df -h / /home/T7
nvidia-smi --query-gpu=memory.used,memory.total,utilization.gpu --format=csv,noheader
ls -d /home/T7/ojh/robot_sim/env_isaaclab/bin/python
cat /home/T7/ojh/robot_sim/env.sh 2>/dev/null | head -30   # kit/omni 缓存重定向到这里看
du -sh /home/dgut/.dsh-bench 2>/dev/null                  # 本窗口占用（当前 397 MB）
```

## 7. 可回收空间（**需要用户点头才动，我不会自行删**）

| 目标 | 大小 | 备注 |
|---|---|---|
| `~/.trae-server` | 3.6 G | Trae 远程 server 多版本；目录内还混有 `bin/ ckg_server/ extensions/ ai-agent/` 等组件目录，**必须先分类** |
| `~/.trae-cn-server` | 6.2 G | 同上 |
| `/tmp/dsh_sessions.tgz` | 214 MB | 9月29日旧包 |
| `/tmp/dshmem-1790703316/` | 223 MB | 旧会话记忆 |
| 合计 | **~10.2 G** | 清完 `/` 从 9.0 G → ~19 G |

另外 `/home/dgut/pusht_runtime` **59 G**、`yuwang` 15 G、`yuwang2` 13 G 是别的工作的大头，与本项目无关。

## 8. 给隔壁窗口 agent 的可粘贴说明

```text
远端主机：dgut@jxxy.taildd42cc.ts.net（Tailscale MagicDNS；tailnet IP 100.88.178.19）。
免密登录已配好（~/.ssh/id_rsa + ~/.ssh/config 里已有 Host jxxy.taildd42cc.ts.net）。
禁止使用校园网 IP 172.31.68.251（不可达）。无 sudo/apt。

传文件只走：本地 write(LF) → scp → ssh "bash /path/x.sh"。
禁止：base64 过控制台、CRLF 脚本喂 bash -s、ssh 命令行里塞多行/$(...)/$?。

长任务：cd <dir> && setsid nohup bash x.sh > x.log 2>&1 < /dev/null &，30 秒后查 pgrep + 日志时间戳。

解释器：数据类用 /home/T7/public/miniconda3/bin/python（3.13.2，有 pandas）；
Isaac 用 /home/T7/ojh/robot_sim/env_isaaclab/bin/python；ultralytics 要加
PYTHONPATH=/home/T7/ojh/robot_sim/experiments/yolo26_p2_ab/_wheel_extract。

磁盘：/ 只剩 ~9 GB，/home/T7 剩 1.2 T。大文件/渲染输出一律写 /home/T7，绝不写 / 或 ~。
当前 GPU 有本窗口的训练任务（RTX A6000，占用 6-7 GB，24 GiB 上限），请勿 kill 他人进程、勿改 WANDB 以外无关配置；
如需 GPU 请按自己的显存预算跑，并设置 WANDB_MODE=disabled。
产物/结论要能自证：给出路径、张数或 sha256。删除远端文件前必须先问用户。
```

## 9. 不要做的事

1. 不要 `sudo`（不接受密码，也没必要）。
2. 不要动 `/home/T7/dgut/robot_sim/` 下的既有资源（图像池/数据集/权重是别轮实验的真源）。
3. 不要覆盖既有产物文件（远端/本地都是"新文件优先"，覆盖前先改名备份）。
4. 不要 kill 不属于自己的进程（本窗口训练正在跑：`full_e50`，预计 11:35Z 结束）。
5. 不要把凭据/密码写进日志或仓库。
