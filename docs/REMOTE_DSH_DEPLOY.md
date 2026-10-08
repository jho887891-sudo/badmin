# 远程 DeepSeek Harness 部署记录（jxxy）

> 日期：2026-09-29　｜　目标机：`dgut@172.31.68.251`（hostname `jxxy`，Ubuntu 22.04.5 LTS，x86_64）
> 原则：**用户级安装、无 sudo、不装 VPN、不改防火墙、不覆盖已有配置、只绑回环**。

---

## 1. 结果速览

| 项 | 值 |
|---|---|
| 安装位置 | `~/.local/lib/node_modules/@deepseek-ai/dsh`（npm 全局前缀 `~/.local`） |
| 可执行入口 | `~/.local/bin/dsh`（**自建 wrapper**，非 npm 软链，原因见 §5） |
| 版本 | **`@deepseek-ai/dsh 0.2.0-rc.2`**（registry `latest`） |
| Node | **v22.23.3**（`~/.nvm/versions/node/v22.23.3`，本次由 nvm 新装） |
| 启动命令 | `~/.dsh-web/start.sh` → 等价于 `cd ~ && ~/.local/bin/dsh web --no-open --host 127.0.0.1 --port 3080` |
| 监听 | **`127.0.0.1:3080`（仅回环，未监听 0.0.0.0，未改防火墙）** |
| 访问 URL | `http://127.0.0.1:3080/?token=<每次启动生成的 token>`（token 打印在 `~/.dsh-web/dsh-web.log`） |
| 鉴权 | 不带 token 访问 `/` → **401**；带 token → 303（种 cookie）→ **200**，页面含 `<title>DeepSeek Harness</title>` 与 `__DSH_BOOT__` |
| 开机自启 | **无**：前台用户进程，重启/重登后需手动 `start.sh`（如需常驻见 §6，需另行批准） |

## 2. 部署前勘察（实测）

| 项 | 结果 |
|---|---|
| OS / 内核 / glibc | Ubuntu 22.04.5 LTS / 5.15.0-191 / glibc 2.35 |
| 身份 | `dgut`（uid 1000；在 sudo 组，但 **`sudo -n` 需要密码** → 全程不用 sudo） |
| 已有 Node | **无系统 node/npm/npx**；但 `~/.nvm` 已存在，含 **v18.17.0（default）、v20.19.0** |
| 其他依赖 | git 2.34.1 ✓、python3 3.10.12 ✓、curl/wget/tar/xz ✓（jq 缺，不需要） |
| 网络出口 | nodejs.org 200、registry.npmjs.org 200、npmmirror 200（无 proxy 环境变量） |
| 端口 | 3080 空闲；22 已监听 |
| 既有 DSH | 无（`~/.dsh` 不存在） → **没有覆盖任何已有安装/配置** |
| 磁盘 | `/` 280 G，部署前 13 G 可用（96%），部署后 12 G 可用 |

## 3. 实际执行的命令（可逐步复现）

```bash
# 1) 装 Node 22（用户级；不动 nvm default，仍是 18.17.0）
export NVM_DIR="$HOME/.nvm"; . "$NVM_DIR/nvm.sh"; nvm install 22

# 2) 用 Node 22 安装 DSH 到 ~/.local 前缀（必须让 node 22 在 PATH 最前）
NODE22="$HOME/.nvm/versions/node/v22.23.3/bin"
PATH="$NODE22:$PATH" npm install -g --prefix "$HOME/.local" @deepseek-ai/dsh

# 3) 用 wrapper 覆盖 npm 软链（node 固定为 22，见 §5）
#    内容见 ~/.local/bin/dsh

# 4) 启动（仅回环）
~/.dsh-web/start.sh
```

## 4. Node 版本是硬约束（踩坑 1）

- 用 nvm 默认的 **Node 18.17.0** 安装时，postinstall 直接失败：
  `TypeError: (intermediate value).resolve is not a function`（`import.meta.resolve` 在 18 上不可用）
- 换 **Node 20.19.0** 能装完，但 npm 报出 **EBADENGINE**：`@deepseek-ai/libreoffice-kit@0.1.2`、`@earendil-works/pi-telemetry@0.87.1` 要求 `node >=22.19.0`，`which-command@0.1.0` 要求 `>=22`。
- 换 **Node 22.23.3** 后：`added/changed 541 packages`，**EBADENGINE 计数 = 0**，`dsh --version` = `0.2.0-rc.2`。
- 注意：`npm` 在 nvm 下是 shell 包装，会从 **PATH** 取 node —— 只写 `$NODE_BIN/npm` 而不改 PATH 时，实际仍会用 nvm 当前的 node（第一轮就是这样误用了 18.17.0）。

## 5. 覆盖事件与修复（踩坑 2，已完全恢复）

- npm 建的是**软链** `~/.local/bin/dsh -> ../lib/node_modules/@deepseek-ai/dsh/lib/bin.js`。
- 第一次写 wrapper 时用了 `cat > ~/.local/bin/dsh`，**穿透软链把 CLI 的 `lib/bin.js` 覆盖成了 shell 脚本**，于是报 `SyntaxError: Invalid or unexpected token`。
- 修复：`npm install -g --prefix ~/.local @deepseek-ai/dsh`（npm 重写包内所有文件，bin.js 复原，`--version` 恢复正常），再用 `mv -T` **原子替换软链本身**（不删除其它文件）。
- 备份/回滚：包装脚本内容就是两行路径，删掉后重跑 `npm install -g` 即可回到 npm 软链形态。

```bash
# ~/.local/bin/dsh （wrapper，2026-09-29 建立）
#!/usr/bin/env bash
# dsh wrapper: pinned to Node 22 because this host nvm default is v18.17.0 while
# @deepseek-ai/dsh requires node >=22.19.0.
exec "/home/dgut/.nvm/versions/node/v22.23.3/bin/node" \
     "/home/dgut/.local/lib/node_modules/@deepseek-ai/dsh/lib/bin.js" "$@"
```

**为什么必须用 wrapper**：本机 `nvm alias default = 18.17.0`，登录 shell 里 `node` 解析到 18（实测 `dsh` 走 npm 软链会报 `/usr/bin/env: ‘node’: No such file or directory` / 版本不符）。wrapper 把 Node 22 写死，**不改用户的 nvm 默认版本**。

## 6. 运行与访问

```bash
# 启动 / 查看 URL（不会重复启动；已运行则只打印现有 URL）
~/.dsh-web/start.sh
# 停止
~/.dsh-web/stop.sh
# 取当前 token
grep -o "http://127.0.0.1:3080/?token=[A-Za-z0-9_-]*" ~/.dsh-web/dsh-web.log | tail -1
```

**本机访问（SSH 端口转发，推荐；不改防火墙、不暴露公网）**：

```bash
# 本机 3080 已被本地 Harness 占用，所以用 3081 映射远程 3080
ssh -f -N -o BatchMode=yes -L 3081:127.0.0.1:3080 dgut@172.31.68.251
# 然后浏览器打开（token 每次启动都会变，需现取）
#   http://127.0.0.1:3081/?token=<token>
```

实测：带 token → HTTP 200 / 34,846 B / `<title>DeepSeek Harness</title>` / `__DSH_BOOT__` 命中 1 次；不带 token → 401。

**开机/重登后**：`dsh web` 是前台用户进程，会随会话结束而停止 → **需要重新执行 `start.sh`**（token 也会变）。若要常驻/自启，可选 `systemd --user` + `loginctl enable-linger`（可能需 sudo）——**需用户批准后再做**。

## 7. 未做/需批准的事项

| 事项 | 状态 |
|---|---|
| 安装 Tailscale（远端未装）用于私网访问 | **未做**（用户规则：装 VPN 需先问） |
| 修改防火墙 / 端口对外暴露 | **未做**（只监听回环） |
| systemd --user 常驻 + linger 开机自启 | **未做**（涉 sudo/常驻，需批准） |
| 清理 npm 缓存 `~/.npm/_cacache`（约数百 MB） | **未做**（删除类动作需批准）；当前 `/` 剩余 12 G |
| 在 Web UI 里填 DeepSeek API Key | **未做**（需用户自己的 key；设置 → 模型，保存即生效，无需重启） |

## 8. 复现/校验清单

| 校验点 | 期望 |
|---|---|
| `bash -lc "command -v dsh; dsh --version"` | `~/.local/bin/dsh` 与 `0.2.0-rc.2` |
| `ss -ltn \| grep 3080` | `127.0.0.1:3080`（**不得**是 `0.0.0.0:3080`） |
| `curl -o /dev/null -w "%{http_code}" http://127.0.0.1:3080/` | `401` |
| 同上加 `?token=…` 并跟随跳转 | `200` 且页面含 `DeepSeek Harness` |
| `pgrep -af "[b]in.js web"` | 1 个 node 进程 |
---

## 9. 远端 Harness 的「本地记忆」（2026-09-29 同步）

DSH 的「记忆」由三层构成，本次按层搬运，**密钥层明确排除**：

| 层 | 内容 | 处理 |
|---|---|---|
| ① 工作区记忆 | 项目文本（`AGENTS.md`、`management/**`、`docs/**`、`configs/**`、`tools/**`、`experiments/**`、`src/**`、`tests/**`、`deliverables/**`、`outputs/**` 的 md/csv/json/txt/yaml）+ 困难负样本评估的 21 张证据图 | **已同步**到远端 `/home/T7/ojh/badmin_project`（3,864 文件 / 29 MB；压缩包 5.91 MB） |
| ② Harness 记忆 | `~/.dsh/.agent-presets/badmin-ptc/`（自定义预设 17,241 + 374 B）、`~/.dsh/settings.yaml`（模型路由，1,237 B，**不含明文密钥**） | **已同步**到远端 `~/.dsh/`（远端此前无这两项，属新增而非覆盖） |
| ③ 会话记忆 | `~/.dsh/sessions/` 136 MB（11+7 个 session 的对话记录）+ `attachments/` 87.5 MB + `storages/workspace.json` 索引 | **未同步**（按本地 Windows 路径编码，远端工作区路径不同；且体积 224 MB）——需用户决定 |
| ⛔ 密钥层 | `~/.dsh/.credentials.yaml`（含 `DEEPSEEK_API_KEY` 记录与浏览器会话 secret） | **绝不复制**；远端用自己的 key（在 Web UI 设置 → 模型里填） |

### 9.1 同步命令（可复跑）

```powershell
# 本地 E:\具身智能\badmin_project -> 远端 /home/T7/ojh/badmin_project
powershell -File tools\sync_dsh_memory_to_remote.ps1
```

排除规则：`_scratch_*`、`.git`、`env_isaaclab`、`IsaacLab`、`node_modules`、`assets`（21 MB，属资源非记忆）、`__pycache__`、`tools/relay.log`（27 MB 日志）；图像**只**包含 `outputs/shuttle_capability/hard_negative_eval/**` 的 21 张证据图（其余图像池是资源，体积 2 GB+）。

### 9.2 远端工作区已指向记忆树

`~/.dsh-web/start.sh` 现在以 `DSH_WORKSPACE`（默认 `/home/T7/ojh/badmin_project`）为工作目录启动，DSH 会把启动目录作为默认文件系统位置：

```bash
# 远端实测
readlink /proc/$(pgrep -f "[b]in.js web" | head -1)/cwd   # -> /home/T7/ojh/badmin_project
```

### 9.3 当前访问信息（2026-09-29 17:29 UTC 重启后）

| 项 | 值 |
|---|---|
| 远端进程 | PID 890921（`cwd = /home/T7/ojh/badmin_project`） |
| 远端地址 | `http://127.0.0.1:3080/?token=T6--DADRoTiiJQiIpyPRL-TQCfDbnQFRboZZm_gkRbE` |
| 本机隧道 | 本地 3081 → 远端 3080（`ssh -f -N -L 3081:127.0.0.1:3080 dgut@172.31.68.251`） |
| 实测 | 无 token → 401；带 token → 200（34,846 B，`<title>DeepSeek Harness</title>`，`__DSH_BOOT__` 命中） |

> token 每次重启都会变；取当前值：`ssh dgut@172.31.68.251 'grep -o "token=[A-Za-z0-9_-]*" ~/.dsh-web/dsh-web.log | tail -1'`

### 9.4 远端存在两棵项目树（需用户决定是否合并）

| 树 | 用途 | 对本地记忆的时效 |
|---|---|---|
| `/home/T7/ojh/robot_sim` | 训练期镜像（`AGENTS.md` 2025-09-11 版本、含 `env_isaaclab`/`IsaacLab`/`assets`、被 `tools/train_yolo26_v1.py` 的 `--map` 指向） | **旧** |
| `/home/T7/ojh/badmin_project` | 本次同步的 Harness 工作区（最新文本记忆） | **新** |

合并方案（未执行，需批准）：把 `badmin_project` 的文本记忆 rsync **进** `robot_sim`（不带 `--delete`，只更新同名文件），再让 `start.sh` 指向 `robot_sim`。
---

## 10. 第 ③ 层「会话记忆」导入（2026-09-29 完成）

### 10.1 远端会话目录命名 = 源码里的 `projectKey()`（不是猜的）

`@deepseek-ai/dsh-session-persistence-jsonl/lib/index.js:874`：分隔符 `/ \ :` 合并成 `-`，非 `[A-Za-z0-9._-]` 的字符转成 `~XXXX`（大写十六进制），整体包成 `--…--`、去前导 `-`、截断到 251 字符。

用本地实存目录名反验证：`E:\具身智能\badmin_project` → `--E-~5177~8EAB~667A~80FD-badmin_project--`（与磁盘一致 ✓）。据此得到远端名：

| 本地工作区 | 远端工作区 | 远端会话目录 |
|---|---|---|
| `E:\具身智能\badmin_project` | `/home/T7/ojh/badmin_project` | `~/.dsh/sessions/--home-T7-ojh-badmin_project--`（**11 个 session**） |
| `E:\具身智能` | `/home/T7/ojh` | `~/.dsh/sessions/--home-T7-ojh--`（**4 个 session**） |

合计 **15 个 session / 48 个文件 / 137 MB**；`~/.dsh/attachments/` **99 文件 / 86 MB**。

### 10.2 注册表合并

`~/.dsh/storages/workspace.json` 先备份为 `workspace.json.bak-20260929-173539`（227 B），再把本地两个 workspace **保留原 id**（`c48c6ef5…` → `/home/T7/ojh`、`41b71220…` → `/home/T7/ojh/badmin_project`）并入，sessionIds 仍为 7 / 11 条。工具：`tools/merge_dsh_workspace_registry.py`。

### 10.3 为什么必须改表头 cwd（源码证据）

- `dsh-api-workspace-files/lib/index.js:385`：`workspaceRoot: header.cwd ?? scope.sandboxPolicy.workspaceRoot` —— 文件 API **以会话表头 cwd 为工作区根**；
- `dsh-acp/lib/index.js:1216`：`if (!await sameDirectory(persisted.cwd, params.cwd)) throw invalidParams("session cwd does not match")` —— 恢复会话时 cwd 不一致**直接报错**。

因此对 42 个 `.jsonl.zstd` 做路径改写（解压 → 替换 → `zlib.zstdCompressSync` 重压）：`E:\具身智能\badmin_project` → `/home/T7/ojh/badmin_project`、`E:\具身智能` → `/home/T7/ojh`、`D:\_eth_data\eth_shuttle_detection` → `/home/T7/dgut/robot_sim/eth_shuttle_detection`。结果 **FILES=42 / CHANGED=42 / TOTAL_REPLACEMENTS=42**，复验表头全部为远端路径。

**备份（可一键回滚）**：`~/.dsh/sessions-orig-20260929-173716`（137 MB，导入+改写前的原始副本）。

> 改写后远端会话内容与本地原文**不再逐字节相同**（路径字符串被替换过）；本地 `~/.dsh` 仍是唯一真源。

### 10.4 服务器状态（2026-09-29 17:41 UTC）

| 项 | 值 |
|---|---|
| 进程 | PID 895610（cwd `/home/T7/ojh/badmin_project`） |
| 监听 | `127.0.0.1:3080` |
| token | `i1Naas4p8KL91MzuonWmqq44fo1any1IIrqGgznsMY0`（每次重启都会变） |
| 实测 | 无 token 401；带 token 200（34,846 B，`<title>DeepSeek Harness</title>`，`__DSH_BOOT__`） |
| 本机隧道 | `ssh -f -N -L 3081:127.0.0.1:3080 dgut@172.31.68.251` → `http://127.0.0.1:3081/?token=…` |

### 10.5 远端临时文件（可清理，需用户批准）

| 路径 | 体积 | 说明 |
|---|---:|---|
| `/tmp/dsh_sessions.tgz` | 214 MB | 上传的压缩包 |
| `/tmp/dshmem-1790703316/` | 223 MB | 解包暂存 |

`/` 当前剩余 11 GB（97%）。
---

## 11. 凭证（DeepSeek API Key）写入远端（2026-09-29）

### 11.1 结论

| 项 | 值 |
|---|---|
| 写入位置 | 远端 `~/.dsh/.credentials.yaml` 的 **`refs.DEEPSEEK_API_KEY`**（该文件此前只有 `records.client-connection/browser-session`，无 `refs:` 段） |
| 权限 | **0600**（223 B，与本地同构：`version` / `refs.<KEY>` / `records.client-connection/browser-session`） |
| 密钥指纹（掩码） | `len=35`，`sha256: C39756DF…`（**报告中不记录明文**） |
| 备份 | `~/.dsh/.credentials.yaml.bak-20260929-174015` |
| 传递方式 | 经 **stdin** 交给远端脚本（不进 argv / 不进 bash history / 不落远端日志） |
| 落库校验 | 掩码读回 `refs.DEEPSEEK_API_KEY` 的 sha256 前缀与写入值一致 |
| **功能校验** | 远端实调 `GET https://api.deepseek.com/models` → **HTTP 200**，返回 `{"object":"list","data":[{"id":"deepseek-flash"…`（说明 key 有效、且远端可直连 DeepSeek） |
| 本地 Harness | **未改动**（本地 `refs.DEEPSEEK_API_KEY` 指纹为 `D97FDCF5…`，与本次写入的 key 不是同一把） |

### 11.2 为什么这样写

- 远端 `settings.yaml` 里 provider 用 `apiKeyEnv: DEEPSEEK_API_KEY` 解析引用，凭证名必须正好是 `DEEPSEEK_API_KEY`；
- DSH 自身有 `credentials/set` RPC（`@deepseek-ai/dsh-api-settings-controller#credentials/set`，Web UI「设置 → 模型」走的就是它）；无头环境下按同一 schema（`refs.<NAME>`）直接写文件，等价且可回滚；
- 写后需**重启** `dsh web` 才会重新读取凭证。

### 11.3 安全须知

- 明文密钥**只存在远端凭证文件**（0600）与你的输入渠道；仓库内任何文件（本文档、DAILY_LOG、脚本）**都不含明文**，只记掩码指纹与备份路径；
- 密钥一旦在聊天/日志里出现过，视为已暴露：如担心，请在 platform.deepseek.com 轮换，然后重复 §11.1 的写入流程（脚本 `experiments/yolo26_v1/remote_write_key.py`，密钥走 stdin）。
---

## 12. Tailscale 组网（2026-09-30 完成，本地 ↔ 远端 **直连**）

### 12.1 两端现状（实测）

| 项 | 本地 | 远端 |
|---|---|---|
| 设备名 | `abc123456` | **`jxxy`** |
| Tailscale IP | **100.79.176.111** | **100.88.178.19**（IPv6 `fd7a:115c:a1e0::e033:b214`） |
| 状态 | Running / Online（Windows 客户端，服务 Automatic） | **Running / Online**，`tailscaled.service` **active + enabled**（MainPID 922720，开机自启） |
| 版本 | 1.102.4 | **1.102.4** |
| tailnet | `taildd42cc.ts.net`（账号 `jho887891@`） | 同 |

### 12.2 可达性与直连/中继判定（你要的三项）

| 测试 | 结果 | 判定 |
|---|---|---|
| 本地 → 远端 `tailscale ping jxxy` | `pong from jxxy (100.88.178.19) via 172.31.68.251:41641 in 8ms` | **direct** |
| 本地 → 远端 **ICMP** `ping 100.88.178.19` | 4/4 收到，**0% 丢包**，min 6 / avg 50 / max 184 ms | 通 |
| 远端 → 本地 `tailscale ping abc123456` | `pong from abc123456 (100.79.176.111) via 10.62.142.122:41641 in 6ms` | **direct** |
| 远端 `tailscale status` 对本地一栏 | `active; direct 10.62.142.122:41641, tx 516 rx 564` | **direct** |

两条腿都是 **direct 直连**，**没有走 DERP 中继**。

### 12.3 安装方式与验证链（官方来源）

1. 官方仓库行（取自官方 keyring list 文件）：`deb [signed-by=…] https://pkgs.tailscale.com/stable/ubuntu jammy main`；
2. `.deb` 取自官方 `pool/tailscale_1.102.4_amd64.deb`，`Size 38695094` 与官方签名索引（`InRelease` 200）一致，**SHA256 `758cd0b2…56f8b` 逐位一致 → `sha256sum -c OK`**；
3. `apt-get install -y <本地已验证 .deb>`（唯一依赖 `iptables` 来自 Ubuntu 官方源）；
4. `systemctl enable --now tailscaled` → **enabled**（重启后自动恢复）；
5. `tailscale set --operator=dgut` → 之后 `dgut` 不需要 sudo 就能操作 tailscale；
6. 登录：授权链接 `https://login.tailscale.com/a/4626b5e01a6f0`（**由用户本人点击授权**，符合“登录必须本人完成”的约定）。

### 12.4 实际生效的设置（确认未越界）

`tailscale debug prefs` 实测：`Hostname=jxxy`、**`CorpDNS=False`**（未改 DNS）、**`RouteAll=False`**、**`ExitNodeID=`（空）**、**`AdvertiseRoutes=None`**（未做 Subnet Router）、`WantRunning=True`、`ShieldsUp=False`。

**明确未做**：Exit Node、Subnet Router、修改学校/实验室网络、修改防火墙规则、开放公网端口、端口转发、暴露 3080。dsh web 仍为 `127.0.0.1:3080`（本阶段前后都实测确认）。

### 12.5 注意事项与风险

- **无 MagicDNS**（`CorpDNS=False`）：普通工具请用 **IP**（`100.88.178.19`），名字 `jxxy` 只在 tailscale 自己的命令里能解析；
- 本机 Tailscale 仍报 `derp17c.tailscale.com` 被校园网锐捷设备**中间人拦截**（证书 `CN=ruijie`）→ **只影响中继回退**，当前直连不受影响；
- 远端 `netcheck` 显示 **`MappingVariesByDestIP: true`**（对称 NAT），直连依赖对端可被拨号；本机今天网络变过两次（`10.62.142.122`、网关 `10.60.0.1`），若直连打洞失败会回退中继，而中继链路正是被拦截的那条 → **后续判分叉时要把这点算进去**；
- 回滚方式（需用户批准）：`sudo systemctl disable --now tailscaled && sudo apt-get remove tailscale`。

### 12.6 可清理的中间文件（合计约 721 MB，均待用户批准）

| 路径 | 体积 | 说明 |
|---|---:|---|
| `~/.local/tailscale-dl/` | 74 MB | 官方 `.deb` + `.tgz` + sha256（安装已完成） |
| `~/.local/tailscale/` | 73 MB | 免 root userspace 尝试的解包残留（进程已确认不存在，只有文件） |
| `/tmp/dsh_sessions.tgz` | 214 MB | 会话记忆上传包 |
| `/tmp/dshmem-1790703316/` | 223 MB | 会话记忆解包暂存 |
| `~/.dsh/sessions-orig-20260929-173716/` | 137 MB | 会话改写前备份（建议留到确认历史可用后） |
---

## 13. 远端 Harness 接入 tailnet（`tailscale serve`，2026-09-30）

### 13.1 为什么不是“绑 Tailscale IP”

先尝试 `dsh web --host 100.88.178.19` → **DSH 拒绝**，日志原文：
```
$.host expected "127.0.0.1" | "0.0.0.0" but got "100.88.178.19" (at host)
```
DSH 的 `--host` 只接受 `127.0.0.1` 或 `0.0.0.0`；而 `0.0.0.0` 会把 3080 摊在校园网 `172.31.68.251` 上（明确禁止）→ 放弃该路线。

### 13.2 最终架构（官方推荐路径）

```
浏览器(本机) ──http://jxxy.taildd42cc.ts.net:3080──> tailscaled(serve) ──proxy──> 127.0.0.1:3080 (dsh web)
                        tailnet only                                    loopback only
```

配置命令（`operator=dgut` 已设，无需 sudo）：
```bash
tailscale serve --bg --http=3080 http://127.0.0.1:3080
tailscale serve status          # -> http://jxxy.taildd42cc.ts.net:3080 (tailnet only) |-- / proxy http://127.0.0.1:3080
```
安装时那条 `tailscale set --operator=dgut` 实际未生效，导致首次 `serve` 报 `Access denied: serve config denied`；用 `sudo tailscale set --operator=dgut` 补设后正常。

`~/.dsh-web/start.sh` 相应加了浏览器信任栅栏白名单（**仍然只绑回环**）：
```bash
dsh web --no-open --host 127.0.0.1 --port 3080 \
  --trusted-host 100.88.178.19:3080 --trusted-host jxxy.taildd42cc.ts.net:3080
```

### 13.3 实测证据（2026-09-30）

| 测试 | 结果 |
|---|---|
| 本机 → MagicDNS 名 + token | **200**，34,846 B，`<title>DeepSeek Harness</title>`，`__DSH_BOOT__` 命中 |
| 本机 → MagicDNS 名、无 token | **401**（栅栏生效） |
| 本机 → 直接 IP `http://100.88.178.19:3080/` | **404** —— `tailscale serve` 按 Host 分发，**必须用节点名** |
| 本机 → 校园网地址 `http://172.31.68.251:3080/` | **连接被拒（000）** → 未暴露到局域网/公网 |
| 远端监听 | `127.0.0.1:3080`(dsh) + `100.88.178.19:3080` / IPv6(tailscaled serve) |

### 13.4 持久性与回滚

- `tailscale serve` 的配置存在 tailscaled 状态中 → **重启后自动恢复**；
- **`dsh web` 不自启**（用户进程）：重启/进程退出后需再跑 `~/.dsh-web/start.sh`，**且 token 会变**（`grep -o "token=[^ ]*" ~/.dsh-web/dsh-web.log | tail -1`）→ 如需开机自启，方案见 §14（未执行，待批准）；
- 回滚：`tailscale serve --http=3080 off`（或 `tailscale serve reset`）；恢复“仅本机回环”只需不再配置 serve。

### 13.5 当前访问地址

```
http://jxxy.taildd42cc.ts.net:3080/?token=<每次启动新生成，见 ~/.dsh-web/dsh-web.log>（自 §14 起也可以直接跑 `~/.dsh-web/url.sh`）
---

## 14. Harness 开机自启（systemd --user + linger，2026-09-30）

### 14.1 单元

`~/.config/systemd/user/dsh-web.service`（仓库内副本 `experiments/yolo26_v1/dsh-web.service`）：
```ini
[Unit]
Description=DeepSeek Harness Web UI (loopback; exposed to the tailnet via tailscale serve)
After=network-online.target
Wants=network-online.target

[Service]
Type=simple
WorkingDirectory=/home/T7/ojh/badmin_project
ExecStart=/home/dgut/.local/bin/dsh web --no-open --host 127.0.0.1 --port 3080 --trusted-host 100.88.178.19:3080 --trusted-host jxxy.taildd42cc.ts.net:3080
Restart=on-failure
RestartSec=5
StandardOutput=journal
StandardError=journal

[Install]
WantedBy=default.target
```

`Type=simple`（systemd 直接托管进程，可自愈重启），日志进 journal；`sudo loginctl enable-linger dgut` 让用户管理器在**无人登录时也随开机启动**。

### 14.2 实测状态

| 项 | 值 |
|---|---|
| unit | `dsh-web.service` **active + enabled**，MainPID 931050，`WorkingDirectory=/home/T7/ojh/badmin_project` |
| linger | **`Linger=yes`** |
| 监听 | `127.0.0.1:3080`(dsh) + `100.88.178.19:3080`/IPv6(tailscale serve) |
| 校园网暴露 | `172.31.68.251:3080` **未绑定** ✓ |
| 本机经 Tailscale 访问 | 正确 token → **200**（34,846 B、`<title>DeepSeek Harness</title>`、`__DSH_BOOT__`）；错/旧 token → **401** |

### 14.3 日常操作

```bash
~/.dsh-web/url.sh                      # 打印当前 local + tailnet 两个 URL（含最新 token）
systemctl --user status dsh-web        # 看状态
systemctl --user restart dsh-web       # 重启（token 会变）
journalctl --user -u dsh-web -n 30     # 看日志
# 完全回滚：
systemctl --user disable --now dsh-web && sudo loginctl disable-linger dgut
```

> 注意：**token 每次启动/重启都会变**，且服务已在开机时自启 —— 所以“开机后拿新 token”用 `~/.dsh-web/url.sh` 一条命令即可。手动 `~/.dsh-web/start.sh` 现在是**备用**手段（它检测到已在运行就只打印现状，不会冲突）。
---

## 15. 浏览器 502 的真因与修复（本机代理绕过，2026-09-30）

### 15.1 现象与定位

浏览器打开 `http://jxxy.taildd42cc.ts.net:3080/?token=…` → **HTTP ERROR 502**；但服务端一切正常（unit active、进程在、`127.0.0.1:3080` + `100.88.178.19:3080` 均在监听）。

定位证据（同一台本机，仅“是否走代理”不同）：

| 请求 | 结果 |
|---|---|
| `curl -x http://127.0.0.1:7890 http://jxxy.taildd42cc.ts.net:3080/` | **502**（复现浏览器现象） |
| `curl --noproxy "*" http://jxxy.taildd42cc.ts.net:3080/` | **401**（到达 Harness，要求鉴权） |
| `curl --noproxy "*" http://127.0.0.1:3081/`（旧 SSH 隧道） | 200 |

根因：本机 **WinINET 代理开启**（`ProxyEnable=1`，`ProxyServer=127.0.0.1:7890`），而 `ProxyOverride` 只含 `localhost;127.*;10.*;172.16–172.31.*;192.168.*;<local>`，**不含 `100.*` 与 `*.ts.net`** → 浏览器把 tailnet 名字交给代理，代理在 tailnet 外解析/连接失败 → 502。

### 15.2 修复（仅改 HKCU 用户项）

给 `HKCU\Software\Microsoft\Windows\CurrentVersion\Internet Settings\ProxyOverride` **追加** `;100.*;*.ts.net`，并用 `InternetSetOption(SETTINGS_CHANGED=39, REFRESH=37)` 通知 WinINET（实测两项均返回 True）。

| | 值 |
|---|---|
| BEFORE | `localhost;127.*;10.*;172.16.*;172.17.*;172.18.*;172.19.*;172.20.*;172.21.*;172.22.*;172.23.*;172.24.*;172.25.*;172.26.*;172.27.*;172.28.*;172.29.*;172.30.*;172.31.*;192.168.*;<local>` |
| AFTER | 同上 `;100.*;*.ts.net` |

修复后实测（不走代理，即浏览器现在的行为）：tailnet URL → **303 → 200**（34,846 B、`<title>DeepSeek Harness</title>`、`__DSH_BOOT__`）。

### 15.3 回滚

```powershell
Set-ItemProperty -Path "HKCU:\Software\Microsoft\Windows\CurrentVersion\Internet Settings" -Name ProxyOverride -Value "localhost;127.*;10.*;172.16.*;172.17.*;172.18.*;172.19.*;172.20.*;172.21.*;172.22.*;172.23.*;172.24.*;172.25.*;172.26.*;172.27.*;172.28.*;172.29.*;172.30.*;172.31.*;192.168.*;<local>" -Type String
```

> 备注：`100.*` 覆盖整个 `100.0.0.0/8`（Tailscale 实际只用 `100.64.0.0/10`）；若日后发现某个公网 `100.x` 服务需要走代理，可把该项换成更窄的模式或直接删掉——`*.ts.net` 已足够覆盖当前的 Harness 访问。
---

## 16. 为什么“离开校园网就感觉连不上”（2026-09-30 诊断）

### 16.1 现状：现在的直连是**校园网内网**直连

```
本机 WLAN 10.62.142.122/14, gw 10.60.0.1, DNS 172.30.253.241/242（校园网）
   └─ Tailscale: 100.79.176.111
        │  tailscale ping jxxy -> pong ... via 172.31.68.251:41641 in 5ms   ← 走的是校园内网地址
远端 jxxy 172.31.68.251（同校园网），Tailscale: 100.88.178.19
```

证据：`Find-NetRoute 172.31.68.251` 走 WLAN（ifIndex 20）；DNS 是 `172.30.253.x`（校园）。也就是说两台机器**都在校园网内**，Tailscale 于是选到了内网直连路径（最快）。

### 16.2 离开校园网后会发生什么

| 候选路径 | 前提 | 现状判定 |
|---|---|---|
| 校园内网直连 `172.31.68.251:41641` | 必须能路由到 `172.31.0.0/16`（校园网或校园 VPN） | **离开即失效** |
| 公网直连（UDP 打洞） | 双方 NAT 可打洞 | 远端 `netcheck` 实测 **`MappingVariesByDestIP: true`（对称 NAT）** → 通常打洞失败 |
| **DERP 中继** | 两端都能连到 DERP 服务器 | **两端都通**：远端对 `derp1/derp9/derp17c.tailscale.com` 均 **HTTP 200**，并列出 25 个区域的延迟（tok 177 ms、hkg 219 ms…）；本机 netcheck 也报出 DERP 延迟；对端 DERP 区域为 `lax` |

结论：**并非“完全不可能”**，而是“校园内网直连消失 + 对称 NAT 难以打洞 → 只能靠 DERP 中继”，而中继链路在校园侧曾被网关做 TLS 中间人（`CN=ruijie`），品质一般。

### 16.3 如果实测确实连不上，优先排查这四条

1. **切换空窗**：网络切换后 Tailscale 需要约 30–60 秒拆除失效的直连、回落到中继 —— 立刻测试会失败；
2. **所在网络限制**：某些公司/酒店网络封 UDP（含 41641）；DERP 走 HTTPS(443) 一般能过，但若连 443 也被拦就彻底无解；
3. **HTTP 代理**（已修，见 §15）：代理未绕过时，即使在校内也是 502；
4. **token 变了**：远端 Harness 每次重启会换 token，页面会 401/空白，看起来像“连不上”——用 `~/.dsh-web/url.sh` 取当前 URL。

### 16.4 建议的下一步（对应“要不要中转”的分叉）

1. **先实测**：离开校园网后运行 `tailscale ping jxxy` 与 `tailscale netcheck`，把输出发我 → 一眼就能判 `direct` / `DERP` / 失败；
2. 若 **DERP 可用** → 无需中转；可选把 tailnet 的 DERP 区域固定到更近的节点（管理台设置）以降低延迟；
3. 若 **DERP 不可用/太慢** → 用你已有的公网机器做中转（你 ssh 配置里有 `223.109.239.11:20736`）：
   - 自建 **DERP**（`derper`）供 tailnet 使用；或
   - 在远端到该公网机之间建 **反向隧道**（`ssh -R`）暴露 Harness（需要在该机器上开端口，属“端口暴露”，须先经你同意）；
4. 若学校提供 **VPN** → 校外连上 VPN 即恢复校园路由，零新增设施，最省事。
### 16.5 延迟实测（2026-09-30，本机切到**手机热点**后）

**当前路径：DERP 中继**（非直连）

| 测量 | 结果 |
|---|---|
| `tailscale ping -c 10 jxxy` | min **375** / avg **404** / max **466** ms，全部 `via DERP(lax)`，末行 `direct connection not established` |
| ICMP `ping 100.88.178.19` × 10 | min 368 / avg **395** / max 434 ms，**0% 丢包** |
| 反向 `tailscale ping`（远端→本机） | 390–473 ms，`via DERP(sfo)` |
| HTTP 首页（36,833 B，带 cookie） | TCP connect ≈ **0.40 s**；TTFB min 0.77 / avg **0.87** s；总时 min 1.52 / avg **1.62** s（5 次） |
| 本机 DERP 延迟 | sfo 180 ms、**lax 193 ms**、hkg 269 ms、sin 369 ms |

**对照：校内直连**（§12.2 实测）`tailscale ping` **5–8 ms**、ICMP avg 6 ms —— 热点下慢了约 **50–80 倍**（内部捷径消失 + 两端对称 NAT 无法打洞 → 只能走中继，且中继在美西）。

**若在公网机自建 DERP 的预期收益**（实测到 VPS `223.109.239.11` 的 RTT）：

| 方向 | RTT |
|---|---|
| 本机（手机热点）→ VPS | min 52 / avg **63** / max 87 ms |
| 远端（校园网）→ VPS | min 40.7 / avg **40.8** / max 41.3 ms（mdev 0.2，极稳） |
| 合计（推算的端到端中继 RTT） | **≈ 104 ms**（现为 ~400 ms，约 **4×** 改善；页面预计 TTFB ~0.25–0.35 s、总时 ~0.5 s） |

⚠️ 前提与风险：该 VPS 的 `20736` 端口**两端都连不上**（`tcp_20736_blocked_or_filtered`，但 ICMP 通）→ 自建 DERP 需要在其上开端口（443/TCP+UDP 或自定义）并配 TLS 证书，**属端口暴露，须先经用户同意**。


### 16.6 两种模式实测对照（同一台本机，切换网络）

| 维度 | 模式 A：校园网/WiFi（`莞工全光无线`, 10.62.142.122/14, 网关 10.60.0.1） | 模式 B：纯手机流量热点（公网 183.46.216.194） |
|---|---|---|
| 到远端的路径 | `via 172.31.68.251:41641`（校园内网一跳） | `via DERP(lax)` / 反向 `DERP(sfo)` |
| `tailscale ping` RTT | **7 ms** | **375–466 ms**（avg 404） |
| ICMP RTT | 6 ms | avg 395 ms |
| 本机 `MappingVariesByDestIP` | false（锥形 NAT） | **true（对称 NAT）** |
| Harness 页面总时 | **0.64 s**（36,833 B） | 1.52–2.02 s |
| 直连是否可能 | 是 | **否**（双端对称 NAT + 双端无 IPv6） |

结论：**“能不能用热点直连”取决于热点上游**——上游仍是校园网（手机中继校园 WiFi，或挂着校园 VPN）→ 直连 7 ms；纯运营商流量 → 只能中继 ~400 ms（可用但慢）。自检一条命令：`tailscale status | Select-String jxxy`（`direct` vs `relay`）。

**运维提示**：离开校园网后，SSH 也必须用 tailnet 名字/IP（`dgut@jxxy.taildd42cc.ts.net` 或 `100.88.178.19`），校园地址 `172.31.68.251` 不可达（本次实测 `connect to host 172.31.68.251 port 22: Connection timed out`）；首次连接需接受主机密钥（本次用 `StrictHostKeyChecking=accept-new`，已写入本机 known_hosts）。

---

## 17. 交接快照（2026-09-30，可用状态）

### 17.1 链路

```
本机浏览器 ──(Tailscale, 私网)──> jxxy(100.88.178.19) ──tailscale serve──> 127.0.0.1:3080 ──> dsh web
            工作区 /home/T7/ojh/badmin_project（我们的项目记忆）
```

### 17.2 关键事实（实测值）

| 项 | 值 |
|---|---|
| 远端 Harness | `@deepseek-ai/dsh 0.2.0-rc.2`，Node v22.23.3，装在 `~/.local`，wrapper `~/.local/bin/dsh` |
| 服务 | `systemctl --user dsh-web`：**active + enabled**，`Linger=yes`（开机自启、无需登录），PID 931050 |
| 监听 | `127.0.0.1:3080`（dsh）+ `100.88.178.19:3080`（tailscale serve，**仅 tailnet**） |
| 访问 | `http://jxxy.taildd42cc.ts.net:3080/?token=<最新>`；取 token：`~/.dsh-web/url.sh` |
| 工作区 | `/home/T7/ojh/badmin_project`（3,864 文本文件 / 29 MB 项目记忆 + 15 个会话历史 / 137 MB + 86 MB 附件） |
| 模型凭证 | `~/.dsh/.credentials.yaml` 的 `refs.DEEPSEEK_API_KEY`（0600；实调 api.deepseek.com **200**） |
| Tailscale | 两端 1.102.4；本机 100.79.176.111、远端 100.88.178.19；直连（`via 172.31.68.251:41641`，校内网） |
| 本机代理 | `ProxyOverride` 已加 `;100.*;*.ts.net` |

### 17.3 常用命令

```bash
ssh dgut@172.31.68.251 '~/.dsh-web/url.sh'                      # 当前 URL + token
ssh dgut@172.31.68.251 'systemctl --user status dsh-web'        # 服务状态
ssh dgut@172.31.68.251 'systemctl --user restart dsh-web'       # 重启（token 变）
ssh dgut@172.31.68.251 'journalctl --user -u dsh-web -n 30'     # 日志
ssh -f -N -L 3081:127.0.0.1:3080 dgut@172.31.68.251            # 备用：SSH 隧道（不受代理影响）
```

### 17.4 未做/待决策

1. 远端中间文件约 **721 MB** 待清理（§12.6）；
2. `robot_sim` ↔ `badmin_project` 两棵树是否合并（§9.4）；
3. 本地 Harness 是否改用与远端同一把 key（当前本地是另一把）；
4. `tailscale serve` 是否升级为 HTTPS（需在管理台开证书）；
5. 校外访问实测（§16.4）；若中继不可用再考虑自建 DERP/反向隧道；
6. 规范 §28 的 `tools/audit_yolo26_v1_run.py` 与 Stage C / Gate 7 仍未做。




```





