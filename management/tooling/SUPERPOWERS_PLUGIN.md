# superpowers 插件（DSH）安装状态与升级评估

记录时间：本次会话。执行者：DSH agent（web profile）。

## 1. 结论（TL;DR）

- **插件已经装好并在生效中**：profile `web` 里 `superpowers-dsh` **v0.1.3**，
  14 个技能当前会话可见可用（本会话的技能目录就是它提供的）。
- **已完成"下载"**：npm 最新版 **0.2.0** 已下载并通过完整性校验，留档待用。
- **不建议按原样升级到 0.2.0**：其 SDD / executing-plans 辅助脚本是 bash，
  而本机 DSH agent 的 PATH 里没有 bash/sh/awk → 升级会导致 Windows 上这两个技能退化。
  是否升级需人工决策（见第 4 节）。

## 2. 现场证据（可自行复核）

### 2.1 当前安装（v0.1.3）

```
> dsh --profile web --dump-config | Select-String superpowers
- id: superpowers-dsh
  name: superpowers-dsh
```

- 位置：`C:\Users\abcd1\.dsh\profiles\web\node_modules\superpowers-dsh`
- profile 依赖声明（`C:\Users\abcd1\.dsh\profiles\web\package.json`）：
  `"superpowers-dsh": "file:C:/Users/abcd1/Desktop/具身智能/superpowers-dsh-worktrees/fix-windows-sdd/superpowers-dsh-0.1.3.tgz"`
- 技能数 14：brainstorming, dispatching-parallel-agents, executing-plans,
  finishing-a-development-branch, receiving-code-review, requesting-code-review,
  subagent-driven-development, systematic-debugging, test-driven-development,
  using-git-worktrees, using-superpowers, verification-before-completion,
  writing-plans, writing-skills
- 该构建来自本地分支 `fix/windows-sdd`（`fix(sdd): cross-platform Node entrypoints`），
  SDD 入口被替换为 `node scripts/*.mjs`（shim 内容：
  `#!/usr/bin/env sh\nset -eu\nexec node "$(dirname "$0")/task-brief.mjs" "$@"`）。

### 2.2 已下载的 0.2.0

```
file=E:\具身智能\badmin_project\_scratch_superpowers\superpowers-dsh-0.2.0.tgz
bytes=1861816
sha256=eaadd524d65586c9ec39bbdc21a2b632283d92941f7edff3ef7156903296a501
sha512=692f5cf07fce46a595c437e329fcd30728173c30d8265981a2a192a281bc8c371ea556e6728197acf509f75d71f28093f9d61e2a4f7384a1992f717a2c9d2c3f
--- installed 0.1.3 tree hash (file count + total bytes) ---
files=57 bytes=2228696 skills=14
--- node ---
v24.14.1
```

- 来源：`https://registry.npmjs.org/superpowers-dsh/-/superpowers-dsh-0.2.0.tgz`
- registry 声明 integrity：`sha512-aS9c8H/ORqWVxDfjKfzTBygXPDDYJlmBoqGSooG8jDcepVbmcoGXrPUJ911x8oCT+dYeKk9zhKGZL3F6LJ0sPw==`
  → 换算 hex = `692f5cf07fce46a595c437e329fcd30728173c30d8265981a2a192a281bc8c371ea556e6728197acf509f75d71f28093f9d61e2a4f7384a1992f717a2c9d2c3f`
  → **与本地实测 sha512 完全一致（校验通过）**
- 79 文件；技能 15 个（新增 `diagnosing-superpowers`）；`dsh.superpowersUpstream = v6.4.2`
- 上游 gitHead = `10448b549702c0be9445beb454f39f6214a63601` = `origin/main` 当前 HEAD（`git ls-remote` 实测）

## 3. 升级会踩的坑（已实测，非推测）

| 项 | v0.1.3（现装） | v0.2.0（npm） |
|---|---|---|
| SDD 入口 | `node scripts/task-brief.mjs` | `bash scripts/task-brief`（`#!/usr/bin/env bash`，用 awk/wc） |
| executing-plans 入口 | Node 版 | `bash scripts/task-start` / `task-done`（同为 bash） |
| Windows 依赖 | 仅需 `node`（本机在 PATH 上：E:\MyCode\node.exe） | 需要 Git Bash + awk（**本机 PATH 上没有**） |

实测（PowerShell `Get-Command`）：

```
bash => NOT ON PATH
sh   => NOT ON PATH
awk  => NOT ON PATH
wc   => NOT ON PATH
sed  => NOT ON PATH
（Git Bash 实体存在：D:\Git\bin\bash.exe、D:\Git\usr\bin\bash.exe，但未加入 PATH）
```

0.2.0 的 `README.md` 自己也写明："**Windows 提示**：技能正文里的 bash 辅助脚本需要 Git for Windows（Git Bash）"。

另外，`fix/windows-sdd` 的修复**没有进上游**：

```
git merge-base --is-ancestor b0e7690 origin/main   -> exit 1（不是祖先）
git merge-base --is-ancestor d7bfbcf origin/main   -> exit 1（不是祖先）
git branch -a --contains b0e7690                   -> 仅 fix/windows-sdd（本地）
```

## 4. 待决策（三选一）

| 方案 | 动作 | 代价 / 风险 |
|---|---|---|
| **A（默认保持）** | 不动 0.1.3 | 0 风险，但拿不到 `diagnosing-superpowers` 与 v6.4.2 技能同步 |
| **B（推荐，若确实要升）** | 在 `fix/windows-sdd` 上 rebase 到 `origin/main`，把 0.2.0 的 5 个 bash 脚本（task-brief / sdd-workspace / review-package / executing-plans task-start / task-done）等价移植为 Node `.mjs` + shim，用 bash 版 vs Node 版**输出对比**验证后再安装本地构建 | 中等工作量；可顺带推上游 PR |
| **C（最小改动）** | 装 0.2.0 原样，并把 `D:\Git\bin` 加进 DSH 的环境 PATH | 依赖 Git Bash + awk 长期在 PATH；仍需重启 `dsh web` |

> 任何安装动作都需要**重启 `dsh web`** 才生效（bundle 层在 profile 启动时挂载），
> 重启会断开当前 GUI 会话。

## 5. 遗留物

- tarball 与解包目录：`E:\具身智能\badmin_project\_scratch_superpowers\`
  （`superpowers-dsh-0.2.0.tgz`、`extract020\`、`compat_check.txt`；该目录匹配 `.gitignore` 的 `_scratch_*`，不进仓库）
- 决策为 A 或 B 完成后，该 scratch 目录可删除。
