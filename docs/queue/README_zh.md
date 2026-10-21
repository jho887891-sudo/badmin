# 执行队列 CPU 侧工具（A01 / A02 / A03 / A10）

本目录的四个工具把 P0–P8 的 96 个任务槽位变成**可校验的机器状态**，而不是聊天记录里的计划。
全部只用 **Python 3 标准库 + pytest**，外部程序只调用 `git` 与 `gh`；
**不使用 Isaac Sim / Isaac Lab / Kit / omni / pxr / torch / numpy，也不启动任何 GPU 进程**。

## 1. 一键复现（纯 CPU）

```
python tools/queue/run_cpu_checks.py
```

它会依次执行并打印每一步的**真实退出码**：

| 步骤 | 命令 | 通过标准 |
|---|---|---|
| 状态结构校验 | `python tools/queue/queue_state.py --validate` | exit 0 |
| 队列/仓库一致性 | `python tools/queue/check_consistency.py` | exit 0 |
| 工程索引新鲜度 | `python tools/queue/a01_engineering_index.py --check` | exit 0 |
| 离线测试 | `python -m pytest tests/queue -q` | exit 0 |

实测（本分支，Windows 10 / Python 3.11.9 / pytest 9.0.2，无 GPU、无 Kit）：

```
python -m pytest tests/queue -q
# 59 passed in 56.03s        exit 0

python tools/queue/queue_state.py --validate
# state OK: 96 slots, schema 1.0          exit 0
python tools/queue/check_consistency.py
# consistent: 96 slots checked            exit 0
python tools/queue/a01_engineering_index.py --check
# index OK: ... matches a fresh scan (stable facts)   exit 0
```

## 2. 单个工具

### A01 · 工程索引（机器可读，禁止手写）

```
python tools/queue/a01_engineering_index.py          # 写 management/morph_one_rs/ENGINEERING_INDEX.json
python tools/queue/a01_engineering_index.py --check  # 稳定事实（spec/文档/证据）不一致则 exit 1
python tools/queue/a01_engineering_index.py --check --strict  # 连分支/PR/worktree 快照漂移也算失败
```

产物记录：每个分支 → HEAD sha / 提交时间 / 主题 / 是否已并入 main / 关联 PR 号；
worktree 列表（路径 + HEAD + 分支）；PR 号与状态；`management/**`、`docs/**` 文档的 sha256；
证据目录（先在当前 worktree 找，找不到就从 `origin/dev/p2-wheel-contact-g1g4` 用 `git ls-tree` 读取，
并标 `source: git:<ref>`）；`origin/HEAD` 缺失时**不假设 main 是默认分支**，而是在 `notes` 里明说。

### A02 · 队列状态 + READY 选择器

```
python tools/queue/queue_state.py --generate      # 从 SPEC 解析 96 槽位 → 状态 + registry
python tools/queue/queue_state.py --status        # 表格
python tools/queue/queue_state.py --ready         # JSON：READY 列表 + BLOCKED 链
python tools/queue/queue_state.py --ready --stage A,P0
python tools/queue/queue_state.py --set P2-05=DONE --artifact <path> --head-sha <sha> --owner main
```

- 状态取值：`READY / RUNNING / DONE / PARTIAL / FAILED / BLOCKED / UNKNOWN`。
- **READY 是推导出来的**：只有当该槽位的**全部依赖都是 DONE** 时才是 READY；
  否则为 BLOCKED，并在 `--ready` 的 `blocking_chain` 里给出 `TASK → 依赖 → … → 根因`，
  `root_cause_status` 给出根因当前状态（READY/RUNNING/PARTIAL/FAILED）。
- **循环依赖 fail-closed**：`compute_queue` 抛 `QueueCycleError`，CLI 退出码 **3**，绝不返回半截 READY 列表。
- 写入是**原子**的（临时文件 + `os.replace`），不会留下半写文件。
- `--set` 拒绝非法迁移：`DONE` 是终态；`DONE` 必须同时给出 `--artifact` 与 `--head-sha`，
  且 artifact 必须真实存在、sha 必须是本仓库可达提交；**已 DONE 的证据不可替换**；
  依赖未 DONE 时也不能直接置 `READY`/`DONE`；同状态重复写入且证据不同会被拒绝（防止并发重复占用）。

依赖策略（写在代码里，测试覆盖）：`P3-*`→`P2-10`；`P4-*`→`P1-09+P2-10+P3-08`；
`P6` 静态部分→`P2-10`（`P6-09` 另加 `P5-09`、`P6-08`）；`P7-*`→`P5-09+P6-09`；
`P8-06`→`P7-13`；其余 `P8` 为设计/文档/模拟数据，**现在就是 READY**（PPO 训练与真机仍禁止）。

### A03 · 一致性门禁

```
python tools/queue/check_consistency.py          # exit 0 = 一致
python tools/queue/check_consistency.py --json
```

失败条件：DONE 缺少 artifact/head_sha；artifact 在工作区与该提交里都不存在；sha 不是可达提交；
两个任务并发声明同一 artifact；run 目录缺 `manifest.json`/`result.json`/`raw_poses.jsonl`；
`latest_run_id` 在工作区与提交里都不存在；registry 与依赖策略不一致；依赖环（exit 3）。

## 3. 当前种子（只记已证明的事实）

| 槽位 | 状态 | 证据 | 说明 |
|---|---|---|---|
| P2-01 | PARTIAL | `.../p2_wc_gates/G2_r02/result.json` @ `9bf73d7` | G2_r02 已 PASS（恰好接触零重力）；报告里“每组重复两次”的措辞仍需更正，因此不记 DONE |
| P2-02 | PARTIAL | `.../p2_wc_gates/G5_r01/manifest.json` @ `6d8fe70` | manifest 的 gravity 与命令行冲突，旧 JSON 保留，尚缺独立审计文件 |
| P2-05 | RUNNING | `management/morph_one_rs/PHASE_P2_WC_G6_COLLISION_IMPLEMENTATIONS.md` @ `9bf73d7` | G6 A/B/C 各 ≥2 run 已存在，仍需独立代理复核后才可 DONE |

其余 93 个槽位均为 **UNKNOWN**（没有可信测量），不虚报 DONE。
注意：P2-WC 分支仍在推进，种子 SHA 是**钉住的快照**；刷新 HEAD 后请重跑 `--generate` 与索引。

## 4. 为什么可以断言“纯 CPU、不会启动 Kit”

`tests/queue/test_engineering_index.py` 里的契约测试做静态验证：

1. `tools/queue/*.py` 的 import 只允许标准库白名单，且**不允许**出现
   `isaacsim / isaaclab / omni / pxr / torch / carb / numpy`；
2. 所有 `subprocess` 调用的字面量参数**不允许**包含 `isaac` 或 Kit 相关字符串（实际只调用 `git`/`gh`）；
3. `docs/queue/README_zh.md`（本文件）必须给出可复现命令。

配合 A01 索引里的 `environment.launches_isaac_or_kit = false`，可审计地证明这些检查不需要 GPU。

## 5. 纪律

不合并 PR、不 force push、不碰 main；不改别人的 `scripts/simulation/*` 与 `management/morph_one_rs/PHASE_P2_*`；
失败证据不删除，历史 RUN 不覆盖。
