# R10-C · 证据契约与断言溯源（C1–C4）

> **性质**：本文件是 CPU 只读审计与校验器的交付记录，不是实验成功证明。
> 工作线：R10 批次 C 线（`C1`–`C4`，见 `management/morph_one_rs/EXECUTION_BATCH_R10_32_20261022.md` @ `14f339f`）。
> 分支：`r10/c-evidence-contracts`（基于 `queue/a01-a03-a10` @ `79c4371`）。仅 CPU，未启动 Isaac/Kit，未改 runner / scripts/simulation / simulation / src，未改 `management/morph_one_rs/EXECUTION_QUEUE_STATE.json`。

## 0. 交付物

| 文件 | 作用 |
|---|---|
| `tools/queue/evidence_contract.py` | C1 类型化证据契约（NORMAL / FAULT）+ C2 完整性与溯源校验器（CLI，退出码 0/1/3） |
| `tools/queue/assertion_registry.py` | C3 断言溯源注册表机检器（CLI，退出码 0/1/3） |
| `management/morph_one_rs/R10_C_ASSERTION_REGISTRY.json` | 新 P0/P5 断言的溯源注册表（ASSERTED / UNKNOWN / NOT_RUN） |
| `tests/fixtures/r10c/build_fixtures.py` | 可复现的 fixture 生成器（生成的 `good/`、`bad/`、`bad_dup/` 一并提交） |
| `tests/queue/test_evidence_contract.py` | C1/C2 正反例回归（21 例） |
| `tests/queue/test_assertion_registry.py` | C3 机检回归（15 例） |

现有 `tools/queue/{queue_state.py, check_consistency.py}` 未修改一行：新校验器是**并列的**独立门禁，不改写既有 96 槽判据。（注：派工单提到的 `tools/queue/experiment_meta.py` 在 `queue/a01-a03-a10@79c4371` 上并**不存在**，实测该分支 `tools/queue` 只有 `__init__.py / a01_engineering_index.py / check_consistency.py / queue_state.py / run_cpu_checks.py`；本线不臆造该文件。）

## 1. C1 · 故障 Run 与正常 Run 分离的证据规则

**规则（类型化，非启发式）**

| 判定 | 触发条件 | 必需文件集 | 结论 |
|---|---|---|---|
| `NORMAL` | 有 `result.json` 且无 `fault_record.json` | `manifest.json` + `result.json` + `raw_poses.jsonl` + `gate_runner_summary.json` | 缺任一 → `NORMAL_INCOMPLETE` FAIL |
| `FAULT` | 有 `fault_record.json`（带类型化 `fault_code`）且**无** `result.json` | `fault_record.json` + `gate_runner_summary.json` + ≥1 个原始子进程日志（`child_*.log` / `gate_child_*.log` / `*.txt`，见下注） | 缺任一 → `FAULT_INCOMPLETE` / `FAULT_NO_RAW_LOG` FAIL |
| `UNDECLARED` | 既无 `result.json` 也无 `fault_record.json` | — | `KIND_UNDECLARED_UNPROVEN` FAIL：**缺 result.json 永远不构成故障证明** |
| `CONFLICT` | 两者同时存在 | — | `KIND_CONFLICT` FAIL：不许一个 Run 同时声称两种身份 |

> 注：仓库 `.gitignore:13` 忽略 `*.log`，因此 **fixture** 里的原始日志用 `child_stdout.txt` / `child_stderr.txt`（契约对两者同等接受）；真实 Run 的日志仍应为 `.log`，但必须进 Run 目录并纳入索引，否则 `.gitignore` 会让它从提交里消失——这本身就是 C2 要防的失效模式。另新增 `.gitattributes` 的 `tests/fixtures/r10c/** -text`：fixture 自带 SHA256 索引，任何 EOL 归一化都会改变字节并让索引失效。

**类型化故障码**（A3 语义）：`EXPECTED_FAULT` / `UNEXPECTED_FAULT` / `CRASH` / `TIMEOUT` / `SIGSEGV` / `KIT_PYTHON_ERROR`；`fault_record.json` 还必须带 `run_id`、`fault_class`（EXPECTED/UNEXPECTED）、`outer_exit`、`child_returncode`——即“child_rc=0 的假成功”必须留在记录里。

**两个方向都有 fixture 证明**（实测见第 4 节）：

- 方向 A（不得因缺 `result.json` 判故障 Run 失效）：`tests/fixtures/r10c/good/F1_r01` 无 `result.json`，判定 0 violation。
- 方向 B（不得因缺 `result.json` 判普通 Run 合格或“自动等于故障”）：`tests/fixtures/r10c/bad/B1_r01_no_result` → `KIND_UNDECLARED_UNPROVEN`。

## 2. C2 · 唯一 RUN_ID / 参数 hash / 环境指纹 / 原始日志 / SHA256

| 检查 | 实现 | 失败码 |
|---|---|---|
| RUN_ID 唯一（跨证据集） | 同一 RUN_ID 被两个目录声明 | `RUN_ID_DUPLICATE` |
| RUN_ID 与目录/内嵌值一致 | 目录名 vs `manifest.run_id` vs `result/fault_record/gate_runner_summary.run_id` | `RUN_ID_DIR_MISMATCH` / `CROSS_RUN_REFERENCE` |
| 参数 hash 可重算 | `sha256(canonical_json(manifest.config))` 与 `manifest.config_sha256` 比对 | `PARAM_HASH_MISSING` / `PARAM_HASH_MISMATCH` |
| 环境指纹存在 | `manifest.env_fingerprint` 且含非空 `python_version` / `platform` / `sim_backend` | `ENV_FINGERPRINT_MISSING` |
| 原始日志存在且**在 Run 目录内** | `gate_runner_summary.child_stdout_log/stderr_log` 必须为相对路径且文件非空 | `RAW_LOG_MISSING` / `LOG_OUTSIDE_RUN` |
| 文件 SHA256 完整性 | `manifest.files{相对路径: sha256}` 逐文件重算 | `SHA256_INDEX_MISSING` / `FILE_MISSING` / `SHA256_MISMATCH` |
| stale / 外来文件 | 目录内未进索引的文件（`manifest.json` 自身除外） | `STALE_UNINDEXED_FILE` |

约定：`manifest.json` 自身不进索引（避免自指哈希）；索引里只能出现 Run 目录内的相对路径（绝对路径即跨机引用，直接失败）。

## 3. C3 · 断言溯源注册表

注册表条目结构（ASSERTED 必须齐全，缺任一项机检即失败）：

```json
{"id": "...", "task": "P0-07", "status": "ASSERTED", "claim": "...",
 "source": {"file": "tools/queue/evidence_contract.py", "line": 128, "quote": "def classify_run(run_dir: Path):"},
 "spec_clause": "C1", "unit": "boolean (契约判定)",
 "evidence": {"trace_path": "tests/fixtures/r10c/good/F1_r01"},
 "threshold": {"value": "violations == 0", "unit": "violation count",
               "pre_registered": true, "registered_commit": "14f339f"}}
```

机检项：`source.file` 必须存在（或给 `source.commit` 且该路径在该提交树内）、`source.line` 为正整数且不越界、**`source.quote` 必须真的出现在该行**（只有行号不构成溯源）；`spec_clause` 必须是批次条款号（A–H + 数字）；`unit` 非空；`evidence` 必须 `run_id` 与 `trace_path` **二选一**且可解析；`threshold` 必须 `pre_registered: true` 并带 `value`/`unit`/`registered_commit`；`ID_DUPLICATE` 失败。

**UNKNOWN 不数值填充**：`unit`/`threshold`/`evidence` 必须为 null；`claim` 先剥离标识符（`P0-07`、`C1`、`R10`、commit sha、版本号、`Issue #13`、`PR #n`）后不得残留任何数字，否则 `UNKNOWN_CARRIES_NUMBER`。当前注册表：6 条 ASSERTED（C1 两条方向、C2 参数 hash / SHA256 / RUN_ID 唯一 / 日志越界）、2 条 P5 UNKNOWN（P5-07 多环境、P5-08 种子/吞吐/显存）、1 条 P5-07 NOT_RUN（无 GPU 权限）。

## 4. C4 · 真实命令与**真实退出码**

环境：Windows / Python 3.11.9 / pytest 9.0.2，工作树 `badmin_wt_r10c`。以下退出码为实测，未做任何美化。

| # | 命令 | 实测结果 | 退出码 |
|---|---|---|---|
| 1 | `python -m pytest tests/queue -q` | `96 passed in 112.63s` | **0** |
| 2 | `python tools/queue/check_consistency.py` | `consistent: 96 slots checked` | **0** |
| 3 | `python tests/fixtures/r10c/build_fixtures.py` | `fixtures rebuilt` | **0** |
| 4 | `python tools/queue/evidence_contract.py --root tests/fixtures/r10c/good` | `evidence contracts satisfied: 2 run(s)` | **0** |
| 5 | `python tools/queue/evidence_contract.py --root tests/fixtures/r10c/bad` | 12 violation(s) / 10 run(s) | **1** |
| 6 | `python tools/queue/evidence_contract.py --run-dir .../bad_dup/a/DUP_r01 --run-dir .../bad_dup/b/DUP_r01` | `RUN_ID_DUPLICATE` | **1** |
| 7 | `python tools/queue/evidence_contract.py --run-dir tests/fixtures/r10c/ghost_r99` | `error: run directory not found` | **3** |
| 8 | `python tools/queue/assertion_registry.py` | `assertion provenance satisfied` | **0** |
| 9 | 同上，`--registry _scratch_registry_bad.json`（把 C1-01 的 line 改成 99999） | `ASSERTED_SOURCE_INVALID ... has only 434 lines` | **1** |
| 10 | 同上，`--registry _scratch_no_such_registry.json` | `error: cannot read registry` | **3** |
| 11 | `python tools/queue/check_consistency.py --state _scratch_state_bad.json`（**草稿副本**，把 P0-02 指向 `GHOST_r99/result.json`） | 2 problem(s)：artifact 不存在 + run 目录缺失 | **1** |

第 5 步 12 条违规的逐条映射（`--root bad`）：`KIND_UNDECLARED_UNPROVEN`(B1) / `FAULT_NO_RAW_LOG`+`RAW_LOG_MISSING`×2(B2) / `KIND_CONFLICT`(B3) / `FAULT_CODE_NOT_TYPED`(B4) / `PARAM_HASH_MISMATCH`(B5) / `ENV_FINGERPRINT_MISSING`(B6) / `SHA256_MISMATCH`(B7) / `STALE_UNINDEXED_FILE`(B8) / `CROSS_RUN_REFERENCE`(B9) / `LOG_OUTSIDE_RUN`(B11)。

**对真实历史证据的只读回放**（把 `3a366e7` 的 `G6A_r01` 用 `git archive` 解到 `_scratch_real/`，只读校验，不改进历史）：

```
python tools/queue/evidence_contract.py --run-dir _scratch_real/.../p2_wc_gates/G6A_r01
```

结果 **5 violation(s)**，退出码 **1**：`PARAM_HASH_MISSING`、`ENV_FINGERPRINT_MISSING`、`SHA256_INDEX_MISSING`、`LOG_OUTSIDE_RUN`×2（`gate_runner_summary.json` 里子进程日志是 `/home/dgut/runs/g/G6A_r01/gate_child.stdout.log` 绝对路径，原始日志根本不在 Run 目录内）。也就是说：**既有 P0-06 证据不满足新 C2 契约**。本线的处理是——契约只对**新** Run 生效，历史证据保持原样、不补写哈希、不搬移日志、不伪造 `result.json`；该差距登记为本报告的已知缺口。

**未修改 live 队列 JSON**：第 11 步只在 `_scratch_state_bad.json` 上做，且该文件已删除；`git status` 里 `management/morph_one_rs/EXECUTION_QUEUE_STATE.json` 无改动。

## 5. UNKNOWN / 未完成

1. **P5-07 / P5-08 的真实 Isaac 多环境证据仍 UNKNOWN**：本线无 GPU 权限，未启动 Kit，注册表里相应条目 `status=UNKNOWN`、无数值，不做任何填充。
2. **历史证据（含 P0-06 `G6A_r01`、P0-07 六次故障注入）没有 `config_sha256` / `env_fingerprint` / 文件索引 / 目录内原始日志**。这些字段只能由**新 Run** 的 runner 侧写入；本线按约束不修改 runner，因此“新旧证据如何共存”仍是治理决定，未在本线擅自判定为 PASS。
3. **`threshold.registered_commit` 只做格式校验（7–40 位 hex），未做可达性校验**（可达性检查每条约一次 `git` 调用，留给 `check_consistency`/`queue_state.sha_exists` 侧）；当前 6 条 ASSERTED 的预注册点统一指向派工单提交 `14f339f`，该提交是本批验收口径的发布点。
4. **C2 未覆盖跨机器路径规范化**：只规定“日志必须在 Run 目录内、必须是相对路径”，不解决 `/home/dgut/...` 与 `/home/T7/...` 的多机重放问题。
5. **`FAULT` 与 `EXPECTED_FAULT` 的语义映射**沿用 A3 名词，但 A 线 runner 尚未在本分支落地对应字段；本线只用 fixture 证明判据，未宣称与 runner 实现已对接。
6. 派工单提到的 `tools/queue/experiment_meta.py` 在该分支不存在（见第 0 节），若 A/H 线后续新增，需要重新对齐 `config_sha256` 的定义。
