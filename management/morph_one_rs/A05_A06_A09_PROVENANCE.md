# A05 / A06 / A09 交付说明（断言来源、运行身份、不可变证据）

**分支**：`queue/a05-a06-a09` **基线**：`origin/main@16f1ae4` + `queue/a01-a03-a10@1cfbbed`
**范围**：纯 CPU、无 GPU、无 Isaac/Kit。只改 `tools/queue/`、`tests/queue/`、本文件。
**规则遵守**：未修改 `EXECUTION_QUEUE_STATE.json` 的任何 task 状态；未触碰 `scripts/simulation/*`、`simulation/`、`src/`、`outputs/.../p2_wc_gates/`、任何 `PHASE_P2_*` / `PHASE_P3_*` 文档；未改写任何历史证据。

---

## 1. 交付物

| 任务 | 文件 | 说明 |
|---|---|---|
| A05 | `tools/queue/experiment_meta.py` | 配置 hash（排序归一化参数上的 sha256）、环境指纹（python/平台/可导入包版本，**不含 GPU 探针**）、资产 SHA 清单、manifest 声称值 vs 命令实际应用值 |
| A06 | `tools/queue/run_evidence.py` | RUN_ID 唯一性、证据目录不可覆盖（对已提交 hash 清单比对）、陈旧/继承 `result.json` fail-closed |
| A09 | `tools/queue/assertion_provenance.py` + `tools/queue/provenance/assertion_registry.json` | 断言来源登记格式 + 加载器 + 生成器（未登记的 gate 直接非零退出） |
| 测试 | `tests/queue/test_experiment_meta.py` / `test_run_evidence.py` / `test_assertion_provenance.py` | 新增 54 项 |
| 兼容 | 从 `queue/a01-a03-a10` 原样继承 `tools/queue/*`、`tests/queue/*`、`QUEUE_TASK_REGISTRY.json`、`EXECUTION_QUEUE_STATE.json`、`docs/queue/README_zh.md`、`ENGINEERING_INDEX.json` | 唯一改动：`test_engineering_index.py` 的 `STDLIB_ALLOWED` 增加 `platform / importlib / shlex`（均为标准库） |

### 契约 1（A05）：配置 / 环境 / 资产可自证

* `config_hash` = `sha256:` + 规范化参数的 sha256。规范化 = 递归按 key 排序、float 归到 12 位小数、tuple 转 list，因此 **key 顺序改变 hash 不变**、**任一数值改变 hash 必变**。
* 环境指纹 = `python_version / python_implementation / platform_* / packages_importable`，`gpu_probe` 恒为 `null`：GPU 状态不可复现、不需要，也不作为通过条件。
* 资产 = 目录内已知后缀文件的 `{path, bytes, sha256}` 列表。
* FAIL（退出码 2，逐条给出 reason）条件：`GRAVITY_MISMATCH`、`GRAVITY_UNCLAIMED`、`COMMAND_UNPARSED`、`RUN_ID_MISMATCH`、`OUT_DIR_MISMATCH`、`LIFT_M_MISMATCH`、`STEPS_MISMATCH`、`DT_S_MISMATCH`、`ASSET_MISSING`、`ASSET_SHA_MISMATCH`、`MISSING_MANIFEST`、`CONFIG_HASH_MISMATCH`。

**真实 bug 已复现（不是修好了历史数据）**：`dev/p2-wheel-contact-g1g4` 上的 `G5_r01/manifest.json` 至今仍写 `gravity_m_s2=[0,0,0]`，而它的 `command` 是 `--gravity-z -9.81`。A05 现在让该实验 **FAIL**（真实退出码见第 2 节）。**历史 JSON 没有被改写，这条证据依然作废。**

### 契约 2（A06）：唯一 RUN_ID + 不可变证据 + 陈旧 JSON 闭锁

* RUN_ID 唯一：目录名即 run_id；目录内 `manifest.run_id` / `result.run_id` 必须等于目录名；两个目录声明同一 run_id（含仅大小写不同）→ `DUPLICATE_RUN_ID`。
* 不可变：`ledger` 子命令把 `{run_id: {path: sha256}}` 冻结成清单；`check` 逐文件比对：内容变了 → `EVIDENCE_MODIFIED`，文件没了 → `EVIDENCE_REMOVED`，清单外新增 → `EVIDENCE_UNLISTED`（默认 fail，`--allow-unlisted` 才放行）；永不就地修复。
* 陈旧 / 继承：`result.json` 的 run_id 与本目录不符、或与 manifest 不符、或与 `--expect-run-id` 不符、或其记录的 `manifest_sha256` 与同目录 manifest 不符 → `STALE_RESULT`（fail closed）。缺失/损坏 → `RESULT_MISSING` / `RESULT_INVALID`。

**与既有 `p2_gate_runner.py` 的关系（引用而非重复实现）**：该 runner 在 `ce278c2`（分支 `test/p0-gate-runner-strict-outdir`）已有写入侧同名规则，本模块不复制其逻辑，只做**事后审计侧**对应：

| 规则 | runner 位置（`ce278c2`） | 本模块 |
|---|---|---|
| 复用 out-dir 拒绝 | `tools/simulation/p2_gate_runner.py:55-59`（REFUSAL_EXIT_CODES）、`:185-238`（decide_out_dir_use） | 只读审计同名证据目录 |
| result.json run_id 不符 | `:602-605`（RUN_ID_MISMATCH） | `STALE_RESULT` 独立复核 |
| 陈旧 result 隔离改名 | `:489`（_quarantine_stale_result） | **不实现改名**，只判失败，历史文件只读 |
| manifest 重力一致性 | `:626-638`（check_manifest_consistency） | A05 在提交后复核同一事实 |

注意：**该 runner 不在本分支**（`origin/main` 无 `tools/simulation/`），因此这里不可能 import 它，只做行号引用。

### 契约 3（A09）：每条 gate 断言的来源可追踪

登记项字段：`unit`、`data_file`、`acceptance`、`acceptance_decision`、`sources[]`、`spec_clauses[]`；每个来源含 `path + line + ref + line_contains`。加载器到 **worktree 或 `git show <ref>:<path>`** 取真实行并校验锚点文本仍在：锚点漂移 → `SOURCE_ANCHOR_MISMATCH`，行不存在 → `SOURCE_LINE_OUT_OF_RANGE`，取不到文件 → `SOURCE_UNRESOLVED`。

生成器扫描 `result.json` 的 `gates`（含 `diagnostics.*`）：**未登记的 gate → `UNREGISTERED_GATE`，退出码 2**；非布尔 gate（缺失/NaN/unknown）→ `NONBOOLEAN_GATE`（空 gate 不得记 PASS）；`acceptance_decision` 只允许 `pass_when_true / pass_when_false`，否则 `DECISION_POLARITY_UNKNOWN`。

已登记 12 条 = 任务点名的 11 条 + `all_finite`（后者是每个真实 `result.json` 都断言的门，不登记则任何真实证据都无法校验，已在登记项 `note` 中写明）。**行号逐条读文件核对过**（用 python 读 `git show` 输出；PowerShell 计数在 `p3_dynamic_mount.py` 上曾偏差 25 行，已按 python 结果改为 388）：

| gate | 源码行（`dev/p2-wheel-contact-g1g4`） | SPEC 条款 |
|---|---|---|
| `all_finite` | `p2_landing_gates.py:59`（同级健全性门） | G5_LANDING:12、G6_PREREG:42 |
| `contact_detected` | `p2_landing_gates.py:59`、`p2_wc_g6_collision_variants.py:408` | G5_LANDING:13、G6_PREREG:43 |
| `settles` | `p2_landing_gates.py:60` | G5_LANDING:14、G6_PREREG:46 |
| `no_tip_over` | `p2_landing_gates.py:64` | G5_LANDING:15、G6_PREREG:45 |
| `no_long_term_penetration` | `p2_landing_gates.py:61` | G5_LANDING:16、G6_PREREG:47 |
| `static_weight_supported` | `p2_landing_gates.py:83` | G5_LANDING:17 |
| `mount_stable_under_load` | `p2_wc_g7b_arm_with_wheels.py:382` | P3_DYNAMIC_MOUNT:25（名为 `mount_stable`）、G5_LANDING:31 |
| `arm_does_not_touch_ground` | `p2_wc_g7b_arm_with_wheels.py:386`、`p3_dynamic_mount.py:388` | P3_DYNAMIC_MOUNT:29 |
| `t0_world_position_within_tol` | `morph_one_p2_real_chassis.py:287`、`p2_wc_g7b_arm_with_wheels.py:360` | G5_LANDING:31、G5_LANDING:5 |
| `t0_world_rotation_within_tol` | `morph_one_p2_real_chassis.py:288`、`p2_wc_g7b_arm_with_wheels.py:361` | G5_LANDING:31 |
| `settle_drift_within_tol` | `morph_one_p2_real_chassis.py:289` | G5_LANDING:5、G5_LANDING:31 |
| `max_drift_within_tol` | `p2_wc_g7b_arm_with_wheels.py:366` | G5_LANDING:31 |

（G5_LANDING = `management/morph_one_rs/PHASE_P2_G5_LANDING_GATE.md`；G6_PREREG = `PHASE_P2_WC_G6_PREREGISTRATION.md`；P3_DYNAMIC_MOUNT = `PHASE_P3_DYNAMIC_MOUNT_GATE.md`。三个文件在本分支不存在，位于 `dev/p2-wheel-contact-g1g4`，登记项带 `ref` 字段，加载器用 `git show` 校验真实行内容。）

---

## 2. 真实命令与真实退出码

本机：Python 3.11.9、pytest 9.0.2、Windows、CPU。全部命令在 `E:/具身智能/badmin_wt_qa2`（分支 `queue/a05-a06-a09`）执行。

### 2.1 测试

`
python -m pytest tests/queue -q
# 112 passed, 2 failed  —  退出码 1（真实值，未美化）
#   新增 54 项全绿；其余 60 项来自 queue 分支

python -m pytest tests/queue/test_experiment_meta.py tests/queue/test_run_evidence.py tests/queue/test_assertion_provenance.py -q
# 54 passed  —  退出码 0
`

**那 2 项失败不是本次改动引入，也不在我的写入范围**：`test_check_consistency.py::test_committed_state_is_consistent` 与 `::test_cli_exit_code_on_committed_state`，失败原因是继承来的 `EXECUTION_QUEUE_STATE.json` 本身有数据缺陷：

`
INCONSISTENT duplicate artifact claim on management/morph_one_rs/PHASE_P2_EVIDENCE_MATRIX.md by P2-08, P2-10
INCONSISTENT duplicate artifact claim on .../p1_wheel_identity/wheel_identity_contract.json by P1-04, P1-09
INCONSISTENT duplicate artifact claim on tools/simulation/p2_gate_runner.py by P0-02, P0-06
INCONSISTENT P2-09: run directory .../g6_summary.json is missing
INCONSISTENT P2-07: latest_run_id 'G7B_grav_r08' exists neither in the worktree nor in commit 8353b68
`

在 `E:/具身智能/badmin_wt_qa`（原 `queue/a01-a03-a10`，`1cfbbed`）跑同一命令的基线：**58 passed, 2 failed — 退出码 1**，即这 2 项在该分支原本就是红的。规则要求 A06 不修改 QUEUE 状态文件的任务状态，故**不擅自改数据**；要修这 2 项必须由主代理更新状态文件（去重 artifact、补 `g6_summary.json`、把 `G7B_grav_r08` 写进队列）。

### 2.2 A05 对着真实 P2 证据跑（钉住那个真实 bug）

`
git archive dev/p2-wheel-contact-g1g4 -- outputs/simulation/rebot_b601_rs/morph_one/p2_wc_gates/G5_r01 ... | tar -x -C E:/temp/a0506demo
python tools/queue/experiment_meta.py validate --experiment E:/temp/a0506demo/outputs/.../p2_wc_gates/G5_r01
# FAIL (A05_VALIDATION): 1 mismatch(es)
#   - GRAVITY_MISMATCH: manifest gravity z is 0.0 but the command applied --gravity-z -9.81
#     (delta 9.81 > 1e-06); the manifest claims ZERO gravity for a run that was NOT
#     zero-gravity, so every zero-gravity conclusion in it is false
# 退出码 2

python tools/queue/experiment_meta.py validate --experiment E:/temp/a0506demo/.../G5L_r01
# PASS (A05_VALIDATION): config_hash sha256:ca943c526d62fdf8d6a5ba77ee2e5f1ecf4e911ecb1d252af966b9f883fec3fa assets 7 env python 3.11.9
# 退出码 0
`

结论：**历史 `G5_r01` 的 manifest 并没有被修好**，A05 现在把它明确判为 `GRAVITY_MISMATCH`；`G5L_r01`（走新落地门禁那次）manifest 与命令一致，通过。

### 2.3 A06 对着真实证据跑

`
python tools/queue/run_evidence.py ledger --runs-root E:/temp/a0506demo/.../p2_wc_gates --out E:/temp/a0506demo/ledger.json
# ledger: ... runs 3 files 18      退出码 0
python tools/queue/run_evidence.py check --runs-root ... --ledger .../ledger.json
# PASS (A06_RUN_EVIDENCE)          退出码 0

# 人为复制 G5L_r01 到 G5L_r02（内含别人的 run_id），模拟继承的 result.json
python tools/queue/run_evidence.py check --runs-root ...
# FAIL (A06_RUN_EVIDENCE): 4 violation(s)   退出码 2
#   DUPLICATE_RUN_ID ...  /  STALE_RESULT: result.json ... carries run_id 'G5L_r01',
#   which is not this directory: the result was inherited from a different run
`

### 2.4 A09

`
python tools/queue/assertion_provenance.py --self-check
# PASS (A09_REGISTRY): 12 gates, every cited source line verified    退出码 0

python tools/queue/assertion_provenance.py --result-json .../G7B_grav_r08/result.json
# PASS (A09_PROVENANCE): gates 7/7 registered                        退出码 0
python tools/queue/assertion_provenance.py --result-json .../G5_r01/result.json
# PASS (A09_PROVENANCE): gates 4/4 registered                        退出码 0
#   这 4 个 gate 的值是 False —— 那是 FAIL，不是来源缺失；A09 只判可追踪性
python tools/queue/assertion_provenance.py --result-json <含未登记 gate 的 json>
# FAIL (A09_PROVENANCE): UNREGISTERED_GATE                           退出码 2
`

---

## 3. 还 UNKNOWN / 未闭环（不粉饰）

1. **`G5_r01` 运行时真实重力未知**：manifest 写 0、命令写 -9.81，运行期重力没有任何字段记录。该 run 按任何口径都是 **INVALID**；A05 只证明自相矛盾，不证明实际施加了 -9.81。
2. **`mount_stable_under_load` 没有一一对应的 P2 SPEC 名字**：P3 文档写 `mount_stable`（`PHASE_P3_DYNAMIC_MOUNT_GATE.md:25`），P2 实现（g7b:382）叫 `mount_stable_under_load`；登记项两条都引，但这不是文献级一一对应。
3. **`all_finite` 的实现行引用偏弱**：它实际由 `p2_gate_runner.read_json_gate` 在 JSONL 层判定，而该 runner 不在本分支；登记项暂指 `p2_landing_gates.py:59` 的同级门，属间接来源。
4. **资产清单口径不统一**：历史 manifest 只有部分实验写 `asset_sha256`，所以声明资产缺失/哈希不符的检查在这些实验上空过（代码有 `test_declared_asset_*` 覆盖）；不能据此宣称历史资产哈希都被校验过。
5. **本分支不含证据目录**：真实证据跑都用 `git archive dev/p2-wheel-contact-g1g4` 解到 `E:/temp/a0506demo`（未提交），因为规则禁止改动 `outputs/.../p2_wc_gates/`；复现命令见 2.2 到 2.4。
6. **行号对工具敏感**：PowerShell 计行与 python `splitlines()` 在 `p3_dynamic_mount.py` 上差 25 行（已按 python 修正为 388）。登记表校验一律走 python 加载器，人工核对不要用 PowerShell 数行。
7. **未验证真实 GPU / Isaac 侧行为**：本切片全程 CPU；`environment_fingerprint()` 明确不含 GPU 探针；A05/A06/A09 不主张任何物理结论。
