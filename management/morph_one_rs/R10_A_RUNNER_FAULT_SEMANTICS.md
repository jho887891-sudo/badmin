# R10-A · Runner 与故障语义（A1–A4）

**分支**：r10/a-runner-faults（基线 test/p0-gate-runner-strict-outdir@ce278c2）· **性质**：CPU 纯离线，无 Isaac、无 GPU、未写 172.31.68.251
**派工单**：management/morph_one_rs/EXECUTION_BATCH_R10_32_20261022.md（chatgpt/r10-32-task-batch-20261022@14f339ff）第 A 节
**本报告只声明可复算的事实**：真实命令、真实退出码、真实回放表；做不到的进第 8 节 UNKNOWN。

---

## 0. 一句话结论

1. runner 现在**只输出五种规范化结局**（SUCCESS / EXPECTED_FAULT / UNEXPECTED_FAULT / CRASH / INCOMPLETE），并且**只有 SUCCESS 允许 exit 0**；故障注入测试 PASS 用 **exit 10**，因此在只看退出码的 CI 里也**绝不会**与"仿真成功"混淆。
2. P0-07 的假成功（Kit 吞异常、child_rc=0）已被两个**独立**硬信号堵住：**typed fault_record.json** 与 **stderr traceback 签名（header + terminator 成对匹配）**；即使 result.json 完整 PASS、child_rc=0，也判 UNEXPECTED_FAULT。
3. 六次真实 P0-07 故障运行全部通过重放分类：**没有一次被判成 SUCCESS**（4 次 EXPECTED_FAULT、1 次 CRASH、1 次取决于日志是否在场的 INCOMPLETE，见 5.1/5.2）。
4. 原有 92 条断言与阈值**一条未改**：python -m pytest tests/simulation/test_p2_gate_runner.py -q → **92 passed，exit 0**。新增 53 条 → **53 passed，exit 0**。

---

## 1. A3 先做：规范化结局词表（代码即契约）

实现位置：tools/simulation/p2_gate_runner.py

| 词表 | 代码 | 退出码 | 精确规则（artifact / 码 / stderr） |
|---|---|---|---|
| SUCCESS | Outcome.SUCCESS | **0** | child_rc=0 **且** result.json 是严格合法 PASS（gate_result 为字符串 "PASS"、gates 非空且全为 bool true）**且** manifest 与实跑参数一致 **且**（要求时）raw_poses.jsonl 逐帧合法 **且** 无 fault 记录 **且** 无 traceback **且** 未声明故障 |
| EXPECTED_FAULT | Outcome.EXPECTED_FAULT | **10** | 该 run **声明**了故障（--expect-fault X 或转发给子进程的 --inject-fault X），**且**观察到故障**且**与声明匹配（typed fault_record，或仅 traceback 的 TRACEBACK_ONLY 弱证据），**且**没有同时宣称 PASS。child_rc 可以是 0 也可以是 1 |
| UNEXPECTED_FAULT | Outcome.UNEXPECTED_FAULT | **1** | 出现故障/失败信号但没有匹配的声明：typed 或 untyped fault_record、traceback、child_rc≠0、或"故障 run 同时给出完整 PASS"（FAULT_RUN_CLAIMED_PASS） |
| CRASH | Outcome.CRASH | **1** | 异常终止：child_rc ∈ {132 SIGILL, 133 SIGTRAP, 134 SIGABRT, 135 SIGBUS, 136 SIGFPE, 137 SIGKILL, 139 SIGSEGV} 或 Windows NTSTATUS（访问违例/栈溢出），或 stderr 出现 Kit crash reporter 签名（[carb.crashreporter-breakpad.plugin] / terminating this process with exit code N / backtrace follows / Segmentation fault / core dumped）。**声明了故障也仍然是 CRASH**：崩溃不能证明被测的类型化故障语义 |
| INCOMPLETE | Outcome.INCOMPLETE | **1** | 无故障信号、无崩溃，但没有可接受的完成证书：超时、result.json 缺失/损坏/自相矛盾、gate_result == "FAIL"（诚实的仿真 FAIL）、manifest 不一致、JSONL 非法、声明了故障却未发生 |

设计说明（必须写清，避免误读）：

* **exit 10 是刻意的**：故障注入测试 PASS 若复用 exit 0，任何只读退出码的消费者都会把它当成"仿真成功"。词表把"测试 PASS"与"仿真 SUCCESS"做成**两个不同的退出码**，并有单元测试断言"只有 SUCCESS 的码是 0"。
* **INCOMPLETE 是残留桶，reason_code 才是细节**：五类词表是冻结的，所以"诚实的仿真 FAIL"（GATE_SAYS_FAIL）与"run 被截断"（RESULT_JSON_MISSING / TIMEOUT）同属 INCOMPLETE，靠 summary["outcome_reason_code"] 区分。消费者**必须**读 (outcome, outcome_reason_code) 这一对，不能只看类名，更不能只看 child_returncode。
* **分类器是纯函数**：classify_run(RunEvidence) -> OutcomeVerdict，只吃数据（不含路径、不做 I/O），因此**已提交的历史 run 可以离线重放**（第 5 节），也让"结论"与"取证据"解耦。
* **每次判定都留规则号**：summary["outcome_rule"]（R1…R11）与 summary["outcome_reason"] 一句话理由，审计时不需要猜。

### 1.1 判定顺序（R1→R11，第一个命中即返回）

| 规则 | 条件 | 结局 / reason_code |
|---|---|---|
| R1 | timed_out | INCOMPLETE / TIMEOUT |
| R2 | 崩溃签名或信号退出码 | CRASH / CRASH_ABNORMAL_TERMINATION |
| R3 | 存在 fault 记录 | 外来 run_id → UNEXPECTED_FAULT / FAULT_RECORD_RUN_ID_MISMATCH；未声明 → FAULT_RECORD_WITHOUT_EXPECTATION；声明不匹配 → FAULT_RECORD_MISMATCH；同时 PASS → FAULT_RUN_CLAIMED_PASS；否则 EXPECTED_FAULT / FAULT_RECORD_TYPED 或 FAULT_RECORD_UNTYPED |
| R4 | 存在 traceback | 未声明 → TRACEBACK_WITHOUT_EXPECTATION；不匹配 → TRACEBACK_MISMATCH；同时 PASS → FAULT_RUN_CLAIMED_PASS；否则 EXPECTED_FAULT / TRACEBACK_ONLY |
| R5 | 声明了故障但无任何信号 | INCOMPLETE / EXPECTED_FAULT_NOT_OBSERVED |
| R6 | child_rc≠0 | UNEXPECTED_FAULT / CHILD_NONZERO_EXIT；若同时有完整 PASS 则为 CHILD_NONZERO_EXIT_WITH_PASS_ARTIFACT |
| R7 | result.json 缺失 | INCOMPLETE / RESULT_JSON_MISSING |
| R8 | result.json 非法 | INCOMPLETE / RESULT_JSON_INVALID |
| R9 | gate_result == "FAIL" | INCOMPLETE / GATE_SAYS_FAIL |
| R10 | manifest / JSONL 违规 | INCOMPLETE / METADATA_INCONSISTENT 或 JSONL_INVALID |
| R11 | 全绿 | SUCCESS / ALL_CHECKS_GREEN |

---

## 2. A2：修掉"child_rc=0 就是成功"的假成功判定

### 2.1 三个硬信号

| 信号 | 实现 | 关键点 |
|---|---|---|
| (a) 强类型故障记录 | read_fault_record(out_dir)，文件名 fault_record.json（P0-07 事实标准），兼容 PR #17 提议的 failure.json | typed 要求 JSON 对象且 fault、code 均为非空字符串；**文件存在本身就是故障信号**：损坏/非对象/缺字段 → typed=False 且带 error，绝不退化成"无故障" |
| (b) stderr traceback 签名 | detect_traceback(stderr_text) | 见 2.2 的**成对**规则 |
| (c) result.json 缺失 | 复用 P0-02 的 fail-closed 读取 | 缺失 → RESULT_JSON_MISSING（不是 PASS） |

### 2.2 traceback 的精确模式（含必须避免的假阳性）

**必须成对**，因为单独一行 ExceptionName: message 在真实 Kit stderr 里会误报——P0F_exception_r01 的 stderr 里就有
"/home/dgut/runs/g/scripts/simulation/p3_dynamic_mount.py:298: DeprecationWarning: The function 'write_root_pose_to_sim' will be deprecated ..."，
这一行**完全符合 terminator 形状**，但它不是故障。

判定规则（TRACEBACK_HEADER_RE / TRACEBACK_FRAME_RE / EXCEPTION_TERMINATOR_RE / FATAL_PYTHON_RE）：

1. 必须出现 header：以 "Traceback (most recent call last):" 结尾的行（允许前导空白）；**且**
2. 以下之一：
   * **TRACEBACK_PAIR（最强）**：header 之后先出现 ≥0 个 frame（形如 File "路径", line N），随后出现**列 0 起头**的 ExceptionType: message 行（例如 RuntimeError: INJECTED_PYTHON_EXCEPTION at step 60），异常类型可提取；
   * **TRACEBACK_TRUNCATED**：只有 header + ≥1 个 frame，没有 terminator（进程或管道把 stderr 截断）；
   * **FATAL_PYTHON_ERROR**：列 0 的 "Fatal Python error:"。
3. 反向证据同样记录：只有 header 而无 frame/terminator，**不算** traceback。

单元测试用 7 个样本钉死这 7 种情形（empty / kit_deprecation_only / bare_exception_line / bare_header / pair / truncated / fatal_python_error），其中 kit_deprecation_only 就是**假阳性对照组**。

### 2.3 声明匹配（token 包含）

--expect-fault python_exception 与 fault_record.code = INJECTED_PYTHON_EXCEPTION、以及 traceback 里的 INJECTED_PYTHON_EXCEPTION 都规范化成小写下划线 token 后做**包含匹配**，因此 python_exception ⊂ injected_python_exception，joint_limit ⊂ joint_limit_refused。
**已知限制**：极短的故障名可能过度匹配；但匹配失败永远是 fail-closed（UNEXPECTED_FAULT），过度匹配**不可能**把故障变成 SUCCESS（见第 8 节 UNKNOWN）。

### 2.4 声明从哪来（否则"注入 run"会被当成"仿真 run"）

| 来源 | 实现 | 记录字段 |
|---|---|---|
| 用户显式 | CLI --expect-fault NAME（可重复） | declared_faults / declared_fault_source = "cli" |
| 转发给子进程的注入开关 | --inject-fault X / --fault-injection X / 等号形式 | declared_fault_source = "forwarded_child_args"（两者都有则 "cli+forwarded_child_args"） |
| 已提交证据的离线重放 | declared_faults_from_command(summary["command"]) | 见第 5 节 |

---

## 3. A1：接口对照表（上游 PR #14/#16/#17 vs 本 runner）

上游代码位置：PR #14 scripts/simulation/p0_subprocess_gate_runner.py（144 行；PR #16 原样集成并加离线 CI；PR #17 增补 failure.json 与 p2_tensor_readback.py）。

| 维度 | 上游 PR #14/#16 | 本 runner | 兼容性 |
|---|---|---|---|
| CLI 目标 | --report（必填，gate JSON 路径）+ --output-dir（必填，日志目录）+ "--" + command REMAINDER | --script + --run-id + --out-dir；子进程参数经 "--" 透传 | **不兼容（有意）**：本 runner 自己拼子进程命令行，才能强制 run_id/out-dir 一致 |
| 超时 | --timeout-sec（默认 900），超时 **outer exit 124** | --timeout（默认 1800），超时 **outer exit 1**，child_rc 记 124 哨兵 | 语义更强：124 只作 child 哨兵，避免"外层 124"与"子进程 124"混淆 |
| 新建目录 | report.exists() 或 output_dir.exists() → argparse error（exit 2） | 严格 run-dir 契约：按 run_id 比对，退出码 2/3/4/5 + gate_runner_refusal.json 审计记录 | 上游"目录已存在就拒绝"是其子集；本 runner 另有 --allow-reused-out-dir 逃生口并把旧 result/fault 隔离 |
| 日志文件名 | child.stdout.log / child.stderr.log，以 "x" 独占创建 | gate_child.stdout.log / gate_child.stderr.log（"w"，因 strict out-dir 保证目录全新） | **不兼容**（文件名与独占语义） |
| gate JSON 结构 | gates 或 acceptance 段任一（非空 bool 字典）+ 可选 gate_result；至少一条显式检查 | **必须** gate_result ∈ {PASS, FAIL} **且** 非空 gates 全 bool true；另有 JSON 合法性/非有限数/run_id 一致性 | 本 runner **更严**：上游接受"只有 acceptance 没有 gate_result"的文档，本 runner 判 INVALID |
| run_id 绑定 | 无 run_id 概念 | --run-id 与 result.json / owner marker / 目录名三方比对 | 新增能力 |
| 元数据一致性 (P0-05) | 无 | --gravity-z / --steps / --dt vs manifest.json | 新增能力 |
| 逐帧数据完整性 (P0-06) | 无 | check_pose_jsonl（单调 step、帧数、有限性、contact 静默 0） | 新增能力 |
| 超时杀进程 | POSIX killpg(SIGKILL)；Windows proc.kill()（**无子孙回收**） | POSIX SIGTERM→SIGKILL；Windows taskkill /F /T + PID 快照前后核对 + descendant_cleanup_verified（可为 null = UNKNOWN） | 本 runner 更强且更诚实 |
| 故障语义 | 无（只有 child rc + gate） | 五类词表 + 纯分类器 + typed fault / traceback / crash 检测 | **新增**，本批核心 |
| summary 字段 | child_command, child_returncode_observed, child_timed_out, spawn_error, gate{status,reason,...}, outer_exit_code, report_path, stdout_path, stderr_path, started_utc, finished_utc | 保留既有字段（child_returncode, json_gate, json_gate_status, child_ok, gates_ok, outer_exit, timed_out, elapsed_s）并**新增** outcome, outcome_rule, outcome_reason_code, outcome_reason, outcome_exit, declared_faults, declared_fault_source, fault_verdict{traceback,crash,fault_record}, quarantined_stale_fault_record | 字段名不同，需适配层；本 runner 的既有字段在 R10-A **一个都没改名、没改含义** |
| 与 PR #17 failure.json | PR #17 让实验端在未捕获异常时写 failure.json 并 flush stderr | 分类器**同时**接受 fault_record.json 与 failure.json 作为类型化故障证据 | 兼容 |

**backward compatibility 结论**：

1. **对本仓库现有调用者**：--script/--run-id/--out-dir/--timeout/--allow-reused-out-dir/--gravity-z/--steps/--dt/--jsonl/--expected-frames/--require-jsonl/--check-jsonl 全部保留且语义不变；outer_exit 仍是 0/1（除**新引入的**声明式故障 run 用 10）。
2. **对上游 PR #14 的调用方**：--report/--output-dir/--timeout-sec 三个参数不等价，需要改调用方式；这是**有意的破坏**，理由是本 runner 的核心保证（run_id 三方绑定、strict out-dir）在上游 CLI 里没有落点。已在 PR 描述中显式标注，不做静默别名。
3. **summary 消费者**：新增字段只增不改；child_ok 仍表示 child_returncode == 0（**故意不改**，它是"Kit 撒谎"的原始证据，改掉会销毁审计线索），但**不得**再用它判定成败——应当读 outcome。

### 3.1 可运行的正例与反例（真实命令、真实退出码）

演示子进程：tools/simulation/gate_demo_child.py（无 Isaac、无 GPU，纯标准库）。

| # | 命令（工作目录 = 本 worktree） | 结局 | reason_code | **真实退出码** |
|---|---|---|---|---|
| 1 | python tools/simulation/p2_gate_runner.py --script tools/simulation/gate_demo_child.py --run-id DEMO_ok --out-dir _scratch_r10a/demo/DEMO_ok -- --demo-gate PASS | SUCCESS | ALL_CHECKS_GREEN | **0** |
| 2 | 同上，但 --run-id DEMO_fail --out-dir _scratch_r10a/demo/DEMO_fail -- --demo-gate FAIL | INCOMPLETE | GATE_SAYS_FAIL | **1** |
| 3 | 同上，但 --run-id DEMO_fault --out-dir _scratch_r10a/demo/DEMO_fault -- --inject-fault joint_limit | EXPECTED_FAULT | FAULT_RECORD_TYPED | **10** |
| 4 | 同上，但 --run-id DEMO_tb --out-dir _scratch_r10a/demo/DEMO_tb -- --demo-traceback --demo-gate PASS | UNEXPECTED_FAULT | TRACEBACK_WITHOUT_EXPECTATION | **1** |
| 5 | 同上，但 --run-id DEMO_expect --out-dir _scratch_r10a/demo/DEMO_expect --expect-fault python_exception -- --inject-fault python_exception | EXPECTED_FAULT | FAULT_RECORD_TYPED | **10** |

第 4 行就是 A2 的现场证明：**result.json 是完整合法 PASS、child_rc=0，只因为 stderr 有 "Traceback (most recent call last):" + "RuntimeError: …" 这一对，判 UNEXPECTED_FAULT、exit 1**。第 3/5 行证明"故障测试 PASS"走 10 而不是 0。

---

## 4. A4：fail-closed 回归矩阵

文件：tests/simulation/test_p2_gate_runner_faults.py（53 条，全部通过）。

| 场景 | 断言要点 | 退出码 |
|---|---|---|
| 超时 | INCOMPLETE / TIMEOUT，child_rc = 124 哨兵，descendant_kill 记录 | 1 |
| SIGSEGV 类（139 / 134 / 137 参数化） | CRASH / CRASH_ABNORMAL_TERMINATION，即使 result.json 是 PASS | 1 |
| Kit crash reporter 文本 | detect_crash 命中 STDERR_KIT_CRASH_REPORTER_LOG / KIT_TERMINATING_WITH_EXIT_CODE | — |
| result.json 缺失（child_rc=0） | INCOMPLETE / RESULT_JSON_MISSING | 1 |
| result.json 损坏 | INCOMPLETE / RESULT_JSON_INVALID，且写出的 summary 仍是严格 JSON | 1 |
| 旧目录残留 | 子进程**未启动**、无 summary、留下 gate_runner_refusal.json，退出码 ∈ {2,3,4,5} | 2–5 |
| 非零 rc + 合法成功产物 | UNEXPECTED_FAULT / CHILD_NONZERO_EXIT_WITH_PASS_ARTIFACT（结构完美的产物不能推翻非零进程状态） | 1 |
| 旧 fault_record 残留 | 被隔离为 fault_record.json.prev-*，不归属新子进程；新子进程干净 PASS → SUCCESS | 0 |
| 外来 run_id 的 fault_record | 先隔离，再按新子进程自身证据判定 | 1 |
| 交叉不变量 | 全矩阵断言 exit 0 当且仅当 SUCCESS | — |

六次 P0F 原始目录**未被修改**：重放全程只读（git show + 临时目录），本分支没有 outputs/.../P0F_* 的 diff。

---

## 5. A2 证明：重放已提交的真实 P0F 证据

工具：tools/simulation/replay_p0f_fault_evidence.py（只读；把证据 git show 到临时目录再分类）。

**重要事实（必须写在最前）**：.gitignore:13 的规则 "*.log" 使子进程的 gate_child.stderr.log **从未进入 git**。因此 6593736 提交里**只有** gate_runner_summary.json / raw_poses.jsonl / authored_p3_scene.usda / fault_record.json（部分 run），**没有 stderr 日志**；stderr 只在产出机器上是**被忽略的本地文件**。所以重放分两种模式，各自把期望值写死在代码里（EXPECTED_COMMITTED / EXPECTED_WITH_LOGS），**不隐藏差别**。

真实命令与**真实退出码**：

    python tools/simulation/replay_p0f_fault_evidence.py
    -> mode=committed_only ... all six classified as expected: YES / none of them is SUCCESS: YES
    -> COMMITTED_ONLY_EXIT=0

    python tools/simulation/replay_p0f_fault_evidence.py --logs-dir <含 *.log 的 checkout>
    -> mode=with_local_ignored_logs ... all six classified as expected: YES / none is SUCCESS: YES
    -> WITH_LOGS_EXIT=0

### 5.1 仅用提交证据（committed_only，SHA 65937365adaacb49b3d7e3e24ecdf38fda1ccf21）

| RUN_ID | child_rc | fault_record | traceback | 结局 | reason_code | exit | 判定 |
|---|---|---|---|---|---|---|---|
| P0F_exception_r01 | 0 | 无 | 无（日志未提交） | INCOMPLETE | EXPECTED_FAULT_NOT_OBSERVED | 1 | MATCH |
| P0F_exception_r02 | 0 | fault_record.json | 无 | EXPECTED_FAULT | FAULT_RECORD_TYPED | 10 | MATCH |
| P0F_exception_r03 | 0 | fault_record.json | 无 | EXPECTED_FAULT | FAULT_RECORD_TYPED | 10 | MATCH |
| P0F_jointlimit_r01 | **139** | 无 | 无 | **CRASH** | CRASH_ABNORMAL_TERMINATION | 1 | MATCH |
| P0F_jointlimit_r02 | 0 | fault_record.json | 无 | EXPECTED_FAULT | FAULT_RECORD_TYPED | 10 | MATCH |
| P0F_jointlimit_r03 | 0 | fault_record.json | 无 | EXPECTED_FAULT | FAULT_RECORD_TYPED | 10 | MATCH |

### 5.2 叠加本地被忽略的 stderr 日志（--logs-dir）

| RUN_ID | child_rc | fault_record | traceback | 结局 | reason_code | exit | 判定 |
|---|---|---|---|---|---|---|---|
| P0F_exception_r01 | 0 | 无 | **TRACEBACK_PAIR** | **EXPECTED_FAULT** | TRACEBACK_ONLY | 10 | MATCH |
| P0F_exception_r02 | 0 | fault_record.json | 无 | EXPECTED_FAULT | FAULT_RECORD_TYPED | 10 | MATCH |
| P0F_exception_r03 | 0 | fault_record.json | 无 | EXPECTED_FAULT | FAULT_RECORD_TYPED | 10 | MATCH |
| P0F_jointlimit_r01 | 139 | 无 | 无 | CRASH | CRASH_ABNORMAL_TERMINATION | 1 | MATCH |
| P0F_jointlimit_r02 | 0 | fault_record.json | 无 | EXPECTED_FAULT | FAULT_RECORD_TYPED | 10 | MATCH |
| P0F_jointlimit_r03 | 0 | fault_record.json | 无 | EXPECTED_FAULT | FAULT_RECORD_TYPED | 10 | MATCH |

**两种模式的共同硬结论**：六次真实故障运行**没有一次被判成 SUCCESS、没有一次 exit 0**。4 次 child_rc 被 Kit 吞成 0 的 run 仍被 typed fault_record 拦下（FAULT_RECORD_TYPED）；唯一真正死亡的 joint_limit r01 保留 SIGSEGV 证据（139 + crash reporter）判 CRASH。

**差异的诚实解释**：异常 r01 当时没有写出 fault_record（见 PHASE_P0_07 第 1 节表格），其故障证据只有 stderr traceback；而 traceback 日志不在 git 里，所以**仅凭提交证据无法证明"类型化故障"**，只能判 INCOMPLETE（EXPECTED_FAULT_NOT_OBSERVED）——这正是 fail-closed 的正确含义：**缺证据既不许 PASS，也不许把未证实的注入说成"预期故障已复现"**。叠加本地日志后才升级为 EXPECTED_FAULT / TRACEBACK_ONLY（弱证据，明确标注）。

---

## 6. 真实测试命令与真实退出码

    cd E:\具身智能\badmin_wt_r10a
    python -m pytest tests/simulation/test_p2_gate_runner.py -q
    -> 92 passed in 101.92s
    -> PYTEST_EXIT=0                      （原有 92 条一条未改、全绿）

    python -m pytest tests/simulation/test_p2_gate_runner_faults.py -q
    -> 53 passed in 53.02s
    -> PYTEST_EXIT=0                      （R10-A 新增）

    python tools/simulation/replay_p0f_fault_evidence.py                 -> 0
    python tools/simulation/replay_p0f_fault_evidence.py --logs-dir ...  -> 0

---

## 7. 改动清单

| 文件 | 类型 | 说明 |
|---|---|---|
| tools/simulation/p2_gate_runner.py | 修改 | Outcome 枚举 + 词表退出码 + RunEvidence/OutcomeVerdict + classify_run + detect_traceback/detect_crash/read_fault_record + --expect-fault + 旧 fault 记录隔离 + summary 新字段。**未修改**任何既有断言含义、阈值或 gate 判据 |
| tools/simulation/replay_p0f_fault_evidence.py | 新增 | P0-07 证据只读重放（git show → 临时目录），两种模式、期望表写死、退出码即结论 |
| tools/simulation/gate_demo_child.py | 新增 | A1 正反例的可运行子进程（无 Isaac） |
| tests/simulation/test_p2_gate_runner_faults.py | 新增 | A2/A3/A4 回归 53 条 + P0F 重放断言 |
| management/morph_one_rs/R10_A_RUNNER_FAULT_SEMANTICS.md | 新增 | 本报告 |

**附带的一处必要修改（已记录理由）**：删除了该文件的 "from __future__ import annotations"。原因：本仓库的回归 fixture 用 importlib.util.module_from_spec **按路径**加载 runner 且不注册 sys.modules，在 PEP 563 下 dataclass 会通过 sys.modules[cls.__module__] 解析字符串注解并抛 AttributeError（已实测）。该 runner 除新代码外没有任何注解，删除后唯一效果是让 dataclass 正常构造。

---

## 8. UNKNOWN / 未做到（不填充、不宣称）

1. **traceback 日志不在版本库里**：.gitignore:13 的 "*.log" 使六次 P0F 的 gate_child.stderr.log 无法从 git 复算。5.2 的 EXPECTED_FAULT 依赖**本地产物**，其 sha256 **无法**与提交对账——这是明确 UNKNOWN，不是"已验证"。建议后续把关键 stderr 以 *.log.txt 或截断文本形式入库（本批不做，避免改动历史证据）。
2. **Windows 侧未复现 Kit 吞异常**：本批全部在 Windows + CPython 3.11.9 上跑纯 Python 子进程，未能在 Windows 上验证 Kit 的 SIGSEGV 行为；SIGSEGV 分支由真实 Linux Kit 的 139 证据 + 合成 rc 参数化测试共同覆盖，**不是** Windows 实机验证。
3. **声明匹配用 token 包含**：极短故障名理论上会过度匹配（例如声明 "a"）。失败方向永远是 fail-closed，但"过度匹配导致把不相关故障认成预期故障"这一风险未量化。
4. **未做**：GPU OOM、Kit 扩展加载失败、看门狗超时后的子孙进程残留、跨 run 的 manifest/trace 溯源（属 C 线）。
5. **未改动**：六次 P0F 原始目录、scripts/simulation/p2_*/p3_*/p4_*、simulation/、src/、队列状态文件。**没有**启动 Isaac/Kit，**没有**写 172.31.68.251。
6. **半故障情形**：fault_record 存在但同时 result.json 是 FAIL 且已声明并匹配 → EXPECTED_FAULT（FAIL 与故障一致，不矛盾，R3E 覆盖）；反之 fault + PASS 必须是 UNEXPECTED_FAULT（R3D）。

---

## 9. 证据入口

* 分支：r10/a-runner-faults（基线 test/p0-gate-runner-strict-outdir@ce278c2）
* 重放证据 commit：65937365adaacb49b3d7e3e24ecdf38fda1ccf21（= p3/gate-v1.1@6593736）
* 原始 run 目录：outputs/simulation/rebot_b601_rs/morph_one/p2_wc_gates/P0F_*/（只读）
* 相关文档：management/morph_one_rs/PHASE_P0_07_FAULT_INJECTION.md（P0-07 事实）、P2_GATE_RUNNER_STRICT_OUTDIR.md（run-dir 契约）、P0_GATE_RUNNER_TESTS.md（既有 92 条来源）
