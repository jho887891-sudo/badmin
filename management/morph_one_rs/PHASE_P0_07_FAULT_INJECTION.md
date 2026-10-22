# P0-07 / P3-05：真实 Kit 故障注入结果（含两个机制性缺陷）

**分支**：`p3/gate-v1.1@2252802` · **GPU**：Isaac Sim 6.0.1 / RTX A6000 · **脚本**：`p3_dynamic_mount.py --inject-fault`
**目的**：验证"进程崩溃/异常绝不能被包装成 PASS"（P0-07），以及关节限位必须**写前拒绝**（P3-05）。

## 1. 三轮真实注入（每轮两个故障，全部 outer exit 1）

| RUN_ID | 注入 | 结果 | child_rc | `fault_record.json` | outer exit |
|---|---|---|---|---|---|
| `P0F_jointlimit_r01` | 关节限位（raise SystemExit 于循环内） | Kit **SIGSEGV** | **139** | 无 | **1** |
| `P0F_exception_r01` | 循环内抛 RuntimeError | traceback 打印后**进程仍 exit 0** | **0** | 无 | **1** |
| `P0F_jointlimit_r02` | 限位（循环前拒绝，`return 3`） | 记录正确，退出码被吞 | **0** | `JOINT_LIMIT_REFUSED`（joint5，目标 2.07 rad > 上限 1.57 rad，targets_written=0） | **1** |
| `P0F_exception_r02` | 记录后 `return 4` | 同上 | **0** | `INJECTED_PYTHON_EXCEPTION`（step 60） | **1** |
| `P0F_jointlimit_r03` | 关闭 Kit 后 `raise SystemExit(3)` | 同上 | **0** | 同上 | **1** |
| `P0F_exception_r03` | 关闭 Kit 后 `raise SystemExit(4)` | 同上 | **0** | 同上 | **1** |

## 2. 两个机制性缺陷（本条最重要）

### 缺陷 A：Kit 吞掉 Python 异常，进程仍然 exit 0
`P0F_exception_r01` 打印了完整 `RuntimeError: INJECTED_PYTHON_EXCEPTION` traceback，
但 `gate_runner_summary.json` 记录的 `child_returncode` 是 **0**。
→ **任何只依据子进程返回码判定成败的检查都会得到假 PASS。**

### 缺陷 B：本构建下 Python 层面的退出码**不可控**
即使先 `SIMULATION_APP.close()` 再 `raise SystemExit(3/4)`，子进程仍是 **0**；
唯一"穿透"的是**信号**（SIGSEGV → 139）。也就是说：`SystemExit` 在本 Kit 构建里**不是一个可靠的失败信号**。
另外，在第一轮里于步进回调内 `raise SystemExit` 直接把 Kit 打成 **SIGSEGV**，
既不干净也可能留下脏的 GPU 状态。

## 3. 目前真正可靠的口径（本轮已验证）

| 可靠 | 不可靠 |
|---|---|
| **外层 exit code**（runner 的 JSON 门禁判定）：6/6 故障运行均为 **1** | `child_returncode`：6 次中有 4 次为 0 |
| **artifacts 状态**：故障 run **没有** `result.json`，runner 因此判 FAIL | 脚本内 Python 退出码 |
| **typed fault 记录**：`fault_record.json` 的 `code` 字段（JOINT_LIMIT_REFUSED / INJECTED_PYTHON_EXCEPTION），并带 `targets_written=0`、`sim_steps_taken` | traceback 文本（只能表明发生过异常，不决定结果） |

**结论**：P0-07 的要求达成——异常与限位拒绝**都没有产生 PASS**，且证据链不依赖不可靠的子进程返回码。
但这同时说明 **runner 必须把 `fault_record.json` 当作硬 FAIL 信号**，并在子进程 stderr 出现 traceback 时拒绝 PASS（当前靠"缺 result.json"这个副作用兜住，属于间接保护）。

## 4. 关节限位拒绝的真实数据

`joint5`：USD 上限 **1.5700000524520874 rad**，注入目标 **2.0700000524520874 rad**，
拒绝发生在**任何写入之前**（`targets_written: 0`，`sim_steps_taken: 0`）。

## 5. 尚未做到

- 没有实现 runner 侧的 `fault_record.json`/traceback 检测（属于 `test/p0-gate-runner-strict-outdir` 分支的后续改动）。
- 未测：GPU OOM 注入、Kit 扩展加载失败、显存泄漏、看门狗超时后的子孙进程残留（A04 只做了只读审计）。
- 未在 Windows 侧验证同一现象。

## 6. 证据

`outputs/simulation/rebot_b601_rs/morph_one/p2_wc_gates/P0F_*/`（6 个 run：`gate_runner_summary.json`、
`fault_record.json`（3 个 run 有）、child stdout/stderr、outer.log）。
