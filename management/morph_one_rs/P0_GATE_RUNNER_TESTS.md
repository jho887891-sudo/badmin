# P0 自动化回归：P2 外层 gate runner（Issue #13）

**结论：PASS**（本分支 pytest 18/18 通过，退出码 0；同时发现并修复了 2 个 fail-open 缺陷，修复前后均有真实证据）

* 仓库 / worktree：`E:\具身智能\badmin_wt_p2`（分支 `test/p0-gate-runner-regression`，从 `459cb12` 建）
* 被测文件：`tools/simulation/p2_gate_runner.py`
  （当前 sha256 `A878D0C13A35776C11F8B17EDACDB62C9ECFD1EDF8FC7E4599025771885D9F6D`）
* 新增测试：`tests/simulation/test_p2_gate_runner.py`
  （sha256 `6AE1C5138F9AE798C117BB8537580A1BAC5244D0811B5A6F30EB9410D112085A`，18 个用例，301 行）
* 全程未使用 GPU / Isaac Sim / torch；只跑 CPU 上的假实验脚本，远程 GPU 主机未参与本轮。

## 1. 契约（本 P0 唯一的验收口径）

    outer exit 0  <=>  子进程 returncode == 0  AND  result.json:gate_result == "PASS"

即 Kit 把失败吞成 0 时，外层仍必须给出非 0。所有用例都在断言这一条或其配套可审计字段。

## 2. 环境

| 项 | 值 |
|---|---|
| python | 3.11.9 |
| pytest | 9.0.2 |
| 解释器 | `C:\Users\abcd1\AppData\Local\hermes\hermes-agent\venv\Scripts\python.exe` |
| 平台 | Windows-10-10.0.26100-SP0 |
| 工作目录 | `E:\具身智能\badmin_wt_p2` |

分支基线说明：任务书写的 worktree 分支是 `dev/morph-one-p2-quat-and-settle`，实际 worktree 当时 check out 在
`dev/morph-one-p2-init-diagnostics`。两者指向**同一个提交** `459cb122f7be68ad60295ee61c4faaf73618b151`，
且 `tools/simulation/p2_gate_runner.py` 在两分支间 diff 为空，因此基线等价，按 `459cb12` 建分支。

## 3. 命令、真实输出、退出码

    $ python -m pytest tests/simulation/test_p2_gate_runner.py -q
    ..................                                                       [100%]
    18 passed in 12.08s

    EXITCODE=0

同一文件的 verbose 运行（`pytest ... -v`）同样 18 passed、退出码 0；用例清单见第 4 节。

## 4. 覆盖点（18 个用例）

| 用例 | 断言的要害 |
|---|---|
| `test_outer_exit_is_pass_only_when_child_ok_and_gate_pass[rc0_gatePASS]` | 0/0/PASS → outer 0（真成功） |
| `...[rc0_gateFAIL_kit_swallowed]` | **rc=0 + FAIL → outer 1**（核心契约，Kit 吞失败） |
| `...[rc1_gatePASS]` | rc=1 + PASS → outer 1（JSON 说 PASS 也不放行） |
| `...[rc1_gateFAIL]` | 1/1/FAIL → outer 1 |
| `test_kit_swallowed_failure_is_rejected_even_though_the_child_exited_zero` | 单独立项的显式回归：`child_returncode=0` 被原样保留，同时 `outer_exit=1`、`json_gate=FAIL` |
| `test_missing_result_json_exits_one_and_marks_gate_missing` | 无 result.json → outer 1，`json_gate="MISSING"`，`json_metrics={}` |
| `test_summary_preserves_raw_child_returncode[0/1/3/7]` | summary 里 `child_returncode` 是**原始值**（不是布尔、不截断），`timed_out=false`，`timeout_s=1800.0`（默认值未被改动），`command` 含子进程脚本绝对路径 |
| `test_child_stdout_and_stderr_are_captured_to_logs` | 子进程 stdout/stderr 落 `gate_child.*.log`，`json_metrics` 原样进 summary |
| `test_arguments_after_separator_are_forwarded_to_the_child` | `-- --lift 0.40 --write-data` 原序透传给子进程 |
| `test_real_process_exit_code[rc0_gatePASS / rc0_gateFAIL / rc1_gatePASS]` | 用**真 subprocess** 跑 `p2_gate_runner.py`，断言 OS 层退出码 0/1/1 与 stdout JSON |
| `test_timeout_is_enforced_and_reported` | 修复项 D2 的回归（见第 5 节） |
| `test_stale_result_json_is_quarantined_and_cannot_pass_a_silent_child` | 修复项 D1 的回归 |
| `test_fresh_run_quarantines_nothing` | 干净目录不被误隔离（`quarantined_stale_result=null`） |

测试实现上不依赖 Isaac Sim：每个用例在 `tmp_path` 里生成一个 `fake_experiment.py`，它解析
`--run-id/--out-dir`（外加 `--fake-gate`、`--fake-write` 之类的自测开关）、写或不写
`result.json`、再 `sys.exit(<指定码>)`；runner 通过 `importlib` 按路径加载后直接调用 `main(argv)`，
因此外层退出码可被逐值断言，另有 3 个用例用真进程验证 `raise SystemExit` 的接线。

## 5. 发现并修复的缺陷（均在本分支提交，修复前后都有实测）

### D1（严重，fail-OPEN）：复用 `--out-dir` 时旧 `result.json` 可被"沉默子进程"继承成假 PASS

runner 启动子进程前从不清空/校验 `<out_dir>/result.json`。于是"上一轮 PASS 的目录 + 本轮子进程什么都没写但 rc=0"
（正是 Kit 吞失败的样子）会被判 **PASS**，即这条 P0 契约在目录复用时被绕开。

* 修复：新增 `_quarantine_stale_result()`，在启动子进程**之前**把旧 `result.json` 改名为
  `result.json.prev-<UTC时间戳>-<run_id>.json`（证据保留，不删除），summary 新增
  `quarantined_stale_result` 记录被隔离文件的路径；子进程没写新文件就只能是 `MISSING` → 退出 1。
* 实测（同一份探针脚本，同一台机器）：

      BEFORE（459cb12 原版 runner）: outer_exit=0  json_gate=PASS     child_rc=0  quarantined=<键不存在>   ← 假 PASS
      AFTER （本分支 runner）      : outer_exit=1  json_gate=MISSING  child_rc=0  quarantined=...\result.json.prev-...json

### D2（静默忽略）：`--timeout` 被解析后从未使用

`subprocess.run(...)` 没传 `timeout`，`args.timeout` 是死参数：子进程挂死时外层**永久阻塞**，也不会留下任何
summary / 证据文件（`gate_runner_summary.json` 只在子进程结束后才写）。

* 修复：`subprocess.run(..., timeout=args.timeout)`；捕获 `subprocess.TimeoutExpired` 后
  `child_returncode=124`（`timeout(1)` 惯例）、`timed_out=true`、stderr 日志追加一行
  `[p2_gate_runner] child exceeded --timeout=... s and was killed`，summary 照常写出；outer 退出 1。
* 实测（`--timeout 0.5` + 睡眠 5 s 的子进程）：

      BEFORE: outer_exit=0  elapsed=5.54s  timed_out=<键不存在>   ← 预算被完全无视
      AFTER : outer_exit=1  elapsed=0.55s  timed_out=True

### diff 摘要（`tools/simulation/p2_gate_runner.py`，+64/-8）

1. 模块 docstring 增加 6 行，写明两条 fail-closed 细节；
2. 新增 `TIMEOUT_RETURNCODE = 124`、`_as_text()`、`_quarantine_stale_result()`；
3. `main()`：`result_path` 提前到启动子进程之前并执行隔离；`subprocess.run` 加 `timeout=` 与
   `try/except TimeoutExpired`；日志写入改用 `stdout_text/stderr_text`；
4. summary 新增 3 个键 `timed_out` / `timeout_s` / `quarantined_stale_result`，控制台 JSON 多打印 `timed_out`。
   **既有字段与判定语义完全不变**（`child_returncode` 仍是原值；`outer_exit = 0 iff rc==0 and gate=="PASS"`）。

兼容性核对：仓库里已有真实证据 `outputs/simulation/rebot_b601_rs/morph_one/p2_real_chassis/RC_lift0300_r02/`
与 `RC_lift0334_r02/` 的 `gate_runner_summary.json` 是**旧 schema（无新增 3 键）**，确认那两次真实运行用的仍是
修复前的 runner；新增键为向后兼容的纯增量，旧 summary 仍可正常读取，未回写、未改动任何历史产物。

## 6. 复现方式（含"修复前"如何重跑出失败）

    cd E:\具身智能\badmin_wt_p2
    python -m pytest tests/simulation/test_p2_gate_runner.py -q          # 本分支: 18 passed, exit 0

    # 用未修复的 runner 跑同一份测试（证明这些用例真的抓得住缺陷；会重写工作区文件，注意 stash/pop）
    git stash push -q -- tools/simulation/p2_gate_runner.py
    python -m pytest tests/simulation/test_p2_gate_runner.py -q          # 实测: 10 failed, 8 passed, exit 1
    git stash pop -q

"修复前"失败为 10 个而非 2 个：除 D1/D2 两个针对性用例外，另有 8 个用例断言的是新增 summary 键
（`timed_out`/`timeout_s`/`quarantined_stale_result`）与新版控制台 JSON，旧版自然缺失——这是 schema 断言，
不是契约回归。**核心契约的 5 项要求（rc0+PASS→0、rc0+FAIL→1、rc1+PASS→1、缺失→1+MISSING、保留原始 rc）
在未修复的 459cb12 上本来就是通过的**（见下节时间线）。

## 7. 过程时间线（便于核对，不是结论分级）

1. 先只写契约用例跑基线：`4 failed, 11 passed` —— 失败原因是我自己的用例写错了
   （把 `summary["command"]` 当成 runner 命令行，实际它是**子进程**命令行）。改成断言子进程脚本路径后
   `15 passed`，退出码 0：即 459cb12 原版 runner 的 5 项契约要求本就成立。
2. 再用临时探针（未入库的 `_scratch_*` 脚本）证伪 D1/D2，得到上面 BEFORE 数字。
3. 修复 runner，补 3 个新用例（timeout / 陈旧 json / 干净目录），`18 passed`，退出码 0。
4. 用 stash 把修复摘掉重跑同一套测试：`10 failed, 8 passed`，退出码 1 —— 即新用例确实锁定修复行为。

## 8. UNKNOWN（未验证，不得当作已通过）

1. **Linux / Kit 真机链路上的超时行为未验证。** 本轮只在 Windows 上用纯 python 子进程验证 `--timeout`。
   POSIX 下 `subprocess.run(timeout=)` 只 kill 直接子进程；Isaac Kit 若再 spawn 自己的 worker，
   孙子进程是否残留 **UNKNOWN**（要 `setsid`+`killpg` 才能保证整棵树，本轮未做，也未在远程实测）。
2. **`result.json` 内容非法（写了一半 / 非 JSON）时的行为未加测试。** 当前实现会抛
   `json.JSONDecodeError`（fail-loud，不经 summary、退出码由异常决定）。若外层还有包装器把这个异常当 0，
   风险 **UNKNOWN**；本轮未改此路径。
3. **D1 修复对既有自动化脚本的影响未在远程实测。** 历史行为是"复用目录可继承旧 PASS"，现在是 `MISSING`；
   凡是靠复用 `--out-dir` 的旧调用链（未在本轮普查）行为会变严（预期方向），实际影响 **UNKNOWN**。
4. **未跑全仓 pytest。** 只跑了本文件（未验证 pytest 9.0.2 与仓库其它历史用例的兼容性）。
5. **提交边界被并发工作混入。** 本分支上 `fd9073b` 由同一 worktree 里的并发 P2 工作提交，把本任务的
   test+runner 修复与 P2 chassis 修复、`RC_lift*_r02` 证据打进了同一个提交。未 push、未改写历史（禁止
   force push），本报告与其后的一次提交都叠在其上。是否把这些改动迁回 P2 分支，留给 review 决定。
6. **未在 GPU 主机 / Isaac Sim 上执行任何一步**（任务禁令，且本测试不需要）；`/home/T7/ojh/robot_sim/...`
   路径未涉及。
