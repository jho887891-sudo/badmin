# P2 gate runner: strict --out-dir contract (P0-04)

Owner: outer gate runner `tools/simulation/p2_gate_runner.py`
Branch: `test/p0-gate-runner-strict-outdir` (base `test/p0-gate-runner-regression`)

## 规则（fail-closed）

一个 `--run-id` 对应一个证据目录。若 `--out-dir` 里已经存在下列任一证据文件，
且其 `run_id` 与本次 `--run-id` 不同（或无法解析/无 run_id），runner **在启动 child 之前拒绝**：

- `result.json`
- `gate_runner_summary.json`
- `manifest.json`
- `.p2_gate_run_owner.json`（runner 自己写的占用标记）

拒绝时：

1. 不启动 child（`child_started=false`），不写 `gate_runner_summary.json`，
   不改动旧证据一个字节；
2. 写 `gate_runner_refusal.json` 作为审计记录；
3. 退出码非 0，`error_code="OUT_DIR_REUSED"`，并带更细的 `refusal_code`：

| refusal_code | 退出码 | 含义 |
|---|---|---|
| `OUT_DIR_RUN_ID_MISMATCH` | 2 | 目录里的证据属于**另一个** run_id |
| `OUT_DIR_RESUME_SAME_RUN_ID` | 3 | 同一 run_id 已有上一次尝试的痕迹（断点续跑需显式放行） |
| `OUT_DIR_NAME_CONFLICT` | 4 | 目录 basename 与 run_id 不符，但有 marker 说明该目录被复用 |
| `OUT_DIR_NOT_EMPTY_UNRECOGNIZED` | 5 | 目录非空但没有可解析的 run 证据，不猜测归属 |

因此"复用目录 + 旧 PASS"不再可能伪造成本次 PASS（历史 fail-open：child exit 0 但
什么都不写时，旧 `result.json` 被当成新结论）。

## 逃生舱 `--allow-reused-out-dir`

显式传入后才允许进入已用目录，且：

- summary 记录 `allow_reused_out_dir=true`、`reused_out_dir=true`、`out_dir_reuse_code`、
  `prior_evidence`、`run_attempt`；
- 旧的 `result.json` 被隔离重命名为 `result.json.prev-<UTC 时间戳>-<run_id>.json`
  （历史兼容策略），本次 gate 只读本次 child 写出的文件。

## 交叉引用

- fail-closed gate 域、NaN/Infinity、gates 布尔断言：见 runner 模块 docstring 与
  `tests/simulation/test_p2_gate_runner.py` 第 12 节；
- 超时与整棵进程树 PID 证据：同文件第 11/15 节；
- manifest 重力一致性（P0-05）、JSONL 完整性（P0-06）：同文件第 13/14 节。
