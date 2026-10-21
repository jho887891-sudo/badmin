# A04：GPU 独占锁 + PID 校验 + 退出后 Kit 后代清理审计

- 分支：`a04/gpu-lock`（基于 `test/p0-gate-runner-strict-outdir` @ `ce278c2`，PR #23）
- 新增：`tools/simulation/gpu_lock.py`、`tools/simulation/kit_descendant_audit.py`
- 测试：`tests/simulation/test_gpu_lock.py`、`tests/simulation/test_kit_descendant_audit.py`
- 证据：`outputs/simulation/a04_evidence/`（竞争、崩溃接管、审计三态、真机只读探测）
- 不改动：原有 `tools/simulation/p2_gate_runner.py` 行为、`scripts/simulation/p*_*`、`src/`、`simulation/`、队列状态、PHASE_P2_*/P3_*/P4_* 文档。

## 1. 为什么单独做一层锁

P0 的 runner 已经做到 **超时 + 后代杀灭 + PID 证据 + 复用 out-dir 拒绝**，但它自己标注的 UNKNOWN 仍然成立：
「Kit 里再次 detach / 被 service host 接管的孙进程不在覆盖范围内」。锁与审计解决的是另外两件事：

1. **同一张 GPU 上只能有一个 Isaac 实例**：冲突必须在启动*之前*被拒绝，而不是事后从损坏的证据里发现；
2. **运行结束之后**，用 run 目录里记录过的 PID 反查「还有谁活着」，而且**绝不杀任何进程**（共享主机上可能是别人的）。

两者都是纯文件 / 纯只读探测，不引入 GPU、Isaac、torch 依赖。

## 2. 锁的合同（`tools/simulation/gpu_lock.py`）

### 2.1 锁文件

`--lock-path` 指向的单个文件，用 `open(O_CREAT|O_EXCL|O_WRONLY, 0o600)` **原子创建**（POSIX 下有 0600 权限测试）。
内容字段（测试逐项断言 + 往返读取）：

| 字段 | 含义 |
|---|---|
| `schema_version` | 当前为 1 |
| `host` | 主机名：跨主机的锁**永不抢占** |
| `pid` / `pid_start_time` | 持有者 PID 与内核启动时间（`/proc/<pid>/stat`，Windows 上为 `null`，见 UNKNOWN） |
| `run_id` | 与 run 目录一一对应的标识 |
| `expected_marker` | 持有者命令行里必须出现的标记（默认 `isaac`） |
| `acquired_at` / `heartbeat_at` | UTC ISO 时间戳；心跳用于发现「活着但已僵死」的持有者 |
| `heartbeat_timeout_s` | 心跳陈旧阈值（默认 30 s，可用 `--heartbeat-timeout` 覆盖） |
| `token` | 本次获取的 uuid4；release 必须同时匹配 pid + token |
| `history` | 每次接管的审计记录（原因、前持有者的 pid/run_id/心跳、判据、探针命令） |

### 2.2 决策：ACQUIRE / TAKEOVER / REFUSE

按顺序判定（每一步都有单元测试）：

1. 锁文件不存在 → `ACQUIRE`（`NO_LOCK`）；
2. 文件不可读 / 空 / 非对象 / 缺字段 → `TAKEOVER`（`LOCK_UNREADABLE`）：`O_EXCL` 先建文件后写内容，崩在这个窗口里必须可恢复；
3. `host` 与当前主机不同 → `REFUSE`（`FOREIGN_HOST`）：无法证明别的机器上的 PID 死活，**不偷**；
4. `pid` 等于自己 → `REFUSE`（`SELF_HELD`）；
5. 记录 PID **确定已死**（POSIX `os.kill(pid,0)` 抛 `ProcessLookupError`；Windows `tasklist /FI "PID eq N"` 无输出）→ `TAKEOVER`（`HOLDER_DEAD`）；
6. PID 活着但**命令行不再包含持有者自己记录的 `expected_marker`** → `TAKEOVER`（`PID_RECYCLED_MARKER_MISMATCH`）：PID 被系统回收给了陌生人，既不能让它继续占锁，也**不能杀它**；
7. 启动时间不一致（POSIX 可用时）→ `TAKEOVER`（`PID_RECYCLED_START_TIME_MISMATCH`）；
8. 心跳年龄 > `heartbeat_timeout_s` → `TAKEOVER`（`HEARTBEAT_STALE`），仍不杀；
9. 活着但命令行读不到（无法验证是否是我们的 Isaac）→ `REFUSE`（`LIVE_HOLDER_UNVERIFIABLE`）：**fail-closed，不偷活人的锁**；
10. 其余（活着 + 标记匹配 + 心跳新鲜）→ `REFUSE`（`LIVE_HOLDER`）。

**任何分支都不会向任何进程发信号**——`gpu_lock.py` 里没有 `os.kill` 发送信号、没有
`taskkill`、没有 `killpg`。杀灭仍然只发生在原 runner 里，锁只负责「拒绝 / 接管」。

### 2.3 接管的竞态安全

先 `os.rename(lock, lock.takeover-<utc>-<pid>)`（原子，两个竞争者也只有一个成功），
再用 `O_EXCL` 建新锁；被移走的旧内容写进 `history[0].prior`，随后删除 tombstone。
若建新锁时 `EEXIST`（别人先拿到），本进程按 REFUSE 退出。若在两次操作之间崩溃，锁**文件不存在**，下一次可直接获取——
不会出现「谁都拿不到」的死锁。

### 2.4 释放

- 上下文管理器 `__exit__`、`atexit`、`SIGINT/SIGTERM` 处理器三条路径；
- 释放前重新读文件，要求 `token` 与 `pid` **同时**匹配，过期持有者删不掉后任的锁（测试 11）；
- 信号路径释放后按 `128+signum` 退出（POSIX 测试断言 `SIGTERM` → 143 且锁文件消失）。

### 2.5 dry-run

`--dry-run` 只做投影：打印决策、`dry_run: true`，**不创建也不修改锁文件**；
投影为 REFUSE 时退出码同样是 2（测试 8 对「无锁」和「有活锁」两种情形都断言文件未被创建/未被改动）。

### 2.6 退出码

| 退出码 | 含义 |
|---|---|
| 0 | 已获取；或 dry-run 投影为 ACQUIRE/TAKEOVER；或 `--status` 查询成功 |
| 1 | 用法 / 内部错误 |
| 2 | 拒绝（LIVE_HOLDER / FOREIGN_HOST / UNVERIFIABLE，dry-run 同样为 2） |
| 3 | `--release` 被拒（pid 或 token 不是自己的） |

## 3. 退出后审计（`tools/simulation/kit_descendant_audit.py`）

输入一个 run 目录；扫描其中 `*.json/.txt/.csv/.log`（含一层子目录），只从**已知 PID 键**取值：
`pid/pids/child_pid(s)/grandchild_pid(s)/descendant(s)_pid(s)/descendants_before_kill/descendants_after_kill/killed_pids/surviving_pids/survivor_pids/recorded_pids`。
返回码、耗时、大小等数字**不会**被当作 PID。

对每个 PID：`alive`、`cmdline`、`marker_found`、分类、**以及实际使用的探针命令原文**
（`tasklist /FI "PID eq N" /FO CSV /NH` 或 `os.kill(N, 0)`，命令行探针 `wmic process where processid=N get commandline /value` 或 `/proc/N/cmdline`）。

| 分类 | 含义 |
|---|---|
| `DEAD` | 记录过的 PID 已不存在 |
| `SURVIVOR_OURS` | 活着且命令行含 marker：**真幸存者** |
| `SURVIVOR_RECYCLED_OR_UNRELATED` | 活着但命令行无 marker：可能是 PID 回收，仍按幸存者上报（fail-closed） |
| `SURVIVOR_UNVERIFIABLE` | 活着但命令行读不到：按幸存者上报 |

| 退出码 | 含义 |
|---|---|
| 0 | `CLEAN`：没有任何记录过的 PID 存活 |
| 2 | `SURVIVORS`：至少一个存活（**非零即告警**） |
| 3 | `NO_PID_RECORDS`：目录里没有任何 PID 证据 = UNKNOWN，同样非零 |
| 1 | 用法错误 / run 目录不存在 |

`"kills_performed": 0` 写在报告里；代码中唯一涉及进程的调用是只读存活探测。测试显式断言：
审计跑完后那个活着的探针进程**仍然活着**（`survivor_alive_after_audit: true`）。

## 4. 真实命令与真实退出码（本机 Windows，Python 3.11.9 / pytest 9.0.2）

```
python -m pytest tests/simulation/test_gpu_lock.py tests/simulation/test_kit_descendant_audit.py -q
=> 18 passed, 2 skipped in 38.02s          exit 0

python -m pytest tests/simulation/test_p2_gate_runner.py -q
=> 92 passed in 83.64s (0:01:23)           exit 0
```

2 个 skip 的原因写在测试里（不是静默跳过）：
`test_signal_releases_the_lock`（POSIX 信号投递）与 `test_posix_permissions_are_user_only`（POSIX 0600 权限位）在 Windows 上无意义。

覆盖的用例（全部用**真实子进程**，不是进程内二次调用）：

1. 锁文件字段往返 + `--status` 只读；2. 两进程竞争（见 §5）；3. 死 PID 接管；
4. 拒绝抢占带 marker 的活持有者；5. PID 回收（marker 不匹配）接管且**不杀**；6. 跨主机拒绝；
7. 心跳陈旧接管并写入 history；8. dry-run 不建文件；9. 异常路径释放；10. `os._exit(9)` 模拟崩溃持有者后恢复；
11. 非本人 pid 的 release 被拒（3）；12. POSIX 权限位。
审计侧：7 个用例，含「真实 spawn-and-exit 的 PID 被判定 DEAD」与「活进程被判 SURVIVORS 且事后仍活」。

## 5. 两进程竞争证据（真实，`outputs/simulation/a04_evidence/gpu_lock_contention.json`）

主机 `abc123456`（Windows），全部为真实 OS 进程：

| 步骤 | 真实结果 |
|---|---|
| 1 持有者 A 获取锁 | `--run-id A04-HOLDER`，pid **65808**，`O_EXCL` 建锁成功，token `eaa53061…`，`heartbeat_at=2026-10-21T18:28:27.922044+00:00` |
| 2 竞争者 B 尝试获取 | 真实第二进程，**退出码 2**，`action=REFUSE / reason=LIVE_HOLDER`；证据 `checks.alive=true`、`marker_found=true`、`heartbeat_age_s=0.601576`、存活探针 `tasklist /FI "PID eq 65808" /FO CSV /NH`、命令行探针 `wmic process where processid=65808 get commandline /value`；锁文件**逐字节未变** |
| 3 持有者 A 正常退出 | 退出码 0；`atexit` 释放，锁文件**消失** |
| 4 dry-run | 退出码 0，`action=ACQUIRE / dry_run=true`，锁文件**未创建** |
| 5 崩溃持有者 | `os._exit(9)`，退出码 **9**，锁文件留存（pid **66476**，run `A04-CRASH`） |
| 6 接管 | `--run-id A04-RECOVER` 在**磁盘上**拿到锁（pid 63000），`history[0].reason=HOLDER_DEAD`、`history[0].prior.pid=66476`、`history[0].checks.alive=false`、`history[0].note=takeover never signals the previous holder` |

## 6. 审计三态证据（`outputs/simulation/a04_evidence/audit_*.json`）

| 场景 | 退出码 | verdict | 备注 |
|---|---|---|---|
| 刚被回收的死 PID（64912） | **0** | `CLEAN` | 分类 `DEAD`，探针原文见报告 |
| 活着的 marker 进程（67120） | **2** | `SURVIVORS` | 分类 `SURVIVOR_OURS`，**审计跑完后进程仍活着** |
| 无任何 PID 证据的 run 目录 | **3** | `NO_PID_RECORDS` | fail-closed |
| 崩溃时记录的 PID 66476（已被系统复用） | **2** | `SURVIVORS` | 分类 `SURVIVOR_UNVERIFIABLE`：号码被复用后命令行读不到，仍按幸存者上报 —— 见 `audit_recycled_pid.json` |

最后一行是本切片**观察到**的真实现象：Windows 上崩溃进程的 PID 在几秒内被复用。这正是「记录 PID 必须配合
marker / 启动时间校验」的直接证据，也是把 `SURVIVOR_*` 一律按 fail-closed 处理的原因。

## 7. 共享 GPU 主机的只读证据（`outputs/simulation/a04_evidence/gpu_host_readonly_probe.txt`）

`ssh -o BatchMode=yes -o ConnectTimeout=8 dgut@172.31.68.251 'hostname; pgrep -af isaac | head -5; nvidia-smi --query-gpu=index,name,memory.used --format=csv,noheader'`
退出码 **0**，真实输出：主机 `jxxy`；**一个**活着的 Kit 实例（kit pid 3111650，父 `runheadless.sh` pid 3111578）；另有别人的 P4 run（pids 3187145/3187160）；`0, NVIDIA RTX A6000, 18729 MiB`。

含义：此刻若在本机对这张 GPU 取锁，决策是 `REFUSE / LIVE_HOLDER`（记录的持有者 pid 活着、命令行为 `isaac`…`、同主机），
**拒绝而不是杀掉**——这正是合同要求的行为。本切片没有启动任何 Isaac，没有 kill 任何进程，没有在主机上写任何文件。

## 8. UNKNOWN（明确不声称的部分）

1. **Kit service-host 孙进程仍未被覆盖**：如果 Kit 把工作进程交给 Windows service host / `pythonw` 之类的宿主，且该 PID **从未**被 runner 记录进 run 目录，审计看不到它；
   锁也不会去杀它。这是原 PR #23 的 UNKNOWN 的**缩小**而非消除——能把「记录过但逃逸的 PID」查出来。
2. **没有做过真实双 Isaac 并发测试**：所有竞争证据是真实的两 OS 进程 + 真实锁文件，但双方都是本工具，不是两个 Isaac Sim 实例。
   「Isaac 在 GPU 上确实互斥到什么程度」仍然未知，本切片只保证**进程级排他**。
3. **Windows 上 `pid_start_time` 为 `null`**：启动时间校验（第 7 条判据）只在 Linux/macOS 生效；Windows 上退化到 marker 校验。
   Windows 的 PID 复用已被 §6 观测到，因此在 Windows 上「marker 不可读」时只能 fail-closed 拒绝。
4. **心跳依赖墙钟**：不同主机间时钟偏移会影响 stale 判定；因此跨主机锁一律 `FOREIGN_HOST` 拒绝，不做心跳接管。
5. **锁是协商式的**：不经过 `gpu_lock.py` 的进程（例如别人手工 `docker run` 起来的 Kit）不受约束；
   §7 的只读探测只能**看到**这类进程，不能阻止它们。
6. **命令行探针是外部命令**：Windows 上 `wmic` 在新版本可能被移除，回退到 PowerShell CIM（更慢，单次约 1 s 量级）；
   如果两者都不可用，决策退化为 `LIVE_HOLDER_UNVERIFIABLE`（拒绝，安全但可能误拒）。
7. **未与 systemd / 任务计划 / 队列调度集成**：锁只提供 CLI 与 Python API；谁来先拿锁仍由上层调用方决定。

## 9. 复现

```bash
cd badmin_wt_a04
python -m pytest tests/simulation/test_gpu_lock.py tests/simulation/test_kit_descendant_audit.py -q   # 18 passed, 2 skipped
python -m pytest tests/simulation/test_p2_gate_runner.py -q                                          # 92 passed

# 手工竞争：两个终端
python tools/simulation/gpu_lock.py --lock-path /tmp/isaac_gpu0.lock --run-id A --hold-seconds 30
python tools/simulation/gpu_lock.py --lock-path /tmp/isaac_gpu0.lock --run-id B   # exit 2, LIVE_HOLDER

# 事后审计
python tools/simulation/kit_descendant_audit.py --run-dir outputs/.../run_001 --json-out audit.json
```
