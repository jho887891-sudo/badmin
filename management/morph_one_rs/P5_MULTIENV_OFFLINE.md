# P5-07 / P5-08：多环境隔离（OFFLINE）证据报告

分支：`p5/multienv-offline`（从 `p5/controller-hold` @ 2afd6f2 起）。本提交的完整 SHA 记录在 Issue #13
的 DONE 评论里（本文件无法写入自身提交的 SHA）。

> **一句话结论**：在**离线 mock** 上，2 路与 4 路 B601-RS 控制器实例的**状态隔离、逐环境确定性随机、
> 写入前安全校验、7 类故障的逐环境封闭、重启/恢复**全部可复现地成立，每一项都有"位相同"的 SHA256
> 状态摘要作为证据。**本报告中的任何吞吐/内存数字都只描述这个离线 mock，与 Isaac Sim、GPU、
> 真实 16/64/128 环境能力无关，见第 6 节。**

---

## 1. 交付物

| 文件 | 作用 |
|---|---|
| `simulation/badminton_scene/multienv.py` | 离线多环境 harness（`MultiEnvHarness`）+ 被注入的 sim 适配器（`OfflineMockSim`，duck-typed：`read_state` / `write_targets` / `reset`）+ 吞吐测量 |
| `tests/simulation/test_p5_multienv.py` | 23 个测试：隔离、复位不串扰、种子可复现、故障封闭、重启恢复、USD 限位守卫、测量标签 |
| `outputs/simulation/rebot_b601_rs/morph_one/p5_multienv/offline_mock_env{1,2,4,8,16}.json` | 第 5 节的原始测量输出（可校验、可重跑） |

未改动：`scripts/simulation/*`、`src/*`、队列状态文件、`PHASE_P2_*/PHASE_P3_*/PHASE_P4_*`、真实机械臂资产、
`172.31.68.251` 上的任何内容。未使用 Isaac Sim / GPU / torch / pxr。

### 复用的既有切片（不重做）

* `dof_index.py`（P5-04：名字→DOF 索引，运行期名字列表驱动）
* `safety.py`（P5-05：包络 + HOLD 锁存）
* `controller.py`（P5-09：`B601RSController` 的 read_state / set_joint_targets / reset / hold）
* `rest_pose.py`（P5B：模型化静止位姿与 B601-RS 限位表）

多环境层只做一件事：把 **N 个独立 controller 实例**各自绑到**独立的适配器行**上，并在每次写入前
强制走 P5-05 包络。

---

## 2. 离线证明了什么（逐条，可复跑）

1. **环境隔离**：A 的 HOLD / 故障 / 复位都不改变 B 的状态摘要（SHA256，位相同，不是"接近"）。
2. **逐环境确定性随机**：每环境种子 = `sha256("<seed>|<env_id>")`；同一 seed 序列 ⇒ 同一状态序列；
   不同 seed ⇒ 不同状态；两个 env 即使拿到**同一个 base seed** 也不同（派生里含 env_id）。
3. **写入前校验**：`step()` 只有一个顺序——`SafetyGate.accept()` 先，`adapter.write_targets()` 后。
   被拒绝的目标不触碰适配器，并以 P5-05 的类型化 code 记录（`non_finite` / `position_limit` /
   `unknown_joint` / `hold_active` / `velocity_limit` / `acceleration_limit`）。
4. **故障注入 7 类，逐环境封闭 + 类型化原因**（见第 4 节矩阵）。
5. **重启/恢复位相同**：`restart_and_resume()` 从 checkpoint（位姿、已接受设定值、HOLD 锁存、RNG 状态、
   计数器）重建该环境 controller，恢复前后的状态摘要**完全相等**；其他环境摘要不变。
6. **RS 静止位姿在 USD 限位内**：测试独立读取 `arm_joint_facts.json`、在测试文件内自行把 deg→rad 换算，
   断言每个环境使用的静止位姿严格落在限位内；反向控制断言 PiPER 的 `joint3 = -1.5` 在 RS 上被**拒绝**
   （P5 F1 的原始故障，作为活体对照保留）。
7. **不存在 NaN/越界写入适配器**：`sim.nonfinite_writes == []`，且随机运行中每一次落入适配器的写入
   都在包络内（遍历 `sim.writes(env)` 断言）。

---

## 3. 独立性证据（真实输出）

状态摘要定义（`MultiEnvHarness.state_digest`）：对 `env_id / seed / attempts / writes / holding / status /
joint_names / 适配器关节位置 / 已接受设定值 / RNG 状态` 做 SHA256，浮点一律用 `float.hex()` 编码，
所以"位相同"是字面意义。

```
$ python _scratch_evidence.py            # 与 test_p5_multienv.py 中等价的证据代码
B_digest_before_hold_A = c5eb450f7bb90ff6762d32f99b664ff3630ceb986c77dd81c809162c073927a5
B_digest_after_hold_A  = c5eb450f7bb90ff6762d32f99b664ff3630ceb986c77dd81c809162c073927a5
B_bit_identical        = True
A_codes_when_frozen    = ['hold_active', 'hold_active', 'hold_active']
A_digest_before        = 6b8bd10267f55328be78ea29db3dd1e4b10ce96d498ba2b72ba54288ab9d892d
A_digest_after         = adcf1e1d3121df09ab342d51fd2a7c368b47403e8df59feb637d43c2cc4d45aa

nan_target         code=non_finite         B_bit_identical=True adapter_writes_delta=0 status=ok
out_of_limit       code=position_limit     B_bit_identical=True adapter_writes_delta=0 status=ok
unknown_joint      code=unknown_joint      B_bit_identical=True adapter_writes_delta=0 status=ok
mid_run_hold       code=hold_active        B_bit_identical=True adapter_writes_delta=0 status=ok
adapter_exception  code=adapter_exception  B_bit_identical=True adapter_writes_delta=0 status=ok
timeout            code=timeout            B_bit_identical=True adapter_writes_delta=1 status=quarantined

N4_digests_distinct = True
N4_env2_unchanged = True | ledgers: {0: 1, 1: 1, 2: 0, 3: 1}
restart_digest_equal = True | hold_survived = True | B_unchanged = True
```

读法：

* A 被 HOLD 冻结时，B 的摘要**一个字节都没变**，而 A 自己的摘要变了（锁存 + 尝试计数）——"隔离"与
  "确实发生了事情"两侧同时成立；A 的 3 条拒绝全部以 `hold_active` 记录。
* `adapter_writes_delta=0` = 该故障**没有触碰适配器**（包络先拒绝）；适配器异常/超时除外：适配器
  异常在改状态前抛出（delta=0），超时的写入**最终还是会落地**（delta=1）——见第 6 节。
* N=4：env 2 未被任何注入影响（摘要不变，事件账本为空），env 0/1/3 各自只有自己的账本。

对应测试：`test_hold_in_a_leaves_b_bit_identical`、`test_reset_in_a_does_not_touch_b`、
`test_no_state_leak_across_resets_of_every_env`、`test_same_seeds_reproduce_the_same_states`、
`test_different_seed_changes_only_that_env`、`test_rng_streams_are_per_env`、
`test_restart_and_resume_is_bit_identical_and_isolated`、
`test_n4_batch_keeps_four_distinct_states_under_mixed_faults`。

---

## 4. 故障注入矩阵

| 注入种类 | 记录类型 | 类型化原因 | 适配器是否被写 | 环境状态 | 其他环境 |
|---|---|---|---|---|---|
| `nan_target` | `RejectedWrite` | `non_finite` | 否 | ok | 摘要不变 |
| `out_of_limit` | `RejectedWrite` | `position_limit` | 否 | ok | 摘要不变 |
| `unknown_joint` | `RejectedWrite` | `unknown_joint` | 否 | ok | 摘要不变 |
| `mid_run_hold` | `RejectedWrite` | `hold_active` | 否 | 保持 HOLD | 摘要不变 |
| `adapter_exception` | `FaultEvent` | `adapter_exception` | 否（改状态前抛出） | `write_diverged=True`，`resume()` 复位基线 | 摘要不变 |
| `timeout` | `FaultEvent` | `timeout` | **会**（迟到写入，`late_write_possible=True`） | `quarantined`，`resume()` 后恢复 | 摘要不变 |
| `restart_resume` | `FaultEvent` | `restart_resume` | 一次 `reset`（重建位姿） | 与重启前**位相同** | 摘要不变、计数器不变 |

`adapter_exception` 与 `timeout` 两条"不干净"的语义都写进事件 detail，不做静默重试：

* 适配器异常时 P5-05 已锁存被接受的设定值而写入失败 ⇒ **plant 落后于 gate**。事件记录
  `write_diverged=True`；`resume()` 用适配器**真实状态**重设速率基线，并返回
  `reconciled_divergence=True`。
* 超时的写入跑在工作线程里，**线程杀不掉**，所以被放弃的写入可能随后落地。环境被隔离（拒绝新指令），
  测试在断言前调用 `wait_for_pending()` 把工作线程 join 回来，因此"实际落地了什么"是可观测的，不是猜的。

---

## 5. OFFLINE-MOCK 吞吐 / 内存（**不是 Isaac 吞吐**）

> **OFFLINE-MOCK — not Isaac throughput.**
> 下表测的是这个 Python mock harness（P5-05 校验 + SHA256 摘要 + 每次写入一个守护线程），
> 没有 Isaac Sim、没有 PhysX、没有 GPU。**不得**据此声称支持真实 16/64/128 环境并发。

命令与真实退出码（PowerShell，工作目录 `E:\具身智能\badmin_wt_p5d`）：

```
$env:PYTHONPATH = "simulation"
$out = "outputs/simulation/rebot_b601_rs/morph_one/p5_multienv"
foreach ($n in 1,2,4,8,16) {
  python -m badminton_scene.multienv --envs $n --steps 200 --seed 7 --out "$out/offline_mock_env$n.json"
}
# env1 EXIT=0 / env2 EXIT=0 / env4 EXIT=0 / env8 EXIT=0 / env16 EXIT=0
```

| envs | env-steps | wall (s) | env-steps/s | peak Python alloc (MiB) | accepted |
|---|---|---|---|---|---|
| 1 | 200 | 0.457 | 437.5 | 0.138 | 200 |
| 2 | 400 | 0.979 | 408.4 | 0.252 | 400 |
| 4 | 800 | 1.780 | 449.3 | 0.480 | 800 |
| 8 | 1600 | 3.724 | 429.6 | 0.936 | 1600 |
| 16 | 3200 | 7.966 | 401.7 | 1.847 | 3200 |

测量口径：

* `env_steps_per_second` = 全部环境的 step 数 / 墙钟；单环境速率约 400–450 env-steps/s（≈2.3 ms/步）。
* `peak_python_bytes` 来自 `tracemalloc`，是**本进程 Python 分配峰值**；不是 GPU 显存，不是 PhysX 显存，
  也不是真实仿真器的内存模型。它随 env 数近似线性（≈0.11 MiB/env），只说明本 harness 自身没有明显的
  每步泄漏式增长，**不构成任何 Isaac 内存结论**。
* 每步约 2.3 ms 中可诊断的主要成本是"每次写入一个守护线程"（超时守护）与摘要哈希；真实仿真里这两项
  会被 PhysX 淹没。这是**已知性能上限**，不是"实时能力"。

---

## 6. UNKNOWN 清单（未证明、不得宣称）

1. **没有真实 Isaac 多环境运行**：从未在 Isaac Sim / Isaac Lab 上跑过 N>1 的 B601-RS 环境，
   因此"PhysX 多环境隔离"本身**未被验证**——本切片验证的是 harness 侧的隔离契约。
2. **没有 GPU 显存数字**：没有 GPU、没有查询，任何显存/带宽/并发规模结论都无从谈起。
3. **没有真实关节名顺序**：真实运行期 `articulation.joint_names` 的顺序仍是 UNKNOWN。本切片用
   `joint1..joint6, gripper_joint1, gripper_joint2` 作为运行期顺序（与 P5/P5B 表一致），真实顺序必须由
   运行时名字列表决定（P5-04），不能从本报告推断。
4. **没有真实动力学**：mock 的"物理"只有"被接受的设定值即关节位置"；速度/加速度/接触/摩擦全未建模。
   P5-05 的速度/加速度上限本身仍是 POLICY CAP（非实测）。
5. **超时的迟到写入**：线程无法杀死，`timeout` 故障下写入可能随后落地；真实 Isaac 的取消语义 UNKNOWN。
6. **重启不是完整重启**：只恢复本 harness 记录的 checkpoint；PhysX 求解器 warm-start / 速度 / 接触缓存
   **完全未记录**，真实 restart 的行为 UNKNOWN。
7. **没有多进程/多机**：所有 env 在**同一进程**内、共享 GIL；进程级隔离与并行伸缩 UNKNOWN。
8. **故障集合不穷尽**：只覆盖 7 类注入；`write_targets` 半途失败、适配器返回错误值、设备中断等未覆盖。

---

## 7. 测试与退出码（真实）

```
python -m pytest tests/simulation/test_p5_multienv.py -q
# 23 passed in 0.95s
# exit code 0

python -m pytest tests/simulation/arm_pose -q
# 119 passed, 5 skipped in 2.56s
# exit code 0

python -m pytest tests/simulation/test_p5_multienv.py tests/simulation/arm_pose -q
# 142 passed, 5 skipped in 2.93s
# exit code 0
```

（Python 3.11.9 / pytest 9.0.2 / Windows。5 个 skip 来自 `arm_pose` 既有切片，与本切片无关。）

---

## 8. 边界与后续

* 本切片**不**修改 P5-05 / P5-09 / P5B 的任何契约，只在它们之上加一层"多环境 + 故障注入 + 证据"。
* 走向真机/真 Isaac 的下一步：把 `OfflineMockSim` 换成 Isaac Lab 适配器（实现同样三个方法
  `read_state / write_targets / reset`），并用同一组测试与同一份状态摘要比对——**在那之前，本报告的
  所有结论只属于离线 harness**。
