# P8-05 + P8-06：训练门禁（预算/预注册/checkpoint/停止条件）+ 算法依赖门禁（mock tests）

> **本轮没有运行 PPO，没有训练循环，没有 GPU，没有 torch，没有向实体机器人发送任何动作。**
> 本轮交付的是**可执行的门禁**（一个会 raise 的对象，不是一份文档），以及**证明它当前拒绝 PPO 的测试**。
> **门禁当前状态：PPO = NOT AUTHORIZED（拒绝）。** 这是**预期结果**，不是缺陷；
> 见第 5 节 `RequirementMissingError` 的真实输出与退出码 1。

前置（未重做，直接复用）：分支 `p8/seeds-baselines` @ `b74994a`（PR #42：种子账本 / curriculum / 基线 / 评测）
与 `p8/rl-interface-schema` @ `c073d2c`（PR #30：观测/动作/reward schema 与本文件的前身 `P8_TRAINING_GATE.md`）。
本文件是**新的独立切片**，不改动上述交付的任何结论。

---

## 1. 本轮交付物

| 文件 | 性质 |
|---|---|
| `src/badminton_brain/rl/training_gate.py` | P8-05 **强制对象**：预注册记录（sha256 固化）+ `TrainingGate.check()` + 预算会计 + checkpoint 契约 + 停止条件 |
| `src/badminton_brain/rl/dependency_gate.py` | P8-06 算法依赖门禁：P5/P6/P7 需要项 + 当前状态 + `require_satisfied()` |
| `tests/badminton_brain/rl/test_rl_training_gate.py` | **73 passed**：全部拒绝路径 / 假时钟预算耗尽 / checkpoint 原子性与哈希 / resume 拒绝 / 授权旗标不能绕过依赖门禁 |
| `tests/badminton_brain/rl/test_rl_dependency_gate.py` | **11 passed**：**今日拒绝 PPO**、矛盾状态拒绝、有证据时可通过（门禁不是硬编码拒绝） |
| 本文件 | 中文合同 + 依赖表 + 证据指针 + 真实命令与退出码 |

`training_gate.py` 与本文件都不含 optimiser / rollout / `env.step()` / `torch` / GPU 调用；
`test_rl_no_training_guard.py` 的 AST 扫描与独立解释器检查仍然覆盖新增模块。

---

## 2. P8-05 契约：七条拒绝路径（每条一个独立异常类）

一次训练**要开始**，必须**同时**满足下列 7 条；任一不满足即抛异常（**没有 warnings-only 路径**）：

| # | 条件 | 拒绝异常 |
|---|---|---|
| 1 | 预注册记录**存在** | `PreRegistrationMissingError` |
| 2 | 记录**完整**（算法/超参/种子哈希/回合预算/墙钟预算/GPU 预算/成功指标+验收阈值/停止条件/评测协议哈希） | `PreRegistrationIncompleteError` |
| 3 | 记录的 **sha256 固化值**与内容一致（未被改动） | `PreRegistrationIntegrityError` |
| 4 | 授权旗标 `ppo_training_authorized` **字面为 True** 且有 `granted_by` | `AuthorizationMissingError` |
| 5 | 上游 P5/P6/P7 在**调用方提供的状态表**中为 `DONE`/`PASS` | `UpstreamGateNotSatisfiedError`（带 `.unsatisfied` 列表） |
| 6 | 本次请求的预算**不超过**已注册预算（回合 / 墙钟 / GPU / 步数） | `BudgetExceedsRegisteredError`（带 `.dimension`） |
| 7 | 种子账本哈希**等于**已注册哈希 | `SeedLedgerMismatchError` |

补充的严格性：

- `True` 必须**是** `True`：`"true"`、`1`、缺省、空 `granted_by` 全部拒绝（"缺省不是许可"）。
- 预注册**必须先落盘成 JSON 再读回**（`TrainingGate.from_file`）才能通过第 3 条：
  内存里手搓的对象没有 pin，会被 `PreRegistrationIntegrityError` 拒绝。
- 预算字段不做"善意截断"：`episode_budget=1.5` 会被 `validate()` 拒绝，而不是被悄悄改成 1。

### 2.1 预算会计（假时钟，从不 sleep）

`BudgetAccountant(registration, clock=...)`，时钟**注入**（默认 `time.monotonic`）。
`admit_step(episodes=..., gpu_s=..., step_wall_s=...)`：

- 在 **步骤开始之前**检查墙钟/GPU/回合/步数，并**计入本步自身成本**（前瞻拒绝）；
- 拒绝时**闩锁** `BUDGET_EXHAUSTED`，之后即使把时钟拨回 0 也继续拒绝（fail-closed，`dimension="latched"`）；
- `snapshot()` 给出已用/剩余（`wall_remaining_s`、`gpu_remaining_s`、`episodes_remaining`、`steps_remaining`）。
- 测试用 `FakeClock` 手工拨表，**整个测试套件不 sleep 1 秒**。

### 2.2 checkpoint 契约

- **原子写**：内容先写同目录 `<name>.tmp-<uuid>` → `os.replace`；任一步失败删临时文件并原样保留旧 checkpoint。
  测试用 monkeypatch 让 `os.replace` 抛错，断言旧文件字节不变、目录里没有残留 tmp。
- **哈希**：`checkpoint_hash` 覆盖（metadata + payload 字节 + 格式串）；`payload_sha256` 单独再核一遍。
  改一个 metadata 字段或换掉 payload 的 base64 都 → `CheckpointIntegrityError`。
- **resume 拒绝跨实验**：`resume_checkpoint(..., registration=..., seed_ledger_hash=..., config_hash=...)`
  比较 checkpoint 里的 `registration_hash` / `seed_ledger_hash` / `config_hash`，任一不同 →
  `CheckpointConfigMismatchError`（`.mismatches` 指明是哪一项）。
- 另外拒绝 resume 一个 `stop_reason ∈ {SAFETY, DIVERGED}` 的 checkpoint（需要**新的授权决定**，不是"继续"）。
- **停止词汇（封闭）**：`BUDGET_EXHAUSTED / PLATEAU / DIVERGED / SAFETY / OPERATOR / CRASH`；
  其它字符串 → `StopReasonError`。进程消失记 `CRASH`（可 resume），**不得**记成成功。
- `StopCriteria.from_registration(...).evaluate(history, steps_used=...)` 把预注册的
  `max_steps / patience_evaluations / plateau_window / min_delta` 变成判定函数（NaN/Inf → `DIVERGED`）。

### 2.3 授权旗标不能绕过依赖门禁

`authorize_training_run(..., dependency_gate=...)` 是**唯一**的授权出口，`dependency_gate` 是**必填关键字参数**
（漏传 → `TypeError`）。它先跑 `TrainingGate.check()`，再无条件调用 `dependency_gate.require_satisfied()`。
因此：**即使**授权旗标为 `True`、**即使**调用方声称 P5/P6/P7 全部 `DONE/PASS`，
只要证据门禁不通过，授权依然失败（测试 `test_the_authorization_flag_alone_does_not_bypass_the_real_dependency_gate`）。

---

## 3. P8-06 依赖门禁：PPO 需要什么，现在有没有

规则（保守，不臆造）：

- 证据文件**不在本分支** → `MISSING_EVIDENCE`（不满足）；
- 文件在、但**没有记录过验收**（真实命令 + 真实退出码）→ `UNKNOWN`（不满足）；
- 文件在**且**表里已记录验收 → `SATISFIED`（唯一放过）；
- 表里写 `SATISFIED` 但文件不在本分支 → `EvidenceContradictionError`（**撒谎会被拒绝**）。

### 3.1 当前依赖表（2026-10-22，本分支 `p8/training-gate`）

| 需求 | 上游 | PPO 需要它的原因 | 证据指针（仅在未合并分支上） | 当前状态 |
|---|---|---|---|---|
| P5-04 | P5 | 关节名→DOF 索引一致性 | `simulation/badminton_scene/dof_index.py` @ `p5/controller-hold` 2afd6f2 | **MISSING_EVIDENCE** |
| P5-05 | P5 | 指令限位 + 速度/加速度/力矩保护 | `management/morph_one_rs/P5_CONTROLLER_AND_HOLD.md` @ 2afd6f2 | **MISSING_EVIDENCE** |
| P5-06 | P5 | reset 负例 fail-closed | `management/morph_one_rs/P5B_RESET_POSE.md` @ 2afd6f2 | **MISSING_EVIDENCE** |
| P5-07 | P5 | 多环境实例隔离 + 随机 reset | `management/morph_one_rs/P5_MULTIENV_OFFLINE.md` @ `p5/multienv-offline` e084efd | **MISSING_EVIDENCE** |
| P5-09 | P5 | 可复用控制器 API（state/targets/reset/hold/IK） | `simulation/badminton_scene/controller.py` @ 2afd6f2 | **MISSING_EVIDENCE** |
| P6-02 | P6 | 球拍→法兰→臂→底盘 frame tree | `outputs/simulation/rebot_b601_rs/morph_one/p6_frames/frame_tree.json` @ `p6/racket-camera-frames` d7672d5 | **MISSING_EVIDENCE** |
| P6-05 | P6 | 多坐标系/单位/手性一致性 | `management/morph_one_rs/P6_RACKET_CAMERA_FRAMES.md` @ d7672d5 | **MISSING_EVIDENCE** |
| P6-09 | P6 | 整机集成 manifest（队列里它是放行 P7 的条件） | **未找到任何分支上的交付物** | **MISSING_EVIDENCE（UNKNOWN 级）** |
| P7-09 | P7 | 可达工作空间 + 关节限位 + IK | `management/morph_one_rs/P7_REACHABILITY_AND_FSM.md` @ `p7/reachability-fsm` adaee81 | **MISSING_EVIDENCE** |
| P7-10 | P7 | fail-closed 安全 FSM（含 HOLD） | 同上 @ adaee81 | **MISSING_EVIDENCE** |
| P7-12 | P7 | 低速击球复现统计（非 RL 基线） | `outputs/badminton_brain/p7_closed_loop/wired_episodes.json` @ `p7/closed-loop-wiring` a3d6de0 | **MISSING_EVIDENCE** |

**11/11 不满足 → PPO 拒绝。** 注意：`P5-* / P6-* / P7-*` 的交付物**只存在于未合并分支**，
没有任何一条以"真实命令 + 退出码"的验收形式进入本分支；表里没有任何一条是"猜"出来的。

---

## 4. 真实命令与真实退出码

工作目录 `E:\具身智能\badmin_wt_p8c`（worktree，分支 `p8/training-gate`，基于 `p8/seeds-baselines` @ b74994a）。

```
$ python --version
Python 3.11.9                                   EXITCODE=0
$ python -m pytest --version
pytest 9.0.2                                    EXITCODE=0
```

（1）原有 8 个测试文件（回归：102 passed 必须保持绿）：

```
$ python -m pytest tests/badminton_brain/rl/test_rl_action_schema.py \
    tests/badminton_brain/rl/test_rl_baselines.py tests/badminton_brain/rl/test_rl_curriculum.py \
    tests/badminton_brain/rl/test_rl_eval_protocol.py tests/badminton_brain/rl/test_rl_no_training_guard.py \
    tests/badminton_brain/rl/test_rl_observation_schema.py tests/badminton_brain/rl/test_rl_reward_terms.py \
    tests/badminton_brain/rl/test_rl_seed_ledger.py -q
102 passed, 6 warnings in 14.21s                EXITCODE=0
```

（2）整个 `tests/badminton_brain/rl`（含本轮新增 84 个测试）：

```
$ python -m pytest tests/badminton_brain/rl -q
186 passed, 6 warnings in 13.95s                EXITCODE=0
```

（3）依赖门禁**当前拒绝 PPO**（这是本轮的核心证据）：

```
$ python -c "import sys; sys.path.insert(0,'src'); \
    from badminton_brain.rl.dependency_gate import default_gate; default_gate().require_satisfied()"
badminton_brain.rl.dependency_gate.RequirementMissingError: PPO training is NOT AUTHORIZED:
11 of 11 upstream requirement(s) are unsatisfied (PPO_TRAINING_STATUS=NOT_AUTHORIZED; an
authorization flag cannot create evidence):
  - P5-04 [MISSING_EVIDENCE] ... evidence artifact absent from this branch ...
  ...
  - P7-12 [MISSING_EVIDENCE] low-speed hitting repeatability statistics ... 
                                               EXITCODE=1
```

（4）P5/P6/P7 分支确实**没有**合并进本分支（逐条）：

```
$ git merge-base --is-ancestor p5/controller-hold HEAD      EXITCODE=1
$ git merge-base --is-ancestor p5/multienv-offline HEAD     EXITCODE=1
$ git merge-base --is-ancestor p6/racket-camera-frames HEAD EXITCODE=1
$ git merge-base --is-ancestor p6/dynamic-fk HEAD           EXITCODE=1
$ git merge-base --is-ancestor p7/reachability-fsm HEAD     EXITCODE=1
$ git merge-base --is-ancestor p7/closed-loop-wiring HEAD   EXITCODE=1
```

（5）证据指针的可复核方式（分支 tip SHA 由 git 自己给出，本文件未编造）：

```
$ git log --oneline -1 p5/controller-hold       # 2afd6f2 feat(arm): DOF index map, safety envelope + HOLD, ...
$ git log --oneline -1 p5/multienv-offline      # e084efd feat(sim): OFFLINE multi-env B601-RS harness ...
$ git log --oneline -1 p6/racket-camera-frames  # d7672d5 docs(p6): report says five error classes ...
$ git log --oneline -1 p7/reachability-fsm      # adaee81 feat(p7): reachability + IK interface and fail-closed FSM
$ git log --oneline -1 p7/closed-loop-wiring    # a3d6de0 feat(p7): wire the closed loop to the verified P6 FK ...
```

---

## 5. 本轮**没有**做的事（明确）

- **没有**运行 PPO / 任何训练 / 任何 rollout；**没有** optimiser、`*.pt`、checkpoint 产物（测试里的 checkpoint 写在 pytest 临时目录）。
- **没有**引入 torch / gym / SB3 / Isaac；**没有**用 GPU。
- **没有**调用 `git merge`，**没有**把 P5/P6/P7 分支合并进本分支，**没有**替它们"预先通过"门禁。
- **没有**填 P8-05 预算表的真实数字：本轮的预注册样例是**测试夹具**（`episode_budget=100` 等），
  **不是**已批准的预算；真实数字仍须"用户再次明确授权"时由人填。
- **没有**声称任何成功率 / 训练指标。
- `docs/observation_action.md`、`docs/reward_design.md`、`docs/algorithm_design.md` 未改。

---

## 6. UNKNOWN / 待澄清

1. **P6-09（整机集成 manifest）在已检查的分支上都找不到交付物**：可能未开工、可能在其它分支。
   没有证据 → 门禁按 `MISSING_EVIDENCE` 处理，**不**猜"大概做了"。
2. **P5/P6/P7 是否"实质完成"未知**：本文件只陈述"其交付物在未合并分支上、且没有验收记录进入本分支"。
   合并/验收与否属于 P5/P6/P7 负责人的判断，本轮不做任何背书。
3. **预算数字（GPU·小时、墙钟上限、episode 预算、checkpoint 频率、patience K）全部 TBD**：
   门禁只能执行**已注册**的数字，注册本身需要用户授权。
4. **一次"中断→resume→指标连续"的实测**：本轮只证明**契约**（哈希/配置/账本一致才允许 resume），
   没有真实训练产物，因此**不能**声称指标连续。
5. **P8-07（真机急停/Watchdog）** 未在范围内；本轮无任何真机通路。

---

## 7. 变更边界（本轮只碰这些）

- 新增：`src/badminton_brain/rl/{training_gate,dependency_gate}.py`
- 新增：`tests/badminton_brain/rl/{test_rl_training_gate,test_rl_dependency_gate}.py`
- 新增：本文件
- **修改**：`tests/badminton_brain/rl/test_rl_no_training_guard.py` —— **一处必要修正**：
  导入子模块会让 `badminton_brain.rl.training_gate` 成为包属性，被旧的"命名空间不得出现 train*"扫描误判为 trainer。
  修正为**跳过模块对象**（`types.ModuleType`）后扫描 API 名，守卫意图不变（torch/gym AST 扫描、
  `.backward()/.step()/.zero_grad()` 扫描、独立解释器导入检查**全部保留**），测试数不变（102 → 102）。
- **未触碰**：`src/perception`、`src/trajectory`、`scripts/simulation/*`、`simulation/`、`PHASE_P2_*`/`PHASE_P3_*`/`PHASE_P4_*` 文档、队列状态文件（`EXECUTION_QUEUE_*`）、`p8/seeds-baselines` 上的任何既有实现。

---

## 8. 修订记录

| 日期 | 版本 | 变更 | 作者 |
|---|---|---|---|
| 2026-10-22 | v0.1 | P8-05 训练门禁强制对象（预注册/预算/checkpoint/停止词汇）+ P8-06 依赖门禁；84 个新测试；门禁**当前拒绝 PPO** | P8 子任务（CPU，无训练） |
