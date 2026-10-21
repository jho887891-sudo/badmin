# P8 训练门禁与预算（P8-01 / P8-02 交付）

> **本文件是一份计划 + 测试证据，不是一个已训练的策略。**
> 本轮交付**没有**运行 PPO、没有训练循环、没有 GPU、没有 Isaac、没有向实体机器人发送任何动作。
> 所有 reward 权重与限值均为 **TEMP_PARAMETERIZED_PROXY / PLACEHOLDER**（未调参、未实测）。
> 队列原文：`management/morph_one_rs/EXECUTION_QUEUE_CONTINUOUS_P0_P8.md` 第 P8 节（分支
> `origin/chatgpt/p2-wc-continuous-queue-20261021`，本分支`main` 上尚无该目录，故本文件新建目录）。
> 授权口径：**PPO / 训练运行 / 真机动作 = NOT AUTHORIZED**；P8-06 需用户再次明确授权，且必须满足 P5/P6/P7 门禁。

---

## 1. 本文件性质（先读这一节）

| 问题 | 答案 |
|---|---|
| 这里有训练好的策略吗？ | **没有。** 没有任何 checkpoint、权重、`*.pt` 文件。 |
| 这里跑过 PPO 吗？ | **没有。** `src/badminton_brain/rl/` 内不存在 optimiser / trainer / rollout / `env.step()`，有静态+运行时测试守卫（见第 8 节）。 |
| 能通过这份代码动真机吗？ | **不能。** action 是仿真专用接口：`SimulatedCommand.to_real_robot()` 必抛 `RealRobotActionForbidden`；本包内无驱动、无串口、无 SDK、无急停。 |
| 那这份交付是什么？ | **接口/schema 设计 + 离线可计算性验证 + 门禁计划**（P8-01 + P8-02）。 |
| 改动了项目 API 吗？ | **没有。** `docs/observation_action.md` / `docs/reward_design.md` 原样未改；本 schema 标记为 `CANDIDATE_NOT_PROJECT_API`，等 P1/P8-06 再决定是否升为项目 API（项目 API 规则：改 API 必须同步该文档）。 |

---

## 2. READY（本轮已完成，可直接复用）

### 2.1 P8-01 观测 schema（`src/badminton_brain/rl/observation_schema.py`）

- 17 个声明式通道，扁维 **70**（含 `shuttle_velocity` history=3 的堆叠）。
- 每个通道显式声明：`name / shape / dtype / unit / bounds / source / max_age_s / history`。
- 覆盖：底盘位姿/速度、机械臂 q/dq、球拍位姿与速度、羽毛球估计 + 3×3 协方差、time-to-hit、可行性 flags、上一步动作。
- `build_observation(mapping) -> np.ndarray`（`(70,)` 或 `(N,70)`，dtype 恒为 float64）与 `validate(...)`（mapping 或已构建数组两条路径）。
- **硬失败**（各有独立异常类与测试）：缺键、多键（拼写错误）、形状错（含 history 缺失）、dtype 错（字符串/复数/bool 混入浮点/浮点冒充 flag）、NaN/Inf、越界、时间戳过期/未来/缺失、单位混淆（deg↔rad、mm/cm↔m、km/h）、批量维度不一致。
- 单位混淆拒绝理由带修复提示（如 "divide by 1000"）；`radians` 等别名被规范化接受，但 `deg` 绝不等于 `rad`。

### 2.2 P8-01 动作 schema（`src/badminton_brain/rl/action_schema.py`）—— **SIMULATION-ONLY**

- 必选 9 维：`base_linear_velocity (2,) m/s` + `base_angular_velocity (1,) rad/s` + `joint_position_delta (6,) rad`
  （与 `docs/observation_action.md` 的 vx/vy/wz + 6 关节占位一致）；可选球拍目标 7 维（位置+朝向，必须成对）。
- 限值全部是 `AssetStatus.TEMP_PARAMETERIZED_PROXY` 且带 `source`：0.60 m/s、1.00 rad/s、2.00 rad/s、±2.6 rad、workspace box。
- **两级约束**：绝对边界（越界直接拒绝，**不 clamp** —— 静默限幅会掩盖策略跑出包络）+ 步间速率限制（0.5 m/s²、1.0 rad/s²、2.0 rad/s²，按 `dt_s` 缩放；即使动作在绝对边界内，跳变过大同样拒绝）。
- `to_command(action, joint_position=..., previous_action=..., dt_s=...)` 必须传入当前关节角（不允许凭空捏造当前状态）。
- 强制仿真标记：`SimulatedCommand(simulation_only=True, target="SIMULATOR_ONLY")`；`simulation_only=False` / `target!="SIMULATOR_ONLY"` 构造即抛 `RealRobotActionForbidden`；`to_real_robot()` 必抛。

### 2.3 P8-02 reward 候选（`src/badminton_brain/rl/reward_terms.py`）

7 条候选项，全部是**记录帧的纯函数**（`Frame` 为 frozen dataclass，不可改）：

| 项 | 输入（记录） | 缺失时行为 |
|---|---|---|
| `distance_to_intercept` | shuttle_position, intercept_point | masked（None） |
| `contact_success` | contact_detected | masked（None） |
| `velocity_matching` | racket_velocity, shuttle_velocity | masked |
| `collision_penalty` | collision_impulse_ns | masked |
| `joint_limit_penalty` | joint_limit_excess_rad | masked |
| `instability_penalty` | base_tilt_rad | masked |
| `timeout_penalty` | t_s | masked |

关键语义（有专门测试）：

- **masked ≠ 0**：`contact_detected=False`（传感器答复"没碰到"，0.0）与 `contact_detected=None`（没记录，masked）必须可区分；测试刻意构造"两者加权总和相同（-1.5）"的场景，只有 mask 能区分，因此 mask 必须上报。
- `TermResult` 三态不存在：`value=None` 必须配 `masked=True`；`masked=True` 不得带数值；NaN/Inf 不允许产出（应改为 masked）。
- 记录帧本身损坏（形状不对、含 NaN）→ **抛 ValueError**，不是 mask：把损坏数据伪装成"缺测"会掩盖数据集问题。
- `offline_computability_report(frames)` 输出逐项 × 逐帧的可计算性矩阵（含每个 mask 的原因），是 P8-02 的离线证据形式。
- 权重（`RewardWeights`）标注 `PLACEHOLDER … NOT tuned`；`docs/reward_design.md` 未改。

### 2.4 测试（`tests/badminton_brain/rl/`）

`70 passed`（见第 8 节真实命令/退出码），包含：形状/dtype 往返、批量往返、17 个通道的越界/单位/时间戳拒绝路径、动作速率限制、reward 求和、masked-vs-zero、"无训练"守卫（AST 扫描 + 独立解释器导入 + 命名空间检查）。

**未做**：没有跑全仓测试，因此**不声称**全仓无回归。

---

## 3. FORBIDDEN（未经新授权一律禁止）

以下行为在**本轮以及可预见的后续轮次**都不得由 agent 自行启动，必须由用户给出新的、明确的书面授权：

1. **PPO / 任何 RL 训练运行**（含 CPU 小规模试跑、超参搜索、curriculum 自动推进）。
2. 引入 `torch` / `gymnasium` / `stable_baselines3` / `isaacgym` / `isaacsim` 等依赖到本仓库（`tests/badminton_brain/rl/test_rl_no_training_guard.py` 会失败）。
3. **GPU 训练** 与 **Isaac Sim/Isaac Lab 场景实例化**（P2/P3/P4 的 GPU 串行门禁仍按队列文件执行，不因 P8 而放宽）。
4. **对实体机器人发送任何动作**（含"只是测试一下串口"、急停联调、手动点动）。
5. 把 `src/badminton_brain/rl/` 的 candidate schema 当成项目 API 使用（不得在 `docs/observation_action.md` 之外"事实上定稿"）。
6. 声称已获得任何任务成功率 / 已训练策略的指标（本仓库当前**没有**任何训练产物，任何此类数字都是伪造）。

---

## 4. P8-06 的前置门禁（P5 / P6 / P7）

P8-06（算法设计 + 离线 mock tests；真正训练须再次授权）在开工前，必须逐条给出**证据 commit + 可复现命令 + 真实退出码**，缺一不可：

| 门禁 | 要求（以队列文件 P5/P6/P7 条目为准） | 证据形式 |
|---|---|---|
| P5 单臂控制 / 多环境 reset | B601-RS 单臂控制与 reset 在仿真中稳定；多环境一致性与状态同步有实测证据 | 指向具体 commit + 复现命令 + 退出码 |
| P6 球拍 / 相机 / 标定 / 整机 | 球拍末端与相机外参标定、整机装配可用；标定值带 `AssetStatus`，未实测项不得写成实测 | 标定文件 + 审计表 |
| P7 检测→跟踪→预测→低速击球 | 离线链路（检测/UKF/轨迹预测/可行性）有量化指标与冻结测试；预测误差口径固定 | 指标 JSON + 冻结测试 |
| P8-03 种子分离 | 第 5 节策略落地为代码 + manifest | seed manifest（含 sha256） |
| P8-04 非 RL 基线 | 有限状态/可达性/简单跟踪基线 + 同口径成功率评测协议 | 基线结果表 |
| P8-05 预算门禁 | 第 6 节表格填成**真实数字**（当前全部 TBD） | 本文件更新 commit |

> 说明：上述门禁条目名以队列文件原文为准；本文件不复制其数值，也不替它们"预先通过"。**P8-01/P8-02 完成不等于 P8-06 解锁。**

---

## 5. 种子分离策略（P8-03 计划，尚未实现）

原则：**训练 / 验证 / 测试的随机种子与场景实例三重分离，且不可泄漏**。

1. **种子集合固定为常量**：`TRAIN_SEEDS` / `VAL_SEEDS` / `TEST_SEEDS` 三段互不相交，写入 `outputs/**/seed_manifest.json`（含每段列表的 sha256）。
2. **场景实例由 (split, seed, task_id) 派生**：`task_seed = sha256(f"{split}|{seed}|{task_id}")[:8]` 转整数；**禁止**跨 split 复用同一 episode 的初始状态、发球参数或随机扰动。
3. **reward / 超参调优只允许看 TRAIN**；VAL 仅用于早停与模型选择；TEST 只在最终评测跑一次，跑完记录次数。
4. **不可泄漏记录**：每次评测产出 `{split, seed, task_id, episode_hash, return, contact_rate}` 行；同一 `episode_hash` 出现在两个 split 即判为泄漏，评测直接失败。
5. **报告口径**：任何"成功率"必须写清 split、seed 列表、episode 数、评测 commit；缺一项视为无效结论。
6. 本节的实现属于 P8-03，**本轮未写代码**，因此当前不存在任何可报告的评测数字。

---

## 6. 预算 / checkpoint / 回滚 / 停止条件（P8-05 占位，全部 TBD）

> 下表是**待填写模板**。所有 TBD 必须在 P8-06 授权时由用户与负责人共同填成真实数字，**agent 不得自行填入估计值**。

| 项 | 值 | 状态 |
|---|---|---|
| GPU 型号 / 数量 | TBD | 待授权 |
| 训练总 GPU·小时上限 | TBD | 待授权（超限即停止） |
| 单次运行墙钟上限 | TBD | 待授权 |
| 并发环境数上限 | TBD | 待授权（须先过 P5 多环境门禁） |
| 训练数据来源 | TBD | 仅允许仿真；真机数据需另立授权 |
| checkpoint 格式 / 字段 | TBD | 必须含：git SHA、seed manifest sha256、reward/limits 版本、环境数、步数、评测摘要 |
| checkpoint 保存频率 | TBD | 建议按"每 N 步 + 每次评测 + 异常前"，N 待定 |
| 恢复（resume）验证 | TBD | 必须有一次"中断→恢复→指标连续"的实测记录 |
| 停止条件 | TBD | 至少包括：超预算、评测退化、NaN/Inf 出现、安全项（碰撞/越界）触发率超阈值、人手中断 |
| 回滚方式 | TBD | 代码回滚 = `git revert`；产物回滚 = checkpoint 作废 + manifest 归档；回滚后必须重跑门禁测试 |
| 安全守卫 | 已定：仿真内仍走 SafetyShield 限幅 | 真机守卫 = **禁止**（见第 3 节） |
| 人工安全区域 / 急停 / Watchdog | TBD | 属 P8-07 文档范围，**仅文档**，不得实现真机通路 |

---

## 7. 停止条件与回滚（占位细化）

- **强制停止**：预算超限、出现 NaN/Inf reward、显存/内存异常增长、评测指标连续 K 次下降（K 待定）、任何安全项触发率越界、用户中断。
- **回滚触发**：门禁测试变红、checkpoint 无法复现、seed manifest 变更无记录。
- **回滚动作**：revert 相关 commit → 作废旧 checkpoint 与 manifest（归档而非删除）→ 重跑 `python -m pytest tests/badminton_brain/rl -q` 与相应门禁测试 → 在 `management/` 记录回滚原因与 SHA。
- 以上全部为**计划文本**，本轮没有任何 checkpoint 被创建或删除。

---

## 8. 证据（真实命令 + 真实退出码，2026-10-21）

工作目录 `E:\具身智能\badmin_wt_p8`（worktree，分支 `p8/rl-interface-schema`，基于 `origin/main`）。

```
$ python --version
Python 3.11.9
$ python -m pytest --version
pytest 9.0.2
```

首次运行（暴露真实缺陷，如实记录）：

```
$ python -m pytest tests/badminton_brain/rl -q
...
3 errors in 3.32s
EXITCODE=2        # reward_terms.py 多行签名语法错误（已修）
$ python -m pytest tests/badminton_brain/rl -q
5 failed, 65 passed in 4.34s
EXITCODE=1        # 批量拼接用 np.stack（宽度不同）、测试期望值写错等（已修）
```

修复后最终结果：

```
$ cd "E:\具身智能\badmin_wt_p8"
$ python -m pytest tests/badminton_brain/rl -q
......................................................................   [100%]
70 passed in 3.05s
EXITCODE=0
```

- 仅运行 `tests/badminton_brain/rl`，**没有**运行全仓测试，故本文件**不声称**全仓无回归。
- 无 GPU、无 Isaac、无网络依赖：测试只 import `numpy` 与 `src/badminton_brain`。
- "无训练"守卫（`test_rl_no_training_guard.py`）：AST 扫描 `src/badminton_brain/rl/*.py` 的 import 根（torch/gym/SB3/Isaac/…）、扫描 `.backward()/.step()/zero_grad()` 调用、检查公开命名空间无 Trainer/Optimizer/PPO、并用**独立解释器**导入本包后断言 `sys.modules` 中不含任何禁止模块。

---

## 9. 变更边界（本轮只碰这些）

- 新增：`src/badminton_brain/rl/{__init__,observation_schema,action_schema,reward_terms}.py`
- 新增：`tests/badminton_brain/rl/test_rl_{observation_schema,action_schema,reward_terms,no_training_guard}.py`
- 新增：本文件 `management/morph_one_rs/P8_TRAINING_GATE.md`
- **未触碰**：`scripts/simulation/*`、`simulation/`、`src/perception/`、`src/trajectory/`、队列状态文件、任何 `PHASE_P2_*` / `PHASE_P3_*` 文档、`docs/observation_action.md`、`docs/reward_design.md`。

---

## 10. 修订记录

| 日期 | 版本 | 变更 | 作者 |
|---|---|---|---|
| 2026-10-21 | v0.1 | 建立 P8-01/P8-02 接口 + 测试 + 门禁/预算占位 | P8 子任务（CPU，无训练） |
