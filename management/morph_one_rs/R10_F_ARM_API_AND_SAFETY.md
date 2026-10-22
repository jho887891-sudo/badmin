# R10-F：机械臂接口与安全（F1–F4）执行报告

- 分支：`r10/f-arm-api-safety`（worktree `badmin_wt_r10f`）
- 基线：`p5/controller-hold @ 2afd6f2`；为使用既有 P7 代码并与本分支集成，先合入 `p7/closed-loop-wiring @ a3d6de0`（merge commit `2abe7102ba5a1b5875cecd45a77431cbf15ebcbb`，无冲突）
- 派工单：`chatgpt/r10-32-task-batch-20261022 @ 14f339ff` 的 **F · 机械臂接口与安全**（F1=P5-09、F2=P7-09、F3=P7-10、F4=P7-11/12）
- 环境：CPU only，Python 3.11.9 + pytest 9.0.2 + numpy；**未启动 Isaac/Kit/GPU/真机**
- 性质：只声明**接口与接口级安全**成立；不含任何物理接触、命中率或真机结论

---

## 1. F1（P5-09）：统一 B601-RS controller API

**新文件**：`simulation/badminton_scene/controller_api.py`（未修改任何既有 P5 文件）

### 1.1 API 契约

| 层 | 成员 | 语义 |
|---|---|---|
| 抽象接口 | `ArmController`（abc.ABC） | 状态 / 关节目标 / reset / HOLD / IK 钩子 |
| 状态 | `sample_state(at_s)`、`state(at_s)` | 返回 `JointReading`（按关节名寻址；**无 torque 字段**——本仓库无该数据） |
| 目标 | `command_joint_targets(targets, at_s)` | 序列或 `{名称: 值}`；通过后返回 `JointCommand`（含每关节限位余量） |
| 复位 | `reset(joint_pos, at_s)` | **不把 None 解释成 rest pose**（那是 `reset_to_rest()` 的职责） |
| HOLD | `hold(at_s)` / `release(at_s)` | 冻结最后一个**已接受**目标，按位相同，不做任何算术 |
| IK 钩子 | `solve_ik(target_pose, at_s, seed=...)` | 声明但无实现，抛 `IkUnavailable`（不返回"看起来像解"的姿态） |
| 具体适配器 | `B601RSControllerAdapter` | 包在既有 `B601RSController` 之上，沿用其 DOF 映射、P5-05 包络、P5B reset 路径 |
| 版本 | `API_VERSION = "r10f/arm-controller/v1"` | 调用方可钉住 |

### 1.2 类型化错误路径（12 个，全部带稳定 `.code`）

`UnknownJointTarget`(unknown_joint)、`TargetWidthError`(target_width)、`NonFiniteTarget`(non_finite_target)、`OutOfLimitTarget`(out_of_limit_target)、`StaleState`(stale_state)、`StateUnavailable`(state_unavailable)、`DoubleReset`(double_reset)、`HoldDuringMotion`(hold_during_motion)、`HoldEngaged`(hold_engaged)、`NotHolding`(not_holding)、`HoldWithoutTarget`(hold_without_target)、`IkUnavailable`(ik_unavailable)。

语义要点（即 F1 的安全性所在）：

- **StaleState**：`state()` 读取缓存读数，超过 `max_state_age_s`（默认 0.5 s）即拒绝；reset 之后缓存读数被**作废**（reset 是"瞬移"，旧读数只描述机械臂曾经在哪）。
- **DoubleReset**：在没有任何进展（无被接受目标、无新状态读取）时连续两次 reset 被拒——复位循环会掩盖它本想清除的故障。
- **HoldDuringMotion**：`hold()` 在目标仍在执行中（setpoint 在途）时被拒；冻结一个机械臂从未到达的目标是"带品牌的缓慢漂移"。`settle()` 由上层监督者声明"已到达"后才允许 HOLD。

### 1.3 拒绝先于写入（**本任务的核心证据**）

ABC 拥有全部校验，只调用 3 个钩子：`_do_read()`、`_do_write(values)`、`_do_reset(rows)`。子类**永远看不到未校验的目标**。校验顺序固定：宽度 → 有限性 → 逐关节限位 → HOLD/在途状态，全部在任何写钩子之前。

测试通过**计数**证明（`tests/r10_f/test_controller_api_contract.py` + `test_controller_adapter_wiring.py`）：

| 拒绝场景 | 触发异常 | `write_count` | 写钩子调用次数 |
|---|---|---|---|
| joint3=3.5（上限 3.14） | `OutOfLimitTarget`（`written=False`） | 0 | 0 |
| gripper_joint1=0.06 m（行程 0.05） | `OutOfLimitTarget` | — | reset 0 |
| 未知关节 `joint7` / 缺失 `joint5` | `UnknownJointTarget` | 0 | 0 |
| 7 个值 / NaN / Inf | `TargetWidthError` / `NonFiniteTarget` | 0 | 0 |
| 在 HOLD 下命令 / reset | `HoldEngaged` | 不变 | 0 |

**真实适配器同一性质**：`B601RSControllerAdapter` + 记录型 `reset_fn`，越界 reset 后 `calls == []`（reset 钩子一次都没被调用）；合法 reset 恰好 1 次，第二次立即 `DoubleReset` 且次数仍为 1。

---

## 2. F2（P7-09）：verified-8DOF 与 proxy-6DOF 可达性对比（每条判定都带标签）

**新文件**：`src/badminton_brain/reachability/backend_compare.py`；运行器 `tools/r10f/run_r10f_backend_compare.py`

每条判定强制携带 5 个标签 + 后端自身 provenance：`backend`（proxy_dh | verified_p6_fk）、`frame_name`（racket_face_proxy | gripper_end）、`target_kind`（RACKET_FACE_PROXY | ARM_END_FRAME）、`source_class`（PROXY_DH_ENGINEERING_V0_1 | VERIFIED_ASSET_KINEMATICS）、`limit_source`，以及 verified 侧 `fk_module_sha256`。

**"一个对象带两个 backend id 就抛错"** 由 `LabelledVerdict.backend_ids: Tuple[str, ...]` + `single_backend_id()` 实现并测试：`backend_ids=("proxy_dh","verified_p6_fk")` → `BackendMixError`；标签与 provenance 不一致同样抛错。

**不静默回退**：请求 verified 后端但 P6 FK 不可导入时 → 该 section 记为 `status=UNAVAILABLE`、`verdicts=[]`、`refusal` 含 `VERIFIED_FK_UNAVAILABLE`；**不会**把 proxy 数字放在 verified 名下。

真实运行（`BADMIN_P6_ROOT=<p6/dynamic-fk worktree>`），输出 `outputs/badminton_brain/r10f/backend_compare.json`：

| backend | frame | scenario | decision | reason |
|---|---|---|---|---|
| proxy_dh | racket_face_proxy | proxy_convergent | NOT_REACHABLE | TIME_TO_TARGET |
| proxy_dh | racket_face_proxy | base_column_proxy | NOT_REACHABLE | SELF_COLLISION_PROXY |
| proxy_dh | racket_face_proxy | outside_box | NOT_REACHABLE | WORKSPACE_BOUND |
| proxy_dh | racket_face_proxy | low_confidence | NOT_REACHABLE | COVARIANCE_ABOVE_THRESHOLD |
| verified_p6_fk | gripper_end | proxy_convergent | NOT_REACHABLE | IK_NOT_CONVERGED |
| verified_p6_fk | gripper_end | base_column_proxy | NOT_REACHABLE | SELF_COLLISION_PROXY |
| verified_p6_fk | gripper_end | outside_box | NOT_REACHABLE | WORKSPACE_BOUND |
| verified_p6_fk | gripper_end | low_confidence | NOT_REACHABLE | COVARIANCE_ABOVE_THRESHOLD |

两次后端均 `status={'proxy_dh': 'OK', 'verified_p6_fk': 'OK'}`（dof 6 / 8）。**注意**：本组场景是"拒绝场景"集合（零种子 + 0.30 s），因此 8 条全为 NOT_REACHABLE；这**不是**可达性质量结论。两后端回答的是**不同问题**（代理拍面点 vs 真实 link6 的 gripper_end），本模块只做对比，不做排名、不做合并。

---

## 3. F3（P7-10）：安全 FSM 负例（可证伪）

**新文件**：`tests/r10_f/test_hold_negative_paths.py`（驱动真实 `InteractionFsm`）

每条负例断言 4 件事：状态 = HOLD、`hold_reason` 是**预期的那一个**、`emitted_commands == []`（零指令）、**latch 成立**（再次调用 stage 被 suppress 且状态不变）。

| # | 场景 | 触发点 | 期望 HOLD 原因 | 发出指令 | latch |
|---|---|---|---|---|---|
| 1 | 关节限位余量（joint3=-0.02，余量 0.02 < 0.05） | `on_execute_command` | `JOINT_LIMIT_MARGIN` | 0 | 是 |
| 2 | 自碰撞代理（面心 0.20 m < 0.24 m 圆柱） | `on_plan` | `UNREACHABLE_TARGET`（detail 含 `SELF_COLLISION_PROXY`） | 0 | 是 |
| 3a | 过期帧（捕获 -0.5 s） | `on_frame` | `STALE_FRAME` | 0 | 是 |
| 3b | 过期目标（intercept 已过去 0.05 s） | `on_prediction` | `TIME_TO_HIT_INVALID` | 0 | 是 |
| 4 | 不可达目标（面心 x=2.4 m 出工作空间） | `on_plan` | `UNREACHABLE_TARGET`（含 `WORKSPACE_BOUND`） | 0 | 是 |
| 5 | 低置信度（sigma 0.40 m > 0.05 m） | `on_frame` | `COVARIANCE_ABOVE_THRESHOLD` | 0 | 是 |
| 6 | 用过期帧 reset | `reset()` | `ResetRejected`，HOLD 保持 | 0 | 是 |

### 3.1 "先写越界目标再拒绝" 的显式反证

`test_the_arm_api_refuses_before_writing_the_vector_the_fsm_refused` 把 FSM 拒绝的同一形态向量（joint3=-0.02）交给 F1 的 controller API：
`pytest.raises(OutOfLimitTarget)`；`arm.writes == []`；`arm.write_count == 0`；`context["joint"] == "joint3"`；`context["written"] is False`。
即：越界目标在写入之前被拒，且**写入计数为 0**。

---
## 4. F4（P7-11/12）：simulation-only 闭环审计与三层声明拆分

**新文件**：`src/badminton_brain/harness/claim_layers.py`

审计对象：`src/badminton_brain/harness/{closed_loop,wired_closed_loop}.py` 与 `outputs/badminton_brain/p7_closed_loop/wired_episodes.json`（12 个 episode，全部 `hit_rate=null`、`real_hit_rate_claimed=false`）。

| 层 | 声明内容 | 所需证据 | 本仓库状态 |
|---|---|---|---|
| `TRAJECTORY_REACHABLE` | IK 收敛 + 关节余量 + P5-05 包络通过的**指令**可达 | 带 backend/frame/provenance 的可达判定 + FSM 事件日志 + 限位内关节向量 | **SUPPORTED**（有已提交 Run） |
| `GEOMETRIC_CONTACT` | 球拍**代理面**与球球面在声明距离内通过 | P6 球拍碰撞代理 + 球半径 + 每 episode 表面距离统计 | **PROXY_ONLY**（代理网格 + 运动学回放球体；无扫掠体、无相对速度） |
| `PHYSICAL_HIT` | 球拍施加冲量、球以回球轨迹离开 | 接触模型（摩擦/恢复系数、拍线柔度）+ 已标定 TCP 与拍面 + 物理引擎步进的球体 + 实测/仿真碰撞后球态 | **BLOCKED**（四项全缺，`evidence_paths=()`） |

**hit_rate 保持 null，且有人报非 null 就失败**：
- `forbid_hit_rate(v)` 对任何非 None 值抛 `HitRateFabrication`；
- `audit_report()` 递归遍历报告，**任意深度**出现非 null `hit_rate` 即抛错（含"每 episode 为 null、summary 却写了 0.31"这种情况）；报告里**完全没有** `hit_rate` 字段也抛错（缺失与"忘记填"不可区分）；`real_hit_rate_claimed=True` 与 `physical_hit_verified=True` 同样抛错。
- 三层聚合报告 `three_layer_claim()` 固定写出 `hit_rate: None`、`real_hit_rate_claimed: False`。

---

## 5. 真实命令与退出码

所有命令在 `E:\具身智能\badmin_wt_r10f` 下执行（CPU，无 Isaac/GPU）。

| # | 命令 | 结果 | 退出码 |
|---|---|---|---|
| 1 | `python -m pytest tests/r10_f -q` | **42 passed** | **0** |
| 2 | `python -m pytest tests/simulation/arm_pose -q` | **119 passed, 5 skipped**（未被本次改动破坏） | **0** |
| 3 | `python -m pytest tests/badminton_brain/test_interaction_fsm.py tests/badminton_brain/test_p7_closed_loop_wiring.py -q` | **68 passed** | **0** |
| 4 | `$env:BADMIN_P6_ROOT=<p6/dynamic-fk>; python tools/r10f/run_r10f_backend_compare.py` | 表格见 §2；写入 `outputs/badminton_brain/r10f/backend_compare.json` | **0** |

r10_f 用例分布：F1 `test_controller_api_contract.py` 13 + `test_controller_adapter_wiring.py` 6；F2 `test_backend_provenance.py` 8；F3 `test_hold_negative_paths.py` 8；F4 `test_claim_layers.py` 7。

---

## 6. UNKNOWN / 未完成（不得由本报告推导）

1. **无 IK 求解器**：`solve_ik()` 仍为抛出式钩子；B601-RS 球拍目标的 IK 与已标定 TCP 均为 UNKNOWN。F1 只发布钩子与拒绝语义。
2. **速度/加速度上限是 POLICY CAP**：`ENVELOPE_SOURCE_NOTE` 明确写出 URDF 无该数据、本机无实测；这两个数字不能当作数据手册值。
3. **F2 未产出可用的可达性质量结论**：8 条判定全为 NOT_REACHABLE，因为所选用例是"拒绝场景"。真正的可达率对比需要一组有意义的候选击球点（属 P7 规划侧），本次未做。
4. **verified 后端需外部 P6 检出**：本仓库不 vendored P6 FK，运行需 `BADMIN_P6_ROOT` 或 `--p6-root`；缺失时该 section 为 UNAVAILABLE（**不**由 proxy 顶替）。CI 上若无 P6 检出，verified 一半为 UNKNOWN。
5. **PHYSICAL_HIT 完全 BLOCKED**：无接触模型、拍面 frame 在 P6 中为 UNKNOWN（frame_tree 边 13/14/15）、无 D455 外参、闭环为运动学回放。`hit_rate` 因此恒为 null。
6. **无真机、无 Isaac、无 GPU**：本报告一切结论均为 L1/L2（接口级/离线仿真级），不得表述为物理验证。
7. **本分支为保证可运行先合入 p7/closed-loop-wiring**：PR 基为 `p5/controller-hold`，因此 PR diff 会包含 p7 的既有提交；这些文件**未被修改**（仅新增本报告的 4 个模块 + 6 个测试文件 + 1 个运行器）。若评审要求纯 F 线 diff，可改为基于 `p7/closed-loop-wiring` 重开分支。

## 7. 变更文件清单

新增（未改动任何既有 P5/P6/P7 源码、未改 queue state、未动 `scripts/simulation/*`）：

- `simulation/badminton_scene/controller_api.py`（F1）
- `src/badminton_brain/reachability/backend_compare.py`、`tools/r10f/run_r10f_backend_compare.py`（F2）
- `src/badminton_brain/harness/claim_layers.py`（F4）
- `tests/r10_f/{conftest.py,test_controller_api_contract.py,test_controller_adapter_wiring.py,test_backend_provenance.py,test_hold_negative_paths.py,test_claim_layers.py}`（F1–F4）
- `outputs/badminton_brain/r10f/backend_compare.json`（F2 运行产物）
- `management/morph_one_rs/R10_F_ARM_API_AND_SAFETY.md`（本文件）
