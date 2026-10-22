# P7-09 + P7-10：可达性 / IK 接口与安全状态机

分支：`p7/reachability-fsm`（基于 `p7/dataset-audit-latency` @ 03af23b）
范围：**仿真/离线**。无训练、无 GPU、无 YOLO 推理、无真机、无物理引擎闭环。

---

## 0. 一句话结论

本切片交付两件事，并且两件事都**在证据不足时拒绝给出肯定答案**：

1. **可达性 + IK 接口**：给定球拍拍面目标位姿（或预测拦截点 + 逼近方向）与关节限制，
   返回 `REACHABLE` / `NOT_REACHABLE` / `UNKNOWN` **以及原因**。IK 只有两种合法行为：
   在文档化容差内收敛并报告残差，或抛 `IkNotImplemented`；**绝不静默返回近似位姿**。
2. **安全状态机**：SEARCH / TRACK / PREDICT / PLAN / EXECUTE / HOLD，全部失败路径
   **FAIL-CLOSED 进入 HOLD 并记录类型化原因**；HOLD **闩锁**，必须"显式复位 + 新鲜有效输入"
   才能离开，且 HOLD 期间**不发出任何指令**。

**不声称**：真实击球率、真实 IK 可用性、真实 TCP、D455 标定、PPO 策略。

---

## 1. 交付物

| 文件 | 作用 |
|---|---|
| `src/badminton_brain/reachability/kinematics.py` | 文档化解析 FK（标准 DH，6R），代理运动学表 |
| `src/badminton_brain/reachability/ik.py` | IK **接口契约** + `DampedLeastSquaresIk` 真求解器 + `UnavailableIk` 占位 |
| `src/badminton_brain/reachability/reachability.py` | 三态判定、原因枚举、2D→3D 拒绝闸门 |
| `src/badminton_brain/safety/interaction_fsm.py` | 六态 FSM、HOLD 闩锁、事件日志、协方差/剩余时间传播 |
| `src/badminton_brain/harness/closed_loop.py` | 预注册仿真闭环 **harness only**（可 mock 运行，缺模型时拒绝给击球率） |
| `tests/badminton_brain/test_reachability_ik.py` | 可达性 / IK 契约测试 |
| `tests/badminton_brain/test_interaction_fsm.py` | 全转移、全 FAIL-CLOSED、闩锁、2D 拒绝 |
| `tests/badminton_brain/test_closed_loop_harness.py` | harness 拒绝语义 + 确定性 |

继承复用（未重做）：P7-08 `timing/latency_budget.py` 的 `MAX_SOURCE_FRAME_AGE_S = 0.100 s`
（本切片直接 import，并有测试钉死相等），P5-05 的 2.0 rad/s / 20 rad/s² 策略上限常量语义。

---

## 2. IK 契约（硬约束）

```text
solve(target, seed) -> IkResult   仅当 residual_position <= tol_position_m 且
                                  residual_rotation <= tol_rotation_rad
                    | raise IkNotImplemented   本求解器根本不实现
```

* `IkResult` **无法**以 `converged=False` 构造（构造即 `ValueError`）；
* 未达容差 → **抛 `IkNotConverged`**，异常里带残差 / 容差 / 迭代数；
* 每个结果都记录 `solver`、`iterations`、`tolerance_*`、`residual_*`，可离线审计。

### DampedLeastSquaresIk 的假设（全部未在真机验证）

* 雅可比为**中心差分数值雅可比**（步长固定 1e-6 rad，确定性），DLS 阻尼 1e-3，单步上限 0.20 rad；
* 关节限位来自 PiPER URDF 公布值（`joint2 ∈ [0, 3.14]`、`joint3 ∈ [-2.967, 0]` 见
  `simulation/piper_stage0/piper_cfg.py`），未在实机或 USD 上复核；
* DH 表为 **ENGINEERING_V0_1 代理**（URDF 不在本仓库），拍面点距 link6 原点 0.50 m，
  与 `simulation/badminton_scene/racket_cfg.py` 的 `head_center_from_tcp = 0.50` 一致；
* 不建模自碰撞、力矩、速度（由可达性层的余量/代理补上）。

> 因此：**IK 收敛只证明求解器正确，不证明真机能到**。

---

## 3. 可达性判定顺序（首个失败即原因）

```text
1 标定可用？           否 -> UNKNOWN / CALIBRATION_MISSING
2 维度 == 3？          否 -> 抛 DetectionDimensionalityError（2D 绝不提升为 3D）
3 目标可构造？         否 -> UNKNOWN / NON_FINITE_TARGET
4 剩余时间 > 0？       否 -> NOT_REACHABLE / TIME_TO_TARGET（零/反向时间）
5 工作空间盒内？       否 -> NOT_REACHABLE / WORKSPACE_BOUND
6 基座柱代理外？       否 -> NOT_REACHABLE / SELF_COLLISION_PROXY
7 sigma <= 阈值？      否 -> NOT_REACHABLE / COVARIANCE_ABOVE_THRESHOLD
8 IK 求解              IkNotImplemented -> UNKNOWN / IK_NOT_IMPLEMENTED
                       IkNotConverged  -> NOT_REACHABLE / IK_NOT_CONVERGED
9 关节余量 >= 阈值？   否 -> NOT_REACHABLE / JOINT_LIMIT_MARGIN
10 所需换位时间足够？  否 -> NOT_REACHABLE / TIME_TO_TARGET
   否则 REACHABLE / IK_CONVERGED
```

自碰撞为**柱体+球代理**（基座半径 0.18 m + 拍面球 0.06 m），**不是碰撞检测**。
时间充分性是**下界**：`max_i |Δq_i| / v_cap_i`，不含底盘行进（底盘由 P6 拦截搜索负责）。

---

## 4. 状态机与转移表

```text
SEARCH  --FRAME-->        TRACK
TRACK   --PREDICTION-->   PREDICT
PREDICT --PLAN-->         PLAN
PLAN    --COMMAND-->      EXECUTE
EXECUTE --CYCLE-->        PLAN
EXECUTE --PREDICTION-->   PREDICT
EXECUTE --FRAME_UPDATE--> EXECUTE   （自环：执行中感知继续刷新，仍走陈旧/协方差检查）
任意非 HOLD 态 --任何失败/异常-->   HOLD
HOLD    --RESET(新鲜有效且带操作者)--> TRACK
```

* HOLD 无自动出口：无定时器、无"自动恢复"；复位后也只能回到 **TRACK**，
  不允许直接回 EXECUTE（测试 `test_valid_reset_returns_to_track_only` 钉死）。
* HOLD 期间任一阶段调用都会被记 `SUPPRESSED`，`on_execute_command` 抛
  `CommandSuppressed`，`commands_emitted` 保持 0。
* 每次转移写入 `FsmEvent(seq, at_s, from, to, trigger, reason, detail, time_to_hit_s, sigma_m)`；
  `report()` 输出 `hold_reasons_seen` 与阈值快照，作为安全证据。

---

## 5. FAIL-CLOSED 对照表（全部有测试）

| 触发条件 | 类型化原因 | 结果 | 测试 |
|---|---|---|---|
| 无标定 / 标定未验证 | `CALIBRATION_MISSING` | HOLD | `test_missing_calibration` |
| 源帧年龄 > 0.100 s | `STALE_FRAME` | HOLD | `test_stale_frame` |
| 帧来自未来（时钟错） | `STALE_FRAME` | HOLD | `test_future_frame` |
| 帧 sigma > 0.05 m | `COVARIANCE_ABOVE_THRESHOLD` | HOLD | `test_covariance_above_threshold_at_the_frame` |
| 预测 sigma > 阈值 | `COVARIANCE_ABOVE_THRESHOLD` | HOLD | `test_prediction_covariance_above_threshold` |
| 规划耗时使 sigma 增长超限 | `COVARIANCE_ABOVE_THRESHOLD` | HOLD | `test_covariance_growth_blows_the_budget_during_planning` |
| 时间命中 = 0 或反向 | `TIME_TO_HIT_INVALID` | HOLD | `test_zero_and_reversed_time_to_hit` |
| 剩余时间 < 0.02 s | `TIME_TO_HIT_INVALID` | HOLD | `test_time_to_hit_below_minimum` |
| 目标不可达（盒/代理/IK） | `UNREACHABLE_TARGET` | HOLD | `test_unreachable_target` |
| 可达性 UNKNOWN（无 IK） | `UNREACHABLE_TARGET` | HOLD | `test_unknown_reachability_is_not_a_licence_to_move` |
| 所需换位时间不足 | `UNREACHABLE_TARGET` | HOLD | `test_time_insufficient_for_the_slew` |
| 关节限位余量不足 | `JOINT_LIMIT_MARGIN` | HOLD | `test_joint_limit_margin_violation_on_the_command` |
| 速度超 P5-05 上限 | `COMMAND_OVER_CAP` | HOLD | `test_velocity_cap_violation` |
| 加速度超 P5-05 上限 | `COMMAND_OVER_CAP` | HOLD | `test_acceleration_cap_violation` |
| 阶段抛异常 | `STAGE_EXCEPTION` | HOLD | `test_exception_in_the_ik_stage` |
| 顺序错误（未预测就规划等） | `STAGE_ORDER` | HOLD | `test_plan_before_prediction_is_fail_closed` |
| 2D 检测进预测阶段 | `DETECTION_NOT_3D` | HOLD | `test_two_dimensional_frame_cannot_reach_the_prediction_stage` |

**2D→3D 拒绝**：`FrameInput(dim=2, position_m=...)` 构造即抛
`DetectionDimensionalityError`（`AssertionError` 子类）；`require_world_3d` 拒绝 `(2,)` 与 `None`；
`assess_reachability(detection_dim=2)` 直接拒绝。没有"假设球高"的兜底深度。

**协方差 / 剩余时间传播**：预测的 `sigma_start_m` 与 `sigma_growth_mps`、`intercept_time_s`
进入 FSM 后按阶段传播（规划耗时即球飞行时间），每个事件记录传播后的 `sigma_m` 与
`time_to_hit_s`；签发指令前再核一次。测试 `PropagationTests` 钉死数值
（0.30 s → 0.20 s，0.010 m → 0.015 m）。

---

## 6. 仿真闭环 harness（仅 harness）

预注册场景：`FIXED_BALL` / `LOW_SPEED_INCOMING`（约 2 m/s 线性来球）/ `PERTURBED`（定种子噪声）。
无物理引擎、无相机、无机器人（报告字段显式为 `null`）。

**拒绝规则**：缺少拍面碰撞代理或缺少标定 → `status = REFUSED`、`hit_rate = null`、
`synthetic_contact_rate = null`、`contacts = null`，并列出 `refusal_reason`。
即使两者齐全，`hit_rate` 也恒为 `null`，`real_hit_rate_claimed = false`，
仅给出 `synthetic_contact_rate`（mock 球心与 mock 拍面中心距离 <= 代理半径的几何统计）。

实测（mock，确定性）：

| 场景 | status | final_state | 发出指令 | contacts | synthetic_contact_rate | hit_rate |
|---|---|---|---|---|---|---|
| FIXED_BALL | MOCK_RUN_COMPLETE | EXECUTE | 12 | 0 | 0.00 | null |
| LOW_SPEED_INCOMING | MOCK_RUN_COMPLETE | EXECUTE | 12 | 3 | 0.25 | null |
| PERTURBED | MOCK_RUN_COMPLETE | EXECUTE | 12 | 3 | 0.25 | null |

> 上表 `synthetic_contact_rate` 是 **mock 几何统计**，不是击球率，不得引用为击球率。

运行：
```powershell
$env:PYTHONPATH="src"; python -m src.badminton_brain.harness.closed_loop --scenario all
$env:PYTHONPATH="src"; python -m src.badminton_brain.harness.closed_loop --scenario PERTURBED --without-proxy
```

---

## 7. 测试证据（真实退出码）

```text
python -m pytest tests/badminton_brain/test_reachability_ik.py \
  tests/badminton_brain/test_interaction_fsm.py \
  tests/badminton_brain/test_closed_loop_harness.py -q
-> 74 passed, 6 subtests passed        exit code 0

python -m pytest tests/badminton_brain -q
-> 412 passed, 1 skipped, 12 subtests passed   exit code 0
```

全部为纯 Python / pytest 9.0.2 / Python 3.11.9 / numpy 2.4.6，合成 fixture，确定性
（固定种子、固定雅可比步长、固定种子关节向量；重复运行逐字节相同）。

---

## 8. UNKNOWN（必须保留）

| UNKNOWN | 影响 | 现状 |
|---|---|---|
| **无验证过的 TCP / 拍面变换** | 拍面位姿与 `t_tcp_contact` 未测量；IK 收敛≠真机到位 | 全部数值为代理，可达性结论仅离线 |
| **无 D455 标定 / 外参** | `Calibration.verified=False` 时一切判定降级为 UNKNOWN | 测试中使用的 `verified=True` 是**合成标定**，不是实测 |
| **无真实击球统计** | 无任何真实闭环击球率；harness 只做 mock 几何 | `hit_rate` 永远为 null |
| **无 PPO** | 规划为确定性规则/mock，不涉及学习策略 | 本切片不含 RL |
| **代理 DH 与真实 URDF 不一致** | 关节解不可直接下发 | 需 URDF 到位后替换 `PROXY_KINEMATIC_TABLE` |
| **速度/加速度上限是策略上限** | 2.0 rad/s / 20 rad/s² 非数据手册值 | 取自 P5-05，仍未标定 |
| **时间下界忽略底盘行进** | 可能把"底盘来不及"判成可达 | 底盘由 P6 拦截搜索负责，未合并 |

---

## 9. 对外契约（供后续切片接）

```python
from badminton_brain.reachability import (
    IkTarget, DampedLeastSquaresIk, UnavailableIk, assess_reachability,
    Calibration, KinematicLimits, ReachabilityDecision, ReachabilityReason,
)
from badminton_brain.safety.interaction_fsm import (
    InteractionFsm, FsmLimits, FrameInput, PredictionInput, PlanInput, FsmState, HoldReason,
)
```

* 任何新求解器：实现 `solve(target, seed) -> IkResult`，失败必须抛；
* 任何新传感器输入：2D 只能构造 `FrameInput(detection_dim=2)`，**不得**带 `position_m`；
* 任何自动恢复逻辑：不允许绕过 `reset(frame=..., prediction=..., operator=...)`。
