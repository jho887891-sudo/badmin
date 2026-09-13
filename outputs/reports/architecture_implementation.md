# ROBOT BRAIN 模块化整体架构 - 实现记录 (v0.1)

依据：`docs/architecture/ROBOT_BRAIN.md`（总体架构与主闭环 S11/S12/S13/S15）、
`COORDINATE_SYSTEM.md`（Court Frame、T_A_B）、`SIMULATION_ENVIRONMENT.md`；
以及 `docs/simulation/BADMINTON_ROBOT.md`（机器人本体模块、TEMP 制度）。

## 1. 实现了什么

| 层 | 接口类 | 输入 → 输出 | 是否实现算法 |
|---|---|---|---|
| ① 感知 Perception | `PerceptionModule` | RobotSensorState → ShuttleMeasurement | 接口 + 契约（算法待感知数据） |
| ② 定位/状态估计 Estimation | `EstimationModule` | ShuttleMeasurement + RobotSensorState → UnifiedState | 接口 + 契约 |
| ③ 轨迹预测 Prediction | `PredictionModule` | UnifiedState → PredictedTrajectory | 接口 + 契约（物理 ODE 已有独立模块 `src/trajectory/shuttle_aerodynamics.py`） |
| ④ 击球决策 Decision | `DecisionModule` | UnifiedState + PredictedTrajectory → (HitDecision, BestIntercept?\|None) | 接口 + 契约 |
| ⑤ 规划/策略 Planning | `PlanningModule` | UnifiedState + decision + intercept + trajectory → WholeBodyTarget | 接口 + 契约 |
| ⑥ 安全 Safety | `SafetyModule` | WholeBodyTarget → SafeCommand | 接口 + 契约（**唯一可执行命令来源**） |
| ⑦ 执行 Execution | `ExecutionModule` | SafeCommand → Feedback | 接口 + 契约 |
| ⑧ 自适应 Adaptation | `AdaptationModule` | Feedback + UnifiedState → 修正量 | 接口 + 契约（在主链之后运行） |

装配与边界：`ModuleRegistry`（一层一模块、重复拒绝、层错拒绝、`replace()` 换实现）
+ `BrainPipeline`（按架构基线顺序装配与运行、逐步计时、层输出类型校验、`reset(env_ids)` 传导）
+ `validate_architecture(mode='development'|'final')`（阶段齐全、实现状态、频率是否已定）。

共享词汇：`src/common/status.py` 定义 `AssetStatus`/`Param`；
`simulation/robots/badminton_robot/badminton_robot_cfg.py` 改为**再导出**同一份（不再各写一套）。

## 2. 强制执行的架构规则（有测试）

| 规则（来源） | 实现方式 | 测试 |
|---|---|---|
| 主闭环顺序不可乱（S15） | `BASELINE_ORDER` 固定 7 层，按序调用 | `test_baseline_order_is_enforced` |
| Safety 不可绕过（S12/S63.8） | 缺少 Safety 层时 `BrainPipeline` 构造即失败 | `test_safety_cannot_be_bypassed` |
| 执行层只吃 SafeCommand | 执行阶段入参由管线注入 | `test_execution_only_receives_safe_command` |
| 模块不得代劳他层（S12） | 层输出类型不匹配 → `BrainBoundaryError` | `test_layer_output_type_mismatch_is_a_boundary_error` |
| 一模块一层、可替换 | 注册表重复/层错拒绝；`replace()` 后管线照常运行 | `test_duplicate_layer_is_rejected` / `test_module_swap_keeps_pipeline_working` |
| Court Frame 唯一（S28） | 所有消息 `frame='court'`，否则拒绝；消息**不含 env_origin 字段** | `test_messages_refuse_non_court_frames` / `test_no_message_exposes_env_origin` |
| 时间戳 = 仿真时间（S43） | 非有限时间戳直接拒绝 | `test_timestamp_must_be_finite_and_records_sim_time` |
| 批量形状 (N, …) | `check_batched`/`check_per_env_scalar` | `test_check_batched_helper` 等 |
| 频率不硬编码 | `PipelineConfig.stage_rate_hz` 每层默认 `REQUIRES_MEASUREMENT` | `test_final_mode_rejects_unresolved_stage_rates` |
| final 模式不许假装完成 | 未实现算法/未定频率 → `ok=False` + errors | `ValidationTests` × 3 |

## 3. TDD 证据（遵循 DEC-011 完整回路）

1. **RED**：先写 `tests/architecture/test_brain_types.py`（10 项）与 `test_brain_pipeline.py`（15 项），
   先出现 import error → 按承诺补“可运行空壳” → 重跑得到**纯断言失败** `FAILED (failures=6)` / `(failures=10, errors=5)`。
2. **GREEN**：分 2 片实现 —— 片 1 `types.py`（10/10 绿）；片 2 `interfaces.py + registry.py + pipeline.py + validation.py`（15/15 绿）。
3. **验证 GREEN**：**全量回归 11 套 = 134 项全通过**：
   `court 16 / shuttlecock 14 / aero 7 / racket 16 / piper-wrapper 6 / usd_glb_common 8 /`
   `robot_config 18 / robot_frames 15 / morph_one_kinematics 9 / brain_types 10 / brain_pipeline 15`。
4. **REFACTOR**：把 `AssetStatus/Param` 收敛为单一来源（`src/common/status.py`），robot 模块改为再导出；
   以既有 42 项机器人测试作为护栏 → 仍全绿。

## 4. 已实现 vs 未实现（诚实边界）

- **已实现**：整机模块化骨架、层间契约、装配与替换机制、边界强制、批量形状与 Court-Frame 纪律、开发/final 双模式校验。
- **未实现（明确不假装）**：各层算法本体（YOLO/UKF/EKF/可行性判据/拦截搜索/规划/PPO/Safety 具体限值/执行驱动）。
  所有模块 `is_implemented=False`，`final` 模式会因此报错 —— 这是设计要求，不是缺陷。
- **仍阻塞**：`docs/architecture/MODULE_INTERFACES.md` 缺失（ISSUE-006），故消息字段集为**临时契约**，
  该文档到位后可能需微调字段名；语义层（层名、输入输出类型、顺序）已按 ROBOT_BRAIN.md 固定。

## 5. 复现命令

```bash
cd /home/T7/ojh/robot_sim
./env_isaaclab/bin/python tests/architecture/test_brain_types.py
./env_isaaclab/bin/python tests/architecture/test_brain_pipeline.py
```
