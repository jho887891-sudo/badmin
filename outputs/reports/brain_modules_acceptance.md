# ROBOT BRAIN 模块化架构 — 实现与验收报告（v0.1）

日期：2026-09-13　仓库：/home/T7/ojh/robot_sim（本地镜像 E:\具身智能\badmin_project，SSOT）
流程：严格按 superpowers 执行 —— writing-plans → subagent-driven-development → test-driven-development；
全部裁决记入 docs/superpowers/plans/2026-09-13-brain-modules.md 的 ledger 与 management/DECISIONS.md。

## 1. 交付物（8 层全部实现，均有测试）

| 层 | 模块文件 | 测试数 | 关键实测 |
|---|---|---|---|
| 感知 | perception/stereo_geometry.py、synthetic_detector.py、perception_module.py（适配器） | 28+7 | 三角化误差 5.3e-15 m；ground truth 不泄漏（有断言） |
| 估计 | estimation/robot_localization.py(EKF)、shuttle_ukf.py(UKF)、estimator.py（适配器） | 22+13+3 | 噪声场景 0.0138 m vs 航位推算 0.0356 m；UKF RMSE 0.00568 vs 0.01424 |
| 预测 | prediction/physics_predictor.py | 17 | 与 shuttle_aerodynamics.rollout 逐点差 0.0；落地点差 1.03e-5 m |
| 决策 | decision/feasibility.py、intercept_search.py、travel_model.py、decision_module.py（适配器） | 33+26+7 | canonical 落点 x≈-1.133；最早拦截 t*=0.405 s 与解析一致 |
| 规划 | planning/expert_planner.py、ppo_policy_stub.py | 23 | 饱和裁剪精确到 max_wheel_speed*R=2.4 m/s；PPO 占位在 final 被拒 |
| 安全 | safety/safety_shield.py | 38 | 6300 次调用扫描零越限；急停/看门狗经管线可达 |
| 执行 | execution/sim_adapter.py | 30 | 与规范 kinematics 逐位一致；坐标系护栏实测 court 直传偏 35.5% |
| 自适应 | adaptation/online_adaptation.py | 26 | drag_scale 1.0→1.24999999（真值 1.25），残差来自真实物理模型 |
| 集成 | apps/full_brain.py（接口发现装配 + 运行时推入） | 11 | 端到端 canonical 场景；estop 差异化断言；预测推入闭环 |
| 契约 | types/interfaces/registry/pipeline/validation | 10+15+6+3 | 层边界/顺序/Court Frame/批量形状/时钟 |

## 2. 全量回归（wave 3，实测）

suites_passed=26  suites_failed=0  tests_total=422
（此后新增 T11 运行时接线测试 2 项，合计 424）

运行方式：
  ./env_isaaclab/bin/python tools/run_all_tests.py
  for t in $(find tests -name 'test_*.py'); do timeout 180 ./env_isaaclab/bin/python $t; done

## 3. 架构规则（有测试强制执行）

| 规则（来源） | 执行方式 | 测试 |
|---|---|---|
| 主闭环顺序（ROBOT_BRAIN S15） | BASELINE_ORDER 七层按序 | test_baseline_order_is_enforced |
| Safety 不可绕过 | 缺层则管线构造即失败 | test_safety_cannot_be_bypassed |
| 执行层只吃 SafeCommand | 由管线注入 | test_execution_only_receives_safe_command |
| 模块不得代劳他层 | 层输出类型不符 → BrainBoundaryError | test_layer_output_type_mismatch_is_a_boundary_error |
| Court Frame 唯一、无 env_origin | 消息 frame 校验 + 无该字段 | test_messages_refuse_non_court_frames |
| 频率不硬编码 | stage_rate_hz 默认 REQUIRES_MEASUREMENT | test_final_mode_refuses_unresolved_stage_rates |
| final 模式不许假装完成 | 未实现/未实测即 error | ValidationTests |

## 4. 评审闭环（SDD：每任务独立评审 + 整体评审）

| 任务 | 结论 | 处置 |
|---|---|---|
| T4 预测 | APPROVE | 3 个变异体全捕获；独立复算一致 |
| T2 定位 | REQUEST CHANGES | 已修：死测试复活（收集 15→22）+ 防孤儿用例；协方差定量断言；7/7 变异体被杀 |
| T5 可行性 | REQUEST CHANGES | 已修：阻断级 NaN 逃逸；重复真源收敛 travel_model.py；短窗 median 方向门 |
| T7 规划 | REQUEST CHANGES | 已修：坐标系（court→body），旧写法轮速偏差 23% |
| T8 安全 | REQUEST CHANGES | 已修：SafetyContext 推入（急停/看门狗不再死代码）+ 两处越限反例 |
| T10 自适应 | REQUEST CHANGES | 已修：雅可比 gain/theta 错误；步间 dt；set_prediction 通道 |
| T9 执行 | REQUEST CHANGES | 修复中（measurement_requirements、TEMP 警告、几何单一真源、一处空测试） |
| 整体 broad | 进行中 | 待结论 |
| T1/T3/T6 | 未派单模块评审 | 由消费方测试（T4/T5/T6/T11）间接验证 |

## 5. 裁决（DECISIONS）

DEC-013 契约扩 odom/IMU｜DEC-014 base_twist 机体系｜DEC-015 残差语义拆分｜
DEC-016 时间基准与时钟推入｜DEC-017 src 只读引用 simulation 规范实现｜DEC-018 TEMP 与 final 拒绝

## 6. 诚实边界（未实现/未实测）

- 未实现的算法：真实检测器（YOLO）、真机驱动、PiPER IK/NMPC、全身协同、PPO（占位，调用即抛错）、
  Safety 投影 QP/碰撞/人机分离/力矩限、多目标优化。
- 未实测的真实参数：相机内外参与 baseline、Morph One 几何与质量特性、球拍实测参数与 T_link6_tcp、
  接触对参数、底盘/机械臂限值、噪声与增益 —— 全部 Param(value=None, REQUIRES_MEASUREMENT, source=...)；
  TEMP 一律 TEMP_PARAMETERIZED_PROXY 并写来源。
- final 模式会拒绝上述未解决项（有测试），即当前状态为「开发可用、上线不放行」，符合设计。

## 7. 复现

cd /home/T7/ojh/robot_sim
./env_isaaclab/bin/python tools/run_all_tests.py                       # 全量回归
./env_isaaclab/bin/python tests/badminton_brain/test_full_brain.py     # 端到端集成

冻结契约 sha256 基线：outputs/reports/contracts.md5（远端无 git，供评审做基线对比）
