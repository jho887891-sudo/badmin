# 设计决策记录（DECISIONS）

> 重要架构/方案决定在此登记，避免"几个月后不知道为什么这么做"。
> 每次新增决策：编号递增（DEC-001、DEC-002…）。

## 决策模板
```
## DEC-XXX 标题
- **日期：**
- **背景：**
- **备选方案：**
- **最终决定：**
- **原因：**
- **优点：**
- **缺点：**
- **未来是否允许修改：** 允许（条件）/ 不允许（理由）
```

---

## DEC-001 项目根目录：直接采用工作区根目录
- **日期：** 2026-09-08
- **背景：** 目录规范示例为 `badminton_robot/`，而工作区 `E:\具身智能\badmin_project` 已是本项目专属目录，且骨架（docs/management/experiments/...）已按规范就位。
- **备选方案：** a) 在工作区内再建 `badminton_robot/` 子目录包裹全部文件；b) 工作区根即项目根。
- **最终决定：** b) 工作区根即项目根，README 以 "badminton_robot" 命名项目。
- **原因：** 避免双重嵌套与路径冗余；目录名本身已表达项目身份；工具默认工作目录即项目根，操作直接。
- **优点：** 路径短；后续所有文件操作、git 操作天然以项目为根。
- **缺点：** 与规范示例目录名不完全逐字一致（仅命名层面）。
- **未来是否允许修改：** 允许 —— 若未来需与其它项目并列存放时，整体迁移到一个 `badminton_robot/` 子目录（纯移动，无技术成本）。

## DEC-002 Git 分支基线：master → main
- **日期：** 2026-09-08
- **背景：** `git init` 默认分支为 master；规范建议 main / dev / feature/* 开发模式。
- **备选方案：** a) 沿用 master；b) 空仓库阶段直接改名为 main。
- **最终决定：** b) 分支改名 main（仓库尚无 commit，改零成本）。
- **原因：** 与团队规范（main / dev / feature/*）一致。
- **优点：** 从第一版历史起命名即统一。
- **缺点：** 无。
- **未来是否允许修改：** 允许。

## DEC-003 首阶段不写代码，先建知识库与盘点现状
- **日期：** 2026-09-08
- **背景：** 项目从零开始，环境与硬件事实未知；用户明确要求先检查现状、建立项目管理结构。
- **备选方案：** 直接开始搭 Isaac Lab 环境或写 PPO 骨架。
- **最终决定：** 先落盘知识库结构（本 DEC 之前的文件清单），再做环境/硬件盘点，再进入 Stage 0 编码。
- **原因：** 本地 = 事实来源；信息先落盘避免只存在于对话上下文；符合"数据优先于猜测"。
- **优点：** 后续每次改动都有可追溯基线。
- **缺点：** 前期多一步文档工作。
- **未来是否允许修改：** 允许（流程性决定）。

## DEC-004 GitHub 远端（origin）与提交身份
- **日期：** 2026-09-08
- **背景：** 需要异地备份与协同源，以及 git 提交署名；用户提供仓库地址 https://github.com/jho887891-sudo/badmin（存在且为空）。
- **备选方案：** a) 不关联远端，仅本地仓库；b) 关联该 GitHub 仓库并统一提交身份。
- **最终决定：** b) 本地仓库关联 origin = 上述仓库；提交身份默认 `jho887891-sudo <jho887891-sudo@users.noreply.github.com>`。
- **原因：** 空仓库零冲突成本；noreply 邮箱是 GitHub 匿名保护约定，避免泄露个人邮箱。
- **优点：** 历史署名一致；可随时 push 备份；本地 = SSOT 不变，远端仅作镜像/协同。
- **缺点：** 展示名与真实姓名不一致（仅展示层面）。
- **未来是否允许修改：** 允许 —— 用户提供真实姓名/邮箱后修改本地 config 即可（只影响后续提交）。

## DEC-005 IsaacLab v3.0.0-beta2.patch1 的 torch 版本：采用官方 pin（2.10.0）而非用户消息中的 2.11.0
- **日期：** 2026-09-08
- **背景：** 用户指定 torch==2.11.0/torchvision==0.26.0（cu128）；但克隆的 IsaacLab tag 的官方文档（docs/source/setup/installation/pip_installation.rst）与 pyproject.toml dependencies 均要求 torch==2.10.0/torchvision==0.25.0/torchaudio==2.10.0（cu128 通道）。isaacsim[all]==6.0.1.0 的默认解析为 torch 2.11（宽松约束下的最新），官方流程随后 `uv pip install -U torch==2.10.0 ...` 强制回到 2.10。
- **备选方案：** a) 按用户消息装 2.11.0/0.26.0；b) 按官方文档装 2.10.0/0.25.0；c) 换更新的 IsaacLab 版本。
- **最终决定：** b) 官方 2.10.0/0.25.0/2.10.0（cu128）—— 因用户的最高原则是"严格按照当前 Isaac Lab 官方版本对应依赖安装，不自行换版"；用户的 2.11/0.26 疑对应更新版 IsaacLab。
- **原因：** 数据优先于消息转述；tag 自带文档与 pyproject 为同一发布的事实来源。
- **优点：** 与该 tag 的 isaaclab-dev 代码验证基线一致，避免隐性 API 不兼容。
- **缺点：** 与用户消息中的版本数字不一致（已在此记录，用户可随时推翻）。
- **未来是否允许修改：** 允许 —— 用户确认后可用 `uv pip install -U torch==2.11.0 torchvision==0.26.0 --index-url .../cu128` 切回 2.11。

## DEC-006 PiPER 仿真资产来源与导入管线：官方 AgileX 描述 + isaacsim 官方 URDF 导入器（fix_base）
- **日期：** 2026-09-09
- **背景：** 需要 PiPER 6 轴机械臂在 Isaac Lab 中单独正确导入（Stage 0）；无现成本地资产
- **备选方案：** a) 用仓库自带 USD（piper_v1/v2.usd 等，面向旧版 Isaac Sim，未采用）；b) 官方 isaacsim.asset.importer.urdf 从 URDF 现转；c) mujoco menagerie 等其它来源
- **最终决定：** b) 采用官方 agilexrobotics/piper_isaac_sim 的 piper_description/urdf/piper_no_gripper_description.urdf（纯 6 轴无夹爪），经官方 URDFImporter（fix_base=True、merge=False、allow_self_collision=False、run_multi_physics_conversion=True）转 USD；脚本 t01
- **原因：** 数据来源权威（厂商）；URDF 含完整 inertial/mass/limits/meshes；官方导入器保证 schema 与 PhysX 兼容；fix_base 固定基座（机械臂安装于底盘/桌面）
- **优点：** 可复现、版本无关；关节名 joint1..6 与限位忠实保留（已对照）
- **缺点：** 导入器默认不产生 stiffness/damping（URDF 无 transmission），增益需在 Isaac Lab 侧指定
- **未来是否允许修改：** 允许 —— 换真机标定 URDF 或官方 v100/夹爪版本时重走 t01 即可
- **相关文件：** scripts/piper_stage0/t01_convert_urdf.py、piper_cfg.py

## DEC-007 Stage0 控制参数：ImplicitActuator stiffness=150 / damping=15（240 Hz，自碰撞关闭）
- **日期：** 2026-09-09
- **背景：** PiPER 轻量臂（单链接质量 ≤1.2 kg），URDF 无传动增益；Stage0 需干净的单/全轴位置控制与稳定性
- **最终决定：** ArticulationCfg actuators：piper_arm ImplicitActuatorCfg(joint_names_expr=joint[1-6], stiffness=150.0, damping=15.0, effort_limit_sim=100.0)；init_state 取中段 rest（j2=1.5、j3=-1.5 避开 URDF 零位贴边界）；SimulationCfg dt=1/240；scene num_envs=1；spawn 自碰撞关闭（enabled_self_collisions=False）
- **原因：** 实测稳定且收敛（单关节 err≤0.046 rad、全轴 err≤0.048 rad、静止 drift 0.022 rad、600 s soak 0 NaN）；避免零位限位接触与自碰撞伪影干扰 Stage0 验证
- **优点：** 结果可复现；为后续加球拍负载/底盘时留出调参基线
- **缺点：** joint2 存在重力下垂稳态误差 ~0.045-0.048 rad（增益或前馈可后续优化）
- **未来是否允许修改：** 允许 —— 载荷/高速/自碰撞研究时按需调整并登记
- **相关文件：** scripts/piper_stage0/piper_cfg.py

## DEC-008 依赖管理：冻结 baseline，RL extras 用官方 editable 安装且不改核心版本
- **日期：** 2026-09-09
- **背景：** 用户要求补齐 RL extras 且不修改 Isaac Sim / Isaac Lab / Python / PyTorch / CUDA / Driver 版本；并冻结当前已验证环境
- **最终决定：** 1) RL 框架按官方路径安装（isaaclab_rl[all] editable）得到 rsl_rl/skrl/rl_games/sb3；2) 任何与 isaacsim-kernel 硬 pin 冲突的包恢复 baseline 版本；3) pip freeze 冻结到 robot_sim/logs/baseline_env_20260909.txt 作为 baseline；4) 后续升级一律先说明风险并记录
- **原因：** 可复现优先；官方兼容表约束核心版本
- **优点：** 训练框架可用且核心环境与 2026-09-08 验证一致
- **缺点：** 冻结清单将随未来合理升级而更新（需记录）
- **未来是否允许修改：** 允许 —— 条件：先记录风险与影响范围
## DEC-009 Robot 模块以 BADMINTON_ROBOT.md 为 Source of Truth（含 TEMP 制度与 Phase Gate）
- **日期：** 2026-09-13
- **背景：** 新增整机 Robot 模块设计规范（1671 行），明确 Frozen Decisions（Court Frame / PiPER 冻结 / 球拍 FixedJoint / Morph One 四转四驱拓扑 / env_origin 不泄漏）与资产真实性等级制度
- **最终决定：** 1) Robot 模块实现严格按该规范的 Phase 0→10 推进，未过当前 Phase 不进入下一 Phase；2) TEMP 参数必须显式标记（TEMP_PARAMETERIZED_PROXY / REQUIRES_MEASUREMENT），禁止把未知真实值补成"真实值"；3) 采用 TDD（RED→GREEN→Refactor）；4) 每个 Phase 以规范 §67 格式汇报，禁止只回 "Done"；5) 现阶段 NO PPO / NO DirectRLEnv / NO ManagerBasedRLEnv
- **原因：** 结构真实、未知显式、资产可替换、接口稳定、128-env 原生
- **优点：** Morph One 真资产到位后可无痛替换，上层 API 不变
- **缺点：** 大量参数与最终 USD 会长期处于 BLOCKED（这是设计意图）
- **未来是否允许修改：** 需用户显式批准（Frozen Decisions 不得自行修改）
## DEC-010 Frame 命名冲突：以别名解析到单一真值（不许存在两套平行定义）
- **日期：** 2026-09-13
- **背景：** `BADMINTON_ROBOT.md` §7 写 `racket_contact_frame` / `camera_center`；`docs/architecture/COORDINATE_SYSTEM.md` §2.5 写 `racket_contact` / `camera_rig`。两份都是规范来源
- **最终决定：** 实现中 **单一真值** 为 `racket_contact` 与 `camera_center`；`racket_contact_frame`、`camera_rig` 作为**别名**解析到同一变换（`FrameTree.get_transform` 入口统一解析别名）。COORDINATE_SYSTEM §2.5 明确要求"禁止同一实体存在多个无必要别名"，故不建立两套独立变换
- **原因：** 避免出现两个名字各有一套数值、后续算法取错 frame
- **优点：** 两套文档的命名都能用，且物理上只有一个真相
- **缺点：** 需要在验证器/文档中显式说明别名关系
- **未来是否允许修改：** 若上游规范统一为一个名字，则删除别名并更新调用点
## DEC-011 强制流程：TDD 完整回路（用户指令，2026-09-13）
- **背景：** 用户要求"以后都用 tdd 的完整路程"
- **最终决定：** 任何生产代码改动（新功能/修 bug/重构）必须走：
  `RED（先写测试并亲眼看它失败，失败须来自断言）→ 验证 RED → GREEN（最小实现）→ 验证 GREEN（含全量回归）→ REFACTOR（保持绿）→ 下一个`；
  每轮开始声明 "Using test-driven-development to ..."；每阶段按 §67 格式汇报并附真实测试输出；禁止先写实现再补测试（写了就删掉重来）
- **与项目规范的关系：** `BADMINTON_ROBOT.md` §65 已要求 TDD，本决策把它升级为对所有代码改动的默认流程
- **优点：** 每个行为都有失败证据；重构有测试护栏；避免"看起来完成"
- **缺点：** 单轮耗时增加（小步多次跑测试）
- **未来是否允许修改：** 仅用户显式指示可豁免（例如明确说"这次是原型/一次性脚本"）
## DEC-012 模块化整体架构的实现形态：注册表 + 管线 + 强制边界（不是一堆 if）
- **日期：** 2026-09-13
- **背景：** 需按 ROBOT_BRAIN.md（S11/S12/S15）把"模块化整体架构"变成可运行、可替换、可验收的代码
- **最终决定：**
  1) 层与消息先用**显式契约类型**固定（`src/badminton_brain/types.py`），全部 Court Frame、批量、带仿真时间戳，且**不含 env_origin**
  2) 层职责用**接口类**表达（8 个），模块只能返回自己那一层的消息类型，违者 `BrainBoundaryError`
  3) 装配用 `ModuleRegistry`（一层一模块、重复/层错拒绝、`replace()` 替换实现），运行用 `BrainPipeline`（按 `BASELINE_ORDER` 顺序、逐步计时）
  4) **Safety 是执行层的唯一上游**：缺 Safety 层时管线构造即失败（PPO/规划无法绕过）
  5) **频率不硬编码**：`PipelineConfig.stage_rate_hz` 每层默认 `REQUIRES_MEASUREMENT`，待接口/实时性测试确定
  6) `validate_architecture(mode='final')` 对"未实现算法/未定频率"直接判失败
  7) `AssetStatus/Param` 收敛到 `src/common/status.py` 单一来源（robot 仿真层再导出）
- **原因：** 满足"结构正式、未知显式、资产可替换、接口稳定、批量原生"；避免算法逻辑散落在场景代码里
- **优点：** 换实现不动上层；边界错误立刻暴露；sim/real 共用同一套契约
- **缺点：** 多一层抽象；消息字段在 MODULE_INTERFACES.md 到位后可能需微调（ISSUE-006）
- **未来是否允许修改：** 允许，但必须同时更新 ROBOT_BRAIN.md / MODULE_INTERFACES.md / 本文件（文档 S15 要求）
## DEC-013 RobotSensorState 增加可选 odom/IMU 通道（契约扩展，向后兼容）
- **日期：** 2026-09-13
- **背景：** 估计层（T2 EKF）需要里程计与 IMU，但冻结的 `RobotSensorState` 只有 base_pose / joint_pos / joint_vel → 估计模块无法接入 `EstimationModule`
- **最终决定：** 在 `src/badminton_brain/types.py` 的 `RobotSensorState` 增加**可选**字段 `odom_twist (N,3)` 与 `imu_yaw_rate (N,)`（默认 None；提供时按批量形状与非有限值校验）；既有调用方无需改动
- **语义：** `odom_twist` = 本步里程计增量通道（vx, vy, wz，机体系），`imu_yaw_rate` = 本步偏航角速率；由 `estimation/estimator.py` 按调用间隔消费
- **配套：** 新增 `estimation/estimator.py`（EkfEstimatorModule）作为唯一桥接点，把 T2 的批量 EKF 与 T3 的 UKF bridge 接进接口
- **验证：** `tests/architecture/test_sensor_channels.py`（6 项）+ 既有契约测试 `test_brain_types`(10)/`test_brain_pipeline`(15) 全绿，证明向后兼容
- **优点：** 估计层可插拔；契约保持 court frame 与批量形状纪律
- **缺点：** 契约文件出现一次受控变更（需同时更新本文件与 plan ledger）
- **未来是否允许修改：** 允许，但任何契约变更都必须有 DEC 条目 + 契约测试证据
## DEC-014 base_twist 坐标系：脑内 court frame，唯 base_twist 是机体系（由规划器旋转）
- **日期：** 2026-09-13
- **背景：** T7 评审实测：把 court 系速度直送机体系四舵轮 IK，yaw=90° 时方向错 90°、轮速偏差最大 23%（5.91 vs 7.70 rad/s）
- **最终决定：** `WholeBodyTarget.base_twist` 与 `SafeCommand.base_twist` 定义为 **robot_base（机体系）** `[vx_body, vy_body, wz]`；court→body 旋转由**规划器**完成（它持有 UnifiedState 的 yaw）；执行层直接送入机体系 IK。其余所有消息字段仍为 Court Frame
- **原因：** 执行层只收到 SafeCommand，拿不到位姿，无法自行旋转；若改为 court 系需扩契约或让执行层缓存状态
- **验证：** T7 增加 yaw=0/30/45/90/135/180° 逐位断言（atol 1e-9）；T9 增加同名回归护栏（机体系直传 vs court 系直传必须得到不同轮速）
- **优点：** 契约不变、执行层无隐式状态
- **缺点：** 一个字段的坐标系与消息其余字段不同，必须靠文档与测试护栏维持
- **未来是否允许修改：** 允许，但必须同时改 T7/T9 两个模块与双方的护栏测试
## DEC-015 Feedback 残差语义拆分：prediction_error（预测残差）vs tracking_residual（执行残差）
- **日期：** 2026-09-13
- **背景：** T10 评审实测：执行适配器把「底盘 twist 跟踪残差（m/s）」写进 `Feedback.prediction_error`，而自适应层按「羽毛球位置误差（米）」消费 → 喂 5.0 即把三参数全部推到步长上限
- **最终决定：** 1) `Feedback.prediction_error` = **预测残差**（预测的羽毛球位置/速度 − 实测，单位米 / 米每秒），由估计/预测侧或应用层提供，**执行层不得伪造**；2) 新增**可选**字段 `Feedback.tracking_residual (N,)` = 执行跟踪残差（底盘 twist/关节可实现性），由执行层填写
- **配套：** T9 已改写入 `tracking_residual` 且 `prediction_error=None`（30 项测试）；T10 改为「应用层推入预测轨迹 `set_prediction()` + 步间 dt」自行计算残差，若 `prediction_error` 非 None 则优先采用；契约测试的假执行模块同步改为新语义
- **验证：** `test_brain_types`(10)/`test_brain_pipeline`(15)/`test_execution_adapter`(30) 全绿
- **优点：** 两类残差不再互相污染；自适应增益语义恢复
- **缺点：** 闭环需要应用层承担一次推入调用（与 SafetyContext 同一模式，已文档化）

## DEC-016 时间基准：消息自带时间戳 + 应用层推入运行时钟
- **日期：** 2026-09-13
- **背景：** 两处评审发现同一根因——冻结管线不提供「当前时间」：Safety 的 `now` 缺省 None（急停/超时不可达）、T10 用同一步内 dt（恒 0）故永不更新、T5 门无法判定陈旧状态
- **最终决定：** 1) 所有消息继续携带仿真时间戳（S43）；2) **应用层负责把运行时钟推入需要的模块**：`SafetyShield.set_context(SafetyContext(now=...))`、`FeasibilityDecisionModule.set_now(now)/now_provider=`；3) 自适应层用**步间 dt**（上一拍状态时间戳 → 本拍），不用同一步内 dt
- **验证：** T8 新增 `set_context`/`context_snapshot`（38 项绿）；决策适配器 `set_now` 使 30 s 陈旧状态被判定为 STALE（7 项绿）；T10 待其修复回归
- **优点：** 不改冻结契约即可让时钟语义正确；推入点集中在应用层
- **缺点：** 运行时循环必须显式推入（否则退化为消息时间戳，已在 docstring 警示）
## DEC-017 单一真源跨层引用：src/ 允许只读引用 simulation/ 的规范实现
- **日期：** 2026-09-13
- **背景：** T7/T9 实现时发现任务书写的 `src/badminton_brain/morph_one/kinematics.py` 不存在；四舵轮运动学的规范实现（已实现已测）在 `simulation/robots/badminton_robot/morph_one/kinematics.py`。同时 T9 复用 `DriveMode/WheelId/PiperCfg.joint_names` 单一真源
- **最终决定：** 允许 `src/` **只读**引用 `simulation/` 的规范实现与枚举（通过惰性加载 helper，不改写、不复制）；禁止在 `src/` 下另建同名副本。理由是复制会造成两个真源（正是本项目反复出现的问题）
- **验证：** T7/T9 均断言 `adapter.kinematics is <simulation 模块>`（同一对象），并有回归测试防止漂移
- **优点：** 单一真源；无重复公式
- **缺点：** `src/` 出现对 `simulation` 的路径依赖（已文档化，且无 Isaac 依赖、已实测）
- **未来是否允许修改：** 若将来把运动学整体迁入 `src/`，须一次性迁移 + 全量回归，不允许长期并存两份

## DEC-018 零参构造与 TEMP 默认几何：开发可用、final 必须拒绝
- **日期：** 2026-09-13
- **背景：** T9/T7/T8 的模块需要零参构造（`Module(N)`）才能被 `full_brain` 装配，但真实几何/限值未实测
- **最终决定：** 允许模块用 `TEMP_PARAMETERIZED_PROXY` 默认值零参构造（并在构造时发出 RuntimeWarning + 把 status 写进参数溯源）；**`validate_architecture(mode='final')` 与各模块的 `unresolved_limits()/measurement_requirements()` 必须把这些项报为未解决**；禁止把 TEMP 值标成 VERIFIED_*
- **验证：** `test_full_brain`（9 项）在 development 模式全绿；`test_final_mode_refuses_unresolved_stage_rates` 与 PPO 占位拒绝用例通过；T8 的 `SafetyShield()` 默认即拒绝启动（REQUIRES_MEASUREMENT），显式 `temp_proxy()` 才可运行
- **优点：** 现在就能开发与集成；上线前不会被 TEMP 蒙混过关
- **缺点：** 运行时会有 TEMP 警告噪声（有意保留，避免"静默使用未实测值"）
## DEC-019 PredictedTrajectory.landed_within_horizon 提升为契约字段；final 校验必须查未实测参数
- **日期：** 2026-09-13
- **背景：** broad 评审 D2：T5 把「地平线截断投影」当真实落点判 OUTSIDE_RESPONSIBILITY（实测 last_z=1.16/1.39 m 却报落对方半场）；D6：validate_architecture(final) 对未实测参数完全无感知（把频率填成已解析后，100% 跑在 TEMP 代理上的脑仍判 ok=True）
- **最终决定：**
  1) `PredictedTrajectory.landed_within_horizon` 由预测器的私有附加属性**提升为契约可选字段**（(N,) 逐 env 标量，可空），因为「球在地平线内是否真的落地」是消费者必须能看到的语义；
  2) `validate_architecture` 在两种模式下都查询每个模块的 `measurement_requirements()` 与 `unresolved_limits()`：development 计为 warning，**final 计为 error**；模块的收集器抛异常也算 error（不允许用它隐藏问题）。
- **验证：** 契约测试 10+15 绿；把频率填成已解析后 `final ok=False`，错误逐层列出（planning 的 contact_jacobian/max_horizon_s/max_joint_offset_rad/max_joint_rate_rad_s…）
- **优点：** 上线门禁真正咬得住未实测项；消费者不会把截断点当落点
- **缺点：** 契约多一个可选字段；final 模式的报错数量变多（有意）

## DEC-020 T11 运行时与测试纪律：推入式通道 + 变异可杀断言
- **日期：** 2026-09-13
- **背景：** broad 评审 D3/D8：T11「canonical 端到端」前 11 步指令恒为 0（决策不可行→规划保持），因此 estop 对比退化为「零 vs 零」；两个致命变异（set_context 置空、process 全零 twist）都能通过；另有测试无断言
- **最终决定：**
  1) T11 的 estop 测试必须用**会动的桩规划器**（registry.replace(AlwaysMovePlanner())）建立非零基线，并断言基线非零 + 急停 env 归零 + 非急停 env 与基线逐位相同；
  2) 所有 T11 测试必须含真实断言（reset 传播测试改为用 spy 断言「8 层都收到且只收到请求的 env_ids」）；
  3) 推入式通道（prediction / SafetyContext / clock）由 `FullBrainRuntime` 统一负责，测试必须覆盖。
- **验证：** test_full_brain 11/11 绿；运行时实测 prediction_available=[True,True]；reset spy 断言通过
- **优点：** 端到端测试不再是空转；变异不再能溜过
- **缺点：** 测试更长（桩规划器 + spy）
## DEC-021 决策层两级模型的权威关系：T5 为快速预筛、T6 为精确可行集
- **日期：** 2026-09-13
- **背景：** T6 报告指出两者口径不同：T5 忽略底盘 yaw、用「任意合法站位」的时间模型；T6 用估计 yaw 旋转工作盒并加底盘三角/梯形最小时间门（更严格）；T5 评审也发现底盘最小时间模型曾被写两份（已收敛为 `decision/travel_model.py`）
- **最终决定：**
  1) `travel_model`（底盘最小时间）**单一真源**，T5/T6 都必须 import（已完成）；
  2) **T6 的可行集为权威**（含 yaw 感知工作盒 + 时间门 + 姿态可行性）；**T5 允许更宽松**，定位为廉价预筛：预筛通过 ≠ 可打，最终以 T6 为准；
  3) 两级都不允许把「地平线截断点」当真实落点（DEC-019 的 `landed_within_horizon` 必须消费）。
- **优点：** 保留快速否决（省算力）同时不产生假阳性最终结论
- **缺点：** 两个判据集的阈值需各自维护（已各自申报 `measurement_requirements()`）

## DEC-022 reason 词表暂不跨模块合并（记为待办 ISSUE-007）
- **日期：** 2026-09-13
- **背景：** T5 有 `HitReason`、T6 有 `InterceptReason`，两者同源（04_HIT_DECISION.md S14）但独立定义，T6 刻意不跨模块 import 以避免并行冲突
- **最终决定：** 现阶段保留两份（各自 `str` 枚举、语义一致、词表已在文档与测试中固定）；T12 之后若仍需统一，抽出 `decision/reasons.py` 作为单一真源并让两侧改为 re-export
- **风险：** 词表漂移（两侧新增词条时可能不一致）—— 已登记为 ISSUE-007，要求在 T12 复核时核对词表交集
## DEC-024 自适应层 delay 的语义与符号（T10 评审修复中暴露并修正）
- **日期：** 2026-09-13
- **背景：** T10 把 delay 的收敛测试从「用实现自己的灵敏度模型自证」改为「用真实 RK4 滚筒生成残差」后立刻暴露符号错误：原实现用 `gradient = -velocity`，会让估计**背离真值并撞到边界**；旧的自证型测试恰好与错误符号互相印证，所以一直通过
- **最终决定（协调者确认 T10 的语义）：** `delay_s` = **进入预测器的状态比当前时刻旧多少**（正值 = 状态陈旧）。状态越旧，前推到「现在」后位置越沿速度方向前进，因此 **d(residual)/d(delay) = +v**，实现取 `gradient = +velocity`。不动点即真实延迟
- **验证（两重独立）：** ①对真实 RK4 求数值导数，`d(rollout position)/d(duration)` 与滚筒末端速度一致（rtol 2%）且与初速度点积 > 0；②闭环收敛：真值 0.02 s，残差由 RK4 生成 → 2.000e-02 / 20 样本 3.31e-06 / 60 样本 3.46e-14，全程远离边界
- **约束：** 若将来 T4 的预测器新增显式 `state_delay_s` 参数且其含义改为「缩短滚筒」，则符号与语义必须**在同一处一并反向**，并在本文件新增 DEC；禁止两层各持一种约定
- **现状：** T4 目前无 delay 参数（从给定状态积分），故当前约定自洽；T10 已在模块 docstring 写明该定义
## DEC-025 球速上限判据采用「窗首样本（来球速度）」变体，而非 S20 的字面形式
- **日期：** 2026-09-13
- **背景：** 04_HIT_DECISION.md S20 字面为「若所有未来可用时刻 s(t) > s_max 才否决」；T5 实现为「首样本（来球速度）> s_max 即否决」，并说明：若按窗内峰值判，自由落体末速会把低速高吊误判为过快（已实测该陷阱）；上下限用同一变量以保持自洽
- **最终决定：** **采纳 T5 的变体**（更保守：对来球本身已过快的球更早否决），并在模块 docstring 与枚举注释中写明这是对 S20 的保守化偏离，而非疏漏
- **理由：** 决策层的假阳性（判定可打但实际打不到）代价远高于假阴性（放弃一个本来可打的球，安全且可复盘）；自由落体末速陷阱是实测过的真实现象
- **代价：** 与文档字面不一致，需在文档侧补一句注解（已登记为 ISSUE-011 的一部分）
- **未来是否允许修改：** 若后续实验证明该变体误杀率过高（用真实回放统计），可改回字面形式并同步调整 too-slow 用例
## DEC-026 传感器事件一律不得击穿闭环（D4 原则推广到辅助几何函数）
- **日期：** 2026-09-13
- **背景：** broad 评审 D4 指出整帧出视场时 `synthetic_detector.project()` 抛异常，直接击穿 `BrainPipeline.step`；T1 修复后（36/36 绿 + 端到端 `PIPELINE SURVIVED OUT-OF-VIEW: True`）又报告同类残留：`subpixel_centroid` 对全暗 patch 仍抛错
- **最终决定：** 确立通则 —— **合法的传感器事件（出视场、遮挡、曝光不足、窗口内无目标）必须返回哨兵（NaN + 无效标志），只有真正的非法输入（形状/类型错误、NaN/Inf、参数非法）才允许抛 BrainBoundaryError**；该通则适用于感知层全部几何函数（detector 与 centroid）
- **理由：** 真实相机在遮挡/曝光不足时必然产生空窗口，若走异常路径会让整个闭环 step 崩溃；且 D4 已证明该失效模式真实可达（第 11 步必崩）
- **配套：** `ShuttleMeasurement.valid_mask`（DEC-023）把「本帧无有效测量」语义贯穿到估计层，避免把哨兵值当真测量
- **代价：** 调用方必须检查无效标志（已在契约与测试中固定）
## DEC-027 落点未知时「挂起判据」而非「直接拒球」，并新增 WAIT 语义枚举
- **日期：** 2026-09-13
- **背景：** broad 评审 D2 指出 T5 把预测地平线截断投影当真实落点判出界（实测 last_z=1.18 m 却报「落在对方半场」）。T5 修复后提出两项判据取舍请协调者确认
- **最终决定（确认 T5 的两项取舍）：**
  1) **落点未知时挂起 OUT_OF_BOUNDS / OUTSIDE_RESPONSIBILITY**（不判、不误杀），继续用方向/速度/工作空间+时间判据；只有当**不存在任何可用样本**时才返回新增稳定枚举 `NO_LANDING_IN_HORIZON`
  2) 新增 `NO_LANDING_IN_HORIZON`（S14 词表外第二处新增，已在 `HitReason` docstring 声明）：语义对应文档 S10 的 **WAIT**——信息不足、后续预测可能变好。不复用 `NO_TIME_MARGIN`，避免把「信息不足」与「来不及」混为一谈
- **落点可信判定：** 优先读契约字段 `landed_within_horizon`（DEC-019，有限即权威，`>0` 为真落地）；标志缺失或被污染成 NaN 时**回退到样本自证**（只有真正到达 z=0 才认为有落点）——延续 C1 的「消息可变、不可信」原则
- **理由：** 截断不等于不可打（球可能正穿过球拍盒）；决策层的假阴性（放弃本可打的球）在信息不足时是可以接受的代价，但**用伪造的理由拒绝**是不可接受的（会掩盖真正的模型缺陷）
- **验证：** 用真实 T4 预测器端到端复现：horizon 0.30/0.545 s（last_z=1.798/1.181 m）→ `NO_LANDING_IN_HORIZON`（修前为 `OUTSIDE_RESPONSIBILITY`）；真实落地出界仍判 `OUT_OF_BOUNDS`；41 项测试绿
- **观察（转 T4 参考，非缺陷）：** 当地平线恰好等于落地时刻时，样本网格末点 z=0.180 m > 0，预测器按自身契约报 `landed_within_horizon=False`；门因此更保守（只少拒、不多拒）
## DEC-028 引入 YOLO 作为感知层粗检测器（仅下载归档，未接入、未安装）
- **日期：** 2026-09-13
- **背景：** 用户指示「下载 yolo」。架构文档 `docs/architecture/01_PERCEPTION.md` 第 22–26 节明确规定 YOLO 的角色是**粗检测**（输出 coarse bbox + confidence + class，回答「羽毛球大概在哪」），并要求显式还原 letterbox/scale 到全幅整流图坐标；**不得**把 bbox 中心当精确球心
- **与旧冻结规格的关系：** 早先 Scene v0.1 冻结规格曾把「YOLO」列入当期禁止项——那是对**场景资产阶段**的限制；本次用户指示 + 架构文档共同构成 Robot Brain 阶段的有效依据（指令优先级高于规格）
- **本次执行：** 仅**下载并归档**，未安装进任何运行环境、未修改任何代码路径：
  - 权重：YOLO11n、YOLOv8n（官方 v8.3.0 发布资产）→ `assets/external/_staging/F_yolo/weights/`
  - 包：`ultralytics-8.4.150-py3-none-any.whl`（`--no-deps`）→ 同目录 `wheels/`
  - 完整性与来源记录：`assets/THIRD_PARTY_ASSETS_YOLO.md`（含 sha256/md5、CRC 全条目校验、来源 URL）
- **许可提示（需项目决策）：** Ultralytics 代码与官方权重为 **AGPL-3.0**（网络传染性；闭源商用需企业许可）。宽松替代：YOLOX / RT-DETR / NanoDet-Plus（Apache-2.0）。
- **关键限制：** COCO 预训练权重**不含羽毛球类**，只能作为微调起点或弱代理；要真正检测必须先做 整流/同步 → 数据采集与标注 → 微调 → 验证。任何在微调前的检测精度声明都无效。
- **环境策略：** 未安装到 `env_isaaclab`（保持 Isaac Lab 环境稳定）；建议后续用独立 `env_vision`（`--system-site-packages` 复用 torch）。
- **接口约束：** 真实检测器只能通过 `StereoPerceptionModule(measure_fn=...)` 注入，保持层边界（`ROBOT_BRAIN.md` S12）。
### DEC-028 补充（2026-09-13，用户确认模型版本）
- 指定模型为 **YOLO26s**：`weights/yolo26s.pt`（20,422,725 B，v8.4.0 发布资产，sha256 646f8bc3…84a1b）
- 校验证据：字节数与 GitHub API 声明值逐字节一致；本地与远端 sha256 相同；zip 718 条目全 CRC 通过
- 早期下载的 yolo11n.pt / yolov8n.pt 保留为备选（可删，约 12 MB）
- `yolo26s.onnx` **未取得**（远端超时 + 本地连接重置）→ 部署路径稍后重试或由 ultralytics 自行导出
- 其余结论（AGPL 许可、COCO 无羽毛球类需微调、未安装未接入、注入点约束）不变