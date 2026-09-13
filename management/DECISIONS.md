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
