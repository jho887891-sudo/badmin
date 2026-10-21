# Morph One × B601-RS × D455 × 羽毛球：全项目连续执行总任务（P0–P8）

**执行版**：v2.1 · 2026-10-21 · **CHECKPOINT ≠ STOP**  
**总任务**：[Issue #13](https://github.com/jho887891-sudo/badmin/issues/13)  
**原总 SPEC**：[EXECUTION_SPEC_P0_P8.md](https://github.com/jho887891-sudo/badmin/blob/chatgpt/full-execution-spec-p2-20261021/management/morph_one_rs/EXECUTION_SPEC_P0_P8.md)  
**轮地接触专题**：[P2-WC SPEC](https://github.com/jho887891-sudo/badmin/blob/chatgpt/p2-wheel-contact-spec-20261021/management/morph_one_rs/PHASE_P2_WHEEL_CONTACT_ACCEPTANCE_SPEC.md)  
**目前证据基准**：PR #20、\`dev/p2-wheel-contact-g1g4@6d8fe70981df4fca974ac33f647e8840d9f0eec5\`。后续执行前**刷新所有 HEAD**，不把基准 SHA 误当最新事实。  
**目标**：连续推进完整仿真机器人闭环，而不是分批提交 3–5 个 Gate 就退出；所有任务均以独立、可复现证据验收。  
**本文件是任务，不是代码或 GPU 已完成的证明。**

## 0. 委派范围、决策规则、持续执行

### 0.1 授权
在代理**已经拥有**的 GitHub / 服务器 / Isaac Sim / GPU 权限范围内，自主创建隔离分支和 worktree、修改代码、编写测试、运行可控实验、修普通 Bug、重构模块、产生数据集分析、开 PR、上传日志和原始证据，**无需每次征求用户确认**。这不会赋予新的实际服务器/系统权限。

严格禁止：force-push 已公开协作分支、覆盖历史失败数据、修改只读厂商资产、自动合并到 main/default/保护分支、未经新授权启动 PPO 训练或真实机器人运动、绕过物理/安全门禁。GPU Isaac 进程始终串行，独立 CPU 工作可并行。

### 0.2 CHECKPOINT ≠ STOP
每项提交可以写 DONE，但其语义只是“该 task 完成”。交付 commit/PR/Issue 评论后**立即查队列并领取下一 READY 工作**；不能因为“G5 跑完”“PR 创建成功”“收到 FAIL”“需要重写工具”就结束。普通错误应自动调试，完成一条工作线时自动转向另一条 READY。无法跨会话后台续跑时，只存检查点并说明真实停止理由，不声称有后台执行。

### 0.3 动态调度，不一次性并行占满 GPU
每轮扫描所有任务并自动计算：
- \`READY\`：依赖已满足，资源权限足够，立即安排；
- \`RUNNING\`：持续调查，不能因 GitHub 上写一次进度就转 DONE；
- \`DONE\`：代码+测试+必要实测与证据均通过，不是“写了脚本”；
- \`PARTIAL\`：部分证明而非最终门禁；
- \`FAILED\`：测到的结果不合格，但仍可调试；
- \`BLOCKED\`：依赖/客观环境缺失，说明精确解锁条件；
- \`UNKNOWN\`：还没有可信测量。

对所有队列执行 \`refresh latest refs → assign task → implement → test → audit → preserve raw evidence → push independent branch → next READY task\`。在 \`management/morph_one_rs/EXECUTION_QUEUE_STATE.json\` 或等效跟踪文件登记 \`task_id,status,dependency,owner,head_sha,artifact,latest_run_id,next_action,blocker\`。默认按解锁条件派发，不要求用户手工传递两个 AI 的聊天记录。

### 0.4 真实退出条件
只有 (a) 所有当前**合法可执行**任务完成，(b) 当前平台实际执行窗口/工具额度到达限制且提供具体证据，(c) 无其他 READY 而存在明确外部依赖或安全授权缺口，才允许结束本次会话。不能把“计划已写好”当客观阻断。

## 1. 首先冻结已有成果，不要浪费运行时间重复成功场景

已在 PR #20 的 \`6d8fe70\` 回读到：
- **G0**：历史 USD double-lift 离线复算，曾错误将轮子悬空而认为穿地。
- **G1** 正间隙零重力 PASS（2 次）；**G2** 恰接触零重力 PASS（当前仅 \`G2_r01\`）；**G3** 穿地负控制 EXPECTED FAIL，有 64.08 N 响应（2 次）；**G4** 单 Cube 落地传感器正控制 PASS（2 次，峰值约 200.9 N）。
- **G5 已实际运行两次**：重力下自由落地，峰值约 2693.5 N；\`raw_poses.jsonl\` 末期 chassis z 约 0.334m，接触力约 **196.2 N**（对应当前 20kg TEST_PROXY 的 mg），后 30 步基本稳定。结果 JSON 仍 FAIL 因误用了**零重力静置**的 t0/相对初值漂移阈值；而 manifest 错写 \`gravity_m_s2=[0,0,0]\`，命令为 \`--gravity-z -9.81\`。**禁止把“G5 原始静置 Gate FAIL”偷改 PASS；应新增专用落地/承重 Gate，与旧 raw 结果并存并重跑含正确 metadata 的新实验。**
- **G7_r01 已运行**：0g、10 rigid/8 DOF、真实 arm 资产 SHA 不变、旧挂载姿态/静置 PASS；**但还没证明真实臂与修正后的四轮 compound collider 在同一场景里于重力下稳定承载。**
- P0 独立分支有 18 项 runner 回归、曾发现 stale result 假 PASS 和 timeout 死参数；P1 确认轮候选 STEP BREP 顶点包围盒与网格 AABB 差异，外圆半径约 103mm 但**有效滚动半径/轴心/轮距未冻结**；P5 静态审计发现 PiPER reset 的 joint3=-1.5rad 对 B601-RS [0,3.14]rad 越限。
- **先读这些现有文件和 sha，局部已 PASS 的实验无需无理由重跑**；以最新源码和实际内容为准。

## 2. 96 个可执行任务槽位：从工程底座到低速闭环

这是一个**任务池而不是 90 次 GPU 启动承诺**。同类任务可在同一 PR/一次可靠运行里合并验收。任务号是稳定 ID，GPU 任务按依赖进入 READY；CPU 任务立即并行。任何新发现 Bug 产生 \`FIX-* \`子任务而不是只写下一轮计划。

### A. 实验平台与数据治理（A01–A10，CPU 优先，可立即开始）
- [ ] **A01**：全仓库 P0–P8 实际文件/分支/PR/HEAD/保护范围清点，输出机器可读工程索引，不假设 main 是默认分支。
- [ ] **A02**：实现动态队列状态文件 schema、依赖拓扑、READY 自动选择器；针对循环依赖与 BLOCKED 传染做离线单测。
- [ ] **A03**：每次 commit/PR/Issue 状态一致性检查，防止同一任务并发重复写入或无证据 DONE。
- [ ] **A04**：GPU 独占锁 + PID 验证 + 退出后 Kit 子孙进程清理审计；不得误杀其他用户进程。
- [ ] **A05**：实验 YAML/JSON 配置、参数 hash、环境指纹、资产 SHA、实际应用参数落地，任何配置失配 FAIL。
- [ ] **A06**：唯一 RUN_ID、不可变原始证据、manifest/summary SHA256、任意旧 JSON 残留 fail-closed。
- [ ] **A07**：跨代理 PR review 约定：隔离 worktree、可 cherry-pick、小 diff、冲突解决、回归测试和 reviewer 签字。
- [ ] **A08**：离线 CI 矩阵：数值/物理几何/文件契约/坏数据注入；GPU 自动化仅在授权环境串行触发。
- [ ] **A09**：每项 gate 的断言来源可追踪：源码行、SPEC 条款、单位、数据文件、验收裁定。
- [ ] **A10**：一键复现实验命令及 README，验证“纯 CPU 检查”不意外启动 Kit。

### P0. 可信运行、退出码与记录（P0-01–P0-09，READY）
- [ ] **P0-01**：合入/复用此前 18 项 runner 回归的**有效补丁**到独立集成分支，不盲合其他实验分支。
- [ ] **P0-02**：child 0 + JSON FAIL、child nonzero + JSON PASS、missing/invalid/NaN/conflict/empty gates 的全部 fail-closed。
- [ ] **P0-03**：\`--timeout\` 真正杀死超时实验；检查 Kit 子孙进程是否残留和 GPU 资源泄漏，记录实际 PID。
- [ ] **P0-04**：重复 out-dir 明确拒绝/独占创建；保留此前“重命名 stale JSON”的历史兼容策略，但新实验不得依赖它形成假 PASS。
- [ ] **P0-05**：报告 \`manifest.gravity\` vs 命令实际 \`gravity-z\` 冲突、steps/dt/config/source hashes 不一致时 INVALID。
- [ ] **P0-06**：JSONL 逐帧完整性、排序、NaN/Inf、四元数顺序、坐标单位、接触 unavailable 与 0N 的区别。
- [ ] **P0-07**：在真实 Kit 上做故障注入：Python 抛异常子进程 0、真正 PASS、真正 FAIL、timeout 和历史目录防串用。
- [ ] **P0-08**：P0 工具 PR #14/#16/#17 与 DeepSeek runner 的功能对照，保留一套明确验收接口并测试 backward compatibility。
- [ ] **P0-09**：生成可重跑的 PASS/FAIL/UNKNOWN 看板，不能自动把全局项目状态归为 PASS。

### P1. CAD 真实轮件身份与变换（P1-01–P1-09，CPU READY）
- [ ] **P1-01**：冻结厂商 STEP、USD、mapping.csv/JSON 的 SHA、metersPerUnit 和坐标转换、原始文件只读。
- [ ] **P1-02**：重演 205.991 vs 180.782 mm 的 BREP 顶点盒/曲面包络/三角网格 AABB 区别，固定三种口径。
- [ ] **P1-03**：四轮 eid→USD prim mapping 的实体逐项对照，检查重复 mapping、axis 旋转和左右镜像。
- [ ] **P1-04**：从 CAD 轴面/圆柱面/网格法线辨认轮子旋转轴、真正轮中心（chassis frame），不是只看顶点盒。
- [ ] **P1-05**：圆柱 \`r=103mm\` 是滚动面还是挡边/法兰的分类证据；未证实继续 UNKNOWN，不擅自称轮径 206mm。
- [ ] **P1-06**：四轮轴心间距/前后轮距/轮序/正转方向，自洽的 CAD/world transform 表和可视化检查。
- [ ] **P1-07**：质量/惯量/材料/接触形状从可用来源核实；厂家未知值保留 UNKNOWN，必要时 proxy 明确标注。
- [ ] **P1-08**：真实外圆几何 vs 目前 90mm 半高代理盒差异量化，重算真实接地高度；不能把旧 \`lift=.334\` 当官方值。
- [ ] **P1-09**：输出 \`wheel_identity_contract.json\`、按实体的 hash、来源、质量、轴心及不确定性，对 P4 提供 gate。

### P2. 装配、碰撞、承重收官（P2-01–P2-10，GPU 串行）
- [ ] **P2-01**：回归 G0–G4 原始证据，补 \`G2_r02\`，纠正“每组已重复两次”的过度表述。
- [ ] **P2-02**：修 \`morph_one_p2_real_chassis.py\` manifest 的实际重力/命令/配置一致性；对旧 G5 追加**独立审计结果**，不覆盖旧 JSON。
- [ ] **P2-03**：设计专门的 G5 **重力落地验收**：正常无初始穿透、观察下落与接触，稳定窗口统计 z、vz、az、接触力；不拿零重力静置最大漂移阈值判自由落体。
- [ ] **P2-04**：新 G5 两次独立运行，至少 120 步（必要时更长直至有充分稳态窗口），输出真实 landing timeline、min ground gap、姿态 roll/pitch、steady-state contact/weight 支撑对照。稳态公差在运行前基于 proxy/数值精度预注册，不能看结果调阈值。
- [ ] **P2-05**：G6 **只改一个因素**做 collision-shape 对照：Compound Cube ↔ 合法的 Mesh ConvexHull/独立形状；不因 CPU/GPU 误用导致假错误。
- [ ] **P2-06**：分别独立验证 USD authored world bbox、runtime shape 激活、ContactSensor、solver depenetration，传感器 0N 不独立证明有效或无效。
- [ ] **P2-07**：G7 上一轮真实 B601-RS 0g 10刚体/8DOF PASS 复用；另做 **真实 arm + 修正四轮 compound 在同一 PhysX 场景**，检查是否碰撞、是否有双重质量/错误刚体嵌套。
- [ ] **P2-08**：整机合法静置 g=0 和重力支持场景分开 Gate；真实臂/底盘相对安装位置与角度分别 ≤1mm/0.001rad，绝不伪称自由落体世界位姿不变。
- [ ] **P2-09**：各实验 2 独立 run_id、raw full steps、outer exit 正负例、资产 hash、日志，另一个代理用独立验收器审计。
- [ ] **P2-10**：一份 **P2 最终证据矩阵**：真实/代理、静置/落地、源资产/派生资产、PASS/FAIL/PARTIAL 与解除依赖条件；达门禁后自动解锁 P3 READY。

### P3. 底盘运动过程中的机械臂安装稳定性（P3-01–P3-08，依赖 P2）
- [ ] **P3-01**：冻结底盘 chassis/arm base 6DoF frame tree、Joint anchor、相对姿态误差与非平凡 yaw/roll 参考。
- [ ] **P3-02**：纯 X/Y 非零平移、独立 Z 位移、纯 yaw 非零转角，分别验证**实际测量**非零，而不是只有目标命令。
- [ ] **P3-03**：平移+转向复合运动及加减速段，记录 chassis/arm 全帧世界与相对 pose。
- [ ] **P3-04**：可控运动施加手段合法、明确标为诊断驱动；**这种驱动不能冒充 P4 真实轮驱动**。
- [ ] **P3-05**：真实臂关节在固定 target 下的位姿稳定、接触/自碰撞、超限检查及 FAIL 注入。
- [ ] **P3-06**：相对安装 6DoF 最大误差、峰值速度/角速度以及 dynamic jerk 指标；执行前从原协议冻结阈值，非见数调参。
- [ ] **P3-07**：不同 initial yaw/arm pose、至少两次独立 run，证明不是单个位姿硬编码修正。
- [ ] **P3-08**：P3 PASS/FAIL 原始录像/轨迹、复现命令与审计输出，达标后让 P4 由 BLOCKED→READY。

### P4. 四轮真实运动与控制（P4-01–P4-11，依赖 P1+P2+P3）
- [ ] **P4-01**：从 P1 wheel identity 建立 wheel prim→axle joint→DOF→motor channel 一一映射，方向/单位/极限自检。
- [ ] **P4-02**：合理质量/惯量/接触摩擦代理，区分物理支持证据和未经厂家授权的估计参数。
- [ ] **P4-03**：轮毂/轮胎与底盘刚体的 articulation graph 有效性、root/motor placement、多环境可解析性。
- [ ] **P4-04**：单轮低速开环旋转，记录关节角、实际轮速与电机目标/扭矩，先无地面后有地面。
- [ ] **P4-05**：四轮同时正反低速旋转、位置与实际角速度同步，方向对照。
- [ ] **P4-06**：平地低速直线前进/后退，\`wheel rpm → chassis displacement\` 由真实物理接触产生。
- [ ] **P4-07**：低速纯转弯、不同转弯半径、轮滑及 yaw rate 的实测闭环。
- [ ] **P4-08**：不同摩擦系数、载荷、地面接触法向变化的单因素对照，不宣称未知轮胎摩擦为真实。
- [ ] **P4-09**：基于 joint/odom/ground truth 的速度控制器、坡道加减速、安全加速度限值（仅仿真）。
- [ ] **P4-10**：里程计与 ground truth 误差、轮滑识别、静止驻停和急停 HOLD（仿真），完整错误路径。
- [ ] **P4-11**：真实驱动不可用时独立代理可继续开发控制接口/单测但不得虚构“轮驱动已完成”；P4 达标自动解锁相关 P6 任务。

### P5. B601-RS 单臂控制、多环境与 reset（P5-01–P5-09，部分 CPU READY）
- [ ] **P5-01**：将单臂关节事实与限位合同、P5 子代理 14 tests 证据整合至可运行测试。
- [ ] **P5-02**：修复 \`simulation/badminton_scene/reset.py\` 的 **PiPER 默认 joint3=-1.5rad** 与 B601-RS \`[0,3.14]rad\` 不一致；按型号/DOF name 定义安全 reset，保留 PiPER 不变行为。
- [ ] **P5-03**：所有 rest pose 和随机 reset 关节值在 USD/URDF/应用合同内，缺有效值 FAIL/HOLD 而非静默 clamp。
- [ ] **P5-04**：joint name→runtime DOF index 显式映射，不使用 USD stage 遍历顺序；角关节 USD gain 的度制/弧度换算验收。
- [ ] **P5-05**：运动命令限位、速度/加速度/扭矩保护，以及 6 轴与夹爪 2 DOF 的独立归属。
- [ ] **P5-06**：单环境 reset 负例：越限、错误名字、NaN、缺关节，确认异常路径不会触发危险目标。
- [ ] **P5-07**：N=2 多环境实例隔离与 reset 随机化；实际支持再测 16/64/128，不能因显存不足硬跑。
- [ ] **P5-08**：多环境吞吐、内存峰值、跨环境 state leakage、复现种子与掉线恢复。
- [ ] **P5-09**：最小可复用 B601-RS controller API（state, joint targets, reset, hold, IK hook）与独立单测。

### P6. 球拍、相机、标定、整机（P6-01–P6-09，部分可离线 READY）
- [ ] **P6-01**：审计球拍 GLB/USD 的单位、碰撞代理、安装法兰/末端 link 几何授权，源资产只读。
- [ ] **P6-02**：球拍→末端→臂→底盘的 frame tree、相对转换、姿态/接触容差和基准配置。
- [ ] **P6-03**：D455 相机真实机型/镜头坐标（optical frame）与机器人 mount frame 区分，未标定外参 UNKNOWN。
- [ ] **P6-04**：相机内参、分辨率、RGB-depth 对齐、帧率、深度单位、畸变参数的版本化元数据读取与校验。
- [ ] **P6-05**：多坐标变换一致性（camera→base→world→racket）和米/毫米、左右手坐标的回归样例。
- [ ] **P6-06**：RGB、Depth、机器人 state 时间戳、延迟/掉帧同步协议；真实硬件采集需相应权限，无硬件用回放测试。
- [ ] **P6-07**：模拟羽毛球 GLB/碰撞体、球拍包围盒和球-拍接触检测最小样例，代理标签不可冒充真实球动力学。
- [ ] **P6-08**：单场景整机模型静态加载、碰撞、重力稳定、所有传感器发布有效和运行时零 NaN。
- [ ] **P6-09**：整机集成 manifest 和多环境能力 gate；完成后自动开放 P7 可控闭环试验。

### P7. 羽毛球检测→跟踪→预测→低速击球（P7-01–P7-13，离线 READY，实体/闭环门禁后解锁）
- [ ] **P7-01**：审核 YOLO 模型/data manifest、训练/验证视频分组防泄漏、标签/负样本、小尺寸 4–6 px 分布。
- [ ] **P7-02**：可复现检测推理基线：置信度、召回、误检、按球像素大小分桶；GPU 不可用则标 NOT RUN。
- [ ] **P7-03**：640/960/1280 以及 crop/tiling 的真实精度-延迟对照，不能只比较训练 loss。
- [ ] **P7-04**：检测器输出统一 \`timestamp, frame_id, bbox, confidence, pixel_covariance\`，接口/单位单测。
- [ ] **P7-05**：EKF/其他时序滤波器状态定义、过程噪声、测量噪声和离群拒绝，支持只见 2D 时保持像面状态。
- [ ] **P7-06**：若有可信深度/双目和标定，才升级三维 \`position/velocity/cov\`；不得把 2D bbox 当真实三维。
- [ ] **P7-07**：0.3–0.8 s 球轨迹预测的统一 horizon、误差/置信区间/观测缺失处理和分场景回放。
- [ ] **P7-08**：视觉测量延迟、球速/小目标噪声的传播，输出预测不确定度与过期帧拒绝。
- [ ] **P7-09**：球拍可达工作空间、机械臂限位、IK、多候选击球时间点、避障/自碰撞。
- [ ] **P7-10**：安全 FSM：SEARCH/TRACK/PREDICT/PLAN/EXECUTE/HOLD；置信度不足、时间超限、不可达时进入 HOLD。
- [ ] **P7-11**：固定静止球、低速确定性来球、低扰动轨迹三种仿真序列，逐级启用真实闭环。
- [ ] **P7-12**：低速击球复现统计：每组 trial 数、检测率、预测位置误差、实际接触率、有效回球率、失败案例。
- [ ] **P7-13**：端到端推理与控制延迟预算、帧同步、异常恢复、冷启动/重复任务一致性回放；不能用离线轨迹 loss 替代击球成功率。

### P8. 训练准备及未来扩展（P8-01–P8-08，准备可做；PPO/真机当前 NOT AUTHORIZED）
- [ ] **P8-01**：仅搭建训练接口设计 \`observation/action/reward/reset/termination\` 和 schema/形状测试，不启动 PPO。
- [ ] **P8-02**：安全 reward 候选与约束指标（距离、碰撞、速度、成功、超时），先离线可计算性验证。
- [ ] **P8-03**：训练/验证/测试随机种子分离、任务 difficulty curriculum、不可泄漏记录。
- [ ] **P8-04**：非 RL 基线（有限状态/可达性/简单跟踪控制）和同口径任务成功率评测协议。
- [ ] **P8-05**：PPO 预算门禁文件：GPU/训练时间上限、安全守卫、可恢复 checkpoint、停止条件和回滚。
- [ ] **P8-06**：算法设计与离线 mock tests；实际 PPO 训练**须用户再次明确授权并满足 P5/P6/P7 门禁**。
- [ ] **P8-07**：真机实验风险清单、急停/Watchdog/人工安全区域/速度限制方案仅写文档，**禁止向实体机器人发送动作**。
- [ ] **P8-08**：未来研究扩展（不冒充已实现）：主动感知、VLA 策略接口/数据记录、感知→计划→动作的失败归因与评测方案。

## 3. 自动领取顺序与拆并行规则

1. **立即 GPU 串行**：P2-02 修配置；P2-03/04 检验 G5 落地后承重（旧轨迹已有接近 196.2N 支持证据，先离线核算）；P2-01 补 G2；P2-05 G6 单因素；P2-07/08 修正四轮+真实臂的联合对照；P2-09/10 最终审计。**当前 G7_r01 只是旧 0g 真实臂静置，不能替代联合 G7**。
2. **立即 CPU 并行**：A01–A10、P0-01–06/08/09、P1-01–09、P5-01–06/09、P6-01–05/07（资料/模拟允许时）、P7-01/04/05/07–10/13（离线允许时）、P8-01–05/07/08（设计/假数据）。
3. **P2 门禁达到时**：解锁 P3 真实非零 6DoF 运动，完成全部 P3 才能将 P4 真轮驱动设 READY；CPU 侧继续完善接口。
4. **P1 轮件身份还 UNKNOWN 时**：P4 可做独立参数化控制与轮驱动 mock tests，但不可认定真实轮半径/轴心已确定、不可运行并宣传完整 P4 PASS。
5. **P4/P5/P6 相应门禁达到时**：解锁 P7 可控低速仿真闭环（不是真机），P7 完成后仍只做 P8 训练准备，待用户新授权。
6. 同一工作线遇阻后，自动选择**其它 READY 项**，不要只把阻断写到“下一轮”。如果 3 个 CPU worktree 并行，请记录任务 owner/文件所有权，避免多个代理在同一个共享目录交错提交。
7. 每项任务只有“确认依赖满足 + 实际代码/配置/实验+正反例/必要数据 + PR + 交叉审计 + gate”才可记 DONE。纯方案写作是 PLAN，不是实现 DONE。

## 4. 不变的实测证据与门禁协议

每个真实实验新目录：\`manifest.json\`, \`result.json\`, \`raw_poses.jsonl\`（逐步），必要的 \`contacts.jsonl\`, 实际 authored USD、\`gate_runner_summary.json\`、child stdout/stderr、notes、完整源/资产 SHA。meta 中 gravity 必须与程序真正应用的 \`--gravity-z\` 和 sim 配置一致，差异直接 INVALID；任何历史 RUN 不覆盖。浮点/四元数原始顺序带来源标签；\`contact_status=unavailable\` 与数值 0 必须分开；所有 Gate 独立布尔断言，缺失/NaN/空 Gate 不予 PASS。

P2 零重力静置：\`t0_world_pos <= 0.001m\`, \`t0_world_rot <=0.001rad\`, 相对安装 \`pos<=0.001m\` / \`rot<=0.001rad\`, 60 步**全程最大**漂移 \`<=0.001m\`。P2 重力落地另立有时间窗和物理支撑的 Gate，**不能用 t0 世界 pose 不变来判自由落体失败**。P3/P4/P6/P7 定量阈值必须**事先**从工程/来源依据预注册，不能根据跑出来的数据事后放宽。

在真正完成 P2 之前，禁止把 P3 真运动、P4 真轮驱动、P6 整机真实动态或 P7 击球闭环标 PASS；可以先做不依赖它的静态/离线代码。任何代理不准以“进度太少”作为理由绕过依赖、安全或事实性证据。

## 5. 汇报要让下一次会话能无缝恢复

每个阶段提交\`workstream_done\`，但\`whole_project_done=false\`除非所有授权目标真的完成。GitHub Issue #13 最终交接必须包括：
- 最新所有分支 SHA 与工作树状态；
- 当前 RUNNING / READY / BLOCKED / FAILED 的 task ID 和优先级；
- 每个新增结果的原始 run ID、真实命令/退出码、具体 Gate、数据来源；
- 是否复用未经合并的 PR #14–#20、base 是否正确；
- 哪些真实资产/参数 UNKNOWN，后续如何取证；
- 结束本会话的**真实**限制（例如实际工具额度/环境报错），不是“我已经完成几个任务了”。

**关键指令**：每次出现 \`task_DONE\` 之后马上查询剩余 READY；只要有可以做的任务且执行窗口未被平台中断，就继续。不因为发布 DONE 评论、PR 或完成 G5/G7 自动返回用户。严禁假装可以离线异步长时间自行运行；若当前会话结束，保存队列由下次触发继续。
