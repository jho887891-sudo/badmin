# P2-WC · Morph One 四轮世界几何、PhysX 碰撞与接触传感器联合验收 SPEC

**版本**：v1.0（2026-10-21）  
**项目**：[jho887891-sudo/badmin](https://github.com/jho887891-sudo/badmin)  
**总任务**：[Issue #13](https://github.com/jho887891-sudo/badmin/issues/13)  
**阶段**：P2 的最后一段（真实测量几何的简化底盘轮地碰撞）；允许进行 P3/P4 的静态准备，不得提前声明 P3/P4 通过。  
**执行**：DeepSeek 负责 Isaac Sim GPU 代码、实验及原始证据；ChatGPT 同样可在独立分支直接改代码、做离线数学/源码审计，双方互评，不由用户搬运数据。

## 0. 执行纪律

此文是可操作技术规格，不是已完成实验的结果。代理须在现有代码/服务器权限内自主读取、修改、测试、提交、继续下一个可做事项；一次普通 Python/Kit 错误不算“客观阻断”。不能擅自更改权限系统、修改真实设备、运行 PPO、强制推送、合并受保护分支。若外部 GPU/硬件不可用，要留下明确的异常证据并切换合法并行任务。

所有真实实验严格**单一 GPU Isaac 实例串行运行**，对遗留进程做安全确认；每个 run_id 独立目录、不可覆盖或删除历史 FAIL 数据。每项实验必须有外层子进程 runner、原始 stdout/stderr、child/outer exit、JSON 布尔 gate、commit SHA、环境版本和全步轨迹。在 GPU 上重现不了的结果标 UNKNOWN，不得混同离线 PASS。

## 1. 已确认基准和本轮目标

基于 DeepSeek \`dev/morph-one-p2-quat-and-settle@459cb122f7be68ad60295ee61c4faaf73618b151\` 以及 ChatGPT [Draft PR #18](https://github.com/jho887891-sudo/badmin/pull/18) HEAD \`424c7e3d33776b934f2579a32c1fa445b8220d36\`。发起新实验前查 GitHub 是否已有新提交，若有则以最新源码与证据比对而不是静默覆盖。

历史数据：
- P0 外层 gate 的真正 PASS/FAIL 正负例已有 GPU 报告；TEST_PROXY 和真实 B601-RS 单臂挂载静置在**其他场景**已有 PASS 证据。
- DeepSeek 旧 RC_lift0300/0334/0350 场景给出 t0 位置/姿态近零、60 步漂移 0；但 ContactSensor 记录 0 N，且此前报告的“轮底高度”不正确。
- **历史 USD 的确定性双重 lift**：RC_lift0300_r01 中 \`Chassis.xformOp:translate.z=0.2596\`，其子 \`wheel1.xformOp:translate.z=0.056\`，轮盒半高 0.09，组合世界轮底 \`0.2596+0.056-0.09=+0.2256 m\`；**不是**原先声称的 -0.034 m。RC_lift0334 得 +0.2936 m，RC_lift0350 得 +0.3256 m。这是直接从已提交原始 USDA 复算出来的证据。
- PR #18 已改为**Chassis 只施加一次 lift；轮子继承父刚体变换，局部 CAD 中心不再增加 lift**，并添加 XformCache 世界轮底检查及静置全程最大漂移检查。**这些改动仅经过离线 CI，没有真实 Isaac GPU 复跑。**
- 四个轮状 STEP 实体候选：435700/759491/773650/871375；其“轮身份”应保留候选性质，不能假装官方 CAD 已确认，尤其不把碰撞盒等同真实轮胎模型。
- 当前简化形状固定 \`WHEEL_CENTRES[...].z=-0.244 m\`，盒半高 \`0.090 m\`，预期各轮最低世界高度：
  \`\`\`
  z_world_bottom = lift + wheel_local_z - wheel_half_height
                 = lift - 0.334 m
  \`\`\`
  这是**简化碰撞盒的几何计算结果**，不是未经校验的真实轮胎半径。

本轮要回答并用数据区分这四个问题：
1. USD 中轮子到底在世界坐标系什么位置？
2. PhysX 实际把哪几个 prim 当成有效 collision shape，是否与 USD authored 相符？
3. Collision shape 与地面相交/加载时，是否确实出现求解器响应？
4. ContactSensor 是否正确绑定对应刚体并报告接触？即使没有接触力读数，能否从独立物理观测证明/否定碰撞？

**不得仅因重力为零且 60 步不动，就声称接触有效。**

## 2. 代码任务：必须实际修改并保存可重放证据

### A. 统一坐标与碰撞体授权
从 PR #18 的修复逻辑开始，DeepSeek 应复查/移植小块代码到**自己新的独立工作树**，注意不覆盖其已有分支。建立/打印 \`/RC\`, \`/RC/Chassis\`, \`/RC/Chassis/geometry\`, 四个 \`wheelN\` 及其 \`geometry\` 的：
- \`parent_path\`、\`UsdGeom.XformCache.GetLocalToWorldTransform\`、local transform、USD 世界位置、世界 AABB 及单位；
- \`UsdPhysics.RigidBodyAPI\`、\`UsdPhysics.CollisionAPI\`、可用的 \`PhysxSchema.PhysxCollisionAPI\` / \`UsdPhysics.MeshCollisionAPI\` 应用状态、物理 scene collision group / filter / contact reporting、物理 shape 是否启用；
- 碰撞几何的类型（Cube 还是 Mesh）、尺寸、scale、local transform xformOpOrder、mesh convexHull/convexDecomposition 近似设置和 PhysX cooking 结果（只记录运行时 API 可用的，无法查询时为 UNKNOWN）。

**关键**：对嵌套 wheel 的局部位置禁止重复加 lift；主体 CAD bbox 中心相对 Chassis CAD 原点的偏移应作为 Geometry local transform，确保 Translate/Scale 顺序正确。不得仅看脚本中的 \`lift - 0.334\` 常量算“实际世界高度”；以生成的 USDA、USD 组合矩阵和 runtime collider 的几何范围交叉核验。
- 作者级 assert：4 个轮子各自 \`|z_world_bottom - (lift-0.334)|<=1e-5m\`；若实际 CAD/代理几何变化，必须重新预注册公式/阈值，而不是调参掩盖错误。
- 同时检查底盘主体 geometry 的世界 AABB、底盘 origin 与质量/惯量对应关系；不要因为只看轮底而漏掉主体先穿地。
- 保存源 STEP SHA 或引用被审计的 CAD 测量路径；资产 SHA/源几何不改写。

### B. 双通道的碰撞证明（不要单看 ContactSensor）
1. **正控制（positive control）**：一个已知有效的简单动态 Cube + Ground，无关节，启用 collision/contact reporting，在低速、低风险仿真内给可观测的向下速度或正常重力，必须同时观察到非穿透的状态演化和可读取的接触报告/冲量（如果传感器 API 存在）。如果正控制也无接触读数，先修传感器绑定或改用 PhysX Contact Report API，不要责怪多轮 collider。
2. **负控制（negative control）**：把 Cube 或轮子放在明确正间隙，零重力、零初速度，预期不出现接触诱导的速度/位置变化；不要求任意时刻接触力必须严格 0（注意噪声/初始化）。
3. **轮子穿透控制（penetration diagnostic）**：在已经验证 USD/runtime world collider bbox 正确后，让 \`lift=0.300\` 对应轮底 \`-0.034m\`，启动后若缺少任何 depenetration/接触/速度/位置反应，必须标 \`collision_response_unknown_or_failed\`。穿透场景本身是不合法的生产初始化，不可用于静置 PASS；它只用于验证碰撞系统确实生效。
4. **有效落地接触（loaded contact）**：从 \`lift=0.350\` 约 +16 mm 间隙开始，保持 shape/质量/solver 设置一致，给**诊断场景专用**的小幅负 z 初速度或开启重力（推荐先最小单刚体；随后四轮），采集撞地前后 \`p_z, v_z, a_z\`、接触状态和每轮接触信息。低速，不能把无下落/无驱动场景的一段静止当作真实接触验证。

默认优先使用当前 build 实际存在且受支持的 ContactSensor / PhysX API，记录 API 名称与失败异常。不能虚构物理报告字段。区分 **raw/contact force**、**solver depenetration**、**world pose motion** 是三个不同观测通道。零重力恰好接触的 0 N 不足以判传感器损坏或碰撞失效；初始穿透下 0 N 也必须结合 shape 是否生效、接触采样时机和 solver 响应解释。

若需要更换 collision implementation，依次实验**实际有效且合法**的替代：
- 轮子为子 Xform + CollisionAPI(Cube)，由父刚体管理；
- 在 USD 类型和接口正确前提下使用显式刚体子碰撞体（若物理层级要求 joint 则一并记录，禁止凭空创建不受控动态子刚体）；
- 对真实 Mesh 使用 \`UsdPhysics.MeshCollisionAPI\`、在可用 PhysX API 中使用 ConvexHull 或 ConvexDecomposition 进行几何代理，并记录 cooking；
- 如 CompoundShape 不支持，保留可用的简单箱体基线并标四轮方案 BLOCKED，而不是删除失败测量。

不要在一个实验里同时改碰撞 API、传感器路径、root API、重力和 geometry，以免无法归因。

## 3. 预注册试验矩阵（按先后顺序，不必每个都重新启动完整工程）

| Run 组 | 几何 | Lift / Gap | g / 初速度 | 本次唯一因素与判定 |
|---|---|---|---|---|
| G0 | 原提交 USDA，离线 | 0.300/0.334/0.350 | 不需要 PhysX | 复算原始双重 lift，保留错误证据，不覆盖旧结果 |
| G1 | PR #18 修正后的四轮 Cube Compound | 0.350 / +0.016m | g=0, v0=0 | 有正间隙，t0 世界位姿与 settle 应稳定；同时记录 runtime shape 路径 |
| G2 | G1 | 0.334 / 0m | g=0, v0=0 | 恰好接触的边界实验；0N 不能单独判 FAIL |
| G3 | G1 | 0.300 / -0.034m | g=0, v0=0 | 已知穿透，期望观察碰撞响应；生产用稳定 Gate 必须 FAIL |
| G4 | 一个独立标准 Cube + Ground | 明确悬空至落地 | g=-9.81 或小 v_z<0 | 校准 ContactSensor/PhysX 接触报告管线，非多轮成功证据 |
| G5 | 修正后四轮底盘 | 0.350 / +0.016m | 受控低速向下/正常重力 | 实际落地过程，证实四轮 compound collider 可与 ground 发生交互 |
| G6 | 与 G5 同样配置 | 同 G5 | 同 G5 | **只改变 CollisionAPI/Shape 实现**，比较 G5，找出可靠碰撞结构 |
| G7 | 与证实有效的底盘装配真实 B601-RS | 合法安全间隙 | g=0, v0=0 | P2 真实臂静置回归，必须没有把已有 10 刚体 / 8DOF 结果破坏 |

可以根据因果发现调整 G4/G5 顺序，但要先写入实验 manifest，不能事后随意改预注册指标。各组关键重复至少两次独立 \`run_id\`，每次独立启动、无并发。每轮 dt=1/120 s、至少 60 physics steps；如果实际帧数、solver、质量或环境改变，全部在 manifest 公开。

## 4. P2 门禁：每项单独计算

### 4.1 原 P2 预注册门禁不变（在合法**非穿透**静置工况）
- \`t0_world_position_error_m <= 0.001\`：底盘和需要检查的刚体相对各自授权世界位置；
- \`t0_world_rotation_error_rad <= 0.001\`：源四元数格式独立转换，USD 为 wxyz，当前构建 observed Isaac runtime body_quat 为 xyzw，必须保留原始数组与来源；
- \`t0_mount_position_error_m <= 0.001\` 且 \`t0_mount_rotation_error_rad <= 0.001\`；
- \`max_{step=0..60} ||p_chassis(step)-p_chassis(0)|| <= 0.001m\`；**使用全程最大值，而不是末帧**；
- 全时间有限值、frame 不缺失、没有非法穿透，外层 gate 根据实际 JSON 结果返回正确状态。

### 4.2 新增独立的 **Collision/Contact Gate**
P2 几何/轮地接触完成，必须同时取得：
1. **authored shape geometry correctness**：4 个轮子的 USD 世界高度及父子坐标正确，原始 USDA 已入库，并保留 runtime geometry/shape 和过滤状态的证据；
2. **physical collision active**：G3 的真实已知穿透或 G5 有效加载的原始时间序列，能独立证实轮子会阻挡/改变真实动力学，且不是仅靠主动 write_root_pose 消除穿透；
3. **contact-report pipeline validation**：G4 正控制的 ContactSensor 或 PhysX 接触报告/冲量链路真实有效。若当前 API 根本不可用，明确 \`contact_sensor_verified=false\`，即使位姿有反应也至多 **PARTIAL**，不能宣称 ContactSensor 通过；
4. **nonpenetrating stable**：G1/G2 的 t0 + 60 步稳定和四轮 geometry overlap 检查独立达标；
5. **real-arm regression**：G7 不能破坏之前的真实 B601-RS 挂载/静置 PASS；
6. **gate correctness**：必须有外层 child exit 和 JSON gate 一致性记录，包含预期 FAIL→outer nonzero 的负控制。不能只读取一个 \`gate_result=PASS\` 字符串而忽略 false gates。

**接触力数值阈值不要凭空发明。** 对加载组先通过 G4 正控制确定传感器采样时间、有效动态范围/噪声，再预先冻结“检测到非零冲量/接触报告”的判据。若缺少这类观测，\`sensor=UNKNOWN\`，独立物理碰撞证据和传感器证据分别报告；不可将缺失值写成 0N。

### 4.3 结果词义不可偷换
- \`PASS\`：该明确定义的 Gate 真实达到，证据完整且可重放；
- \`FAIL\`：数值不合格或已知穿透生产初始化；
- \`PARTIAL\`：几何/动力学部分得到证据，但 contact API 或 runtime collision 匹配仍缺；
- \`UNKNOWN\`：该组未运行、数据缺失或 API 未观测到；
- \`BLOCKED\`：有具体错误、环境权限或资源阻断证明，且已记录尝试和合法并行任务。

**P0 PASS、代理场景 P2 PASS、真实臂旧 P2 PASS 不会自动推导多轮复合碰撞 PASS**。

## 5. 必须提交的工程产物

1. 修复后的 author/physics 代码，测试、各 PR 分支和完整 SHA；对比新旧有效形状层级图（以路径/坐标表为主）。
2. 每个 RUN 目录包含：
\`\`\`text
outputs/simulation/rebot_b601_rs/morph_one/p2_wheel_contact/<RUN_ID>/
    authored_chassis.usda
    manifest.json                # full sha, run id, Isaac/PhysX/driver versions, seed, dt, gravity, mass, geometry SHA, config
    world_geometry.json          # each wheel authored/world bbox + collision API/shape paths, runtime status
    raw_poses.jsonl              # reset0, step1...step60, all relevant bodies and velocity
    contacts.jsonl               # sensor values/events and source API; no API means explicit unavailable
    result.json                  # separate geometry / static / collision / sensor gates
    gate_runner_summary.json     # raw child exit, JSON gate, outer exit
    child.stdout.log
    child.stderr.log
    notes.md                     # positives, negatives, unknowns, causal explanation
\`\`\`
若运行时只允许单一 JSONL，也可以保留统一的逐步遥测但必须可独立抽取 pose/contact 字段。run_ID 不能复用，输出目录禁止原地覆盖。数值单位采用 m、rad、s、N，时间戳与 physics step 对齐。
3. 与 PR #18 的代码数学契约对齐的离线 pytest/unittest 和至少一份真实 PhysX 新旧对照；本地和 GPU 日志必须分开报告。
4. Issue #13 的 START/DONE 评论：实测命令、每条 Gate、整个运行的耗时、日志路径、改动 SHA、反例、反事实控制和下轮优先级，不要只宣布总 PASS。
5. 若还没修完，也要提交能复现的失败实验及可审计的“已排除原因”，不要删除历史 RC 三次原始 USD/JSON。

## 6. 下一阶段解锁与可并行范围

只有新 P2 的 **世界/安装/静置/碰撞/接触证据**满足以上门禁，才允许将整体 P2 标为完整 PASS，并准备 P3 动态刚性挂载试验；P3 要求非零平移/非零 yaw 的真实测量结果，不用 root pose 脚本伪装轮驱动。

P4 必须在 P1 真实轮件身份和 P2 接触有效后，验证 wheel joint / actuator / 轮地摩擦 / 轮速→底盘位移的动力学，不能提前声称真四轮驱动已完成。P6/P7/P8/PPO/实机保持已有阶段阻断规则。

P2 GPU 排队或临时失败期间可以做 P1 STEP 轮件映射、P5 B601-RS 单臂静态代码、P7 YOLO/EKF/时间戳等离线模块，但不能冒充依赖整机的闭环结果。

## 7. 代理自主执行与退出标准

- DeepSeek 直接阅读本 SPEC、Issue #13、PR #18 的真实代码，在独立工作树开展任务；必要时实际修复 PR #18 的 bug，而非只赞同结论。
- **首先完成 G1+G3+G4，然后 G2+G5+G6+G7**；每一次失败保存证据、修改后继续。普通依赖/接口/数据格式错误自行解决。
- ChatGPT 在独立分支继续编写/审查验证器与源码，对 GPU 数据独立计算，不由用户复制粘贴中转。
- 只在已授权且可执行工作全部完成或遇到真正运行窗口/资源上限时结束当前会话，直接给出真实状态；不可许诺在无人触发情况下后台继续执行。
- 未获用户额外授权，不合并保护分支、不更改真实硬件、不自动运行 PPO 或危险高能动作。

**阶段验收一句话**：不是“机器人没有动”，而是“USD 世界坐标正确 + PhysX shape 确实生效 + 受控加载产生可信碰撞响应 + ContactSensor 管线有正控制 + 真实臂装配回归稳定”。只有这些相互独立的证据形成闭环，才将 P2 视为完全完成。
