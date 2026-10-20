# Morph One × B601-RS × D455 × 羽毛球：双 AI 联合开发执行 SPEC（v1.0）

> **优先级**：本文件是 Issue [#13](https://github.com/jho887891-sudo/badmin/issues/13) 的可执行技术规格补充。出现冲突时，以明确冻结的验收门槛、真实资产不可篡改约束和 #13 最新经审计决议为准。**SPEC 是工作指令，不是仿真已成功的证据**。
>
> **执行入口**：在 GitHub 直接阅读本文件 → 按工作流 START 评论领取 → 独立分支开发 → 运行/保存证据 → PR → 另一代理 review → gate 判定 → 在 #13 DONE 留痕。**禁止要求用户反复复制代理输出**。
>
> 基准：\`dev/morph-one-p2-init-diagnostics\` @ \`e947bad1d27b1a6c9bceb544c27cb8705291c798\`；P0 工具 [PR #14](https://github.com/jho887891-sudo/badmin/pull/14)；四元数及位姿计算 [PR #15](https://github.com/jho887891-sudo/badmin/pull/15)。以上两项均未合并，使用前需检查当前 HEAD 是否改变。协作主线不得误用仓库 GitHub 默认分支 \`feature/shuttle-detection\` 作为 P2 合并目标。

## 0. 使命、边界和禁止事项

总体目标：准确导入 Morph One M1-4WD4S 的 CAD/STEP，完成真实的四轮底盘动力学与 reBot B601-RS 的 PhysX 合体，挂载球拍及 D455，先用检测→跟踪→预测→可行性判别→低速击球形成可复现闭环；当场景稳定且具有独立验收证据后，才考虑多环境强化学习和实机验证。

- 现阶段优先把 **P0 可信退出码 + P2 正确初始世界位姿和零重力无指令稳定性** 做到可复现。**未过 P2 不能开始 P3/P4/P6 中依赖真实整机动力学的部分，不能跑 PPO 或实机动态击球**。
- 真正的 B601-RS USD/URDF、原始 STEP、源网格、SDK/厂商数据应当视为只读证据；在独立派生资产/overlay 实施修改，提供源 SHA 和派生 diff。TEST_PROXY 的结果不能被宣传为真实资产通过。
- 不把“运行无异常”“关节误差小”“离线检测准确”替代碰撞、真实动力学、闭环击球和安全成功率。
- 不虚构真实轮子身份、摩擦/质量/转动惯量、速度、硬件极限、运行时间、GPU 指标、相机外参或机器人实际成功率。
- **没有用户新的合并/实机授权，不向仓库 main/default/base 分支强制推送，不自行合并 PR，不操作真机，不启动高能运动。** 修改可发生在独立分支/草稿 PR。

## 1. 共同工作方式与所有权

| 角色 | 可以直接修改 | 必须亲自验证 | 必须审别人 |
|---|---|---|---|
| ChatGPT | 标准库 Python 工具、位姿/轨迹度量、gate、测试、SPEC、仿真脚本中的独立修复分支 | 本地能运行的离线单测与历史 JSON 复算；GPU 不可访问时标 NOT RUN | DeepSeek 的 PhysX 实施、原始 JSON、退出码与科学结论 |
| DeepSeek | Isaac Sim/Lab 程序、独立实验工况、装配/运动学、GPU 调试、同样可以修改通用工具 | 真实 GPU 上的 KIT、物理态、全步轨迹、运行退出码 | ChatGPT 的 #14/#15 与后续工具、避免盲用 |
| 用户 | 冻结目标/范围、重大架构决定、合并及实机授权 | 无需搬运日志 | 接收只有证据支撑的总结 |

禁止两方同时直接推送相同开发分支。并行子任务按如下界面分隔：
- ChatGPT owns \`chatgpt/p0-*\`、\`chatgpt/p2-*-metrics\`、\`chatgpt/*spec*\`，主要写纯 Python 仪表、验证器、回归；如需修 PhysX 脚本则新建分支/PR。
- DeepSeek owns \`dev/morph-one-p2-quat-and-settle\`（已经在 [#13 START](https://github.com/jho887891-sudo/badmin/issues/13#issuecomment-6293847205) 登记）及 GPU 实验；如果本地 worktree 尚未 push，先公开最新提交 SHA 让两方不重复劳动。
- 冲突解决：在 PR 做差异审查，按明确 \`git cherry-pick\` / 手动移植小块，并运行测试；不把 PR #14/#15 草率整分支合并到错误 base。
- 任何代理在 DONE 评论中写清楚：已做、没做、假设、失败、与原实验不一致的地方；不能拿另一个代理的声明冒充自己实测。

## 2. 现状冻结与已知失败

历史环境报告：Isaac Sim 6.0.1、Isaac Lab 3.0.0-beta2、RTX A6000、Python 3.12.13、TGS、dt=1/120 s、重力 \`(0,0,0)\`；**仅代表已记录环境，不保证当前机上完全相同**。

- \`p2_b0853ef/p2_joint-local.json\`：TEST_PROXY。FixedJoint 的 \`physics:localPos0 = (0.12,-0.05,0.20)m\`，reset0 局部安装平移残差约 \`4.4e-8m\`，角残差约 0；**相对安装 PASS**。
- USD authored \`Chassis\` 的世界位置为 \`(0,0,0)\`；reset0 实际为 \`(0.000044573,0.000191213,0.037923977)m\`，世界位置差约 \`0.037924485m\`。按正确四元数分量顺序，世界旋转差约 \`0.014896rad\`。**绝对世界 t0 不 PASS**。
- 同一个 TEST_PROXY 在 60 个 settle 步漂移约 \`0.369020227m\`，在零重力下明显上浮，FixedJoint 相对残差仍保持约 \`1e-8m\`；**settle FAIL，根因 UNKNOWN**。未知的是初始化速度、joint/contact 约束冲突、驱动写入、root API 配置、接触求解等哪个造成；不得先入为主。
- \`body-only\` 不授权 \`physics:localPos0\` 时，位置残差约 \`0.238537209m\`，是假稳定；必须保留 joint-local 授权。
- 之前按 wxyz 误读本构建的 \`robot.data.body_quat_w\`（实际观测数组为 xyzw）导致约 179° 虚假异常；USD \`Gf.Quat\` 则是真正 wxyz。**不要只看变量名决定四元数约定**。
- PR #12 旧 shell 的 \`| grep | tail\` 掩盖 Python 非零码，此外 Kit \`close()\` 退出状态存在未隔离取证的风险。P0 仍 PARTIAL。PR #14 是外层 gate runner 修正候选，不是已在 Kit 证实的最终结论。
- STEP 中轮件身份与 bbox \`205.991mm\` 对 \`180.782mm\` 差异仍 UNKNOWN；不能把候选轮当官方轮。闭环击球成功率/可复现 PPO 指标尚未知。

## 3. P0：唯一实验验收协议（必须先行）

### 3.1 运行结构
运行器负责开启独立的**新目录**，保存 child stdout、stderr、原始 child return code、JSON gate 和最终 outer return code。
- \`child exit=0 AND JSON explicit PASS AND no invalid/false gate → outer 0\`；不满足时 outer 非零；timeout 专用 124。
- \`gate_result=PASS\` 与 \`gates={...false}\` 矛盾时 FAIL，不准任选其一。缺报告、NaN、空 gates、非布尔门禁、旧目录重复提交都 FAIL/INVALID。
- 实验结果**一份记录对应一个唯一 RUN_ID**，包含日期、commit、参数、seed、真实软件版本、显卡、dt、g、headless 状态和资产 checksum；绝不覆写既有输出，禁止只截取过滤后日志。
- 子进程 Kit 可因自身原因 exit=0 但 JSON FAIL，outer wrapper 必须失败并保留两者；\`os._exit\` 不应绕过文件写盘/Kit 资源释放，除非额外隔离证据证明安全。
- 最小验证矩阵：①简单 Python PASS/exit0；②简单 Python FAIL/exit0；③PASS/exit7；④缺报告；⑤JSON 语法错；⑥冲突字段；⑦超时；⑧重复目录拒绝；⑨**GPU/Kit 真正 PASS 与预期 FAIL 双例**。
- 离线代码接口可沿用 #14 \`scripts/simulation/p0_subprocess_gate_runner.py\`，完整标准库回归包含 \`tests/simulation/test_p0_subprocess_gate_runner.py\`。如果另有实现则给出两套实际比对。

### 3.2 Gate P0
\`P0=PASS\` 条件是：离线测试成功并提供真实 pytest/unittest 退出码；Kit 独占运行正负例成功，stdout/stderr/child exit/outer exit/raw JSON 全部存在，关闭后无残留 Kit 进程且不会覆盖历史证据。**无 GPU Kit 证据时标 PARTIAL**。

## 4. P2：复现、测量、最小消融和根因修复

### 4.1 几何、坐标与度量必须分开
使用 \`Chassis\` 作为底盘世界位姿参照，而不是随 ArticulationRootAPI 附着对象改变就换一个“根”：
- USD \`UsdGeom.XformCache\` 读出的姿态为 **wxyz**；根据实际所用 Isaac Lab API 明确标 \`raw_quat_xyzw\` 及 \`normalized_quat_wxyz\`；其他 API 需独立核对。
- 使用一致的右手坐标系、米与弧度。四元数长度必须有限且非零；q/-q 视作同一姿态；世界旋转差用绝对四元数内积的最短角 \`2 acos(min(1, abs(dot(q1,q2))))\`。
- 世界误差：\`e_world_pos = ||p_chassis_reset0 - p_chassis_USD||\`，\`e_world_rot = angle(q_chassis_reset0,q_chassis_USD)\`。
- 相对固定安装：\`p_CB = R_C^T (p_B-p_C)\`，与授权 joint anchor \`t_CB\` 比；\`q_CB = inv(q_C) * q_B\`，与授权相对姿态 \`q_CB_expected\` 比（真实 anchor 带旋转时先正确组合 local frame，不能盲设 identity）。
- **两个相互独立 gate**：世界 t0 PASS 不推出相对安装 PASS，反之亦然。使用 #15 \`scripts/simulation/p2_pose_metrics.py\` 作为独立计算参考。
- 输入来源和原始四元数一定要保存到 JSON；证明测量的 yaw=90° 与 roll=60° 非单位姿态时不会发生分量误判。

### 4.2 全时间线记录，不准靠末帧猜
记录时点：USD authored、live spawned（未 step）、\`sim.reset()\` 后 reset0、write_data_to_sim 前/后、step1、step2，之后每一步直到 step60（≥60 步）；优先记录真实时间戳/step index，不给 reset0 和 step1 共用索引。
每个时点、每个 body：
- 位姿 \`p_w, q_raw, q_wxyz\`；root/body \`v_lin_w, v_ang_w\`；关节 \`q, dq\`；各自由度 effort/target（API 可读则录）；
- 关节世界 anchor、rel pos/rot residual、底盘相对初始漂移、root world transform；基础质量、kinematic/static/awake 状态；
- 地面/体间 contact count、法向/冲量、penetration 或约束修正指标（**仅 API 实际支持时**）；不支持填 \`unavailable\` 且注明，不得伪造 0；
- \`write_data_to_sim\` 调用次数、模式与 drive gains、ArticulationRootAPI 位置和是否包含无关节 root；
- 逐步数据保留为 jsonl/csv；摘要 min/max/final 只是索引，不可替代原始数据。

### 4.3 因果诊断顺序（一次只改一个因素）
**所有变体在合法 PhysX/USD 图下比较，固定相同基准与 checksum。** 对非法配置只记 ERROR/NOT APPLICABLE，不拿结果与 PASS 工况对照。

| 编号 | 场景 | 改变的唯一变量 | 应记录与判定 |
|---|---|---|---|
| C0 | 两个刚体 + 1 个 FixedJoint，零重力、无地面、无 drive | 最小基准 | t0/step1/60 世界姿态与线角速度 |
| C1 | C0 同几何，开启地面 | 地面存在 | 判断 contact/depeneration 引发平移与否 |
| C2 | C0 中允许的碰撞过滤开/关 | 过滤状态 | 无假定“所有自碰撞已排除” |
| C3 | C0 同 geometry/anchors，只合法变更 root API：非刚体 Xform / Chassis rigid | root 属性放置 | 将非法 root 配置记录不可比较 |
| C4 | C0 加入第一 RevoluteJoint 和 arm link1 | 一节关节 | 定位链条引入的初始约束错误 |
| C5 | C4 加入第二 RevoluteJoint 与 link2 | 第二节关节 | 比较 step0 动能、初值 |
| C6 | C5 同物理结构，去掉 write_data_to_sim | 写入调用 | 验证写入是否导致速度/effort |
| C7 | C5 同结构，开启 write_data_to_sim | 写入调用 | 记录 write 前后与 physics step 后数据 |
| C8 | C5 控制器同结构，驱动 stiffness/damping 按已记录值与零增益对比 | 仅 gain | 检查 drive 目标/初值冲突 |
| C9 | 同 C5 的独立世界初值重置/不重置比较 | 状态初始化方式 | 证明修复不是硬把漂移藏到初值之下 |

先做 C0 与 C5，确定“基本结构就失败”还是“加关节才失败”，再按最可能因素分支展开；不得省略必要对照并宣称根因已证实。**如果 C0 就漂移**，优先研究 FixedJoint anchor、刚体惯量、root API、非期望速度及约束投影；**如果 C0 不漂移而 C4/C5 漂移**，优先查关节局部锚点、初始角度、重叠、驱动目标与 self-collision。

### 4.4 解决根因时的原则
- 必须以“单一变量修复 → FAIL→PASS”的原始序列支持结论。不能直接把机器人设 kinematic、强制每帧写固定 root pose 或删掉全部碰撞后声称动力学修好。
- 允许使用**仅为了验证根因**的实验性开关，但要标为 diagnostic-only，不作为真实底盘替代实现。
- 如果出现“无重力但速度非零”，用 \`v(t)\` 与 \`Δv(t)\` 区分 reset 带入初速度、解算器投影和持续外力/写入；仅看到近似线性位置还不充分。
- PASS 需要重放至少两个独立 run_id（同基准）、固定 seed，展示逐步小误差；有 GPU 非确定性则说明取样上界而非删失败记录。

### 4.5 预注册 Gate P2（独立四条）
1. \`t0_world_position_error_m <= 0.001\`（Chassis 及要求的各刚体对授权 pose）；
2. \`t0_world_rotation_error_rad <= 0.001\`；
3. \`t0_mount_position_error_m <= 0.001\`，\`t0_mount_rotation_error_rad <= 0.001\`，固定关节局部 pose 正确；
4. \`settle_60_step_chassis_drift_m <= 0.001\`，全步 finite，未授权接触/穿透为 0 或异常项被明确判定 FAIL；未取到 contact 时不能断言“无碰撞”。

至少保存 \`C0\`、\`C5\`、使异常消失的消融组、未消失对照组与重复组的原始曲线和进程码。若任何一项不满足，P2 整体 FAIL，P3 BLOCKED。

## 5. 可独立先做而无需等 P2 的任务（避免闲置）

### P1 真实 CAD 解析/轮件身份
输入原始 STEP 和解析表。静态核查实体身份、单位、轴线、零件装配关系、cad-to-usd mapping。量化 \`205.991\` 与 \`180.782\` mm 差别来自单位、bbox、几何并集、轮胎 vs 轮毂、候选误认中的哪类；**找不到官方确认时写 UNKNOWN**。不得因为 P1 脚本绿就给四轮驱动盖章。

### P5 B601-RS 资产静态一致性/多环境测试
可提前做不依赖整机的单臂 DOF/关节名/limit/joint order、USD-URDF 对齐、reset 独立性、Articulation vs InteractiveScene 接口诊断、N=1/2/16/64/128 环境独立隔离计划。只有 **GPU 实测** 才能对吞吐、回放漂移、PhysX 稳定性下结论。保存每 env 的 joint/pose/error；不得以一个环境“看起来正常”推广到多环境。

### P7 离线视觉/感知（与 P2 硬件依赖解耦部分）
对已有小 4–6 px 羽毛球检测数据先做可复现数据统计、train/val 按视频/场景分组防泄漏、YOLO 输出时间戳→EKF 轨迹接口、预测 horizon=0.3–0.8 s 的测量协议。**无真实机载数据时只能称离线基线**，不能称击球成功。预测接口建议 \`(timestamp, frame_id, camera_intrinsics, detections, cov, estimated_position, velocity, horizon, confidence)\`；数据未标定不可伪造世界坐标。

## 6. 依赖解锁后的阶段规格（P3→P8）

| 阶段 | 先决条件 | 必交工程产物 | 明确门禁 |
|---|---|---|---|
| P3 动态挂载 6DoF | P2 PASS | 底盘 x/y/z 纯平移、纯 yaw、复合运动各一可控场景，timestamped ground truth 与局部安装误差 | **真实非零** yaw 与位移；命令值不算实测；各工况 6DoF 相对误差满足独立阈值及不可忽略的角运动 |
| P4 四轮真实动力学 | P1 轮件身份有可信依据、P2/P3 稳定 | wheel asset→joint→drive 配置、接触/摩擦/轮速/里程计与 commanded velocity 对照 | ground wheel contact 有证据，角速度/位移因真实轮子运动而发生；禁止 \`write_root_velocity_to_sim\` 伪装轮驱动 |
| P5 可训练的 B601-RS 多环境 | 独立单臂正确 + 环境隔离 | 关节 limit、独立 reset、timeout、崩溃恢复、吞吐/memory 统计 | N≥2 无跨环境污染，16/64/128 实测能力按环境资源上报；瓶颈不隐瞒 |
| P6 CAD+ arm+ racket + D455 整机 | P1/P3/P4 与 P5 基本门禁 | frame tree、真实/代理资产对应表、机械臂与底盘碰撞、相机内外参授权、传感器时间同步 | 单位/坐标/装配偏移正确，碰撞无非法穿透；proxy 不冒充官方真实资产 |
| P7 低速闭环 | P6 基础可靠 | 小目标检测/时序滤波/预测/可达性/IK/控制、失败回放与硬停止 | 固定球→慢速确定性来球→有限扰动；报告每组试次数、检测率/击中率/合法回球率，安全 HOLD 可复现 |
| P8a PPO | P5/P6/P7 的训练闭环及预算门禁通过 | observation/action/reward/reset、固定种子、train/test split、baseline、reward 与碰撞安全日志 | 训练曲线、同预算基线对照与闭环成功率；禁止用离线动作误差替代 |
| P8b 真机 | 实机资料、安全审批和独立测试许可 | 急停、watchdog、限速、限加速度、操作员隔离、人工复位、数据保护 | **独立人身安全门禁**；仿真成功≠真机许可 |

注意：P3 的具体数值角/平移误差阈值在实施前按 #13 原协议冻结；不能事后看到跑不过才放宽。P4/P6/P7 的绝对验收阈值需要由真实资产、公差/任务需求授权后预注册，若暂无证据则保持 UNKNOWN，不随意编造。

## 7. 所有实验共用证据目录和摘要 Schema

每次运行独立目录：
\`\`\`text
outputs/simulation/rebot_b601_rs/morph_one/<run_id>/
  manifest.json                 # SHA/config/seed/versions/command/asset checksums
  authored_proxy.usda           # 若为 TEST_PROXY；实际源资产只引用 SHA
  raw_poses.jsonl               # 原始逐步 body/joint 状态、来源/单位/时间戳
  result.json                   # 逐指标数值 + explicit gates + gate_result
  child.stdout.log
  child.stderr.log
  gate_runner_summary.json      # child/outer exit 与时间戳
  notes.md                      # 只解释，不伪造缺失值
\`\`\`

\`manifest.json\` 至少包含 \`stage, run_id, commit, base_commit, scene_variant, asset_hashes, isaac_sim_version, isaac_lab_version, python_version, torch_version, gpu, physics_dt_s, gravity_m_s2, seed, command, root_api_target, include_ground, collision_mode, drive_mode, write_data_to_sim\`。

\`raw_poses.jsonl\` 每行至少 \`{"step":int,"t_s":float,"event":str,"bodies":{"Chassis":{"position_m":[...],"raw_quaternion":[...],"quat_layout":"xyzw","source_api":"...","linear_velocity_m_s":[...],"angular_velocity_rad_s":[...]}},"joints":{"name":{"position_rad":...,"velocity_rad_s":...}},"contact_status":"measured|unavailable"}\`。注意每步有可变 DOF 时也需记录 joint_names/index 映射，不用未授权假字段冒充已测。

\`result.json\` 至少包含 \`{"gates":{"t0_world_position_within_tol":false, "t0_world_orientation_within_tol":false, "t0_mount_position_within_tol":true, "t0_mount_orientation_within_tol":true, "settle_drift_within_tol":false, "no_non_finite":true}, "gate_result":"FAIL","metrics":{...},"source_run_id":"..."}\`。\`gate_result\` 必须与 gates 一致；缺关键字段就是 INVALID，不能靠默认真值通过。

## 8. 对 DeepSeek 的直接执行清单（无需任何用户中转）

1. 阅读 #13、此 SPEC、#14/#15 源码及测试。先在 #13 评论列出最新真实工作树 HEAD、Isaac/GPU 版本、预定 RUN_ID 和将使用的分支。如原 START 第二轮还在进行，**续做，不重复重建工程**。
2. 先跑 #14/#15 离线单测并 review，报告精确命令和 exit；针对实际必要的接口 cherry-pick 经审计代码。不要求统一吞并所有 PR。
3. 实现 P2 姿态正确转换、世界 t0 gate、逐步速度和接触采集；从 C0→C5 先跑，异常的位置在加入哪一项后出现就围绕那一项做单因素消融。
4. 取得真正 P2 FAIL→PASS 时保留失败与成功的完整原始证据；没 PASS 不能报 DONE PASS，更不能去 P3。若发现一个需用户做实际决策的硬件/安全 blocker，记录 UNKNOWN，继续可独立完成的 P1/P5/P7 工作，不必反复询问普通参数。
5. 同时与 ChatGPT 的 PR 互审，给出源代码行号、测试反例、必要补丁建议和实际运行反馈；任何“通过”必须区分离线、proxy、真实资产、GPU 仿真、实机这五种级别。
6. 结束每个独立工作单元，将 SHA、PR、命令、raw JSON/logs、exit、门禁结果直接写到 #13；**不要等待用户帮忙贴日志**。

## 9. DONE/FAIL 接受标准与停机规则

\`\`\`text
START: stage/workstream, owner, base full SHA, branch, expected file scope,
       pre-registered gates, experiment plan, no-go list
DONE:  stage/workstream, output full SHA/PR, file list, run_id, source hashes,
       python/Isaac/GPU versions, exact commands + child/outer exits,
       gate-by-gate metrics, raw evidence paths, failed cases, UNKNOWN,
       reviewer, remaining blockers
\`\`\`

验收裁定由**数据 + 可重放代码 + 另一代理审计**共同决定。出现以下任一情况立即停止扩大范围并明确 FAIL/BLOCKED：非法物理穿透、无根因的世界漂移、关节不稳定/NaN、用 proxy 顶替真资产、声称动作但状态不变、日志缺失、错误退出码被掩盖、提交改动破坏历史证据、实机保护未授权。

本次 SPEC/代码提交本身**不构成 P0、P2、P3 或任何硬件门禁自动 PASS**；没有持续自动执行/通知部署时，GitHub 上新增评论不会自行触发 DeepSeek 进程。代理须主动读取并执行，不能向用户承诺其睡眠期间任务会自己运转。
