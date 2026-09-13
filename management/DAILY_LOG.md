# 每日开发记录（DAILY_LOG）

> 每天结束前更新，新条目加在顶部。请勿只更新这里而不更新 PROJECT_STATUS / TODO。

---

# 2026-09-09

**今日目标：** 用户新指令：PiPER Stage 0（Isaac Lab 中单独正确导入 PiPER，articulation/joint/actuator/limit/坐标系/最小控制验收；不训练 RL、不动 Morph/动力学）

**完成：**
- 旧编排残留进程核查：远端无残留自动化进程（无需再互相 SIGTERM）
- 补齐 Isaac Lab RL extras（官方 editable 方式安装 isaaclab_rl[all]，RC=0）：rsl-rl-lib 5.0.1 / skrl 2.1.0 / rl-games 1.6.1 / sb3 2.9.0 / gym 0.23.1；**Isaac Sim / Isaac Lab / Python / PyTorch / CUDA / Driver 版本零改动**；随后按 baseline 恢复 isaacsim-kernel/isaaclab 硬 pin（aiohttp==3.13.4、click==8.1.7、websockets==12.0、Pillow==12.2.0），重跑 Cartpole 冒烟正常
- baseline 冻结：远端 robot_sim/logs/baseline_env_20260909.txt（pip freeze 325 包）+ IsaacLab tag v3.0.0-beta2.patch1
- PiPER 6 轴（no gripper）Stage 0 全验收通过（脚本存于本地 scripts/piper_stage0/ 与远端 projects/piper_stage0/scripts/）：
  - 官方 AgileX 描述 agilexrobotics/piper_isaac_sim → piper_no_gripper_description.urdf（6 revolute：joint1..joint6，limits 与 effort/velocity 见 joints.csv）
  - t01 官方 isaacsim.asset.importer.urdf 转 USD（fix_base）→ projects/piper_stage0/assets/piper_no_gripper.usd；ArticulationRoot=/piper/Geometry，joints=/piper/Physics/joint1..6(+root_joint)
  - t02：Articulation 创建（InteractiveScene/Articulation）✅；6 关节名/顺序与 URDF 一致 ✅；limits 与 URDF 全部一致 ✅；无 NaN ✅；静止稳定（drift 0.022 rad、|v|max 0.057 rad/s）✅；单关节位置控制 6/6 ok（err≤0.046 rad）✅；全 6 轴位置控制 ok（err≤0.048 rad）✅；joint pos/vel 输出
  - t03：600 s / 144000 步连续运行（8 s 交替双姿态），nan_count=0、anomalies=[]、passed=true ✅；输出 joint_state_final.csv / soak_log.csv
  - 测试增益：stiffness 150 / damping 15（ImplicitActuator，240 Hz；见 DEC-007）

**发现问题：** RL extras 的 pip 解析与 isaacsim-kernel 硬 pin 冲突（aiohttp/click/websockets/Pillow 被改动）→ 已按 baseline 恢复（ISSUE-005）；t01 输出为嵌套子目录 .usda 致首版断言失败、t02/t03 limits 张量形状差异与 SimulationApp API 小坑（ISSUE-006）

**解决问题：** 见 ISSUES.md ISSUE-005/006

**未完成：** Morph One / 羽毛球动力学 / PPO（用户禁止）；Observation/Action v0.1 未开始

**Git commit：** 待本地提交（本次未自动 commit）

**远端运行任务：** PiPER Stage0 测试（logs/piper_t0*.log；outputs/20260909_*）

**Checkpoint：** PiPER 单臂仿真闭环（导入→控制→soak）✅；projects/piper_stage0/assets/piper_no_gripper.usd


# 2026-09-08

**今日目标：** 检查项目目录与当前状态，建立项目管理结构（本轮不写算法代码）

**完成：**
- 检查工作区现状：目录骨架（docs/management/experiments/runs、configs、scripts、src、tests、assets、tools）已存在但全空；Git 仓库已初始化但 0 commit（分支 master）
- 建立根状态文件：README / PROJECT_STATUS / TODO / CHANGELOG / ROADMAP
- 建立 docs/ 8 个设计文档种子 + management/ 5 个管理文件种子 + experiments/EXPERIMENT_INDEX.md
- 初始化 .gitignore；git 分支 master → main

**修改文件：** README.md、PROJECT_STATUS.md、TODO.md、CHANGELOG.md、ROADMAP.md、.gitignore、docs/×8、management/×5、experiments/EXPERIMENT_INDEX.md、各空目录 .gitkeep

**实验：** 无

**发现问题：** 工作区已有空目录骨架与空 git 仓库，但无任何文件/commit，也无环境与硬件记录

**解决问题：** 补齐知识库种子文件，作为后续一切工作的落盘基线

**未完成：** 环境盘点（GPU/CUDA/Isaac 版本、远端主机）、硬件盘点、Observation/Action v0.1

**明日第一件事：** 完成环境与硬件事实盘点（本地跑检查命令，登记到 docs/deployment.md 与 hardware_interface.md）

**Git commit：** a8bf4d1（chore: init project knowledge base structure，分支 main）

**远端运行任务：** 无

**Checkpoint：** 无

**（同日补充，约 18:26）**
- SSH 到远端开发机验证通过：dgut@172.31.68.251（hostname jxxy）；GPU = RTX A6000 49GB、驱动 550.163.01；Python 3.10.12；docker/nvidia-docker 可用；CUDA toolkit / Isaac Sim / Isaac Lab / conda 均未安装 → 事实已登记 docs/deployment.md
- 远端候选目录 /home/T7/ojh 已查看（含 YOLO 权重与实验目录），**尚未定为项目工作区**
- 关联 GitHub origin = https://github.com/jho887891-sudo/badmin（远程空仓库，用于备份/协同）
- 提交身份默认：jho887891-sudo <jho887891-sudo@users.noreply.github.com>（可改为真实姓名/邮箱）
- 补充 commit：（见 git log）

**（安装推进补充 2026-09-08）**
- 决定安装目标：/home/T7/dgut/robot_sim（根分区 100% 满+inode 满，唯一大空间=NTFS/FUSE 的 T7，1.6T 可用）
- 完成：uv venv Python 3.12.13；git clone IsaacLab @ v3.0.0-beta2.patch1（需 -c http.proxy= 绕过失效代理，ISSUE-001）
- 通道核验：uv 可直连 pypi.nvidia.com 解析 isaacsim==6.0.1.0（curl 301→.cn 为误导，ISSUE-002）；IsaacLab 官方兼容表 v3.0.0* ↔ Isaac Sim 6.0.1 ✓
- torch 版本裁决：IsaacLab 官方 pin torch==2.10.0/0.25.0 cu128（DEC-005）；用户消息 2.11.0/0.26.0 为偏差，已记录
- master 脱离脚本 11:25 启动（PID 2776157）：PHASE1 isaacsim 下载完成（158 包，~1h，CDN 峰值 3.8MB/s）；NTFS/FUSE 解包极慢（uv 线程 FUSE request_wait；ntfs-3g 长时间 I/O）；PHASE2/3 待跑
- 后台编排任务 pwsh-4：轮询至 MASTER DONE 后自动执行验证脚本（torch/cuda/isaaclab import/Cartpole 源检查）

**（2026-09-08 晚：安装完成 + 最小验证通过）**
- torch 2.10.0+cu128 经官方源长时重试终于下载成功（期间速率 ~60KB/s→后段 340KB/s）；isaaclab.sh -i 历经 imgui/TMPDIR（ISSUE-004）、robomimic git 代理（ISSUE-001）、rl_games git 超时等问题
- 核心验证全部通过：python3.12/torch 2.10.0+cu128 CUDA True/import isaaclab/Cartpole-Direct-v0 20 步正常（verify_cartpole.py）
- 遗留：mimic/rl-games/teleop 等可选 extras 因 github 间歇不可达未装完（不阻塞当前阶段）
**（2026-09-08 深夜：权威最终验证，SETUP.md 已重写为一致版本）**
- 官方最小测试通过：`./isaaclab.sh -p scripts/environments/zero_agent.py --task Isaac-Cartpole-Direct-v0 --num_envs 128 --viz none`（本 tag 任务名带 -v0）→ env 建立完成，obs Box(128,4)/act Box(128,1)，GPU +2.4GB、util 2-16% 持续 12min 无错
- 版本实测：isaacsim 6.0.1.0；isaaclab 6.1.14（editable @ v3.0.0-beta2.patch1）；torch 2.10.0+cu128 avail=True；python 3.12.13；robomimic 0.4.0 已装
- 关键运行前提（已写入 SETUP.md/env.sh）：root 100% 满 → HOME/TMPDIR/XDG/UV/PIP 全指 /home/T7；EULA 首次接受；--viz none
- **勘误**：此前记录"rl[rsl-rl] 就绪"不实 —— rsl_rl/rl_games/skrl 仍未装（extras 阶段未执行到），待网络稳定后 `source env.sh && bash isaaclab.sh -i` 补齐
- 未开始羽毛球项目开发（遵守边界）

**（2026-09-09 晨：Badminton Scene v0.1 动工）**
- 远端 projects/badminton_scene 包已建（badminton_scene/scene_layout.py 唯一真相源 + court/net/robot/shuttle/camera/racket/contact/physics_materials 各 cfg + scene_cfg 装配 + reset.py）
- 复用 Stage0 piper_no_gripper.usd（未重转），240Hz；scene 用 InteractiveScene + 原生 pxr 静态几何
- t01_scene_1env.py ✅：1 env 构建 + 90 步 0 NaN，93 prim；输出 outputs/t01_20260909_072346/（prim_tree.txt/state_1env.csv/results.json）
- 已修 API：InteractiveSceneCfg class_name 参数、MeshCuboid spawner 空路径→改原生 pxr、TargetZone z 缺失
- 待续：t02 几何核验、racket(piper_with_racket.usd)、contact、canonical 轨迹、reset、128-env、soak、evidence 图片

**（2026-09-09 晨 续）**
- 生成组合资产 piper_with_racket.usd（reference piper_no_gripper.usd + Racket 于 link6，FixedJoint 用 physics:body0/1 relationship 修正），robot_cfg 已指向它
- t02 ✅ 几何核验；t03 ✅ racket frames（bodies 含 RacketBody；驱动关节后 link6 与 racket 位移差<5e-3）
- 证据：outputs/t02_20260909_072449/geometry_check.json、outputs/t03_20260909_072825/racket_frames.json
- 待续：t04 canonical 轨迹、t05 contacts、t06 reset 1000x、t07 128env、t08 soak、evidence 图片与 scene_layout.json/docs/coordinate

**（2026-09-09 续：t04/t05 PASS）**
- 修复关键步进模式：每步需 scene.write_data_to_sim()+sim.step()+scene.update(dt)（Stage0 一致）；新增 step_env 助手
- t04 ✅ CANONICAL_INCOMING_001：越网最低 z=1.543m（>网顶 1.524），~step190 落地；evidence outputs/t04_*/shuttle_trajectory.csv
- t05 ✅ ContactSensorCfg 真实接触：SHUTTLE_GROUND(step72)/SHUTTLE_NET(step52)/SHUTTLE_RACKET(step25)；contact_events.csv（force+速度前后）；说明：per-partner force_matrix_w 对静态伙伴为 0，采用 net_forces_w+受控场景分类（记录于 results.json）
- 注：ContactSensor 需在 PHYSICS_READY 前创建；shuttle 需 activate_contact_sensors=True
- 待续：t06 reset1000、t07 128env、t08 soak、evidence 图片/layout json/docs

**（2026-09-09 续：t07 PASS / t08 运行中）**
- create_scene 支持多 env（每 env 静态几何）；世界坐标写根状态（env_origins+local）
- t07 ✅ 128 env：joint_pos/vel (128,6)、shuttle pos/vel (128,3)、env_origins (128,3)、nan=0、within_cell、net prims=128（outputs/t07_*/results.json；速度由 physx get_transforms 差分，已注明）
- t08 soak 128env×600s sim 已 nohup 启动（outputs/t08_run.log）
- 待续：soak 结果、contact per-partner matrix 增强验证、reachability、截图、scene_layout/coordinate evidence 收尾

**（2026-09-10：真实场景截图完成）**
- 用 IsaacLab Camera（RTX，--enable_cameras）真实渲染，非示意图；t09_screenshots.py；证据含投影可见性（screenshots.json）与颜色分析（screenshot_content_check.json）
- 6 张：world_frame/scene_perspective/scene_top/scene_side/racket_frames (1280x720) + scene_128env (1600x900)
- 取景自检：persp/side 自动换机位后场地绿占比 0.49/0.55；top 0.76；128env 0.27；racket 特写为真实渲染但坐标轴像素未检出（记录 PARTIAL）
- 注意：两个 Isaac 实例互斥（soak 与截图需串行）；soak 已恢复运行

**（2026-09-10 P0 证据修复）**
- Net 几何修正（spec 2）：collision_bottom_z=0.764、top=1.55、thickness 0.02；visual band 同步；canonical 重跑 PASS（越网 min z=1.543>1.524，无碰网；落地 step190）；t05 低轨迹仍产生 SHUTTLE_NET
- t07 重写并 PASS：128env 形状 (128,6)/(128,3)；仅改 env17 其余在自然漂移容差内不变；reset_idx([3,17,81]) 恰好只改这三者；cross_env_collision=0；NaN/Inf=0；GPU before/with128 记录（vLLM 未停）
- 截图重生成 6 张（当前 Net 几何）：5 张 required-object 可见性检查 ok=True；racket_frames 用差分法证明 tcp/contact 坐标轴可见（8133/6499 px）── 严格纯色阈值因色调映射为 0，故以差分证据为准（axis_pixels_rgb 保留记录）
- t08 v2（含 spawn/flight/contact/reset 循环 + nan/inf/reset_fail/cross_env/cuda/physx 计数）已启动，目标 600s 仿真

**（2026-09-10 P0 续：t08 soak 调试）**
- 修：GroundPlane 覆盖不足→大静态地面（400×400×0.4）；shuttle 初始 z=0 位于地面内部→抬到 0.30；球代理无滚动阻力导致弹射后无限滚出本格（cross_env 虚高）→加 linear_damping=1.0/angular_damping=3.0；CSV/心跳周期 1200 与 reset 周期冲突→beat 改 240
- 现状：t08 运行中，t=173s/600s，nan=0 inf=0 reset_ok=34 reset_fail=0 cross_env=0 contacts 累计 17792；速率~0.167×，预计再 ~43 分钟

**（2026-09-10 t08 soak PASS）**
- 128 env × 600s 仿真完成：steps=144000、wall=1328s、nan=0、inf=0、reset_ok=119、reset_fail=0、cross_env_collision=0、contact_events=61440、cuda_fatal=0、physx_fatal=0、anomalies=0、passed=true
- 证据：outputs/t08_20260910_171556/{results.json, run.log, soak.csv, gpu_stats.txt}
- 期间修复：大静态地面（400×400×0.4，防穿透/覆盖不足）、shuttle 初始 z=0.30、加 linear/angular damping（消除无限滚动导致的 cross-env 虚高）、beat 周期与 reset 周期解耦
- 另：Superpowers 插件已装（profile web, superpowers-dsh@0.1.1，14 skills），但需重启 profile 才生效

**（t08 soak 完成 PASS）**
- 128 env，sim_time 600.0s，144,000 steps，wall 1328.4s
- nan=0 / inf=0 / reset_fail=0（reset_ok=119）/ cross_env_collision=0 / contact_events=61,440 / cuda_fatal=0 / physx_fatal=0 / anomalies=0 → passed=true
- 产物（远端 outputs/t08_20260910_171556/，本地 evidence/t08_soak/）：results.json、run.log、soak.csv(482 行)、gpu_stats.txt
- 另：截图相机 bug 已定位（USD xformOp:orient 需 GfQuatd；set_world_poses 未生效）并修复，重拍待做；先前的 screenshots.json 可见性断言已撤回

**（2026-09-13 BADMINTON_ROBOT.md 规范接入 + Phase 0 审计）**
- 收到 Robot 模块设计规范 BADMINTON_ROBOT.md（1671 行，Source of Truth，文档自述"待用户审阅"）
- 归位：docs/simulation/BADMINTON_ROBOT.md；docs 重组为 docs/architecture/（01–08 + COORDINATE_SYSTEM/SIMULATION_ENVIRONMENT/INTEGRATION_TESTING/ROBOT_BRAIN）+ docs/simulation/
- Phase 0 审计（只读）：冻结软件 ✅（Isaac Sim 6.0.1.0 / Isaac Lab v3.0.0-beta2.patch1 / py3.12.13 / torch 2.10.0+cu128）；PiPER 资产 ✅（joints root_joint+joint1..6，7 bodies，mpu=1.0）；场景约定 ✅（240 Hz、Court Frame、env_spacing 15.0）；球拍视觉 ✅ 存在、final 被 REQUIRES_MEASUREMENT 阻塞（符合设计）；现有测试 **59 项全通过**（16+14+7+16+6）
- 发现缺口：docs/architecture/MODULE_INTERFACES.md 缺失（规范 §64/Appendix C 要求先读）；Morph One 资产不存在（规范允许 TEMP，但拓扑必须四转四驱）；规范里的资产路径与仓库实际路径不一致（需 adapter/映射）
- 未进入 Phase 1（等用户确认规范 + 补 MODULE_INTERFACES.md）

**（2026-09-13 Phase 1 + Phase 2 实现完成，TDD）**
- Phase 1 Config：`simulation/robots/badminton_robot/badminton_robot_cfg.py` + `validation/robot_validator.py`；测试 `tests/simulation/robots/test_robot_config.py` **18 项通过**
  - RED：先跑测试得 `ModuleNotFoundError: No module named 'robots'` → 实现 → GREEN
  - AssetStatus 8 级真实性枚举、Param(value+status+source)、四转四驱 8 个语义关节、PiPER 冻结关节序、双目基线校验、development/final 双模式（final 遇到 TEMP/UNMEASURED 直接失败）
- Phase 2 Frames：`frames/robot_frames.py`（Transform/FrameTree/quat 工具/整机链条/env_origin 剥离）；测试 `test_robot_frames.py` **15 项通过**（含 round-trip <1e-12、S41 组合式校验）
  - TDD 中也修了自己的 bug：`from_xyz_quat` 签名与测试不符（按测试为准修实现）、别名 frame 解析顺序
- 记录：`outputs/reports/phase1_config_report.md`、`phase2_frames_report.md`（规范 §67 格式）；TEMP 清单 `morph_one/TEMP_README.md`
- 未开始 Phase 3（Morph One 运动学）——下一步

**（2026-09-13 用户指令：以后都用 TDD 的完整路程）**
- 立即执行一轮完整 TDD 回路，产物 `tools/usd_glb_common.py`（GLB→USD 共享实现）
  - RED：8 项测试对 stub 先失败（2 failures + 6 errors，全部"功能缺失"）
  - 验证 RED → GREEN 分 3 片实现（解析 / 精度兼容 / 网格材质），每片后重跑看绿色增长
  - 验证 GREEN：**全量回归 9 套 109 项全通过**
  - REFACTOR：两个提取器改用共享 helper（168→78 行、144→88 行，消除重复实现）
  - 验证 GREEN(2)：真实提取重跑，tri/bbox/材质与重构前一致；最终羽毛球资产重建成功
- 报告：`outputs/reports/tdd_cycle_usd_glb_common.md`

**（2026-09-13 按 docs/architecture 实现模块化整体架构，TDD 完整回路）**
- 读出架构契约：主闭环 `Perception → Estimation(EKF/UKF) → Prediction → HitFeasibility → InterceptSearch → Planner/PPO → Safety → Execution → Feedback`；规则：模块不得代劳、PPO 不得绕过 Safety、频率不硬编码、MODULE_INTERFACES.md 管消息结构
- 新增 `src/badminton_brain/`：`types.py`（9 类消息契约 + Court Frame/批量/时间戳校验）、`interfaces.py`（8 层接口）、`registry.py`（一层一模块 + replace）、`pipeline.py`（按基线顺序装配与运行、层输出类型校验、逐步计时、reset(env_ids)）、`validation.py`（development/final）
- 新增 `src/common/status.py`：`AssetStatus/Param` 单一来源；robot 模块改为再导出（消除重复定义，42 项测试护栏不变）
- TDD：RED（先空壳 → 断言级失败 failures=6 / failures=10+errors=5）→ GREEN 分 2 片（10/10、15/15）→ 全量回归 **11 套 134 项全绿** → REFACTOR（共享词汇）→ 再回归全绿
- 报告：`outputs/reports/architecture_implementation.md`；`src/README.md` 更新
- 未实现（明确）：各层算法本体；所有模块 is_implemented=False，final 模式报错

**（2026-09-13 严格按 superpowers 实现 Robot Brain 模块：计划 + 并行派发）**
- 加载 skill：`writing-plans`、`dispatching-parallel-agents`、`subagent-driven-development`、`test-driven-development`
- 计划落盘：`docs/superpowers/plans/2026-09-13-brain-modules.md`（T1–T12，含文件结构、验收点、环境约束、规则、ledger）
- 目标更新：goal 改为"实现 8 层模块"，revision 2，max_goal_rounds=24，已 resume
- 并行派发 10 个全新 implementer 子智能体（各自独立文件、纯 numpy、无 GPU）：
  T1 感知几何 40cee11d｜T2 机器人 EKF c053066a｜T3 羽毛球 UKF eca65f2d｜T4 物理预测 6ba71651｜T8 Safety 143d8024
  T5 可行性门 6606ef65｜T6 拦截搜索 44cd9c69｜T7 规划器 61717545｜T9 执行适配器 c95b6743｜T10 在线自适应 fa3c36ec
- 协调者本人在做 T11：`tests/badminton_brain/test_full_brain.py` 骨架已就位（RED：apps.full_brain 未实现）
- 后续（SDD 连续执行）：等各 agent 回报 → 每任务派独立评审子智能体 → 集成 T11 → 全量回归 → T12 整体评审与记录
