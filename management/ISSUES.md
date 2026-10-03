# 问题与解决记录（ISSUES）

> 规则：单次排错超过 15 分钟必须记录；出现类似错误**先搜索本文件**，不要重复踩坑。

## ISSUE 模板
```
## ISSUE-XXX
- **日期：**
- **模块：**
- **现象：**
- **错误日志：**
- **原因：**
- **解决方案：**
- **修改文件：**
- **验证方法：**
- **是否彻底解决：** 是 / 否（原因）
- **相关 commit：**
```

---

当前记录：无


## ISSUE-001 远端 git 全局代理失效（127.0.0.1:10080 拒连）
- **日期：** 2026-09-08
- **模块：** 环境/网络
- **现象：** git clone/ls-remote 报 "Failed to connect to 127.0.0.1 port 10080: Connection refused"
- **错误日志：** fatal: unable to access ... Failed to connect to 127.0.0.1 port 10080 after 0 ms
- **原因：** ~/.gitconfig 配了 http/https.proxy=http://127.0.0.1:10080，但该代理当前未运行（jxxy 上无监听）
- **解决方案：** git 命令加 `-c http.proxy= -c https.proxy=` 绕过（curl/github 直连可达）；勿改动用户全局 git 配置
- **修改文件：** 无（命令行级绕过）
- **验证方法：** git -c http.proxy= ls-remote 成功；IsaacLab clone 成功
- **是否彻底解决：** 部分 —— 代理未恢复期间每次 git 命令需带 -c 参数
- **相关 commit：** 无

## ISSUE-002 NVIDIA pip 源 curl 探测误导（301→.cn 404，但 uv 实际可用）
- **日期：** 2026-09-08
- **模块：** 环境/网络
- **现象：** curl https://pypi.nvidia.com/simple/isaacsim/ 返回 301 到 pypi.nvidia.cn，.cn 上所有包 404，一度误判 NVIDIA 源不可用
- **错误日志：** pypi.nvidia.cn/simple/isaacsim/ -> HTTP 404
- **原因：** 301 为地域分流；curl 未跟随的路径与 uv 客户端行为不一致；uv（含 --extra-index-url https://pypi.nvidia.com --index-strategy unsafe-best-match）实测可正常解析 isaacsim==6.0.1.0（166 包 dry-run 通过）
- **解决方案：** 用 uv 直接安装即可；探测通道请以 uv/pip 实测为准，勿仅凭 curl 结论
- **修改文件：** 无
- **验证方法：** uv pip install --dry-run "isaacsim[all,extscache]==6.0.1.0" 成功
- **是否彻底解决：** 是（通道确认）
- **相关 commit：** 无


## ISSUE-003 官方 cu128 torch 源间歇 AccessDenied；国内镜像无 cu128
- **日期：** 2026-09-08
- **模块：** 环境/网络/依赖
- **现象：** uv pip install torch==2.10.0(+cu128) --index-url https://download.pytorch.org/whl/cu128 反复失败：14:00-14:12 曾成功下载 torchvision/nvidia-cudnn-cu12，之后 torch-2.10.0+cu128 wheel(874MB) 下载 3+ 次均失败；curl 直取返回 S3 AccessDenied XML
- **错误日志：** <?xml ...><Code>AccessDenied</Code>；uv "Downloading torch (874.4MiB)" 循环
- **原因：** download.pytorch.org 的 S3/CF 对该出口 IP 大对象间歇拒绝（时间窗口性）；国内镜像（TUNA/阿里/上交/中科大/华为云）均未同步 cu128（止于 cu126；华为云 200 为 SPA 兜底非真文件）
- **解决方案：** 用户拍板走官方源；uv 后台长时重试（retry_torch_run.sh，间隔 ~7min×40≈5h），命中放行窗口即装成；装成后 isaaclab.sh -i 的 `_ensure_cuda_torch` 精确匹配 torch==2.10.0+cu128 会跳过（install.py:277 起）
- **修改文件：** 无（远端脚本）
- **验证方法：** 重试日志 RC=0 后 python import torch 打印 2.10.0+cu128 且 cuda.is_available()=True
- **是否彻底解决：** 是 —— retry_torch_run.sh 于 16:30:51 RC_1=0，torch 2.10.0+cu128 装成且 avail=True（ISSUE-003 关闭）
- **相关 commit：** 无


## ISSUE-004 isaaclab.sh -i 失败：imgui 编译写 /tmp 撞满根分区
- **日期：** 2026-09-08
- **模块：** 安装/构建
- **现象：** ./isaaclab.sh -i 安装 isaaclab_teleop 子模块时构建 imgui 失败，日志：Cannot create temporary file in /tmp/: No space left on device；随后 SIGABRT，ISAACLAB_I_RC=1（16:59:29）
- **原因：** 根分区 / 容量 100% 满（仅 356K）；cc 编译默认临时目录 /tmp 位于根分区 → ENOSPC。其余子模块（isaaclab/ppisp/assets/contrib/...）此前均已 -e 装好
- **解决方案：** TMPDIR/TMP/TEMP 指向 /home/T7/dgut/robot_sim/cache/tmp（T7 盘）后重启 ./isaaclab.sh -i（run3.sh，PID 2901987）；imgui 将可正常构建
- **修改文件：** 无（远端脚本/环境变量）
- **验证方法：** isaaclab3.log 出现 ALL3_DONE 且 rc=0
- **是否彻底解决：** 是 —— TMPDIR 指 T7 后 run4/run5 中 isaaclab_teleop 构建通过（isaaclab_teleop-0.5.2 装成）（ISSUE-004 关闭）
- **相关 commit：** 无

## ISSUE-005 补齐 RL extras 时 pip 改动 isaacsim-kernel/isaaclab 硬 pin 依赖
- **日期：** 2026-09-09
- **模块：** 依赖管理
- **现象：** 安装 isaaclab_rl[all]（官方 editable 方式）后 pip 报告依赖冲突：isaacsim-kernel 6.0.1.0 要求 aiohttp==3.13.4/click==8.1.7/websockets==12.0/Pillow==12.2.0（isaaclab 亦要求 Pillow==12.2.0），而 extras 装成 aiohttp 3.13.3/click 8.5.0/websockets 17.1/Pillow 11.3.0
- **原因：** isaaclab_rl extras 内部分包固定（如 aiohttp==3.13.3）与 isaacsim-kernel 的严格 pin 冲突；python -m pip 未做全局解算
- **解决方案：** 安装完成后显式恢复 baseline pin：aiohttp==3.13.4、click==8.1.7、websockets==12.0、Pillow==12.2.0（uv pip install）
- **修改文件：** 无（仅 venv 包版本）
- **验证方法：** import 版本核对 + Cartpole 冒烟（env 8 个、scene 建立、无错）+ torch/isaaclab 仍 2.10.0+cu128 / 6.1.14
- **是否彻底解决：** 是 —— rl 库全部可 import（rsl_rl/skrl/rl_games/sb3），核心版本零改动
- **相关 commit：** 无

## ISSUE-006 PiPER Stage0 开发中的 4 个小坑（USD 路径/数组形状/API/命名）
- **日期：** 2026-09-09
- **模块：** PiPER Stage0 脚本
- **现象：**
  1) t01：URDFImporter.import_urdf 返回嵌套子目录内 .usda（如 usd_piper_no_gripper/piper_no_gripper_abs_1/*.usda），首版按 usd_dir/*.usd 断言失败
  2) t02/t03：robot.data.joint_pos_limits 形状 (n,2)，而 joint_vel_limits/joint_effort_limits 为 (1,n) 等不同形状 → 直接 squeeze/下标报 IndexError
  3) t03：simulation_app.is_stopped() 不存在（isaaclab SimulationApp 无该方法），应用 sim.is_stopped()
  4) 任务/寄存器名注意：Isaac Lab 环境注册名带 -v0；headless 用 --viz none（Stage-0 冒烟经验）
- **原因：** API 版本细节（IsaacLab 6.1.14 / Isaac Sim 6.0.1）+ 未先核对返回路径与张量形状
- **解决方案：** t01 直接使用返回路径并 Usd.Stage.Export 规范化二进制 USD；t02/t03 增加 norm_2d/reshape(-1) 形状归一；t03 改用 sim.is_stopped()
- **修改文件：** scripts/piper_stage0/t01_convert_urdf.py、t02_articulation_checks.py、t03_soak_10min.py（本地 + 远端）
- **验证方法：** t02/t03 全 phase 通过（含 600s soak，nan=0）
- **是否彻底解决：** 是
- **相关 commit：** 无
## ISSUE-005 USD 25.11：xformOp 精度与 customData 类型（**RESOLVED 2026-09-13**）
- **现象：** 新建 `xformOp:orient` 在不同条件下分别是 `Quatf` 或 `Quatd`，写入另一精度直接抛 `Tf.ErrorException`；`SetCustomDataByKey` 不接受 Python list
- **触发记录：** ① 本会话早期相机 `cam.set_world_poses` 无效 → 改直接写 USD 需 `GfQuatd`；② 用户 `build_racket.py` 写 `GfQuatd` 到 `Quatf` op 失败；③ 我的球拍预览脚本写 `Gf.Quatf` 到 `Quatd` op 失败；④ `SetCustomDataByKey("face_normal_local", [1.0,0.0,0.0])` 失败
- **处理：** 统一做法 —— **读取属性类型再按类型写入**（`_set_orient(op, quat)`）；customData 用 `Vt.FloatArray`。已在 `tools/build_racket.py` 与本项目脚本中落地
- **影响范围：** 所有 USD authoring 代码（含后续 Robot 模块的 TEMP 资产）
- **状态：** OPEN（作为编码规范固化；建议后续抽成公共 helper）

## ISSUE-006 规范引用的 `docs/architecture/MODULE_INTERFACES.md` 缺失
- **现象：** `BADMINTON_ROBOT.md` §64 与 Appendix C 要求实现前必读该文档，但仓库中不存在
- **当前处理：** 未擅自编写（避免自造接口），Phase 1/2 只依赖 `COORDINATE_SYSTEM.md` 与规格本身；接口层（Robot State / Command / 与其他模块的数据契约）如与该文档冲突需要回改
- **需要：** 用户提供该文档，或授权我按规范补写并送审
- **状态：** OPEN（潜在返工风险）
### ISSUE-005 解决记录（2026-09-13）
- **解决方式：** 抽出 `tools/usd_glb_common.py`，其中 `set_orient_compat(op, quat_wxyz)` 先读属性类型再按类型写入（Quatf/Quatd 皆可）；两个 GLB 提取器与后续 USD authoring 统一调用它，不再各写一份
- **覆盖测试：** `tests/tools/test_usd_glb_common.py::OrientCompatTests`（新建 op 不报错 + 旋转方向正确）
- **状态：** RESOLVED（若后续在别处再出现同类写法，视为新缺陷）
### ISSUE-006 追加说明（2026-09-13）
- 模块化架构已按 ROBOT_BRAIN.md 实现，但**消息字段集属临时契约**：层名/输入输出类型/顺序已固定，
  字段细节待 `docs/architecture/MODULE_INTERFACES.md` 到位后核对；若该文档给出不同字段名，只需改 `src/badminton_brain/types.py` 与测试，管线与接口不变
## ISSUE-007 reason 词表存在两份（HitReason / InterceptReason）
- **现象：** 决策层两个模块各自定义同源 reason 枚举（T5 `HitReason`、T6 `InterceptReason`），语义一致但代码独立
- **风险：** 后续新增词条时两侧漂移，上层按字符串分支可能漏判
- **当前处置：** 保留（DEC-022），已在测试中固定两侧词表；T12 复核时核对交集与拼写
- **归属：** 决策层（T5/T6 文件所有者）
- **状态：** OPEN（低危，计划性技术债）

## ISSUE-008 感知有效性掩码未贯穿到估计层
- **现象：** T1 的 `StereoPerceptionModule` 对无效行填 `[0,0,0]` 并暴露 `last_valid_mask`，但**下游无人消费**；因此「球出视场」与「球在原点」当前不可区分（T1 报告残留风险，D4 修复后出视场不再崩溃但有此语义缺口）
- **影响：** 球出视场时估计层可能把 (0,0,0) 当测量使用（T3 UKF 会把状态往原点拉）
- **建议：** 在 `ShuttleMeasurement` 上增加可选 `valid_mask (N,)` 契约字段（需 DEC），或在估计层消费 `perception.last_valid_mask`
- **状态：** OPEN（中危，需在接入真机/长回合前解决）
## ISSUE-009 odom_twist 字段名与语义（每步增量）不一致
- **现象：** T2 评审指出 `RobotSensorState.odom_twist` 被 `estimation/estimator.py` 当作**每步增量** [dx, dy, dyaw] 消费，但字段名与类型校验的措辞像「速度（twist）」。当前所有调用点都填 0，故未暴露；一旦感知/仿真层填真实速度，EKF 会把速度当位移积分
- **当前处置：** 已在 `types.py` 契约注释中**钉死语义**（per-step increment，机体系，EKF 只积分一次；`imu_yaw_rate` 为同一区间角速率），并保留 DEC-013 记录
- **建议：** 接入真实里程计前重命名为 `odom_increment`（两处调用点 + 契约测试），避免长期歧义
- **状态：** OPEN（低危，已用注释与 DEC 缓解；属于命名债）
## ISSUE-010 base_twist 的物理帧无法在消息中编码（仅文档 + 护栏）
- **现象：** `types.check_court_frame` 强制每条消息 `frame == 'court'`，因此 `SafeCommand.frame` 始终是 `'court'`，而其 `base_twist` 字段的**物理帧是 robot_base（机体系，DEC-014）**。该差异无法在消息里表达，只能靠 docstring 与护栏测试锁定
- **现有护栏：** T9 `FrameConventionGuardTests`（5 项）：文档即契约守卫、court→body 映射复算、原样透传证明（`record.body_twist` 与传入逐位相同）、负向护栏（court 直传会让四个舵角各错 90°）、wz≠0 时轮速差异可见（>1%）
- **注意：** wz=0 的纯平移下，两帧的轮速**大小**相同（只差舵角 90°），因此轮速差护栏必须带 wz≠0；舵角护栏在 wz=0 也有效
- **建议：** 若将来需要机器可校验的帧标注，应解冻 `types.py`，为 `WholeBodyTarget`/`SafeCommand` 增加显式 `twist_frame` 字段（需 DEC + 契约测试）
- **状态：** OPEN（低危，已被文档与护栏覆盖；属契约表达力债）
## ISSUE-011 机器人能力模型（底盘/机械臂限值）尚未收敛为单一真源
- **现象：** 同一物理能力在多个模块各自以 TEMP 代理表达且互相不一致：T5 的工作空间盒（忽略底盘 yaw、站位区各内缩 20/30 cm）、T6 的 yaw 感知工作盒 + 底盘三角/梯形时间门、T7 的 standoff/关节偏置与速率、T8 的 `base_twist_axis_max`/`base_translation_speed_max`/racket 工作空间盒。T5 与 T7/T8 评审均指出这一漂移风险
- **影响：** 换用实测 PiPER/Morph One 参数时需要逐模块改，容易漏改导致判据互相矛盾（例如决策层认为可打、安全层却 HOLD）
- **建议（T12 之后处理）：** 抽一个 `robot_capability` 单一真源（`Param` 化：底盘速度/加速度限、机械臂工作空间盒、关节限位/速率、接触可达锥），由 decision/planning/safety 三方 import；配合 `measurement_requirements()` 统一暴露未实测项
- **当前缓解：** 三方都提供 `measurement_requirements()`，未实测项在 `final` 模式一律报错（DEC-019），因此不会静默上线
- **状态：** OPEN（中危，架构债；属 T12 后第一优先）

### 文档注解待办（随 ISSUE-011 一并处理）
- `docs/architecture/04_HIT_DECISION.md` S20 需补注：实现采用「窗首样本」保守变体（DEC-025）
## ISSUE-012 T6 的绝对/相对时间「自动判别」启发式应当换成显式不变量
- **现象：** `decision/intercept_search.py` 在 `times[0] < trajectory.timestamp` 时**静默**切换到相对基准。当前预测器满足 `times[0] == timestamp`（T4 已按 DEC-016 修正），走的是正确分支；但一旦某个生产者让 `timestamp` 略微领先 `times[0]`，搜索会把绝对 `time_s`(约 12 s) 与 `now_s`(约 0) 相减 → 得到约 12 s 可达时间 → **一切都被判可行**
- **建议：** 把启发式改为显式不变量检查（不满足 `times[0] == timestamp` 直接 raise），或由调用方显式传 `now`
- **风险：** 静默降级为「永远可行」，是危险的失效模式
- **状态：** OPEN（中危；当前不触发，但缺少护栏）

## ISSUE-013 空气动力长度 L=6.5 m 在三个模块各声明一份
- **现象：** `adaptation/online_adaptation.py`（AERODYNAMIC_LENGTH_M）、`estimation/shuttle_ukf.py`（1.0/6.5）、`prediction/physics_predictor.py`（L=6.5 与 k=1/L）各带一份 source。物理 ODE 本身是单一真源（三者都 import shuttle_aerodynamics），但常数有 3 份副本
- **建议：** 收敛为由 `trajectory/shuttle_aerodynamics` 导出的单个 Param（决策层的同类重复已由 `decision/travel_model.py` 收敛，可作范例）
- **状态：** OPEN（低危；漂移风险）

## ISSUE-014 UnifiedState.base_twist / racket_contact_twist 的机体系语义未进契约/DEC
- **现象：** `estimation/robot_localization.py` 的 base_twist() 明确定义为**机体系** [vx,vy,vz,wx,wy,wz]，`estimator.py` 又把它原样填进 racket_contact_twist；而 `types.UnifiedState` 的 docstring 写「one consistent Court-Frame state」且 check_court_frame 强制 frame=='court'。当前 src 内无人消费这两个字段（尚未暴露），但语义漂移与 DEC-014 同类
- **建议：** 与 DEC-014 一并补 DEC：明确 UnifiedState.base_twist / racket_contact_twist 为机体系，或改为显式字段名；同时 racket_contact_twist 缺少 omega×r 力臂项应标 TEMP
- **状态：** OPEN（低危；尚未被消费）
## ISSUE-015 T11 集成测试曾在高负载下出现一次性 errors=6（无法复现）
- **现象：** 某轮批量运行中 `test_full_brain.py` 报 `Ran 11 tests ... FAILED (errors=6)`；随后**连续多次**单独复跑（含 CPU 负载下）均 `OK (11 tests)`，无法复现。同一时段仓库有 5+ 个 agent 并行写盘（reviewers 做 mutation testing）
- **可能原因（未证实）：** ①并发期间文件被其他 agent 短暂改写（此前已两次观察到 `types.py` 等被并发修改）；②全仓唯一墙钟使用者 `pipeline.py` 的计时探针 + T11 的 clock/e-stop 用例在多 agent 争抢 CPU 时触及时钟容差
- **当前状态：** wave 4 / 5 / 6 三次全量回归中 `test_full_brain` 均为 11 OK；D7/D8 修复后新增了端到端护栏 `test_end_to_end_full_runtime_keeps_the_slow_loop_alive`（12 拍 20 Hz 断言慢环存活）
- **建议：** 后续 CI 化时要求「记录文件 hash + 单进程独占」再判定红绿；若再复现，优先检查 SafetyShield 的 clock tolerance 断言与并发写盘
- **状态：** OPEN（低危，可观测性问题；不影响当前结论）
## ISSUE-016 远端与本地均无法稳定下载 GitHub 发布资产 CDN 大文件
- **现象：** 远端 jxxy 可访问 api.github.com 与 pypi.org，但 release-assets.githubusercontent.com 时通时断：5–6 MB 的 yolo11n/yolov8n 直连成功；20 MB 的 yolo26s.pt 直连 278 s 连接超时；38 MB 的 yolo26s.onnx 远端与本地重试均失败（本地为 curl (56) Recv failure: Connection was reset）
- **当前处置：** 走本地下载 → scp 上传 → 两端核对字节数与 sha256（yolo26s.pt 已按此完成并通过校验）
- **影响：** 后续较大第三方资产（模型权重、数据集、USD/GLB）默认应走本地中转路径，并在 THIRD_PARTY_ASSETS_*.md 记录来源与哈希
- **状态：** OPEN（环境约束，已有可用绕行方案；ONNX 仍未取得）

## ISSUE-017 训练集背景与冻结能力测试集同源泄漏（**RESOLVED 2026-09-14**）
- **现象：** P4 训练集背景池 `bg_train` 中的 `tbg_011.jpg` 与冻结的 P3 `SYNTHETIC_ON_REAL_BG` 能力测试集背景 `bg_021.jpg` 是**同一张源图**。两条独立证据一致：① 两端元数据的 Commons 标题/URL 完全相同（`File:Interior overview of the indoor sports hall at OAU Gymnasium.jpg`）；② 32×32 亮度归一化互相关 **r = 1.0000**
- **危害：** 若照此训练再在冻结 P3 集上评估，30 张背景中约 1 张（≈160 张里的 8–9 张）是模型见过的背景 → **系统性高估能力**，且集中在 `≤8px` 这一每百分点都关键的区间；违反协议「训练集与冻结测试集严格隔离」
- **处置：** 移出 `tbg_011.jpg` 至 `train_data/bg_excluded/`；`bg_train_val_metadata.csv` 该行 `split` 改为 `bg_excluded` 并写明理由；复检最大相关降至 **0.7753**（低于 0.95 同源阈值），标题/URL 交集归零
- **遗留建议：** 泄漏审计应**自动化**并纳入生成脚本前置门禁（当前为一次性检查脚本）；后续新增背景池必须重跑
- **状态：** RESOLVED（已隔离并复检；自动化门禁待补）

## ISSUE-018 合成数据集「受控像素尺寸」从未真正受控（**RESOLVED 2026-09-14**）
- **现象：** 旧生成脚本用 `dist = fx × 0.0778 m / target_px` 反推相机距离来「控制」目标像素尺寸。实测标称值与实际严重不符：标称 6px 实得 4px、12px 实得 10px、24px 实得 18px（偏差 17–33%）；标称 **2px 时实际覆盖率最高仅 0.44，按「≥50% 覆盖」判据 GT 框直接消失**
- **双重根因：** ① 0.0778 m 是羽毛球**长度轴**，朝向相机时被透视压缩，真正投影的是 61.9 mm 裙径；② 羽毛裙是稀疏网格，可见 bbox 远小于顶点包围盒
- **危害：** 「2–32px 全尺寸覆盖」是这套能力集存在的唯一理由（P3 报告的核心就是分尺寸桶 recall），尺寸没控住 → 分桶结论的可信度受损；P3 报告只能事后测量 `1.00–25.50` 正是此因
- **处置：** 新增闭环标定 `calibrate_distance()`：渲染 → 测量真实 GT 足迹 → 按比值迭代距离（容差 5%）。已由测试验证 2/4/8/16px 目标命中 ±25% 内
- **附带裁决：** GT 判据由「像素覆盖率 ≥50%」改为**物体足迹**（被触碰的像素即属于目标，`SUPPORT_COVERAGE`）—— 亚像素尺度下 50% 判据会让 GT 消失；同时 manifest 记录 `coverage_sum`/solid_px 供分析方自选判据
- **状态：** RESOLVED（有测试护栏）

## ISSUE-019 manifest 漏记实际生效参数，破坏可追溯性（**RESOLVED 2026-09-14**）
- **现象：** 旧 manifest 的 `noise_sigma` 列**全为空**，但代码确实施加了 σ∈[1,5] 的噪声；`background` 列存的是**样本名**而非背景文件名（导致训练/验证泄漏审计无法进行）
- **处置：** `render_sample()` 统一「抽一次值 → 传给 composite → 记录同一个变量」；新增护栏测试断言 manifest 记录能**逐像素复现**自己的图像。修复过程中该测试又抓出第二个缺陷：记录对 `shadow_gain`/`noise_sigma`/`motion_angle` 做了 `round()`，回代产生 ±1 像素差异 → 已改为精确存储（姿态角度等仅供展示的字段仍保留舍入）
- **状态：** RESOLVED（有测试护栏）

## ISSUE-020 Windows 平台 I/O 陷阱：非 ASCII 路径与默认编码（**RESOLVED 2026-09-14**）
- **现象：** ① 本仓库路径含中文（`E:\具身智能\...`），`cv2.imread`/`cv2.imwrite` 对此**返回 None / 静默失败**，首轮 6 个背景全部「unreadable」；② 本地 Python 默认编码为 GBK，读取远程生成的 UTF-8 CSV 抛 `UnicodeDecodeError`；生成的 manifest 若用默认编码写盘，同步到 Linux 会出现 mojibake
- **处置：** `tools/shuttle_render.py` 提供 `imread_unicode`/`imwrite_unicode`（走字节缓冲 + `np.fromfile`/`tofile`），CLI 全部改用之并显式 `encoding="utf-8"` 写 CSV；相关脚本 `15_check_sheet.py` 一并修正
- **状态：** RESOLVED（有测试护栏；后续所有涉及路径/CSV 的脚本应沿用同一套 helper）

## ISSUE-022 尺寸标定在错误的采样倍率下测量（**RESOLVED 2026-09-14**）
- **现象：** 首版 `calibrate_distance()` 在 `measure_supersample=2` 下测量足迹，而成品图用 `supersample=3` 渲染 —— **标定的是一个谁也不会渲染的量**。实测 target=8px 时成品为 **10.00px（ratio 1.250）**，比不标定的朴素公式（6.48px）**更差**
- **为何没被发现：** 当时的尺寸测试容差为 ±25%，而朴素公式的误差恰好约 25%（0.81–0.88）→ 测试无法区分"标定了"与"完全没标定"
- **发现方式：** 变异测试（DEC-020 要求的「变异可杀」纪律）—— 把标定打回朴素公式后测试**依然通过**（SURVIVED），从而暴露测试过松
- **处置（三重）：**
  1. `calibrate_distance` 的测量倍率改为与调用方一致（默认 3，`render_sample` 显式传自身 `supersample`）
  2. 返回迭代中**误差最小**的距离而非最后一次，吸收小尺寸端主导的 ±1px 量化振荡
  3. 测试容差收紧至 ±12%，并新增 `test_calibration_never_loses_to_the_naive_formula`（逐目标断言标定不劣于朴素公式）
- **验证后实测：** target 2/4/8/12/16/24px 的成品 ratio 由 0.81–1.00 提升为**全部 1.000**；三个变异（标定/泄漏门禁/可复现性）全部 KILLED
- **状态：** RESOLVED（有测试 + 变异双重护栏）

## ISSUE-023 接触阴影盖在物体自己身上 + 白羽球渲染得比背景还暗（**RESOLVED 2026-09-14**）
- **现象（两个独立缺陷，均由分尺寸桶目视审计发现）：**
  1. **阴影自遮挡：** 接触阴影用固定偏移 `np.roll(alpha, (6, 8))` + σ=4 模糊，再对整幅图相乘 —— 对数像素的小目标，阴影比物体本身还大且**盖在物体上**。实测 `shadow_gain=0.319` 使物体自身亮度下降 **26.1%**
  2. **白羽球偏暗：** `AMBIENT=0.34` 使完全背光的羽毛面只有 **80/255**，**比它所处的中间调背景更暗** —— 与"白羽球是球场上最亮的物体"的物理事实相反
- **危害：** `3–7 px` 区间（占训练集 42%）的样本呈现为暗斑而非羽毛球；小目标正是本项目的关键区间
- **处置：** ① 阴影按物体尺寸缩放（偏移 `clip(0.8×extent, 1, 8)`、模糊 `clip(0.35×extent, 0.6, 4)`）并乘以 `(alpha <= 0)` **禁止落在物体自身**；② `AMBIENT` 0.34→0.56、`KEY` 0.66→0.44，使最暗白羽面 ≥ 128（中间调）
- **护栏：** 新增 `test_contact_shadow_never_darkens_the_shuttle_itself`（物体像素与无阴影逐位相同、且背景上阴影仍可见）与 `test_the_darkest_white_feather_face_is_not_darker_than_mid_grey`（RED 实测 80.1 < 128）
- **修复后目视复验：** 6.9/9.5/13.4/19.8 px 样本均为明确发亮的羽毛球（修复前 13.4 px 为暗块）
- **状态：** RESOLVED（有测试护栏）

## ISSUE-021 训练背景池未做目视核验：混入非照片、全黑帧与**含真实羽毛球**的图片（**RESOLVED 2026-09-14**）
- **现象：** P4 训练背景池（`bg_train` 32 + `bg_val` 7，2026-09-14 建立）**从未经过 P0 式逐张目视核验**，实际混入 7 张不合格图：
  - **非照片 4 张**：`tbg_001.png`/`tbg_003.png`（球场线条示意图）、`tbg_002.png`（球场 SVG 渲染图）、`tbg_017.jpg`（赛程/文字表格）
  - **全黑帧 2 张**：`tbg_018.jpg`、`tbg_005.png`（透明源图铺平为黑底，实测 mean=0.0 / std=0.0）
  - **含真实羽毛球的照片 1 张：`tbg_006.jpg`** —— 放大核验可见 **9+ 个真实羽毛球**（黄色裙羽 + 白色球头）散落绿色球场
- **危害：** `tbg_006.jpg` 属**标注污染**：管线只给合成的那一个球写标签，背景中的真球成为**无标注的真目标** → 训练会教检测器「忽略真实羽毛球」，并使后续假阳性/漏检统计系统性失真。非照片与全黑帧则让样本不代表真实域，与 P0 阶段「剔除 14 张非照片」的自身标准自相矛盾
- **处置：** 7 张全部移入 `train_data/bg_excluded/`；`bg_train_val_metadata.csv` 对应行 `split` 改为 `bg_excluded` 并逐条写明理由。新增两个**前置硬门禁**（失败即非零退出、不渲染）：`find_blank_backgrounds()`（空白/单色/不可读帧）与 `find_leaked_backgrounds()`（与冻结测试集同源，按内容而非文件名比对）
- **流程教训：** P3 的 30 张冻结背景当初做了目视核验并写进报告，训练池却没有 —— **凡进入数据管线的图像，无论训练集还是测试集，都必须过同一道目视核验**，且核验必须留证据（接触表 + 结论），不能只靠"来源看起来干净"
- **残余风险（显式登记）：** 剩余 25+6 张背景的核验尺度相当于 300px 缩略接触表；`tbg_006` 的真球在该尺度下**可见**（说明该方法有效），但未做全分辨率逐张复核。后续应补一次全分辨率核验再进入正式训练
- **状态：** RESOLVED（污染项已隔离 + 门禁已自动化；全分辨率复核待补）

## ISSUE-024 渲染器自带羽毛球尺寸常数，且无人校验外部视觉体与项目模型是否一致（**RESOLVED 2026-09-14**）
- **现象：** `tools/shuttle_render.py` 把尺寸标定种子写成**硬编码实测值** `SHUTTLE_LENGTH_M = 0.0778`（从导入 GLB 包围盒量得）。这正是本仓库登记过的失败模式（ISSUE-013：L=6.5 m 在三个模块各有一份副本）
- **触发：** 用户质疑「不是有羽毛球模型吗」→ 复查确认项目**确有自建模型**：`tools/build_shuttlecock.py` + `configs/shuttlecock.yaml`（16 羽毛、BWF 尺寸、软木半球凸包 + 裙部开口锥壳、质量分布、气动参数）
- **关键澄清（避免误判）：** 该模型是**物理资产**，其视觉体按项目自身策略必须是外链，因此用 GLB 渲染**不是绕过模型**：
  - `configs/shuttlecock.yaml`：`visual.mode: EXTERNAL_REFERENCE`、`local_asset_path: assets/third_party/shuttlecock_visual.usd`（即导入 GLB 的提取物）
  - `build_shuttlecock.py:136-137`：`visual.mode` 不是 `EXTERNAL_REFERENCE` 就**直接抛错**
  - `build_shuttlecock.py:337`：程序化网格 `MakeInvisible()` —— 是**碰撞几何，不可见**
  - `build_shuttlecock.py:416-433`：程序化视觉仅为 `TEMP_ProceduralDebug`（球+锥），需显式开关，注释要求 final 前替换
- **真实缺陷（已修）：** ① 渲染器不应自带尺寸常数 → 改为 `load_shuttle_spec()` 经项目自有的 `load_config`/`build_physics_description` 读取配置，`canonical_length_m()` 缓存；② **此前没有任何检查确认外部视觉体与项目 BWF 模型尺寸一致** —— 一旦换掉视觉体，所有样本会被静默错误定标
- **证据：** 新增 `ShuttleSpecTests`（3 项）。实测导入视觉体与配置模型高度一致：**裙径 61.9 mm vs 61.8 mm（差 0.16%）**、总长 77.8 mm vs **79.25 mm（差 1.83%）**，容差 5%
- **连带影响（显式登记）：** 种子由 0.0778 → 0.07925（+1.86%），实测约 27% 样本最终落点差 **±1px**（最大 1.006 px，小尺寸端量化格）→ 为保持「manifest 记录可逐位复现」契约，**数据集已重新生成并复验**
- **状态：** RESOLVED（有测试护栏；数据集按新种子重生成）

## ISSUE-025 记录把 `target_px` 舍入到 3 位小数，破坏逐位回放（**RESOLVED 2026-09-14**）
- **现象：** `render_sample()` 记录 `target_px` 时用了 `round(..., 3)`。但 `target_px` 是**回放输入**（驱动标定循环），与已修的 `shadow_gain`/`noise_sigma`/`motion_angle_deg`（ISSUE-019）同类，上一轮修复漏了它
- **影响量化：** 1e-4 px 的差异会翻转标定的停止点，从而改变一个像素的覆盖。实测 `train_00399` 用**记录值**回放差 **0.000103**，用**精确值**回放差 **0.000000** → 图像本身正确，错的是记录精度
- **为何测试没抓到：** 既有的 `test_sample_record_reproduces_the_exact_image` 只回放 `composite()` 的参数，没有从记录**整样本重渲染**，因此覆盖不到 `target_px`
- **处置：** ① `target_px` 改为精确存储；② 新增 `test_sample_record_stores_replay_parameters_exactly` 与 `test_replaying_a_sample_from_its_record_is_bit_exact`（从记录值整样本重渲染并断言逐位相同）
- **已交付数据集的修复方式（不重渲染）：** `target_px` 由 CLI 的确定性配方（`default_rng(seed).lognormal(log 9, 0.5, n)` 后 clip）可精确重导 → 回填 train 397/400、val 120/120 行；回填后 5 个抽样（含 `train_00399`）回放**全为 0.000000**，数据集验证恢复 `ALL CHECKS PASSED`
- **状态：** RESOLVED（有测试护栏；记录已回填并复验）
## ISSUE-026 Stage B 从 Stage A 续训时学习率过高，破坏已收敛参数（**FAILED_HIGH_LR_STAGE_B**）
- **日期：** 2026-09-27
- **模块：** 训练 / 优化器超参（Stage B 主训练）
- **现象：** 从 Stage A `best.pt` 续训 Stage B（全网络解冻、`lr0=0.01`）后，前 3 个 epoch 指标**持续坍塌**：mAP50 0.337 → 0.437 → 0.226，Recall 0.496 → 0.390 → 0.189，`train/box_loss` 连续上升 1.455 → 1.575 → 1.660，均远低于 Stage A 基线（mAP50 0.871 / Recall 0.760 / box_loss 1.431）。
- **失败实验（已保留，禁止覆盖或删除）：**
  - run id：`gate6B_fromA_imgsz1024_b16_w8_20260927-130317`
  - 路径：`/home/T7/ojh/robot_sim/runs/shuttle_yolo26_v1/gate6B_fromA_imgsz1024_b16_w8_20260927-130317`
  - 内含 `results.csv`（3 行）、`best.pt`(58.9 MB)、`last.pt`(58.9 MB)、`resolved_config.yaml`、`args.yaml`
- **错误日志（`results.csv` 原文，epoch/time/box/cls/l1/P/R/mAP50/mAP50-95/val_box/val_cls/val_l1）:**
```
1,4640.85,1.45492,0.75779,0.00182,0.31375,0.49571,0.33691,0.1995,1.53891,2.32459,0.00247
2,9065.47,1.57457,0.98920,0.00200,0.67154,0.39049,0.43652,0.26384,1.39498,2.31767,0.00209
3,12938.8,1.65959,1.16127,0.00213,0.80622,0.18896,0.22574,0.13940,1.40337,2.85320,0.00211
```
  记录到的 lr（8 个参数组）：epoch1 `0.00999522 / 0.00333174`（交替）、epoch2 `0.0197124 / 0.00657081`、epoch3 `0.0291468 / 0.00971559`。
- **对照基线校正（避免混用不同 epoch 的指标）：** Stage A `best.pt` 的 `train_metrics.fitness = 0.52278`，与 `results.csv` 中 **epoch 6** 的 mAP50-95 完全一致 → **Stage A 基线 = epoch 6**：
  box_loss 1.43067 / cls_loss 0.66014 / P 0.94111 / R 0.76012 / mAP50 0.871 / mAP50-95 0.52278 / val_box 1.23579 / val_cls 0.92225 / val_l1 0.00178。
  后续一切 Stage B 对照都只用 epoch 6 这一行，不再用 Stage A 的 epoch 7/8。
- **原因（2026-09-27 用源码与实测 LR 审计查明，第一版判断只说对了一半）：**
  1. **`optimizer=auto` 会完全忽略 `lr0`**。`ultralytics/engine/trainer.py:1137-1146`：`if name == "auto": LOGGER.info("...ignoring 'lr0=...' and 'momentum=...'")`；随后 `lr_fit = 0.002*5/(4+nc)`，并 `name, lr, momentum = ("MuSGD", 0.01, 0.9) if iterations > 10000 else ("AdamW", lr_fit, 0.9)`。
  2. 70 epoch × 2,091 step = **146,370 iterations > 10,000** → auto 选 **MuSGD，lr=0.01**（与我配置的 `lr0` 无关，数值上碰巧相同）。
  3. MuSGD 路径还会给检测头（`cv3`/`one2one_cv3`）**学习率 ×3**（`trainer.py:1195`：`{"params": p1, **x, "lr": lr * 3}`），并把参数组拆成 8 组 → **头部分组的 LR = 0.03**，实测 `results.csv` 的 `lr/pg0` 在 epoch 3 达到 **0.0291468**，与该推算一致。
  4. 于是真实情况是：**全网络解冻 + 检测头 0.03 的 MuSGD 更新**，把 Stage A 已收敛（mAP50 0.871）的权重直接打散。
  5. 另一个必须记录的事实：3 epoch 的 smoke 若仍用 `auto`，因为 iterations = 6,273 < 10,000 会走 **AdamW lr=0.002** —— **与真实 70 epoch 运行不是同一条代码路径**，用这种 smoke 验证 LR 修复是无效的（本次已作废重跑）。
- **解决方案：**
  ① `stages.B.lr0: 0.01 → 0.001`；
  ② **`run.optimizer: auto → MuSGD`（显式固定）**——否则 lr0 依然被忽略，这是让 lr0 生效的必要条件；`run.warmup_bias_lr: 0.0`（与 auto 分支原本强制的值一致）；
  ③ 本轮相对失败实验**只改「有效学习率」这一个变量**（优化器家族仍是 auto 当初选的 MuSGD），imgsz(1024)/batch(16)/workers(8)/dataset/sampler/augmentation/Stage A checkpoint/模型结构全部不变；
  ④ 先跑 **3 epoch smoke** 验收（且必须显式 MuSGD，保证与 70 epoch 走同一路径），通过后再续剩余 epoch；
  ⑤ 训练器新增 LR/内存审计回调：记录 `lr_config.json`（warmup_epochs / warmup_bias_lr / scheduler / optimizer / 每个参数组的 lr·initial_lr·weight_decay·张量数）与 `lr_probe.jsonl`（逐 step 记录所有参数组 LR）、`mem_probe.jsonl`（MemAvailable / SwapFree / cgroup / RSS），并在 `MemAvailable < 5 GB` 时自动停止训练。
- **修改文件：** `configs/shuttle_detection/yolo26_p2_v1.yaml`（`stages.B.lr0`）、`tools/train_yolo26_v1.py`（新增参数组 LR 审计回调与内存护栏）
- **验证方法：** 3 epoch smoke 的 7 条验收 —— ① `box_loss` 不连续明显上升；② Recall 不出现 0.75→0.5→0.2 级坍塌；③ mAP50 不长期低于 Stage A 的一半（<0.435）；④ 所有 pg 的实际 LR 符合预期（任何组都不得升到 0.01/0.02/0.03 以上）；⑤ 无 NaN；⑥ 权重确实从 Stage A `best.pt` 加载；⑦ backbone 确实已解冻（`freeze=0` 且 backbone 参数 `requires_grad=True`）。
- **是否彻底解决：** 否（待 3 epoch smoke 验收；若 smoke 通过才续训剩余 Stage B）
- **相关 commit：** 无（本次改动尚未提交）

## ISSUE-027 `hard_negatives2/excluded.txt` 的排除名单被数据集构建器忽略，7 张图仍进入训练集
- **日期：** 2026-09-28
- **模块：** 数据集构建 / 数据卫生（V1 统一清单）
- **现象：** 困难负样本评估（`outputs/shuttle_capability/reports/HARD_NEGATIVE_EVAL.md`）核对清单身份时发现，`outputs/shuttle_capability/hard_negatives2/excluded.txt` 里明确排除的 7 张图，**在 `v1_dataset_manifest.csv` 中全部存在且 `split=train`**，即它们参与了 Stage A 与 Stage B 的训练。
- **证据（清单 30,321 行逐行匹配）：**
```
hn2_010.jpg -> manifest source=hard_negative split=train location=hard_negatives2
hn2_011.jpg -> manifest source=hard_negative split=train location=hard_negatives2
hn2_012.jpg -> manifest source=hard_negative split=train location=hard_negatives2
hn2_013.jpg -> manifest source=hard_negative split=train location=hard_negatives2
hn2_015.jpg -> manifest source=hard_negative split=train location=hard_negatives2
hn2_016.jpg -> manifest source=hard_negative split=train location=hard_negatives2
hn2_017.jpg -> manifest source=hard_negative split=train location=hard_negatives2
excluded-but-trained: 7
```
  复现：`python tools/audit_neg_membership.py`（`PYTHONUTF8=1`）。
- **原因：** `tools/build_yolo26_v1_dataset.py` 只按**冻结目录**硬排除（`controlled_capability / challenge_test / synthetic_on_real_bg / synthetic_3d / real_images / real_video / real_match_frames / real_train`），**从未读取 `excluded.txt`**。建立 `excluded.txt` 的那次人工筛图与清单构建是两条独立流程，没有接线。
- **影响评估：** ① 结论层面：`hard_negative` 集合“已被训练”的判定不变（96 张 100% split=train）；② 程度层面：困难负样本上 Stage A→Stage B 的 FP 从 2→0 只能算记忆性证据，且被这 7 张**本不该出现**的图污染（其中 `hn2_056`/`hn2_057` 恰是 Stage A 仍有误检的两张）；③ 泄漏层面：这 7 张不含正样本标注，不构成正样本泄漏，只影响“困难负样本”这一集合的定义纯度。
- **解决方案：** 在 `tools/build_yolo26_v1_dataset.py` 中新增 `excluded.txt` 读取（按文件名过滤，缺失文件即报错，不静默），然后重建 manifest/digest/sampler。**注意：这会改变已冻结清单 sha256，必须等 Stage B 训练结束之后再执行**，不得在训练途中改动清单（当前 Stage B 仍在跑，PID 3839453）。
- **修改文件：** 待改 `tools/build_yolo26_v1_dataset.py`；新增证据脚本 `tools/audit_neg_membership.py`；新增报告 `outputs/shuttle_capability/reports/HARD_NEGATIVE_EVAL.md`
- **验证方法：** 重建后重跑 `tools/audit_neg_membership.py`，要求 `excluded-but-trained: 0`；并比对新的 manifest 行数（应为 30,321 − 7 = 30,314，若允许与正样本重复计数则以实际为准）。
- **是否彻底解决：** 否（已定位、未修；受 Stage B 训练约束，排在训练结束后）
- **相关 commit：** 无

## ISSUE-028 本地推理环境 `D:\_yolo26v1` 被外部删除（torch/ultralytics 全失），本地评估一度不可执行
- **日期：** 2026-09-29
- **模块：** 环境 / 本地评估链路
- **现象：** Stage B 训练结束后准备做三方对比时，`D:\_yolo26v1\venv\Scripts\python.exe` 报“不会被识别为 cmdlet/可执行程序”；`Test-Path` 为 False，`dir /a /b D:\` 无任何 `yolo*` 条目，`E:` 亦无同名目录 → **整个 `D:\_yolo26v1`（venv + 解包的 ultralytics wheel `_wheel_extract` + `_deps` + `YOLO_CONFIG_DIR=cfg`）被外部删除，不是改名、不在回收站**。
- **同时确认未受影响：** `D:\_eth_dl\ckpts\*`（4 个权重）、`D:\_eth_dl\venv`（ETH 下载用，仅 numpy/PIL/pyyaml）、`D:\_eth_data\eth_shuttle_detection`、仓库 `outputs/**` 全部完好；`D:` 现有 23.9 GB 空闲。
- **影响：** 本地无法做任何 ultralytics 推理（尺寸分桶评估、困难负样本 FP 复算、Gate 7 的 640/960/1280 对比全部受阻）；`D:\_yolo26v1` 下的本地脚本（`launch_*.ps1`、`run_*.ps1`）一并丢失。
- **原因：** 非本会话所为（本会话对本地只做读取与仓库内写入）；最可能是机器清理/迁移。**无法从本地文件系统取证删除者** → 记录为“外部删除”。
- **临时绕行尝试（据实记录）：** ① 改用远端 GPU + 远端 venv `env_isaaclab/bin/python`：`import torch` 在 `timeout 900` 内**未完成**（`/dev/vdb2` 84% 占用、loadavg 13.9、GPU 上另有 3 个他人进程 24.2 GB），远端路径不可靠；② 远端 `v1_dataset` 清单已核对与本地**同 sha256**（`11d85026…`）且负样本目录齐备（36+60+30），故远端资产本身没问题，瓶颈只在解释器导入速度。
- **解决方案：** 重建**本地**评估环境 —— `D:\_eval26\venv`（Python 3.11）：先装 `ultralytics==8.4.150 + polars`（PyPI），**再**装 CUDA 版 `torch/torchvision --index-url https://download.pytorch.org/whl/cu126`（顺序不能反，否则 ultralytics 的依赖会拉入 CPU 版 torch）；装完用 `torch.cuda.is_available()` + `get_device_name(0)` 验收（本机 GPU = RTX 4060 Laptop 8 GB，之前报告里写的 “A6000” 是本机标注错误，已在 `HARD_NEGATIVE_EVAL.md` 更正）。脚本：`D:\_eval26\setup_eval_env.ps1`，日志 `D:\_eval26\setup.log`。
- **经验教训：** 本地评估环境必须**可重建**——把安装配方（含精确版本与索引 URL）写进仓库脚本，而不是只留 venv；本轮重建前，唯一记录在案的配方只存在于已删除目录的 `launch_*.ps1` 里。
- **是否彻底解决：** 是（重建脚本已落盘并可重跑）；但重建完成前，所有本地评估处于阻塞状态
- **相关 commit：** 无
## ISSUE-029 固定评估集被写成“从未参与训练/调参/模型选择”，与事实不符（文档口径缺陷）
- **日期：** 2026-10-01
- **模块：** 文档 / 评估口径（不影响代码与权重）
- **现象：** `outputs/shuttle_capability/reports/FROZEN_TEST_EVAL.md` 第 5 行写“8 个冻结集合**从未参与训练、调参或模型选择**；本报告是它们的首次使用”，`STAGE_B_FINAL_REPORT.md` 与 `docs/EVAL_FROZEN_TEST.md` 也有“从未参与训练/调参的冻结测试集”这类表述。
- **原因：** ① 措辞沿用早期“冻结测试集”的说法，把“评估时未参与梯度更新”误写成“未参与模型选择”——实际上这些集合上的数字被反复用于 Stage A/B 的取舍（A best vs B best@e10 的判定本身就是用它们做的）；② `management/ISSUES.md` ISSUE-027 已证明 `hard_negatives2/excluded.txt` 的 7 张图实际进入了训练，隔离并非 100%；③ 集合被多次复看后，“首次使用”“完全未见过”的强断言不再成立。
- **影响：** 若后续（如 Stage C 调参、置信度阈值选点）继续引用这些集合并称其为“最终测试集”，同一批数据会被同时用于选择与判定，结论无法与选择偏差区分。
- **解决方案：** 统一改为 **fixed evaluation set** / **development holdout**；新增 `docs/LOCALIZATION_AUDIT.md` §7 作为术语口径来源；在 `FROZEN_TEST_EVAL.md`、`STAGE_B_FINAL_REPORT.md`、`docs/EVAL_FROZEN_TEST.md` 顶部插入**不改写历史**的更正说明；明确：**若 Stage C 依据这些集合上的任何结论调参或选点，最终判断必须另用一个全新的、未被看过的 holdout。**
- **修改文件：** `docs/LOCALIZATION_AUDIT.md`（新增）、`outputs/shuttle_capability/reports/FROZEN_TEST_EVAL.md`、`outputs/shuttle_capability/reports/STAGE_B_FINAL_REPORT.md`、`docs/EVAL_FROZEN_TEST.md`、`management/ISSUES.md`
- **是否彻底解决：** 是（口径已更正并落盘；历史报告保留原文 + 更正说明）
- **相关 commit：** 无
## ISSUE-030 train 与 val 的 GT 框高度/宽高比规范不一致，导致 val 上的高度偏差被误读为模型问题
- **日期：** 2026-10-01
- **模块：** 数据集 / 标注规范（评估口径已排除）
- **现象：** val 上三个 checkpoint 的 pred_h/gt_h 均值一致地 ≈ **1.092**，而 pred_w/gt_w ≈ 1.000；即"预测框高度系统性偏大 9.2%"。表面像模型回归偏差或评估坐标 bug。
- **排查：** 见 reports/BBOX_HEIGHT_BIAS_ROOT_CAUSE.md。坐标链路（GT 解码 / letterbox / scale-back）回环误差 1.14e-13，与安装版 ultralytics 的参数与回代函数差 0.0，且 gain_x == gain_y（各向同性，无法只差高度）；原生 vs 缓存逐框 ≤0.303 px；AP 口径与既有报告逐位一致 → 评估链路无 bug。
- **原因：** **同一宽度桶内，val 的 GT 框比 train 更扁更矮**（1024 输入系中位高）：[14,20) 13.27→12.27、[20,28) 17.60→15.47、[28,40) 23.34→18.67 px；aspect 1.286→1.400 / 1.323→1.483 / 1.360→1.760。模型复现 train 规范，故在 val 上高度多 1.3 px（对 8–16 px 目标即 +9%～+11%）。与规范一致的来源（eth_iphone 1.020、synthetic 1.015）无此偏差，不一致的 eth_main（96% val GT）为 1.095 → 是数据侧规范差异，而非模型/评估。**叠加** aspect 回归到均值（GT aspect 0.677→预测 0.801、1.960→预测 1.521）。
- **影响：** val 上的 height/AP90/AP95 结论会被这一项系统性偏差污染：counterfactual 显示只纠正高度整轴可把 a_best AP90 0.090→0.515、AP95 0.009→0.203（诊断值，非正式指标）。任何"提高高 IoU 性能"的方案若不先统一规范，都会在错误的基准上比较。
- **解决方案（本轮不做，仅登记）：** ① 追查 train/val 标注规范差异来源（同一 eth_main 内部，疑与子集标注来源/COCO 转换有关），统一规范后重导 val；② 在统一规范前，val 上的 AP90/AP95 对比需附带本偏差说明。**本轮纪律禁止修改标签**，故只登记不改动。
- **修改文件：** tools/audit_bbox_height_bias.py、tests/test_bbox_coordinate_roundtrip.py、outputs/shuttle_capability/metrics/bbox_*.csv、outputs/shuttle_capability/reports/BBOX_HEIGHT_BIAS_ROOT_CAUSE.md、management/DAILY_LOG.md
- **是否彻底解决：** 否（已定位并量化，处置方案待定；标签未改动）
- **相关 commit：** 无
## ISSUE-031 我们的 val split 有 62.7% 图像就是 ETH 官方模型的训练帧（第三方基线评估的泄漏风险）
- **日期：** 2026-10-01
- **模块：** 数据集 / 第三方基线评估
- **现象：** 用 ETH 官方 `best.pt` 在我们的固定集合上评估时，val 取得 R 0.9288 / mAP50-95 0.7766，远高于其在 controlled_capability（0.0024 / 0.0123）与 challenge_test（0.0000 / 0.0072）上的表现，值得怀疑。
- **排查：** 依据 ETH 官方每档 yaml 的 `train: images/train` 与其 config.json 的 12 个训练 location + `diff_levels.train=[easy, medium]` 做**文件级**归属判定（tools/eval_eth_official_baseline.py 的 leakage_classify）。结果：val 4,413 图中 **A 类 2,765 张（62.7%）位于 ETH 训练目录**（ml_6 easy 674 + medium 449、ml_3 easy 606 + medium 398、uetlibergstrasse_1 easy 591 + medium 47），B 类 263、coco_val_easy 1,235、外部 150。
- **原因：** 我们的 v1 数据集在构建时直接从 ETH 公开数据集各 location/difficulty 目录取样，未按“是否属于 ETH 官方训练档”过滤；而 ETH 官方仓库的训练/验证划分是按 location+difficulty+子目录（images/train|val）定义的，两者口径不同，导致大量 ETH 训练帧进入我们的 val。
- **影响：** ① ETH 模型在 val 上的任何指标**不能作为其泛化能力**（本质含训练集记忆，A 类子集 R 0.9978）；② 我们的 val 也不能用作“外部模型公平比较集”；③ 若用 val 比较我们与 ETH，会严重高估 ETH、低估我们。
- **解决方案（本轮只登记，不改数据）：** ① 第三方模型评估统一使用 `eth_unseen` 口径（剔除 A 类；本轮已产出 `val|eth_unseen` 行，495 GT）与两个**完全外部**的合成集；② 后续如需真实域外部评估集，应新建独立于 ETH 训练档的固定集合（我们的 iphone/自采帧 + 合成-on-real-bg），并在 manifest 里显式记录 provenance（location/difficulty/subdir 或来源 URL）以便复核；③ 不要把 val 称为 untouched/final test。
- **修改文件：** tools/eval_eth_official_baseline.py、tests/test_eth_eval_parity.py、outputs/shuttle_capability/metrics/eth_data_leakage_audit.csv、eth_data_leakage_inventory.csv、reports/ETH_OFFICIAL_BASELINE_AUDIT.md、management/DAILY_LOG.md
- **是否彻底解决：** 否（已定量定位；评估口径已切换为 eth_unseen，数据集本身未改动）
- **相关 commit：** 无

## ISSUE-032 ETH-only V1 数据集冻结的三处产物缺陷（上报字段、尺寸统计、负样本抽样）
- **日期：** 2026-10-03
- **模块：** 数据集构建 / 实验产物
- **现象：** 首次全量冻结（远端 `~/.dsh-bench/eth_only_v1_*`）audit 显示 PASS，但逐字段核验发现三处不一致：① `val_locations` 列出 `uetlibergstrasse_1`，而 `val_location_counts` 只有 `glc_2 + uetlibergstrasse_2`；② `size_distribution.csv` 里 13,992 张正样本全部落在 `<4`，`median_equiv_size_640` 恒为 0.333；③ 重跑时 train 从 14,543 涨到 19,493、负样本从 551 涨到 5,501。
- **原因：** ① `build()` 把"被选中的 location 组合"当作 val locations 上报（候选池排除评估帧后该 location 已 0 帧，但名字仍在组合里）；② `inventory_eth_dataset` 用 `equiv_size_640(1, 1, W, H)` 占位（因为逐图箱尺寸未知），使该列恒为 `sqrt(1*1)*640/max(W,H)`；③ 负样本抽样只由 `--negative-fraction` 触发，`build()` 不读 recipe 的 `official_fraction_coco_train`，CLI 漏传即静默退化为全量 5,500。
- **影响：** ②使尺寸分布完全失真（会误导后续按尺寸分桶的性能比较）；③会以 10 倍负样本训练出一个与计划配方不同的模型（负样本比例 0.1→0.28），且 audit 仍是 PASS 不会报警；①只是上报字段错误，拆分本身正确（val 2,920 / 0.17266 三次一致）。
- **解决方案：** ① 改为从实际 val 帧推导 `val_locations`；② 按标签框逐框计算 `equiv_size_640(w_px,h_px,W,H)`，每图新增 `min/median/max_equiv_size_640` 与 `equiv_sizes`，尺寸分布改为**箱级**统计；③ `negative_fraction` 缺省时按 recipe `official_fraction_coco_train` 取值并记录 `negative_fraction_source`；重跑脚本加硬断言（train/val/coco/val_locs）确保不再静默漂移。
- **修改文件：** tools/build_eth_only_v1_dataset.py、tests/test_build_eth_only_v1_dataset.py、tools/remote/run_build_v1b.sh、tools/remote/verify_v1_artifacts.py、tools/remote/check_local_artifacts.py、data/eth_only_v1*、outputs/shuttle_capability/metrics/eth_only_v1_{data_audit.json,size_distribution.csv}
- **验证方法：** `python tools/remote/check_local_artifacts.py`：audit PASS、`val_locations` 与 val manifest 实际 location 集合一致、箱级桶计数合计 **13,992** = 正样本数、negatives 551（coco 550）、16 项 builder 测试全绿（含新增的 recipe 负样本比例测试）
- **是否彻底解决：** 是（产物已重生成并提交；错误版本保留在远端两个备份目录，未删除）
- **相关 commit：** 52cd0dc

## ISSUE-033 ultralytics 把 run 目录嵌套在 runs/detect/<project> 下，按 project/name 读 results.csv 会静默丢失选点
- **日期：** 2026-10-03
- **模块：** 训练启动器 / 实验产物
- **现象：** 3 epoch smoke 的日志显示 `Logging results to /home/dgut/.dsh-bench/runs/detect/runs_eth_only_v1/smoke_e3`，而启动器按 `<project>/<name>` 去找 `results.csv` → 训练完成后选点记录只会写成 `{"error": "results.csv missing after training"}`。
- **原因：** 当 `project` 是相对路径时，ultralytics 会把它并到默认根 `runs/detect/` 之下（`get_save_dir`），实际保存目录并不等于 `project/name`。
- **影响：** 若未发现，50 epoch 全量训练后"只用内部 val mAP50-95 选点"这一步会退化为无记录（best.pt 仍有，但选点证据、best epoch、recall 全部丢失），且不会报错——正是本轮契约最在意的失败模式。
- **解决方案：** 启动器改为从 trainer 对象读取权威 `save_dir`（`resolve_save_dir(project, name, model)`），并把 `save_dir` 写进 manifest；新增 2 项测试。
- **修改文件：** tools/train_eth_only_v1.py、tests/test_train_eth_only_v1.py
- **验证方法：** `pytest tests/test_train_eth_only_v1.py` 46 项全绿；smoke 重跑后 manifest 的 `save_dir`/`selection` 与日志中的实际目录一致
- **是否彻底解决：** 是（`integration_e1c_manifest.json` 的 `save_dir`/selection 与实际运行目录一致）
- **相关 commit：** f2bed61、a7ce718

## ISSUE-034 训练启动器三处缺陷（非法参数、W&B 自动上传、选点列名不匹配），由 smoke+集成跑逐个暴露
- **日期：** 2026-10-03
- **模块：** 训练启动器
- **现象：** ① 1 epoch 集成跑直接报 `'weight' is not a valid YOLO argument`；② 3 epoch smoke 日志出现 `wandb: uploading artifact ...`（本机从未申请 W&B）；③ 去掉 weight 后，训练跑完 1 epoch 却在选点处抛 `results.csv ... has no 'mAP50-95' column`。
- **原因：** ① `build_train_args` 误传 `weight=<path>`（模型已由 `YOLO(weights)` 加载，无需该键）；② ultralytics 8.4.150 的 `cfg/default.yaml` 与 `cfg/__init__.py` **根本没有 `wandb` 键**——集成由回调自动注册，唯一可靠的关闭方式是环境变量，故 `wandb=False` 同样非法；③ 契约写 `selection_metric: mAP50-95`，而 ultralytics 的结果列名是 `metrics/mAP50-95(B)`，两者字面不等。
- **影响：** 若不修：50 epoch 全量会在**最后一步选点**失败（best.pt 仍在但选点证据丢失，正是本实验契约最在意的失败模式）；且训练产物会被上传到第三方 W&B 服务（离线可复现性与数据泄漏风险）。
- **解决方案：** 移除 `weight=`；改为 `disable_third_party_loggers()`（`WANDB_MODE=disabled`/`WANDB_DISABLED=true`/`COMET_MODE=DISABLED`）并把生效值写入 manifest 的 `third_party_loggers`；`metric_aliases()` 把两种写法归一到同一集合，manifest 记录 `selection.column`；新增 2 项测试：**所有 train kwargs 必须存在于 ultralytics `cfg/default.yaml` 的键集合**（fixture 与真实 contract+recipe 各一次），专防此类"跑到远端才炸"的拼写/键名错误。
- **修改文件：** tools/train_eth_only_v1.py、tests/test_train_eth_only_v1.py、tools/remote/run_smoke_v1.sh、tools/remote/run_full_v1.sh、tools/remote/run_integration_v1.sh
- **验证方法：** `integration_e1c_manifest.json`：`save_dir` 取 trainer 权威路径、`selection.column = metrics/mAP50-95(B)`、`value 0.60435` @epoch1（recall 0.82232）、`gpu_cap.fraction 0.5063`（设备 47.4 GiB）、`diagnostic_reasons ["epochs 1 != contract 50"]`；同一日志 `grep -c wandb` = **0**（修复前的 launch 有大量上传行）；启动器测试 54 → 58 项全绿
- **是否彻底解决：** 是（四类缺陷全部修完并有测试/产物双证据）
- **相关 commit：** a7ce718、8d834fc、ef313d8


