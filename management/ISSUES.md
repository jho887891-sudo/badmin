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