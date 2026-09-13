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