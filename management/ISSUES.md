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