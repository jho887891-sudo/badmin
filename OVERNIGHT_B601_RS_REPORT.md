# OVERNIGHT B601-RS REPORT (2026-10-18)

起点提交 9d08a53。本轮解决了一个真实阻塞（F1：整机场景无法保持 rest pose），并完成了资产依赖
审计与可移植模型包。所有结论都来自实际运行，没有凭推断下结论的条目。

分支隔离：F1 的两个修复提交留在 feature/shuttle-detection（项目的日常工作分支），随后新开
feature/rebot-rs-sim-asset 承接资产包工作。另一会话在途的文件（deliverables/rebot_b601dm/*、
management/rebot_b601dm/DELIVERABLES.md、scripts/simulation/render_shuttle_3d_probe.py、
outputs/three_d_render_probe/、outputs/yolo26s_speed_opt/render_manifest.json）全程未 stage、未改动。
没有推送任何分支。

---

## 1. 根因及证据

### 结论

不是刚度、不是重力、不是自碰撞、不是关节限位、也不是步进顺序。根因是**装配路径**：当 B601-RS 通过
InteractiveScene 注册进场景时，PhysX 对这些关节不施加任何驱动扭矩，整臂表现为被动链；6 kg 机械臂
在重力下瘫落，joint3 被压到 0 以下（软限位让到 -0.067 rad），末端落到安装面下 0.19 m。同一份 USD 用
普通 Articulation 绑定则跟随目标到 1e-2 rad。

### 隔离矩阵（每行都是一次真实运行，1200 步 = 5 s @240 Hz）

| 装配 | 驱动写入 | 结果 |
| --- | --- | --- |
| 手工 Articulation，/World/Robot | 运行期 k=400 c=10 | **保持，0.0103 rad** |
| 手工 Articulation，/World/envs/env_0/Robot/RebotB601RS | 运行期 k=400 c=10 | **保持，0.0103 rad** |
| InteractiveScene，仅机器人 | 运行期 k=400 c=10 | 1.2867 rad 塌落 |
| 同上 + replicate_physics=False | 运行期 k=400 c=10 | 1.2867 rad（五位小数全同）|
| 同上 + scene.reset() | 运行期 k=400 c=10 | 1.7217 rad |
| 同上，去掉 spawn 属性覆盖 | 运行期 k=400 c=10 | 1.7217 rad |
| 同上，拍平自包含 USD（103 MB）| 运行期 k=400 c=10 | 1.2867 rad |
| 同上，stiffness 扫 100/200/400/800 | 运行期 | 1.42 / 2.89 / 1.84 / 2.74 rad（非单调）|
| 同上，重力关闭（rigid_props）| 运行期 | 与重力开完全相同 |
| 同上，自碰撞开启 | 运行期 | 与关闭相同 |
| 同上，三种 write/update 组合 | 运行期 | 完全相同，1.2867 rad |

### 支撑事实（都是读出来的）

* 实时 stage 上有 8 个 drive 属性、**唯一** articulation 根 /World/envs/env_0/Robot/RebotB601RS/Geometry；
* 资产里驱动的 authored 值为 stiffness 1.7453 / damping 0.0175（导入器把请求的 100 / 1.0 除以 57.2958），
  targetPosition 0；重转换时传 5730 / 573 会精确落成 100.007 / 10.0007，证实这是单位换算而非随机值；
* write_joint_stiffness_to_sim / write_joint_damping_to_sim / set_joint_position_target / _initialize_impl
  四种写入都改变不了场景结果；
* 结果对增益、目标、属性覆盖、复制开关、reset、资产拍平、步进顺序全部不敏感（五位小数一致），只有
  "改用直接装配"这一项能改变结果。

### 修复（最小，且每一步都验证过）

1. 正式机械臂改为**直接装配**：spawn + Articulation 绑定，InteractiveScene 只保留场地/传感器/穿梭球。
2. 安装位姿必须在 **spawn 时**给定（translation）。事后用 write_root_pose_to_sim 会脱开固定基座，
   整机掉到地下——这一步我踩了两次，实测最低刚体 -0.28 m 与 -0.55 m。
3. step_env() 负责这些 direct articulation 的写入与刷新（它们不属于 InteractiveScene）。

纪律约束遵守情况：**没有**加大 stiffness（保持配置的 100 / damping 1）、**没有**改任何官方限位、
**没有**忽略越界或屏蔽检查。

---

## 2. 修改的文件与提交

| 提交 | 内容 |
| --- | --- |
| bcf8dee | F1 修复 + 验收证据 + 新测试（simulation/badminton_scene/{robot_cfg,scene_cfg}.py、scripts/simulation/check_arm_rest_pose_hold.py、tools/merge_hold_evidence.py、tests/simulation/rebot_b601_rs/test_rest_pose_hold.py、outputs/simulation/rebot_b601_rs/HOLD_*、ARM_MIGRATION_RECORD.md）|
| e109294 | 完整运行日志入库 + 断言 DOF 目标确实到达 PhysX |
| 0d5bfe7 | 可移植资产包：tools/verify_asset_package.py、management/rebot_b601_rs/ASSET_PACKAGE.md、asset_package_manifest.json、asset_package_MANIFEST.sha256、outputs/simulation/rebot_b601_rs/asset_package_verify.json |

具体文件作用：

| 文件 | 作用 |
| --- | --- |
| simulation/badminton_scene/robot_cfg.py | 抽出 rebot_rs_spawn_cfg / rebot_rs_actuators，新增 spawn_rebot_rs_directly(translation) |
| simulation/badminton_scene/scene_cfg.py | 正式臂直接装配；step_env 支持 direct_articulations；num_envs != 1 显式报错 |
| scripts/simulation/check_arm_rest_pose_hold.py | 验收探针：两种装配、1200 步、检查点 0/1/5/30/60/240/600/1200、曲线 CSV |
| tools/merge_hold_evidence.py | 把两次运行合并成 HOLD_rest_pose.json |
| tests/simulation/rebot_b601_rs/test_rest_pose_hold.py | TDD 门：5 s 保持、joint2/3 限位、末端高度、驱动参数、目标到达 PhysX |
| tools/verify_asset_package.py | 资产包校验器（manifest-only / --root 全量重算 sha256）|
| management/rebot_b601_rs/asset_package_manifest.json 与 _MANIFEST.sha256 | 59 文件的校验清单 |

---

## 3. 实际测试命令、结果与日志位置

### 命令（远端 GPU 主机 dgut@172.31.68.251）

    ssh dgut@172.31.68.251 bash /home/T7/dgut/robot_sim/run_hold_f1.sh
    ssh dgut@172.31.68.251 /home/T7/ojh/robot_sim/env_isaaclab/bin/python /home/T7/dgut/robot_sim/verify_asset_package.py --manifest <pkg>/manifest.json --root <pkg>

### 命令（本地）

    python tools/merge_hold_evidence.py
    python -m pytest tests/simulation/rebot_b601_rs -q
    python -m pytest tests -q            (需 PYTHONPATH 指向本地 _scratch_pylibs，否则 pandas 测试无法收集)
    python tools/verify_asset_package.py --manifest management/rebot_b601_rs/asset_package_manifest.json

### F1 验收结果（两种装配完全一致）

| 指标 | 独立机械臂 | 整机场景 |
| --- | --- | --- |
| 5 s 保持 | 是 | 是 |
| 最差保持误差 | 0.0423 rad | 0.0423 rad |
| joint3 区间（1200 步）| 0.945–1.000 rad（下限 0，从未越过）| 同 |
| 末端相对安装面 z | 0.482–0.497 m | 同 |
| 末端下坠 | 0.015 m | 0.015 m |
| 最低刚体 z | 0.30（即基座，无穿地）| 0.30 |

检查点（joint3 / 保持误差 / 末端相对安装面 z）：

| step | 0 | 1 | 5 | 30 | 60 | 240 | 600 | 1200 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| q3 | 1.0000 | 0.9991 | 0.9898 | 0.9449 | 0.9668 | 0.9612 | 0.9620 | 0.9622 |
| err3 | 0.0000 | 0.0009 | 0.0102 | 0.0551 | 0.0332 | 0.0388 | 0.0380 | 0.0378 |
| ee_rel_z | 0.4968 | 0.4966 | 0.4944 | 0.4822 | 0.4881 | 0.4867 | 0.4868 | 0.4869 |

第 30 步的 0.055 是欠阻尼（damping=1）造成的单次过冲，60 步内回到 0.033 并稳定；这是我保留、
没有靠加大阻尼去掩盖的事实。

### 回归

* tests/simulation/rebot_b601_rs：**36 passed**
* 全量 tests：**1682 passed / 13 skipped / 4 failed**，4 个失败全部是 tests/tools/test_usd_glb_common.py
  （需要 pxr，本地 Windows Python 没有；该文件与本轮改动无关，git status 干净）

### 资产包校验

| 检查 | 结果 |
| --- | --- |
| 断链引用 | 0 |
| 引用是否出包 | 无 |
| 绝对路径命中 | 70 处，全部在转换器 doc 字符串（/tmp 构建路径），无功能依赖 |
| URDF 网格闭包 | 42/42 存在 |
| 结构 | 6 revolute / 2 prismatic / 1 fixed / 6 个角驱动属性 |
| sha256sum -c MANIFEST.sha256 | MANIFEST_OK，59 文件全匹配 |
| 校验器 manifest-only（本地）| PASS |
| 校验器 --root 全量重算（主机）| PASS，59/59 |

### 日志与证据位置

| 内容 | 位置 |
| --- | --- |
| 完整运行日志 | outputs/simulation/rebot_b601_rs/HOLD_log_standalone.txt、HOLD_log_scene.txt（原件在主机 migration_check/hold_f1/run_*.log）|
| 关节角度曲线（60 Hz，301 行）| outputs/simulation/rebot_b601_rs/HOLD_curve_standalone.csv、HOLD_curve_scene.csv |
| 结构化结果 | outputs/simulation/rebot_b601_rs/HOLD_rest_pose.json + HOLD_run_{standalone,scene}.json |
| 资产包校验 | outputs/simulation/rebot_b601_rs/asset_package_verify.json |
| 审查记录 | management/rebot_b601_rs/ARM_MIGRATION_RECORD.md（E/F1 节）、ASSET_PACKAGE.md |

---

## 4. 尚未解决的问题

1. **InteractiveScene 内的驱动绑定缺陷只是被绕过，没有从上游解释。** 证据齐了（矩阵 + stage 事实），
   但没定位到 IsaacLab 具体哪一行/哪个时序。
2. **num_envs 只支持 1。** 直接装配绑定到具体的 /World/envs/env_0 路径，scene_cfg 对 num_envs != 1
   直接抛 NotImplementedError（不静默降级）。这条阻塞并行环境与 RL 训练。
3. **default_joint_pos 全为 0。** init_state.joint_pos 对直接装配不生效，当前由 create_scene 显式
   write_joint_state_to_sim 写入 rest pose；任何依赖 default_joint_pos + action 的动作管理都需要注意。
4. **applied_joint_efforts 接口不存在**（ArticulationView 无该方法），施加力矩只能间接证明（驱动参数 +
   保持误差 + 目标到达 PhysX）。
5. **关节速度 50/40 仍未核实**（原 F2）：与 DM 那次 rpm 当 rad/s 同一特征，现标 OFFICIAL_URDF，
   require_real=True 会抛错。
6. **球拍适配件仍未测量**（configs/racket.yaml mount_transform REQUIRES_MEASUREMENT），因此没有 B601-RS
   的球拍装配 USD；本轮按指示未触碰球拍。
7. 4 个 pxr 依赖测试在本地无法通过（环境缺 pxr，非代码缺陷）。

---

## 5. 明天最值得继续推进的一项

**把 InteractiveScene 路径的驱动绑定问题定位到上游，恢复 num_envs > 1。**

理由：这是当前唯一的"绕过而非解决"，且直接卡住并行环境、课程式训练和 RL（PPO）接入。现在有了一套
很干净的隔离工具（同一 USD，两种装配，1200 步，五位小数可比），把它从"直接装配"换回"场景注册"，
只需要在 IsaacLab 的 cloner / actuator 初始化时序上加一次观测（例如给 InteractiveScene 加一个
drive-binding 断言探针），就能知道是克隆前写入被丢、还是 articulation view 绑到了另一个对象。

具体第一步（约 30 分钟）：在 scene_cfg 里对注册路径的机器人做一次"写入 -> 读回"断言——写完 k=400 后
立刻读 PhysX 侧生效增益与 DOF 目标，若读回仍是 1.745 / targetPosition 0，就能直接断言"写入落空"，
再顺着 cloner 的调用顺序找丢在哪一步。
