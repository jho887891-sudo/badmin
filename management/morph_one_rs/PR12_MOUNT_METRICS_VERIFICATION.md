# PR #12 复验：PhysX 六自由度安装误差指标

被验提交：`96ef90f614300174fb700aafe8abde541e1552aa`（指标修复 `cfb7f45`、文档 `96ef90f`）
worktree：`evidence/pr12-physx-metrics-verification`（独立建立）
环境：Isaac Sim 6.0.1 + Isaac Lab 3.0.0-beta2，TGS（solver_type=1, external forces every iteration），dt=1/120 s。
B601-RS 真实资产全程只读引用，未改写（第 5 节给出 sha256 前后比对）。

---

## 0. 结论速览

| 项 | 状态 | 依据 |
|---|---|---|
| 新增六自由度误差指标是否产生 | **VERIFIED** | 审计脚本原样跑，JSON 里出现 `max_fixed_mount_local_translation_error_m`、`max_fixed_mount_relative_rotation_error_rad`、`root_vertical_delta_after_settle_m`、`root_vertical_span_during_drive_m` |
| 指标口径是否正确（底盘局部坐标、四元数最短角、settle/drive 分离） | **VERIFIED（代码层面）** | 四元数逆变换把位移转到底盘坐标系；角度用 `2·acos(|dot|)` 处理 q≡−q；竖直量从 settle 后的起点算 |
| 代理底盘与臂基座初始重叠是否消除 | **VERIFIED（几何）** | 底盘碰撞盒升到 z∈[0.02,0.10]，臂基座在 z∈[0.16,0.24]，无重叠；另发现并修掉了"底盘压在地面以下"的第二处重叠 |
| 非零安装偏移 | **VERIFIED（已加）** | 偏移 `(0.12, −0.05, 0.20) m`，`setup_mount_offset_error_m` 反映真实偏差 |
| 驱动阶段安装相对位置/旋转误差（底盘局部系） | **VERIFIED（刚体约束成立）** | 位置 **3.1e-08 m**，旋转 **0.0 rad**（joint-local 授权方式） |
| settle 前后位置变化 / drive 阶段净竖直位移 | **已测** | joint-local：settle 漂移 **0.3549 m**，drive 净竖直 **−1.19e-07 m** |
| 初始构型是否等于授权 USD（"无初始穿模/无 settle 瞬态"） | **FAILED** | setup 偏移误差 **0.2612 m**，t=0 底盘已被转到 ~179° yaw、z=0.0379 m |
| 旋转（yaw 注入）工况 | **FAILED / UNKNOWN** | 指令 wz=0.3 rad/s，实测底盘 yaw 变化 **0.0 rad**（joint-local）；body-only 则整体自旋 2.20 rad |
| 实际生效关节增益 | **VERIFIED（本轮读到）** | `robot.data.joint_stiffness = [300, 300]`、`joint_damping = [20, 20]` |

---

## 1. 审计脚本原样运行（未改一行）

    python morph_one_proxy_physx_probe.py --variant real_arm --gravity-off --gains verified \
        --joint-step --out out/audit_real_arm.json
    -> EXIT_A=0

| 字段 | 值 |
|---|---|
| num_bodies_physx | 10（9 个臂刚体 + 底盘） |
| num_joints_physx / dofs_physx | 8 / 8 |
| chassis_in_articulation | true |
| max_fixed_mount_local_translation_error_m | 0.0 |
| max_fixed_mount_relative_rotation_error_rad | 0.0 |
| root_vertical_delta_after_settle_m | 0.000883 |
| twist_written | true |

    代理件 arm_rooted：4 刚体 / 2 DOF / chassis_in=true / 两个安装误差 0.0 / root_vertical_delta 0.0，EXIT_B=0

即：**新增指标确实产生、数值形态合理（10 刚体 8 DOF 与上一轮一致）**。但要注意一个口径问题：
审计脚本把参考位姿取在 `step(SETTLE_STEPS)` **之后**，所以它看不见 settle 期间发生的初始构型偏差——
这正是第 2 节用独立沙盒要测的东西。
## 2. 独立开发沙盒（重叠已消除、非零偏移、settle/drive 分离）

新文件：`scripts/simulation/morph_one_mount_metrics_sandbox.py`（TEST_PROXY，非 Morph One 硬件）。

几何（自上而下，全部无重叠）：

| 体 | 半尺寸 m | 中心（体坐标系） | 质量 kg |
|---|---|---|---|
| Chassis（代理底盘） | 0.25 × 0.175 × 0.04 | (0, 0, +0.06)，即 z∈[0.02, 0.10] | 15 |
| Arm/base_link | 0.04 × 0.04 × 0.04 | 体原点在安装偏移 (0.12, −0.05, 0.20) | 2 |
| link1 / link2 | 0.14/0.11 × 0.025/0.02 | 沿 +x 接出 | 1 / 0.5 |

安装偏移 **非零**：`(0.12, −0.05, 0.20) m`。比较两种 USD 授权方式：

* `joint-local`：FixedJoint 的 `physics:localPos0` = 偏移，localRot 单位四元数；
* `body-only`：FixedJoint 不给局部位姿，只靠两个体的 xform。

### 2.1 结果（三种工况，零重力，验证增益 300/20）

| 工况 | 授权方式 | setup 偏移误差 m | settle 漂移 m | settle 安装误差 m | drive 安装位置误差 m | drive 安装旋转误差 rad | drive 底盘位移 m | drive 净竖直位移 m | drive yaw 变化 rad |
|---|---|---|---|---|---|---|---|---|---|
| translate（vx=0.4） | joint-local | 0.261245 | 0.354905 | 0.367858 | **3.1e-08** | **0.0** | **0.600003** | −1.19e-07 | 0.0 |
| rotate（wz=0.3） | joint-local | 0.261245 | 0.354905 | 0.367858 | 8.1e-09 | 0.0 | 6.0e-06 | −1.19e-07 | **0.0** |
| combined（0.4+0.3） | joint-local | 0.261245 | 0.354905 | 0.367858 | 3.1e-08 | 0.0 | 0.600003 | −1.19e-07 | **0.0** |
| translate | body-only | 0.238537 | 0.006809 | 0.0 | 0.0 | 0.0 | 0.570973 | **0.195167** | **2.201670** |
| rotate | body-only | 0.238537 | 0.006809 | 0.0 | 8.0e-09 | 0.0 | 0.0 | 0.185485 | 2.171532 |

三条硬结论：

1. **刚体安装约束本身在驱动阶段成立**：joint-local 方式下，底盘局部系里的安装位置误差 3.1e-08 m、
   旋转误差 0.0 rad，全程 180 步没有增长；净竖直位移 −1.19e-07 m（数值零）。
2. **初始构型不等于授权 USD（FAILED）**：t=0（`sim.reset()` 之后、任何步进之前）实测
   底盘位姿 = (3.4e-05, 1.8e-04, 0.0379) m、四元数 (0.00756, 9.4e-05, 3.5e-04, 0.99997) ≈ **179° yaw**，
   而授权值是原点 + 单位四元数；底盘局部系下的相对位置因而是
   `(−0.1208, +0.0513, +0.1992)` —— 与授权的 `(0.12, −0.05, 0.20)` 相比 x/y **符号被翻转**。
   由此产生 settle 期 0.3549 m 的位移与 0.3679 m 的瞬时安装误差。**根因 UNKNOWN**：
   授权 USD 里两个体都在预期位姿，但 PhysX 初始化后的体姿态不同，需要专门做一次
   "USD xform ↔ `robot.data.body_*_w`" 的逐步对照才能定位（本轮预算内未完成）。
3. **两种授权方式各有各的坏法**：body-only 的 settle 漂移反而小（0.006809 m、settle 期安装误差 0.0），
   但驱动阶段整体自旋 2.20 rad、竖直漂移 0.195 m，动态上不可用；joint-local 则相反（驱动阶段刚性极好，
   初始化偏差大）。也就是说哪种授权方式才正确仍未确定，两个变体都不能作为验收基线。
4. **旋转工况未成立（FAILED）**：joint-local 方式下注入 wz=0.3 rad/s 后底盘 yaw 变化 0.0 rad、
   位移 6.0e-06 m —— 角速度注入没有产生任何转动；body-only 方式则相反，整体自旋 2.20 rad
   （因为它把安装偏移当成了要消除的约束误差）。两种方式都**没有**给出"底盘旋转时安装误差如何"的可信数据。

### 2.2 本轮自己引入并修掉的一个假象

第一版沙盒把底盘碰撞盒放在 z∈[−0.10, −0.02]，**压在地面平面以下**，settle 期被静态地面弹开
（漂移 0.3505 m）。升到 z∈[0.02, 0.10] 后该假象消失，但第 2.1 条的 settle 漂移仍在（0.3549 m），
说明它不是地面穿透造成的，而是第 2 条的初始化偏差造成的。这条弯路如实保留，因为它区分了两种不同原因。

## 3. 实际生效的关节增益：读到 300 / 20

审计脚本用 `view.get_joint_stiffness()` 取不到（`ArticulationView` 没有该方法），因此只能标 UNKNOWN。
本沙盒改读 `robot.data.joint_stiffness` / `joint_damping`，成功：

    status = READ_FROM_ARTICULATION_DATA
    joint_names = ["joint1", "joint2"]
    stiffness   = [300.0, 300.0]
    damping     = [20.0, 20.0]

即 `--gains verified`（300/20，与仓库 `rs_physics.py` + `scene_layout.REBOT_RS` 一致）**确实生效**。
上一轮日志里出现的 `2.9e+03 / 286.479` 是"未指定 `--gains verified`"时的默认值，两者已被区分开。

## 4. 口径与解释边界（必须随结果一起传播）

* **根速度注入不是轮驱动**：本沙盒没有任何轮胎接触、没有摩擦、没有电机/减速器模型，`vx=0.4 m/s` 只是
  给 articulation 根体一个初速度。**不得**把它当作四转四驱（4WD4WS）的成功率或底盘运动能力证明。
* 代理质量/惯量（15 / 2 / 1 / 0.5 kg）是随手取的测试值，**不是** Morph One 的真实参数。
* 旋转工况未成立（第 2.1 节第 3 条），因此"底盘旋转时安装约束是否仍然成立"仍是 **UNKNOWN**。
* 初始构型偏差（第 2.1 节第 2 条）意味着**任何**基于"settle 后取参考"的指标都会把这个偏差判成 0；
  PR #12 的指标口径在这一点上是正确的（它如实只报 settle 之后的量），但**不能**用它的 0.0 去反推
  "初始化没问题"。

## 5. 真实 B601-RS 资产未被修改

    臂资产 sha256（本轮前后一致）：96d4f2ec1126dec6eaf9e70ab664ba6cc30fa81bdec86178a3e50b7a23bdd510

沙盒与审计脚本都只通过 `references` 引用真实臂资产，并在覆盖层里改 `root_joint` 的 `body0`；资产文件本身
没有被写入（本轮未出现任何 Write/Export 到该路径）。

## 6. 复现命令与退出码

    # 审计版本原样
    python morph_one_proxy_physx_probe.py --variant real_arm --gravity-off --gains verified \
        --joint-step --out out/audit_real_arm.json            # EXIT=0
    python morph_one_proxy_physx_probe.py --variant arm_rooted --gravity-off \
        --out out/audit_arm_rooted.json                        # EXIT=0
    # 独立沙盒
    python morph_one_mount_metrics_sandbox.py --mode translate --out out/sandbox_translate.json       # EXIT=0
    python morph_one_mount_metrics_sandbox.py --mode rotate    --out out/sandbox_rotate.json          # EXIT=0
    python morph_one_mount_metrics_sandbox.py --mode combined  --out out/sandbox_combined.json        # EXIT=0
    python morph_one_mount_metrics_sandbox.py --mode translate --mount-authoring body-only \
        --out out/sandbox_bodyonly_translate.json               # EXIT=0
    python morph_one_mount_metrics_sandbox.py --mode rotate --mount-authoring body-only \
        --out out/sandbox_bodyonly_rotate.json                   # EXIT=0

证据：`outputs/simulation/rebot_b601_rs/morph_one/pr12_metrics/`（6 个运行 JSON、4 个授权 USDA、3 份原始日志）。

## 7. 交给下一阶段的技术建议

1. **先修初始化偏差**：在 `sim.reset()` 前后逐体打印 `robot.data.body_pos_w/body_quat_w` 与 USD 里的
   `xformOp` 期望值，定位"为什么底盘在 t=0 已经是 179° yaw"。在修好之前，settle 期数据一律不可用于验收。
2. **旋转工况换驱动方式**：`write_root_velocity_to_sim` 的角速度分量在本 articulation 上无效（yaw 变化 0.0）。
   可改为逐步写根位姿（`write_root_pose_to_sim`）或用一个小型"虚拟万向轮"力矩模型，再测安装误差。
3. 指标口径建议保留 PR #12 的做法（底盘局部系 + 最短角 + settle 后起点），但**追加**一个 `setup_error`
   字段（把 t=0 与授权值的偏差也报出来），否则又会漏掉本轮发现的这一类问题。
4. 四转四驱能力验证必须另起一条线：需要轮胎接触模型与摩擦参数，不能用根速度注入代替。
