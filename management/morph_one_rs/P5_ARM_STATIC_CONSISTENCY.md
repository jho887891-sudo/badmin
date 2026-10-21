# P5 — B601-RS 单臂静态一致性核查

Status: **PASS on the asset contract, 1 real defect found (reset rest pose), 2 dependency gaps recorded as UNKNOWN.**

本轮只做被指派单元：用 pxr 在 CPU 上静态读资产，与仓库里的关节顺序/限位定义逐项对照，并用 pytest 把结论钉住。
没有启动 Isaac Sim、没有 torch.cuda、没有写资产、没有合并 PR、没有 force push、没有碰 main。

| 项 | 值 |
| --- | --- |
| 分支 | `p5/arm-static-consistency`，从 `origin/main` @ `16f1ae4` 建 |
| 资产（只读） | `/home/T7/dgut/robot_sim/assets/robots/badminton_robot/rebot_b601_rs/usd/rebot_b601_rs.usd/ReBot_Arm_RS/ReBot_Arm_RS.usda` |
| 资产 sha256（前 = 后） | `96d4f2ec1126dec6eaf9e70ab664ba6cc30fa81bdec86178a3e50b7a23bdd510` |
| 解释器 | `/home/T7/ojh/robot_sim/env_isaaclab/bin/python`（Python 3.12.13, pxr = USD 0.25.11） |
| 定义来源（见 U2） | `src/simulation/rebot_b601_rs/joint_limits.py` @ `d912202`, sha256 `69b078e5520abae4a33a67a35a9ef9ee2d1d40809bd30ec83aa6e8398a20ac09` |
| 事实导出 | `outputs/simulation/rebot_b601_rs/morph_one/p5_static/arm_joint_facts.json`, sha256 `42e57775639e14d8f462feb84eac603bf913def37886b8ec09c0d90dfa1d7df6` |
| 导出脚本 | `scripts/simulation/p5_arm_static_facts.py`, sha256 `e29a5f2c2ba246b1aa41bb9fcab719740e3c700a84789c890bec3def1ac5e2a8` |
| 测试 | `tests/simulation/cad/test_arm_static_consistency.py`, sha256 `a18603adfea418f6a69c6b655f3f8b82c8aec5f59809c4a246d0107881e9a35d` |

---

## 0. 结论速览

| 问题 | 结论 | 依据 |
| --- | --- | --- |
| 关节计数 1 Fixed + 6 Revolute + 2 Prismatic = 9 | **PASS** | 第 2 节事实表 |
| 8 个关节限位与文档一致 | **PASS**（最大偏差 8.3e-8 rad / 3.4e-9 m） | 第 4 节 |
| URDF 关节顺序与 USD/PhysX 运行时顺序一致 | **PASS（URDF = 运行时）**，但 **USD stage 遍历顺序不一致**，见 F2 | 第 3 节 |
| 仓库里依赖关节顺序的代码 | 存在，`safety_shield.py` 是唯一硬依赖，见 3.3 | 第 3.3 节 |
| reset 后 rest pose 是否有硬编码值与 limits 冲突 | **有，1 处真实缺陷**：`reset.py:10` 的 PiPER 姿态 (joint3 = -1.5 rad ∉ [0, 3.14]) | 第 5 节 / F1 |
| 静态一致性 pytest | **PASS**：定义在场 14 passed；仅 origin/main 3 passed / 11 skipped，两次退出码均 0 | 第 6 节 |

缺陷与未解项：**F1** reset 硬编码越限（真实缺陷，已用金丝雀测试钉住）、**F2** USD 遍历顺序 ≠ DOF 顺序（陷阱，已钉住）、
**F3** 驱动增益是度制（100/57.2958）、**F4** 9 = 9 的假相等（URDF 的固定关节是 `j_gripper_end`，USD 的是 `root_joint`）、
**U1–U5** 见第 7 节。

---

## 1. 怎么跑出来的（可复查）

### 1.1 事实导出（pxr only，CPU）

```bash
# 脚本先拷到主机，再在主机上跑（脚本本身只读资产）
scp scripts/simulation/p5_arm_static_facts.py dgut@172.31.68.251:/home/dgut/runs/p5/p5_arm_static_facts.py
ssh dgut@172.31.68.251 \
  "cd /home/dgut/runs/p5 && /home/T7/ojh/robot_sim/env_isaaclab/bin/python ./p5_arm_static_facts.py \
     --usda /home/T7/dgut/robot_sim/assets/robots/badminton_robot/rebot_b601_rs/usd/rebot_b601_rs.usd/ReBot_Arm_RS/ReBot_Arm_RS.usda \
     --out /home/dgut/runs/p5/arm_joint_facts.json"
```

真实输出（完整 stdout 亦存于主机 `/home/dgut/runs/p5/facts_stdout.txt`）：

```text
joints=9 rigid_bodies=9 articulation_roots=1 unmodified=True
  /ReBot_Arm_RS/Physics/gripper_joint1   PhysicsPrismaticJoint axis=Z limits={0.0, 0.05000000074505806} m
  /ReBot_Arm_RS/Physics/gripper_joint2   PhysicsPrismaticJoint axis=Z limits={0.0, 0.07150000333786011} m
  /ReBot_Arm_RS/Physics/joint1           PhysicsRevoluteJoint  axis=Z limits={-160.4281768798828, 160.4281768798828} deg
  /ReBot_Arm_RS/Physics/joint2           PhysicsRevoluteJoint  axis=Z limits={0.0, 179.90875244140625} deg
  /ReBot_Arm_RS/Physics/joint3           PhysicsRevoluteJoint  axis=Z limits={0.0, 179.90875244140625} deg
  /ReBot_Arm_RS/Physics/joint4           PhysicsRevoluteJoint  axis=Z limits={-89.95437622070312, 89.95437622070312} deg
  /ReBot_Arm_RS/Physics/joint5           PhysicsRevoluteJoint  axis=Z limits={-89.95437622070312, 89.95437622070312} deg
  /ReBot_Arm_RS/Physics/joint6           PhysicsRevoluteJoint  axis=Z limits={-179.90875244140625, 179.90875244140625} deg
  /ReBot_Arm_RS/Physics/root_joint       PhysicsFixedJoint     axis=None limits={None, None}
stage traversal order: joint6, gripper_joint1, gripper_joint2, joint5, joint4, joint3, joint2, joint1, root_joint
wrote /home/dgut/runs/p5/arm_joint_facts.json
EXIT=0
```

### 1.2 资产未被写入（sha256 前后一致）

```text
=== sha256 BEFORE ===
96d4f2ec1126dec6eaf9e70ab664ba6cc30fa81bdec86178a3e50b7a23bdd510  .../ReBot_Arm_RS.usda
=== sha256 AFTER (两次 pytest 之后) ===
96d4f2ec1126dec6eaf9e70ab664ba6cc30fa81bdec86178a3e50b7a23bdd510  .../ReBot_Arm_RS.usda
```

脚本自己也在 `asset.usda_sha256_before` / `asset.usda_sha256_after` / `asset.unmodified: true` 里记了同一对值，
所以 JSON 单独拿出来也能自证“没改过资产”。

### 1.3 资产结构（为什么只读一个 1251 B 的 .usda 就够）

```text
ReBot_Arm_RS.usda (1251 B, root layer, defaultPrim = ReBot_Arm_RS, upAxis Z, metersPerUnit 1)
  references ./payloads/base.usda
  variantSet "Physics" = "physx"  ->  payload ./payloads/Physics/physx.usda
  (另有 physics.usda / mujoco.usda 两个变体，本轮未选，故未读)
payloads/base.usda 28045 B, robot.usda 3538 B, instances.usda 20270 B, geometries.usd 19.8 MB
```

`Usd.Stage.Open` 默认加载 payload，所以 9 个 joint / 9 个 rigid body / 驱动属性都来自**组合后**的 stage，
而不是某个未经组合的裸层。变体选择在 JSON 的 `asset.variant_selection` 里记为 `/ReBot_Arm_RS|Physics = physx`。

---

## 2. 关节事实（`arm_joint_facts.json` 摘要）

计数：`joints=9`，`by_type = {Fixed: 1, Revolute: 6, Prismatic: 2}`，`rigid_bodies=9`，`articulation_roots=1`。
ArticulationRootAPI 在 **`/ReBot_Arm_RS/Geometry`**（唯一一处）。

| joint | 类型 | body0 | body1 | axis | lower / upper | drive stiffness | damping | maxForce |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| `joint1` | Revolute | base_link | link1 | Z | -160.4281768798828 / 160.4281768798828 **deg** | 1.7453292608261108 | 0.01745329238474369 | 36 |
| `joint2` | Revolute | link1 | link2 | Z | 0.0 / 179.90875244140625 **deg** | 1.7453292608261108 | 0.01745329238474369 | 36 |
| `joint3` | Revolute | link2 | link3 | Z | 0.0 / 179.90875244140625 **deg** | 1.7453292608261108 | 0.01745329238474369 | 36 |
| `joint4` | Revolute | link3 | link4 | Z | -89.95437622070312 / 89.95437622070312 **deg** | 1.7453292608261108 | 0.01745329238474369 | 14 |
| `joint5` | Revolute | link4 | link5 | Z | -89.95437622070312 / 89.95437622070312 **deg** | 1.7453292608261108 | 0.01745329238474369 | 14 |
| `joint6` | Revolute | link5 | link6 | Z | -179.90875244140625 / 179.90875244140625 **deg** | 1.7453292608261108 | 0.01745329238474369 | 14 |
| `gripper_joint1` | Prismatic | link6 | gripper_left | Z | 0.0 / 0.05000000074505806 **m** | 100.0 | 1.0 | 500 |
| `gripper_joint2` | Prismatic | link6 | gripper_right | Z | 0.0 / 0.07150000333786011 **m** | 100.0 | 1.0 | 500 |
| `root_joint` | Fixed | `/ReBot_Arm_RS` | `/ReBot_Arm_RS/Geometry/base_link` | — | — | — | — | — |

9 个 rigid body（`mass` kg）：base_link 1.1774、link1 0.4655、link2 1.5520、link3 1.2520、link4 0.4600、
link5 0.20120457、link6 **0.75**、gripper_left 0.0752、gripper_right 0.0752，合计 **6.008504584431648 kg**。
link6 = 0.75 而不是 URDF 的 0.1，是因为 `--merge-joints` 把 `j_gripper_end` 的 0.65 kg 折进了 link6；合计与文档 6.0085 kg 一致（F4）。

每个 drive 还带 `targetPosition = 0`、`targetVelocity = 0`、`type = force`（见 JSON `joints[].authored_attrs`），
即资产自带的驱动目标是 0，不是某个 rest pose —— 这与 `management/rebot_b601_rs/ARM_MIGRATION_RECORD.md` F1 记录的
“asset drive gains authored divided by 57.2958 with targetPosition 0”一致。

---

## 3. 问题一：URDF 关节顺序 == USD/PhysX 运行时顺序？

### 3.1 三种“顺序”不是一回事（实测）

| 顺序 | 实际序列 | 出处 |
| --- | --- | --- |
| **URDF 声明顺序**（9 关节） | joint1, joint2, joint3, joint4, joint5, joint6, **j_gripper_end(Fixed)**, gripper_joint1, gripper_joint2 | 主机上 `xml.etree` 解析 `urdf/ReBot_Arm_RS.urdf`（第 1 节同法，只读） |
| **USD 运动链顺序**（8 活动关节） | joint1…joint6, gripper_joint1, gripper_joint2 | JSON `joints[].body0/body1` 串链；pytest `test_kinematic_chain_order_matches_joint_limits_order` |
| **USD stage 遍历顺序**（9 关节） | joint6, gripper_joint1, gripper_joint2, joint5, joint4, joint3, joint2, joint1, root_joint | JSON `stage_traversal_order`；pytest `test_stage_traversal_order_is_not_the_dof_order` |
| **PhysX/IsaacLab 运行时 DOF 顺序**（8 DOF） | joint1…joint6, gripper_joint1, gripper_joint2 | 既往真跑记录 `outputs/simulation/rebot_b601_rs/VERIFY_usd_load.json:98-107` |

结论：**URDF 顺序与 PhysX 运行时 DOF 顺序一致**（都是 joint1…joint6 后接两个手指），
但 **USD stage 的遍历顺序既不是 URDF 顺序也不是字母序**（F2）。任何人若用 `stage.Traverse()` 的位置当 DOF 下标，
会把 joint6 当成 DOF 0、把固定关节算成第 9 个 DOF。

运行时顺序这一条本轮**没有重新测**（GPU 被禁），它是既往真跑记录里的数据；因此第 7 节把它单列为 U1。

### 3.2 两个“9”不是同一组 9（F4）

URDF 是 9 个关节：6 revolute + 9 号位那个 **mid-chain fixed `j_gripper_end`** + 2 prismatic。
USD 也是 9 个：6 revolute + 2 prismatic + **base fixed `root_joint`**。
`--merge-joints` 把 `j_gripper_end` 折进 link6、`--fix-base` 又补了一个 `root_joint`，于是计数相同而成员不同。
只数个数会得到一个假的 PASS，所以测试里显式断言“固定关节必须是 `root_joint`，且 `j_gripper_end` 不存在”。

### 3.3 仓库里依赖顺序的代码（file:line）

**硬依赖（位置即语义，顺序错了会算错）**

| 文件:行 | 依赖形态 | 后果 |
| --- | --- | --- |
| `src/badminton_brain/safety/safety_shield.py:213` / `:216` | 逐关节数组按 `[joint1..joint6]` 排列：min `[-2.80, 0.00, 0.00, -1.57, -1.57, -3.14]`，max `[2.80, 3.14, 3.14, 1.57, 1.57, 3.14]` | 第 k 列必须就是 joint(k+1) |
| `src/badminton_brain/safety/safety_shield.py:72` | `JOINT_COUNT = 6` | 只接受 6 维手臂向量，8 DOF 的整机向量会被拒 |
| `src/badminton_brain/safety/safety_shield.py:392-399` | `_env_matrix` 要求形状严格 `(N, 6)`，否则抛 `BrainBoundaryError` | 传 8 维（含手指）直接报错，不会静默错位 |
| `src/badminton_brain/safety/safety_shield.py:514` / `:517` | `state.joint_pos` / `context.joint_position` 按列取 | 调用方必须按 joint1..joint6 排 |
| `src/badminton_brain/safety/safety_shield.py:568-569` | `lo`/`hi` 与 `joints` 逐列比较、裁剪 | 行列错配 = 上下限套错关节 |
| `src/badminton_brain/safety/safety_shield.py:312-313` | `_safe_rest_q = clip(zeros(6), lo, hi)` | 兜底 rest 是“限位内的零位”，与场景 rest pose 不同（见 5.3） |
| `src/badminton_brain/execution/sim_adapter.py:216` / `:225` | 接口文档与默认值把 `joint_position_target` 定为 `(6,)` | 6 维顺序契约 |
| `tests/simulation/rebot_b601_rs/test_safety_limits_agree.py:26-27` | 用 `ARM_JOINTS` 顺序拼期望数组，与盾的数组**按位置**比对 | 顺序漂移这里会红（这是它存在的意义） |
| `tools/generate_rs_limits_yaml.py:72` | 按 `JOINT_ORDER` 生成 YAML 行序 → `configs/simulation/rebot_b601_rs_joint_limits.yaml` | 序列化的顺序也跟 `JOINT_ORDER` 走 |
| `tests/simulation/rebot_b601_rs/test_multienv_hold.py:77` | `len(trace) == 8` | 断言运行时 articulation 恰好 8 个 DOF（固定关节不占 DOF） |

**按名字映射（与顺序无关，顺序变了也不会错）**

| 文件:行 | 做法 |
| --- | --- |
| `simulation/badminton_scene/reset.py:21-22` | `for k, name in enumerate(robot.joint_names)` + `REST_JOINT.get(name)` |
| `simulation/badminton_scene/scene_cfg.py:191-195` | `[REBOT_RS["rest_joint_pos"].get(nm, 0.0) for nm in _direct_robot.joint_names]` |
| `simulation/badminton_scene/multienv_arm.py:101` | `[self.rest_joint_pos.get(nm, 0.0) for nm in self.joint_names]` |
| `simulation/badminton_scene/robot_cfg.py:92` | `joint_pos=dict(REBOT_RS["rest_joint_pos"])`（IsaacLab 按名字解析） |
| `scripts/simulation/measure_racket_speed.py:94` | `dof_names.index(name)` |
| `scripts/simulation/check_arm_rest_pose_hold.py:70` | `[REST.get(nm, 0.0) for nm in robot.joint_names]` |

即：**场景/复位层已经改成按名字，唯一的位置依赖在安全盾（brain 层）与它的对照测试里。**
安全盾注释里已经写明“数组必须跟踪 `joint_limits.py`”，但它对 DOF 顺序的假设（joint1..joint6）目前只由
`test_safety_limits_agree.py` 间接保护 —— 那个测试保护的是“表 vs 盾”，不是“盾 vs 资产”。本轮新增的
`test_arm_static_consistency.py` 把“资产 vs 表”这一半补上了，两半合起来才闭合。

---

## 4. 问题二：8 个关节限位是否与文档一致

文档 = `src/simulation/rebot_b601_rs/joint_limits.py`（`_URDF_ANGLE` 第 166-173 行 / `joint_limits` 第 230-239 行），
其值来自官方 URDF `<limit>`；资产 = 上表 USD 值（**角关节单位是度**）。

| joint | 文档 (rad / m) | URDF `<limit>` (rad / m) | USD (deg / m) | USD 换算 (rad / m) | 偏差 |
| --- | --- | --- | --- | --- | --- |
| joint1 | [-2.8, 2.8] | [-2.8, 2.8] | [-160.4281768798828, 160.4281768798828] | [-2.7999998995, 2.7999998995] | 1.0e-7 |
| joint2 | [0, 3.14] | [0, 3.14] | [0, 179.90875244140625] | [0, 3.1400000833] | 8.3e-8 |
| joint3 | [0, 3.14] | [0, 3.14] | [0, 179.90875244140625] | [0, 3.1400000833] | 8.3e-8 |
| joint4 | [-1.57, 1.57] | [-1.57, 1.57] | [-89.95437622070312, 89.95437622070312] | [-1.5700000416, 1.5700000416] | 4.2e-8 |
| joint5 | [-1.57, 1.57] | [-1.57, 1.57] | [-89.95437622070312, 89.95437622070312] | [-1.5700000416, 1.5700000416] | 4.2e-8 |
| joint6 | [-3.14, 3.14] | [-3.14, 3.14] | [-179.90875244140625, 179.90875244140625] | [-3.1400000833, 3.1400000833] | 8.3e-8 |
| gripper_joint1 | [0, 0.05] | [0, 0.05] | [0, 0.05000000074505806] | 同左 | 7.5e-10 |
| gripper_joint2 | [0, 0.0715] | [0, 0.0715] | [0, 0.07150000333786011] | 同左 | 3.3e-9 |

**结论：PASS。** 8/8 关节在 float32 转换误差内一致（最大 8.3e-8 rad），没有任何一个限位被放宽或收窄。

附带核对到的三条（都在 JSON 里可查）：

* `drive:angular:physics:maxForce` = 36/36/36/14/14/14，与 URDF `<limit effort>` 逐关节相同；`joint_limits.py` 的
  torque 用的是 RS06=36 N·m / RS00=14 N·m 峰值，两者独立方向一致（与迁移记录第 30-33 行同一论证）。
* `drive:linear:physics:maxForce` = 500（手指），`joint_limits.py` 明确记录 500 是**牛顿的指力不是电机扭矩**，provenance = UNKNOWN。资产与文档的说法不冲突。
* 角关节驱动增益 1.7453292608261108 = 100 / 57.2958、阻尼 0.01745329238474369 = 1.0 / 57.2958（**F3**）：
  资产存的是“度制”增益。把 1.745 当成 rad 制读会低估 57.3 倍，把 100 当成资产值会高估 57.3 倍 —— 两个方向都会错。
  手指是 linear，`stiffness = 100 / damping = 1` 原样保留（不缩）。

---

## 5. 问题三：reset 后 rest pose 有没有被硬编码成与 limits 冲突的值

### 5.1 有 —— 1 处真实缺陷（F1）

```text
simulation/badminton_scene/reset.py:10
REST_JOINT = {"joint1": 0.0, "joint2": 1.5, "joint3": -1.5, "joint4": 0.0, "joint5": 0.0, "joint6": 0.0}

simulation/badminton_scene/reset.py:19-22
    if joint_pos is None:
        jp = torch.zeros((n, robot.num_joints), device=dev)
        for k, name in enumerate(robot.joint_names):
            jp[:, k] = REST_JOINT.get(name, 0.0)
simulation/badminton_scene/reset.py:25
    robot.write_joint_state_to_sim(jp, torch.zeros_like(jp), env_ids=env_ids)
```

这是 **PiPER 的 rest pose**（同一组数字也出现在 `simulation/badminton_scene/scene_layout.py:101` 的 `PIPER` 块）。
在 B601-RS 上 `joint3` 的限位是 **[0, 3.14] rad**，`-1.5 rad` 在限位之外；`joint2 = 1.5` 则在 [0, 3.14] 之内。
`reset_idx()` 在调用方不传 `joint_pos` 时就把这组值写进仿真（`reset.py:22` → `:25`）。

可达路径（都以默认臂 `rebot_b601_rs` 建场景：`simulation/badminton_scene/robot_cfg.py:137` + `:140`）：

| 调用点 | 传 `joint_pos`? | 能否走到 |
| --- | --- | --- |
| `scripts/t06_reset_local.py:39` | 否 | `--num_envs 1` 时可达；默认 8 会先撞下面的守卫 |
| `scripts/t07_local.py:53` / `:68` | 否 | `--num_envs 1` 时可达；默认 128 同上 |
| `scripts/t08_local.py:69` | 否 | `--num_envs 1` 时可达；默认 128 同上 |

“默认环境数走不到”必须写清楚，否则会低估这条缺陷：B601-RS 的正式场景目前只支持 `num_envs == 1`
（`simulation/badminton_scene/assembly_route.py:16-20` 对 `num_envs != 1` 直接抛 `NotImplementedError`），
所以 `num_envs == 1` 恰恰是**唯一被支持的配置**，而这三个脚本只要按支持的方式跑，就在这条越限写入的路径上。

三处调用都**没有**传 `joint_pos`，所以默认走 `REST_JOINT`。后果按严重度：最轻是 PhysX/关节限位把 joint3 顶回 0，
最重是写进一个越限状态后在随后的步进里被解算器强行拉回、产生非物理的速度尖峰。
**本轮只能静态证明“写入值越限”这一步；运行时后果未测**（U3）。

这条已经被写进测试当**金丝雀**：`tests/simulation/cad/test_arm_static_consistency.py` 的
`test_reset_rest_joint_is_the_piper_pose_canary` 断言“`joint3 = -1.5` 仍然在限位外”。
也就是说：修好了这个缺陷，测试会**红**，逼着下一次改动同时更新本报告，而不是让记录悄悄过期。
（同样的金丝雀做法本仓库已有先例：`management/rebot_b601_rs/ARM_MIGRATION_RECORD.md` 第 I.4 条的 state-sync 缺陷。）

### 5.2 正式臂自己的 rest pose 是合规的

`simulation/badminton_scene/scene_layout.py:80-81`：

```text
rest_joint_pos={"joint1": 0.0, "joint2": 0.3, "joint3": 1.0,
                "joint4": 0.0, "joint5": 0.0, "joint6": 0.0},
```

joint2 = 0.3 ∈ [0, 3.14]，joint3 = 1.0 ∈ [0, 3.14]，其余为 0，都在限位内。改成这个值是为了让所有连杆都在基座平面之上
（`outputs/simulation/rebot_b601_rs/rest_pose_probe.json`），注释第 76-79 行也写明了“PiPER 的 (1.5, -1.5) 在这条臂上不可达”。
测试 `test_formal_arm_rest_pose_is_inside_the_documented_limits` 覆盖这一条。

### 5.3 另外两处“不是缺陷但要写下来”

* `src/badminton_brain/safety/safety_shield.py:312-313` 的兜底 rest 是 `clip(zeros(6), lo, hi)` = 全零，
  在限位内（joint2/joint3 的下限就是 0），**不与 limits 冲突**；但它与场景实际 spawn 的 (0.3, 1.0) 不是同一个姿态。
  也就是说：盾的 HOLD 兜底点 ≠ 场景 rest pose。目前不影响限位合规，记录备查。
* `simulation/badminton_scene/multienv_arm.py:101` 与 `scene_cfg.py:192` 用 `.get(nm, 0.0)`，手指两关节拿到 0.0，
  正好等于手指下限（[0, 0.05] / [0, 0.0715] 的左端点）。合规，但**贴着边界**；若将来把某个手指下限改成 >0，这两行会越限。

---

## 6. pytest（纯 pxr，CPU，主机上真跑）

### 6.1 测试覆盖什么

`tests/simulation/cad/test_arm_static_consistency.py`（14 个用例，缺 pxr 时 `pytest.importorskip` 整文件跳过）：

| 用例 | 断言 |
| --- | --- |
| `test_joint_census_is_one_fixed_six_revolute_two_prismatic` | 9 = 1 Fixed + 6 Revolute + 2 Prismatic，且类型集合恰好这三种 |
| `test_the_fixed_joint_is_the_base_fix_not_the_merged_gripper_flange` | 固定关节是 `root_joint`；`j_gripper_end` 已被 merge-joints 折掉（F4） |
| `test_revolute_limits_equal_the_documented_ranges` | 6 个 revolute 的度值 == 文档弧度换算值（1e-2 deg），且弧度读取一致（1e-6 rad） |
| `test_every_revolute_limit_stays_inside_the_documented_range` | 包含关系：USD 区间不得越出文档区间 |
| `test_prismatic_finger_limits_match` | 两指 [0,0.05] / [0,0.0715] m，且 kind == prismatic |
| `test_joint_name_set_matches_joint_limits` | 活动关节名集合 == `JOINT_LIMITS` 键集合（8 个） |
| `test_kinematic_chain_order_matches_joint_limits_order` | 由 body0→body1 串链得到的顺序 == `JOINT_ORDER`（不靠下标） |
| `test_stage_traversal_order_is_not_the_dof_order` | 金丝雀：stage 遍历顺序 ≠ `JOINT_ORDER`（F2） |
| `test_revolute_drive_gains_are_degree_scaled` | stiffness == 100/57.2958、damping == 1/57.2958、maxForce == URDF effort（F3） |
| `test_prismatic_drive_gains_are_not_degree_scaled` | 手指 stiffness 100 / damping 1 / maxForce 500，不做度缩 |
| `test_articulation_root_is_unique` | 唯一 ArticulationRootAPI，且落在 `/Geometry` |
| `test_rigid_body_mass_total_matches_the_table` | 9 个刚体、总质量 == `total_model_mass()`（1e-4 kg，表里是 4 位小数） |
| `test_formal_arm_rest_pose_is_inside_the_documented_limits` | `REBOT_RS` rest pose 6 个值都在限位内 |
| `test_reset_rest_joint_is_the_piper_pose_canary` | 金丝雀：`reset.py` 的 `joint3 = -1.5` 仍在限位外（F1） |

`joint_limits.py` 用 `importlib` 按**文件路径**加载（先看 `P5_JOINT_LIMITS`，再看 `REPO_ROOT/src/simulation/rebot_b601_rs/joint_limits.py`），
找不到就 `pytest.skip` 并打印原因，不会为凑绿去编数字。资产缺失同理 skip（资产是资源，不在仓库里）。

### 6.2 两次真跑（主机 CPU）

```bash
ssh dgut@172.31.68.251   # 只允许 CPU / pxr
# RUN A：把定义参考件按 d912202 铺到 /home/dgut/runs/p5/repo 下
cd /home/dgut/runs/p5/repo
/home/T7/ojh/robot_sim/env_isaaclab/bin/python -m pytest \
  tests/simulation/cad/test_arm_static_consistency.py -q -rs -p no:cacheprovider

# RUN B：只有 origin/main 的内容（没有 joint_limits.py / REBOT_RS 块）
cd /home/dgut/runs/p5/repo_mainonly
/home/T7/ojh/robot_sim/env_isaaclab/bin/python -m pytest \
  tests/simulation/cad/test_arm_static_consistency.py -q -rs -p no:cacheprovider
```

RUN A 真实输出与退出码：

```text
..............                                                           [100%]
14 passed in 0.60s
RUN_A_EXIT_CODE=0
```

RUN B 真实输出与退出码：

```text
..ssssssss.sss                                                           [100%]
SKIPPED [1] ...:160: joint_limits.py not found (tried: .../repo_mainonly/src/simulation/rebot_b601_rs/joint_limits.py). The B601-RS limit table lives on origin/feature/rebot-rs-sim-asset @ d912202, not on origin/main; set P5_JOINT_LIMITS to compare against it.
... (同一条原因共 11 次)
3 passed, 11 skipped in 1.41s
RUN_B_EXIT_CODE=0
```

完整输出留档：主机 `/home/dgut/runs/p5/pytest_p5_output.txt`（`tee` 落盘）。

---

## 7. UNKNOWN / 未解项

| # | UNKNOWN | 为什么现在不能定 | 怎么才能定 |
| --- | --- | --- | --- |
| U1 | **PhysX 运行时 DOF 顺序没有在本轮重新观测** | 本轮禁止使用 GPU（不得启动 Isaac Sim），只能引用既往真跑记录 `VERIFY_usd_load.json:98-107` | 在有权限的轮次跑一次 articulation 并 dump `robot.joint_names`，与本报告第 3.1 表比对 |
| U2 | **`joint_limits.py` 与 `ARM_MIGRATION_RECORD.md` 不在 `origin/main` 上** | `git ls-tree -r origin/main` 里 `rebot_b601_rs` 路径数为 **0**；两个文件只在 `origin/feature/rebot-rs-sim-asset` @ `d912202`（已 push、无 PR） | 把该分支合入 main，或让 P5 分支改基到 `d912202`（本轮按指示未改基） |
| U3 | reset.py 越限写入的**运行时后果**（是报错、被夹回、还是产生速度尖峰）未测 | 需要跑仿真 / 真机 | 有权限时用 t06/t08 复现并记录 joint3 轨迹 |
| U4 | URDF `axis = 0 0 ±1` 与 USD `physics:axis = Z` 的符号等价性 | USD 把符号折进了 `physics:localRot0/localRot1`（如 joint1 的 `localRot0 = (0,-1,0,0)` 即绕 X 转 180°），本轮是**按四元数推断**，没有做数值验证 | 用 pxr 变换矩阵把关节轴投到世界系逐关节比对符号 |
| U5 | 真机限位 | 官方文档没有给出实测角度范围，`joint_limits.py` 自己也标了 `usable_for_capability=False` | 需要实测这台单元 |

另有两条不算 UNKNOWN、但值得记住的事实：

* 资产的 `physics:breakForce` / `physics:breakTorque` 都是 `Infinity`（USD 的“永不破坏”回退值），不是有人显式配的阈值；JSON 里按字符串 `"Infinity"` 存，保证是严格 JSON。
* `--fix-base` 产生的 `root_joint` 是 Fixed 且 `body0 = /ReBot_Arm_RS`（Xform 根，不是刚体），所以它是“把臂钉在地上”的那一个，而不是 URDF 里那个 mid-chain 的 `j_gripper_end`（F4）。

---

## 8. 复现清单

```bash
# 1) 导出事实（主机 CPU，只读）
python scripts/simulation/p5_arm_static_facts.py \
  --usda <...>/ReBot_Arm_RS.usda --out outputs/simulation/rebot_b601_rs/morph_one/p5_static/arm_joint_facts.json

# 2) 测试（没有 pxr 的环境会整文件 skip；没有 joint_limits.py 的环境会跳过对照用例）
P5_ARM_USDA=<...>/ReBot_Arm_RS.usda \
P5_JOINT_LIMITS=src/simulation/rebot_b601_rs/joint_limits.py \
python -m pytest tests/simulation/cad/test_arm_static_consistency.py -q
```

校验量：资产 sha256 `96d4f2ec…bdd510`（前后一致）、事实 JSON sha256 `42e57775…a1d7df6`、
定义件 sha256 `69b078e5…3def1ac5e2a8`、导出脚本 sha256 `e29a5f2c…0dfa1d7df6` 见第 1 节表。
这三个提交内文件的 sha256 可以按**提交内容**复算（已核对：HEAD blob 与工作树同一哈希，即入库存的是 LF 版）：

```bash
for f in scripts/simulation/p5_arm_static_facts.py \
         tests/simulation/cad/test_arm_static_consistency.py \
         outputs/simulation/rebot_b601_rs/morph_one/p5_static/arm_joint_facts.json; do
  git cat-file blob <commit>:$f | sha256sum
done
```

注意：Windows 检出时 `core.autocrlf` 可能把工作树文件转成 CRLF，那么直接对文件算出的哈希会与上表不同；以上表（LF 版）为准。

## 9. Git

分支 `p5/arm-static-consistency`，base = `origin/main` @ `16f1ae4`。本轮只提交：
`scripts/simulation/p5_arm_static_facts.py`、`outputs/simulation/rebot_b601_rs/morph_one/p5_static/arm_joint_facts.json`、
`tests/simulation/cad/test_arm_static_consistency.py`、`management/morph_one_rs/P5_ARM_STATIC_CONSISTENCY.md`。
不含任何资产二进制（资产在主机 `/home/T7/dgut/robot_sim/assets/...`，按 AGENTS.md 资源不落本地）。
