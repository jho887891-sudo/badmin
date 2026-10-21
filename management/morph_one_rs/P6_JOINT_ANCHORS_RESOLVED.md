# P6-02-OI-1 结案：机械臂关节 body1 侧锚点（physics:localPos1 / physics:localRot1）

- **任务**：关闭 Issue #13 的 P6-02-OI-1
- **分支**：`p6/joint-anchor-fix`（基线 `p6/racket-collision-proxy` @ `0e87c5e`）
- **结论**：**已解决**。关节边组合（composition B）与 pxr 参考位姿的偏差由 **0.33242 m / π rad** 降到 **1.852511e-09 m / 2.151961e-08 rad**，在 1e-6 m、1e-6 rad 双门限内 → **PASS**
- **未改动**：`mounted_chain.json` 一个字节都没动（历史数字与 5 条 UNKNOWN 边全部保留）；GPU 主机只读，未写入任何内容

---

## 1. 缺失的到底是什么

P6 帧树把每条转动关节边记成一个位姿 `translation_m` + `rotation_quat`，但 USD 的 `PhysicsJoint` 不是「一个位姿」，而是**同一个关节坐标系被钉在两侧刚体上的两个位姿**：

| 记号 | USD 属性 | 含义 |
|---|---|---|
| `X0` | `physics:localPos0` / `physics:localRot0` | 关节坐标系在 **body0** 中的位姿 |
| `X1` | `physics:localPos1` / `physics:localRot1` | 关节坐标系在 **body1** 中的位姿 |

从 body0 走到 body1 的正确变换是：

```
T_body0_to_body1 = X0 * inv(X1)
```

P6 帧树只记了 `X0`，等价于**默认 `X1` = 单位**。对本臂而言这个默认只在 joint2 和 root_joint 上成立；joint1/3/4/5/6 的 `localRot1` 是绕 +X 的 180° 翻转 `(0, -1, 0, 0)`（wxyz）。漏掉 `X1` 就等于把这 5 条关节的子连杆**整体翻了个面**。

### 为什么之前没被发现

1. composition A 与 B 的旋转矩阵都满足 det(R) = +1、都正交归一 → 所有「手性/单位」检查全绿；
2. 5 次 180° 翻转在旋转上部分相互抵消，最终累计成**恰好 π rad** 的整体朝向差，位置差 0.33242 m；
3. 单看任意一条关节边都自洽，只有**整链末端位置**与 pxr 参考对比才暴露。

所以这是**缺数据**，不是手性/单位 bug —— 与原 issue 的判断一致，现在被证实并修复。

---

## 2. 提取到的 body0 / body1 数值（含源码行号）

- 来源层：`payloads/Physics/physics.usda`（P3-01 已确认关节定义在该 payload）
- 层 sha256：`8d18a8c9271e0905df72530d177157bbdebeb2fceaae3de7628524240fce02b6`（**远端 sha256sum 与本地 sha256 逐字相同**，读取前后各校验一次）
- 臂资产 sha256：`96d4f2ec1126dec6eaf9e70ab664ba6cc30fa81bdec86178a3e50b7a23bdd510`（**一致**）
- 远端目录：`/home/T7/dgut/robot_sim/assets/robots/badminton_robot/rebot_b601_rs/usd/rebot_b601_rs.usd/ReBot_Arm_RS/`
- 读取方式：`ssh dgut@172.31.68.251 cat ...` → 落到本地 **gitignored** 的 `_scratch_p6c/`，解析用纯 Python 正则（无 pxr 依赖）
- 单位：米 / 弧度；四元数 **wxyz**（USD 约定）；层内 `metersPerUnit = 1.0`，全程不乘任何比例因子

| 关节 | USD 类型 | axis | body0 → body1 | localPos0 (m) | localPos1 (m) | localRot0 (wxyz) | localRot1 (wxyz) | X1=单位? | 源码行 (def/axis/body0/body1/Pos0/Pos1/Rot0/Rot1) |
|---|---|---|---|---|---|---|---|---|---|
| root_joint | PhysicsFixedJoint | — | `/ReBot_Arm_RS` → `.../base_link` | (0, 0, 0) | (0, 0, 0) | (1, 0, 0, 0) | (1, 0, 0, 0) | 是 | 308 / — / 311 / 313 / 314 / 315 / 316 / 317 |
| joint1 | PhysicsRevoluteJoint | Z | `base_link` → `link1` | (-0.00034283, -0.00098683, 0.075) | (0, 0, 0) | (0, -1, 0, 0) | **(0, -1, 0, 0)** | 否 | 124 / 132 / 134 / 136 / 137 / 138 / 139 / 140 |
| joint2 | PhysicsRevoluteJoint | Z | `link1` → `link2` | (0.020343, 0.027237, 0.07) | (0, 0, 0) | (-0.70710546, 0.7071081, 0, 0) | (1, 0, 0, 0) | 是 | 147 / 155 / 157 / 159 / 160 / 161 / 162 / 163 |
| joint3 | PhysicsRevoluteJoint | Z | `link2` → `link3` | (-0.236, 2.7755576e-17, 0) | (0, 0, 0) | (0, -1, 0, 0) | **(0, -1, 0, 0)** | 否 | 170 / 178 / 180 / 182 / 183 / 184 / 185 / 186 |
| joint4 | PhysicsRevoluteJoint | Z | `link3` → `link4` | (0.228, -0.072746, 0.0045) | (0, 0, 0) | (0, -1, 0, 0) | **(0, -1, 0, 0)** | 否 | 193 / 201 / 203 / 205 / 206 / 207 / 208 / 209 |
| joint5 | PhysicsRevoluteJoint | Z | `link4` → `link5` | (0.087, -0.048, -0.03075) | (0, 0, 0) | (-0.7071081, -0.70710546, 0, -0) | **(0, -1, 0, 0)** | 否 | 216 / 224 / 226 / 228 / 229 / 230 / 231 / 232 |
| joint6 | PhysicsRevoluteJoint | Z | `link5` → `link6` | (0.0365, 2.1175824e-22, 0.048) | (0, 0, 0) | (-3.3148634e-17, -0.70710546, -3.7116984e-17, 0.7071081) | **(0, -1, 0, 0)** | 否 | 239 / 247 / 249 / 251 / 252 / 253 / 254 / 255 |

要点：

1. **7 条关节的 `localPos1` 全是原点**，真正缺失的信息**只在旋转上**，即 `localRot1`；
2. 5 条转动关节的 `localRot1` 是 `(0, -1, 0, 0)`（绕 +X 的 180°），只有 joint2 和 root_joint 是单位四元数；
3. `localRot0` 与 P6 帧树里已记录的值**逐位一致** → 反向证实「帧树记的就是 body0 侧」，诊断成立；
4. 每条关节的 4 个锚点都带**源文件行号**（见上表与 `joint_anchors.json` 的 `source_lines`），任何人拿同一 sha256 的层都能逐行核对。

---

## 3. 组合误差：before / after

组合链：`chassis → arm_base → base_link → link1..link6 → gripper_end`，参考真值为 pxr 组合出的 `gripper_end` 世界位姿 `(0.301710169999, 2.62633e-07, 0.217746720407)` m。

| 组合 | 用了什么 | 位置最大偏差 | 朝向测地角 | 判定（阈值 1e-6 m / 1e-6 rad） |
|---|---|---|---|---|
| **B v1**（历史，`mounted_chain.json`） | 只有 `X0`，丢掉 `X1` | **0.33242 m** | **π rad（180.000°）** | FAIL |
| **B v2**（本次修复） | `X0 * inv(X1)`，两侧都用 | **1.852511e-09 m** | **2.151961e-08 rad** | **PASS** |

- 平移改善倍数：**1.79e8**
- 残差来源：USD 层里的 `point3f` / `quatf` 是 **float32**，1.85e-9 m 就是 float32 舍入，比门限低 5 个数量级；不是残留的约定错误（否则应为 O(0.1 m) 量级）。
- **自证**：用「只留 `X0`」的同一段代码重算，能复现 `mounted_chain.json` 里记录的 B 矩阵，最大元素差 **4.588e-13** → 说明 v1 与 v2 的差别**只**来自 body1 锚点，没有混入其它改动。
- 朝向差的 π 是一处强证据：5 次 180° 翻转累计成整体翻转，位置偏差 0.33242 m 只是它的副产品。

---

## 4. 产物（全部为新增文件，无覆盖）

| 文件 | 作用 |
|---|---|
| `outputs/simulation/rebot_b601_rs/morph_one/p6_frames/joint_anchors.json` | **新增**。7 条关节的 `X0`、`X1`、`T_body0_to_body1`、关节类型/轴、**源行号**、层 sha256。sha256(LF) = `13030fb3eb5daaba6db28406c7f7388e49e56427f08bc1670d0d6e8a9ab9467b` |
| `outputs/simulation/rebot_b601_rs/morph_one/p6_frames/mounted_chain_v2.json` | **新增**。以**指针**继承 `mounted_chain.json`（不复制数字），追加 body1 扩展、composition B v2、before/after 误差、issue 结案段。sha256 = `bf7b6361cca80543aa902cfdbf16ce36ac4f1af31945594473c1846b1dc234a6` |
| `tools/p6_frames/build_joint_anchors.py` | 生成器：解析 USD 文本 → 写上面两个 JSON。无 pxr、无 GPU、可重跑 |
| `tests/simulation/p6_joint_anchors_test.py` | 24 条回归测试（见第 5 节） |
| 本文件 | 结案报告 |

**历史不被重写**：`mounted_chain.json` 的 sha256(LF) = `7228b55930de8208d4ab195d6c9ceacc688905aa61ea4eb83df6d80892747772`，由 v2 记录并由测试逐次校验；它里面的 `open_issue.status` 仍是 `OPEN`、`max_abs_delta_m` 仍是 `0.33242`。**结案记录写在旁边，不覆盖在上面。**

---

## 5. 测试与真实退出码

```
python -m pytest tests/simulation -k p6_ -q
73 passed, 196 deselected          exit code 0
```

```
python -m pytest tests/simulation/p6_joint_anchors_test.py -q
24 passed                          exit code 0
```

新增测试的四道闸门：

1. **漂移闸门** `test_composition_b_v2_agrees_with_pxr_within_1e_6`：从**已提交的 JSON** 重新推导（不读回文件里存的数字，且**故意不 import 生成器**，避免「用同一段错代码测自己」），要求 ≤1e-6 m 且 ≤1e-6 rad。
2. **UNKNOWN 闸门**：5 条 UNKNOWN 边（1/2/13/14/15）必须仍然 `value_ref = null`、无任何数值 payload、自由文本里也不许夹带数字；v2 的 `unknown_edges_unchanged` 只允许出现边号整数，出现任何浮点即失败。**给安装偏移猜一个数就会当场失败。**
3. **历史闸门**：`mounted_chain.json` 的 LF 归一 sha256 必须等于 v2 记录值；A / B v1 两个矩阵必须与基线**逐字相等**；`open_issue` 必须仍是 `OPEN` / 0.33242。
4. **重解析闸门** `test_reparsed_layer_reproduces_every_recorded_anchor`：若本地存在该层的只读副本（`P6_PHYSICS_USDA` 或 `_scratch_p6c/physics.usda`），就从原始文本重新解析并与已记录锚点逐值比对；副本缺失时 `pytest.skip`（**不是静默通过**）。

### 红-绿已验证（测试不是空跑）

把 joint3 的 `localRot1` 改回单位四元数（即重现 v1 的 bug），漂移闸门立刻失败：

```
AssertionError: joint-edge composition drifted 1.454934e-01 m from the pxr end pose
(tolerance 1.0e-06 m); the body1 anchors are missing or wrong
1 failed, 1 passed            exit code 1
```

随后已还原，套件回到 0 失败。

---

## 6. 仍然 UNKNOWN（本次**没有**回答，也不该回答）

本次只解决了**机械臂内部**这一段。以下边界原样保持 UNKNOWN，**没有**填入任何数字：

| 边 | 边界 | 仍然缺的量 |
|---|---|---|
| 1 | `world → chassis` | 真机部署时 Morph One 底盘的世界位姿 (x, y, yaw) |
| 2 | `chassis → arm_base` | **物理安装偏移与偏航**（B601-RS 底座在 Morph One 顶板上的孔位/朝向，`T_chassis_armbase`）。P3-01 的恒等变换是 **sim authored**，只说明仿真里两个坐标系重合，**不是**测量值 |
| 13 | `gripper_end → racket_mount` | `T_flange_tcp`：球拍手柄轴从法兰何处伸出、绕轴如何定相（`configs/racket.yaml` 仍为 `REQUIRES_MEASUREMENT`） |
| 14 | `racket_mount → racket_root` | 手柄插入深度、拍面 +X 法线绕轴定相 |
| 15 | `racket_root → racket_contact_frame` | 实测拍面中心 / 甜区中心（`head_center_from_tcp=0.50` 明确标为近似） |

另外**完全未触碰**：

- **D455 双目外参**（`camera_metadata.json` 里 intrinsics / extrinsics / depth_scale / distortion 全部仍为 UNKNOWN）；
- 资产 `assets/`、`simulation/`、`src/`、`scripts/simulation/p2_*/p3_*/p4_*`、P2/P3/P4 文档、队列状态文件。

一句话：**机械臂的关节几何现在是可复现的；机器人在世界里的位置、球拍怎么装在法兰上、相机在哪，仍然要实测。**

---

## 7. 如何复现 / 如何核对

```bash
# 1) 只读取层（GPU 主机不写入任何东西）
ssh dgut@172.31.68.251 "sha256sum <...>/payloads/Physics/physics.usda"
ssh dgut@172.31.68.251 "cat <...>/payloads/Physics/physics.usda" > _scratch_p6c/physics.usda
# 2) 校验本地副本 == 远端（必须逐字相同）
sha256sum _scratch_p6c/physics.usda   # 8d18a8c9...
# 3) 重算两个 JSON
python tools/p6_frames/build_joint_anchors.py --physics-usda _scratch_p6c/physics.usda --remote-sha256 8d18a8c9271e0905df72530d177157bbdebeb2fceaae3de7628524240fce02b6
# 4) 回归
python -m pytest tests/simulation -k p6_ -q
```

拿不到 GPU 主机也能核对：`joint_anchors.json` 记录了层的 sha256 与**每一行行号**，`mounted_chain_v2.json` 记录了 `joint_anchors.json` 的 sha256；两者都由测试每次校验，sha256 一变即失败。
