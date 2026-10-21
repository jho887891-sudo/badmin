# P6 动态正运动学 FK(q)：B601-RS 8 自由度链

任务：P6 条目 C3（R+1）。分支 `p6/dynamic-fk`，基线 `p6/joint-anchor-fix` @ 3540109（PR #37）。
本文件是本切片唯一新增的文档；**没有修改** `mounted_chain.json`、`mounted_chain_v2.json`、`joint_anchors.json`、球拍 proxy、`assets/`、`scripts/simulation/p2_*/p3_*/p4_*`、`simulation/`、`src/` 既有文件、queue 状态文件或 `PHASE_P2_*/P3_*/P4_*`。

新增文件：

| 文件 | 作用 |
|---|---|
| `src/simulation/rebot_b601_rs/kinematics/fk.py` | 纯 numpy FK + 解析雅可比（新包，不改任何既有模块） |
| `src/simulation/rebot_b601_rs/kinematics/reference.py` | 两条**独立**参照路径（pxr Gf + UsdGeom.XformCache；纯四元数代数） |
| `src/simulation/rebot_b601_rs/kinematics/__init__.py` | 包标记 |
| `tests/simulation/p6_dynamic_fk_test.py` | 23 个纯 Python 测试 |
| `scripts/simulation/p6_dynamic_fk_validate.py` | 生成 `outputs/simulation/rebot_b601_rs/morph_one/p6_fk/fk_validation.json` |
| `management/morph_one_rs/P6_DYNAMIC_FK.md` | 本文件 |

---

## 1. FK 定义

一个 USD `PhysicsJoint` 把关节系钉在**两个**刚体上：

~~~
X0 = (physics:localPos0, physics:localRot0)     关节系在 body0 中的位姿
X1 = (physics:localPos1, physics:localRot1)     关节系在 body1 中的位姿
T_body0_body1(q) = X0 * M(q) * inv(X1)
~~~

`M(q)` 是**关节系内**的关节坐标：旋转关节绕关节轴转 q，移动关节沿关节轴平移 q。本资产每个关节的 `physics:axis` 都写作 `Z`，于是 `M(q) = Rz(q)`（revolute）或 `Tz(q)`（prismatic）。

整链（11 个位姿，8 个驱动关节 + 1 个固定根关节 + 1 个作者坐标系）：

~~~
arm_root --(root_joint, fixed, I)--> base_link
base_link --joint1--> link1 --joint2--> link2 --joint3--> link3
         --joint4--> link4 --joint5--> link5 --joint6--> link6
link6 --gripper_joint1(移动)--> gripper_left
link6 --gripper_joint2(移动)--> gripper_right
link6 --gripper_end(作者 xformOp, 固定)--> gripper_end
~~~

递归：`T_world_child = T_world_parent @ joint_local(name, q[name])`，`joint_local = X0 @ M(q) @ inv(X1)`。

**参考系**：所有位姿在 **arm root（/ReBot_Arm_RS）** 中，P3 证据记录该系与底盘系在 retarget 后同一（`mounted_chain.json` edge 2 仍是 UNKNOWN）。`fk(q, base=...)` 可换基座，其余代码不假设任何基座。

**q 的映射**：`q` 是 8 个**按名字**给出的值（joint1..joint6、gripper_joint1、gripper_joint2）。`coerce_q` 接受 `dict`（缺名/多名/非有限值直接报错），也接受长度 8 的序列但按模块文档顺序解释；**顺序与 USD stage 无关**（stage 里还有 `root_joint`，且关节 prim 顺序不是驱动顺序）。测试 `test_q_is_addressed_by_name_not_by_index` 用逆序 dict 验证结果完全一致。

**四元数约定**：锚点源头是 USD **wxyz**。`quat_to_matrix(q, order=...)` 强制调用方声明来源约定；`load_committed_anchors` 先读文件里声明的约定并核对（不符抛 `ConventionError`）。xyzw 与 wxyz 的互换**永不隐式**。

## 2. 使用的锚点数值（含来源行）

来源：`outputs/simulation/rebot_b601_rs/morph_one/p6_frames/joint_anchors.json`（sha256 钉在
`96d4f2ec…bdd510` / `8d18a8c9…ce02b6`，取自 GPU 主机只读 `ssh cat`）与
`…/p6_frames/evidence/robot_usd_joint_chain.json`（同一 `physics.usda` 的解析记录）。

| 关节 | 类型 | 轴 | localPos0 (m) | localRot0 (wxyz) | localPos1 | localRot1 (wxyz) | 源行（physics.usda） |
|---|---|---|---|---|---|---|---|
| root_joint | fixed | — | (0, 0, 0) | (1,0,0,0) | (0,0,0) | (1,0,0,0) | def 308 / pos0 314 / pos1 315 / rot0 316 / rot1 317 |
| joint1 | revolute | Z | (-0.00034283, -0.00098683, 0.075) | (0,-1,0,0) | (0,0,0) | (0,-1,0,0) | def 124 / axis 132 / body0 134 / body1 136 / 137..140 |
| joint2 | revolute | Z | (0.020343, 0.027237, 0.07) | (-0.70710546,0.7071081,0,0) | (0,0,0) | (1,0,0,0) | def 147 / axis 155 / 160..163 |
| joint3 | revolute | Z | (-0.236, 2.7755576e-17, 0) | (0,-1,0,0) | (0,0,0) | (0,-1,0,0) | def 170 / axis 178 / 183..186 |
| joint4 | revolute | Z | (0.228, -0.072746, 0.0045) | (0,-1,0,0) | (0,0,0) | (0,-1,0,0) | def 193 / axis 201 / 206..209 |
| joint5 | revolute | Z | (0.087, -0.048, -0.03075) | (-0.7071081,-0.70710546,0,-0) | (0,0,0) | (0,-1,0,0) | def 216 / axis 224 / 229..232 |
| joint6 | revolute | Z | (0.0365, 2.1175824e-22, 0.048) | (-3.31e-17,-0.70710546,-3.71e-17,0.7071081) | (0,0,0) | (0,-1,0,0) | def 239 / axis 247 / 252..255 |
| gripper_joint1 | prismatic | Z | (1.5351144e-07, 7.3385e-05, 0.124271) | (0.7071081,-0.70710546,-2.5973434e-06,2.597353e-06) | (0,0,0) | (1,0,0,0) | physics.usda gripper_joint1（证据文件记录） |
| gripper_joint2 | prismatic | Z | (1.5458967e-07, -7.3385e-05, 0.124271) | (0.7071081,0.70710546,2.5973434e-06,2.597353e-06) | (0,0,0) | (1,0,0,0) | physics.usda gripper_joint2（证据文件记录） |

作者坐标系 `gripper_end`（UsdGeom xformOp，不是关节）：
translate (0, 0, 0.16621)，orient wxyz (2.597488447e-06, -0.7071054578, -2.597497996e-06, -0.7071080804)。

关节限位（SI，写入 `fk.LIMITS`）：joint1 ±2.7999999539821165、joint2/joint3 [0, 3.1400000406473483]、
joint4/joint5 ±1.5700001075901369、joint6 ±3.1400000406473483、gripper_joint1 [0, 0.05] **米**、gripper_joint2 [0, 0.0715] **米**。
注意：`physics.usda` 里夹爪限位的 `limit_unit_as_authored` 字段写作 "degree"，但数值是米（0..0.05 / 0..0.0715），该单位字段是错的；证据文件据此算出的 `limit_upper_rad`（0.00087 / 0.00125）**不可用**，本模块按米处理。

**五个 180° body1 翻转**（localRot1 = (0,-1,0,0) wxyz，即绕 +X 180°）不是特例分支：它们就是 `inv(X1)`。joint1/joint3/joint4/joint5/joint6 的 X1 非单位，joint2 与 root_joint 的 X1 是单位。这也解释了为什么 q=0 时子连杆的 local 位姿并不等于 X0（例如 joint1 的 local 退化成纯平移）。

## 3. 零位残差

q = 0 时 `gripper_end` 与本仓库已提交的 pxr 参考（`evidence/p3_mount_frame_tree_at_0bbed19.json` → `/arm_chain_rest_pose/gripper_end_world_rest_pose`，来自 UsdGeom.XformCache.GetLocalToWorldTransform）对比：

| 量 | FK(q=0) | pxr 参考 | 差值 |
|---|---|---|---|
| x | 0.3017101699988416 | 0.301710169999 | |
| y | 2.62394596854716e-07 | 2.62633e-07 | |
| z | 0.21774671855448924 | 0.217746720407 | |
| 最大绝对位置差 | | | **1.8525107559064224e-09 m** |
| 姿态测地角 | | | **1.2287812309317777e-07 rad** |
| det(R) | 1.0000000000000002 | +1 | |

残差与 `mounted_chain_v2.json` 记录的 1.85e-9 m **逐位相同**：这是 USD 层里 float32 锚点的量化误差，不是约定错误（比 1e-6 容差低 5 个数量级）。

## 4. 非零位验证表

三个非零构型（第三个每个关节都"贴限位但在限位内"）：

| 构型 | joint1..joint6 (rad) | 夹爪 (m) | gripper_end 位置 (m) | det(R) 范围 | R 正交误差 | 在限位内 |
|---|---|---|---|---|---|---|
| A 混合符号 | -0.70, 0.90, -1.20, 0.40, -0.90, 1.70 | 0.02, 0.03 | (-0.056326, -0.254461, -0.114903) | [1.0, 1+9e-16] | 1.11e-15 | 否（见下） |
| B 全正 + 夹爪张开 | 0.30, 0.45, 0.60, 0.50, 0.35, 0.20 | 0.05, 0.0715 | (0.252882, -0.005525, 0.521544) | [1-1e-16, 1+1.3e-15] | 1.11e-15 | 是 |
| C 贴限位内侧 | 2.79, 3.13, 3.13, -1.56, 1.56, -3.13 | 0.049, 0.07 | (-0.385798, -0.359342, 0.131298) | [1.0, 1+1.3e-15] | 1.55e-15 | 是 |

- 构型 A 的 joint3 = -1.20 **故意**落在 joint2/joint3 的非对称限位 [0, π] 之外：它只用来说明 FK 的数值行为（det(R)=+1、正交、有限），**不是一个可达位姿**。构型 C 才是"贴限位但在限位内"的证据。
- 每个构型都另有一条树自洽检查：每个连杆位姿 == 父连杆位姿 × 该关节 local（< 1e-12）。
- "运动确实发生"的正向检查（`test_a_moving_joint_actually_moves_the_chain`）：逐个关节 +0.2 rad，末端位置必须移动 > 1 mm；两个夹爪关节必须只动自己的爪、不动 `gripper_end`。
- **实测发现**：joint6 只让末端**自转**、不平移（Δp = 0.0 m，Δθ = 0.2 rad）。原因不是 bug：作者的 `gripper_end` 局部平移是 (0, 0, 0.16621)，正好落在 joint6 轴线上（到轴线距离 = 0），所以 joint6 对该点是纯滚转。pxr 参照给出同一分裂（2.78e-17 m / 0.2000000000 rad），因此测试把 joint6 单列为"纯滚转"分支并写入 `fk_validation.json` 的 `joint6_is_a_pure_roll`。**任何声称 joint6 会平移末端的模型都与 pxr 矛盾。**

## 5. 雅可比检查

`J(q)` 为 6×8 几何雅可比（arm root 系）：1..3 行 d(position)/dq（m/rad，夹爪列 m/m），4..6 行 d(rotation vector)/dq（rad/rad，夹爪列 rad/m）。解析列：转动关节 `[z_w × (p_link − p_joint); z_w]`，移动关节 `[z_w; 0]`，z_w 取关节系世界姿态的第三列（真实世界轴，含 180° 翻转的影响）。

| 构型 | 解析 vs 中心差分（h = 1e-6） | 解析 vs FK 自身前向差分（h = 1e-6） |
|---|---|---|
| A | **1.063487076180536e-10** | 2.2237706162608895e-07 |
| B | **1.1119111187340991e-10** | 2.0797906441494263e-07 |
| C | **1.4094125866392915e-10** | 2.1813478939987363e-07 |

- 中心差分转动行用 log(R(q+h) R(q−h)^T)（对旋转向量求导，无欧拉角奇异），步长 1e-6 已写在 docstring 与 `jacobian_central(h=)` 参数里。
- 前向差分残差 ~2e-7 就是 O(h) 截断误差（h=1e-6），不是不一致。
- 夹爪列的角度行恒为 0（|·| < 1e-12），且每个夹爪只驱动自己的爪（列范数 > 0.99）。

## 6. 独立参照校验（两条，均为非零 q）

| 构型 | pxr Gf + UsdGeom.XformCache 最大位置差 (m) | pxr 最大测地角 (rad) | 纯四元数代数路径最大矩阵元差 |
|---|---|---|---|
| A | **8.834515341504812e-10** | 2.1073424255447017e-08 | 6.66e-16 |
| B | **1.4014189149424983e-09** | 0.0 | 4.44e-16 |
| C | **6.257481754978755e-10** | 0.0 | 7.77e-16 |

**参照路径怎么来的（可复现）**：

1. **pxr 路径**（`reference.reference_pxr`）：在**内存中**（`Usd.Stage.CreateInMemory`，不落盘、不碰资产）按真实臂层级建 prim：`/arm/base_link/link1/…/link6/{gripper_left,gripper_right,gripper_end}`，每个连杆写一个 `xformOp:transform`；每个矩阵由 **pxr 自己的数学**生成 —— 锚点原始 wxyz 四元数直接进 `Gf.Quatd → Gf.Rotation → Gf.Matrix4d.SetTransform`，关节运动用 `Gf.Rotation(Gf.Vec3d(0,0,1), deg)` / `Gf.Matrix4d.SetTranslate`，逆用 `Gf.Matrix4d.GetInverse`，最后由 `UsdGeom.XformCache.GetLocalToWorldTransform` 完成**父链累加**。也就是说：四元数→矩阵、4×4 求逆、矩阵乘法、层级复合全部由 pxr 的 C++ 代码完成，与 `fk.py` 的 numpy 公式无关。
   - 这里踩到并记录了一个真实约定坑：**Gf/UsdGeom 的变换是转置存贮的**（点左乘行向量 p·M，平移在最后一行），所以列向量写法 X0·M(q)·inv(X1) 在 Gf 里必须写成 inv(X1)·M(q)·X0，读回时再取转置。写错这个顺序时位置差 0.24–0.53 m、姿态差接近 π —— 这也是"零位通过≠正确"的又一例（q=0 时该顺序错误对 joint1/joint6 这类对称锚点不可见）。
2. **纯四元数路径**（`reference.reference_quat`）：旋转用 Hamilton 四元数乘积复合、平移与旋转分开携带、SE(3) 逆用（共轭四元数, −R^T t），不以整 4×4 矩阵相乘。与 FK 差 ~1e-16，即同一几何、不同算术顺序。

本轮**没有**用 Isaac，也**没有**在 GPU 主机上做任何 GPU 计算；本轮也未重新 ssh 取层（锚点已由 PR #37 只读抓取并以 sha256 钉住，本切片直接消费提交的 JSON）。

## 7. 测试与真实退出码

~~~
python -m pytest tests/simulation/p6_dynamic_fk_test.py -q
23 passed in 1.86s                                        exit 0

python -m pytest tests/simulation -k p6_ -q -rs
95 passed, 1 skipped, 196 deselected in 3.28s             exit 0
SKIPPED [1] tests/simulation/p6_joint_anchors_test.py:507: no read-only copy of
            physics.usda locally; anchors are pinned by sha256 instead
~~~

- 基线（不含本切片）：`python -m pytest tests/simulation -k "p6_ and not dynamic_fk" -q` → **72 passed, 1 skipped**（= 73 个用例，exit 0）。加上本切片的 23 个 → **95 passed, 1 skipped**（= 96 个用例）。新增 23 个全绿，既有 P6 套件保持绿。
- **唯一的 skip 是既有的**（PR #37 的测试需要本地 `physics.usda` 副本，本机没有，改用 sha256 钉锚点），本切片没有引入新的 skip。
- `python scripts/simulation/p6_dynamic_fk_validate.py` exit 0，写出 `outputs/simulation/rebot_b601_rs/morph_one/p6_fk/fk_validation.json`（本文所有数字均取自该文件，可逐一复核）。

## 8. UNKNOWN（本切片**没有**解决，也不声称解决）

1. **没有经过标定的 TCP / 法兰变换。** `gripper_end` 只是资产里作者的一个 xformOp 坐标系，不等于真实法兰面或真实夹持中心；没有测量、没有误差界。
2. **没有实测球拍面。** 球拍挂载链 `mounted_chain.json` edge 13/14/15（gripper_end→racket_mount→racket_root→racket_contact_frame）仍是 UNKNOWN，且夹爪是 0..0.05 / 0..0.0715 m 的滑动副，被夹住的球拍**不是**固定工具（`racket.py:LOCKED_JOINTS` 已拒绝）。所以本文的末端位姿**不是**拍面位姿，更不是击球点位姿。
3. **没有 D455 外参**，也没有相机与法兰/底盘之间的手眼关系。
4. **这是运动学，不是碰撞或动力学验证。** 没有网格碰撞检查、没有质量/惯量、没有摩擦、没有关节柔性或齿隙、没有驱动器限速/限矩；`within_limits` 只比较标量限位，不保证轨迹可达、不保证不撞、不保证力矩够。
5. **底盘到臂根仍是 UNKNOWN**（`mounted_chain.json` edge 1/2）：本文的"世界"是臂根系；把它当底盘系依赖 P3 的 retarget 结论，而不是新的测量。
6. **夹爪限位的单位字段是错的**（"degree" 但数值是米）；本模块按米处理，这是对资产的**读法**而非对资产的修改，需要原作者确认。
7. **joint2/joint3 的非对称限位 [0, π]** 使负角度不可达；构型 A 因此是非物理的数值测试（第 4 节已标注）。
