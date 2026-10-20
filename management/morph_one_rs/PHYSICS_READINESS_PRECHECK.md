# Morph One + B601-RS 物理建模前置核查（只读证据）

分支：`feat/morph-one-physics-sandbox`　基线：`b0853efe375bda931b84ed401c7ccff2d6ce950e`
本文件只回答"能不能、需不需要破坏结构、缺什么数据"，**不建立整机物理装配**，也不声明任何真实质量/惯量/摩擦。

---

## 0. 结论速览

| 问题 | 状态 | 一句话结论 |
|---|---|---|
| C1 逐件刚体拆分可行性 | **VERIFIED（结构事实）** | 282 个网格实体只以"1 个 prototype 集合 + 1 个 instance"形式存在，资产内**零**刚体/碰撞体/关节 |
| C2 是否必须破坏 instance/prototype | **VERIFIED** | 必须二选一：在 prototype 里写碰撞体（所有 instance 共享），或解除 instanceable；本例 prototype:instance = **282:282（1:1）**，解除实例化**不损失任何复用** |
| C3 单 articulation 容纳底盘+机械臂 | **CANDIDATE（已做组合验证，未做动力学）** | 机械臂自带 1 个 ArticulationRootAPI + 9 刚体 + 9 关节；重定目标后组合层仍**只有 1 个** articulation root，7/7 自检通过 |
| C4 机械臂既有固定根关节如何处理 | **VERIFIED** | `/ReBot_Arm_RS/Physics/root_joint`（PhysicsFixedJoint）body0 = `/ReBot_Arm_RS`（**不是刚体**）、body1 = `/ReBot_Arm_RS/Geometry/base_link`；挂底盘＝把 body0 改成底盘刚体 |
| C5 关节限位 / 增益 / 求解器设置能否保留 | **VERIFIED** | 8 个关节带限位、8 个带驱动，全部写在资产里；**资产自身增益 ≠ 仓库运行时增益**（见第 5 节） |
| C6 必须等真实数据的量 | **UNKNOWN** | 质量、惯量、质心、摩擦、安装刚度、轮胎半径、转向电机力矩/减速比，资产里一个都没有 |
| 附带风险（新发现） | **VERIFIED_MISMATCH** | 资产 `upAxis = Z`，但其几何按 Y 竖直建模（与工作流 B 的测量一致）→ 直接 spawn 会侧躺 |

---

## 1. 底盘资产结构（pxr 0.25.11 实测）

资产：`/home/T7/dgut/runs/cad_dev/usd/M1-4WD4S_model2601_dev.usda`（58,659,744 B）

| 项 | 值 |
|---|---|
| defaultPrim | `/tn__M14WD4S_model2601_dev_VV0` |
| metersPerUnit | **0.001（毫米）** |
| upAxis（元数据） | **Z** |
| prim 总数 | 1,141（Mesh 564 / Xform 567 / Scope 4 / Material 3 / Shader 3） |
| 三角面顶点数 | 455,255（经 564 个 mesh 计数得 910,510，即 prototype 与 instance proxy 各算一次） |
| instanceable prim | **1 个**：`/tn__M14WD4S_model2601_dev_VV0/tn__M1100V3_qAD` |
| instance / instance proxy | 1 / 567 |
| prototype 分组 | `/tn__M14WD4S_model2601_dev_VV0/Prototypes`：282 mesh、455,255 点 |
| 引用侧 | `/tn__M14WD4S_model2601_dev_VV0/tn__M1100V3_qAD`：282 mesh，**全部**为 instance proxy |
| 物理 schema | 空——无 UsdPhysics.Scene / RigidBodyAPI / CollisionAPI / MassAPI / Joint / ArticulationRootAPI |

**关键结构事实**：整个模型是"一个 prototype 集合被一个 instance 引用一次"。282 个 prototype mesh 对 282 个
instance proxy mesh 是 **1:1**，所以解除 instanceable 不会让任何几何被复制多份。

## 2. C1：逐件刚体拆分

* 可以做：282 个 mesh 每个都有稳定路径（`.../Prototypes/<name>`）与唯一 STEP 实体号映射
  （PR #8 已验收 282/282 配对），足以把每个零件或每个模块指派为刚体。
* 目前**不能**直接做：资产里没有任何刚体、碰撞体、质量、关节；`physics_authoring_present = false`。
* 拆分粒度建议（未实施）：按模块（2 个舵轮模块 + 主体模块 + 4 个未命名轮组件）先做 7 个刚体，再按零件细化。

## 3. C2：是否必须破坏 instance/prototype

必须处理，理由来自 USD 语义而不是偏好：

1. 写在 prototype mesh 上的 CollisionAPI 会**被该 prototype 的所有 instance 共享**；若需要"每个实例不同的
   质量/摩擦/碰撞近似"，共享就不够。
2. 本例 prototype 与 instance 是 1:1 使用，**共享没有收益**，因此推荐路径是**解除实例化**（把
   `/tn__M1100V3_qAD` 的 instanceable 关掉）。几何本来只有一份，取消后仍只有一份，体积不会翻倍。
3. 另一条更省事的路：保留 instanceable，把碰撞体/mass 写进 prototype。二者都能成立，取决于"是否需要逐件
   不同参数"。本报告不替研究负责人决定，只列出两者的代价。

补充（单位）：资产以毫米为单位（mpu 0.001），把它引用进米制场景**必须**显式缩放。沙箱里对
`/MorphOneMount/Chassis` 实际写入的 `xformOp:scale = 0.001` 就是这条规则的证据；机械臂资产 mpu = 1.0，
缩放 1.0（无需转换）。

## 4. C3 / C4：单 articulation 与既有根关节

机械臂资产：`.../rebot_b601_rs.usd/ReBot_Arm_RS/ReBot_Arm_RS.usda`（mpu 1.0，upAxis Z）

| 项 | 值 |
|---|---|
| ArticulationRootAPI | **1 个**：`/ReBot_Arm_RS/Geometry` |
| 刚体 | 9 个（base_link → link1..link6 → gripper_left / gripper_right） |
| 关节 | 9 个：1 Fixed + 6 Revolute + 2 Prismatic |
| 组合层 | ReBot_Arm_RS.usda, geometries.usd, materials.usda, instances.usda, base.usda, robot.usda, physx.usda, physics.usda |

既有固定根关节（定义在 physics.usda 里）：

    /ReBot_Arm_RS/Physics/root_joint   PhysicsFixedJoint
      body0 = /ReBot_Arm_RS                     <- 机械臂的 Xform，不是刚体
      body1 = /ReBot_Arm_RS/Geometry/base_link  <- 真正的第一刚体
      localPos0 = (0,0,0)   localPos1 = (0,0,0)

因为 body0 不是刚体，这个关节等价于"把 base_link 钉在世界坐标"——这正是现在单独 spawn 机械臂能工作的
原因，也正是**挂到底盘上必须改的地方**：把 body0 指向底盘根刚体（或引入中间安装刚体），body1 保持
`base_link` 不变。

**已做的组合验证**（`mount_override.usda`，1,262 B，只在新层里写一个 over，不改源资产）：

* `over "Arm/Physics/root_joint" { rel physics:body0 = </MorphOneMount/Chassis> }` 生效：
  body0 = `/MorphOneMount/Chassis`、body1 = `/MorphOneMount/Arm/Geometry/base_link`，两者都解析成功；
* 组合后 articulation root 仍然**只有 1 个**（`/MorphOneMount/Arm/Geometry`）；
* 底盘资产与机械臂资产 sha256 **前后一致**（`assets_unchanged = true`）。

代理 articulation（`mount_proxy.usda`，4,738 B）自检：1 个 articulation root、DOF = 2（两个 revolute）、
每个关节 body0/body1 都能解析、含 1 个 FixedMount、4 个刚体（底盘板 + 代理底座 + 两个连杆）。

**这不是动力学验证**：没有任何 PhysX 步进，`physics_stepped = false` 写在 JSON 里。真实惯性/接触行为必须
在 Isaac 里另做一次（占单 GPU 槽位），在此之前"单 articulation 方案可行"只能算 CANDIDATE。

## 5. C5：必须保留的既有配置

资产自身写死的（任何底盘侧改动都不得触碰）：

| 关节 | 类型 | 轴 | 限位（资产单位：度 / 米） | 资产驱动（stiffness / damping / maxForce） |
|---|---|---|---|---|
| joint1 | Revolute | Z | −160.428177 … 160.428177 | 1.745329 / 0.017453 / 36.0 |
| joint2 | Revolute | Z | 0 … 179.908752 | 同上 |
| joint3 | Revolute | Z | 0 … 179.908752 | 同上 |
| joint4 | Revolute | Z | −89.954376 … 89.954376 | 1.745329 / 0.017453 / 14.0 |
| joint5 | Revolute | Z | −89.954376 … 89.954376 | 同上 |
| joint6 | Revolute | Z | −179.908752 … 179.908752 | 同上 |
| gripper_joint1 | Prismatic | Z | 0 … 0.05 | 100.0 / 1.0 / 500.0 |
| gripper_joint2 | Prismatic | Z | 0 … 0.0715 | 100.0 / 1.0 / 500.0 |

**重要区分**：仓库运行时配置（`simulation/badminton_scene/rs_physics.py` + `scene_layout.REBOT_RS`）用的是
TGS（`solver_type = 1`）、`enable_external_forces_every_iteration = True`、stiffness 300 / damping 20，并有实测
依据（静止保持积分误差 0.134867 → 0.000143 rad）。这与资产里写的 1.745329 / 0.017453 **不是同一个东西**：
前者是经过 A/B 实测选定的运行时覆盖，后者是资产出厂默认。合并底盘时**只允许在运行时 cfg 上做**，不得
改写资产增益与限位。

## 6. C6：必须等真实数据的量（全部 UNKNOWN）

1. 底盘总质量、四个轮组件质量、主体模块质量与各自质心/惯性张量（资产内 MassAPI 计数 = 0）。
2. 轮胎与地面的静/动摩擦系数、滚动阻力（资产内 CollisionAPI 计数 = 0）。
3. 轮胎标称外径与宽度（工作流 B 只测到 103.0 mm 的圆特征半径与 48.653 × 180.782 × 157.783 mm 包围盒）。
4. 转向电机额定力矩、减速比、最大转向角速度；轮毂电机额定转速与力矩曲线。
5. 机械臂与底盘之间安装件的刚度/阻尼、螺栓预紧，是否需要柔性建模。
6. 各关节真实摩擦/回差（现在只有仿真驱动参数，没有任何实测辨识值）。

## 7. 附带发现：upAxis 与几何竖直轴不一致

* 资产元数据 `upAxis = Z`；工作流 B 用真实几何测得**竖直轴是 Y**（四轮实体 y 恒为 −244.000 mm 且最低点
  = 模型 Y 最小 −334.391 mm；主体模块 Y 厚度仅 132.837 mm）。
* 转换过程没有旋转几何（转换后包围盒仍为 X 580.0 / Y 372.996 / Z 802.852 mm，轴向分配与 STEP 一致），
  所以这是"元数据声明"与"内容"不一致，而不是几何问题。
* **方向符号审计**：采用右手定则，绕 X 轴 **+90°** 时 (0,+1,0) → (0,0,+1)，即 CAD 的 +Y-up 才会映射到世界 +Z-up；绕 X 轴 −90° 时会映射到 −Z（上下颠倒）。旧版推荐的 −90° 号数错误。
* **证据边界**：Y-up 是由轮状实体位置、主体形状和最小外包围轴推断的强候选，USD 元数据 Z-up 与几何候选存在差异；没有厂家坐标系说明或独立可视化前，不把重力方向或最终安装朝向宣称 VERIFIED。当前 `mount_override.usda` **仅做了 0.001 缩放，没有旋转**，不具备直立真实仿真的验收意义。
* **Articulation 边界**：计数为 1 个 `ArticulationRootAPI`、joint body0/body1 路径可解析，只证明 USD 静态引用可组合；PhysX 是否把底盘和机械臂视作一个 articulation 尚需真实步进、DOF/链接数/跟随误差检查。
* 如果按 Y-up 候选姿态不做旋转，底盘将在 Z-up 世界侧躺。**必须先决定**：若确认 CAD +Y 为上，spawn 时绕 X 轴 **+90°** 旋转（候选，零成本、可回滚），
  还是重新导出 Z-up 资产（要重跑转换与全部验收）。此项在拿到厂商姿态图或一次渲染确认前保持
  **VERIFIED_MISMATCH / 待决定**。

## 8. 工具与复现

| 文件 | 作用 |
|---|---|
| `scripts/simulation/morph_one_physics_readiness.py` | 只读打开资产，统计 instance/prototype/物理 schema/分组，产出 C1–C6 事实与结论标签 |
| `scripts/simulation/morph_one_mount_sandbox.py` | 生成 TEST_PROXY 代理 articulation + 真实资产的 over 挂载覆盖层，含 7 项自检与源资产 sha256 前后比对 |
| `tests/simulation/cad/test_physics_readiness.py` | 5 项（需 pxr，缺失时 skip）：单位/计数/物理探测/结论标签/代理 articulation/代理自标注 |

运行（Isaac 自带解释器，CPU 即可）：

    python scripts/simulation/morph_one_physics_readiness.py \
      --chassis-usd M1-4WD4S_model2601_dev.usda --arm-usd ReBot_Arm_RS.usda \
      --output-json physics_readiness.json
    python scripts/simulation/morph_one_mount_sandbox.py \
      --chassis-usd ... --arm-usd ... --output-usda mount_proxy.usda \
      --override-usda mount_override.usda --output-json mount_sandbox.json --self-check

结果：`checks_passed 7 / 7`，`assets_unchanged true`，`proxy_dof 2`，`SANDBOX_EXIT 0`；
`pytest tests/simulation/cad/test_physics_readiness.py` → **5 passed**（pxr 0.25.11，Isaac 解释器）。

证据（仓库内 `outputs/simulation/rebot_b601_rs/morph_one/physics_b0853ef/`）：
`physics_readiness_chassis.json`、`physics_readiness_arm.json`、`mount_sandbox.json`、
`mount_proxy.usda`、`mount_override.usda`。

## 9. 下一步（安全顺序）

1. 决定第 7 节的 upAxis 方案（旋转 vs 重新导出），并做一次可视化确认——这是所有后续动力学的先决条件。
2. 决定第 3 节的两条路径之一（prototype 内写碰撞体 / 解除实例化）；建议后者，因为 1:1 复用无收益。
3. 用代理件先跑一次真正的 PhysX articulation 步进（单 GPU 槽位，标注 TEST_PROXY），确认
   "底盘刚体 + 固定关节 + 机械臂 8 DOF"运动学可控、无自碰撞爆炸；成功后再替换为真实几何。
4. 向厂商/实测索取第 6 节 6 项数据；在此之前不建立任何质量/惯量/摩擦参数。
