# PhysX 动态验证（B 线）：代理 articulation → 真实 B601-RS 移动挂载 → 朝向

分支 `feat/morph-one-physx-dynamic-validation`　基线 `9bafbfe051b05ff9b58c6bcfee477dafa78e71bf`（= PR #10 tip）
环境：Isaac Sim 6.0.1 + Isaac Lab 3.0.0-beta2，`/home/T7/ojh/robot_sim/env_isaaclab/bin/python`，
TGS（`PhysxCfg(solver_type=1, enable_external_forces_every_iteration=True)`），dt = 1/120 s，每条 180 步 = 1.5 s。
**本轮所有 GPU 运行都是真实 PhysX 步进**，除特别标注外不是静态 USD 检查。代理件一律标注 TEST_PROXY。

---

## 0. 结论速览

| 项 | 状态 | 关键数字 |
|---|---|---|
| B1 代理装配形成 articulation | **VERIFIED** | 4 刚体 / 2 关节 / 2 DOF，底盘在 articulation 内 |
| B1 底盘可被驱动 | **VERIFIED** | 零重力下 1.5 s 位移 **0.600 m**（指令 0.4 m/s） |
| B1 固定安装关节相对位姿误差 | **VERIFIED** | 全程 **0.000000 m**（max 也是 0） |
| B1 原有 root API 是否把底盘排除在 articulation 外 | **否（上一轮的担心被证伪）** | 三种 root 放置都是 4 刚体、底盘在内 |
| 在 `/Mobile` 再加一个 root API（想"提到最上层"） | **FAILED（被引擎拒绝）** | 见第 4 节原始报错 |
| B2 真实 B601-RS 移动挂载 | **VERIFIED** | **10 刚体 / 8 DOF**，底盘在 articulation 内，安装误差 0.0 |
| B2 机械臂关节低速控制（保留 300/20 增益） | **VERIFIED** | 指令 0.1 rad → 0.099954 rad，稳态误差 **4.6e-05 rad** |
| B2 异常碰撞 / 发散 | 有异常碰撞，无发散 | 代理底盘方块与真实臂基座几何重叠 → **0.479 m** 竖直漂移；无 NaN、无发散 |
| B3 朝向（0.001 缩放 + 绕 X **+90°**） | **VERIFIED** | 世界包围盒 (0.58, 0.802852, 0.372996) m，竖直轴变成 **Z** |
| B3 四个候选轮状实体的 USD 网格 bbox 与 STEP 顶点 bbox | **UNKNOWN / 需解决** | Y/Z 205.991 mm vs 180.782 mm（差 25.2 mm） |

---

## 1. 代理装配（B1）

几何（TEMP，全为方块，绝无"真机"含义）：底盘 0.5×0.35×0.06 m / 12 kg；臂基座 0.08³ m / 2 kg；连杆
0.28×0.05×0.05 与 0.22×0.04×0.04；两个 RevoluteJoint（限位 ±90°、±120°，驱动 50/5/100）位于臂子树内；
一个 FixedJoint 把底盘刚体接到臂基座。

### 1.1 三种 articulation root 放置（重力开）

| 变体 | root API 位置 | PhysX 刚体 | 关节 | DOF | 底盘在内 | root 位移 m | 安装误差 m | max\|q̇\| rad/s | NaN | 发散 |
|---|---|---|---|---|---|---|---|---|---|---|
| arm_rooted | `/Proxy/Arm`（与真实资产同位） | 4 | 2 | 2 | 是 | 0.113225 | 0.0 | 20.173269 | 0 | 否 |
| whole_rooted | `/Proxy`（整机上层） | 4 | 2 | 2 | 是 | 0.113225 | 0.0 | 20.173269 | 0 | 否 |
| chassis_rooted | `/Proxy/Chassis`（底盘自身） | 4 | 2 | 2 | 是 | 0.041995 | 0.0 | 15.894356 | 0 | 否 |

刚体顺序不同（arm/whole：`base_link, link1, Chassis, link2`；chassis：`Chassis, base_link, link1, link2`）说明
articulation 的"根刚体"随 API 位置改变，但**成员集合不变**：三种放置都把底盘包含进来。

### 1.2 零重力（把"能不能动"从自由落体里分离出来）

| 变体 | root 位移 m | 水平位移 m | 末端 z m | yaw 变化 rad | 底盘位移 m | 臂基座位移 m | max\|q̇\| rad/s | NaN | 发散 |
|---|---|---|---|---|---|---|---|---|---|
| arm_rooted | 0.600000 | 0.600000 | 0.549297 | 1.685521 | 0.600000 | 0.600000 | 20.097458 | 0 | 否 |
| whole_rooted | 0.600000 | 0.600000 | 0.549297 | 1.685521 | 0.600000 | 0.600000 | 20.097458 | 0 | 否 |
| chassis_rooted | 0.600000 | 0.600000 | 0.392024 | −0.040812 | 0.600000 | 0.600000 | 23.471865 | 0 | 否 |

* 位移 **0.600 m = 0.4 m/s × 1.5 s**，与指令完全一致 → 悬浮基座确实被 PhysX 驱动，底盘和臂基座同步移动
  （两者位移逐位相同，差值 0.0）。
* arm_rooted 与 whole_rooted 的数值**逐位相同**，说明这两种 root 放置对动力学没有区别。
* 代理的关节瞬态很大（|q̇| ≈ 20 rad/s，末端高度漂移 0.55 m）：这是代理件 + Isaac Lab 默认隐式增益造成的
  **代理假象**，不是真实机械臂的性质。记录在案，不作为结论使用。同时如实记录：日志里 Isaac Lab 打印的关节
  信息表显示实际生效刚度 **2.9e+03**、阻尼 **286.479**，覆盖了我在 USD 里写的 50/5——即"代理关节并不由我
  写入的增益驱动"。

---

## 2. 真实 B601-RS 移动挂载（B2）

场景（`real_arm.usda`，1,198 B）：`/Mobile` 下放一个代理底盘方块刚体（0.6×0.4×0.1 m / 25 kg），
**引用真实臂资产**（只读），并把臂自带的 `/Mobile/Arm/Physics/root_joint` 的 **body0 重定目标到底盘刚体**
（只在新层里写 over，资产本身未被改写）。arm 的 ArticulationRootAPI 保持在资产原位
`/Mobile/Arm/Geometry`。

| 运行 | 刚体 | 关节 | DOF | 底盘在内 | root 位移 m | 安装误差 m | max\|q̇\| rad/s | NaN | 发散 |
|---|---|---|---|---|---|---|---|---|---|
| 重力开、默认增益 | **10** | 8 | **8** | 是 | 0.016158 | 0.0 | 0.117452 | 0 | 否 |
| 零重力、**验证增益 300/20**、含关节阶跃 | **10** | 8 | **8** | 是 | 0.585757 | 0.0（max 0.0） | 0.018824 | 0 | 否 |

零重力 + 验证增益那一次的细节：

    chassis_moved_m            = 0.523199        # 底盘实际位移（指令 vx 0.4 m/s，含 0.3 rad/s 转向）
    arm_base_moved_m           = 0.523199        # 与底盘逐位相同
    fixed_mount_relative_error = 0.000000        # 固定安装关节全程零误差
    max_abs_joint_pos          = 0.001579 rad
    root_end_z_m               = 0.478883        # 见第 5 节：这是异常碰撞造成的竖直漂移
    non_finite_samples         = 0
    diverged                   = false

关节低速控制：PhysX 给出的关节顺序是

    ['joint4','joint3','joint5','joint2','joint6','joint1','gripper_joint1','gripper_joint2']

注意**不是** URDF 顺序，所以我下发的"索引 0"实际是 **joint4**（如实记录，不改成 joint1 来凑结论）。
指令 0.1 rad → 终值 **0.099954 rad**，**稳态误差 4.6e-05 rad**，过程最大 |q̇| 1.333686 rad/s，全程有限。

即：**在不改动真实机械臂关节拓扑、限位与资产内容的前提下，真实 B601-RS 可以和底盘刚体组成同一个
PhysX articulation（10 刚体 / 8 DOF），固定安装约束零误差，关节低速可控。**

---

## 3. root API 位置的结论（回答本轮特别检查项）

1. "原有 ArticulationRootAPI 位置是否导致底盘被排除在 articulation 外" → **不会**。代理件三种放置（臂子树 /
   整机上层 / 底盘自身）都得到 4 刚体、底盘在内；真实臂在"保持资产原位"的情况下得到 10 刚体、8 DOF。
2. "是否可以把 root 提到整机上层再挂" → 若**同时**保留资产自带的 root API，Isaac Lab/PhysX 直接拒绝：

       RuntimeError: Expected exactly one ArticulationRootAPI prim under '/World/Mobile'
       (resolved from '/World/Mobile'), found 2: ['/World/Mobile', '/World/Mobile/Arm/Geometry'].

   也就是说这条"换 root"的路必须先把资产自带的 root API 去掉（要改资产或写删除 API 的覆盖层），而第 2 节
   已经证明**根本不需要**这么做。这是本轮最有价值的一条结论。

---

## 4. 几何朝向（B3）

资产：`M1-4WD4S_model2601_dev.usda`，`metersPerUnit = 0.001`（毫米），元数据 `upAxis = "Z"`。
按几何最小跨度判定的竖直轴是 **Y**（上一轮工作流 B 的结论在本轮工具里复现）。

施加 **scale 0.001 + 绕 X 轴 +90°** 后：

| 项 | CAD（mm） | 世界（m，缩放+旋转后） |
|---|---|---|
| 包围盒 min | (−290.0, −346.996, −400.0) | (−0.29, −0.402852, −0.346996) |
| 包围盒 max | (290.0, 26.0, 402.852) | (0.29, 0.4, 0.026) |
| 包围盒尺寸 | (580.0, 372.996, 802.852) | **(0.58, 0.802852, 0.372996)** |
| 最小跨度轴（=竖直） | **Y** | **Z** ✓ |

四个候选轮状实体（STEP eid 435700 / 759491 / 773650 / 871375）旋转后中心：

    ( 0.250002,  0.297, -0.244)   ( 0.250002, -0.297, -0.244)
    (-0.250002,  0.297, -0.244)   (-0.250002, -0.297, -0.244)   单位 m

四者 z 完全相同（−0.244），最低点 z = **−0.346996**，正好等于模型世界包围盒的 min z——即"四个候选轮
落在地面平面、竖直方向为 +Z"。**+90° 是对的，−90° 会得到 −Z（倒扣）**，与 PR #10 的修正一致。
仍按候选对待：这是由 CAD 几何与最小跨度推断的装配约定，不是厂家声明的坐标系；且 +90° 之后 CAD 前进方向
落在世界 −Y（车头朝向仍需另定，见下一阶段建议）。

**一处必须提出的不一致**：这四个 eid 映射到的 USD 网格，其包围盒是 48.791 × **205.991 × 205.991** mm，
而 STEP 顶点包围盒是 48.653 × **180.782 × 157.783** mm（X 差 0.14 mm，Y/Z 差 25.2 mm）。两者不是同一外形，
说明这些"未命名轮状实体"的映射只是 CANDIDATE 级，或网格把邻近几何合并进来了。**在解决之前，任何用网格
包围盒反推轮径/轮宽的做法都不成立。**

---

## 5. 异常与限制（不掩盖）

1. **异常碰撞**：真实臂那一次运行里，代理底盘方块（0.6×0.4×0.1 m，中心 z=0.25）与臂基座附近几何重叠，
   depenetration 使整体在 1.5 s 内竖直漂移 **0.4789 m**（零重力下本应 z 不变）。art 本身没有 NaN、没有发散、
   安装误差仍是 0.0，但这属于"必须修的代理场景缺陷"：下次应把底盘方块移到不与臂重叠的位置。
2. **代理关节瞬态**：见 1.2，|q̇| 20 rad/s 属代理假象；真实臂在 300/20 增益下是 0.0188 rad/s。
3. **关节顺序**：PhysX/Isaac Lab 的关节索引顺序不是 URDF 顺序，下发改哪个关节必须按 `joint_names` 查。
4. **未做的**：没有做接触/摩擦标定、没有用真实底盘几何做碰撞、没有做多关节轨迹跟踪、没有动 PPO 与真机。
5. **不属于本轮证据的**：代理件的质量/惯量是我随手取的（12 kg / 2 kg / 1 kg / 0.5 kg），**不得**当作 Morph One
   的真实参数。

---

## 6. 工具与复现

| 文件 | 作用 |
|---|---|
| `scripts/simulation/morph_one_proxy_physx_probe.py` | 授权代理/真实臂挂载场景并在 PhysX 里真实步进，输出 JSON（变体：arm_rooted / whole_rooted / chassis_rooted / real_arm / real_arm_mobile_root） |
| `scripts/simulation/morph_one_orientation_probe.py` | 只读计算 0.001 缩放 + 绕 X 旋转后的世界包围盒、上轴、四个候选轮实体位置 |
| `tests/simulation/cad/test_orientation_probe.py` | 4 项回归（+90°→+Z / −90°→−Z / 上轴取几何而非元数据 / 缺映射不编造），已在带 pxr 的解释器里 **4 passed** |

命令（GPU 主机）：

    export PYTHONPATH=$(ls -d /home/T7/ojh/robot_sim/IsaacLab/source/* | tr '\n' ':')
    $PY morph_one_proxy_physx_probe.py --variant arm_rooted  --out out/arm_rooted.json
    $PY morph_one_proxy_physx_probe.py --variant arm_rooted  --gravity-off --out out/arm_rooted_zerog.json
    $PY morph_one_proxy_physx_probe.py --variant whole_rooted --gravity-off --out out/whole_rooted_zerog.json
    $PY morph_one_proxy_physx_probe.py --variant chassis_rooted --gravity-off --out out/chassis_rooted_zerog.json
    $PY morph_one_proxy_physx_probe.py --variant real_arm --out out/real_arm.json
    $PY morph_one_proxy_physx_probe.py --variant real_arm --gravity-off --gains verified --joint-step \
        --out out/real_arm_verified_gains.json
    $PY morph_one_orientation_probe.py --chassis-usd <usda> --mapping <mapping.json> --out out/orientation_plus90.json

证据：`outputs/simulation/rebot_b601_rs/morph_one/physx_validation/`（6 个运行 JSON、4 个授权 USDA、
4 份裁剪后的运行日志、朝向 JSON）。

---

## 7. 交给下一阶段的技术建议

1. **采用"保持资产 root API 原位 + 仅重定 root_joint body0"** 作为挂载方案（已由 PhysX 证明可行，且不需要
   任何资产改动）；不要走"整机再加一个 root API"的路（本轮已被引擎拒绝）。
2. 下一个 GPU 任务优先修代理场景几何（底盘方块不重叠），重跑一次"无异常碰撞"的基线，再做底盘加减速与转向
   时的安装误差与关节跟踪曲线。
3. 朝向：+90° 绕 X 已确认把几何立成 Z-up；还需决定世界坐标里"车头朝哪个轴"，并把该决定写进场景配置常量，
   避免每次 spawn 手工旋转。
4. 轮实体 bbox 不一致（205.991 vs 180.782 mm）必须在用任何轮径/轮宽前解决：要么确认这两个未命名实体与网格
   的映射是否正确，要么用 STEP 顶点包围盒作为唯一测量来源。
5. 在拿到真实质量/惯量/摩擦之前，不要把代理件的任何质量参数带进整机模型。
