# P3-01 底盘 ↔ 机械臂基座 6DoF 帧树冻结（离线半）

**任务**：P3-01「冻结底盘 chassis/arm base 6DoF frame tree、Joint anchor、相对姿态误差与非平凡 yaw/roll 参考」。
**性质**：本文件只交付 P3 的**离线/静态**一半（真实资产的帧树与锚点 + 参考夹具 + 测试）。
P3 的动态部分（P3-02…P3-08，真实运动中测量）**未做、未宣称**；门禁文件
`management/morph_one_rs/PHASE_P3_DYNAMIC_MOUNT_GATE.md`（分支 `dev/p2-wheel-contact-g1g4`，L25）**未被修改**。

**分支**：`p3/mount-frame-tree`

**产物**：

- `outputs/simulation/rebot_b601_rs/morph_one/p3_mount/mount_frame_tree.json`（帧树 + 锚点契约）
- `outputs/simulation/rebot_b601_rs/morph_one/p3_mount/mount_fixtures.json`（3 个参考夹具的期望位姿）
- `tools/p3_mount/se3.py`、`tools/p3_mount/extract_mount_frame_tree.py`、`tools/p3_mount/gen_mount_fixtures.py`
- `tests/simulation/p3_mount_frame_test.py`

---

## 1. 证据来源：真实只读资产，不是猜测

真实资产（GPU 主机只读；本次**未写入任何文件、未启动 Isaac Sim、未使用 GPU**）：

```
/home/T7/dgut/robot_sim/assets/robots/badminton_robot/rebot_b601_rs/usd/rebot_b601_rs.usd/ReBot_Arm_RS
```

用 `ssh -o BatchMode=yes 172.31.68.251 "cat <layer>"` 把 ASCII 层抓到**被 .gitignore 忽略的 `_scratch_p3a/`**
（`.gitignore:83 _scratch_*/`），并逐层比对远端 `sha256sum`：本地与远端**逐字节相同**，
所以「解析本地副本」就是「解析真实资产」。根层 sha256 与任务给定的
`96d4f2ec1126dec6eaf9e70ab664ba6cc30fa81bdec86178a3e50b7a23bdd510` 完全一致。

| 层 | 作用 | 字节 | sha256 | 远端 sha256sum 一致 |
|---|---|---|---|---|
| `ReBot_Arm_RS.usda` | 根层：defaultPrim、单位、upAxis、Physics variant 选择 | 1251 | `96d4f2ec1126…bdd510` | 是 |
| `payloads/base.usda` | prim 层级与 authored xformOp（base_link…gripper_end） | 28045 | `032ef9d0953b…0cf557ef` | 是 |
| `payloads/robot.usda` | 仅 Isaac*API override，无变换 | 3538 | `757b36cbb7dc…f6acb630c` | 是 |
| `payloads/Physics/physx.usda` | 生效 variant 的 payload，仅 PhysxJointAPI | 2127 | `e51eae9dbb0d…1dc6bda` | 是 |
| `payloads/Physics/physics.usda` | 关节 authoring（含 `root_joint` 锚点） | 15832 | `8d18a8c9271e…24fce02b6` | 是 |

单位自证：`metersPerUnit = 1`、`kilogramsPerUnit = 1`、`upAxis = "Z"` → `units_verdict = PASS`。

**关键 authoring 事实（容易搞错）**：根层用 `variants = { string Physics = "physx" }` 选中 `physx.usda`，
而 `physx.usda` 通过 `subLayers = [@./physics.usda@]` **再引用** `physics.usda`。
因此 `root_joint` 的 authoring 位于 `physics.usda`，它是**生效层**，不是死层。

## 2. 帧树（名称全部来自真实 USD）

```
world
 └─(自由 6DoF 根位姿：P3 诊断驱动 write_root_pose_to_sim)
    chassis            /RW/Chassis            代理底盘刚体，20 kg，非 articulation root
     └─(root_joint, PhysicsFixedJoint)        ← 本任务的锚点
        arm_base        /ReBot_Arm_RS         资产 defaultPrim，retarget 后是 root_joint 的 body0
         └─(刚性，恒等)
            base_link   /ReBot_Arm_RS/Geometry/base_link
             └─ link1 ─ link2 ─ link3 ─ link4 ─ link5 ─ link6    （6 个转动关节的链）
                └─ gripper_end  /ReBot_Arm_RS/Geometry/base_link/link1/link2/link3/link4/link5/link6/gripper_end
```

- `chassis` 的 20 kg 与「不是 articulation root」来自 P2 证据脚本
  `scripts/simulation/p2_wc_g7b_arm_with_wheels.py`（分支 `dev/p2-wheel-contact-g1g4`，`CHASSIS_MASS_KG = 20.0`；
  该脚本把 `root_joint` 的 `physics:body0` 改指 `/RW/Chassis`，**不改** localPos/localRot）。
- `arm_base` 在资产里就叫 `/ReBot_Arm_RS`（defaultPrim）；资产中**不存在**名为 `arm_base` 的 prim，
  本契约的 `arm_base` 就是该 prim 的帧，不做重命名。

## 3. root_joint 锚点：真实 authoring 数值（含行号与 sha256）

来源层 `payloads/Physics/physics.usda`，sha256 `8d18a8c9271e0905df72530d177157bbdebeb2fceaae3de7628524240fce02b6`；
`def PhysicsFixedJoint "root_joint"` 在 **L308**，prim 路径 `/ReBot_Arm_RS/Physics/root_joint`。

| 字段 | 值 | 单位 | 行号 |
|---|---|---|---|
| `physics:body0` | `/ReBot_Arm_RS` | prim path | L311 |
| `physics:body1` | `/ReBot_Arm_RS/Geometry/base_link` | prim path | L313 |
| `physics:localPos0` | `(0, 0, 0)` | m | L314 |
| `physics:localPos1` | `(0, 0, 0)` | m | L315 |
| `physics:localRot0` | `(1, 0, 0, 0)`（wxyz） | 无量纲 | L316 |
| `physics:localRot1` | `(1, 0, 0, 0)`（wxyz） | 无量纲 | L317 |

**chassis ↔ arm_base 相对变换（推导，不是抄写）**：

```
T_chassis_arm_base = T(localPos0, localRot0) · T(localPos1, localRot1)^-1 = I · I^-1 = I
```

- 位置 `[0, 0, 0]` m，`position_norm_m = 0.0`；姿态 `[1, 0, 0, 0]` (wxyz)，`rotation_angle_rad = 0.0`。

这**不是**「算出来刚好是恒等」，而是资产把两个 local pose 都 author 成恒等：底盘体帧与机械臂基座帧**严格重合**，
retarget 只改 body0 的指向，不动锚点。直接后果（必须写清）：**机械臂装在底盘原点，而不是某个 CAD 安装点**；
真实安装偏置在资产里**没有任何 authoring**，只能由场景侧 body0 位姿提供（见 §6）。

## 4. 容差：来自预注册门禁，不是本任务自定

| 量 | 值 | 出处 |
|---|---|---|
| 位置 | `0.001 m` | `management/morph_one_rs/PHASE_P3_DYNAMIC_MOUNT_GATE.md` **L25**（门禁行 `mount_stable`，分支 `dev/p2-wheel-contact-g1g4`） |
| 姿态 | `0.001 rad` | 同上 L25 |

原文（L25）：`| mount_stable | 相对安装位姿相对 step0 的最大偏差 ≤ 0.001 m 且 ≤ 0.001 rad | 与 P2 零重力静置同量级；运动中不放松 |`

契约 JSON 以 `tolerances.{position_m, rotation_rad, source{file,line,branch,quote}}` 冻结，测试断言这两个数字**不得被改动**。
本任务**没有**修改门禁文件。

## 5. 参考夹具：三个，其中两个非平凡

夹具只施加「world → chassis」的根位姿（正是 P3 诊断驱动的量），其余帧由冻结契约推导。旋转约定 `R = Rz(yaw) · Rx(roll)`，四元数 wxyz。

| 夹具 | chassis 位置 (m) | yaw / roll (rad) | chassis 转角 (rad) | 非平凡 |
|---|---|---|---|---|
| `F1_identity` | (0, 0, 0) | 0 / 0 | 0.0 | 否（对照） |
| `F2_yaw_035` | (0.30, 0, 0) | 0.35 / 0 | **0.35** | **是** |
| `F3_yaw_030_roll_010` | (−0.15, 0.20, 0.10) | 0.30 / 0.10 | **0.316108960343** | **是** |

期望位姿（节选；全精度见 `mount_fixtures.json`）：

| 夹具 | 帧 | 位置 (m) | 姿态 wxyz |
|---|---|---|---|
| F1 | chassis / base_link | (0, 0, 0) | (1, 0, 0, 0) |
| F1 | gripper_end | (0.301710169999, 2.62633e-07, 0.217746720407) | (1, −6.9738e-08, −1.7116e-08, 0) |
| F2 | chassis / base_link | (0.30, 0, 0) | (0.984726538905, 0, 0, 0.174108137594) |
| F2 | gripper_end | (0.583418210829, 0.10345600249, 0.217746720407) | (0.984726538905, −6.5693e-08, −2.8996e-08, 0.174108137593) |
| F3 | chassis / base_link | (−0.15, 0.20, 0.10) | (0.98753537156, 0.049417957074, 0.007468793718, 0.149251373721) |
| F3 | gripper_end | (0.144658793498, 0.2683942156, 0.31665892) | (0.987535375134, 0.04941789076, 0.007468766408, 0.149251373396) |

- `base_link` 在**每个**夹具里都与 `chassis` 完全重合（差 0）：这是恒等安装的直接可检验后果。
- **期望安装相对位姿在三个夹具下不变**：`position_norm_m = 0.0`、`rotation_angle_rad = 0.0`，跨夹具最大偏差 `0.0 m / 0.0 rad`
  （远小于门禁 0.001 / 0.001）。只转恒等位姿的「证明」没有意义，故 F2/F3 带真实非零 yaw/roll 与真实平移。

臂链静止位姿（`payloads/base.usda` authored xformOp 逐跳合成，pxr 复核）：`gripper_end` 在 arm_base 帧下
= 位置 `(0.301710169999, 2.62633e-07, 0.217746720407)`、姿态 `(1, −6.9738e-08, −1.7116e-08, 0)`；
逐跳乘积与 pxr `GetLocalToWorldTransform` 之差为 `0.0 m / 0.0 rad`（`cross_checks[1]`）。逐跳表见契约
`frame_tree.edges` 与 `arm_chain_rest_pose.hops`，每跳含 `{position, quat_wxyz, det_R, source_file, locator}`。

## 6. UNKNOWN（不猜，写清缺哪个 datum）

1. **真实 CAD 安装点**（机械臂在 M1-4WD4S 底盘上的安装偏置）：UNKNOWN。缺：机械臂 USD 不引用任何底盘资产；
   本仓库没有 M1-4WD4S CAD 的机器可读安装帧。故场景只能用恒等安装。
2. **底盘↔臂基座连接件的刚度/预载/力矩**：UNKNOWN。缺：资产内无 FEA / 螺栓预载数据；P3 只能测运动学刚性。
3. **真实部署中 body0 是「底盘体原点」还是「安装支架 prim」**：UNKNOWN。缺：无整机装配 USD；
   若是支架 prim，相对变换会整体平移，0.001 m 判据必须在那个装配上重新冻结。

## 7. 测试覆盖（纯 python + numpy；pxr 可选）

`tests/simulation/p3_mount_frame_test.py`，10 个用例：

1. 契约/夹具文件存在且 schema 正确；
2. 资产溯源是真实资产（根层 sha256 = `96d4f2ec…bdd510`、单位 PASS、`isaac_sim_used=false`、`gpu_used=false`、每层都有 sha256）；
3. **帧链名称与顺序**（world/chassis/arm_base/base_link/link1..6/gripper_end）与锚点 prim/类型/body0/body1；
4. 锚点数值就是 authoring 的恒等，且 `unknown` 里确实记了「CAD 安装点」；
5. **det(R) = +1**：所有臂链跳、安装旋转、三个夹具 × 四个帧的旋转，误差 < 1e-9；
6. 逐跳乘积 == pxr 记录的静止位姿（< 1e-9），夹具侧自检为 0；
7. **夹具非平凡性**：F2/F3 转角 > 0.1 rad 且分别 ≥ 0.35 / 0.30 rad、平移非零；F1 恰为 0（对照）；
8. **安装相对位姿在三个夹具下不变**（跨夹具最大偏差 < 1e-9，且各自 ≤ 门禁 0.001/0.001），并断言容差数字与门禁 L25 未被改；
9. 世界位姿与夹具命令一致（chassis 位置/旋转，base_link 与 chassis 重合）；
10. pxr 交叉复核（**可选**）：本机有 pxr 0.26.11 且设了 `P3_MOUNT_ASSET_DIR` 时才重新打开资产重算 `gripper_end`；
    否则测试**明确打印「只校验已提交证据、未重新打开资产」**并跳过，不会假装复核过。

## 8. 真实命令与真实退出码

在 `E:\具身智能\badmin_wt_p3a`（worktree，分支 `p3/mount-frame-tree`）实跑：

```powershell
# 0) 建 worktree（在 E:\具身智能\badmin_project 执行）
git worktree add "E:\具身智能\badmin_wt_p3a" -b p3/mount-frame-tree origin/main   # exit 0

# 1) 只读取证：ssh cat 层到 gitignored scratch，并核对 sha256
ssh -o BatchMode=yes 172.31.68.251 "sha256sum <REMOTE>/ReBot_Arm_RS.usda"           # exit 0，96d4f2ec…bdd510
ssh -o BatchMode=yes 172.31.68.251 "cat <REMOTE>/ReBot_Arm_RS.usda" > _scratch_p3a/ReBot_Arm_RS.usda
Get-FileHash -Algorithm SHA256 ...                                                   # 本地哈希 = 远端哈希

# 2) 抽取帧树契约（pxr 可选）
python tools/p3_mount/extract_mount_frame_tree.py --asset-dir _scratch_p3a `
    --out outputs/simulation/rebot_b601_rs/morph_one/p3_mount/mount_frame_tree.json   # exit 0

# 3) 生成参考夹具
python tools/p3_mount/gen_mount_fixtures.py `
    --contract outputs/simulation/rebot_b601_rs/morph_one/p3_mount/mount_frame_tree.json `
    --out outputs/simulation/rebot_b601_rs/morph_one/p3_mount/mount_fixtures.json      # exit 0

# 4) 测试
python -m pytest tests/simulation/p3_mount_frame_test.py -q                           # exit 0，10 passed
python tests/simulation/p3_mount_frame_test.py                                        # exit 0，10 passed, 0 failed
$env:P3_MOUNT_ASSET_DIR="E:\p3a_scratch"; python -m pytest tests/simulation/p3_mount_frame_test.py -q   # exit 0，10 passed（pxr 复核生效）
```

实际输出要点：

```text
units: PASS upAxis=Z metersPerUnit=1.0 variant=physx
root_joint body0=/ReBot_Arm_RS body1=/ReBot_Arm_RS/Geometry/base_link
chassis->arm_base pos=[0.0, 0.0, 0.0] m norm=0.000000000000 rot=0.000000000000 rad
pxr available: True (pxr 0.26.11)
chain self-check: dpos=0.000e+00 m drot=0.000e+00 rad
F1_identity          chassis_rot=0.000000 rad  mount_relative pos=0.000000 m rot=0.000000 rad
F2_yaw_035           chassis_rot=0.350000 rad  mount_relative pos=0.000000 m rot=0.000000 rad
F3_yaw_030_roll_010  chassis_rot=0.316109 rad  mount_relative pos=0.000000 m rot=0.000000 rad
10 passed in 1.80s
```

## 9. 边界（不要过度解读）

- 本文件**只**冻结离线帧树/锚点/夹具。`mount_stable` 门禁本身需要 P3-02 的真实运动遥测才能判 PASS；
  这里能说的只有：「给定真实资产的 authoring，安装相对位姿在三个非平凡夹具下数学上不变，且容差与出处已冻结」。
- `F1_identity` 转角为 0，是**对照**而非「非平凡参考」；P3-01 要求的非平凡 yaw/roll 由 F2/F3 承担。
- 臂链静止位姿取自 USD 的 authored xformOp（关节值不参与帧树），故 `gripper_end` 世界位姿是**静态参考**，不是运行时 FK 结果。
- 未跑任何 Isaac Sim / GPU 任务，未在 172.31.68.251 上写任何文件。
