# P5B — reset 姿态按机型取值（B601-RS 越限修复）

Status: **PASS。P5 记录的 F1（reset 硬编码 PiPER 姿态）已修复；PiPER 数值逐位不变；本机 52 passed / 1 skipped（pxr 缺席）。**

本轮只做被指派单元：把 `reset.py` 里那张"所有机型共用"的 rest pose 表换成按机型的、按**关节名**寻址的表，
并用纯 Python + pytest 钉住。没有启动 Isaac Sim、没有 torch.cuda、没有 GPU、没有写资产、没有改 `scripts/simulation/*`。

| 项 | 值 |
| --- | --- |
| 分支 | `p5/rs-reset-pose`，base `p5/arm-static-consistency` @ `ca30af0` |
| 资产（只读，未改） | `/home/T7/dgut/robot_sim/assets/robots/badminton_robot/rebot_b601_rs/usd/rebot_b601_rs.usd/ReBot_Arm_RS/ReBot_Arm_RS.usda` |
| 资产 sha256（前 = 后） | `96d4f2ec1126dec6eaf9e70ab664ba6cc30fa81bdec86178a3e50b7a23bdd510` |
| 限位事实来源（有 pxr 时的替代物） | `outputs/simulation/rebot_b601_rs/morph_one/p5_static/arm_joint_facts.json`（P5 的 pxr 导出，自述 sha256 前=后） |
| 本机解释器 | Python 3.11.9, pytest 9.0.2, 无 pxr（USD 未安装） |
| 生产代码 | `simulation/badminton_scene/rest_pose.py`（新）+ `simulation/badminton_scene/reset.py`（改） |

> 本机**没有 pxr**，所以本轮所有"RS 限位"结论都来自**已提交的 pxr 导出** `arm_joint_facts.json`，
> 不是来自实时资产；实时资产复查测试 `tests/simulation/arm_pose/test_rs_rest_pose_against_usda.py` 在本机 SKIP（原因写明），
> 有 pxr 的机器上会真跑。这一点在第 9 节 UNKNOWN 里再复述一次，不隐藏。

---

## 0. 结论速览

| 项 | 结论 | 依据 |
| --- | --- | --- |
| RS rest pose 8 个关节全部严格在 B601-RS 限位内 | **PASS**（最近者 joint2，余量 0.300 rad = 9.6% 行程） | 第 2 节表 |
| 余量规则"距任一限位 ≥5% 行程" | **PASS**（8/8 满足；最紧 gripper_joint1 余量 0.025 m = 50% 行程） | 第 2 节表 |
| PiPER rest pose 与原始常量逐位相同 | **PASS（是，bit-identical）** | 第 3.2 节 |
| 未知机型 / 缺关节名 / NaN / 调用方越限 一律抛异常 | **PASS**（4 类各有用例，且抛前不写 sim） | 第 4、7 节 |
| 反向控制：拿回原始 `reset.py` 跑同一套测试 | **FAIL（符合预期）**：arm_pose 退出码 **1**（11 failed / 41 passed / 1 skipped）；cad 金丝雀 退出码 **1**（1 failed / 13 skipped） | 第 3.3 节 |
| 原有 cad 金丝雀（P5 版断言"缺陷仍在"） | **已翻转为断言"已修复"，本机实测 PASS（退出码 0）** | 第 7 节 |

---

## 1. 缺陷（P5 F1，真实存在）

`reset.py` 原来把 PiPER Stage-0 的姿态当成所有机型的姿态：

```python
REST_JOINT = {"joint1": 0.0, "joint2": 1.5, "joint3": -1.5, "joint4": 0.0, "joint5": 0.0, "joint6": 0.0}
...
jp[:, k] = REST_JOINT.get(name, 0.0)     # 未知关节名静默补 0.0
```

而 B601-RS 的 `joint3` 限位是 `[0.0, 3.14] rad`（USD 里写作 `[0, 179.90875244140625] deg`），
`-1.5 rad` 在限位外。仓库内三个调用方（`scripts/t06_reset_local.py`、`t07_local.py`、`t08_local.py`）
都不传 `joint_pos`，因此 `--num_envs 1` 的 reset 路径**每次**都写了一个越限状态；
`.get(name, 0.0)` 还让"表里没有的关节"变成 0.0 悄悄下发（两个手指关节正好落在下限上）。

反向控制实测（原始 `reset.py`，`git show ca30af0:...`，用 numpy 版 torch 替身跑 `reset_idx`）：

```json
{ "written": {"joint1": 0.0, "joint2": 1.5, "joint3": -1.5, "joint4": 0.0, "joint5": 0.0,
              "joint6": 0.0, "gripper_joint1": 0.0, "gripper_joint2": 0.0},
  "outside":  {"joint3": [-1.5, 0.0, 3.1400000832579256]} }
```

即：越限的只有 `joint3`；两个手指被写成 0.0（= 下限，不越限但坐在硬限位上）。

---

## 2. 修复：按机型、按关节名的表

`simulation/badminton_scene/rest_pose.py` 提供 `REST_POSES`：一个机型一条 `RestPoseSpec`
（`joint_names` / `rest` / `limits` / `aliases`），值只按**名字**取。

余量规则（`required_margin`）：**距最近限位 ≥ 5% 行程**，在关节自己的单位里（revolute: rad，prismatic: m）。

```
margin(limit, v)        = min(v - lower, upper - v)
required_margin(limit)  = 0.05 * (upper - lower)
```

为什么是"相对 5%"而不是固定 0.05 rad：两个手指行程只有 0.05 m / 0.0715 m，5% 是 0.0025 m / 0.0036 m；
若按固定 0.05 rad(=0.05 m) 要求，手指的**半个行程都不够**，规则会自相矛盾。

实际数值与余量（`REST_POSES` 全表）：

| 机型 | 关节 | rest | 限位 | 单位 | 行程 | 余量 | 需要 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| piper | joint1 | 0.0 | [-2.618, 2.168] | rad | 4.7860 | 2.1680 | 0.2393 |
| piper | joint2 | 1.5 | [0.0, 3.14] | rad | 3.1400 | 1.5000 | 0.1570 |
| piper | joint3 | -1.5 | [-2.967, 0.0] | rad | 2.9670 | 1.4670 | 0.1484 |
| piper | joint4 | 0.0 | [-1.745, 1.745] | rad | 3.4900 | 1.7450 | 0.1745 |
| piper | joint5 | 0.0 | [-1.22, 1.22] | rad | 2.4400 | 1.2200 | 0.1220 |
| piper | joint6 | 0.0 | [-2.0944, 2.0944] | rad | 4.1888 | 2.0944 | 0.2094 |
| rebot_b601_rs | joint1 | 0.0 | [-2.8, 2.8] | rad | 5.6000 | 2.8000 | 0.2800 |
| rebot_b601_rs | joint2 | 0.3 | [0.0, 3.14] | rad | 3.1400 | **0.3000** | 0.1570 |
| rebot_b601_rs | joint3 | 1.0 | [0.0, 3.14] | rad | 3.1400 | 1.0000 | 0.1570 |
| rebot_b601_rs | joint4 | 0.0 | [-1.57, 1.57] | rad | 3.1400 | 1.5700 | 0.1570 |
| rebot_b601_rs | joint5 | 0.0 | [-1.57, 1.57] | rad | 3.1400 | 1.5700 | 0.1570 |
| rebot_b601_rs | joint6 | 0.0 | [-3.14, 3.14] | rad | 6.2800 | 3.1400 | 0.3140 |
| rebot_b601_rs | gripper_joint1 | 0.025 | [0.0, 0.05] | m | 0.0500 | 0.0250 | 0.0025 |
| rebot_b601_rs | gripper_joint2 | 0.03575 | [0.0, 0.0715] | m | 0.0715 | 0.0357 | 0.0036 |

最紧的是 RS `joint2`：余量 0.300 rad，占 3.14 rad 行程的 9.6%，仍 > 5%。
两个手指取**行程中点**（不是 0.0——0.0 恰好是两个手指的下限，坐在硬限位上的姿态不是 rest pose）。

PiPER 的限位来自 `simulation/piper_stage0/t02_articulation_checks.py:41-49` 记录的 AgileX URDF 值
（`joint1` 非对称，不做"对称化"），rest 值来自 `simulation/piper_stage0/piper_cfg.py` 的 Stage-0 姿态。

---

## 3. 反向控制（真跑，真退出码）

本机命令（在 `E:\具身智能\badmin_wt_p5b`）：

### 3.1 修复后（基线）

```powershell
python -m pytest tests/simulation/arm_pose -q
python -m pytest tests/simulation/cad/test_arm_static_consistency.py -q
```

```text
52 passed, 1 skipped in 1.18s        # 退出码 0
1 passed, 13 skipped in 0.20s        # 退出码 0   （pxr 缺席 → 13 个 stage 用例 SKIP；金丝雀 1 passed）
```

### 3.2 PiPER 逐位不变（同一套测试里的断言）

- `test_piper_rest_pose_is_bit_identical_to_the_stage0_values`：`struct.pack("<d", v)` 逐位比对；
- `test_reset_on_the_piper_arm_is_bit_identical_to_the_stage0_pose`：走 `reset_idx` 真代码路径后比对写入值。

两条都 **PASS**，即 **PiPER 不变 = 是**。

### 3.3 反向控制：把原始 `reset.py` 换回来

```powershell
git checkout ca30af0 -- simulation/badminton_scene/reset.py     # 原始 726d1ec
python -m pytest tests/simulation/arm_pose -q
python -m pytest tests/simulation/cad/test_arm_static_consistency.py -q
git checkout HEAD -- simulation/badminton_scene/reset.py
```

真实结果：

```text
11 failed, 41 passed, 1 skipped in 2.06s     # 退出码 1
1 failed, 13 skipped in 0.83s                # 退出码 1
```

失败里最关键的一条（就是本任务要的"RS rest pose 在限位内"）：

```text
FAILED test_reset_idx_contract.py::test_reset_on_the_rs_arm_writes_an_in_limit_model_aware_pose
FAILED test_reset_idx_contract.py::test_reset_on_the_rs_arm_does_not_write_the_piper_pose
FAILED test_reset_pose_canary.py::test_reset_py_no_longer_hardcodes_one_arm_agnostic_rest_pose
FAILED test_arm_static_consistency.py::test_reset_pose_is_model_aware_canary
  AssertionError: reset.py does not take its rest pose from rest_pose.py: one pose for every arm is
  the P5 F1 defect (joint3 = -1.5 rad is outside the B601-RS [0, 3.14])
```

即：**原始 `reset.py` 在同一套测试下退出码 1**，修复版退出码 0。

---

## 4. 失败即抛出（不留静默路径）

`reset_idx()` 成功路径返回 `None`，所以"返回错误码"会被调用方无声忽略——三个脚本会继续 step 一个没走 reset 的 sim。
因此所有检查都是**抛异常**，且都在**写 sim 之前**：

| 情形 | 异常 | 触发点 |
| --- | --- | --- |
| `robot.joint_names` 匹配不到任何机型（含多个机型同时匹配） | `UnknownRobotModel` | `resolve_model()` |
| `robot_key=` 显式指定了未知机型 | `UnknownRobotModel` | `resolve_model() → _spec()` |
| `robot_key=` 与运行时的名字集合矛盾 | `ResetPoseError` | `resolve_model()` |
| 机型表覆盖不了运行时 DOF 名（缺失 / 多余） | `UnknownJoint` | `rest_joint_pos()` / `validate_rest_pose()` |
| rest 值越限、贴限位、非有限数 | `RestPoseLimitViolation` | `validate_rest_pose()` |
| 调用方 `joint_pos` 越限 | `RestPoseLimitViolation` | `reset.py::_reject_out_of_limits` |
| 调用方 `joint_pos` 含 **NaN** | `RestPoseLimitViolation` | 同上（见下） |
| `robot.num_joints` ≠ `len(joint_names)` | `ResetPoseError` | `reset_idx()` |
| `joint_pos` 列数 ≠ 运行时 DOF 数 | `ResetPoseError` | `reset_idx()` |

**NaN 为什么要单独处理**：`NaN < lo` 与 `NaN > hi` **都是 False**，纯区间比较看不见 NaN，
它会一路写进 PhysX 并污染整个 articulation 状态。所以用反身判据 `jp != jp` 单独抓一次，
用例 `test_a_joint_pos_containing_nan_raises_before_writing` 钉住（抛之前 `joint_state_writes == []`）。

"越限"判定对**调用方显式目标**是**闭区间**（`lower <= v <= upper`）：显式命令恰好停在限位上是合法请求，
只有 rest pose 才要求"离限位 ≥5% 行程"。两者规则不同，是有意的。

---

## 5. 关节名 ↔ DOF 索引映射规则（重要陷阱）

**规则：一律按关节名索引，永不按位置。** `reset.py` 里写的是

```python
joint_names = tuple(str(name) for name in robot.joint_names)
model = resolve_model(joint_names, robot_key=robot_key)
target = rest_joint_pos(model, joint_names)          # {name: value}，键 = 运行时名字
for column, name in enumerate(joint_names):          # 列顺序 = robot.joint_names 的顺序
    jp[:, column] = target[name]
```

也就是说：**列顺序由运行时 `robot.joint_names` 决定，值由名字决定**。表的书写顺序不参与映射。

**USD stage 遍历顺序 ≠ DOF 顺序**（P5 F2，已用 `test_stage_traversal_order_is_not_the_dof_order` 钉住）。
实测（`arm_joint_facts.json`）：

```text
stage_traversal_order = [joint6, gripper_joint1, gripper_joint2, joint5, joint4, joint3, joint2, joint1, root_joint]
```

它既不是 URDF 顺序，也不是字母序。若有人按遍历位置取下标，`joint6` 会拿到 `joint1` 的值——
在 B601-RS 上这不是精度问题，是直接越限。测试 `test_rest_pose_only_depends_on_joint_names_not_on_their_order`
把"把名字列表整体倒序后，得到同一个 {name: value}"也钉住了。

另外 `root_joint` 是 **Fixed**，不是 DOF；所以"9 个关节"里只有 8 个有行程（P5 F4：URDF 的 9 个里
固定关节是 `j_gripper_end`，被 `--merge-joints` 折进 link6 了）。任何"按关节个数"推断 DOF 数量的代码都是错的。

---

## 6. 度 vs 弧度（角度关节）

| 位置 | 单位 | 说明 |
| --- | --- | --- |
| B601-RS USD（`physics:lowerLimit/upperLimit` on `PhysicsRevoluteJoint`） | **度** | UsdPhysics 约定；例：`joint3 [-0.0, 179.90875244140625] deg` |
| URDF `<limit lower upper>` | **弧度** | 例：`joint3 [0.0, 3.14] rad` |
| `rest_pose.py` 的表 | **弧度 / 米** | `JointLimit.unit` 随值一起走，防止单位被误读 |
| `reset_idx()` 写给 `write_joint_state_to_sim` 的值 | **弧度** | Isaac Lab 关节状态是弧度；表值直接下发，不做二次换算 |
| `drive:angular:physics:stiffness/damping` | **按度缩放** | 资产里是 `100/57.2958`、`1.0/57.2958`（P5 F3），与 rest pose 无关但同源，别把 100 读成"100 rad 增益" |

因此**唯一的换算点**在读取 USD 限位处：`math.radians()` 一次，之后全程弧度。
本仓库出现这个换算的地方：`tests/simulation/arm_pose/pose_data.py::read_usd_limits`、
`tests/simulation/cad/test_arm_static_consistency.py::_recorded_usd_limits`、
`tests/simulation/arm_pose/test_rs_rest_pose_against_usda.py::usd_limits`。
把度当弧度读是 **57.3 倍**误差：`joint3` 的上限会从 3.14 变成 179.9，等于把"可动关节"当成"几乎不动的关节"。

---

## 7. 怎么跑（可复查）

```powershell
# 新契约（纯 Python，无需 torch / pxr / GPU）
python -m pytest tests/simulation/arm_pose -q
# → 52 passed, 1 skipped in 1.18s    退出码 0

# 原有 cad 静态一致性 + P5B 金丝雀（pxr 缺席时 stage 用例 SKIP，金丝雀仍会跑）
python -m pytest tests/simulation/cad/test_arm_static_consistency.py -q
# → 1 passed, 13 skipped in 0.20s    退出码 0

# 有 pxr 的机器上（本轮未跑，见第 9 节）
P5_ARM_USDA=<...>/ReBot_Arm_RS.usda python -m pytest \
    tests/simulation/arm_pose/test_rs_rest_pose_against_usda.py \
    tests/simulation/cad/test_arm_static_consistency.py -q
```

用例清单（`tests/simulation/arm_pose/`，全部纯 Python：numpy 版 torch 替身 + 已提交的 pxr 导出）：

| 文件 | 钉住什么 |
| --- | --- |
| `test_model_aware_rest_pose.py` | 按名寻址、顺序无关、5% 余量、RS 8 关节在限位内、PiPER 逐位不变、机型解析/别名、缺关节/未知关节/NaN/贴限位统统抛 |
| `test_reset_idx_contract.py` | `reset_idx` 在 RS 上写入合法姿态、PiPER 逐位不变、spawn 偏移不变、未知机型/缺名/越限/NaN/列数错/num_joints 矛盾 抛且不写 sim |
| `test_reset_pose_canary.py` | 金丝雀（文本级）：`REST_JOINT` 表、`0.0` 静默兜底、PiPER 字面量都不许再出现 |
| `test_rs_rest_pose_against_usda.py` | 实时资产复查（需 pxr；本机 SKIP），并校验探测证据 JSON 与实时读数一致 |
| `conftest.py` / `fake_torch.py` / `pose_data.py` | 无 Isaac 的运行时替身；期望值用**字面量**写死，避免"用被测表证明被测表" |

---

## 8. 改动文件与 sha256

| 文件 | 状态 | sha256 |
| --- | --- | --- |
| `simulation/badminton_scene/rest_pose.py` | 新增 | `d180eada0046c77e43787886c131979a0d5835f713ec33d9c100d9b9cbce27b2` |
| `simulation/badminton_scene/reset.py` | 改 | `0496712943feb72662f700ff453cec30738ba6205cc3e8b5e19964787c86bf9c` |
| `tests/simulation/arm_pose/conftest.py` | 新增 | `72a35d44664b122a19b2072626e685b1d5afb033b330a427a4a20f887461c6a1` |
| `tests/simulation/arm_pose/fake_torch.py` | 新增 | `98230219b2f78c7e17903003b8a5b1c6312e3a9f30bfdcd43efbcccba5bc08a3` |
| `tests/simulation/arm_pose/pose_data.py` | 新增 | `028a83ce6c032a43f02930b7af6986b362991b95826fd79f193f1b2896af6c59` |
| `tests/simulation/arm_pose/test_model_aware_rest_pose.py` | 新增 | `dc3986ffbee6ef5424db7ac2b4fd7a7ef2f9b359ec90395beb63833d17a4c4d6` |
| `tests/simulation/arm_pose/test_reset_idx_contract.py` | 新增 | `64bd6d0249e9e2aabbad8f825ac5b6cc1757accbe2056144749be56d89c1f7e3` |
| `tests/simulation/arm_pose/test_reset_pose_canary.py` | 新增 | `a69e658abbc8054475f3566609c0dc17b06287bf3f7114369771ca74171241bf` |
| `tests/simulation/arm_pose/test_rs_rest_pose_against_usda.py` | 新增 | `7470da93ab184c0794de85281994f324254b09565e173485549127ef5e5f1a92` |
| `tests/simulation/cad/test_arm_static_consistency.py` | 改 | `f1f48466fd66e7b8f76e8444fb0e75188e90c2f37ba92f624f3f9116455ab9c3` |

`test_arm_static_consistency.py` 的两处改动（都不改它原本的结论）：
1. `pxr` 从 `pytest.importorskip`（模块级 SKIP，整个文件退出码 5）改成"可缺席"：`stage` fixture 缺席时才 SKIP。
   这样金丝雀在没有 USD 的机器上**也能跑、也能红**——否则这条修复只会在拥有资产的那台机器上被验证。
2. P5 版金丝雀（断言"缺陷仍在"，见 `P5_ARM_STATIC_CONSISTENCY.md` 第 5.1 节）翻转为断言"已修复"；
   限位来源改为已提交的 pxr 导出（`joint_limits.py` 在 `origin/feature/rebot-rs-sim-asset`，不在本分支）。

---

## 9. UNKNOWN / 未做

- **U1 实时资产未在本机复读**：本机无 pxr。RS 限位结论来自已提交导出 `arm_joint_facts.json`
  （自述 `usda_sha256_before == after == 96d4f2ec...`）。实时复查测试已写好并本机 SKIP；
  "在主机上跑通的退出码"本轮**没有**，不冒充 PASS。
- **U2 Isaac Lab 运行时未验证**：本轮没有启动 Isaac Sim（主 agent 拥有 GPU），
  `reset_idx` 的验证走的是 numpy 版 torch 替身 + fake robot/shuttle。真机/真 sim 上的
  `robot.joint_names` 顺序、`num_joints` 一致性**未实测**；代码对二者都做了显式检查（矛盾就抛）。
- **U3 限位是模型值不是实测值**：两个机型的限位都来自 URDF/USD 官方值，
  `usable_for_capability=False` 的告诫没有被削弱。
- **U4 rest pose 的"最好"取值未做优化**：RS 用正式场景已用的 `joint2=0.3, joint3=1.0`，
  手指取行程中点。是否更利于击球起始姿态，需要后续规划层评估，本轮只保证**合法且可重复**。
- **U5 `scripts/simulation/*` 未改**（不在本 agent 文件范围）。
