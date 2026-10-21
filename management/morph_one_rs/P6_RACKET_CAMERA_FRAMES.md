# P6-01 / P6-02 / P6-04 / P6-05：球拍资产、相机元数据与 frame tree 合同

本文件只写能自证的结论。凡是仓库里没有来源的数字，一律标 UNKNOWN，并写清缺的是哪个量。

- 分支：`p6/racket-camera-frames`，基线 `origin/main` = `16f1ae4`
- 全部资产**只读**：`assets/**` 未修改一个字节（用 sha256 对照证明，见下）
- CPU-only：没有 Isaac Sim、没有 GPU、没有渲染；真机 USD 只在 GPU 主机上做**只读** `ssh cat` / `sha256sum`

---

## 0. 环境与一次真实的环境改动

| 项 | 值 |
|---|---|
| Python | 3.11.9 |
| pytest | 9.0.2 |
| pxr（usd-core） | **初始不存在**（`python -c "from pxr import Usd"` → `ModuleNotFoundError: No module named 'pxr'`，exit 1） |

因此本地执行了一次 `python -m pip install --quiet usd-core`（exit 0），之后 `from pxr import Usd` 可用。
这一改动**只影响本机解释器**，不进仓库。若别的机器没有 pxr：
`tools/p6_frames/build_p6_frame_contract.py` 与 `tests/simulation/p6_*_test.py` **不需要 pxr**就能跑，
因为它们读的是已提交的 evidence JSON + 文件 sha256；只有 `audit_racket_usd.py` 的 USD 部分需要 pxr
（它已经把结论落成 `evidence/racket_usd_audit.json`，缺 pxr 时会写 `status=UNKNOWN` 而不是编造数值）。

---

## 1. P6-01 球拍资产审计

命令与真实退出码：

```text
python tools/p6_frames/audit_racket_usd.py                       # exit 0
wrote outputs/.../p6_frames/evidence/racket_usd_audit.json
usd PASS metersPerUnit 1.0 upAxis Z collision 0 sha_ok True glb_sha_ok True
```

| 项 | 结论 | 证据 |
|---|---|---|
| 单位 `metersPerUnit` | **PASS** = 1.0 | `UsdGeom.GetStageMetersPerUnit`，USDC 二进制 crate，头部 `PXR-USDC` |
| `upAxis` | **PASS** = `Z` | 同上 |
| 包围盒（米） | **PASS** | 拍框 `x ±0.01584, y ±0.10797, z 0.0000–0.6827`；线床 `x ±0.0010, y ±0.10073, z 0.44391–0.67526`；整体长 **0.6827 m**、头宽 **0.21594 m** |
| 是否有碰撞体 | **PASS（确实没有）** | 全 stage `CollisionAPI`/`Physics*` prim 数 = **0**；两个 Mesh 都没有 CollisionAPI。此资产**不可碰撞**，P6-07 必须自建代理 |
| 法兰 / 末端 link 在哪 | **UNKNOWN（资产里不存在）** | 资产只有 `/RacketVisual` + 两个 Mesh，无 joint、无 mount、无法兰几何。原点在**手柄末端**（z=0 就是最低点），**不是**法兰贴合面 |
| sha256 | **PASS** | `643fb67003c6429f98ec132d2db056c83ed02811b594ad27904b618bf7c1ec29`，与 `assets/THIRD_PARTY_ASSETS_RACKET.md:17` 记录一致 |
| GLB 自身单位/缩放 | **PASS（单位由规范固定）/ 缩放 REVIEW** | glTF 2.0 规范：长度单位=米、右手、+Y up，文件**没有**单位字段；实测 17 个 node **全部没有** scale/rotation/translation，即没有藏起来的 Sketchfab 缩放。但 — 见下一行 |
| GLB/USD 的**尺寸是否等于真实球拍** | **UNKNOWN** | 视觉 682.7 mm vs 产品 675 mm、BWF 上限 680 mm；`configs/racket.yaml:measured_unit.status = REQUIRES_MEASUREMENT`，至今未缩放。任何从视觉网格量出来的尺寸都不是真球拍的量 |

派生事实（供下游引用，别再重新猜）：
- 拍面坐标系：`+Z 手柄→头`、`+X 面法向朝对方`、`+Y = Z×X`（右手，行列式 +1，测试里实测）。
- 视觉资产 868 tri（`Obj_Racket_0` 1632 pts + `Obj_Strings_0` 44 pts），材质 `White` / `Strings`。

---

## 2. P6-02 frame tree（机器可读）

产物：`outputs/simulation/rebot_b601_rs/morph_one/p6_frames/frame_tree.json`

真机臂 USD 的取证（**只读**，无 Isaac Sim、无 GPU）：

```text
ssh dgut@172.31.68.251 "sha256sum .../ReBot_Arm_RS/ReBot_Arm_RS.usda"
96d4f2ec1126dec6eaf9e70ab664ba6cc30fa81bdec86178a3e50b7a23bdd510   # 与任务给定值一致 → 同名同体

ssh ... "cat .../payloads/Physics/physics.usda"  > _scratch_p6/physics.usda.local
ssh ... "cat .../payloads/base.usda"             > _scratch_p6/base.usda.local
ssh ... "cat .../ReBot_Arm_RS.usda"              > _scratch_p6/ReBot_Arm_RS.usda.local
# 本地副本 sha256 与远端 sha256sum 逐字节一致：8d18a8c9…(physics) / 032ef9d0…(base) / 96d4f2ec…(root)
```

> 为什么要下载再解析：USDA 是**文本**，本地解析不需要 pxr，也不需要在那台机器上装任何东西、写任何文件。
> 本地副本 sha256 == 远端 sha256，所以"解析副本"就是"解析资产"。

命令与真实退出码：

```text
python tools/p6_frames/extract_robot_usd_frames.py --layer ... --out .../evidence/robot_usd_joint_chain.json  # exit 0
wrote .../robot_usd_joint_chain.json joints: 9 units: PASS
python tools/p6_frames/build_p6_frame_contract.py                                                             # exit 0
  edges 15 asserted 10 unknown 5
```

### 2.1 臂链：全部**由资产断言**（名字按名字映射，不靠 stage 遍历顺序）

`ReBot_Arm_RS`（defaultPrim）→ `base_link` → `link1` … → `link6`，每个 body 的 prim 路径都写在合同里。

| 边 | joint | 平移 (m，`physics:localPos0`) | 旋转 `wxyz` | 限位（USD 原始=度 / 换算=rad） |
|---|---|---|---|---|
| arm_root_prim → base_link | root_joint（fixed，identity） | (0,0,0) | (1,0,0,0) | — |
| base_link → link1 | joint1 | (-0.00034283, -0.00098683, 0.075) | (0,-1,0,0) | ±160.42818° / ±2.8000 rad |
| link1 → link2 | joint2 | (0.020343, 0.027237, 0.07) | (-0.70710546, 0.7071081, 0, 0) | 0…179.90875° / 0…3.1400 rad |
| link2 → link3 | joint3 | (-0.236, ~0, 0) | (0,-1,0,0) | 0…179.90875° / 0…3.1400 rad |
| link3 → link4 | joint4 | (0.228, -0.072746, 0.0045) | (0,-1,0,0) | ±89.95438° / ±1.5700 rad |
| link4 → link5 | joint5 | (0.087, -0.048, -0.03075) | (-0.7071081, -0.70710546, 0, 0) | ±89.95438° / ±1.5700 rad |
| link5 → link6 | joint6 | (0.0365, ~0, 0.048) | (-3.31e-17, -0.70710546, -3.71e-17, 0.7071081) | ±179.90875° / ±3.1400 rad |
| link6 → gripper_left | gripper_joint1（prismatic，**禁止当 mount**） | (1.5351144e-07, 0.000073385, 0.124271) | (0.7071081, -0.70710546, -2.597e-06, 2.597e-06) | 0…**0.05 m** |
| link6 → gripper_right | gripper_joint2（prismatic，**禁止当 mount**） | (1.5458967e-07, -0.000073385, 0.124271) | (0.7071081, 0.70710546, 2.597e-06, 2.597e-06) | 0…**0.0715 m** |
| link6 → gripper_end | 无 joint（**不是刚体**） | (0, 0, 0.16621) | (2.597e-06, -0.70710546, -2.597e-06, -0.70710808) | — |

来源：`payloads/Physics/physics.usda`（sha256 `8d18a8c9…`）+ `payloads/base.usda`（sha256 `032ef9d0…`），
localPos/localRot 是 joint 在**零位**时的子帧位姿。

三个必须写下来的发现：
1. **USD 的转限位是"度"**。`physics:upperLimit = 160.42818` 是 2.8 rad，不是 160 rad。合同里两个值都存。
2. **gripper_end 不是刚体**：它是 `instanceable` 的 Xform 子节点，没有 rigid body、没有 joint。可以当**几何帧**用，但当动力学 body 用会静默生成一个 static child；动力学 body 只能是 `link6`。
3. **两侧夹爪行程不对称**（0.05 m vs 0.0715 m，与 B601-DM 资产两个 0.0715 不同）。这直接支持 `src/simulation/rebot_b601dm/racket.py` 里 `LOCKED_JOINTS` 的判断：夹在滑动爪上的球拍不是固定工具。

### 2.2 跨安装面的边：全部 **UNKNOWN**（合同里数值为 null）

| 边 | 状态 | 缺的量 | 填它需要什么 |
|---|---|---|---|
| world → chassis_morph_one | UNKNOWN | 底盘在场地的实际位姿 (x,y,yaw) | 场地标定/里程计 + 场地注册；`scene_layout.py` 的 `world_pos=(-1.60,0,0)` 是场景占位值 |
| chassis_morph_one → arm_base_link | UNKNOWN | `T_chassis_armbase`：B601-RS 底座的螺栓孔偏移与偏航 | 该底盘+该臂的 CAD 或实测安装记录；**不能**复用 `scene_layout.py` 的 PiPER `mount_rel=(0,0,0.30)` |
| gripper_end → racket_mount | UNKNOWN | `T_flange_tcp`（手柄轴从法兰哪里出去、绕轴怎么定相） | `configs/racket.yaml piper_attachment.mount_transform`（现在 translation/quaternion 都是 null，status=REQUIRES_MEASUREMENT）+ `docs/RACKET.md:117` 的测量流程 |
| racket_mount → racket_root | UNKNOWN | 手柄末端插进 mount 多深 + 绕手柄轴的定相 | 实物测量。资产答不了：`/RacketVisual` 原点=手柄末端，是工具上的一个点，不是贴合面 |
| racket_root → racket_contact_frame | UNKNOWN | 甜区中心在 racket_root 下的位置 | 实测；`racket_cfg.py head_center_from_tcp=0.50` 自标 "approx"、`measured_unit` 仍是 REQUIRES_MEASUREMENT |

> 线床包围盒（z 0.44391–0.67526 m）**能**算出一个 0.56 m 的中心，但视觉比真拍长 7.7 mm，所以那个中心不是真拍的中心，合同**不**把它当成断言值。这是"能算"和"能断言"的区别。

**合同统计**：15 条边，10 条由资产断言（含 sha256 + 定位串），5 条 UNKNOWN（都带 missing_datum），0 条靠猜。

---

## 3. P6-04 D455 相机元数据

产物：`outputs/simulation/rebot_b601_rs/morph_one/p6_frames/camera_metadata.json`

| 字段 | 结论 | 依据 |
|---|---|---|
| 机型 Intel RealSense D455 | **UNKNOWN**（未证） | 型号只出现在任务文本和 `configs/tiny_recovery_yolo26s_v1.yaml:40`（"D455 hard negatives"，数据集范围）；仓库里没有序列号/固件/rs-enumerate 记录 |
| 内参 fx, fy, cx, cy | **UNKNOWN** | `management/shuttle_detection/plan5_input_requirements.md:57` 原文：**"The repository holds no calibrated fx"** |
| 分辨率 | **UNKNOWN** | 没提交任何 stream profile |
| RGB-depth 对齐 | **UNKNOWN** | 不知道 pipeline 是否 `align(rs.stream.color)`；对齐关掉时 RGB 像素 (u,v) 与 depth 像素 (u,v) 是两条不同射线 |
| 帧率 | **UNKNOWN** | 无配置、无时间戳日志（P6-06 负责） |
| 深度单位 / depth scale | **UNKNOWN** | 这是**每台设备**的量，须 `depth_sensor.get_depth_scale()` 读取；硬编码 0.001 会在设备报 0.0001 时把球位置错 10 倍 |
| 畸变模型+系数 | **UNKNOWN** | 同上；出厂标定是最低要求，棋盘格重标是诚实上限 |
| 深度→彩色外参 | **UNKNOWN** | `get_extrinsics_to(color_profile)` 未记录 |
| mount→camera 外参 | **UNKNOWN** | 需要手眼标定或实测 CAD 安装；FOV 不能代替位姿 |

被**明确拒绝**的替代物（合同里逐条记录，防止以后被误用）：

| 来源 | 值 | 为什么不能用 |
|---|---|---|
| `simulation/badminton_scene/scene_layout.py:87` | `rig_world=(-1.40,0,1.20), pitch=-4°, baseline=0.29, ±0.145` | 文件自己写着 "ENGINEERING_V0_1 camera rig placeholder (no sensor/render)"，且无内参/无 D455 身份；0.29 m 是 rig 几何占位，不是实测基线 |
| `simulation/badminton_scene/camera_rig_cfg.py:1` | mast/50 mm cube | 从上面的占位值推出来的盒子，没有任何相机模型 |
| `docs/ALGORITHM_AUDIT.md:56` | `fx=700 px`（"代理标定"） | 渲染/合成数据用的工程代理标定，不是任何一台 D455 的测量 |
| `outputs/shuttle_detection/capability/CONTROLLED_MATRIX_RUN1.md:20` | `fx=11200 px` | 长焦渲染相机，用来画尺寸阶梯，与 D455 物理无关 |

**已经证成的部分（这才是 P6-04 的产出）**：光心坐标系与机器人 mount 坐标系的**轴向约定**是确定的、可测的：
`+Z 出镜头 / +X 图像右 / +Y 图像下`（右手，det=+1），与 REP-103 body 帧 `+X 前 / +Y 左 / +Z 上` 相差

```text
R_mount→optical = [[0,-1,0],
                   [0, 0,-1],
                   [1, 0,0]]      quat wxyz = (0.5, 0.5, -0.5, 0.5)
mount +X → optical +Z ;  mount +Y → optical -X ;  mount +Z → optical -Y
depth 点 (X,Y,Z)_optical 在 mount 帧是 (Z, -X, -Y)
```

这是**约定，不是标定**：轴向由两套约定固定，相机在机器人上的**位姿**仍然是 UNKNOWN（合同里 state 分开写）。

填表流程（写进合同 `fill_procedure`）：
1. `rs-enumerate-devices -c > d455_<serial>_<fw>_enumerate.txt`
2. pyrealsense2 取选定 profile 的 colour 内参+畸变、depth 内参、`get_depth_scale()`、`get_extrinsics_to`
3. 写 `configs/camera/d455_<serial>_<mode>.json`（含 profile 名与固件）并提交、算 sha256
4. 让 `build_p6_frame_contract.py` 指向该文件，这些字段就不再是 UNKNOWN
5. `T_mount_camera` 单独做手眼标定、单独提交（那是机器人测量，不是相机事实）

---

## 4. P6-05 旋转/单位回归测试

```text
python -m pytest tests/simulation/p6_frame_tree_test.py -q
.............................                                            [100%]
29 passed in 0.54s
exit=0
```

覆盖的五类错误（都是"不报错但结果错"的那种）：

1. **米 vs 毫米**：球拍整体长必须落在 `0.5 < L < 1.0` 且 ≈0.6827；臂杆平移 `< 1 m`。若把米当毫米，236 mm 会变 236 m。另外单测了"视觉比真拍大"这一**事实**（682.7 > 680 BWF 上限、> 675 产品长），防止有人用容差把它抹平。
2. **wxyz vs xyzw，按来源区分**：合同里每个旋转都写成 `{order, values}`，且 `usd_physics_and_usdgeom=wxyz`、`isaac_body_quat_w=xyzw`、`gltf=xyzw`、`racket.yaml:mount_transform.quaternion_xyzw=xyzw`；测试里 **wxyz 解码器与 xyzw 解码器是两个独立函数**。用 joint2 的真实值 `(-0.70710546, 0.7071081, 0, 0)` 断言：按 wxyz 解是绕 +X 的 **-90°**（`R22=0, R12=+1, R21=-1`），按 xyzw 解会变成绕 (-0.707,0.707,0) 的 **180°**（`R22=-1`）。另有一个不对称四元数验证两个解码器**必须**给出不同矩阵、且换序后互相复现。
3. **右手/左手**：所有断言边的 `det(R)=+1`（≥10 条），并给出反例 `diag(1,1,-1)` 的 det=-1；球拍 `+Y=Z×X` 的基也断言 det=+1。
4. **camera optical vs robot mount**：光学帧矩阵必须把 mount +X 映到 optical +Z、+Y 映到 -X、+Z 映到 -Y，det=+1；深度点 `(1,2,3)_optical → (3,-1,-2)_mount`；所有相机字段 `status=UNKNOWN` 且数值为 null；占位值 0.29/1.20/-4.0/0.145 **不得**作为契约值出现（只允许出现在"拒绝记录"里）。

5. **链式复合**（第 5 类）：joint1…joint6 的 6 个旋转矩阵连乘后仍须 `det=+1` 且 `RᵀR=I`（误差 1e-9）。单条边的约定写错不会自己报错，但连乘会离开正交群，这是抓"某一条边悄悄错了"的检查——同一条测试里还顺带用真实 sha256 校验了 GLB 原始文件（不是恒真断言）。

结构性断言：每条边要么 `ASSERTED_BY_ASSET`（带 file+sha256+locator+数值），要么 `UNKNOWN`（数值全 null 且必须写 missing_datum）；`frame_tree.json` 的矩阵必须能由 evidence 里的四元数当场重算出来；磁盘上 `racket_visual.usd` 的 sha256 必须同时等于审计记录和合同里的值；夹爪两条边 `mount_allowed=false`。

---

## 5. 复现（全部真命令，无 GUI/无 GPU）

```bash
git -C "E:\具身智能\badmin_project" worktree add "E:\具身智能\badmin_wt_p6" -b p6/racket-camera-frames origin/main   # exit 0
cd E:\具身智能\badmin_wt_p6
python tools/p6_frames/audit_racket_usd.py                                     # exit 0（需 pxr）
python tools/p6_frames/extract_robot_usd_frames.py --layer ... --out ...       # exit 0（纯文本，9 joints）
python tools/p6_frames/build_p6_frame_contract.py                              # exit 0（15 edges / 10 asserted / 5 unknown）
python -m pytest tests/simulation/p6_frame_tree_test.py -q                     # exit 0，29 passed
```

没有 pxr 的机器：跳过第 1 条（其结论已在 `evidence/racket_usd_audit.json`），其余三条照跑。

## 6. 仍未完成 / 待取证（不冒充 DONE）

- **P6-03**（相机机型/镜头光心 vs mount、未标定外参 UNKNOWN）：本文件只把**轴向约定**与**两个帧的区别**钉死；真机型的确认、手眼外参都还是 UNKNOWN，属于 P6-03。
- **P6-06** 时间戳/延迟/掉帧协议：未做。
- **P6-07** 球拍包围盒与球-拍接触最小样例：本文件只证明球拍资产**没有**碰撞体、必须自建代理，最小样例未做。
- 真机臂的 `geometries.usd`（19.8 MB）**没有**下载也没有解析：本轮只用到关节/位姿层，几何层对 frame tree 不必要。
- 三个跨安装面变换（chassis→arm、flange→tcp、mount→racket_root）**没有任何一个**变成已知；谁要说"球拍在世界系的位置已知"，那是假的。
