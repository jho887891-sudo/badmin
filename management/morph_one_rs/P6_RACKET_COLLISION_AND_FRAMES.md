# P6-07 + P6-02：球拍碰撞代理 + 安装锚点合并帧树

- 分支：`p6/racket-collision-proxy`（worktree `E:\具身智能\badmin_wt_p6b`，基线 `p6/racket-camera-frames` @ d7672d5）
- 本轮新增（不改动 P6/P3 已交付的任何文件）：
  - `tools/p6_frames/build_racket_collision_proxy.py` → `outputs/simulation/rebot_b601_rs/morph_one/p6_frames/racket_collision_proxy.json`
  - `tools/p6_frames/merge_mount_chain.py` → `outputs/.../p6_frames/mounted_chain.json`
  - `outputs/.../p6_frames/evidence/p3_mount_frame_tree_at_0bbed19.json`（P3-01 原文按字节钉住的副本）
  - `tests/simulation/p6_racket_proxy_test.py`（20 项）
- 全部为纯 Python + numpy（本地另用 pxr 0.26.11 **只读**打开 USD）；**未使用 Isaac Sim，未用 GPU，未写 GPU 主机，未改动 `assets/`**。

## 1. 先说结论

| 项 | 结果 |
|---|---|
| 球拍碰撞代理 | 43 个原始体（handle 4 / shaft 1 / frame 27 / string_bed 11，排序后共 43），单位米，全部带证据指针 |
| 视觉表面包含性 | 20 000 个表面采样点 **100.000000 %** 落在代理内或距其表面 ≤ 1 mm，最大未覆盖距离 **0.614 mm**（容差 1 mm） |
| bbox 体积覆盖率 | **7.45 %**（球拍 bbox 的绝大部分本来就是空气） |
| 代理体积 / 网格壳体积 | 0.000425 m³ / 0.001467 m³ = **0.290**（两者都是估计量，见 §3.4） |
| 代理“多占的空气”（leak） | 最大 **15.36 mm**，p95 8.65 mm，均值 **2.37 mm** |
| 合并后帧链 | 15 条边：**10 asserted / 5 UNKNOWN**，UNKNOWN 恒为 `[1, 2, 13, 14, 15]` |
| 组合校验 A（UsdGeom xformOps） | det(R) = **+1.0**，正交误差 0.0，末端位姿与 P3 的 pxr 组合结果偏差 **0.0 m** |
| 组合校验 B（P6 关节锚点） | det(R) = **+1.0**，但与 A 相差 **0.33242 m** → 记为 **OPEN 问题 P6-02-OI-1**，不作数据使用 |

## 2. 真实命令与真实退出码

Windows PowerShell 下 `pytest tests/simulation/p6_*.py` **不会**由 shell 展开通配符，pytest 收到字面量后直接报错——这是实测：

```text
PS> python -m pytest tests/simulation/p6_*.py -q
ERROR: file or directory not found: tests/simulation/p6_*.py
EXIT_GLOB=4
```

等价的两种正确写法（均已实跑）：

```text
PS> python -m pytest "tests/simulation/p6_frame_tree_test.py" "tests/simulation/p6_racket_proxy_test.py" -q
49 passed in 1.81s
EXIT_ALL=0

PS> python -m pytest tests/simulation -k "p6_" -q
49 passed, 196 deselected in 2.83s
EXIT_K=0
```

拆分计数（基线未回归 + 新增独立）：

```text
PS> python -m pytest tests/simulation/p6_frame_tree_test.py -q      # 基线 P6-05，未改一行
29 passed in 0.51s
EXIT_OLD=0

PS> python -m pytest tests/simulation/p6_racket_proxy_test.py -q    # 本轮新增
20 passed in 1.44s
EXIT_NEW=0
```

代理构建与合并工具的真实退出码：

```text
PS> python tools/p6_frames/build_racket_collision_proxy.py
primitives: 43
containment: inside=0.640050 within_tol=1.000000 max_uncovered=0.000614 m
bbox coverage fraction = 0.074525 (bbox vol 0.004670336 m3)
leak: max 0.015355 m, mean 0.002366 m
proxy vol 0.000425222 m3, shell vol 0.001466939 m3
EXITCODE=0

PS> python tools/p6_frames/merge_mount_chain.py
edges: 15 asserted=10 unknown=5 ids=[1, 2, 13, 14, 15]
A (P3 xformOps): det(R)=1.0 ortho_err=0.0 vs pxr end pose max delta=0.0 m
B (P6 joint anchors, edges [3, 4, 5, 6, 7, 8, 9, 12]): det(R)=1.0 ortho_err=0.0 vs A max delta=0.33242 m
open issue P6-02-OI-1: OPEN
EXITCODE=0
```

构建器是**会失败**的：`within_tolerance_fraction < 1.0` 时返回 1，报告与 JSON 里都写死“此文件不得被当作 collider 使用”。

## 3. P6-07 碰撞代理

### 3.1 输入与证据

| 输入 | sha256 | 说明 |
|---|---|---|
| `assets/third_party/racket_visual.usd` | `643fb67003c6429f98ec132d2db056c83ed02811b594ad27904b618bf7c1ec29` | 43 217 B，USDC，metersPerUnit=1.0，upAxis=Z |
| `/RacketVisual/Obj_Racket_0` | 同上 | 1 632 点 / 828 三角面，handle+shaft+hoop（**无碰撞体**） |
| `/RacketVisual/Obj_Strings_0` | 同上 | 44 点 / 40 三角面，拍线床（**无碰撞体**） |

两个 mesh 的 local-to-world 经 pxr 读回后与单位阵逐元素一致（abs ≤ 1e-9）；若不是单位阵，构建器直接 abort——否则用 `racket_root` 坐标拟合出来的原始体会悄悄错位。

**代理不是厂商碰撞模型**：源本身是 CC-BY-4.0 低模视觉资产，总长 682.7 mm vs 产品 675 mm / BWF 上限 680 mm；本文件只声明几何，不声明质量/惯量/摩擦/恢复系数。

### 3.2 分段（全部由顶点环读出，无一个数来自数据手册）

| 量 | 值 | 来源 |
|---|---|---|
| 握把半径 | 0.015845 m | `z < 0.4·z_tip` 环的最大径向距离 |
| 中杆半径 | 0.003896 m | 0.1·z_tip < z < 0.6·z_tip 各环径向距离的**最小值** |
| 握把顶端 | 0.156690 m | 径向仍大于 (grip+shaft)/2 的最后一个环 |
| 分叉起点 | 0.437280 m | 首个轮廓宽度 > 2·r_shaft 的环 |
| 拍头底端 | 0.444180 m | 轮廓宽度 > 0.25·最大宽度的首个环 |
| 拍尖 | 0.682700 m | 最大 z |
| 拍框管半径 | 0.009015 m | 拍头环的 max((outer−inner)/2, max|x|)（修复后值，原拟合 0.003896 m） |

### 3.3 原始体顺序（ordered set；JSON 里 `order` 严格非降）

1. `racket_handle`（capsule，z 0 → 0.15669，r = 15.845 mm）
2. `racket_grip_taper_01..03`（3 段 capsule 逼近握把→中杆锥度，每段取该段较粗端半径）
3. `racket_shaft`（capsule，z 0.15669 → 0.43728，r = 3.896 mm，**未被放大**）
4. `racket_frame_fork_p/n`（2 条 capsule：中杆轴 → 拍头底部实测管心线）
5. `racket_frame_rim_p01..13 / rim_n01..13`（26 条 capsule，轴 = 每环 (outer+inner)/2 的实测管心线，13 节点线性插值）
6. `racket_frame_crown`（横跨拍尖的 1 条 capsule：拍顶是 YZ 平面内的横向圆弧，只靠左右两条竖链会漏尖）
7. `racket_string_bed_slab_01..11`（11 个 box，相邻实测线床环之间一板，Y 半宽 = 该板内线床顶点 max|y|，X 半厚 = max|x| = 1 mm）

线床按**实体膜**建模（本项目球速下拍线不会漏球），因此拍头内部的“洞”不是碰撞空洞——这一点是设计选择，已写进每个 slab 的 `dimension_derivation`。

### 3.4 精度声明（可复算，不是形容词）

- **包含性**：20 000 个按面积加权的三角面重心采样点（seed 20261021），`inside_fraction = 0.640050`，`within_tolerance_fraction = 1.000000`，`max_uncovered_distance_m = 0.000614`（容差 `CONTAINMENT_TOL_M = 0.001`，取中杆半径的十分之一量级）。含义：视觉球拍不会从代理里“捅出去”。
- **修复过程**：先按测量值建体，再逐点“只放大该点最近的那一个原始体”，每轮记录 `primitive / channel / from / to`，最多 12 轮（实跑 2 轮），共 42 条 growth 记录，全部在 `accuracy.repair.growths`。**没有整体盲目放大**：中杆半径在加入 fork 原始体后保持 3.896 mm 未变。
- **bbox 体积覆盖率**：在视觉 bbox（0.004670336 m³）内均匀撒 200 000 点，落在代理内的比例 = **0.074525**。球拍 bbox 本身绝大部分是空气，这个数天然偏低；它衡量的是“代理有没有退化成实心块”。
- **代理/壳体积**：0.000425 m³ / 0.001467 m³ = 0.290。两者都是估计量：代理是**并集**（重叠部分重复计入），壳体积是开放低模壳的有符号四面体和。
- **leak（多占的空气）**：在代理内部 4 000 个采样点，到最近视觉三角面的距离 max 15.36 mm / p95 8.65 mm / 均值 2.37 mm，最大点出现在线床 slab 顶部（椭圆床在 slab 内收窄处）。

### 3.5 明确“不是什么”

- 不是厂商/CAD 碰撞模型（源是低模视觉资产，且长 682.7 mm ≠ 675 mm 产品）；
- 不是物理材质模型（无质量/惯量/摩擦/恢复）；
- 不提供**安装**信息：手柄→法兰的变换仍是 UNKNOWN（帧树 edge 13/14），代理没有添加任何安装几何；
- 不是形变模型（拍线按刚性实体膜处理）。

## 4. P6-02 合并帧树（`mounted_chain.json`）

### 4.1 唯一一条链 + 不重复任何数值

链序（`chain_order`）：
`world → chassis → arm_base → base_link → link1 → link2 → link3 → link4 → link5 → link6 → gripper_end → racket_mount → racket_root → racket_contact_frame`

- **数值只有一份**：每条 asserted 边只带 `value_ref = {file, sha256, json_pointer}`，指向已提交的证据文件；**文件里没有第二个 3 维向量**。测试 `test_numeric_edges_are_pointers_and_the_pointers_verify` 会校验指针 sha256 与源文件字节一致，并断言该边自身不得再出现 `translation_m / rotation_quat / rotation_matrix / values`。
- **命名去重**：P6 的 `arm_root_prim` 与 P3 的 `arm_base` 是同一个 `/ReBot_Arm_RS`，合并后只保留一个节点名 `arm_base`，别名关系写在 `naming_reconciliation`，不再分裂成两条链。
- **P3-01 原文钉住**：`git show origin/p3/mount-frame-tree:.../p3_mount/mount_frame_tree.json > evidence/p3_mount_frame_tree_at_0bbed19.json`，内容未改。文本证据的 sha256 一律按 **LF 归一化后的字节**计算（本 working tree 是 CRLF、对象库是 LF，只在某一种行尾下成立的哈希不是证据）：P3 副本 `96366ec79a684e24bf3f3b67deefef29faf798f694cf8d290ecf41a1f7343e9c`，P6 帧树 `7764d0d1a78430f7c3505301afa4e49f4ab83ec2b31ec7fa28ae5ca2d04443bf`。已实测：把文件转成 CRLF 后归一化哈希不变（`de2186c6...` → `96366ec7...` 归一化后一致），所以换行约定不会让校验在新 clone 上假失败。二进制资产（racket USD）仍用**原始字节** sha256。

### 4.2 P3 安装锚点如何进入链（而**没有**把 UNKNOWN 变成已知）

edge 2 `chassis → arm_base`：

- **物理安装偏移：仍然 UNKNOWN**（状态、`missing_datum`、`would_come_from` 全部保持 P6 原文，无任何数字载荷）。
- 另附 `p3_01_anchor`（子字段，指向 P3 文件）记录 **SIM 作者身份**：`root_joint` 的 `physics:localPos0 = localPos1 = 0`、`localRot0 = localRot1 = 恒等`（层 `payloads/Physics/physics.usda`，sha256 `8d18a8c9...`），即“仿真里底盘原点 == 臂基座原点”。合并时**重新验证**：`local_rot0_equals_local_rot1 = true`、`translation_from_anchor_m = [0,0,0]`、`matches_p3_derived_transform = true`。
- 结论写死在 `authoring_consequence`：仿真组合出 identity 是**作者选择**，Morph One 顶板上的真实螺栓孔位偏移仍未测，**任何安装偏移都不许凭空补**。

### 4.3 组合校验

| 组合 | 组成 | det(R) | 正交误差 | 与参照 |
|---|---|---|---|---|
| A（链上记录口径） | P3-01 的 UsdGeom xformOp hops（9 跳） | **+1.0** | 0.0 | 与 P3 的 pxr 组合末端位姿 `(0.301710169999, 2.62633e-07, 0.217746720407)` 偏差 **0.0 m** |
| B（诊断口径） | P6 的 8 条关节/几何锚点边 | **+1.0** | 0.0 | 与 A 相差 **0.33242 m** |

**OPEN 问题 P6-02-OI-1（本轮新发现，必须上报）**：USD 里一条关节的局部位姿是**两段**（body0 侧 `localPos0/localRot0`、body1 侧 `localPos1/localRot1`），而 P6 帧树每条 revolve 边只记录了 body0 侧，并把它当作“子连杆位姿”。因此 B 组合**无法**复现 A。两者 det(R) 都是 +1，所以这**不是**手性问题，而是缺 `localPos1/localRot1`。处理方式：链上记录口径用 A；B 只作诊断，并在 `open_issue` 里写明下一步（从只读臂 USD 抽出 1..6 号关节的 `localPos1/localRot1` 后重算 A vs B）。**没有为了让数字好看而修改 P6 的任何一条边。**

## 5. UNKNOWN 清单（仍然 UNKNOWN，且测试会拦住“猜一个填上”）

| edge | from → to | 缺的数据 | 一旦被填数会怎样 |
|---|---|---|---|
| 1 | world → chassis | Morph One 在真实场地的位姿 (x, y, yaw) | `test_an_unknown_edge_may_not_carry_any_number` 立刻失败 |
| 2 | chassis → arm_base | 顶板螺栓孔位与偏航（**物理**安装） | 同上；即使 P3 identity 存在，状态仍必须是 UNKNOWN |
| 13 | gripper_end → racket_mount | T_flange_tcp：手柄轴从法兰伸出的位置与绕轴时钟相位 | 同上 |
| 14 | racket_mount → racket_root | 手柄插入深度 + 拍面法向时钟相位 | 同上（视觉资产原点在**柄尾**，不是配合面） |
| 15 | racket_root → racket_contact_frame | 甜区/拍面中心在 racket_root 下的实测坐标 | 同上 |

另有相机侧未决项（沿用 P6 交付，本轮未改，也未假装已知）：**D455 内参（fx, fy, cx, cy, 畸变）与双目外参仍然全部 UNKNOWN**，唯一被证明的只是“光学坐标系 vs 机械安装坐标系”的约定；`camera_metadata.json` 里的数值一个都没变。

## 6. 回归风险与未做的事

- 未修改 `assets/`、`scripts/simulation/p2_*/p3_*/p4_*`、`simulation/`、`src/`、队列状态文件、`PHASE_P2_*/P3_*/P4_*` 文档；P6/P3 已交付的 JSON 一个字节未动（新增文件全部是 `??` 未跟踪文件）。
- 未接 Isaac：本代理**尚未**在 PhysX 里实测接触行为；它只保证“几何上包住视觉模型”，接触参数仍由后续 M1 阶段决定。
- 未解决 P6-02-OI-1（缺 `localPos1/localRot1`），因此**不要**用 B 口径去驱动动力学。
- 未验证真实球拍尺寸：视觉 682.7 mm 比产品 675 mm 长 7.7 mm，代理继承了这个放大，未缩放（`configs/racket.yaml measured_unit` 仍是 REQUIRES_MEASUREMENT）。
