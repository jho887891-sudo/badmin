# P1 轮件 AXLE 几何 v2（车架系轴心 / 滚动面分类 / 轮距轴距）

- 分支：`p1/wheel-axle-geometry`（base `p1/wheel-rolling-surface` @ `c3ec799`，PR #25）
- 任务：master spec P1 / R-next 执行项 **6 + 7**（把轮身份证据推进到 P4 可用的逐轮 AXLE 几何）
- 机器可读产物：`outputs/simulation/rebot_b601_rs/morph_one/p1_wheel_identity/wheel_geometry_v2.json`
  （schema `badmin.morph_one.wheel_geometry_v2/1`，每个量都带 verdict / unit / frame / residual / source sha256）
- 本文不重复 PR #25 已定论的事实（`P1_WHEEL_ROLLING_SURFACE.md`）：轮心三法一致、自旋轴 (1,0,0)、
  r=103 只是"最大圆特征"、镜像残差 2.0872 mm < 网格 p95 2.9877 mm、13 mm 代理盒差，全部沿用。
- 本轮只做三件 PR #25 没做的事：**(1) 把轴心写进车架系；(2) 用 BREP 面群给滚动面定性；(3) 给出带符号外法向 + 镜像残差**。

## 1. 方法

只用 CPU、纯标准库、只读。没有 CAD 内核、没有 Isaac、没有 GPU、没有写主机。

| 线 | 输入 | 回答什么 |
|---|---|---|
| A. BREP 面群 | STEP `M1-4WD4S_model2601.step`（115,496,178 B，sha256 `388cb365…`）四个实体的 `ADVANCED_FACE` 闭包 | 每个面的解析曲面类型/半径、轴向带、径向带、面积 |
| B. 已录制点云合同 | PR #25 的 `wheel_identity_contract.json` + `cloud_geometry.json` | 轮心、自旋轴、镜像残差（不重复测量） |
| C. 文档化 CAD→USD 变换 | `scripts/simulation/morph_one_orientation_probe.py` 的 `rot_x(90°)` + `SCALE=0.001` | 车架系坐标 |

新增 `scripts/simulation/p1_wheel_axle_brep.py`（纯 python，复用仓库自带 STEP 记录解析器）：

1. 对每个轮件做 `forward_closure`，取全部 `ADVANCED_FACE`；
2. 面 → 曲面实体（`PLANE` / `CYLINDRICAL_SURFACE` / `CONICAL_SURFACE` / `TOROIDAL_SURFACE` / `SPHERICAL_SURFACE`）取解析参数；
3. 面 → `FACE_OUTER_BOUND` → `EDGE_LOOP` → `ORIENTED_EDGE` → `EDGE_CURVE` → 顶点与 `CIRCLE` 弧，
   按 `.T./.F.` 处理走向，用圆弧扫角给出该面的**周向覆盖率**；
4. 轴向带/径向带按**该面自身轴线**投影（螺栓孔柱面与轮轴平行但不同轴，不能按轮轴算）；
5. 面积按解析积分：柱面 `r·Δθ·Δh`、锥面 `r̄·Δθ·slant`、环面
   `Δθ_major · rₘ · [R(θ₂−θ₁) + rₘ(sinθ₂−sinθ₁)]`（θ 由环面小半径参数给出）、平面用 Newell 多边形；
6. 车架系：`v_chassis = R_x(+90°) · (0.001 · v_cad)`，即 `(x,y,z) → (x, −z, y)`。

**变换来源说明（如实标注）**：任务书写的是"p2_chassis_frame_math.py 里已用的变换"。本仓库里
`scripts/simulation/p2_chassis_frame_math.py` 只有 lift/半高算术，**不含**旋转矩阵；
0.001 + `rot_x` 的实际出处是 `scripts/simulation/morph_one_orientation_probe.py`（`SCALE = 0.001`、
`rot_x(vec, degrees)`），`p1_usd_wheel_prim_measure.py` 也以它为基准。本报告按后者执行，并在此显式记录这一差异。

## 2. 关键数字

### 2.1 逐轮轴心 + 轴向量（CAD 系 / 车架系）

轴向为已核实 `(1,0,0)`；轴线写成"点 + t·方向"。CAD 系单位 mm，车架系单位 m。

| STEP eid | CAD 轴心 (mm) | 车架系轴心 (m) | 车架系外法向 | spin_sign |
|---|---|---|---|---|
| 435700 | (250.002, −244.000, −297.000) | (0.250002, 0.297000, −0.244000) | (+1, 0, 0) | **+1** |
| 759491 | (250.002, −244.000, 297.000) | (0.250002, −0.297000, −0.244000) | (+1, 0, 0) | **+1** |
| 773650 | (−250.002, −244.000, −297.000) | (−0.250002, 0.297000, −0.244000) | (−1, 0, 0) | **−1** |
| 871375 | (−250.002, −244.000, 297.000) | (−0.250002, −0.297000, −0.244000) | (−1, 0, 0) | **−1** |

残差（两条路径：STEP 解析圆 vs USD 点云拟合）：

| 残差 | 值 | 说明 |
|---|---|---|
| 轴心路径残差 | **0.002 mm** | 点云 bbox 中心 250.002 vs STEP 圆圆心 250.0；变换是相似变换（正交 + 均匀 0.001），所以车架系里仍是 2e−06 m |
| 变换往返残差 | **0.000 mm** | `T⁻¹(T(p)) − p` 逐分量 0.0 |
| 轴方向夹角 | **0.0043°**（PCA）/ **0.1650°**（最小跨度扫描） | 对 STEP 圆轴 |
| 外法向镜像残差 | **0.000°** | 四件方向全平行 chassis x |

**注意（重要且此前未写）**：STEP 里 `CIRCLE` 的放置轴在 +x 侧存成 `(+1,0,0)`、在 −x 侧存成 `(−1,0,0)`。
轴"线"本身无向，所以**存储符号不能当自旋方向**；外法向一律由实测轮心 x 的符号赋值（见 `spin_sign`）。

### 2.2 滚动面分类：环面（crowned tread），不是圆柱，不是 103 mm 棱边

每个轮件 190 个面 = 124 柱面 + 58 平面 + 8 环面。**包含 103 mm 圆的那 4 个面全部是 `TOROIDAL_SURFACE`**：

| 量 | 值 |
|---|---|
| 环面主半径 R | **77.738095 mm** |
| 环面小半径 r | **25.261905 mm** |
| 赤道半径 R + r | **103.000000 mm**（与实测 103.0 mm 圆残差 **0.000000 mm**） |
| 径向带 | **92.500 → 103.000 mm** |
| 轴向带宽 | **41.000 mm**（轴向带 [−20.502, 20.498] mm，跨轮中面） |
| 小半径参数带 | **±0.9467 rad = ±54.2426°** |
| 该组面积（每轮） | **29,870.69 mm²**（4 个 180° 周向面） |
| 面数 | 4 |
| 实体中最大柱面半径 | **68.500 mm**（比 103 小 **34.500 mm**） |

面群面积表（每轮，mm²）：

| 面群 | 面数 | 面积 | 面积未知面 | 轴向带宽 | 径向带 | 含 103 圆 |
|---|---|---|---|---|---|---|
| TOROIDAL R=77.738（踏面） | 4 | 29,870.69 | 0 | 41.000 | 92.5–103.0 | **是** |
| TOROIDAL R=80.819（外肩弧） | 4 | 26,151.30 | 0 | 48.653 | 68.5–92.5 | 否 |
| CYLINDRICAL r=6.5（孔/槽） | 118 | 43,701.34 | 6 | 48.653 | 72.7–83.1 | 否 |
| CYLINDRICAL r=63.0 | 2 | 6,328.46 | 1 | 31.975 | 63.0 | 否 |
| CYLINDRICAL r=68.5 | 4 | 3,454.03 | 0 | 40.000 | 68.5 | 否 |
| PLANE | 58 | 1,883.19 | 0 | 31.975 | 63.0–78.9 | 否 |

裁定：**滚动面 = 环面冠状踏面（verified）**。PR #25 说的"r=103 只占 0.585 mm 轴向带"与本节不冲突：
0.585 mm 是"达到最大半径的点"的带宽，而承载滚动的**面**横跨 41 mm 并从 92.5 mm 抬升到 103 mm。

被否定的假设：

1. "103 mm 棱边邻接的是柱面" → **否定**：实体最大柱面 68.5 mm，92.5/103 处没有柱面；
2. "49 mm 宽平柱面踏面" → **否定**：踏面是环面，边缘掉到 92.5 mm；
3. "206 mm 法兰盘 + 平面环" → **否定**：平面总面积只有 1,883.19 mm²。

### 2.3 车架系轮距 / 轴距 / 滚动高度

| 量 | 值 | 不确定度 | 方向 |
|---|---|---|---|
| track_width（轮距） | **0.500004 m** | ±4e−06 m | 车架系 **x**（= CAD x） |
| wheelbase（轴距） | **0.594000 m** | ±4e−06 m | 车架系 **y**（= CAD z） |
| 四轮中心共面跨度 | **0.000000 m** | — | 车架系 z |
| 几何未加载滚动高度 | **0.347000 m** | 同轮半径不确定度 | = 0.244 + 0.103 |

**13 mm 依然成立**（P1-08）：P2/P4 代理盒半高 0.090 m → `lift = 0.334 m`；几何给出 0.347 m，
差 **0.013 m**。代理 lift 仍然**不是**真实滚动高度。

**上轴冲突（PR #25 §2.6）本轮部分解决**：文档化变换把 CAD y（四轮共面的几何竖直）映射到**车架系 z**，
与 USD stage 声明的 `upAxis = Z` **一致**。所以"USD 声明 vs 几何推断"并非真冲突，而是 PR #25 在
未施加旋转的 CAD 系里比较造成的。但"授权重力方向"仍缺（→ UNKNOWN，见 §5）。

### 2.4 四件是镜像对，不是四个相同件

| 证据 | 值 |
|---|---|
| 踏面环面轴向带镜像残差 | **0.000000 mm**（[−20.502, −0.002]/[−0.002, 20.498] ↔ 符号翻转后逐段精确相等） |
| STEP 存储轴符号 | +x 侧 `+1`，−x 侧 `−1` → 放置被镜像 |
| 点云镜像 Hausdorff（PR #25） | 2.0872 mm < 网格 p95 2.9877 mm（**不足以**证明物理非对称） |

裁定：**镜像左右对（verified，在 BREP 面签名层面）**；"物理零件号是否同一件"仍为 candidate。

## 3. 逐量裁定表

| 量 | 裁定 | 值 | 单位 | 系 | 源（sha256） | 残差 |
|---|---|---|---|---|---|---|
| 轴方向（逐轮） | **verified**（仅方向） | (1,0,0) | unit | CAD / 车架 | contract `…` | 0.0043° PCA / 0.165° 扫描 |
| 轴心（CAD） | **candidate** | 见 §2.1 | mm | CAD | contract | 0.002 mm |
| 轴心（车架） | **candidate** | 见 §2.1 | m | 车架 | v2 | 往返 0.000 mm |
| 外法向 + spin_sign | **candidate** | ±x，见 §2.1 | unit | 车架 | v2 | 镜像 0.000° |
| 滚动面类型 | **verified** | TOROIDAL_SURFACE | — | CAD BREP | faces_v2 `f758aeb3…` | 赤道 0.000000 mm |
| 踏面几何 R/r/带/面积 | **verified** | 77.738095 / 25.261905 / 41.0 mm / 29,870.69 mm² | mm, mm² | CAD | faces_v2 | 面积法见 §1 |
| 轮半径（未加载赤道） | **candidate** | 103.000 mm | mm | CAD/车架 | faces_v2 + contract | 2.2e−05 mm（点云 103.000022） |
| 轮半径（受载滚动） | **UNKNOWN** | — | — | — | — | 缺受载接触测量 |
| track / wheelbase | **candidate** | 0.500004 / 0.594000 | m | 车架 | v2 | ±4e−06 m + 未冻结基准 |
| 四轮共面 | **verified** | 0.000000 | m | 车架 | v2 | — |
| 零件身份 | **verified（镜像）/ candidate（件号）** | 镜像对 | — | CAD | faces_v2 | 0.000000 mm |
| 未加载滚动高度 | **candidate** | 0.347000 | m | 车架 | v2 | 与代理差 0.013 m |
| up 轴 | **candidate** | 车架 z（= CAD y） | — | 车架 | probe | 与 USD `upAxis=Z` 一致 |
| 质量/惯量/摩擦/胎面材料 | **UNKNOWN** | — | — | — | — | 缺厂家参数或实测件 |

## 4. P4 消费合同 v2

**可作为 TEST_PROXY 参数消费（全部 candidate，必须显式参数化 + 敏感性扫描）**

| id | 值 | 约束 |
|---|---|---|
| `wheel_axle_direction_chassis` | 单位向量 chassis x | 只用方向；自旋符号取 `spin_sign`，不得硬编码 ± |
| `wheel_centre_chassis` | (±0.250002, ±0.297000, −0.244000) m | 只用相对几何；原点**不是**标定基准 |
| `track_width` | 0.500004 m | 沿 chassis x |
| `wheelbase` | 0.594000 m | 沿 chassis y |
| `wheel_unloaded_radius` | 0.103 m（环面赤道 R+r） | 仅代理；受载半径 UNKNOWN |
| `tread_axial_band` | 0.041 m 宽，居中于轮平面 | 用于接触带宽度代理，不是轮胎模型 |
| `wheel_spin_sign` | +x 轮 +1，−x 轮 −1 | 相对外法向；前向仍 UNKNOWN |

**禁止当真值冻结**

1. 0.103 m 当物理轮半径；
2. 受载滚动高度 / 任何胎面压缩量、接触斑尺寸；
3. `lift = 0.334 m`（比几何 0.347 m 低 13 mm，从未实测）；
4. 0.090 m 代理碰撞盒半高当真实踏面半宽；
5. 轮质量、惯量张量、摩擦系数、胎面柔度；
6. 车架系原点当标定基准、chassis z 当授权重力方向；
7. 任何由轴族聚类推出的 axle joint 轴 / 电机通道；
8. 转向自由度、转向范围、转向轴位置；
9. 环面踏面当轮胎材质或真实接触模型证据。

**Gate**：读到 UNKNOWN 却按冻结值请求 → fail closed；任何 candidate 必须有扫描；自旋符号必须取
`outward_axle_direction.spin_sign`。

## 5. 仍然 UNKNOWN（按缺哪次测量列出）

| 量 | 缺什么测量 |
|---|---|
| 受载滚动半径 | 受载接触试验（或胎面刚度 + 整车载荷） |
| 接触斑几何 | 只有未加载冠状面；缺压缩/接触测量 |
| 车架基准（原点 + 授权 up 轴） | 厂家车架帧声明或物理基准标定 |
| 前向（哪一侧是车头） | 整车前向标注 |
| 转向轴位置/范围 | joint 定义或实测转向位姿扫描 |
| 轮质量/惯量/摩擦 | 厂家参数或实测件 |
| 胎面材料 | 材料规格；BREP 不含材质 |
| 7 个面的面积（每轮） | 这些面由样条裁剪，柱面/锥面弧覆盖法给不出面积（已在 `area_unknown_faces` 记数） |

## 6. 命令与真实退出码

```bash
# 1) 纯 python STEP BREP 面群普查（只读、无内核、无 GPU）
python scripts/simulation/p1_wheel_axle_brep.py \
  --step "E:/具身智能/badmin_project/_scratch_downloads/M1-4WD4S_model2601.step" \
  --centres-json outputs/simulation/rebot_b601_rs/morph_one/p1_wheel_identity/wheel_centres_cad_mm.json \
  --out-json outputs/simulation/rebot_b601_rs/morph_one/p1_wheel_identity/wheel_brep_faces_v2.json
# BREP_EXIT=0    （首次运行 BREP_EXIT=1：工作树内没有 _scratch_downloads，FileNotFoundError；
#                 修成绝对路径后 0。另修了一处 VERTEX_POINT 解析 bug，见 §7）

# 2) 生成 v2 记录
python scripts/simulation/p1_wheel_axle_geometry.py
# BUILD_EXIT=0   （此前两次 BUILD_EXIT=1：一次语法错误、一次缺 verdict/constraint/residual 字段——
#                 后者是被第 4 步的测试抓到的，不是被掩盖的失败）

# 3) 陈旧性检查（提交的 JSON 必须等于重算结果）
python scripts/simulation/p1_wheel_axle_geometry.py --check
# OK: ... is up to date -> CHECK_EXIT=0

# 4) 形状/单位/残差上界测试（纯 python）
python -m pytest tests/simulation/p1_wheel_geometry_v2_test.py -q
# 13 passed in 0.20s -> PYTEST_EXIT=0

# 5) 推送
git push -u origin p1/wheel-axle-geometry
# push_exit=0
```

## 7. 方法学上的自我修正（记录在案）

- 面积法第一版对环面用 `2π·r·R·Δθ`（假设小半径方向整圈），把踏面面积高估 **2.6 倍**（155,056 → 59,740）。
  改为按小半径参数 θ 积分 `Δθ·rₘ·[R(θ₂−θ₁) + rₘ(sinθ₂−sinθ₁)]`，得到 29,870.69 mm²。
- 周向覆盖率第一版按"轮轴"聚合圆弧，导致 118 个螺栓孔柱面 `Δθ = 0`、面积 None。
  改为按**面自身轴线**聚合（孔轴与轮轴平行但不同轴），面积恢复。
- 外法向第一版直接取"存储轴 × 轮心符号"，因 −x 侧存储轴本身是 `(−1,0,0)`，会把左侧写成 +x。
  改为只用实测轮心符号赋值，并把存储符号单独记成镜像证据。

## 8. 证据路径与 sha256（LF 归一化）

| 文件 | 内容 | sha256 |
|---|---|---|
| `outputs/.../p1_wheel_identity/wheel_geometry_v2.json` | **v2 合同本体**（逐量 verdict/unit/frame/residual） | `66b24074…` |
| `outputs/.../p1_wheel_identity/wheel_brep_faces_v2.json` | 190×4 个 BREP 面的类型/半径/带/面积 | `f758aeb3…` |
| `outputs/.../p1_wheel_identity/wheel_centres_cad_mm.json` | 四轮 CAD 轴心（BREP 普查输入） | `f8db5451…` |
| `scripts/simulation/p1_wheel_axle_brep.py` | STEP BREP 面群普查 | `a83fe742…` |
| `scripts/simulation/p1_wheel_axle_geometry.py` | v2 记录生成器（含 `--check`） | `5c5034d9…` |
| `tests/simulation/p1_wheel_geometry_v2_test.py` | 13 项 schema/单位/残差测试 | `6ef335bc…` |

外部资源（**不入库**，可再生）：STEP `_scratch_downloads/M1-4WD4S_model2601.step`，
115,496,178 B，sha256 `388cb3653a864066a35a54065f71938635c7e9a221cb4b897b31fbd44ff266c5`。
主机只读原始证据（未修改、未写入）：`dgut@172.31.68.251:/home/dgut/runs/wheels/out/`，
sha256 已由 PR #25 记入合同 `inputs`；本轮**没有**再访问主机（数据已随 PR #25 本地录制）。
