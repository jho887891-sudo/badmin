# P1 轮件滚动面与轮身份合同（P1-04 / P1-05 / P1-08 / P1-09）

- 分支：`p1/wheel-rolling-surface`（base `p1/wheel-identity-205-vs-181` @ `76c7022`）
- 机器可读产物：`outputs/simulation/rebot_b601_rs/morph_one/p1_wheel_identity/wheel_identity_contract.json`
- 本文只记录**已录制证据**测得的东西。任何候选值都不得升格为冻结参数；凡是证不了的量一律 UNKNOWN，并写明缺哪一次测量。
- 轮件身份：STEP eid `435700 / 759491 / 773650 / 871375`，四件在 CAD 中的 bbox 中心分别为
  `(±250.002, -244.000, ±297.000)` mm。

## 1. 方法

只用 CPU、只读、不启动 Isaac Sim、不碰 GPU。四个轮件的形状用三条互相独立的测量线交叉确认：

| 线 | 工具 / 数据 | 回答什么 |
|---|---|---|
| A. 解析曲面 | STEP BREP 的 `CIRCLE` / `CYLINDRICAL_SURFACE` 实体（`step_remeasure.json`） | 最大圆/柱面半径、圆心、轴线 |
| B. 三角网格点云 | USD `Mesh` 顶点经 `localToWorld` × `metersPerUnit` × 1000（`cloud_geometry.json`） | 独立自旋轴、包围盒、径向包络、圆心拟合 |
| C. 整机轴族聚类 | `wheel_kinematics.json` 的 `corners` / `axis_clusters`（282 solids → 2224 clusters → 2755 segments） | 四角转轴/驱动轴候选、镜像对称 |

新增脚本 `scripts/simulation/p1_wheel_cloud_geometry.py`（pxr + numpy）：

1. **最小投影跨度方向搜索**：4001 个 Fibonacci 半球方向 + 3 次局部收缩，取投影跨度最小的方向 → 独立自旋轴候选；
2. **协方差主轴**（特征值分解）：最小特征值对应方向是第二个独立自旋轴候选，且与 CAD 轴夹角直接给出度数值；
3. **近轴径向包络**：绕候选轴按 5° 分箱取每箱 `r_max`（滚动面判据：恒半径 = 圆柱滚动面）；
4. **边缘圆拟合**（Kasa）：只取 `r >= r_max - 0.05 mm` 的点做最小二乘圆拟合 → 不依赖任何 BREP 放置实体得到轮心与半径；
5. **镜像残差**：把源点云对指定平面镜像后与对角轮点云做**精确最近邻 Hausdorff**（分块暴力，无 scipy）；
6. **网格分辨率**：点云内到**另一个**顶点的最近距离分布，用来判断镜像残差是否只是网格分辨率。

`scripts/simulation/p1_wheel_identity_contract.py` 只读上面这些录制件，生成合同 JSON，并内置断言：
候选半径不得升格、UNKNOWN 必须带 `missing_measurement`、代理盒与 103 mm 特征差必须为 13 mm。

## 2. 关键数字

### 2.1 轮心与包围盒（CAD 坐标系，mm）

| 量 | 值 | 来源 |
|---|---|---|
| 点云包围盒 | 48.7911 × 205.9914 × 205.9914 | `cloud_geometry.json` |
| BREP 顶点包围盒 | 48.6527 × 180.782 × 157.7827 | `comparison.json` |
| BREP **曲面**包络包围盒 | 50.5238 × 206.000 × 206.000 | `comparison.json` |
| 边缘圆拟合半径 | **102.999793**（rms 残差 0.000147，最大 0.000221） | `cloud_geometry.json` |
| 点云 `r_max` | 103.000022 / 103.000019 / 103.000015 / 103.000010 | 同上 |
| 轮心（拟合） | `(±250.002, -244.000, ±297.000)`，与 bbox 中心偏差 0.000 mm | 同上 |
| 轴向宽度 | 49.2491 / 49.2491 / 49.2479 / 49.2479（中点 -0.0004） | 同上 |
| 径向跨度 | `r_min` 62.9543 → `r_max` 103.000022，跨度 40.046；p50 80.07、p90 98.71 | 同上 |

### 2.2 自旋轴（三个独立估计，全部指向 CAD x 轴）

| 估计 | 方向 | 与最近 CAD 轴夹角 |
|---|---|---|
| STEP 最大圆轴线 | `(1, 0, 0)` | 0.000° |
| 点云最小跨度扫描 | `(0.999996, 0.000497, -0.002795)` | **0.1626°** |
| 点云协方差最小主轴 | `(1.0, -0.000069, 0.000029)` | **0.0043°** |

三个估计两两一致到 < 0.5° 容差内；最小跨度扫描量得 49.2491 mm，Fibonacci 网格最小值 51.9075 mm（局部收缩把跨度压下去 2.66 mm）。

### 2.3 "r=103 mm" 到底是什么

- 每个轮件的 `r_max` 特征只占据 **0.5847 mm 宽的轴向带**（`rim_point_count = 64`），而轮子轴向总宽 49.25 mm。
  因此这个 r=103 mm 圆是一条**环形棱边（轮缘唇 / 挡边）**，不是一个 49 mm 宽的滚动踏面。
- 轴向切片看，`r_max = 103.0` 只出现在 `-256.9 … -238.4` mm 两段薄切片；相邻切片的最大半径是 **102.282 mm**（另一条略小的圆柱面）。
- 方位角包络：72 个 5° 箱全部有点，中位数 **102.9998 mm**，8 个箱低到 84.03 mm。
  这 8 个低箱严格等间隔 **45°** 出现（72/8 = 9 箱），而外圆三角化只有 64 个顶点（360/64 = 5.625°）——
  即 5° 分箱与 64 边形采样发生**拍频混叠**，低箱里根本没有外圆顶点，不是轮子上的缺口。此结论已写进合同
  `circular_features.envelope_aliasing`，避免后人把它当几何缺陷。

结论：**`r = 103 mm` 只能报作 "max circular feature radius candidate"**，不能当已核实的轮半径/直径 206 mm。

### 2.4 四角转轴 / 驱动轴候选（说明这不是普通四轮）

- 四个角都有**竖直（y）圆族**：点 `(±250.0, -99.61 / -100.68, ±297.0)`，半径 ≤ 40 mm，`axial_span = [-214.02, -1.30]` mm，每个角 106/107 个特征 → 舵轮（转向）候选。
- 每个角的**驱动轴候选**：方向 `(1, 0, 0)`，半径 103.0 mm，每个角 324 个特征，轴线点 `(±231.24, -244.0, ±297.0)`。
  轴线点 x = 231.24 与轮心 x = 250.002 的 18.76 mm 差异**不构成矛盾**：x 向轴线上任意一点都代表同一条轴线，起决定作用的是 y = -244.0、z = ±297.0。
- `wheel_kinematics.json` 记录 `symmetry_pairs = 4`、`symmetry_consistent = 4`、`corners_with_steering_candidate = 4`。

⇒ 四轮在 CAD 里是**同一个转向位姿**下的静态姿态。转向自由度、转向范围、以及"轮子在任意转向角下的轴心"都**未被证明**（UNKNOWN）。

### 2.5 镜像对称残差

| 平面 | 源 → 目标 | Hausdorff | 平均最近邻 |
|---|---|---|---|
| z=0 | 435700 → 759491 | 2.0872 mm | 0.9329 mm |
| x=0 | 435700 → 773650 | 2.0872 mm | 0.9329 mm |
| x=0 | 759491 → 871375 | 2.0872 mm | 0.9329 mm |
| z=0 | 773650 → 871375 | 2.0872 mm | 0.9329 mm |

点云自身到**另一个**顶点的最近距离：`min = 0`（网格存在重合顶点）、`p95 = 2.9877 mm`、`max = 3.23 mm`。
镜像残差 2.087 mm **小于**网格分辨率 p95 2.99 mm，所以**不能**据此声称真实非对称；只能声称"四件在网格分辨率内互为镜像"。
绕轴一个顶点步长的相位差无法排除，也不做声明。合同里 `mirror_symmetry.verdict = PASS`（解释已固定）。

### 2.6 竖直轴：几何推断与 USD 声明冲突

- 四个轮心共享同一个 `y = -244.000` 平面（跨度 0.000 mm）⇒ 唯一与"四轮共面"自洽的竖直轴是 CAD **y**。
- 但 `usd_remeasure.json` 记录 stage `upAxis = "Z"`，`wheel_kinematics.json` 的 `chassis_frame` 用"最小总跨度"推断出 y（整体尺寸 580.0 × 360.391 × 802.852 mm）。
- 这是一处**必须在 P2/P4 前解决的冲突**：USD 声明与几何推断不一致，且两者都不是厂家授权的重力方向。

## 3. 反向 / 稳健性核查

### 3.1 只用 BREP 顶点 vs 完整点云（P1-02 口径复演）

| 口径 | 薄轴 | 面内两轴 | 与点云差 |
|---|---|---|---|
| 只取 BREP 250 个顶点 | 48.6527 | 180.782 / 157.7827 | 薄轴 -0.138 mm，面内 **-25.2 / -48.2 mm** |
| BREP 曲面包络（解析曲面求界） | 50.5238 | 206.000 / 206.000 | 薄轴 +1.73 mm，面内 **+0.0086 mm** |
| 完整三角点云（6354 点） | 48.791061 | 205.991409 / 205.991409 | 基准 |

结论：**只拿 BREP 顶点盒当轮件身份是错的**（面内低估 25–48 mm，正是 205.991 vs 180.782 的来路）；
解析曲面包络与三角点云在面内一致到 0.0086 mm，最大圆半径 103.0 vs 103.000022 mm 差 2.2e-05 mm。
两条独立的"曲面口径"互证，所以 206 mm 圆**确实存在**；但"存在"≠"滚动面"。

### 3.2 反向检查：有没有更大的特征被漏掉？

- 四件 `r_max` 最大 103.000022 mm，与解析最大圆半径 103.0 mm 一致，没有更大者。
- 最大**柱面**半径仅 68.5 mm（`step_remeasure.json`），`circle_count = 136`、`cylindrical_surface_count = 124`。
  即 r=103 只有一条圆（棱边），而**没有一个半径为 103 mm 的柱面**——这正是"是棱边不是踏面"的解析侧证据。
- 点云 122 个 shell、6354 点、7900 面，四件点数完全相同，无遗漏子件。

## 4. 逐量裁定

| 量 | 裁定 | 值 / 来源 | 依据与不确定度 |
|---|---|---|---|
| **wheel_radius** | **UNKNOWN** | 候选：max circular feature radius **103.0 mm**（STEP 解析）/ 103.000022 mm（点云），残差 2.2e-05 mm | r=103 特征只占 0.585 mm 轴向带（=棱边），且无 r=103 柱面。缺：该面是否接地（接触带分类）+ 胎面材料 + 受载半径 |
| **spin_axis** | **PASS**（仅方向） | `(1,0,0)`，三法一致，最大夹角 0.163° | 不确定度 0.5°；轴线**位置**与转向角下的轴未证 |
| **wheel_centre** | **PARTIAL** | `(±250.002, -244.000, ±297.000)` mm（CAD 系） | 三法一致到 0.002 mm；但**无车架基准标定**，车架系坐标 UNKNOWN |
| **track_width** | **PARTIAL** | 候选 **500.004 mm**（左右轮心 x 距） | 继承车架基准缺失；且只在录制的转向位姿下有意义 |
| **wheelbase** | **PARTIAL** | 候选 **594.000 mm**（前后轮心 z 距） | 同上 |
| **up_axis** | **PARTIAL** | 几何竖直 = CAD y（四轮共面 + 跨度 0.000） | 与 USD `upAxis = Z` 冲突，无授权基准 |
| **wheel_mass / inertia / friction** | **UNKNOWN** | — | 录制证据中无密度/质量/胎面材质 |
| **loaded_rolling_height** | **UNKNOWN** | 拒绝候选：`lift = 0.334 m` | 0.334 m 来自 90 mm 半高 TEST_PROXY 盒；P2/P4 碰撞盒半高 0.090 m 与 103 mm 特征相差 **13 mm** |

### P1-08 的 13 mm 差异（必须显式写出）

> P2/P4 目前使用的碰撞盒半高 **0.090 m** 与 103 mm 圆柱/圆特征相差 **13 mm**，
> 因此任何由代理盒推出的底盘高度都偏低 13 mm；`lift = 0.334 m` **不是**真实滚动高度，不得冻结。

## 5. P4 使用合同（摘要，全文见 JSON `p4_usage_contract`）

**今天可以参数化消费（必须带 candidate 标签 + 敏感性扫描）**

| id | 值 | 状态 |
|---|---|---|
| `wheel_spin_axis_direction` | CAD 系单位向量 ±x（仅方向） | PASS |
| `wheel_radius_parameter` | 0.103 m，标签 `candidate=None-verified` | CANDIDATE |
| `track_width_parameter` | 0.500004 m | CANDIDATE |
| `wheelbase_parameter` | 0.594 m | CANDIDATE |
| `wheel_prim_per_eid` | 4 个 `usd_path` | PASS（仅身份，不是 joint、不是电机通道） |
| `mirror_symmetry_check` | 镜像回归上界 2.0872 mm | PASS |

**禁止作为冻结真值**

1. 轮径 206 mm / 轮半径 0.103 m 当物理参数；
2. `lift = 0.334 m` 当真实滚动高度；
3. 0.090 m 碰撞盒半高当真实轮半高（差 13 mm）；
4. 轮质量、惯量张量、摩擦系数、胎面柔度；
5. 车架系轮距 / 轴距；
6. 任何由轴族聚类推出的 axle joint 轴或电机通道；
7. USD stage 声明的 `upAxis` 当重力方向；
8. 转向自由度存在性与减速比。

**P4 gate**：读到 verdict = UNKNOWN 的量却以冻结值请求 → fail closed；用于里程计的轮半径必须是显式参数 + 扫描；不得用诊断驱动冒充真实轮驱动（P3-04）。

## 6. 证据路径

| 文件 | 内容 |
|---|---|
| `outputs/simulation/rebot_b601_rs/morph_one/p1_wheel_identity/comparison.json` `.csv` | BREP 顶点 vs 曲面包络 vs 点云三口径 |
| `.../step_remeasure.json` `.csv` | STEP 解析圆/柱面、最大圆半径与圆心 |
| `.../usd_remeasure.json` `.csv` | USD Mesh 点数/面数/shell、径向统计、stage 单位与 upAxis |
| `.../cloud_geometry.json` `.csv` | 本轮新增：主轴、包络、圆拟合、镜像残差、网格分辨率 |
| `.../wheel_kinematics_extract.json` | 从主机 `wheel_kinematics.json` 摘录的 corners / symmetry / 四轮 part 记录（含来源 sha256） |
| `.../wheel_identity_contract.json` | **合同本体**（含每个量 verdict、P4 合同、UNKNOWN 与差异清单） |
| `.../MANIFEST.sha256` | 上述文件的 sha256 |

> `MANIFEST.sha256` 的哈希按 **LF 归一化**后的内容计算（本工作区 `core.autocrlf = true`：
> 直接对工作区文件做 `Get-FileHash` 会对含 CRLF 的文本文件得出不同字节哈希，从而与 git blob 不一致）。
> 用 LF 归一化后，旧的 `usd_remeasure.json` 条目 `93c33998…` 与 git blob 完全对上。

主机原始证据（只读、未修改，sha256 已记入合同 `inputs`）：`/home/dgut/runs/wheels/out/`
（`wheel_kinematics.json` 2848695 B `c2a942af…`、`wheel_parts.csv` 125457 B `033b8a5d…`、
`wheel_axis_candidates.csv` 275269 B `ad19b950…`、`wheel_axes_candidates.usda` 92304 B `1cf7794c…`）。

## 7. 命令与真实退出码

```bash
# 1) 主机上空跑点云测量（纯 CPU / pxr，只读 usda，未启动 Kit / 未用 GPU）
ssh dgut@172.31.68.251 '/home/T7/ojh/robot_sim/env_isaaclab/bin/python /tmp/p1_wheel_cloud_geometry.py \
  --usda /home/T7/dgut/runs/cad_dev/usd/M1-4WD4S_model2601_dev.usda \
  --mapping /tmp/_scratch_mapping.json \
  --out-json /tmp/p1_cloud_geometry.json --out-csv /tmp/p1_cloud_geometry.csv'
# CLOUD_EXIT=0

# 2) 生成合同
python scripts/simulation/p1_wheel_identity_contract.py
# build_exit=0

# 3) 形状/单位/反升格测试
python -m pytest tests/simulation/p1_wheel_identity_contract_test.py -q
# 7 passed in 2.17s -> pytest_exit=0

# 4) 合同陈旧性检查（提交的 JSON 必须等于重新生成的结果）
python scripts/simulation/p1_wheel_identity_contract.py --check
# OK: ... is up to date -> check_exit=0

# 5) 推送
git push -u origin p1/wheel-rolling-surface
# push_exit=0
```

> 说明：脚本自身在主机上以 `CLOUD_EXIT=0` 退出过两次失败后修好（先有对称配对把对角轮也当成镜像对、
> 后有 `report` 未定义两处 bug），最终一次为 0；失败那两次的 traceback 已在会话中定位，不是被掩盖的失败。

## 8. 仍然 UNKNOWN（按缺哪次测量列出）

| 量 | 缺什么测量 |
|---|---|
| wheel_radius | r=103 面的接地分类（轴向/径向接触带）+ 胎面材质 + 受载半径 |
| wheel_centre（车架系） | 车架基准（root frame 原点 + up 轴声明）与四轮 mount 变换 |
| track_width / wheelbase | 同上 + 定义轮距/轴距时的转向位姿 |
| wheel_mass / inertia / friction | 厂家参数或带标注的代理 |
| loaded_rolling_height | 受载半径 / 胎面柔度 |
| 转向轴线与范围 | joint 定义 + 任一转向位姿下的实测 |
