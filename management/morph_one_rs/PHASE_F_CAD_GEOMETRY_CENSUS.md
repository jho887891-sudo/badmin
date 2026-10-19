# Phase F — 可复现 CAD 几何普查与依赖图（Task 3）

本阶段回答三个问题，全部在**不修改源模型**的前提下回答：

1. 这份 115 MB 的 STEP 里到底有什么？（可复现普查）
2. 上一轮标记的 469 个"候选异常点"到底属于哪些实体？（依赖图归因，而不是猜）
3. 能否从整机里切出一个**自洽的子集**用于下游冒烟？（子集抽取）

硬约束：**不删除、不吸附、不修补任何点**。所谓 outlier 在本工具里始终只是
*候选*——它是文本统计的产物，不是几何有效性的证明。

工具：`scripts/simulation/step_geometry_census.py`（纯标准库，无 CAD 内核依赖）
测试：`tests/simulation/cad/test_step_geometry_census.py`（14 项，全部离线可跑）

---

## 1. 三种模式与复现

    # 普查：实体分类、单位、产品结构、点统计、候选 outlier
    python scripts/simulation/step_geometry_census.py --input M1-4WD4S_model2601.step \
      --mode census --output-dir out/census

    # 归因：候选 outlier -> 引用它的 solid（逆向依赖图 BFS）
    python scripts/simulation/step_geometry_census.py --input ... --mode attribute --output-dir out/census

    # 子集：只导出指定 solid + 产品/单位脚手架，条目表自动重写并自检悬空引用
    python scripts/simulation/step_geometry_census.py --input ... --mode subset --output-dir out/census \
      --solid-id 435700 --solid-id 759491 --subset-out out/smoke.step

实测资源占用（GPU 主机，115,496,178 B 输入）：

| 模式 | wall clock | 峰值 RSS |
|---|---|---|
| census | 21.37 s | 1.07 GiB |
| attribute（469 点全归因） | 24.71 s | 1.07 GiB |
| subset（单 solid） | 36.43 s | 1.20 GiB |

> 主机与他人 Isaac Sim 会话共用 GPU，这些数字只说明容量需求，不是性能结论。

---

## 2. 普查结果，以及与上一轮 inventory 的逐项对账

输入文件事实（可复核）：

| 项 | 值 |
|---|---|
| 大小 | 115,496,178 B |
| SHA-256 | `388cb3653a864066a35a54065f71938635c7e9a221cb4b897b31fbd44ff266c5` |
| schema | `CONFIG_CONTROL_DESIGN`（AP203） |
| 单位 | `SI_UNIT(.MILLI., .METRE.)`，3 个 SI 单位声明 |
| 产品 | 1 个 PRODUCT，名为 `M1-1.00V3`（PRODUCT_DEFINITION 名称是乱码字节） |
| 装配层级 | `NEXT_ASSEMBLY_USAGE_OCCURRENCE = 0`（单产品、无装配实例） |
| 实体总数 | **1,070,664** |
| 其中复杂实例 | **870** |
| CARTESIAN_POINT | 720,342 |

**对账**：与上一轮基于"逐行正则"的 `step_inventory.json` 相比，所有**简单实体类型计数
逐项完全一致**（差值全为 0），例如 CARTESIAN_POINT 720,342、DIRECTION 73,606、
ORIENTED_EDGE 72,170、EDGE_CURVE 36,085、AXIS2_PLACEMENT_3D 27,496、VERTEX_POINT 22,871、
ADVANCED_FACE 14,714、FACE_OUTER_BOUND 14,714、CIRCLE 13,324、PLANE 6,200、
CYLINDRICAL_SURFACE 5,692、CLOSED_SHELL 340、MANIFOLD_SOLID_BREP 281、SPHERICAL_SURFACE 100。

唯一差异是**复杂实例**：旧口径把它们算进"行数/被引类型"，新解析器把它们作为独立记录暴露，
于是实体数 1,070,664 比旧的行数 1,069,794 多 **870**，而这个 870 恰好等于复杂实例总数：

| 复杂实例组成 | 数量 |
|---|---|
| BOUNDED_CURVE + B_SPLINE_CURVE + B_SPLINE_CURVE_WITH_KNOTS + RATIONAL_B_SPLINE_CURVE ... | 580 |
| BOUNDED_SURFACE + B_SPLINE_SURFACE + ... + RATIONAL_B_SPLINE_SURFACE | 286 |
| GEOMETRIC_REPRESENTATION_CONTEXT + GLOBAL_UNCERTAINTY/UNIT_ASSIGNED_CONTEXT + ... | 1 |
| LENGTH_UNIT / NAMED_UNIT / SI_UNIT 等组合 | 3 |

即：差异**有解释**，不是解析丢失。（旧文档里 "398 CLOSED_SHELL" 的口径应为
340 个 CLOSED_SHELL + 58 个 ORIENTED_CLOSED_SHELL，本文按精确类型分别计数。）

---

## 3. 点统计与 469 个候选异常点

| 项 | 值 |
|---|---|
| 点总数 | 720,342 |
| 全模型包围盒 | 1,000,299.0 × 998,022.9 × 895,042.2 mm（≈ 1 km 量级） |
| 判据 | 超出 [p0.1, p99.9] 区间、再向两侧各放宽该区间跨度的 50%，任一轴越界即为候选 |
| 候选数 | **469** |
| 逐轴越界计数 | x 405 / y 333 / z 350（一个点可同时越界多轴） |

**重要且容易被误读**：文本级包围盒达到 ~1 km，但**这并不意味着存在 1 km 长的实体**。
实测（Task 2 的整机转换，见 Phase E）：USD 里 282 个网格的世界包围盒只有
**580.0 × 372.996 × 802.852 mm**，且**没有任何网格的任一跨度超过 1,000 mm**。
也就是说那些 ±499 m 的点是**曲面控制点/辅助点**（B_SPLINE 的控制点不必落在曲面上），
并不会生成几何。因此：

* "469 个候选"是**点层面的统计事实**，不是几何缺陷的证明；
* 也不能反过来当作"模型没问题"的证明——它只说明这些点没有materialize成可见几何。

下节的归因只回答一个更窄的问题：这些点**归谁所有**。

---

## 4. 依赖图归因：469 个候选点，100% 可归因

做法：对每个候选点沿**逆向**依赖链上溯（POINT → VERTEX_POINT → EDGE_CURVE → … →
CLOSED_SHELL → MANIFOLD_SOLID_BREP / BREP_WITH_VOIDS），因此"归因"是**证明所有权**，
不是空间邻近猜测。

结果：

| 项 | 值 |
|---|---|
| 已归因候选点 | 469 / 469（未归因 0） |
| 涉及 solid 数 | 17 |
| #894929 命中 | 113 点（24.1%） |
| 其余 16 个 solid | 34、34、34、34、26、26、26、26、22 … 逐级递减 |

**结构性异常（本轮最重要发现）**

| 事实 | 数值 |
|---|---|
| #894929 类型 | `BREP_WITH_VOIDS`（全模型**唯一**一个） |
| 其 BREP 闭包实体数 | 743,477 |
| 闭包含点数 | 613,330（占全模型点的 **85.1%**） |
| 其包围盒 | X [-498,959, 499,175]、Y [-499,028, 498,995]、Z [-432,111, 432,521] mm |
| 对比：其余 281 个 solid 中最大闭包 | 11,633 实体 |

也就是说：**一个实体吞掉了 85% 的点和 743k 个实体，并且横跨 ±499 m**；其余 281 个 solid
都正常。469 个候选点里有 113 个直接属于它，另外 356 个分散在 16 个正常尺寸的 solid 上
（这 16 个需要用 CAD 端确认是否只是"离群但合法"的曲面控制点）。

**交叉验证（关键）**：整机转换后，#894929 确实被转成了网格（282 = 281 + 1），而它在 USD 里的
实际尺度并不异常——所有网格跨度都 < 1,000 mm。因此对 #894929 的正确表述是：

* 它**结构上**异常（唯一 BREP_WITH_VOIDS、闭包 743,477 实体、含 85.1% 的点、点级包围盒横跨 ±499 m）；
* 但它**几何上**没有产生 1 km 级的曲面；
* "它是否该清理"仍是 CAD 侧的 UNKNOWN，本阶段既不删除也不裁剪。

不删点的理由不变：删除会破坏 BREP 拓扑；下游要做筛选，只能按 solid 白名单抽取子集，
或在 USD 层面处理。


---

## 5. 附带发现：模型高度重复，适合实例化

282 个 solid 只有 **40 种**不同闭包规模，重复非常密集：

| 闭包实体数 | solid 个数 |
|---|---|
| 530 | 106 |
| 520 | 22 |
| 2,467 | 18 |
| 162 / 292 / 720 / 974 | 各 8 |
| 1,108 | 6 |
| 11,633 | 4（#435700、#759491、#773650、#871375） |

那 4 个闭包 11,633（7,763 点）的 solid 是除 #894929 之外最大的零件，且**尺寸完全一致**，
形状上符合"同一部件出现四次"。但 AP203 里 `NEXT_ASSEMBLY_USAGE_OCCURRENCE = 0`、只有 1 个
PRODUCT，**没有任何命名/装配层级信息**，因此无法从该文件证明它们就是四个轮模块。
本阶段按 UNKNOWN 记录，不做语义断言。

---

## 6. 子集抽取：自洽性可自检

`extract_subset` 的规则：只保留指定 solid 的**完整**前向闭包（不裁剪其内部任何
实体），再加上产品/单位脚手架；representation 的**位置参数条目表**会被重写为只引用保留下来的
solid。并且每一步都会自检"是否存在指向已删除实体的引用"（dangling）。

两个必须踩过的坑（都已修复并被测试覆盖）：

1. 条目表是**位置参数**（TYPE('name',(#a,#b),#ctx)），文件里并没有 items= 这种写法；
2. 条目表里除了 solid 还可能有 `AXIS2_PLACEMENT_3D`（世界坐标系），它只被条目表引用。
   如果要求"所有引用都必须先存在"，representation 会被整段跳过，产出**无几何**的美观空壳
   （实测：556 B / 2 个 Xform / 0 网格）。修复后同一流程产出 3.86 MB / 5 网格 / 26,092 点。

实测子集（供 Task 2 冒烟）：

| 子集 | 大小 | 实体 | 悬空引用 | sha256 |
|---|---|---|---|---|
| 5 solids（435700 759491 773650 871375 453492） | 5,191,797 B | 47,008 | 0 | afe2555f…fab6ef4 |
| 单 solid（480214，退化薄片） | 5,458 B | 61 | 0 | 8f641193…43f000 |

---

## 7. 明确的 UNKNOWN（不得靠猜补齐）

1. #894929 是否属于"应清理的多余几何"——需 CAD 端（SolidWorks/HOOPS 诊断）确认。
2. 那 16 个含候选点的正常 solid 是否真的几何异常——本工具只能证明"点属于它"。
3. 任何零件/装配的**语义命名**（轮模块、舵机支架…）——文件里没有。
4. 供应商原始设计意图、公差、材料——AP203 文本中不存在。

---

## 8. 产物清单

| 路径 | 内容 |
|---|---|
| `scripts/simulation/step_geometry_census.py` | 普查 / 归因 / 子集抽取（本阶段新增） |
| `tests/simulation/cad/test_step_geometry_census.py` | 14 项离线单元测试 |
| `outputs/simulation/rebot_b601_rs/morph_one/cad_census/step_census.json` | 普查报告 |
| `outputs/simulation/rebot_b601_rs/morph_one/cad_census/cartesian_point_outliers.csv` | 469 个候选点（id + 坐标） |
| `outputs/simulation/rebot_b601_rs/morph_one/cad_census/step_attribute.json` | 归因结果与命中直方图 |
| `outputs/simulation/rebot_b601_rs/morph_one/cad_census/solid_sizes.json` | 282 个 solid 的闭包/点数规模 |
| `outputs/simulation/rebot_b601_rs/morph_one/cad_census/step_subset.json` | 子集抽取统计（含 dangling 自检） |

子集 STEP 与 USD 均留在主机 `/home/T7/dgut/runs/cad/`，不入库（资源不落本地原则）；
入库的是它们的 sha256 与统计，可据此复核。
