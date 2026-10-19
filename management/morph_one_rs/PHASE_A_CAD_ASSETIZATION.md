# Phase A - Morph One CAD assetization (facts, classification policy, blockers)

Status: **文件与结构已核实；几何导出与零件分类阻塞**（本机无 CAD 内核，Isaac Sim 的 CAD 转换器无脚本 API）。
因此本阶段**没有**产出 USD 几何、**没有**任何零件被标记为官方确认，四个转向/车轮模块**未识别**。

## 1. 文件核验（实际执行）

| 项目 | 值 |
| --- | --- |
| 路径 | /home/T7/dgut/robot_sim/assets/robots/badminton_robot/morph_one/cad/M1-4WD4S_model2601.step |
| 字节数 | 115,496,178 |
| SHA-256（重算） | 388cb3653a864066a35a54065f71938635c7e9a221cb4b897b31fbd44ff266c5 |
| 期望 SHA-256 | 388cb3653a864066a35a54065f71938635c7e9a221cb4b897b31fbd44ff266c5 |
| 一致性 | **MATCH** |

## 2. 结构与单位（文本层证据）

| 项目 | 值 | 依据 |
| --- | --- | --- |
| 格式 | ISO-10303-21, **STEP AP203 (CONFIG_CONTROL_DESIGN)** | FILE_DESCRIPTION / FILE_SCHEMA |
| 导出器 | SwSTEP 2.0 / SolidWorks 2022 | FILE_NAME |
| 原始文件名 | M1-1.00V3.STEP（导出时间 2026-03-06T09:12:09） | FILE_NAME |
| 长度单位 | **毫米**：LENGTH_UNIT + SI_UNIT ( .MILLI., .METRE. ) | 第 124426 行 |
| 角度/立体角单位 | rad / sr，无 CONVERSION_BASED_UNIT | 第 140870 / 720129 行 |
| 产品数 | PRODUCT = 1（'M1-1.00V3'），PRODUCT_DEFINITION = 1 | 第 396151 行 |
| **装配结构** | **NEXT_ASSEMBLY_USAGE_OCCURRENCE = 0** | 全文计数 |
| 形状表达 | 1 × ADVANCED_BREP_SHAPE_REPRESENTATION ('M1-1.00V3', (…281 个 body…)) | 第 963193 行 |
| PRODUCT_DEFINITION 名称 | 乱码（原始应为中文，编码损坏） | 第 709922 行 |

结论：这是**一个扁平的多实体零件文件**（单 product、无 assembly occurrence、无 part 实例、无关节关系），
与任务描述一致。281 个实体以同一个 shape representation 内的裸列表存在，**没有任何名称或分组**。

### 实体普查（全文逐行统计，共 1,069,794 条实体）

| 实体 | 数量 |
| --- | --- |
| CARTESIAN_POINT | 720,342 |
| DIRECTION | 73,606 |
| ORIENTED_EDGE | 72,170 |
| EDGE_CURVE | 36,085 |
| AXIS2_PLACEMENT_3D | 27,496 |
| VERTEX_POINT | 22,871 |
| LINE / VECTOR | 18,614 / 18,614 |
| EDGE_LOOP | 16,602 |
| ADVANCED_FACE | 14,714 |
| FACE_OUTER_BOUND | 14,714 |
| CIRCLE | 13,324 |
| PLANE | 6,200 |
| CYLINDRICAL_SURFACE | 5,692 |
| CLOSED_SHELL | 398 |
| **MANIFOLD_SOLID_BREP** | **281** |
| BREP_WITH_VOIDS | 1 |

## 3. 几何范围与异常坐标（可排除部分的依据）

统计基于全部 720,342 个 CARTESIAN_POINT 的坐标（未取整）：

| 轴 | min | p0.1% | 中位数 | p99.9% | max | 中心 99.8% 跨度 |
| --- | --- | --- | --- | --- | --- | --- |
| X | -500,149.49 | -287.50 | 169.98 | 287.60 | 500,149.49 | 575.10 mm |
| Y | -499,027.90 | -325.05 | -25.37 | 26.00 | 498,995.00 | 351.05 mm |
| Z | -447,521.10 | -383.00 | 57.32 | 401.15 | 447,521.10 | 784.15 mm |

判定规则（本轮自定义，写入脚本可复现）：某坐标超出 `p0.1%` / `p99.9%` 区间再各放宽 50% 跨度，即视为异常。

* 命中 **469 个点**（占 0.065%），其坐标量级在 **±250 m ~ ±500 m**（例：(-353753.64, 305969.28, 177089.32)）。
* 这些点属于明显非机械几何（CAD 导出残渣/构造几何），**必须从任何 PhysX 导入中排除**；
  因为无法逐实体归因（无内核、无名称），本轮只能给出**点级**排除清单，不能给出实体级排除清单。
* 中央 99.8% 盒体 575.1 × 351.0 × 784.1 mm：Z 向 784 mm 明显高于“底盘”量级，提示该文件可能含
  桅杆/立柱或非底盘零件 —— 这也正是**不能按外形猜分类**的理由。

原始数据：outputs/simulation/rebot_b601_rs/morph_one/step_inventory.json（5,064 B，含全量统计、离群点样本）。

## 4. 零件分类结果：全部 UNKNOWN

| 类别 | 数量 | 结论 | 依据 |
| --- | --- | --- | --- |
| 车架 (frame) | 0 已识别 / 282 未识别 | UNKNOWN | 无名称、无装配分组、无逐实体几何 |
| 四个轮模块 (wheel module ×4) | 0 已识别 | UNKNOWN | 同上；CIRCLE/CYLINDRICAL_SURFACE 计数不能反推半径与轴向 |
| 转向部分 (steering) | 0 已识别 | UNKNOWN | 同上 |
| 其他零件 | 282 | UNKNOWN | 同上 |

**判定口径**：只有同时具备（a）逐实体几何、（b）可区分的轴向/半径或装配分组，才可分类。当前两者皆无，
因此一律 UNKNOWN，不按外形猜测，也不把猜测写成官方验证。

## 5. 为什么没有 USD（转换阻塞的实测证据）

| 尝试 | 结果 |
| --- | --- |
| 本机 Python CAD 内核（OCP / OCC / cadquery / FreeCAD / gmsh） | 全部缺失（importlib 检查） |
| CLI 工具（FreeCADCmd / gmsh / assimp / meshlabserver / openscad） | 全部缺失 |
| trimesh | 已安装，但**不能读 STEP** |
| Isaac Sim `omni.kit.converter.cad-209.4.0` | 扩展存在且声明支持 Step/Iges，但 `find -name '*.py'` 为空 → 纯编译二进制、无脚本 API，只能走 UI，本轮无法在预算内脚本化 |

因此**连“静态 Visual CAD”也无法交付**：静态可视化同样需要三角化。按任务规则，这一条属于
“不可证明即停止”的范畴，所以我停在这里，没有伪造任何几何或关节。

## 6. 解除阻塞的最小条件（下一步）

1. 在 GPU 主机安装 `pythonocc-core`（OCP）或 FreeCAD CLI（任一即可）；或
2. 用 Isaac Sim 的 CAD Converter UI 人工导入一次并保存 USD（需人在场，本轮不可行）。

拿到逐实体网格后，分类算法（可复现）：按 entity → 网格包围盒 + 惯性主轴 + 圆柱面半径聚类；
四轮模块判据 = 存在两条以上同半径圆柱面且轴线近似平行于底盘 Y 轴、包围盒近似圆柱；
再把 469 个离群点所在实体整体剔除（届时可做实体级排除）。**在这些证据出现之前，Phase B/C 的
“基于已识别几何建立关节”无法成立**，详见 PHASE_BCD_BLOCKERS_AND_DESIGN.md。
