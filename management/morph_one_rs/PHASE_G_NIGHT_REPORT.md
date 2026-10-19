# Phase G — 夜间自主开发报告（P0–P4）

分支：dev/morph-one-cad-pipeline　　基线：c5469e8bd8d19b76c9fef21f172f66ab0b8369aa
worktree：E:/具身智能/badmin_wt_cad_pipeline
本轮所有结论都来自真实运行（pytest / 115 MB STEP / 58 MB USD），没有一条来自"代码看起来对"。

---

## P0 回归验证

命令与结果：

    pytest tests/simulation/rebot_b601_rs -q
    89 passed in 2.00s

先前四项"证据漂移"失败已全部消失，无 traceback 需要上交。
未修改任何验收代码：本轮对 tests/simulation/rebot_b601_rs 的改动为 0（只新增
tests/simulation/cad/ 下的新测试）。

全量回归（含新增测试）：

    pytest tests/simulation -q
    265 passed, 46 skipped

---

## P1 CAD 转换（延续 + 本轮实体级审计）

### P1.1 文件核对

    sha256sum .../M1-4WD4S_model2601.step
    388cb3653a864066a35a54065f71938635c7e9a221cb4b897b31fbd44ff266c5   <- 与预期一致
    115,496,178 B

### P1.2 路线与接口（含本轮新增的 extension.toml 证据）

| 检查项 | 结论 |
|---|---|
| Kit CLI | venv 提供 isaacsim 控制台入口；app 配置 isaacsim.exp.full.kit 第 135 行启用 omni.kit.converter.cad |
| omni.services.convert.cad | 已安装 508.0.4，但官方 config 明写 "No public API"（只能当 service 用） |
| 可编程接口 | omni.kit.converter.hoops_core.get_instance().create_converter_task(input, output, Dict[str,str]) |
| 无界面 STEP 支持 | 是（filter 正则含 .STEP/.STP/.IGES/.IGS），本轮实跑验证 |
| 小 STEP 冒烟 | 已做：5-solid 子集 5,191,797 B -> 0 错误、5 网格 |
| 原始 Morph One | 已做：exit 0，见下 |

关键新证据（缺陷根因，不是我猜的）：

    omni.kit.converter.cad-209.4.0/config/extension.toml
    [dependencies]
      "omni.kit.converter.hoops_core" = { version = "511.3.2", exact = true }
      "omni.kit.converter.dgn_core"   = { version = "512.3.3", exact = true }

也就是说 CAD bundle 用 exact 版本同时锁定两套基于同名 libTD_Root.so 的核，
于是进程内动态链接必然发生 SONAME 抢占：dgn_core 那份不含
_ZN13OdConstStringC1EPKw，hoops 的 libTD_DbCore.so 解析失败 -> exit 127。
绕过方式（本轮沿用并再次验证）：

    export LD_PRELOAD=<site-packages>/isaacsim/extscache/omni.kit.converter.hoops_core-511.3.2+110.1.2.lx64.r.cp312.u7f4/bin/libTD_Root.so

这是环境绕过，不是修复；换 Isaac Sim 版本必须重测。

### P1.3 整机转换结果（真实产出）

| 项 | 值 |
|---|---|
| 退出码 | 0，status_error_code = 0 |
| wall clock / 峰值 RSS | 53.86 s / 4.28 GiB |
| USD | M1-4WD4S_model2601.usda，58,658,300 B |
| prim / mesh | 574 / 282（= 281 MANIFOLD_SOLID_BREP + 1 BREP_WITH_VOIDS，逐一对应，无静默丢件） |
| 顶点 | 455,255 |
| 可渲染实例 | 282（1 个 instanceable 根，569 个 abstract 原型 prim） |
| 世界包围盒 | 580.0 × 372.996 × 802.852 mm |
| 单位 / 朝向 | metersPerUnit = 0.001（mm->m 正确）、upAxis = Z、kilogramsPerUnit = 1 |

### P1.4 本轮新增：极端几何与 PhysX 侵入审计

由 scripts/simulation/analyze_usd_geometry.py 在真实 USD 上运行得到（不是手抄）：

| 审计项 | 值 | 含义 |
|---|---|---|
| max abs local coordinate | 400.0 mm | 局部坐标最大值，远小于 STEP 里 ±499 m 的离群点 |
| mesh_local_extent_max | [580.0, 158.0, 800.0] mm | 最大网格就是主机体 |
| mesh_local_extent_min | [0.222, 0.004, 0.104] mm | 最小网格仍非退化 |
| meshes_with_local_extent_over_1000 | 0 | 没有任何 1 m 以上的网格 |
| PhysX CollisionAPI / RigidBodyAPI / MassAPI / ArticulationRootAPI | 0 / 0 / 0 / 0 | 转换结果不含任何物理 schema，异常几何未进入 PhysX |

结论：STEP 里 469 个统计离群点**没有**变成极端几何，也**不可能**进入 PhysX
（该 USD 目前纯渲染，没有 collider、没有质量）。同时说明：要用于物理仿真，
后续必须显式添加碰撞体——这是**未完成**项，不在本轮范围。

469 个离群点一个都没删（本轮只新增分析，未写回任何 STEP/USD）。

---

## P2 备用转换路线（未启动，理由记录）

P1 已成功产出真实 USD，因此按任务书条件（"如果 Isaac Sim CAD Converter 无法使用"）
不启动 FreeCAD/OCP 路线，也未安装任何新内核。

---

## P3 STEP 几何分析工具

### P3.1 新增能力一：点级全链路追溯（trace 模式）

    python scripts/simulation/step_geometry_census.py --input <step> --mode trace

对 469 个候选离群点逐个做逆向 BFS，输出"点到 solid"的最短路径与祖先角色分类。

真实运行结果（25.56 s，峰值 1.12 GiB）：

| 分类 | 数量 |
|---|---|
| auxiliary_geometry（辅助几何） | 469 / 469 |
| brep_vertex（真实顶点） | 0 |
| bspline_control_point（样条控制点） | 0 |
| unknown | 0 |

直接父类型分布：

| 直接父类型 | 点数 |
|---|---|
| LINE | 419 |
| AXIS2_PLACEMENT_3D | 50 |

**这一条是本轮最重要的纠错**：上一轮我推测离群点是"B 样条控制点"，实测**否**。
它们是 LINE 的原点与 AXIS2_PLACEMENT_3D 的原点。STEP 的 LINE(pnt, dir) 只需要
"直线上的任意一点"作原点，把原点放在 900 m 外完全合法，真正的线段由两端
VERTEX_POINT 约束。所以：

* 统计离群（点坐标）≠ 几何异常（曲面/实体尺寸），两者被实证区分开了；
* 任何"删除 469 个点"的操作都既无必要也有害（会改动 LINE/PLACEMENT 记录）。

样例路径（真实输出）：

    CARTESIAN_POINT -> AXIS2_PLACEMENT_3D -> PLANE -> ADVANCED_FACE -> CLOSED_SHELL -> BREP_WITH_VOIDS
    CARTESIAN_POINT -> LINE -> EDGE_CURVE -> ORIENTED_EDGE -> EDGE_LOOP -> FACE_OUTER_BOUND -> ... -> MANIFOLD_SOLID_BREP

即 P3 目标 1（solid -> shell -> face -> edge -> vertex/point 可追溯）与目标 2
（异常坐标属于哪个实体/面/边）均已实现并可复现。

### P3.2 新增能力二：逐实体几何（step_solid_geometry.py）

    python scripts/simulation/step_solid_geometry.py --input <step> --output-dir <out>

对每个 solid 提取：解码后的零件名、装配分组、闭包规模、**仅用 BREP 顶点**测量的
包围盒、圆特征数量与半径统计、圆轴方向、形状提示（ratio 规则）、以及
**平移无关的形状指纹**（用于证明"是否同一零件的副本"，而不是靠实体数量猜）。

真实运行结果（25.26 s，峰值 1.12 GiB）：

| 项 | 值 |
|---|---|
| solids | 282 |
| 有名实体 / 无名实体 | 247 / 35 |
| 全模型顶点包围盒 | 580.0 × 360.391 × 802.852 mm |
| 重复形状组（>=2 副本） | 54 组 |
| 顶层装配根 | M1.2-00V3 底盘模块-1: 184 个；M1.1-00V3 主体模块-1: 63 个；UNKNOWN: 35 个 |

**交叉验证（很强的一条）**：顶点法得到 580.0 × 360.391 × 802.852 mm，CAD 内核
网格法得到 580.0 × 372.996 × 802.852 mm——X 与 Z **完全一致**，Y 差 12.6 mm 且方向
符合"顶点包围盒是下界"的预期。两条完全独立的管线互相印证。

分组（按装配路径第 2 段）：

| part_group | solids | 顶点包围盒（mm） |
|---|---|---|
| M1.2.2-00V3 舵轮前组件-1 | 92 | 547.3 × 203.4 × 178.0 |
| M1.2.2-00V3 舵轮前组件-2 | 92 | 547.3 × 203.4 × 178.0 |
| M1.1-00V3 主体模块-1 | 63 | 395.7 × 132.8 × 800.9 |
| UNKNOWN（无名字） | 35 | 580.0 × 360.4 × 800.0 |

零件命名（GBK 解码后）示例：轮毂电机固定板、舵轮轴连接板、左/右舵轮立板、
舵轮模组挡板、减震器弹簧、GB/T 70.1-2008 内六角圆柱头螺钉 等。

### P3.3 明确未能证明的事（UNKNOWN）

1. 四个 11,633 闭包的大实体（#435700 / #759491 / #773650 / #871375）：
   拓扑计数完全相同（250 顶点、136 圆）、圆半径统计完全相同（min 6.5 / median 6.5 /
   max 103.0）、顶点包围盒完全相同（48.7 × 180.8 × 157.8 mm），但**形状指纹四者互异**，
   因此**不能**声明它们是同一零件的四份副本（可能是镜像/不同朝向）。名字为 NONE。
   最大圆特征半径 103.00 mm（直径 206 mm）只出现在这 4 个实体上——这是**测量事实**，
   但**没有**把它叫做轮半径，也没有据此推导轮距/轴距。
2. #894929（唯一 BREP_WITH_VOIDS，743,477 闭包实体、613,330 点）：顶点包围盒
   580 × 158 × 800 mm、3,830 个圆特征、最大半径 40 mm、slab_like、名字 NONE。
   它究竟是"应清理的多余几何"还是"合并导入的主体"，仍须 CAD 侧确认。
3. 哪些零件是左/右、前/后：名字里有"左舵轮立板/右舵轮立板"等字样，但装配里
   只有两个舵轮前组件（-1 / -2），无法从文件证明其物理前后位置。
4. 轮半径、转向轴高度、轴距、质量、惯量：**一律未推导**（文件里没有这些量，
   且任务禁止伪造）。

---

## P4 提交

新增源代码：

| 路径 | 说明 |
|---|---|
| scripts/simulation/step_geometry_census.py | 普查 / 归因 / 子集 / **trace 追溯** / GBK 解码 |
| scripts/simulation/step_solid_geometry.py | 逐实体几何、分组、重复形状指纹 |
| scripts/simulation/analyze_usd_geometry.py | USD 单位/网格/极端几何/PhysX 审计 |
| tests/simulation/cad/test_step_geometry_census.py | 20 项 |
| tests/simulation/cad/test_solid_geometry.py | 8 项 |

证据（文本，入库）：

| 路径 | 内容 |
|---|---|
| outputs/.../morph_one/cad_census/step_census.json | 普查（1,070,664 实体 / 720,342 点 / 469 候选） |
| outputs/.../morph_one/cad_census/cartesian_point_outliers.csv | 469 个候选点 |
| outputs/.../morph_one/cad_census/step_attribute.json | 候选点 -> solid 归因 |
| outputs/.../morph_one/cad_census/step_trace.json | 候选点 -> 面/边/顶点 + 角色分类 |
| outputs/.../morph_one/cad_census/step_solid_geometry.json/.csv | 282 个 solid 的几何事实 |
| outputs/.../morph_one/cad_census/solid_names.json | 282 个零件名（GBK 已解码） |
| outputs/.../morph_one/cad_census/solid_sizes.json | 282 个 solid 的闭包规模 |
| outputs/.../morph_one/cad_census/step_subset.json | 子集抽取统计（悬空引用=0） |
| outputs/.../morph_one/cad_conversion/full_step.json | 整机转换证据（sha256/状态/校验） |
| outputs/.../morph_one/cad_conversion/full_step_usd_geometry.json | 整机 USD 几何统计 |
| outputs/.../morph_one/cad_conversion/usd_audit.json | 极端几何与 PhysX 审计 |
| outputs/.../morph_one/cad_conversion/smoke_medium.json | 冒烟转换证据 |
| outputs/.../morph_one/cad_conversion/smoke_small.json | 退化样本（0 网格，如实记录） |

资源（不入库，留在 GPU 主机）：

    /home/T7/dgut/runs/cad/usd/M1-4WD4S_model2601.usda      58,658,300 B
    /home/T7/dgut/runs/cad/usd/smoke_medium.usda             3,858,470 B
    /home/T7/dgut/runs/cad/census/smoke_medium.step          5,191,797 B  sha256 afe2555f...fab6ef4
    /home/T7/dgut/runs/cad/census/smoke_small.step               5,458 B  sha256 8f641193...43f000

复现命令（在 GPU 主机，PYTHONPATH 指向 /home/T7/ojh/robot_sim/IsaacLab/source/*）：

    # 普查 / 归因 / 追溯 / 子集
    python scripts/simulation/step_geometry_census.py --input <step> --mode census    --output-dir <out>
    python scripts/simulation/step_geometry_census.py --input <step> --mode attribute --output-dir <out>
    python scripts/simulation/step_geometry_census.py --input <step> --mode trace     --output-dir <out>
    python scripts/simulation/step_geometry_census.py --input <step> --mode subset    --output-dir <out> --pick-largest-solids 1

    # 逐实体几何
    python scripts/simulation/step_solid_geometry.py --input <step> --output-dir <out>

    # 转换（必须先 LD_PRELOAD，见 P1.2）
    LD_PRELOAD=<hoops_core>/bin/libTD_Root.so python scripts/simulation/convert_step_to_usd.py \
      --input <step> --output <usd> --report <json> --headless --kit_args "--enable omni.kit.converter.cad"

    # USD 审计
    python scripts/simulation/analyze_usd_geometry.py --usd <usd> --report <json>

---

## 本轮失败与阻塞

| 问题 | 处理 |
|---|---|
| trace 模式首次运行 AttributeError（缺 --trace-limit 参数） | 工具报错不静默；补参数后重跑成功 |
| 顶点路径枚举在共享点上可能爆炸 | 改为 BFS 前驱重建最短路径，469 点 25 s 跑完 |
| circle_facts 遇到退化 DIRECTION 会漏掉轴 | 改为遍历 placement 的所有方向引用，跳过退化值 |
| "四个大实体是否同一零件" | 指纹显示互异 -> 按 UNKNOWN 记录，不编故事 |

无致命阻塞。P1/P3 均产出可复核证据。

---

## 下一步建议（交给 ChatGPT 审计）

1. 审计 P1.2 的依赖冲突结论（exact 锁版 -> 同名库抢占），确认是否应改为只启用 hoops 子集
   或向 NVIDIA 反馈。
2. 审计 P3 的 trace 语义：是否同意"LINE 原点/placement 原点 => 统计离群不等于几何异常"。
3. 决定 #894929 与 4 个大 NONE 实体的处置口径（清理 / 保留 / 交 CAD 侧）。
4. 物理可用性仍是空白：需要在 USD 上加碰撞体并验证，本轮未做、也未声称做过。
5. 底盘接口参数（轮半径/转向轴高/轮距/轴距/质量/惯量）仍须测量或用厂商数据，
   本轮只提供了"可测的对象与分组"，没有给出任何数值结论。
