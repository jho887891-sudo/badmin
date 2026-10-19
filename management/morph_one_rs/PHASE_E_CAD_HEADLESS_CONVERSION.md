# Phase E — 无界面 CAD→USD 转换（Task 2）

本阶段只回答一个问题：**在没有任何 GUI 交互的情况下，能否把 Morph One 的 STEP 转成 USD，
并且让产物可核对？** 不涉及机械臂连接、不涉及 PPO、不涉及真机。

结论先行：**可以**，入口是一套 Kit 扩展 API；但本机 Isaac Sim 6.0.1 的 pip 安装里存在一个
**确定性动态库冲突缺陷**，不绕过它转换必定失败（exit 127）。本文给出缺陷的精确证据、绕过方法
和验证方法，所有数字都来自落盘文件，不来自日志里的 "success" 字样。

---

## 1. 转换入口（不是 GUI，也不是 HTTP）

| 项 | 事实 |
|---|---|
| 服务型扩展 | `omni.services.convert.cad-508.0.4` 已安装，官方文档明写 **"No public API"**，只能作为 Kit *service* 经 job 提交 |
| 可编程 API | `omni.kit.converter.hoops_core.get_instance()` → `HoopsCoreConverter.create_converter_task(input, output, file_format_args)` |
| 参数类型 | `file_format_args: Dict[str, str]`；键为 camelCase 选项名，值必须字符串化，布尔必须小写 |
| STEP 支持 | 是。filter 正则 `(.*\.STEP$)|(.*\.STP$)|(.*\.IGES$)|(.*\.IGS$)`（hoops_core/impl/filters.py:48-50） |
| 纯 venv 导入 | **不可行**。`omni.kit.converter.*` 只在运行的 Kit 应用内存在 |

证据来源：`omni/kit/converter/hoops_core/impl/helper.py`、
`omni/kit/converter/common/common.py:208-221`（`config_path_to_args` 负责把选项 JSON 变成 `Dict[str,str]`）。

因此脚本 `scripts/simulation/convert_step_to_usd.py` 的做法是：用 IsaacLab `AppLauncher`
起一个 **headless Kit**（无窗口；GLFW 初始化失败只告警，不影响转换），在应用内启用扩展并直接
调用该 API。

复现（远端 GPU 主机）：

    export PYTHONPATH=/home/T7/ojh/robot_sim/IsaacLab/source/isaaclab:/home/T7/ojh/robot_sim/IsaacLab/source/isaaclab_assets:/home/T7/ojh/robot_sim/IsaacLab/source/isaaclab_tasks
    python scripts/simulation/convert_step_to_usd.py --input file.step --output file.usda --report report.json --headless --kit_args "--enable omni.kit.converter.cad"

> 注意：venv 的 editable 安装指向已不存在的 `/home/T7/dgut/robot_sim/IsaacLab`，
> 直接 `import isaaclab` 会 `ModuleNotFoundError`。必须显式给 `PYTHONPATH`（源码实际在
> `/home/T7/ojh/robot_sim/IsaacLab`），或走 `isaaclab.sh -p`。

---

## 2. 本机缺陷：两份同名 `libTD_Root.so` 互相掩盖

第一次真实转换直接死在原生库加载阶段：

    python: symbol lookup error: .../omni.kit.converter.hoops_core-511.3.2.../bin/libTD_DbCore.so:
    undefined symbol: _ZN13OdConstStringC1EPKw
    Exit status 127

排查结论（可自行复核）：

1. 该符号由 `hoops_core-511.3.2/bin/libTD_Root.so` **定义**：
   `nm -D --defined-only .../libTD_Root.so` 中出现 `T _ZN13OdConstStringC1EPKw`。
2. `libTD_DbCore.so` 与 `libTD_Gi.so` 都**需要**它（`U _ZN13OdConstStringC1EPKw`）。
3. 两份库 SONAME 完全相同，都是 `libTD_Root.so`（`readelf -d`）。
4. 另一份在 `omni.kit.converter.dgn_core-512.3.3/bin/`，且**不含**该符号
   （对其 `nm -D --defined-only` 计数为 0）。
5. 两份库都只有 `RPATH=$ORIGIN`，没有反向隔离。

`dgn_core` 按字母序先于 `hoops_core` 进入加载器路径，同名 SONAME 让动态链接器复用了
**不含该符号**的那份，于是 `libTD_DbCore.so` 解析失败。

绕过方式（本轮实测有效，且只影响本进程）：

    export LD_PRELOAD=<site-packages>/isaacsim/extscache/omni.kit.converter.hoops_core-511.3.2+110.1.2.lx64.r.cp312.u7f4/bin/libTD_Root.so

这是**环境缺陷的绕过**，不是修复；升级 Isaac Sim 或改动扩展版本后必须重测。

---

## 3. 第二个陷阱：网格在抽象 prim 里，默认遍历看不见

转换成功后，`stage.Traverse()` 只看到 5 个 prim（2 Xform + Scope + Material + Shader）、
`mesh_count = 0`——但 USD 文件有 3.86 MB，且包围盒非空。用 `TraverseAll()` 才看到真相：

    smoke_medium (Xform, kind=assembly)
      Prototypes (class Scope, abstract=True)          <- 原型几何存放处
        tn__M1100V3_qAD/NONE/Mesh ... 共 5 个 Mesh
      Looks/Diffuse (Material) -> PreviewSurface (Shader)
      tn__M1100V3_qAD (Xform, instanceable=True)       <- 实例引用原型

即：**几何放在 `Usd class` 抽象 prim 里，由 instanceable prim 引用**。默认遍历谓词
（active ∧ defined ∧ ¬abstract）会整段跳过，所以"0 网格"是校验口径造成的假阴性，不是转换失败。

因此 `convert_step_to_usd.py` 的校验同时报告两组数：

* `mesh_count` / `total_mesh_points` / `total_mesh_faces`：`TraverseAll()` 统计的**原型**几何；
* `rendered_mesh_instances` / `rendered_mesh_points`：`Usd.PrimRange(root, Usd.TraverseInstanceProxies(Usd.PrimDefaultPredicate))`
  统计的**实际会被渲染**的实例数；
* `world_bbox`：`UsdGeom.BBoxCache` 对伪根求并，已包含实例变换。

---

## 4. 冒烟测试证据（真实子集，不是手写小文件）

冒烟输入由 Task 3 的依赖图工具从**同一个真实 STEP** 里切出，自洽且有自检：

| 项 | 值 |
|---|---|
| 子集 | 5 个 solid（435700 / 759491 / 773650 / 871375 / 453492），47,008 实体 |
| 子集文件 | `smoke_medium.step`，5,191,797 B，sha256 `afe2555f858eae97f734b012d25006c8bbbe4436aa61bae7a876b2fc8fab6ef4` |
| 悬空引用 | 0（`extract_subset` 自检） |
| 转换退出码 | 0，`status_error_code = 0` |
| 输出 | `smoke_medium.usda`，3,858,470 B |
| 网格 | 5 个 Mesh / 26,092 点 / 32,260 面 |
| 可渲染实例 | 5 |
| 包围盒 | 548.79 × 205.99 × 799.99 mm |
| 单位 / 朝向 | `metersPerUnit = 0.001`（mm→m 正确），`upAxis = Z` |
| 结论串 | `converted and verified: 5 prototype meshes / 26092 points, 5 rendered mesh instances, bbox size [548.79, 205.99, 799.99]` |

落盘证据：`outputs/simulation/rebot_b601_rs/morph_one/cad_conversion/smoke_medium.json`
（含输入/输出 sha256、扩展启用状态、`file_format_args`、转换状态、USD 结构化校验）。

### 整机转换（115 MB，本轮已完成）

| 项 | 值 |
|---|---|
| 输入 | `M1-4WD4S_model2601.step`，115,496,178 B，sha256 `388cb365…f266c5` |
| 转换退出码 | 0，`status_error_code = 0` |
| wall clock / 峰值 RSS | 53.86 s / 4.28 GiB |
| 输出 | `M1-4WD4S_model2601.usda`，58,658,300 B |
| prim / 网格 | 574 prim，**282** 个 Mesh（= 281 MANIFOLD_SOLID_BREP + 1 BREP_WITH_VOIDS，与 STEP 实体计数一致） |
| 网格顶点 | 455,255 |
| 可渲染实例 | 282（1 个 instanceable 根） |
| 世界包围盒 | **580.0 × 372.996 × 802.852 mm** |
| 单位 / 朝向 | `metersPerUnit = 0.001`，`upAxis = Z` |
| 超过 1 m 的网格 | **0** |

网格数与 STEP 里 282 个 solid **逐一对应**，这是"没有静默丢件"的最有力证据；
包围盒量级（0.58 × 0.37 × 0.80 m）也与文本统计的中央区间（575.1 × 351.0 × 784.1 mm）
相互独立地吻合，说明单位换算与缩放正确。

另有一个**退化样本**：单 solid 480214 子集转出 556 B、2 个 Xform、0 网格。该 solid 只有 2 个面、
14 个点，属薄片/退化体，不能据此说转换器有问题，但也**不能**拿它当"成功"证据——这正是本文
坚持用 5 solid 子集做冒烟的原因。

---

## 5. 已验证 / 未验证的边界

**已验证**：headless 转换链路（进程内 API）、整机 115 MB 转换、产物结构化核对
（见第 4 节表格，证据 `cad_conversion/full_step.json` 与 `full_step_usd_geometry.json`）。

**未验证**：

1. **Isaac Sim 内的实际可用性**：未在 Isaac Lab 场景里引用该 USD，未验证渲染、碰撞、质量/惯量。
3. **官方 service 路线**（`omni.services.convert.cad` + `job.cad_converter.kit`）未端到端验证；
   本轮只验证了进程内 API 路线。
4. **内存/时间不是性能结论**：主机与他人 Isaac Sim 会话共用 GPU，wall-clock 仅供容量参考。

---

## 6. 产物清单

| 路径 | 内容 |
|---|---|
| `scripts/simulation/convert_step_to_usd.py` | headless 转换 + 结构化自校验（本阶段新增） |
| `outputs/simulation/rebot_b601_rs/morph_one/cad_conversion/smoke_medium.json` | 冒烟转换证据 |
| `outputs/simulation/rebot_b601_rs/morph_one/cad_conversion/smoke_small.json` | 退化样本证据（0 网格，如实记录） |
| `outputs/simulation/rebot_b601_rs/morph_one/cad_conversion/full_step.json` | 整机转换证据（sha256、状态、USD 校验） |
| `outputs/simulation/rebot_b601_rs/morph_one/cad_conversion/full_step_usd_geometry.json` | 整机 USD 几何统计（282 网格、包围盒、>1 m 网格数=0） |

生成的 USD 留在 GPU 主机 `/home/T7/dgut/runs/cad/usd/`，不入库（资源不落本地原则）。
