# Phase I — 米制场景装配与 STEP→USD 零件映射

- 基线 SHA：2270cb3535b9b9473c8c99a620dad93791a56770（审计 PR #6）
- 开发分支：dev/morph-one-metric-scene-and-mapping
- 工作区：E:/具身智能/badmin_wt_dev_2270cb3
- 未修改任何 ChatGPT 源码；本轮只新增 3 个模块、3 个测试文件与证据。

---

## P0 回归测试

    pytest tests/simulation/cad tests/simulation/rebot_b601_rs \
           tests/simulation/rebot_b601dm/test_deliverables_index.py -q
    1 failed, 142 passed, 3 skipped in 5.98s

- 工作区污染：仍为**零**。运行前后 git status --porcelain 去掉我自己的快照文件后完全一致
  （快照见 dev_2270cb3/p0/git_status_before.txt / after.txt）。
- 唯一失败项依旧是审计侧回归：
  tests/simulation/cad/test_solid_geometry.py::test_build_report_aggregates_groups_and_caveats
  断言 len(caveats) == 3，实际 4。成因 commit 85e204a 只改了 step_solid_geometry.py。
  我未修改该源码，也未修改该断言。
- 3 skipped 是我新增的 pxr 依赖测试在系统解释器上跳过；在 Isaac venv 上
  11 passed（含这 3 项），日志见 dev_2270cb3/p4/... 与下节说明。

---

## P1 STEP 追溯复验（attribute + trace）

审计版 step_geometry_census.py（sha256 55358c8d...）全量重跑：

| 项 | 结果 |
|---|---|
| attribute | exit 0，25.24 s，1.12 GiB |
| trace | exit 0，26.44 s，1.12 GiB |
| 候选点 | 469 |
| 归因 | 469 / 469，未归因 0 |
| 涉及 solid | 17（与历史一致） |
| kind | auxiliary_geometry 469 |
| 直接父类型 | LINE 419 / AXIS2_PLACEMENT_3D 50 |

**新旧差异：无。** 三份产物与历史版本在 git blob（LF）层面逐字节相同：

    step_attribute.json          a849792c56c1866c...
    step_trace.json              6d502443b38fcb01...
    cartesian_point_outliers.csv eb7b8addacb96783...

因此不需要列出"原始/修正"的 Entity ID 对照：没有任何一条归因发生变化。

---

## P2 CAD 转换复验（新输出路径 + 四条验收）

输出：/home/T7/dgut/runs/cad_dev/usd/M1-4WD4S_model2601_dev.usda（此前不存在）

| 项 | 值 |
|---|---|
| 退出码 | 0 |
| verification_checks | 5/5 true（meshes=282、rendered=282、mpu=0.001、up=Z、非空网格） |
| 网格 / 顶点 / 实例 | 282 / 455,255 / 282 |
| 世界包围盒 | 580.0 × 372.996 × 802.852（mm 单位） |
| metersPerUnit / upAxis | 0.001 / Z |
| 输出大小 | 58,659,744 B |
| SHA-256 | 3b9962913ec517637df2a57df58acb5650fe995b437088539e26f00912af74b7 |
| 用时 / 峰值 RSS | 46.7 s / 4.81 GiB |

第三次同参数转换得到第三个不同 sha256（前三轮：58,658,300 / 58,660,611 / 58,659,744 B），
再次确认**转换器输出不是字节级可复现**；验收应固定在计数/单位/朝向/包围盒。

---

## P3 米制场景装配验证（本轮新功能）

新增：
- scripts/simulation/morph_one_metric_scene.py —— 建场景与测量（纯函数 + pxr 内部导入）
- scripts/simulation/run_metric_scene_validation.py —— Isaac Sim headless 执行 + 报告
- tests/simulation/cad/test_morph_one_metric_scene.py —— 11 项测试（8 项无 pxr，3 项需 pxr）

测量方法（不是理论换算）：遍历 **instance proxy**，把每个 Mesh 的每个点乘以其
local-to-world 矩阵，得到世界包围盒；同时用 UsdGeom.BBoxCache 作第二意见，
两者必须一致（报告里 bbox_methods_agree）。

| 用例 | 场景 mpu | 引用缩放 | 组合对角 | 世界包围盒 | 物理尺寸 (m) | 匹配期望 |
|---|---|---|---|---|---|---|
| A 不缩放 | 1.0 | 无 | 1.0,1.0,1.0 | 580.0 × 372.996 × 802.852 | 580.0 × 372.996 × 802.852 | 否 |
| B 缩放 0.001 | 1.0 | 0.001 | 0.001,0.001,0.001 | 0.580000028 × 0.372995715 × 0.802851662 | **0.5800 × 0.3730 × 0.8029** | **是** |

- B/A 比值：0.001（三轴一致）；与期望值偏差约 3e-8 m。
- 两用例各 282 个 mesh 实例，PhysX Collision/RigidBody/Mass/Articulation 全为 **0**。
- 逐 prim 采样里可见 A 与 B 的世界尺寸恰好相差 1000 倍（例：0.222450256 → 0.00022245），
  证明读的是真实世界坐标而非理论值。
- verdict.ok = true。
- 该 0.580 × 0.373 × 0.803 m 是**转换几何外包围盒**，不是确认过的车架尺寸。
- 未添加碰撞、质量、惯量或任何关节/机械臂约束。

截图：**不可用**，原因是 headless 应用没有 viewport（ModuleNotFoundError: omni.kit.viewport），
已在报告 JSON 的 screenshots 字段如实记录，未伪造图片。

---

## P4 STEP → USD 零件映射（本轮新功能）

新增脚本 scripts/simulation/map_step_solids_to_usd_meshes.py，用两个**相互独立**的信号：

1. **名字谱系**（CAD 结构，不是几何）：转换器把 STEP 名字写进 prim 名。
   归一化必须处理三件事：标点/中文消失、GB\u2571T 被重新编码成 GBuT（多出小写字母，
   故先删小写再删符号）、同名件被加去重后缀（NONE / NONE_1 ... / _<hash>）。
2. **几何夹逼**：STEP 侧给出下界（BREP 顶点，必定在曲面上）与上界（闭包内全部
   Cartesian 点，并用每个 CIRCLE 的半径外扩）；正确配对必须满足
   `lower ⊆ mesh ⊆ upper`（世界坐标）。容差 2.5 mm 来自 tessLOD=2 的弦高误差，
   与零件尺寸无关（一个 8 mm 螺钉的网格就可能差 2 mm）。

### 结果（真实数据）

| 项 | 值 |
|---|---|
| STEP solid / USD mesh | 282 / 282 |
| 配对 | **224** |
| VERIFIED | **199** |
| CANDIDATE | **25** |
| UNKNOWN（两侧各 58 未匹配） | 58 |
| 名字谱系证据 | 199 对 |
| 仅几何证据（NONE 组） | 25 对 |
| bijection_proven | **false** |

按装配根统计已配对：M1.2-00V3 底盘模块 138、M1.1-00V3 主体模块 61、NONE 25。

**为什么 282 == 282 不构成证明**（工具明确输出 bijection_proven = false）：

- 名字键只有 240 个不同值；**35 个 solid 的名字字面就是 NONE**，USD 侧同样 35 个，
  只能靠几何配对，故最高只能到 CANDIDATE。
- 另有 4 组「3 solid / 3 mesh 同名」（左右立板等被中文区分，ASCII 归一后塌陷），
  配对依赖几何，标 CANDIDATE。
- 58 个 UNKNOWN 全部是「名字键在两侧都存在、但几何夹逼不通过或被抢走」，
  报告里逐条给出 closest_key_ratio = 1.0 与原因，不做猜测性配对。

### 附带的可量化发现

- 60 对中顶点盒与网格盒差异显著（>2.5 mm）的 14 对里，最大的是舵轮轴连接板
  （顶点盒比网格小 16.2 mm，因为 36 mm 圆孔/圆弧鼓出）；另有 4 个无名件差 8.95 mm。
  这解释了为什么"用顶点包围盒直接比对尺寸"会失败，也再次印证圆弧必须按半径外扩。
- 位置误差（配对件的最小角点与世界坐标）中位数 1.8 mm，203/235 在 3 mm 内，
  无一超过 50 mm —— 说明转换器的世界放置是可信的，问题只在尺寸下界。

### 未做的事（不伪造）

- 没有把任何零件标注成"车轮/轮轴/车架尺寸"；仅按装配名字给出结构性归属。
- 没有从 4 个边界相同的无名实体推断四个轮轴（它们的形状指纹互不相同，已在 Phase G 记录）。
- 未测量轮半径、轴距、质量、惯量。

---

## 产物与复现

证据目录（全部文本，已入库）：
outputs/simulation/rebot_b601_rs/morph_one/dev_2270cb3/
  p0/   git status 前后快照 + pytest_full.log
  p1/   attribute/trace 产物与运行日志（与历史逐字节相同）
  p2/   convert_report.json（含 sha256 与 5 项检查）+ convert.log
  p3/   metric_scene_report.json + metric_scene.log + 两个场景 usda
  p4/   step_to_usd_mapping.json / .csv + p4_mapping.log
  MANIFEST.sha256

资源（留 GPU 主机，不入库）：
  /home/T7/dgut/runs/cad_dev/usd/M1-4WD4S_model2601_dev.usda  58,659,744 B
  sha256 3b9962913ec517637df2a57df58acb5650fe995b437088539e26f00912af74b7

复现命令（PYTHONPATH 指向 /home/T7/ojh/robot_sim/IsaacLab/source/*）：

    # P1
    python scripts/simulation/step_geometry_census.py --input <step> --mode attribute --output-dir <out>
    python scripts/simulation/step_geometry_census.py --input <step> --mode trace     --output-dir <out>
    # P2（需先 LD_PRELOAD hoops 的 libTD_Root.so）
    python scripts/simulation/convert_step_to_usd.py --input <step> --output <new.usda> \
      --report <json> --headless --max-seconds 7200 \
      --expect-meshes 282 --expect-rendered-instances 282 \
      --expect-meters-per-unit 0.001 --expect-up-axis Z \
      --kit_args "--enable omni.kit.converter.cad"
    # P3
    python scripts/simulation/run_metric_scene_validation.py --visual-usd <usd> \
      --output-dir <dir> --report <json> --screenshot --headless
    # P4
    python scripts/simulation/map_step_solids_to_usd_meshes.py --step <step> --usd <usd> \
      --output-dir <dir> --report <json>

---

## 仍 UNKNOWN / 阻塞

1. 任务一那条审计侧断言失败需要审计方修复（或明确允许我调整断言）。
2. 58 个实体未能配对，其中 10 个是 NONE 无名件、12 个来自 4 组同名件；其余为几何夹逼
   不通过。逐条证据在 p4 JSON 的 unmatched_solids / rejected_pairs 里。
3. 米制场景只验证引用、单位、坐标；未加碰撞/质量/关节，因此"能否用于物理仿真"仍未验证。
4. 0.580 × 0.373 × 0.803 m 是几何外包围盒，不是车架尺寸；转向/驱动参数仍全部 UNKNOWN。
5. 无 pxr 的系统解释器上会跳过 3 项测试属预期；需在 Isaac venv 上跑全量。

## 建议 ChatGPT 审计的位置

- scripts/simulation/map_step_solids_to_usd_meshes.py 的 sandwich / pair_cost / 分级规则
  （尤其 2.5 mm 容差与"同名组内按几何选伴"是否会掩盖错配）。
- scripts/simulation/morph_one_metric_scene.py 的 world_bbox_from_points（instance proxy 遍历）。
- tests/simulation/cad/test_step_to_usd_mapping.py 的 11 项断言是否足够严格。
