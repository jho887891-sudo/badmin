# Phase H — 审计 SHA 2270cb3 复验报告

- 审计 SHA：2270cb3535b9b9473c8c99a620dad93791a56770（分支 review/morph-one-cad-pipeline-audit-20261020）
- 复验工作区：E:/具身智能/badmin_wt_2270cb3
- 证据分支：evidence/morph-one-cad-revalidation-2270cb3
- 我未修改审计源码；本轮只新增 3 个独立验证程序与 3 项测试。

---

## 任务一 回归测试

命令与结果：

    pytest tests/simulation/cad tests/simulation/rebot_b601_rs \
           tests/simulation/rebot_b601dm/test_deliverables_index.py -q
    123 passed, 1 failed in 9.46s

### 1.1 工作区污染：已修复（验证通过）

运行前后各取一次 git status --porcelain：

| 时点 | 内容 |
|---|---|
| 运行前 | 仅我自己写的快照文件 .git_status_before.txt |
| 运行后 | 仅 .git_status_before.txt / .git_status_after.txt |

去掉我自己的快照文件后两边**完全一致**，`deliverables/.../DELIVERABLES.md`、
`management/rebot_b601dm/DELIVERABLES.md` 都没有再被改写。
即审计的 9ff2f63 + 702b9e6（让生成器接受隔离输出路径、测试改用 tmp_path）确有效果。

### 1.2 唯一失败项：需要 ChatGPT 修复（我未改动）

    FAILED tests/simulation/cad/test_solid_geometry.py::test_build_report_aggregates_groups_and_caveats
    assert len(report["caveats"]) == 3
    E   assert 4 == 3

成因定位（git 证据）：

    git show --stat 85e204a
    scripts/simulation/step_solid_geometry.py | 18 +++++++++++-------
    1 file changed, 11 insertions(+), 7 deletions(-)

85e204a（"mark quantized vertex fingerprints as duplicate candidates"）在
`build_report` 的 caveats 里增加了第 4 条（rotated/mirrored copies may differ），
但没有同步更新我那条 `len(caveats) == 3` 的断言。该断言在 a042f51 时是绿的，
所以这是**审计提交引入的回归**，不是我的新代码造成。

按本轮规则我没有修改它（既不改正交源码，也不改测试标准去迁就），
建议审计方二选一：把断言更新为 4 条并断言具体文本，或把断言改成"至少包含这三条"，
避免再次因为追加 caveat 而失败。

完整 traceback 见
`outputs/simulation/rebot_b601_rs/morph_one/revalidation_2270cb3/p0_failing_traceback.log`。

---

## 任务二 STEP 实体归因复验

使用审计版 `scripts/simulation/step_geometry_census.py`
（sha256 `55358c8dcb3a854c1b1d387d5d634bfc417691b4073717531922f992d781c592`）重新执行
attribute 与 trace（30.1 s / 27.1 s，各约 1.12 GiB）。

### 2.1 结论：469 个候选点全部仍是辅助几何，solid 归属没有任何变化

| 指标 | 第二轮（修复前） | 本轮（审计修复后） | 变化 |
|---|---|---|---|
| 候选点 | 469 | 469 | 无 |
| 已归因 / 未归因 | 469 / 0 | 469 / 0 | 无 |
| 涉及 solid 数 | 17 | **17** | 无 |
| 直方图首位 | 894929 -> 113 | 894929 -> 113 | 无 |
| kind 分类 | auxiliary_geometry 469 | **auxiliary_geometry 469** | 无 |
| 直接父类型 | LINE 419 / AXIS2_PLACEMENT_3D 50 | **LINE 419 / AXIS2 50** | 无 |

更硬的证据：两次产物在 git blob（LF）层面**逐字节相同**

    step_attribute.json          a849792c56c1866c...
    step_trace.json              6d502443b38fcb01...
    cartesian_point_outliers.csv eb7b8addacb96783...

（注意：Windows 工作区若开启 autocrlf，检出文件会被转成 CRLF，哈希随之改变
——本轮实测 CRLF 行数为 676 / 1994 / 470，而 blob 内为 0。核对时必须用 git blob，
不能用工作区文件。这一点建议写进审计清单。）

### 2.2 为什么"修复了却没有任何变化"——结构化解释

新增独立程序 `scripts/simulation/verify_attribution_depths.py`：
对每个候选点按**深度**逐层上溯，分别给出

* legacy_owners = 第一个出现 owner 的深度上的 solid（旧实现最多只能看到这些）
* full_owners   = 所有深度的并集（审计后的返回）

实测（21.5 s）：

| 指标 | 值 |
|---|---|
| 检查点数 | 469 |
| legacy_solids / full_solids | 17 / 17 |
| **存在多个 owner 深度的点数** | **0** |
| **早期 break 会丢 owner 的点数** | **0** |
| 首个 owner 出现深度分布 | 深度 5: 50 点（placement 路径）／深度 8: 419 点（LINE 路径） |
| 与第二轮直方图比较 | histograms_equal = true，两向差集皆空 |

即：该模型的每个候选点，其全部 owner 都出现在**同一深度**，所以"找到第一个就停"
在数学上不可能丢掉 owner。修复本身仍然是正确且必要的（见 2.3），但对本数据无行为变化。

### 2.3 修复确实覆盖了真实缺陷（用新夹具证明）

我的验证工具另配了 3 项测试（`tests/simulation/cad/test_verify_attribution_depths.py`），
夹具让一个点通过**共享 placement** 在深度 4 到达 solid 600、通过顶点链在深度 5 到达
solid 100：

    legacy_owners = [600]
    full_owners   = [100, 600]
    points_where_early_stop_would_lose_an_owner = 1

这说明：审计的 5d4fd18 修的是一类真实缺陷，只是 Morph One 这份数据恰好不触发。
本轮没有为了"对齐旧结果"而改动任何归因算法或断言。

---

## 任务三 CAD 转换复验（带显式验收条件）

新输出路径：`/home/T7/dgut/runs/cad_round3/usd/M1-4WD4S_model2601_round3.usda`（此前不存在）

    --expect-meshes 282 --expect-rendered-instances 282 \
    --expect-meters-per-unit 0.001 --expect-up-axis Z

| 项 | 值 |
|---|---|
| 退出码 | 0 |
| wall clock / 峰值 RSS | 2:50.57 / 4.79 GiB |
| verification_checks | has_nonempty_meshes / expected_prototype_mesh_count / expected_rendered_instance_count / expected_meters_per_unit / expected_up_axis **全部 true** |
| 网格 / 顶点 / 实例 | 282 / 455,255 / 282 |
| 世界包围盒 | 580.0 × 372.996 × 802.852（mm 单位） |
| metersPerUnit / upAxis | 0.001 / Z |
| 输出大小 | 58,660,611 B |
| 输出 sha256 | `181cd7957910a6059d6f4309e73193707e3fc6c1121aec4e82ac5e9578de9db7` |

**新发现（需要审计方定调）**：与第二轮同参数产物相比，几何计数与包围盒完全一致，
但字节数从 58,658,300 变成 58,660,611（+2,311 B，sha256 不同）。
即**该转换器输出不是字节级可复现的**。验收口径应固定在"计数 + 单位 + 朝向 + 包围盒"，
不能写成"sha256 必须相同"，否则每次重跑都会"失败"。

---

## 任务四 米制场景装配验证（本轮新增实验）

场景文件（极小，已入库）：`revalidation_2270cb3/A_no_scale_metre_stage.usda` 等 3 个。
三者都只做引用，不添加碰撞/质量/关节/机械臂约束。

| 用例 | 场景 metersPerUnit | 引用节点缩放 | 组合后局部→世界对角 | 世界包围盒（场景单位） | 物理尺寸（m） | 匹配 0.580×0.373×0.803 |
|---|---|---|---|---|---|---|
| A 不缩放（米制场景） | 1.0 | 无 | 1.0, 1.0, 1.0 | 580.0 × 372.996 × 802.852 | **580.0 × 372.996 × 802.852** | 否 |
| B 缩放 0.001（米制场景） | 1.0 | 0.001 | 0.001, 0.001, 0.001 | 0.580000028 × 0.372995715 × 0.802851662 | **0.5800 × 0.3730 × 0.8029** | **是** |
| C 不缩放（毫米制场景） | 0.001 | 无 | 1.0, 1.0, 1.0 | 580.0 × 372.996 × 802.852 | 0.5800 × 0.3730 × 0.8029 | 是（因为整场景就是 mm） |

B/A 比值：0.001（三个轴一致）。B 与期望值偏差约 3e-8 m，远小于任何测量精度。

**Isaac Sim 内部复测一致**：用 headless Isaac Sim 打开 A、B 两个场景读取同一量：

| 场景 | stage mpu | mesh prims | 世界包围盒（m） | Collision/RigidBody/Mass/Articulation |
|---|---|---|---|---|
| A | 1.0 | 282 | 580.0 × 372.996 × 802.852 | 0 / 0 / 0 / 0 |
| B | 1.0 | 282 | 0.580000028 × 0.372995715 × 0.802851662 | 0 / 0 / 0 / 0 |

### 结论（同时纠正我自己此前的一个错误假设）

* **USD 不会在引用时自动做单位换算**：虽然被引用的 USD 声明 metersPerUnit = 0.001，
  但用例 A 组合出的变换对角仍是 1.0，尺寸仍是 580（在米制场景里就等于 580 m）。
  我此前在脚本注释里写的"USD 会按 metersPerUnit 自动换算"是**错的**，已被本实验推翻，
  注释与报告已按实测改写。
* 因此在**米制场景**中引用该 Visual USD，**必须显式施加 0.001 缩放**才能得到
  0.580 × 0.373 × 0.803 m；用例 B 与任务书的期望一致。
* 用例 C 也"正确"只是因为整个场景被声明为毫米制，不是可移植做法；工程上应统一用米制 + 0.001 缩放。
* 以上数值是**转换几何的外包围盒**，不是已确认的车架尺寸；本实验只验证引用、坐标与单位。

---

## 交付物

| 路径（证据分支内） | 内容 |
|---|---|
| management/morph_one_rs/PHASE_H_REVALIDATION_2270cb3.md | 本报告 |
| scripts/simulation/verify_attribution_depths.py | 归因深度验证（新增） |
| scripts/simulation/verify_metric_scene_reference.py | 米制场景 A/B/C 对照（新增） |
| scripts/simulation/verify_metric_scene_in_isaac.py | Isaac Sim 内部同款测量（新增） |
| tests/simulation/cad/test_verify_attribution_depths.py | 3 项测试（新增） |
| outputs/.../revalidation_2270cb3/p0_pytest.log | 任务一完整输出（含失败） |
| outputs/.../revalidation_2270cb3/p0_failing_traceback.log | 失败 traceback |
| outputs/.../revalidation_2270cb3/step_attribute.json | 审计版 attribute 结果 |
| outputs/.../revalidation_2270cb3/step_trace.json | 审计版 trace 结果 |
| outputs/.../revalidation_2270cb3/attribution_depth_verification.json | 深度验证（469 点逐点） |
| outputs/.../revalidation_2270cb3/convert_round3.json / .log | 任务三报告与日志 |
| outputs/.../revalidation_2270cb3/metric_scene_report.json | 任务四 A/B/C 结果 |
| outputs/.../revalidation_2270cb3/metric_scene_in_isaac.json / .log | Isaac Sim 内部复测 |
| outputs/.../revalidation_2270cb3/*.usda | 三个米制/毫米制场景定义 |
| outputs/.../revalidation_2270cb3/MANIFEST.sha256 | 上述文件校验和 |

资源（留在 GPU 主机，不入库）：

    /home/T7/dgut/runs/cad_round3/usd/M1-4WD4S_model2601_round3.usda   58,660,611 B
    sha256 181cd7957910a6059d6f4309e73193707e3fc6c1121aec4e82ac5e9578de9db7
    /home/T7/dgut/robot_sim/assets/.../M1-4WD4S_model2601.step        115,496,178 B
    sha256 388cb3653a864066a35a54065f71938635c7e9a221cb4b897b31fbd44ff266c5

复现命令（GPU 主机，PYTHONPATH 指向 /home/T7/ojh/robot_sim/IsaacLab/source/*）：

    # 任务二
    python scripts/simulation/step_geometry_census.py --input <step> --mode attribute --output-dir <out>
    python scripts/simulation/step_geometry_census.py --input <step> --mode trace     --output-dir <out>
    python scripts/simulation/verify_attribution_depths.py --input <step> --output-dir <out> \
           --compare-with <out>/step_attribute.json

    # 任务三（必须先 LD_PRELOAD hoops 的 libTD_Root.so）
    python scripts/simulation/convert_step_to_usd.py --input <step> --output <new.usda> \
      --report <report.json> --headless --max-seconds 7200 \
      --expect-meshes 282 --expect-rendered-instances 282 \
      --expect-meters-per-unit 0.001 --expect-up-axis Z \
      --kit_args "--enable omni.kit.converter.cad"

    # 任务四
    python scripts/simulation/verify_metric_scene_reference.py --visual-usd <usd> --output-dir <dir>
    python scripts/simulation/verify_metric_scene_in_isaac.py --scene <A.usda> --scene <B.usda> \
      --visual-usd <usd> --report <json> --headless

---

## 仍然 UNKNOWN / 下一步

1. 任务四只验证了引用、单位与坐标。**未**添加碰撞体、质量、关节、机械臂固定约束，
   因此"能否用于物理仿真"仍未验证。
2. 0.580 × 0.373 × 0.803 m 是几何外包围盒，**不是**已确认的车架尺寸；轮半径、转向轴高度、
   轮距、轴距、质量、惯量仍全部 UNKNOWN。
3. 转换输出非字节级可复现（本轮实测 +2,311 B），建议验收口径固定为计数/单位/朝向/包围盒。
4. 任务一的 caveats 断言失败需要审计方修复（或明确允许我改断言）。
