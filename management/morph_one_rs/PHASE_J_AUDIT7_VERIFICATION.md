# Phase J — 第七轮审计（PR #8）修复验证

- 目标 SHA：b0853efe375bda931b84ed401c7ccff2d6ce950e（分支 review/morph-one-metric-mapping-audit-20261020）
- 验证工作区：E:/具身智能/badmin_wt_b0853ef
- 证据分支：evidence/morph-one-audit7-verification-b0853ef
- 我未修改任何 ChatGPT 源码；本轮新增 1 个独立验证工具 + 1 个测试文件。

## 提交树一致性

| 项 | 值 |
|---|---|
| HEAD | b0853efe375bda931b84ed401c7ccff2d6ce950e |
| HEAD 树对象 | dbc2fe4c615f44a9012b6dba3469b3810d34690c |
| 工作区未跟踪/修改 | 0（检出后立即为空） |
| 远端 ref 指向 | refs/heads/review/morph-one-metric-mapping-audit-20261020 = b0853efe… |
| 与我上一轮 dev 分支关系 | 我的 3ce42c0 是本提交的祖先（审计基于我的工作追加 8 个提交） |

---

## P0 完整回归

| 运行 | 命令 | 退出码 | 结果 |
|---|---|---|---|
| 系统 Python | pytest tests/simulation/cad tests/simulation/rebot_b601_rs tests/simulation/rebot_b601dm/test_deliverables_index.py -q | 0 | **148 passed, 3 skipped** |
| 同上 + 我的新测试 | 同上 | 0 | **155 passed, 3 skipped** |
| Isaac venv（pxr 测试） | python -m pytest tests/simulation/cad/test_morph_one_metric_scene.py -q | 0 | **12 passed** |

- **原先失败的 test_build_report_aggregates_groups_and_caveats 现已通过**（单独运行亦为 1 passed）。
  审计在 tests/simulation/cad/test_solid_geometry.py 里同步了 caveats 断言。
- 3 skipped 是本轮我新增的 pxr 依赖测试在系统解释器上的预期跳过；在 Isaac venv 上全绿。
- 工作区污染：**零**。运行前后 git status --porcelain 仅含我自己的快照文件
  （p0/git_status_before.txt、p0/git_status_after.txt）。

---

## P1 STEP → USD 实体映射重算

输入（未改动，均可校验）：

    STEP 115,496,178 B  sha256 388cb365…f266c5
    USD  58,659,744 B   sha256 3b9962913ec517637df2a57df58acb5650fe995b437088539e26f00912af74b7
    审计版 mapper sha256 b4a3f90b3fe7412248a4e20cc872552729f151ac4e2b8517edae8fec6d7a5d33

退出码 0，耗时 28.4 s，峰值 1.12 GiB。

### 1. 新配对总数

**282 / 282**（旧：224 / 282）。

### 2. 分级数量

| 级别 | 数量 |
|---|---|
| STRONG_CANDIDATE | 235 |
| CANDIDATE | 47 |
| UNKNOWN | 0 |
| VERIFIED | 0 —— 审计已把 VERIFIED 保留给"逐件源 ID/精确对应"，本轮不使用 |

bijection_proven = **false**（审计硬编码为 false，遵循真实证据：名字键只有 240 个不同值，
35 个零件名字字面为 NONE、4 组各有 3 件同名，无法证明一一对应）。
rejected_pair_count = 632（保留前 50 条诊断样本）。

### 3. 与旧结果（224 配对 / 58 未匹配）的具体差异

| 类别 | 数量 | 说明 |
|---|---|---|
| UNCHANGED | **224** | 旧配对全部保留，**没有任何一对换了伙伴** |
| INEQUALITY_FIX | **58** | 旧未匹配的 58 个实体全部配对成功，且其新伙伴在旧规则下**不可行** |
| MATCHING_ALGORITHM | **0** | 本数据上增广路径匹配没有改变任何伙伴选择 |
| LOST | 0 | 无 |

### 4. 因"包围盒最小坐标不等式修复"而改变的实体

审计把下界判据由

    below = max(0, lower_min - mesh_min, lower_max - mesh_max)      # 修复前
    below = max(0, mesh_min - lower_min, lower_max - mesh_max)      # 修复后

前者把"网格向负方向鼓出顶点盒"误判为违规（曲面包络本来就会向外鼓），
同时**漏检**"网格没有盖住顶点盒"的情形。修复方向正确。

归因方法（可复核）：我独立复现了修复前的判据与旧贪心选择，在相同 STEP/USD/容差
（2.5 mm、1%）下产出 **224 对**，与已提交的旧结果**逐一相同（一致率 1.0）**；
因此"每个新增配对在旧规则下是否可行"这一判断是可信的。

- 58 个新增配对全部满足 new_partner_feasible_under_old_rule = false。
- 这些实体在旧规则下的可行候选数：0 个（56 个实体）、1 个（2 个实体）；
  在新规则下：1 个（54 个）、2 个（4 个）。
- 逐实体 ID 与新伙伴 Prim 路径见 p1/mapping_supersession.csv
  （列含 cause / step_eid / old_usd_path / new_usd_path / 两侧候选数 / 名称）。

### 5. 因"双边最大匹配算法"而改变的实体

**0 个**。旧贪心在每组内本就按代价升序取，而 58 个新增件在旧规则下没有可行边，
匹配算法无从改派。审计换算法仍是必要的健壮性改进（其新增测试覆盖了
"A 有两个候选、B 只有一个候选时贪心会抢走 B 的唯一候选"这一情形），只是本数据不触发。

### 6. 逐实体对照

- p1/step_to_usd_mapping.csv：282 行，step_eid / 名称 / 归一化键 / usd_path / 分级 / 包围盒 / 组规模。
- p1/mapping_supersession.csv：282 行，额外给出 cause 列与新旧路径并列，可逐行核对。

### 7. 分级变化（说明 VERIFIED 的语义改变）

| 旧 → 新 | 数量 |
|---|---|
| VERIFIED → STRONG_CANDIDATE | 199 |
| （无）→ STRONG_CANDIDATE | 36 |
| （无）→ CANDIDATE | 22 |
| CANDIDATE → CANDIDATE | 25 |

199 个"唯一名字键 + 几何夹逼"的配对并没有被判为错误，而是审计把 VERIFIED 一词收紧为
"源 ID 级精确对应"后的语义重命名。这与证据一致，我认同该收紧。

---

## P2 米制场景复验（新输出目录）

命令退出码 0，34.9 s，峰值 2.55 GiB；输出目录 /home/T7/dgut/runs/cad_audit7/metric_scene_audit7。

| 检查项 | 结果 |
|---|---|
| verdict.ok | **true** |
| both_have_valid_scene_checks | **true** |
| bbox_methods_agree（A / B） | **true / true** |
| stage_meters_per_unit（A / B） | **1.0 / 1.0** |
| up_axis（A / B） | **Z / Z** |
| CollisionAPI / RigidBodyAPI / MassAPI / ArticulationRootAPI | **0 / 0 / 0 / 0**（两场景各四项均存在且为 0） |
| both_free_of_physics_apis | true |
| B_is_thousandth_of_A | true（0.001, 0.001, 0.001） |
| A_matches_expected / B_matches_expected | false / **true** |

实测包围盒：

| 用例 | 世界包围盒（场景单位） | 物理尺寸 |
|---|---|---|
| A 不缩放 | 580.0 × 372.995697021 × 802.851623535 | 580.0 × 372.996 × 802.852 m（不匹配期望） |
| B 缩放 0.001 | 0.580000028 × 0.372995715 × 0.802851662 | **0.5800 × 0.3730 × 0.8029 m** |

两用例各 282 个 mesh 实例、575 个 prim；逐点变换得到的包围盒与 UsdGeom.BBoxCache 一致。
截图仍不可用（headless 无 omni.kit.viewport），已在 JSON 中如实记录，未伪造。
本轮只做单位与坐标组合验证，未添加碰撞、质量、惯量或任何关节/机械臂约束。

---

## 交付物（本分支内）

| 路径 | 内容 |
|---|---|
| management/morph_one_rs/PHASE_J_AUDIT7_VERIFICATION.md | 本报告 |
| scripts/simulation/verify_mapping_supersession.py | 独立归因工具（新增） |
| tests/simulation/cad/test_verify_mapping_supersession.py | 7 项测试（新增，全绿） |
| outputs/.../morph_one/audit7_b0853ef/p0/ | 前后 git status + 全量 pytest 日志 |
| outputs/.../morph_one/audit7_b0853ef/p1/ | 新映射 JSON/CSV、归因 JSON/CSV、运行日志 |
| outputs/.../morph_one/audit7_b0853ef/p2/ | 米制场景 JSON、日志、两个场景 USDA |
| outputs/.../morph_one/audit7_b0853ef/MANIFEST.sha256 | 上述文件校验和 |

复现命令：

    # 归因（需 Isaac venv 的 pxr）
    python scripts/simulation/verify_mapping_supersession.py --step <step> --usd <usd> \
      --old-report <旧 mapping.json> --new-report <新 mapping.json> --output-dir <out>
    # 米制场景
    python scripts/simulation/run_metric_scene_validation.py --visual-usd <usd> \
      --output-dir <新目录> --report <json> --screenshot --headless

---

## 仍需注意（不建议当作已证明）

1. STRONG_CANDIDATE 仍是"唯一归一化名字 + 近似包围盒"，不是源 ID 级对应；审计自己也把
   VERIFIED 保留给将来的逐件源 ID。容差 2.5 mm 是经验值，未经厂商误差界标定。
2. 47 个 CANDIDATE 中包含 35 个名字为 NONE 的零件与 12 个来自 4 组同名件，只能靠几何选伴。
3. 米制场景仅验证引用/单位/坐标；物理可用性（碰撞体、质量、关节）仍未验证。
4. 0.580 × 0.373 × 0.803 m 是转换几何外包围盒，不是确认过的车架尺寸。
