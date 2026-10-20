# PR #11 独立复验：轮组对称性改为比较水平投影

被验对象：`eaf7c1e20ea97c25c019cb3e23b896b604bea200`（修复 `2c39636`、测试 `0b2e417`、文档 `eaf7c1e`）
独立 worktree：`evidence/pr11-wheel-symmetry-verification`　基线 `b0853efe`
本文件只记录复验结果，不修改被验代码。

---

## 1. 单测

    python -m pytest tests/simulation/cad -q
    -> 77 passed, 3 skipped

## 2. 真实 STEP 重算（同一份 STEP，同一容差）

    工具 sha256（PR #11 版）已核对
    solids 282 | clusters 2224 | segments 2755 | HIGH 73 | up_axis y | 4/4 corners

与 PR #11 之前的运行**完全一致**（2 224 条候选轴 / 2 755 个轴向分段 / 73 条 HIGH / 四角 steering 候选
161-123-144-123 与 drive 候选 488-408-409-384 均未变）——说明这次修复只改了对称性**度量方式**，没有动几何。

## 3. 四组镜像的水平投影残差（本轮重点）

新增字段 `symmetry_metric = "horizontal_axis_footprint"`，同时保留原始三维偏移诊断：

| 配对 | 镜像轴 | 水平投影残差 mm | 三维残差 mm（保留） | 竖直锚点偏移 mm | 半径差 mm | 判定 |
|---|---|---|---|---|---|---|
| ++ ↔ −+ | X | [0.0, 0.0, 0.0] | [0.0, 1.0692, 0.0] | 1.0692 | 0.0 | consistent |
| +− ↔ −− | X | [0.0, 0.0, 0.0] | [0.0, 2.2810, 0.0] | 2.2810 | 0.0 | consistent |
| ++ ↔ +− | Z | [0.0, 0.0, 0.0] | [0.0, 0.0, 0.0] | 0.0 | 0.0 | consistent |
| −+ ↔ −− | Z | [0.0, 0.0, 0.0] | [0.0, 1.2118, 0.0] | 1.2118 | 0.0 | consistent |

原始三维偏移**没有被删除**（`mirror_residual_mm` 与 `anchor_up_offset_mm` 仍在），这一点符合要求。

## 4. 结论边界（不得扩大解释）

* **可以说**：以四角的转向候选轴为对象，它们的**水平足迹**在 X/Z 两个方向上镜像残差恰为 0.0000 mm，四个角的
  最大测量圆半径完全一致（40.0 mm，差 0.0）。
* **不能说**：这说明真实机械结构已经验证。三点限制必须写在同一份结论里：
  1. 被比较的对象是**候选轴**（同轴圆特征聚类的结果），不是实测关节；
  2. 上一轮 1.0692 / 2.2810 / 1.2118 mm 的三维差异是**不同零件上的不同特征被选为"最强锚点"**造成的，
     它不代表结构不对称，也不代表对称——它只是说明"用任意轴点比较高度"没有意义，这正是 PR #11 修正的点；
  3. 半径相同只说明四角有同样大小的同轴圆特征，不等于四个轮胎规格相同。
* 因此本轮对称性结论的分级是 **CANDIDATE（水平足迹一致）**，而"真实四轮结构对称"仍是 **UNKNOWN**，
  需要厂商图纸或实机测量。

## 5. 结论分级

| 项 | 状态 |
|---|---|
| 轮组轴线单测 | **VERIFIED**（77 passed / 3 skipped） |
| 完整 CAD 候选轴重算可复现 | **VERIFIED**（2 224 / 2 755 / 73 / 4 corners 与修复前一致） |
| 四组镜像水平投影残差为 0 | **VERIFIED（就候选轴足迹而言）** |
| 原始三维轴点偏移保留 | **VERIFIED**（三字段并存） |
| 真实机械结构对称 | **UNKNOWN**（未验证，且不得由上一行推出） |

## 6. 复现

    python -m pytest tests/simulation/cad -q
    python scripts/simulation/morph_one_wheel_kinematics.py --step <STEP> --mapping <mapping.json> --output-dir out

证据文件：`outputs/simulation/rebot_b601_rs/morph_one/pr11_b0853ef/wheel_kinematics_pr11.json`（含
`symmetry` 四行完整字段）、`.../wheel_axis_candidates_pr11.csv`、`.../wheel_parts_pr11.csv`、
`.../runner_pr11.out`。
