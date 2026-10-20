# PR #9 回归复验：指数方向向量 + 负零归一化

被验提交：`3a679b35fc326ed8eadfdaf299f56ce19d28c321`（代码修复 `65a94a8`、测试 `3a679b3`）
worktree：`evidence/pr9-signed-zero-verification`（独立建立，未与其他分支混合）
本文件只记录复验，不修改被验代码。

---

## 1. CAD 测试：全部通过，退出码 0

    python -m pytest tests/simulation/cad -q
    -> 56 passed, 3 skipped in 1.06s          EXIT=0

    python -m pytest tests/simulation -q
    -> 295 passed, 49 skipped in 18.23s       EXIT=0

定点确认两条关键用例：

    python -m pytest tests/simulation/cad/test_solid_geometry.py -q -k "negative_zero or exponential_direction"
    -> 2 passed, 9 deselected                 EXIT=0
__PART2__
* `test_circle_facts_preserves_axis_with_exponential_direction`（上一轮失败的那条）现在**通过**；
* `test_step_direction_axis_bins_canonicalize_negative_zero`（本轮新增）**通过**。

## 2. 真实 STEP 重新统计

同一份 `M1-4WD4S_model2601.step`（115,496,178 B），三个版本各跑一次：

| 指标 | 修复前 `b0853ef` | PR #9 首版 `e771ed1` | PR #9 现在 `3a679b3` |
|---|---|---|---|
| DIRECTION 实体 | 73,606 | 73,606 | 73,606 |
| 成功解析 | 31,401 | 73,606 | **73,606** |
| 解析失败 | 42,205 | 0 | **0** |
| CIRCLE/CYLINDRICAL 特征 | 13,324 | 13,324 | 13,324 |
| 有 dominant_circle_axes 的实体 | 234 / 282 | 276 / 282 | **276 / 282** |
| 有圆但拿不到轴 | 42 | 0 | **0** |
| 不同轴标签数 | 135 | 105 | **76** |
| 含 `-0` 子串的标签 | 150 | 505 | 48 |
| 分量恰为 `-0` 的标签（裸负零） | 未单独统计 | 未单独统计 | **0（出现 0 次）** |
| 探针实体 #254973 | `None` | `(1, -0, -0)` | **`(1, 0, 0)`** |

修复后最常见的 6 个轴标签（都是规范化的整数分量）：

    (-1,0,0) 140   (0,1,0) 133   (1,0,0) 129
    (0,-1,0) 121   (0,0,1) 39    (0,0,-1) 39
## 3. 负零造成的重复方向标签是否归零：**已归零**

判定口径要区分两件事，本轮两种都测：

* **裸负零**（某个分量恰好是 `-0`，例如 `(1,-0,-0)`）：正则 `(?<![0-9.])-0(?![0-9.])` 判定
  -> **0 个标签、0 次出现**。
* **`-0` 子串**（例如 `(0,1,-0.144)`、`(1,0,-0.257)`、`(-1,0,-0.595)`）：仍有 31 个标签 / 48 次出现，
  但它们含的是**负的非零分量**，不是有符号零。上一轮我用「子串包含 `-0`」作检测器，会把这些也算进去；
  本轮把口径拆开，因此**不能**把 48 说成负零残留。

标签种类由 105 降到 76，正是「同一轴向被 `-0` 拆成两桶」被消除后的结果。

## 4. 原映射数量与实体对应：不受影响

* 实体总数 282、CIRCLE/CYLINDRICAL 13,324、DIRECTION 73,606 全部未变；
* 映射工具 `map_step_solids_to_usd_meshes.py` 不引用 `_direction_vector` / `dominant_circle_axes` /
  `circle_facts`（上一轮已 grep 验证），方向解析的改动不会传导到映射；
* 因此既有 282/282 配对结论保持不变。

## 5. 过程与退出码

| 步骤 | 命令要点 | 退出码 |
|---|---|---|
| CAD 单测（本机） | `pytest tests/simulation/cad -q` | 0 |
| 全 simulation 单测（本机） | `pytest tests/simulation -q` | 0 |
| 定点两条用例 | `pytest -k negative_zero or exponential_direction` | 0 |
| 真实 STEP 统计（GPU 主机 CPU） | `a1stats.py <root> final` | 0（STATS_EXIT=0） |
| 负零分类（GPU 主机 CPU） | `a2zero.py <root>` | 0 |

## 6. 结论分级

| 项 | 状态 |
|---|---|
| 负零断言不再失败（CAD 测试全绿） | **VERIFIED**（56 passed / 3 skipped / exit 0） |
| 指数方向向量解析 | **VERIFIED**（73,606 / 73,606） |
| dominant_circle_axes 重建 | **VERIFIED**（276 / 282，0 个「有圆无轴」） |
| 负零造成的重复标签 | **VERIFIED（归零）**（裸负零 0 标签 / 0 次） |
| 原映射数量与实体对应 | **不受影响** |
| 剩余 `-0` 子串 | 负的非零分量，非缺陷（口径已区分） |

证据文件（本分支 `outputs/simulation/rebot_b601_rs/morph_one/pr9b_signed_zero/`）：
`pytest_cad.txt`、`pytest_full_simulation.txt`、`dominant_axes_pr9_final.json`、`zero_classification.json`、
`final_run.log`。
