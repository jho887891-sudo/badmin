# PR #9 独立复验：STEP 科学计数法方向向量

被验对象：`e771ed167affd894742fed3f64f75c0e5c58ada7`（含修复提交 `df30447` 与测试提交 `e771ed1`）
独立 worktree：`evidence/pr9-exponent-direction-verification`　基线 `b0853efe`
本文件只记录复验结果，不修改被验代码。

---

## 1. 单测结果：**PR 自己新加的测试是红的**

    python -m pytest tests/simulation/cad -q
    -> 1 failed, 54 passed, 3 skipped

失败项（PR #9 自己新增的那条）：

    tests/simulation/cad/test_solid_geometry.py::test_circle_facts_preserves_axis_with_exponential_direction
    assert '(-0,0,1)' == '(0,0,1)'      # 实际 '(-0,0,1)'

也就是说：指数分量**确实被解析出来了**（否则断言会看到空列表），但轴标签里出现了**负零** `-0`，与期望的
`0` 不相等。属于"修复有效但收尾没做完"，不是原缺陷回归。

## 2. 真实 STEP 前后对比（同一份 115,496,178 B 的 STEP）

方法：分别用 `b0853ef` 与 `e771ed1` 两个版本的 `step_solid_geometry.py` 在同一台机器上跑同一份 STEP，
统计 DIRECTION 解析率、每实体 dominant_circle_axes、以及轴标签分布。

| 指标 | 修复前 b0853ef | 修复后 PR #9 | 变化 |
|---|---|---|---|
| DIRECTION 实体数 | 73,606 | 73,606 | 不变 |
| 成功解析 | 31,401 | **73,606** | +42,205 |
| 解析失败 | 42,205（57.3%） | **0** | 全部修复 |
| CIRCLE/CYLINDRICAL 特征总数 | 13,324 | 13,324 | 不变（修复不影响特征识别） |
| 有 dominant_circle_axes 的实体 | 234 / 282 | **276 / 282** | +42 |
| 有圆但拿不到轴 | 42 | **0** | 全部修复 |
| 不同轴标签数 | 135 | 105 | −30（原先大量轴因解析失败被丢弃/错并） |
| 含 `-0` 的轴标签数 | 150 | **505** | +355（负零问题暴露出来） |
| 出现 `-0` 标签的实体数 | 134 | 264（共 276 有轴） | — |
| 探针实体 #254973 | `None` | `(1, -0, -0)` | 修复生效 |

修复后标签前四名：`(1,0,0)=52`、`(1,-0,-0)=46`、`(-1,-0,-0)=43`、`(0,1,0)=42`——**语义相同但字符串不同**。

## 3. 遗留缺陷（FAILED，需 ChatGPT 处理，我未修改其代码）

* 现象：`dominant_circle_axes` 的轴标签由三个分量格式化而来，`-4.79e-14` 这种量格式化后成为 `"-0"`。
* 影响：标签被当作**分组/统计的键**，于是同一物理轴向会分裂成 `(1,0,0)` 与 `(1,-0,-0)` 两个键。真实数据上
  264/276 个实体至少有一个这样的标签，505 个标签受影响；这会污染"最常出现的轴向"统计，也会让 `(0,0,1)` 与
  `(0,0,-1)` 的计数方式变得难以解释。
* 最小修法（供审计参考，未提交到 PR 分支）：在生成标签前把每个分量归一化，例如 `value = float(value) + 0.0`
  （IEEE754 里 `-0.0 + 0.0 == 0.0`），或先 `round(v, 9) + 0.0` 再格式化；PR 自带的测试即可转绿。
* 严重度：**不改变几何与映射结论**，只影响标签一致性；但会让 PR #9 的测试套件保持红色，不能直接合并。

## 4. 原映射数量与实体对应是否受影响：**不受影响（VERIFIED）**

三条证据：

1. 解析函数作用域很窄：`_TRIPLE_RE` 只被 `_direction_vector()` 使用，后者只被 `step_solid_geometry.py:218`
   的 AXIS2_PLACEMENT_3D 分支调用；映射工具 `map_step_solids_to_usd_meshes.py` 里
   **没有任何** `_direction_vector` / `dominant_circle_axes` / `circle_facts` 引用（grep 无命中）。
2. 映射依赖的量未变：实体总数 282、CIRCLE/CYLINDRICAL 13,324、DIRECTION 73,606、点的集合均不变；映射用顶点
   包围盒与点集匹配，不走方向解析。
3. 既有验收数字仍是上一轮的 282/282 配对（235 `STRONG_CANDIDATE` + 47 `CANDIDATE`、0 `UNKNOWN`），本轮
   复跑中没有任何映射侧的输入发生变化。

## 5. 结论分级

| 项 | 状态 |
|---|---|
| E/D 指数方向向量解析 | **VERIFIED**（73,606/73,606，探针 #254973 由 None 变为向量） |
| 真实 STEP 的 dominant_circle_axes 重建 | **VERIFIED**（276/282，42 个"有圆无轴"实体归零） |
| 修复前后统计对比 | 已记录（第 2 节表） |
| PR #9 自带测试 | **FAILED**（负零标签，1 failed） |
| 原映射数量与实体对应 | **不受影响（VERIFIED）** |

## 6. 复现

    # 单测（CPU）
    python -m pytest tests/simulation/cad -q
    # 真实 STEP 前后对比（GPU 主机 CPU 跑，约 60 s/次）
    python /tmp/a1stats.py <repo-root-before> before
    python /tmp/a1stats.py <repo-root-after>  after

证据文件：`outputs/simulation/rebot_b601_rs/morph_one/pr9_b0853ef/dominant_axes_before_b0853ef.json`、
`.../dominant_axes_after_pr9.json`。
