# PR #8 修复复核：pxr 环境回归记录（工作流 A，第二轮）

分支 `evidence/morph-one-audit7-verification-b0853ef`，基线 `b0853efe375bda931b84ed401c7ccff2d6ce950e`。
本文件只追加**运行记录**，不改动任何被审计的代码与断言。

---

## 1. 本轮新增的一项：在带 pxr 的 Isaac 解释器里跑回归

第一轮的 pytest 记录跑在本机（无 pxr），pxr 相关用例被 skip。本轮把 `b0853ef` 的树搬到 GPU 主机，用 Isaac Sim
自带解释器（带 pxr）实际跑一遍，消除"跳过"这一不确定性。

| 项 | 值 |
|---|---|
| 解释器 | `/home/T7/ojh/robot_sim/env_isaaclab/bin/python`（CPython 3.12.13） |
| pxr | **0.25.11** |
| pytest | 9.1.1 |
| 树来源 | GPU 主机无 GitHub 出口 → 本机 `git archive --format=tar.gz b0853ef`（16,013,217 B、943 文件），scp 后解包，未做任何本地修改 |
| 命令 | `pytest tests/simulation/cad tests/simulation/rebot_b601_rs tests/simulation/rebot_b601dm/test_deliverables_index.py -q` |

结果：

    151 passed in 1.81s
    PYTEST_EXIT=0

即：**cad（含 pxr 用例）+ rebot_b601_rs + deliverables index 全量 151 项通过，0 失败、0 跳过**。

### 1.1 中间发现（如实记录，未掩盖）

第一次打包时我把 `docs/` 排除在外（为了减小传输体积），结果 2 项失败：

    FAILED tests/simulation/rebot_b601dm/test_deliverables_index.py::test_every_listed_file_exists
    FAILED ...::test_the_regenerated_index_still_lists_only_existing_files
    assert not ['docs/superpowers/plans/2026-09-17-rebot-b601dm-badminton-capability-plan.md']
    2 failed, 149 passed

结论：这不是代码缺陷，而是**验收断言确实有效**——交付索引里的路径一旦在磁盘上不存在，它就会失败。把 `docs/`
包含进来后立刻 151 全绿。这条记录同时说明：任何"只搬部分目录"的复核方式都会污染结果，后续复核必须整树搬运。

## 2. 第一轮结论（本分支 `PHASE_J_AUDIT7_VERIFICATION.md` 与 `outputs/simulation/rebot_b601_rs/morph_one/audit7_b0853ef/{p0,p1,p2}`）

保持不变，摘要如下（详见原文件，未修改）：

| 复核项 | 结果 |
|---|---|
| `test_build_report_aggregates_groups_and_caveats` | 通过 |
| `sandwich()` 下界 | 修正后正确（原实现把边界判成"内部"） |
| 二分图匹配是否丢合法配对 | 修正后不再丢；旧的"贪婪+阈值"规则复现出 224 对，与历史一致 |
| STEP→USD 映射 | 282/282 配对，235 `STRONG_CANDIDATE` + 47 `CANDIDATE`，0 `UNKNOWN`，`bijection_proven = false` |
| 新旧映射差异 | 224 `UNCHANGED`、58 `INEQUALITY_FIX`（旧规则下均不可接受）、0 `MATCHING_ALGORITHM`、0 `LOST` |
| 度量场景复验 | `verdict.ok = true`、`both_have_valid_scene_checks = true`、`bbox_methods_agree = true`、四个 PhysX API 存在且计数为 0 |
| 截图 | 不可用（`ModuleNotFoundError: No module named 'omni.kit.viewport'`）——如实记录，不伪造截图 |
| 工作区洁净度 | 复核前后 `git status` 干净 |

## 3. 复核口径提醒（给审计方）

* 转换输出**不是字节可复现**的（同一输入三次得到 58,658,300 / 58,660,611 / 58,659,744 B），所以验收只能看
  计数、单位、轴向、包围盒，不能看 sha256 相等。
* Windows 上 `core.autocrlf` 会把文本证据变成 CRLF，比对哈希要拿 git blob（LF）而不是工作区文件。
* 本轮所有产物均为**证据**，未修改 `scripts/`、`tests/`、`simulation/` 中的任何被审计文件（`git status` 检查过）。
