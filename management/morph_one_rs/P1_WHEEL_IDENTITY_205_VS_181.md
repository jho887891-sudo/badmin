# P1 轮件身份：205.991 mm 与 180.782 mm 差异量化

- 任务：Issue #13 · P1「分别检查 4 个候选轮状实体的 STEP 曲面几何和 USD 网格包围盒；澄清先前约 205.991 mm vs 180.782 mm 的差异」
- 分支：`p1/wheel-identity-205-vs-181`，base `459cb122f7be68ad60295ee61c4faaf73618b151`（`dev/morph-one-p2-quat-and-settle`）
- 本轮结论：**PARTIAL** —— 差异来源已量化并归因为「包围盒计算方式不同」，不是单位错误、也不是映射错配；但**轮径、轴心、轮距仍为 UNKNOWN**，本轮不确认任何真实轮径。

## 0. 一句话结论

`48.653 × 180.782 × 157.783 mm` 是**只用 BREP 顶点（VERTEX_POINT）**量出来的盒子（250 个点）；
`48.791 × 205.991 × 205.991 mm` 是**三角化网格全部顶点**量出来的盒子（6354 个点）。
STEP 的解析曲面（轮胎外圆柱面）在文件里以 `CIRCLE`/`CYLINDRICAL_SURFACE` + 半径 103.0 mm 存储，
**不在顶点集合里**，所以顶点盒在轮平面内只量到 180.782 / 157.783。两侧在半径上互相吻合
（STEP 解析半径 103.0 mm，网格点云 r_max = 103.000022 mm，差 2.2e-5 mm），在薄轴上也吻合
（48.653 vs 48.791 mm）。**因此 205.991 与 180.782 不是同一个量的两种说法之争，而是两种不同量的测量结果。**

## 1. 结论分级

| 项目 | 分级 | 依据 |
| --- | --- | --- |
| 差异来源归因：包围盒计算方式（顶点盒 vs 网格盒） | **PASS** | USD 值落在 STEP 的 [顶点盒, 曲面包络] 括号内，4/4 个 eid |
| 假设：单位错误（mm vs m、0.001 缩放） | **FAIL**（被证伪） | STEP 声明 `SI_UNIT(.MILLI.,.METRE.)`；USD `metersPerUnit=0.001`；半径残差 2.2e-5 mm |
| 假设：STEP→USD 映射候选错配 | **FAIL**（被证伪） | 圆心与网格盒中心 4/4 一致（≤0.00196 mm），最大圆半径 4/4 一致 |
| 假设：网格把多个实体合并成一个 prim | **UNKNOWN** | 无法用 bbox 判定；122 个连通分片是未焊接的曲面片，不是物理零件 |
| 假设：轮毂 vs 轮胎是两个不同实体 | **UNKNOWN** | 每个 eid 只是一个 BREP solid + 一个 mesh，半径从 63.0 mm 连续到 103.0 mm |
| **轮径 / 轴心 / 轮距的真值** | **UNKNOWN** | 4 个 eid 名称均为 `NONE_*`，映射等级仅 `CANDIDATE`；无实物测量或图纸佐证 |

## 2. 输入与身份（全部可校验）

| 对象 | 值 |
| --- | --- |
| STEP 文件 | `/home/T7/dgut/robot_sim/assets/robots/badminton_robot/morph_one/cad/M1-4WD4S_model2601.step` |
| STEP sha256 / bytes | `388cb3653a864066a35a54065f71938635c7e9a221cb4b897b31fbd44ff266c5` / 115,496,178 |
| 本地复核副本（gitignored `_scratch_downloads/`，与远端逐字节相同） | `E:\具身智能\badmin_project\_scratch_downloads\M1-4WD4S_model2601.step` |
| 转换后 USD | `/home/T7/dgut/runs/cad_dev/usd/M1-4WD4S_model2601_dev.usda`，sha256 `3b9962913ec517637df2a57df58acb5650fe995b437088539e26f00912af74b7` |
| 映射文件 | `/home/dgut/runs/wheels/mapping.json`，sha256 `460ebbb0d2a1faea3f81792fe67f5334d98fb53e62b70bf95d1fdbbe360ec096`，282 对 |
| 4 个 eid 的 prim | `…/NONE_15/Mesh`(435700)、`NONE_23/Mesh`(759491)、`NONE_7/Mesh`(773650)、`NONE_31/Mesh`(871375) |
| 轮件 STEP 名称 | 全部 `NONE`（未命名），映射等级 `CANDIDATE` |

**注意（证据可用性）**：本分支（base 459cb12）里
`outputs/simulation/rebot_b601_rs/morph_one/wheels_b0853ef/wheel_parts.csv` **不存在**，它在
`53d71ff`（`feat/morph-one-wheel-kinematics` 一线）。用下列命令读取，本轮全部交叉验证都基于它：

```
git show 53d71ff:outputs/simulation/rebot_b601_rs/morph_one/wheels_b0853ef/wheel_parts.csv
```

本分支内的 USD 侧原始证据 `outputs/simulation/rebot_b601_rs/morph_one/physx_validation/orientation_plus90.json`
（sha256 `fa89d11c47d089839c8e49262f55f30c31897c8f68b0aef01508ed5c6e9dd3b1`）确实记录了
`cad_bbox_size_mm = [48.791, 205.991, 205.991]`，四个 eid 各一份，与本轮 pxr 复测完全一致。

## 3. 方法：不猜一个数，而是夹一个区间

STEP 是 BREP，曲面是解析表达：轮的外圆柱面存成 `CYLINDRICAL_SURFACE` + 两条 `CIRCLE`（数值半径 103.0），
圆本身**不打散成点**。因此：

```
vertex_bbox  <=  真实几何包围盒  <=  surface_envelope_bbox
（只用 VERTEX_POINT）              （再并上每个 CIRCLE/ELLIPSE 的整圆包络、
                                   CYLINDRICAL_SURFACE 的轴向壳、B 样条控制点凸包）
```

下界与上界都由脚本算出，没有手写常数；USD 侧则用 pxr 读 `UsdGeom.Mesh.points`，
乘 prim 的 local-to-world 矩阵得世界空间点集，再用 `UsdGeom.BBoxCache` 交叉校验。

## 4. 结果对照表（`comparison.csv` / `comparison.json`）

| eid | STEP 顶点数 | USD 点数 | STEP 顶点盒 (mm) | STEP 曲面包络 (mm) | USD 网格盒 (mm) | R_step | R_usd |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 435700 | 250 | 6354 | 48.653 × 180.782 × 157.783 | 50.524 × 206.000 × 206.000 | 48.791 × 205.991 × 205.991 | 103.0 | 103.000022 |
| 759491 | 250 | 6354 | 48.653 × 180.782 × 157.783 | 50.524 × 206.000 × 206.000 | 48.791 × 205.991 × 205.991 | 103.0 | 103.000022 |
| 773650 | 250 | 6354 | 48.653 × 180.782 × 157.783 | 50.524 × 206.000 × 206.000 | 48.791 × 205.991 × 205.991 | 103.0 | 103.000022 |
| 871375 | 250 | 6354 | 48.653 × 180.782 × 157.783 | 50.524 × 206.000 × 206.000 | 48.791 × 205.991 × 205.991 | 103.0 | 103.000022 |

四个 eid 的上述数值**逐项相同**（几何是同一条轮子在 (±250, −244, ±297) 的四份副本），
所以下表只列一次：

| 量 | 值 | 说明 |
| --- | --- | --- |
| STEP VERTEX_POINT | 250 | 顶点盒的样本数 |
| STEP CARTESIAN_POINT（闭包内） | 7763 | 圆/样条的控制点也在这里，但多数不在实体表面上 |
| STEP ADVANCED_FACE | 190 | 面数 |
| STEP CIRCLE / CYLINDRICAL_SURFACE | 136 / 124 | 合计 **260** = `wheel_parts.csv` 的 `circular_features` |
| USD point / triangle / 连通分片 | 6354 / 7900 / 122 | 分片=未焊接三角化的曲面片，非物理零件 |
| STEP 最大圆 | R=103.0 mm，圆心 (±250, −244, ±297)，轴 ±x | 与 USD 圆盘轴 `x` 一致 |
| STEP 次大圆 | R=92.5 mm（在 x 方向偏移 20.5 mm 处） | 对应 USD 最大分片之外的一圈：直径 184.9923 ≈ 2×92.5 |
| USD 点云半径 | min 62.999669 / p50 80.040 / p90 98.705 / **max 103.000022** | 半径分布 63→103 mm 连续 |
| USD 圆盘轴 / 盒中心 | `x` / (250.00196, −243.999993, −296.999993) | 与 STEP 圆心差 ≤ 0.00196 mm |
| USD authored extent | ±[24.395531, 102.995705, 102.995705] | 与世界盒逐位相同 ⇒ 父级无额外缩放 |

### 4.1 关于"point count 与 STEP 顶点数是否一致"

**不一致，而且不应该一致**，这三个数字是三个不同的东西：

- USD 网格点 = 6354（三角化后的顶点，含每条边/面重复点）
- STEP VERTEX_POINT = 250（拓扑顶点，解析曲面不打散）
- STEP CARTESIAN_POINT（闭包内）= 7763（含圆/样条的控制点）—— 与网格点也不等

另外需要更正任务书里的一处口径：`wheel_parts.csv` 那一列的 `260` **不是顶点数**，
它是 `circular_features`（CIRCLE 136 + CYLINDRICAL_SURFACE 124 = 260，本轮已逐项复现）。
拿 260 去和 USD 的 point count 比较会得到无意义的结论。

## 5. 差异的量化（每个数字都能从上面两张表重算）

| 关系 | 计算 | 结果 |
| --- | --- | --- |
| 半径一致性 | 103.000022 − 103.0 | **+2.2e-5 mm**（网格内接圆，量级符合三角化） |
| 轮平面内 USD vs 解析包络 | 205.991409 − 206.000 | **−0.008591 mm**（弦高差） |
| 轮平面内 USD vs 顶点盒 | 205.991409 − 157.7827 | +48.208709 mm（这就是"差异"，且是方法差） |
| 轮平面内括号宽度 | 206.000 − 157.7827 | 48.2173 mm（USD 落在括号内 ⇒ True） |
| 薄轴 | 48.791061 − 48.6527 | +0.138361 mm；括号 = [48.6527, 50.5238] |
| 圆心一致性 | USD 盒中心 − STEP 圆心 | [0.00196, 7e-6, 7e-6] mm |

关键是薄轴：**48.653（STEP 顶点）与 48.791（USD 网格）只差 0.14 mm**。
如果差异是单位/缩放问题，薄轴不可能几乎重合；如果映射错配到别的实体，薄轴与圆心都不可能对齐。

## 6. 明确不予确认的内容（防止下游误引用）

- **轮径 UNKNOWN。** 只能说：这 4 个 `NONE_*` solid 内部存在半径 103.0 mm 的圆形几何，
  且转换后的 mesh 点云在同一轴上 r_max = 103.000022 mm。这**不等于**"轮径 = 206 mm"，
  因为无法从 bbox 判断半径 103.0 的圆柱面就是滚动面（也可能是法兰/挡边）。
- **轴心、轮距 UNKNOWN。** 本轮只测了每个 prim 自身的盒子与半径，没有建立四轮几何中心到
  chassis frame 的变换，也没有与实物核对。
- **"网格把多个实体合并"未被证实也未被排除（UNKNOWN）。**
  122 个连通分片来自未焊接的三角化（各 BREP 面独立打点），不能当作零件数；
  反过来，bbox 也无法证明 mesh 只含一个物理零件。
- **127 行/282 对的映射覆盖率不构成任何机械身份结论**（与 Issue #13 的措辞一致）。

## 7. 可复现命令、真实输出与退出码

### 7.1 环境

见 `logs/env_provenance.txt`（由命令生成，非手写）：

```
local_python: Python 3.11.9
remote_host=jxxy  remote_user=dgut  Python 3.12.13
pxr (Usd.GetVersion) = (0, 25, 11)   # 见 usd_remeasure.json 的 pxr_version 字段
local_step_sha256: 388cb3653a864066a35a54065f71938635c7e9a221cb4b897b31fbd44ff266c5
3b9962913ec517637df2a57df58acb5650fe995b437088539e26f00912af74b7  /home/T7/dgut/runs/cad_dev/usd/M1-4WD4S_model2601_dev.usda
460ebbb0d2a1faea3f81792fe67f5334d98fb53e62b70bf95d1fdbbe360ec096  /home/dgut/runs/wheels/mapping.json
388cb3653a864066a35a54065f71938635c7e9a221cb4b897b31fbd44ff266c5  /home/T7/dgut/robot_sim/assets/robots/badminton_robot/morph_one/cad/M1-4WD4S_model2601.step
```

从 0 开始复现 STEP 副本（资源不落本地，只临时放到 gitignored 目录）：

```
scp dgut@172.31.68.251:/home/T7/dgut/robot_sim/assets/robots/badminton_robot/morph_one/cad/M1-4WD4S_model2601.step _scratch_downloads/
```

### 7.2 STEP 侧（本地，纯 CPU，无 CAD 内核）

```
python scripts/simulation/p1_step_solid_surface_bbox.py \
  --step _scratch_downloads/M1-4WD4S_model2601.step \
  --eids 435700,759491,773650,871375 \
  --out-json outputs/simulation/rebot_b601_rs/morph_one/p1_wheel_identity/step_remeasure.json \
  --out-csv  outputs/simulation/rebot_b601_rs/morph_one/p1_wheel_identity/step_remeasure.csv
```

真实输出（`logs/step_remeasure.log`，EXIT=0，ELAPSED_S=43.3）：

```
STEP file      : <local repo>/_scratch_downloads/M1-4WD4S_model2601.step (115496178 bytes)
                 （日志里该行路径含中文目录，被终端码页显示成乱码；字节数与 sha256 见 env_provenance.txt）
file_schema    : CONFIG_CONTROL_DESIGN
declared units : {"si_unit_tokens": [" $, .RADIAN. ", " $, .STERADIAN. ", " .MILLI., .METRE. "], "conversion_based_units": [], "global_unit_context": [" ( #120813, #136850, #699449 "], "file_schema": "CONFIG_CONTROL_DESIGN"}
parse warnings : 0

eid      verts  circ  cyl  maxR_mm  vertex bbox (mm)              envelope bbox (mm)
------------------------------------------------------------------------------------
435700   250    136   124  103.0    48.653 x 180.782 x 157.783     50.524 x 206.000 x 206.000
759491   250    136   124  103.0    48.653 x 180.782 x 157.783     50.524 x 206.000 x 206.000
773650   250    136   124  103.0    48.653 x 180.782 x 157.783     50.524 x 206.000 x 206.000
871375   250    136   124  103.0    48.653 x 180.782 x 157.783     50.524 x 206.000 x 206.000
# EXIT=0
```

### 7.3 USD 侧（GPU 主机，只用 pxr，未启动 Isaac Sim、未触碰 torch.cuda）

```
scp scripts/simulation/p1_usd_wheel_prim_measure.py dgut@172.31.68.251:/home/dgut/runs/p1_wheel_identity/
ssh dgut@172.31.68.251 'cd /home/dgut/runs/p1_wheel_identity && \
  /home/T7/ojh/robot_sim/env_isaaclab/bin/python p1_usd_wheel_prim_measure.py \
  --usda /home/T7/dgut/runs/cad_dev/usd/M1-4WD4S_model2601_dev.usda \
  --mapping /home/dgut/runs/wheels/mapping.json \
  --eids 435700,759491,773650,871375 \
  --out-json p1_usd_wheel_prim_measure.json --out-csv p1_usd_wheel_prim_measure.csv; echo REMOTE_EXIT=$?'
```

真实输出（`logs/usd_remeasure.log`，SSH_EXIT=0 / REMOTE_EXIT=0）：

```
usda                 : /home/T7/dgut/runs/cad_dev/usd/M1-4WD4S_model2601_dev.usda
stage metersPerUnit  : 0.001
stage upAxis         : Z
mapping pairs        : 282 (unique eids 282)

eid      pts     faces  shells  world bbox (mm)                  r_max_mm  axis
-------------------------------------------------------------------------------
435700   6354    7900   122     48.791 x 205.991 x 205.991       103.000022 x
759491   6354    7900   122     48.791 x 205.991 x 205.991       103.000022 x
773650   6354    7900   122     48.791 x 205.991 x 205.991       103.000022 x
871375   6354    7900   122     48.791 x 205.991 x 205.991       103.000022 x
REMOTE_EXIT=0
```

### 7.4 对照表与假设裁定

```
python scripts/simulation/p1_wheel_identity_compare.py \
  --step-json outputs/.../p1_wheel_identity/step_remeasure.json \
  --usd-json  outputs/.../p1_wheel_identity/usd_remeasure.json \
  --out-json  outputs/.../p1_wheel_identity/comparison.json \
  --out-csv   outputs/.../p1_wheel_identity/comparison.csv
```

真实输出（`logs/comparison.log`，EXIT=0）：

```
eid      stepverts usdpts  stepVertex bbox (mm)             envelope bbox (mm)             usd bbox (mm)                    Rstep    Rusd
-----------------------------------------------------------------------------------------------------------------------------------------
435700   250       6354    48.653 x 180.782 x 157.783      50.524 x 206.000 x 206.000     48.791 x 205.991 x 205.991       103.0    103.000022
...

hypothesis verdicts:
  unit_mismatch                      FAIL
  bbox_computation_method            PASS
  merged_multiple_entities           UNKNOWN
  mapping_candidate_mismatch         FAIL
  hub_vs_tyre_two_entities           UNKNOWN
  overall                            PARTIAL
# EXIT=0
```

`comparison.json` 里每个判定都带 `evidence`（数值）与 `reasoning`（为什么这样判），
不是写死的字符串：例如 `bbox_computation_method` 要求"USD 落在 STEP 括号内"且"括号宽度 > 1 mm"同时成立。

### 7.5 回归：本轮能否重现历史 `wheel_parts.csv`

```
python scripts/simulation/p1_check_wheel_parts_csv.py --rev 53d71ff \
  --csv outputs/.../p1_wheel_identity/step_remeasure.csv \
  --json outputs/.../p1_wheel_identity/step_remeasure.json \
  --repo-root . --eids 435700,759491,773650,871375
```

真实输出（`logs/crosscheck.log`，EXIT=0）：

```
435700  size old=[48.653, 180.782, 157.783] new=[48.653, 180.782, 157.783] equal=True
      circular_features old=260  CIRCLE=136 + CYLINDRICAL_SURFACE=124 = 260  equal=True
      max_circle_radius old=103.0 new=103.0 equal=True
      this round adds: vertex_points=250 cartesian_points=7763 envelope_bbox=[50.5238, 206.0, 206.0]
...
CROSSCHECK PASS
# EXIT=0
```

这四行同时说明：历史 CSV 的 48.653/180.782/157.783 与本轮顶点盒**逐位相同**，
所以"180.782 是顶点盒"不是推测，而是可重放的复现结果。

## 8. 未解决 / 下一步（不建议跳过）

1. **滚动面判定**：需要区分半径 103.0 的圆柱面是轮胎滚动面还是挡边/法兰。
   建议用 CAD 内核（或 pxr 面法线+面积统计）按曲面分组，看外圆柱面是否为最大面积面且覆盖整周。
   在此之前轮径保持 UNKNOWN。
2. **物理轮径的唯一可信来源**：实物测量（卡尺/图纸）或厂家规格。本轮所有数字都是 CAD 侧。
3. **网格合并问题**：若要让"一个 prim = 一个物理零件"成立，需要按 BREP 面→mesh subset
   反查（`UsdGeom.Subset` 或 face index 映射），本轮未做。
4. **轴心/轮距**：需要把四个 prim 中心 (±250.002, −244.0, ±297.0) mm 与
   `X+90°` → chassis frame → world 的既定变换一起验证（P1 的另一条 checklist），本轮未做。

## 9. 本轮工件

| 路径 | 作用 |
| --- | --- |
| `scripts/simulation/p1_step_solid_surface_bbox.py` | STEP 侧顶点盒/曲面包络/点云盒，纯 stdlib |
| `scripts/simulation/p1_usd_wheel_prim_measure.py` | USD 侧 pxr 测量（metersPerUnit、点数、面数、分片、半径分布） |
| `scripts/simulation/p1_wheel_identity_compare.py` | 对照表 + 假设裁定（PASS/FAIL/UNKNOWN 由数值计算） |
| `scripts/simulation/p1_check_wheel_parts_csv.py` | 与历史 `wheel_parts.csv` 的逐列回归 |
| `outputs/simulation/rebot_b601_rs/morph_one/p1_wheel_identity/step_remeasure.{json,csv}` | STEP 复测原始输出 |
| `outputs/simulation/rebot_b601_rs/morph_one/p1_wheel_identity/usd_remeasure.{json,csv}` | USD 复测原始输出（含 pxr 版本、矩阵、extent） |
| `outputs/simulation/rebot_b601_rs/morph_one/p1_wheel_identity/comparison.{json,csv}` | 对照表与裁定 |
| `outputs/simulation/rebot_b601_rs/morph_one/p1_wheel_identity/logs/` | 5 份原始日志（含命令、EXIT、耗时、环境 sha256） |
| `outputs/simulation/rebot_b601_rs/morph_one/p1_wheel_identity/MANIFEST.sha256` | 上述工件的 sha256 |

未在本分支提交任何资源文件（STEP/USDA/mapping 均在远端），本地 `_scratch_downloads/` 已被 `.gitignore` 忽略。
