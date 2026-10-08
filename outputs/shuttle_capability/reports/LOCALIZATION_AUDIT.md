# LOCALIZATION AUDIT — YOLO26 V1 羽毛球超小目标检测

> 本文档由 `tools/audit_localization_yolo26_v1.py` 自动生成（表格与判定规则）+ 人工结论（第 0 节，需手动维护）。
> 只读审计：不训练、不改模型/配置/数据/标签/采样器/增强；只读现有 checkpoint 与标注数据；无标注集不参与 bbox 定位。

## 0. 结论（人工分析）

> **人工结论（2026-10-01）**。本节由人工撰写，`tools/audit_localization_yolo26_v1.py` 重跑会清空本节（工具只生成 §1–§10）。

### 0.1 一句话结论

**Stage B（`b_best` = stageB_best_e10）相对 Stage A（`a_best` = stageA_best e6）的 mAP50-95 提升，主要来自 Recall，而不是真正的 bbox 定位改善。**

- 在 `controlled_capability/images`（**固定开发留出集**，也正是产生 `FROZEN_TEST_FAVORS_B_BEST_E10_OVER_A_BEST` 的那个集合）上：
  mAP50-95 +0.0712，几乎全部由召回贡献（R 0.1734→0.2527，TP 359→523）；同一批 TP 的**框质量反而变差**：
  IoU mean −0.0163 / median −0.0065，中心误差中位数 0.698→1.111 px640（P90 6.25→26.99），宽度相对误差中位数 +0.0360，
  IoU∈[0.50,0.60) 的“勉强 TP” 7→48 → **CASE A（召回驱动）**。
- 在 `val`（域内开发集）上方向相反：AP75 +0.0176、AP90 +0.0244、IoU 中位数 +0.0114、中心误差 −0.0934 px640，
  但召回 −0.0798、AP50 −0.0216，净 mAP50-95 仅 +0.0019（噪声级）→ **CASE B（真实定位改善）但以召回为代价，净值不足以替换基线**。
- `challenge_test/images` 只有 194 个 GT（TP 9→14），**样本量不足以下结论**。
- `b_last`(e70) 不是候选：recall 全面崩塌（val 0.6031、controlled 0.2449、challenge 0.1649）。

### 0.2 三个集合的主指标（conf_op 0.25；AP 曲线 conf 0.001；IoU≥0.50 为一对一 TP）

| 集合 | ckpt | TP | Recall | AP50 | AP75 | AP90 | AP95 | mAP50-95 | IoU 中位 | 中心误差中位(px640) | 近失 0.30–0.50 |
|---|---|---|---|---|---|---|---|---|---|---|---|
| val (3,260 GT) | a_best | 2802 | 0.8595 | **0.9051** | 0.6855 | 0.0904 | 0.0087 | 0.5907 | 0.8170 | 0.532 | 21 |
| val | b_best | 2542 | 0.7798 | 0.8835 | **0.7031** | **0.1148** | 0.0083 | **0.5926** | **0.8284** | **0.439** | 20 |
| val | b_last | 1966 | 0.6031 | 0.7564 | 0.5898 | 0.0946 | 0.0061 | 0.5044 | 0.8274 | 0.439 | 7 |
| controlled_capability (2,070 GT) | a_best | 359 | 0.1734 | 0.3659 | 0.2413 | 0.0647 | 0.0078 | 0.2303 | **0.8520** | **0.698** | 1 |
| controlled_capability | b_best | 523 | **0.2527** | **0.4366** | **0.3472** | **0.0894** | **0.0128** | **0.3015** | 0.8455 | 1.111 | **48** |
| controlled_capability | b_last | 507 | 0.2449 | 0.3212 | 0.2672 | 0.1201 | 0.0126 | 0.2342 | 0.8724 | 1.503 | 7 |
| challenge_test (194 GT) | a_best | 9 | 0.0464 | 0.1840 | 0.0889 | 0.0157 | 0.000054 | 0.0972 | 0.7920 | 0.415 | 0 |
| challenge_test | b_best | 14 | 0.0722 | 0.2339 | 0.0963 | 0.0155 | 0.0 | 0.1170 | 0.8131 | 0.404 | 2 |
| challenge_test | b_last | 32 | 0.1649 | 0.2234 | 0.1410 | 0.0214 | 0.0 | 0.1328 | 0.7140 | **40.047** | 2 |

`b_best` 在 `controlled_capability` 上的代价是精度：P 0.8842→0.4847（`detections_op` 406→1079，FP 47→556）。

### 0.3 尺寸分桶（equiv_size_640；要求对比 6–8 / 8–12 / 12–16，附最相关桶）

| 集合 | 桶 | GT | R(a_best) | R(b_best) | ΔR | IoU med a→b | 中心px640 a→b | AP50 a→b |
|---|---|---|---|---|---|---|---|---|
| val | 6-8 | 251 | 0.7928 | 0.7968 | +0.0040 | 0.7920→0.7959 | 0.514→0.461 | 0.8451→0.8699 |
| val | 8-12 | 1724 | 0.8846 | 0.8573 | −0.0273 | 0.8149→0.8258 | 0.482→0.395 | 0.9302→0.9325 |
| val | 12-16 | 755 | 0.9033 | 0.7351 | **−0.1682** | 0.8269→0.8448 | 0.586→0.464 | 0.9377→0.8899 |
| controlled_capability | 6-8 | 54 | 0.2037 | 0.1667 | **−0.0370** | 0.7901→0.7593 | 0.214→0.307 | 0.2877→0.3071 |
| controlled_capability | 8-12 | 405 | 0.0765 | 0.0840 | +0.0074 | 0.8543→0.8443 | 0.457→0.492 | 0.2516→0.3334 |
| controlled_capability | 12-16 | 825 | 0.1261 | 0.1636 | +0.0376 | 0.8965→0.8795 | 0.427→0.456 | 0.3161→0.4566 |
| controlled_capability | <4 / 4-6 | 160 / 106 | 0.0688 / 0.1321 | 0.0500 / 0.1226 | −0.0188 / −0.0094 | 0.7592→0.7323 / 0.7172→0.7655 | 0.269→0.377 / 0.432→0.481 | 0.0966→0.0661 / 0.1967→0.1799 |
| controlled_capability | 32-64 / >64 | 120 / 240 | 0.5917 / 0.2333 | 0.8500 / 0.6667 | **+0.2583 / +0.4333** | 0.7839→0.8219 / 0.7913→0.7482 | 2.914→2.118 / 7.608→12.609 | 0.8625→0.9375 / 0.6399→0.6156 |

要点：**B 在固定留出集上的召回增益集中在“大目标”（32–64 px +0.258、>64 px +0.433），而最需要解决的 <4 / 4–6 / 6–8 px 三个超小目标桶的召回反而下降**；
val 上则是 12–16/16–24 段召回大幅下降（−0.168/−0.224）、同时 IoU 中位数在该段上升（定位更准但找回更少）。

### 0.4 主导误差来源（有数据的排序）

1. **漏检（Recall）是第一大来源**：`controlled_capability` 上 class D（无检测或最佳 IoU<0.10）占 1707/2070 = 82.5%（b_best 1543/2070 = 74.5%）；val 上 D 占 13.9%（b_best 21.9%）。
2. **框尺寸误差 > 中心误差**：val 上 width_ratio 均值 0.9978/1.0003、**height_ratio 均值 1.0919/1.0918**（预测框系统性比 GT 高约 9.2%，三个 checkpoint 一致），
   width_rel 中位 0.052–0.059、height_rel 中位 0.094–0.099、area_ratio 中位 1.084–1.091。
   量化上限：若中心完美，平均 IoU ≤ 1/(0.9978×1.0919) = **0.918**，与观测到的 AP85 0.285 / AP90 0.090 / AP95 0.009 的断崖一致 → **AP90/AP95 上不去主要是“框偏大”，不是“框偏心”**。
3. **中心误差在 TP 上很小**：归一化中心误差中位 0.0411–0.0489（val）、0.0295–0.0452（controlled）；近失 0.30–0.50 的数量在 val 只有 20–21 个（占 GT 0.6%）。
   但 b_best 在 `controlled_capability` 上把近失从 1 个推到 **48 个**、中心误差 P90 从 6.25 推到 26.99 px640 → **B 新召回的目标里有一批框明显更松**。
4. **超小目标接近几何极限**：<4 px 桶在 val 与 fixed set 上召回 0.042–0.069（b_best 更差：0.042 / 0.050）；4 px 物体要达 IoU 0.5，框的尺寸/中心误差必须同时小于约 15%/0.5 px，已达标注与像素分辨率极限。
5. **标签噪声**：本轮**没有**支持它是主导因素的证据（class C 仅 0–4 个；近失总量很小），但也**无法量化**（没有双标/复核数据），故不作为结论。

### 0.5 可信度与口径注意（避免误读）

- **IoU 中位数是“条件在 TP 上”的量**，会随召回变化产生选择偏差：`b_last` 在 `controlled_capability` 上 IoU 中位最高（0.8724）却召回最低（0.2449），
  在 `challenge_test` 上它把 24 个 >64 px 目标全部“找回”（该桶 IoU 中位仅 0.667、中心误差中位 45.97 px640；其 32 个 TP 整体中心误差中位 40.05 px640）——
  大框即使中心偏 40 px 仍能满足 IoU≥0.5 → 单看 IoU 中位数会得出错误结论，必须与 AP-vs-IoU 曲线、分桶召回一起读。
- `val` 是 `b_best` 的**选点集**（best epoch 由 val fitness 选出），因此 val 指标对 B 有利；即便如此，B 的 val AP50 与召回仍低于 A。
- `challenge_test` 仅 194 GT，A/B 的差异（9 vs 14 TP）不足以支撑结论。
- 术语：这三个集合是 **fixed evaluation set / development holdout**，不是“完全未参与模型选择的最终测试集”（见 `docs/LOCALIZATION_AUDIT.md` §7、`management/ISSUES.md` ISSUE-029）。

### 0.6 自证：口径一致性交叉校验

| 校验项 | 本轮结果 | 既有记录 | 结论 |
|---|---|---|---|
| a_best / val | TP 2802, R 0.859509, AP50 0.905095, mAP50-95 0.590744, GT 3260, 4413 图 | `metrics/size_bucket_metrics_val.csv` stageA_best OVERALL: 3260, 2802, 458, 0.859509, 0.905095, 0.590744 | 逐位一致 |
| a_best / controlled_capability | TP 359, FP 47, P 0.884236, R 0.173430, AP50 0.365888, mAP50-95 0.230295 | `reports/FROZEN_TEST_EVAL.md`: 359, 47, 0.8842, 0.1734, 0.3659, 0.2303 | 逐位一致 |
| b_best / controlled_capability | TP 523, AP50 0.436648, mAP50-95 0.301460 | `metrics/frozen_test_matrix.csv` / FROZEN_TEST_EVAL: 523, 0.4366, 0.3015 | 一致 |
| b_last / controlled_capability | TP 507, AP50 0.321176 | 同表: 507, 0.3212 | 一致 |
| a/b/b_last / challenge_test | 9 / 14 / 32 TP；AP50 0.184045 / 0.233914 / 0.223368 | 同表: 9/14/32；0.1840 / 0.2339 / 0.2234 | 一致 |

产物 sha256（前 16 位，`outputs/shuttle_capability/` 下）：

| 文件 | bytes | sha256(16) |
|---|---|---|
| `metrics/localization_ap_by_iou.csv` | 1742 | `c781a1fa68b6619c` |
| `metrics/localization_iou_distribution.csv` | 1208 | `3d3852cf41fd9766` |
| `metrics/localization_error_summary.csv` | 7680 | `aa0d0412b409c2b9` |
| `metrics/localization_size_buckets.csv` | 9432 | `caae073c58cb861b` |
| `metrics/localization_checkpoint_compare.csv` | 7067 | `afa37f5fc5de0d2b` |
| `metrics/localization_near_miss.csv` | 20996 | `21129e8580e83fc5` |
| `metrics/localization_audit.json` | 23,049,205 | `ad946fa845d2f7cc` |
| `localization_examples/index.csv` | 105,687 | `8e74560d36a77f7c` |

示例图 348 张（170.8 MB）：`localization_examples/{low_iou,center_error,width_error,height_error,near_miss}/{set}/`，全部为 `b_best`；
`challenge_test` 只有 14 个 TP，故 4 类各 14 张、近失 2 张。要同时导出 `a_best` 的示例：`--examples-ckpt a_best`（预测已缓存，约 3 分钟）。
预测缓存：`_scratch_localization_audit/*.json`（临时，可删；删除后用 §10 的同一条命令可完整重生成）。

### 0.7 结论与纪律

- **回答本次任务的核心问题：Stage B 的 mAP50-95 提升主要来自 Recall，不是真正的 bbox localization 改善。**
  在固定留出集上，召回 +0.0792 换来 IoU 质量下降（mean −0.0163）、勉强 TP（IoU<0.6）7→48、中心误差中位 0.698→1.111 px640；
  在 val 上确有真实定位改善（AP75/AP90/IoU 中位上升、中心误差下降）但召回 −0.0798，净值 mAP50-95 +0.0019 属噪声。
- 因此：**不替换基线（保留 Stage A），不因该 +0.0712 就升级 B**；`b_best` 只在“多找回大目标”这一条上领先，且以精度（P 0.88→0.48）和超小目标召回为代价。
- **本轮不启动 Stage C**（用户约束）。若后续 Stage C 依据本报告调参、选阈值或做后处理，最终判断必须另用一个**全新的、未被看过的 holdout**，否则无法与选择偏差区分。
- 本轮只新增/修改评估工具、测试与报告；**未修改任何模型、训练配置、数据、标签、loss、optimizer、augmentation，也未删除任何 checkpoint**。

## 1. 口径与数据集

- AP：按置信度排序的标准 PR 曲线（all-point），单图 AP 再按 GT 数做加权平均（与仓库既有 val / fixed evaluation set 报告同口径）。
- `conf_op = 0.25`（TP/FP/FN/精度/召回的工作点）；AP 曲线用 `conf_floor = 0.0010` 的全部检测。
- TP 定义：IoU >= 0.50 的一对一匹配；近失匹配 0.30-0.50 记为 class B，**从不计入 TP**。
- `equiv_size_640 = sqrt(w_px*h_px) * 640/max(W,H)`（半开区间分桶），不按 1024 输入像素分桶。
- `normalized_center_error = center_error / sqrt(gt_w*gt_h)`；`center_err_px640` 是把原始像素误差折算到 640 短边输入后的等效像素。
- 术语：这些固定评估集是 `fixed evaluation set` / `development holdout`；若后续 Stage C 依据本报告调参，最终判断必须另用全新的未见 holdout。

评估集（本文件只列出有标注、可用于 bbox 定位的集合）：

| set | images | GT |
|---|---|---|
| challenge_test/images | 227 | 194 |
| controlled_capability/images | 2070 | 2070 |
| val | 4413 | 3260 |

## 2. 检查点指纹

| name | epoch | role | sha256(前16) | bytes | path |
|---|---|---|---|---|---|
| a_best | 6 | baseline | 3214aaf15fd9d9a2 | 20125499 | D:/_eth_dl/ckpts/stageA_best.pt |
| b_best | 10 | candidate | 419a3eefa7e4ca44 | 20136763 | D:/_eth_dl/ckpts/stageB_best_e10_stripped.pt |
| b_last | 70 | overfit_control | 42f0043a1b902b40 | 20136763 | D:/_eth_dl/ckpts/stageB_last_e70.pt |

## 3. AP50..AP95（标准 PR 曲线）

| checkpoint | set | GT | TP@0.5 | FP@0.5 | R@0.5 | P@op | AP50 | AP55 | AP60 | AP65 | AP70 | AP75 | AP80 | AP85 | AP90 | AP95 | mAP50-95 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| a_best | challenge_test/images | 194 | 9 | 4 | 0.0464 | 0.6923 | 0.1840 | 0.1781 | 0.1557 | 0.1364 | 0.1077 | 0.0889 | 0.0659 | 0.0395 | 0.0157 | 0.0001 | 0.0972 |
| a_best | controlled_capability/images | 2070 | 359 | 47 | 0.1734 | 0.8842 | 0.3659 | 0.3579 | 0.3456 | 0.3199 | 0.2849 | 0.2413 | 0.1871 | 0.1280 | 0.0647 | 0.0078 | 0.2303 |
| a_best | val | 3260 | 2802 | 1269 | 0.8595 | 0.6883 | 0.9051 | 0.8983 | 0.8809 | 0.8488 | 0.7914 | 0.6855 | 0.5135 | 0.2848 | 0.0904 | 0.0087 | 0.5907 |
| b_best | challenge_test/images | 194 | 14 | 10 | 0.0722 | 0.5833 | 0.2339 | 0.2229 | 0.1903 | 0.1540 | 0.1255 | 0.0963 | 0.0805 | 0.0515 | 0.0155 | 0.0000 | 0.1170 |
| b_best | controlled_capability/images | 2070 | 523 | 556 | 0.2527 | 0.4847 | 0.4366 | 0.4289 | 0.4186 | 0.4031 | 0.3799 | 0.3472 | 0.2942 | 0.2038 | 0.0894 | 0.0128 | 0.3015 |
| b_best | val | 3260 | 2542 | 1168 | 0.7798 | 0.6852 | 0.8835 | 0.8775 | 0.8630 | 0.8343 | 0.7819 | 0.7031 | 0.5404 | 0.3193 | 0.1148 | 0.0083 | 0.5926 |
| b_last | challenge_test/images | 194 | 32 | 65 | 0.1649 | 0.3299 | 0.2234 | 0.2131 | 0.2031 | 0.1947 | 0.1652 | 0.1410 | 0.0976 | 0.0684 | 0.0214 | 0.0000 | 0.1328 |
| b_last | controlled_capability/images | 2070 | 507 | 211 | 0.2449 | 0.7061 | 0.3212 | 0.3174 | 0.3112 | 0.3012 | 0.2877 | 0.2672 | 0.2251 | 0.1787 | 0.1201 | 0.0126 | 0.2342 |
| b_last | val | 3260 | 1966 | 806 | 0.6031 | 0.7092 | 0.7564 | 0.7509 | 0.7420 | 0.7204 | 0.6749 | 0.5898 | 0.4497 | 0.2589 | 0.0946 | 0.0061 | 0.5044 |

## 4. IoU 分布（IoU >= 0.5 的 TP）

| checkpoint | set | n_tp | mean | median | std | P10 | P25 | P50 | P75 | P90 | bin 0.50-0.60 | bin 0.60-0.70 | bin 0.70-0.80 | bin 0.80-0.90 | bin 0.90-0.95 | bin >=0.95 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| a_best | challenge_test/images | 9 | 0.7782 | 0.7920 | 0.1111 | 0.5890 | 0.7718 | 0.7920 | 0.8802 | 0.8881 | 2 | 0 | 3 | 3 | 1 | 0 |
| a_best | controlled_capability/images | 359 | 0.8284 | 0.8520 | 0.0968 | 0.6852 | 0.7579 | 0.8520 | 0.9074 | 0.9354 | 7 | 40 | 77 | 130 | 90 | 15 |
| a_best | val | 2802 | 0.8050 | 0.8170 | 0.0832 | 0.6923 | 0.7584 | 0.8170 | 0.8642 | 0.9011 | 64 | 260 | 847 | 1339 | 264 | 28 |
| b_best | challenge_test/images | 14 | 0.7866 | 0.8131 | 0.1105 | 0.6140 | 0.7191 | 0.8131 | 0.8851 | 0.8967 | 1 | 2 | 3 | 7 | 1 | 0 |
| b_best | controlled_capability/images | 523 | 0.8121 | 0.8455 | 0.1162 | 0.6175 | 0.7483 | 0.8455 | 0.8976 | 0.9322 | 48 | 40 | 94 | 216 | 105 | 20 |
| b_best | val | 2542 | 0.8185 | 0.8284 | 0.0789 | 0.7090 | 0.7734 | 0.8284 | 0.8765 | 0.9096 | 34 | 192 | 672 | 1290 | 328 | 26 |
| b_last | challenge_test/images | 32 | 0.7123 | 0.7140 | 0.1181 | 0.5311 | 0.6432 | 0.7140 | 0.7876 | 0.8564 | 6 | 10 | 8 | 7 | 1 | 0 |
| b_last | controlled_capability/images | 507 | 0.8348 | 0.8724 | 0.1083 | 0.6830 | 0.7707 | 0.8724 | 0.9205 | 0.9399 | 24 | 39 | 106 | 125 | 192 | 21 |
| b_last | val | 1966 | 0.8179 | 0.8274 | 0.0761 | 0.7193 | 0.7745 | 0.8274 | 0.8716 | 0.9102 | 23 | 117 | 572 | 991 | 245 | 18 |

## 5. 定位误差（IoU >= 0.5 的 TP）

| checkpoint | set | n_tp | center_px mean | center_px median | center_px P90 | center_px640 mean | center_px640 median | center_px640 P90 | norm center median | w_rel median | h_rel median | area_ratio median | w_ratio mean | h_ratio mean |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| a_best | challenge_test/images | 9 | 1.14 | 0.62 | 2.27 | 0.76 | 0.41 | 1.51 | 0.045 | 0.096 | 0.039 | 0.930 | 0.938 | 0.932 |
| a_best | controlled_capability/images | 359 | 4.50 | 1.05 | 12.50 | 2.34 | 0.70 | 6.25 | 0.045 | 0.043 | 0.063 | 1.003 | 0.993 | 1.019 |
| a_best | val | 2802 | 2.06 | 1.59 | 3.99 | 0.69 | 0.53 | 1.33 | 0.049 | 0.059 | 0.095 | 1.084 | 0.998 | 1.092 |
| b_best | challenge_test/images | 14 | 1.15 | 0.61 | 3.28 | 0.77 | 0.40 | 2.19 | 0.052 | 0.117 | 0.079 | 0.929 | 0.920 | 1.002 |
| b_best | controlled_capability/images | 523 | 11.52 | 1.82 | 53.99 | 5.83 | 1.11 | 26.99 | 0.045 | 0.079 | 0.061 | 1.029 | 0.979 | 1.003 |
| b_best | val | 2542 | 1.80 | 1.31 | 3.63 | 0.60 | 0.44 | 1.22 | 0.041 | 0.052 | 0.094 | 1.091 | 1.000 | 1.092 |
| b_last | challenge_test/images | 32 | 113.12 | 112.13 | 198.04 | 40.46 | 40.05 | 70.73 | 0.094 | 0.147 | 0.066 | 0.855 | 0.852 | 0.993 |
| b_last | controlled_capability/images | 507 | 18.91 | 2.94 | 75.68 | 9.50 | 1.50 | 37.84 | 0.029 | 0.058 | 0.052 | 1.038 | 0.995 | 1.019 |
| b_last | val | 1966 | 1.71 | 1.30 | 3.23 | 0.57 | 0.44 | 1.08 | 0.041 | 0.052 | 0.099 | 1.084 | 0.995 | 1.094 |

### 5.1 中心误差与尺寸误差分位数（px / px640 / 归一化 / 相对误差）

| checkpoint | set | metric | mean | median | std | P75 | P90 | P95 |
|---|---|---|---|---|---|---|---|---|
| a_best | challenge_test/images | center_dist_px | 1.1385 | 0.6224 | 1.2380 | 1.5487 | 2.2725 | 3.2850 |
| a_best | challenge_test/images | center_dist_px_640 | 0.7590 | 0.4150 | 0.8253 | 1.0325 | 1.5150 | 2.1900 |
| a_best | challenge_test/images | normalized_center_error | 0.0684 | 0.0445 | 0.0641 | 0.0845 | 0.1259 | 0.1802 |
| a_best | challenge_test/images | center_abs_dx | 0.6838 | 0.2649 | 1.2729 | 0.3155 | 1.2546 | 2.7595 |
| a_best | challenge_test/images | center_abs_dy | 0.6385 | 0.5316 | 0.5771 | 0.6551 | 1.5863 | 1.6628 |
| a_best | challenge_test/images | width_rel_err | 0.1029 | 0.0963 | 0.0773 | 0.1227 | 0.1812 | 0.2292 |
| a_best | challenge_test/images | height_rel_err | 0.0818 | 0.0390 | 0.0860 | 0.0994 | 0.1489 | 0.2259 |
| a_best | challenge_test/images | area_ratio | 0.8762 | 0.9296 | 0.1487 | 0.9694 | 1.0258 | 1.0684 |
| a_best | challenge_test/images | width_ratio_signed | 0.9385 | 0.9605 | 0.1130 | 0.9957 | 1.0910 | 1.0936 |
| a_best | challenge_test/images | height_ratio_signed | 0.9319 | 0.9645 | 0.0972 | 1.0088 | 1.0186 | 1.0288 |
| a_best | challenge_test/images | log_width_ratio | -0.0711 | -0.0403 | 0.1242 | -0.0044 | 0.0871 | 0.0895 |
| a_best | challenge_test/images | log_height_ratio | -0.0766 | -0.0362 | 0.1134 | 0.0088 | 0.0184 | 0.0283 |
| a_best | controlled_capability/images | center_dist_px | 4.4972 | 1.0536 | 7.3007 | 5.6612 | 12.4957 | 23.6717 |
| a_best | controlled_capability/images | center_dist_px_640 | 2.3396 | 0.6982 | 3.6118 | 2.8306 | 6.2479 | 11.8358 |
| a_best | controlled_capability/images | normalized_center_error | 0.0530 | 0.0452 | 0.0334 | 0.0703 | 0.1023 | 0.1202 |
| a_best | controlled_capability/images | center_abs_dx | 2.3907 | 0.6777 | 4.0377 | 2.0828 | 8.2139 | 11.4101 |
| a_best | controlled_capability/images | center_abs_dy | 3.3118 | 0.6616 | 6.3670 | 4.1922 | 7.3493 | 19.4322 |
| a_best | controlled_capability/images | width_rel_err | 0.0678 | 0.0430 | 0.0604 | 0.1052 | 0.1634 | 0.1900 |
| a_best | controlled_capability/images | height_rel_err | 0.0927 | 0.0626 | 0.0813 | 0.1414 | 0.2411 | 0.2626 |
| a_best | controlled_capability/images | area_ratio | 1.0200 | 1.0034 | 0.2044 | 1.1081 | 1.3525 | 1.4345 |
| a_best | controlled_capability/images | width_ratio_signed | 0.9926 | 0.9959 | 0.0905 | 1.0367 | 1.1160 | 1.1426 |
| a_best | controlled_capability/images | height_ratio_signed | 1.0187 | 0.9971 | 0.1218 | 1.0814 | 1.2109 | 1.2592 |
| a_best | controlled_capability/images | log_width_ratio | -0.0117 | -0.0041 | 0.0931 | 0.0360 | 0.1098 | 0.1333 |
| a_best | controlled_capability/images | log_height_ratio | 0.0114 | -0.0029 | 0.1189 | 0.0783 | 0.1914 | 0.2305 |
| a_best | val | center_dist_px | 2.0646 | 1.5873 | 1.9102 | 2.5252 | 3.9859 | 5.4311 |
| a_best | val | center_dist_px_640 | 0.6913 | 0.5319 | 0.6370 | 0.8425 | 1.3350 | 1.8135 |
| a_best | val | normalized_center_error | 0.0581 | 0.0489 | 0.0403 | 0.0758 | 0.1130 | 0.1414 |
| a_best | val | center_abs_dx | 1.5993 | 1.1623 | 1.6149 | 2.0939 | 3.4559 | 4.8578 |
| a_best | val | center_abs_dy | 0.9676 | 0.6911 | 1.3452 | 1.2189 | 1.8991 | 2.5314 |
| a_best | val | width_rel_err | 0.0796 | 0.0591 | 0.0745 | 0.1102 | 0.1735 | 0.2282 |
| a_best | val | height_rel_err | 0.1122 | 0.0954 | 0.0874 | 0.1585 | 0.2272 | 0.2738 |
| a_best | val | area_ratio | 1.0912 | 1.0844 | 0.1723 | 1.1834 | 1.2796 | 1.3687 |
| a_best | val | width_ratio_signed | 0.9978 | 1.0005 | 0.1090 | 1.0604 | 1.1184 | 1.1620 |
| a_best | val | height_ratio_signed | 1.0919 | 1.0871 | 0.1085 | 1.1540 | 1.2251 | 1.2726 |
| a_best | val | log_width_ratio | -0.0083 | 0.0005 | 0.1120 | 0.0586 | 0.1119 | 0.1501 |
| a_best | val | log_height_ratio | 0.0830 | 0.0835 | 0.0989 | 0.1433 | 0.2030 | 0.2411 |
| b_best | challenge_test/images | center_dist_px | 1.1484 | 0.6054 | 1.1912 | 0.8998 | 3.2792 | 3.6528 |
| b_best | challenge_test/images | center_dist_px_640 | 0.7656 | 0.4036 | 0.7941 | 0.5998 | 2.1861 | 2.4352 |
| b_best | challenge_test/images | normalized_center_error | 0.0760 | 0.0523 | 0.0609 | 0.0896 | 0.1789 | 0.1993 |
| b_best | challenge_test/images | center_abs_dx | 1.0210 | 0.4693 | 1.2079 | 0.8289 | 3.1601 | 3.5312 |
| b_best | challenge_test/images | center_abs_dy | 0.3978 | 0.2999 | 0.2793 | 0.5940 | 0.8158 | 0.9054 |
| b_best | challenge_test/images | width_rel_err | 0.1073 | 0.1174 | 0.0858 | 0.1596 | 0.2158 | 0.2353 |
| b_best | challenge_test/images | height_rel_err | 0.0869 | 0.0786 | 0.0558 | 0.0984 | 0.1195 | 0.1729 |
| b_best | challenge_test/images | area_ratio | 0.9261 | 0.9286 | 0.1710 | 1.0782 | 1.1135 | 1.1561 |
| b_best | challenge_test/images | width_ratio_signed | 0.9203 | 0.9214 | 0.1119 | 1.0152 | 1.0271 | 1.0597 |
| b_best | challenge_test/images | height_ratio_signed | 1.0015 | 1.0496 | 0.1032 | 1.0806 | 1.0961 | 1.0982 |
| b_best | challenge_test/images | log_width_ratio | -0.0906 | -0.0829 | 0.1233 | 0.0151 | 0.0267 | 0.0573 |
| b_best | challenge_test/images | log_height_ratio | -0.0043 | 0.0482 | 0.1105 | 0.0775 | 0.0918 | 0.0937 |
| b_best | controlled_capability/images | center_dist_px | 11.5174 | 1.8193 | 20.1170 | 7.8214 | 53.9865 | 61.5666 |
| b_best | controlled_capability/images | center_dist_px_640 | 5.8251 | 1.1108 | 10.0246 | 3.9107 | 26.9932 | 30.7833 |
| b_best | controlled_capability/images | normalized_center_error | 0.0584 | 0.0451 | 0.0434 | 0.0782 | 0.1270 | 0.1557 |
| b_best | controlled_capability/images | center_abs_dx | 9.3777 | 0.8656 | 17.7303 | 6.1249 | 44.4173 | 54.8821 |
| b_best | controlled_capability/images | center_abs_dy | 5.4161 | 1.0092 | 10.2814 | 4.6057 | 17.5884 | 31.0226 |
| b_best | controlled_capability/images | width_rel_err | 0.0969 | 0.0789 | 0.0821 | 0.1356 | 0.2410 | 0.2708 |
| b_best | controlled_capability/images | height_rel_err | 0.0839 | 0.0614 | 0.0715 | 0.1160 | 0.1983 | 0.2332 |
| b_best | controlled_capability/images | area_ratio | 0.9941 | 1.0286 | 0.2190 | 1.1334 | 1.2326 | 1.3310 |
| b_best | controlled_capability/images | width_ratio_signed | 0.9793 | 0.9995 | 0.1253 | 1.0707 | 1.1189 | 1.1487 |
| b_best | controlled_capability/images | height_ratio_signed | 1.0027 | 1.0238 | 0.1102 | 1.0684 | 1.1180 | 1.1713 |
| b_best | controlled_capability/images | log_width_ratio | -0.0296 | -0.0005 | 0.1347 | 0.0683 | 0.1123 | 0.1386 |
| b_best | controlled_capability/images | log_height_ratio | -0.0037 | 0.0235 | 0.1154 | 0.0662 | 0.1116 | 0.1581 |
| b_best | val | center_dist_px | 1.8005 | 1.3059 | 1.7804 | 2.1000 | 3.6313 | 4.9501 |
| b_best | val | center_dist_px_640 | 0.6043 | 0.4386 | 0.5933 | 0.7045 | 1.2167 | 1.6500 |
| b_best | val | normalized_center_error | 0.0516 | 0.0412 | 0.0389 | 0.0677 | 0.1030 | 0.1306 |
| b_best | val | center_abs_dx | 1.3888 | 0.9527 | 1.5945 | 1.7161 | 2.9925 | 4.3770 |
| b_best | val | center_abs_dy | 0.8373 | 0.5746 | 1.1133 | 1.0071 | 1.6822 | 2.2306 |
| b_best | val | width_rel_err | 0.0741 | 0.0524 | 0.0709 | 0.1029 | 0.1667 | 0.2174 |
| b_best | val | height_rel_err | 0.1070 | 0.0944 | 0.0777 | 0.1488 | 0.2071 | 0.2466 |
| b_best | val | area_ratio | 1.0934 | 1.0906 | 0.1573 | 1.1739 | 1.2709 | 1.3311 |
| b_best | val | width_ratio_signed | 1.0003 | 1.0025 | 0.1025 | 1.0546 | 1.1137 | 1.1579 |
| b_best | val | height_ratio_signed | 1.0918 | 1.0890 | 0.0952 | 1.1464 | 1.2054 | 1.2457 |
| b_best | val | log_width_ratio | -0.0051 | 0.0025 | 0.1049 | 0.0532 | 0.1077 | 0.1466 |
| b_best | val | log_height_ratio | 0.0840 | 0.0853 | 0.0874 | 0.1366 | 0.1868 | 0.2197 |
| b_last | challenge_test/images | center_dist_px | 113.1211 | 112.1321 | 80.1323 | 179.3453 | 198.0406 | 240.2839 |
| b_last | challenge_test/images | center_dist_px_640 | 40.4570 | 40.0472 | 28.5397 | 64.0519 | 70.7288 | 85.8157 |
| b_last | challenge_test/images | normalized_center_error | 0.0965 | 0.0937 | 0.0423 | 0.1257 | 0.1523 | 0.1721 |
| b_last | challenge_test/images | center_abs_dx | 85.8254 | 73.0835 | 73.7096 | 150.6755 | 180.0318 | 200.5780 |
| b_last | challenge_test/images | center_abs_dy | 62.3368 | 63.6601 | 50.3248 | 101.9139 | 118.3888 | 138.3155 |
| b_last | challenge_test/images | width_rel_err | 0.1787 | 0.1466 | 0.1301 | 0.2875 | 0.3617 | 0.3923 |
| b_last | challenge_test/images | height_rel_err | 0.0760 | 0.0663 | 0.0619 | 0.0915 | 0.1735 | 0.1840 |
| b_last | challenge_test/images | area_ratio | 0.8575 | 0.8552 | 0.2280 | 1.0901 | 1.1506 | 1.1699 |
| b_last | challenge_test/images | width_ratio_signed | 0.8517 | 0.8534 | 0.1639 | 0.9932 | 1.0806 | 1.0947 |
| b_last | challenge_test/images | height_ratio_signed | 0.9932 | 0.9951 | 0.0978 | 1.0763 | 1.0926 | 1.1392 |
| b_last | challenge_test/images | log_width_ratio | -0.1798 | -0.1589 | 0.1987 | -0.0068 | 0.0775 | 0.0904 |
| b_last | challenge_test/images | log_height_ratio | -0.0119 | -0.0049 | 0.1019 | 0.0735 | 0.0885 | 0.1303 |
| b_last | controlled_capability/images | center_dist_px | 18.9091 | 2.9362 | 33.7688 | 14.5740 | 75.6833 | 91.4813 |
| b_last | controlled_capability/images | center_dist_px_640 | 9.4975 | 1.5033 | 16.8618 | 7.2870 | 37.8417 | 45.7406 |
| b_last | controlled_capability/images | normalized_center_error | 0.0460 | 0.0295 | 0.0379 | 0.0640 | 0.1006 | 0.1228 |
| b_last | controlled_capability/images | center_abs_dx | 11.4072 | 1.6700 | 25.0700 | 8.7144 | 33.0371 | 60.5450 |
| b_last | controlled_capability/images | center_abs_dy | 12.8527 | 1.4868 | 23.9596 | 7.0733 | 58.2881 | 75.3889 |
| b_last | controlled_capability/images | width_rel_err | 0.0873 | 0.0576 | 0.0874 | 0.1201 | 0.2429 | 0.2782 |
| b_last | controlled_capability/images | height_rel_err | 0.0750 | 0.0521 | 0.0701 | 0.1114 | 0.1641 | 0.2185 |
| b_last | controlled_capability/images | area_ratio | 1.0206 | 1.0382 | 0.1910 | 1.1253 | 1.2451 | 1.3255 |
| b_last | controlled_capability/images | width_ratio_signed | 0.9949 | 1.0175 | 0.1234 | 1.0564 | 1.1081 | 1.1842 |
| b_last | controlled_capability/images | height_ratio_signed | 1.0190 | 1.0139 | 0.1008 | 1.0731 | 1.1438 | 1.1841 |
| b_last | controlled_capability/images | log_width_ratio | -0.0136 | 0.0173 | 0.1335 | 0.0549 | 0.1026 | 0.1691 |
| b_last | controlled_capability/images | log_height_ratio | 0.0138 | 0.0138 | 0.1004 | 0.0705 | 0.1343 | 0.1690 |
| b_last | val | center_dist_px | 1.7070 | 1.3049 | 1.8233 | 2.0670 | 3.2338 | 4.4678 |
| b_last | val | center_dist_px_640 | 0.5726 | 0.4386 | 0.6070 | 0.6946 | 1.0804 | 1.4893 |
| b_last | val | normalized_center_error | 0.0497 | 0.0406 | 0.0356 | 0.0648 | 0.0945 | 0.1187 |
| b_last | val | center_abs_dx | 1.2848 | 0.9083 | 1.3532 | 1.6873 | 2.7392 | 3.8277 |
| b_last | val | center_abs_dy | 0.8418 | 0.5721 | 1.4310 | 1.0553 | 1.7132 | 2.1703 |
| b_last | val | width_rel_err | 0.0695 | 0.0517 | 0.0663 | 0.0924 | 0.1507 | 0.2030 |
| b_last | val | height_rel_err | 0.1119 | 0.0994 | 0.0834 | 0.1613 | 0.2213 | 0.2558 |
| b_last | val | area_ratio | 1.0893 | 1.0840 | 0.1533 | 1.1711 | 1.2614 | 1.3261 |
| b_last | val | width_ratio_signed | 0.9954 | 0.9972 | 0.0959 | 1.0467 | 1.1011 | 1.1427 |
| b_last | val | height_ratio_signed | 1.0936 | 1.0936 | 0.1035 | 1.1600 | 1.2210 | 1.2558 |
| b_last | val | log_width_ratio | -0.0093 | -0.0028 | 0.0979 | 0.0456 | 0.0963 | 0.1334 |
| b_last | val | log_height_ratio | 0.0851 | 0.0895 | 0.0944 | 0.1484 | 0.1997 | 0.2278 |

## 6. 尺寸分桶（equiv_size_640，TP@0.5 与逐桶 AP）

| checkpoint | set | bucket | GT | TP@0.5 | Recall | IoU mean | IoU median | center_px640 median | norm center median | w_rel median | h_rel median | AP50 | AP75 | AP90 | AP95 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| a_best | challenge_test/images | <4 | 40 | 0 | 0.0000 | n/a | n/a | n/a | n/a | n/a | n/a | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| a_best | challenge_test/images | 4-6 | 0 | 0 | n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a |
| a_best | challenge_test/images | 6-8 | 66 | 4 | 0.0606 | 0.7660 | 0.7965 | 0.21 | 0.031 | 0.118 | 0.097 | 0.3131 | 0.1439 | 0.0303 | 0.0000 |
| a_best | challenge_test/images | 8-12 | 20 | 1 | 0.0500 | 0.7718 | 0.7718 | 1.18 | 0.099 | 0.090 | 0.110 | 0.2250 | 0.1500 | 0.0000 | 0.0000 |
| a_best | challenge_test/images | 12-16 | 44 | 4 | 0.0909 | 0.7921 | 0.8361 | 0.79 | 0.065 | 0.068 | 0.023 | 0.1837 | 0.0985 | 0.0227 | 0.0000 |
| a_best | challenge_test/images | 16-24 | 0 | 0 | n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a |
| a_best | challenge_test/images | 24-32 | 0 | 0 | n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a |
| a_best | challenge_test/images | 32-64 | 0 | 0 | n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a |
| a_best | challenge_test/images | >64 | 24 | 0 | 0.0000 | n/a | n/a | n/a | n/a | n/a | n/a | 0.1023 | 0.0172 | 0.0015 | 0.0004 |
| a_best | controlled_capability/images | <4 | 160 | 11 | 0.0688 | 0.7445 | 0.7592 | 0.27 | 0.081 | 0.113 | 0.063 | 0.0966 | 0.0616 | 0.0031 | 0.0000 |
| a_best | controlled_capability/images | 4-6 | 106 | 14 | 0.1321 | 0.7068 | 0.7172 | 0.43 | 0.082 | 0.165 | 0.117 | 0.1967 | 0.0824 | 0.0000 | 0.0000 |
| a_best | controlled_capability/images | 6-8 | 54 | 11 | 0.2037 | 0.8165 | 0.7901 | 0.21 | 0.029 | 0.111 | 0.073 | 0.2877 | 0.2414 | 0.0407 | 0.0000 |
| a_best | controlled_capability/images | 8-12 | 405 | 31 | 0.0765 | 0.8363 | 0.8543 | 0.46 | 0.045 | 0.041 | 0.062 | 0.2516 | 0.1686 | 0.0358 | 0.0000 |
| a_best | controlled_capability/images | 12-16 | 825 | 104 | 0.1261 | 0.8776 | 0.8965 | 0.43 | 0.035 | 0.025 | 0.037 | 0.3161 | 0.2217 | 0.0792 | 0.0121 |
| a_best | controlled_capability/images | 16-24 | 106 | 34 | 0.3208 | 0.8798 | 0.8887 | 0.59 | 0.030 | 0.044 | 0.047 | 0.4969 | 0.4500 | 0.1494 | 0.0094 |
| a_best | controlled_capability/images | 24-32 | 54 | 27 | 0.5000 | 0.9050 | 0.9245 | 0.77 | 0.027 | 0.025 | 0.030 | 0.6134 | 0.5741 | 0.4074 | 0.0741 |
| a_best | controlled_capability/images | 32-64 | 120 | 71 | 0.5917 | 0.7753 | 0.7839 | 2.91 | 0.055 | 0.091 | 0.187 | 0.8625 | 0.5205 | 0.0417 | 0.0083 |
| a_best | controlled_capability/images | >64 | 240 | 56 | 0.2333 | 0.7809 | 0.7913 | 7.61 | 0.078 | 0.052 | 0.065 | 0.6399 | 0.3144 | 0.0356 | 0.0002 |
| a_best | val | <4 | 24 | 1 | 0.0417 | 0.5023 | 0.5023 | 0.56 | 0.141 | 0.644 | 0.036 | 0.1488 | 0.0000 | 0.0000 | 0.0000 |
| a_best | val | 4-6 | 71 | 26 | 0.3662 | 0.6999 | 0.7051 | 0.44 | 0.078 | 0.096 | 0.152 | 0.5323 | 0.2088 | 0.0000 | 0.0000 |
| a_best | val | 6-8 | 251 | 199 | 0.7928 | 0.7809 | 0.7920 | 0.51 | 0.073 | 0.066 | 0.084 | 0.8451 | 0.5582 | 0.0478 | 0.0000 |
| a_best | val | 8-12 | 1724 | 1525 | 0.8846 | 0.8054 | 0.8149 | 0.48 | 0.049 | 0.053 | 0.107 | 0.9302 | 0.7220 | 0.0700 | 0.0054 |
| a_best | val | 12-16 | 755 | 682 | 0.9033 | 0.8096 | 0.8269 | 0.59 | 0.043 | 0.063 | 0.081 | 0.9377 | 0.7080 | 0.1364 | 0.0146 |
| a_best | val | 16-24 | 343 | 302 | 0.8805 | 0.8173 | 0.8330 | 0.79 | 0.042 | 0.064 | 0.074 | 0.9172 | 0.7230 | 0.1370 | 0.0117 |
| a_best | val | 24-32 | 52 | 40 | 0.7692 | 0.8168 | 0.8270 | 1.28 | 0.047 | 0.070 | 0.043 | 0.8168 | 0.6058 | 0.1346 | 0.0192 |
| a_best | val | 32-64 | 29 | 20 | 0.6897 | 0.8145 | 0.7990 | 1.97 | 0.050 | 0.058 | 0.068 | 0.7021 | 0.5572 | 0.1379 | 0.0690 |
| a_best | val | >64 | 11 | 7 | 0.6364 | 0.7899 | 0.7503 | 4.47 | 0.058 | 0.072 | 0.049 | 0.7403 | 0.4545 | 0.0909 | 0.0909 |
| b_best | challenge_test/images | <4 | 40 | 0 | 0.0000 | n/a | n/a | n/a | n/a | n/a | n/a | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| b_best | challenge_test/images | 4-6 | 0 | 0 | n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a |
| b_best | challenge_test/images | 6-8 | 66 | 7 | 0.1061 | 0.7917 | 0.8094 | 0.39 | 0.059 | 0.119 | 0.069 | 0.3378 | 0.1582 | 0.0303 | 0.0000 |
| b_best | challenge_test/images | 8-12 | 20 | 1 | 0.0500 | 0.8981 | 0.8981 | 0.29 | 0.024 | 0.029 | 0.068 | 0.2183 | 0.1500 | 0.0000 | 0.0000 |
| b_best | challenge_test/images | 12-16 | 44 | 6 | 0.1364 | 0.7620 | 0.7629 | 1.12 | 0.092 | 0.140 | 0.088 | 0.2811 | 0.1174 | 0.0227 | 0.0000 |
| b_best | challenge_test/images | 16-24 | 0 | 0 | n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a |
| b_best | challenge_test/images | 24-32 | 0 | 0 | n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a |
| b_best | challenge_test/images | 32-64 | 0 | 0 | n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a |
| b_best | challenge_test/images | >64 | 24 | 0 | 0.0000 | n/a | n/a | n/a | n/a | n/a | n/a | 0.2646 | 0.0028 | 0.0000 | 0.0000 |
| b_best | controlled_capability/images | <4 | 160 | 8 | 0.0500 | 0.7575 | 0.7323 | 0.38 | 0.113 | 0.036 | 0.053 | 0.0661 | 0.0187 | 0.0000 | 0.0000 |
| b_best | controlled_capability/images | 4-6 | 106 | 13 | 0.1226 | 0.7577 | 0.7655 | 0.48 | 0.101 | 0.067 | 0.085 | 0.1799 | 0.0879 | 0.0094 | 0.0000 |
| b_best | controlled_capability/images | 6-8 | 54 | 9 | 0.1667 | 0.7682 | 0.7593 | 0.31 | 0.046 | 0.149 | 0.113 | 0.3071 | 0.1975 | 0.0278 | 0.0000 |
| b_best | controlled_capability/images | 8-12 | 405 | 34 | 0.0840 | 0.8383 | 0.8443 | 0.49 | 0.048 | 0.056 | 0.057 | 0.3334 | 0.2556 | 0.0444 | 0.0049 |
| b_best | controlled_capability/images | 12-16 | 825 | 135 | 0.1636 | 0.8654 | 0.8795 | 0.46 | 0.037 | 0.047 | 0.040 | 0.4566 | 0.3848 | 0.1030 | 0.0158 |
| b_best | controlled_capability/images | 16-24 | 106 | 35 | 0.3302 | 0.8915 | 0.8887 | 0.42 | 0.023 | 0.037 | 0.059 | 0.4827 | 0.4701 | 0.1698 | 0.0283 |
| b_best | controlled_capability/images | 24-32 | 54 | 27 | 0.5000 | 0.9247 | 0.9361 | 0.63 | 0.021 | 0.021 | 0.022 | 0.6389 | 0.6111 | 0.4630 | 0.0926 |
| b_best | controlled_capability/images | 32-64 | 120 | 102 | 0.8500 | 0.8127 | 0.8219 | 2.12 | 0.043 | 0.108 | 0.096 | 0.9375 | 0.7819 | 0.1458 | 0.0083 |
| b_best | controlled_capability/images | >64 | 240 | 160 | 0.6667 | 0.7343 | 0.7482 | 12.61 | 0.085 | 0.122 | 0.079 | 0.6156 | 0.4084 | 0.0798 | 0.0104 |
| b_best | val | <4 | 24 | 1 | 0.0417 | 0.6394 | 0.6394 | 0.43 | 0.108 | 0.353 | 0.069 | 0.1076 | 0.0000 | 0.0000 | 0.0000 |
| b_best | val | 4-6 | 71 | 25 | 0.3521 | 0.7016 | 0.6824 | 0.42 | 0.078 | 0.116 | 0.158 | 0.4371 | 0.1549 | 0.0000 | 0.0000 |
| b_best | val | 6-8 | 251 | 200 | 0.7968 | 0.7875 | 0.7959 | 0.46 | 0.062 | 0.069 | 0.097 | 0.8699 | 0.6058 | 0.0518 | 0.0040 |
| b_best | val | 8-12 | 1724 | 1478 | 0.8573 | 0.8186 | 0.8258 | 0.39 | 0.041 | 0.046 | 0.106 | 0.9325 | 0.7675 | 0.1030 | 0.0087 |
| b_best | val | 12-16 | 755 | 555 | 0.7351 | 0.8298 | 0.8448 | 0.46 | 0.035 | 0.061 | 0.079 | 0.8899 | 0.7082 | 0.1589 | 0.0119 |
| b_best | val | 16-24 | 343 | 225 | 0.6560 | 0.8291 | 0.8533 | 0.68 | 0.038 | 0.057 | 0.069 | 0.8206 | 0.6368 | 0.1540 | 0.0029 |
| b_best | val | 24-32 | 52 | 26 | 0.5000 | 0.8229 | 0.8281 | 1.47 | 0.057 | 0.057 | 0.048 | 0.6667 | 0.4840 | 0.1154 | 0.0192 |
| b_best | val | 32-64 | 29 | 23 | 0.7931 | 0.8224 | 0.8472 | 1.78 | 0.042 | 0.046 | 0.074 | 0.7969 | 0.6724 | 0.1034 | 0.0000 |
| b_best | val | >64 | 11 | 9 | 0.8182 | 0.8309 | 0.8369 | 3.51 | 0.050 | 0.086 | 0.057 | 0.8523 | 0.7273 | 0.1818 | 0.0000 |
| b_last | challenge_test/images | <4 | 40 | 0 | 0.0000 | n/a | n/a | n/a | n/a | n/a | n/a | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| b_last | challenge_test/images | 4-6 | 0 | 0 | n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a |
| b_last | challenge_test/images | 6-8 | 66 | 2 | 0.0303 | 0.8462 | 0.8462 | 0.37 | 0.055 | 0.041 | 0.068 | 0.1742 | 0.1591 | 0.0455 | 0.0000 |
| b_last | challenge_test/images | 8-12 | 20 | 1 | 0.0500 | 0.8680 | 0.8680 | 0.38 | 0.032 | 0.071 | 0.076 | 0.1500 | 0.1000 | 0.0000 | 0.0000 |
| b_last | challenge_test/images | 12-16 | 44 | 5 | 0.1136 | 0.8516 | 0.8477 | 0.38 | 0.031 | 0.092 | 0.055 | 0.1364 | 0.1364 | 0.0227 | 0.0000 |
| b_last | challenge_test/images | 16-24 | 0 | 0 | n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a |
| b_last | challenge_test/images | 24-32 | 0 | 0 | n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a |
| b_last | challenge_test/images | 32-64 | 0 | 0 | n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a |
| b_last | challenge_test/images | >64 | 24 | 24 | 1.0000 | 0.6657 | 0.6670 | 45.97 | 0.100 | 0.223 | 0.066 | 0.9514 | 0.3693 | 0.0060 | 0.0000 |
| b_last | controlled_capability/images | <4 | 160 | 4 | 0.0250 | 0.6995 | 0.7024 | 0.43 | 0.130 | 0.091 | 0.127 | 0.0437 | 0.0125 | 0.0000 | 0.0000 |
| b_last | controlled_capability/images | 4-6 | 106 | 11 | 0.1038 | 0.6752 | 0.7245 | 0.53 | 0.106 | 0.144 | 0.128 | 0.1415 | 0.0566 | 0.0000 | 0.0000 |
| b_last | controlled_capability/images | 6-8 | 54 | 7 | 0.1296 | 0.6583 | 0.6114 | 0.45 | 0.061 | 0.217 | 0.219 | 0.1713 | 0.0787 | 0.0000 | 0.0000 |
| b_last | controlled_capability/images | 8-12 | 405 | 23 | 0.0568 | 0.8032 | 0.7980 | 0.54 | 0.053 | 0.055 | 0.097 | 0.1284 | 0.1012 | 0.0198 | 0.0000 |
| b_last | controlled_capability/images | 12-16 | 825 | 65 | 0.0788 | 0.8632 | 0.8720 | 0.36 | 0.028 | 0.061 | 0.050 | 0.1934 | 0.1683 | 0.0509 | 0.0061 |
| b_last | controlled_capability/images | 16-24 | 106 | 34 | 0.3208 | 0.8869 | 0.8936 | 0.52 | 0.028 | 0.046 | 0.047 | 0.4104 | 0.4009 | 0.1792 | 0.0189 |
| b_last | controlled_capability/images | 24-32 | 54 | 25 | 0.4630 | 0.9185 | 0.9315 | 0.63 | 0.023 | 0.022 | 0.034 | 0.5926 | 0.5741 | 0.4074 | 0.0741 |
| b_last | controlled_capability/images | 32-64 | 120 | 100 | 0.8333 | 0.9186 | 0.9194 | 1.00 | 0.021 | 0.038 | 0.023 | 0.9083 | 0.8833 | 0.7417 | 0.0500 |
| b_last | controlled_capability/images | >64 | 240 | 238 | 0.9917 | 0.7936 | 0.7955 | 8.95 | 0.042 | 0.081 | 0.086 | 0.9896 | 0.7564 | 0.2861 | 0.0375 |
| b_last | val | <4 | 24 | 1 | 0.0417 | 0.5784 | 0.5784 | 0.62 | 0.156 | 0.447 | 0.137 | 0.0469 | 0.0000 | 0.0000 | 0.0000 |
| b_last | val | 4-6 | 71 | 18 | 0.2535 | 0.6983 | 0.7072 | 0.40 | 0.083 | 0.159 | 0.192 | 0.2968 | 0.0850 | 0.0000 | 0.0000 |
| b_last | val | 6-8 | 251 | 159 | 0.6335 | 0.7970 | 0.8011 | 0.40 | 0.054 | 0.068 | 0.098 | 0.7251 | 0.5040 | 0.0578 | 0.0040 |
| b_last | val | 8-12 | 1724 | 1185 | 0.6874 | 0.8151 | 0.8217 | 0.40 | 0.040 | 0.045 | 0.122 | 0.8247 | 0.6678 | 0.0731 | 0.0023 |
| b_last | val | 12-16 | 755 | 411 | 0.5444 | 0.8340 | 0.8503 | 0.48 | 0.035 | 0.060 | 0.066 | 0.7493 | 0.5704 | 0.1497 | 0.0132 |
| b_last | val | 16-24 | 343 | 153 | 0.4461 | 0.8324 | 0.8440 | 0.73 | 0.040 | 0.062 | 0.057 | 0.6346 | 0.4849 | 0.1283 | 0.0146 |
| b_last | val | 24-32 | 52 | 16 | 0.3077 | 0.8561 | 0.8616 | 0.99 | 0.038 | 0.051 | 0.057 | 0.5000 | 0.4231 | 0.0769 | 0.0000 |
| b_last | val | 32-64 | 29 | 15 | 0.5172 | 0.7899 | 0.8657 | 2.43 | 0.045 | 0.029 | 0.066 | 0.7241 | 0.4310 | 0.1724 | 0.0000 |
| b_last | val | >64 | 11 | 8 | 0.7273 | 0.8185 | 0.8376 | 1.96 | 0.025 | 0.109 | 0.060 | 0.8636 | 0.6818 | 0.1818 | 0.0000 |

## 7. A vs B 分桶增量（B 减 A；正数表示 B 更大）

| set | bucket | GT_a | GT_b | R_a | R_b | dR | IoU_med_a | IoU_med_b | dIoU_med | ctr640_a | ctr640_b | dctr640 | w_rel_a | w_rel_b | dw_rel | h_rel_a | h_rel_b | dh_rel |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| challenge_test/images | ALL | 194 | 194 | 0.0464 | 0.0722 | 0.0258 | 0.7920 | 0.8131 | 0.0211 | 0.41 | 0.40 | -0.01 | 0.096 | 0.117 | 0.021 | 0.039 | 0.079 | 0.040 |
| challenge_test/images | <4 | 40 | 40 | 0.0000 | 0.0000 | 0.0000 | n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a |
| challenge_test/images | 4-6 | 0 | 0 | n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a |
| challenge_test/images | 6-8 | 66 | 66 | 0.0606 | 0.1061 | 0.0455 | 0.7965 | 0.8094 | 0.0128 | 0.21 | 0.39 | 0.19 | 0.118 | 0.119 | 0.001 | 0.097 | 0.069 | -0.028 |
| challenge_test/images | 8-12 | 20 | 20 | 0.0500 | 0.0500 | 0.0000 | 0.7718 | 0.8981 | 0.1263 | 1.18 | 0.29 | -0.89 | 0.090 | 0.029 | -0.060 | 0.110 | 0.068 | -0.042 |
| challenge_test/images | 12-16 | 44 | 44 | 0.0909 | 0.1364 | 0.0455 | 0.8361 | 0.7629 | -0.0732 | 0.79 | 1.12 | 0.33 | 0.068 | 0.140 | 0.072 | 0.023 | 0.088 | 0.066 |
| challenge_test/images | 16-24 | 0 | 0 | n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a |
| challenge_test/images | 24-32 | 0 | 0 | n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a |
| challenge_test/images | 32-64 | 0 | 0 | n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a |
| challenge_test/images | >64 | 24 | 24 | 0.0000 | 0.0000 | 0.0000 | n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a |
| controlled_capability/images | ALL | 2070 | 2070 | 0.1734 | 0.2527 | 0.0792 | 0.8520 | 0.8455 | -0.0065 | 0.70 | 1.11 | 0.41 | 0.043 | 0.079 | 0.036 | 0.063 | 0.061 | -0.001 |
| controlled_capability/images | <4 | 160 | 160 | 0.0688 | 0.0500 | -0.0188 | 0.7592 | 0.7323 | -0.0268 | 0.27 | 0.38 | 0.11 | 0.113 | 0.036 | -0.077 | 0.063 | 0.053 | -0.010 |
| controlled_capability/images | 4-6 | 106 | 106 | 0.1321 | 0.1226 | -0.0094 | 0.7172 | 0.7655 | 0.0484 | 0.43 | 0.48 | 0.05 | 0.165 | 0.067 | -0.098 | 0.117 | 0.085 | -0.031 |
| controlled_capability/images | 6-8 | 54 | 54 | 0.2037 | 0.1667 | -0.0370 | 0.7901 | 0.7593 | -0.0308 | 0.21 | 0.31 | 0.09 | 0.111 | 0.149 | 0.038 | 0.073 | 0.113 | 0.040 |
| controlled_capability/images | 8-12 | 405 | 405 | 0.0765 | 0.0840 | 0.0074 | 0.8543 | 0.8443 | -0.0100 | 0.46 | 0.49 | 0.04 | 0.041 | 0.056 | 0.015 | 0.062 | 0.057 | -0.006 |
| controlled_capability/images | 12-16 | 825 | 825 | 0.1261 | 0.1636 | 0.0376 | 0.8965 | 0.8795 | -0.0171 | 0.43 | 0.46 | 0.03 | 0.025 | 0.047 | 0.022 | 0.037 | 0.040 | 0.003 |
| controlled_capability/images | 16-24 | 106 | 106 | 0.3208 | 0.3302 | 0.0094 | 0.8887 | 0.8887 | -0.0000 | 0.59 | 0.42 | -0.17 | 0.044 | 0.037 | -0.007 | 0.047 | 0.059 | 0.012 |
| controlled_capability/images | 24-32 | 54 | 54 | 0.5000 | 0.5000 | 0.0000 | 0.9245 | 0.9361 | 0.0116 | 0.77 | 0.63 | -0.14 | 0.025 | 0.021 | -0.004 | 0.030 | 0.022 | -0.009 |
| controlled_capability/images | 32-64 | 120 | 120 | 0.5917 | 0.8500 | 0.2583 | 0.7839 | 0.8219 | 0.0380 | 2.91 | 2.12 | -0.80 | 0.091 | 0.108 | 0.017 | 0.187 | 0.096 | -0.091 |
| controlled_capability/images | >64 | 240 | 240 | 0.2333 | 0.6667 | 0.4333 | 0.7913 | 0.7482 | -0.0431 | 7.61 | 12.61 | 5.00 | 0.052 | 0.122 | 0.070 | 0.065 | 0.079 | 0.014 |
| val | ALL | 3260 | 3260 | 0.8595 | 0.7798 | -0.0798 | 0.8170 | 0.8284 | 0.0114 | 0.53 | 0.44 | -0.09 | 0.059 | 0.052 | -0.007 | 0.095 | 0.094 | -0.001 |
| val | <4 | 24 | 24 | 0.0417 | 0.0417 | 0.0000 | 0.5023 | 0.6394 | 0.1371 | 0.56 | 0.43 | -0.13 | 0.644 | 0.353 | -0.291 | 0.036 | 0.069 | 0.033 |
| val | 4-6 | 71 | 71 | 0.3662 | 0.3521 | -0.0141 | 0.7051 | 0.6824 | -0.0227 | 0.44 | 0.42 | -0.02 | 0.096 | 0.116 | 0.020 | 0.152 | 0.158 | 0.005 |
| val | 6-8 | 251 | 251 | 0.7928 | 0.7968 | 0.0040 | 0.7920 | 0.7959 | 0.0039 | 0.51 | 0.46 | -0.05 | 0.066 | 0.069 | 0.002 | 0.084 | 0.097 | 0.013 |
| val | 8-12 | 1724 | 1724 | 0.8846 | 0.8573 | -0.0273 | 0.8149 | 0.8258 | 0.0109 | 0.48 | 0.39 | -0.09 | 0.053 | 0.046 | -0.007 | 0.107 | 0.106 | -0.001 |
| val | 12-16 | 755 | 755 | 0.9033 | 0.7351 | -0.1682 | 0.8269 | 0.8448 | 0.0179 | 0.59 | 0.46 | -0.12 | 0.063 | 0.061 | -0.002 | 0.081 | 0.079 | -0.002 |
| val | 16-24 | 343 | 343 | 0.8805 | 0.6560 | -0.2245 | 0.8330 | 0.8533 | 0.0203 | 0.79 | 0.68 | -0.10 | 0.064 | 0.057 | -0.007 | 0.074 | 0.069 | -0.006 |
| val | 24-32 | 52 | 52 | 0.7692 | 0.5000 | -0.2692 | 0.8270 | 0.8281 | 0.0011 | 1.28 | 1.47 | 0.19 | 0.070 | 0.057 | -0.013 | 0.043 | 0.048 | 0.005 |
| val | 32-64 | 29 | 29 | 0.6897 | 0.7931 | 0.1034 | 0.7990 | 0.8472 | 0.0482 | 1.97 | 1.78 | -0.19 | 0.058 | 0.046 | -0.013 | 0.068 | 0.074 | 0.006 |
| val | >64 | 11 | 11 | 0.6364 | 0.8182 | 0.1818 | 0.7503 | 0.8369 | 0.0866 | 4.47 | 3.51 | -0.95 | 0.072 | 0.086 | 0.014 | 0.049 | 0.057 | 0.008 |

## 8. Near-miss 分类（每个 GT 的最佳一对一 IoU）

class A = IoU >= 0.50（TP）；class B = 0.30-0.50（近失，**不计 TP**）；class C = 0.10-0.30；class D < 0.10 或无匹配。

| checkpoint | set | class | count | share | mean IoU |
|---|---|---|---|---|---|
| a_best | challenge_test/images | A | 9 | 0.0464 | 0.778 |
| a_best | challenge_test/images | B | 0 | 0.0000 | n/a |
| a_best | challenge_test/images | C | 0 | 0.0000 | n/a |
| a_best | challenge_test/images | D | 185 | 0.9536 | 0.000 |
| a_best | controlled_capability/images | A | 358 | 0.1729 | 0.828 |
| a_best | controlled_capability/images | B | 1 | 0.0005 | 0.412 |
| a_best | controlled_capability/images | C | 4 | 0.0019 | 0.206 |
| a_best | controlled_capability/images | D | 1707 | 0.8246 | 0.000 |
| a_best | val | A | 2781 | 0.8531 | 0.805 |
| a_best | val | B | 21 | 0.0064 | 0.460 |
| a_best | val | C | 4 | 0.0012 | 0.219 |
| a_best | val | D | 454 | 0.1393 | 0.000 |
| b_best | challenge_test/images | A | 14 | 0.0722 | 0.787 |
| b_best | challenge_test/images | B | 2 | 0.0103 | 0.495 |
| b_best | challenge_test/images | C | 0 | 0.0000 | n/a |
| b_best | challenge_test/images | D | 178 | 0.9175 | 0.000 |
| b_best | controlled_capability/images | A | 477 | 0.2304 | 0.828 |
| b_best | controlled_capability/images | B | 48 | 0.0232 | 0.392 |
| b_best | controlled_capability/images | C | 2 | 0.0010 | 0.158 |
| b_best | controlled_capability/images | D | 1543 | 0.7454 | 0.000 |
| b_best | val | A | 2526 | 0.7748 | 0.819 |
| b_best | val | B | 20 | 0.0061 | 0.428 |
| b_best | val | C | 0 | 0.0000 | n/a |
| b_best | val | D | 714 | 0.2190 | 0.000 |
| b_last | challenge_test/images | A | 30 | 0.1546 | 0.722 |
| b_last | challenge_test/images | B | 2 | 0.0103 | 0.403 |
| b_last | challenge_test/images | C | 0 | 0.0000 | n/a |
| b_last | challenge_test/images | D | 162 | 0.8351 | 0.000 |
| b_last | controlled_capability/images | A | 501 | 0.2420 | 0.836 |
| b_last | controlled_capability/images | B | 7 | 0.0034 | 0.473 |
| b_last | controlled_capability/images | C | 0 | 0.0000 | n/a |
| b_last | controlled_capability/images | D | 1562 | 0.7546 | 0.000 |
| b_last | val | A | 1923 | 0.5899 | 0.819 |
| b_last | val | B | 7 | 0.0021 | 0.460 |
| b_last | val | C | 1 | 0.0003 | 0.203 |
| b_last | val | D | 1329 | 0.4077 | 0.000 |

## 9. 判定规则（自动，容差 0.01）

CASE A = AP50 上升但 AP75/AP90 与 IoU 中位数基本不动（纯粹多找到了目标）；CASE B = AP75/AP90 与 IoU 中位数一起上升、中心/尺寸误差下降（真正的定位改善）；CASE C = 两者同时改善。

| set | verdict | dAP50 | dAP75 | dAP90 | dAP95 | d_IoU_median | d_center_px640_med | d_norm_center_med |
|---|---|---|---|---|---|---|---|---|
| challenge_test/images | INDETERMINATE (mixed / no clear AP shift) | 0.0499 | 0.0074 | -0.0002 | -0.0001 | 0.0211 | -0.0114 | 0.0077 |
| controlled_capability/images | INDETERMINATE (mixed / no clear AP shift) | 0.0708 | 0.1059 | 0.0248 | 0.0050 | -0.0065 | 0.4126 | -0.0001 |
| val | CASE B (real bbox localization gain) | -0.0216 | 0.0176 | 0.0244 | -0.0004 | 0.0114 | -0.0934 | -0.0077 |

## 10. 复现命令与产物

```
python tools/audit_localization_yolo26_v1.py --device 0
```

- 检查点与标注数据均为只读输入；本审计不产生任何训练/权重/标签变更。
- 输出 CSV：localization_ap_by_iou.csv / localization_iou_distribution.csv / localization_error_summary.csv /
  localization_size_buckets.csv / localization_checkpoint_compare.csv / localization_near_miss.csv；
  另有 localization_audit.json（含每个 GT 的匹配记录）与 localization_examples/（示例图 + index.csv）。
- 预测缓存位于 `_scratch_localization_audit/`（临时，可删；删除后用同一命令可完整重生成）。
- 示例图共 348 张（low_iou / center_error / width_error / height_error / near_miss，各 30 张/集合）。

