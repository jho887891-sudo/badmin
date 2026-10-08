# STAGE_B_FINAL_REPORT —— Stage B 训练收官与三方对比（Stage A best / Stage B best@e10 / Stage B last@e70）

> 生成时间：2026-09-29　｜　训练远端：`dgut@172.31.68.251`（RTX A6000 49,140 MiB）；评估本地：Windows + RTX 4060 Laptop 8 GB（`D:\_eval26\venv`：torch 2.14.0+cu126 / ultralytics 8.4.150）
> 所有结论可由 `outputs/shuttle_capability/metrics/{size_bucket_*,hard_negative_*,stage_b_gate_final.json}` 与 `outputs/shuttle_capability/hard_negative_eval/**` 自证（sha256 见 §9）。

---

## 1. 结论速览

| 问题 | 结论 | 证据 |
|---|---|---|
| Stage B 训练过程是否健康？ | **是**（无 NaN、LR 全程在设计目标内、无坍塌、内存安全） | §4，Gate 判定 `PASS_CONTINUE_STAGE_B` |
| Stage B 是否比 Stage A 更好？ | **没有可交付的改进** | §5：mAP50 −0.0216、Recall −0.0798、R_6-16 −0.0634；仅 mAP50-95 +0.0019（0.3%，噪声级） |
| best epoch 是哪个？ | **epoch 10**（fitness 0.58263）；mAP50-95 峰值在 epoch 11（0.55305 vs e10 0.55291） | §3 |
| epoch 70 能用吗？ | **不能**：val mAP50 0.7564、Recall 0.6031（比 Stage A 低 0.2564） | §5、§6 |
| 小目标（6–16 px）表现？ | Stage B e10 比 A 低 6.3 个百分点（0.8179 vs 0.8813）；**12–32 px 段退步最大**（−0.17～−0.27） | §6 |
| 误检（负样本）表现？ | 未见过 COCO 背景上 B e10 最好（0.0010 vs A 0.0100 FP/图）；冻结真实背景上 A 与 B e10 基本持平（二者最高误检是同一张图同一物体） | §7 |
| 下一步 | 用**固定评估集合（fixed evaluation set / development holdout）**在 A best 与 B best@e10 之间做判定；不要采用 e70 | §8 |

**判定 token：`STAGE_B_HEALTHY_BUT_NO_VAL_GAIN`** —— 训练过程健康、已跑完 70 epoch，但**不建议把 Stage B 产物作为基线替换 Stage A**，除非固定评估集合给出相反证据。
> **术语更正（2026-10-01）**：本节早期版本写作“8 个冻结集合从未参与训练、调参或模型选择”。该表述不严谨，已被 `docs/LOCALIZATION_AUDIT.md` §7 的口径取代：
> 这些集合在 Stage A/B 的模型选择与多份报告中被反复查看使用，应称为 **fixed evaluation set** / **development holdout**，不得称为“完全未参与模型选择的最终测试集”。
> 另外 `management/ISSUES.md` ISSUE-027 已记录：`excluded.txt` 中 7 张图实际进入了训练，隔离并非 100%。
> 若后续 Stage C 依据这些集合上的任何结论调参或选点，最终判断**必须另用一个全新的、未被看过的 holdout**，否则无法与选择偏差区分。

---

## 2. Stage B 运行事实

| 项 | 值 |
|---|---|
| run id | `gate6B_lr0001_musgd_b16_w8_20260927-235730` |
| run 目录（远端） | `/home/T7/ojh/robot_sim/runs/shuttle_yolo26_v1/gate6B_lr0001_musgd_b16_w8_20260927-235730` |
| 起止 | 2026-09-27T15:58Z 起，2026-09-29T02:42Z 结束（`results.csv` 最后写入 02:39:43） |
| `metrics.json.elapsed_s` | **124,977.6 s（34.72 h）** |
| epoch 数 | **70/70**（`results.csv` 71 行含表头） |
| 关键超参 | imgsz 1024、physical batch 16、`nbs=32`（有效 batch 32）、workers 8、freeze **0**、`optimizer=MuSGD`（显式）、`lr0=0.001`（头部 ×3 = 0.003）、`warmup_epochs=3`、`warmup_bias_lr=0`、`cos_lr=false`、`lrf=0.01`、seed 42 |
| 初始化 | `…/gate6A_imgsz1024_b16_w8_20260926-083331/weights/best.pt`（Stage A epoch 6） |
| 训练图数 | 33,456 / epoch（`metrics.json.train_images`） |
| 峰值显存 | **17,455 MB** |
| 内存（`mem_probe.jsonl`，1,463 条） | MemAvailable 最低 **20.6 GB**、最高 32.7 GB；trainer RSS 峰值 **9.59 GB**；swap in/out 峰值 **1,254.5 / 1,319.4 KB/s** |
| 内存护栏 | MemAvailable < 5 GB 或 swap > 20,480 KB/s 或 cgroup > 0.90 → **全程未触发** |
| NaN | **0 行**（results.csv 全表扫描） |
| LR 实测（`lr_probe.jsonl`，146,440 条） | 全程最大 **0.00291468**（检测头组，限 0.0033）；末期 7.24e-05 / 2.41e-05 |

### 2.1 最终权重（已被 ultralytics `strip_optimizer` 转为推理格式）

| 文件 | 字节 | sha256 | 含义 |
|---|---:|---|---|
| `weights/best.pt` | **20,136,763** | `419a3eefa7e4ca44b7da29457fddcc142ed56a04de0735f116cb46121097208b` | epoch 10 权重 |
| `weights/last.pt` | **20,136,763** | `42f0043a1b902b40780c9781cc36bb0559b828c7b3722b2da3d114830d7f98a4` | epoch 70 权重 |

两者大小相同是因为**都已被剥离优化器状态**（训练中途它们是 58.9 MB：`best.pt` 58,948,377 B、`last.pt` 58,957,721 B）。本地副本：`D:\_eth_dl\ckpts\stageB_best_e10_stripped.pt`、`stageB_last_e70.pt`（sha256 与远端逐一相符）。

---

## 3. 训练曲线与 best epoch 判定

| epoch | train box_loss | P | R | mAP50 | mAP50-95 | 备注 |
|---:|---:|---:|---:|---:|---:|---|
| 1 | 1.38454 | 0.91561 | 0.74632 | 0.84932 | 0.51743 | 续训起点（Stage A e6 基线：R 0.76012 / mAP50 0.87100 / mAP50-95 0.52278 / box 1.43067）|
| **10** | 1.26328 | 0.90564 | **0.77301** | **0.85013** | 0.55291 | **best.pt（fitness 0.58263）** |
| 11 | — | — | — | — | **0.55305** | mAP50-95 峰值（fitness 未超 e10） |
| 65 | 0.86037 | 0.68106 | 0.63528 | 0.66335 | 0.43643 | |
| **70** | **0.83955** | 0.67554 | 0.62638 | **0.64597** | **0.42474** | 终点 |

全 70 epoch 汇总：`max_P = 0.95056`、`max_R = 0.77446 (e6)`、`max_mAP50 = 0.85013 (e10)`、`max_mAP50-95 = 0.55305 (e11)`、`min_mAP50 = 0.64597 (e70)`、NaN 0。

**过拟合判据（同一次运行内自证）**：train box_loss 从 e10 的 1.26328 单调降到 e70 的 0.83955（−33.5%），而 val mAP50 从 0.85013 降到 0.64597（−0.20416）、mAP50-95 0.55291 → 0.42474（−0.12817）、recall 0.77301 → 0.62638。**训练损失下降 + 验证指标下降 = 过拟合**（无 NaN、LR 正常衰减，不是发散）。

---

## 4. 训练过程 Gate 判定（`tools/check_stage_b_gate.py`）

```
D:\_eval26\venv\Scripts\python.exe tools\check_stage_b_gate.py --run D:\_eth_dl\stageb_run_local --label final_stage_b_epoch70 --required-epochs 70 --out outputs\shuttle_capability\metrics\stage_b_gate_final.json
```

| 检查项 | 阈值/目标 | 实测 | 结论 |
|---|---|---|---|
| verdict | — | **`PASS_CONTINUE_STAGE_B`** | — |
| epochs_finished | 70 | 70 | ✓ |
| LR 越界（正常组 ≤0.0011 / 头部 ≤0.0033） | ×1.10 容差 | `lr_violations = []` | ✓ |
| NaN / Inf | 必须 false | `nan = false` | ✓ |
| backbone 解冻 | 必须 true | `backbone_unfrozen = true` | ✓ |
| 初始化权重 | 指向 Stage A best.pt | 相符 | ✓ |
| 指标坍塌（catastrophic / trend） | 均为 false | 均为 false | ✓ |
| 内存 | MemAvailable ≥5 GB、swap ≤20,480 KB/s | 20.6 GB / 2,077.7 KB/s | ✓ |

> `PASS_CONTINUE_STAGE_B` 是这套工具的“训练过程无异常”判定，**不代表模型更好**；模型优劣见 §5–§7。

---

## 5. 三方对比 A —— val 总体（4,413 张 / 3,260 GT，conf 0.25，IoU 0.5，imgsz 1024）

来源：`outputs/shuttle_capability/metrics/size_bucket_metrics_val.json`（三检查点同一轮推理、同一工具版本）。

| 检查点 | TP | FP | FN | P | R | F1 | mAP50 | mAP50-95 | R_6-16 | R_4-16 | 中心距≤25px R（次要） |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| **stageA_best (e6)** | 2,802 | 1,269 | 458 | 0.6883 | **0.8595** | **0.7644** | **0.9051** | 0.5907 | **0.8813** | **0.8683** | **0.8666** |
| stageB_best_e10 | 2,542 | 1,168 | 718 | 0.6852 | 0.7798 | 0.7294 | 0.8835 | **0.5926** | 0.8179 | 0.8061 | 0.7853 |
| stageB_last_e70 | 1,966 | **806** | 1,294 | **0.7092** | 0.6031 | 0.6519 | 0.7564 | 0.5044 | 0.6429 | 0.6330 | 0.6055 |

**Δ（相对 stageA_best）**

| 检查点 | ΔR | ΔmAP50 | ΔmAP50-95 | ΔR_6-16 | ΔFP |
|---|---:|---:|---:|---:|---:|
| stageB_best_e10 | −0.0798 | −0.0216 | **+0.0019** | −0.0634 | −101（−8.0%） |
| stageB_last_e70 | −0.2564 | −0.1487 | −0.0863 | −0.2384 | −463（−36.5%） |

**读法**：Stage B 唯一领先的是 mAP50-95（+0.0019，0.3%），而它正是 ultralytics fitness 中权重 0.9 的那一项 —— 这解释了 B 运行内部为何选出 e10。但项目目标是**超小目标召回**（R_6-16），该项 Stage B 比 A 低 6.34 个百分点。e70 的精度看似最高（P 0.7092）只是因为**框变少了**（FN 从 458 涨到 1,294）。

---

## 6. 三方对比 B —— 尺寸分桶（equiv_size_640，半开区间；LOW_SAMPLE 阈值 30 GT）

完整表（含逐桶 Δ 与汇总行）：`outputs/shuttle_capability/metrics/size_bucket_three_way_compare_val.csv`。

| 桶（px） | GT | A Recall | B_e10 Recall | B_e70 Recall | A AP50 | B_e10 AP50 | B_e70 AP50 | A AP50-95 | B_e10 AP50-95 | B_e70 AP50-95 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| <4 ⚠ | 24 | 0.0417 | 0.0417 | 0.0417 | 0.1488 | 0.1076 | 0.0469 | 0.0220 | 0.0259 | 0.0091 |
| 4-6 | 71 | 0.3662 | 0.3521 | 0.2535 | 0.5323 | 0.4371 | 0.2968 | 0.2633 | 0.2001 | 0.1299 |
| 6-8 | 251 | 0.7928 | **0.7968** | 0.6335 | 0.8451 | **0.8699** | 0.7251 | 0.5149 | **0.5392** | 0.4554 |
| 8-12 | 1,724 | 0.8846 | 0.8573 | 0.6874 | 0.9302 | **0.9325** | 0.8247 | 0.6087 | **0.6302** | 0.5511 |
| 12-16 | 755 | **0.9033** | 0.7351 | 0.5444 | **0.9377** | 0.8899 | 0.7493 | **0.6200** | 0.6071 | 0.5106 |
| 16-24 | 343 | **0.8805** | 0.6560 | 0.4461 | **0.9172** | 0.8206 | 0.6346 | **0.6199** | 0.5587 | 0.4287 |
| 24-32 | 52 | **0.7692** | 0.5000 | 0.3077 | **0.8168** | 0.6667 | 0.5000 | **0.5419** | 0.4465 | 0.3452 |
| 32-64 ⚠ | 29 | 0.6897 | **0.7931** | 0.5172 | 0.7021 | **0.7969** | 0.7241 | 0.4794 | **0.5366** | 0.4592 |
| >64 ⚠ | 11 | 0.6364 | **0.8182** | 0.7273 | 0.7403 | **0.8523** | 0.8636 | 0.4721 | 0.5909 | **0.5909** |
| **OVERALL** | 3,260 | **0.8595** | 0.7798 | 0.6031 | **0.9051** | 0.8835 | 0.7564 | 0.5907 | **0.5926** | 0.5044 |
| COMBINED 6-16 | 2,730 | **0.8813** | 0.8179 | 0.6429 | — | — | — | — | — | — |
| COMBINED 4-16 | 2,801 | **0.8683** | 0.8061 | 0.6330 | — | — | — | — | — | — |
| SECONDARY（中心距≤25px） | 3,260 | **0.8666** | 0.7853 | 0.6055 | — | — | — | — | — | — |

⚠ = LOW_SAMPLE（GT < 30）：`<4`(24)、`32-64`(29)、`>64`(11)；`24-32`(52) 偏小，引用需谨慎。

**分桶结论**：

1. Stage B 在 **6–8 px** 与 **8–12 px** 桶的 AP50 上更好（+0.0248 / +0.0023），是与“超小目标”目标一致的唯一正向信号；
2. 但 **12–16 / 16–24 / 24–32 px** 段大幅退步（Recall −0.1682 / −0.2245 / −0.2692；AP50 −0.0477 / −0.0967 / −0.1502），足以抵消小目标收益；
3. `32-64` / `>64` 的“进步”**样本量不足**（29 / 11 GT），不能作为结论；
4. e70 在所有桶上都退步，6–16 px 组合召回掉到 0.6429（比 A 低 0.2384）。

---

## 7. 三方对比 C —— 困难负样本假阳性（imgsz 1024，conf floor 0.01）

数据身份（冻结清单 `v1_dataset_manifest.csv` sha256 `11d850263b370fc67b78b1264609a5a3d2f79d9f235f9d943bfb322de91ed539` 逐行核对）：

| 集合 | 图数 | 是否参与训练 | 说明 |
|---|---:|---|---|
| `hard_negative` | 96 | **是（100% split=train）** | 记忆性检验，非泛化证据 |
| `normal_neg_coco_val` | 1,000 | 否（split=val） | 泛化证据 |
| `normal_neg_bg_val` | 6 | 否（split=val） | 样本量过小 |
| `frozen_real_backgrounds` | 30 | 否（`real_images` 被 builder 硬排除） | 从未进入任何 split |

### 7.1 FP 计数（`FP/图` / `含FP图数`）

| 集合 | 阈值 | A best | B best@e10 | B last@e70 |
|---|---|---|---|---|
| hard_negative (96) | 0.25 | 0.0208 / 2 | **0 / 0** | **0 / 0** |
| hard_negative (96) | 0.50 | 0 / 0 | 0 / 0 | 0 / 0 |
| coco_val (1000) | 0.25 | 0.0100 / 8 | **0.0010 / 1** | 0.0040 / 4 |
| coco_val (1000) | 0.50 | 0.0020 / 2 | **0 / 0** | 0.0030 / 3 |
| bg_val (6) | 0.25 | 0.1667 / 1 | 0.8333 / 1 | **0 / 0** |
| bg_val (6) | 0.50 | 0 / 0 | 0.5000 / 1 | **0 / 0** |
| frozen_real_bg (30) | 0.25 | 0.0667 / 2 | 0.1000 / 2 | **0 / 0** |
| frozen_real_bg (30) | 0.50 | 0.0333 / 1 | 0.0667 / 2 | **0 / 0** |

@0.75 三个检查点在四个集合上**全为 0**。

### 7.2 FP 置信度分布（`conf ≥ 0.01`；这些集合无正样本标注，所有预测即 FP）

| 检查点 | 集合 | ≥0.01 总数 | 0.01-0.05 | 0.05-0.10 | 0.10-0.25 | 0.25-0.50 | 0.50-0.75 | 0.75-1.0 | max |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| A | hard_negative | 61 | 47 | 8 | 4 | 2 | 0 | 0 | 0.4014 |
| B_e10 | hard_negative | 37 | 29 | 5 | 3 | 0 | 0 | 0 | 0.1868 |
| B_e70 | hard_negative | 5 | 4 | 1 | 0 | 0 | 0 | 0 | 0.0801 |
| A | coco_val | 252 | 189 | 33 | 20 | 8 | 2 | 0 | 0.6546 |
| B_e10 | coco_val | 46 | 33 | 7 | 5 | 1 | 0 | 0 | 0.2900 |
| B_e70 | coco_val | 15 | 7 | 2 | 2 | 1 | 3 | 0 | **0.6816** |
| A | bg_val | 10 | 9 | 0 | 0 | 1 | 0 | 0 | 0.3662 |
| B_e10 | bg_val | 21 | 10 | 2 | 4 | 2 | 3 | 0 | 0.6840 |
| B_e70 | bg_val | 10 | 7 | 2 | 1 | 0 | 0 | 0 | 0.1747 |
| A | frozen_real_bg | 32 | 21 | 5 | 4 | 1 | 1 | 0 | 0.5921 |
| B_e10 | frozen_real_bg | 32 | 24 | 2 | 3 | 1 | 2 | 0 | 0.5928 |
| B_e70 | frozen_real_bg | 2 | 1 | 0 | 1 | 0 | 0 | 0 | 0.1628 |

**解读**：

- B_e10 把 COCO 背景的长尾从 252 条削到 46 条、@0.25 的 FP 从 10 降到 1、最大误检置信度 0.6546 → 0.2900，是真实的正向变化；
- 但 B_e10 在 `frozen_real_backgrounds` 与 `bg_val` 上反而更多（3 vs 2、5 vs 1），其中 `bg_032.jpg`（A 0.5921 / B_e10 0.5928）**框住的是一颗真实羽毛球**（该“背景”图带球，属标注缺失，见 §10.3）；
- B_e70 在所有集合上 FP 都最少（frozen 0、bg 0、coco 4）——**这是欠自信的假象**：它的整体召回掉了 25.6 个百分点，框本来就少；
- 三个检查点在 @0.75 全部为 0 → 生产可把阈值抬到 0.75 消掉这批 FP，但代价是召回（e70 在 0.25 就已掉到 0.6031）。

### 7.3 最高置信度 FP 图片（已存盘，仅保存确实含 ≥0.25 FP 的图）

`outputs/shuttle_capability/hard_negative_eval/{stageA_best,stageB_best_e10,stageB_last_e70}/top20_fp/` → **13 + 4 + 4 = 21 张**。

失败模式（目视 4 例，跨数据集复现）：**6–30 px 的高对比小圆/方块** —— 天花板射灯、壁灯、红色消防标志；另有 1 例（`hn2_056`）是桌面小器件。**没有**整图级乱框。

---

## 8. 判定与下一步

**判定：`STAGE_B_HEALTHY_BUT_NO_VAL_GAIN`**

- 训练过程健康（§4）：无 NaN、LR 全程 ≤0.00291468（限 0.0033）、无坍塌、内存余量 ≥20.6 GB、正确从 Stage A best.pt 续训；
- 交付层面无改进（§5–§6）：`R_6-16` −0.0634、`mAP50` −0.0216、`R` −0.0798，只有 `mAP50-95` +0.0019；
- 唯一正向信号：6–8 px（+0.0248 AP50）与 8–12 px（+0.0023 AP50）——**需要更多证据**。

**下一步（按优先级）**

1. **冻结测试集终局判定**：用 `controlled_capability / challenge_test / synthetic_on_real_bg / synthetic_3d / real_images / real_video / real_match_frames / real_train` 这 8 个**从未参与训练、调参、模型选择**的集合，在 `stageA_best` 与 `stageB_best_e10` 之间做一次判定（`tools/eval_yolo26_v1.py --split frozen_test` 的接线已存在但尚未启用），主判据建议 `R_6-16` + `mAP50-95` 双指标。**不得**用 val 做这一步的裁判（val 已被两次 best 选择用过）。
2. **不要采用 e70**（`stageB_last_e70.pt` 仅作对照保留）。
3. **ISSUE-027**：给 `tools/build_yolo26_v1_dataset.py` 接上 `hard_negatives2/excluded.txt` 排除逻辑并重建清单（训练已结束，可以动清单了）。
4. **ISSUE-028 / 环境**：本地评估环境已重建为 `D:\_eval26\venv`，重建脚本入库；旧的 `D:\_yolo26v1` 外部删除原因不明。
5. **Stage C（15 epoch，mosaic 0）** 与 **Gate 7（640/960/1280）** 排在冻结测试集判定之后。

---

## 9. 结果自证

### 9.1 本地产物与 sha256（全部在仓库内）

| 文件 | 字节 | sha256 |
|---|---:|---|
| `outputs/shuttle_capability/metrics/size_bucket_metrics_val.json` | 12,234 | `33ccf9465bf49ad7cc90e608fc1663ddf6286b1d30ef5083576d681a3ed7b7e4` |
| `outputs/shuttle_capability/metrics/size_bucket_metrics_val.csv` | 2,280 | `7ec4aed281cc2f4db471d8a68541ebb1eb57ee48062eb75b2c65c63daef0dc23` |
| `outputs/shuttle_capability/metrics/size_bucket_three_way_compare_val.csv` | 2,514 | `85c3f7f26b7a7c6db2ff5bcadca73a5bcda3de0b7a22a769659a95eeb37fdd6e` |
| `outputs/shuttle_capability/metrics/size_bucket_checkpoint_compare_val.csv`（工具原生前两检查点对比） | 853 | `ff79c00fc125c0d02b26aaf69bc464cbc84eed8073ee8ddadfeca72ec5dc2f52` |
| `outputs/shuttle_capability/metrics/stage_b_gate_final.json` | 57,732 | `b6cc58f2cd510c60783032ced368b82899f9ce94b71ca0f7f0f6a733161e5b85` |
| `outputs/shuttle_capability/metrics/hard_negative_eval_hard_negative.json` | 2,848 | `7e544bb32272d0bc3b5dc7b66be2a90d681606565009a573ba09407476786c8b` |
| `outputs/shuttle_capability/metrics/hard_negative_eval_normal_neg_coco_val.json` | 3,083 | `7704fe02917f721c8cd953489395ae102a70512cd5df1620b293fad9eebb9928` |
| `outputs/shuttle_capability/metrics/hard_negative_eval_normal_neg_bg_val.json` | 2,998 | `67e6c6ffb178e9b4aa34ad49d666ac27c9d80dda5e1fdbd3364752de70effed2` |
| `outputs/shuttle_capability/metrics/hard_negative_eval_frozen_real_backgrounds.json` | 3,101 | `1f2d32c150d16aaebef9665dc0ad39e9c7ea3a26534887be72b8d9e341392064` |
| `outputs/shuttle_capability/metrics/hard_negative_predictions.csv`（逐框原始预测） | 81,185 | `3f33b7da4066bf54cbc3c6716a9984c9aa2b6e106aa3b3fc6bf729009f7c222b` |
| `outputs/shuttle_capability/metrics/hard_negative_fp_confidence_hist.csv` | 819 | `bb27f27b7e3728e16d2a3c2df11c4e1e108d9910029eb308a3b4e729511fafd9` |
| `outputs/shuttle_capability/metrics/val_images.txt`（本次推理的 4,413 张图清单） | 361,411 | `2e80ed1e57012619ee44f478e8a1fb5ad126ad8cbb854a401d28d7c271f02b44` |

### 9.2 资源位置（**不在本地**；`*.pt` 属资源）

| 资源 | 远端位置 | 校验量 |
|---|---|---|
| Stage A best（20,125,499 B） | `/home/T7/ojh/robot_sim/runs/shuttle_yolo26_v1/gate6A_imgsz1024_b16_w8_20260926-083331/weights/best.pt` | 本地副本 sha256 `3214aaf15fd9d9a238dce81db8932467b3c76df7251cba5c954a2b29e9b95a33` |
| Stage B best e10（20,136,763 B） | `…/gate6B_lr0001_musgd_b16_w8_20260927-235730/weights/best.pt` | sha256 `419a3eef…97208b` |
| Stage B last e70（20,136,763 B） | 同上 `weights/last.pt` | sha256 `42f0043a…d7f98a4` |
| ETH 数据集（29,377 张） | `/home/T7/dgut/robot_sim/eth_shuttle_detection` | 清单 30,321 行（sha256 `11d85026…ed539`，与本地一致） |
| 负样本原图 | `/home/T7/ojh/robot_sim/outputs/shuttle_capability/{hard_negatives,hard_negatives2}/raw`、`real_images/backgrounds` | 36 / 60 / 30 张 |
| ETH 官方代码+权重 | `/home/T7/dgut/robot_sim/eth_official_code/` | 73 文件 / 268,910,149 B，逐文件 sha256 见 `eth_official_repo_inventory.csv` |

### 9.3 再生方式（逐字节复现）

```powershell
# 0) 环境（可重建）：D:\_eval26\setup_eval_env.ps1 + D:\_eval26\install_cuda.ps1
# 1) 三方 val 尺寸分桶（约 27 min，RTX 4060）
& experiments\yolo26_v1\run_stageb_final_eval_local.ps1
# 2) 困难负样本 4 集合 × 3 检查点（约 6 min）
& experiments\yolo26_v1\run_hardneg_sets.ps1
# 3) 汇总表（不跑推理）
D:\_eval26\venv\Scripts\python.exe tools\compare_size_buckets.py --repo .
D:\_eval26\venv\Scripts\python.exe tools\compare_hard_negative_eval.py --repo .
D:\_eval26\venv\Scripts\python.exe tools\hard_negative_confidence_hist.py --repo .
D:\_eval26\venv\Scripts\python.exe tools\render_top_fp.py --repo . --min-confidence 0.25 --topk 20
```

---

## 10. 事故、数据卫生与已知缺口（须随报告一起读）

### 10.1 监控空档（诚实记录）

原只读监控进程在 2026-09-28T15:40:54（记下 epoch 40 后）死亡 —— 与“会话后台作业/子进程被父进程退出带走”一致；改用 `Start-Process` 的 detached 单实例（PID 59732）也只存活到下一次工具调用结束。**因此 epoch 41–70 没有实时事件日志**。替代证据：① `results.csv` 70 行全表扫描（无 NaN、指标单调）；② `mem_probe.jsonl` 1,463 条（MemAvailable 最低 20.6 GB）；③ `lr_probe.jsonl` 146,440 条（最大 0.00291468）。**本报告结论不依赖实时监控**。

### 10.2 工具缺陷（已修）

`tools/eval_hard_negative.py` 原先以 `mode="a"` 追加写 `hard_negative_predictions.csv` → **重复运行会重复计数**（本次第一次跑就出现过 3 个集合只有 2 检查点的混态）。已改为**幂等**：重算的 `(checkpoint, set)` 行被替换、其他集合保留。另确认该工具 `--images` 是**单值参数**（一次只评一个集合），故运行脚本按集合调用 4 次（`run_hardneg_sets.ps1`）。

### 10.3 标注缺失（数据侧问题，不是模型问题）

`frozen_real_backgrounds` 中 `bg_032.jpg` 的“FP”（A 0.5921 / B_e10 0.5928，两代几乎相同）**框住的是一颗真实羽毛球** —— 该目录被当作纯背景，标签为空。与历史 ISSUE-021 同源，需补标或剔除。

### 10.4 ISSUE-027（未修）

`hard_negatives2/excluded.txt` 的 7 张排除图（hn2_010/011/012/013/015/016/017）**仍在清单里且 split=train**，即被 Stage A/B 训练过；builder 从未读取该排除名单。修法排在训练结束之后（现在已可执行）。

### 10.5 ISSUE-028（本地环境被外部删除）

`D:\_yolo26v1`（venv + 解包的 ultralytics wheel + `_deps` + `cfg`）在 2026-09-29 被发现整体消失，非本会话所为；已重建为 `D:\_eval26\venv` 并把安装配方写进仓库脚本。远端 `env_isaaclab/bin/python -c "import torch"` 当日 `timeout 900` 未完成（FUSE 读挂起、loadavg 13.9），故本轮评估在本地 GPU 完成。

### 10.6 尚未做的事

- **冻结测试集评估**（8 个集合）—— 终局判定必需，尚未运行；
- `tools/audit_yolo26_v1_run.py`（规范 §28 要求）—— 尚未实现，`stage_b_gate_final.json` 目前承担了部分职能；
- `eval_yolo26_v1.py` 的 hard-negative 段与 `--window-epochs` —— 未实现（本次用独立工具代替）；
- Stage C 与 Gate 7 —— 未开始。
