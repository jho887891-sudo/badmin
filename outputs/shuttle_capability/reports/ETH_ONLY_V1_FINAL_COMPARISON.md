# ETH-Only YOLO26s 1024 Baseline V1 — 统一对比报告（Task 7）

> 实验：`eth_only_yolo26s_1024_v1`｜契约 `configs/eth_only_yolo26s_1024_v1.yaml`｜spec `docs/superpowers/specs/2026-10-02-eth-only-yolo26s-1024-baseline-v1-design.md`
> 报告日期：2026-10-05｜数据窗口：训练 2026-10-03 03:55:49Z–11:19:33Z，评估 2026-10-03–05，代价实测 2026-10-05 02:08:05Z–02:16:21Z
> **口径声明**：计划原文（`ETH_ONLY_YOLO26S_1024_BASELINE_V1_IMPLEMENTATION_PLAN.md`）当初以附件形式提供、**未入仓库**，第 7 节的"八问"是按冻结契约与 spec 重述的；若与计划措辞不同，以计划为准并回来修订本节。

## 0. 摘要（先看这里）

1. **ETH-only（纯 ETH 真实数据、无合成、无 iPhone、剔除我们全部评估帧）训出来的 YOLO26s，在唯一公平的真实域集合 `val|eth_unseen` 上明显弱于 ETH 官方 yolov8s**：R 0.1636 vs 0.5434、mAP50-95 0.1909 vs 0.4261（Δ 远超计划的可行动阈值 0.05）。
2. **它同时弱于我们既有的 v1 模型**（a_best R 0.5576 / b_best R 0.4869）——差距不是"合成 vs 真实"造成的，而是**训练数据覆盖面**：本实验只用了 ETH 12 个 location 中的 7 个 + 550 张 coco 负样本（batch 8/nbs 32），而 v1 混合物还有合成、iPhone、困难负样本。
3. **去掉合成数据后，合成域能力基本归零**：controlled_capability R 0.0053（v1 的 a_best/b_best 是 0.1734/0.2527）、challenge_test R 0.0052（v1 是 0.0464/0.0722）。→ v1 的合成域能力来自其合成训练数据，**不是真实域泛化**；这条是本实验最硬的结论。
4. **代价上它比 P2 便宜**：imgsz 1024/batch1 15.62 ms（P2 18.19–18.95）、batch64 吞吐 126.9 img/s（P2 ~80）、batch64 显存 10.1 GB（P2 16.6 GB）；但仍比 ETH 官方 yolov8s 慢（8.28 ms / 157.5 img/s）。
5. **泄漏纪律成立且可复核**：训练池剔除 2,765 张评估帧，`eval_overlap_images 0`、`sha256_overlap []`、`location_overlap []`、audit PASS；选点仅用内部 location-disjoint val 的 mAP50-95（epoch 21），且 fitness argmax 与之一致。

## 1. 契约与证据链（可自证）

| 项 | 值 |
|---|---|
| 权重（初始化） | 官方 `yolo26s.pt` **20,422,725 B / sha256 `646f8bc3fe0a656803d95c294f7852321748cb29d13466a1af8862e2db384a1b`**（训练启动前校验通过） |
| 训练数据真源 | 远端 `/home/T7/dgut/robot_sim/eth_shuttle_detection`（本仓库只有 manifest 与列表） |
| 冻结 split | `data/eth_only_v1_train_manifest.csv` 14,543 行 / `data/eth_only_v1_val_manifest.csv` 2,920 行 |
| 训练产物（资源，远端为真源） | `/home/dgut/.dsh-bench/runs/detect/runs_eth_only_v1/full_e50/weights/best.pt` **20,344,069 B / sha256 `7a836a2621686affbdd3f1da7c3a0a57c4b86f14432835be2bc562c2899fc6f2`** |
| 训练 manifest | `outputs/shuttle_capability/metrics/eth_only_v1_full_e50_manifest.json`（3350 B, sha16 `cf99df172841168b`） |
| 训练曲线（结果） | `outputs/shuttle_capability/metrics/eth_only_v1_full_e50_results.csv`（6143 B, sha16 `a8688b9f81ff8482`） |
| 第三方差照权重 | ETH 官方 `runs/final-model/best.pt` 134,312,133 B / sha256 `f1aea7dec784a24d53d475f89510ef45fc13255df6298c6a0ea2fbd4419c7d6d`（评估前重新校验一致；本地副本已删，远端为真源） |
| 评估器 | `tools/eval_yolo26_v1.py`（SSOT）+ `tools/eval_eth_official_baseline.py`（同模块复用，parity 由 `tests/test_eth_only_v1_eval_parity.py` 钉死） |
| 相关提交 | `52cd0dc`(数据冻结) `f2bed61`(启动器) `a7ce718` `8d834fc` `ef313d8`(3 处启动器修复) `3c955e6`(训练产物) `1d0e384`(早期工具入库) `743d5e4`(公平对比) |

## 2. 数据：冻结 split 的无泄漏证据

- train 14,543 = **13,992** ETH 真实正样本（easy+medium，7 个 location：cab_1/cab_2/glc_1/ml_4/ticino_1/ticino_2 + coco_train 负样本 550）+ 1 张空标签正样本位置图；val **2,920**（glc_2 1,613 + uetlibergstrasse_2 1,307），`val_positive_fraction` **0.17265846736045412**
- 剔除我们**全部**评估帧 **2,765** 张（ml_3 1,004 / ml_6 1,123 / uetlibergstrasse_1 638，见 ISSUE-031）；审计 `eval_overlap_images 0`、`eval_hash_overlap_images 0`、`sha256_overlap []`、`location_overlap []`、`nonempty_negative_labels 0`、`unreadable_images 0`、audit **PASS**
- 箱级尺寸分布（13,992 boxes）：<4 161 / 4-6 1,282 / 6-8 4,522 / 8-12 5,599 / 12-16 1,691 / 16-24 596 / 24-32 97 / 32-64 41 / >64 3
- ⚠ **内部 val 不是公平比较集**：它的 2,920 行**全部来自 ETH 官方 `images/train` 子目录**（`in_official_training=True`）→ 只能用于本模型选点，不能用于与 ETH 官方对比（对比一律用 `val|eth_unseen`）

## 3. 训练与选点

- 配方：plain YOLO26s、`imgsz 1024`、`freeze 0`、`AdamW lr0 1e-4`、`epochs 50`、`nbs 32`、`batch 8`（回退 6/4 未触发）、ETH 官方增强/loss 值；与 ETH 官方的**唯一偏离**是 batch/nbs（32/64 → 8/32，24 GiB 预算所迫）
- 选点（内部 val 2,920 帧，**只用 mAP50-95**）：**epoch 21 → mAP50-95 0.67733 / mAP50 0.96384 / R 0.91448**；fitness argmax 同为 e21（已核对）
- 曲线：e1 0.4086 → e5 0.6483 → e10 0.6530 → e20 0.6671 → **e21 0.6773** → e30 0.6745 → e40 0.6706 → e50 0.6709；top6 = 0.6756–0.6773（平台期，无过拟合崩塌）
- 运行：24 GiB 单进程上限（fraction 0.5063）、`WANDB_MODE=disabled`（否则 ultralytics 会自动上传，见 ISSUE-034）、显存 6.0–6.7 GB、7.4 h

## 4. 四集合 × 五模型（同一评估器：conf 0.25 / AP floor 0.001 / NMS iou 0.7 / max_det 300 / imgsz 1024）

数据来源：`eth_vs_ours_eth_only_v1.csv`（本次）与 `eth_vs_ours.csv`（上一轮，同为同一工具产出）。

| 集合 | 模型 | images | GT | P | R | AP50 | mAP50-95 | Recall_<8 | TP_<8 |
|---|---|---|---|---|---|---|---|---|---|
| val（含 2,765 张 ETH 训练帧） | eth_official | 4413 | 3260 | 0.9780 | 0.9288 | 0.9396 | 0.7767 | 0.7341 | 254 |
| val | a_best | 4413 | 3260 | — | 0.8595 | 0.9051 | 0.5907 | 0.6532 | 226 |
| val | b_best | 4413 | 3260 | — | 0.7798 | 0.8835 | 0.5926 | 0.6532 | 226 |
| val | **eth_only_v1_best** | 4413 | 3260 | 0.9752 | 0.5549 | 0.7638 | 0.5391 | 0.3555 | 123 |
| **val\|eth_unseen（公平）** | eth_official | 1648 | 495 | 0.8102 | **0.5434** | 0.6133 | **0.4261** | 0.1714 | 18 |
| **val\|eth_unseen（公平）** | a_best | 1648 | 495 | — | **0.5576** | 0.6779 | 0.3801 | 0.2000 | 21 |
| **val\|eth_unseen（公平）** | b_best | 1648 | 495 | — | 0.4869 | 0.6516 | 0.3708 | **0.2571** | 27 |
| **val\|eth_unseen（公平）** | **eth_only_v1_best** | 1648 | 495 | 0.6864 | **0.1636** | 0.3510 | **0.1909** | 0.0762 | 8 |
| controlled_capability | eth_official | 2070 | 2070 | 0.0463 | 0.0024 | 0.0175 | 0.0124 | 0.0 | 0 |
| controlled_capability | a_best | 2070 | 2070 | — | 0.1734 | 0.3659 | 0.2303 | 0.1125 | 36 |
| controlled_capability | b_best | 2070 | 2070 | — | **0.2527** | 0.4366 | 0.3015 | 0.0938 | 30 |
| controlled_capability | **eth_only_v1_best** | 2070 | 2070 | 0.2292 | 0.0053 | 0.0669 | 0.0455 | 0.0 | 0 |
| challenge_test | eth_official | 227 | 194 | 0.0 | 0.0 | 0.0103 | 0.0072 | 0.0 | 0 |
| challenge_test | a_best | 227 | 194 | — | 0.0464 | 0.1840 | 0.0972 | 0.0377 | 4 |
| challenge_test | b_best | 227 | 194 | — | **0.0722** | 0.2339 | 0.1170 | 0.0660 | 7 |
| challenge_test | **eth_only_v1_best** | 227 | 194 | 0.0909 | 0.0052 | 0.0934 | 0.0535 | 0.0 | 0 |

关键差（新模型 − 对照物，计划阈值 |Δ|<0.02 不确定 / 0.02–0.05 待确认 / >0.05 可行动）：

| 对照 | 集合 | ΔR | Δ mAP50-95 | 判定 |
|---|---|---|---|---|
| eth_official | val\|eth_unseen | **−0.3798** | **−0.2352** | 可行动（ETH 更好） |
| a_best | val\|eth_unseen | −0.3940 | −0.1892 | 可行动（a_best 更好） |
| b_best | val\|eth_unseen | −0.3232 | −0.1799 | 可行动（b_best 更好） |
| eth_official | controlled_capability | +0.0029 | +0.0331 | R 不确定 / AP 待确认；绝对量级都为 0 |
| eth_official | challenge_test | +0.0052 | +0.0463 | 同上 |
| b_best | controlled_capability | **−0.2474** | **−0.2560** | 可行动（b_best 更好） |
| b_best | challenge_test | **−0.0670** | **−0.0635** | 可行动（b_best 更好） |

## 5. 小目标与尺寸桶

- `val|eth_unseen` 的 <8px GT 有 105 个：ETH 官方召回 18 个、b_best 27、a_best 21、**eth_only_v1_best 只有 8 个（Recall 0.0762）** → 新模型在小目标上是"最差的一档"，不是"更保守"能解释的（它整体 R 也只有 0.1636）。
- 分桶明细：`outputs/shuttle_capability/metrics/eth_only_v1_val_size_buckets.{json,csv}`（val）、`eth_vs_ours_eth_only_v1_size_buckets.csv`（四集合）。

## 6. 代价（同一协议：A6000、fp32、process 显存上限 24 GiB、同点交错重复）

产物：`outputs/shuttle_capability/metrics/gpu_bench_a6000_24g_eth_only_v1.{json,csv}`（本次，含 4 模型，02:08:05Z–02:16:21Z）与上一轮 `gpu_bench_a6000_24g.{json,csv}`（3 模型）。

| 模型 | 参数 | GFLOPs | imgsz1024/batch1 | batch64 吞吐 | batch64 显存 | batch32 显存 |
|---|---|---|---|---|---|---|
| eth_official (yolov8s) | 11,166,560 | 73.76 | **8.28 ms / 120.8 fps** | **157.5 img/s** | **9,196 MB** | 4,693 MB |
| eth_only_v1_best (yolo26s) | 9,948,638 | 59.19 | 15.62 ms / 64.0 fps | 126.9 img/s | 10,127 MB | 5,159 MB |
| a_best (yolo26-P2) | 9,663,464 | 69.09 | 18.19 ms / 55.0 fps | 80.0 img/s | 16,639 MB | 8,415 MB |
| b_best (yolo26-P2) | 9,663,464 | 69.09 | 18.95 ms / 52.8 fps | 79.9 img/s | 16,639 MB | 8,415 MB |

- 24 GiB 部署卡：四者都不触上限（最高 16.6 GB 在 batch 64）；新模型在 batch 32/64 分别 5.2 / 10.1 GB。
- 新模型 vs ETH 官方：batch1 延迟 **1.9×**（15.62 vs 8.28 ms），batch64 吞吐 **−19%**（126.9 vs 157.5 img/s）。
- 新模型 vs P2（a_best）：FLOPs 少 14%（59.19 vs 69.09 GFLOPs），batch1 延迟低 14%（15.62 vs 18.19 ms），batch64 吞吐高 **1.6×**（126.9 vs 80.0 img/s）→ P2 的成本主要来自高分辨率特征图，而非参数量（9.95 M vs 9.66 M）。
- **每 FLOP 效率**：batch1 下 yolov8s 73.76 GFLOPs / 8.28 ms ≈ **8.9 TFLOP/s**，plain yolo26s 59.19 / 15.62 ≈ **3.8 TFLOP/s** → 同卡 fp32 下 YOLO26s 的算子效率约为 yolov8s 的 43%，这是"新模型更慢"的主因（不是参数量或 FLOPs）。

## 7. 计划八问（按冻结契约重述，逐条给证据与判定）

**Q1 ETH-only 真实数据训练的 YOLO26s 能否在未见过的真实域上超过 ETH 官方 yolov8s？**
不能。`val|eth_unseen` R 0.1636 vs 0.5434、mAP50-95 0.1909 vs 0.4261（Δ 远超 0.05 阈值）。
**Q2 去掉合成数据后，合成域能力还剩多少？**
基本归零：controlled R 0.0053（v1 b_best 0.2527）、challenge R 0.0052（v1 b_best 0.0722）。Δ 均属"可行动"。
**Q3 数据/评估口径是否无泄漏且可复核？**
是。train/val location-disjoint、sha256 无交集、剔除 2,765 张评估帧、audit PASS；内部 val 全部来自 ETH `images/train`，因此**只用于选点**，不参与任何对比。
**Q4 选点是否只用内部 val，未碰任何评估集？**
是。`selection_scope=internal_validation_only`，epoch 21（mAP50-95 argmax，且 = fitness argmax），manifest 记录四个评估集 `never_used_for_selection`。
**Q5 小目标（<8px）表现如何？**
`val|eth_unseen` 105 个 <8px GT，新模型只召回 8 个（0.0762），低于 ETH 官方（18）与 a/b_best（21/27）。
**Q6 代价是否满足 24 GiB 部署卡？**
满足：最坏 batch64 10.1 GB（上限 24 GiB 从未触发）；batch1 15.62 ms / 64 fps，可满足单目实时需求。代价相对位置：比 ETH 官方慢 1.9×（batch1 延迟），比我们的 P2 快 1.6×（batch64 吞吐）。
**Q7 与既有 v1 模型相比，谁是更好的工程选择？**
按测量值：真实域 `val|eth_unseen` a_best 0.5576 > b_best 0.4869 > eth_official 0.5434(R)/0.4261(mAP) > eth_only_v1_best 0.1636；合成域 b_best 全面最好；代价上新模型最省显存/吞吐更好。→ 若目标是本项目的能力集（含合成域），**b_best 仍是当前更好的选择**；新模型的价值在于提供了一个"无泄漏、可复现的纯真实数据下界"。
**Q8 下一个实验应该做什么？**
见第 9 节（推荐 A：泄漏干净化的 v1 混合物重训；B：把 batch/nbs 对齐 ETH 做算力对等；C：低成本先做 val|eth_unseen 的误差分析）。

## 8. 局限（不写成结论的话）

1. **不是纯"数据口径"实验**：新模型同时比 ETH 官方少了 5 个 location、全部 iPhone/合成/困难负样本，并把 batch/nbs 从 32/64 降到 8/32、学生网络换了（yolov8s→yolo26s）。因此 Q1 的差距不能单独归因于"ETH-only 数据"。
2. `val|eth_unseen` 只有 **495 GT**（<8px 仅 105），置信区间宽；任何 ±0.02 级别的差异在这个集合上都不足以下结论（本轮主要差异都 >0.05，故结论仍稳）。
3. controlled_capability / challenge_test 对**所有**模型都近乎地板（R<0.26），只能做排序不能做能力声明；ETH 官方在这两个集合上基本失效（R 0.0024/0.0000）。
4. 合成集与真实集不可混比；比较只在同一集合、同一评估器内成立（本报告已按此组织）。
5. 报告与计划原文的"八问"措辞可能不同（计划未入库）；如不一致需回来改第 7 节。

## 9. 下一个实验的决策（按性价比排序）

**A（推荐，直接回答本轮暴露的问题）：把 v1 混合物"泄漏干净化"后重训**
- 做法：沿用 v1 的数据构成（ETH 真实 + 合成 + iPhone + 困难负样本），但用已冻结的 2,765 张剔除名单把评估帧全部移出训练，并保持 location-disjoint 划分；配方与本次一致（50 epoch / imgsz 1024 / batch 8 / nbs 32），另跑一个 batch 32/nbs 64 的对等版本。
- 预期回答：v1 的领先到底来自"数据多样性"还是"泄漏"——本轮已证明它**不来自** ETH-only 的真实域泛化。
- 成本：每条 ~7.4 h（远端共享 A6000，24 GiB 上限），两条可串行一晚完成。
**B（对照，算力对齐）**：把 ETH-only 配方提到 batch 32 / nbs 64，看差距缩小多少（区分"数据"与"优化预算"）。
**C（零训练成本，先做）**：对 `val|eth_unseen` 的 414 个 FN 做误差分析（尺寸/位置/置信度分布 + 与 ETH 官方 TP 的交集），产出"失败模式"表，为 A/B 的数据配比提供依据。

## 10. 复现命令（照抄）

```bash
# 1) 数据冻结（远端，约 7 min；真源 /home/T7/dgut/robot_sim/eth_shuttle_detection）
bash tools/remote/run_build_v1b.sh                 # 内置 SELF-CHECK：train 14543 / val 2920 / coco 550
# 2) 训练（远端，50 epoch，约 7.4 h）
bash tools/remote/run_full_v1.sh full_e50           # WANDB_MODE=disabled、24 GiB 上限、内部 val 选点
# 3) 评估（本地；权重只放 _scratch_*，用完即删）
scp dgut@jxxy.taildd42cc.ts.net:/home/dgut/.dsh-bench/runs/detect/runs_eth_only_v1/full_e50/weights/best.pt _scratch_eth_only_v1/best.pt
python tools/eval_yolo26_v1.py --repo . --split val --ckpt-name eth_only_v1_best --out-dir _scratch_eth_only_v1_eval\val_full
python tools/eval_eth_official_baseline.py --repo . --eth-ckpt _scratch_eth_official/best.pt \
  --models eth_official,eth_only_v1_best --out-dir _scratch_eth_only_v1_eval\fair --no-examples --no-latency
# 4) 代价（远端，约 8 min）
bash tools/remote/run_bench_eth_only_v1.sh
python tools/remote/bench_json_to_csv.py gpu_bench_a6000_24g_eth_only_v1.json gpu_bench_a6000_24g_eth_only_v1.csv
```
