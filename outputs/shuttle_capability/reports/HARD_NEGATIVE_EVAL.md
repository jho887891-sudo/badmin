# HARD_NEGATIVE_EVAL —— 困难负样本假阳性评估（Stage A best vs Stage B best@epoch10）

> 生成时间：2026-09-28　｜　评估设备：本地 Windows，`--device cuda:0` = **本机 RTX 4060 Laptop 8 GB**（torch 2.14.0+cu126，venv 位于 `D:\_yolo26v1\venv`）
> 评估点：Stage B **仍在训练中（未修改、未中断）**，本报告用的是 epoch 10 的 `best.pt` 快照。
> 所有结论均可由 `outputs/shuttle_capability/metrics/hard_negative_*` 与 `outputs/shuttle_capability/hard_negative_eval/**` 自证。

---

## 1. 目的与范围

回答四个问题：

1. Stage B 相比 Stage A，在**未见过**的普通负样本（COCO 背景 + 项目背景）上的假阳性（FP）是变好还是变差？
2. 在**训练过的**困难负样本上，FP 是否被压下去（记忆性检验，不是泛化结论）？
3. FP 的置信度分布长什么样，工作点 0.25 / 0.50 / 0.75 各留多少 FP？
4. 置信度最高的那些 FP 到底长什么样（图片证据 + 失败模式）？

本报告**只评假阳性**，不评召回；召回/尺寸分桶结论见 `outputs/shuttle_capability/reports/SIZE_BUCKET_EVAL_VAL.md`。
FP **不做尺寸分桶**（沿用计划 A：负样本图无 GT 框，分桶无意义）。

---

## 2. 结论速览（TL;DR）

| 问题 | 答案 |
|---|---|
| Stage B 在未见过普通负样本上的 FP | **不劣化**：COCO val 1000 张 0.0100 → 0.0010 FP/图（@0.25），2 FP → 0（@0.5） |
| Stage B 在项目背景 val（6 张，未见过） | 单图异常：A 1 FP → B 5 FP（@0.25），但**含 FP 的图都是同一张**（image FP rate 均 1/6 = 0.1667），@0.75 双双归零 |
| Stage B 在训练过的困难负样本（96 张） | **变好**：2 FP → 0 FP（@0.25） |
| 冻结真实羽毛球背景（30 张，从未训练） | 略差：2 → 3 FP（@0.25），1 → 2 FP（@0.5），最大 FP 置信度几乎不变（0.59207 → 0.59277） |
| 判定 | `PASS_FP_NO_REGRESSION`，附带 `WATCH_bg_negative_single_image`（样本量 6，统计力不足） |
| 附带发现 | 冻结背景集里那张 0.593 的“FP”**框住的是一颗真实羽毛球**——那是**标注缺失**，不是误检（见 §10.2） |
| 附带发现 | `hard_negatives2/excluded.txt` 里排除的 7 张图**仍然在训练清单里**（split=train），已被 Stage B 训练（见 §11） |

---

## 3. 评估口径与可复现命令

- 推理输入尺寸：`imgsz=1024`（与 Stage A / Stage B 训练一致，不做 640 降级）
- 置信度采集下限：`--conf-floor 0.01`（保证置信度分布能看到长尾）
- 报告工作点：`0.25 / 0.50 / 0.75`
- FP 定义：负样本图上**任何**输出框都算 FP（这些集合无 GT 框；评估器会跳过带正样本标注的图，本次 `skipped_because_positive_label` 全为 0）
- 统计量：`total_FP`、`FP_per_image`（FP/总图数）、`images_with_FP`、`image_FP_rate`（含 FP 的图/总图数）、FP 置信度 max / mean / median / P95
- 检查点：`stageA_best` = Stage A epoch 6 `best.pt`（20,125,499 B）；`stageB_best_e10` = Stage B epoch 10 `best.pt`（58,948,377 B）

```powershell
# 1) 主评估（4 个负样本集 × 2 个检查点）
D:\_yolo26v1\venv\Scripts\python.exe tools\eval_hard_negative.py --repo . \
  --images manifest:hard_negative:train --set-name hard_negative \
  --images manifest:coco_bg:val        --set-name normal_neg_coco_val \
  --images manifest:bg_negative:val    --set-name normal_neg_bg_val \
  --images dir:outputs/shuttle_capability/real_images/backgrounds --set-name frozen_real_backgrounds \
  --ckpt stageA_best=D:\_eth_dl\ckpts\stageA_best.pt \
  --ckpt stageB_best_e10=D:\_eth_dl\ckpts\stageB_best.pt \
  --thresholds 0.25,0.50,0.75 --conf-floor 0.01 --imgsz 1024 --device cuda:0 --topk 20

# 2) A/B 对比表 + FP 置信度分布 + Top-K 图片重绘
D:\_yolo26v1\venv\Scripts\python.exe tools\compare_hard_negative_eval.py --repo .
D:\_yolo26v1\venv\Scripts\python.exe tools\hard_negative_confidence_hist.py --repo .
D:\_yolo26v1\venv\Scripts\python.exe tools\render_top_fp.py --repo . --min-confidence 0.25 --topk 20
```

四个数据集的身份来自冻结清单 `outputs/shuttle_capability/v1_dataset/v1_dataset_manifest.csv`（30,321 行，sha256 `11d850263b370fc67b78b1264609a5a3d2f79d9f235f9d943bfb322de91ed539`）。

---

## 4. 数据集身份：这些困难负样本到底训练过没有？

**结论：训练过。** 96 张困难负样本**全部**是 `split=train`，即 Stage A 与 Stage B 都见过它们。

`v1_dataset_manifest.csv` 按 source × split 统计：

| source | train | val | total | 本次评估用法 |
|---|---:|---:|---:|---|
| `hard_negative` | 96 | **0** | 96 | 困难负样本集（**已训练**） |
| `bg_negative` | 25 | 6 | 31 | 普通负样本（6 张 val 未训练） |
| `coco_bg` | 5,500 | 1,000 | 6,500 | 普通负样本（1,000 张 val 未训练） |
| `eth_main` | 17,481 | 3,028 | 20,509 | 正样本主体（未用于本报告） |
| `eth_iphone` | 2,133 | 235 | 2,368 | 正样本主体（未用于本报告） |
| `synthetic` | 673 | 144 | 817 | 合成正样本（未用于本报告） |

另有 `outputs/shuttle_capability/real_images/backgrounds`（30 张）**从未进入任何 split**（builder 硬排除目录 `real_images`），属于真正的“冻结未见背景”。

> 因此：**只有 `normal_neg_coco_val`(1,000) 、`normal_neg_bg_val`(6) 、`frozen_real_backgrounds`(30) 可以当泛化证据；`hard_negative`(96) 只能当记忆/拟合证据。**

---

## 5. 总表：Stage A vs Stage B（阈值 0.25 / 0.50 / 0.75）

来源：`outputs/shuttle_capability/metrics/hard_negative_eval_compare.csv`（sha256 `2FF53A29…F7A805D1`）。

| 集合 | 训练过 | 图数 | 阈值 | A FP | A FP/图 | A 含FP图 | A 图FP率 | A 最大FP置信度 | B FP | B FP/图 | B 含FP图 | B 图FP率 | B 最大FP置信度 | ΔFP/图 |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| hard_negative | ✔ | 96 | 0.25 | 2 | 0.0208 | 2 | 0.0208 | 0.40142 | **0** | **0.0000** | 0 | 0.0000 | — | **−0.0208** |
| hard_negative | ✔ | 96 | 0.50 | 0 | 0.0000 | 0 | 0.0000 | — | 0 | 0.0000 | 0 | 0.0000 | — | 0 |
| hard_negative | ✔ | 96 | 0.75 | 0 | 0.0000 | 0 | 0.0000 | — | 0 | 0.0000 | 0 | 0.0000 | — | 0 |
| normal_neg_coco_val | ✘ | 1000 | 0.25 | 10 | 0.0100 | 8 | 0.0080 | 0.65460 | **1** | **0.0010** | 1 | 0.0010 | 0.28996 | **−0.0090** |
| normal_neg_coco_val | ✘ | 1000 | 0.50 | 2 | 0.0020 | 2 | 0.0020 | 0.65460 | **0** | **0.0000** | 0 | 0.0000 | — | **−0.0020** |
| normal_neg_coco_val | ✘ | 1000 | 0.75 | 0 | 0.0000 | 0 | 0.0000 | — | 0 | 0.0000 | 0 | 0.0000 | — | 0 |
| normal_neg_bg_val | ✘ | 6 | 0.25 | 1 | 0.1667 | 1 | 0.1667 | 0.36617 | 5 | 0.8333 | 1 | 0.1667 | 0.68402 | +0.6667 |
| normal_neg_bg_val | ✘ | 6 | 0.50 | 0 | 0.0000 | 0 | 0.0000 | — | 3 | 0.5000 | 1 | 0.1667 | 0.68402 | +0.5000 |
| normal_neg_bg_val | ✘ | 6 | 0.75 | 0 | 0.0000 | 0 | 0.0000 | — | 0 | 0.0000 | 0 | 0.0000 | — | 0 |
| frozen_real_backgrounds | ✘（从未入集） | 30 | 0.25 | 2 | 0.0667 | 2 | 0.0667 | 0.59207 | 3 | 0.1000 | 2 | 0.0667 | 0.59277 | +0.0333 |
| frozen_real_backgrounds | ✘（从未入集） | 30 | 0.50 | 1 | 0.0333 | 1 | 0.0333 | 0.59207 | 2 | 0.0667 | 2 | 0.0667 | 0.59277 | +0.0333 |
| frozen_real_backgrounds | ✘（从未入集） | 30 | 0.75 | 0 | 0.0000 | 0 | 0.0000 | — | 0 | 0.0000 | 0 | 0.0000 | — | 0 |

**读表要点**

- 最可信的一行（1,000 张未见过的 COCO 背景）：Stage B 把 FP 从 10 降到 1（@0.25）、2 降到 0（@0.50），最大误检置信度从 0.6546 降到 0.2900——**误检的“自信程度”整体下降**。
- `normal_neg_bg_val` 的 +0.6667 是**单张图贡献 5 个框**造成的（含 FP 的图数量没变，仍然 1/6）。样本量 6，**不足以支撑任何结论**，因此只登记为观察项。
- `frozen_real_backgrounds` 的 +1 个框来自同一张图（`bg_032`），且该框框住的是**真实羽毛球**（见 §10.2）——属标注缺失，不算模型退化。

---

## 6. 普通负样本（未见过）明细

### 6.1 `normal_neg_coco_val`（1,000 张，COCO 背景，split=val）

| 检查点 | 阈值 | FP 总数 | FP/图 | 图 FP 率 | 最大置信度 | 均值 | 中位数 | P95 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Stage A best | 0.25 | 10 | 0.0100 | 0.0080 | 0.65460 | 0.41095 | — | — |
| Stage A best | 0.50 | 2 | 0.0020 | 0.0020 | 0.65460 | 0.58621 | — | — |
| Stage B best@10 | 0.25 | 1 | 0.0010 | 0.0010 | 0.28996 | 0.28996 | 0.28996 | 0.28996 |
| Stage B best@10 | 0.50 | 0 | 0 | 0 | — | — | — | — |

### 6.2 `normal_neg_bg_val`（6 张，项目背景，split=val）

| 检查点 | 阈值 | FP 总数 | FP/图 | 图 FP 率 | 最大置信度 | 均值 | 中位数 | P95 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Stage A best | 0.25 | 1 | 0.1667 | 0.1667 | 0.36617 | 0.36617 | 0.36617 | 0.36617 |
| Stage B best@10 | 0.25 | 5 | 0.8333 | 0.1667 | 0.68402 | 0.50984 | 0.56309 | 0.67117 |
| Stage B best@10 | 0.50 | 3 | 0.5000 | 0.1667 | 0.68402 | 0.62229 | 0.61976 | 0.67760 |

5 个框全部落在 `tbg_015.jpg`（1920×2560 室内中庭照，见 §10.1）；其余 5 张图在 0.25 阈值下 0 FP。

### 6.3 冻结真实背景 `frozen_real_backgrounds`（30 张，`real_images/backgrounds`，从未进入任何 split）

| 检查点 | 阈值 | FP 总数 | FP/图 | 图 FP 率 | 最大置信度 |
|---|---:|---:|---:|---:|---:|
| Stage A best | 0.25 | 2 | 0.0667 | 0.0667 | 0.59207 |
| Stage A best | 0.50 | 1 | 0.0333 | 0.0333 | 0.59207 |
| Stage B best@10 | 0.25 | 3 | 0.1000 | 0.0667 | 0.59277 |
| Stage B best@10 | 0.50 | 2 | 0.0667 | 0.0667 | 0.59277 |

两代模型的最高置信度几乎一致（0.59207 vs 0.59277）且指向**同一张图同一个物体**——说明这不是 Stage B 新长出来的误检，而是**两代都存在的标注/负样本定义问题**。

---

## 7. 困难负样本（96 张，**训练过**）明细

| 检查点 | 阈值 | FP 总数 | FP/图 | 图 FP 率 | 最大置信度 |
|---|---:|---:|---:|---:|---:|
| Stage A best | 0.25 | 2 | 0.0208 | 0.0208 | 0.40142 |
| Stage A best | 0.50 | 0 | 0 | 0 | — |
| Stage B best@10 | 0.25 | **0** | **0** | **0** | — |
| Stage B best@10 | 0.50 | 0 | 0 | 0 | — |

Stage A 在 0.25 阈值下的 2 个 FP 位于 `hn2_056.jpg`（0.40142）与 `hn2_057.jpg`（0.278…）；Stage B 把它们压到 0.25 以下。

**这是记忆性证据，不是泛化证据**（96 张全部 split=train、且是 epoch 10 的 best 快照，而 Stage A 只训了 8 epoch）。不要用它宣称“误检率降低 100%”。

---

## 8. FP 置信度分布（采集下限 0.01）

来源：`outputs/shuttle_capability/metrics/hard_negative_fp_confidence_hist.csv`。每一行是该 (检查点, 集合) 在 `conf ≥ 0.01` 下的**全部**预测框（这些集合全是负样本，所以全部是 FP）。

| 检查点 | 集合 | ≥0.01 总数 | 0.01–0.05 | 0.05–0.10 | 0.10–0.25 | 0.25–0.50 | 0.50–0.75 | 0.75–1.00 | 最大 |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Stage A | hard_negative | 61 | 47 | 8 | 4 | 2 | 0 | 0 | 0.40142 |
| Stage B | hard_negative | 37 | 29 | 5 | 3 | 0 | 0 | 0 | 0.18677 |
| Stage A | coco_val | 252 | 189 | 33 | 20 | 8 | 2 | 0 | 0.65460 |
| Stage B | coco_val | 46 | 33 | 7 | 5 | 1 | 0 | 0 | 0.28996 |
| Stage A | bg_val | 10 | 9 | 0 | 0 | 1 | 0 | 0 | 0.36617 |
| Stage B | bg_val | 21 | 10 | 2 | 4 | 2 | 3 | 0 | 0.68402 |
| Stage A | frozen_real_backgrounds | 32 | 21 | 5 | 4 | 1 | 1 | 0 | 0.59207 |
| Stage B | frozen_real_backgrounds | 32 | 24 | 2 | 3 | 1 | 2 | 0 | 0.59277 |

（两组 `hard_negative` 行取自 `hard_negative_fp_confidence_hist.csv`；Stage B 在困难负样本上的最大预测置信度 0.18677 < 0.25，因此 @0.25 起 FP=0。）

**分布形状**：两代模型都是**长尾**——绝大多数 FP 落在 0.01–0.05（噪声级），真正够到工作点 0.25 的只有个位数。Stage B 在 COCO val 上的长尾被削掉约 82%（252 → 46），且 0.25–0.50 段从 8 → 1。
唯一变胖的是 `bg_val`（10 → 21，且 0.50–0.75 段 0 → 3），全部来自同一张 `tbg_015.jpg`。

---

## 9. Top-K 最高置信度 FP 图片（已存盘）

只对**确实含 FP（≥0.25）**的图排序保存（避免把“0 检测”的图当成证据凑数）：

- `outputs/shuttle_capability/hard_negative_eval/stageA_best/top20_fp/` —— **13 张**
- `outputs/shuttle_capability/hard_negative_eval/stageB_best_e10/top20_fp/` —— **4 张**

Stage A（13）：

```
normal_neg_coco_val 0.655 000000139258 | 0.518 000000063355 | 0.444 000000552031 | 0.376 000000440895
normal_neg_coco_val 0.371 000000232770   | 0.309 000000010039 | 0.275 000000275855 | 0.262 000000104880
hard_negative      0.401 hn2_056        | 0.278 hn2_057
normal_neg_bg_val  0.366 tbg_015
frozen_real_backgrounds 0.592 bg_032     | 0.462 bg_016
```

Stage B（4）：

```
normal_neg_coco_val     0.290 000000440895
normal_neg_bg_val       0.684 tbg_015
frozen_real_backgrounds 0.593 bg_032 | 0.568 bg_009
```

文件命名：`<ckpt>_<set>_conf<max置信度>_<原图名>.jpg`；图上用红框画出所有 ≥0.25 的 FP 并标注置信度，左上角黄字写 `ckpt | set | FP#rank max_conf=`。

---

## 10. 失败模式分析（目视 4 例）

### 10.1 `tbg_015.jpg` @0.684（Stage B，bg_val，室内中庭 1920×2560）——**主导模式：小尺寸高对比圆/方斑块**

5 个 FP 的位置分别是：中庭天花板**嵌入式圆形射灯**、柱子上的白色圆形壁灯、深色柱面上的白色小灯、绿植旁的圆形吊灯边缘、以及墙上的**红色方形消防标志**。
共同点：

- 尺寸都在 **6–30 px**（模型的主战场尺寸带），形状近似圆形或小方块；
- 与背景亮度对比极高（白色灯 vs 深色顶棚 / 红色标志 vs 浅色墙）；
- 语义上都不是羽毛球，而是**“小而圆且亮的斑点”**——这是 P2 高分辨率头在 1024 输入下的典型形态假阳性。

### 10.2 `bg_032.jpg` @0.593（Stage B，frozen_real_backgrounds）——**标注缺失，不是误检**

红框正好框住画面中**一颗真实羽毛球**（白色球头朝下、在球拍上方）。训练清单里 `real_images/backgrounds` 的**标签为空**，所以评估器把它算成 FP。
→ 结论：这一族“FP”是**负样本定义错误**（背景图里混进了含球帧），与 ISSUE-021“背景池混入含真实羽毛球的图片”同源。**它污染了 4 个集合中的 1 个**，需在下一版清单里剔除或补标。

### 10.3 `000000139258.jpg` @0.655（Stage A，COCO val，浴室镜面）——**镜面反射的圆形顶灯**

FP 框住镜中天花板的**圆形射灯**。与 10.1 同一模式，跨数据集复现，说明这是模型（而非某个数据集）的固有弱点。Stage B 在同一张图上已经不再出框（本次 Stage B 的 coco 最大 FP 只有 0.290）。

### 10.4 `hn2_056.jpg` @0.401（Stage A，困难负样本，黑白合影）——**桌面小物件**

FP 框住长桌上的**小器件**（笔筒/镇纸一类）。该图**已在训练集中**，Stage A 仍然误检；Stage B 训过后（epoch 10）不再出现 ≥0.25 的框。这解释了为什么 `hard_negative` 集合的 FP 会降到 0——**属于记忆**。

**共性小结**：高置信度 FP = ①小尺寸高对比亮斑（灯、反光、标志）②球类/羽毛类纹理的圆形物③少量标注缺失。**不是**大面积背景误判，也不是整图级“到处乱框”。

---

## 11. 数据卫生发现：`hard_negatives2/excluded.txt` 的 7 张图仍被训练

`outputs/shuttle_capability/hard_negatives2/excluded.txt` 列了 7 个文件名，但 `v1_dataset_manifest.csv` 中它们**全部存在且 split=train**：

| 文件名 | manifest source | split | location |
|---|---|---|---|
| hn2_010.jpg | hard_negative | train | hard_negatives2 |
| hn2_011.jpg | hard_negative | train | hard_negatives2 |
| hn2_012.jpg | hard_negative | train | hard_negatives2 |
| hn2_013.jpg | hard_negative | train | hard_negatives2 |
| hn2_015.jpg | hard_negative | train | hard_negatives2 |
| hn2_016.jpg | hard_negative | train | hard_negatives2 |
| hn2_017.jpg | hard_negative | train | hard_negatives2 |

→ `tools/build_yolo26_v1_dataset.py` **没有读取 excluded.txt**（只按冻结目录硬排除）。影响评估：`hard_negative` 集合的“训练过”结论仍然成立，但这 7 张**本应被排除**的图参与了 Stage A/B 训练。已登记为 **ISSUE-027**（见 `management/ISSUES.md`）。
**修法（待用户确认）**：在 builder 里加入 `hard_negatives2/excluded.txt` 的显式排除，并重建 manifest（会改变已冻结清单 sha256，需与 Stage B 训练解耦：**Stage B 正在跑，不得中途改清单**）。

---

## 12. 结果自证（文件清单 / sha256 / 资源位置）

### 12.1 本地产物（全部在仓库内，可随 git 提交）

| 文件 | 字节 | sha256 |
|---|---:|---|
| `outputs/shuttle_capability/metrics/hard_negative_eval_compare.csv` | 2,036 | `2FF53A290FD6AFE5A0A4D5324ABB846E5546749FA2BB146A1FFB6972F7A805D1` |
| `outputs/shuttle_capability/metrics/hard_negative_eval_hard_negative.csv` | — | `169A59EA9D5F41117366D8DE27596D6F51DCF05948691328E02B559C516D4E8D` |
| `outputs/shuttle_capability/metrics/hard_negative_eval_hard_negative.json` | — | `670F895C07DFBB44C99B03C396356B1E49BAEAD200B4E027D82A8991FA0E48FF` |
| `outputs/shuttle_capability/metrics/hard_negative_eval_normal_neg_coco_val.csv` | — | `9280DCB418A05A79F033A5EDFC89A58BBB1723C34FE7A33C61686405284923B9` |
| `outputs/shuttle_capability/metrics/hard_negative_eval_normal_neg_coco_val.json` | — | `84FB8657D684A1A90437724B65356CBABCA9A9E70E747781D8A3A9A2FE399F1A` |
| `outputs/shuttle_capability/metrics/hard_negative_eval_normal_neg_bg_val.csv` | — | `4FB05C7314BA5099CFBF4EE9E9E54B7EE3B2277E64B95D15035179E3A9F5B158` |
| `outputs/shuttle_capability/metrics/hard_negative_eval_normal_neg_bg_val.json` | — | `CA540C6100049042BD57DFA43E133FDD438A1A59ABB3605B19E82033E66B6950` |
| `outputs/shuttle_capability/metrics/hard_negative_eval_frozen_real_backgrounds.csv` | — | `60EB3A1B6E7639E8E8BD48D91ABB1BEC0E267F72C0900C763D0E18BE46510F2D` |
| `outputs/shuttle_capability/metrics/hard_negative_eval_frozen_real_backgrounds.json` | — | `4532232320BBA5285CDF62784A1AD0708122C4417C7EEB9F83E25F4AD093B891` |
| `outputs/shuttle_capability/metrics/hard_negative_predictions.csv` | 76,116 | （逐框原始预测，见文件） |
| `outputs/shuttle_capability/metrics/hard_negative_fp_confidence_hist.csv` | 581 | （分布表） |
| `outputs/shuttle_capability/hard_negative_eval/**/top20_fp/*.jpg` | 17 张 | 原图路径 + 逐框坐标/置信度见 `hard_negative_predictions.csv`（列：`checkpoint,set,image_path,confidence,x1,y1,x2,y2,prediction_count`），可由 `tools/render_top_fp.py` 逐帧重绘 |

### 12.2 资源位置（**不在本地**）

| 资源 | 位置 | 校验量 |
|---|---|---|
| Stage A best 权重（20,125,499 B） | 远端 `/home/T7/ojh/robot_sim/runs/shuttle_yolo26_v1/gate6A_imgsz1024_b16_w8_20260926-083331/weights/best.pt`；本地副本 `D:\_eth_dl\ckpts\stageA_best.pt` | 字节数一致 |
| Stage B epoch10 best（58,948,377 B） | 远端 `…/gate6B_lr0001_musgd_b16_w8_20260927-235730/weights/best.pt`；本地副本 `D:\_eth_dl\ckpts\stageB_best.pt` | 字节数一致 |
| 负样本图像 | 远端 `/home/T7/dgut/robot_sim/eth_shuttle_detection`（+ `hard_negatives*`、`real_images`） | manifest 30,321 行 |

### 12.3 再生方式

删掉 `outputs/shuttle_capability/metrics/hard_negative_*` 与 `outputs/shuttle_capability/hard_negative_eval/` 后，按 §3 的 4 条命令重跑即可**逐字节复现**（推理 `imgsz=1024`、`conf-floor=0.01`、`device cuda:0`、`seed` 无关；保存的 top-K 图片由 `hard_negative_predictions.csv` 重绘，不再跑一次推理）。

---

## 13. 判定与下一步

**判定：`PASS_FP_NO_REGRESSION`**（在未见过的大样本普通负样本上 FP 单调下降；训练过的困难负样本 FP 归零；冻结真实背景的 +1 框可归因于标注缺失）。

**保留观察项：`WATCH_bg_negative_single_image`** —— `tbg_015.jpg` 单图 5 个框（@0.25 最大 0.684）。该集合只有 6 张，**不构成统计证据**；建议把 `bg_negative` 的 val 扩到 ≥50 张后再判。

**下一步（不改 Stage B）**

1. 等 Stage B 跑完 70 epoch，取**真实 best epoch**（不得假定 epoch 10），做三方对比：Stage A best / Stage B 最终 best / Stage B last ——（A）val 总体（B）val 尺寸分桶（C）困难负样本 FP。
2. 在 `tools/build_yolo26_v1_dataset.py` 中加入 `excluded.txt` 排除逻辑（ISSUE-027），**在 Stage B 结束之后再重建清单**。
3. 把 `frozen_real_backgrounds` 中含真实羽毛球的图（如 `bg_032`、`bg_009`）补标或从负样本池剔除（ISSUE-021 遗留）。
4. 若要真正评估“小目标误检”，应补一版**困难负样本 val 子集**（当前 96 张 100% 在训练集里，无法回答泛化问题）。
