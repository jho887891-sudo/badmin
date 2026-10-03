# 每日开发记录（DAILY_LOG）

> 每天结束前更新，新条目加在顶部。请勿只更新这里而不更新 PROJECT_STATUS / TODO。

---

# 2026-10-03（续 14）Task 3 收官（集成跑跑通全链路）+ **Task 4 50 epoch 全量已启动**

**约束**：仍未用任何中间指标做选点或调参；smoke/集成跑均标记 `diagnostic_only=true`、`eligible_for_final_report=false`（新增"epochs 与契约 50 不同即不可作为最终结果"的判定）；旧产物一律保留。

**Task 3 端到端验证**（`integration_e1c`，1 epoch 集成跑，03:42:08→03:52:48Z，10.7 min）
- manifest 证据：contract sha256 `fd9f95dc…`；权重 **20,422,725 B / `646f8bc3…`** 校验通过；数据列表 train **14,543** / val **2,920**（sha256 `d0947c2c…` / `1863b6a1…`，与本地提交逐位一致）
- 冻结配方落盘：imgsz 1024 / AdamW / lr0 1e-4 / nbs 32 / freeze 0 / batch 8（fallback [8,6,4] 实测第 1 次即成功）；ETH 官方增强与 loss（mosaic 1.0、mixup 0.7、scale 0.5、fliplr 0.5、hsv_h 0.015、box 7.5/cls 0.5/dfl 1.5）
- 24 GiB 单进程上限生效：`gpu_cap.fraction 0.5063`（设备 47.4 GiB）；选点 `column metrics/mAP50-95(B)`、`value 0.60435`、`recall 0.82232`、scope `internal_validation_only`；best.pt 20,337,797 B / sha256 `96263a5b…`
- **smoke+集成跑共抓出 4 个真实缺陷**（save_dir 见 ISSUE-033；`weight=` 非法参数、W&B 自动上传产物、选点列名不匹配见 ISSUE-034）→ 全部修复并各自补测试；启动器测试 58 项全绿

**Task 4 已启动**：`run_full_v1.sh full_e50`（50 epoch，batch 8，imgsz 1024，workers 8，`WANDB_MODE=disabled`，24 GiB 上限，内部 val 选点）
- 启动证据：`YOLO26s summary: 260 layers, 9,948,638 parameters` + `Transferred 696/708 items from pretrained weights`（确认从官方 `yolo26s.pt` 初始化）
- 速度实测（smoke，诊断）：冷启动 1.8 it/s → 页缓存热后 4.2–4.6 it/s，约 **12 min/epoch + 1.5 min 验证** → 50 epoch 预计 **10–12 h**；显存 6.0–6.7 GB（上限 24 GiB 从未触发）
- 完成判据：`full_e50_manifest.json` 的 `selection`（内部 val mAP50-95 的 best epoch）+ `best_checkpoint.sha256`；随后 Task 5 用这张 ckpt 做四个评估集（其中 controlled_capability / challenge_test 必须本地跑，ckpt 只落 `_scratch_*` 并在用后删除）
- **内部 val 的来源必须记住（易踩坑）**：`data/eth_only_v1_val_manifest.csv` 2,920 行**全部**来自 ETH 官方 `images/train` 子目录（`split=train`、`in_official_training=True`、`glc_2_easy 1506 / uetlibergstrasse_2_easy 1244 / glc_2_medium 107 / uetlibergstrasse_2_medium 63`）→ 对本模型是干净的（location-disjoint + 已剔除全部评估帧），但**它们就是 ETH 官方模型的训练帧**，因此该内部 val **只能用于本模型选点，绝不可用于与 ETH 官方模型的同集合公平比较**；ETH 公平对照必须继续用 `val|eth_unseen`（ISSUE-031 口径）

**提交**：`a7ce718`（wandb/资格判定）、`8d834fc`（非法参数 + 参数合法性测试）、`ef313d8`（选点列名归一）

# 2026-10-03（续 13）ETH-only YOLO26s 1024 V1：**Task 2 数据冻结（修正 3 个真实缺陷）+ Task 3 24 GiB 训练启动器**

**约束**：本轮未训练/未 fine-tune 任何模型（只跑了 3 epoch 诊断 smoke）、未改 ETH 官方 checkpoint/测试集/GT/evaluator、未按模型单独调阈值、未用任何中间指标做选择、旧产物一律保留（两次错误产物转存远端 `~/.dsh-bench/v1_buggy_backup/`、`v1_badneg_backup/`，未删）。

**Task 2 冻结产物**（资源真源远端 `/home/T7/dgut/robot_sim/eth_shuttle_detection`；本地 SSOT `data/` + `outputs/shuttle_capability/metrics/`，commit `52cd0dc`）
- train **14,543** 行 = 13,992 ETH 真实正样本（easy+medium，7 个 location）+ 550 `coco_train` 负样本（官方 fraction 0.1）+ 1 张空标签正样本位置图；val **2,920** 行，location-disjoint = `glc_2` 1,613 + `uetlibergstrasse_2` 1,307；`val_positive_fraction` **0.17265846736045412**
- 泄漏剔除 **2,765** 张（ml_3 1,004 / ml_6 1,123 / uetlibergstrasse_1 638）：`eval_overlap_images 0`、`sha256_overlap []`、`location_overlap []`、`nonempty_negative_labels 0`、`unreadable_images 0`、audit **PASS**
- 箱级尺寸分布（13,992 boxes，逐一对齐正样本数）：<4 161 / 4-6 1,282 / 6-8 4,522 / 8-12 5,599 / 12-16 1,691 / 16-24 596 / 24-32 97 / 32-64 41 / >64 3；min 2.0 / max 108.87
- 本地校验脚本 `tools/remote/check_local_artifacts.py`；sha16：train `16daa2a01de1608d`、val `0ed405719af1d627`、audit `94c1ff07c8929bdf`
**本轮发现并修复的 3 个真实缺陷**（均在"观测任何指标之前"，详见 ISSUE-032）
- ① `val_locations` 按"被选中的 location 组合"上报，多列 `uetlibergstrasse_1`（该 location 被强制排除后实际 0 帧）→ 改为由实际 val 帧推导；② 每图 `equiv_size_640` 误用 `(1,1)` 占位 → 全部落 `<4`、尺寸分布无意义 → 改为按标签框逐框计算并写入 `equiv_sizes`；③ coco 负样本 0.1 抽样只认 CLI 参数，重建命令漏传即退化为全量 5,500 张 → 缺省时改按 recipe `official_fraction_coco_train`，并记录 `negative_fraction_source`
- 最终产物一次性重跑，脚本内含硬断言 `SELF-CHECK OK`（train 14,543 / val 2,920 / coco 550 / val_locs `['glc_2','uetlibergstrasse_2']`）

**Task 3 启动器**（`tools/train_eth_only_v1.py`，commit `f2bed61`；46 项测试全绿且不需要 GPU/torch）
- 冻结 imgsz/nbs/optimizer/lr0（改动抛 `RecipeViolationError`，contract 优先于 recipe json）；batch 仅在 CUDA OOM 时 8→6→4，其它异常直接抛出；训练前校验权重 sha256+字节数（**20,422,725 B / `646f8bc3…`**，dry-run 实测一致）与 data yaml（nc 1 / class0 `shuttlecock` / 两个列表非空）
- **24 GiB 单进程显存上限**（`torch.cuda.set_per_process_memory_fraction`）；checkpoint 选择只读内部 val `metrics/mAP50-95(B)`（recall 并列记录）；smoke/dry-run 产物强制 `diagnostic_only=true`、`eligible_for_final_report=false`
- 第 4 个缺陷由 3 epoch smoke 抓到：ultralytics 把 run 写到 `runs/detect/<project>/<name>`，按 `project/name` 读 `results.csv` 会**静默丢掉选点** → 改为从 trainer 对象取 `save_dir`（ISSUE-033）
**smoke 实测（诊断用，不作为结论）**：batch 8 / imgsz 1024 显存 **6.2 GB**（24 GiB 上限远未触发）；数据在 NTFS/FUSE 上约 **1–2 it/s**（8 workers）= 磁盘受限 → 50 epoch 全量预计 **10–19 h**（与计划估计一致）
**下一步**：smoke 走完 → `run_full_v1.sh` 起 50 epoch 全量（内部 val 选点）→ Task 5 evaluator 一致性 + 四个评估集

---

# 2026-10-01（续 12）ETH Zurich 官方 YOLOv8s 第三方基线审计：**其自身域又准又快，我们的合成能力域它完全失效；val 有 62.7% 是它的训练帧**

**约束**：只做评估。未训练、未 fine-tune、未改 checkpoint/测试集/GT/evaluator、未按模型单独调阈值、未用论文数字替代实跑；旧结果全部保留。

**权证**：`runs/final-model/best.pt`，**134,312,133 B，sha256 f1aea7dec784a24d53d475f89510ef45fc13255df6298c6a0ea2fbd4419c7d6d**（与仓库 `eth_official_repo_inventory.csv` 及 `IMAGE_RESOURCES.md` 记录逐位一致；远端唯一真源 `/home/T7/dgut/robot_sim/eth_official_code/shuttle_detection/`，本地仅 `_scratch_eth_official/` 临时副本）。**格式不兼容已记录**：该 ckpt 是自建 dict（keys: epoch/model_name/model_state_dict/optimizer_state_dict/train_loss/val_f1/val_loss），非 ultralytics 格式；最小兼容处理 = 用本机 ultralytics 的 yolov8s.yaml 重建同架构 + `load_state_dict`（missing=unexpected=0），**未改任何权重/结构**。state_dict 有 3 个 cv3 分类分支，val 全部 5,645 个检测均为 class 0（实际单类，与其 config.json 的 `dist_threshold=25.0, confidence=0.5, imgsz=1024, model_name=yolov8s` 一致）。

**公平口径**：同一 evaluator（同一 match_greedy/ap_for_gt_set）、同一图像/GT（同一 manifest 与冻结目录）、imgsz=1024、工作点 conf=0.25、AP 曲线下限 0.001、NMS iou=0.7/max_det=300/rect=False、同一等价尺寸桶（equiv_size_640）、同一坐标与硬件、fp32、同一延迟协议（batch=1，warmup 50，measured 200）。**未为任何模型调阈值**。

**结果（conf 0.25）**：
- `controlled_capability`（2,070 GT，全合成）：ETH **TP 5 / FP 103 / R 0.0024 / mAP50-95 0.0123 / Recall_<8 0.0000 / FP-图 0.0498**；a_best 0.1734/0.2302/0.1125；b_best 0.2527/0.3014/0.0937。
- `challenge_test`（194 GT）：ETH **TP 0 / R 0.0000 / mAP50-95 0.0072**；a_best 0.0464/0.0971；b_best 0.0722/0.1170；b_last 0.1649/0.1327。
- `val`（3,260 GT）：ETH **TP 3,028 / FP 68 / P 0.9780 / R 0.9288 / AP50 0.9395 / AP75 0.9021 / AP90 0.4947 / AP95 0.1107 / mAP50-95 0.7766 / IoU 中位 0.9037 / 中心 0.316 px640**；a_best R 0.8595 / mAP50-95 0.5907；b_best 0.7797/0.5926。
- `val` 去掉 ETH 训练帧后（495 GT）：ETH R **0.5434** / mAP50-95 0.4260；a_best R **0.5575** / 0.3801；b_best 0.4868/0.3707；**<8 px：b_best 0.2571 > a_best 0.2000 > ETH 0.1714**（4–6 px 桶 ETH 仅 0.0270）。

**泄漏审计（判据：ETH 官方每档 yaml 的 train: images/train + config.json 的 12 个训练 location 与 diff_levels=[easy,medium]）**：val 4,413 图中 **A=2,765（62.7%）就位于 ETH 训练目录**（R 0.9978）、B=263（R 0.9468）、C=1,235 coco_val_easy（R 0.1429）、C=150 我们的数据（R 0.0333）；controlled/challenge **100% 在 ETH 数据集之外**（背景图探针 30/30 与 33/33 亦确认非 ETH 帧）→ **ETH 在 val 上的分数不能作为泛化能力**。

**height_ratio 归属（修正上一轮）**：ETH val 0.9892、未见子集 1.0089 → **没有 +9% 高度偏置**（`POSSIBLE_MODEL_OR_TRAINING_SPECIFIC_EFFECT`）；**同一批 495 个未见 GT** 上 ETH 1.0089 vs a_best 1.0692 / b_best 1.0707 ⇒ 同图同标签不同模型，该正偏主要来自我们的模型/训练侧（上一轮把它主要归因于 train/val 标注规范的表述据此修正）。

**成本/速度（同卡 batch=1 fp32）**：ETH 11,166,560 参数 / 73.76 GFLOPs / **19.69 ms / 50.8 FPS / 229 MB**；a_best 9,663,464 / 69.09 / 60.14 ms / 16.6 FPS / 417 MB；b_best 57.58 ms / 17.4 FPS / 495 MB。→ **P2 的代价不在 FLOPs（反而低 6.3%）而在延迟 2.9x、显存 2.2x**。

**ETH-style 中心距指标（单独口径）**：val ETH F1 0.9601 vs IoU-based 0.9528；我们的模型两者差 +0.11~0.12（0.8881 vs 0.7644）。**ETH-style F1 != IoU F1**（top-1/图、单 GT/图、至多 1 FP/图）。

**产物**：`tools/eval_eth_official_baseline.py`、`tests/test_eth_eval_parity.py`（**16/16 PASS**）、`reports/ETH_OFFICIAL_BASELINE_AUDIT.md`（464 行 / 44,600 B / sha16 952fd6f851ff8b4f）、metrics/`eth_official_{overall,ap_by_iou,size_buckets,localization,false_positive,latency,center_metric}.csv`、`eth_vs_ours{,_ap_by_iou,_size_buckets,_localization}.csv`、`eth_data_leakage_audit.csv`、`eth_data_leakage_inventory.csv`、`eth_checkpoint_manifest.json`、`examples/eth_official/**`（285 张）+ `examples/model_disagreement/**`（40 张）。

**未做**：未训练/未微调/未改权重与标签、未覆盖任何旧 metrics 或报告、未启动 Stage C。

---
# 2026-10-01（续 11）ROOT CAUSE AUDIT：+9.2% 高度偏差 = **train/val 标注规范不一致**（不是评估 bug）

**用户约束（严格遵守）**：禁止启动任何训练 / 禁止改 checkpoint / 标签 / 训练配置 / 模型结构 / loss / augmentation；只允许读数据、读缓存预测、读评估代码、新增诊断工具、做一致性验证、输出根因报告。

**新增**：tools/audit_bbox_height_bias.py（只读诊断；坐标数学与 round-trip、GT/预测按 source/size/aspect 分解、像素空间误差、counterfactual IoU/AP、外观代理、train-vs-val 标注风格、ultralytics parity）+ tests/test_bbox_coordinate_roundtrip.py（**10 项全绿**，含 Test 1-7 与 parity）。预测侧直接复用上一轮的缓存（_scratch_localization_audit/loc_<ckpt>_val.json），**本轮零训练、零权重改动**。

**坐标链路审计（逐行取自实际代码）**：
- GT：tools/eval_yolo26_v1.py:401-402 归一化→像素，**不经 letterbox**，回环最坏误差 **1.14e-13**。
- letterbox：data/augment.py LetterBox.get_params 1745-1776 / apply_image 1792-1803 / _update_labels 1883-1886；**r = min(1024/H, 1024/W) 单一比例**，padding 只平移。
- 回代：models/yolo/detect/predict.py:121 → utils/ops.py:146-165；**gain_y = gain_x = gain（各向同性）**。
- 与安装版 ultralytics 数值交叉验证：letterbox 参数差 **0.0**、我的 scale-back vs utils.ops.scale_boxes 逐坐标差 **0.0**、padded shape 差 **0.0**。
- 实测像素路径取整残差最坏 h_ratio **+6.25e-04（偏小）**，方向与 +9.2% 相反。

**偏差是否真实**：真实存在（三个 ckpt 一致：h_ratio mean 1.0919 / 1.0918 / 1.0936，w_ratio 0.9978 / 1.0003 / 0.9954）。原生 ultralytics（list source, rect=False）vs 缓存（txt source）：conf≥0.25 一对一匹配 13 对，逐坐标最大差 **0.303 px**（亚像素；差异来自批次取整 padding），仅 1 个阈值边缘不匹配。

**根因（CONFIRMED）**：同一宽度桶内，val 的 GT 框比 train 系统性更扁更矮（1024 输入系中位高）：[14,20) **13.27 → 12.27**、[20,28) **17.60 → 15.47**、[28,40) **23.34 → 18.67** px；aspect 1.286→1.400 / 1.323→1.483 / 1.360→1.760。模型学 train 规范 → val 上高度平均多预测 **+1.3 px（1024 系）**，对 8–16 px 目标即 **+9%～+11%**。**对照组**：eth_iphone h_ratio 1.020、synthetic 1.015（其 train/val 规范一致），eth_main 1.095（96% 的 val GT）。
**叠加机制（CONFIRMED）**：aspect 回归到均值 —— GT aspect 0.677→预测 0.801、1.960→预测 1.521；扁框（aspect≥1.4，占 TP 50%）dh +1.5～+2.2 px，高框（<0.8）dh −1.0 px。
**不是「固定 1.092 倍率」**：按桶 h_ratio 1.119 / 1.110 / 1.088 / 1.040 / **0.973** / 1.064 / **0.988**（4-6…>64）；dh 中位 +0.70 / +1.38 / +1.33 / +1.13 / **−0.83** / +1.59 / **−4.66** px。1024 系下 dh 众数 = **+1.0 px**（28.5% TP）、中位 +1.28；dw 众数 0.0、中位 +0.008（宽度无系统偏差）。
**排除项**：motion blur（清晰度代理 low 1.1068 vs high 1.1106，Pearson −0.0008）、「GT 比外观紧」（blob/GT h 0.773、w 0.697）、matching/AP 口径（orig AP50…AP95 与 localization_ap_by_iou.csv 逐位一致）。

**counterfactual（仅诊断）**：a_best orig AP75/85/90/95 = 0.6855/0.2848/0.0903/0.0086 → 只纠正 height（整轴）**0.8627/0.7167/0.5150/0.2030**、只纠正 width（整轴）0.8838/0.6939/0.4452/0.1420、只纠正 center 0.7624/0.4425/0.2272/0.0687、GT 框上限 0.9184。**归因（AP90 增量）**：height 尺寸 +0.160 > center +0.137 > width 尺寸 +0.070；**AP95 反转为 center +0.060 > height +0.034 > width +0.018** → 上一轮「center 不是主因」需修正为「center 与 height 并列」。

**下一步建议（未执行）**：① 优先查清并统一 train/val 标注规范（B，证据最强；**本轮禁止改标签，只给方案**）；② 输入分辨率（C，POSSIBLE，次要）；③ A/D/E（修 evaluator / Stage C / 改 box loss）**无证据支持**；**在根因处置方案确定前不启动 Stage C**。

**产物**：reports/BBOX_HEIGHT_BIAS_ROOT_CAUSE.md（501 行 / 38,797 B / sha256 前 16 位 7854f06dac7ecedd）；metrics/bbox_{coordinate_roundtrip.json, height_bias_by_source.csv, height_bias_by_size.csv, pixel_error_distribution.csv, ratio_vs_iou.csv, counterfactual_iou.csv, counterfactual_tp_detail.csv, gt_convention_train_vs_val.csv, gt_convention_train_vs_val_by_source_width.csv, aspect_vs_h_ratio.csv, ultralytics_parity.csv}；示例图 150 张（bbox_bias_examples/{high_height_ratio,typical_109,ratio_near_1}/）。

**未做**：未训练、未改 checkpoint/标签/配置/模型结构/loss/augmentation，未覆盖旧报告与旧 metrics，未启动 Stage C。

---

# 2026-10-01（续 10）YOLO26 定位审计：B 的 mAP50-95 提升**主要来自 Recall**，不是真正的 bbox 定位改善

**用户约束（严格遵守）**：只读现有 checkpoint 与评估数据；**禁止启动 Stage C**、禁止改模型/训练配置/数据/标签/loss/optimizer/augmentation；只允许新增/修改评估工具与报告。

**新增工具**：`tools/audit_localization_yolo26_v1.py`（诊断专用）：AP50..AP95（标准 PR 曲线）、TP 的 IoU 分布（含 6 桶直方图）、near-miss A/B/C/D（**0.30–0.50 永不计 TP**）、中心/宽高误差（px、px640、归一化、对数比）、9 个 `equiv_size_640` 桶逐桶 AP、A-vs-B 分桶增量、6 个 CSV + JSON + 示例图 + 自动报告。**22 项单元测试全绿**（`tests/test_audit_localization_yolo26_v1.py`，含边界：0.50 属 A、0.4999 属 B、0.49 不记 TP、AP 单调、完美预测 AP=1、1024↔640 缩放）。

**运行**：`python tools/audit_localization_yolo26_v1.py --device 0`（a_best / b_best / b_last × val + controlled_capability/images + challenge_test/images = 6,710 图 × 3 ≈ 20,130 次推理，本地 RTX 4060，约 75 分钟；预测缓存 `_scratch_localization_audit/`，可删可重建）。**未使用任何无标注集合做 bbox 定位**（工具会直接拒绝未标注集合）。

**结果（工作点 conf 0.25；AP 曲线 conf 0.001；TP = IoU≥0.50 一对一）**：
- `val`（3,260 GT）：a_best `TP2802 / R0.8595 / AP50 0.9051 / AP75 0.6855 / AP90 0.0904 / AP95 0.0087 / mAP50-95 0.5907 / IoU中位 0.8170 / 中心 0.532 px640`；**b_best `2542 / 0.7798 / 0.8835 / 0.7031 / 0.1148 / 0.0083 / 0.5926 / 0.8284 / 0.439`**；b_last `1966 / 0.6031 / 0.7564 / 0.5898 / 0.0946 / 0.0061 / 0.5044 / 0.8274 / 0.439`。
  → val 上 B **确有定位改善**（AP75 +0.0176、AP90 +0.0244、IoU 中位 +0.0114、中心误差 −0.0934 px640），但**召回 −0.0798、AP50 −0.0216**，净 mAP50-95 **+0.0019（噪声级）**。
- `controlled_capability/images`（2,070 GT，即产生 `FROZEN_TEST_FAVORS_B_BEST_E10_OVER_A_BEST` 的集合）：a_best `359 / R0.1734 / P0.8842 / AP50 0.3659 / AP75 0.2413 / AP90 0.0647 / mAP50-95 0.2303 / IoU中位 0.8520 / 中心 0.698 px640`；**b_best `523 / 0.2527 / 0.4847 / 0.4366 / 0.3472 / 0.0894 / 0.3015 / 0.8455 / 1.111`**；b_last `507 / 0.2449 / 0.7061 / 0.3212 / 0.2672 / 0.1201 / 0.2342 / 0.8724 / 1.503`。
  → **mAP50-95 +0.0712 几乎全部由召回贡献（R +0.0792）**，而同一批 TP 的框质量**变差**：IoU 均值 −0.0163 / 中位 −0.0065、IoU∈[0.50,0.60) 的“勉强 TP” **7→48**、中心误差中位 **0.698→1.111 px640（P90 6.25→26.99）**、宽度相对误差中位 **+0.0360**；精度代价 P **0.8842→0.4847**（检测数 406→1079、FP 47→556）。
- `challenge_test/images`（194 GT）：a_best `9 TP / AP50 0.1840`；b_best `14 / 0.2339`；b_last `32 / 0.2234` 但**中心误差中位 40.05 px640**（>64 px 桶 24/24 全召回、该桶 IoU 中位仅 0.667）→ **样本量不足，不下结论**。
- **逐桶（关键）**：B 在固定留出集上的召回增益集中在**大目标**（32-64 px ΔR **+0.258**、>64 px ΔR **+0.433**），而**最关键的 <4 / 4-6 / 6-8 px 三个超小目标桶反而下降**（ΔR −0.0188 / −0.0094 / −0.0370）；val 上 B 的召回下降集中在 12-16（−0.168）与 16-24（−0.224），同时这些桶的 IoU 中位数上升。

**判定（用户要求的显式结论）**：**Stage B 的 mAP50-95 提升主要来自 Recall，而不是真正的 bbox localization 改善。** 固定留出集 = `CASE A（召回驱动）`；val = `CASE B（真实定位改善）但以召回为代价、净值噪声级`。→ **不替换基线（保留 Stage A）、不因 +0.0712 升级 B、不启动 Stage C。**

**主导误差来源（按数据排序）**：① **漏检**（class D：controlled 82.5%，b_best 74.5%；val 13.9% → b_best 21.9%）；② **框偏大而非偏心**（val 上 height_ratio 均值 **1.0919**，三个 checkpoint 一致，宽向无偏 0.9978；即使中心完美的平均 IoU 上限 ≈ **1/(0.9978×1.0919)=0.918**，与 AP85 0.285 / AP90 0.090 / AP95 0.009 的断崖吻合）；③ **B 新召回的大目标框更松**（near-miss 0.30–0.50：A 1 个 → **B 48 个**）；④ **超小目标接近几何极限**（<4 px 桶召回 0.042–0.069，B 更差）；⑤ **标签噪声无证据也无法量化**（class C 仅 0–4 个）——不列为结论。

**口径自证（逐位一致）**：a_best/val ↔ `metrics/size_bucket_metrics_val.csv`（2802 / 0.859509 / 0.905095 / 0.590744）；a_best/controlled ↔ `reports/FROZEN_TEST_EVAL.md`（359 / 47 / 0.8842 / 0.1734 / 0.3659 / 0.2303）；b_best 523 / 0.4366、b_last 507 / 0.3212、challenge 9/14/32 与 `metrics/frozen_test_matrix.csv` 全部一致。

**术语更正**：新增 `docs/LOCALIZATION_AUDIT.md`（工具口径 + §7 术语纪律）与 `management/ISSUES.md` **ISSUE-029**，并在 `FROZEN_TEST_EVAL.md` / `STAGE_B_FINAL_REPORT.md` / `docs/EVAL_FROZEN_TEST.md` 插入**不改写历史**的更正说明：这些集合统一称为 **fixed evaluation set / development holdout**，不得称为“从未参与模型选择的最终测试集”。**若 Stage C 依据本报告调参/选阈值，最终判断必须另用一个全新的、未被看过的 holdout。**

**产物**：`metrics/localization_{ap_by_iou,iou_distribution,error_summary,size_buckets,checkpoint_compare,near_miss}.csv`（sha256 前 16 位 `c781a1fa68b6619c` / `3d3852cf41fd9766` / `aa0d0412b409c2b9` / `caae073c58cb861b` / `afa37f5fc5de0d2b` / `21129e8580e83fc5`）+ `localization_audit.json`（23,049,205 B / `ad946fa845d2f7cc`）+ `localization_examples/`（348 张 / 170.8 MB / index.csv `8e74560d36a77f7c`，全部 b_best）+ `reports/LOCALIZATION_AUDIT.md`（§0 人工结论 + §1–§10 自动表格）。

**未做**：未修改任何模型/训练配置/数据/标签/loss/optimizer/augmentation；未删除任何 checkpoint；未启动 Stage C。

---
# 2026-09-30（续 9）冻结测试集评估完成：**判定 B best@e10 优于 A best**（与 val 结论相反）

**运行**：`tools/eval_yolo26_v1.py --split frozen_test --ckpt-name a_best --ckpt-name b_best --ckpt-name b_last`（本地 RTX 4060，3,037 张 × 3 检查点，约 22 分钟）。

**有标签集合（工作点 conf 0.25）**：
- `controlled_capability`（2,070 GT）：a_best `TP359/FP47/P0.8842/R0.1734/mAP50 0.3659/mAP50-95 0.2303/R_6-16 0.1137`；**b_best `TP523/FP556/P0.4847/R0.2527/mAP50 0.4366/mAP50-95 0.3015/R_6-16 0.1386`**；b_last `507/211/0.7061/0.2449/0.3212/0.2342/0.0740`。Δ(b−a)：R **+0.0792**、mAP50 **+0.0708**、mAP50-95 **+0.0712**、R_6-16 **+0.0249**、P −0.3995。
- `challenge_test`（194 GT）：a_best `9/4/0.6923/0.0464/0.1840/0.0972/0.0692`；**b_best `14/10/0.5833/0.0722/0.2339/0.1170/0.1077`**；b_last `32/65/0.3299/0.1649/0.2234/0.1328/0.0615`。Δ(b−a)：R **+0.0258**、mAP50 **+0.0499**、mAP50-95 **+0.0198**、R_6-16 **+0.0385**、P −0.1090。
- **逐桶**：B 的收益集中在 **8–16 px 主战场**（`controlled_capability` 12-16 桶 R +0.038、AP50 +0.141）与 ≥32 px；**<6 px 超小目标反而略退**；`<4` 桶在 `controlled_capability`(160 GT) 与 `challenge_test`(40 GT) 三代模型**全为 0**（最大能力空洞）。

**无标签集合**（FP/图 @0.25）：`synthetic_3d` A 0.714 → **B 0.917** → B70 0.679；`synthetic_on_real_bg` 0.356 → 0.419 → 0.200；`real_images/backgrounds` 0.067 → 0.100 → 0.000；`real_images/raw` 0.068 → **0.034** → 0.169；`real_video/frames` 0.007 → 0.027 → 0.007；`real_match_frames` 0.047 → 0.055 → 0.051；`real_train/raw` 0.045 → **0.000** → 0.091。→ **额外误检集中在合成域，真实视频/比赛帧只 +0.008~0.02**。

**判定 token：`FROZEN_TEST_FAVORS_B_BEST_E10_OVER_A_BEST`（附 `PRECISION_COST_CONCENTRATED_ON_SYNTHETIC`）**
- 与 val 的 `STAGE_B_HEALTHY_BUT_NO_VAL_GAIN` **不矛盾**：val 在训练分布内（ETH 真实帧）→ B 过拟合掉点；冻结集在分布外/困难场景 → B 召回与 AP 改善，代价是精确率。
- **下一步**：Stage C 从 `stageB_best_e10_stripped.pt` 续训（不是 A best）；复评同时看 val + frozen；优先攻 `<4 px`；先做一次 `--conf` 扫描定工作点；`b_last` 只作对照。

**产物**：`outputs/shuttle_capability/reports/FROZEN_TEST_EVAL.md`（209 行 / 13,593 B，12 节）；`metrics/frozen_test_{matrix,labeled_buckets,negatives,checkpoints}.csv`（sha256 见报告 §10）+ `size_bucket_metrics_frozen_test.json`（53,798 B）。

---
# 2026-09-30（续 8）热点/校园网两种模式对照：直连 7 ms vs 中继 404 ms

- **用户问“能用热点直连远端吗”** → 现场实测发现本机当时**其实在校园 WiFi**（SSID `莞工全光无线`，10.62.142.122/14，gw 10.60.0.1，DNS 172.30.253.x；`tracert 172.31.68.251` 走校园内网 10.60.0.1→192.168.48.2→192.168.62.2）。
- **模式 A（校园网/上游为校园网的热点）**：`tailscale status` = `active; **direct** 172.31.68.251:41641`，`tailscale ping` **7 ms**，Harness 页面 **200 / 36,833 B / 0.64 s**，`MappingVariesByDestIP: false`。
- **模式 B（纯手机流量热点）**：`relay "lax"`，ping 375–466 ms（avg 404），ICMP avg 395 ms，页面 1.52–2.02 s，`MappingVariesByDestIP: true`（对称 NAT）+ 两端均无 IPv6 → **直连无望**。
- **结论**：热点能否直连取决于**上游**：上游仍是校园网（手机中继校园 WiFi / 校园 VPN）→ 直连 7 ms；纯运营商流量 → 只能 DERP 中继 ~400 ms。自检：`tailscale status | Select-String jxxy`。

**产物**：`docs/REMOTE_DSH_DEPLOY.md` §16.6。

---
# 2026-09-30（续 7）补齐 `--split frozen_test`：冻结测试集评估接口 + 单元测试 + 冒烟

- **实现**（`tools/eval_yolo26_v1.py`，**未改 val 分支逻辑**，只加函数与分支）：`discover_frozen_groups()`（自动发现 8 个冻结集，支持一个集合多图像目录 → 多组）、`load_gt_boxes()`（YOLO 标签解析，抽出供两条路径共用）、`fp_stats()`（无标签集：FP/图、图 FP 率、置信度分布）、`percentile()`（与 `eval_hard_negative.py` 同口径）、`resolve_checkpoints()` + `load_checkpoint_registry()`（统一检查点接口）、`file_sha256()`、`predict_group()`、`run_frozen()`。
- **CLI 新增**：`--ckpt-registry`（默认 `configs/shuttle_detection/checkpoints_v1.yaml`）、`--ckpt-name`（可重复）、`--frozen-sets`（子集）、`--neg-thresholds`；val 路径也改用统一的 `resolve_checkpoints()`。
- **注册表** `configs/shuttle_detection/checkpoints_v1.yaml`：`a_best`(epoch 6, baseline) / `b_best`(epoch 10, candidate) / `b_last`(epoch 70, overfit_control) / **`c_best` 占位**（path 空 → 打印 SKIP 不报错）。
- **产物**：`size_bucket_metrics_frozen_test.json`（全量）+ `size_bucket_metrics_frozen_<set>.json`（val 同 schema，可直接喂 `tools/compare_size_buckets.py` 做 N 路对比）+ `frozen_test_matrix.csv` + `frozen_test_labeled_buckets.csv` + `frozen_test_negatives.csv` + `frozen_test_checkpoints.csv`（含 sha256）。
- **测试**：新增 `tests/test_eval_yolo26_v1_frozen.py` **26 项全绿**（集合发现/标签目录判定/GT 解析与分桶/FP 统计含阈值边界与空集/百分位同口径/注册表解析含占位与报错/仓库自带注册表可解析）；旧 `tests/test_eval_yolo26_v1.py` **13 项全绿**（无回归）。
- **冒烟**：① val + 注册表名，20 张 → 正常出产物；② frozen_test 全部 8 个集合（每组 2 张，1 检查点）→ 2 个有标签组出 Recall/AP/分桶、7 个无标签组出 FP 统计，4 个 CSV + 3 个 JSON 全部生成（`_scratch_eval_smoke/`，**未覆盖**正式 val 产物）。
- **全量测试说明**：`tools/run_all_tests.py` 报 53 套件 / 18 失败 —— 全部为**环境性**（`ModuleNotFoundError: pxr`，USD 绑定只在 Isaac Sim 环境；另 17 个套件 `rc=1, 0 tests` 为导入期失败），**与本改动无关**；涉及本工具的套件全绿。
- **文档**：`docs/EVAL_FROZEN_TEST.md`（命令、集合分层、产物、验证方式、口径注意）。

**下一步（等用户点头）**：跑**完整**冻结集评估（3,037 张 × a_best/b_best/b_last 三个检查点，本地 4060 预计 30–50 分钟）→ 出 `FROZEN_TEST_EVAL.md` 终局判定。

---
# 2026-09-30（续 6）手机热点下延迟实测（DERP 中继 ≈400 ms）

- **本机切到手机热点**（WLAN 10.37.233.227，公网出口 183.46.216.194，`MappingVariesByDestIP: true`）→ 校园捷径失效、两端对称 NAT 打洞失败 → **走 DERP 中继**。
- **实测**：`tailscale ping -c 10 jxxy` → min 375 / **avg 404** / max 466 ms（`via DERP(lax)`，`direct connection not established`）；ICMP ×10 → avg **395 ms**、0% 丢包；反向 → `via DERP(sfo)` 390–473 ms；**HTTP 首页**（36,833 B）connect ≈0.40 s、**TTFB avg 0.87 s**、**总时 avg 1.62 s**。对照校内直连 5–8 ms。
- **自建 DERP 收益预估**（实测到 VPS `223.109.239.11`）：本机→VPS avg **63 ms**、远端→VPS avg **40.8 ms**（mdev 0.2）→ 端到端中继约 **104 ms**，比现在快约 **4×**。但该 VPS 的 `20736` 端口两端都连不上（ICMP 通、TCP 被过滤）→ 需在其上开端口 + 配证书，**属端口暴露，待用户批准**。
- **运维提示**：离开校园网后 SSH 也要用 tailnet 名字/IP；`172.31.68.251` 不可达（实测 22 端口 timeout）。首次连 `jxxy.taildd42cc.ts.net` 用 `StrictHostKeyChecking=accept-new` 接受了主机密钥。

**产物**：`docs/REMOTE_DSH_DEPLOY.md` §16.5。

---
# 2026-09-30（续 5）交接快照：远端 Harness 全链路可用

- **端到端复核**：unit `active+enabled`、`Linger=yes`、PID 931050；serve `http://jxxy.taildd42cc.ts.net:3080 (tailnet only)`；`~/.dsh-web/url.sh` 正常；本机 `curl --noproxy "*" "http://jxxy.taildd42cc.ts.net:3080/?token=…"` → **303 → 200**（34,846 B、`<title>DeepSeek Harness</title>`、`__DSH_BOOT__`）。
- **注意**：早先那条本地 SSH 隧道（`127.0.0.1:3081`）已随其 ssh 进程退出而失效（`000`）；需要用就重跑 `ssh -f -N -L 3081:127.0.0.1:3080 dgut@172.31.68.251`（备用路径，不受本机代理影响）。
- **交接快照**写入 `docs/REMOTE_DSH_DEPLOY.md` §17（链路图、关键事实表、常用命令、6 项待决策）。

---
# 2026-09-30（续 4）“离开校园网就连不上”的机制诊断

- **现状**：`tailscale ping jxxy` → `via 172.31.68.251:41641 in 5ms`，即走**校园内网**；本机 WLAN `10.62.142.122/14`（gw `10.60.0.1`、DNS `172.30.253.x`）经校园网路由到 `172.31.0.0/16`（`Find-NetRoute 172.31.68.251` 确认走 WLAN）。两台机器都在校园网内。
- **离开校园网后的三条候选路径**：① 内网直连 → 失效；② 公网打洞 → 远端 `netcheck` 实测 **`MappingVariesByDestIP: true`（对称 NAT）**，通常失败；③ **DERP 中继 → 两端都通**（远端对 `derp1/derp9/derp17c` 均 200，并列出 25 个区域延迟；对端 DERP 区域 `lax`）。
- **结论**：不是“不可能连”，而是“只能走中继、且校园侧曾对 DERP 做 TLS 中间人（CN=ruijie）”。实测连不上时优先排查：切换空窗 30–60 s、所在网络封 UDP/443、HTTP 代理未绕过（§15 已修）、**token 变了**（用 `~/.dsh-web/url.sh` 取新 URL）。
- **下一步**：用户在校外跑 `tailscale ping jxxy` + `tailscale netcheck` 发我 → 判 direct/DERP/失败；若 DERP 不可用则用其公网机（`223.109.239.11:20736`）自建 DERP 或反向隧道（后者属端口暴露，须先获批）。

**产物**：`docs/REMOTE_DSH_DEPLOY.md` §16。

---
# 2026-09-30（续 3）浏览器 502 定位并修复：本机代理绕过列表缺 tailnet

- **现象**：浏览器打开 tailnet URL → **HTTP ERROR 502**；服务端正常（unit active、进程在、双监听、serve 正常）。
- **定位**：本机 WinINET 代理开启（`ProxyEnable=1`、`ProxyServer=127.0.0.1:7890`），`ProxyOverride` 不含 `100.*`/`*.ts.net` → 浏览器把 tailnet 名字交给代理，代理在 tailnet 外解析失败 → 502。三方对照：`curl -x http://127.0.0.1:7890 <tailnet URL>` → **502**（复现）；`curl --noproxy "*" <tailnet URL>` → **401**（到达服务）；`curl --noproxy "*" http://127.0.0.1:3081/`（旧 SSH 隧道）→ **200**。
- **修复**：`HKCU\…\Internet Settings\ProxyOverride` 追加 `;100.*;*.ts.net`（**仅用户项**，未动 HKLM/系统），并 `InternetSetOption(39/37)` 通知 WinINET（返回 True）。修复后 `curl --noproxy "*" <tailnet URL>` → **303 → 200**（34,846 B + `<title>` + `__DSH_BOOT__`）。
- **回滚**：把 `ProxyOverride` 还原为 `localhost;127.*;10.*;172.16.*…172.31.*;192.168.*;<local>`（原文见 `docs/REMOTE_DSH_DEPLOY.md` §15.3）。

**产物**：`docs/REMOTE_DSH_DEPLOY.md` §15；脚本 `D:\_eth_dl\add_proxy_bypass.ps1`、`D:\_eth_dl\notify_wininet.ps1`（本机临时，不在仓库内）。

---
# 2026-09-30（续 2）远端 Harness 开机自启：systemd --user + linger

- **安装 `~/.config/systemd/user/dsh-web.service`**（`Type=simple`，`WorkingDirectory=/home/T7/ojh/badmin_project`，`ExecStart=~/.local/bin/dsh web --no-open --host 127.0.0.1 --port 3080 --trusted-host 100.88.178.19:3080 --trusted-host jxxy.taildd42cc.ts.net:3080`，`Restart=on-failure`，日志进 journal）；仓库副本 `experiments/yolo26_v1/dsh-web.service`。
- **`sudo loginctl enable-linger dgut`** → `Linger=yes`（无人登录也随开机启动）。
- **实测**：unit `active + enabled`，MainPID 931050，`ExecMainStartTimestamp=2026-09-29 18:41:29 UTC`；`systemctl --user restart dsh-web` 成功（证明 systemd 真的在托管，而非仅一次性拉起）；监听 `127.0.0.1:3080` + `100.88.178.19:3080`(serve)；**校园网 `172.31.68.251:3080` 未绑定**；本机经 Tailscale 带正确 token → **200**（34,846 B、`<title>DeepSeek Harness</title>`、`__DSH_BOOT__`），旧 token → **401**。
- **当前 token**：`aUa6tFRicmPG6_l_UNLcOZX4OxILsAFF4xDbCiZ8G9M`（每次重启会变）。新增 `~/.dsh-web/url.sh` 一条命令打印 local + tailnet 两个 URL。
- 回滚：`systemctl --user disable --now dsh-web && sudo loginctl disable-linger dgut`。

**产物**：`docs/REMOTE_DSH_DEPLOY.md` §14；`experiments/yolo26_v1/{dsh-web.service,remote_install_unit.sh,remote_dsh_url.sh}`。

---
# 2026-09-30（续）远端 Harness 接入 tailnet：`tailscale serve`（本机浏览器可直接打开）

- **试错**：`dsh web --host 100.88.178.19` 被 DSH 拒绝（`$.host expected "127.0.0.1" | "0.0.0.0"`）；`0.0.0.0` 会把 3080 摊到校园网 → 弃用。
- **落地方案**：Harness 仍只绑 `127.0.0.1:3080`，由 `tailscale serve --bg --http=3080 http://127.0.0.1:3080` **只对 tailnet** 反代；`start.sh` 加 `--trusted-host 100.88.178.19:3080` 与 `--trusted-host jxxy.taildd42cc.ts.net:3080`。此前安装时 `tailscale set --operator=dgut` 未生效（`serve config denied`），已用 sudo 补设。
- **本机实测**：`http://jxxy.taildd42cc.ts.net:3080/?token=…` → **200**（34,846 B、`<title>DeepSeek Harness</title>`、`__DSH_BOOT__`）；无 token → **401**；直接 IP → **404**（serve 按 Host 分发，必须用节点名）；校园网 IP `172.31.68.251:3080` → **连接被拒**（未暴露）。
- **运行时状态（本次核查）**：远端 `dsh web` **PID 927381 正在运行**，cwd `/home/T7/ojh/badmin_project`，`127.0.0.1:3080` + serve 的 `100.88.178.19:3080` 均在监听；当前 token `IYohgKsMRVmoX8l2_bhKujUoXF3rDUy5skbrkJMjXPg`。
- **持久性**：serve 配置随 tailscaled 状态自动恢复；`dsh web` 是用户进程，**开机不自启**，重启后需跑 `~/.dsh-web/start.sh` 且 token 会变（未做 systemd 自启，待批准）。

**产物**：`docs/REMOTE_DSH_DEPLOY.md` §13；脚本 `experiments/yolo26_v1/{remote_dsh_web_start_tailnet.sh,remote_serve_dsh.sh,remote_dsh_isrunning.sh}`。

---
# 2026-09-30 Tailscale 组网完成：本地 ↔ 远端 **direct 直连**

**结果：**
- **本地** `abc123456`（100.79.176.111，Windows，1.102.4，服务 Automatic，本来就有）；**远端** 新装 `jxxy`（**100.88.178.19**，1.102.4，systemd `tailscaled` active + **enabled** 开机自启，MainPID 922720）。同一 tailnet `taildd42cc.ts.net`（`jho887891@`）。
- **直连证据**（三条互相印证）：本地 `tailscale ping jxxy` → `pong … via 172.31.68.251:41641 in 8ms`；远端 `tailscale ping abc123456` → `pong … via 10.62.142.122:41641 in 6ms`；远端 status 对本地显示 `active; direct 10.62.142.122:41641, tx 516 rx 564`。**ICMP** `ping 100.88.178.19` 4/4 收到、0% 丢包、avg 50 ms。→ **不是 DERP 中继**。
- **安装链（官方来源，可自证）**：官方仓库 `pkgs.tailscale.com/stable/ubuntu jammy main`；`.deb` 的 `Size=38695094` 与 **SHA256 `758cd0b2…56f8b`** 与官方签名索引（`InRelease` 200）逐位一致 → `sha256sum -c OK`；`apt-get install -y` 本地已验证 `.deb`（依赖 `iptables` 自动装）；`systemctl enable --now tailscaled`；`tailscale set --operator=dgut`；登录链接由**用户本人**点击授权。
- **未越界确认**：`CorpDNS=False`（未改 DNS）、`RouteAll=False`、`ExitNodeID` 空、`AdvertiseRoutes` 无 → **未配 Exit Node / Subnet Router**；未改防火墙、未开公网端口、未做端口转发；`dsh web` 仍只监听 `127.0.0.1:3080`。
- **注意**：无 MagicDNS（用 IP `100.88.178.19`）；本机 DERP 仍被校园网锐捷设备拦截（只影响中继回退）；远端 `MappingVariesByDestIP=true`（对称 NAT），直连依赖打洞成功。

**过程记录（含我的一次失误）：**
- 先按“免 root userspace 模式”尝试我这一侧能做的部分，但**被中止**；随后经用户授权用 sudo（密码仅经 **stdin** 传给 `sudo -S`，未落盘/未回显/未进 history，但已在聊天中出现，建议轮换）完成**标准系统安装**，并清掉 userspace 残留进程（现在只有 `/usr/sbin/tailscaled` 一个进程）。
- 我上一条消息说“已挂后台自动验证”其实**没有真正启动**那个任务，属我的疏漏；后续直接实测完成验证。

**产物：** `docs/REMOTE_DSH_DEPLOY.md` §12；远端中间文件合计约 **721 MB** 待用户决定是否清理（§12.6）；脚本 `experiments/yolo26_v1/{ts_install.sh,ts_install_root.sh,remote_download_ts.sh,remote_download_deb.sh,remote_ts_verify.sh}`。

---
# 2026-09-29（续 4）第 ③ 层「会话记忆」同步到远端

- **传输**：本地 `~/.dsh/{sessions,attachments}` 打包 **213.74 MB** → 远端；解包后就位：`~/.dsh/sessions/--home-T7-ojh-badmin_project--`（**11 个 session**）、`~/.dsh/sessions/--home-T7-ojh--`（**4 个 session**），合计 **48 个文件 / 137 MB**；`~/.dsh/attachments/` **99 文件 / 86 MB**。
- **命名规则不是猜的**：读源码 `@deepseek-ai/dsh-session-persistence-jsonl/lib/index.js:874` 的 `projectKey()`（分隔符 `/:` 合并为 `-`、非 `[A-Za-z0-9._-]` 转 `~XXXX` 大写十六进制、包 `--…--`、去前导 `-`、截 251）；用本地实存目录名 `--E-~5177~8EAB~667A~80FD-badmin_project--` 反验证规则成立，据此算出远端名 `--home-T7-ojh-badmin_project--` / `--home-T7-ojh--`。
- **索引合并**：`~/.dsh/storages/workspace.json` 先备份（`workspace.json.bak-20260929-173539`，227 B），再把本地两个 workspace（id 不变，故 7 + 11 个 sessionIds 仍然挂得上）改写路径后并入：`c48c6ef5…` → `/home/T7/ojh`、`41b71220…` → `/home/T7/ojh/badmin_project`。合并后服务重启正常，注册表未被服务改写。
- **远端实测可解析**：自写 `node` 探针用 `zlib.zstdDecompressSync` 读表头成功，例：`{"type":"session","id":"session-10a03551…","cwd":"E:\具身智能","createdAt":1788414951258}`。
- **遗留风险（已记录，未擅自修）**：transcript 内 `cwd` 仍是**本地 Windows 路径**，而注册表指向远端路径 —— 会话是否能在 UI 里正常「继续」取决于 DSH 用表头 cwd 还是注册表路径（前者会指向不存在的路径）。修法（需用户批准 + 备份 137 MB）：解压每个 `session.jsonl.zstd` → 把 `E:\具身智能\badmin_project` 换成 `/home/T7/ojh/badmin_project`、`E:\具身智能` 换成 `/home/T7/ojh` → 用 Node 22 的 zstd 重新压缩 → 再用同一探针复验可解析。
- **当前访问**：PID 894020，`127.0.0.1:3080`，新 token `stc4A19tsHT_I5YxbPlNrlVW5oSSQXPTu5yBNsH7XYs`；无 token 401 / 带 token 200（34,846 B）。本机隧道 3081 仍可用。
- **（续 4 补）表头 cwd 改写完成**：源码证据表明表头 cwd 参与工作区根与 resume 校验（`dsh-api-workspace-files/lib/index.js:385`、`dsh-acp/lib/index.js:1216`），故对 **42 个 `session.jsonl.zstd`** 解压→替换→重压（`E:\具身智能\badmin_project`→`/home/T7/ojh/badmin_project`、`E:\具身智能`→`/home/T7/ojh`、ETH 数据集路径→远端路径），**FILES=42 / CHANGED=42 / 替换 42 处**；复验 15 个会话表头 cwd 全为远端路径。备份 `~/.dsh/sessions-orig-20260929-173716`（137 MB，可回滚）。
- **最终状态**：远端 PID 895610，`127.0.0.1:3080`，token `i1Naas4p8KL91MzuonWmqq44fo1any1IIrqGgznsMY0`，无 token 401 / 带 token 200。远端 `/` 剩 11 GB；我的临时文件 `/tmp/dsh_sessions.tgz`(214 MB) 与 `/tmp/dshmem-1790703316/`(223 MB) 保留待用户决定是否清理。
- **（续 5）API Key 写入远端**：在远端 `~/.dsh/.credentials.yaml` 新增 `refs.DEEPSEEK_API_KEY`（原文件只有浏览器会话记录），权限保持 **0600**，写入前备份 `.credentials.yaml.bak-20260929-174015`；密钥经 **stdin** 传递（不进 argv/history/日志），**仓库与文档中不记录明文**（只记 `len=35 / sha256 C39756DF…`）。
- **双重验证**：① 掩码读回指纹一致；② 远端实调 `https://api.deepseek.com/models` → **HTTP 200**（返回 deepseek-flash 等模型列表）。**本地 Harness 未改动**（本地那把 key 指纹 `D97FDCF5…`，不是同一把）。
- **当前访问**：PID 897587，`127.0.0.1:3080`，token `OhHyc8JDmSKD_HqfFyHh3NCSALMysXzl31FFr-ULWrw`；无 token 401 / 带 token 200；本机隧道 3081 同样 200。

---
# 2026-09-29（续 3）把本地记忆同步给远端 Harness

**DSH 记忆分三层，按层搬运：**
- **① 工作区记忆（已同步）**：本地仓库的文本层 + 证据图 → 远端 `/home/T7/ojh/badmin_project`，**3,864 个文件 / 29 MB**（压缩包 5.91 MB；`.txt`×3140 7.89 MB、`.csv`×141 10.36 MB、`.py`×283 2.23 MB、`.md`×146 0.95 MB、证据 `.jpg`×21 3.34 MB）。关键文件校验：`AGENTS.md` 2,137 B、`management/DAILY_LOG.md` 63,916 B、`management/ISSUES.md` 41,733 B、`outputs/shuttle_capability/reports/STAGE_B_FINAL_REPORT.md` 21,612 B。排除：`_scratch_*`、`.git`、`env_isaaclab`、`IsaacLab`、`node_modules`、`assets`、`__pycache__`、`relay.log`，以及除证据图之外的全部图像池。
- **② Harness 记忆（已同步，均为新增非覆盖）**：`~/.dsh/.agent-presets/badmin-ptc/{agent.cordis.yml 17,241 B, preset.yml 374 B}` + `~/.dsh/settings.yaml` 1,237 B（模型路由 `deepseek-vision`；**无明文密钥**）。
- **③ 会话记忆（未同步，待决定）**：`~/.dsh/sessions/` 136 MB + `attachments/` 87.5 MB + `storages/workspace.json`（索引按本地路径编码，远端工作区路径不同，故不一定能挂上）。
- **⛔ 密钥层**：`~/.dsh/.credentials.yaml`（含 `DEEPSEEK_API_KEY` 记录与浏览器会话 secret）**未复制**。

**远端已切换到记忆工作区**：`~/.dsh-web/start.sh` 改为以 `DSH_WORKSPACE`（默认 `/home/T7/ojh/badmin_project`）为 cwd 启动，实测 `readlink /proc/890921/cwd → /home/T7/ojh/badmin_project`；重启后 PID 890921，`127.0.0.1:3080`，新 token `T6--DADRoTiiJQiIpyPRL-TQCfDbnQFRboZZm_gkRbE`，无 token 401 / 带 token 200（34,846 B + `__DSH_BOOT__`）。本机隧道（本地 3081→远端 3080）同样验证 200。

**遗留**：远端存在两棵树 —— `/home/T7/ojh/robot_sim`（训练期镜像，旧 `AGENTS.md`，被训练脚本 `--map` 指向）与 `/home/T7/ojh/badmin_project`（本次新同步）；合并方案（rsync 进 `robot_sim`，不带 `--delete`）**未执行，待用户批准**。

**产物**：`tools/sync_dsh_memory_to_remote.ps1`（可复跑的同步脚本）、`docs/REMOTE_DSH_DEPLOY.md` §9（记忆分层与访问信息）。

---
# 2026-09-29（续 2）远程部署独立 DeepSeek Harness（jxxy）

**完成（用户级、无 sudo、只绑回环）：**
- 远程 `dgut@172.31.68.251`（Ubuntu 22.04.5 / x86_64）：勘察确认 **无系统 node**，但 `~/.nvm` 已有 v18.17.0(default)/v20.19.0；git 2.34.1、python3 3.10.12、出口可达 nodejs.org / registry.npmjs.org；3080 空闲；**无既有 dsh**（未覆盖任何东西）。
- **Node 版本硬约束（踩坑）**：18.17.0 装不上（postinstall `import.meta.resolve` 不是函数）；20.19.0 能装但 3 个依赖报 EBADENGINE（`@deepseek-ai/libreoffice-kit`、`@earendil-works/pi-telemetry` 要 `node>=22.19.0`）→ 用 nvm 新装 **Node 22.23.3**（**未改** `nvm alias default`，仍 18.17.0）。
- **覆盖事件（踩坑，已修复）**：`~/.local/bin/dsh` 是 npm 软链，第一次用 `cat >` 写 wrapper 时**穿透软链覆盖了 CLI 的 `lib/bin.js`** → `SyntaxError`；用 `npm install -g` 重写包内文件复原，改用 `mv -T` 原子替换软链。
- 安装结果：`~/.local/lib/node_modules/@deepseek-ai/dsh` = **`@deepseek-ai/dsh 0.2.0-rc.2`**（npm 全局前缀 `~/.local`），入口 wrapper `~/.local/bin/dsh`（写死 Node 22 绝对路径），`bash -lc "dsh --version"` → `0.2.0-rc.2`。
- 启动：`~/.dsh-web/start.sh`（= `cd ~ && dsh web --no-open --host 127.0.0.1 --port 3080`，`setsid nohup` 常驻会话外），PID 870909，**只监听 127.0.0.1:3080**；停止脚本 `~/.dsh-web/stop.sh`。
- **访问 token 是运行时生成的**（`dsh web: http://127.0.0.1:3080/?token=daCRXzxaN8PU0u-6ccwhR9_CYSlRDVZNiZ6PJNRo2TI`，日志 `~/.dsh-web/dsh-web.log`）：不带 token → **401**，带 token → 303 种 cookie → **200**（34,846 B，`<title>DeepSeek Harness</title>`、`__DSH_BOOT__` 命中）。**此前根据 `--help` 说“无 token”是错的**，已更正。
- **本机访问实测通过**：`ssh -f -N -L 3081:127.0.0.1:3080 dgut@172.31.68.251`（本机 3080 被本地 Harness 占用，故用 3081）→ 本机 curl 得 **200**（同样 34,846 B / 标题 / boot 标记），无 token **401**。
- **未做（需批准）**：装 Tailscale、改防火墙/对外暴露、`systemd --user` + `enable-linger` 开机自启、清理 `~/.npm/_cacache`、填写 API Key。

**产物：** `docs/REMOTE_DSH_DEPLOY.md`（部署记录与复现/校验清单）；远端 `~/.dsh-web/{start.sh,stop.sh,dsh-web.log,dsh-web.pid}`。

---
# 2026-09-29（续）远程机失联 + 远程 DSH 部署前置勘察

**完成（只读勘察，未做任何修改）：**
- **远程机已失联**：`ping 172.31.68.251` = False、TCP/22 = False（本会话早些时候可正常 SSH）。本机网络：有线网卡 `以太网` 拿到 **169.254.87.60（APIPA，无 DHCP）**，WLAN `10.37.233.227`；`172.31.68.0/24` 通路消失。
- **Tailscale 现状（本机已装）**：本机 `100.79.176.111 abc123456`（windows），另有 `100.120.181.60 x70`（android，7 天前离线）；**tailnet 里没有目标 Linux 机**。且本机 Tailscale 健康检查报 DERP 连接被劫持（对端证书 `CN=ruijie,OU=SEC,O=Ruijie`）→ 校园网锐捷网关在拦截，属网络策略，**不应绕过**。
- **本地 DSH 版本与官方安装方式（来自官方仓库 README 与 npm registry，未凭记忆）**：本机装的是 `@deepseek-ai/dsh 0.1.5-rc.1`（npm 全局，`C:\Users\abcd1\AppData\Roaming\npm`，25,432 文件 / 222.39 MB）；registry `dist-tags`: **latest = 0.2.0-rc.2**、next = 0.2.0-rc.2、alpha = 0.1.7-alpha.2；**包内未声明 `engines`**。官方两种方式：`npx @deepseek-ai/dsh web`（默认 `http://127.0.0.1:3080`，SSH 下只打印 URL）或源码 `pnpm install && pnpm run build && pnpm dsh web`。
- **`dsh web` 选项实测**（`dsh web --help`）：`--host` / `--port`（0 = 随机）/ `--no-open` / `--trusted-host`；**没有 token 参数**——访问控制靠「绑定的 host + `/api` 浏览器信任栅栏（browser-trust fence）」，不是访问令牌。
- **待用户决策**：目标机是哪台、如何连接（SSH/Tailscale）、远端 Node 装法（用户级 / sudo / 不装只用 npx）、Web 暴露方式（SSH 端口转发 / Tailscale 私网）。按用户要求，涉及 sudo、防火墙、装 Tailscale/VPN、重启、删除或覆盖配置的动作**先问再做**。

---
# 2026-09-29 Stage B 训练完成（70/70）+ 全流程 Gate 判定 + 本地评估环境重建

**完成：**
- **Stage B 正常结束**：`gate6B_lr0001_musgd_b16_w8_20260927-235730`，`results.csv` **70 行**，`metrics.json.elapsed_s = 124,977.6`（34.72 h），peak VRAM **17,455 MB**，effective batch 32，freeze 0，`init_weights = …/gate6A_imgsz1024_b16_w8_20260926-083331/weights/best.pt` ✓，无 OOM、无 NaN。
- **best 仍是 epoch 10**（fitness 0.58263）：epoch 10 mAP50 **0.85013** / mAP50-95 0.55291 / R 0.77301；epoch 11 的 mAP50-95 略高（0.55305）但 fitness 未超；epoch 70 掉到 mAP50 0.64597 / mAP50-95 0.42474 / R 0.62638，而 train box_loss 从 1.26328 一路降到 0.83955 → **典型过拟合，不是发散**。全 70 epoch：max_P 0.95056、max_R 0.77446(e6)、min_mAP50 0.64597(e70)、NaN 行 0。
- **最终权重（已被 ultralytics strip_optimizer 转成推理格式，各 20,136,763 B）**：`best.pt` sha256 `419a3eef…97208b`（=epoch 10）、`last.pt` sha256 `42f0043a…d7f98a4`（=epoch 70）；均已取回本地 `D:\_eth_dl\ckpts\stageB_best_e10_stripped.pt` / `stageB_last_e70.pt`，sha256 与远端逐一相符。
- **全流程 Gate 判定**：`tools/check_stage_b_gate.py --run … --required-epochs 70` → **`PASS_CONTINUE_STAGE_B`**，`lr_violations=[]`（正常组限 0.0011 / 头部限 0.0033，实测全程最大 0.00291468）、`nan=false`、`collapse=false`、`backbone_unfrozen=true`、`init` 正确；内存 `MemAvailable 20.6–32.7 GB`、trainer RSS ≤ 9.59 GB、swap 峰值 2,077.7 KB/s（护栏 20,480）→ 未触发。结果 `outputs/shuttle_capability/metrics/stage_b_gate_final.json`。
- **远端资产核对**：`/home/T7/ojh/robot_sim/outputs/shuttle_capability/v1_dataset/v1_dataset_manifest.csv` sha256 与本地**完全一致**（`11d85026…ed539`），负样本目录齐备（hard_negatives/raw 36 + hard_negatives2/raw 60 + real_images/backgrounds 30）。
- **本地评估环境被外部删除并已重建（ISSUE-028）**：`D:\_yolo26v1` 整目录消失（venv + 解包 wheel + `_deps` + `cfg`），本轮已重建 `D:\_eval26\venv`（Python 3.11）：`ultralytics 8.4.150` + `polars 1.44.2` + **`torch 2.14.0+cu126` / `torchvision 0.29.0+cu126`**，验收 `cuda_ok=True`、`gpu=NVIDIA GeForce RTX 4060 Laptop GPU`。配方落盘 `D:\_eval26\setup_eval_env.ps1` + `install_cuda.ps1`（顺序关键：先 ultralytics 后 CUDA torch，否则被依赖拉成 CPU 版）。
- **远端评估路径已排除**：远端 `env_isaaclab/bin/python -c "import torch"` 在 `timeout 900` 内未完成（`/dev/vdb2` 84%、loadavg 13.9、GPU 上另有他人 3 进程 24.2 GB），FUSE 读挂起 → 改本地。5 个评估工具已同步远端并逐文件 sha256 校验一致（`278b9069…` 等）。
- **更正**：`HARD_NEGATIVE_EVAL.md` 头部原写“本机 cuda:0 RTX A6000”属笔误，本机 GPU 是 **RTX 4060 Laptop 8 GB**（远端才是 A6000），已改。
- **三方对比已完成**（Stage A best e6 / Stage B best e10 / Stage B last e70），val 4,413 张 / 3,260 GT（conf 0.25、IoU 0.5）：
  - **整体**：A `TP 2802 / FP 1269 / P 0.6883 / R 0.8595 / mAP50 0.9051 / mAP50-95 0.5907`；B_e10 `2542 / 1168 / 0.6852 / 0.7798 / 0.8835 / **0.5926**`；B_e70 `1966 / 806 / 0.7092 / 0.6031 / 0.7564 / 0.5044`
  - **判定 `STAGE_B_HEALTHY_BUT_NO_VAL_GAIN`**：唯一领先项 mAP50-95 +0.0019（0.3%，噪声级），其余全负（R −0.0798、mAP50 −0.0216、R_6-16 −0.0634）；e70 不可用
  - **尺寸分桶**：B_e10 仅在 6–8（AP50 +0.0248）与 8–12（+0.0023）桶更好；12–16 / 16–24 / 24–32 段 Recall −0.1682 / −0.2245 / −0.2692（退步足以抵消小目标收益）；32–64 / >64 的“进步”样本量不足（29 / 11 GT）
  - **困难负样本（3 检查点）**：`coco_val`(1000, 未训练) FP/图 A 0.0100 → B_e10 **0.0010** → B_e70 0.0040；`hard_negative`(96, 训练过) A 0.0208 → 两者 0；`frozen_real_bg`(30) A 0.0667 → B_e10 0.1000 → B_e70 0（同一张 `bg_032` 的 0.592 是**真实羽毛球被当背景**＝标注缺失）；`bg_val`(6) A 0.1667 → B_e10 0.8333（同一张图）→ B_e70 0；@0.75 三检查点全为 0
  - Top-FP 图片：A 13 + B_e10 4 + B_e70 4 = **21 张**（`hard_negative_eval/*/top20_fp/`）
- **工具修复**：`tools/eval_hard_negative.py` 的 `hard_negative_predictions.csv` 由“追加写”改为**幂等写**（重算的 (checkpoint,set) 替换、其他保留），否则重复运行会重复计数；并记录 `--images` 为单值参数（按集合调用 4 次，`experiments/yolo26_v1/run_hardneg_sets.ps1`）
- **报告**：`outputs/shuttle_capability/reports/STAGE_B_FINAL_REPORT.md`（292 行 / 21,612 B，10 节，含逐文件 sha256 与再生命令）

**产物：** `outputs/shuttle_capability/metrics/stage_b_gate_final.json`；权重 `D:\_eth_dl\ckpts\{stageB_best_e10_stripped,stageB_last_e70}.pt`；`management/ISSUES.md`（新增 ISSUE-028）。

---
# 2026-09-28（续）困难负样本假阳性评估（Stage A best vs Stage B best@e10）+ Stage B 只读监控

**完成：**
- **Stage B 未修改、未中断**：进程 PID 3839453 存活，`gate6B_lr0001_musgd_b16_w8_20260927-235730` 跑到 **epoch 33/70**（results.csv 34 行含表头），GPU util 48% / 26,442 MiB；`best.pt` 仍为 epoch 10 快照（fitness 0.58263）。全程**只读**（`tail`/`wc`/`/proc`），未写任何远端文件、未发信号。
- **Task A 只读事件监控**：本地监视器 PID 51592 存活，日志 `D:\_eth_dl\stageb_monitor.log`，按 11 类事件触发才写（BEST_REFRESHED / FITNESS_HIGH / METRIC_ANOMALY / RECALL_COLLAPSE / NAN / LR_ANOMALY / MEM_LOW / SWAP_IO_HIGH / CGROUP_HIGH / PROCESS_EXIT / EPOCH_CHECKPOINT）。本次事件只有 2 条：`MONITOR_START` + `BEST_REFRESHED epoch=10 fitness=0.58263 mAP50=0.85013 mAP50-95=0.55291 recall=0.77301`。
- **Task B 困难负样本 FP 评估**（本地 GPU 推理，imgsz 1024、conf floor 0.01）：4 个负样本集 × 2 个检查点，工作点 0.25/0.50/0.75
  - `normal_neg_coco_val`（1,000 张，**未训练**）：FP/图 A 0.0100 → B **0.0010**（@0.25）、0.0020 → **0**（@0.5）；最大误检置信度 0.65460 → 0.28996
  - `normal_neg_bg_val`（6 张，未训练）：A 1 FP → B 5 FP（**全部落在同一张 `tbg_015.jpg`**，含 FP 图数不变 1/6），@0.75 双双归零
  - `hard_negative`（96 张，**100% split=train，训练过**）：A 2 FP → B **0**（@0.25）
  - `frozen_real_backgrounds`（30 张，从未入集）：A 2 → B 3 FP（@0.25），最大置信度几乎不变 0.59207 → 0.59277
- **判定 `PASS_FP_NO_REGRESSION`** + 观察项 `WATCH_bg_negative_single_image`（bg_val 仅 6 张，无统计力）
- **失败模式（目视 4 例）**：主导误检是**6–30 px 的高对比小圆/方块**（天花板射灯、壁灯、红色消防标志），跨数据集复现（中庭照、COCO 浴室镜面）；`bg_032.jpg` 的 0.593“FP”**框住的是一颗真实羽毛球 → 标注缺失，不是误检**；`hn2_056.jpg` 的 0.401 FP 是桌面小器件，Stage B 训后消失（记忆性）
- **新增数据卫生发现（ISSUE-027）**：`hard_negatives2/excluded.txt` 的 7 张排除图（hn2_010/011/012/013/015/016/017）**仍在 `v1_dataset_manifest.csv` 里且 split=train** → builder 从未读取 excluded.txt，这 7 张被 Stage A/B 训练了。修法排在 Stage B 结束之后（改清单会动冻结 sha256）

**产物：**
- 报告 `outputs/shuttle_capability/reports/HARD_NEGATIVE_EVAL.md`（306 行 / 21,065 B，13 节）
- 指标 `outputs/shuttle_capability/metrics/hard_negative_eval_{hard_negative,normal_neg_coco_val,normal_neg_bg_val,frozen_real_backgrounds}.{csv,json}`、`hard_negative_eval_compare.csv`（sha256 `2FF53A29…F7A805D1`）、`hard_negative_fp_confidence_hist.csv`、`hard_negative_predictions.csv`（76,116 B）
- 图片证据 `outputs/shuttle_capability/hard_negative_eval/{stageA_best,stageB_best_e10}/top20_fp/`（13 + 4 = **17 张**，只保存**确实含 ≥0.25 FP** 的图）
- 工具 `tools/eval_hard_negative.py`、`tools/compare_hard_negative_eval.py`、`tools/hard_negative_confidence_hist.py`、`tools/render_top_fp.py`、`tools/audit_neg_membership.py`
- ISSUE：新增 `ISSUE-027`（`management/ISSUES.md`）
- **资源搬迁（AGENTS.md 合规）**：仓库根临时目录 `_scratch_eth_shuttle/`（ETH 官方代码 + 官方 `final-model/best.pt` 128 MB）**未被 .gitignore 覆盖**，且 `*.pt` 属资源 → 已 scp 到远端 `/home/T7/dgut/robot_sim/eth_official_code/`，**73/73 文件 sha256 逐一比对一致**（268,910,149 B）后删除本地副本；逐文件清单 `outputs/shuttle_capability/metrics/eth_official_repo_inventory.csv`，位置登记在 `management/shuttle_detection/IMAGE_RESOURCES.md` §5

**纪律：** 全体推理在本地进行，未占用远端 GPU；Stage B 训练进程零干预；负样本集身份全部由冻结清单 30,321 行逐行核对（`tools/audit_neg_membership.py`）。

---
# 2026-09-28 Stage B：LR 真因定位 + 诊断 smoke + 正式 70-epoch run

**完成：**
- **真因定位（ISSUE-026 更新）**：失败 Stage B 不是「lr0=0.01 选大了」，而是 **`optimizer=auto` 完全忽略 lr0**（`trainer.py:1137-1146`）→ 70 epoch（iterations>10000）自动进入 **MuSGD lr=0.01**，且 `trainer.py:1195` 对检测头 `cv3/one2one_cv3` **×3** → 头部实际 **0.03**（实测 epoch3 为 0.0291468），全网络解冻后被这一学习率打散
- **次生发现**：3 epoch 的 smoke 若仍用 auto，因 iterations=6,273<10,000 会走 **AdamW lr=0.002**，与 70 epoch 不是同一路径 → 先前的 smoke 作废重跑
- **修正**：`run.optimizer: auto → MuSGD`（显式，让 lr0 生效）+ `run.warmup_bias_lr: 0.0` + `stages.B.lr0: 0.01 → 0.001`；其余全部不变（imgsz/batch/workers/dataset/sampler/augmentation/init/结构/seed）
- **训练器新增审计与护栏**：`lr_config.json`（warmup/optimizer/scheduler/8 个参数组的 lr·initial_lr·wd·张量数）、`lr_probe.jsonl`（逐 step 全参数组 LR）、`mem_probe.jsonl`（MemAvailable/SwapFree/swap 速率/cgroup/RSS）；危险条件 = MemAvailable<5GB 或 swap>20MB/s 或 cgroup>90% → 立即 `save_model()` + 正常停机 + 写 MEMORY_GUARD.txt
- **新工具** `tools/check_stage_b_gate.py`（只读判定：NaN / LR 越界 / backbone 解冻 / init 来源 / box_loss·Recall·mAP 坍塌形态）
- **诊断 smoke PASS**：`gate6B_smoke_musgd_lr0001_20260927-211053`，3 epoch，e3 达 mAP50 **0.87153** / mAP50-95 **0.52277**（追平 Stage A e6 的 0.87100 / 0.52278），`max_consecutive_box_loss_rise=0`；已标记 `DIAGNOSTIC_ONLY_STAGE_B_WARMUP`（不续训、不删除）
- **正式 Stage B 已起**：`gate6B_lr0001_musgd_b16_w8_20260927-235730`，`--stage B --imgsz 1024 --batch 16 --workers 8 --epochs 70 --init Stage A best.pt(e6)`，显式 MuSGD lr0=0.001（头部 0.003）、freeze=0、nbs=32
- **epoch 1-5 Gate 判定 = `PASS_CONTINUE_STAGE_B`**：8 组实际 LR 最大 0.00291468/0.00097156（无 0.01/0.02/0.03）、无 NaN、backbone 100% 解冻、init 正确、box_loss 最长连续上升 1 轮、Recall 最低 0.574（warmup 内）
- **截至 epoch 27**：best mAP50 **0.85013**(e10)、**best mAP50-95 0.55305**(e11) → 比 Stage A e6 的 0.52278 **高 5.8%**；box_loss 1.385→**1.107**

**修改文件：**
- `configs/shuttle_detection/yolo26_p2_v1.yaml`（显式 optimizer=MuSGD、warmup_bias_lr=0.0、stages.B.lr0=0.001）
- `tools/train_yolo26_v1.py`（`--init` / `--workers` / LR·内存审计回调 / 扩展内存护栏）
- 新增 `tools/check_stage_b_gate.py`、`outputs/shuttle_capability/reports/STAGE_B_EPOCH5_GATE.md`（11 节，含 12 项要求与 PASS 判定）
- `management/ISSUES.md`（ISSUE-026 真因与解决方案更新）

**测试结果：** 诊断 smoke 3 epoch PASS；正式 Stage B epoch 1-5 Gate PASS（判定脚本输出 `PASS_CONTINUE_STAGE_B`）；epoch 27 仍在跑

**发现的问题：**
- **观察项**：epoch 18→27 mAP50 由 0.818 缓降至 0.799、mAP50-95 由 0.541→0.530，而 train box_loss 仍在下降 → 轻度过拟合迹象（不改本次判定，best.pt 按 fitness 在 e10-e11）
- 守护脚本 `stage_b_runid.txt` 曾写入空值，导致 5-epoch 监控一直读到 results_lines=0；已手工回填并直接跑判定（教训：监控脚本依赖的文件启动后要校验非空）
- 自动化策略里 `util<=80` 曾挡住正式起训（当时他方占 99%）→ 已改为「显存为硬约束、util 仅记录」

**下一步建议：** 让正式 run 继续（剩 43 epoch，约 15 h），跑完用冻结测试集出最终报告；期间按 epoch 观察过拟合是否加剧，必要时用 `best.pt`（e10-e11）而不是最后一个 epoch 的权重
# 2026-09-22 YOLO26 V1 Gate 6 起训（Stage A）与被 OOM 中止

**完成：**
- 数据集全量同步到远端并校验：`eth_remote_images = eth_local_images = 29,377`（17.4 GB）；repo 侧 train_data / hard_negatives / v1_dataset / tools / configs 同步完成
- Gate 6 正式起训（空间自适应选 batch）：起训时 A6000 `free=39.48 GB, util=20%` → 选 **imgsz=1024 / batch=16 / nbs=32 / freeze=11（Stage A 冻结 backbone）**，run `gate6A_imgsz1024_b16_20260922-132656`
- **epoch 1 + 验证完整跑完**：`P 0.863 / R 0.654 / mAP50 0.766 / mAP50-95 0.392`，val/box_loss 1.423，epoch 耗时 1,671.7 s，峰值显存 12.1–12.2 GB（标定公式预测 12.4 GB）；`best.pt`/`last.pt` 各 45.8 MB 已落盘
- 打通的三个远端坑：①PYTHONPATH → 新增 `experiments/yolo26_v1/run_train_v1.sh` 固化环境；②manifest 里是 Windows 绝对路径 → 训练器加 `--map LOCAL=HOST`；③远端缺 polars → 装入 `_deps`（未动 env_isaaclab 环境）

**发现的问题：**
- **训练在 epoch 2 开始约 3 分钟后中止（06:07 UTC），日志无 error/traceback**。宿主 `free -g` 实测 free 0 GB、swap 13/17 GB 在用、`systemd-oomd` 活跃 → **原因 UNKNOWN（内核 OOM 日志需 root），最可能是内存压力击杀**，非代码错误
- 远端此刻被他人任务占用（A6000 free 6.1 GB / util 99%），重训需要等空间

**下一步建议：** 重训前把 dataloader `workers` 降到 2 或先跑 imgsz=640 以降低宿主内存；恢复条件与命令见 `outputs/shuttle_capability/reports/YOLO26_V1_GATE_REPORT.md` Gate 6 节
# 2026-09-21（下午）YOLO26 V1 七道 Gate

**今日目标：** 用户下达《YOLO26 羽毛球超小目标统一训练方案 V1 Spec》并要求「7 个 gate 一起走，不用回复」

**完成：**
- **Gate 1 PASS**：现役 YOLO26 = **yolo26s**（sha256 646f8bc3…84a1b，nc=80）；Ultralytics **8.4.150**；loss = `E2ELoss`(one2many+one2one `v8DetectionLoss`)，assigner = `TaskAlignedAssigner`(topk 10/7, α0.5 β6.0)；**STAL 在 8.4.150 源码中 `\bSTAL\b` 零命中 → 记为不存在，不猜**
- **Gate 2 PASS**：官方 `yolo26-p2.yaml` 存在；`yolo26s-p2.yaml` → 4 检测层 strides 4/8/16/32，9,765,856 参数，28.023 GFLOPs@640（基线 3 层 10.01M / 23.079）
- **Gate 3 PASS**：COCO 权重 → P2 迁移 **6,075,238 参数（61.96%）**，**backbone layer 0–10 覆盖率 100%**，layer 19+ 为 0%（P2 插入导致索引位移，neck/head 重初始化，预期）；停机条件未触发
- **Gate 4 PASS**：`tools/build_yolo26_v1_dataset.py` 产出 unified manifest **30,321 行**（eth_main 20,509 / iphone 2,368 / coco_bg 6,500 / synthetic 817 / bg 31 / hard_neg 96），sha256 `11d85026…`；**location-disjoint 冻结划分**（val = 帧数最少的 3 个 location：uetlibergstrasse_1、ml_3、ml_6）写入 `configs/shuttle_detection/eth_location_split_v1.yaml`；**ScaleAwareSampler** 用 manifest repeat weight 实现（权重 min(3,√(Nmax/Ni))，<4/24-32/32-64/>64 = 3.0），epoch 长度 33,456，负样本比例 **20.0%**，困难负样本 96 张×3；8 个冻结目录硬门禁排除，0 命中
- **Gate 5 PASS**：本地 RTX 4060 上 `tools/train_yolo26_v1.py --stage smoke --imgsz 1024 --batch 2 --subset 400` 跑通（`runs/shuttle_yolo26_v1/smoke_imgsz1024_20260921-170646/`）：box_loss 2.877 单调下降无 NaN、P2 4 层结构训练+验证完整、**峰值显存 2.44 GB**、val 来自 location-disjoint 划分无泄漏；1 epoch mAP=0（符合预期，不作能力结论）
- 新增配置 `configs/shuttle_detection/yolo26_p2_v1.yaml`（spec §27 单一真源：模型/阶段/增强/数据/评测/冻结集）
- 新增工具 `tools/train_yolo26_v1.py`（Stage A/B/C + smoke，config 驱动，写出 resolved_config/environment/sampler/metrics）
- 本地环境：`D:\_yolo26v1\venv` = Python 3.11.9 + torch **2.14.0+cu126** + ultralytics 8.4.150（解包，未安装）

**修改文件：**
- 新增 `tools/build_yolo26_v1_dataset.py`、`tools/train_yolo26_v1.py`、`experiments/yolo26_v1/{gates_1_3_audit.py,gate1b_mechanism.py,probe_env.py,diag.sh,run_*.sh}`
- 新增 `configs/shuttle_detection/{yolo26_p2_v1.yaml,eth_location_split_v1.yaml}`
- 新增 `outputs/shuttle_capability/reports/{YOLO26_V1_GATE_REPORT.md,YOLO26_V1_MODEL_AUDIT.md,YOLO26_V1_ENVIRONMENT.md}`
- 新增 `outputs/shuttle_capability/metrics/{yolo26_v1_gate1_model_audit.json,yolo26_v1_gate1b_mechanism.json,yolo26_v1_gate2_p2_config.json,yolo26_p2_weight_transfer.json}`
- 新增 `outputs/shuttle_capability/v1_dataset/{v1_dataset_manifest.csv,v1_dataset_digest.json,v1_sampler.json,train.txt,val.txt,train_weighted.txt}`
- 新增 `runs/shuttle_yolo26_v1/smoke_imgsz1024_20260921-170646/`（smoke 运行产物）

**测试结果：** Gate 1–5 全部 PASS，均有 JSON/CSV/日志证据；Gate 6/7 未开始

**发现的问题：**
- **Gate 6/7 BLOCKED（共享资源）**：远端 A6000 被另一用户 4B 训练任务占满（40,207/49,140 MiB、99% util），`/home/T7` NTFS 利用率 98.9%、await 365 ms；我在远端的审计进程被内核阻塞在 FUSE `request_wait_answer`（3 分 19 秒仅 9 秒 CPU）→ 改为本地执行；全程只用 `nvidia-smi` 只读，未 kill 任何他人进程
- ETH 数据集同步暂停在 1.5 GB（17.4 GB 总量，可断点续传）；单文件 200 MB 测试可达 20 MB/s，但大量小文件 + 对端 IO 饱和时降到 1–2.6 MB/s
- ultralytics 8.4.150 的 `read_results_csv` 依赖 `polars`，缺依赖会在训练结束后崩溃（训练本身已完成）；已补装

**下一步建议：** 等远端空闲（A6000 空闲显存 ≥12 GB、vdb await <50 ms）→ 续传数据集 → `train_yolo26_v1.py --stage A/B/C --imgsz 1024` → 再跑 640/960/1280；同时补 `tools/eval_yolo26_v1.py` 与 `tools/audit_yolo26_v1_run.py`
# 2026-09-21

**今日目标：** 用户指令：必须下载 ETH/RSL 羽毛球机载检测数据集（Google Drive）与作者代码（GitHub），并核实其真实字段（不引用论文，实测为准）

**完成：**
- **代码已下载**：`_scratch_eth_shuttle/code/shuttle_detection`，commit `745a82916e919313e6db319a12ea873082192666`，39 文件 / 128.1 MiB / AGPL-3.0；含 git-lfs 的 `runs/final-model/best.pt`（128 MiB）与 `config.json`（yolov8s / imgsz=1024 / epochs=50 / dist_threshold=25.0 px / confidence=0.5）
- **数据集已下载**：直连 `drive.google.com` 被 DNS 投毒（解析到 199.59.148.222，443 不可达）→ 走本机代理 `127.0.0.1:7890`（curl 需 `--ssl-no-revoke`）；自写 24 线程断点续传下载器（embeddedfolderview 枚举 + drive.usercontent 逐文件）；**本轮 47,281 文件 / 16,882,706,990 字节 / fail=0 / 4,172.7 s**；落盘 `D:\_eth_data\eth_shuttle_detection`（资源位）
- **实测总览**：52,577 个文件 / 18,648,975,713 字节（17.368 GB）；图像 29,377；txt 23,161。分组：11 地点 **20,509 张 1920×1200**（20,508 个框）、**coco_train_easy 5,500 + coco_val_easy 1,000 无 labels 目录 = 纯背景负样本**、**old/iphone_20251015 2,368 张 1920×1080（1,068 框 + 1,300 空标签负样本）**
- **与论文 Fig.2 对账**：10/11 地点完全一致；cab_2 实测 3406 vs 论文 3407；总数 **20,509 vs 20,510**；**差异已定位到具体文件**：`ticino_1_easy/ticino_1_00212.jpg` 无对应标签；难度分级实测 easy 15085 / medium 4593 / hard 831（论文 15085 / 4594 / 831）
- **解决论文未给字段**：①**分辨率全部 1920×1200**（主数据集，无一例外）；②标注为 Ultralytics YOLO `0 cx cy w h`，单类 `names: {0: shuttle}`，每图恰好 1 框；③**没有官方 train/val/test 划分**——11 地点所有图都在 `images/train/`，各 `val/` 为空目录，划分由代码按地点交叉验证决定（README 明确说明）
- **bbox 尺度实测（640×640 输入等效，eq=sqrt(w·h)×640/1920）**：p1 3.67 / p5 5.50 / p25 7.21 / **p50 9.04** / p75 11.44 / p95 17.15 / p99 24.71 / mean 9.90 / max 108.87 px；分桶 **6-8 26.91% + 8-12 42.65% + 12-16 14.70% = 84.3% 落在 6-16 px**；16-24 5.55%、24-32 0.84%、32-64 0.29%、>64 0.01%
- **extra iphone 组的意外价值**：2,368 张中 1,068 个真实框，其中 **32-64 px 71 框、>64 px 27 框**，是本次唯一含大目标的真实来源

**修改文件：**
- 新增 `tools/audit_eth_dataset.py`（数据集审计工具，--root/--out）
- 新增 `outputs/shuttle_capability/reports/ETH_DATASET_AUDIT.md`（9 节实测报告）
- 新增 `outputs/shuttle_capability/metrics/eth_dataset_audit.json`、`eth_dataset_counts.csv`、`eth_dataset_size_buckets_640.csv`、`eth_dataset_resolution.csv`、`eth_dataset_group_digest.csv`（37 组各 文件数/图像数/txt 数/字节/组内 sha256 摘要）

**测试结果：** 下载 fail=0；审计全量遍历 20,509 张图 + 20,508 框，无读图失败；manifest（每文件 path/bytes/sha256）52,577 行，sha256 = `9371BFA6E52F2F046C933F90F8DC3E2197D3AB9663D0B35F56615FF8239AD433`

**发现的问题：**
- **ETH 在 640 输入下 84% 的框只有 6-16 px**，16-64 px 合计约 7.7%、>64 px 0.14% → 它**填不上你 16-64 px 的真实数据缺口**（只能补 6-16 px 段）
- 远端 `172.31.68.251` 间歇性拒连（`banner exchange: Connection refused` / connect timeout），17.4 GB 资源**暂不能同步到 `/home/T7/dgut/robot_sim/`**（与既有 `management/rebot_b601dm/GATE3_BLOCKED_HOST_UNREACHABLE.md` 同源）
- 标注噪声风险：ETH 为半自动标注流水线（论文自报准确率 85.7%）+ 人工目视难度分级，**训练前必须分层抽检**
- 数据集许可：Drive 目录内未发现 LICENSE/README，**数据使用许可 UNKNOWN**（代码是 AGPL-3.0）

**下一步建议：** 远端可用时同步 + 校验张数/sha256（未校验前不删本地副本，遵 AGENTS.md）→ 按 easy/medium/hard 分层抽检 → 用作者 `best.pt` 在本地冻结测试集上跑一次基线

---

# 2026-09-09

**今日目标：** 用户新指令：PiPER Stage 0（Isaac Lab 中单独正确导入 PiPER，articulation/joint/actuator/limit/坐标系/最小控制验收；不训练 RL、不动 Morph/动力学）

**完成：**
- 旧编排残留进程核查：远端无残留自动化进程（无需再互相 SIGTERM）
- 补齐 Isaac Lab RL extras（官方 editable 方式安装 isaaclab_rl[all]，RC=0）：rsl-rl-lib 5.0.1 / skrl 2.1.0 / rl-games 1.6.1 / sb3 2.9.0 / gym 0.23.1；**Isaac Sim / Isaac Lab / Python / PyTorch / CUDA / Driver 版本零改动**；随后按 baseline 恢复 isaacsim-kernel/isaaclab 硬 pin（aiohttp==3.13.4、click==8.1.7、websockets==12.0、Pillow==12.2.0），重跑 Cartpole 冒烟正常
- baseline 冻结：远端 robot_sim/logs/baseline_env_20260909.txt（pip freeze 325 包）+ IsaacLab tag v3.0.0-beta2.patch1
- PiPER 6 轴（no gripper）Stage 0 全验收通过（脚本存于本地 scripts/piper_stage0/ 与远端 projects/piper_stage0/scripts/）：
  - 官方 AgileX 描述 agilexrobotics/piper_isaac_sim → piper_no_gripper_description.urdf（6 revolute：joint1..joint6，limits 与 effort/velocity 见 joints.csv）
  - t01 官方 isaacsim.asset.importer.urdf 转 USD（fix_base）→ projects/piper_stage0/assets/piper_no_gripper.usd；ArticulationRoot=/piper/Geometry，joints=/piper/Physics/joint1..6(+root_joint)
  - t02：Articulation 创建（InteractiveScene/Articulation）✅；6 关节名/顺序与 URDF 一致 ✅；limits 与 URDF 全部一致 ✅；无 NaN ✅；静止稳定（drift 0.022 rad、|v|max 0.057 rad/s）✅；单关节位置控制 6/6 ok（err≤0.046 rad）✅；全 6 轴位置控制 ok（err≤0.048 rad）✅；joint pos/vel 输出
  - t03：600 s / 144000 步连续运行（8 s 交替双姿态），nan_count=0、anomalies=[]、passed=true ✅；输出 joint_state_final.csv / soak_log.csv
  - 测试增益：stiffness 150 / damping 15（ImplicitActuator，240 Hz；见 DEC-007）

**发现问题：** RL extras 的 pip 解析与 isaacsim-kernel 硬 pin 冲突（aiohttp/click/websockets/Pillow 被改动）→ 已按 baseline 恢复（ISSUE-005）；t01 输出为嵌套子目录 .usda 致首版断言失败、t02/t03 limits 张量形状差异与 SimulationApp API 小坑（ISSUE-006）

**解决问题：** 见 ISSUES.md ISSUE-005/006

**未完成：** Morph One / 羽毛球动力学 / PPO（用户禁止）；Observation/Action v0.1 未开始

**Git commit：** 待本地提交（本次未自动 commit）

**远端运行任务：** PiPER Stage0 测试（logs/piper_t0*.log；outputs/20260909_*）

**Checkpoint：** PiPER 单臂仿真闭环（导入→控制→soak）✅；projects/piper_stage0/assets/piper_no_gripper.usd


# 2026-09-08

**今日目标：** 检查项目目录与当前状态，建立项目管理结构（本轮不写算法代码）

**完成：**
- 检查工作区现状：目录骨架（docs/management/experiments/runs、configs、scripts、src、tests、assets、tools）已存在但全空；Git 仓库已初始化但 0 commit（分支 master）
- 建立根状态文件：README / PROJECT_STATUS / TODO / CHANGELOG / ROADMAP
- 建立 docs/ 8 个设计文档种子 + management/ 5 个管理文件种子 + experiments/EXPERIMENT_INDEX.md
- 初始化 .gitignore；git 分支 master → main

**修改文件：** README.md、PROJECT_STATUS.md、TODO.md、CHANGELOG.md、ROADMAP.md、.gitignore、docs/×8、management/×5、experiments/EXPERIMENT_INDEX.md、各空目录 .gitkeep

**实验：** 无

**发现问题：** 工作区已有空目录骨架与空 git 仓库，但无任何文件/commit，也无环境与硬件记录

**解决问题：** 补齐知识库种子文件，作为后续一切工作的落盘基线

**未完成：** 环境盘点（GPU/CUDA/Isaac 版本、远端主机）、硬件盘点、Observation/Action v0.1

**明日第一件事：** 完成环境与硬件事实盘点（本地跑检查命令，登记到 docs/deployment.md 与 hardware_interface.md）

**Git commit：** a8bf4d1（chore: init project knowledge base structure，分支 main）

**远端运行任务：** 无

**Checkpoint：** 无

**（同日补充，约 18:26）**
- SSH 到远端开发机验证通过：dgut@172.31.68.251（hostname jxxy）；GPU = RTX A6000 49GB、驱动 550.163.01；Python 3.10.12；docker/nvidia-docker 可用；CUDA toolkit / Isaac Sim / Isaac Lab / conda 均未安装 → 事实已登记 docs/deployment.md
- 远端候选目录 /home/T7/ojh 已查看（含 YOLO 权重与实验目录），**尚未定为项目工作区**
- 关联 GitHub origin = https://github.com/jho887891-sudo/badmin（远程空仓库，用于备份/协同）
- 提交身份默认：jho887891-sudo <jho887891-sudo@users.noreply.github.com>（可改为真实姓名/邮箱）
- 补充 commit：（见 git log）

**（安装推进补充 2026-09-08）**
- 决定安装目标：/home/T7/dgut/robot_sim（根分区 100% 满+inode 满，唯一大空间=NTFS/FUSE 的 T7，1.6T 可用）
- 完成：uv venv Python 3.12.13；git clone IsaacLab @ v3.0.0-beta2.patch1（需 -c http.proxy= 绕过失效代理，ISSUE-001）
- 通道核验：uv 可直连 pypi.nvidia.com 解析 isaacsim==6.0.1.0（curl 301→.cn 为误导，ISSUE-002）；IsaacLab 官方兼容表 v3.0.0* ↔ Isaac Sim 6.0.1 ✓
- torch 版本裁决：IsaacLab 官方 pin torch==2.10.0/0.25.0 cu128（DEC-005）；用户消息 2.11.0/0.26.0 为偏差，已记录
- master 脱离脚本 11:25 启动（PID 2776157）：PHASE1 isaacsim 下载完成（158 包，~1h，CDN 峰值 3.8MB/s）；NTFS/FUSE 解包极慢（uv 线程 FUSE request_wait；ntfs-3g 长时间 I/O）；PHASE2/3 待跑
- 后台编排任务 pwsh-4：轮询至 MASTER DONE 后自动执行验证脚本（torch/cuda/isaaclab import/Cartpole 源检查）

**（2026-09-08 晚：安装完成 + 最小验证通过）**
- torch 2.10.0+cu128 经官方源长时重试终于下载成功（期间速率 ~60KB/s→后段 340KB/s）；isaaclab.sh -i 历经 imgui/TMPDIR（ISSUE-004）、robomimic git 代理（ISSUE-001）、rl_games git 超时等问题
- 核心验证全部通过：python3.12/torch 2.10.0+cu128 CUDA True/import isaaclab/Cartpole-Direct-v0 20 步正常（verify_cartpole.py）
- 遗留：mimic/rl-games/teleop 等可选 extras 因 github 间歇不可达未装完（不阻塞当前阶段）
**（2026-09-08 深夜：权威最终验证，SETUP.md 已重写为一致版本）**
- 官方最小测试通过：`./isaaclab.sh -p scripts/environments/zero_agent.py --task Isaac-Cartpole-Direct-v0 --num_envs 128 --viz none`（本 tag 任务名带 -v0）→ env 建立完成，obs Box(128,4)/act Box(128,1)，GPU +2.4GB、util 2-16% 持续 12min 无错
- 版本实测：isaacsim 6.0.1.0；isaaclab 6.1.14（editable @ v3.0.0-beta2.patch1）；torch 2.10.0+cu128 avail=True；python 3.12.13；robomimic 0.4.0 已装
- 关键运行前提（已写入 SETUP.md/env.sh）：root 100% 满 → HOME/TMPDIR/XDG/UV/PIP 全指 /home/T7；EULA 首次接受；--viz none
- **勘误**：此前记录"rl[rsl-rl] 就绪"不实 —— rsl_rl/rl_games/skrl 仍未装（extras 阶段未执行到），待网络稳定后 `source env.sh && bash isaaclab.sh -i` 补齐
- 未开始羽毛球项目开发（遵守边界）

**（2026-09-09 晨：Badminton Scene v0.1 动工）**
- 远端 projects/badminton_scene 包已建（badminton_scene/scene_layout.py 唯一真相源 + court/net/robot/shuttle/camera/racket/contact/physics_materials 各 cfg + scene_cfg 装配 + reset.py）
- 复用 Stage0 piper_no_gripper.usd（未重转），240Hz；scene 用 InteractiveScene + 原生 pxr 静态几何
- t01_scene_1env.py ✅：1 env 构建 + 90 步 0 NaN，93 prim；输出 outputs/t01_20260909_072346/（prim_tree.txt/state_1env.csv/results.json）
- 已修 API：InteractiveSceneCfg class_name 参数、MeshCuboid spawner 空路径→改原生 pxr、TargetZone z 缺失
- 待续：t02 几何核验、racket(piper_with_racket.usd)、contact、canonical 轨迹、reset、128-env、soak、evidence 图片

**（2026-09-09 晨 续）**
- 生成组合资产 piper_with_racket.usd（reference piper_no_gripper.usd + Racket 于 link6，FixedJoint 用 physics:body0/1 relationship 修正），robot_cfg 已指向它
- t02 ✅ 几何核验；t03 ✅ racket frames（bodies 含 RacketBody；驱动关节后 link6 与 racket 位移差<5e-3）
- 证据：outputs/t02_20260909_072449/geometry_check.json、outputs/t03_20260909_072825/racket_frames.json
- 待续：t04 canonical 轨迹、t05 contacts、t06 reset 1000x、t07 128env、t08 soak、evidence 图片与 scene_layout.json/docs/coordinate

**（2026-09-09 续：t04/t05 PASS）**
- 修复关键步进模式：每步需 scene.write_data_to_sim()+sim.step()+scene.update(dt)（Stage0 一致）；新增 step_env 助手
- t04 ✅ CANONICAL_INCOMING_001：越网最低 z=1.543m（>网顶 1.524），~step190 落地；evidence outputs/t04_*/shuttle_trajectory.csv
- t05 ✅ ContactSensorCfg 真实接触：SHUTTLE_GROUND(step72)/SHUTTLE_NET(step52)/SHUTTLE_RACKET(step25)；contact_events.csv（force+速度前后）；说明：per-partner force_matrix_w 对静态伙伴为 0，采用 net_forces_w+受控场景分类（记录于 results.json）
- 注：ContactSensor 需在 PHYSICS_READY 前创建；shuttle 需 activate_contact_sensors=True
- 待续：t06 reset1000、t07 128env、t08 soak、evidence 图片/layout json/docs

**（2026-09-09 续：t07 PASS / t08 运行中）**
- create_scene 支持多 env（每 env 静态几何）；世界坐标写根状态（env_origins+local）
- t07 ✅ 128 env：joint_pos/vel (128,6)、shuttle pos/vel (128,3)、env_origins (128,3)、nan=0、within_cell、net prims=128（outputs/t07_*/results.json；速度由 physx get_transforms 差分，已注明）
- t08 soak 128env×600s sim 已 nohup 启动（outputs/t08_run.log）
- 待续：soak 结果、contact per-partner matrix 增强验证、reachability、截图、scene_layout/coordinate evidence 收尾

**（2026-09-10：真实场景截图完成）**
- 用 IsaacLab Camera（RTX，--enable_cameras）真实渲染，非示意图；t09_screenshots.py；证据含投影可见性（screenshots.json）与颜色分析（screenshot_content_check.json）
- 6 张：world_frame/scene_perspective/scene_top/scene_side/racket_frames (1280x720) + scene_128env (1600x900)
- 取景自检：persp/side 自动换机位后场地绿占比 0.49/0.55；top 0.76；128env 0.27；racket 特写为真实渲染但坐标轴像素未检出（记录 PARTIAL）
- 注意：两个 Isaac 实例互斥（soak 与截图需串行）；soak 已恢复运行

**（2026-09-10 P0 证据修复）**
- Net 几何修正（spec 2）：collision_bottom_z=0.764、top=1.55、thickness 0.02；visual band 同步；canonical 重跑 PASS（越网 min z=1.543>1.524，无碰网；落地 step190）；t05 低轨迹仍产生 SHUTTLE_NET
- t07 重写并 PASS：128env 形状 (128,6)/(128,3)；仅改 env17 其余在自然漂移容差内不变；reset_idx([3,17,81]) 恰好只改这三者；cross_env_collision=0；NaN/Inf=0；GPU before/with128 记录（vLLM 未停）
- 截图重生成 6 张（当前 Net 几何）：5 张 required-object 可见性检查 ok=True；racket_frames 用差分法证明 tcp/contact 坐标轴可见（8133/6499 px）── 严格纯色阈值因色调映射为 0，故以差分证据为准（axis_pixels_rgb 保留记录）
- t08 v2（含 spawn/flight/contact/reset 循环 + nan/inf/reset_fail/cross_env/cuda/physx 计数）已启动，目标 600s 仿真

**（2026-09-10 P0 续：t08 soak 调试）**
- 修：GroundPlane 覆盖不足→大静态地面（400×400×0.4）；shuttle 初始 z=0 位于地面内部→抬到 0.30；球代理无滚动阻力导致弹射后无限滚出本格（cross_env 虚高）→加 linear_damping=1.0/angular_damping=3.0；CSV/心跳周期 1200 与 reset 周期冲突→beat 改 240
- 现状：t08 运行中，t=173s/600s，nan=0 inf=0 reset_ok=34 reset_fail=0 cross_env=0 contacts 累计 17792；速率~0.167×，预计再 ~43 分钟

**（2026-09-10 t08 soak PASS）**
- 128 env × 600s 仿真完成：steps=144000、wall=1328s、nan=0、inf=0、reset_ok=119、reset_fail=0、cross_env_collision=0、contact_events=61440、cuda_fatal=0、physx_fatal=0、anomalies=0、passed=true
- 证据：outputs/t08_20260910_171556/{results.json, run.log, soak.csv, gpu_stats.txt}
- 期间修复：大静态地面（400×400×0.4，防穿透/覆盖不足）、shuttle 初始 z=0.30、加 linear/angular damping（消除无限滚动导致的 cross-env 虚高）、beat 周期与 reset 周期解耦
- 另：Superpowers 插件已装（profile web, superpowers-dsh@0.1.1，14 skills），但需重启 profile 才生效

**（t08 soak 完成 PASS）**
- 128 env，sim_time 600.0s，144,000 steps，wall 1328.4s
- nan=0 / inf=0 / reset_fail=0（reset_ok=119）/ cross_env_collision=0 / contact_events=61,440 / cuda_fatal=0 / physx_fatal=0 / anomalies=0 → passed=true
- 产物（远端 outputs/t08_20260910_171556/，本地 evidence/t08_soak/）：results.json、run.log、soak.csv(482 行)、gpu_stats.txt
- 另：截图相机 bug 已定位（USD xformOp:orient 需 GfQuatd；set_world_poses 未生效）并修复，重拍待做；先前的 screenshots.json 可见性断言已撤回

**（2026-09-13 BADMINTON_ROBOT.md 规范接入 + Phase 0 审计）**
- 收到 Robot 模块设计规范 BADMINTON_ROBOT.md（1671 行，Source of Truth，文档自述"待用户审阅"）
- 归位：docs/simulation/BADMINTON_ROBOT.md；docs 重组为 docs/architecture/（01–08 + COORDINATE_SYSTEM/SIMULATION_ENVIRONMENT/INTEGRATION_TESTING/ROBOT_BRAIN）+ docs/simulation/
- Phase 0 审计（只读）：冻结软件 ✅（Isaac Sim 6.0.1.0 / Isaac Lab v3.0.0-beta2.patch1 / py3.12.13 / torch 2.10.0+cu128）；PiPER 资产 ✅（joints root_joint+joint1..6，7 bodies，mpu=1.0）；场景约定 ✅（240 Hz、Court Frame、env_spacing 15.0）；球拍视觉 ✅ 存在、final 被 REQUIRES_MEASUREMENT 阻塞（符合设计）；现有测试 **59 项全通过**（16+14+7+16+6）
- 发现缺口：docs/architecture/MODULE_INTERFACES.md 缺失（规范 §64/Appendix C 要求先读）；Morph One 资产不存在（规范允许 TEMP，但拓扑必须四转四驱）；规范里的资产路径与仓库实际路径不一致（需 adapter/映射）
- 未进入 Phase 1（等用户确认规范 + 补 MODULE_INTERFACES.md）

**（2026-09-13 Phase 1 + Phase 2 实现完成，TDD）**
- Phase 1 Config：`simulation/robots/badminton_robot/badminton_robot_cfg.py` + `validation/robot_validator.py`；测试 `tests/simulation/robots/test_robot_config.py` **18 项通过**
  - RED：先跑测试得 `ModuleNotFoundError: No module named 'robots'` → 实现 → GREEN
  - AssetStatus 8 级真实性枚举、Param(value+status+source)、四转四驱 8 个语义关节、PiPER 冻结关节序、双目基线校验、development/final 双模式（final 遇到 TEMP/UNMEASURED 直接失败）
- Phase 2 Frames：`frames/robot_frames.py`（Transform/FrameTree/quat 工具/整机链条/env_origin 剥离）；测试 `test_robot_frames.py` **15 项通过**（含 round-trip <1e-12、S41 组合式校验）
  - TDD 中也修了自己的 bug：`from_xyz_quat` 签名与测试不符（按测试为准修实现）、别名 frame 解析顺序
- 记录：`outputs/reports/phase1_config_report.md`、`phase2_frames_report.md`（规范 §67 格式）；TEMP 清单 `morph_one/TEMP_README.md`
- 未开始 Phase 3（Morph One 运动学）——下一步

**（2026-09-13 用户指令：以后都用 TDD 的完整路程）**
- 立即执行一轮完整 TDD 回路，产物 `tools/usd_glb_common.py`（GLB→USD 共享实现）
  - RED：8 项测试对 stub 先失败（2 failures + 6 errors，全部"功能缺失"）
  - 验证 RED → GREEN 分 3 片实现（解析 / 精度兼容 / 网格材质），每片后重跑看绿色增长
  - 验证 GREEN：**全量回归 9 套 109 项全通过**
  - REFACTOR：两个提取器改用共享 helper（168→78 行、144→88 行，消除重复实现）
  - 验证 GREEN(2)：真实提取重跑，tri/bbox/材质与重构前一致；最终羽毛球资产重建成功
- 报告：`outputs/reports/tdd_cycle_usd_glb_common.md`

**（2026-09-13 按 docs/architecture 实现模块化整体架构，TDD 完整回路）**
- 读出架构契约：主闭环 `Perception → Estimation(EKF/UKF) → Prediction → HitFeasibility → InterceptSearch → Planner/PPO → Safety → Execution → Feedback`；规则：模块不得代劳、PPO 不得绕过 Safety、频率不硬编码、MODULE_INTERFACES.md 管消息结构
- 新增 `src/badminton_brain/`：`types.py`（9 类消息契约 + Court Frame/批量/时间戳校验）、`interfaces.py`（8 层接口）、`registry.py`（一层一模块 + replace）、`pipeline.py`（按基线顺序装配与运行、层输出类型校验、逐步计时、reset(env_ids)）、`validation.py`（development/final）
- 新增 `src/common/status.py`：`AssetStatus/Param` 单一来源；robot 模块改为再导出（消除重复定义，42 项测试护栏不变）
- TDD：RED（先空壳 → 断言级失败 failures=6 / failures=10+errors=5）→ GREEN 分 2 片（10/10、15/15）→ 全量回归 **11 套 134 项全绿** → REFACTOR（共享词汇）→ 再回归全绿
- 报告：`outputs/reports/architecture_implementation.md`；`src/README.md` 更新
- 未实现（明确）：各层算法本体；所有模块 is_implemented=False，final 模式报错

**（2026-09-13 严格按 superpowers 实现 Robot Brain 模块：计划 + 并行派发）**
- 加载 skill：`writing-plans`、`dispatching-parallel-agents`、`subagent-driven-development`、`test-driven-development`
- 计划落盘：`docs/superpowers/plans/2026-09-13-brain-modules.md`（T1–T12，含文件结构、验收点、环境约束、规则、ledger）
- 目标更新：goal 改为"实现 8 层模块"，revision 2，max_goal_rounds=24，已 resume
- 并行派发 10 个全新 implementer 子智能体（各自独立文件、纯 numpy、无 GPU）：
  T1 感知几何 40cee11d｜T2 机器人 EKF c053066a｜T3 羽毛球 UKF eca65f2d｜T4 物理预测 6ba71651｜T8 Safety 143d8024
  T5 可行性门 6606ef65｜T6 拦截搜索 44cd9c69｜T7 规划器 61717545｜T9 执行适配器 c95b6743｜T10 在线自适应 fa3c36ec
- 协调者本人在做 T11：`tests/badminton_brain/test_full_brain.py` 骨架已就位（RED：apps.full_brain 未实现）
- 后续（SDD 连续执行）：等各 agent 回报 → 每任务派独立评审子智能体 → 集成 T11 → 全量回归 → T12 整体评审与记录

**（2026-09-13 轮2：并行实现进行中，实测状态）**
- 已落盘模块测试实测：safety_shield 26 OK｜expert_planner 18 OK｜online_adaptation 15 OK｜physics_predictor 16 OK
- 仍在收敛：execution_adapter 24（failures=9 errors=11）｜feasibility 25（failures=1）｜shuttle_ukf 11（errors=11，模块编辑中）
- 未落盘：perception（T1）；T11 full_brain 4 errors（预期：缺 perception/decision/execution）
- 全量回归实测：17 套 224 项，5 套 FAIL（含 T11）—— 均为进行中状态，非最终结论
- T4 已交付（RED: ModuleNotFoundError→断言级；GREEN 16/16；与 rollout 逐点差 0.0；landing Δpoint=1.03e-5 m、Δt=1.98e-6 s）→ **已派独立评审子智能体 cde60461**
- 协调者工具：`tools/run_all_tests.py` 作为全量回归门（本轮用它取得上述实测）

**（2026-09-13 轮2 续：协调者补齐两处集成缺口 + 派 7 个评审）**
- 契约扩展：`RobotSensorState` 增加可选 `odom_twist (N,3)` / `imu_yaw_rate (N,)`；架构测试 10+15 仍全绿（向后兼容）
- 新增 `estimation/estimator.py`（EkfEstimatorModule）：把 T2 的批量 EKF 接到 `EstimationModule`；测试 6/6 通过；羽毛球估计暂为测量直通并显式标 TEMP（待 T3 UKF 注入）
- 新增 `decision/decision_module.py`（FeasibilityDecisionModule）：T5 门 + T6 搜索 → 批量级 (HitDecision, BestIntercept|None)；测试 5 项通过（端到端可行性用例在 T6 空壳期间显式 skip）
- 评审派发：T2 0fe56b7e｜T7 03fc7c8e｜T5 b43cb41b｜T9 aee7cf91｜T10 b3f23862（此前 T4 cde60461、T8 83a9148e）
- 仍在实现中：T1 perception、T3 shuttle_ukf、T6 intercept search（当前仍为空壳）

**（2026-09-13 轮2 续2：T1/T3/T9/T5/T10 已交付，协调者接线三处）**
- T1 感知几何 28/28 绿（三角化误差 5.3e-15 m）｜T3 羽毛球 UKF 13/13 绿（位置 RMSE 0.00568 vs 原始测量 0.01424，k 误差 11%）｜T9 执行适配器 24/24 绿｜T5 可行性门 25/25 绿｜T10 在线自适应 15/15 绿
- 协调者新增接线（均有测试）：
  `perception/perception_module.py` StereoPerceptionModule 7/7（合成路径用 T1 探测器 + 三角化，**ground truth 不泄漏**；real 路径需 measure_fn + 实测标定）
  `estimation/estimator.py` 默认自动注入 T3 的 ShuttleEstimatorBridge（接线测试 3/3；禁用时显式标 TEMP 直通）
  `decision/decision_module.py` FeasibilityDecisionModule 5 项（1 项 skip：T6 仍为空壳）
- 清理误入仓库的 `_scratch_psd.py`
- 待完成：T6 拦截搜索（仍空壳，agent 运行中）→ 之后 T11 集成 + T12 整体评审

**（2026-09-13 轮2 续3：8 层实现完毕 + T11 端到端跑通 + 评审闭环）**
- **8 层全部实现**：registry 探测确认 8 个模块（perception/estimation/prediction/decision/planning/safety/execution/adaptation）
- **T11 端到端 7/7 通过**：canonical 来球 → 感知(合成代理)→EKF+UKF→预测→决策→规划→安全→执行→反馈→自适应，连续 5 步不发散，逐步计时 8 段
- 评审闭环：T4 **APPROVE**（变异体 3/3 被抓、独立复算一致）｜T7 **REQUEST CHANGES → 已修 23/23 绿**（court→body 修复，并实测证明坐标系错会导致轮速偏差 23%）｜T2 **REQUEST CHANGES**（发现一个**死测试**=假 PASS + 协方差断言无判别力 → 已派修）｜T8 **REQUEST CHANGES**（**HIGH：SafetyContext 无人构造=死代码** + 两处可复现越限反例 → 已派修）
- 协调者裁决（已入 plan ledger + DECISIONS）：DEC-014 base_twist 机体系；times 绝对仿真时间（T4 已改，17/17 绿）；SafetyContext 由应用层推入（T11 将加 estop 端到端测试）
- 发现的流程问题：我曾把**本地过期文件 scp 回远端**覆盖了自己的补丁 → 已改为一律"远端改完立即同步回本地"
- 未决：T9 护栏测试（已派）、T5/T9/T10 评审结论待回、T12 整体评审与最终回归

**（2026-09-13 轮2 续4：整体评审 + 仓库卫生）**
- 按 SDD 派 **broad reviewer**（bac9c744）：跨模块一致性（契约纪律/坐标系/时间基/单一真源/TEMP 纪律/假 PASS/全量回归/可维护性）
- 修复中：T2（死测试+协方差断言）、T8（SafetyContext 接线 + 两处越限反例）、T9（机体系护栏测试）
- 评审进行中：T5、T9、T10、broad
- 仓库卫生：删除评审残留 `_t5_probe2.py` 等；本地新增 `.gitignore`（`_scratch_*`、`_probe_*`、`__pycache__`）
- 启动第二轮全量回归（每套 180s 超时，后台 job）

**（2026-09-13 轮2 续5：评审修复收口）**
- T8（38/38 绿）：`set_context`/`context_snapshot` 上线；急停/超时经冻结管线端到端可达；两处越限反例已修（兜底姿态取限位盒内点；HOLD 目标逐个 FK 盒内校验）；`base_twist_max` 拆为逐轴 + 合成速度
- T9（30/30 绿）：按 DEC-015 改写入 `tracking_residual`，`prediction_error=None`
- T5 评审发现**阻断级** NaN 逃逸（预测样本含 NaN 时判 FEASIBLE）→ 已派修；并派 C3 重复真源收敛
- T10 评审：D1 雅可比错误（gain/θ）→ 已派修；D2/D3 由 DEC-015/016 裁决
- 协调者：契约测试假执行模块改新语义（15/15 绿）；决策适配器加 `set_now`（7/7 绿）；生成 `outputs/reports/contracts.md5`（冻结契约 sha256 基线，弥补远端无 git）
- 新增 DEC-015（残差语义拆分）、DEC-016（时间基准与运行时钟推入）

**（2026-09-13 轮2 续6：回归 wave2 结果）**
- 全量回归实测：**24 套通过 / 2 套失败 / 405 项测试**（含 26 套测试文件）；失败两套（execution_adapter、online_adaptation）是**在回归运行之后才完成修复**的，属时序问题而非代码问题
- 已完成修复回归：T2 22/22（7 变异体全杀）、T7 23/23、T8 38/38、T9 30/30、T4 17/17、T11 9/9（含 estop 差异化端到端）
- 新增裁决：DEC-017（src 只读引用 simulation 规范实现，禁止复制）、DEC-018（TEMP 默认值可用于开发，final 必须拒绝）
**（2026-09-13 轮2 续7：全量回归 26/26 绿 422 项 + 运行时接线 + T12 报告）**
- wave 3 全量回归实测：suites_passed=26 suites_failed=0 tests_total=422（全绿）
- T10 收尾 26/26：set_prediction 预测推入通道（真值 1.25 倍阻力下 drag_scale 收敛 1.24999999）
- 协调者接线：apps/full_brain.py 新增 FullBrainRuntime（每步把 prediction 推入自适应层、可推 SafetyContext/时钟）；T11 增至 11/11 绿，实测 prediction_available=[True,True] → 慢环闭环真正转起来
- T9 评审 REQUEST CHANGES（缺 measurement_requirements、TEMP 无警告、几何未走单一真源、一处空测试）→ 已派修
- 产出 T12 验收报告：outputs/reports/brain_modules_acceptance.md（8 层交付/回归/架构规则/评审闭环/裁决/诚实边界/复现）
**（2026-09-13 轮2 续8：broad 评审处置）**
- broad 评审结论：REQUEST CHANGES，但**契约层本身干净**（层边界运行时强制、Safety 不可绕过、无 env_origin 泄漏、TEMP 由 Param 结构性保证、物理/运动学单一真源）；问题全在跨模块组合
- D1（HIGH，time_s 语义）：裁决为绝对仿真时间 → T6 改为绝对 + T7 改为 `t_go = time_s - state.timestamp`（now=3 与 12 输出逐位相同），26/26 绿
- D2（HIGH，截断点当落点）：已派 T5；并把 `landed_within_horizon` 提升为契约字段（DEC-019）
- D3/D8（T11 空转与无断言测试）：我修复 —— estop 测试改用会动的桩规划器建立非零基线 + 差异化断言；reset 测试改为 spy 断言 8 层收到且只收到请求 env_ids（11/11 绿）
- D4（出视场击穿管线）：T1 修复（出视场=正常传感器事件，valid=False，不再抛错），36/36 绿，并有端到端证据 `PIPELINE SURVIVED OUT-OF-VIEW: True`
- D5（球拍位姿与安全工作空间 TEMP 冲突）：我把估计器的 TEMP 球拍偏移改为 (0.30,0,1.20)，落到 TEMP 工作空间盒内
- D6（final 门禁看不见未实测参数）：我在 validation.py 增加对每层 `measurement_requirements()`/`unresolved_limits()` 的查询，final 计 error；实测把频率填成已解析后 `final ok=False` 并逐层列出未实测参数
- D7（自适应空转）：已派 T10 诊断（运行时已推入 prediction，实测 prediction_available=[True,True]）
- 新增 DEC-019（契约字段提升 + final 参数门禁）、DEC-020（推入式通道 + 变异可杀断言纪律）
- 启动最终回归 wave 4（后台）
**（2026-09-13 目标达成待关闭：Robot Brain 8 层模块全部实现并通过评审收口）**
- 最终全量回归：**26 套 / 460 项测试全绿**（wave 4）
- 8 层 + 集成模块文件全部存在（已用 find 实测列出）；`final` 模式实测 `ok=False, errors=33`（逐层列出未实测参数）→ 门禁有效
- 评审收口：6 份模块评审 + 1 份整体评审；T4 APPROVE；T2/T5/T7/T8/T9/T10 修复并回归绿；broad D1–D8 全关闭
- 契约扩展 2 次（DEC-013、DEC-019），均带 DEC 条目与契约测试证据；冻结资产未改动；全程未使用 Isaac/GPU（纯 numpy）
- 未实现/未实测项全部显式登记（ISSUE-006/007/008 + 各模块 measurement_requirements）
- 验收报告：outputs/reports/brain_modules_acceptance.md（10 节）
**（2026-09-13 收尾：契约语义固化 + 最终回归 wave 5）**
- 语义固化（消除契约歧义，均有契约测试护栏）：`odom_twist` = 每步增量（非速度，ISSUE-009 命名债）；`WholeBodyTarget.horizon_s` = 批量最小 deadline（标量）；`ShuttleMeasurement.valid_mask`（DEC-023，出视场时估计层不再把 (0,0,0) 当测量）
- 启动最终回归 wave 5（覆盖 wave 4 之后落地的 valid_mask / T11 物理真值 / 语义注释等改动）
- 全部评审意见（T2/T5/T7/T8/T9/T10 模块评审 + broad 整体评审 D1–D8）均已处置或显式登记；无未处置的 REQUEST CHANGES
**（2026-09-13 收尾：全部评审条目处置完毕 + 测试卫生 + 收官回归 wave 6）**
- broad 评审完整版列出的 D1–D14：D1–D8 已修复并回归绿；D9/D10/D12 登记为 ISSUE-012/013/014；D11 的未使用导入已清理，并确认 DEC-015 在册（评审快照早于写入）
- 测试卫生（D8）：删除两处**陈旧条件 skip**（decision 测试里 T6 早已落地、full_brain 里安全上下文 API 早已具备）——可行分支测试现在真正执行，不再被静默吞掉
- 契约注释（T9 评审 INFO）：SafeCommand 内显式标注 DEC-014 帧裁决（base_twist 机体系 vs 消息 frame='court'）
- 回退修正：estimator 的 state_snapshot 改用公有 state/covariance（消除与 T2 私有属性耦合）
- 启动收官回归 wave 6（含契约 sha256 漂移检查）
- 累计裁决 DEC-013…025（13 条）、问题 ISSUE-006…014（9 条）
**（2026-09-13 收官：wave 6 全绿 26/460）**
- 收官回归 wave 6 实测：**suites_passed=26 suites_failed=0 tests_total=460**（全部评审修复 + 测试卫生 + 语义固化之后的最终确认）
- 契约基线刷新：`outputs/reports/contracts.md5` 现附「哪些 DEC 改了哪个文件」的说明；**registry.py 与 pipeline.py 自 10:16 起从未改动**（说明架构骨架本身不需要为任何评审发现让步）
- 全部评审条目处置完毕：T4 APPROVE；T2/T5/T7/T8/T9/T10 修复回归绿；broad D1–D8 修复、D9–D14 显式登记
- 累计：DEC-013…025（13 条）｜ISSUE-006…014（9 条）
**（2026-09-13 目标关闭：Robot Brain 8 层模块实现完成）**
- 目标正式标记 complete（第 3 轮）
- 完成证据：wave 4/5/6 三次全量回归一致 —— **26 套 / 460 项测试全绿**；本轮复验 integration 11 / pipeline 15 / types 10 全绿
- 8 层模块（perception/estimation/prediction/decision/planning/safety/execution/adaptation）+ 3 协调者适配器 + FullBrainRuntime 全部可运行
- 流程：writing-plans → 10 个全新 implementer（各自 RED→GREEN）→ 6 份模块评审 + 1 份整体评审 → 全量回归；评审发现全部处置或显式登记
- 纪律：TEMP/REQUIRES_MEASUREMENT 结构性保证（Param.__post_init__）；final 模式实测 ok=False errors=33；未进 PPO（仅抛错占位）；未改冻结资产；全程纯 numpy 不占 GPU
- 遗留（非阻塞、已登记）：ISSUE-006…015（10 条）；DEC-026 的 subpixel_centroid 哨兵化由 T1 收尾中
**（2026-09-13 打包交付：robot_sim 源码包，排除资产）**
- 产物：`robot_sim_src_20260913-1443.zip`（**7.9 MB / 295 文件**），远端 `/home/T7/ojh/`，本地 SSOT `E:\具身智能\`
- SHA256（两端一致）：`08e636800295c03e039ce596983dd14c021c850fb72c44b053d11cf31fd238b9`
- 包含：`AGENTS.md` `env.sh` `configs/` `docs/` `experiments/` `management/` `outputs/` `scripts/` `simulation/` `src/` `tests/` `tools/`
- 排除：`assets/`（用户要求，670 MB；包内 assets 条目数实测 0）、`env_isaaclab/`（Python venv，机器相关且体积大）、`IsaacLab/`（第三方源码克隆）、`cache/`、`home/`（机器状态）、`.pytest_cache/`、`__pycache__/`、`*.pyc/*.pyo/*.log`
- 说明：首次用 `zip -r ... -x` 全树排除时因遍历 `assets/`（670 MB、文件极多，NTFS via fuseblk 很慢）超时；改为**显式文件清单 + `zip -@`** 后 295 文件秒级完成（该残留进程已终止）
**（2026-09-13 收官确认：wave 7 全绿 26/463）**
- 全量回归 wave 7 实测：**suites_passed=26 suites_failed=0 tests_total=463**（DEC-026 感知哨兵化之后；相对 wave 6 的 460 增加 3 项，即新增的哨兵语义测试）
- 四次全量回归一致（wave 4/5/6 = 460，wave 7 = 463）→ 契约扩展、场景真值修正、测试卫生、感知 API 变更均无回归
- DEC-026 已由协调者独立验证：空窗 → CentroidResult(valid=False, uv=NaN)；亮斑 → valid=True；含 NaN patch → 仍抛 BrainBoundaryError
- 验收报告更新至 15 节（新增第 14 节 DEC-026 落地验证、第 15 节四次回归一致）

**（2026-09-14 P4-A：校准合成训练集 + 渲染器单一真源）**
- **回答「为什么不用羽毛球三维模型」时发现问题前提不成立**：该 GLB 的羽毛球网格（`Obj_Feather` 3490v/1920f + `Obj_Cork` 3008v/1502f）**只有 POSITION + NORMAL，没有 UV、没有贴图**（`material[0] "White"` 无 baseColorTexture）；GLB 内唯一贴图属于**球拍线**。→ 几何在用（顶点法线也开始用），但"用真外观"在数据上不存在
- 新建渲染器单一真源 `tools/shuttle_render.py`（TDD）：授权法线重心插值平滑着色、3x 超采样抗锯齿、GT 足迹判据、**闭环尺寸标定**、合成（阴影/运动模糊/噪声）、泄漏与空图门禁、Unicode I/O
- **46 项单元测试全绿**；**3 个关键变异全部 KILLED**（DEC-020）——首个变异曾 **SURVIVED**，由此挖出 ISSUE-022
- 产出训练集 **train 400 + val 120**：实测 **2.45–32.00 px**（median 8.94），尺寸控制中位误差 **0.4%**、100% 落在 ±15% 内；从 manifest 重渲染**逐位复现**（差 0.0000）
- **修复 6 个缺陷**（2 个高危）：ISSUE-017 训练背景 `tbg_011.jpg` 与冻结 P3 集 `bg_021.jpg` **同源**（标题/URL + 像素相关 r=1.0000）；ISSUE-021 背景池混入 3 张球场示意图、1 张文字表格、2 张全黑帧、**1 张含 9+ 真实羽毛球**的照片（标注污染）；ISSUE-018 尺寸从未受控；ISSUE-022 标定测量倍率错配；ISSUE-023 阴影自遮挡 + 白羽球比背景暗；ISSUE-019 manifest 漏记参数；ISSUE-020 Windows 非 ASCII 路径/编码
- **三道硬门禁上线**（失败即不渲染）：split 隔离 `25/6/0`、冻结集内容比对 `0 duplicated`、空图检测
- 背景池由 32+7 清理为 **25+6**（7 张移入 `bg_excluded/`，元数据逐条记录理由）；冻结 P3 背景（30 张）首次落入本地 SSOT
- 纪律：全程 TDD + 变异验证；纯 numpy/OpenCV，**未占用 GPU、未启动 Isaac**；产物本地 SSOT，**远端尚未同步**（报告 §7 显式登记）
- 报告：`outputs/shuttle_capability/reports/P4A_REPORT.md`；审计图 `gt_audit_train.png` `bg_pool_audit.png` `bg_tbg006_real_shuttles.png`

**（2026-09-14 P4-A 续：用户质疑「不是有羽毛球模型吗」→ 复查项目模型，修 ISSUE-024/025）**
- **用户质疑成立但需澄清**：项目**确有**自建羽毛球模型（`tools/build_shuttlecock.py` + `configs/shuttlecock.yaml`：16 羽毛、BWF 尺寸、软木半球凸包 + 裙部开口锥壳、质量分布、气动长度）。它是**物理资产**，其视觉体按项目自身策略**必须是外链** —— 四条代码证据：`configs` 的 `visual.mode: EXTERNAL_REFERENCE` 指向 `assets/third_party/shuttlecock_visual.usd`（即导入 GLB 提取物）；`build_shuttlecock.py:136-137` 非 EXTERNAL_REFERENCE 直接抛错；`:337` 程序化网格 `MakeInvisible()` 不可见；`:416-433` 程序化视觉仅 `TEMP_ProceduralDebug`（球+锥）。→ **用 GLB 渲染符合项目策略，不是绕过模型**
- **但质疑暴露两个真实缺陷（ISSUE-024）**：① 渲染器自带尺寸常数 `SHUTTLE_LENGTH_M = 0.0778`（正是本仓库 ISSUE-013 登记的失败模式）→ 改为 `load_shuttle_spec()` 从配置读取；② **无人校验外部视觉体与项目模型尺寸一致** → 新增测试
- **实测一致性（支持"视觉体与模型对得上"）**：裙径 **61.9 mm vs 61.8 mm（差 0.16%）**、总长 77.8 mm vs **79.25 mm（差 1.83%）**
- **连带影响**：标定种子 0.0778→0.07925（+1.86%），约 27% 样本落点差 ±1px → 为保持可复现契约**重新生成数据集**并复验
- **又抓出 ISSUE-025**：记录把 `target_px` 舍入到 3 位小数（与 ISSUE-019 同类，上轮漏项）。实测 `train_00399` 用记录值回放差 **0.000103**、用精确值差 **0.000000** → 图像正确、记录精度错。因其可由 CLI 确定性配方精确重导，**不重渲染**直接回填 train 397/400 + val 120/120 行，回放复验全为 **0.000000**
- 最终：**51 项测试全绿**；数据集验证 **ALL CHECKS PASSED**（train 400：2.45–32.86 px，median 8.94，尺寸控制中位 0.3%；val 120：2.45–31.40 px）