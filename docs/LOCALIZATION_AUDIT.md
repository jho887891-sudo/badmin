# 定位审计接口（Localization Audit, YOLO26 V1 羽毛球超小目标）

本文档描述 `tools/audit_localization_yolo26_v1.py` 的口径与接口。它回答一个具体问题：

> **当 mAP50-95 在某两个 checkpoint 之间上升时，涨的是“多找到了目标”（Recall），
> 还是“同一批目标的框更准了”（真正的 bbox 定位改善）？**

## 1. 只读边界（硬约束）

- 只**读取**现有 checkpoint 与已标注评估数据；**不训练**、不写权重、不动 optimizer / lr / loss。
- 不修改模型结构、训练配置、数据集、标签、采样器、数据增强。
- **无标注集合（图像池 / 背景 / 视频帧等）不参与任何 bbox 定位指标**：传入未标注集合会被直接拒绝退出。
- 新增/修改的只有评估工具、测试与报告（即本工具、`tests/test_audit_localization_yolo26_v1.py`、`outputs/shuttle_capability/**`）。
- 在本报告得出定位结论之前，**不启动 Stage C**。

## 2. 输入

| 输入 | 默认值 | 说明 |
|---|---|---|
| checkpoint 注册表 | `configs/shuttle_detection/checkpoints_v1.yaml` | `a_best` / `b_best` / `b_last` / `c_best`(占位) |
| 有标注集合 | `val`, `controlled_capability/images`, `challenge_test/images` | `val` 来自 v1 数据集 manifest；后两者来自 `outputs/shuttle_capability/` 下的 fixed evaluation set |
| 数据集 manifest | `outputs/shuttle_capability/v1_dataset/v1_dataset_manifest.csv` | 只取 `split == val` 行 |
| 推理参数 | `--imgsz 1024`，`--conf-floor 0.001`（AP 曲线），`--conf-op 0.25`（工作点） | 与既有 val / fixed evaluation set 报告一致 |

## 3. 指标口径

- **AP50..AP95**：标准置信度排序 PR 曲线（all-point），单图 AP 再按该图 GT 数做加权平均；与 `tools/eval_yolo26_v1.py` 同口径。
- **TP**：IoU ≥ 0.50 的一对一匹配。
- **Near-miss 分档**（每个 GT 取最佳一对一 IoU，`--conf-op` 以上的检测）：
  `A: IoU ≥ 0.50`（= TP）、`B: 0.30–0.50`、`C: 0.10–0.30`、`D: < 0.10 或无匹配`。**B/C 永远不计入 TP**。
- **尺寸桶**：`equiv_size_640 = sqrt(w_px·h_px) · 640 / max(W,H)`，半开区间 `<4 / 4-6 / 6-8 / 8-12 / 12-16 / 16-24 / 24-32 / 32-64 / >64`。
  分桶永远按 640 等效尺寸，**不按 1024 输入像素**。
- **中心误差**：同时给出原始像素与 `px640`（= 原始像素 × 640/max(W,H)）两套，另给
  `normalized_center_error = center_error / sqrt(gt_w·gt_h)`，避免小目标的大数值误差被误读。
- **尺寸误差**：宽/高相对误差、面积比、带符号宽高比与其对数（均值/中位数/P75/P90/P95）。

## 4. 输出

写入 `outputs/shuttle_capability/metrics/`：

| 文件 | 内容 |
|---|---|
| `localization_ap_by_iou.csv` | 每个 checkpoint × 集合的 AP50/55/…/95、mAP50-95、TP/FP/FN、P/R |
| `localization_iou_distribution.csv` | TP 的 IoU 分布：mean/median/std/P10/P25/P50/P75/P90 + 6 个直方图桶 |
| `localization_error_summary.csv` | 中心误差（px / px640 / 归一化 / |dx| / |dy|）、宽高相对误差、面积比、带符号比与对数比 |
| `localization_size_buckets.csv` | 9 个等效尺寸桶的 GT/TP@0.5/Recall/IoU/中心误差/宽高误差/AP50·75·90·95 |
| `localization_checkpoint_compare.csv` | `--compare-a`(默认 a_best) 与 `--compare-b`(默认 b_best) 的分桶增量（含 ALL 行） |
| `localization_near_miss.csv` | A/B/C/D 计数、占比、平均 IoU、中心误差与宽高误差（含分桶明细） |
| `localization_audit.json` | 全部结果 + 每个 GT 的匹配记录 + 推理参数与命令行 |

示例图写入 `outputs/shuttle_capability/localization_examples/`：

```
localization_examples/
  {low_iou,center_error,width_error,height_error,near_miss}/{set}/{NN}_{image}.jpg
  index.csv
```

每张图：红=该 checkpoint 在 `--conf-op` 以上的预测框，绿=GT，青=本例被审计的 GT，黄=与之配对的预测框，
右上角为该 GT 的放大窗。`near_miss` 目录取 0.30–0.50 的近失样本。每类默认 30 张。

报告写入 `outputs/shuttle_capability/reports/LOCALIZATION_AUDIT.md`，第 0 节为人工结论（工具重跑会清空该节）。

## 5. 复现命令

```bash
# 全量：a_best / b_best / b_last × (val + 2 个有标注 fixed evaluation set)
PYTHONUTF8=1 python tools/audit_localization_yolo26_v1.py --device 0

# 小样本冒烟（不覆盖正式产物）
PYTHONUTF8=1 python tools/audit_localization_yolo26_v1.py --ckpt-name a_best \
    --sets val,controlled_capability/images --max-images 8 \
    --out-dir _scratch_localization_audit/smoke_metrics \
    --examples-dir _scratch_localization_audit/smoke_examples \
    --report _scratch_localization_audit/smoke_report.md

# 测试
PYTHONUTF8=1 python tests/test_audit_localization_yolo26_v1.py -v
```

预测缓存在 `_scratch_localization_audit/`（临时目录，可删；删除后用同一命令可完整重生成；缓存键为 checkpoint sha256 + imgsz + conf + 图像数）。

## 6. 判定规则（自动，容差 0.01）

| 结论 | 条件 |
|---|---|
| **CASE A** 只涨召回 | AP50 上升，但 AP75/AP90 与 TP 的 IoU 中位数基本不动 |
| **CASE B** 真正定位改善 | AP75/AP90 上升，TP 的 IoU 中位数上升，中心/尺寸误差下降 |
| **CASE C** 两者同时改善 | AP50 与 AP75/AP90 同时上升 |
| INDETERMINATE | 都不显著 |

## 7. 术语纪律与后续

- 这些固定集合应称为 **fixed evaluation set** / **development holdout**；不要称为“完全未参与模型选择的最终测试集”。
  它们已经在 Stage A/B 的模型选择与报告中被反复查看使用。
- 如果 Stage C 依据本报告的结论调参（含置信度阈值、后处理、训练策略），那么最终判断**必须另用一个全新的、未被看过的 holdout**，
  否则“提升”无法与选择偏差区分。
- 本审计只做诊断，不产生任何训练侧改动。