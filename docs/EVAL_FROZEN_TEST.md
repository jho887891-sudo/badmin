# 固定评估集评估接口（`--split frozen_test`，flag 名保留）

> 工具：`tools/eval_yolo26_v1.py`　｜　注册表：`configs/shuttle_detection/checkpoints_v1.yaml`
> 目的：用 8 个**固定评估集合（fixed evaluation set / development holdout）**给 A/B/C 各阶段产物做对比判定。
> 注意：这些集合已被反复查看使用，不构成“完全未参与模型选择的最终测试集”；口径见 `docs/LOCALIZATION_AUDIT.md` §7。

## 1. 一条命令跑三方/四方对比

```bash
python tools/eval_yolo26_v1.py --repo . --split frozen_test \
    --ckpt-name a_best --ckpt-name b_best --ckpt-name b_last \
    --device cuda:0 --imgsz 1024 --conf 0.25 --neg-thresholds 0.25,0.50,0.75
```

* `--ckpt NAME=PATH`：临时（ad-hoc）检查点，同名时**优先于**注册表；
* `--ckpt-name NAME`：按注册表解析，可重复，**顺序即输出顺序**；后续 Stage C 只要把 `c_best.path` 填上就能直接加进来；
* 注册表条目缺 `path`（占位）时打印 `SKIP`，**不会**中断本次评估。

## 2. 集合的分层（自动发现，不写死清单）

| 类别 | 集合 | 判定口径 |
|---|---|---|
| **有标签** | `controlled_capability`（images+labels）、`challenge_test`（images+labels） | Recall / AP50 / AP50-95 / 尺寸分桶（equiv_size_640，9 桶）/ COMBINED 6-16、4-16 / 次要中心距指标 |
| **无标签** | `synthetic_on_real_bg`、`synthetic_3d`、`real_images/backgrounds`、`real_images/raw`、`real_video/frames`、`real_match_frames/images`、`real_train/raw` | FP/image、image FP rate（默认阈值 0.25/0.50/0.75）+ FP 置信度分布 |

发现规则：每个集合取 `images`/`frames`/`backgrounds`/`raw` 中存在的图像目录（一个集合可有多个 → 拆成多组，如 `real_images/backgrounds` 与 `real_images/raw`）；
标签目录按 `<set>/labels` → `<image_dir>/labels` → 同目录 的顺序找，**必须有 ≥1 个与图像同名的 `.txt`** 才算“有标签”，否则按无标签处理。

## 3. 产物（默认 `outputs/shuttle_capability/metrics/`）

| 文件 | 内容 |
|---|---|
| `size_bucket_metrics_frozen_test.json` | 全量：每个检查点 × 每个集合（有标签给完整分桶/AP，无标签给 FP 统计）+ 检查点元数据（sha256/字节/run/epoch/role） |
| `size_bucket_metrics_frozen_<set>.json` | 单集合、**val 同 schema** → 直接喂 `tools/compare_size_buckets.py` 得到 N 路分桶对比 |
| `frozen_test_matrix.csv` | 统一长表：`checkpoint × set` 一行，召回/AP/组合召回/FP 三阈值一屏看完 |
| `frozen_test_labeled_buckets.csv` | 有标签集合的逐桶 GT/TP/FN/Recall/AP50/AP50-95/low_sample |
| `frozen_test_negatives.csv` | 无标签集合逐阈值 FP/image、image FP rate、FP 置信度 max/mean/median/P95 |
| `frozen_test_checkpoints.csv` | 参与评估的检查点清单（name/path/sha256/bytes/run/epoch/role/imgsz/conf） |

## 4. 验证

```bash
# 单元测试（纯函数：发现/标签解析/FP 统计/百分位/注册表解析）
python tests/test_eval_yolo26_v1_frozen.py -v        # 26 tests
# val 无回归
python tests/test_eval_yolo26_v1.py -v               # 13 tests
# 冒烟（只跑少量图，输出到 _scratch_* 避免覆盖正式产物）
python tools/eval_yolo26_v1.py --repo . --split val         --ckpt-name a_best --max-images 20 --out-dir _scratch_eval_smoke/val
python tools/eval_yolo26_v1.py --repo . --split frozen_test --ckpt-name a_best --max-images 2  --out-dir _scratch_eval_smoke/frozen
```

## 5. 注意

* 无标签集合只回答“误检多少”，**不能**算召回（没有 GT）；有标签集合才给 Recall/AP；
* AP 曲线用 `--ap-conf`（默认 0.001）下限，P/R/F1 用工作点 `--conf`（默认 0.25）；
* 尺寸永远按 **equiv_size_640** 分桶，绝不按 1024 输入像素分桶，保证 640/960/1024/1280 可比；
* FP 不分桶（plan A）。
