# ETH Stage 1 | BBox-only pixel-exact Copy-Paste

Independent entry point: tools/generate_eth_stage1_bbox_preview.py

### Contract

* Use the existing verified ETH TRAIN inventory, expected 13,992 boxes by default.
* Crop source RGB pixels inside the existing bbox; paste onto non-white solid 1024 x 1024 background.
* **No Mask, no GrabCut, no alpha matting, no resize, no blending.** The small amount of original background inside the bbox remains. The entire crop including those pixels is bit-for-bit preserved.
* Images are created outside the source dataset. No training or model weights are touched.
* This is a separate preview and data generation path; it does not bypass the existing formal-training gates.

### Example: 100 images, then human QA

~~~bash
python tools/generate_eth_stage1_bbox_preview.py \
  --inventory /home/dgut/eth_audit/out/inventory_20261026T074352Z/instances.csv \
  --output-dir /home/dgut/eth_audit/out/stage1_bbox_preview_$(date +%Y%m%dT%H%M%S) \
  --mode preview --count 100 --seed 42 \
  --time-bins 10 --width 1024 --height 1024
~~~

Outputs: images/000001.png ... 000100.png, labels/, contact_sheet.png,
manifest.csv, audit.json, SHA256SUMS. Overview uses nearest-neighbor
**review-only** zooms; the 100 exported PNGs contain no target scaling.

### Full 13,992 unique source coverage

~~~bash
python tools/generate_eth_stage1_bbox_preview.py \
  --inventory /path/to/verified/instances.csv \
  --output-dir /NEW/full-output-dir \
  --mode full --seed 42
~~~

The full mode uses each instance once, which is NOT equal scene/time exposure.

To maintain all unique instances and equalize exposure across observed
scene x time cells via recorded repetitions:

~~~bash
python tools/generate_eth_stage1_bbox_preview.py \
  --inventory /path/to/verified/instances.csv \
  --output-dir /NEW/balanced-output-dir \
  --mode balanced-full --seed 42 --time-bins 10 \
  --max-outputs 100000
~~~

Balanced-full fails if any expected scene/time cell is empty or if requested
image count exceeds the explicit max-outputs budget; it does not silently omit
cells. This mode may generate more than 13,992 images. Review space before run.

### Source validation, time semantics, and safety

SHA256 checks verify both image and label bytes. Any mismatch aborts before
output creation; existing output folders are rejected. The input inventory
must contain only train / eth_main / eth_train instances and the six declared
locations. Per-image original and output bbox and hashes are recorded.

Within each location + set_dir, final numeric filename token is treated as a
frame index and binned into equal frame-index ranges. This is a proxy for
capture progress, **NOT a verified physical-time or uniform FPS claim**.
The report includes actual exposure counts for every observed scene/time bin.
An empty time bin blocks balanced-full. Previews balance as far as discrete
unique samples allow. Appearance quality is UNKNOWN before human inspection.

Run CPU tests with:

~~~bash
python -m unittest -v tests/test_eth_stage1_bbox_preview.py
~~~

Do not claim tests or dry-run samples establish real-data success. Check
contact_sheet.png, suspicious bbox rectangular artifacts, output label bounds,
and audit.json after executing on the remote ETH mount.

Never commit generated PNGs, datasets, weights or server-local outputs to Git.
Do not change main or the four-stage official Mask-reviewed training path.

## 2026-10-27 真实服务器预览回传（独立 BBox-only 路线）

> 来源：真实 Linux 服务器执行者的回传。GitHub 已独立确认代码 SHA 和生成器实现；**下面的远端运行次数、文件存在性及 SHA256 校验由服务器报告，GitHub 不能直接读取服务器目录复算**。因此状态分为 SOURCE_VERIFIED / SERVER_REPORTED_PASS / VISUAL_QA_UNKNOWN。

| 维度 | 核验与来源 |
|---|---|
| 代码与固定版本 | [`5d9ab53d586a2cac05ce7568954c349220818445`](https://github.com/jho887891-sudo/badmin/commit/5d9ab53d586a2cac05ce7568954c349220818445)，`feature/eth-four-stage-copypaste`；GitHub SOURCE_VERIFIED |
| 服务器工作区 | `/tmp/badmin-eth-work`；未修改代码、未产生训练提交（执行者报告） |
| CPU Gate | 新增测试 **8/8 PASS**；inventory 回归退出码 **0**；日志 `/tmp/eth_stage1_cpu_tests_5d9ab53.log` |
| 真实 ETH 清单 | **13,992 个实例**；SHA256 `1bc3887d4236c69257ddd2e0f3c5fb9542ac93e4da7fa4c7960e4a66fe9761ed` |
| 预览样本 | **100 张 PNG + 100 个 TXT**；6 场景 `cab_1,cab_2,glc_1,ml_4,ticino_1,ticino_2` |
| 时间覆盖 | **10 个 frame-index 桶**；这是录像采集进度的代理，不代表等秒/等物理时间 |
| 输出 | `/home/dgut/eth_audit/out/stage1_bbox_preview_5d9ab53_20261027T094901` |
| 输出性质 | 每张输出画布 **1024×1024**，Copy-Paste 不缩放（`scale=1.0`），无 Mask/GrabCut/alpha/插值；仅 review 用 `contact_sheet.png` 最近邻缩略或放大 |
| 文件完整性 | 执行者报告 `SHA256SUMS` 全部匹配；contact sheet SHA256 `1ec9e13a951a5a28db5198699f72a302854dc931c3dfb9f122833a97ad5b51e8` |
| 审计字段 | `audit.status=PASS_DATA_GENERATION_NOT_TRAINING`；`model_trained=false`；`optimizer_steps=0` |

**已实现的精确语义**：只裁切原图 bbox 内的 RGB 像素并平移到纯色画布，源 bbox 内的原背景也被一并保留。它与四阶段正规 Stage 1 的“人工审核 Mask 前景”不同，只能称为 **BBox-only 对照/预览数据**，不能计入 `approved_masks` 或直接当作已批准训练集。

### 下一道唯一 Gate：真实图像视觉 QA

1. **先看原有** `contact_sheet.png`（100 张全图 + ROI 放大）。人工逐图检查 bbox 是否真正包含羽毛球、球可辨识程度、框内背景形成的矩形边缘、球的截断与源场景带入。不要把接触表中的最近邻放大图误认作原始导出图片。
2. 记录逐样本 review：`ordinal`、`instance_id`、`source bbox`、`output bbox`、`visibility`（VISIBLE / AMBIGUOUS / NOT_VISIBLE）、`rectangle_artifact`（YES / NO）、`label_correct`（YES / NO）、`decision`（ACCEPT / REJECT / NEEDS_REVIEW）、`reason`；不可见/不确定按保守规则不准入训练。
3. 给出场景×时间桶×尺寸分层接受率，单独统计微小目标（例如 bbox 4–6 px、8–12 px）的失败原因。无需重新生成 13,992 张图片来完成这个 Gate。
4. 如果审查显示大量明显矩形伪特征，**暂停 BBox-only 扩展和训练**，回到已实现的 Mask 候选→人工审核路线；如果大部分合格，仍只能进入一个独立于正式四阶段训练的**受控 BBox-only 消融实验申请**。训练与独立 GT 视频评测都需后续 Gate，不由此报告自动授权。
5. 任何质量/检测性能数字在拿到 review.csv 或真实 GPU/评测日志前保持 **UNKNOWN**；不以 100/100 图片生成成功代替视觉可用率或 Recall。

**最小服务器回传**：`audit.json`（含 `by_scene`/`by_scene_time_bin`）、`manifest.csv` 审计汇总、`contact_sheet.png` 人工审核结果（不是只报文件存在）、review.csv 及 SHA256 和退出码。源数据/生成 PNG 保留服务器，不向 GitHub 提交大文件。
