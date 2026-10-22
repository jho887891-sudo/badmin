# R11 D03 | 真正的 YOLO 全图 vs 分块推理对照（仅执行指令）

## 背景和边界
- 对标历史：`R11_M1_broadcast_640`，同一个 `best.pt` 在 `doubles0.mp4` **336 帧 0 检出**。这只能说明在该视频及检查过的置信度下没有模型检测框；**尚未证明**根因是域偏移、像素尺寸、运动模糊、视频根本没有清晰羽毛球或预处理。
- 本 PR 为 D03 真实实验提供**实际代码**：`src/perception/shuttle_detection/tiled_inference.py` + `run_visual_pipeline.py --tile-size`，不改模型、权重、GT、检测阈值或输入视频。
- 图像被固定规则切成原像素窗口，**不读取 GT**。每块调用一次真实 `YOLO.predict()`；将框移回源图像，跨块 class-aware NMS。全尺寸 1280×720、tile-size 640、overlap 0.25 → **6 个块/帧**，而不是一次前向；**可能显著增加时延**。
- 本轮 **7 个 CPU 测试已在独立本地 TDD 环境通过**，不是 GPU YOLO 复测。能否改善广播检出率：`NOT_RUN`。

## DeepSeek / GPU 操作人员：只执行，禁止自行改代码

**先核对 GPU 独占 A04 锁、GPU 无其他 Kit 实验。严格 checkout PR HEAD 完整 SHA**（从 PR #59 等 GitHub 实际链接获得；禁止自动合并任何受保护分支）。只读检查下面资产真实存在，然后使用空的新 Run 目录。建议 `bash -o pipefail` 运行并记实际退出码。

```bash
# 运行前核对本次 PR 的 HEAD SHA 和已有资产
git rev-parse HEAD
sha256sum /home/dgut/.dsh-bench/runs/detect/runs_cont_v1/full_cont_e20/weights/best.pt
sha256sum /home/dgut/runs/g/data/videos/doubles0.mp4

python -m pytest tests/perception/shuttle_detection/test_tiled_inference.py -q

# 对照 1：原图一次 YOLO，conf 0.01（这不是检测恢复结论）
python scripts/vision/run_visual_pipeline.py \
 --source /home/dgut/runs/g/data/videos/doubles0.mp4 \
 --weights /home/dgut/.dsh-bench/runs/detect/runs_cont_v1/full_cont_e20/weights/best.pt \
 --imgsz 640 --conf 0.01 --iou 0.45 --device cuda:0 \
 --out-dir /home/dgut/runs/g/R11_D03_full640_20261022

# 对照 2：同一个视频/权重/imgsz/conf；只有分块方式变化
python scripts/vision/run_visual_pipeline.py \
 --source /home/dgut/runs/g/data/videos/doubles0.mp4 \
 --weights /home/dgut/.dsh-bench/runs/detect/runs_cont_v1/full_cont_e20/weights/best.pt \
 --imgsz 640 --conf 0.01 --iou 0.45 --device cuda:0 \
 --tile-size 640 --tile-overlap 0.25 \
 --out-dir /home/dgut/runs/g/R11_D03_tiled640_20261022
```

已有同名 Run 目录时换一个唯一 RUN_ID，**不能删除/覆盖旧 Run**。没有权重或视频则报告缺失文件，不得用别的模型/视频替代后声称完成这一对照。

## 验收解释（不要过度宣称）
- 两组对照必须输出 `manifest.json`（含模型和视频 SHA、tile 配置）、`metrics.json`（`frames_processed`、有框帧占比、全量 latency p50/p95/p99/max）、逐帧 `detections.jsonl` 和 `tracks.jsonl`（包含 `detector_diagnostics`）、实际 `overlay.mp4`、命令及退出码。
- `full_frame` 的 `detector_diagnostics.tiles=1`；`tiled` 对 1280×720、640 tile 的 `tiles=6`，并逐帧给出 candidate_count/kept_count。不得从 overlay 漂亮程度推断精度。
- 没有人审正样本与 bbox 真值时，`frames_with_detection/336` 只能称检测**有框帧占比**，不能称 Recall/AP，框可能全是误检。应在 D01 提供“可见羽毛球/模糊/不可见”人工标注，再复核每个候选框是否是真球。
- `tiled` 多次前向可能耗时更高，不保证实际机器人可用。完整端到端时延和未来 0.3–0.8s 预测稳定性单独验证。
- 这是图像二维路径；`IMAGE_PLANE_2D` 不得送入 3D 控制，`hit_rate=null`；不执行 PPO、真机控制。
