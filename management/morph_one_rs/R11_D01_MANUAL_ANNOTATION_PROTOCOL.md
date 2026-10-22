# R11 D01: 模型盲标的比赛视频人工真值流程（2026-10-22）

**项目阶段**：R11 视觉主线 · D01（真实比赛视频 GT）  
**实现**：`tools/vision/annotate_r11_video.py` · `tests/vision/test_r11_annotation.py`  
**验证层级**：`CPU_UNIT_VERIFIED` + `SYNTHETIC_VIDEO_CODEC_SMOKE`。**尚未人工审核 doubles0.mp4 的 336 帧，不构成 REAL_GT_READY。**

## 1. 要解决什么问题

现有 PR #56 的 `R11_M1_broadcast_640` 在 `doubles0.mp4` 336 帧中有框帧数为 0。它**不能单独证明**模型的实际 Recall 是 0%，因为视频中哪些帧存在可评价的真球、球像素多大、哪些帧只是白色模糊点都没有经过人工审阅。PR #59 的 GT-free 分块 YOLO 只能提供新的模型预测，PR #60 的评测器虽然可以算 TP/FP/FN，也必须先有独立的人工标注。

因此新增 **OpenCV 桌面人工审核工具**，完全不加载 YOLO checkpoint，也不显示模型建议框。这个工具能创建 PR #60 `tools/vision/evaluate_r11.py` 直接读取的 `gt.jsonl`，避免把模型输出反过来当模型的“真值”。

## 2. 工作流程和按键

在**有图形桌面的机器**上执行（Windows、Linux 桌面均可，服务器无图形显示不能执行交互标注）：

```bash
python tools/vision/annotate_r11_video.py \
  --video /path/to/doubles0.mp4 \
  --output /path/to/doubles0_human_gt.jsonl \
  --annotator operator1 \
  --display-scale 0.75 \
  --start-frame 0 --end-frame 336 --stride 1
```

Windows 示例（以实际视频所在路径为准）：

```powershell
python tools/vision/annotate_r11_video.py --video "D:\Projects\badmin\data\doubles0.mp4" --output "D:\Projects\badmin\results\doubles0_human_gt.jsonl" --annotator operator1 --display-scale 0.75
```

| 操作 | 实际结果 |
|---|---|
| 鼠标左键按下拖动并松开 | 在**原始图像像素坐标**记录一个人画的 bbox；可画多个，不会读取 YOLO |
| `p` | **positive**，需要至少一个人工 bbox。无框不允许按正例 |
| `n` | **negative**，人工确认该帧没有**可评价的羽毛球**，必须没有候选 bbox |
| `u` | **unscorable**，目标不清楚、严重模糊、无法判断；不能作为 FN 或负例 |
| `s` | **skip/unreviewed**，不写入 GT；**不得自动记为无球** |
| `x` | 撤销当前帧最近一个 bbox |
| `m` | 开关**独立窗口**里的鼠标附近原像素 **6 倍最近邻放大**（不生成新细节、绝不覆盖原视频画面） |
| `q` / `Esc` | 保存已有已审帧并结束；下次相同输入 SHA、帧数、FPS 与标注员编号可继续 |

建议：对于 4–6 px 羽毛球，先移动鼠标使用放大预览检查，再决定是否可以人工确认和框选。若无法辨认应标 `u`，而不是为了完成 336 帧强行画框。不要使用 GT-oracle 自动 ROI、AI 超分恢复图像的“细节”，或者用 YOLO 预测代替人审。

## 3. 产物与严格来源约束

一次标注操作会生成两个文件：

- `<output>.jsonl`：只包含实际按 `p/n/u` **明确确认过**的原始帧，包含 `frame_id`、`timestamp_s`、`timestamp_provenance`、`status`、`image_size` 和 `objects:[{bbox_xyxy_px:[x1,y1,x2,y2]}]`。严格兼容当前 R11 D02/A03 的 `gt.jsonl` schema。
- `<output>.jsonl.meta.json`：视频文件 **原始字节 SHA256**、分辨率、声明 FPS、视频帧数、标注方式、标注员自填标识；显式包含 `held_out_split_verified=false`、`independent_annotation_verified=false`。**不要把自填 annotator 标识当成第三方独立校验。**

状态的含义：

| 人工判定 | 写盘 GT 状态 | 进入评测吗 |
|---|---|---|
| 有可评价目标，亲手画框 | `status=labeled` + `objects` 非空 | 是，正例 |
| 人工确认没有可评价目标 | `status=labeled` + `objects=[]` | 是，负例 |
| 画面不可判断 | `status=unscorable` + `objects=[]` | 不进入 TP/FP/FN |
| 人工跳过，未审核 | **不写 GT 行** | 完全不评价 |

工具使用 **顺序从帧 0 解码**，即使只审核后半段也不依赖不精确的视频 keyframe seek；帧号、时间与真实解码顺序保持对应。视频有 presentation timestamp 时记录 `PTS_VIDEO`；否则由声明 FPS 推算并记录 `ASSUMED_FPS`，不能冒充硬件同步。GUI 显示窗口使用 `WINDOW_AUTOSIZE`，防止任意调整窗口宽高导致鼠标标框投影错位。为避免 4–6px 羽毛球被标注界面的状态文字或放大区域遮住，**状态栏绘制在原视频下方、放大镜位于第二个窗口**，原视频画面未画框时不覆盖任何像素。鼠标误点状态栏不会写 bbox。Linux 无 DISPLAY/WAYLAND_DISPLAY 时提前拒绝 GUI，防止 OpenCV HighGUI 因 headless 环境意外终止。

**不可覆写规则**：一帧一旦保存人工判断，自动恢复会跳过该帧，不允许通过同一个输出路径篡改旧标签。需要纠错时另建输出副本、标明修订者和修订依据，禁止悄悄替换已用于报告的 GT。已有标注但删除了 `.meta.json`，或视频 SHA/分辨率/FPS 不一致时，立即 fail-closed。输出使用同目录临时文件 + 原子替换，每帧提交保存，手动退出和异常中断时已确认帧留存。

**人工真值的可信度限制**：本工具只能证明数据格式被实际人工操作流程管理，无法从机器角度证明是否真的看到了球、是否独立二次复核、是否与训练 session 隔离。后续实验仍需抽查标注样例和人工/数据来源审计。

## 4. 后续 D02/A03 的 CPU 命令

当 GT 确实由人审核并能访问 full/tiled 两组 **同一段真实视频**的检测结果后：

```bash
python tools/vision/evaluate_r11.py \
  --mode detection \
  --ground-truth /path/to/doubles0_human_gt.jsonl \
  --input /path/to/R11_D03_full640_20261022/detections.jsonl \
  --expected-frames 336 --centre-tolerance-px 3 \
  --split-provenance unknown --annotation-provenance human_verified \
  --output /path/to/eval_full640.json

python tools/vision/evaluate_r11.py \
  --mode detection \
  --ground-truth /path/to/doubles0_human_gt.jsonl \
  --input /path/to/R11_D03_tiled640_20261022/detections.jsonl \
  --expected-frames 336 --centre-tolerance-px 3 \
  --split-provenance unknown --annotation-provenance human_verified \
  --output /path/to/eval_tiled640.json
```

注意：评测器的 `--annotation-provenance human_verified` 是操作者的声明，**不会自动校验 `meta.json` 或第三方人审**；因此在完整实验报告中必须提供两个文件的 SHA256 与抽查样例，并明确是否有第二人复核。没有人工 GT 时，这些命令不允许用模型框/伪标签来替代。

## 5. 本次测试与下一道验收门槛

**实际运行**：

- `pytest -q tests/vision/test_r11_annotation.py`：**26 passed, exit 0**。至少一次 RED→GREEN 覆盖无模块、顺序解码、缺 GT 文件的恢复拒绝、4–6 px 原像素放大、屏幕坐标映射。
- `python -m py_compile tools/vision/annotate_r11_video.py`：exit 0。
- 真实 OpenCV `mp4v` 编码器/解码器读写一个 **12 帧合成视频**，对其手动决定写入 3 个合成 GT 行并关闭恢复：**PASS**，输出中正确保留了 `positive/negative/unscorable` 和原始视频指纹。这只验证格式和解码，不是人审 336 帧。
- **未执行真实 GUI 鼠标交互测试**（当前工作环境无图形桌面），Windows GUI 行为尚需实际人工小样检查。
- **没有下载用户 DeepSeek 服务器的视频**，因此未产生 `doubles0.mp4` 的真实人工 GT，没改善 0/336，也不宣称提升 M1=PARTIAL。

下一验收：先在人能看到的机器打开 `doubles0.mp4`，真实标注若干可确认和不可确认帧，观察 `.jsonl` 时间戳、原图像素框，做一次**小样人工核对**；然后继续所有 336 帧（允许部分 `unscorable`），再用 PR #60 精确同源数据的 D02/A03 统计。DeepSeek Harness 仅运行固定 SHA 对照，不负责自行设计或推断人工标签，也不运行 PPO/真机。
