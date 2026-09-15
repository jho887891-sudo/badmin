# 08 — 最终独立验收 Spec

## 1. 目标

冻结前对最终候选模型进行独立验收。

核心问题：

> 在真实机器人需要的工作范围内，这个检测器是否同时满足检测能力、时间连续性和实时性能要求？

## 2. 进入条件

必须满足：

- Spec 01 PASS
- Spec 02 PASS
- Spec 03 PASS
- Spec 04 PASS
- Spec 05 `SYSTEM_REQUIREMENTS_MAPPED`
- 如曾失败，Spec 06 / 07 已完成对应闭环

## 3. 验收前冻结候选配置

测试前固定：

- candidate model weight
- model config
- input resolution
- confidence threshold
- IoU / NMS config
- Top-K
- preprocessing
- postprocessing
- dataset version
- code commit

测试过程中不得边测边调参数。

## 4. 测试集

### 4.1 FIXED_CORE_TEST

- 永不训练；
- 冻结后不修改；
- 所有正式版本必须跑；
- 用于版本公平比较。

### 4.2 CHALLENGE_TEST

来源包括：

- 真机；
- 真实视频；
- 极端背景；
- 极小目标；
- 高速模糊；
- 极端姿态；
- 历史失败案例。

可以增长，但样本本身仍不直接训练。

## 5. 三类数据分别验收

### Synthetic Controlled
验证尺寸、姿态、位置、模糊、遮挡、背景。

### Real Image
验证真实视觉域。

### Real Video
验证真实运动、模糊、连续漏检和重新捕获。

任何一类系统性硬失败：

`FINAL_ACCEPTED = FALSE`

## 6. 左右相机分别验收

分别报告：

- Precision
- Recall
- size-bucket Recall
- Top-K hit
- miss streak
- reacquisition

## 7. 标准检测指标

- Precision
- Recall
- F1
- mAP50
- mAP50-95

## 8. 项目任务指标

- Top-1 hit rate
- Top-K hit rate
- GT coverage rate
- bbox center error
- size-bucket Recall

## 9. 视频时间指标

- longest_miss_streak_frames
- longest_miss_streak_ms
- mean_miss_streak
- reacquisition_frames
- reacquisition_ms
- stable_detection_duration
- frame-level Recall

## 10. 实时性能

在最终部署 GPU 上测完整检测链路：

```text
image received
→ preprocessing
→ inference
→ decoding / NMS / Top-K
→ result ready
```

至少报告：

- median latency
- P95 latency
- P99 latency
- throughput
- peak VRAM

不得用模型 forward 时间替代端到端检测延迟。

## 11. 与 Baseline 的公平对比

如果最终模型经过调整，必须比较：

- target-size Recall
- real-image Recall
- real-video Recall
- longest miss streak
- Top-K hit
- Precision
- end-to-end latency
- peak VRAM

必须同时写出“改善了什么”和“退化了什么”。

## 12. Final Acceptance Gate

以下全部满足才允许通过：

```text
minimum target size requirement       PASS
single-frame requirement              PASS
temporal continuity requirement       PASS
real-image requirement                PASS
real-video requirement                PASS
left-camera requirement               PASS
right-camera requirement              PASS
precision requirement                 PASS
Top-K requirement                     PASS
latency requirement                   PASS
VRAM requirement                      PASS
fixed core test                       PASS
challenge test                        NO UNEXPLAINED SYSTEMIC HARD FAILURE
```

通过：

`FINAL_ACCEPTED`

失败：

```text
FINAL_REJECTED
→ 生成新的 Failure Record
→ 回到 Spec 06
```
