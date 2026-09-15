# 04 — 真实图片与真实视频验证 Spec

## 1. 目标

验证：

```text
三维受控数据上测出的能力
→ 是否真正成立于真实相机数据
```

## 2. 真实图片

真实图片必须带 GT bbox。

至少记录：

- `equivalent_size_px`
- 姿态
- 背景
- 光照
- 清晰度
- 遮挡
- `camera_id`
- 检测结果
- confidence
- Top-K rank

重点比较：

```text
Synthetic controlled result
vs
Real image result
```

如果相同尺寸和相近姿态下真实数据明显更差，优先判定为潜在真实域差异，而不是直接归因于“小目标结构不足”。

## 3. 左右相机

默认共用同一套权重，但分别统计：

- Left Precision / Recall
- Right Precision / Recall
- Left size-bucket Recall
- Right size-bucket Recall
- Left Top-K hit
- Right Top-K hit
- Left / Right miss streak

如果存在持续系统差异：

```text
先检查相机曝光、镜头、安装、采集配置和数据分布
→ 再通过数据补充验证
→ 只有差异仍无法消除，才允许考虑双模型
```

## 4. 真实视频

视频至少覆盖：

- 飞近；
- 飞远；
- 横向；
- 斜向；
- 不同速度；
- 真实运动模糊；
- 部分遮挡；
- 离开画面；
- 重新进入画面。

有逐帧 GT 时正式统计：

- frame-level Precision / Recall
- `longest_miss_streak_frames`
- `longest_miss_streak_ms`
- `mean_miss_streak`
- `reacquisition_frames`
- `reacquisition_ms`
- `stable_detection_duration`

没有逐帧 GT 时只能标记为诊断性连续性测试，不得伪造正式 Recall。

## 5. Top-K 验证

必须同时报告：

- Top-1 是否正确
- Top-K 是否包含真实羽毛球

两种失败必须分开：

```text
Top-1 错，但真实目标仍在 Top-K
≠
真实目标完全不在 Top-K
```

## 6. 输出

必须形成六组结果：

1. Synthetic Capability
2. Real Image Capability
3. Real Video Capability
4. Left Camera Capability
5. Right Camera Capability
6. Temporal Continuity

## 7. 验收标准

本 Spec 通过要求：

```text
真实图片能力已经被量化
+ 真实视频能力已经被量化
+ 左右相机差异已经被量化
+ 时间连续性已经被量化
+ 没有未解释的评测缺口
```

本 Spec 通过后仍不能宣布最终可用，因为系统硬指标尚需 Spec 05 映射。

## 8. 退出状态

`REAL_CAPABILITY_MEASURED`
随后整体状态可标记为：
`BASELINE_MEASURED`
