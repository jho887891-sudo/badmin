# 05 — 系统需求映射 Spec

## 1. 目标

把机器人系统真实需求转换成检测模块硬指标。

不得先拍脑袋定义：

- 最小目标像素；
- Recall 阈值；
- Precision 阈值；
- 最大连续漏检；
- 最大检测延迟。

## 2. 最小目标尺寸

输入：

- 羽毛球真实尺寸；
- 相机内参；
- 图像分辨率；
- 最远有效工作距离；
- 实际场地和工作空间。

推导：

```text
真实几何关系
→ 最不利有效位置的成像尺寸
→ 加有依据的安全余量
→ minimum_required_target_px
```

安全余量必须由可说明因素产生，例如姿态变化、距离误差、标定误差、缩放和真实成像波动；不得随意给百分比。

## 3. Recall 与连续漏检要求

从后级状态估计可容忍的无测量时间反推：

```text
max_tolerable_measurement_gap_ms
÷ camera_frame_period_ms
→ max_allowed_miss_streak_frames
```

然后制定：

- `required_recall`
- `max_allowed_miss_streak_frames`
- `max_allowed_miss_streak_ms`
- `max_reacquisition_time_ms`

## 4. Precision 与 Top-K 要求

错误候选链路：

```text
False Positive
→ 后级候选匹配
→ 可能形成错误三维测量
→ 若未被门控拒绝
→ 污染状态估计
```

因此根据后级过滤能力和计算预算确定：

- `required_precision`
- `confidence_operating_point`
- `top_k`

## 5. 实时性能要求

整条链路：

```text
采集
→ 检测
→ 精定位
→ 三角化
→ 状态估计
→ 预测
→ 决策
→ 控制
```

从总延迟预算分配：

`max_detection_latency_ms`

正式检测延迟定义为：

```text
图像进入检测模块
→ preprocessing
→ inference
→ decode / NMS / Top-K
→ 下游可读取结果
```

还需定义：

`max_vram_mb`

## 6. 输出

正式需求表至少包含：

- `minimum_required_target_px`
- `required_recall`
- `max_allowed_miss_streak_frames`
- `max_allowed_miss_streak_ms`
- `max_reacquisition_time_ms`
- `required_precision`
- `confidence_operating_point`
- `top_k`
- `max_detection_latency_ms`
- `max_vram_mb`

当前尚未确定的值必须标记为：

`REQUIREMENT_DERIVED_LATER`

这表示该值有明确推导来源，而不是任意占位。

## 7. 状态判断

参数不完整：

`REQUIREMENT_PARTIAL`

全部完成：

`SYSTEM_REQUIREMENTS_MAPPED`

随后比较：

```text
MEASURED_CAPABILITY
vs
SYSTEM_REQUIREMENT
```

全部满足 → 进入 Spec 08。

任一硬指标不满足 → 进入 Spec 06。
