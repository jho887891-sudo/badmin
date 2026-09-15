# 03 — 三维受控能力测试 Spec

## 1. 目标

测清当前基线检测器在单一变量可控条件下的能力边界。

问题不是“mAP 多高”，而是：

```text
目标变小 / 姿态变化 / 模糊增加 / 遮挡增加
→ 检测能力从哪里开始稳定下降
```

## 2. 目标尺寸定义

保留：

- `bbox_width_px`
- `bbox_height_px`
- `bbox_area_px`

定义主要标量：

```text
equivalent_size_px = sqrt(bbox_width_px * bbox_height_px)
```

原因：

```text
羽毛球三维姿态改变
→ bbox 宽高比例变化
→ 单独宽度或高度不能稳定代表目标视觉尺寸
→ 面积平方根提供统一尺度
```

## 3. 初始尺寸分桶

- `<4 px`
- `4–6 px`
- `6–8 px`
- `8–12 px`
- `12–16 px`
- `16–24 px`
- `24–32 px`
- `>32 px`

如果实际数据分布要求调整桶边界，必须记录调整原因，并保证不同模型版本使用同一套正式分桶。

## 4. 受控变量

### 4.1 像素尺寸

系统扫描目标从明显可见到极小目标的范围。

### 4.2 三维姿态

必须真正改变三维姿态，而不是只做二维旋转。

至少覆盖：

- 球头朝相机；
- 羽毛端朝相机；
- 侧面；
- 斜侧面；
- yaw；
- pitch；
- roll；
- 飞行旋转中的中间姿态。

### 4.3 图像位置

至少覆盖：

- center
- left
- right
- top
- bottom
- four corners

### 4.4 模糊

在相同目标尺寸和姿态下，比较：

- clear
- light blur
- medium blur
- heavy blur

### 4.5 遮挡

至少分：

- none
- light
- partial

### 4.6 背景与光照

必须记录背景和光照条件，避免不同变量混在一起导致错误归因。

## 5. 标准检测指标

- Precision
- Recall
- F1
- mAP50
- mAP50-95

## 6. 项目任务指标

同时计算：

- `Top-1 hit rate`
- `Top-K hit rate`
- GT 是否被候选框覆盖
- bbox center error
- confidence

## 7. 单变量实验原则

一次实验尽量只改变一个主要变量。

例如：

```text
固定姿态、背景、光照
→ 只改变 equivalent_size_px
→ 测尺寸本身导致的能力变化
```

避免：

```text
尺寸更小 + 背景更复杂 + 模糊更重
→ 最终无法判断到底哪个变量导致失败
```

## 8. 输出

至少输出：

- size-bucket metrics
- pose-conditioned metrics
- position-conditioned metrics
- blur-conditioned metrics
- occlusion-conditioned metrics
- baseline capability curves

## 9. 验收标准

本 Spec 的通过条件不是“满足最终机器人要求”，而是：

```text
各主要受控变量下的能力曲线已可靠测出
+ 没有明显标注/评测异常
+ 能力边界可复现
```

## 10. 退出状态

`CONTROLLED_CAPABILITY_MEASURED`
