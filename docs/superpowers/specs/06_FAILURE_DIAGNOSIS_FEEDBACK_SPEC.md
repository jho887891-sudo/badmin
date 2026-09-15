# 06 — 失败诊断与数据回流 Spec

## 1. 目标

本 Spec 只负责：

```text
检测失败
→ 记录
→ 分类
→ 找根因证据
→ 决定需要哪一级调整
```

不负责直接修改模型。

## 2. Failure Record

每次正式失败至少记录：

- sample_id
- source_type
- camera_id
- video_id / frame_id
- target_size_px
- pose
- motion_blur
- background
- lighting
- occlusion
- prediction
- confidence
- top_k_rank
- failure_type
- suspected_root_cause
- evidence
- recommended_adjustment_level

## 3. 失败类型

至少包括：

- `SMALL_TARGET`
- `POSE`
- `MOTION_BLUR`
- `DOMAIN_GAP`
- `FALSE_POSITIVE`
- `EDGE_POSITION`
- `OCCLUSION`
- `LEFT_RIGHT_ASYMMETRY`
- `TEMPORAL_INSTABILITY`
- `COMPUTE_LIMIT`
- `UNKNOWN`

## 4. 根因检查原则

禁止：

```text
6–8 px Recall 低
→ 直接得出“模型结构不行”
```

必须先检查：

```text
对应尺寸训练样本是否足够
→ 对应姿态是否覆盖
→ 真实样本是否足够
→ 标签是否正确
→ resize 后还有多少有效像素
→ confidence 是否误删正确候选
→ 是否为真实域差异
```

只有证据排除这些因素后，才允许把根因升级到结构能力边界。

## 5. 正样本回流

固定测试样本本身不得直接训练。

正确流程：

```text
固定/挑战测试暴露失败模式
→ 另外采集或生成同类型新样本
→ FAILURE_RETURN_POOL
→ 清洗
→ 标注
→ 加入训练
```

## 6. 困难负样本回流

```text
False Positive
→ 分类误检来源
→ 收集新的同类背景样本
→ HARD_NEGATIVE_POOL
→ 重训
```

例如：

- 灯光高亮；
- 白线；
- 球拍反光；
- 衣物白点；
- 远处亮点。

## 7. Challenge Test 规则

挑战集可以持续增加，但挑战样本本身不进入训练。

```text
挑战样本暴露模式
→ 另采同分布训练数据
→ 重训
→ 原挑战样本继续用于验证
```

## 8. 输出

每轮诊断输出：

- `failure_summary`
- `failure_distribution`
- `root_cause_hypothesis`
- `evidence`
- `recommended_adjustment_level`
- `required_new_data`
- `retest_scope`

## 9. 退出状态

`FAILURE_DIAGNOSED`
随后进入 Spec 07。
