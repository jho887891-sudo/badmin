# 09 — 冻结与交付 Spec

## 1. 目标

冻结通过最终验收的完整检测能力版本，而不是只保存一个权重文件。

## 2. 冻结模型

必须保存：

- final weight
- model architecture
- model config
- class definition
- `nc=1`

并记录模型文件 hash。

## 3. 冻结推理配置

包括：

- input resolution
- confidence threshold
- IoU threshold
- Top-K
- preprocessing
- normalization
- resize / letterbox policy
- postprocessing

原因：

```text
权重相同
→ 推理配置不同
→ Precision / Recall / latency 仍会变化
```

所以不能只冻结 `.pt`。

## 4. 冻结数据版本

记录：

- TRAIN version
- VAL version
- FIXED_CORE_TEST version
- CHALLENGE_TEST snapshot
- 数据来源
- 样本数量
- 尺寸分布
- 合成 / 真实比例
- manifest / hash

## 5. 冻结能力报告

最终报告必须明确写：

```text
稳定检测范围：
>= X px

能力明显下降区间：
X–Y px

低于：
< Y px
不保证稳定检测

最困难姿态：
...

最困难视频条件：
...

Left Camera：
...

Right Camera：
...

最长连续漏检：
...

端到端检测延迟：
...

显存峰值：
...
```

其中 X/Y 必须来自实际测试和系统需求映射，不得编造。

## 6. 冻结已知限制

例如：

- 未覆盖的极端曝光；
- 某高速运动条件样本不足；
- 某尺寸以下不保证检测；
- 相机外观标定仍存在的已知偏差。

已知限制的目的：

```text
告诉下游哪些输入可以信任
→ 哪些条件必须降级、拒绝或保持警惕
```

## 7. 冻结接口

至少明确：

```text
timestamp
camera_id
candidate_rank
bbox
confidence
valid
```

并定义：

- bbox 坐标约定；
- 像素原点；
- 图像分辨率；
- candidate 排序；
- empty detection 语义；
- invalid frame 语义。

必须明确：

```text
没有候选
≠
输出一个全零 bbox
```

## 8. 与后续精定位模块的合同

检测器负责：

- 粗候选区域；
- 置信度；
- Top-K 候选。

检测器不负责：

- 精确中心；
- 三维位置；
- 轨迹。

特别说明：

```text
bbox center 可能存在像素级偏差
→ 后续不能直接把 bbox center 当作最终精确羽毛球中心
```

## 9. Release Identity

建立：

`DETECTOR_RELEASE_ID`

它必须唯一对应：

```text
code commit
+ weight hash
+ model config
+ inference config
+ dataset version
+ evaluation report
```

## 10. 回归测试冻结

保存：

- regression test command
- expected metrics
- allowed tolerance

未来任何：

- 代码修改；
- 重新训练；
- 部署环境修改；

都必须重新跑检测回归测试。

## 11. 新版本回流规则

```text
FROZEN_v1
→ 真机发现新失败
→ CHALLENGE_TEST 增加
→ 另外采集/生成同类训练数据
→ candidate_v2
→ 完整 Spec 08 验收
→ PASS
→ FROZEN_v2
```

不得偷偷覆盖 v1。

## 12. 最终退出状态

`FROZEN`

此时羽毛球二维检测模块才正式允许交付给后续精定位模块。
