# 00 — 羽毛球检测总 Spec

## 1. 目的

本 Spec 定义羽毛球二维检测子项目的完整生命周期：

```text
数据与资产可信
→ 建立 nc=1 基线检测器
→ 测出受控条件下的检测能力边界
→ 在真实图片和真实视频中验证
→ 把机器人系统需求映射成检测硬指标
→ 判断当前检测器是否满足任务
→ 不满足时执行失败诊断和三级调整
→ 独立最终验收
→ 冻结并交付给后续精定位模块
```

最终目标不是“训练出一个模型”，而是得到一套可复现、可测量、可回归、可冻结的羽毛球检测能力。

## 2. 范围

### 2.1 本子项目负责

输入单目图像后输出羽毛球候选区域：

```text
image
→ detector
→ Top-K candidate bounding boxes
→ confidence
→ validity
```

正式输出字段至少包括：

- `timestamp`
- `camera_id`
- `candidate_rank`
- `bbox`
- `confidence`
- `valid`

### 2.2 本子项目不负责

- 亚像素中心定位；
- 左右相机候选匹配；
- 双目三角化；
- 三维位置估计；
- UKF 状态估计；
- 轨迹预测；
- 击球决策；
- 机械臂或底盘控制。

因果边界：

```text
检测器负责找“哪里可能有羽毛球”
→ 后续精定位负责得到更准确的二维图像坐标
→ 双目模块再由左右精确坐标恢复三维位置
```

因此不能用后级算法掩盖检测器本身的能力不足。

## 3. 已确认设计决策

### 3.1 最终验收方式

必须达到真实任务需要的检测能力才能冻结，不是“能力测清楚即可”。

### 3.2 最小目标尺寸来源

```text
真实相机参数
+ 羽毛球真实尺寸
+ 最远有效检测距离
+ 实际工作空间
→ 推导最小实际成像尺寸
→ 加有依据的安全余量
→ minimum_required_target_px
```

不得预先拍脑袋指定 6 px、8 px 等数值。

### 3.3 检测阈值来源

Recall、Precision、连续漏检容忍度、重新捕获时间和检测延迟预算都由整条机器人系统需求反推，不预先固定 95%、20 ms 等数值。

### 3.4 验收指标体系

同时使用：

1. 单帧标准检测指标；
2. 面向后续粗定位任务的候选框指标；
3. 视频时间连续性指标；
4. 实际部署 GPU 上的端到端检测延迟与显存指标。

### 3.5 数据域

三类数据分别验收：

- 三维受控数据；
- 真实图片；
- 真实视频。

任何一类出现系统性硬失败，都不能以其他数据域的高分抵消。

### 3.6 左右相机

左右相机默认共用同一套模型权重，但必须分别统计性能。

如果一侧长期明显更差：

```text
先检查曝光、镜头、相机配置和数据分布
→ 再通过数据补充验证
→ 只有差异仍无法消除，才允许考虑左右相机使用不同模型
```

### 3.7 Top-K 候选

检测器保留少量 Top-K 候选，而不是过早强制只保留最高置信度框。

原因：

```text
Top-1 是误检
→ 如果检测层已经删掉真实候选
→ 后级无法恢复

保留 Top-K
→ 后续双目几何与时间连续性仍可筛选
```

K 的具体值由验证数据和后级计算预算共同确定。

### 3.8 置信度阈值

置信度工作点由误检和漏检的系统代价决定，不直接沿用常见默认阈值。

### 3.9 极小目标判定

标准 IoU 指标和项目任务指标同时保留。

原因：

```text
4–8 px 极小目标
→ bbox 边界偏移 1 px
→ IoU 相对变化很大
→ 但粗框仍可能足够支持后续精定位
```

### 3.10 测试集体系

- `FIXED_CORE_TEST`：首次正式验收前冻结，永不训练，所有版本回归。
- `CHALLENGE_TEST`：持续增长，用于暴露新能力缺口，原样本仍不直接训练。

### 3.11 基线结构

Spec 不永久绑定具体模型型号。

当前允许以现有标准检测结构建立基线，例如 YOLO26s；一级和二级调整尽量保持同一基本结构；更换模型或增加特殊检测分支属于三级调整。

### 3.12 初始化方式

通用预训练权重作为正式基线初始化；随机初始化仅在怀疑预训练先验限制学习时作为诊断对照。

### 3.13 合成与真实数据

采用分阶段 + 混合训练：

```text
Synthetic-only
→ Synthetic + Real
→ 必要时 Real-heavy
```

并显式控制采样比例。

### 3.14 负样本

使用普通负样本 + 失败驱动困难负样本。

### 3.15 系统参数暂缺

允许先建立 baseline 和完整能力曲线，但状态最多为：

`BASELINE_MEASURED`

只有 Spec 05 完成系统需求映射后，才允许判断 `FINAL_ACCEPTED`。

## 4. 项目状态机

```text
ASSET_READY
    ↓
BASELINE_TRAINED
    ↓
CONTROLLED_CAPABILITY_MEASURED
    ↓
REAL_CAPABILITY_MEASURED
    ↓
BASELINE_MEASURED
    ↓
SYSTEM_REQUIREMENTS_MAPPED
    ↓
满足要求？
├─ YES → FINAL_EVALUATION
└─ NO  → FAILURE_DIAGNOSED
           ↓
        LEVEL_1_ADJUSTED
           ↓
         复测
           ↓
       仍不满足？
           ↓
        LEVEL_2_ADJUSTED
           ↓
         复测
           ↓
       仍不满足？
           ↓
        LEVEL_3_ADJUSTED
           ↓
         复测
           ↓
        FINAL_EVALUATION
           ↓
        FINAL_ACCEPTED
           ↓
          FROZEN
```

## 5. 子 Spec

1. `01_DATA_ASSET_READINESS_SPEC.md`
2. `02_BASELINE_TRAINING_SPEC.md`
3. `03_CONTROLLED_CAPABILITY_SPEC.md`
4. `04_REAL_IMAGE_VIDEO_SPEC.md`
5. `05_SYSTEM_REQUIREMENT_MAPPING_SPEC.md`
6. `06_FAILURE_DIAGNOSIS_FEEDBACK_SPEC.md`
7. `07_THREE_LEVEL_ADJUSTMENT_SPEC.md`
8. `08_FINAL_EVALUATION_SPEC.md`
9. `09_FREEZE_HANDOFF_SPEC.md`

## 6. 总退出门

只有以下所有条件成立，羽毛球检测子项目才可以结束：

```text
真实任务最小目标尺寸要求                 PASS
单帧检测指标                           PASS
不同像素尺寸能力                       PASS
三维受控数据                           PASS
真实图片                               PASS
真实视频                               PASS
视频连续漏检要求                       PASS
左右相机分别                           PASS
Top-K 候选要求                         PASS
Precision / 误检要求                   PASS
实际部署 GPU 端到端检测延迟            PASS
显存预算                               PASS
固定核心测试集                         PASS
挑战测试集无未解释系统性硬失败          PASS
```

通过后进入 `09_FREEZE_HANDOFF_SPEC`。
