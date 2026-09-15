# 羽毛球检测 Spec 套件

本目录是一套完整的“总 Spec + 小 Spec”体系，只负责羽毛球二维检测，不负责亚像素定位、双目三角化、UKF、轨迹预测、击球决策或控制。

## 执行顺序

```text
00_SHUTTLE_DETECTION_MASTER_SPEC
        ↓
01_DATA_ASSET_READINESS_SPEC
        ↓
02_BASELINE_TRAINING_SPEC
        ↓
03_CONTROLLED_CAPABILITY_SPEC
        ↓
04_REAL_IMAGE_VIDEO_SPEC
        ↓
05_SYSTEM_REQUIREMENT_MAPPING_SPEC
        ↓
能力满足？
├─ 是 → 08_FINAL_EVALUATION_SPEC
└─ 否 → 06_FAILURE_DIAGNOSIS_FEEDBACK_SPEC
          ↓
        07_THREE_LEVEL_ADJUSTMENT_SPEC
          ↓
        回到原测试协议复测
          ↓
        08_FINAL_EVALUATION_SPEC
          ↓
        09_FREEZE_HANDOFF_SPEC
```

## 核心规则

1. 检测目标收缩为 `nc=1: shuttlecock`。
2. 先建立可复现基线，再测能力边界，不默认增加 P2。
3. 三维受控数据、真实图片、真实视频必须分别验收，不能用平均分互相抵消。
4. 单帧指标与时间序列指标同时作为正式验收依据。
5. 左右相机默认共用同一套权重，但必须分别统计性能。
6. 检测模块输出 Top-K 候选，不在检测层过早强制只保留一个框。
7. 固定核心测试集永久冻结；挑战测试集可以持续增长。
8. 固定核心测试集和挑战测试集中的样本不得直接回流训练；只能另外采集/生成同类训练样本。
9. 失败调整严格按“一级局部 → 二级中局 → 三级模型框架”升级。
10. 每次升级必须提交上一层不足的证据。
11. 最终模型必须同时满足检测能力、视频连续性、端到端检测延迟和显存约束。
12. 系统阈值尚未确定时，模型最多标记为 `BASELINE_MEASURED`，不能标记为 `FINAL_ACCEPTED`。
