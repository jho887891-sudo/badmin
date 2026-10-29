# Roadmap

## 训练路线：Curriculum Learning（阶段 0 → 9）

> 不从完整对打直接开始。每阶段**稳定收敛并记录实验**后再进入下一阶段。
> 某阶段不收敛时，按序排查（不要盲目加网络复杂度）：
> Observation → Action scaling → Reward → Reset → 接触物理 → 仿真步长 → 控制频率 → Normalization → Curriculum → Domain Randomization

| Stage | 内容 | 验收标准（建议） | 状态 |
|---|---|---|---|
| 0 | 机器人静止：验证环境与动作 | env 可 reset、可随机采样、动作回路闭环 | 未开始 |
| 1 | 固定球轨迹，机械臂学习碰球 | PPO 可收敛，稳定碰球 | 未开始 |
| 2 | 随机少量球轨迹，机械臂学习击球 | 泛化到少量随机轨迹 | 未开始 |
| 3 | 学习控制回球方向 | 目标方向命中率指标 | 未开始 |
| 4 | 加入底盘移动 | 底盘+臂分时协同可用 | 未开始 |
| 5 | Whole-Body Control | 底盘+臂同时协同 | 未开始 |
| 6 | 连续两拍 | 两拍成功率 | 未开始 |
| 7 | 连续 rally | rally 平均/最长拍数指标 | 未开始 |
| 8 | Domain Randomization | 仅在收敛后引入，逐项有依据 | 未开始 |
| 9 | Sim-to-Real | 真机低速验证（遵守真机安全清单） | 未开始 |

## 里程碑（M）

| 里程碑 | 内容 | 状态 |
|---|---|---|
| M0 | 项目基线：知识库结构 + git 基线 | 进行中（本次） |
| M1 | Stage 0 仿真环境可运行 | 未开始 |
| M2 | Stage 1 PPO 碰球 baseline（含首个正式实验记录） | 未开始 |
| M3 | Stage 2~3 随机轨迹击球与回球方向 | 未开始 |
| M4 | Stage 4~5 底盘协同 / Whole-Body Control | 未开始 |
| M5 | Stage 6~7 连续两拍与 rally | 未开始 |
| M6 | Stage 8~9 Domain Randomization + Sim-to-Real 首部署 | 未开始 |

> 详细里程碑跟踪见 `management/MILESTONES.md`；实验记录见 `experiments/`。
