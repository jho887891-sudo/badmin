# 风险登记（RISKS）

> 持续维护；新风险出现即登记。等级：高 / 中 / 低。

| ID | 风险 | 等级 | 缓解措施 |
|---|---|---|---|
| R-01 | 环境依赖被随意升级（PyTorch / Isaac Sim / Isaac Lab / CUDA）导致实验不可复现 | 高 | 升级前必须先说明风险并记录；环境清单冻结于 docs/deployment.md |
| R-02 | 误删远端数据：未确认路径 rm -rf、误删数据集 / checkpoint | 高 | 禁止未确认删除；删除类操作先报告；远端执行前查 git 状态 |
| R-03 | 真机高速挥拍安全事故 | 高 | 首部署强制：低速模式、action/joint velocity/torque 限幅、workspace、急停可用、清场、Safety Shield 开启（见 README 原则 8） |
| R-04 | 实验不可复现（记录缺失 / 种子丢失 / 版本漂移） | 中 | 每次训练按规范登记 experiments/EXPERIMENT_INDEX.md（含 git commit / seed / 环境版本） |
| R-05 | 仿真与真机差距未量化（sim2real gap） | 中 | docs/sim2real.md 的 A/B/C 参数分类持续维护；DR 每项有依据 |
| R-06 | 只有代码没有设计记录，几个月后无法理解"为什么" | 中 | DECISIONS + docs 与代码同步更新纪律 |
| R-07 | 观测/动作/reward 改动未同步文档导致 API 漂移 | 中 | observation_action.md / reward_design.md 属项目 API，代码改动强制同步 |
