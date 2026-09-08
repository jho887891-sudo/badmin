# Sim-to-Real

**状态：占位（待填写）**

职责：维护仿真参数三类清单与 Domain Randomization 依据。

## 参数分类
- **A. Nominal parameters**：名义参数（仿真标称值）
- **B. Identified parameters**：实测辨识参数（真机测量/标定得出）
- **C. Randomized parameters**：Domain Randomization 范围（仿真训练用）

## Domain Randomization 至少考虑项（每项必须写：来源 / 默认值 / 范围 / 为什么）

| 参数 | 默认值 | 范围 | 来源 | 为什么设这个范围 |
|---|---|---|---|---|
| mass | 待填 | 待填 | 待填 | 禁止无依据乱随机 |
| COM | 待填 | | | |
| joint friction | 待填 | | | |
| joint damping | 待填 | | | |
| motor delay | 待填 | | | |
| control delay | 待填 | | | |
| sensor latency | 待填 | | | |
| observation noise | 待填 | | | |
| base friction | 待填 | | | |
| racket mass / inertia | 待填 | | | |
| shuttlecock drag | 待填 | | | |
| shuttlecock initial velocity | 待填 | | | |
| wind disturbance | 待填 | | | |

> 约定：每一个随机项都要有依据（来源/默认/范围/为什么）；**不要无依据乱随机**。DR 只在阶段收敛后引入（Stage 8）。
