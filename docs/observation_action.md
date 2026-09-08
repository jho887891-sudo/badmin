# Observation / Action（项目 API）

**状态：草案未定（v0.1 属 P1 任务）**

> ⚠️ 项目 API 规则：**Observation / Action 属项目 API，任何修改必须同步本文档，禁止只改代码不改文档。**
> 每次改动记录：维度变化、版本号、日期、原因、相关 commit。

规划观测（示例占位，维度以最终定稿为准）：
| 项 | 说明 | 维度 |
|---|---|---|
| ball relative position | 球相对位置 | 3 |
| ball velocity | 球速度 | 3 |
| base velocity | 底盘速度 | 3 |
| joint position | 关节位置 | 6 |
| joint velocity | 关节速度 | 6 |
| previous action | 上一步动作 | 9 |
| **Total** | | **待定** |

规划动作（示例占位）：
| 项 | 维度 |
|---|---|
| base vx / vy / wz | 3 |
| arm joint delta ×6 | 6 |
| **Total** | 9 |

版本记录表（从 v0.1 起逐版追加）：
| 版本 | 日期 | 变更内容 | 维度变化 | commit |
|---|---|---|---|---|
| （待起草） | | | | |
