# src/ — Robot Brain 源码

按 AGENTS.md，本目录放**算法与控制核心**（仿真与真机共用，见 ROBOT_BRAIN.md S13）。

## 已实现（v0.1 架构骨架）

| 路径 | 内容 |
|---|---|
| `common/status.py` | 共享真实性词汇 `AssetStatus` / `Param`（机器人仿真层与本层共用同一份） |
| `badminton_brain/types.py` | 架构消息契约：`ShuttleMeasurement` / `RobotSensorState` / `UnifiedState` / `PredictedTrajectory` / `HitDecision` / `BestIntercept` / `WholeBodyTarget` / `SafeCommand` / `Feedback`，全部 Court Frame + 批量形状校验 |
| `badminton_brain/interfaces.py` | 8 个层接口（Perception/Estimation/Prediction/Decision/Planning/Safety/Execution/Adaptation） |
| `badminton_brain/registry.py` | `ModuleRegistry`：一层一模块、重复/层错拒绝、`replace()` 换实现 |
| `badminton_brain/pipeline.py` | `BrainPipeline`：按架构基线顺序装配与运行、逐步计时、层输出类型校验、`reset(env_ids)` 传导 |
| `badminton_brain/validation.py` | `validate_architecture()`：development / final 双模式校验 |
| `trajectory/shuttle_aerodynamics.py` | 羽毛球二次阻力 + RK4 飞行模型（k = 1/L，L = 6.5 m） |

## 尚未实现（不假装完成）

感知（YOLO/亚像素/三角测量）、状态估计（EKF/UKF）、可行性判据与拦截搜索、规划/NMPC/PPO、
Safety 具体限值、执行驱动。所有模块 `is_implemented=False`，`final` 模式会明确报错。

文档：`docs/simulation/BADMINTON_ROBOT.md`（机器人本体）、`docs/architecture/ROBOT_BRAIN.md`（总体架构）。
实现记录：`outputs/reports/architecture_implementation.md`。
