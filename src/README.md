# src/ — Robot Brain 核心源码（当前为空）

按 AGENTS.md，本目录放**算法与控制核心**（仿真之外的真机/算法代码）：

| 子目录（规划） | 内容 |
|---|---|
| `perception/` | 双目视觉、目标检测、球体三维定位 |
| `state_estimation/` | EKF/UKF 状态估计、坐标变换 |
| `trajectory/` | 羽毛球轨迹预测（含空气阻力模型） |
| `hit_decision/` | 击球可行性判断、击球点选择 |
| `planning/` | 运动规划（底盘+机械臂协同） |
| `control/` | Whole-body 控制、阻抗/轨迹跟踪 |
| `safety/` | Safety Shield、安全约束与急停逻辑 |
| `rl/` | PPO 训练与策略部署（Isaac Lab 侧入口放 simulation/） |
| `ros2/` | ROS2 节点与硬件接口 |

**状态**：尚未开始（当前阶段只做 Scene v0.1 仿真场景与资产管线，未进入 PPO/感知）。
仿真侧代码在 `simulation/`，可直接运行的入口在 `scripts/`。
