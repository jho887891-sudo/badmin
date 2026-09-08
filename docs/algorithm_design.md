# Algorithm Design

**状态：占位（待填写）**

职责：记录各算法模块的设计与选型理由（"为什么这么做"），并在关键改动时同步更新。

覆盖模块：
- 羽毛球动力学模型（重力 / 空气阻力 / 自旋，参数分类见 `docs/sim2real.md`）
- 双目视觉与 3D 定位（相机标定、三角测量/深度）
- EKF / 轨迹预测
- 击球点预测
- 机械臂挥拍策略
- 底盘 + 机械臂协同（Whole-Body Control）
- RL 算法：PPO / Actor-Critic（含是否 asymmetric、是否 privileged，记录于 DECISIONS）
- Safety Shield

写作约定：每个设计记录：目标 → 输入输出 → 方法 → 关键参数 → 验证方式 → 备选方案与选择理由。
