# TODO

优先级：**P0 = 当前阻塞 ｜ P1 = 本周必须完成 ｜ P2 = 后续 ｜ P3 = 想法**

## P0
- [ ] 环境盘点：GPU / CUDA / PyTorch / Python / Isaac Sim / Isaac Lab 版本、远端主机与项目路径（结果写入 `docs/deployment.md`，禁止猜测）
- [ ] 硬件盘点：机械臂 / 底盘 / 双目型号与接口、URDF 来源、ROS topic、CAN ID（写入 `docs/hardware_interface.md`）

## P1
- [ ] git 分支基线规范落地并完成首次 commit（main / dev / feature/*）
- [ ] 起草 Observation / Action 规范 v0.1（含维度表，`docs/observation_action.md`）
- [ ] 坐标系约定 v0.1（世界 / 底盘 / 机械臂基座 / 末端 / 相机 / 球拍，`docs/coordinate_system.md`）
- [ ] 定义 Stage 0 验收标准：Isaac Lab 环境可 reset / 随机采样 / 动作回路可跑

## P2
- [ ] 羽毛球动力学模型（重力 / 空气阻力 / 自旋）
- [ ] 球拍接触模型（racket contact frame 变换）
- [ ] PPO baseline（Stage 1：固定轨迹 → 机械臂碰球）
- [ ] EKF 轨迹预测与击球点预测设计

## P3
- [ ] Safety Shield 设计草图（限幅 / 急停联动 / workspace 约束）
- [ ] Domain Randomization 参数化清单初稿（`docs/sim2real.md`）
- [ ] TensorRT / ONNX 部署路径预研

> 规则：每完成一项立即更新；从"做过了什么"反查历史看 `management/DAILY_LOG.md` 与 git log。
