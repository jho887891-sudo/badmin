# Project Status

> 保持简短：30 秒看懂项目在哪。每日工作结束或关键节点更新。

- **Current Stage:** 仿真环境已就绪并冻结 baseline；**PiPER Stage 0 完成**（导入/articulation/限位/坐标系/控制全部验收通过）；RL Stage 0 未开始
- **Current Goal:** PiPER Stage 0 验收（已完成）→ 下一阶段待用户指令（硬件参数盘点 / Observation-Action v0.1 / Morph 集成等，均未开始）
- **Working:** Isaac Sim 6.0.1.0 + Isaac Lab v3.0.0-beta2.patch1（包 6.1.14）+ torch 2.10.0+cu128 于远端 jxxy 运行稳定；PiPER 6 轴（no gripper）URDF→USD 导入 + Isaac Lab Articulation 全验收通过（含 10 分钟 soak）；RL 训练框架 rsl_rl/skrl/rl_games/sb3 已补齐；baseline（pip freeze 325 包）已冻结于远端 robot_sim/logs
- **Not Working:** 无算法/RL 训练代码；Morph One / 羽毛球动力学 / PPO 未开始（遵守边界）；真机硬件参数未确认
- **Current Experiment:** 无训练（experiments/EXPERIMENT_INDEX.md 尚未登记）
- **Best Model:** 无
- **Best Metrics:** 无
- **Main Blocker:** 无（PiPER 单臂仿真闭环已通）。注意：根分区仍 100% 满（未清理，需用户批准）；vLLM 占用 ~24.8GB 显存
- **Next Step:** 等用户指令进入下一阶段（候选：真机硬件盘点并登记 docs/hardware_interface.md → Observation/Action v0.1；后续再上 Morph/动力学/PPO）
- **Last Updated:** 2026-09-09
