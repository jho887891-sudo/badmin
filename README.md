# 低速羽毛球人机对打机器人（badminton_robot）

> 移动全向底盘 + 6 轴机械臂 + 双目视觉的低速羽毛球人机对打机器人 —— 算法研发项目。
> **本地目录 = 项目事实来源（Single Source of Truth）；远端 GPU 机 = 计算执行环境。**
> 重要研发信息一律落盘，不得只保存在聊天上下文中。

## 30 秒速览（先读这些，再动手）

| 想了解 | 文件 |
|---|---|
| 项目现在在哪 | `PROJECT_STATUS.md` |
| 下一步做什么 / 优先级 | `TODO.md`、`ROADMAP.md` |
| 为什么这么设计 | `management/DECISIONS.md` |
| 踩过什么坑 | `management/ISSUES.md` |
| 每天做了什么 | `management/DAILY_LOG.md` |
| 训练实验登记 | `experiments/EXPERIMENT_INDEX.md` |

## 目录结构

```
badminton_robot/            ← 本工作区根目录
├── README.md / PROJECT_STATUS.md / TODO.md / CHANGELOG.md / ROADMAP.md
├── docs/                   ← 技术设计文档（架构/算法/坐标系/Obs-Action/Reward/sim2real/部署/硬件接口）
├── management/             ← 项目管理（每日记录/决策/问题/风险/里程碑）
├── experiments/            ← 实验索引 + runs/ 详细报告
├── configs/ scripts/ src/ tests/ assets/ tools/
```

## 核心工作原则

1. 开始新任务前先读状态文件，禁止在不了解现状时大规模重写。
2. 小步修改：先说清改哪个文件、为什么、影响范围 → 改完跑最小验证。
3. Observation / Action / Reward 属于项目 API：改代码必须同步 `docs/observation_action.md` 与 `docs/reward_design.md`，禁止只改代码不改文档。
4. RL 采用 Curriculum Learning（Stage 0→9，见 `ROADMAP.md`）；某阶段不收敛时优先查 Observation / Action scaling / Reward / Reset / 接触物理 / 步长 / 控制频率 / Normalization / Curriculum / DR，不要盲目加网络复杂度。
5. 环境事实禁止猜测：CUDA 版本、Isaac Sim/Lab 版本、GPU 型号、URDF 路径、ROS topic、CAN ID、机械臂/底盘接口 —— 一律先跑检查命令或读文件，数据优先。
6. 实验编号 `EXP-YYYYMMDD-XXX`，每次 RL 训练按规范登记（见 `experiments/EXPERIMENT_INDEX.md`）。
7. Git：main / dev / feature/*，commit 前缀 `feat: fix: refactor: docs: test: exp: chore:`；不得为整洁删除开发历史。
8. 真机安全：未经低速验证的 policy 禁止高速运行 —— 低速模式、限制 action / joint velocity / torque、workspace、急停可用、清场、Safety Shield 开启。

## 技术栈（待环境盘点后填写，禁止猜测）

Isaac Sim / Isaac Lab / PyTorch / CUDA / ROS2 / TensorRT-ONNX（版本待登记至 `docs/deployment.md`）

## 主要算法方向

双目视觉 3D 定位 → EKF / 轨迹预测 → 击球点预测 → 挥拍策略 → 底盘+机械臂协同（Whole-Body Control）→ Safety Shield；全程贯穿 Domain Randomization 与 Sim-to-Real。
