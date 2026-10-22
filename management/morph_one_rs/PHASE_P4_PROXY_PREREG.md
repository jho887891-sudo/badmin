# P4_PROXY 预注册（参数化四轮驱动软件验证 · 在运行之前登记）

**登记时间**：2026-10-21，**在任何 P4_PROXY GPU 运行之前**
**定级**：本文件与 `scripts/simulation/p4_proxy_wheel_drive.py` 只覆盖 **P4_PROXY**。
`P4_REAL` 仍然 **BLOCKED**（真实装载滚动半径、接触斑、质量/惯量/摩擦、胎面材料未验收）。
**禁止**把本文件任何结果写成"真实轮径/厂商物理模型 PASS"。

## 1. 参数来源（逐项标注）

| 参数 | 值 | 来源 | 级别 |
|---|---|---|---|
| 四轮轴心（chassis 系） | (±0.250002, ±0.297, −0.244) m | P1 v2（PR #33），残差 0.002 mm | **verified** |
| 自转轴 | world X | P1 v2，PCA 0.0043° | **verified** |
| 左右 spin 符号 | +1 / +1 / −1 / −1 | P1 v2 外法向约定 | **verified（方向）** |
| 滚动半径 | **0.103 m**（环面赤道 R+r = 103.000000 mm） | P1 v2，残差 0.000000 mm | **geometric unloaded（非装载）** |
| 碰撞代理外形 | 圆柱（半径 0.103、半宽 0.0243） | 本文件 | **TEST_PROXY**（环面→圆柱简化） |
| 轮质量 / 底盘质量 | 1.0 / 16.0 kg（合计 20 kg） | 本文件 | **TEST_PROXY（无厂商质量数据）** |
| 地面初始间隙 | 0.02 m（lift = 0.347 + 0.02） | 本文件 | TEST_PROXY |
| 驱动方式 | `set_joint_velocity_target`（仅轮关节） | 本文件 | — |

## 2. 硬约束（运行时自证）

- **绝不写 root pose/velocity**：脚本内 `ROOT_WRITES` 计数器恒为 0，并作为门禁 `no_root_pose_write`；
  底盘只能因轮地接触而移动。任何"直接写底盘位姿制造位移"的做法都属于作弊，禁止。
- 轮体是显式刚体 + 绕 X 的 `RevoluteJoint`（不能是 chassis 的子 Xform，否则无法驱动）。
- 每个 run 独立目录、单一 Isaac 实例、串行、外层 `p2_gate_runner.py` 判定。

## 3. 预注册门禁与阈值（冻结）

| Gate | 判据 | 说明 |
|---|---|---|
| `all_finite` | 全部遥测有限 | 健全性 |
| `no_root_pose_write` | `root_writes == 0` | 禁止伪装驱动 |
| `wheel_speed_tracks_target` | 末 60 步平均轮速相对误差 **≤ 0.20**（仅对有非零目标的轮） | 速度级驱动是否真的建立 |
| `contact_detected` | 轮/地接触力峰值 **> 1.0 N** | 与 G4/G6 同判据 |
| `chassis_displaced_along_command` | forward: `Δx ≥ +0.01 m`；backward: `Δx ≤ −0.01 m` | 位移必须由接触产生且方向正确 |
| `yaw_changed` | 差速剖面 |Δyaw| **≥ 0.05 rad** | 转向可行性 |

**诊断量（不作门禁）**：`distance_per_wheel_revolution_m`（滚动半径一致性）、`tail_chassis_speed_m_s`、
分轮 omegas、倾斜、滑移。阈值不得依据实测放宽；若实测显示阈值不合理，必须新增 v1.1 预注册而不是改本文件。

## 4. 运行计划（GPU 串行）

| RUN_ID | profile | 目标轮速 rad/s | 步数 | 期望 |
|---|---|---|---|---|
| `P4p_spin1_r01` | spin_single | (2, 0, 0, 0) | 240 | 只转一轮；轮速跟踪；底盘**不应**明显平移 |
| `P4p_fwd_r01` | forward | (4, 4, 4, 4) | 360 | +x 位移 ≥0.01 m |
| `P4p_back_r01` | backward | (−4, −4, −4, −4) | 360 | −x 位移 ≤−0.01 m |
| `P4p_turn_r01` | turn | (−3,−3,3,3) | 360 | |Δyaw| ≥ 0.05 rad |

前两个 profile 各补第二次独立 run（`_r02`）以满足"≥2 独立 run"。

## 5. 与 P4_REAL 的关系

P4_PROXY 通过**只**验证"轮关节→电机通道→速度目标→接触→底盘位移"这条软件/符号链路的正确性，
包括左右符号、轴心位置、半径参数化是否被算法正确消费。
它**不能**证明真实轮胎的滚动半径、滑移率、摩擦或承载能力——这些仍然 UNKNOWN，需要 P1 的装载测量或厂家数据。
