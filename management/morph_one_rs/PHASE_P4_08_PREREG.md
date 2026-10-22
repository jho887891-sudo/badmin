# P4-08 预注册：轮地摩擦 / 载荷 / 轮速单因素矩阵（TEST_PROXY）

**登记时间**：2026-10-22，**在任何 P4-08 GPU 运行之前**
**基线**：PR #35 `p3/gate-v1.1@5b74eb1` 的 P4_PROXY 轮关节驱动（`set_joint_velocity_target`，**零 root 写入**）
**定级**：**P4_PROXY**。`P4_REAL` 仍 BLOCKED（无厂家受载滚动半径、接触材料、质量/惯量/摩擦数据）。

## 1. 冻结的仪器审计（每次 run 写入 manifest/result）

| 项 | 值 | 来源 |
|---|---|---|
| 轮轴心（chassis 系） | (±0.250002, ±0.297, −0.244) m | P1 v2 verified（残差 0.002 mm） |
| 自转轴 | world +X；滚动方向 world Y | P1 v2 verified |
| 几何滚动半径 | 0.103 m（环面赤道 R+r，**未装载**） | P1 v2 verified |
| 碰撞代理 | 圆柱 r=0.103、半宽 0.0243 | **TEST_PROXY** |
| 质量 | chassis 16 kg + 4×1 kg 轮 = 20 kg | **TEST_PROXY** |
| 重力 | −9.81 m/s² | — |
| dt / solver | 1/120 s、TGS（solver_type=1） | 与 P2/P3 一致 |
| 初始间隙 | 0.02 m（lift = 0.347+0.02） | TEST_PROXY |
| 地面材质 | `RigidBodyMaterialCfg(static=dynamic=--friction, restitution=0)` | 单因素旋钮 |
| 驱动 | `set_joint_velocity_target`，仅轮关节；`root_writes == 0` 是门禁 | — |

## 2. 单因素矩阵（每个 run 只变一项；每组 ≥2 独立 RUN_ID）

| 组 | RUN_ID 模板 | friction | payload | 轮速 rad/s | profile |
|---|---|---|---|---|---|
| A 摩擦 低 | `P4f_low_rNN` | **0.2** | 0 | 4（四轮同向） | roll_neg_y |
| A 摩擦 基准 | `P4f_base_rNN` | **0.8** | 0 | 4 | roll_neg_y |
| A 摩擦 高 | `P4f_high_rNN` | **1.4** | 0 | 4 | roll_neg_y |
| B 载荷 基准 | `P4L_0_rNN` | 0.8 | 0 kg | 4 | roll_neg_y |
| B 载荷 +10 kg | `P4L_10_rNN` | 0.8 | **+10 kg**（0.30×0.25×0.15 m，原点上方 0.15 m） | 4 | roll_neg_y |
| C 轮速 1 | `P4S_1_rNN` | 0.8 | 0 | **1** | roll_neg_y |
| C 轮速 4 | `P4S_4_rNN` | 0.8 | 0 | **4** | roll_neg_y |
| D 无驱动负控制 | `P4N_still_rNN` | 0.8 | 0 | **0** | hold_still |
| E 反向控制 | `P4R_pos_rNN` | 0.8 | 0 | **−4** | roll_pos_y |

摩擦系数 0.2/0.8/1.4 是**人为选定的 TEST_PROXY 档位**（低/中/高的干燥地面量级），**不是**厂家轮胎数据。

## 3. 预注册门禁（冻结，不得事后放宽）

| Gate | 判据 |
|---|---|
| `all_finite` | 全部遥测有限 |
| `no_root_pose_write` | `root_writes == 0` |
| `wheel_speed_tracks_target` | 末 60 步平均轮速相对误差 ≤ **0.20** |
| `contact_detected` | 轮/地接触力峰值 > **1.0 N** |
| `chassis_displaced_along_command` | 沿命令方向位移 ≥ **0.01 m**（反向剖面符号相反） |
| `no_motion_without_drive` | hold_still：`‖Δxy‖ ≤ 1e-3 m` 且 `|Δyaw| ≤ 1e-3 rad` |
| `measured_rolling_radius_consistent` | 直线剖面：`|r_eff − 0.103|/0.103 ≤ 0.10`，其中 `r_eff = |Δs_measured| / |Δθ_measured|` |
| `yaw_changed` | turn：\|Δyaw\| ≥ 0.05 rad |

**新的滚动半径口径**：`r_eff = |Δ底盘Y| / |Δ关节角实测(rad)|`（不再用命令转速做分母；旧字段标记 deprecated/UNKNOWN）。
它**包含打滑**，因此是"有效滚动半径"，**不是轮胎半径**；只在直线剖面且 |Δθ|>1e-3 时给出。

## 4. 通过/不通过的物理含义（预注册，避免事后解释）

- 摩擦 0.2 若出现 `measured_rolling_radius_consistent` FAIL（r_eff 偏大）→ 说明**打滑**，符合低摩擦预期；
  若反而 PASS 且位移不变，说明当前质量/驱动下摩擦未进入打滑区，必须在报告里如实写出"本矩阵未能区分摩擦"，不得宣称"摩擦无关"。
- 载荷 +10 kg（30 kg 总重）若位移/time 曲线显著变化，只证明**敏感性趋势**；
  **不得**与 G7B 的 26.0085 kg 真实臂场景对比成同一结论（配置不同）。
- 轮速 1 vs 4 rad/s 的 `r_eff` 若差异 > 5%，说明 `r_eff` 依赖速度（打滑随速度变化），必须记录为速度相关量。

## 5. 失败保护

- 任一 run 若 `no_root_pose_write` 为假 → 该 run 作废（说明 harness 用 root 写入冒充驱动）。
- 出现 OOM / Kit 启动超时 / 宿主被他人占用：记录 `uptime`、`nvidia-smi`、Kit 扩展日志与进程 RSS，
  切换 CPU READY 工作，**不占用他人 CPU/GPU 进程**。
- 负面结果一律保留，不删除、不重跑覆盖。

## 6. 覆盖与不足（预先声明）

每个配置 **2 个独立 RUN_ID**；矩阵不包含坡道、越障、单轮悬空、温度、轮胎磨损、真实胎压——
这些属于 `P4_REAL`/后续任务，本文件不作任何暗示。
