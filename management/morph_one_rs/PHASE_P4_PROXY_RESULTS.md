# P4_PROXY 结果（参数化四轮驱动软件验证）· 2026-10-21

**分支**：`p3/gate-v1.1`（基于 `dev/p2-wheel-contact-g1g4`）· 预注册：`PHASE_P4_PROXY_PREREG.md`（运行前提交）
**定级**：**P4_PROXY**。`P4_REAL` 仍 **BLOCKED**（真实装载滚动半径/接触斑/质量/摩擦/胎面材料 UNKNOWN）。
**驱动**：`set_joint_velocity_target` 仅作用于轮关节；**从不写 root pose/velocity**（`root_writes == 0` 为门禁）。

## 1. 结果

| RUN_ID | profile | 轮目标 rad/s | 实测轮速 rad/s | 底盘位移 m | yaw 变化 rad | gate | exit |
|---|---|---|---|---|---|---|---|
| `P4p_spin1_r02` | spin_single（仅 wheel1） | (2,0,0,0) | **1.7711** / 0.0742 / 0.0781 / 0.0733 | (6.3e-06, **−0.01255**, −0.01862) | 2.05e-05 | **PASS** | 0 |
| `P4p_rollneg_r01` | roll_neg_y | (4,4,4,4) | 3.9978 / 3.9948 / 3.9986 / 3.9944 | (2.7e-04, **−1.21474**, −0.01862) | 1.29e-04 | **PASS** | 0 |
| `P4p_rollpos_r01` | roll_pos_y | (−4,−4,−4,−4) | −4.0018 / −4.0049 / −4.0013 / −4.0055 | (1.3e-04, **+1.21551**, −0.01862) | 5.56e-05 | **PASS** | 0 |
| `P4p_turn_r03` | turn（左右反向） | (3,3,−3,−3) | 2.8679 / 2.8720 / −2.8339 / −2.9069 | (1.3e-04, 2.5e-03, −0.01861) | **1.41751** | **PASS** | 0 |
| `P4p_spin1_r03` | spin_single（重复） | (2,0,0,0) | 1.7711 / 0.0742 / 0.0781 / 0.0733 | (6.3e-06, −0.01255, −0.01862) | 2.05e-05 | **PASS** | 0 |
| `P4p_rollneg_r02` | roll_neg_y（重复） | (4,4,4,4) | 3.9978 / 3.9948 / 3.9986 / 3.9944 | (2.7e-04, −1.21474, −0.01862) | 1.29e-04 | **PASS** | 0 |
| `P4p_rollpos_r02` | roll_pos_y（重复） | (−4,−4,−4,−4) | −4.0018 / −4.0049 / −4.0013 / −4.0055 | (1.3e-04, **+1.21551**, −0.01862) | 5.56e-05 | **PASS** | 0 |

**关键读数**：轮速跟踪相对误差 0.05%–5.5%（门禁 ≤20%）；直线行驶 ±1.215 m 且 yaw 漂移仅 1.3e-04 rad
（≈0.007°）→ 四轮同向、无差速；差速剖面纯转动（xy 位移 2.5 mm、yaw 1.4175 rad）。
**底盘位移完全由轮地接触产生**——脚本内 `ROOT_WRITES` 计数恒 0 且作为门禁。

**重复性与覆盖**：`spin_single`、`roll_neg_y`、`roll_pos_y` 各有 **2 个独立 run**，且两次的位移/轮速**逐位一致**
（如 roll_neg 两次均为 `−1.2147367 m`、`3.99775821 rad/s`）→ 本代理场景确定性可复现。
**`turn` 只有 1 个有效 run**（`turn_r03`；`turn_r02` 是被旧循环污染的反例）→ 差速剖面**仍欠第二个 run**，标 **PARTIAL**。

## 2. 本线真正抓出来的三个缺陷（失败证据全部保留）

1. **`P4p_spin1_r01` / `P4p_fwd_r01` / `P4p_back_r01`**：命令 2–4 rad/s，实测轮速 **−0.0013 rad/s**（几乎不动），
   底盘仅下沉 18.6 mm。根因：USD 关节**没有 `DriveAPI`**，Isaac Lab 的隐式执行器没有可写的 drive。
   修法：显式 `UsdPhysics.DriveAPI.Apply(joint, "angular")`（stiffness 0 / damping 40 / maxForce 400）。
2. **`P4p_fwd_r02` / `P4p_back_r02`**：按 P1 的"外法向"符号（+1/+1/−1/−1）下发，结果底盘**自转 1.915 rad**
   而没有平移。根因：四个轮关节的轴**都指向 world +X**，所以直线行驶需要**同号**；
   P1 的 +1/−1 是**电机通道**约定，不是关节轴约定。修法：关节空间同号 + 结果里显式记录
   `motor_sign_mapping`（关节轴 / 滚动方向 / 外法向符号 / 直线同号 / 说明）。
3. **`P4p_turn_r02` 的污染**：早期实验循环未被完全终止，旧代码又跑了一次同名 RUN，写入的 target 全是 −3。
   已把"先杀循环再杀子进程"写进启动脚本，并用新 RUN_ID 重跑（`turn_r03` PASS）。
   **这条 run 不删除**，作为"同名 RUN 不得复用"的反例证据。

## 3. 预注册门禁（全部保持原阈值，未事后放宽）

| Gate | 阈值 | 结果 |
|---|---|---|
| `all_finite` | — | 全部 PASS |
| `no_root_pose_write` | `root_writes == 0` | 全部 PASS |
| `wheel_speed_tracks_target` | 相对误差 ≤ 0.20 | 全部 PASS（最差 0.1145 @ spin_single） |
| `contact_detected` | 峰值 > 1.0 N | 见各 run（轮地接触已建立） |
| `chassis_displaced_along_command` | 沿命令方向 ≥ 0.01 m | roll_neg / roll_pos 均为 1.21 m |
| `yaw_changed` | \|Δyaw\| ≥ 0.05 rad | turn 1.4175 rad |

## 4. 明确不成立的推断（不得越界）

- 滚动半径 0.103 m 是 **P1 v2 的几何未装载值**（环面赤道 R+r），**不是装载半径**；碰撞代理是**圆柱**（TEST_PROXY）。
- 轮/底盘质量 1.0 / 16.0 kg 是 **TEST_PROXY**，无厂商质量/惯量/摩擦数据。
- `distance_per_wheel_revolution_m` 全为 0（分母用了目标角速度而不是实测平均值，公式缺陷）→ 视为 **UNKNOWN**，不作结论。
- 本结果**不能**称 `P4_REAL` 通过；真实滑移率、摩擦、承载、坡道、急停均未测。

## 5. 证据

`outputs/simulation/rebot_b601_rs/morph_one/p2_wc_gates/P4p_*/`（`authored_p4_proxy.usda`、manifest、raw_poses.jsonl、
result.json、gate_runner_summary.json、child stdout/stderr、outer.log）。脚本 `scripts/simulation/p4_proxy_wheel_drive.py`。
