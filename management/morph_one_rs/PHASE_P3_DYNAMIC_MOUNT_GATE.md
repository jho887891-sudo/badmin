# P3 动态挂载验收门禁（预注册 · v1.0）

**提交时间**：2026-10-21，**在任何 P3 运行之前**
**依赖**：P2 静态/落地/碰撞门禁（G0–G7、G7B）。P3 使用**真实 B601-RS + 修正四轮 compound** 场景。
**驱动性质**：P3 的运动来自 **root pose 诊断驱动**（`write_root_pose_to_sim`），**不是**轮驱动，
禁止当作 P4 证据（JSON 里 `drive_mode` 与 `diagnostics.note` 双处标注）。

## 1. 运动剖面（每个剖面 ≥2 个独立 run_id，串行单实例，dt=1/120，g=0）

| profile | 位移 | 偏航 | 斜坡 | 步数 | 目的 |
|---|---|---|---|---|---|
| `static` | 0 | 0 | 1.0 s | 360 | 参考：不动时安装残差基线 |
| `translate_xy` | +0.30 m x | 0 | 1.5 s | 360 | P3-02 纯平移 |
| `translate_z` | +0.10 m z | 0 | 1.0 s | 360 | P3-02 纯 Z 位移 |
| `yaw` | 0 | +0.35 rad | 1.5 s | 360 | P3-02 纯非零偏航 |
| `compound` | +0.20/−0.15 m | +0.30 rad | 2.0 s | 360 | P3-03 复合运动 |

斜坡为余弦（两端速度为零），**实测**位移/偏航必须来自运行时 `body_pos_w/body_quat_w`，不读命令值。

## 2. 预注册门禁（冻结，不得事后放宽）

| Gate | 判据 | 依据 |
|---|---|---|
| `all_finite` | 全部遥测有限 | 健全性 |
| `mount_stable` | 相对安装位姿相对 step0 的**最大**偏差 ≤ **0.001 m** 且 ≤ **0.001 rad** | 与 P2 零重力静置同量级；运动中不放松 |
| `joint_hold` | 8 DOF 关节相对初始目标最大偏差 ≤ **0.05 rad** | 隐式驱动 300/20，固定 target，允许 2.9° 跟随误差 |
| `translation_actually_measured` | 实测底盘位移 max ≥ **0.9 ×** 命令平移幅值 | 防"只有命令没有运动" |
| `yaw_actually_measured` | 实测偏航 max ≥ **0.9 ×** 命令偏航 | 同上 |
| `arm_does_not_touch_ground` | 手臂子树接触力峰值 ≤ **1.0 N**（可测时） | 复用 G4/G6 的 1 N 判据 |

`static` 剖面没有平移/偏航门禁项（幅值为 0，不适用）。命令跟踪误差（`tracking_*_error_max`）
只作为**诊断量**记录，不作为门禁（它受 root pose 写入时机与求解器延迟影响）。

## 3. 判词

全部 gate 为真 → 该 run **PASS**（仅代表"该剖面下安装刚性保持 + 运动确实发生"）。
运动发生但安装残差超限 → **PARTIAL**（记录具体超限剖面）。不动或未测 → **FAIL / UNKNOWN**。

## 4. 写 root pose 的四元数约定（必须自证）

单位四元数无法区分 wxyz/xyzw。脚本先写一个已知 **0.2 rad** 偏航，读回底盘实际偏航：
若 `|yaw-0.2| > 0.01` 则切换到 xyzw 并记录 `wxyz_probe_yaw_rad`；实际使用顺序写入
`result.json.diagnostics.quat_order_for_write_root_pose`。禁止凭记忆假定顺序。

## 5. 实现与证据

`scripts/simulation/p3_dynamic_mount.py`；每个 run 目录含 `authored_p3_scene.usda`、`manifest.json`、
`raw_poses.jsonl`（逐步：命令位姿、全部刚体位姿/速度、关节 q/dq、相对安装位姿、接触）、`result.json`、
`gate_runner_summary.json`、child stdout/stderr；真实臂资产 sha256 前后比对，只读。
