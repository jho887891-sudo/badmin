# P3 动态挂载实验结果（诊断 root-pose 驱动 · 2026-10-21）

**分支**：`dev/p2-wheel-contact-g1g4`（PR #20）· 预注册：`PHASE_P3_DYNAMIC_MOUNT_GATE.md`（未修改）
**场景**：真实 B601-RS（只读引用，sha256 未变）+ 修正四轮 compound 底盘，g=0、lift 0.40 m、dt=1/120、360 步
**驱动**：`write_root_pose_to_sim` 诊断驱动，**不是轮驱动**，不得作为 P4 证据（JSON 两处标注）

## 1. 结果（run `*_r09`，串行单实例，外层 gate runner 判定）

| RUN_ID | 剖面 | 命令 | 实测位移 | 实测偏航 | 安装漂移 max | 关节偏差 | gate | 外层 exit |
|---|---|---|---|---|---|---|---|---|
| P3_static_r09 | static | — | 0.0 | 0.0 | 0.0 m / 0.0 rad | 0.0 | **PASS** | 0 |
| P3_xy_r09 | translate_xy | +0.30 m x | **0.299999984 m** | 0.0 | **2.98e-08 m / 0.0 rad** | 0.0 | **PASS** | 0 |
| P3_yaw_r09 | yaw | +0.35 rad | 0.090590384 m（见 §3 说明） | **0.349999844 rad** | **6.16e-08 m / 8.69e-08 rad** | 0.0 | **PASS** | 0 |
| P3_compound_r09 | compound | +0.20/−0.15 m +0.30 rad | **0.18121759 m**（要求 ≥0.225） | 0.299999995 rad | 6.66e-08 m / 8.94e-08 rad | 0.0 | **FAIL** | 1 |

- `translate_z` 剖面（`P3_z_r09`）在本文件写作时仍在运行 → **UNKNOWN**。
- 所有 run：10 刚体 / 8 DOF、真实臂 sha256 前后一致、手臂不触地（0 N）、全部字段有限。

**结论词**：`static` **PASS**；`translate_xy` **PASS**；`yaw` **PASS**；`compound` **FAIL**（见 §3，是**真实**的门禁拦截）；`translate_z` **UNKNOWN**。

## 2. 四元数约定已被实测确认（不再是假设）

探针两种顺序都写、都测（先 `sim.forward()` 再读，避免过期位姿）：

| 候选顺序 | 实测偏航 rad | 与命令 0.2 rad 的绝对误差 |
|---|---|---|
| wxyz | 1.1e-07 | 0.1999999 |
| **xyzw** | **0.199999865** | **1.35e-07** |

→ **本 build 的 `write_root_pose_to_sim` 使用 xyzw**（与 `body_quat_w` 读取一致，与 USD `Gf.Quat` 的 wxyz 不同）。
探针结果逐 run 落盘在 `result.json.diagnostics.quat_order_for_write_root_pose`。

## 3. compound FAIL 的原因：**底盘与 articulation root 之间有 0.208 m 偏移**（真实建模事实）

- 本场景的 articulation root 是**真实臂自己的 root prim**（`/PM/Arm/...`），底盘是通过 `root_joint` 固连的
  **子刚体**；实测底盘相对 root 的偏移约 **0.208 m**。
- 因此"把 root 写到某个位姿"**不等于**"把底盘放到那个位姿"：
  - 纯偏航时，底盘绕 root 走出圆弧 → `yaw` 剖面实测到 0.0906 m 的**非命令**平移（诊断量，不影响该剖面的门禁）；
  - 复合剖面里，圆弧位移与命令平移部分抵消，底盘只到 0.1812 m（要求的 72.5%）→
    `translation_actually_measured` = false → **FAIL 是正确的**：它抓住了"命令的不是底盘"这一真实建模错误。
- **正确做法（未实现，下一步）**：命令要按 `root = chassis_cmd − R(yaw)·offset` 反解，
  或用 `write_root_pose_to_sim` 前先读 chasis 的实际世界位姿做闭环补偿。这属于 harness 修正，不是物理失败。

## 4. 速度/加加速度指标不可用（必须说明）

诊断驱动是**逐步传送 + 每步清零根速度**，所以 `peak_lin_speed / peak_ang_speed / peak_jerk` 全为 **0.0**。
这不是"运动平稳"，而是运动学传送的必然结果。**P3 的有效判据是位置/姿态追踪与相对安装刚性，不是速度/加加速度**；
要测 jerk 必须换成速度级驱动（属于 P4，且需要真实轮参数）。

## 5. 本轮修掉的三个 harness 缺陷（详见 `PHASE_P3_HARNESS_DEFECTS.md`，另加第 4 个）

1. 探针在 `sim.reset()` 之前运行，错误顺序 ≈180° 翻转泄漏进正式段（`P3_static_r01`：起点 0.216/0.145 而非 0/0）。
2. `robot.update()` 不推进物理 → 读到过期位姿，探针选错顺序（`P3_static_r03/r05` 同样翻转）。
3. 传送根位姿不清零速度、且从未下发关节目标 → "静止"剖面里关节被扯动 0.20–0.24 rad。
4. **把 root 当底盘命令**（本轮新发现）：root 与底盘不同点 → 命令位移不落到底盘上（§3）。

## 6. 证据

`outputs/simulation/rebot_b601_rs/morph_one/p2_wc_gates/P3_{static,xy,yaw,compound}_r09/`
（`authored_p3_scene.usda`、`manifest.json`、`raw_poses.jsonl`（361 帧，含每步命令位姿、10 个刚体位姿/速度、
关节 q/dq、相对安装位姿、接触）、`result.json`、`gate_runner_summary.json`、child stdout/stderr、`outer.log`）。
脚本：`scripts/simulation/p3_dynamic_mount.py`；离线坐标系契约：`p3/mount-frame-tree` @ `0bbed19`（PR #29）。
