# P3 Gate v1.1 修订预注册（在 v1.1 的 GPU 运行之前登记）

**登记时间**：2026-10-21，**在任何 v1.1 GPU 运行之前**
**基线**：P3 gate v1.0 = `PHASE_P3_DYNAMIC_MOUNT_GATE.md`（`dev/p2-wheel-contact-g1g4@5266cfc`，未修改）
**来源**：ChatGPT Draft PR #31（承诺 `edf2184d98f503d45bc91dec3c445e728aa1c723`）；
其自报"GPU 尚未验证"，本文件登记 DeepSeek 侧的移植与扩展，并明确历史 run 继续绑定 v1.0。

## 1. 为什么必须修订（两条**已确认**的 fail-open）

| # | v1.0 缺陷 | 证据 | v1.1 修法 |
|---|---|---|---|
| 1 | `translation_amplitude = hypot(spec.x, spec.y)` **完全忽略 Z** → `translate_z` 剖面根本没有平移门禁；且仅比较位移**范数**，等幅反向/换轴运动也能 PASS | PR #31 审阅 `scripts/simulation/p3_dynamic_mount.py`（v1.0 L~425）；`P3_z_r09` 就是在这种口径下判 PASS | 改用**有符号 3D 投影**：`achieved_along_command = Δp · ĉ`，要求 ≥ `0.9 × |command|`；保留旧范数为诊断字段名 `realized_translation_max_m` |
| 2 | `set_joint_targets()` 用 `except Exception: pass` **吞掉**目标下发失败 → 关节目标从未下发也可能 `joint_hold` PASS | 同上 v1.0 L~275 | 去掉静默兜底，API 异常直接 fail-closed（异常传播 → child 非 0 → 外层 exit 1） |

## 2. DeepSeek 侧的扩展（在 PR #31 之上）

- **逐帧取最大投影**（`directed_displacement_track`）：PR #31 只比较首帧与末帧，
  会在"中途到位又漂回"时误判。现对**每一帧**投影并取最大，同时记录 `best_frame`、
  `achieved_along_command_final_m`、`cross_track_m`。
- **GPU 负控制开关** `--freeze-axis {none,x,y,z,yaw}`：故意丢掉某个命令分量且写入 manifest/result 的
  `negative_control` 字段，用于在真机 GPU 上证明"缺 Z 运动必须 FAIL"这条门禁真的接线了。
- **cross-track 仅作诊断**，不在本版升级为门禁（避免在尚未取得数据前增加新判据）。

## 3. 判据变化（逐条）

| Gate | v1.0 | v1.1 |
|---|---|---|
| `translation_actually_measured` | `max‖Δp‖ ≥ 0.9·hypot(x,y)` | `max_τ(Δp_τ·ĉ)/|c| ≥ 0.9`，`c = (x,y,z)`；零命令返回不适用 |
| `yaw_actually_measured` | `max|Δyaw| ≥ 0.9|yaw|` | 不变 |
| `joint_hold` | 目标下发异常被吞 | **异常 fail-closed** |
| `mount_stable` / `no_tip_over` / `arm_does_not_touch_ground` | 不变 | 不变 |
| 阈值 0.001 m / 0.001 rad / 0.05 rad / 0.9 | 不变 | **不变**（未借修订放宽任何阈值） |

## 4. 历史数据绑定（不得改写）

- `P3_*_r09`、`P3_*_r11` 等全部历史 run 的 `result.json` **继续绑定 v1.0 口径**，不重新计算、不改写。
- v1.1 只对**新** `RUN_ID` 生效；新 run 的 `result.json.diagnostics` 带
  `directed_translation`（含 `best_frame`）与 `negative_control` 字段以示区别。
- `P3_z_r09` 在 v1.0 口径下 PASS 这件事**保留**，并在报告中标注"v1.0 口径未含 Z 有符号判据"。

## 5. v1.1 运行计划（GPU 串行）

| RUN_ID | 剖面 | freeze-axis | 期望 | 目的 |
|---|---|---|---|---|
| `P3_z_r12` | translate_z | none | PASS | 第 2 个 translate_z 独立 run（v1.1 口径） |
| `P3_xy_r12` | translate_xy | none | PASS | 第 2 个 translate_xy 独立 run |
| `P3_neg_frozen_z_r01` | translate_z | **z** | **FAIL（外层 exit 1）** | 负控制：缺 Z 运动必须被拦截 |
| `P3_neg_frozen_xy_r01` | translate_xy | **x** | **FAIL（外层 exit 1）** | 负控制：缺 X 运动必须被拦截 |

全部 360 步、单实例串行、独立 run 目录、真实 child/outer exit、外层 `p2_gate_runner.py` 判定。

## 6. 未改变的口径

P3 的驱动仍是 `write_root_pose_to_sim` 诊断式写入，每步清零根速度 → 速度/jerk 恒为 0，
**不得**作为真实动力学或 jerk 证据；P3 PASS **不等于** P4。
