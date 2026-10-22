<!-- 本文件由 scripts/simulation/p3_summary.py 自动生成，请勿手改 -->

# P3 证据汇总（自动生成，非手抄）

重新生成：`python scripts/simulation/p3_summary.py`（同时写出 `p3_summary.json` 与本表）

## 0. 结论边界（必读）

> P3 drive is a DIAGNOSTIC root-pose teleport with the root velocity zeroed every step: peak_lin_speed_m_s / peak_ang_speed_rad_s / peak_jerk_m_s3 are zero (or teleport artefacts) BY CONSTRUCTION and are NOT dynamics or stability evidence. P3 PASS is never P4 wheel-drive evidence.

## 1. 每一 run 的证据（按 gate 代际分组，绝不混算）

### v1.0（11 个 run）

| run_id | profile | gate 模式 | gate_result | outer_exit | child_rc | 失败门禁 | 逐门禁 |
|---|---|---|---|---|---|---|---|
| P3_compound_r09 | compound | positive | FAIL | 1 | 0 | translation_actually_measured | all_finite=T, arm_does_not_touch_ground=T, joint_hold=T, mount_stable=T, translation_actually_measured=F, yaw_actually_measured=T |
| P3_compound_r11 | compound | positive | PASS | 0 | 0 | - | all_finite=T, arm_does_not_touch_ground=T, joint_hold=T, mount_stable=T, translation_actually_measured=T, yaw_actually_measured=T |
| P3_static_r01 | static | positive | FAIL | 1 | 0 | joint_hold | all_finite=T, arm_does_not_touch_ground=T, joint_hold=F, mount_stable=T |
| P3_static_r09 | static | positive | PASS | 0 | 0 | - | all_finite=T, arm_does_not_touch_ground=T, joint_hold=T, mount_stable=T |
| P3_static_r11 | static | positive | PASS | 0 | 0 | - | all_finite=T, arm_does_not_touch_ground=T, joint_hold=T, mount_stable=T |
| P3_xy_r09 | translate_xy | positive | PASS | 0 | 0 | - | all_finite=T, arm_does_not_touch_ground=T, joint_hold=T, mount_stable=T, translation_actually_measured=T |
| P3_z_r09 | translate_z | positive | PASS | 0 | 0 | - | all_finite=T, arm_does_not_touch_ground=T, joint_hold=T, mount_stable=T |
| P3_yaw_r09 | yaw | positive | PASS | 0 | 0 | - | all_finite=T, arm_does_not_touch_ground=T, joint_hold=T, mount_stable=T, yaw_actually_measured=T |
| P3_yaw_r11 | yaw | positive | PASS | 0 | 0 | - | all_finite=T, arm_does_not_touch_ground=T, joint_hold=T, mount_stable=T, yaw_actually_measured=T |
| P3_yaw_r13 | yaw | positive | PASS | 0 | 0 | - | all_finite=T, arm_does_not_touch_ground=T, joint_hold=T, mount_stable=T, yaw_actually_measured=T |
| P3_yaw_r14 | yaw | positive | PASS | 0 | 0 | - | all_finite=T, arm_does_not_touch_ground=T, joint_hold=T, mount_stable=T, yaw_actually_measured=T |

### v1.1（16 个 run）

| run_id | profile | gate 模式 | gate_result | outer_exit | child_rc | 失败门禁 | 逐门禁 |
|---|---|---|---|---|---|---|---|
| P3_compound_r13 | compound | positive | PASS | 0 | 0 | - | all_finite=T, arm_does_not_touch_ground=T, joint_hold=T, mount_stable=T, translation_actually_measured=T, yaw_actually_measured=T |
| P3_compound_r14 | compound | positive | PASS | 0 | 0 | - | all_finite=T, arm_does_not_touch_ground=T, joint_hold=T, mount_stable=T, translation_actually_measured=T, yaw_actually_measured=T |
| P3_static_r15 | static | positive | PASS | 0 | 0 | - | all_finite=T, arm_does_not_touch_ground=T, joint_hold=T, mount_stable=T |
| P3_static_r16 | static | positive | PASS | 0 | 0 | - | all_finite=T, arm_does_not_touch_ground=T, joint_hold=T, mount_stable=T |
| P3_neg_frozen_x_r03 | translate_xy | negative_control(freeze_axis=x) | FAIL | 1 | 0 | translation_actually_measured | all_finite=T, arm_does_not_touch_ground=T, joint_hold=T, mount_stable=T, translation_actually_measured=F |
| P3_neg_frozen_xy_r02 | translate_xy | negative_control(freeze_axis=x) | FAIL | 1 | 0 | translation_actually_measured | all_finite=T, arm_does_not_touch_ground=T, joint_hold=T, mount_stable=T, translation_actually_measured=F |
| P3_neg_frozen_y_r01 | translate_xy | negative_control(freeze_axis=y) | PASS | 0 | 0 | - | all_finite=T, arm_does_not_touch_ground=T, joint_hold=T, mount_stable=T, translation_actually_measured=T |
| P3_neg_frozen_y_r02 | translate_xy | negative_control(freeze_axis=y) | FAIL | 1 | 0 | negative_control_is_meaningful | all_finite=T, arm_does_not_touch_ground=T, joint_hold=T, mount_stable=T, negative_control_is_meaningful=F, translation_actually_measured=T |
| P3_xy_r13 | translate_xy | positive | PASS | 0 | 0 | - | all_finite=T, arm_does_not_touch_ground=T, joint_hold=T, mount_stable=T, translation_actually_measured=T |
| P3_xy_r14 | translate_xy | positive | PASS | 0 | 0 | - | all_finite=T, arm_does_not_touch_ground=T, joint_hold=T, mount_stable=T, translation_actually_measured=T |
| P3_neg_frozen_z_r02 | translate_z | negative_control(freeze_axis=z) | FAIL | 1 | 0 | translation_actually_measured | all_finite=T, arm_does_not_touch_ground=T, joint_hold=T, mount_stable=T, translation_actually_measured=F |
| P3_z_r13 | translate_z | positive | PASS | 0 | 0 | - | all_finite=T, arm_does_not_touch_ground=T, joint_hold=T, mount_stable=T, translation_actually_measured=T |
| P3_z_r14 | translate_z | positive | PASS | 0 | 0 | - | all_finite=T, arm_does_not_touch_ground=T, joint_hold=T, mount_stable=T, translation_actually_measured=T |
| P3_neg_frozen_yaw_r01 | yaw | negative_control(freeze_axis=yaw) | FAIL | 1 | 0 | yaw_actually_measured | all_finite=T, arm_does_not_touch_ground=T, joint_hold=T, mount_stable=T, yaw_actually_measured=F |
| P3_yaw_r15 | yaw | positive | PASS | 0 | 0 | - | all_finite=T, arm_does_not_touch_ground=T, joint_hold=T, mount_stable=T, yaw_actually_measured=T |
| P3_yaw_r16 | yaw | positive | PASS | 0 | 0 | - | all_finite=T, arm_does_not_touch_ground=T, joint_hold=T, mount_stable=T, yaw_actually_measured=T |

v1.1 有向位移（directed_translation，v1.1 判据）：

| run_id | profile | 命令位移 (m) | achieved_ratio | final_ratio | best_frame | cross_track (m) | 通过 |
|---|---|---|---|---|---|---|---|
| P3_compound_r13 | compound | 0.254951 | 0.997424 | 0.997424 | 240 | 0.00269164 | PASS |
| P3_compound_r14 | compound | 0.254951 | 0.997424 | 0.997424 | 240 | 0.00269164 | PASS |
| P3_neg_frozen_x_r03 | translate_xy | 0.3 | 0.000776468 | 0.00044665 | 303 | 0.000112786 | FAIL |
| P3_neg_frozen_xy_r02 | translate_xy | 0.3 | 2.42505e-05 | 2.39985e-05 | 78 | 4.5273e-05 | FAIL |
| P3_neg_frozen_y_r01 | translate_xy | 0.3 | 1.00103 | 1.00078 | 180 | 0.000124074 | PASS |
| P3_neg_frozen_y_r02 | translate_xy | 0.3 | 1.00103 | 1.00078 | 180 | 0.000124074 | PASS |
| P3_xy_r13 | translate_xy | 0.3 | 1.00144 | 1.00127 | 180 | 0.000136932 | PASS |
| P3_xy_r14 | translate_xy | 0.3 | 1.00103 | 1.00078 | 180 | 0.000124074 | PASS |
| P3_neg_frozen_z_r02 | translate_z | 0.1 | 0.000336468 | 0.00033617 | 54 | 3.11668e-05 | FAIL |
| P3_z_r13 | translate_z | 0.1 | 0.975779 | 0.975779 | 120 | 0.000187002 | PASS |
| P3_z_r14 | translate_z | 0.1 | 0.972361 | 0.972361 | 120 | 0.000192566 | PASS |

## 2. 分代际聚合（不跨代际平均）

| 代际 | run 数 | PASS | FAIL | 有向判据 run 数 | 最小 achieved_ratio | 安装位偏差 max (m) | 安装姿态偏差 max (rad) | jerk max (m/s³) |
|---|---|---|---|---|---|---|---|---|
| v1.0 | 11 | 9 | 2 | 0 | - | 6.6639e-08 | 9.4243e-08 | 6209.07 |
| v1.1 | 16 | 11 | 5 | 11 | 2.42505e-05 | 3.0825e-08 | 9.1857e-08 | 904.133 |

## 3. 覆盖审计（每个剖面每个代际需要 2 个独立 PASS）

| 代际 | profile | 独立 PASS | 要求 | 缺口 | 看到的 run |
|---|---|---|---|---|---|
| v1.0 | compound | 1 | 2 | **缺口** | P3_compound_r09, P3_compound_r11 |
| v1.0 | static | 2 | 2 | - | P3_static_r01, P3_static_r09, P3_static_r11 |
| v1.0 | translate_xy | 1 | 2 | **缺口** | P3_xy_r09 |
| v1.0 | translate_z | 1 | 2 | **缺口** | P3_z_r09 |
| v1.0 | yaw | 4 | 2 | - | P3_yaw_r09, P3_yaw_r11, P3_yaw_r13, P3_yaw_r14 |
| v1.1 | compound | 2 | 2 | - | P3_compound_r13, P3_compound_r14 |
| v1.1 | static | 2 | 2 | - | P3_static_r15, P3_static_r16 |
| v1.1 | translate_xy | 2 | 2 | - | P3_xy_r13, P3_xy_r14 |
| v1.1 | translate_z | 2 | 2 | - | P3_z_r13, P3_z_r14 |
| v1.1 | yaw | 2 | 2 | - | P3_yaw_r15, P3_yaw_r16 |

### 覆盖缺口清单

- v1.0 / compound：独立 PASS 1 < 2（PASS: P3_compound_r11；run: P3_compound_r09, P3_compound_r11）
- v1.0 / translate_xy：独立 PASS 1 < 2（PASS: P3_xy_r09；run: P3_xy_r09）
- v1.0 / translate_z：独立 PASS 1 < 2（PASS: P3_z_r09；run: P3_z_r09）

## 4. 负控制审计（缺运动必须 FAIL）

| run_id | profile | frozen_axis | gate_result | outer_exit | 假的门禁 | achieved_ratio | 判定 |
|---|---|---|---|---|---|---|---|
| P3_neg_frozen_x_r03 | translate_xy | x | FAIL | 1 | translation_actually_measured | 0.000776468 | 通过（按设计失败） |
| P3_neg_frozen_xy_r02 | translate_xy | x | FAIL | 1 | translation_actually_measured | 2.42505e-05 | 通过（按设计失败） |
| P3_neg_frozen_y_r01 | translate_xy | y | PASS | 0 | - | 1.00103 | **失败：负控制竟然 PASS** |
| P3_neg_frozen_y_r02 | translate_xy | y | FAIL | 1 | negative_control_is_meaningful | 1.00103 | **失败：负控制竟然 PASS** |
| P3_neg_frozen_z_r02 | translate_z | z | FAIL | 1 | translation_actually_measured | 0.000336468 | 通过（按设计失败） |
| P3_neg_frozen_yaw_r01 | yaw | yaw | FAIL | 1 | yaw_actually_measured | - | 通过（按设计失败） |

## 5. 诊断量：root<->chassis 偏置与安装漂移

- `diagnostics.root_to_chassis_offset_m` 取值：[0.261476607]（跨 run 极差 0）
- 带该诊断字段的 run：21 / 27；声明做偏置补偿的 run：21
- 安装位偏差 max：6.6639e-08 m；安装姿态偏差 max：9.4243e-08 rad（门限 0.001 m / 0.001 rad）
- `raw_poses.jsonl` 帧数观测：[361]，steps+1 期望：[361]
- v1.0 安装漂移：6.6639e-08 m / 9.4243e-08 rad
- v1.1 安装漂移：3.0825e-08 m / 9.1857e-08 rad

## 6. 自动告警与证据完整性

- 无错误：每个 run 都有 `result.json` 与 `gate_runner_summary.json`，每个负控制都按设计失败
- 告警：P3 drive is a DIAGNOSTIC root-pose teleport with the root velocity zeroed every step: peak_lin_speed_m_s / peak_ang_speed_rad_s / peak_jerk_m_s3 are zero (or teleport artefacts) BY CONSTRUCTION and are NOT dynamics or stability evidence. P3 PASS is never P4 wheel-drive evidence.
- 告警：{'run_id': 'P3_neg_frozen_y_r01', 'run_dir': 'P3_neg_frozen_y_r01', 'code': 'vacuous_negative_control', 'message': "P3_neg_frozen_y_r01 froze 'y' but that command component was zero (0.0); gate_result='PASS'. The run proves nothing."}
- 告警：{'run_id': 'P3_neg_frozen_y_r02', 'run_dir': 'P3_neg_frozen_y_r02', 'code': 'vacuous_negative_control', 'message': "P3_neg_frozen_y_r02 froze 'y' but that command component was zero (0.0); gate_result='FAIL'. The run proves nothing."}
- 告警：COVERAGE GAP: compound has 1 independent PASS under v1.0 (need 2)
- 告警：COVERAGE GAP: translate_xy has 1 independent PASS under v1.0 (need 2)
- 告警：COVERAGE GAP: translate_z has 1 independent PASS under v1.0 (need 2)
- 告警：AMBIGUOUS: P3_compound_r09 has a two-entry quat probe but no v1.1 marker; bound to v1.0 per amendment section 4
- 告警：AMBIGUOUS: P3_compound_r11 has a two-entry quat probe but no v1.1 marker; bound to v1.0 per amendment section 4
- 告警：AMBIGUOUS: P3_static_r09 has a two-entry quat probe but no v1.1 marker; bound to v1.0 per amendment section 4
- 告警：AMBIGUOUS: P3_static_r11 has a two-entry quat probe but no v1.1 marker; bound to v1.0 per amendment section 4
- 告警：AMBIGUOUS: P3_xy_r09 has a two-entry quat probe but no v1.1 marker; bound to v1.0 per amendment section 4
- 告警：AMBIGUOUS: P3_z_r09 has a two-entry quat probe but no v1.1 marker; bound to v1.0 per amendment section 4
- 告警：AMBIGUOUS: P3_yaw_r09 has a two-entry quat probe but no v1.1 marker; bound to v1.0 per amendment section 4
- 告警：AMBIGUOUS: P3_yaw_r11 has a two-entry quat probe but no v1.1 marker; bound to v1.0 per amendment section 4
- 告警：AMBIGUOUS: P3_yaw_r13 has a two-entry quat probe but no v1.1 marker; bound to v1.0 per amendment section 4
- 告警：AMBIGUOUS: P3_yaw_r14 has a two-entry quat probe but no v1.1 marker; bound to v1.0 per amendment section 4

