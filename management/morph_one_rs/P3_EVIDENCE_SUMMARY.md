# P3 证据汇总（自动生成口径，P3-08 / R+1 C1）

**分支**：`p3/evidence-summary`（worktree 自 `p3/gate-v1.1` @ `60a7f9a`，PR #35）
**生成器**：`scripts/simulation/p3_summary.py`（唯一数据来源）
**机器可读产物**：`outputs/simulation/rebot_b601_rs/morph_one/p2_wc_gates/p3_summary.json`
**自动表格**：`outputs/simulation/rebot_b601_rs/morph_one/p2_wc_gates/p3_summary.md`
**测试**：`tests/simulation/test_p3_summary.py`

本文件中的**每一个数字**都由上述脚本从 run 目录现场读取生成，不存在手工抄录；
本文件只解释"这些数字意味着什么"。重跑口径：

```bash
python scripts/simulation/p3_summary.py
# runs=12 v1.0=8 v1.1=4 coverage_gaps=8 negative_controls=2 errors=0
# exit 0
```

---

## 0. 结论边界（最重要的一段，必须先读）

**P3 的全部结果都来自诊断式 root-pose 驱动（`drive_mode = DIAGNOSTIC_ROOT_POSE (NOT wheel drive)`）**：
每步用 `write_root_pose_to_sim` 把底盘根位姿"瞬移"到指令位姿，并**每步清零根速度**。
因此 `peak_lin_speed_m_s` / `peak_ang_speed_rad_s` / `peak_jerk_m_s3` 是**由构造方式决定的**
（要么恒 0，要么是瞬移跳变的有限差分假象，例如静态剖面 `P3_static_r01` 报出
`peak_jerk_m_s3 = 6209.07`、`peak_lin_speed_m_s = 1.49`，而该 run 的命令振幅恒为 0）。

> **结论：P3 的速度/jerk 数值不是动力学证据，P3 的 PASS 也不是动态稳定性证据。**
> P3 证明的是"门禁接线正确、安装关系在瞬移驱动下不散架、有向位移判据真的能抓错"，
> 仅此而已。真实轮驱动力学必须由 P4 的 wheel-drive proxy 承担；**P3 PASS ≠ P4**。

脚本把这段声明写进 `caveats` 与 `warnings`（每次运行都打印），并有测试
`test_diagnostic_drive_caveat_is_always_emitted` 守住，防止有人把它删掉。

---

## 1. 代际划分（v1.0 与 v1.1 绝不混算）

按 `PHASE_P3_GATE_V1_1_AMENDMENT.md` §4，历史 run 继续绑定 v1.0，v1.1 只对新 RUN_ID 生效。
脚本用**产物**而不是 run 名判代际：

| 判据 | 归属 |
|---|---|
| `metrics.directed_translation` 存在（含有符号 3D 投影、`best_frame`） | v1.1 |
| `diagnostics.negative_control` 非空（freeze-axis 开关） | v1.1 |
| 以上皆无 | v1.0 |

**为什么不用 quat 探针判代际**：`diagnostics.quat_order_for_write_root_pose` 的两值探针
（`probe_measured_yaw_rad` 含 `wxyz`/`xyzw` 两项）**不是 v1.1 独有产物**——它在 v1.0 时代的
`P3_*_r09` 里就已经存在，而这些 run 仍带着 0.208 m 的 root 写入缺陷
（`t0_pose_write_error_m = 0.208394530817`，r11/r13 才降到 `6.16e-08`）。
若把"quat 探针两值"当作 v1.1 判据，会把 r09 与 r13 混成一代，这正是本脚本要防的事故。
脚本因此把该探针记录为 `quat_probe_two_entries`，并在它**单独出现**时打
`AMBIGUOUS: ... bound to v1.0` 告警（本次 7 个 run 命中）。
需要按字面规则复核时可用 `--spec-quat-rule` 开关，对照跑一次即可看到差异。

**本次划分结果**：v1.0 = 8 个 run，v1.1 = 4 个 run，互不相交（有测试断言）。

---

## 2. 每一 run 的证据（自动生成，摘自 p3_summary.md）

### v1.0（8 个）

| run_id | profile | gate_result | outer_exit | child_rc | 失败门禁 |
|---|---|---|---|---|---|
| P3_compound_r09 | compound | FAIL | 1 | 0 | translation_actually_measured |
| P3_compound_r11 | compound | PASS | 0 | 0 | - |
| P3_static_r01 | static | FAIL | 1 | 0 | joint_hold |
| P3_static_r09 | static | PASS | 0 | 0 | - |
| P3_static_r11 | static | PASS | 0 | 0 | - |
| P3_xy_r09 | translate_xy | PASS | 0 | 0 | - |
| P3_yaw_r09 | yaw | PASS | 0 | 0 | - |
| P3_yaw_r11 | yaw | PASS | 0 | 0 | - |

v1.0 **没有** `directed_translation` 字段：`translate_z` 剖面在 v1.0 口径下**根本没有平移门禁**
（`hypot(x, y)` 不看 Z），所以 v1.0 的 translate_z PASS 不能作为 Z 运动证据。

### v1.1（4 个）

| run_id | profile | gate 模式 | gate_result | outer_exit | child_rc | 失败门禁 |
|---|---|---|---|---|---|---|
| P3_neg_frozen_xy_r02 | translate_xy | negative_control(freeze_axis=x) | FAIL | 1 | 0 | translation_actually_measured |
| P3_xy_r13 | translate_xy | positive | PASS | 0 | 0 | - |
| P3_neg_frozen_z_r02 | translate_z | negative_control(freeze_axis=z) | FAIL | 1 | 0 | translation_actually_measured |
| P3_z_r13 | translate_z | positive | PASS | 0 | 0 | - |

v1.1 有向位移（`achieved_ratio = 沿命令投影 / 命令位移`，门限 0.9）：

| run_id | profile | 命令位移 (m) | achieved_ratio | final_ratio | best_frame | cross_track (m) | 通过 |
|---|---|---|---|---|---|---|---|
| P3_neg_frozen_xy_r02 | translate_xy | 0.3 | 2.42505e-05 | 2.39985e-05 | 78 | 4.5273e-05 | FAIL |
| P3_xy_r13 | translate_xy | 0.3 | 1.00144 | 1.00127 | 180 | 1.36932e-04 | PASS |
| P3_neg_frozen_z_r02 | translate_z | 0.1 | 0.000336468 | 0.00033617 | 54 | 3.11668e-05 | FAIL |
| P3_z_r13 | translate_z | 0.1 | 0.975779 | 0.975779 | 120 | 1.87002e-04 | PASS |

---

## 3. 分代际聚合（不跨代际平均）

| 代际 | run 数 | PASS | FAIL | 有向判据 run 数 | 最小 achieved_ratio | 安装位偏差 max (m) | 安装姿态偏差 max (rad) | jerk max (m/s³) |
|---|---|---|---|---|---|---|---|---|
| v1.0 | 8 | 6 | 2 | 0 | - | 6.6639e-08 | 9.4243e-08 | 6209.07 |
| v1.1 | 4 | 2 | 2 | 4 | 2.42505e-05 | 3.0328e-08 | 8.9407e-08 | 307.027 |

脚本**只**产出 `aggregates.v1.0` 与 `aggregates.v1.1`，没有 overall/all 之类的合池字段
（`test_aggregates_are_per_generation_and_never_pooled` 断言这一点）。
两代的 jerk 数字（6209 vs 307）本身就是"不可比"的证明：它们是瞬移跳变的假象，不是物理量。

---

## 4. 覆盖审计（本次最重要的发现）

规则：**每个剖面、每个代际，都需要 ≥2 个独立 PASS**（独立 = 不同 run 目录且
`raw_poses.jsonl` 内容哈希不同，防止"同一 run 复制两份"）。

| 代际 | profile | 独立 PASS | 要求 | 缺口 |
|---|---|---|---|---|
| v1.0 | compound | 1 | 2 | **缺口** |
| v1.0 | static | 2 | 2 | - |
| v1.0 | translate_xy | 1 | 2 | **缺口** |
| v1.0 | translate_z | 0 | 2 | **缺口** |
| v1.0 | yaw | 2 | 2 | - |
| v1.1 | compound | 0 | 2 | **缺口** |
| v1.1 | static | 0 | 2 | **缺口** |
| v1.1 | translate_xy | 1 | 2 | **缺口** |
| v1.1 | translate_z | 1 | 2 | **缺口** |
| v1.1 | yaw | 0 | 2 | **缺口** |

### 覆盖缺口清单（8 条）

- **v1.0 / translate_xy**：只有 `P3_xy_r09` 一次 PASS —— 这正是"单一 run 不足以支撑结论"的典型。
- **v1.0 / translate_z**：**0 次独立 PASS**。修订预注册 §4 引用 `P3_z_r09` 作为 v1.0 口径下的
  translate_z PASS，但该目录**不在 `p3/gate-v1.1` 分支上**（见 §7）。
- **v1.0 / compound**：只有 `P3_compound_r11` 一次 PASS（r09 在 v1.0 口径下 FAIL）。
- **v1.1 / translate_xy、v1.1 / translate_z**：各只有 r13 一次 PASS（v1.1 计划中的
  `P3_xy_r12` / `P3_z_r12` 第二次独立 run 不存在）。
- **v1.1 / compound、static、yaw**：v1.1 口径下**一次 run 都没有**。v1.1 只覆盖了平移剖面，
  偏航与复合剖面在 v1.1 口径下无任何证据。

也就是说：**v1.1 目前只有"平移"一项单 run 证据 + 两个负控制**，凡是需要"两次独立复现"的结论
在 v1.1 口径下都不成立。这条覆盖审计就是让"translate_xy / translate_z 只有一个 run"这类问题
在 CI 里自动暴露，而不是靠人眼数目录。

---

## 5. 负控制审计（必须按设计失败）

| run_id | profile | frozen_axis | gate_result | outer_exit | 假的门禁 | achieved_ratio | 判定 |
|---|---|---|---|---|---|---|---|
| P3_neg_frozen_xy_r02 | translate_xy | x | FAIL | 1 | translation_actually_measured | 2.42505e-05 | 通过（按设计失败） |
| P3_neg_frozen_z_r02 | translate_z | z | FAIL | 1 | translation_actually_measured | 0.000336468 | 通过（按设计失败） |

判定条件（脚本硬断言，任一不满足即 exit 1 + 报 `negative_control_passed`）：

1. `gate_result == "FAIL"`；
2. `gate_runner_summary.json.outer_exit == 1`（外层 gate runner 真的非零退出，不是 Kit 吞异常）；
3. 该 frozen axis 对应的运动门禁**确实为 false**（x/y/z → `translation_actually_measured`）。

**结果：两个负控制都按设计失败，负控制审计通过（无 `negative_control_passed` 错误）。**
反向自检由 `test_negative_control_that_passed_fails_the_script` 保证：把一个负控制改成 PASS
后脚本真实退出码为 **1**，并写出 `negative_control_passed` 错误——即"负控制通过 = harness 故障"。

---

## 6. 诊断量：root↔chassis 偏置与安装漂移

- `diagnostics.root_to_chassis_offset_m`：**0.261476607 m**（7 个 run 全部同值，跨 run 极差 **0**
  → 无偏置漂移；root 系分量 `root_to_chassis_offset_root_frame_m = [0.215999865, 0.145000108, -0.026249634]`）。
- 声明做了偏置补偿的 run：**7 / 12**（r09 那批没有该字段，属于未补偿时代）。
- 安装漂移（`mount_position_deviation_max_m` / `mount_rotation_deviation_max_rad`）：
  全库 max = **6.6639e-08 m / 9.4243e-08 rad**；按代际 v1.0 = 6.6639e-08 / 9.4243e-08，
  v1.1 = 3.0328e-08 / 8.9407e-08；门限 0.001 m / 0.001 rad（余量约 4 个数量级）。
- `raw_poses.jsonl` 帧数：观测 **361**，`steps+1 = 361` 期望一致（12/12 run）。

---

## 7. 证据完整性与已知 UNKNOWN

脚本的硬失败条件（真实退出码）：

| 条件 | 退出码 |
|---|---|
| 全部 run 都有 `result.json` + `gate_runner_summary.json`，且负控制全部按设计失败 | **0** |
| 某个 run 缺 `result.json` | **1**（`missing_json`） |
| 某个 run 缺 `gate_runner_summary.json` | **1**（`missing_gate_runner_summary`） |
| 负控制竟然 PASS（或 outer_exit ≠ 1、或对应门禁不为 false） | **1**（`negative_control_passed`） |
| 目录里没有任何 `P3_*` run | **2** |

本次真实运行：`runs=12 v1.0=8 v1.1=4 coverage_gaps=8 negative_controls=2 errors=0`，**exit 0**。

**UNKNOWN / 需人工确认（脚本无法自证）**：

1. **`P3_z_r09` 不在本分支**：修订预注册 §4 引用它作为 v1.0 translate_z PASS，但在
   `p3/gate-v1.1` 的 run 目录里不存在；它只出现在 `dev/p2-wheel-contact-g1g4` 分支的
   `93fcc54`（"add the P3_z_r09 run that was reported PASS but never fetched into the repository"）。
   脚本**不跨分支取数**（否则"证据"会变成"任意分支的历史"），因此 v1.0/translate_z 记为 0 覆盖。
   是否把该 run 迁到 P3 证据分支，需 owner 决定。
2. **`P3_static_r01` 的 jerk = 6209 / lin = 1.49**：静态剖面命令为 0 却出现有限差分速度，
   说明这些数字纯粹来自 root 写入跳变。**不得**据此推断任何动力学结论。
3. **v1.1 只有 4 个 run**：偏航/复合/静态剖面在 v1.1 口径下无证据；v1.1 的
   "两次独立复现"要求（`P3_xy_r12` / `P3_z_r12`）**未完成**。
4. 负控制只覆盖"缺 X"与"缺 Z"，**缺 Y 与缺 YAW 的负控制不存在**（`--freeze-axis y|yaw` 未运行）。
5. `gate_runner_summary.json` 里的 `child_returncode` 全部为 0（含 FAIL 的负控制），即失败由
   JSON 门禁判定、而不是 child 崩溃——这与"Kit 会吞异常"的历史缺陷一致，故 `outer_exit` 才是可信退出码。

---

## 8. 复现

```bash
# 汇总（写 json + md，exit 0）
python scripts/simulation/p3_summary.py
# 汇总（只看结果，不写文件；用字面 quat 规则复核代际划分）
python scripts/simulation/p3_summary.py --no-write --spec-quat-rule
# 测试
python -m pytest tests/simulation/test_p3_summary.py -q
python -m pytest tests/simulation/test_p3_motion_gate.py tests/simulation/test_p3_motion_gate_track.py -q
```

本切片**只新增文件**：`scripts/simulation/p3_summary.py`、`tests/simulation/test_p3_summary.py`、
`outputs/.../p2_wc_gates/p3_summary.json`、`outputs/.../p2_wc_gates/p3_summary.md`、本文件。
未修改任何历史 run 目录、任何 `PHASE_P3_*` 文档、`p3_dynamic_mount.py`、`p3_motion_gate.py`
或队列状态文件。
