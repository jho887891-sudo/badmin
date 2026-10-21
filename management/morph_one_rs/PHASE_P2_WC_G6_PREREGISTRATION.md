# P2-WC G6 预注册 · 只改碰撞实现（CollisionAPI/Shape）与 G5 的单变量对照

**版本**：v1.0（2026-10-21，**在任何 G6 GPU 运行之前提交**）
**阶段**：P2-WC G6（Issue #13 队列第 5 项，PR #20 分支）
**基线**：G5 = 修正后四轮 box compound 碰撞 + `g=-9.81`，`G5_r01`：lift 0.400 m、间隙 +0.066 m、峰值接触 2693.495361 N、t0 误差 1.383805e-03 m、60 步漂移 0.064618954925 m、JSON gate FAIL（静态门禁不适用于重力落地）。
**目的**：在**同一场景、同一几何包络、同一质量、同一求解器**下，只替换碰撞实现，判定哪种碰撞结构可靠；并以 60 步变体 A 复现 G5 峰值作为“换 harness 不改物理”的等价性控制。

本文件提交后，第 3 节的指标与阈值**不得事后放宽**；任何偏离都必须新增一条“事后追加”小节并明确标注，不得覆盖本文件的历史版本。

## 1. 三个碰撞实现（每轮只改这一项）

| 代号 | `--impl` | 定义 | 与 G5 的关系 |
|---|---|---|---|
| A | `box_child_xform` | Chassis 单刚体（20 kg）+ 4 个 `wheelN` 子 Xform cube 碰撞体（继承父变换，无关节） | **与 G5 逐行等价** |
| B | `box_child_rigid` | 4 个 `wheelN` 为**显式刚体**（各 1 kg）+ `UsdPhysics.FixedJoint` 固连到 Chassis（16 kg），总质量仍 20 kg | 仅改碰撞体所属层级/实现 |
| C | `mesh_convex_hull` | 4 个 `wheelN` 子 Xform 下用**生成的椭圆柱 Mesh**（32 段，半轴 = 盒半高 0.0243/0.0789/0.0900），`UsdPhysics.MeshCollisionAPI(convexHull)` 烹饪 | 包络与最低点与 A 相同，只改 Shape 类型与 cooking |

**不可归因的耦合必须公开**：B 必须把 20 kg 拆成 16+4×1 kg（显式刚体无法零质量），这是该实现无法避免的第二处差异，已在 `manifest.json` 的 `impl_definition` 与 `chassis_mass_kg/wheel_mass_kg/total_mass_kg` 字段中记录。A/C 质量与 COM 与 G5 完全一致（COM 在刚体原点，CAD 主体 bbox 偏移**不建模**，属已知简化）。

**几何仍然是 TEST_PROXY**：轮盒 `24.3 × 78.9 × 90.0 mm` 半尺寸来自 CAD 包络，不是真实轮胎；最低点 `z = lift - 0.334 m` 由脚本内 `assert |authored_world_bottom - (lift-0.334)| <= 1e-5` 强制校验。

## 2. 冻结的运行计划（串行，单一 Isaac 实例，`dt=1/120`，TGS）

| RUN_ID | impl | steps | lift | g_z | 说明 |
|---|---|---|---|---|---|
| `G6A_eq_r01` | A | 60 | 0.400 | −9.81 | **等价性控制**：必须复现 G5_r01 峰值 |
| `G6A_r01` | A | 360 | 0.400 | −9.81 | 落地 + 稳定判定 |
| `G6A_r02` | A | 360 | 0.400 | −9.81 | 重复（独立启动） |
| `G6B_r01` | B | 360 | 0.400 | −9.81 | 落地 + 稳定判定 + **分轮接触** |
| `G6B_r02` | B | 360 | 0.400 | −9.81 | 重复 |
| `G6C_r01` | C | 360 | 0.400 | −9.81 | 落地 + 稳定判定 |
| `G6C_r02` | C | 360 | 0.400 | −9.81 | 重复 |

全部经 `tools/simulation/p2_gate_runner.py` 外层判定（child rc=0 **且** JSON `gate_result=PASS` 才外层 0）。每次独立进程、每次独立 `run_id` 目录，串行无并发。

## 3. 预注册指标与阈值（冻结，不得事后放宽）

落地组的 JSON gate（`--gate-mode landing`）逐项布尔：

| Gate | 判据 | 依据 |
|---|---|---|
| `all_finite` | 全部字段有限 | 基本健全性 |
| `contact_detected` | 全步 `max_contact_force_n` 峰值 **> 1.0 N** | G4 正控制（1 kg 落地峰值 200.9 N）；20 kg 静态重力 ≈196 N。1 N 比两者低 2–3 个数量级，只判“有无” |
| `per_wheel_contact_all_four` | 4 轮各自峰值 **> 1.0 N** | 仅在 **B（可绑定 ContactSensor 的显式刚体）** 可测；A/C 该字段为 `null`、不参与 PASS 判定，记为 **UNKNOWN** |
| `no_tip_over` | 全步 `tilt_rad` 最大值 **< 0.05 rad**（2.9°） | 四轮落地的几何对称性判据 |
| `settles` | 末 30 步（0.25 s）内 `max|v| < 0.02 m/s` **且** 底盘 z 跨度 **< 0.002 m** | G4 正控制停稳在 z=0.0500；静态门禁 0.001 m 不适用于落地 |
| `no_long_term_penetration` | 末 30 步内轮底最大穿透 **< 0.002 m** | 轮底由 `chassis_pos + R·(轮心局部) − 0.090` 计算 |

离线确定性判据（`p2_wc_g6_compare.py`，同一 impl 的两次独立 run）：

| 检查 | 容差 |
|---|---|
| 峰值接触力相对差 | ≤ 1 % |
| 末态底盘 z 绝对差 | ≤ 1 mm |
| 首次接触步差 | ≤ 1 步 |

**等价性控制**：`G6A_eq_r01`（60 步）峰值与 `G5_r01` 2693.495361 N 相对差 ≤ 1 %，否则判 **FAIL**（说明新 harness 改变了物理，G6 全部结论不可比）。

## 4. 判定词（严格按 SPEC 4.3）

- 单个 impl：两 run 全 gate PASS 且确定性 PASS → **PASS**；有接触但稳定性 gate 未过 → **PARTIAL**；无接触 → **FAIL**；per-wheel 不可测 → 该项 **UNKNOWN**（不阻断整体，但必须写出）。
- G6 整体：给出“可靠碰撞结构 = X”的结论句，或明确“三种实现均不可靠”的 FAIL 证据；不得用单次峰值代替稳定承重。
- 任何 `UNKNOWN` 字段保留在 JSON 中，禁止写入 0。

## 5. 反事实与失败证据保留

- 变体 C 若烘焙失败/静默无碰撞，保留原始 `child.stderr.log`、`authored_chassis.usda` 与 `UNKNOWN` 记录，不删除、不覆盖。
- 变体 B 若因 FixedJoint 触发关节/articulation 异常，保留异常原文，并在 `notes.md` 中写明“B 实现 BLOCKED 的具体原因”；A/C 结论不受影响。
- 历史 `G1–G5`、`RC_*` 证据只读，不重写。

## 6. 交付物

每个 run 目录：`authored_chassis.usda`、`manifest.json`、`raw_poses.jsonl`（逐步：底盘位姿/速度、轮体位置、tilt）、`contacts.jsonl`（分传感器接触力 + contact_api）、`result.json`、`gate_runner_summary.json`、`child.stdout.log`/`child.stderr.log`、`notes.md`。
汇总：`g6_summary.json` + `PHASE_P2_WC_G6_COLLISION_IMPLEMENTATIONS.md`。
