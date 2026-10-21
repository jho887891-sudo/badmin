# P2 G5 重力落地 / 承重验收门禁（预注册 · v1.0）

**提交时间**：2026-10-21，**在任何 G5L / landing 模式运行之前**
**背景**：`G5_r01/r02`（PR #20）在 `g=-9.81`、lift 0.400 m 下真实落地，峰值 2693.495361 N，
但结果 JSON 判 FAIL —— 因为它套用了**零重力静置**的 `t0 世界位姿` 与 `60 步最大漂移 <= 0.001 m` 门禁。
自由落体必然改变世界位姿，故旧判定**不能**用于落地工况。G5 的旧 JSON 保留不改写，本文件新增专用门禁。

## 1. 阈值（冻结，不得事后放宽）

| Gate | 判据 | 依据 |
|---|---|---|
| `all_finite` | 全部遥测字段有限 | 健全性 |
| `contact_detected` | 全步接触力峰值 **> 1.0 N** | G4 正控制峰值 200.9 N（1 kg 落地）；20 kg 静态重力 196.2 N。1 N 低 2–3 个数量级，只判“有无” |
| `settles` | 末 **30 步**（0.25 s）内 `max|v| < 0.02 m/s` 且底盘 z 跨度 **< 0.002 m** | G4 正控制停稳于 z=0.0500；0.001 m 是静置门禁值，落地允许 2 mm |
| `no_tip_over` | 全步 `tilt_rad` 最大 **< 0.05 rad** | 四轮对称落地判据 |
| `no_long_term_penetration` | 末 30 步轮底最大穿透 **< 0.002 m** | 轮底 = 底盘 z + R·(轮心) − 0.090 |
| `static_weight_supported` | 末 30 步**平均**接触力与 `m·|g| = 196.2 N` 相对误差 **≤ 5 %** | 稳态支撑必须等于重量；单次峰値不能替代 |

轮底用 `half_height = 0.090 m` 的 **TEST_PROXY 盒**计算，不是真实轮胎；该代理值已在报告中标明，
真实轮径/滚动面仍是 UNKNOWN（P1 未闭环）。

## 2. 判词

- 全部 gate 为真 → 该 run **PASS**（仅代表“该代理几何在重力下落地并稳定承载”）。
- 有接触但 `settles`/`static_weight_supported` 未过 → **PARTIAL**（有碰撞响应，无稳定承重）。
- 无接触 → **FAIL**。
- 传感器不可用 → `contact_detected` 记 false 并要求 `contact_api` 明确标注，不得写 0 N 冒充读数。

## 3. 与其它门禁的关系

- 本门禁**不替代** P2 零重力静置门禁（`t0_world_pos <= 0.001 m`、`t0_world_rot <= 0.001 rad`、相对安装 `<= 0.001 m/0.001 rad`、60 步全程最大漂移 `<= 0.001 m`）。
- 静态门禁结果在 landing 模式的 JSON 中作为 `diagnostics.static_gates_as_diagnostics` **保留**，但在 landing 模式**不参与** `gate_result`。
- 每次实验仍需外层 `tools/simulation/p2_gate_runner.py` 判定（child rc=0 且 JSON gate PASS 才外层 0）。

## 4. 运行计划（串行，dt=1/120，TGS，g=-9.81）

| RUN_ID | 脚本 | 参数 | 用途 |
|---|---|---|---|
| `G5L_r01` | `p2_wc_g6_collision_variants.py --impl box_child_xform` | lift 0.400, steps 360 | 与 G5 完全相同的碰撞实现，改用落地门禁 |
| `G5L_r02` | 同上 | 同上 | 独立重复 |
| `G6A_r01/r02` | 同上 | 同上 | 已运行，作为第三/第四组同一实现的独立证据 |

## 5. 实现

`scripts/simulation/p2_landing_gates.py`（纯函数，无 Isaac）+ `tests/simulation/test_p2_landing_gates.py`（9 项单测，退出码 0）。
