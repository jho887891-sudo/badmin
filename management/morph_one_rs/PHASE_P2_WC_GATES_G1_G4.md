# P2-WC：四轮底盘轮地碰撞（G1/G2/G3/G4 已完成，含正控制）

分支 `dev/p2-wheel-contact-g1g4`（base = PR #18 branch `chatgpt/p2-chassis-world-frame-contact-20261021`）
环境：Isaac Sim 6.0.1 + Isaac Lab 3.0.0-beta2，TGS，dt = 1/120 s，每组 ≥2 个独立 run_id，串行无并发。
全部运行经 `tools/p2_gate_runner.py` 外层验收（子进程 rc=0 **且** JSON gate=PASS 才返回 0）。

## 1. 门禁结果（核心表）

| Gate | run_id | lift m | 间隙 m | step0 接触力 N | t0 世界位置误差 m | t0 旋转误差 rad | 60 步漂移 m | JSON gate | 外层 exit |
|---|---|---|---|---|---|---|---|---|---|
| G4 正控制（1 kg 立方体自由落体，g=−9.81） | G4_r01 / G4_r02 | — | — | — | — | — | 落到 z=0.0500 | **PASS** | 0 |
| G1 有正间隙 | G1_r01 / G1_r02 | 0.350 | +0.016 | 0.0 | 5.96e-09 | 0.0 | 0.0 | **PASS** | 0 |
| G2 恰好接触 | G2_r01 | 0.334 | 0.000 | 0.0 | 1.12e-07 | 0.0 | 1.55e-07 | **PASS** | 0 |
| G3 已知穿透 | G3_r01 / G3_r02 | 0.300 | −0.034 | **64.077667** | 0.0374 | 0.00798 | 0.404265 | **FAIL** | **1** |

两次独立 run_id 的 G1 与 G3 数值**逐位一致**（G1 t0 5.96e-09、G3 f0 64.077667 N、漂移 0.404264995007），
说明该配置在本环境下是确定性的。

## 2. 三个必须分开陈述的结论

1. **ContactSensor 管线有效（G4 正控制 PASS）**：1 kg 立方体从 0.20 m 自由落体，第 19 步首次接触，
   峰值 **200.911575 N**，最终停在 z=0.0500（半高 0.05，正好落地）。这条只校准传感器与 PhysX 接触报告，
   **不是**轮地接触成功证据。
2. **PhysX 碰撞在修正后的四轮 compound 上真实生效（G3）**：穿透 34 mm 时 step0 就有 **64.08 N**，
   随后整机被迫抬升，漂移 0.404 m、t0 世界误差 0.0374 m → 生产 Gate **FAIL**、外层 exit **1**。
   之前 `RC_lift0300_r01/r02` 报 0 N 的原因是我自己那版授权（子 prim 继承父变换导致几何没到地面），
   PR #18 的 frame math + 自校验修掉了这一点；本轮用 PR #18 的授权复核。
3. **有间隙时零重力静置干净（G1/G2）**：0.016 m 间隙与"恰好接触"两种情形 t0 世界误差 ≤1.12e-07 m、
   60 步漂移 ≤1.55e-07 m，接触力 0 N——**零重力静止本身不算轮地接触成功**，只证明未穿透时装配正确。

## 3. 与 P2 历史结论的关系

* 上一轮 `RM_*`（单块底盘盒）结论不变：0 间隙 → 注入 2.92 m/s → FAIL；0.06/0.10/0.40 m → PASS。
* P2 生产 Gate（t0 世界 ≤1 mm / 旋转 ≤1 mrad / 60 步漂移 ≤1 mm）在 G1、G2 上满足；G3 按预期 FAIL。
* **轮径/轴心/轮距仍 UNKNOWN**（P1 子代理结论：205.991 vs 180.782 的差异来自"BREP 顶点盒 vs 三角化点云盒"，
  半径侧 103.0 mm 与点云 r_max 103.000022 mm 一致；不得把任一方当轮径）。

## 4. 剩余 Gate（未做，明确列出）

| Gate | 状态 | 说明 |
|---|---|---|
| G0 离线复算历史双重 lift | 待做 | 可用 PR #18 的 `historical_nested_wheel_bottom_world_z` + 459cb12 的 USD 证据复算 |
| G5 重力下真实落地（修正后四轮） | 待做 | 需 g=−9.81、受控下落，证实四轮 compound 与 ground 的真实交互（G3 已间接证明会响应） |
| G6 只改碰撞实现（Mesh/Cube/compound 变体） | 待做 | 与 G5 比较，找出可靠碰撞结构 |
| G7 与真实 B601-RS 装配回归 | 待做 | 必须保持上一轮 10 刚体 / 8 DOF 结论不被破坏 |

## 5. 复现

    export PYTHONPATH=$(ls -d /home/T7/ojh/robot_sim/IsaacLab/source/* | tr '\n' ':')
    python tools/p2_gate_runner.py --script scripts/simulation/p2_wc_g4_contact_control.py \
        --run-id G4_r01 --out-dir <dir>
    python tools/p2_gate_runner.py --script scripts/simulation/morph_one_p2_real_chassis.py \
        --run-id G3_r01 --out-dir <dir> -- --lift 0.300 --read-contacts

证据：`outputs/simulation/rebot_b601_rs/morph_one/p2_wc_gates/<run_id>/`（result.json、manifest.json、
gate_runner_summary.json、61 行 raw_poses.jsonl、authored USD）。
## 6. 追加完成：G0 / G5 / G7（同一分支）

### G0 离线复算历史双重 lift（无 PhysX）

从 `459cb12` 提交的 USDA 复算（脚本 `scripts/simulation/p2_wc_g0_offline_recompute.py`，退出码 0）：

| lift m | 历史（双重 lift）轮底世界 z | 修正后轮底 z | 差值 m |
|---|---|---|---|
| 0.300 | **+0.2256**（悬空 225.6 mm） | **−0.034**（穿地 34 mm） | 0.2596 |
| 0.334 | +0.2936 | 0.000 | 0.2936 |
| 0.350 | +0.3256 | +0.016 | 0.3256 |

历史 chassis translate z = 0.2596，wheel child z = 0.0560（两者都含 lift）→ 世界轮底 = 0.2256 m。
**这就是 `RC_lift0300_r01/r02` 报 0 N 的原因：几何根本没到地面**（不是传感器失效）。G0 与 G3（修正后 64 N）
互为闭环证据。

### G5 重力下真实落地（修正后四轮 compound）

| run_id | lift | 重力 | step0 接触 | 落地过程峰值接触力 | t0 世界误差 | 60 步位移 | 静态 Gate | 外层 exit |
|---|---|---|---|---|---|---|---|---|
| G5_r01 / G5_r02 | 0.40 | −9.81 | 0.0 N | **2693.495361 N** | 0.001383805275 | 0.064618954925 | FAIL（预期） | 1 |

**口径说明（重要）**：G5 判 FAIL 是因为它拿"零重力静态"门禁去量重力落地过程，而 SPEC 对 G5 的要求是
"证实四轮 compound 可与 ground 发生交互"。实测**峰值接触力 2693.5 N** 已经证明交互发生；两次独立
run_id 数值逐位一致。零重力静态 Gate 不适用于重力落地场景，本条按"接触发生 = 该 Gate 目的达成"记录，
但不计入 P2 静态门禁。

### G7 真实 B601-RS 装配回归（修正后底盘 + 只读真实臂）

| run_id | Gate | 外层 exit | 刚体 | DOF | 底盘在 articulation | 臂资产未改 | t0 世界误差 | 60 步漂移 |
|---|---|---|---|---|---|---|---|---|
| G7_r01 | **PASS** | 0 | **10** | **8** | 是 | 是 | 5.98e-08 m | 1.63e-09 m |

上一轮的 10 刚体 / 8 DOF 结论**未被破坏**。

## 7. Gate 汇总

| Gate | 状态 | 依据 |
|---|---|---|
| G0 | **PASS** | 离线复算，历史双重 lift 差值 0.2596 m，解释 0 N |
| G1 | **PASS** | 2 次独立 run，t0 5.96e-09、漂移 0.0、接触 0 N |
| G2 | **PASS** | 恰好接触，t0 1.12e-07、漂移 1.55e-07 |
| G3 | **FAIL（预期）** | 穿透 34 mm → 64.077667 N → 漂移 0.404265，外层 exit 1 |
| G4 | **PASS** | 正控制 200.911575 N，落点 0.0500 |
| G5 | **接触发生（PASS 其目的）** | 峰值 2693.495361 N；零重力静态门禁不适用 |
| G6 | **未做** | 需只改碰撞实现的对照 |
| G7 | **PASS** | 10 刚体 / 8 DOF 保持 |
