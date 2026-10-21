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
