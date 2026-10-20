# P2 补充矩阵：C6/C7/C8 与真实臂离地间隙扫描

分支 `dev/morph-one-p2-quat-and-settle`　环境：Isaac Sim 6.0.1 + Isaac Lab 3.0.0-beta2，TGS，dt=1/120 s，零重力。
全部经 `tools/simulation/p2_gate_runner.py` 外层验收（子进程 rc=0 且 JSON gate=PASS 才返回 0）。

## 1. C6/C7/C8（SPEC 4.3，一次只改一个因素）

| Run | write_data_to_sim | 增益 | Gate | 外层 exit | 60 步漂移 m |
|---|---|---|---|---|---|
| C6_r01 | 关 | none | PASS | 0 | 0.0 |
| C7_r01 | 开 | none | PASS | 0 | 0.0 |
| C8a_r01 | 开 | none（零增益基线） | PASS | 0 | 0.0 |
| C8b_r01 | 开 | 300 / 20（验证增益） | PASS | 0 | 0.0 |

结论：**写入调用与驱动增益都不引入漂移**。此前观察到的上浮因此可以完全归因于初始地面穿透一项，
与 `write_data_to_sim` 或增益无关。

## 2. 真实 B601-RS 上的离地间隙扫描

底盘代理碰撞盒半高 0.05 m（体原点在 z=0 时 z∈[-0.05,+0.05]）。

| Run | 离地间隙 m | Gate | 外层 exit | t0 世界位置误差 m | 60 步漂移 m |
|---|---|---|---|---|---|
| RM_nolift_r01 | 0（穿地 50 mm） | **FAIL** | **1** | 0.049226 | 0.485598 |
| RM_lift0p06_r01 | 0.06 | **PASS** | 0 | 4.38e-08 | 1.63e-09 |
| RM_lift0p10_r01 | 0.10 | **PASS** | 0 | 3.87e-08 | 1.63e-09 |
| RM_lift_r01 | 0.40 | **PASS** | 0 | 5.98e-08 | 1.63e-09 |

**冻结的装配规则**：spawn 时任何碰撞几何必须与地面平面严格分离；本例阈值与几何一致——只要间隙 > 0.05 m
（即底盘半高）就 PASS，0 间隙（穿地）必然 FAIL。真实底盘（P4/P6）必须以真实轮-地接触高度放置，
不能用"先穿地再让 PhysX 顶出来"。

## 3. 阶段状态

| 阶段 | 状态 |
|---|---|
| P0 | **PASS**（外层 gate 正负例实测：FAIL→1、PASS→0；Kit 吞状态行为记录在 summary） |
| P2 TEST_PROXY | **PASS**（C0/C3x/C4/C5/C5x/C1/C0g_lift/C5g_lift/C5g_nolift + C6/C7/C8a/C8b 共 13 次运行） |
| P2 真实 B601-RS 挂载与静置 | **PASS**（10 刚体 / 8 DOF / 资产未改 / t0 与 60 步漂移全部达标） |
| P2 剩余 | 真实底盘简化碰撞（282 实体）落地高度、contact/penetration 信号（当前 UNKNOWN）、与 PR #15 度量对齐 |
| P3/P4/P6/P7/P8 | BLOCKED |

## 4. 复现

    python p2_gate_runner.py --script morph_one_p2_c0_minimal.py --run-id C7_r01 --out-dir <dir> -- --joints 2 --write-data
    python p2_gate_runner.py --script morph_one_p2_c0_minimal.py --run-id C8b_r01 --out-dir <dir> -- --joints 2 --write-data --gains verified
    python p2_gate_runner.py --script morph_one_p2_real_arm_mount.py --run-id RM_lift0p06_r01 --out-dir <dir> -- --lift 0.06

证据：`outputs/simulation/rebot_b601_rs/morph_one/p2_matrix/{C6,C7,C8a,C8b}_r01/`、
`.../p2_real_arm/RM_lift0p06_r01/`、`.../p2_real_arm/RM_lift0p10_r01/`（各含 result.json、manifest.json、
gate_runner_summary.json、raw_poses.jsonl）。
