# P2 最终证据矩阵（P2-10）

**分支**：`dev/p2-wheel-contact-g1g4`（PR #20）
**矩阵口径**：每一行是一个**已入库**的 run 或离线检查；"证据层"区分离线复算 / 零重力静置 / 重力落地 / 真实资产。
**结论词**只用 PASS / FAIL / PARTIAL / BLOCKED / UNKNOWN。**旧 FAIL 数据一律保留不改写。**

## 1. 几何与授权层（离线，CPU）

| 项 | 结论 | 关键数字 | 证据 |
|---|---|---|---|
| G0 历史双重 lift 复算 | **PASS** | lift 0.300 时轮底 **+0.2256 m**（悬空）vs 修正 **−0.034 m**，差 0.2596 m | `p2_wc_g0_offline_recompute`、PR #18 原始 USDA |
| 四轮 authored 世界轮底（修正后） | **PASS** | 每个 run 均由 `XformCache` 断言 `|bottom − (lift−0.334)| ≤ 1e-5`；G6 全部 = 0.066000003 | 各 run 的 `result.json.diagnostics` |
| 轮件真实身份（轮径/轴心/轮距） | **UNKNOWN** | 030062→ 见 P1：spin_axis PASS(±x)、radius UNKNOWN（103 mm 只是圆特征边，0.5847 mm 轴向带）、track 500.004 mm PARTIAL | `P1_WHEEL_ROLLING_SURFACE.md`、`wheel_identity_contract.json` |
| 碰撞盒 vs 真实轮 | **PARTIAL（已知简化）** | 半高 0.090 m 比 103 mm 特征半径小 **13 mm** → `lift 0.334` 不是官方落地高度 | P1-08 合同字段 |

## 2. 零重力静置层（g=0，静态门禁）

| RUN_ID | 工况 | t0 位置误差 m | t0 旋转 rad | 60 步全程最大漂移 m | JSON gate | 外层 exit |
|---|---|---|---|---|---|---|
| G1_r01 / G1_r02 | lift 0.350，+16 mm 间隙 | 5.96e-09 | 0 | 0.0 | PASS | 0 |
| G2_r01 | lift 0.334，恰接触 | 1.12e-07 | 0 | 1.55e-07 | PASS | 0 |
| **G2_r02**（本轮补跑） | 同上（独立重复） | 1.1184e-07 | 0 | 1.55365e-07 | PASS | 0 |
| G3_r01 / G3_r02 | lift 0.300，穿地 34 mm（**负控制**） | 0.0374 | 0 | 0.404265 | **FAIL（预期）** | **1** |
| G7_r01 | 真实 B601-RS + 单盒代理底盘 | 5.98e-08 | 1.25e-07 | 1.63e-09 | PASS | 0 |

零重力静置层结论：**修正后四轮几何在非穿透工况下 t0 与 60 步全程稳定；穿地负控制被正确判 FAIL 且外层 exit=1。**
**仅凭这一层不能声称轮地接触有效**。

## 3. 碰撞/传感器有效层

| RUN_ID | 工况 | 观测 | 结论 |
|---|---|---|---|
| G4_r01 / r02 | 1 kg 立方体自由落体（正控制） | 第 19 步首触，峰值 200.911575 N，停 z=0.0500 | **PASS**：ContactSensor 管线真实有效 |
| G3 系列 | 已知穿透 34 mm | step0 **64.077667 N**、root 初速 0.80853 m/s、整机抬升 0.404 m | **PASS（碰撞真实生效）**，但生产 Gate FAIL |
| G6A/B/C | 见第 4 节 | 三实现均有真实接触 | **PASS** |

## 4. 重力落地 / 承重层（`g=-9.81`，落地门禁预注册）

| RUN_ID | impl | 峰值 N | 首触步 | 末态 z m | 末 30 步 max\|v\| m/s | 末 30 步穿透 m | 稳态支撑 N（补充观测） | gate | exit |
|---|---|---|---|---|---|---|---|---|---|
| G5L_r01 | A（=G5） | 2693.495361328 | 12 | 0.333999693394 | 5.13395e-04 | 3.36e-07 | **196.200016**（=mg，8e-08） | PASS | 0 |
| G6A_r01 / r02 | A | 2693.495361328 | 12 | 0.333999693394 | 5.13395e-04 | 3.36e-07 | 196.200016 | PASS | 0 |
| G6B_r05 / r06 | B | 855.51348877 | 12 | 0.33399990201 | 4.67113e-04 | 9.8e-08 | 分轮合计 **196.2002** | PASS | 0 |
| G6C_r01 / r02 | C | 2701.190917969 | 12 | 0.333999663591 | 5.84411e-04 | 3.36e-07 | 196.200012 | PASS | 0 |
| G5_r01 / r02（历史） | A（60 步） | 2693.495361328 | 12 | 0.333999693394 | — | — | 196.187…196.212（末 12 步） | **FAIL（旧静态阈值，保留不改写）** | 1 |

**确定性**：A/B/C 各两次独立 run 的峰值相对差 **0.0**、末态 z 差 **0.0**、首触步差 **0**（预注册容差 1% / 1 mm / 1 步）。
**G5 等价性**：60 步变体 A 与 `G5_r01` 峰值逐位一致 → 新 harness 未改变物理。

## 5. 真实 B601-RS 联合层（P2-07/P2-08）

| RUN_ID | 工况 | 结论 |
|---|---|---|
| G7_r01 | 真实臂 + 单盒代理底盘，g=0，10 刚体 / 8 DOF，臂 sha256 未变 | **PASS**（旧证据，复用） |
| G7B_static_r01/r02 | 真实臂 + 四轮 compound，g=0 | **FAIL 建场**（漏 scp `p2_landing_gates.py` → ModuleNotFoundError），证据保留 |
| G7B_static_r03/r04 | 同上，修 import 后 | **FAIL 建场**：两个 `ArticulationRootAPI`（`/World/RW/Chassis` 与 `/World/RW/Arm/Geometry`）→ PhysX 拒绝；证据保留 |
| G7B_static_r05 / G7B_grav_r01/r02 | 同上，去掉第二 root | **BLOCKED（环境）**：宿主被其他用户占用（`load average 13–18`、`%st 32.6`、swap 已用 7.4 GB、他人 Isaac full app 223% CPU + 训练 189%），Kit 单次启动从 ~40 s 退化到 **5–20 min 且多次停滞**（`omni.kit.extension.log` 长时间无新事件）；run 被主动 kill 并保留日志 |
| G7B_grav_r07 | 单跑重试 | 见 `G7B_grav_r07/result.json`（本文件写作时仍在启动中；若完成则为唯一一次真实臂 + 四轮 + 重力证据） |

**P2-07/P2-08 结论**：真实臂与修正四轮 compound 的**同场景重力承重尚未取得可判定的 PASS/FAIL 证据**，
当前状态 **BLOCKED（环境资源竞争）**，脚本与预注册已入库，宿主空闲后可直接重跑。

## 6. Gate 正确性层

| 项 | 结论 |
|---|---|
| 外层 runner 双保险（child rc + JSON gate） | **PASS**（G3 与 G5 历史 FAIL 均正确给出外层 exit 1） |
| 18→92 项 runner 回归 | **PASS**（`test/p0-gate-runner-strict-outdir` @ `ce278c2`，92 passed exit 0；反向对照 79 failed） |
| JSONL 完整性检查器在**真实**证据上运行 | **PASS**（`G6A_r01`：361 帧、contact numeric 361、`valid=true`、exit 0） |
| 预注册后不得放宽 | **PASS**（G6 与 G5 落地门禁均在运行前提交；G5 旧 JSON 未改写） |

## 7. 解除依赖条件（写清楚，避免下轮重新推理）

- **P2-10 自身**：待 `G7B_grav_r07` 出结果后，把第 5 节最后一行替换为真实 PASS/FAIL；在此之前 P2-10 = **PARTIAL**。
- **P3 解锁**：P2-10 DONE 后立即解锁；`p3_dynamic_mount.py` 与 `PHASE_P3_DYNAMIC_MOUNT_GATE.md` 已入库待跑。
- **P4 解锁**：仍阻断于 P1-09（轮径/轴心/轮距 UNKNOWN），不可用 `lift 0.334`、`206 mm`、`0.090 m 盒` 当真实轮参数。
- **P6/P7 静态项解锁**：队列模型把 P6-01/P6-05 挂在 P2-10 上，P2-10 完成后自动变 READY/DONE。
