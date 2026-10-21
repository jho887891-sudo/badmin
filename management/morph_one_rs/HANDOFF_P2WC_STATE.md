# 交接状态（P2-WC G6 之后）— 供新会话零上下文接续

本文件是唯一入口：读完这一页即可继续。总任务队列见
`management/morph_one_rs/EXECUTION_QUEUE_CONTINUOUS_P0_P8.md`（chatgpt/p2-wc-continuous-queue-20261021）
与 `EXECUTION_QUEUE_STATE.json`（`queue/a01-a03-a10` 分支）。

## 1. 分支与 SHA（全部已 push）

| 分支 | HEAD | 内容 | 状态 |
|---|---|---|---|
| `dev/p2-wheel-contact-g1g4`（PR #20，base = PR #18 分支） | 见 `git log` | P2-WC G0–G7 + G6 三实现对照 + 落地门禁 + 证据 | 待审 |
| `p5/rs-reset-pose`（PR #22，base `p5/arm-static-consistency`） | `667230693448370cca2c11d5eceab778519b2aec` | B601-RS 型号感知 reset pose（修 PiPER joint3=-1.5 越限） | PASS（52 passed / exit 0；反向对照 11 failed） |
| `p1/wheel-identity-205-vs-181` | `76c7022` | 205.991 vs 180.782 归因 | PARTIAL |
| `p5/arm-static-consistency` | `ca30af0` | 单臂静态一致性 + reset 越限 canary | PASS |
| `test/p0-gate-runner-regression` | `5e143d0` | 18 用例 gate runner 回归 | PASS |
| `test/p0-gate-runner-strict-outdir` | 进行中（子代理） | P0-02..P0-06 fail-closed 硬化 | RUNNING |
| `p1/wheel-rolling-surface` | 进行中（子代理） | P1-04/05/08/09 轮件身份合同 | RUNNING |
| `queue/a01-a03-a10` | 进行中（子代理） | A01/A02/A03/A10 队列基建 | RUNNING |

## 2. P2-WC Gate 现状（G0–G7 全部有结论）

| Gate | 状态 | 关键数字 |
|---|---|---|
| G0 | PASS（离线） | 历史双重 lift：lift 0.300 轮底 **+0.2256 m**（悬空）vs 修正 −0.034 m，差 0.2596 m |
| G1 | PASS | lift 0.350（+16 mm 间隙）t0 5.96e-09 m、漂移 0.0、接触 0 N（2 run） |
| G2 | PASS | lift 0.334 恰接触：`G2_r01` 1.12e-07 / 漂移 1.55e-07，`G2_r02`（本轮补跑）1.1184e-07 / 1.55365e-07 → **重复已补齐** |
| G3 | FAIL（预期负控制） | 穿透 34 mm → step0 64.077667 N → 漂移 0.404265 → 外层 exit 1 |
| G4 | PASS | 1 kg 立方体自由落体，第 19 步首触，峰值 200.911575 N，停 z=0.0500 |
| G5 | **落地门禁 PASS**（旧静态 gate FAIL 保留不改） | 峰值 2693.495361 N，首触第 12 步，末态 z 0.333999693394，末 30 步平均接触 **196.200016 N = mg**，穿透 3.36e-07 m |
| G6 | PASS | 三实现全 PASS 且确定性 0.0 差；A/B/C 见 `PHASE_P2_WC_G6_COLLISION_IMPLEMENTATIONS.md` |
| G7 | PASS（0g 旧证据） | 真实臂 + 修正底盘：10 刚体 / 8 DOF、臂 sha256 未变、t0 5.98e-08、漂移 1.63e-09 |
| G7B（P2-07/08 新增） | 见 `p2_wc_gates/G7B_*` | 真实臂 + 四轮 compound 同场景：静置（g=0）与承重（g=-9.81）分开判定 |

### G6 结论（可靠碰撞结构）
- **A `box_child_xform`（G5 原实现）= 可靠基线**：与 G5 峰值逐位一致（2693.495361328 N，相对差 0.0），
  稳态支撑 196.200016 N（相对误差 8e-08）。
- **B `box_child_rigid`（4 个显式轮刚体 + FixedJoint，总质量仍 20 kg）= 唯一可分轮测法向力**：
  分轮峰值 855.51 / 534.95 / 740.07 / 316.48 N，分轮稳态均值合计 **196.2002 N = mg**；
  但 chassis 传感器只有 60.81 N，**不可当总支撑力**。
- **C `mesh_convex_hull` = 合法等价替代**：峰值 2701.190917969 N（+0.29%），稳态 196.200012 N。
- 新发现的两条真实缺陷（失败证据已入库）：嵌套刚体拿不到 contact reporter（`G6B_r01/r02`）；
  只有 FixedJoint 的 articulation 仍触发 `init_state` 正则失败（`G6B_r03/r04`）。

## 3. 环境与命令（GPU 主机 dgut@172.31.68.251）

    export PYTHONPATH=$(ls -d /home/T7/ojh/robot_sim/IsaacLab/source/* | tr '\n' ':')
    PY=/home/T7/ojh/robot_sim/env_isaaclab/bin/python
    # 落地实验（G6 / G5L）
    $PY tools/p2_gate_runner.py --script scripts/simulation/p2_wc_g6_collision_variants.py \
        --run-id <ID> --out-dir /home/dgut/runs/g/<ID> --timeout 1200 -- \
        --impl box_child_xform --lift 0.400 --gravity-z -9.81 --steps 360 --read-contacts --gate-mode landing
    # 真实臂 + 四轮（P2-07）
    $PY tools/p2_gate_runner.py --script scripts/simulation/p2_wc_g7b_arm_with_wheels.py \
        --run-id <ID> --out-dir /home/dgut/runs/g/<ID> --timeout 1800 -- \
        --lift 0.40 --steps 360 --gate-mode landing --gravity-z -9.81 --read-contacts

远端依赖：`scripts/simulation/p2_landing_gates.py`、`p2_chassis_frame_math.py` 必须与主脚本一起 scp，
否则 `ModuleNotFoundError`（本轮踩过一次）。运行目录 `/home/dgut/runs/g/`。

## 4. 必须知道的坑（都踩过并已修，勿回退）

1. Kit 会把失败写成 exit 0 → 一律用 `tools/simulation/p2_gate_runner.py` 外层判定。
2. CUDA tensor 不能直接 `np.asarray` → `detach().cpu().numpy()`。
3. 无 DOF 场景（含只有 FixedJoint 的 articulation）必须清空 `init_state.joint_pos/joint_vel` 字典。
4. USD 四元数 wxyz / `body_quat_w` xyzw（本 build）→ 按来源分别转换。
5. 子 prim 继承父变换 → 轮碰撞体偏移相对 chassis，父体只加一次 lift；B 结构下轮体为**兄弟**刚体，需自带 lift。
6. 碰撞几何 spawn 时必须与地面严格分离，否则 `sim.reset()` 注入 0.22–2.92 m/s 上浮速度。
7. `ContactSensor` 需 `activate_contact_sensors=True` 且 prim 必须是刚体；**嵌套刚体拿不到 reporter**。
8. PhysX/Isaac Lab 关节索引顺序 ≠ URDF/USD 遍历顺序，只能按名字映射。
9. 主机无 GitHub 出口；PowerShell 里不要把 `$` 内联进 ssh，写 `.sh` 先 scp 再执行。
10. 落地工况不能用零重力静置门禁判 FAIL；用 `p2_landing_gates.py`（预注册，9 单测）。
11. 新脚本若 import 同目录模块，必须与依赖一起 scp，否则远端 `ModuleNotFoundError`。

## 5. 证据位置（仓库内）

- `outputs/simulation/rebot_b601_rs/morph_one/p2_wc_gates/<RUN_ID>/`：G1/G2/G3/G4/G5/G5L/G6A/B/C/G7 全量
  （`authored_*.usda`、manifest、raw_poses.jsonl、contacts.jsonl、result.json、gate_runner_summary.json、
  child stdout/stderr、notes.md、landing_analysis.json）
- `p2_wc_gates/g6_summary.json`：G6 汇总（表、确定性、G5 等价性、补充承重）
- 报告：`PHASE_P2_WC_GATES_G1_G4.md`、`PHASE_P2_WC_G6_COLLISION_IMPLEMENTATIONS.md`、
  `PHASE_P2_G5_LANDING_GATE.md`、`PHASE_P2_WC_G6_PREREGISTRATION.md`、`PHASE_P2_ROOT_CAUSE_GROUND_PENETRATION.md`、
  `PHASE_P2_REAL_ARM_AND_GATE.md`、`P0_GATE_RUNNER_TESTS.md`、`P1_WHEEL_IDENTITY_205_VS_181.md`、
  `P5_ARM_STATIC_CONSISTENCY.md`、`P5B_RESET_POSE.md`

## 6. 仍未闭环（UNKNOWN，不得当已知参数）

- 轮径 / 轴心 / 轮距 / 真实滚动面 = **UNKNOWN**（P1 进行中）；碰撞盒半高 0.090 m 是 TEST_PROXY，比 103 mm 特征半径小 13 mm。
- PhysX cooked shape 查询 API = UNKNOWN（本 build 无暴露）。
- B 结构下 chassis 传感器语义（60.81 N vs 196.2 N）= UNKNOWN。
- 真实臂在重力下的承重/关节保持 = G7B 本轮判定（见 `G7B_grav_*`）。

## 7. 纪律

不合并 PR、不 force push、不碰 main、不跑 PPO、不操作真机；真资产只读并记录 sha256；
结论只写 PASS / FAIL / PARTIAL / BLOCKED / UNKNOWN；失败证据不删。

## 8. 2026-10-21 本轮新增（P2-10 / P3 准备 / 并行 CPU 线 / 队列）

| 项 | 结果 | 证据 |
|---|---|---|
| P2-02 manifest 重力修正 | DONE | `morph_one_p2_real_chassis.py` 记录实际 `--gravity-z`；旧 G5 JSON 未改写 |
| P2-03 G5 落地门禁预注册 | DONE | `PHASE_P2_G5_LANDING_GATE.md`、`p2_landing_gates.py`、9 单测 exit 0 |
| P2-01 G2 重复 | DONE | `G2_r02` PASS（1.1184e-07 / 1.55365e-07） |
| P2-05/P2-06 G6 + 分通道碰撞证明 | DONE | G6A/B/C 各 2 run、`g6_summary.json` |
| P2-07/P2-08 真实臂 + 四轮同场景 | **BLOCKED（环境）→ 重试中** | `G7B_static_r01..r05`（两个真实建场缺陷已修）、`G7B_grav_r07` |
| P2-10 证据矩阵 | PARTIAL（等 G7B） | `PHASE_P2_EVIDENCE_MATRIX.md` |
| P3 动态挂载 | 脚本+预注册已入库，未跑 | `p3_dynamic_mount.py`、`PHASE_P3_DYNAMIC_MOUNT_GATE.md` |
| P0-02..P0-06 | DONE（子代理） | `test/p0-gate-runner-strict-outdir` @ `ce278c2`，92 passed |
| P5-02 reset 越限修复 | DONE（子代理） | `p5/rs-reset-pose` @ `6672306`，52 passed |
| P1-04/05/08/09 轮件合同 | DONE/PARTIAL（子代理） | `p1/wheel-rolling-surface` @ `c3ec799`；`wheel_radius` 仍 UNKNOWN |
| P6-01/02/04/05 | DONE/PARTIAL（子代理） | `p6/racket-camera-frames` @ `d7672d5`（球拍 USD 无碰撞几何；D455 全部 UNKNOWN） |
| P7-04/05/07 | PARTIAL（子代理） | `p7/detector-ekf-interfaces` @ `8150b2a`，50 passed；拖曳不匹配时 CI 覆盖 0%（负结果已钉住） |
| A01/A02/A03/A10 队列基建 | DONE（子代理） | `queue/a01-a03-a10` @ `490e34a`，60 passed；96 槽位状态文件 |

**环境观测（写报告时必须带上）**：本轮宿主被其他用户占用，`load average` 峰值 13–18、`%st` 32.6%、
swap 已用 7.4 GB；他人 `isaacsim.exp.full.streaming.kit` 223% CPU + 训练任务 189% CPU。
Kit 单次启动从 ~40 s 退化到 **5–20 min 并多次停滞**（`omni.kit.extension.log` 长时间无新事件）。
G6 全部 8 个 run 在此之前完成（每 run ~55 s），因此 G6 结论不受影响；G7B 受影响。
