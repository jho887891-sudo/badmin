# 交接状态（P2-WC 之后）— 供新会话零上下文接续

本文件是唯一入口：新会话读完这一页即可继续。

## 1. 分支与 SHA（全部已 push）

| 分支 | HEAD | 内容 | 状态 |
|---|---|---|---|
| dev/p2-wheel-contact-g1g4（PR #20，base = PR #18 branch） | 6d8fe70981df4fca974ac33f647e8840d9f0eec5 | P2-WC G0/G1/G2/G3/G4/G5/G7 + 报告 + 证据 | 待审 |
| dev/morph-one-p2-quat-and-settle（= dev/morph-one-p2-init-diagnostics） | 1f5ef7a 起 | P2 根因、真实臂挂载、C6/C7/C8 | 已交付 |
| test/p0-gate-runner-regression | 5e143d0 | 18 个 gate runner 用例 + 修 2 个 fail-open | PASS |
| p1/wheel-identity-205-vs-181 | 76c7022 | 205.991 vs 180.782 归因 | PARTIAL |
| p5/arm-static-consistency | ca30af0 | 单臂静态一致性 + reset.py 越限缺陷 | PASS |

## 2. P2-WC Gate 现状

| Gate | 状态 | 关键数字 |
|---|---|---|
| G0 | PASS | 历史双重 lift：lift 0.300 轮底 +0.2256 m vs 修正 -0.034 m，差 0.2596 m |
| G1 | PASS | t0 5.96e-09 m、漂移 0.0、接触 0 N（2 次独立 run） |
| G2 | PASS | 恰好接触：t0 1.12e-07、漂移 1.55e-07 |
| G3 | FAIL（预期） | 穿透 34 mm -> step0 64.077667 N -> 漂移 0.404265 -> 外层 exit 1 |
| G4 | PASS | 1 kg 立方体自由落体正控制：200.911575 N、第 19 步首触、停 z=0.0500 |
| G5 | 接触已发生 | 峰值 2693.495361 N；零重力静态门禁不适用于重力落地 |
| G6 | 未做 | 只改碰撞实现的对照（Cube vs Mesh/凸包），与 G5 比较 |
| G7 | PASS | 真实臂 + 修正底盘：10 刚体 / 8 DOF、臂 sha256 未变、t0 5.98e-08、漂移 1.63e-09 |

下一步只有 G6；G6 完成后 P2 静态门禁闭环，才可考虑 P3。P4 仍需 P1 轮件身份（轮径/轴心/轮距 = UNKNOWN）。

## 3. 环境与命令（GPU 主机 dgut@172.31.68.251）

    export PYTHONPATH=$(ls -d /home/T7/ojh/robot_sim/IsaacLab/source/* | tr '\n' ':')
    PY=/home/T7/ojh/robot_sim/env_isaaclab/bin/python
    $PY tools/p2_gate_runner.py --script scripts/simulation/morph_one_p2_real_chassis.py --run-id G6_r01 --out-dir /home/dgut/runs/g/G6_r01 -- --lift 0.350 --read-contacts

主机会话目录：/home/dgut/runs/g/（P2-WC）、/home/dgut/runs/p2/（早期 P2）。

## 4. 必须知道的坑（都踩过并已修，勿回退）

1. Kit 会把失败写成 exit 0 -> 一律用 tools/p2_gate_runner.py 外层判定（子进程 rc=0 且 JSON gate=PASS 才 0）。
2. CUDA tensor 不能直接 np.asarray -> 必须先 detach 再 cpu() 再 numpy()。
3. C0 无关节时 Isaac Lab 默认 init_state.joint_pos 正则 .* 匹配失败 -> 构造后清空该字典；本 build 没有 ArticulationCfg.InitState。
4. USD 四元数是 wxyz，robot.data.body_quat_w 在本 build 是 xyzw -> 按来源分别转换，禁止共用同一 convention 比较。
5. 子 prim 继承父变换 -> 轮子碰撞体偏移必须相对 chassis；绝对 CAD 中心会双重计入 lift（历史 0 N 真因）。
6. 碰撞几何 spawn 时必须与地面严格分离，否则 sim.reset() 注入 0.22-2.92 m/s 上浮速度（P2 根因）。
7. ContactSensor 需要 activate_contact_sensors=True，且 prim 必须是刚体。
8. PhysX/Isaac Lab 关节索引顺序 != URDF 顺序（USD stage 遍历顺序也不是），只能按名字映射。
9. 主机无 GitHub 出口 -> 传文件用 scp；远端 shell 脚本先 scp 再执行，不要在 PowerShell 里内联 $ 变量。

## 5. 证据位置（仓库内）

- outputs/simulation/rebot_b601_rs/morph_one/p2_wc_gates/<run_id>/（result/manifest/gate_runner_summary/61 行 raw_poses.jsonl/authored USD）
- outputs/simulation/rebot_b601_rs/morph_one/p2_matrix/<run_id>/（P2 代理矩阵 C0-C8）
- outputs/simulation/rebot_b601_rs/morph_one/p2_real_arm/<run_id>/（真实臂挂载与间隙扫描）
- outputs/simulation/rebot_b601_rs/morph_one/p1_wheel_identity/、p5_static/（子代理证据）
- 报告：PHASE_P2_WC_GATES_G1_G4.md、PHASE_P2_ROOT_CAUSE_GROUND_PENETRATION.md、PHASE_P2_REAL_ARM_AND_GATE.md、PHASE_P2_C678_AND_CLEARANCE.md、P0_GATE_RUNNER_TESTS.md、P1_WHEEL_IDENTITY_205_VS_181.md、P5_ARM_STATIC_CONSISTENCY.md

## 6. 纪律

不合并 PR、不 force push、不碰 main、不跑 PPO、不操作真机；真资产只读并记录 sha256；结论只写 PASS / FAIL / PARTIAL / BLOCKED / UNKNOWN；失败证据不删。
