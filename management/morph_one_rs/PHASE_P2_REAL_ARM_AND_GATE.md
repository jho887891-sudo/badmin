# P2 真实资产验证：地面间隙规则 + 外层 Gate（FAIL 不再伪装成 PASS）

分支 `dev/morph-one-p2-quat-and-settle`　环境：Isaac Sim 6.0.1 + Isaac Lab 3.0.0-beta2，TGS，dt=1/120 s，零重力。

## 1. 真实 B601-RS 资产上的对照（只读引用，未改写）

| Run | 地面 | 抬高 m | Gate（JSON） | 外层 exit | 子进程 exit | t0 世界位置误差 m | t0 世界旋转误差 rad | 安装位置误差 m | 安装旋转误差 rad | 60 步漂移 m | step0 root 速度 m/s |
|---|---|---|---|---|---|---|---|---|---|---|---|
| RM_lift_r01 | 有 | 0.40 | **PASS** | **0** | 0 | 5.98e-08 | 1.25e-07 | 0.0 | 0.0 | **1.63e-09** | 0.0 |
| RM_nolift_r01 | 有 | 0 | **FAIL** | **1** | **0** | 0.049226 | 0.001272 | 3e-11 | 0.0 | 0.485598 | **2.924878** |

两次运行都是：10 刚体 / 8 DOF / 底盘在 articulation 内 / 真实臂资产 sha256 前后一致（`arm_unchanged=True`）。

要点：

* **同一份代码、同一阈值**，只改"初始离地间隙"这一个因素 → FAIL 变 PASS；
* 无间隙时 reset 阶段就注入 **2.92 m/s** 向上速度（代理场景是 0.46–0.53 m/s，真实臂/25 kg 底盘更大），
  随后 0.486 m 平移；安装约束本身始终正确（3e-11 m）；
* 有 0.40 m 间隙时 t0 世界位姿误差 ≤1.25e-07、安装误差 0.0、60 步漂移 **1.63e-09 m**，四条 SPEC 4.5 Gate 全过。

## 2. P0：外层 subprocess gate 已落地并证明有效

新工具 `tools/simulation/p2_gate_runner.py`：在**普通 Python 父进程**里启动 Kit 实验，分别读取
(1) 子进程返回码、(2) 实验自己写的 JSON gate；**只有两者同时为 0/PASS 才返回 0**，并把原始子进程状态
原样写进 `gate_runner_summary.json`。

实证（本轮两次真实运行）：

    RM_nolift_r01: child_returncode = 0（Kit 把失败吞成 0）  json_gate = FAIL  ->  outer_exit = 1
    RM_lift_r01:   child_returncode = 0                      json_gate = PASS  ->  outer_exit = 0

即：**FAIL 不会再被 Kit 的 0 退出状态掩盖**。这正是 Issue #13 反复要求的那条契约。

## 3. 复现

    export PYTHONPATH=$(ls -d /home/T7/ojh/robot_sim/IsaacLab/source/* | tr '\n' ':')
    python p2_gate_runner.py --script morph_one_p2_real_arm_mount.py \
        --run-id RM_lift_r01 --out-dir <dir> -- --lift 0.40
    python p2_gate_runner.py --script morph_one_p2_real_arm_mount.py \
        --run-id RM_nolift_r01 --out-dir <dir>

证据：`outputs/simulation/rebot_b601_rs/morph_one/p2_real_arm/<run_id>/`（result.json、manifest.json、
61 行 raw_poses.jsonl、gate_runner_summary.json、gate_child.stderr.log、authored_mount.usda）。

## 4. 阶段判定

| 阶段 | 状态 | 说明 |
|---|---|---|
| P0 | **PASS（本轮）** | 外层 gate 正负例均实测：FAIL→1、PASS→0；Kit 吞状态的行为被如实记录 |
| P2（TEST_PROXY） | **PASS** | 9 个单因素变体，四条 Gate 全过（见 PHASE_P2_ROOT_CAUSE_GROUND_PENETRATION.md） |
| P2（真实 B601-RS） | **PASS（挂载+静置部分）** | 10 刚体 / 8 DOF，t0 世界与安装误差、60 步漂移全部达标 |
| P2 剩余 | **待补** | 真实底盘（282 实体简化碰撞）落地高度、接触/穿透信号读取、与 PR #15 度量对齐 |
| P3/P4/P6/P7/P8 | BLOCKED | 依赖 P2 真实底盘部分与 P1 轮件身份 |

## 5. 下一轮最小实验

1. 用真实底盘简化碰撞体（先 7 个模块级凸包/盒）替换代理底盘，跑同一 `--lift` 扫描，确定安全落地高度区间；
2. 读取 `contact/penetration` 信号（PhysX contact report 或 `get_net_contact_forces`），把"未授权穿透 = 0"
   从 UNKNOWN 变成实测；
3. 与 PR #15 的 source-aware 度量对齐（本轮 harness 为等价自研实现），并把 `--write-data` / gain 对照
   （SPEC C6/C7/C8）补齐。
