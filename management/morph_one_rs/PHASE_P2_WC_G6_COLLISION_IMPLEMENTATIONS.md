# P2-WC G6 · 只改碰撞实现：Cube compound / 显式刚体 / Mesh ConvexHull 对照

**分支**：`dev/p2-wheel-contact-g1g4`（PR #20，base = PR #18 分支）
**预注册**：`management/morph_one_rs/PHASE_P2_WC_G6_PREREGISTRATION.md`（**运行前提交**，`83543be`）
**落地门禁预注册**：`management/morph_one_rs/PHASE_P2_G5_LANDING_GATE.md`（`a7c8b28`）
**GPU**：dgut@172.31.68.251 / RTX A6000 / Isaac Sim 6.0.1 / Isaac Lab 3.0.2-beta / dt=1/120 / TGS / 串行单实例
**几何**：TEST_PROXY（轮盒半尺寸 24.3 × 78.9 × 90.0 mm，最低点 `lift − 0.334 m`）——**不是真实轮胎**

## 1. 结论（一句话）

三种碰撞实现在 `g=-9.81`、lift 0.400 m（+66 mm 间隙）下**都**产生了真实接触、稳定停驻与**等于重量的稳态支撑**；
其中 **box_child_xform（G5 原有实现）是与 G5 逐位等价的可靠基线**；**只有显式刚体变体 B 能给出分轮法向力**，
因此 P4 若需要轮载/摩擦分配必须用 B 结构；Mesh ConvexHull（C）是合法且几乎等价的替代实现。

## 2. 单变量定义与配对条件（全部冻结于预注册文件）

| 代号 | `--impl` | 碰撞实现 | 质量 | 与 G5 的关系 |
|---|---|---|---|---|
| A | `box_child_xform` | Chassis 单刚体 + 4 个 `wheelN` 子 Xform cube | 20 kg | **逐行等同 G5** |
| B | `box_child_rigid` | 4 个 `wheelN` 显式刚体（各 1 kg）+ `FixedJoint` 固连 Chassis（16 kg） | 16+4×1 = 20 kg | 仅改碰撞体层级/实现 |
| C | `mesh_convex_hull` | 4 个 `wheelN` 子 Xform 下椭圆柱 Mesh（32 段）+ `MeshCollisionAPI(convexHull)` | 20 kg | 仅改 Shape 类型与 cooking |

公共：lift 0.400 m、`g_z=-9.81`、360 步（3.0 s）、ground plane、`PhysxCfg(solver_type=1)`、COM 在刚体原点（与 G5 一致）。
B 的 20 kg 拆分是该实现无法避免的第二处差异，已在 `manifest.json` 与 `result.json` 的
`chassis_mass_kg/wheel_mass_kg/total_mass_kg` 中公开。

## 3. 结果（每 impl ≥2 个独立 run_id，串行，外层 gate runner 判定）

| RUN_ID | impl | 峰值接触 N | 首触步 | 末态 z m | 末 30 步 max\|v\| m/s | 倾斜 rad | 末 30 步穿透 m | JSON gate | 外层 exit |
|---|---|---|---|---|---|---|---|---|---|
| G6A_eq_r01（60 步等价控制） | A | **2693.495361328** | 12 | 0.333999693394 | 5.13395e-04 | 8.7677e-05 | 3.36e-07 | PASS | 0 |
| G6A_r01 | A | 2693.495361328 | 12 | 0.333999693394 | 5.13395e-04 | 8.7677e-05 | 3.36e-07 | PASS | 0 |
| G6A_r02 | A | 2693.495361328 | 12 | 0.333999693394 | 5.13395e-04 | 8.7677e-05 | 3.36e-07 | PASS | 0 |
| G5L_r01（落地门禁） | A | 2693.495361328 | 12 | 0.333999693394 | 5.13395e-04 | 8.7677e-05 | 3.36e-07 | PASS | 0 |
| G6B_r05 | B | 855.51348877 | 12 | 0.33399990201 | 4.67113e-04 | 9.3463e-05 | 9.8e-08 | PASS | 0 |
| G6B_r06 | B | 855.51348877 | 12 | 0.33399990201 | 4.67113e-04 | 9.3463e-05 | 9.8e-08 | PASS | 0 |
| G6C_r01 | C | 2701.190917969 | 12 | 0.333999663591 | 5.84411e-04 | 1.87813e-04 | 3.36e-07 | PASS | 0 |
| G6C_r02 | C | 2701.190917969 | 12 | 0.333999663591 | 5.84411e-04 | 1.87813e-04 | 3.36e-07 | PASS | 0 |

**确定性**（预注册容差：峰值 ≤1%、末态 z ≤1 mm、首触步 ≤1）：三种 impl 全部 **PASS**，实测峰值相对差 **0.0**、
末态 z 差 **0.0**、首触步差 **0**。

**G5 等价性控制**：`G6A_eq_r01`（60 步）峰值 2693.495361328 N 与 `G5_r01` **完全相同**（相对差 0.0 ≤ 1%），
末态 z、漂移（0.064618955 m）与 G5 逐位一致 → 新 harness **没有**改变物理，G6 与 G5 可直接比较。

## 4. 承重（补充观测，**非** G6 预注册 gate 项，事后离线计算，单独标注）

| RUN_ID | 末 30 步平均接触力 N | 期望重量 m·g N | 相对误差 | 分轮平均值 N |
|---|---|---|---|---|
| G5L_r01 | **196.200016** | 196.2 | 8.0e-08 | UNKNOWN（A 无分轮传感器） |
| G6A_r01 / r02 | 196.200016 | 196.2 | 8.0e-08 | UNKNOWN |
| G6C_r01 / r02 | 196.200012 | 196.2 | 6.2e-08 | UNKNOWN |
| G6B_r05 / r06 | 60.809241（**仅 chassis 传感器**） | 196.2 | 0.690 | 60.809 / 41.027 / 49.878 / 44.486，**合计 196.2002** |

→ A/C 的稳态支撑力**等于 20 kg 重量**（误差 ~1e-7），不是“某一次峰値”。
→ B 的分轮传感器合计 **196.2002 N = mg**，但 **chassis 传感器只有 60.81 N**（数值恰好等于 wheel1）：
**在 B 结构下不能只读 chassis 传感器当作总支撑力**；必须对 4 个轮体求和。该 API 语义（父体是否收到 FixedJoint 约束力报告）仍为 **UNKNOWN**。

## 5. 本轮新发现的真实缺陷（都已保留失败证据，不删除）

1. **嵌套刚体在本 build 非法**：把 `RigidBodyAPI` 施加在 Chassis 刚体**子 prim** 上（`G6B_r01/r02`），
   `ContactSensor` 初始化报 `RuntimeError: Sensor at path '/World/RC/Chassis/wheel1' could not find any bodies with contact reporter API`。
   Isaac Lab 的 `activate_contact_sensors` 遍历时**遇到第一个 RigidBodyAPI 就停止下降**，因此子刚体永远拿不到 reporter。
   修正：B 的四个轮体改为 Chassis 的**兄弟**刚体（各自承担 lift），用 `FixedJoint` 固连，并用 `XformCache` 断言
   世界轮底仍等于 `lift-0.334`（`per_wheel_world_bottom_z_m = 0.066000003`，全部一致）。
2. **只有 FixedJoint 的 articulation 仍触发 `init_state` 正则失败**（`G6B_r03/r04`）：
   `ValueError: Not all regular expressions are matched ... .*: [] Available strings: []`（`joint_names` 为空）。
   修正：**所有变体**都在构造后清空 `joint_pos/joint_vel` 字典（坑 #3 的完整版本）。

## 6. UNKNOWN（不得写成 0 或 PASS）

- PhysX **cooked shape 查询**：本 build 无暴露 API → `cooked_shape_query = UNKNOWN`。
- B 结构下 chassis 传感器语义（60.81 N vs 196.2 N）→ **UNKNOWN**（观察已记录）。
- 轮碰撞盒半高 **0.090 m 是 TEST_PROXY**；真实滚动面/轮径/轴心/轮距仍 **UNKNOWN**（P1 未闭环），
  因此 `lift = 0.334` **不是**官方落地高度，不得写入 P4 参数。
- A/C 的分轮法向力不可测（只能测 chassis 合力）→ 分轮项 **UNKNOWN**，不是 0 N。

## 7. 复现命令（全部经外层 gate runner，GPU 串行）

见 `EXECUTION` 一节下方代码块（同一 impl 两个 run_id，串行）：

```bash
export PYTHONPATH=$(ls -d /home/T7/ojh/robot_sim/IsaacLab/source/* | tr '\n' ':')
PY=/home/T7/ojh/robot_sim/env_isaaclab/bin/python
# 每个 impl 各跑 2 个独立 run_id，串行，单 Isaac 实例
$PY tools/p2_gate_runner.py --script scripts/simulation/p2_wc_g6_collision_variants.py \
    --run-id G6_IMPL_r01 --out-dir /home/dgut/runs/g/G6_IMPL_r01 --timeout 1200 -- \
    --impl IMPL --lift 0.400 --gravity-z -9.81 --steps 360 --read-contacts --gate-mode landing
# 离线汇总（CPU，无 Isaac）
python scripts/simulation/p2_wc_g6_compare.py --run-dir A_r01 --run-dir A_r02 \
    --baseline-g5 G5_r01/result.json --equivalence-run G6A_eq_r01 --out g6_summary.json --write-notes
```

## 8. 证据位置

- Run 目录：`outputs/simulation/rebot_b601_rs/morph_one/p2_wc_gates/<RUN_ID>/`
  （`authored_chassis.usda`、`manifest.json`、`raw_poses.jsonl`、`contacts.jsonl`、`result.json`、
  `gate_runner_summary.json`、`gate_child.std{out,err}.log`、`notes.md`、`landing_analysis.json`、`outer.log`）
- 失败证据：`G6B_r01`（嵌套刚体）、`G6B_r02`（同）、`G6B_r03`（init_state 正则）、`G6B_r04`（同）——原样保留
- 汇总：`p2_wc_gates/g6_summary.json`
- 离线单测：`tests/simulation/test_p2_landing_gates.py` → `9 passed`，exit 0

## 9. 对后续阶段的可消费结论

- **P3（底盘 6DoF 运动 + 臂安装稳定性）**：用 **A 结构**（与 G5 等价、已验证、最少活动件）。
- **P4（四轮真实驱动/摩擦）**：若需要分轮法向力 → 必须用 **B 结构**，且支撑力要对 4 轮求和；
  轮半径/轴心/轮距仍 **UNKNOWN**，P4 参数化控制可做，真值不可冻结。
- **C** 可作为合法替代实现（峰值 +0.29%，承重完全一致），但没有额外收益，保留为备选。
