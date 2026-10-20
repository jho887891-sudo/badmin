# P2 根因已定位并有 FAIL→PASS 单因素证据：初始地面穿透

分支 `dev/morph-one-p2-quat-and-settle`　harness `scripts/simulation/morph_one_p2_c0_minimal.py`
环境：Isaac Sim 6.0.1 + Isaac Lab 3.0.0-beta2，TGS，dt=1/120 s，零重力，TEST_PROXY，每步全序列入库。

## 1. 结论

**零重力下"无指令仍上浮"的根因是：初始时刻代理碰撞几何与静态地面平面相互穿透。**
PhysX 在 `sim.reset()` 期间解算这个穿透，**在 step0 之前**就给整机注入了向上的速度（实测
**0.456 – 0.529 m/s**），随后表现为长时间匀速平移（0.25–0.37 m）。它与
ArticulationRootAPI 放置、关节链、连杆自碰撞**都无关**。

## 2. 单因素消融矩阵（每个变体只改一个因素）

| Run | 地面 | 抬高 m | 关节数 | root API | Gate | t0 世界位置误差 m | settle 漂移 m | step0 root 速度 m/s |
|---|---|---|---|---|---|---|---|---|
| C0_r03 | 无 | 0 | 0 | Chassis 刚体 | **PASS** | 0.0 | **0.0** | 0.0 |
| C3x_r01 | 无 | 0 | 0 | 非刚体 Xform | **PASS** | 0.0 | **0.0** | 0.0 |
| C4_r01 | 无 | 0 | 1 | Chassis | **PASS** | 3.7e-09 | **0.0** | 0.0 |
| C5_r01 | 无 | 0 | 2 | Chassis | **PASS** | 7.5e-09 | **0.0** | 0.0 |
| C5x_r01 | 无 | 0 | 2 | Xform | **PASS** | 7.5e-09 | **0.0** | 0.0 |
| C0g_lift_r01 | 有 | 0.30 | 0 | Chassis | **PASS** | 0.0 | **0.0** | 0.0 |
| C5g_lift_r01 | 有 | 0.30 | 2 | Chassis | **PASS** | 7.5e-09 | **0.0** | 0.0 |
| C1_r01 | 有 | 0 | 0 | Chassis | **FAIL** | 0.0413 | 0.287077 | **0.529100** |
| C5g_nolift_r01 | 有 | 0 | 2 | Chassis | **FAIL** | 0.0411 | 0.251871 | **0.456180** |

读法：

* **只改"地面"这一个因素**（C0 → C1）：PASS→FAIL，step0 速度 0 → 0.529 m/s。
* **只改"抬高"这一个因素**（C1 → C0g_lift；C5g_nolift → C5g_lift）：FAIL→PASS，step0 速度 0.456/0.529 → 0。
* 关节数 0→1→2、root API 刚体↔Xform，**都不引起失败**。
* 所有 PASS 变体的 t0 世界误差 ≤7.5e-09 m、安装误差 ≤1.2e-08 m、60 步漂移 **严格 0.0**。

几何原因很直接：C0/C1 的底盘碰撞盒中心在体原点、半高 0.04 m，即 z∈[-0.04, +0.04]，**穿地 40 mm**；
`--lift 0.30` 把它整体抬到地面之上后，同一份代码、同一物理配置即 PASS。

## 3. 这也解释了此前的 0.369 m

上一轮 `morph_one_p2_init_diagnostics.py` 里 reset0 实测底盘已到 z=0.0379 m 并持续上浮——同样的机制
（场景里有地面 + 某个体初始与地面相交 → reset 期间注入速度）。当时把该现象归因为"初始化偏差"是**不完整**
的，正确表述是"初始地面穿透"，而不是 ArticulationRootAPI 或 USD 授权方式的问题。

## 4. 这不是"掩盖问题"

* 没有固定根节点、没有每帧覆写位姿、没有删除碰撞体、没有改阈值。
* 判定用的是 SPEC 4.5 的四条 Gate：t0 世界位置 ≤1 mm、t0 世界旋转 ≤1 mrad、安装位置/旋转 ≤1 mm/1 mrad、
  60 步零重力无指令漂移 ≤1 mm。所有 PASS 变体是**同一代码 + 同一阈值**下的真实测量。
* 修法是"场景授权规则"：spawn 时碰撞几何必须与地面留有间隙（或按真实轮地接触高度放置），
  属于 P4/P6 必须遵守的装配约束，而不是物理参数调参。

## 5. 复现

    export PYTHONPATH=$(ls -d /home/T7/ojh/robot_sim/IsaacLab/source/* | tr '\n' ':')
    python morph_one_p2_c0_minimal.py --run-id C0_r03      --out-dir <dir>
    python morph_one_p2_c0_minimal.py --run-id C1_r01      --out-dir <dir> --include-ground
    python morph_one_p2_c0_minimal.py --run-id C0g_lift_r01 --out-dir <dir> --include-ground --lift 0.30
    python morph_one_p2_c0_minimal.py --run-id C5g_lift_r01 --out-dir <dir> --include-ground --joints 2 --lift 0.30
    python morph_one_p2_c0_minimal.py --run-id C5g_nolift_r01 --out-dir <dir> --include-ground --joints 2

证据：`outputs/simulation/rebot_b601_rs/morph_one/p2_matrix/<run_id>/`（每次运行 result.json、manifest.json、
authored_proxy.usda、child.stdout.log、child.stderr.log；C0_r03 / C5g_lift_r01 / C5g_nolift_r01 另含
61 行 `raw_poses.jsonl` 全序列）。

## 6. 遗留（下一轮）

1. **真实资产侧**：用同一 harneess 指向真实 B601-RS（只读引用）+ 真实底盘简化碰撞，确认"落地高度"规则
   可复现 PASS；这决定 P4/P6 的 spawn 规则。
2. **P0 外层 runner**：Kit 仍会把失败状态写成 0（本目录 `C0_r01/C0_r02` 的 python 异常 → 观测 exit 0），
   需接 PR #14 的 subprocess gate 做正负例。
3. 与 PR #15 的 source-aware 度量对齐（本 harness 自研等价实现，未 cherry-pick 其代码）。
