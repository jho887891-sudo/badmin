# P2 阶段报告：PhysX 初始位姿与 settle 正确性（含 P0 退出码契约）

分支 `dev/morph-one-p2-init-diagnostics`（base `96ef90f614300174fb700aafe8abde541e1552aa`）
对应 Issue #13 的 P0 + P2。环境：Isaac Sim 6.0.1 + Isaac Lab 3.0.0-beta2，TGS，dt=1/120 s，**零重力**，TEST_PROXY 几何。

## 0. 门禁结果

| 门禁 | 结果 | 依据 |
|---|---|---|
| P0 退出码契约 | **PARTIAL** | 管道掩盖已复现并修正；Kit close() 覆盖进程状态已观测，但 os._exit 修复的验证运行与上一进程重叠，本轮**未取得干净证据** |
| P2 t=0 授权对齐 | **PASS（在自洽的四元数约定下）** | 安装约束残差 **3.5e-08 m**、相对转角 **0.0 rad** |
| P2 settle 稳定 | **FAIL** | 60 步内整机漂移 **0.369020 m**（预注册阈值 0.001 m） |
| P2 body-only 授权方式 | **FAIL（按设计暴露）** | 偏移被吞掉，残差恒为 **0.238537 m** ＝ 偏移长度 |
| P3 旋转/平移挂载 | 未开始（依赖 P2 settle PASS） | — |

---

## 1. P0：为什么 acceptance=false 却报 EXIT=0

两个互相独立的成因，都已经用最小复现钉住（证据 `p0_exitcode_demo.txt`）：

    管道掩盖：
      python3 -c '...; sys.exit(1)' | tail -3     -> masked_exit=0     （问号变量取的是 tail 的状态）
      同上，改读 PIPESTATUS[0]                      -> pipestatus0=1
    改为重定向：
      python3 -c '...; sys.exit(1)' > out 2>&1    -> redirect_exit=1
      python3 -c '...; sys.exit(0)' > out 2>&1    -> redirect_exit_ok=0

也就是说：上一轮 runner 用 `... 2>&1 | grep ... | tail`，报告里的 `MODE_EXIT=0` 是 `tail` 的退出码；
脚本本身按 `0 if all(acceptance) else 1` 确实返回 1，只是被管道吃掉了。

**第二个成因（本轮新发现，更严重）**：即使改用重定向、脚本 `return 1`，实测进程退出码仍是 **0**——
Isaac/Kit 的 `SIMULATION_APP.close()` 会接管进程终止并把状态写成 0。本轮改为：

    SIMULATION_APP.close()
    os._exit(code)          # 关闭之后再离开，任何框架都无法覆盖门禁状态

修复后的验证运行**不干净**：日志里同时出现 `PROCESS_EXIT_CODE=1` 与 `REAL_EXIT=0`（上一进程尚未退出、两次运行重叠）。因此本轮记为 PARTIAL：机制已定位，修复待一次独占、无重叠的运行重新取证。
**审计脚本 `morph_one_proxy_physx_probe.py` 仍是旧写法（close 后 return），其退出码同样不可信**；
本轮没有改它（保持审计对照不被污染），列为给 ChatGPT 的必改项。
---

## 2. P2 t=0：上一轮报的 179° yaw 是**我的读法错误**，不是初始化错误

诊断脚本在四个时刻逐体记录位姿：授权 USD（读回）、spawn 后（活动 stage 组合结果）、`sim.reset()` 后第 0 步、
第 1 步与 settle 每步。安装约束残差用两种四元数约定各算一遍：

| 时刻 | 约定 wxyz | 约定 xyzw | 自洽结论 |
|---|---|---|---|
| spawn 后（未步进） | — | — | 见 JSON |
| reset0 安装位置残差 | 0.261244825 m | **3.5e-08 m** | **xyzw 才是运行时数组的真实顺序** |
| reset0 底盘系相对位置 | (-0.1208, +0.0513, +0.1992) | **(0.11999, -0.04999, 0.19999)** | 与授权 (0.12,-0.05,0.20) 一致 |
| reset0 安装相对转角 | 0.0 rad | 0.0 rad | 两种约定都给出 0（两个体同姿态） |

结论：本构建里 `robot.data.body_quat_w` 的四个分量必须按 **(x, y, z, w)** 读。按 (w,x,y,z) 读时，
单位四元数 (1,0,0,0) 被看成"绕 X 转 180°"，于是出现我上一轮报告里的"≈179° yaw"和"偏移 x/y 反号"。
**这两个现象都是读法造成的假象，t=0 的实际安装位姿是正确的（1e-8 量级）。**

同期记录的、与四元数顺序无关的事实：reset0 底盘位置相对授权值是 `(+3.4e-05, +1.8e-04, +0.03785)` m，
即 `sim.reset()` 已经把整机抬起约 **37.9 mm**。这一项是真实存在、需要解释的（见第 3 节）。

## 3. P2 settle：FAIL，且已排除三个候选原因

零重力、60 步（0.5 s）内底盘轨迹（每 10 步采样）：

    reset0 z=0.037855 -> step1 0.043265 -> 0.048699 -> 0.104469 -> 0.162529
                      -> 0.222282 -> 0.283011 -> 0.343979   （单位 m）
    整段漂移 = 0.369020 m，单调上升，约 0.66 m/s

关键观察：**安装约束残差全程保持 1e-8 ~ 8e-8 量级**（xyzw 约定下逐采样点都在 1e-8），
即整机是"刚性整体匀速上浮"，不是约束被拉开。这一点把问题从"挂载"转移到"整机受到净速度/外力"。

本轮已经排除的候选原因：

| 候选 | 试验 | 结果 |
|---|---|---|
| 四元数顺序读错 | 两种约定对比 | **已排除**（读法错误只影响报告，不产生运动） |
| 链条自碰撞（代理上臂与 0.22 m 前臂相互穿透 0.14 m） | 第一版几何 vs 分离版几何 | **已排除**：漂移仅从 0.360954 变为 0.369020 m |
| 底盘压在地面以下（第一版 z∈[-0.10,-0.02]） | 抬到 z∈[0.02,0.10] | **已排除**：漂移仍为 0.35 m 量级 |
| 底盘与臂基座重叠 | 分离布局 | 已排除（两者本来就分离） |

**剩余根因：UNKNOWN**。下一个要测的假设（本轮未做，按 roadmap 要求不猜着改）：
1. 零重力下，以非刚体 Xform（`/Sandbox`）为 ArticulationRootAPI 的浮动 articulation，PhysX 是否退化成
   "运动学基座"或给根体写入非零速度；
2. `sim.reset()` 之后根体速度是否已非零（本轮只记了位置/姿态，**没记** `root_lin_vel_w` / `joint_vel` 在
   reset0 的值，这是本轮遗漏，下一轮第一件事就是补上）；
3. IsaacLab 每步 `write_data_to_sim()` 是否把某个内部速度缓冲重复写入根体。

## 4. body-only 授权方式：偏移被吞掉（FAIL，按设计）

| 授权方式 | t=0 安装位置残差（两种约定） | settle 漂移 |
|---|---|---|
| joint-local（FixedJoint 写 localPos0=偏移） | wxyz 0.2612 m / **xyzw 3.5e-08 m** | 0.369020 m |
| body-only（不写局部位姿） | **两种约定都是 0.238537 m**（＝|偏移|） | 0.001015 m |

即：不写 `physics:localPos0` 时，PhysX 把两个体的原点拉到一起，安装偏移被直接吞掉；settle 看起来更"稳"
（0.001 m）是被拉直后的假象。**结论（有证据）：固定关节必须显式授权局部位姿**，不能用体变换代替。

## 5. 本轮边界（未做/未碰）

* 未修改审计脚本（退出码缺陷仍在，已列必改项）；未修改真实 B601-RS 资产（本脚本根本不开它）。
* 未做 P3（平移/旋转挂载）：P2 settle FAIL 时按 roadmap 不得推进。
* 未做 P1 轮件身份（205.991 mm vs 180.782 mm 差异）——仍为 UNKNOWN，未因任何运行而改变。

## 6. 给 ChatGPT 的下一步（按优先级）

1. **退出码**：把 `close()` 之后 `os._exit(code)` 的写法推广到审计脚本，并加一条 subprocess 级断言：
   门禁 FAIL 的运行必须给出非零进程码。
2. **settle 根因**：在 reset0 / step1 / step2 记 `root_lin_vel_w`、`root_ang_vel_w`、`joint_vel` 与
   `robot.data.body_lin_vel_w`；若 reset 后根速度非零即可直接定位；若为零，则比较 ArticulationRootAPI 放在
   `/Sandbox`（Xform）、放在底盘刚体、以及不设 root API 三种情况的差别。
3. P2 settle PASS 之前不要做 P3/P4；P3 的旋转工况还需换掉 `write_root_velocity_to_sim`（上一轮已证角速度分量
   无效）。

## 7. 复现

    export PYTHONPATH=$(ls -d /home/T7/ojh/robot_sim/IsaacLab/source/* | tr '\n' ':')
    python morph_one_p2_init_diagnostics.py --variant joint-local --out out/p2_joint-local.json
    python morph_one_p2_init_diagnostics.py --variant body-only   --out out/p2_body-only.json
    # 运行包装（保留真实退出码，不用管道）
    python ... > out/x.stdout 2> out/x.stderr; echo "REAL_EXIT=$?"

证据：`outputs/simulation/rebot_b601_rs/morph_one/p2_b0853ef/`（两个变体 JSON、授权 USD、stdout/stderr、
`p0_exitcode_demo.txt`）。
