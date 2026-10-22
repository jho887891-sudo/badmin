# P7-11 / P7-12 仿真闭环接线（simulation-only）

本文件是 P7-11（把**验证过的 FK** 接进可达性模块）与 P7-12（把闭环 harness 端到端接通）的交付说明。
分支：`p7/closed-loop-wiring`（worktree：`E:\具身智能\badmin_wt_p7d`，基线 `p7/reachability-fsm @ adaee81`）。
无训练、无 GPU、无 Isaac、无真机；全部离线 numpy 计算。

---

## 1. 接线图（数据流）

```
场地真值球（解析，非物理引擎）
  │
  ├─(1) 检测接口  perception/synthetic_detector.SyntheticStereoDetector.detect()
  │        → SyntheticDetection(uv_left, uv_right, valid)          [像素域]
  ├─(2) 三角化   perception/stereo_geometry.triangulate()
  │        → 三维场地坐标点 (3,)                                    [2D→3D 的唯一入口]
  ├─(3) 估计接口  estimation/shuttle_ukf.ShuttleEstimatorBridge.update(ShuttleMeasurement)
  │        → 位置/速度/协方差（fuse_velocity=False：一帧只测位置）
  ├─(4) 预测接口  prediction/physics_predictor.PhysicsTrajectoryPredictor.predict()
  │        → PredictedTrajectory(times, position, velocity)
  │        拦截点规则：t ≥ now+0.45 s 且 z ≤ 1.0 m 的首个样本（HIT_PLANE_CROSSING），
  │        否则取最接近 z=1.0 m 的样本（NEAREST_TO_HIT_PLANE）
  │        接近方向：预登记的“回球方向” +X（DECLARED_RETURN_DIRECTION）
  │
  ├─(5) 安全 FSM  safety/interaction_fsm.InteractionFsm
  │        on_frame → on_prediction → on_plan → on_execute_command
  │        任一阶段失败即锁死 HOLD（不可自动恢复）
  │
  ├─(6) 可达性判定  reachability/reachability.assess_reachability()
  │        solver = 所选 backend 的阻尼最小二乘 IK（FK 与 Jacobian 都来自同一 backend）
  │        limits = kinematic_limits_for(backend)（DOF/关节限位/速度上限随 backend）
  │
  └─(7) 球拍碰撞代理  P6 已提交 JSON（43 个 primitive）按文档规则摆到规划位姿
           → SIM_PROXY_GEOMETRY_ONLY 间隙/接触判定（几何，不是物理接触）
```

关键约束（代码里强制）：

* **一个 episode 只允许一个 backend**。每条判定记录 `backend`、`backend_source_class`、`frame_name`；
  `summarise()` 与单次运行都会在发现两个 backend id 时抛 `BackendMixError`。
* **要求 verified 却拿不到时直接报错**，绝不静默回退到 proxy DH：`VerifiedFkUnavailable`（含 `VERIFIED_FK_UNAVAILABLE` 前缀与查找路径）。
* **2D→3D 拒绝**不变：`FrameInput(detection_dim=2, position_m=...)` 直接抛 `DetectionDimensionalityError`；
  2D 帧可以进入 TRACK，但 `on_prediction` 一定以 `DETECTION_NOT_3D` 锁 HOLD；`assess_reachability(detection_dim=2)` 同样抛错。
* **不报命中率**：`hit_rate` 恒为 null，`real_hit_rate_claimed=False`，即使一次运行完整跑完也如此。

---

## 2. 两个 backend：同接口、不同的“真实”

| | `ProxyDhBackend` | `VerifiedFkBackend` |
|---|---|---|
| 实现 | P7-09 `reachability/kinematics.py` 的 ENGINEERING_V0_1 proxy DH | **P6 真 FK**：运行时从 P6 checkout 载入 `src/simulation/rebot_b601_rs/kinematics/fk.py`（不复制代码） |
| DOF | 6（全转动） | 8（6 转动 + 2 个夹爪移动副） |
| 判定所针对的帧 | `racket_face_proxy`（proxy 球拍面中心，link6 + 0.50 m） | `gripper_end`（P6 已验证的臂端帧） |
| 关节限位 | PiPER 公开范围（未实测） | P6 `joint_anchors.json` / `robot_usd_joint_chain.json` 的 as-authored 限位 |
| 数据来源标记 | `PROXY_DH_ENGINEERING_V0_1` | `VERIFIED_ASSET_KINEMATICS`（记录 fk.py 的 sha256） |
| 解析可达半径 | 1.266 m | 0.959 m |

**最重要的诚实结论**：P6 只验证了臂端（gripper_end），**球拍面 / 球拍安装帧仍然是 UNKNOWN**
（`frame_tree.json` 的 edge 13/14/15 没有位姿）。所以 verified backend 回答的是
“臂端能不能到”，**不是**“拍面能不能到”；两者不可直接相减当作精度差。

---

## 3. 数字对比（真实输出，见 `outputs/badminton_brain/p7_closed_loop/backend_compare.json`）

命令（真实退出码 0）：

```powershell
$env:BADMIN_P6_ROOT = "E:\\具身智能\\badmin_wt_p6fk"   # p6/dynamic-fk @ a0a0fb3 的 worktree
python tools\p7\run_p7_closed_loop_wiring.py compare
# EXIT=0
```

预登记目标集：6 个点 × 2 个接近方向 × 2 个 backend = 24 次判定；时间预算 0.448 s；
每个 backend 用同一套声明多起点种子（harness 姿态 + mid-range）。

| 指标 | proxy_dh | verified_p6_fk |
|---|---|---|
| 判定计数 | `NOT_REACHABLE:IK_NOT_CONVERGED` 12/12 | `REACHABLE:IK_CONVERGED` 1/12，``NOT_REACHABLE:IK_NOT_CONVERGED` 11/12 |
| IK 位置残差（收敛样本） | 无（0 个） | min/max/p50 = **2.121e-06 m**（tol 1e-04 m） |
| 解析可达半径 | **1.266 m**（Σ(|a|+|d|)+0.50 m 工具） | **0.959 m**（P6 q=0 链上相邻帧距离之和） |
| 采样可达半径（4000 次，种子 20261022） | 1.2457 m | 0.9071 m |
| 两者判定不一致的用例数 | 1（(0.36,0.05,0.45) + 接近方向 +X：verified 可达、proxy 不可达） | — |

解读（不夸大）：

* 24 次判定里 23 次都是 IK 不收敛。这说明两套运动学在**真实拦截点线**（x≈0.35–0.46 m、z=0.45–1.35 m）
  上都到不了：proxy 的 1.266 m 与 verified 的 0.959 m 都是**包围球上界**，不是可达集；
* 唯一收敛的一例是低点 (0.36,0.05,0.45) 的 verified 判定，残差 2.1e-6 m——说明 adapter 用的是真 FK
  （否则不可能在 8 自由度真链上收敛到微米级）；
* 差异的根本原因是 **帧不同（拍面 vs 臂端）、DOF 不同（6 vs 8）、限位不同**，不能简化成“proxy 偏大/偏小”。

---

## 4. 闭环运行（P7-12）：能报的指标，不报命中率

命令（真实退出码 0）：

```powershell
$env:BADMIN_P6_ROOT = "E:\\具身智能\\badmin_wt_p6fk"
python tools\p7\run_p7_closed_loop_wiring.py all
# wrote ...backend_compare.json
# wrote ...wired_episodes.json
# EXIT=0
```

9 个 episode（3 场景 × 3 种子）×2 个 backend，标定用**假定的单位阵**（`Calibration(court_from_base=I, verified=True)`，
只为跑通链路；这不是标定，报告里写死了 ASSUMED）：

| | proxy_dh | verified_p6_fk |
|---|---|---|
| 结束于 HOLD 的 episode | 9/9 | 9/9 |
| HOLD 原因 | `UNREACHABLE_TARGET` 9 | `UNREACHABLE_TARGET` 6、`COMMAND_OVER_CAP` 3 |
| 可达性判定 | `NOT_REACHABLE:IK_NOT_CONVERGED` 9 | `REACHABLE:IK_CONVERGED` 12、`NOT_REACHABLE:WORKSPACE_BOUND` 6 |
| 执行的指令数 | 0 | 3 |
| SIM_PROXY 几何评估 / 接触 | 0 / 0 | 12 / 12 |
| `hit_rate` | **null** | **null** |

为什么 proxy 一次都到不了 EXECUTE：它是对**代理拍面**做 IK，在预测拦截点（z≈0.40–0.75 m、x≈0.35–0.46 m）
上不收敛；verified 在同样点上收敛（残差 ~1e-5 m），于是能进入 PLAN/EXECUTE 并产生几何评估。

`COMMAND_OVER_CAP` 是**接线本身**的发现，不是 FSM 的问题：harness 每帧从当前位置重解一个新位姿，
没有轨迹生成器，所以第二帧的集点跳变超过了 P5-05 的 2.0 rad/s 上限，FSM 正确地拒绝执行。

### SIM_PROXY_GEOMETRY_ONLY 是什么、不是什么

* 是：把已提交的 43 个 primitive（`racket_collision_proxy.json`，sha256 记入报告）按文档规则
  摆到规划位姿——即“拍面中心 = 规划点”，`racket_root = face − 0.5472 m · z_axis`
  （0.5472 m 由 JSON 里 string_bed 盒中心推出，不是编的），然后算球心到表面的最小有符号距离，减去声明的
  球半径 0.030 m；最近 primitive 必须是 `string_bed` 才算“拍面接触”。
* 不是：物理接触。没有质量、没有恢复系数、没有线床形变、没有摩擦，也没有被测的球半径。

---

## 5. 测试（纯 pytest，真实退出码）

```powershell
$env:BADMIN_P6_ROOT = "E:\\具身智能\\badmin_wt_p6fk"
python -m pytest tests\badminton_brain\test_p7_closed_loop_wiring.py -q
# 32 passed  EXIT=0

python -m pytest tests\badminton_brain -q
# 444 passed, 1 skipped, 12 subtests passed  EXIT=0     (基线 412 + 新增 32)
```

覆盖点：

1. backend 选择与**响亮的失败**（`fk_path` 指向不存在的文件 → VerifiedFkUnavailable`；
   无候选 root 且无环境变量 → 同样抛；harness 里的 verified 请求会原样上抛）；
2. 双 backend 判定对比（同一目标集，判定里带 backend id / solver id，收敛解的残差 ≤ 容差）；
3. 2D→3D 拒绝仍成立（FrameInput、assess_reachability、2D 帧进 PREDICT 阶段锁 HOLD）；
4. HOLD 在每种注入故障下锁死：缺标定、过期帧、协方差超限、时间倒流、2D、不可达、指令超速；
   锁死后所有阶段 SUPPRESSED、指令被 `CommandSuppressed` 拒绝、过期 reset 被 `ResetRejected` 拒绝；
5. harness 拒绝命中率（缺标定 → `REFUSED` + hit_rate=null，但仍给出真实 FSM 产生的 HOLD 原因/状态计数）；
6. 已提交的 racket proxy JSON 能载入（43 个 primitive、sha256、由几何推出的拍面中心），
   且几何检查能**拒绝**离球 0.8 m 的位姿；被截断/异 schema 的 JSON 会被拒。

**重要**：需要 P6 checkout 的测试（verified backend、racket proxy JSON）在没有 checkout 时
**失败而不是 skip**——测试文件里 `require_p6()` 会带着它找过的路径抛 AssertionError。
这是刻意的：静默跳过的安全测试等于不存在。

---

## 6. UNKNOWN 清单（必须随结果一起引用）

1. **没有标定过的 TCP**：P6 只验证了 authored `gripper_end`，球拍面/安装帧仍是 UNKNOWN
   （`frame_tree.json` edges 13/14/15）。verified backend 的判定**不是**球拍面的判定。
2. **没有 D455 标定**：stereo 用的是 ENGINEERING_V0_1 合成基线；`Calibration(verified=True)`
   是**假定**的单位阵，只为把链路跑通。真实外参未知 ⇒ 一切“场地坐标”结论都带这个前提。
3. **没有物理接触模型**：几何间隙 ≠ 接触；球半径 0.030 m 是声明的代理值，不是测量值；
   线床被当作实心膜（沿用 P6 proxy 的声明）。
4. **没有真实命中率**，也不打算有：`hit_rate` 永远 null。
5. **没有轨迹生成器**：每帧重解位姿，会在第二帧触发 `COMMAND_OVER_CAP`；这不是控制的终点。
6. **IK 不收敛 = 拒绝**，不是“差不多”：proxy 在 12/12 目标上拒绝，verified 在 11/12 上拒绝；
   可达半径 1.266 / 0.959 m 是**包围球上界**，不代表可达集。
7. **没有真机、没有 Isaac、没有 GPU**：本切片全部是 numpy 离线计算。

---

## 7. 文件与产物

| 路径 | 说明 |
|---|---|
| `src/badminton_brain/reachability/backends.py` | 两个 backend + 通用 DLS IK + 显式选择与拒绝 |
| `src/badminton_brain/reachability/reachability.py` | 仅新增 `dof` / `joint_names` 字段（默认 6，旧行为不变） |
| `src/badminton_brain/harness/wired_closed_loop.py` | 端到端接线 harness + 已提交 proxy 的几何检查 |
| `tools/p7/run_p7_closed_loop_wiring.py` | 证据脚本（compare / episodes），退出码 0 / 2 / 3 |
| `tests/badminton_brain/test_p7_closed_loop_wiring.py` | 32 个测试 |
| `outputs/badminton_brain/p7_closed_loop/backend_compare.json` | 双 backend 判定对比（含 sha/路径/可达半径） |
| `outputs/badminton_brain/p7_closed_loop/wired_episodes.json` | 9+9 episode 的完整报告（含逐条 FSM 事件） |

## 8. 如何复现

```powershell
git -C "E:\\具身智能\\badmin_project" worktree add --detach "E:\\具身智能\\badmin_wt_p6fk" p6/dynamic-fk
$env:BADMIN_P6_ROOT = "E:\\具身智能\\badmin_wt_p6fk"
cd "E:\\具身智能\\badmin_wt_p7d"
python -m pytest tests\\badminton_brain -q
python tools\\p7\\run_p7_closed_loop_wiring.py all
```

P6 artifacts 的定位规则（无隐式回退）：`$BADMIN_P6_FK_PATH` → `$BADMIN_P6_ROOT` → 本仓库`；
只有真的存在 `src/simulation/rebot_b601_rs/kinematics/fk.py` 的 root 才算候选。
