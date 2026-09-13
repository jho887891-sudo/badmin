# ALGORITHM_AUDIT.md — 羽毛球机器人算法审计（v0.1）

日期：2026-09-13　审计人：协调者（DSH 会话）
被审计范围：`src/badminton_brain/**`（Robot Brain 8 层 + 适配器 + 运行时）、`src/trajectory/**`、
`simulation/robots/badminton_robot/**`、以及资产阶段产出的物理/接触模型。

## 0. 审计方法与证据等级

本文档区分三类证据，**不把代理值当实测**：

| 等级 | 含义 | 本文档中的标注 |
|---|---|---|
| A | 协调者**本轮亲自实测**（工具输出可见） | 【实测】 |
| B | 模块作者自报 + 独立评审子智能体复算一致 | 【评审复算】 |
| C | 仅作者自报，未获独立复算 | 【自报】 |

另有统一的过程证据：**变异测试**（改坏实现看测试是否被杀）—— T2 8/8、T4 3/3、T5 2/2、T7 2/2、
T8 7/8（第 8 个为等价变异）、T9 2/2、T10 5/5。假 PASS 排查（死测试、空断言、条件 skip）已执行并修复。

全量回归：**26 套 / 463 项 / 0 失败**（wave 7）【实测】，四次连续一致（460/460/460/463）。

## 1. 算法清单总表

| # | 算法 | 位置 | 真实性 | 关键证据 | 主要风险 |
|---|---|---|---|---|---|
| 1 | 羽毛球二次阻力 ODE + RK4 | `src/trajectory/shuttle_aerodynamics.py` | 文献可溯源（Darbois-Texier 2012，L=6.5 m） | 与解析/细步解一致；被 4 个模块共享 | 无自旋/升力；L 非本项目实测 |
| 2 | 双目三角测量（DLT） | `perception/stereo_geometry.py` | 真实几何 | 合成往返误差 5.329e-15 m【实测】 | 标定全为 TEMP；深度误差随 z² 增长 |
| 3 | 亚像素质心（强度加权） | 同上 | 真实几何 | 高斯 patch 误差 1.554e-06 px【评审复算】 | 依赖阈值/信噪；无实际图像验证 |
| 4 | 合成探测器（针孔投影 + 确定性噪声） | `perception/synthetic_detector.py` | **代理**（非 YOLO、无权重） | 出视场=合法事件（D4/DEC-026） | **不得**用于声称感知精度 |
| 5 | 机器人定位 EKF（平面 6 维，Joseph） | `estimation/robot_localization.py` | 真实滤波器 | 噪声场景 0.0138 m vs 航位推算 0.0356 m【评审复算】 | 无 z/roll/pitch；IMU 线加速度未用 |
| 6 | 羽毛球 UKF（增广 7/10 维） | `estimation/shuttle_ukf.py` | 真实滤波器 | 位置 RMSE 0.00568 vs 原始测量 0.01424【评审复算】 | k 与航向风分量近简并；k 误差 ~11% |
| 7 | 轨迹预测（包装 rollout） | `prediction/physics_predictor.py` | 真实（直接调用冻结物理） | 与 rollout 逐点差 0.0；spy 证明是调用非照抄【评审复算】 | 地平线内未落地时 landing_point 是截断投影（有标志） |
| 8 | 击球可行性判据（规则集） | `decision/feasibility.py` | 真实规则 + 代理阈值 | 6 类否决 + canonical 接受；落点未知时挂起（DEC-027） | 阈值 17/21 为 TEMP；工作空间盒忽略底盘 yaw |
| 9 | 拦截点搜索（候选评分 + Hermite 插值） | `decision/intercept_search.py` | 真实搜索 | 最早可行点与解析入窗 t*=0.405 s 一致【评审复算】 | 无 IK/可操作性/迟滞；score 权重为 TEMP |
| 10 | 专家规划器（站位 + 一阶雅可比步） | `planning/expert_planner.py` | **近似**（明确非 IK/NMPC） | 舵轮限位精确裁剪 2.4 m/s【评审复算】 | 接触雅可比是 TEMP 代理；全身协同缺失 |
| 11 | PPO 策略 | `planning/ppo_policy_stub.py` | **未实现**（调用即抛错） | final 模式拒绝该占位 | 无观测/动作/奖励规格，不可训练 |
| 12 | Safety Shield（限幅 + 优先级） | `safety/safety_shield.py` | 真实限幅逻辑 | 6300 次随机扫描 0 越限【评审复算】；急停端到端可达【实测】 | 限值全 TEMP；无 QP/碰撞/人机分离/力矩 |
| 13 | 四舵轮 IK + S11 最小转向 | `simulation/.../morph_one/kinematics.py` | 真实运动学 | 轮速与手写 IK 精确相等（0.0）【评审复算】 | 几何（轮位/轮径）为 TEMP |
| 14 | 执行适配器（两种驱动模式） | `execution/sim_adapter.py` | 真实映射 | 与规范 kinematics 逐位一致；模式差异可测 | TEMP 几何；`SafeCommand.frame` 无法编码机体系（ISSUE-010） |
| 15 | 在线自适应（高斯-牛顿 + 限幅） | `adaptation/online_adaptation.py` | 真实慢环 | drag 收敛 1.24999999（真值 1.25），残差由真实物理生成 | 单方向残差下 drag 与沿速度风分量不可分辨 |
| 16 | 架构执行器（注册表/管线/校验） | `interfaces/registry/pipeline/validation` | 真实 | 层边界运行时强制；Safety 不可绕过【实测】 | 无（架构层本身未改动过） |

## 2. 逐算法详述与失效模式

### 2.1 羽毛球空气动力学（最底层，被 4 个模块共享）

- 模型：`p' = v`，`v' = g - k|v-w|(v-w)`，`k = 1/L = 1/6.5 = 0.153846 1/m`，RK4，内部步长可配（测试用 1 ms）。
- 单一真源：预测器、UKF、自适应、决策 travel model 全部 import 同一模块；**无任何一处重写 ODE**（评审 grep 确认）。
- 文献可溯源：`L` 与 `k` 标 `TRACEABLE_REFERENCE` 并注明来源，**明确不是本项目实测**。
- 失效模式：忽略自旋、升力、拍面摩擦与球速相关的恢复系数；风为零（TEMP）或由自适应估计。
- 结论：**可作为仿真基准**；真机标定前不可声称绝对精度。

### 2.2 双目几何（三角测量 / 亚像素质心 / 合成探测器）

- 三角测量为 DLT，输入需经 ROI 与有效对过滤；误差与 z²σ/(f·b) 同阶。
- 代理标定（fx=700 px、b=0.29 m、σ=0.5 px）下的深度 RMS 约 23/47/78 mm @1/2/3 m【自报】——
  **这是设计输入而非精度声明**：真实精度必须用标定后的内参/基线重算。
- 失效模式（已按 DEC-026 治理）：出视场、遮挡、曝光不足一律走**哨兵**（`valid=False` + NaN），
  不再抛异常击穿闭环；`ShuttleMeasurement.valid_mask`（DEC-023）把该语义贯穿到估计层。
- 未实现：真实检测器（YOLO）、真实立体匹配/整流、时间同步。

### 2.3 状态估计（EKF + UKF）

- EKF：平面 6 维 `[x,y,yaw,vx,vy,wz]`，Joseph 形式更新，yaw 新息跨 ±π 环绕；
  协方差侧有**定量**断言（`P == FPFᵀ+Q`、`P−APAᵀ == KRKᵀ`、Jacobian 符号双向固定）【评审复算】。
- UKF：增广 `[p,v,k(,wind)]`，过程模型复用冻结 `rk4_step`（与它逐位一致 ≤1e-12）；
  测量更新使用报文自带的协方差；reset(env_ids) 隔离逐位成立。
- 可辨识性限制（作者已写入文档串，审计确认）：单一速度方向下，**阻力系数 k 与沿速度方向的风分量近简并**，
  因此不能在单次 1 s 飞行内同时唯一辨识二者；侧向风分量可辨识（实测误差 0.165 m/s vs 沿航向 0.863 m/s）【自报】。
- 失效模式：出视场若不被掩码过滤会把状态拉向原点（**已修**，DEC-023）；IMU 线加速度未被使用（需扩 bias 状态）。

### 2.4 预测 → 决策 → 规划 → 安全 → 执行 链条

统一的坐标与时间纪律（审计重点）：

- **Court Frame 唯一**：所有消息 `frame='court'`，`env_origin` 不参与任何计算（全仓 grep 仅注释提及）。
- **时间基**：`PredictedTrajectory.times` 与 `arrival_time` 为**绝对仿真时间**（DEC-016）；
  `BestIntercept.time_s` 亦为绝对；规划器自行换算 `t_go = time_s − state.timestamp`（D1 修复，
  实测 now=3 与 now=12 输出逐位相同）。
- **base_twist 是机体系**（DEC-014），court→body 旋转由规划器完成；执行层原样送四舵轮 IK。
  已知代价：`SafeCommand.frame` 仍为 'court'，该差异只能文档化 + 护栏（ISSUE-010）。

各环节的算法性质与风险：

- **预测**：真实物理；地平线截断时 `landed_within_horizon=False`（DEC-019 契约字段），
  决策层必须消费该标志（D2 修复后已消费，实测 last_z=1.18 m 不再谎报出界）。
- **可行性门**：规则集（有效性/新鲜度/球速上下限/方向/出界/责任区/工作空间+时间）。
  球速上限采用**窗首样本（来球速度）**而非窗内峰值（DEC-025，避免自由落体末速误判高吊）；
  方向门用短窗 median(vx)（S19）；落点未知时**挂起落点判据**并返回 WAIT 语义枚举（DEC-027）。
- **拦截搜索**：在预测栅格上取候选、按公开 score 公式选优，确定性（位级可复现），
  无可行点返回 `None` 且不填假值；**未做** IK/可操作性/roll 搜索/迟滞。
- **规划器**：站位速度 + 偏航率 + 一阶（Moore-Penrose）关节步；接触雅可比是**显式 TEMP 代理**，
  **不是 IK/NMPC**；舵轮限位用等比例精确裁剪（饱和位移 = max_wheel_speed·R = 2.4 m/s）。
  转向限位在默认 π 下是恒真命题（作者已如实标注，强制归 Safety）。
- **Safety Shield**：优先级 ESTOP > PROTECTIVE_STOP > CONTROLLED_STOP > HOLD > PROJECT > PASS；
  硬关节限位永不违反；工作空间 HOLD 目标须逐个通过 FK 盒校验（D5/F2 修复）。
  已知取舍：经校验的 HOLD 可能超过 `dq_max·dt` 一步（冻结/退避优先于软速度投影，已文档化）。
- **执行适配器**：两种驱动模式（BODY_TWIST_ACTUATOR / STEER_DRIVE_WHEEL_MODEL），
  wheel 模式使用 S11 最小转向解与舵角记忆；`Feedback.tracking_residual` 与 `prediction_error` 分离（DEC-015）。

### 2.5 在线自适应

- 高斯-牛顿步 + 每步上限 + 边界限幅；残差优先用显式 `prediction_error`（米），否则由应用层推入的预测计算。
- 审计发现并已修正的真实缺陷：drag 雅可比多乘 `k_total` 导致**实际增益 = gain/θ**（D1）；
  delay 雅可比符号错误（应 `+v`，自证型测试曾与错误互相印证）——修复后由真实 RK4 生成残差验证收敛【评审复算】。
- 当前闭环状态：**应用层推入预测**（`FullBrainRuntime.set_prediction`）后慢环真正更新（`updates=[24,24]`）【评审复算】；
  但自适应输出**目前无 in-loop 消费者**（模块已导出 `consumer` 字段说明为离线环预留）——这是**未闭合的一段**。

## 3. 横向问题（按优先级）

| 优先级 | 问题 | 影响 | 建议 |
|---|---|---|---|
| P0 | 未实测参数共 33 项（`final` 实测 `ok=False, errors=33`）——相机内外参、Morph One 几何/质量、球拍实测值与 `T_link6_tcp`、底盘/机械臂限值、接触对参数 | 上线不放行；所有绝对精度结论不成立 | 先做标定与称重/测量，再用 `Param(..., VERIFIED_MEASURED)` 替换 |
| P1 | **机器人能力模型未收敛为单一真源**（ISSUE-011）：工作盒/限值在 decision/planning/safety 各写一份 | 换实测参数时易漏改，判据互相矛盾（决策认为可打、安全却 HOLD） | 抽 `robot_capability` 单一真源，三方 import |
| P1 | `docs/architecture/MODULE_INTERFACES.md` 缺失（ISSUE-006） | 消息字段集属**临时契约**，字段名可能需调整 | 补该文档或确认现有字段 |
| P2 | 自适应输出无消费者（闭环最后一段未闭合） | 参数修正不会反馈给预测/估计 | 在估计/预测侧消费 `drag_scale/wind/delay`（注意可辨识性限制） |
| P2 | 时间基「自动判别」启发式（ISSUE-012） | 静默降级为「永远可行」的危险失效模式 | 改为显式不变量检查 |
| P3 | L=6.5 三处副本（ISSUE-013）；`UnifiedState.base_twist` 机体系未进契约（ISSUE-014）；`odom_twist` 命名债（ISSUE-009）；reason 词表双份（ISSUE-007） | 漂移风险 | 按各自 ISSUE 收敛 |

## 4. 明确**未实现**的算法（避免误读）

- 真实感知：YOLO/目标检测、立体匹配与整流、相机时间同步、真实标定。
- 真机执行：Morph One/PiPER 驱动、通信、真实 IMU/编码器接入。
- 机械臂逆运动学 / NMPC / 全身协同 / 拍面姿态控制（规划器只做一阶代理）。
- PPO 训练（占位，调用即抛错）；观测/动作/奖励规格未定。
- Safety 的投影 QP、预测性碰撞、速度感知动态包络、围栏、人机分离、力矩限。
- 击球后的回合管理、迟滞/稳定选择、多目标优化。

## 5. 复现命令

```bash
cd /home/T7/ojh/robot_sim
./env_isaaclab/bin/python tools/run_all_tests.py            # 全量：26 套 / 463 项
./env_isaaclab/bin/python tests/badminton_brain/test_full_brain.py   # 端到端集成（11 项）
./env_isaaclab/bin/python -c "import sys;sys.path.insert(0,'src');\
from badminton_brain.apps.full_brain import build_full_brain;\
from badminton_brain.validation import validate_architecture;\
import numpy as np;\
t=lambda n,s: np.stack([np.array([5.2,0.3,2.1])+np.array([-8,-0.1,1.2])*s for _ in range(n)]);\
r,p=build_full_brain(num_envs=2,truth_provider=t);\
print(validate_architecture(r,p.config,mode='final').ok)"   # 应为 False（未实测参数）
```

契约 sha256 基线：`outputs/reports/contracts.md5`（含「哪个 DEC 改了哪个契约文件」的说明）。
验收报告：`outputs/reports/brain_modules_acceptance.md`（15 节）。

## 6. 审计结论

1. **数值与逻辑正确性**：核心算法（空气动力学、几何、EKF/UKF、运动学 IK、限幅安全）在**合成与代理参数**下
   通过了独立复算与变异测试；未发现仍在生效的正确性缺陷（本次审计发现并已修复 8 处，含一处阻断级 NaN 逃逸、
   一处坐标系错误导致 23% 轮速偏差、一处急停死代码）。
2. **可信度边界**：所有**绝对**精度/能力结论（多远能接到、多快能挥、感知多少毫米）**都尚未成立**，
   因为它们依赖 33 项未实测参数；`final` 模式会拒绝放行，这是设计而非缺陷。
3. **闭环完整度**：感知→估计→预测→决策→规划→安全→执行 已闭合并可端到端运行；
   **自适应回注（最后一段）尚未闭合**，且存在 P1 级的「能力模型未单一真源」与「接口文档缺失」。
