# P4-11 轮速指令接口（wheel-drive command interface）+ 失败即关闭校验

**分支**：`p4/command-interface»（worktree 自 `p3/gate-v1.1 @ 5e4e92e»，PR #35）
**定级**：**P4_PROXY**（沿用 P4-08/09 的圆柱踏面代理）。**不是 P4_REAL**。
**范围**：只新增「机体系指令 → 四轮关节速度目标」这一层的接口与测试 + GPU 脚本的失败即关闭前置校验；
**未改动任何既有门禁、任何阈值、任何 run 目录、PHASE_P4_08/09 文档、队列状态文件**。

---

## 1. 契约（contract）

文件：`scripts/simulation/p4_command_interface.py»（**纯标准库**：不 import numpy/torch/Isaac/USD）
测试：`tests/simulation/test_p4_command_interface.py»

### 1.1 `WheelCommand» 字段

| 字段 | 含义 | 单位/取值 |
|---|---|---|
| `wheel_targets_rad_s» | 四轮关节速度目标（顺序 = P1 v2 轴位表 wheel1..wheel4） | rad/s，维度必须 == 4 |
| `speed_m_s» | 直行速度（**>0 表示沿 -Y 前进**） | m/s |
| `travel_direction» | 由 `speed_m_s» 推出的方向 | `"-Y"» / `"+Y"» / `"none"» |
| `yaw_rate_rad_s» | 原地转向偏航角速度 | rad/s（>0 = 从 +Z 看逆时针） |
| `limit_mode» | **硬性**限幅策略，**无默认值可猜** | `"clamp"» / `"reject"» |
| `radius_m» | 滚动半径 | m |
| `requested_speed_m_s» / `requested_yaw_rate_rad_s» | 限幅**前**的原始请求（留证） | m/s, rad/s |
| `clamped» | 是否真的被钳位 | bool |

构造后**必合法**：所有校验在 `__post_init__» 一次性完成，不存在「半合法」对象。

### 1.2 构造器

- `WheelCommand.from_speed(v_m_s, radius_m, limit_m_s, limit_mode)»
  → 四轮同号 `omega = v / r»；`clamp» 把幅值钳到 limit 并置 `clamped=True»；`reject» 抛错。
- `WheelCommand.from_yaw_rate(yaw_rate_rad_s, radius_m, limit_rad_s, limit_mode, half_track_m=0.250002)»
  → `omega_i = -(psi_dot * x_i) / r»，`x_i» 取自 P1 v2 轴位表。
  `half_track_m» **不允许**改成别的值（会与轴位表自相矛盾，直接抛错）。
- `WheelCommand.from_audit(audit)» → audit 字典回转成等价对象（审计闭环）。

### 1.3 `audit()»

返回可直接塞进 run manifest / `result.json» 的纯 JSON 字典（schema `p4_11_wheel_command/v1»），
含：四轮目标（键名与 GPU 的 `wheel_targets_rad_s_per_wheel» 一致）、限幅模式、钳位标志、方向、
机体系速度、半径、半轮距、**符号约定块**、限值块、**GPU 证据指针**、**UNKNOWN 列表**。

---

## 2. 符号 / 轴向约定（GPU 已证，禁止「修正」）

1. 四个轮关节都绕**世界 +X** 轴（P1 v2 PCA 0.0043° 验证）。
2. **+omega 绕 +X → 底盘沿 -Y 前进**（接地点向 +Y 走，底盘被反推 -Y）。**滚动方向是 Y，不是 X。**
3. 直行需要**四轮同号**。P1 v2 的 `+1/+1/-1/-1» 是**电机通道**约定（左右各乘一次符号），
   **不是关节轴约定**；把它当关节目标会让底盘偏航 1.915 rad 而不平移（`P4p_fwd_r02» 实测）。
4. 接口的 ±速度直接对应 GPU 的前馈式：GPU 里 `omega_ff = -vy_target / r»、`vy_target = -v_cmd»，
   等价于本接口的 `omega = +v / r»（v>0 → -Y）。
5. 转向：横向偏置 `x_i = ±0.250002 m» 的轮需要局部表面速度 `psi_dot * x_i»，故
   `omega_i = -(psi_dot * x_i) / r»。

### 2.1 GPU 证据（已提交，逐条可核）

| 证据 | 记录内容 | 与接口的一致性 |
|---|---|---|
| `P4p_rollneg_r02» | targets `[4,4,4,4]» → `chassis_displacement_m[1] = -1.2147367» | 四轮同号 → **-Y** ✅ |
| `P4p_rollpos_r02» | targets `[-4,-4,-4,-4]» → `+1.215509057» | 反号 → **+Y** ✅ |
| `P4p_turn_r03» | targets `[3,3,-3,-3]» → yaw 0 → `-1.417513835» rad | 由 `psi_dot<0» 反推得到同一数组（1e-6 内）✅ |
| `P4V_030_r10» | cmd 0.30 m/s → `dY = -1.041028857», `dtheta = +10.0358» rad | 符号与前馈一致 ✅ |
| `P4V_clamp_r11» | cmd 0.80 / limit 0.50 → `commanded_after_clamp_m_s = 0.50» | `clamp» 语义一致 ✅ |

「红/绿」测试：同一构造器把速度取反，**四轮目标与行进方向同时反号**
（`test_red_green_flipping_the_speed_sign_inverts_travel_direction»）。

---

## 3. 失败即关闭（fail closed）：typed errors，绝不静默修复

全部异常继承 `CommandValidationError»，带稳定 `.code»，调用方可分支而不用匹配字符串。

| 触发条件 | 异常 | `code» |
|---|---|---|
| NaN / ±inf 速度、偏航率、轮目标 | `NotFiniteError» | `not_finite» |
| 半径 ≤ 0 | `InvalidRadiusError» | `invalid_radius» |
| limit ≤ 0 | `InvalidLimitError» | `invalid_limit» |
| limit 为 NaN / inf | `InvalidLimitError» | `invalid_limit» |
| `limit_mode» 不在 {clamp, reject} | `InvalidLimitModeError» | `invalid_limit_mode» |
| 轮目标个数 ≠ 4（含字符串、标量） | `InvalidWheelCountError» | `invalid_wheel_count» |
| `reject» 且 `abs(v) > limit» | `SpeedLimitExceededError» | `speed_limit_exceeded» |
| `reject» 且 `abs(yaw) > limit» | `YawLimitExceededError» | `yaw_limit_exceeded» |
| 轮目标超出 50 rad/s 且落在 rpm 带 | `UnitConfusionError» | `unit_confusion» |
| 轮目标任何单位都不可能（>477.5 rad/s） | `WheelTargetsOutOfRangeError» | `wheel_targets_out_of_range» |
| 缺失某个 `joint_wheelN» | `InvalidWheelCountError» | `invalid_wheel_count» |

### 3.1 单位陷阱防护（documented ranges）

上限来源：GPU 脚本的 actuator `velocity_limit = 50 rad/s»、几何半径 0.103 m、半轮距 0.250002 m
→ 车速天花板 **5.15 m/s**、偏航天花板 **20.60 rad/s**。

| 量 | 判定区间 | 处置 |
|---|---|---|
| 速度 | 5.15 < 幅值 ≤ 5150 | `UnitConfusionError»，报「看起来是 mm/s」，并给出正确值（300 → 0.300 m/s） |
| 速度 | 幅值 > 5150 | `UnitConfusionError»，任何单位都不可能 |
| 半径 | 10 ≤ r ≤ 1000 | `UnitConfusionError»，报「看起来是 mm」（103 → 0.103 m） |
| 半径 | r < 0.005 或 > 5 | `UnitConfusionError»（超出本底盘文档区间） |
| 偏航率 | 20.60 < 幅值 ≤ 1180 | `UnitConfusionError»，报「看起来是 deg/s」 |
| 轮目标 | 50 < 幅值 ≤ 477.46 | `UnitConfusionError»，报「看起来是 rpm」 |

**单位防护在 clamp 与 reject 两种模式下都会抛**：钳位会把 1000 倍的单位错误藏起来，这正是要禁止的。

---

## 4. 失败即关闭的集成路径（GPU 脚本）

`scripts/simulation/p4_proxy_wheel_drive.py» 新增（**仅新增，不改任何门禁/阈值**）：

1. 顶部 `from p4_command_interface import ...»（在 `AppLauncher» 之前，模块是纯标准库，可安全导入）。
2. 新增参数 `--limit-mode {clamp,reject}»，**默认 `clamp»** = P4-09 原有行为（钳位），
   因此既有命令行与既有门禁数值一字不变。
3. `# P4-11 PREFLIGHT BEGIN ... END» 代码块，位置在 `ARGS = PARSER.parse_args()» 之后、
   **`LAUNCHER = AppLauncher(ARGS)» 之前**：
   - 调用 `WheelCommand.from_speed(abs(velocity_cmd), nominal_radius, speed_limit, limit_mode)»；
   - 任何 `CommandValidationError» → 打印 JSON（`aborted: true», `sim_steps: 0», `stage_authored: false»,
     `root_writes: 0»）+ `GATE_RESULT=ABORTED»，然后 `raise SystemExit(2)»；
   - 因此 **USD 舞台没被 author、`SimulationContext» 没被创建、`sim.step()» 一次都没走**。
4. 闭环块里加一条一致性硬检查：接口算出的有效速度必须与脚本自己的内联钳位
   `min(abs(v_cmd), speed_limit)» 在 1e-12 内相等，否则 `SystemExit»（防止「校验的指令 ≠ 执行的指令」）。

### 4.1 真命令 + **真退出码**（本机实跑）

```powershell
# 1) P4-11 测试套件
python -m pytest tests/simulation/test_p4_command_interface.py -q
# 73 passed in 0.74s
# exit code = 0

# 2) reject 前置校验当作独立进程跑（用 GPU 脚本里同一段代码，verbatim）
python _scratch_preflight_demo.py reject
# {"run_id":"P4I_demo","profile":"velocity_control","aborted":true,"limit_mode":"reject",
#  "reason":"SpeedLimitExceededError","error_code":"speed_limit_exceeded",
#  "detail":"speed 0.8 m/s exceeds limit 0.5 m/s with limit_mode='reject': abort before any
#            simulation write ...","stage_authored":false,"sim_steps":0,"root_writes":0}
# GATE_RESULT=ABORTED
# exit code = 2   <-- 非零，且在任何仿真写入之前

# 3) clamp 前置校验（默认行为，不中止）
python _scratch_preflight_demo.py clamp
# PREFLIGHT_OK limit_mode=clamp effective=0.500000 m/s clamped=True
# exit code = 0

# 4) 相邻回归（未受影响的 P2/P3 门禁测试）
python -m pytest tests/simulation/test_p2_chassis_world_frame.py tests/simulation/test_p2_landing_gates.py ^
    tests/simulation/test_p3_motion_gate.py tests/simulation/test_p3_motion_gate_track.py -q
# 38 passed, 8 subtests passed in 1.77s
# exit code = 0
```

---

## 5. GPU 交叉核对（接口能否复现 GPU 真正跑过的东西）

**真实数据**：`outputs/simulation/rebot_b601_rs/morph_one/p2_wc_gates/» 下已提交的 P4V/P4p/P4f `result.json»
（测试里**没有**任何 mock / 合成 JSON；证据目录缺席时测试会 **FAIL** 而不是跳过）。

| 核对项 | 结果 |
|---|---|
| `P4V_030_r10 / P4V_050_r09 / P4V_clamp_r11» 的 `wheel_targets_rad_s_per_wheel» 由接口重算 | **[0,0,0,0] 精确相等**（velocity_control 的 profile 目标表全零、闭环每步另建张量；测试里写明了这一点，不冒充非零核对） |
| `commanded_speed_m_s + nominal_radius + limit 0.50» → 接口有效速度 vs 记录的 `commanded_after_clamp_m_s» | **≤ 1e-6**（4 个 P4V run 全过，含 `0.80→0.50» 的 clamp run，「clamped」布尔一致） |
| 记录的 `measured_rolling_radius_m» 由 `abs(dY)/abs(dtheta)» 重算 | **最大偏差 4.93e-7 < 1e-6**（`P4V_030_r10» wheel4；`P4V_050_r09»/`clamp_r11» 为 4.79e-7） |
| `P4p_turn_r03» 的 `[3,3,-3,-3]» 由 `from_yaw_rate(psi_dot = -3*0.103/0.250002, r=0.103)» 反推 | **≤ 1e-6**，且方向 `-Z» 与实测 yaw `-1.4175 rad» 同号 |
| `P4p_rollneg_r02» / `P4p_rollpos_r02» 的 targets 由 `from_speed(omega*r)» 重算 | **≤ 1e-6**，且预测的 ±Y 与实测 `dY» 同号 |
| 所有已提交 P4V `manifest.json» 里的命令行过新前置校验 | 8/8 不中止；且接口算出的有效速度与记录值 ≤ 1e-6 |

**结论**：接口不是「另一套数学」，它复现了 GPU 真实跑过的目标值（零目标精确、非零目标 1e-6）、
钳位语义（1e-6）与滚动半径关系（4.93e-7）。

---

## 6. UNKNOWN / 明确边界

1. **GPU 未验证**：新增的 `--limit-mode reject» 前置校验**从未在 Isaac 里跑过**（本机无 GPU/Isaac）。
   已验证的只是：同一段代码文本被抽出后作为独立进程执行 → 退出码 2；以及**结构顺序**
   （代码块文本位置严格早于 `LAUNCHER = AppLauncher(ARGS)»，块内可执行语句不含
   `sim.step»/`SimulationContext»/`author(»/`makedirs»）。
   「Isaac `AppLauncher» 构造本身无副作用」这一点仍为 **UNKNOWN**。
2. 接口输出的是**运动学**目标：真实负载半径、滑移、电机扭矩/电流、编码器噪声、通信延迟都未建模。
3. 原地转向在四轮固定轴底盘上必须横向刮蹭，**实测偏航率远小于指令**
   （`P4p_turn_r03» 指令 `psi_dot ≈ -1.236 rad/s»，实测约 `-0.472 rad/s»）。接口只下单运动学，不保证达成。
4. 单位防护只挡**跨量级**错误（mm/s、mm、deg/s、rpm）；若错误单位换算后的量级**恰好落在合法区间内**
   （例如 1.2 deg/s 被写成 1.2 rad/s），**无法区分，仍为 UNKNOWN**，需上层显式声明单位。
5. `radius_m» 允许区间 [0.005, 5] m 是文档化的工程区间，不是标定结果；
   P4-08 实测有效半径随速度/摩擦在 0.10103–0.10161 m 变动，接口默认 `0.101459» 只是 P4-08 的取值。
6. 停车路径（命令归零 → 速度收敛到 1e-4 m/s 量级）沿用 P4-09 的门禁结论，**不是安全级急停**。
7. `_scratch_preflight_demo.py» 只用于本次取证，**未提交**；GPU 侧复现命令见第 7 节。

---

## 7. 复现清单

```powershell
python -m pytest tests/simulation/test_p4_command_interface.py -q     # 73 passed, exit 0
python -m py_compile scripts/simulation/p4_command_interface.py       # exit 0
python -m py_compile scripts/simulation/p4_proxy_wheel_drive.py       # exit 0
```

GPU 侧（有卡时，用来补上第 6 节第 1 条）：

```bash
$PY tools/p2_gate_runner.py --script scripts/simulation/p4_proxy_wheel_drive.py \
    --run-id P4I_reject_r01 --out-dir <DIR> --timeout 1500 -- --profile velocity_control \
    --read-contacts --velocity-cmd 0.80 --speed-limit 0.50 --limit-mode reject
# 预期：进程退出码 2、stdout 含 GATE_RESULT=ABORTED、out-dir 内没有 result.json / raw_poses.jsonl
```
