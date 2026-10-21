# P8-03 / P8-04：种子分离 + 难度课程 + 非强化学习基线

**分支**：`p8/seeds-baselines`（从 `p8/rl-interface-schema` @ `c073d2c`，PR #30 切出）
**范围**：仅 CPU + numpy + pytest。**没有训练、没有 PPO、没有优化器、没有 GPU、没有 Isaac、没有真机。**
**状态**：准备件（preparation-only）。下面的任何成功率都是**合成运动学玩具环境中的非 RL 基线数字**，
**不是 RL 性能，也不得被引用为 PPO/学习型策略的指标**。

---

## 1. 交付内容

| 文件 | 内容 |
|---|---|
| `src/badminton_brain/rl/seed_ledger.py` | 主种子 → train/val/test 种子集，**可证明互斥**；场景派生种子；JSON 账本 + SHA-256；重叠区间/越界/重复一律拒绝 |
| `src/badminton_brain/rl/curriculum.py` | 课程定义：等级 → 参数区间；等级自带 split；**验证/测试等级在结构上无法进入训练** |
| `src/badminton_brain/rl/baselines.py` | 两个确定性非 RL 基线：有限状态/可达性驱动 + 比例跟踪控制器；均走 P8-01 严格动作校验与 P5-05 安全包络 |
| `src/badminton_brain/rl/evaluation.py` | **公平评测协议**：同一 episode 列表、同一观测/动作校验路径、同一成功/失败定义；账本与 episode 列表不一致时拒绝运行 |
| `tests/badminton_brain/rl/` 下 4 个新测试文件 | 32 个新测试 |
| 本文件 | 设计 + 真实命令 + 真实退出码 + 真实（合成）数字 |

---

## 2. 种子账本设计（P8-03）

### 2.1 可证明互斥：靠构造，不靠抽样

- 每个 split 独占一段**半开整数带**：`train = [0, B)`、`val = [B, 2B)`、`test = [2B, 3B)`，`B = BAND_WIDTH = 2**40`。
  三条带互不相交 ⇒ 一个种子**不可能**同时属于两个 split。这是区间算术结论，与抽样、哈希碰撞概率无关。
- 带内**也不重复**：`seed(i) = band_lo + slot(i) * 64`，`slot(i) = (a*i + b) mod SLOTS_PER_BAND`，其中
  `SLOTS_PER_BAND = 2**40/64 = 2**34`，**a 被强制为奇数**。模数是 2 的幂，奇数乘法在该模下是双射
  ⇒ 不同 `i` 必得不同 slot。`a, b` 由 `SHA-256(master_seed, task_family, split)` 派生
  ⇒ 换主种子则所有派生种子改变，但互斥性不变。
- 容量上限：每个 split 最多 `SLOTS_PER_BAND` 个种子；请求更多则抛 `SeedExhaustionError`，**不会**静默去重或缩小集合。

### 2.2 场景派生种子仍然互斥

若某个场景按 `seed + 参数` 派生（例如光照、球号），派生种子定义为 `base + k`，`0 <= k < MAX_DERIVATIVES_PER_BASE = 64`，`k = SHA-256("derivative", 参数) mod 64`。
即：**一个基础种子的全部派生种子都落在它自己的 64 宽窗口内**，因此

1. 不会与同 split 的其它基础种子的派生种子碰撞（slot 不同 ⇒ 窗口不重叠）；
2. 更不会与其它 split 碰撞（窗口仍在所属带内）。

参数名哈希到 `k` 只是"选窗口内的哪个位置"，无法逃出窗口。测试对 train/val/test 各 40 个基础种子 × 4 个参数做了集合交验。

### 2.3 账本 JSON + 哈希

`SeedLedger.to_dict()` 记录 version / task_family / master_seed / band_width / max_derivatives_per_base / counts / **每个 split 的全部种子**，外加 `ledger_hash`（对去掉哈希字段的规范化 JSON 求 SHA-256）。
`verify()` 会拒绝：种子越出本带、同 split 重复、跨 split 重复、数量与记录不符；`from_dict()` 另加版本/窗口宽度/哈希校验。

**任何单个种子改变 ⇒ 哈希改变**（测试分别替换 train/val/test 中的种子各一次验证）。

**拒绝重叠请求**：`allocate_split_ranges({"train": (0,100), "val": (50,150)})` 抛 `OverlappingSplitRangeError`（消息含两个 split 与重叠区间）；`SeedLedger.derive(band_layout=...)` 同样走该校验，实测传重叠布局即拒绝。

---

## 3. 难度课程（P8-03）：验证/测试等级进不了训练

- `CurriculumLevel(name, level, split, parameters)`：`split ∈ {train, val, test}` 是**构造参数**。
- 三道结构性闸门：
  1. `CurriculumSchedule.training_levels()` 只返回 `split == "train"` 的等级；
  2. `require_training_level(name)` 对 val/test 等级抛 `CurriculumSplitViolation`（训练清单无法用 evals 拼出来）；
  3. `episodes_from_ledger(..., split=, level=)` 要求等级 split 与种子 split 一致，否则抛 `CurriculumSplitViolation`（验证集不能"借"训练种子，反之亦然）。
- 默认等级（合成的接触点瞄准任务族 `racket_contact_targeting`）：

| 等级 | split | target_x (m) | target_y (m) | target_z (m) | reach_radius (m) |
|---|---|---|---|---|---|
| L1-near | train | [-1.20, -0.80] | [-0.40, 0.40] | [0.45, 0.75] | [1.60, 2.60] |
| L2-mid | train | [-1.80, -1.00] | [-0.80, 0.80] | [0.35, 0.95] | [1.60, 2.60] |
| L3-far | train | [-2.20, -0.60] | [-1.20, 1.20] | [0.35, 0.95] | [1.60, 2.60] |
| V1-edge | val | [-2.40, -1.80] | [1.20, 1.40] | [0.90, 1.00] | [1.60, 2.60] |
| V2-tight-reach | val | [-2.20, -1.20] | [-1.00, 1.00] | [0.35, 0.95] | [0.60, 1.10] |
| T1-corner | test | [-2.55, -2.20] | [-1.45, -1.20] | [0.20, 0.30] | [1.60, 2.60] |
| T2-late | test | [-1.60, -0.60] | [-0.60, 0.60] | [0.55, 0.85] | [2.40, 2.60] |

参数采样确定：`np.random.default_rng(seed)` + 参数名排序遍历 ⇒ 同一 (level, seed) 在任何机器上同值。
`reach_radius_m` 只用来合成 `Decision.feasibility` 四个 flag（可达性判决的替身），不改变动力学。

---

## 4. 两个非 RL 基线（P8-04）

两者都是**已验证 P8-01 观测向量**的纯函数，返回 **P8-01 动作映射**（`ActionValue`，带单位），因此与将来的学习型策略走完全相同的校验路径。

1. **`reachability-fsm`（有限状态 / 可达性驱动）**
   `LocalReachabilityFSMStub` 把 `feasibility_flags = [reachable, in_workspace, timing_ok, prediction_valid]` 映成 fail-closed 状态机：任一不满足 → `HOLD_*` 且输出零动作；全满足 → `TRACK`，对 `time_to_hit` 外推的拦截点做饱和 bang-bang 逼近。
   **明确声明**：该 stub 只是 flag 中继，**不是可达性计算**，其数字不得称为 P7 可达性性能。
   P7 的真实 FSM（`p7/reachability-fsm` @ `adaee81`，`safety/interaction_fsm.py`）不是本分支祖先，本分支**无法 import**；`p7_fsm_status()` 在运行时记录该事实（`importable: False`，`verdict_source: LocalReachabilityFSMStub`），并有测试在它变得可 import 时**主动失败**，提醒去写真正的 P7 适配器，而不是悄悄换掉判决来源。
2. **`tracking-p`（比例跟踪）**
   对"球拍-目标"误差做比例控制：平面部分 → 底盘速度，竖直部分 → 一个关节增量；每步增量按 `*_accel_max * dt` 限幅，再经 `to_command(previous_action=...)` 复核，最后进 P5-05 的 `SafetyShield`（TEMP proxy 限值，未测量）。

两个基线使用的目标源**完全相同**（`intercept_target()`：shuttle 估计的线性外推），合成 episode 构造时让该外推值等于 episode 目标，因此成功判据也完全相同。

---

## 5. 公平评测协议（P8-04）

`evaluate(policy, episodes, ledger)` 的固定循环：

1. 用 `observation_schema.build_observation` 构造观测 → `observation_schema.validate`（严格编码路径）；
2. 策略返回动作映射；
3. `action_schema.to_command`（单位/形状/边界/速率限幅）；
4. `WholeBodyTarget` → `SafetyShield.process` → 记录 `last_actions[0]`（PASS/PROJECT/HOLD/STOP…）；
5. 应用**安全后**的命令到合成状态，按同一判据计分。

定义（两者完全一致，不存在"对某个基线更宽松"的版本）：

- **success**：在 horizon 内某一时刻，球拍接触点与 episode 目标距离 ≤ `tolerance_m`（默认 0.05 m）；
- **timeout**：跑满 horizon 仍未成功；
- **safety_reject**：`SafetyShield` 判决不是 `PASS`（说明该基线要求了包络外的东西），episode 立即终止并计数；
- **action_reject**：动作校验异常（不应发生；出现即为缺陷，单独计数而不是隐藏）。

**拒绝运行的条件**（`LedgerEpisodeMismatchError`）：episode 列表的 `ledger_hash` 与传入账本不一致；或某个 episode 的种子不在该 split 的种子集合里；或 split 未知。
此外 `assert_identical_episodes(a, b)` 用 fingerprint 强制"两个基线必须跑同一份 episode 列表"。

**合成动力学（明确标注，非机器人模型、非物理）**：
`base_position += dt * twist[0:2]`；`joint_position = safe joint target`；`racket = base + (0.30, 0.00, 0.60 + 0.40*sin(q2))`；底盘原点 `STATION_ORIGIN = (-1.50, 0, 0)`（保证初始球拍点落在 P8-01 的球拍工作区盒内）。

---

## 6. 真实命令与真实退出码

环境：Windows，Python **3.11.9**，pytest **9.0.2**，numpy；工作目录 `E:\具身智能\badmin_wt_p8b`。以下为逐字粘贴的真实输出。

代码块 1（切分支前基线，在 `p8/rl-interface-schema` 上）：

    > python -m pytest tests/badminton_brain/rl -q
    70 passed in 3.39s
    EXIT=0

代码块 2（本切片完成后，70 旧 + 32 新）：

    > python -m pytest tests/badminton_brain/rl -q
    102 passed in 11.95s
    EXIT=0

代码块 3（合成数字脚本 _scratch_numbers.py，运行后已删除，可按第 5 节重建）：

    > python _scratch_numbers.py
    LEDGER_HASH 11ce3d8a43e5e60a429b37cea4a00d51e26f090587bdccdd8ed928b94295778c
    EXIT=0

账本参数：`master_seed=20261022`，`task_family="racket_contact_targeting"`，`counts={"train":60,"val":20,"test":20}`，`BAND_WIDTH=2**40`。
L1-near 10 个 episode 的列表 fingerprint：`a904baa2a37fe60f4112c199fc81fdd1cdea7e58ab9d77c9dc1935c890e8bfee`。

---

## 7. 基线数字（**合成**，每级 10 个 episode，非 RL 性能）

| 等级 | split | 基线 | n | success | timeout | safety_reject | action_reject | success_rate | 平均最小距离 (m) | FSM 状态 |
|---|---|---|---|---|---|---|---|---|---|---|
| L1-near | train | reachability-fsm | 10 | 8 | 2 | 0 | 0 | 0.8 | 0.0490 | TRACK |
| L1-near | train | tracking-p | 10 | 10 | 0 | 0 | 0 | 1.0 | 0.0474 | TRACK |
| L2-mid | train | reachability-fsm | 10 | 6 | 4 | 0 | 0 | 0.6 | 0.0522 | TRACK |
| L2-mid | train | tracking-p | 10 | 10 | 0 | 0 | 0 | 1.0 | 0.0472 | TRACK |
| L3-far | train | reachability-fsm | 10 | 5 | 5 | 0 | 0 | 0.5 | 0.4634 | HOLD_UNREACHABLE + TRACK |
| L3-far | train | tracking-p | 10 | 10 | 0 | 0 | 0 | 1.0 | 0.0459 | TRACK |
| V1-edge | val | reachability-fsm | 10 | 0 | 10 | 0 | 0 | 0.0 | 1.6502 | HOLD_UNREACHABLE |
| V1-edge | val | tracking-p | 10 | 10 | 0 | 0 | 0 | 1.0 | 0.0467 | TRACK |
| V2-tight-reach | val | reachability-fsm | 10 | 0 | 10 | 0 | 0 | 0.0 | 0.9403 | HOLD_UNREACHABLE |
| V2-tight-reach | val | tracking-p | 10 | 10 | 0 | 0 | 0 | 1.0 | 0.0471 | TRACK |
| T1-corner | test | reachability-fsm | 10 | 0 | 10 | 0 | 0 | 0.0 | 1.7806 | HOLD_UNREACHABLE |
| T1-corner | test | tracking-p | 10 | 10 | 0 | 0 | 0 | 1.0 | 0.0467 | TRACK |
| T2-late | test | reachability-fsm | 10 | 6 | 4 | 0 | 0 | 0.6 | 0.0521 | TRACK |
| T2-late | test | tracking-p | 10 | 10 | 0 | 0 | 0 | 1.0 | 0.0470 | TRACK |

读法（重要）：

- 这些数字**只说明**：在合成玩具环境里，两个基线都跑得通、都遵守动作 schema 与安全包络（`safety_reject=0`、`action_reject=0`），并且"fail-closed 的 FSM 在可达性预算不够时就真的不动"是可测量的（V1/V2/T1 全 timeout，L3 部分 `HOLD_UNREACHABLE`）。
- `tracking-p` 接近 1.0 的合成成功率**没有**任何物理意义：环境没有动力学、没有感知噪声、没有延迟、没有真机误差；它只证明"harness 能把一个会动的控制器跑完并计分"。
- **不得**把表中的任何成功率引用为 RL/PPO 性能。本切片未运行任何学习过程。

---

## 8. 测试覆盖（32 个新测试）

- 种子：大 N（train 2000 / val 800 / test 800）三向集合互斥 + 每种子都在自己带内；决定性（同参数两次同哈希、换主种子/换任务族则不同）；派生种子留在基础种子的 64 宽窗口且跨 split 不交；单种子改变 ⇒ 哈希改变（三个 split 都测）；重叠区间请求被拒（顶层 API 与 `SeedLedger.derive(band_layout=...)` 两条路径）；手工改账本被拒（跨 split 搬种子、窗口内改值但哈希不符）；带容量耗尽被拒、非法 split 被拒。
- 课程：`training_levels()` 恰为 L1/L2/L3；对 V/T 调 `require_training_level` 抛异常；`require_eval_level("L1-near")` 抛异常；采样确定且在声明区间内；参数名一致性。
- 基线：两个基线都产出 P8-01 合法动作、都通过 `to_command`；连续 12 步满足加速度限幅；整批评测 `safety_reject=0` 且 shield 全为 PASS；FSM 在拒绝判决下输出零动作；比例控制器零误差时不动；`ObservationView` 拒绝未验证向量；P7 FSM 可 import 性被断言（与常量不一致即失败）；`torch` 未被 import。
- 评测协议：同账本可复现 fingerprint；换账本被拒；伪造 `ledger_hash` 被拒；种子不在 split 集合被拒；等级 split 与种子 split 不一致被拒；两基线共享同一 fingerprint；计数之和等于总数；报告标注 `synthetic=True`、`NON_RL_BASELINE_NO_LEARNING`。

---

## 9. UNKNOWN / 未验证项

1. **P7 真实 FSM 未接入**：`interaction_fsm.py` 在 `p7/reachability-fsm` 上，本分支不可 import；基线用的是 flag 中继 stub。真正的 P7 判决适配器（`InteractionFSM` → `ReachabilityVerdict`）尚未编写。
2. **安全包络限值是 TEMP proxy**：`SafetyLimits.temp_proxy()` / `ActionLimits.temp_proxy()` 的关节限、速度限、工作区盒全部是占位值，未从 PiPER URDF / Morph One 实测获得。合成环境特意把状态放进盒内，因此**没有**覆盖"包络外行为"这一真实场景。
3. **合成动力学是玩具**：3 个可动自由度、无动力学/摩擦/延迟/噪声；不构成仿真，不能外推成功率的物理含义。
4. **成功率与容差耦合**：`tolerance_m=0.05` 与 baseline 死区（0.01/0.05 m）同量级，平均最小距离 0.046–0.052 m 说明成功判定贴近容差边界；真实容差应由接触物理决定（未测）。
5. **课程等级只对合成任务族定义**：真实击球任务族的参数（落点、速度、旋转、时间窗）尚未定义，现有区间不可解释为场地范围声明。
6. **未测**：种子派生对下游数据加载/场景生成的实际可复现性（本切片只验证种子本身与 episode 构造）；多机一致性（只在单机 Windows/py3.11.9 验证）。
7. **训练授权状态不变**：`management/morph_one_rs/P8_TRAINING_GATE.md` 的"PPO NOT AUTHORISED"未被本切片改变；本切片没有 step()/reset()/rollout/gym API，也没有优化器调用。
