# P5 控制器与 HOLD：显式 DOF 映射 · 安全包络 · reset 负向对照 · 最小控制器 API

- 分支：`p5/controller-hold`（基于 `p5/rs-reset-pose` @ 6672306）
- 范围：master spec **P5-04 / P5-05 / P5-06 / P5-09**（R-next item 11）
- 环境：本机 CPU，Python 3.11.9 + pytest 9.0.2；**无 GPU、无 Isaac Sim、无 pxr、无 torch 安装**
- 前置工作（未重做，只复用）：P5 `p5/arm-static-consistency` @ ca30af0、P5B `p5/rs-reset-pose` @ 6672306（PR #22）
- 真机资产未改动：`96d4f2ec1126dec6eaf9e70ab664ba6cc30fa81bdec86178a3e50b7a23bdd510`

---

## 1. 交付物

| 文件 | spec | 作用 |
|---|---|---|
| `simulation/badminton_scene/dof_index.py` | P5-04 | 只用**运行时 joint-name 列表**建立 name → runtime DOF index 双射；重复 / 缺失 / 多余一律 raise |
| `simulation/badminton_scene/safety.py` | P5-05 | 位置 / 速度 / 加速度 / 有限性校验 + HOLD 锁存；拒绝一律抛带 joint + value 的类型化异常 |
| `simulation/badminton_scene/controller.py` | P5-09 | 最小可复用 B601-RS 控制器：读状态 / 下目标 / reset / hold / IK 钩子（显式 NotImplemented） |
| `tests/simulation/arm_pose/test_dof_index.py` | P5-04 | 12 条：乱序、重复、缺失、多余、与 USD 遍历序的**反向对照** |
| `tests/simulation/arm_pose/test_safety_envelope.py` | P5-05 | 22 条：每条拒绝路径 + HOLD 目标位级不变 |
| `tests/simulation/arm_pose/test_reset_negative_controls.py` | P5-06 | 13 条：NaN / inf / 未知名 / 越限 / 宽度错 / 空 payload / 控制器 None |
| `tests/simulation/arm_pose/test_controller_api.py` | P5-09 | 20 条：numpy/torch shim 驱动，含一次**真实** `reset_idx()` 端到端 |

分层（每层可单独使用，每层单独有测试）：

    rest_pose.py    模型表（限位 / 静止位姿），零运行时依赖      ← P5B，未改
    dof_index.py    name → runtime DOF index（P5-04）
    safety.py       包络 + HOLD（P5-05），按 runtime DOF 顺序工作
    controller.py   把一台 articulation 绑起来，拥有唯一的写 sim 出口（P5-09）

设计上刻意把「名字→索引」与「按位置校验」分成两层：SafetyGate 只按 runtime 顺序工作、从不自己重排，
因此它不可能拿 joint3 的限位去校验 joint1 的值 —— 这正是本切片要防的失效模式。

## 2. 真实命令与真实退出码

在本 worktree 根目录执行（PowerShell）：

    python -m pytest tests/simulation/arm_pose/test_dof_index.py tests/simulation/arm_pose/test_safety_envelope.py tests/simulation/arm_pose/test_reset_negative_controls.py tests/simulation/arm_pose/test_controller_api.py -q
    → 67 passed in 1.11s        exit code 0

    python -m pytest tests/simulation/arm_pose -q
    → 119 passed, 5 skipped in 2.19s   exit code 0

单文件计数（同一命令族，逐文件跑）：

| 命令 | 结果 | exit |
|---|---|---|
| `python -m pytest tests/simulation/arm_pose/test_dof_index.py -q` | 12 passed | 0 |
| `python -m pytest tests/simulation/arm_pose/test_safety_envelope.py -q` | 22 passed | 0 |
| `python -m pytest tests/simulation/arm_pose/test_reset_negative_controls.py -q` | 13 passed | 0 |
| `python -m pytest tests/simulation/arm_pose/test_controller_api.py -q` | 20 passed | 0 |

回归约束：P5B 的 52 条仍然全绿（119 − 67 = 52），5 个 skip 是 P5B 已有的条件跳过，不是本次新增。
**没有**跳过、没有 xfail、没有 `pytest.mark.skip` 用来掩盖任何一条负向用例。

## 3. P5-04：显式 joint-name → runtime DOF index

三种顺序在本资产上互不相同（P5 F2，pxr 导出 `arm_joint_facts.json` 的 `stage_traversal_order`）：

    USD 遍历序   joint6, gripper_joint1, gripper_joint2, joint5, joint4, joint3, joint2, joint1, root_joint
    URDF 序      joint1 … joint6, gripper_joint1, gripper_joint2
    runtime DOF  由 articulation.joint_names 给出（唯一能用于索引 joint_pos 的顺序）

`JointIndex.build(runtime_names, required=...)` 只吃 runtime 列表，产出 name → index 双射，否则 raise：

- 重复 → `DuplicateJointName`（消息列出该名字的全部位置，如 `'joint3' at DOF positions [3, 7]`）
- 缺失 → `MissingJoint`；多余 → `ExtraJoint`
- 空列表 → `JointIndexError`；宽度不符 → raise（绝不截断）
- 名字**精确匹配**：不 strip、不 lower、不做别名。`'joint1 '` 是另一个关节，测试证明它会失败。

**顺序无关性的证明**（不是声明，是可执行断言）：
1. 打乱 runtime 顺序后，`index_of(name)` 随 runtime 位置走，`mapping` 的键集合不变；
2. 同一份按模型表顺序给出的 payload，经两个不同 runtime 顺序的 index `reorder()` 后得到**不同**的 runtime 向量，
   但 `as_mapping()` 回到**完全相同**的 {name: value}；每个值都落在该名字的 runtime 下标上；
3. 反向对照（用 USD 遍历序）：`dict(zip(usd_order, values)) != dict(zip(runtime_order, values))`
   —— 朴素按位置 zip 会整体错位；而 `as_mapping(values, order=usd_order)` 与前者一致、
   `as_mapping(values)`（默认 runtime 序）与后者一致。即：模块从不"顺手假设"payload 已是 runtime 序。

## 4. P5-05：安全包络 + HOLD

校验顺序（首个失败即抛，消息只讲一个原因）：

1. 宽度：每个 runtime DOF 恰好一个值（`CommandWidthError`）
2. 有限性：NaN / ±inf 先于任何比较（`NonFiniteCommand`）——NaN < lo 为 False，纯范围检查看不见 NaN
3. 位置：来自**模型表**的逐关节限位（`PositionLimitViolation`），端点包含（与 reset.py 一致）
4. 速度：|目标 − 上一目标| / dt ≤ v_max[joint]（`VelocityLimitViolation`）
5. 加速度：二阶差分 / dt²（`AccelerationLimitViolation`，仅当已有完整历史）

每条拒绝都携带 `.joint / .value / .limit / .index / .code`，并有 `as_dict()`
结构化表单供日志使用；`rejected_count` 可用于遥测。测试断言的是属性，不只是消息文本。

HOLD：

- `hold()` 冻结**最后一次被接受的目标对象本身**（`held is accepted` 断言同一对象，`struct.pack` 位级比较），不做任何算术、不复制重算；
- HOLD 期间任何新命令 → `HoldActive`（连 1e-9 的微动也拒绝）；`last_valid` 保持不变；
- 未接受过目标就 `hold()` → `HoldWithoutTarget`；未 held 就 `release()` → `NotHolding`；
- `release()` 后恢复接受命令，最后有效目标仍是速率基线；
- reset/teleport 后必须 `seed()` 重新锚定速率基线（否则第一条命令会拿旧目标做速率比较），控制器在 reset 内部自动 seed。

## 5. P5-06：reset 负向对照（不静默 clamp）

复用已有异常类型（来自 `badminton_scene.rest_pose`，由 reset.py 再导出），**未重定义**；
测试断言 `type(exc) is rest_pose.RestPoseLimitViolation`，因此任何人都不能用"长得像"的类替换它。

| 负向用例 | 抛出 | 写 sim？ |
|---|---|---|
| NaN 目标 | `RestPoseLimitViolation`（消息含 `nan` 与 `joint5`） | 否 |
| +inf 目标 | `RestPoseLimitViolation`（inf == inf 逃过 NaN 检查，但逃不过范围） | 否 |
| 未知 joint 名（`shoulder_pan`） | `UnknownRobotModel` | 否 |
| 越限目标（joint3 = −1.5，即 P5 F1 缺陷） | `RestPoseLimitViolation` | 否 |
| 宽度错（6 列） | `ResetPoseError`（消息含 6 与 8） | 否 |
| 空 payload（`[[]]` 与 `[]`） | `ResetPoseError` | 否 |
| 控制器 `reset(None)` | `ResetPoseError` | 否（reset_fn 未被调用，测试断言 calls == []） |

关于 `reset_idx(joint_pos=None)` 的口径（必须说清楚，否则"None 必须 raise"会与 P5B 冲突）：
**`reset_idx` 的 None 是 P5B 的显式语义**——"写入本机型自己的静止位姿"，是一个经过 USD 限位校核的分支，
不是 clamp、不是垃圾默认值；有专门的对照用例断言它写出的位姿在 USD 限位内（joint2 = 0.3、joint3 = 1.0）。
控制器的 API 没有这个哨兵：`B601RSController.reset(None)` 视为调用方 bug 而 raise，
"回到静止位姿"必须显式调用 `reset_to_rest()`。两处口径都在测试里钉死。

## 6. P5-09：最小可复用 B601-RS 控制器 API

    arm = B601RSController(robot, shuttle=None, reset_fn=None, tensor_factory=None, dt=1/60)
    arm.describe()                 # model / joint_names / dof_index / 限位 / 包络来源
    arm.read_state()               # ArmState：按名字取关节值；缺状态就 raise，绝不补零
    arm.set_joint_targets(x)       # 序列或 {name: value}；经 P5-05 校验后锁存
    arm.reset(pos, env_ids=...)    # 经 P5B reset_idx 写入；自己先做有限性/限位校核
    arm.reset_to_rest(env_ids=...) # 显式走模型静止位姿
    arm.hold() / arm.release() / arm.held_target
    arm.seed(pos)                  # reset/teleport 后重新锚定速率基线
    arm.solve_ik(target)           # 抛 IkNotImplemented（NotImplementedError 子类）

无 Isaac 依赖：模块导入时不 import torch / numpy；reset 路径惰性 import torch，或由 `tensor_factory` 注入。
测试里 `test_a_reset_through_the_real_reset_idx_is_driven_by_the_shim` 用
`reset_fn=reset_mod.reset_idx` + tensor_factory=reset_mod.torch.tensor` 走完
controller → P5B reset_idx → 伪造 PhysX 写入 的**真代码路径**，断言写入值逐关节落在 pxr 导出的 USD 限位内，
并对 joint3 复算 5% 行程余量规则。这也同时证明替换顺序错乱的 payload 不会污染 joint_pos 向量
（命名映射用例：倒序字典输入仍得到 runtime 序的向量）。

IK 钩子：`solve_ik()` 直接抛 `IkNotImplemented`，且断言**无副作用**（`last_valid is None`、`commands == []`）。
返回当前位置或零向量在链路下游与"已解出的位姿"无法区分，所以选择拒绝。

## 7. 仍然 UNKNOWN（不可据本文声称的部分）

1. **没有真实 GPU / 多环境运行**：全部证据来自 CPU + shim。多 env（`--num_envs > 1`）下的
   reset/目标广播、PhysX 驱动的实际跟随、真实 runtime `joint_names` 顺序，都**未在本机验证**；
   runtime DOF 顺序的"唯一权威"仍是 Isaac 加载后的 `articulation.joint_names`，本切片只保证"拿到列表后不再出错"。
2. **没有力矩数据**：无力矩传感器、无实测关节力矩；`ArmState` 刻意不含 torque 字段。
3. **速度 / 加速度上限是 POLICY CAP，不是实测或数据手册值**：URDF 未记录这些关节的速度限，
   也没有任何一台实机做过特性标定。代码里 `ENVELOPE_SOURCE` 常量、`describe()` 输出、以及每条
   速率异常消息都带着这句"未验证"声明。**在替换为实测/厂商值之前，该包络不能作为安全认证依据。**
4. **IK 未实现**：无 IK 求解器、无标定 TCP/球拍坐标系，`solve_ik()` 只保证"拒绝"。
5. **未做真机 / 真 Isaac 联调**：没有在 172.31.68.251 上运行、没有向任何 USD 写入；本切片对资产的唯一接触是
   读已提交的 pxr 导出 JSON（只读）。sha256 `96d4f2ec…bdd510` 与 P5 记录一致，未改动。
6. **HOLD 是命令层锁存，不是安全级急停**：它冻结的是"最后被接受的目标"，不切断执行器电源、
   不与驱动器的 STO/看门狗交互；轨迹层/驱动器层的保持行为未验证。
7. **P5-06 覆盖的是"目标值合法性"**：不覆盖 reset 之后 shuttle/root 状态的物理一致性（P5B 已有条款，本切片未扩展）。

## 8. 边界声明（按任务分区）

本次只改：`simulation/badminton_scene/{dof_index,safety,controller}.py`、
`tests/simulation/arm_pose/test_{dof_index,safety_envelope,reset_negative_controls,controller_api}.py`、
本报告、以及 `simulation/badminton_scene/__init__.py` 未改。
未触碰：`scripts/simulation/*`、`src/*`、队列状态文件、`PHASE_P2_*`/`PHASE_P3_*` 文档、
P5B 已有的 `reset.py` / `rest_pose.py` 及其测试。
