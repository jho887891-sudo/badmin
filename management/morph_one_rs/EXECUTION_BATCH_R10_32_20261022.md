# R10 执行批次：32 个执行子项、8 条工作线（2026-10-22）

> **性质：上传到仓库的派工单，不是实验已成功或代理自动启动的证明。**
> **权威队列**：[96 项 SPEC](https://github.com/jho887891-sudo/badmin/blob/chatgpt/p2-wc-continuous-queue-20261021/management/morph_one_rs/EXECUTION_QUEUE_CONTINUOUS_P0_P8.md) · [状态 JSON](https://github.com/jho887891-sudo/badmin/blob/queue/a01-a03-a10/management/morph_one_rs/EXECUTION_QUEUE_STATE.json) · [Issue #13](https://github.com/jho887891-sudo/badmin/issues/13)。

## 基线事实与防止错误派工

- 本批来自用户 2026-10-22 最新报告：**38 DONE / 32 READY / 13 PARTIAL / 13 BLOCKED（READY/BLOCKED 为调度口径）**。状态 JSON 原始字段当前为 **38 DONE / 13 PARTIAL / 45 UNKNOWN**，必须运行队列选择器重新派生 READY/BLOCKED，不能把两个口径错误相加。
- 分支证据入口：`p3/gate-v1.1@6593736`、`p6/frame-consistency-audit@e330785`、`p8/training-gate@d061559`、`queue/a01-a03-a10@79c4371`。开始时必须刷新完整 SHA。
- **任务号校正**：官方 **P5-07/08 是真实多环境与吞吐/隔离**；**P5-09 是 B601-RS controller API**。不可用 P5-09 冒充 Isaac N=2。
- P4-08、P4-09、P6-02、P6-05、P7-11、P8-05 有已登记成果，子项仅做增量审计/补缺与接口连接，不允许重复宣称完成；P7-12 和 P8-06 有 PARTIAL，不能升为完整物理/训练 PASS。
- A04 实际是协作锁而非强互斥；GPU/Kit 同一时刻仅一条工作线。GPU 测试须由有实际 Isaac 权限的运行者完成；CPU 可隔离并行。
- 该批 32 项是**既有 96 项任务的执行切片与机制缺陷子工作**，**不是向全局队列额外添加 32 个新稳定 ID**；只有真实 DoD 已满足才能更新原队列状态。

## 分工与验收（32 个子项）

### A · Runner 与故障语义

**独占工作范围**：CPU · tools/simulation/p2_gate_runner.py 与独立回归测试。各线新建独立 worktree/分支，修改同文件必须通过 Issue 协调，不能并发写入。

| 子项 | 对应原任务 | 动作 | 最小验收/证据 |
|---|---|---|---|
| A1 | P0-08 | 读 PR #14/#16/#17 与现用 DeepSeek runner，逐接口列出差异与 backward compatibility | 接口对照表 + 可运行正反例；不重新实现已通过的 P0-02/03/04 |
| A2 | P0-07 → P0-08 | 修 Kit Python 异常被吞且 child_rc=0 的假成功判定 | 回放真实 P0F 异常记录：traceback 或强类型故障不得 SUCCESS |
| A3 | P0-08 / A06 | 建立 SUCCESS、EXPECTED_FAULT、UNEXPECTED_FAULT、CRASH、INCOMPLETE 的规范化语义 | 故障注入测试 PASS 与仿真运行 SUCCESS 严格分离，Run ID/产物强校验 |
| A4 | P0-08 / P0-02/03/04 | 验证超时、SIGSEGV、缺 result、损坏 JSON、旧目录残留、非零 rc+成功产物 | 全部 fail-closed 回归；保留六次 P0F 原始数据 |

### B · 真实 Isaac 多环境

**独占工作范围**：GPU 独占 · tools/simulation / P5 场景。各线新建独立 worktree/分支，修改同文件必须通过 Issue 协调，不能并发写入。

| 子项 | 对应原任务 | 动作 | 最小验收/证据 |
|---|---|---|---|
| B1 | A04 | 核验互斥锁、PID 与 Kit 子孙进程；协作锁不能冒充原子锁 | 无竞争占用、无误杀其他用户进程，安全获取和释放 GPU 锁 |
| B2 | P5-07 | 真实 Isaac/Kit 启动 B601-RS N=2，不用 mock 代替 | 两个可独立寻址 articulation 的原始 Run、命令、退出码及 state |
| B3 | P5-07 | env0/env1 交叉施加非零命令，分别 reset 0/1 与联合 reset | 非目标环境状态不被污染；拒绝 env 索引广播与 stale state |
| B4 | P5-08 | 复现种子、吞吐、峰值显存、NaN、掉线后重启与数据一致性 | 带环境指纹的真实指标；16/64/128 仅在合法资源范围且前置条件成立时推进 |

### C · 证据契约和断言溯源

**独占工作范围**：CPU · tools/queue、A06/A09 审计工具，避免修改 A 线 runner 主文件。各线新建独立 worktree/分支，修改同文件必须通过 Issue 协调，不能并发写入。

| 子项 | 对应原任务 | 动作 | 最小验收/证据 |
|---|---|---|---|
| C1 | A06 | 将故障 Run 与正常 Run 分别建证据规则 | 不能因故障运行天然无 result.json 就判失效，也不能因缺 result.json 自动证明故障 |
| C2 | A05 / A06 | 校验唯一 RUN_ID、参数 hash、环境指纹、原始日志和文件 SHA256 | stale 文件、跨 Run 引用及摘要值失配均 FAIL |
| C3 | A09 | 新 P0/P5 断言关联源码行、SPEC 条款、单位、Run/trace 与预注册阈值 | 每个 ASSERTED 都可单独重算；UNKNOWN 不数值填充 |
| C4 | A03 / A08 | 运行 check_consistency 和 CI 坏样本，追踪新证据 | 队列修复只按有效事实提交；不因校验器规则假造成功产物 |

### D · P4 动力学根因

**独占工作范围**：CPU 分析可并行，真实 Isaac 接触实验排队使用 A04。各线新建独立 worktree/分支，修改同文件必须通过 Issue 协调，不能并发写入。

| 子项 | 对应原任务 | 动作 | 最小验收/证据 |
|---|---|---|---|
| D1 | P4-09（已 DONE，新增诊断） | 分析命令 0.30/0.50 m/s 与仿真 0.31811/0.53007 的约 6% 差异 | 按轮速、joint、odom、ground truth 独立复算 r_eff，不改历史 DONE/原始数据 |
| D2 | P4-08（已 DONE，审计） | 复用已存在的 36 个摩擦/载荷条件 Run 判明滑移与有效半径变化 | 证明采样维度及记录一致；必要时只补缺失的有控制单因素实验 |
| D3 | P4-06/07 | 检查正反直线与转向时四轮方向、量纲和控制饱和 | 分别区分代理参数与物理接触来源，禁用诊断驱动冒充真轮驱动 |
| D4 | P4-10/11 | 补里程计误差、驻停 HOLD 和 P4_REAL 缺口判定 | 有独立可重复的验收表，不能因 6% 漂移被手改阈值而 PASS |

### E · 整机几何与传感器

**独占工作范围**：CPU 解析+只读 P3/P6 历史证据；需 Isaac 的部分排 GPU 队。各线新建独立 worktree/分支，修改同文件必须通过 Issue 协调，不能并发写入。

| 子项 | 对应原任务 | 动作 | 最小验收/证据 |
|---|---|---|---|
| E1 | P6-05（已 DONE，补审计） | 复现 composition A 的 pxr 跳数和跨分支来源 | 保留 mounted_chain 的 0.33242m OPEN 与 v2 的 1.853e-9m，不覆盖 UNKNOWN |
| E2 | P3-05/07 | 非零 arm pose、非平凡 yaw 下的挂载相对六自由度稳定性 | 原始 pose、相对误差、有效负控制，区分 P3 诊断驱动与真实轮驱动 |
| E3 | P6-06 | RGB/Depth/state 时间戳、延迟、掉帧、错配回放测试 | D455 未实标外参则保持 UNKNOWN，2D 不偷升 3D |
| E4 | P6-08/09 | 整机最小合法静态场景、集成 manifest、多环境能力 gate | 明确 P5-09 API 前置与 P6-08 静态门禁；缺证据保持 BLOCKED/PARTIAL |

### F · 机械臂接口与安全

**独占工作范围**：CPU controller/IK/FSM；实际物理接触须前置门禁和 GPU。各线新建独立 worktree/分支，修改同文件必须通过 Issue 协调，不能并发写入。

| 子项 | 对应原任务 | 动作 | 最小验收/证据 |
|---|---|---|---|
| F1 | P5-09 | 统一 B601-RS controller API：state/targets/reset/hold/IK hook | 单元与错误路径测试；注意 P5-09 不是 N=2 任务 |
| F2 | P7-09 | 对比 verified 8DOF FK 与 proxy 6DOF IK/可达空间 | 每条结果标明 backend/frame/provenance；不得混合 backend 或静默 fallback |
| F3 | P7-10 | 关节限位、自碰撞、过期目标、不可达和低置信度触发安全 HOLD | 有负例能证伪，禁止先写越界目标再拒绝 |
| F4 | P7-11/12 | 审计现有 simulation-only 闭环并拆分“轨迹可达”“几何接触”“物理击球” | 保留 hit_rate=null 的已有证据；无接触证据不得宣称有效回球 |

### G · 视觉与轨迹预测

**独占工作范围**：CPU 数据审计/滤波器；检测 GPU 基准使用 A04 串行窗口。各线新建独立 worktree/分支，修改同文件必须通过 Issue 协调，不能并发写入。

| 子项 | 对应原任务 | 动作 | 最小验收/证据 |
|---|---|---|---|
| G1 | P7-01 | 审计 YOLO 真实/合成数据 manifest、4–6 px 小球分桶、重复样本和视频泄漏 | 按采集源/视频分组，不把同源帧跨训练验证集 |
| G2 | P7-02/03 | 640/960/1280 与裁剪/tiling 的 mAP、recall、误检、延迟对照 | 只报告真正执行的组合与环境；GPU 无权使用则 NOT RUN |
| G3 | P7-05/07 | 2D EKF 跟踪与 0.3–0.8s 轨迹预测：缺帧、离群值、时钟错位 | 分 horizon 的误差与不确定度回放，禁止用合成 3D 当真实实测 |
| G4 | P7-04/06/08 | 检测→时间戳/协方差→轨迹→可达性接口，拒绝过期帧和未标定 3D | shape/单位/frame 一致；无外参时输出 2D_ONLY/HOLD |

### H · 门禁、集成与队列治理

**独占工作范围**：CPU 审计优先；集成 Isaac 验证排 A04。各线新建独立 worktree/分支，修改同文件必须通过 Issue 协调，不能并发写入。

| 子项 | 对应原任务 | 动作 | 最小验收/证据 |
|---|---|---|---|
| H1 | P8-06 | 逐项核查 11 条 P5/P6/P7 训练前置与证据指针 | 已证实缺失继续 MISSING_EVIDENCE，不通过自写 PASS 标志解锁 PPO |
| H2 | P8-05 | 回归训练授权参数不可绕过、checkpoint 原子/哈希与停止条件 | 仅运行 CPU tests；PPO、torch 训练和真实机器动作均不执行 |
| H3 | P7-13 / P6-09 | 测量最小感知→预测→IK→HOLD/动作接口的时延与故障恢复 | 若整机门禁未开则仅接口级，明确 NOT_PHYSICAL_CONTACT |
| H4 | P0-09 / A03 | 汇总 PR、Run、SHA、断言、阻塞链，重算 READY 并发布本批结果 | 不改写已有 DONE；所有变更有可审计证据；检查持续运行与停止理由 |

## 依赖与调度

1. **CPU 即刻可做**：A（Runner）、C（只读证据契约与独立校验器）、D（已有 P4 数据独立复算）、E（历史坐标证据）、F（controller/IK/FSM）、G（数据/滤波）、H（训练门禁与队列）。禁止 CPU 审计意外启动 Kit。
2. **GPU 串行**：B 在核验 A04 锁后真实 Isaac N=2；D/E/F/G/H 的 GPU 实验按门禁、依赖和锁排队。Kit 退出码 0 不能保证测试成功，B 的最终状态须使用 A 线修复的判据或独立检查原始 fault 与 result。
3. **连接顺序**：P5-09 controller API + P6-08 集成静态场景 → P6-09 manifest/gate → P7 可控闭环；无门禁时可做离线/接口开发，不能标物理验证 DONE。
4. **交付规则**：每项 `START` 记录 owner/base_SHA、代码路径、验收断言、适用验证层级；提交时提供 commit/PR、真实命令与退出码、Run ID、原始证据、正反例、check_consistency。DONE 只表示该原任务 DoD 完整满足，不能把子项完成自动换算成队列 DONE。
5. **CHECKPOINT ≠ STOP**：每条 READY 工作完成之后继续领取其它合法 READY；发生真正阻塞记录证据、转向独立 READY。文件/评论发布并不会自动启动 DeepSeek 进程，执行须由已经连接该仓库任务入口的 Harness 主动消费。

## 保留的关键失败与权限边界

- P0-07：6 次真实 Kit 故障注入全部 outer exit 1；Python RuntimeError traceback 但 child_rc 0，SystemExit 不可靠；保留 typed fault 及 SIGSEGV 原日志。故障注入 gate PASS 不等于底层模拟执行 SUCCESS。
- P6：历史 `mounted_chain` 0.33242m、OPEN，v2 1.853e-09m，两者各属不同证据；未知外参仍 UNKNOWN。
- P4：0.30→0.31811、0.50→0.53007（约 6%），作为待分析偏差，禁止调阈值掩盖。
- P7：几何/代理闭环不等于物理接触和真实回球，`hit_rate=null` 不得凭空补值。
- P8：dependency_gate 的 11 条缺失证据使 PPO 拒绝，为预期安全结果；**无用户新明确授权不得启动 PPO、真机动作，不得自动合并受保护/default 分支。**
- 不覆盖历史故障产物，不伪造 `result.json`，不把 mock/offline 证据宣传为真实 Kit/物理验证。

## 本批完成后的统一回报

输出 32 子项逐条 `DONE / PARTIAL / BLOCKED / FAILED / UNVERIFIED`、对应原任务号、owner、branch/commit/PR、验证 L1–L5、证据 Run ID、实验命令/退出码、下一动作。最后运行 `tools/queue/check_consistency.py`（按当前真实 CLI），公布原 96 项队列刷新后的数据，并在 Issue #13 发布进度。不因完成此派工单而更改原始 96 项的状态。
