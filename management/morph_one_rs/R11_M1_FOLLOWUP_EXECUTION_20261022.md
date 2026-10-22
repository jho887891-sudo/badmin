# R11-M1-FOLLOWUP：真实视觉纠偏、广播检出恢复、三维/Isaac N=1 推进（2026-10-22）

> **正式派工，不是完成报告。** 本文件是 [R11 视觉主线](https://github.com/jho887891-sudo/badmin/blob/chatgpt/r11-vision-first-20261022/management/morph_one_rs/EXECUTION_R11_VISION_FIRST_20261022.md) 的后续执行切片，不新增 96 项权威队列稳定 ID，不改变 `38 DONE/13 PARTIAL/45 UNKNOWN` 的原始快照。最终以最新队列与真实实测刷新为准。
>
> **主验收目标：真实比赛/真实机载样式视频中，出现由 YOLO 实际推理产生、有人审真值验证的羽毛球检测，并能连续接到 2D EKF 和未来预测。** 同时逐步推进 G01 与 S01，但不可用其阻塞纯视频 M1。

## 0. 当前证据快照，必须保留

- PR [#56](https://github.com/jho887891-sudo/badmin/pull/56) · `r11/vision-m1@a967c82533eb67c509d92a091acf97367f1d6d59`：真实 YOLO 权重前向、真实 ETH 连续图像/比赛视频、`overlay.mp4`、`detections.jsonl`、`tracks.jsonl`、`metrics.json`、`manifest.json`。**M1=PARTIAL**，不是靠跑通程序就自动 PASS。
- ETH 60 帧有预测框 59/60；这是 **frames_with_detection**，不是人工 GT recall，且来源 train 帧，不算泛化。
- 广播比赛 doubles0.mp4 336 帧现有权重 conf 0.25/0.10/0.05/0.01 均为 0 检出。确认事实只到‘这段画面模型不出框’，领域差距/不可见球/缩放太小/模糊/输入处理等属于尚待区分的解释。
- 基于已入库 `R11_M1_ethseq_v2/tracks.jsonl` 的独立复算：旧 `sigma_major_px` 是**错误地对整个 4×4 [x,y,vx,vy] 协方差矩阵**求最大特征值，混用 px² 与 (px/s)²。默认 0.3s σ 中位被写 614.46px，正确 `sqrt(lambda_max(P_position_2x2))` ≈ **95.33px**；`ethseq_calib` 旧字段 15.17px，对应正确位置 σ ≈ **4.37px**。这是**已有数据的独立算术复核**，不是重跑 GPU。
- 噪声对照存在**时间轴混杂变量**：`ethseq_v2` 开头 0.533333→4.333333s 为 3.8s 间隙，`ethseq_calib` 则 0→0.033333s；旧 7.78px→18.51px 不能归因为单独的 process noise 差异。两种原始 Run 必须保留。
- PR #56 `metrics.json` 并不包含可复算的未来人工 GT 匹配明细；检测 p50≈13ms 也不能盖过原始 run 的 p99≈1892ms、max≈4575ms，需要冷启动与稳态拆分。
- PR #56 Review 已详细指出输入来源错误标记、`git_head=UNKNOWN`、图片清单未按图片内容 hash、目录输入 hash 失败路径。评论入口：[Issue #13](https://github.com/jho887891-sudo/badmin/issues/13#issuecomment-6337111330)。
- PR #27 的检测接口/EKF与 PR #51 的 IK/FSM 均可复用；后者还有运动中 HOLD 不等于真制动的既存审查缺陷。P6 球拍 TCP 和真实 D455 mount extrinsic 仍未验证。

## 1. 本批任务：16 个执行子项 / 4 条工作线

### A：M1 数值、时钟与证据可信度

Owner 建议：`vision-m1-correctness`。CPU 可优先修改/回放，GPU 只在复测 YOLO 时通过 A04 锁。开始前在 Issue #13 记录 owner、基准 SHA、worktree 和路径所有权，避免多个代理互相覆盖。

| 子项 | 原 SPEC 映射 | 任务动作 | 可证伪验收 |
|---|---|---|---|
| **A01** | P7-05/07/08 | 修复 horizon 的协方差单位：只用 4×4 状态协方差的 position 2×2 块算位置椭圆/σ，速度 2×2 块独立报 px/s；按正确位置协方差绘图 | 原已提交 ethseq_v2 的 0.3s σ 中位从错误 614.46px 改为经独立复算约 95.33px；calib 从错误 15.17px 对应约 4.37px；新 Run 不覆盖旧 Run；单测用速度协方差很大但位置方差小的负例，二维 coverage 与 ellipse 正确区分 1σ/95% |
| **A02** | P7-05/07 | 相同真实像素输入、完整相同帧顺序和时间戳、同一代码 SHA、同一帧位置 GT，仅变更 process_accel_std 2000→45 的单因素重跑 | 同一 timestamp source/序列、同一 evaluation pairs；不再将旧 7.78→18.51px 直接归因为噪声；显式报告 ethseq_v2 0.5333→4.3333s 的 3.8s 首帧跳变 |
| **A03** | P7-07/08 | 建立独立 future-time evaluator，对 0.3/0.5/0.8s 预测只匹配与目标未来时刻足够接近的 GT 帧，先记录标注误差/采样容差；不许未来帧泄漏给预测器 | 按每个 horizon 输出 N_valid、median/p95/p99（可得时）、位置覆盖率和置信校准；标注 GT 与‘未来 YOLO 预测框’伪 GT 另列；稀疏 val 不得插值造真值；同段 in-sample 不报泛化 |
| **A04** | P7-04/P7-13 | 统一 FrameSource 到 tracks/metrics/manifest 的来源标签，目录/文件列表/视频可重现指纹，记录脚本 Git SHA 与输入真实逐帧内容 hash；修目录输入 path hash 与 unknown git_head 路径 | ETH 序列 tracks.source_class=REAL_RGB_FRAMES；目录和 .txt 列表的原始帧变更使 provenance hash 改变；无可信 commit ref 则标 UNKNOWN_WITH_SOURCE_SHA 而非造 SHA；跨 Run/旧产物不得复用 |
| **A05** | P7-13 | 建立 M1 的‘零框 / 非零框 / 漏检 / 乱时钟 / video writer 失败’端到端负控；区分冷启动、模型 warmup 与稳态 decode/detect/track/predict/overlay 延迟 | 0/336 可判 PIPELINE_EXECUTED_NO_DETECTIONS 而非功能 M1 PASS；显式显示 p50/p95/p99,max，ETH 老 run detect p99≈1892ms/max≈4575ms 的根因/冷启动归因要由实验验证 |

### D：恢复真实比赛视频检测与可信评估

Owner 建议：`vision-broadcast-recovery`。标注与数据审计 CPU 并行；YOLO GPU 使用 A04 独占。开始前在 Issue #13 记录 owner、基准 SHA、worktree 和路径所有权，避免多个代理互相覆盖。

| 子项 | 原 SPEC 映射 | 任务动作 | 可证伪验收 |
|---|---|---|---|
| **D01** | P7-01/02 | 对 doubles0.mp4 的 336 帧逐帧抽样检查并建立人审真值：球存在/不可见/遮挡/运动模糊、bbox/质心、可见面积、原生 px 尺寸、标注可信度；没有标签不称召回率 | 标注数据及抽检截图/视频 sha，声明审阅范围与模糊不可判的帧；确认总可评价带球帧数分母。如果视频确实不可见球，则该片段不用于恢复正检测的主验收，补同分布有可见球素材 |
| **D02** | P7-02/03 | 保持现有 20-epoch checkpoint 与冻结标注不变，对原帧 640/960/1024/1280、conf 与 NMS、输入 letterbox/裁切/通道顺序做受控检测诊断 | 按有真值的 4–6/6–10/10–20/>20原生像素桶报告 TP/FP/FN、precision/recall 及真推理耗时；记录低 conf score 分布，不能单以低阈值有无框臆断根因 |
| **D03** | P7-03 | ROI crop/tiling/多尺度对照，每一方案记录裁剪来源，拼回原坐标与 NMS，禁止从 GT 生成生产路径的 oracle ROI | 如果用 GT-assisted crop，必须标 ORACLE_UPPER_BOUND，不计入可部署召回；正常运行只可依赖不泄漏的图像规则/前序 tracker ROI；同一视频与同一 GT 对照速度与误检 |
| **D04** | P7-01/03 | 根据 D01–D03 诊断构建最小有针对性的广播场景标注集与采样策略，优先真实帧、背景硬负样本、4–6px目标；需要时才开展受控小规模 YOLO 微调 | 训练/验证/测试按独立视频/session 隔离，`old/iphone_20251015` 已知跨 split OPEN 必须处理；仅在推理基线和数据分布证据支持时开启有限 YOLO 微调（不是 PPO）；保存 config/权重/epochs/曲线，测试集不得用于调参 |
| **D05** | P7-02/05/07 | 以独立真实视频、人工可评真值和固定模型/代码，交付‘有真球检测→EKF→像面未来预测’演示，并与 0 检出场景并列 | 至少报告目标可评价帧数、检测 TP/FP/FN、目标跟踪关联与未来预测 n/error/coverage，以及 overlay/JSONL；如果仍 0，诚实 BLOCKED，禁止填人工 bbox 为 YOLO 结果 |

### G：有来源的 3D 几何与击球接口

Owner 建议：`vision-geometry`。CPU 离线回放优先，真实相机使用需确认设备权限。开始前在 Issue #13 记录 owner、基准 SHA、worktree 和路径所有权，避免多个代理互相覆盖。

| 子项 | 原 SPEC 映射 | 任务动作 | 可证伪验收 |
|---|---|---|---|
| **G01** | P6-03/04/06 | 审计 D455 当前能实际获得的 RGB/Depth、内参畸变、同步/深度单位与外参来源；真实采集、合成夹具和 Isaac 校准必须分开 | D455 历史 USB/RGB+Depth 测通不等于已完成机器人 mounted extrinsic；输出可用性和 3D 可信级别，外参 UNKNOWN 则只允许 2D_ONLY/HOLD |
| **G02** | P7-06 | 只从匹配的有效深度或同步且标定的双目计算 3D 测量/协方差并验证不确定度与离群拒绝；小球无有效深度时拒绝 3D | 分别输出 DEPTH_VALID/STEREO_VALID/2D_ONLY/INVALID_DEPTH，正确 m/m² 单位、frame、time、calibration SHA；无有效 3D 观测不造 Z/速度 |
| **G03** | P7-09/10 | 将经验证的 3D 轨迹预测接到现有 P6 verified gripper_end FK/P5 controller API 和安全 FSM，列出可达与 HOLD 原因 | 球拍面 TCP UNKNOWN 与 gripper_end 可达区分；2D-only、深度失效、超时/越界时目标下发 write_count=0；禁止真机发送动作 |

### S：Isaac N=1 相机驱动的真实物理演示

Owner 建议：`vision-isaac-n1`。通过 A04 获得真实 Kit GPU 时段；不要并行占 GPU。开始前在 Issue #13 记录 owner、基准 SHA、worktree 和路径所有权，避免多个代理互相覆盖。

| 子项 | 原 SPEC 映射 | 任务动作 | 可证伪验收 |
|---|---|---|---|
| **S01** | P6-08/P7-11 | 独立启动最小 N=1 Isaac 场景，包含可渲染 RGB 摄像头、球、球拍碰撞几何，并输出有效的真实相机 RGB 帧与时钟/日志；不等待 R10 N=2 | 真实 Kit 启动和有效帧/截图或视频 + stage/source hash、exit/status；失败收 Kit 进程日志与 run id，不拿 mock/scaffolding 冒充 |
| **S02** | P7-11/12 | 同一 `run_visual_pipeline` YOLO 检测 Isaac 渲染 RGB，2D 跟踪，再经已声明几何生成测量/规划；独立评测器才可读球 GT | 删除/屏蔽策略端 GT 仍能执行感知和提出可解释拒绝，真实检测为 0 要保留失败，不能用 synthetic detector 或 GT boxes 混入 YOLO 路径 |
| **S03** | P7-12/13 | 在相机/几何/控制安全满足时尝试 N=1 受限球拍物理碰撞；先简化球拍可行，但分开记录目标运动、PhysX 接触事件与有效回球 | 无实际 PhysX contact 就 hit_rate=null；代理标 SIM_PROXY_PHYSX_CONTACT，完整 B601-RS 需真实 racket TCP + 可控 arm；不得把几何 overlap 当触球，也不启动 PPO/真机 |

## 2. 执行优先级与依赖（立即行动，不写二次总计划）

1. **两个 P0 立即并行**：A01/A02（位置 σ + 同钟单因素）；D01/D02（336 帧真值与失败根因）。A03–A05 和 D03–D05 可在无冲突工作树继续推进。先从有真像素及真 YOLO 推理产物直接做实验，不要等待 G/S。
2. **G01 与 S01 同时独立准备**：G01/P6 可通过 CPU 回放启动，S01 先查场景与资源，取得 A04 GPU 锁后串行启动。N=2 P5-07/08 不是 N=1 的硬前置条件，M1 更不等待它。
3. **M1 成功的判定**：两层不同：`PIPELINE_RUNTIME_PASS` 表示模型实际执行/保存完整输出；`M1_FUNCTIONAL_PASS` 还必须展示至少一段**可评价的真实非训练来源的有球录像**内 YOLO 对人审真球的正确检测+追踪和预测结果，列出 GT 匹配分母及 TP/FP/FN（指标是否够实际击球另评）。0 框可判 runtime 通过，但不能判功能通过。若 336 帧其实没有可见目标，应换可信真实测试视频并如实报告 336 帧不可评。
4. **不预设模型必须有多少 % recall**：先完整测量并决定可行性。对无法满足的指标可提出后续视觉迭代，但禁止根据测试集改阈值、人工填框或把自己训练序列的 59/60 误称 held-out 效果。
5. **时间/不确定度正确性**：4×4 的 x/y/vx/vy 混合单位状态不能整体求特征值解释为位置 σ；二维位置 95% 联合椭圆应用 `chi2_df2(0.95)=5.991` 的尺度，不能把 2σ 圈误称为 95% 联合区域。未来时刻误差只与匹配到的有效真实未来时刻相比。
6. **输入真实可溯源**：保存真实视频/逐图像内容 sha、模型 sha、代码 SHA、时间戳 provenance、原始 YOLO 逐帧结果、实际设备、CUDA/Ultralytics 版本、计时各阶段。所有失败 run 保留，不在旧目录重写数据。
7. **安全边界**：没有有出处的 3D 几何就输出 2D_ONLY/HOLD，禁止 2D bbox→3D 伪转换、GT→策略泄漏；未授权 PPO、真实机械臂运动、默认/受保护分支自动合并；Kit GPU 独占且真实日志 fail-closed。
8. **验收回报**：每个子项 owner、branch/SHA/PR、Run ID、raw/overlay/GT 标注路径、完整命令和 exit、前后比较的同口径指标、负例及剩余问题；新工作线局部完成不自动增加原 96 项队列 DONE。一次会话结束时按真实已运行情况保存 checkpoint，不能声称后台继续执行。

## 3. 首个检查点必须交付的东西

- A01/A02 的可独立复算 0.3/0.5/0.8 秒位置协方差/GT 对照修订报告及两组同钟重跑证据。
- D01 对广播视频真实可见球的有无/原生 px 尺寸分布与标注样例，D02 至少一组合法模型真实推理对照；不准把 0 框只解释为“领域差距”。
- G01 的真实/合成/仿真标定事实矩阵，S01 的 Isaac N=1 真渲染执行结果（或确切错误日志与后续排查步骤）。
- 尽快提供**视觉效果视频**，而不只是更多 pytest PASS；无可评价视频时明确缺什么数据。
