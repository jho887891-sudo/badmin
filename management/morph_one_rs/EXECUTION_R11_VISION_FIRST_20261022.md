# R11：视觉路线优先执行单（Visual-First Vertical Slice）

> **2026-10-22 新优先级指令**。这是下发给项目执行代理的**开发任务**，不是已完成报告；本批 24 子项映射到原 P0–P8 96 项 SPEC，不自动增加 DONE 或更改原队列状态。
>
> 仓库：[jho887891-sudo/badmin](https://github.com/jho887891-sudo/badmin) · 总入口：[Issue #13](https://github.com/jho887891-sudo/badmin/issues/13) · [R10 原派工](https://github.com/jho887891-sudo/badmin/pull/50)。

## 0. 最高目标与停止伪闭环

**必须让真实像素流进入 YOLO，再经追踪、三维几何和安全规划，使仿真控制由视觉估计驱动，最终尝试 PhysX 接触。** 禁止直接从解析/仿真球的位置真值输入策略后报告“视觉闭环”。

### 三个里程碑与验收

| 里程碑 | 真实输入/主链 | 必交结果 | 何时才能宣称完成 |
|---|---|---|---|
| **M1 真实视频 2D 视觉闭环** | 真实录制 RGB 视频/相机 → 已有 YOLO 权重实际推理 → 2D EKF → 0.3–0.8s 像面预测 | 可回放 `overlay.mp4`、`detections.jsonl`、`tracks.jsonl`、实际 latency、模型/视频 SHA | **真实模型 + 真实逐帧输入**均运行，独立复现；仅接口/合成 fixtures 不算 M1 |
| **M2 可测三维 + 预测击球点** | 同步 RGB/Depth 或双目 → 校准几何 → 3D 状态和预测 → 多候选 IK + 安全拒绝 | `intercepts.jsonl`、标定来源及残差/置信区间、可达/不可达统计 | 确有可信三维**观测**，不靠 GT；仅 Isaac 标定时标 `SIM_CALIBRATED_3D`，不得称真实 D455 标定完成 |
| **M3 单场景 Isaac 真物理接触** | Isaac N=1 渲染 RGB → **同一 YOLO/估计/预测代码** → 规划/动作 → PhysX 物理接触 | `camera_frames`/视频、每帧链路、运动及 `contact_events.jsonl`、试次数/接触率 | 必须有 PhysX 接触事件；若仅简单执行器球拍，报告 `SIM_PROXY_PHYSX_CONTACT`，不能宣称完整 B601-RS 视觉击球 |

依赖顺序：M1 可完全独立于 Isaac/机械臂执行。M2 需要真实或仿真已标定几何才能升 3D；没有有效 3D 时只许输出像面轨迹与 HOLD。M3 首先用**单场景 N=1**，先不要被 P5-07/08 的 N=2 崩溃阻塞；完整 B601-RS 打球另受 P6/P7 的机械臂、球拍 TCP、安全门禁约束。

## 1. 已有成果，优先复用而不是推倒重做

- PR #27 / `p7/detector-ekf-interfaces`：`src/perception/shuttle_detection/detection_output.py`、`src/badminton_brain/estimation/shuttle_state_ekf.py`、`src/badminton_brain/prediction/horizon_prediction.py`，**只证明接口及合成测试**。
- PR #51 / `r10/f-arm-api-safety`：已有 `simulation/badminton_scene/controller_api.py`、IK/FSM、size bucket evaluator、offline latency；`scripts/shuttle_detection/evaluate_size_buckets.py` 的预测是**随机合成夹具，并未加载 YOLO 模型**。严禁拿这个评估的合成数字当真实检测性能。
- 现有 P7 数据审计显示历史 `old/iphone_20251015` session 跨 train/val 的泄漏仍 OPEN；`eq640_min` 在仓库无权威定义，不能称为原始目标像素。
- PR #47/51 的模拟闭环多处直接从解析球源开始，`SIM_PROXY_GEOMETRY_ONLY` 不等于物理碰撞，`hit_rate` 继续为 null；P6 verified FK 仅保证 `gripper_end`，球拍 TCP 未认证。
- R10 已报告 P0-08 6 次 P0F 回放无 SUCCESS；A/C 有 fault schema 不一致等跨线审查红灯，M3 新运行必须使用 fail-closed 的原始证据判定，不能只相信 child_rc=0。
- R10 B 线 Isaac N=2 第 7 次步进期间静默终止、child_rc=0、无 result；当前只能判 INCOMPLETE，不能宣称已定位为 crash。N=2 降级为**非 M1/M2 阻断工作**。
- 已有 96 项队列基线：38 DONE /13 PARTIAL/45 UNKNOWN（来自 2026-10-22 R10 回报；执行前刷新 Git SHA 和队列 READY 派生）。本任务不改写历史事实。

## 2. 六条并行工作线，24 个可执行子项

### V · 真实数据与输入来源

**建议 owner**：`vision-io`，独立 worktree；必须在 Issue #13 领取且确认无路径冲突。CPU 工作并行；需要真实 YOLO GPU 的运行和 Isaac/Kit 使用 A04 独占锁排队。

| 子项 | 原队列任务 | 执行动作 | 验收证据 |
|---|---|---|---|
| **V01** | P7-01 | 清点可运行 YOLO 权重、真实视频/逐帧图像、D455 回放、对应 GT 标注、设备信息与 sha256；不可用资源明确报缺失 | 资源清单：路径或外部可复现获取方式、帧数/FPS/真实时间戳、分辨率、分组、hash；不得造数据 |
| **V02** | P7-01 | 审计 train/val/test 会话分组：显式处理 old/iphone_20251015 已知 session 跨 split；冻结至少一个不泄漏的回放/评估集 | 无同一录制 session 跨 test/train；缺 test 时标 OPEN、先用探索集但不报泛化指标 |
| **V03** | P7-04/P6-06 | 实现视频、图像序列、相机的统一 FrameSource，输出实际 capture_ts、frame_id、camera_id、image_size、frame_data | 禁止用 frame_id/FPS 伪称硬件同步时间；缺 PTS 显式 ASSUMED；音视频读取失败标记 |
| **V04** | P7-13 | 实现统一 CLI 的输入/写盘/退出路径和 provenance：run_id、视频/权重 SHA、软件版本、完整原始逐帧记录 | 演示 CLI：python scripts/vision/run_visual_pipeline.py --source <real_video> --weights <best.pt> --out-dir <new_run>; 输出原始检测日志和轨迹可视化 |

### D · 真实 YOLO 推理与尺寸分桶

**建议 owner**：`vision-detector`，独立 worktree；必须在 Issue #13 领取且确认无路径冲突。CPU 工作并行；需要真实 YOLO GPU 的运行和 Isaac/Kit 使用 A04 独占锁排队。

| 子项 | 原队列任务 | 执行动作 | 验收证据 |
|---|---|---|---|
| **D01** | P7-02/P7-04 | 接入已有 YOLO checkpoint 做逐帧实际推理；适配 DetectionObservation(timestamp_s, frame_id, camera_id, bbox_xyxy_px, confidence, pixel_covariance) | 实际打开图片/实际模型前向；记录权重 sha256、输入变换、NMS 参数、设备、每帧检测耗时；不得合成预测记录替代 |
| **D02** | P7-02 | 输出 4–6px、6–10px、10–20px、>20px（也含 <4px）召回、precision、FP/FN、负样本误报；IoU50 和中心距规则并列 | 仅对有真实人工标签的有效 split 统计；按 native 原像素重算目标尺寸，禁止拿未定义 eq640_min 当原始像素 |
| **D03** | P7-03 | 在同一冻结视频/标注和设备上比较 640/960/1280，以及合适的 ROI crop/tiling；记录完整耗时与分桶效果 | 真实推理至少先跑已有可用配置；不可执行的配置标 NOT_RUN；不按测试表现事后改变阈值来制造改善 |
| **D04** | P7-02/P7-08 | 设计检测置信度、bbox 协方差与误检/漏检诊断；根据现有数据做误差校准而非固定虚构低方差 | 投影与 2D tracker 可用；低置信度、出框、不可信尺寸一律拒绝进入高置信 3D 执行 |

### T · 2D EKF 跟踪与像面预测

**建议 owner**：`vision-tracker`，独立 worktree；必须在 Issue #13 领取且确认无路径冲突。CPU 工作并行；需要真实 YOLO GPU 的运行和 Isaac/Kit 使用 A04 独占锁排队。

| 子项 | 原队列任务 | 执行动作 | 验收证据 |
|---|---|---|---|
| **T01** | P7-05 | 复用 PR #27 的 shuttle_state_ekf.py 和 detection_output.py，接入真正逐帧 YOLO 输出；实现状态关联、多目标/误检裁决 | 2D 状态位置/速度/协方差在像素域，输出 [cx,cy,vx,vy]、cov、关联 ID，不能伪造三维速度 |
| **T02** | P7-05/P7-08 | 处理不规则 dt、空检测/遮挡、重复/乱序时间戳、重放过期检测、创新门控/噪声漂移 | 负例验证不复用过期目标；遮挡时不确定度增加，轨迹过期丢弃/HOLD |
| **T03** | P7-07 | 实现基于真实观测的 2D horizon 0.3/0.5/0.8s 预测，明确输出 px、px/s、px² 和置信区间 | 像面预测可以画在视频上，但必须标 IMAGE_PLANE_2D，不可交给米制 IK/机械臂 |
| **T04** | P7-07/P7-08 | 逐帧将 bbox、轨迹 ID、历史中心、预测点/区间、漏检状态、处理延迟叠加到导出视频，并写 track.jsonl | 从真实视频可重跑得到 overlay.mp4/track.jsonl；每帧 timestamp 与输入来源匹配；不能只输出静态图/合成 fixtures |

### G · 标定几何与三维状态

**建议 owner**：`vision-geometry`，独立 worktree；必须在 Issue #13 领取且确认无路径冲突。CPU 工作并行；需要真实 YOLO GPU 的运行和 Isaac/Kit 使用 A04 独占锁排队。

| 子项 | 原队列任务 | 执行动作 | 验收证据 |
|---|---|---|---|
| **G01** | P6-03/P6-04/P6-06 | 审计 D455 intrinsics、depth 单位/有效率、相机内外参、optical/world/base 坐标、时间同步，列出真实或仿真校准来源 | 真实 mount extrinsic 未测保持 UNKNOWN；synthetic @self 或 assumed I 不是 CALIBRATED_REAL |
| **G02** | P7-06 | 从同步双目或可信且有效的深度测量恢复米制 3D 位置+协方差，传播像素噪声和深度不确定度 | 每帧输出 3D_VALID / 2D_ONLY / INVALID_DEPTH + provenance；严禁拿 2D bbox 或仿真 GT 直接冒充测量 |
| **G03** | P7-06/P7-07 | 以实际 3D 测量序列估计速度与 0.3–0.8s 三维预测，输出机器坐标系中的候选截点、时戳与可信度 | 标出时空坐标转换和几何来源；无可靠 3D 测量时返回 HOLD/NO_3D，不回退到猜测 Z |
| **G04** | P6-05/P7-06 | 用 Isaac 相机合法内外参 + 渲染 RGB/Depth 建立独立评测器，仿真真实 pose 仅供事后 metrics | policy 进程输入禁用 shuttle/world GT；测试证明删掉真值通道仍能预测；允许拿相机校准参数但记录来源 |

### I · 视觉到击球决策安全接口

**建议 owner**：`vision-to-ik`，独立 worktree；必须在 Issue #13 领取且确认无路径冲突。CPU 工作并行；需要真实 YOLO GPU 的运行和 Isaac/Kit 使用 A04 独占锁排队。

| 子项 | 原队列任务 | 执行动作 | 验收证据 |
|---|---|---|---|
| **I01** | P7-09 | 统一 PredictedIntercept 契约：target_xyz_m、frame_id、intercept_time、covariance、confidence、source_class、calibration_hash | 不合法单位、缺外参、过期帧、不可达截点、未知 racket TCP 一律拒绝驱动 |
| **I02** | P7-09/P5-09 | 对接 verified P6 FK / P5 controller API，分清 gripper_end 与 racket_face；按多个未来击球时刻试探 IK | 不能把 verified gripper_end 可达性称为已验证球拍面可达；TCP 未标定时只能做受限的 SIM_PROXY 规划 |
| **I03** | P7-10 | 把 SEARCH→TRACK→PREDICT→PLAN→EXECUTE/HOLD 接到真实视觉状态，不允许用解析轨迹真值喂策略 | 必须演示 2D_ONLY/STALE/OUT_OF_REACH/UNCALIBRATED/WRITE_REFUSED 等负例，unsafe 写入计数 0 |
| **I04** | P7-10/P7-13 | 修复 PR #51 控制器安全语义：settle 必须测量后验证到位；运动中 HOLD 的拒绝不冒充实际制动 | 将 command suppression / measured-pose hold / emergency stop 支持级别分开；无硬件制动证明只声明接口级安全 |

### S · 单环境 Isaac 视觉物理接触演示

**建议 owner**：`vision-sim`，独立 worktree；必须在 Issue #13 领取且确认无路径冲突。CPU 工作并行；需要真实 YOLO GPU 的运行和 Isaac/Kit 使用 A04 独占锁排队。

| 子项 | 原队列任务 | 执行动作 | 验收证据 |
|---|---|---|---|
| **S01** | P6-08/P7-11 | 优先建单环境 N=1、最小相机/羽毛球/球拍/有效物理碰撞场景；避免将未解决的 N=2 作为 M1/M2 阻塞 | Kit 真实启动、每帧相机渲染有效、时间戳、真实日志，不能只做 mock USD 静态检查 |
| **S02** | P7-11/P7-12 | 单环境视觉感知接入 S01 的真实渲染 RGB；相机 2D→3D 只能经声明的几何测量链 | 策略不接收球的 GT 位置；输出完整帧→Detection→State→Intercept→Action 记录 |
| **S03** | P7-11/P7-12 | 当标定/TCP/IK 成立时执行受安全边界约束的物理击球；如先用代理球拍 actuator，清楚标 SIM_PROXY_PHYSX_CONTACT | 接触必须由 PhysX contact reporter/传感器证实，记录真实 contact event、接触对象、时间、场景 hash，不可用几何重叠替代 |
| **S04** | P7-12/P7-13 | 分别评估静止球、低速确定轨迹、小扰动 3 类；每组记录尝试数、视觉检测率、预测/拦截误差、真实接触率、有效回球率 | 无物理接触报告时 hit_rate=null；区分 SIMPLE_PADDLE_PROXY、B601RS_FULL_ROBOT、REAL_ROBOT；GPU 失败则保持 BLOCKED，保存 Kit 崩溃日志 |

## 3. 验收协议和禁止偷换输入

1. **不允许 GT 泄漏**：任何运行保存 `input_source_class` = REAL_RGB_VIDEO / REAL_D455 / ISAAC_RENDERED_RGB；`measurement_source` = YOLO_BOX / VALID_DEPTH / CALIBRATED_STEREO。球 GT 在仿真只进 **独立评价器**，不能进入 detector/tracker/predictor/IK 的输入。通过故障注入证明断开 GT 通道后策略仍能运行。
2. **清楚区分三种时钟**：视频 capture/PTS、主机单调接收时间、处理时间；相机帧/深度与关节状态必须匹配采样时间。只有 frame index 时不能虚构同步精度。
3. **统一单位和不确定度**：bbox/track/prediction2d 用 px、px/s、px²；只有有内外参+有效深度/双目时可产出 3D m、m/s、m²；投影/变换必须带 `frame_id` 和版本化校准 SHA。无可信 3D 时 `2D_ONLY/NO_3D/HOLD`。
4. **实时测量，不接受玩具指标**：实际运行独立记录 decode/detect/track/predict/plan/control 延时（p50/p95/p99）、有效帧率、漏检序列长度、跟踪断裂、预测误差按 horizon 分桶。合成录像/测试夹具只可作 L2，不可兑付 M1 的真实视频或 M3 的实际 PhysX 接触。
5. **最小 CLI 验收目标**：

   `python scripts/vision/run_visual_pipeline.py --source <existing_video_path> --weights <existing_best_pt> --out-dir <new_unique_run_dir>`

   实际路径由读取仓库/资产清单决定；不写死不存在的模型权重。应至少得到源/权重元信息、原始视频采集清单、`detections.jsonl`、`tracks.jsonl`、`overlay.mp4`、`metrics.json`、日志、错误原因与 exit status。没有真实资源时报告具体阻塞并继续开发可检验接口，不得伪称 M1 PASS。
6. **M3 最小物理场景允许分层**：第一层独立简单球拍执行器 PhysX 接触 `SIM_PROXY_PHYSX_CONTACT`；第二层完整 B601-RS + 经核实的 racket TCP + 安全速度控制 `B601RS_FULL_ROBOT_PHYSX_CONTACT`。两个标签不得互换。撞击接触事件和有效回球率分别统计，二者也不能互换。
7. **关键负控**：空检测、假阳性、帧丢失、过期帧、时间倒流、深度无效、外参 UNKNOWN、视差不足、坐标帧混淆、无法 IK、未标定球拍 TCP、越限写入、子进程静默退出。要求写入数/动作执行为 0 或返回已定义的安全拒绝；`hold()` 抛异常不等于物理急停成功。
8. **持续交付**：每子项记录 owner/base SHA、代码与复现命令、真实 exit、传感器/视频/模型 SHA、Run ID、原始帧/JSONL、正反例、验证层级、PR。只凭本任务文件完成不得更改原队列状态。

## 4. 执行调度与资源约束

- **第一批立即启动**：V01/V02/V03、D01、T01/T02、G01、I01、S01 的单环境最小框架（S01 需要 Kit 且获 A04 锁）；优先从已有真实视频跑出首个可视化，不等所有校准和整机门禁。
- **第二批**：D02/D03/D04、T03/T04 完成 **M1**；G02/G03/G04 的有效测量链 + I02/I03/I04 在可用标定/控制接口上推进 **M2**。
- **第三批**：S02→S03→S04，完成 **M3** 物理接触的对应实际层级；若没有有效相机/Isaac 实验环境，先补 evidence + CPU 负控，保持 BLOCKED，继续其它 READY。
- **基础设施止损**：R10 A/C 合并前在新运行入口仅补必要的 fail-closed 运行真实性检查，禁止无限扩展 unrelated runner/队列特性。P4 剩余约 +1.27% 和 N=2 扩展不是 M1 主路径；留档但不抢视觉主线 GPU。
- **权限**：禁止 PPO 训练、真机机器人动作、覆盖历史失败证据、自动合并 main/default/受保护分支或 force-push 协作分支。真实 Isaac GPU 串行，视频 GPU 推理同受 A04 保护；独立 CPU 可以并行。

## 5. 交付表与优先级退出规则

持续更新 Issue #13。每个里程碑给出 **PASS / PARTIAL / BLOCKED / FAILED / NOT_RUN**、与其相称的真实验证层级，绝不按任务完成数推断里程碑。须演示：输入视频/帧 → 实际 YOLO 框 → 2D EKF 跟踪 → 有证据的 3D（若可用）→ 候选击球点 + 拒绝原因 → 单环境真实 PhysX contact（若实际发生）。

**先交能真实复现的 M1 视频和日志，再扩展 M2/M3；单个 PR/报告完成不等于 STOP。** 会话无法继续时保存准确检查点，不声称有后台自主运行。
