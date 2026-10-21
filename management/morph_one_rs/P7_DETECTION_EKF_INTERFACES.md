# P7-04 / P7-05 / P7-07 接口定义与单测（检测输出 / 2D 优先 EKF / 轨迹预测）

- 分支：p7/detector-ekf-interfaces（基于 origin/main）
- 任务：P7-04 检测输出统一 schema；P7-05 只见 2D 时成立的 EKF 状态定义（含过程/测量噪声、门控/离群拒绝、丢失测量时的行为）；P7-07 0.3–0.8 s 预测接口（horizon + 误差/置信区间 + 过期帧拒绝）
- 环境：本地 Windows，Python 3.11.9，pytest 9.0.2，numpy 2.4.6；**CPU only，无 GPU，不训练，未重跑任何检测**
- 全部证据均为**合成数据**：本文所有数字都是接口/一致性统计量，**不是**真实检测精度，也不是真实飞行精度。

---

## 1. 为什么要先做接口（P7 的现状）

仓库里已有的相关实现：

| 已有 | 位置 | 与 P7 的关系 |
|---|---|---|
| 数据集审核契约（split/source/issue） | src/perception/shuttle_detection/contracts.py | 只描述**数据集**，不描述检测输出 |
| 3D 测量契约（米 / 球场系） | src/badminton_brain/types.py: ShuttleMeasurement | 只有 3D；2D bbox 直接塞进去就等于伪造 3D（P7-06 明确禁止） |
| 3D UKF（位置/速度/阻力，含风） | src/badminton_brain/estimation/shuttle_ukf.py | **必须有立体点**才能工作，做不了"只见 2D"的估计器 |
| 物理预测器（RK4 滚动） | src/badminton_brain/prediction/physics_predictor.py | 接受任意 horizon、无不确定度、无法表达"基于过期帧" |
| 立体几何/标定 | src/badminton_brain/perception/stereo_geometry.py | 相机内参/外参都是 TEMP 工程基线，真实标定 UNKNOWN |

因此这三个任务补的是**缺的那一层**，不改动已有契约：

    detector --DetectionObservation(px)--> ShuttleStateEKF(IMAGE_PLANE_2D)
                                              |
                                              +-- upgrade_to_3d()  (需要可信深度 + 已标定相机)
                                              v
                                          COURT_3D 状态 (m, m/s, cov)
                                              |
                                              v
                                    PolynomialHorizonPredictor (0.3-0.8 s)
                                              |
                                              v
                                    TrajectoryPrediction（含 95% 区间）

**兼容性影响：无破坏性变更。** 三个模块都是新增文件，不修改任何现有 dataclass / 函数签名；ShuttleMeasurement、PredictedTrajectory、ShuttleUKF、PhysicsTrajectoryPredictor 行为不变（回归测试见第 5 节）。

---

## 2. P7-04：检测输出 schema（src/perception/shuttle_detection/detection_output.py）

### 2.1 契约

| 字段 | 类型/单位 | 校验 |
|---|---|---|
| timestamp_s | float, s | 有限、>= 0 |
| frame_id | int | >= 0，必须是整数（bool 也拒绝） |
| camera_id | str | 非空、非纯空白 |
| bbox_xyxy_px | (4,) float, **px** | 有限；x2>x1、y2>y1；x1,y1 >= 0；不允许归一化坐标 |
| confidence | float, [0,1] | 有限且在闭区间内 |
| pixel_covariance | 2x2 float, **px^2**（bbox 中心的协方差） | 有限、对称、半正定（PSD）、对角 >= 像素下限 |
| image_size | ImageSize(width_px, height_px) | 整数且 > 1 |
| validity | ValidityFlags | detected / in_image / confidence_ok / covariance_ok / timestamp_ok / reject_reason |

ValidityFlags.usable 只有在**所有**标志为真且 reject_reason == "NONE" 时为真；reject_reason 取自封闭词表（NOT_DETECTED / LOW_CONFIDENCE / GEOMETRY_OUT_OF_IMAGE / NON_FINITE / BAD_COVARIANCE / STALE / UNIT_CONFUSION）。

关键设计：in_image 是**派生量**，构造时会与 bbox/图像尺寸重新比对，**标志与几何矛盾直接报错**——"可以撒谎的 flag 比没有 flag 更糟"。

### 2.2 JSON 往返

to_json_dict() 输出纯 JSON 类型（无 numpy），并自带 units 块：

    "units": {"bbox_xyxy_px": "px", "pixel_covariance": "px^2",
              "timestamp_s": "s", "confidence": "1"}

validate_detection_payload() / from_json_dict() 是**唯一校验入口**，拒绝：
1. schema / schema_version 不匹配；
2. 缺 units 或 units 不是 px / px^2；
3. 出现米制键名（bbox_xyxy_m、position_m、velocity_mps、depth_m 等）→ 判定为单位混淆；
4. strict=True 时的未知键（不允许夹带私货，扩展请改 schema）；
5. 任何字段级违规（NaN/Inf、范围、PSD、flag 矛盾）。

### 2.3 像素 / 米绝不允许混淆的三道闸

1. **键名**：米制键名出现即拒绝。
2. **单位块**：payload 必须自报 px / px^2。
3. **协方差像素下限** MIN_PIXEL_VARIANCE_PX2 = 0.01 px^2（σ = 0.1 px）：亚像素质心估计在理想大目标上也只到 ~0.1 px，而 4–6 px 小球的 bbox 检测不可能更好（management/shuttle_detection/closure_questions_interim.md 的 4–6 px 尺寸桶）；而真实立体 σ = 0.01–0.1 m 对应 1e-4–1e-2 m^2，正好落在下限及以下。该闸门**故意保守**：它只会拒绝"不可能属于真实检测的像素协方差"，是单位闸门，不是精度声明。

---

## 3. P7-05：2D 优先的 EKF（src/badminton_brain/estimation/shuttle_state_ekf.py）

### 3.1 状态定义

| 模式 | 状态 | 维度 | 单位 |
|---|---|---|---|
| IMAGE_PLANE_2D | x = [u, v, u_dot, v_dot] | 4 | px, px/s |
| COURT_3D | x = [px, py, pz, vx, vy, vz] | 6 | m, m/s |

- 滤波器**只能以 2D 构造**；ShuttleStateEKF(mode="COURT_3D") **直接抛错**。3D 状态只能通过 upgrade_to_3d() 进入。
- 模型：匀速（CV），F = [[I, dt*I],[0, I]]；过程噪声用分段常加速度离散化（Bar-Shalom, Li & Kirubarajan 2001, §6.2.2）：Q_axis = a^2 * [[dt^4/4, dt^3/2],[dt^3/2, dt^2]]。
- 更新用 Joseph 形式 P = (I-KH)P(I-KH)^T + KRK^T 保持对称半正定。

### 3.2 显式参数（全部是 Param，带 status 与来源）

| 参数 | 默认值 | 单位 | 状态 | 理由（摘要） |
|---|---|---|---|---|
| PROCESS_ACCEL_STD_PX_S2 | 2000 | px/s^2 | TEMP_PARAMETERIZED_PROXY | 冻结气动模型 k=1/6.5 1/m，v≈4.5 m/s 时阻力减速 3.1 m/s^2；1280x720、fx≈700 px 的工程基线相机约 700 px/m ⇒ ≈2200 px/s^2。取 2000 为量级值（透视放大处偏乐观）。 |
| MEASUREMENT_STD_PX | 2.0 | px | TEMP_PARAMETERIZED_PROXY | 本项目小目标桶为 4–6 px，2.0 px ≈ 该桶上界的三分之一，作为 bbox 中心抖动的合理代理；**有 per-detection pixel_covariance 时以它为 R**。 |
| MEASUREMENT_STD_PX_MEASURED | None | px | REQUIRES_MEASUREMENT | 真实 bbox 抖动从未测量（静态目标重复帧 / 标注残差按尺寸桶统计）。 |
| PROCESS_ACCEL_STD_MP_S2 | 3.0 | m/s^2 | TEMP_PARAMETERIZED_PROXY | 同一条 k*v^2 界（3.1 m/s^2）覆盖无阻力状态的未建模阻力；shuttle_ukf 用 1.0 + 显式阻力状态，本滤波器没有阻力状态，故取更大值。 |
| MEASUREMENT_STD_M | None | m | REQUIRES_MEASUREMENT | 立体噪声未表征；**COURT_3D 下拒绝发明 R**，必须由调用方传入深度/三角化协方差（measurement_covariance=）。 |
| MEASUREMENT_STD_M_MEASURED | None | m | REQUIRES_CALIBRATION | 需要标定 + 静态目标残差研究。 |
| INITIAL_POSITION_STD_PX / VELOCITY_STD_PX_S | 8 px / 600 px/s | — | TEMP | 初始化不给协方差时的宽松先验（约 2 倍小目标桶；~1 s 穿过画面 ⇒ 600 px/s）。 |
| INITIAL_POSITION_STD_M / VELOCITY_STD_MPS | 0.1 m / 1.0 m/s | — | TEMP | 仅 3D 升级在无协方差输入时使用。 |
| GATE_CHI2_2DOF_099 / 3DOF_099 | 9.2103 / 11.3449 | — | TRACEABLE_REFERENCE | chi^2(2)/chi^2(3) 0.99 分位（标准表）：门控按设计丢弃 ~1% 内点。 |
| MAX_COAST_STEPS | 5 | 帧 | TEMP | 50 Hz 下 0.1 s，即"还能靠预测撑多久"；超过即报 LOST，不假装知道球在哪。 |
| OUTLIER_STREAK_REINIT | 3 | 次 | TEMP | 连续 3 次被门控拒绝 ⇒ 认为预测错了而不是测量错了，用最新测量**重新初始化**并在 UpdateResult 里报告 REINIT。 |
| DEPTH_RATE_STD_PROXY_MPS | 2.0 | m/s | TEMP_PARAMETERIZED_PROXY | 有深度但无深度变化率时，3D 速度无法诚实升级，用该代理膨胀速度协方差并在结果中 velocity_from_depth_rate=False。 |

### 3.3 门控 / 离群拒绝

d^2 = y^T S^-1 y（y 为新息，S = HPH^T + R），d^2 > gate_chi2 则：
- **不做更新**（状态与协方差逐位不变，单测用 assert_allclose 断言）；
- UpdateResult(accepted=False, reason="GATE", mahalanobis_d2, nis, ...) 明确返回；
- 连续达到 OUTLIER_STREAK_REINIT 次 → reason="REINIT"，位置重置为最新测量、速度置零并给出对应协方差。

### 3.4 测量停止时的行为（文档化 + 单测）

只调用 predict(dt) 时：状态按 CV 模型**精确**推进（不发明任何测量），协方差按 Q 单调增长，steps_without_measurement 递增，status 依次为 TRACKING → COASTING（<= MAX_COAST_STEPS）→ LOST。永久丢失不会自动重置，必须外部 reset()。

### 3.5 2D → 3D 升级（P7-06 门禁）

upgrade_to_3d(depth_m, depth_sigma_m, camera, depth_source, ...) 只在以下条件同时满足时执行，否则**抛类型化异常**：

| 拒绝条件 | 异常 |
|---|---|
| depth_source 不在 STEREO_TRIANGULATED / STRUCTURED_LIGHT / LIDAR | DepthNotTrustworthyError（MONOCULAR_GUESS 永远拒绝） |
| depth/sigma 非有限或 <= 0 | DepthNotTrustworthyError |
| camera.calibration != "CALIBRATED" | UncalibratedCameraError（只有显式 allow_uncalibrated=True 的合成测试可绕过，并在结果里记录 provenance） |
| 滤波器已在 3D / 未初始化 | StateModeError / ShuttleEKFError |

升级数学：针孔 p_cam = d*((u-cx)/fx, (v-cy)/fy, 1)，p_court = R*p_cam + t，对输入 [u, v, u_dot, v_dot, d, d_dot] 求 6x6 雅可比 J，P3 = J * blkdiag(P2, sigma_d^2, sigma_ddot^2) * J^T——3D 协方差**同时继承像面协方差与深度误差**，而不是把 2D 协方差当 3D 用。速度含 (u_dot/fx)*d + ((u-cx)/fx)*d_dot 项，因此**没有 d_dot 时速度只能靠代理膨胀**（见上表）。

---

## 4. P7-07：horizon 受控的预测接口（src/badminton_brain/prediction/horizon_prediction.py）

### 4.1 契约

| 项 | 约束 |
|---|---|
| horizon_s | **预注册区间 [0.3, 0.8] s**（ROBOT_BRAIN.md 4.3，physics_predictor 的 DEFAULT_HORIZON_S=0.6 同源）。越界 → PredictionRejected("HORIZON_OUT_OF_RANGE")，**不静默外推**。该区间在任何测量之前固定，事后不放宽。 |
| 时间基准 | times_s[0] == reference_timestamp_s，times_s[-1] == reference + horizon_s，严格递增 |
| 输出 | position_m (T,3)、velocity_mps (T,3)、position_covariance_m2 (T,3,3)、error_bound_m (T,) = z*sqrt(lambda_max)、ci_lower/upper_m (T,3) = p -/+ z*sigma_axis |
| 置信水平 | 0.95（z = 1.959963984540054），只支持 0.90/0.95/0.99，其他值拒绝（不做近似） |
| 一致性不变式 | 构造时**重新计算** error_bound 与 CI 并与传入值比对，任何篡改 → DEGENERATE_SAMPLES |
| 过期帧 | measurement_age_s = now_s - max(t_i)；> MAX_MEASUREMENT_AGE_S → PredictionRejected("STALE_MEASUREMENT")；负年龄（观测在未来）→ "FUTURE_OBSERVATION" |
| 样本量 | < MIN_FIT_SAMPLES(4) → INSUFFICIENT_SAMPLES；时间戳重复或不递增 → DEGENERATE_SAMPLES；拟合法方程 cond > 1e12 → DEGENERATE_SAMPLES |

MAX_MEASUREMENT_AGE_S = 0.10 s 是 TEMP 代理（50 Hz 的 5 帧；理由：0.3 s 最小 horizon 减去 ~0.1 s 感知/推理延迟后仍要留 0.2 s 给决策规划，因此比自身延迟预算更旧的帧不能进入决策级预测）。真实端到端延迟属 P7-13，对应 MAX_MEASUREMENT_AGE_S_MEASURED = None (REQUIRES_MEASUREMENT)。

参考实现 PolynomialHorizonPredictor：按每帧 sigma 加权的二次多项式拟合（常加速度的精确模型），用 var(t) = j(t)^T (A^T W A)^-1 j(t) 传播到外推点。它**不是**真实预测器，只是让接口可测的参考实现；真实飞行应接 physics_predictor 的 RK4 模型并替换 uncertainty 来源。

---

## 5. 测试与真实结果（合成数据）

命令与真实退出码（本机 Python 3.11.9 / pytest 9.0.2 / numpy 2.4.6）：

    python -m pytest tests/perception/shuttle_detection/test_detection_output.py \
                     tests/badminton_brain/test_shuttle_state_ekf.py \
                     tests/badminton_brain/test_horizon_prediction.py -q
    => 50 passed in 7.06s   exit code 0

    python -m pytest tests/badminton_brain/test_shuttle_ukf.py -q      # 回归：已有 3D UKF
    => 13 passed in 6.47s   exit code 0

### 5.1 P7-04（23 个用例，全部合成 payload）

JSON 往返（含真实 json.dumps/loads）；单位块/键名/像素下限三道闸；NaN/Inf 逐字段拒绝；confidence 范围；bbox 顺序/形状/负坐标；frame_id/camera_id 类型；image_size 必须像素；in_image 派生与"flag 不能撒谎"；协方差对称/PSD（奇异 PSD 允许）/形状；reject_reason 词表封闭；usable 需要全部标志。

### 5.2 P7-05（17 个用例，全部合成真值）

| 检查 | 预注册窗口 | 实测 |
|---|---|---|
| 均值 NEES（模型匹配，dof 4，N=3165） | [3.4, 4.6] | **4.0632** |
| 均值 NIS（dof 2） | [1.5, 2.3] | **1.9751** |
| 门控丢弃内点比例（99% 门） | <= 3% | **35/3200 = 1.09%**（符合设计值） |
| 确定性 CV 真值下均值 NEES（保守性） | < 4.6 | **2.0901**（保守，不冒进） |
| 离群注入（≈20 sigma） | 必须 GATE 且状态/协方差逐位不变 | 通过；连续 3 次 → REINIT |
| 丢失测量 6 步 | 协方差 trace 单调增长、CV 精确推进、COASTING→LOST | trace 5736.8 → 25135.5，状态与解析 CV 一致 |
| 3D 升级（已标定相机 + 立体深度 + 深度率） | 位置/速度误差门槛 | **位置 0.0004 m，速度 0.0582 m/s** |
| 无标定 / 单目深度 / sigma_d=0 / 缺 sigma_d_dot | 必须拒绝且模式仍为 2D | 全部通过 |
| 3D 下不给 R | 必须抛错（不发明米制默认值） | 通过 |

### 5.3 P7-07（10 个用例，全部合成真值）

合成抛物线（sigma=2 mm，50 Hz，0.48 s / 25 帧窗口），200 次蒙特卡洛，真实误差 vs 报告不确定度：

| horizon | RMSE | 报告 1.96 sigma 界 | 95% CI 每轴覆盖率 |
|---|---|---|---|
| 0.3 s | 10.61 mm | 11.84 mm | 0.955 |
| 0.4 s | 14.45 mm | 16.81 mm | 0.967 |
| 0.5 s | 20.91 mm | 22.63 mm | 0.937 |
| 0.6 s | 25.97 mm | 29.30 mm | 0.952 |
| 0.7 s | 31.56 mm | 36.81 mm | 0.960 |
| 0.8 s | 39.55 mm | 45.17 mm | 0.953 |

误差与不确定度都随 horizon 单调增长，**报告界略大于实测 RMSE（不低报）**，覆盖率 ≈ 0.95 ⇒ 统计意义上是校准的。预注册上限（RMSE < 20 mm @0.3 s，< 80 mm @0.8 s）在测量前固定。

模型失配对照（真值 = 冻结气动模型 k=1/6.5，拟合仍是二次多项式，50 次 MC）：

| horizon | RMSE | 报告界 | 95% CI 全轴覆盖率 |
|---|---|---|---|
| 0.3 s | 130.6 mm | 11.8 mm | 0.000 |
| 0.5 s | 374.8 mm | 22.6 mm | 0.000 |
| 0.8 s | 1138.6 mm | 45.2 mm | 0.000 |

**这是本轮最重要的负面结果**：报告的区间只含**统计不确定度**，不含模型误差；一旦模型失配（真实阻力项），CI 完全失去覆盖（0%）。测试里把这一点**钉死为已知限制**（断言覆盖率 < 0.2），不允许后人靠放宽区间"修好"。上报上限 1.25*（0.5*k*|v0|^2*T^2）= 1.55 m 由声明参数推导、运行前固定，不是看着数字改的。证据：outputs/p7_interfaces/p7_07_prediction_error_vs_horizon.json

过期/越界拒绝：horizon 取 {0.29, 0, -0.3, 0.81, 1.5, NaN} 全部 HORIZON_OUT_OF_RANGE/NON_FINITE_INPUT；年龄 > 0.10 s → STALE_MEASUREMENT（0.099 s 仍接受）；观测在未来 → FUTURE_OBSERVATION；3 帧 → INSUFFICIENT_SAMPLES；重复时间戳 → DEGENERATE_SAMPLES；NaN 观测无法构造。

---

## 6. 没有证明什么（必须与结论一起读）

1. **没有任何真实检测精度结论。** 三个模块的测试数据全部合成；confidence 只是检测器自报值，没有任何测试把它转成"正确的概率"。
2. **没有真实 3D 升级能力。** 真实相机内参/外参在仓库里是 TEMP 工程基线（stereo_geometry.py 的 *_measured 条目为 REQUIRES_CALIBRATION）；本任务只证明了"没有标定就拒绝升级"这条门禁能生效，以及**合成**针孔几何下的数值正确性。位置 0.0004 m 是自己造的真值，不是标定精度。
3. **MEASUREMENT_STD_PX=2.0、PROCESS_ACCEL_STD=2000/3.0、MAX_MEASUREMENT_AGE_S=0.10、MAX_COAST_STEPS=5** 都是 TEMP 代理；对应的 REQUIRES_MEASUREMENT 条目仍为 None。真实标定/抖动/延迟测量做出来之前，这些数字不得被当作已表征参数。
4. **P7-07 的置信区间只是统计区间**，模型失配下 0% 覆盖（上表）。绝不能当作真机安全界；安全界必须另立（P7-10 安全 FSM + P7-13 延迟预算）。
5. **没有闭环、没有控制、没有击球成功率。** P7-08/09/10 及实体门禁未受影响；本轮未运行任何检测推理、训练或 GPU 任务。
6. **没有做多目标/多球跟踪**（单球单滤波器），也没有做相机间关联；互相关跟踪（P7-08 的观测缺失与延迟传播）尚未接入。

## 7. UNKNOWN / 待取证

| UNKNOWN | 如何取证 |
|---|---|
| 真实 bbox 中心抖动（按 4–6 px / 6–8 px / 8–12 px 分桶） | 静态吊球重复帧 + 标注残差统计，填入 MEASUREMENT_STD_PX_MEASURED |
| 真实立体/三角化协方差 | 棋盘格标定 + 静态目标深度残差；填 MEASUREMENT_STD_M |
| 真实相机内参/外参（含安装俯仰） | 标定会话；目前全是 TEMP 基线 |
| 端到端延迟（采集→检测→预测） | P7-13 延迟预算实测；替换 MAX_MEASUREMENT_AGE_S |
| 真实轨迹预测误差 | 需要真实飞行 3D 序列（资源在远端，本地无），本轮**不**提供 |
| 阻力系数 k 的本批球实测值 | 飞行序列参数辨识；目前用文献 L=6.5 m |

## 8. 复现

    python -m pytest tests/perception/shuttle_detection/test_detection_output.py -q
    python -m pytest tests/badminton_brain/test_shuttle_state_ekf.py -q -s
    python -m pytest tests/badminton_brain/test_horizon_prediction.py -q -s   # 会重写 outputs/p7_interfaces/*.json

证据文件：outputs/p7_interfaces/p7_07_prediction_error_vs_horizon.json（含 synthetic_only=true 与说明）。
