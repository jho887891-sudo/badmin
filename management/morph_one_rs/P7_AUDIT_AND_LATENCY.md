# P7-01 / P7-02 / P7-03 / P7-08 数据集审计、尺寸分桶离线评估、延迟与过期帧传播

- 分支：`p7/dataset-audit-latency`（`git worktree add ... origin/main`，基于 origin/main）
- 任务：P7-01 已提交 manifest 的数据集审计 + 泄漏测试；P7-02/03 尺寸分桶离线评估协议；P7-08 端到端延迟预算 + 过期帧拒绝
- 环境：本地 Windows，Python 3.11.9，pytest 9.0.2，numpy；**CPU only，无 GPU，未跑 YOLO 推理，未训练，未触碰 scripts/simulation 与 PHASE_P2/P3、队列状态文件**
- 前置阅读（未改动、未重做）：PR #27 / 分支 `p7/detector-ekf-interfaces` @ 8150b2a 的
  `src/perception/shuttle_detection/detection_output.py`、`src/badminton_brain/estimation/shuttle_state_ekf.py`、
  `src/badminton_brain/prediction/horizon_prediction.py` 及其报告 `management/morph_one_rs/P7_DETECTION_EKF_INTERFACES.md`。
  **注意：PR #27 尚未合入 main，因此本分支没有这三个文件**；评估模块用"可导入则用之、否则同契约兼容校验"的方式衔接（见 3.3），不复制、不改写它们。

---

## 0. 一句话结论

数据集层面**没有发现跨 split 的帧级重复**（同路径 / 同名 / 同视频片段+帧号：0），但发现 **1 个仍在的 session 级泄漏**（`old/iphone_20251015` 同一段 iPhone 录制同时出现在 train 与 val），已在审计中**显式标记为 OPEN_LEAK_UNRESOLVED 而不是被静音**；评估与延迟两块**全部是 OFFLINE/SYNTHETIC**，不构成任何真实检测精度或真实延迟结论。

---

## 1. 交付物清单

| 类型 | 路径 |
|---|---|
| 审计模块（P7-01） | `src/perception/shuttle_detection/committed_dataset_audit.py` |
| 审计 CLI | `scripts/shuttle_detection/audit_committed_datasets.py` |
| 审计证据 | `outputs/p7_audit/dataset_audit.json`、`outputs/p7_audit/dataset_audit_maincheckout.json` |
| 评估模块（P7-02/03） | `src/perception/shuttle_detection/size_bucket_eval.py` |
| 评估 CLI + 夹具生成 | `scripts/shuttle_detection/evaluate_size_buckets.py` |
| 合成夹具（已提交） | `outputs/p7_eval/fixtures/{ground_truth,predictions}.jsonl` + `fixtures.json` |
| 评估证据 | `outputs/p7_eval/size_bucket_eval.json` |
| 延迟/过期（P7-08） | `src/badminton_brain/timing/latency_budget.py`、`src/badminton_brain/timing/__init__.py` |
| 延迟证据生成 | `scripts/badminton_brain/latency_probe.py` |
| 延迟证据 | `outputs/p7_latency/latency_budget_report.json` |
| 测试（68 个） | `tests/perception/shuttle_detection/test_committed_dataset_audit.py`(22)、`test_size_bucket_eval.py`(27)、`tests/badminton_brain/test_latency_budget.py`(19) |

---

## 2. P7-01：已提交 manifest 的数据集审计

### 2.1 为什么只审 manifest

已有的 `dataset_audit.py` 会打开每张图与每个 label（内容级审计），但本机资源并不完整：ETH 图像在 `D:\_eth_data`，Isaac 池已丢失（435 行悬空，见 `management/shuttle_detection/DATASET_RESOLUTION.md` 第 4 节）。一个"遇到缺失文件就崩"的审计对 30321 行有 manifest 的数据**什么都报告不了**。因此本模块只读 git 已提交的 manifest 行，把缺文件降级为 **DANGLING 计数**（附最多 10 行样例），从不抛异常。

范围（写进 JSON，防止后人误读）：

    scope                  = COMMITTED_MANIFEST_ONLY
    file_existence_checked = true   （每行一次 stat；缺失是数据，不是错误）
    image_content_checked  = false
    native_px_sizes        = false
    checks_not_performed   = [图像内容重复 / label 内容错误 / 原生像素尺寸 / test split]

### 2.2 命令与真实退出码

    python scripts/shuttle_detection/audit_committed_datasets.py
    => exit code 0

    python scripts/shuttle_detection/audit_committed_datasets.py --repo-root "E:\具身智能\badmin_project" \
        --out outputs/p7_audit/dataset_audit_maincheckout.json
    => exit code 0   （以主检出为资源根复核 dangling；见 2.5）

    python -m pytest tests/perception/shuttle_detection/test_committed_dataset_audit.py -q
    => 22 passed in 32.86s, exit code 0

### 2.3 三个已提交 manifest 的真实数字

| manifest | rows | train | val | test | 负样本 | DANGLING（本 worktree） | DANGLING（主检出为根） |
|---|---|---|---|---|---|---|---|
| `v1_dataset/v1_dataset_manifest.csv` | 30321 | 25908 | 4413 | **0** | 8085 | 0 | 0 |
| `train_data/manifest_train.csv` | 400 | 400 | 0 | 0 | 未记录 | 400 | 0 |
| `train_data/manifest_val.csv` | 120 | 0 | 120 | 0 | 未记录 | 120 | 0 |

- **没有 test split**：已提交 manifest 中不存在 test 划分（`test_split_present=false`）；"冻结测试集"只以 CSV 指标形式存在，不是一份带 split 的 manifest。这是 P7-01 的**已知缺口**，不是本审计的遗漏。
- `v1_dataset_manifest.csv` 的 sha256 = `11d850263b370fc67b78b1264609a5a3d2f79d9f235f9d943bfb322de91ed539`（写进 JSON，供无资源时核对）。
- 该 manifest 行数 30321 与已提交 `v1_dataset_digest.json` 的 `rows` 一致（有测试断言，防漂移）。
- 来源构成：eth_main 20509、coco_bg 6500、eth_iphone 2368、synthetic 817、hard_negative 96、bg_negative 31。

### 2.4 分组泄漏：能查到的和查不到的

分组键由**记录路径**推导（`session` = `images/` 之前的目录；`clip` = session + 文件名去掉 ≥4 位尾随帧号；`declared_group` = manifest 的 `location` 列）。这是启发式，不是数据集的元数据，JSON 的 `caveats` 已声明。

| 检查 | 结果 |
|---|---|
| 同一规范化路径出现在 2 个 split | **0** |
| 同一 basename（大小写无关）出现在 2 个 split | **0** |
| 同一 clip + 同一帧号出现在 2 个 split | **0** |
| clip 级跨 split 分组 | **0**（6674 个 clip group） |
| session 级跨 split 分组 | **1**（42 个 session group） |

唯一的 session 级交叉：

    session:_eth_data/eth_shuttle_detection/old/iphone_20251015
    train: 若干帧, val: 若干帧   （同一段 iPhone 录制）

判定：**OPEN_LEAK_UNRESOLVED**。同一段录制（同场景/同光照）同时喂 train 与 val，属于"记住场景"而不是"泛化目标"；但逐 clip 标识符并不交叉，所以它不是帧级重复。修复方式（重划分已提交 manifest / 整段剔除）属于数据集变更，超出本轮 CPU-only 审计切片，必须由数据集负责人决定。**审计没有把它静音**：

- `KNOWN_SESSION_CROSSINGS` 白名单把它记为 `status=OPEN_LEAK_UNRESOLVED` + 理由；
- 测试 `test_group_leakage_is_only_the_reviewed_and_open_list` **任何新增的、未登记的分组交叉都会 FAIL**；
- 测试 `test_known_crossings_cannot_rot` 要求：登记项若消失，测试同样 FAIL（防止白名单腐化成"永久放行"）。

> 说明：`declared_group`（`location` 列）上还有 `backgrounds`、`iphone_20251015`、`synthetic` 三个池同时含 train/val。前两者与上面同源/同类，`backgrounds` 是背景池（train:bg_train / val:bg_val 两个不同目录），不构成同一素材跨 split；因此**默认只对 session/clip 两级做阻断**，declared_group 交叉仍在 JSON 中完整列出，供人判断。

### 2.5 DANGLING：真实数字与口径

- 以**本 worktree 为根**：`manifest_train.csv` 400/400、`manifest_val.csv` 120/120 = **520 行悬空**；`v1_dataset_manifest.csv` 0 行悬空（其 image 列是绝对路径 `E:\...`，本机存在）。
- 以**主检出 `E:\具身智能\badmin_project` 为根**复核：两份 synthetic manifest 的 **dangling 都变成 0**。
- 结论（必须写清楚）：这 520 行**不是丢失的资源**，而是"新 worktree 里没有 gitignored 的图像池"。审计用 `--repo-root` 就能把这点验证出来。判定口径：**DANGLING 是相对某个资源根的量**，报告必须带根，否则数字没有意义。
- 同一根因也解释了本轮两个"预存在失败"（见第 5 节）。

### 2.6 负样本与小目标尺寸分布

负样本（`n_boxes == 0`）：8085 / 30321 = 26.66%。`n_boxes` 与 `bucket=='negative'` 列的一致性检查：**0 处不一致**。

尺寸分布（按 P7-02 的桶边界，半开区间 [lo, hi)）：

| 桶 | <4px | 4-6px | 6-10px | 10-20px | >20px | 未知（负样本） |
|---|---|---|---|---|---|---|
| v1_dataset（`eq640_min` 基准） | 386 | 1738 | 11275 | 7926 | 911 | 8085 |
| manifest_train（`equiv_size_px`） | 15 | 69 | 169 | 134 | 13 | 0 |
| manifest_val（`equiv_size_px`） | 5 | 18 | 48 | 41 | 8 | 0 |

`v1_dataset` 的尺寸统计：min 1.633、p50 9.154、p95 18.5733、max 538.287（n=22236）。

**口径警告（必须与数字一起读）**：尺寸取自 manifest 的 `eq640_min` 列，而该列**在仓库中没有任何代码或文档定义其含义**（JSON: `basis_definition_in_repo = UNDOCUMENTED`）。因此：

- 它是"manifest 自报的 640 等价尺寸"，**不等于**原生像素尺寸；
- 本审计**不**用 `width/640` 之类的假设把它换算成原生像素（那是猜测，不是测量）；
- 所以"4-6 px 样本有 1738 个"这句话只能读作"manifest 记录的等价尺寸落在 4-6"；真实原生像素下的尺寸分布 = **UNKNOWN**。

---

## 3. P7-02 / P7-03：尺寸分桶离线评估协议

### 3.1 协议

- 输入：GT JSONL + predictions JSONL（各一行一个 JSON 对象），**逐行 schema 校验后才计算任何指标**；
- 尺寸桶：`<4px`（报告但不属于预注册四桶）、`4-6px`、`6-10px`、`10-20px`、`>20px`，半开区间；
- 尺寸定义：`size_px = sqrt(bbox_w * bbox_h)`（GT 可显式给 `size_px`，否则由 bbox 推导）；检测框按**自身尺寸**入桶，GT 按自身尺寸入桶（两者都记录，避免"用 GT 尺寸给预测分桶"的循环）；
- 匹配：贪心一对一。**两套规则都报告，都不称作"真值"**：
  - `iou50`：IoU ≥ 0.5（主规则）；
  - `centre`：中心距 ≤ max(1 px, 0.5·size_px)。理由：4-6 px 小框上 IoU 极其苛刻（5×5 框平移 2 px，IoU 掉到 0.43 = 完全漏检；4×4 框平移 1 px 掉到 0.39）。把两条曲线并列，才能暴露"指标本身比检测器更敏感"的问题。
- 输出：每桶 precision / recall / AP（all-point 插值）、TP/FP/FN、平均匹配 IoU，以及延迟统计（n / mean / p50 / p90 / p95 / p99 / max）。
- 证据等级：`claim = OFFLINE-SYNTHETIC`、`real_detector_accuracy = false`，并在 `notes` 里写明"不是真实检测精度、不是闭环、不是 D455 实测"。

### 3.2 输入校验（拒绝而不是打分）

以下每一项都有单测覆盖（`test_size_bucket_eval.py`，27 个用例，全部通过）：`UNLABELED_FRAME`（`labeled` 必须是 `True`，1/"yes"/None 都拒绝）、`UNKNOWN_FRAME`（预测的 frame_key 不在已标注 GT 中）、`DUPLICATE_FRAME`、`MISMATCHED_FRAME`（camera_id / image_size 与 GT 不一致）、`BAD_BBOX`（形状/x2>x1/负坐标）、`BAD_NUMBER`（NaN/Inf、confidence 越界、latency<0）、`UNIT_CONFUSION`（像素 payload 里出现 `position_m` 等米制键，或 2×2 协方差对角低于 0.01 px² 的像素下限）、`UNKNOWN_KEY`（strict）、`BAD_SCHEMA`（schema/schema_version 不符、非 JSON 行）。**所有问题一次性列出**，不是只报第一条。

### 3.3 与 PR #27（P7-04 schema）的衔接

`detection_output.py` 尚在未合并分支上。评估模块这样做：

    try:  from src.perception.shuttle_detection.detection_output import validate_detection_payload
          => detection_schema_source = "P7_MODULE"
    except ImportError:
          => detection_schema_source = "FALLBACK_COMPAT"  （实现同三条像素闸门）

本次实测 `detection_schema_source = FALLBACK_COMPAT`（PR #27 未合入 main），证据 JSON 里如实记录。**PR #27 合入后不需要改评测逻辑**，只是校验器换成权威实现；三条闸门的行为两侧一致（有测试）。这不是第二套 schema，只是过渡兼容层。

### 3.4 合成夹具（已提交，可复现）

    python scripts/shuttle_detection/evaluate_size_buckets.py --make-fixtures
    => exit code 0   （seed=20261021，120 帧，50 Hz，1280×720，camera=d455_left）

夹具是"一个 recall 随目标尺寸下降的**模拟检测器**"：声明 recall `4-6px=0.35, 6-10px=0.60, 10-20px=0.80, >20px=0.95`（`<4px=0.20`），负样本帧有 12% 概率产生假阳性，框抖动 σ=0.35 px，延迟对数正态（中位数 22 ms）。**这些数字是夹具参数，不是任何检测器的行为。**

    python scripts/shuttle_detection/evaluate_size_buckets.py
    => exit code 0

### 3.5 结果（OFFLINE-SYNTHETIC，不是检测精度）

计数：GT 120 帧 / 153 个目标 / 13 个负样本帧；预测 98 条 / 85 帧；被拒绝帧 0。

| 桶 | GT | 预测 | TP | FP | FN | Precision | Recall | AP | 平均匹配 IoU | 夹具声明 recall |
|---|---|---|---|---|---|---|---|---|---|---|
| <4px | 24 | 3 | 2 | 1 | 22 | 0.6667 | 0.0833 | 0.0833 | 0.7495 | 0.20 |
| 4-6px | 19 | 5 | 4 | 1 | 15 | 0.8000 | **0.2105** | 0.2000 | 0.8526 | 0.35 |
| 6-10px | 34 | 22 | 21 | 1 | 13 | 0.9545 | **0.6176** | 0.6110 | 0.8874 | 0.60 |
| 10-20px | 43 | 37 | 35 | 2 | 8 | 0.9459 | **0.8140** | 0.8081 | 0.9251 | 0.80 |
| >20px | 33 | 31 | 30 | 1 | 3 | 0.9677 | **0.9091** | 0.9091 | 0.9572 | 0.95 |
| 全部 | 153 | 98 | 92 | 6 | 61 | 0.9388 | 0.6013 | 0.5932 | 0.9200 | — |

- **harness 自检**：四桶 recall 与夹具声明值之差最大 0.1395（4-6px），全部 < 0.20 的测试容差；recall 随尺寸单调上升（有测试断言）。这证明"协议能把尺寸相关的召回差异读出来"，**不证明任何检测器**。
- `centre` 规则本次与 `iou50` 完全一致（抖动只有 0.35 px），说明该规则在预测已经很准时不会虚增；真正拉开差距的是小框大偏移的场景（单测覆盖）。
- 延迟（98 条预测）：mean 24.78 ms、p50 22.94、p90 37.04、p95 43.27、p99 58.76、max 65.26 ms。**这是夹具文件里的批处理延迟，不是相机到指令的延迟**（JSON 内有同句 note）。

---

## 4. P7-08：端到端延迟预算与过期帧拒绝

### 4.1 预注册（在跑任何数据之前固定，写进模块常量与本文件）

    MAX_SOURCE_FRAME_AGE_S = 0.100 s      # 源帧年龄上限；age <= 0.100 接纳，> 0.100 拒绝
    STAGE_BUDGET_S = capture 0.010 | detect 0.030 | filter 0.005 | predict 0.010 | command 0.005
    RESERVE_S      = 0.040                # 不可被单阶段占用
    END_TO_END_BUDGET_S = 0.100           # 0.060 + 0.040

取值理由：0.10 s 与 P7-07 的 `MAX_MEASUREMENT_AGE_S` 同源（0.30 s 最小 horizon − 约 0.10 s 感知/推理 − 约 0.10 s 决策规划）；**全部是 TEMP 代理**，`MEASURED_*` 仍为 UNKNOWN（P7-13 未做）。

### 4.2 语义与诚实边界

- 五个阶段时间戳 = `capture → detect → filter → predict → command`，必须非递减、有限，否则 `LatencyError`；
- `duration(stage) = t_stage − t_prev`（时间戳标记该阶段的**完成**时刻）；
- `capture` 是时钟原点，**它自身的耗时无法从这五个时间戳恢复**，因此 `latency_report` 把它标为 `NOT_MEASURABLE_FROM_STAGE_TIMESTAMPS`，**不**报告成 0（"报告成 0"等于伪造一个达标）；
- 预算表必须自洽：阶段预算之和 + 预留 = 端到端预算，否则 `LatencyError`（防止有人偷偷改小某一阶段）。

### 4.3 命令与真实退出码

    python scripts/badminton_brain/latency_probe.py
    => exit code 0

    python -m pytest tests/badminton_brain/test_latency_budget.py -q
    => 19 passed, exit code 0

### 4.4 结果（合成链，90 帧，50 Hz）

| 阶段 | 预算 | 均值 | 最大 | 超预算帧数 | 状态 |
|---|---|---|---|---|---|
| capture | 10 ms | 不可测 | 不可测 | — | NOT_MEASURABLE |
| detect | 30 ms | — | **80.0 ms** | **15 / 90** | OVER_BUDGET |
| filter | 5 ms | — | **5.899 ms** | **14 / 90** | OVER_BUDGET |
| predict | 10 ms | — | **38.01 ms** | **4 / 90** | OVER_BUDGET |
| command | 5 ms | — | **6.066 ms** | **7 / 90** | OVER_BUDGET |
| 端到端 | 100 ms | **41.78 ms** | **87.0 ms**（frame 30） | 0 / 90 | WITHIN_BUDGET |

刻意注入的尖峰（frame 30 detect 80 ms、frame 61 predict +30 ms）都被抓到。**本轮真实的负面结果**：即使把抖动设成"看起来很温柔"的量级（filter 4 ms ±0.8 ms、command 4 ms ±0.8 ms），预注册的 5 ms 预算也会在 14/90 与 7/90 帧被击穿。这说明**预注册的单阶段预算对 filter/command 偏紧**——我们**不**在看过结果之后把预算调宽（那就是事后调参），而是把这件事作为结论上报：这两个预算需要 P7-13 实测后重新预注册。

### 4.5 过期帧拒绝（不是静默复用）

`StalenessGuard.admit(prediction_id, frame_id, source_frame_timestamp_s, now_s)`，拒绝时抛 `StaleFrameRejected`（携带 `Admission` 记录，异常本身就是审计痕迹，调用方无法悄悄吞掉）：

| 场景 | 结果 | 原因 |
|---|---|---|
| 源帧年龄 = 0.100 s（正好在阈值上） | **ADMITTED** | 边界包含（预注册） |
| 源帧年龄 = 0.1001 s | **REFUSED** | `SOURCE_FRAME_AGE_EXCEEDED` |
| 同一 prediction 在 0.050 s 时二次消费 | **REFUSED** | `SOURCE_FRAME_REUSE_AFTER_STALE` |
| 源帧时间戳在 now 之后 | **REFUSED** | `SOURCE_FRAME_IN_FUTURE`（时钟错误） |

关键点：**同一个预测被消费两次永远不会静默通过**——即使它当时还没过期，也会以 `SOURCE_FRAME_REUSE_AFTER_STALE` 拒绝。策略语义是"要么用刚采的帧重算，要么明确失败"。

---

## 5. 测试与真实退出码

本分支新增 68 个测试：审计 22、评估 27、延迟 19。

    python -m pytest tests/perception/shuttle_detection/test_committed_dataset_audit.py \
                     tests/perception/shuttle_detection/test_size_bucket_eval.py \
                     tests/badminton_brain/test_latency_budget.py -q
    => 68 passed in 31.90s   exit code 0

任务要求的回归：

    python -m pytest tests/perception/shuttle_detection tests/badminton_brain -q
    => 2 failed, 877 passed, 10 skipped, 99 warnings in 88.43s   exit code 1

两个失败**均为预存在**，与本轮改动无关（本轮只新增文件，未改任何既有模块）：

| 失败用例 | 真实原因 |
|---|---|
| `test_challenge_test.py::CommittedChallengeManifestTests::test_no_image_content_is_shared_with_the_frozen_core` | `FileNotFoundError: outputs/shuttle_capability/controlled_capability/images/s1_size_lt4_0.jpg`（图像池是 gitignored 资源，只存在于主检出，新 worktree 没有） |
| `test_dataset_leakage.py::AuditDatasetTests::test_frozen_p4a_train_and_val_pass_the_audit` | `report.result.error_count = 1040 != 0`，同因：1040 张图在新 worktree 中不存在 |

佐证：`outputs/shuttle_capability/controlled_capability/images` 在主检出 `E:\具身智能\badmin_project` 存在（`s1_size_lt4_0.jpg` 亦存在），在 `badmin_wt_p7b` 不存在。这正是第 2.5 节同一条"资源不在 git 里"的口径。**未修改这两个用例，也未改任何不相关代码。**

---

## 6. 哪些是 OFFLINE/SYNTHETIC，哪些是 UNKNOWN（必须与结论一起读）

**真实（来自已提交数据，可复核）**

1. manifest 的行数/划分/来源/负样本/尺寸分布 —— 全部由已提交 CSV 统计，sha256 记录在证据 JSON 中。
2. 帧级跨 split 重复 = 0；clip 级跨 split = 0；session 级跨 split = 1（iphone_20251015，OPEN）。
3. "已提交 manifest 中不存在 test split" 这一事实。
4. 本轮 68 个新测试与 2 个预存在失败的通过/失败状态与退出码。

**OFFLINE / SYNTHETIC（不是真实精度、不是真实延迟）**

1. 第 3.5 节的 precision/recall/AP 全部来自**合成的模拟检测器夹具**，只衡量 harness；
2. 第 3.5 节的延迟统计是**夹具文件里的批处理延迟**；
3. 第 4.4 节的全部延迟数字来自**脚本生成的合成时间戳**，没有相机、没有推理、没有机器人；
4. 第 4.5 节只证明拒绝策略生效，不证明真机上"球没被漏"。

**UNKNOWN（无数据，不得填充）**

| UNKNOWN | 现状 | 取证方式 |
|---|---|---|
| 真实检测器在 4-6 / 6-10 / 10-20 / >20 px 上的 P/R/AP | 无真实预测 JSONL（权重与推理都未在本轮出现） | 用真实权重在冻结集上产出 predictions JSONL，直接喂本 harness |
| manifest `eq640_min` 列的定义 | 仓库无任何代码/文档定义 | 找 `v1_dataset` 生成脚本的原作者确认，或重算 |
| 真实原生像素尺寸分布 | 需要 label 文件内容（本审计不读） | 内容级审计 + 原生分辨率统计 |
| 真实 D455 采集→时间戳延迟 | 常量是 TEMP 代理 | P7-13 打点实测 |
| 真实检测/EKF/预测/执行耗时 | 同上 | P7-13；替换 `STAGE_BUDGET_S` 前必须重新预注册 |
| 端到端闭环成功率 | 未做闭环、未控制真机 | sim-to-real 闭环实验 |
| 是否存在**图像内容级**跨 split 重复 | 本审计不读图像字节 | `dataset_audit.py` 内容级审计（需要图像池） |

---

## 7. 复现

    # P7-01 审计（两种资源根口径）
    python scripts/shuttle_detection/audit_committed_datasets.py
    python scripts/shuttle_detection/audit_committed_datasets.py --repo-root "E:\具身智能\badmin_project" \
        --out outputs/p7_audit/dataset_audit_maincheckout.json
    python -m pytest tests/perception/shuttle_detection/test_committed_dataset_audit.py -q

    # P7-02/03 夹具 + 评估
    python scripts/shuttle_detection/evaluate_size_buckets.py --make-fixtures
    python scripts/shuttle_detection/evaluate_size_buckets.py
    python -m pytest tests/perception/shuttle_detection/test_size_bucket_eval.py -q

    # P7-08 延迟/过期
    python scripts/badminton_brain/latency_probe.py
    python -m pytest tests/badminton_brain/test_latency_budget.py -q

证据文件 sha256（本轮实际计算）：

    outputs/p7_audit/dataset_audit.json                        fabe9b9a3f241c338d96fdcea88972fa6681a809b9899e0d564c0bc8a7ea40a7
    outputs/p7_eval/size_bucket_eval.json                      fdccf04797942480e10e3b2987478d6e22eabd3ca885028e23a94ee094963124
    outputs/p7_eval/fixtures/ground_truth.jsonl                15beb988883e876ad377977470bca62f3088ee58882bffa58c317acbff19c70a
    outputs/p7_eval/fixtures/predictions.jsonl                 bc3bcd0af4fc3f140a0dc674895c2c144030554bf756d841622180183444fdb6
    outputs/p7_latency/latency_budget_report.json              5d14a672f5f468a6bcbcc56529a54052fab6c183024c3ea4fe10ce98276dc8a1
