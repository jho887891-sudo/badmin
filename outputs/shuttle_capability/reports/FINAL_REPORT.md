# FINAL_REPORT.md — 羽毛球检测能力评估（v1：能力测试阶段）

日期：2026-09-13　范围：**仅目标检测**（不含定位/三角化/UKF/控制/规划）
产物：`outputs/shuttle_capability/`　脚本：`experiments/shuttle_capability/`

## A. 目标收缩（协议 §1）

| 项 | 结果（实测 target_check.json） |
|---|---|
| 当前权重 | yolo26s.pt（COCO） |
| 当前 nc | **80** |
| 是否含 shuttlecock 类 | **否** |
| 是否含 badminton 类 | **否** |
| 最接近的类 | sports ball (32) / tennis racket (38) |
| 判定 | **TARGET_CLASS_NOT_PRESENT** |
| TARGET_2 | **UNKNOWN / NOT_SPECIFIED**（项目未定义，未猜测） |

→ 任务定义应收缩为 **nc=1，class 0 = shuttlecock**；COCO 80 类不作为最终任务定义。

## B. 能力测试集（协议 §3）

| 类型 | 状态 | 数量 |
|---|---|---|
| A. 三维模型/素材生成 | ✅ 已建立（受控光栅化，GT 精确） | **84 张正样本 + 40 张负样本** |
| B. 真实羽毛球图片 | ❌ **BLOCKED — 数据不存在** | 0（本地检索：无任何羽毛球照片） |
| C. 真实羽毛球视频帧 | ❌ **BLOCKED — 数据不存在** | 0 |

**Capability Test Set 已冻结**：84 张合成样本 + 40 张负样本，**不会回流入训练**（协议 §3/§35）。

合成集构造（`01_gen_synthetic.py`）：使用归档的真实羽毛球 GLB 网格（**6498 顶点 / 3422 三角面**，
球体尺寸 78 mm），numpy z-buffer 软件光栅化 → 轮廓掩膜 → **精确 GT bbox**；
覆盖：距离 0.8–26 m、yaw/pitch/roll 随机三维姿态、9 个图像位置、4 种背景、
以及 **SYNTHETIC_BLUR_TEST** 子集（sigma 1.0–3.0 随机方向运动模糊）。

尺寸定义（协议 §5）：`equivalent_size_px = sqrt(bbox_w_px * bbox_h_px)`，同时保留 w/h。
实测分布：**min 1.41 px / median 10.99 px / max 67.19 px**（覆盖 <4px 到 >32px）。

## C. 当前模型能力实测（协议 §2/§6/§28）

配置：`yolo26s.pt`，imgsz 640/960/1280，conf 扫描 0.05–0.90，IoU≥0.5 判定命中。

### C1. Recall_any（任意类别框与 GT 的 IoU≥0.5）

见 `reports/CAPABILITY_BEFORE.md` 完整表。要点（imgsz=1280）：

| 尺寸桶 | c≥0.05 | c≥0.25 | c≥0.5 | c≥0.9 |
|---|---|---|---|---|
| <4px (21) | 0.14 | 0.00 | 0.00 | 0.00 |
| 4–6px (7) | 0.71 | 0.00 | 0.00 | 0.00 |
| 6–8px (8) | 0.50 | 0.00 | 0.00 | 0.00 |
| 8–12px (7) | 0.43 | 0.14 | 0.00 | 0.00 |
| 12–16px (7) | 0.71 | 0.14 | 0.14 | 0.00 |
| 16–24px (10) | 0.90 | 0.30 | 0.00 | 0.00 |
| 24–32px (6) | 0.67 | 0.33 | 0.17 | 0.00 |
| >32px (18) | 0.94 | 0.44 | 0.17 | 0.00 |

### C2. **关键：这些「命中」不是羽毛球检测**

对 conf≥0.05 下所有 IoU≥0.5 的命中做类别归属分析（`03_analyze_before.py`）：

```
kite        14 hits  mean_conf 0.126
bird         9 hits  mean_conf 0.233
clock        9 hits  mean_conf 0.390
umbrella     5 hits  mean_conf 0.139
bottle       2      person 2      potted plant 1   snowboard 1   vase 1
stop sign    1      toilet 1      fork 1 (conf 0.797)   car 1   boat 1
sports ball  1 hit  mean_conf 0.081
```

即：**没有任何一个类别是羽毛球**；它们是模型在白色轮廓上产生的**低置信度幻觉**，
且随置信度阈值升高全部消失（c≥0.25 时 ≤24px 全为 0）。

### C3. Recall_sports_ball（COCO class 32）

几乎恒为 0：c≥0.05 时仅 6–8px 0.12、12–16px 0.14、24–32px 0.17，其余全 0；
c≥0.25 后**全部为 0**。→ 用 COCO 的 sports ball 代替羽毛球**不可行**。

### C4. 分辨率影响（协议 §20）

overall recall_any：640 → 0.5595，960 → 0.5714，1280 → 0.5952（conf 0.05）。
在可用置信度（≥0.25）下三者都接近 0 —— **提高输入分辨率并不能让无类别模型学会检测羽毛球**。

### C5. 误检（协议 §23）

40 张无球帧（纯背景）：conf≥0.05 下 **0 个框（0.00 FP/frame）**。
**局限（必须声明）**：本负样本集背景过于简单（纯色/渐变/噪声），
**不能**代表真实困难负样本（球网、白线、灯、白鞋、反光）；真实负样本需真实图像才能评估。

## D. 当前模型是否够用（协议 §14）

**判定：`CURRENT_MODEL_NEEDS_TRAINING`**

理由（全部有实测支撑）：
1. 没有 shuttlecock 类别（nc=80，TARGET_CLASS_NOT_PRESENT）；
2. 在 conf≥0.25 的可用阈值下，**所有尺寸桶 ≤24px 的检测率≈0**，24–32px 仅 0.33，>32px 0.44；
3. 低阈值下的命中经类别分析证实为**无关类别的幻觉**，非羽毛球检测；
4. 提高输入分辨率不改变这一结论。

**没有任何 mAP/Precision 数字被报告**：因为不存在 shuttlecock 类，
对该类的 mAP 在定义上不可计算 —— 按协议不得编造（`ACCEPTANCE_THRESHOLD_NOT_SPECIFIED`，
项目亦未定义「稳定检测」的任务阈值）。

## E. BLOCKED 项（协议 §3 要求三类数据，缺两类）

| 实验 | 状态 | 具体原因 | 解锁条件 |
|---|---|---|---|
| B. 真实图片能力测试 | **BLOCKED** | 项目与本地机器均无任何真实羽毛球图片 | 提供真实图片（含来源元数据） |
| C. 真实视频能力测试（连续性/漏检/重捕获/运动模糊真值） | **BLOCKED** | 无真实羽毛球视频 | 提供视频（逐帧或按频率 GT） |
| 真实困难负样本（网/线/灯/反光） | **BLOCKED** | 同上 | 真实无球帧 |
| 姿态/距离/位置与真实数据的一致性对比（§34 Q11） | **BLOCKED** | 缺 B/C | 同上 |
| 训练前 baseline 的 mAP/F1 | **不适用** | 无 shuttlecock 类 | 训练后与训练前用同一独立测试集对比 |

## F. 失败原因分析 → 后续数据构建方向（协议 §15/§16/§17）

已实测的失败模式（用于驱动数据构建，而不是无脑加 epoch）：

| 失败类别 | 证据 | 需要补的数据 |
|---|---|---|
| **目标类别缺失** | nc=80，无 shuttlecock | 全部训练数据必须标注为 class 0 = shuttlecock |
| **极小目标（<8px）** | conf≥0.25 时 0.00 | 用 3D 模型**精确生成** 4–8px 与 8–12px 样本（球远/小） |
| **低置信度幻觉** | kite/bird/clock 等 14 类 | 加入困难负样本（白点/高光/白线/网孔）以压制 |
| **姿态相关性未知** | 合成集姿态已覆盖但模型无类别 | 训练后按 yaw/pitch/roll 分桶复核 |
| **运动模糊** | SYNTHETIC_BLUR 子集已生成 | 训练时加入真实高速模糊（真实视频） |
| **背景域差异** | 合成背景仅 4 种、过于简单 | 用真实球场渲染/真实照片做背景合成 |

## G. 下一步（按协议顺序，等你确认）

1. **[需要你] 提供真实数据**：真实羽毛球图片（多品牌/角度/距离/背景/光照/清晰与模糊）
   与真实羽毛球视频（近→远、远→近、横向、斜向、遮挡、出入画面）。这是当前唯一硬阻塞。
2. 我将按失败模式构建训练集：3D 模型生成的**受控小目标与长尾姿态**（自动 GT） + 真实图片/视频帧；
   两类分别统计（source_type = SYNTHETIC_3D / REAL_IMAGE / REAL_VIDEO），禁止合成淹没真实。
3. 数据清洗独立成阶段（hash/感知哈希去重、坏图、越界框、漏标/错标、train-val-test 按 session 隔离）。
4. 训练 baseline 检测结构（**不先上 P2**），完成后用**同一冻结 Capability Test Set**重测 before/after。
5. 只有当「极小羽毛球」仍是主要瓶颈时，才进入 P2 检测分支对比实验（协议 §32）。

## H. 产物清单（`outputs/shuttle_capability/`）

```
reports/target_check.json            # §1 目标收缩实测
reports/CAPABILITY_BEFORE.md         # §6 能力表（conf × 尺寸桶）
metrics/synthetic_manifest.csv       # 84 张合成样本元数据（距离/姿态/位置/尺寸/模糊/背景）
metrics/size_bucket_metrics_before.csv
metrics/capability_before.csv        # 逐样本 × conf 的命中矩阵
metrics/class_hits_before.txt        # 命中类别归属（幻觉证据）
predictions_before/predictions_before.csv    # 逐图 best-IoU 汇总
predictions_before/detections_before.csv     # 逐检测框（类别/conf/IoU/命中）
synthetic_3d/images/                 # 84 张能力测试图（含 SYNTHETIC_BLUR 子集）
synthetic_3d/negatives/              # 40 张无球帧
annotations/synthetic_3d/            # YOLO 格式精确 GT
```

## I. 真实图片能力测试（新增，协议 §12）— 来源：Wikimedia Commons（自由许可）

数据获取（合法公开来源，已记录元数据 `real_images_metadata.csv`）：
- 下载 **59 张**真实图片；许可分布：CC BY-SA 4.0 (20)、Public domain (15)、CC BY 2.0 (6)、
  CC BY-SA 3.0 (6)、CC0 (6)、CC BY-SA 2.5 (3)、CC BY 4.0 (1)、GODL-India (1)、No restrictions (1)
- 记录字段：file / title / license / artist / 原始宽高 / 字节数 / Commons 页面 URL / 检索词

GT（标注来源必须声明）：`gt_source = auto_threshold_union`（白-低饱和连通域并集 + 4 px 外扩），
**`verified_by_agent = False`**（尚未逐张目视确认）→ 本组数字标记为 **PROVISIONAL_GT**。

### 实测（`yolo26s.pt`, imgsz=1280, conf≥0.05, IoU≥0.5）

| 尺寸桶 | n | recall_any | recall_sports_ball | max_conf |
|---|---|---|---|---|
| >32px | 49 | 0.408 | **0.000** | 0.948 |

- **全部 49 张都落在 >32px 桶**（59.3 – 1858.3 px）：公开图库被**影棚产品特写**主导，
  没有小目标真实样本 —— 这正是必须依赖**真实视频帧**才可能覆盖小尺寸的原因。
- `recall_sports_ball = 0.000`：即使球大到 59–1858 px，模型也**从未**以 sports ball 类且 IoU≥0.5 命中。
- `recall_any = 0.408` 与合成集同源问题：是**无关类别的框**碰巧与 GT 重叠（max_conf 0.948 说明模型确实自信地检测到了*某个东西*，但不是羽毛球类）。

### 与合成结果的一致性（协议 §34 Q11 的部分回答）

合成（>32px, c≥0.05）：recall_any 0.94；真实（>32px, c≥0.05）：recall_any 0.41。
真实场景更难（背景复杂、姿态自然、无合成轮廓的纯净度），但**两者的 sports ball recall 都≈0** —— 
结论一致：**COCO 模型不具备羽毛球检测能力**。

## J. 真实视频能力测试（协议 §13）— **BLOCKED（临时）**

- 已检索到 **19 段自由许可的真实羽毛球视频**（Commons：CC0 / CC BY 3.0 / CC BY-SA 4.0，含比赛与训练场景，
  列表见 `real_video_candidates.csv`，其中最小的 1.1 MB、最大 752 MB）。
- **下载受阻**：Wikimedia 对连续请求返回 **HTTP 429（rate limited）**，多次退避重试仍失败。
  → 标记 `REAL_VIDEO_DOWNLOAD_RATE_LIMITED`，将在更长退避（≥30 min）或换用其他自由来源后继续。
- 视频侧（连续性/漏检长度/重捕获/运动模糊真值）**尚未产生任何数字**，不编造。

## K. 当前判定（更新）

**`CURRENT_MODEL_NEEDS_TRAINING`** —— 依据（全部实测）：
1. 无 shuttlecock 类（nc=80）；
2. 合成集：可用置信度（c≥0.25）下 ≤24px 全为 0，24–32px 0.33，>32px 0.44；
3. **真实集：即使 59–1858 px，sports ball recall = 0.000**；
4. 低阈值命中经类别归属分析证实为无关类别幻觉（kite/bird/clock/…）；
5. 提高输入分辨率（640→1280）不改变结论。

### 下一步（无阻塞部分可立即继续）

1. **真实图片 GT 目视确认**（把 PROVISIONAL_GT 升级为 verified）——我可以逐张过复核图；
2. **构造真实小目标样本**：用已下载的真实**球场/比赛照片**做背景，把 3D 羽毛球按受控像素尺寸合成进去
   （记录 source_type=SYNTHETIC_ON_REAL_BG，与纯合成、真实分开统计）；
3. **视频**：等限流解除后下载 → 抽帧 → 逐帧目视定位球 → 标注 → 跑当前模型（real-video 能力 + 连续性）；
4. 上述完成后进入**失败驱动训练集构建 → 清洗 → 训练 baseline（不上 P2）→ 同一冻结测试集重测**。
