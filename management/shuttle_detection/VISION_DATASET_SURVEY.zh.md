# 羽毛球机器人视觉数据集调研（机载/自运动相机 · 单帧检测 · 640×640 输入）

调研日期：2026-03（检索环境：联网，primary source 优先）
调研人：算法研发助手（执行）
本地文件：management/shuttle_detection/VISION_DATASET_SURVEY.zh.md

> 本文所有关键结论都标注了出处。凡当前无法从一手来源确认的字段，一律写 UNKNOWN，不做猜测。

## 0. 一句话结论

做「羽毛球机器人机载视觉」这一类工作的人，训练时真正用的数据分成三种，比例极不均衡：

1. **绝大多数是自采（self-collected），且不公开**：YOLOBR(2021)、FTOC(2019)、YO-CSA-T(2025)、ETH/RSL 的原始拍摄、ETH 四足机械臂 ANYmal-D(2025) 全是自采。论文里通常只写 we collect / we label，没有可下载的标准数据集。
2. **公开数据几乎全是「固定机位转播视角」**：TrackNet / Shuttlecock Trajectory Dataset、RacketVision、ShuttleSet、Roboflow 上那些 shuttlecock 项目，都是电视转播或固定机位。
3. **截止可查证据（2026-03），公开的、真正面向「机器人自运动/第一视角」的羽毛球检测数据集只有 1 个**：ETH RSL 的 One-Shot Badminton Shuttle Detection dataset（20,510 帧 / 11 个地点）。该论文自己写道：唯一的公开数据集（指 TrackNet 系）在视角和分辨率上都不适合机载相机。

对你的直接含义：**真实机载检测数据没有现成的可白嫖大集合，你必须走「少量公开 egocentric 数据 + 自己按目标尺度自采 + 半自动标注」这条路**。

## 1. 检索方法与证据等级

- 检索面：arXiv（abs / html / ar5iv 全文）、GitHub API（README、文件树、config）、Roboflow 官方博客、Kaggle 公开 API、HuggingFace、EuropePMC/PMC、Semantic Scholar API、ScienceDirect 摘要页、Springer/ACM 摘要页、Google Drive 目录。
- 证据等级：1 原论文 > 2 官方项目主页 > 3 官方 GitHub > 4 官方数据集页面 > 5 二手资料。本文以 1/3/4 为主。
- 明确排除（虽然搜索命中，但与你的任务无关）：比赛精彩片段/击球事件识别、动作分类、球员姿态、胜负分析、纯转播视频分类。
- 反面例子（不要因为名字里带 shuttle 就收录）：Autoshuttle (Zhou, Salaar, Schmidt, Dehghani, Arbeiter) 是**自动驾驶接驳车**数据集，不是羽毛球；UCI Statlog Shuttle 是航天飞机数据。

## 2. 逐篇核实：真正相关的工作用了什么数据

### 2.1 ETH RSL —— One-Shot Badminton Shuttle Detection for Mobile Robots

- 论文：arXiv:2603.06691，Dipner / Talbot / Tuna / Cramariuc / Hutter，RSL ETH Zurich，2026-03-04 提交，2026-03-17 更新。
- 官方项目页：https://sites.google.com/leggedrobotics.com/shuttlecockfinder （本次环境无法访问，fetch failed）
- 官方 GitHub：https://github.com/leggedrobotics/shuttle_detection （AGPL-3.0，2026-01-22 创建，2026-03-04 最后推送，7 stars）

| 项目 | 内容 |
|---|---|
| 任务 | 单帧检测（one-shot detection，YOLOv8s 微调）；明确面向下游 tracking / trajectory estimation / re-initialization |
| 相机 | UNKNOWN（论文可访问片段未给型号；未提及 ZED/RealSense） |
| 相机安装位置 | 机器人机载 / egocentric（论文核心卖点：non-stationary robots, ego perspective） |
| 相机是否移动 | 标注用素材是**静止相机**拍的；论文用定性实验证明模型可迁移到 moving camera |
| 图像分辨率 | **UNKNOWN**（未从一手来源确认）；训练 imgsz=1024（GitHub config.py 确认） |
| FPS | UNKNOWN |
| 数据来源 | **自采**（不是公开标准数据集），作者自己采集后做成公开数据集 |
| 数据集名称 | 未命名正式数据集名（论文只称 our dataset），非公开标准数据集 |
| 帧数 | **20,510 帧**，11 个地点/背景（CAB1 2296 / CAB2 3407 / GLC1 2501 / GLC2 1648 / ML3 1148 / ML4 1366 / ML6 1238 / Ticino1 2572 / Ticino2 2382 / Uetlibergstrasse1 642 / Uetlibergstrasse2 1310） |
| 难度分级 | Easy 15,085 (73.5%) / Medium 4,594 (22.4%) / Hard 831 (4.1%)（论文 Fig.2） |
| 羽毛球标注类型 | **YOLO bbox**（半自动生成 → CVAT 人工校正 → 导出 YOLO 格式） |
| 标注质量 | 自动标注流水线准确率 **85.7%**（背景差分 + 对手分割 + 时域滤波）→ 约 14.3% 噪声，必须复核 |
| 是否公开 | 公开（项目页 + Google Drive；注意 GitHub README 里的下载链接是占位符 https://example.com，**README 本身给不出有效链接**） |
| 下载地址 | 项目页 / 用户提供的 Drive 目录 id=1neBfC7-Fp-l53muxZiIbIET6lYH5xv9s（本次环境无法访问 drive.google.com，未能列目录） |
| 许可证 | 代码 AGPL-3.0；**数据集许可 UNKNOWN** |
| 训练配置 | YOLOv8s，imgsz=1024，epochs=50，batch=32，AdamW lr=1e-4，额外混入 COCO-train 的 10%（select_coco.py） |
| 评价口径 | **不是 IoU**！config.py 里 dist_threshold=25.0 px、confidence=0.5 —— 预测中心与 GT 中心距离 < 25 px 即命中 |
| 结果 | F1 0.86（与训练相似环境）/ 0.70（完全未见环境）；论文明确说性能强烈依赖羽毛球尺寸与背景纹理复杂度 |
| 与你的任务相似度 | **高**（唯一的公开 egocentric 移动机器人羽毛球检测数据集） |

对你的 6 个重点问题的答案：
1. 是否真的移动机器人/自运动相机视角：**方向上是**，但用来标注的素材是静止机位，移动相机只是定性验证过。
2. 总帧数：20,510。
3. 分辨率：**UNKNOWN**。
4. 是否 YOLO bbox：**是**，原生 YOLO 格式。
5. easy/medium/hard 定义：论文只给分布；GitHub `scripts/sort_difficulty.py` 显示是**人工目视**把 easy 里的图拖到 medium/hard（放大裁切 + 画框交互），不是纯几何阈值。
6. recording location 数量：**11 个**（上面已列分布）。
7. 能否直接用于 YOLO：**能**（格式一致，imgsz 会由你控制）。
8. 是否存在真实多尺度 bbox：**是**（不同地点/距离，论文把 size 当核心变量分析），但具体尺度直方图 UNKNOWN。
9. 是否比普通固定比赛视角更适合机器人：**是**，明显更适合（egocentric + 11 背景 + 室内外 + 难度分级）；但它训练用 imgsz=1024，你按 640 用时小目标会更难。

### 2.2 YOLOBR —— Detecting the shuttlecock for a badminton robot: A YOLO based approach

- 论文：Cao, Liao, Song, Chen, Li，Expert Systems with Applications，DOI 10.1016/j.eswa.2020.113833（2020 online / 2021 卷期）。
- 任务：机器人实时检测；改造 Tiny YOLOv2 的 loss（适配小目标）与结构，对比 Faster R-CNN / SSD / Tiny YOLOv2 / YOLOv3。
- 数据：**自采，未公开**。摘要与所有可访问页面均未给数据集名称、规模、分辨率、划分方式。
- 是否用 ZED：**UNKNOWN**（摘要未提；不能因为同课题组另一篇 FTOC 用了 ZED 就推断）。
- 是否公开数据集：**否**（未发现任何公开数据链接）。
- 图像数量 / 羽毛球尺寸范围 / 训练测试划分：**UNKNOWN**（正文付费墙）。

### 2.3 FTOC —— Using FTOC to track shuttlecock for the badminton robot

- 论文：Chen, Liao, Li, Lin, Xue, Zhang, Guo, Cao，Neurocomputing 2019，DOI 10.1016/j.neucom.2019.01.023。
- 任务：**跟踪**（不是检测），融合异构线索 + AdaBoost 的 FTOC；最后落到 **ZED 双目相机** 的机器人实时 3D 羽毛球跟踪系统。
- 相机：**确认 ZED 双目**（摘要原文：a real-time ZED binocular camera based 3D shuttlecock tracking system for a robot）。
- **关于 BR dataset：没有找到任何公开证据证明存在名为 BR 的公开羽毛球数据集。** 已检索论文摘要页、S2、引文、中文/英文搜索，均无。结论：**UNKNOWN，且更可能是论文内部自采/自命名数据，不构成可下载的公开数据集。** 不要把它当作数据集名到处引用。
- ZED 分辨率 / 数据规模 / 是检测还是跟踪数据：跟踪数据为主；**分辨率与规模 UNKNOWN**。
- 是否公开：**否**（未发现公开链接）。

### 2.4 TrackNet / TrackNetV2 / TrackNetV3 + Shuttlecock Trajectory Dataset

- 原论文：TrackNet，arXiv:1907.03698（IEEE AVSS 2019）。heatmap + 连续 3 帧，输入 **640×360**，网球转播视频为主，另外自行标注羽毛球数据（论文文本出现 We have labeled 18,2xx frames of badminton，本次抓取到 18,2 处被 HTML 截断 → 明确记为「约 1.8 万帧，数字未完整确认」）。原文给出的下载页 https://nol.cs.nctu.edu.tw/ndo3je6av9/ （历史页面，当前状态 UNKNOWN）。
- 官方数据集页（TrackNetV3 仓库指向）：https://hackmd.io/Nf8Rh1NrSrqNUzmO0sQKZw —— **本次环境无法访问（fetch failed）**，因此当前链接有效性 UNKNOWN。
- 数据集结构（TrackNetV3 README 确认）：`Professional/` + `Amateur/` + `Test/` 三段；每场比赛一个目录，含 `frame/`(PNG) 与 `csv/*_ball.csv`（每帧球位置 + visibility）。
- 标注类型：**中心点 / 高斯热图，不是 bbox**。
- 相机：**固定机位、转播俯视视角**（MonoTrack 原文：overhead, static, broadcast-view camera）。
- 规模各来源不一致，需并列记录：
  - ETH 论文（二次引用）：55,563 帧，18 场比赛视频；
  - MonoTrack（CVPRW2022）：**77k** 标注帧，**26** 场单打，其中 3 场 test 共 **12k** 帧；
  - RacketVision 论文 Table 1：TrackNet 720p / 10 games / 19k；TrackNetV2 720p / 19 games / **78k** 帧。
- 分辨率：RacketVision 记为 720p；FPS **UNKNOWN**；许可证 **UNKNOWN**。
- 是否适合直接训练 YOLO：**不适合当 bbox GT**（点标注）。可转成小框做弱监督，但框大小是人为设定的，学不到真实尺度。TrackNetV3 论文里确实放了 YOLOv7 基线（badminton 57.82% accuracy）——说明可转换，但精度远低于热图法。
- 更适合：**轨迹跟踪/预测**（与你的第二阶段吻合），不是机载单帧检测。
- Kaggle 镜像：`phuc25111/shuttlecock-tracknetv2`（182 GB，声称 Apache-2.0，6 个版本，29 次下载）——二手镜像，许可与完整性需自行校验。

### 2.5 MultiSenseBadminton（Scientific Data 2024）

- 论文：Seong et al., Sci Data 11:343, 2024, DOI 10.1038/s41597-024-03144-z（GIST + MIT），CC BY 4.0。
- 规模：**7,763 次挥拍 / 25 名球员**（初学-中级-专家），传感器含 **eye tracking（眼动）、body tracking、EMG 肌电、足底压力**，另有视频录制。
- 标注：**击球类型、技术水平、击球声音、落点、击球位置**，以及问卷/访谈。
- Eye Video 是否是真正第一视角：提供的是**眼动追踪数据 + 视频录制**，本次无法读取方法学细节（PMC/Springer 均被拦截/重定向），因此严格说：**眼动数据存在**，但「是否是可用的第一视角视频、分辨率、FPS」= **UNKNOWN**。
- 是否存在逐帧羽毛球 bbox：**没有任何证据**；所有公开描述里都没有逐帧球框标注。结论：**不能直接训练检测器**。
- 对你还有什么价值：人体第一视角下的羽毛球**外观/尺度先验**（需自己重标）、击球事件与技术水平标签（可用于第二阶段的事件对齐/评测）、以及眼动-球位置关系（可用于预测人类何时看到球）。
- 获取：figshare 集合 id 6725706（本次 API 返回 403，未核到文件清单/许可细节）。

### 2.6 RacketVision（AAAI 2026，arXiv:2511.17045）

- 多球拍运动基准（乒乓/网球/羽毛球），**1,672 clips / 435,179 帧 / 12,755 s**；羽毛球部分 **461 clips / 114,753 帧 / 23,003 个球标注**。
- 数据来源：**YouTube 职业比赛转播**，1080p，固定机位。
- 标注：球位置是**单点 (x,y) + 可见性标志**，只标注每 clip **均匀采样 20% 的帧**（稀疏标注）；球拍是 bbox + 5 关键点。
- 任务：球跟踪（单帧 / 多帧 N=5）、球拍姿态、轨迹预测（长 N=80/M=20，短 N=20/M=5）。
- 评测：Precision / Recall / **MDE（像素，按 1920×1080）** / mAP@50。羽毛球列基线：RTMDet mAP 64.2、YOLO11 72.1、WASB 58.7、TrackNetV3 67.7。
- 公开：HuggingFace `linfeng302/RacketVision`（本次 HF 域名无法访问，**许可 UNKNOWN**）。
- 对你：**不适合机载检测**（转播视角、点标注、帧稀疏）；但**非常适合轨迹预测阶段**的预训练与评测对齐。

### 2.7 YO-CSA-T（arXiv:2501.06472）—— 最值得你抄「怎么自采双目数据」的样本

- 自建**双目系统**：2 台单目工业相机 A7200CU130（华瑞科技），**基线 0.8 m，架高约 1.8 m，放在机器人一侧场地的后场**（模拟成年男性视角）——**固定机位，不是机载**；不用 ZED，理由是 ZED 上不了 90 fps 以上。
- 数据：**自采**，10 个不同场馆/环境，共 **32,539 张**；从原图中随机裁 **640×640** 且保证球在裁切区（球不一定居中）→ 天然的**多尺度 + 位置多样性**。
- 标注：YOLO bbox；论文指出球框近似正方（符合羽毛球 68-78 mm 高 × 58-68 mm 径）。
- 测试：另采 12 段 160 fps 发球/击球视频。
- 结果：YO-CSA mAP@0.75 90.43%、mAP@0.5:0.95 74.0%、recall 99.02%、243.9 FPS。
- 是否公开：**未发现任何发布链接 → 视为不公开（UNKNOWN）**。

### 2.8 四足机械臂打羽毛球（arXiv:2505.22974，Learning coordinated badminton skills for legged manipulators）

- 硬件：ANYmal-D 底盘 + DynaArm + **ZED X 双目全局快门** + 机载 Jetson AGX Orin。
- **关键事实：他们真机感知没有用任何学习式检测器**，而是 **HSV 颜色阈值（羽毛球橙色）+ 双目三角化 + EKF**；感知链路 60 Hz，相机快门到球位置计算延迟实测 **60-160 ms**。
- 数据：为拟合**感知噪声模型**，用**动捕系统**在相机移动、球固定的条件下自采真实数据（回归检测概率/位置误差与距离、角速度的关系）。
- 对你的含义：真机上「先颜色阈值 + 几何」往往比「先训检测器」更快更稳；检测器更适合作为**远距离/复杂背景下的补充**，并且**必须把延迟（60-160 ms）与相机自运动误差建模进训练**。

### 2.9 其他（补充/弱相关，非主结果）

- MonoTrack（CVPRW 2022，arXiv:2204.01899）：基于 TrackNetV2 数据集做击球分段 + 3D 轨迹重建；给 TrackNetV2 数据补了场地四角与击球人标签。**转播视角，2D 点 → 3D 轨迹**。
- ShuttleSet（KDD 2023，arXiv:2306.04948）+ Coach AI：**击球级别**战术数据集（转播），不是逐帧检测。
- A Badminton Recognition and Tracking System Based on Context Multi-feature Fusion（arXiv:2306.14492）：使用 Coach AI 数据。
- Badminton trajectory tracking method based on binocular stereo vision and YOLOv8（Discover Applied Sciences 2026，DOI 10.1007/s42452-026-08354-1，Yuan Zhang）：**摘要不可得（S2 返回空）**，细节 UNKNOWN，只能列为待读。
- EgoTouch（HuggingFace `zhouzhoujy/EgoTouch`）含目录 `Outdoor/pick_up_badminton_shuttlecock/...`：第一视角捡球视频，**未见检测标注证据，未核实**。
- BMT-Bench 论文中出现一个 Labeled Badminton partial dataset 的说法：**未核实**（该论文是视频生成基准，NSF 页面抓取失败）。

## 3. 已知候选核实结论（你点名要查的 5 项）

1. **One-Shot Badminton Shuttle Detection for Mobile Robots**：真实、公开、机载视角导向、20,510 帧、YOLO bbox、11 地点、easy/medium/hard = 73.5/22.4/4.1%，难度靠人工目视排序；但**原始分辨率与 FPS 未确认**、**数据集许可未确认**、README 下载链接是占位符。结论：**是目前最值得拿的主真实数据集，但要承担约 14% 的标注噪声并自行复核。**
2. **YOLOBR**：自采、未公开、非 ZED 证据不足；规模/尺寸/划分 UNKNOWN。结论：**只能当方法参考（小目标 loss 改造 + 结构改造），不能当数据来源。**
3. **FTOC**：ZED 双目确认；任务是跟踪不是检测；**BR dataset 查无实据 → UNKNOWN**；分辨率/规模 UNKNOWN。
4. **TrackNet 系**：Shuttlecock Trajectory Dataset 是公开的、固定转播机位的**点标注**数据；官方页面当前不可访问；帧数口径有 55,563 / 77k / 78k 三种说法（分别来自 ETH 论文 / MonoTrack / RacketVision）；**不能直接当 YOLO bbox GT**，更适合跟踪与预测预训练。
5. **MultiSenseBadminton**：7,763 挥拍 / 25 人，多传感器 + 视频 + 事件级标注；**没有逐帧羽毛球 bbox**，不能直接训检测器；眼动视频的分辨率/FPS UNKNOWN；价值在于第一视角外观先验与事件标签。

## 4. 其他数据源（我另找到的）

| 来源 | 是什么 | 真实羽毛球检测可用性 |
|---|---|---|
| Roboflow Universe `mathieu-cartron/shuttlecock-cqzy3` | 5.6K train / 1.6K valid / 816 test，统一 resize 到 **640×640**，两类（Shuttlecock / Null） | 有你现有那份；经你核验：等效尺寸仅 5.2-8.6 px、只有 8 种框形状、99.8% 集中在两种框 → **高度疑似点标注转固定框，框不是真实边界尺度**。只能当外观/负样本补充，不能当尺度真值 |
| Kaggle `ayushsinha731/shuttle-badminton-photos` | 2.2 GB，**由两个 Roboflow 数据集合并清洗**，YOLO + COCO 标注 | 二手再分发，**许可 Unknown**，标注继承上游问题 → 不建议作主源 |
| Kaggle `phuc25111/shuttlecock-tracknetv2` | 182 GB，声称 Apache-2.0，6 个版本 | TrackNet 系数据的镜像，点标注，二手 |
| Kaggle 其他 shuttle/badminton 检索结果 | 绝大多数是航天飞机、酒店评论、排名 CSV | 与检测无关，排除 |
| HuggingFace | 本次 HF API/网页全部抓取失败；仅确认存在 `linfeng302/RacketVision`、`zhouzhoujy/EgoTouch` | 许可与内容 UNKNOWN |
| Zenodo / Figshare | 命中 MultiSenseBadminton 集合（figshare 403）、A New Perspective for Shuttlecock Hitting Event Detection（Zenodo 14677727，事件检测，非检测框） | 无新的 bbox 检测数据集 |
| Papers With Code | 未返回可用的 shuttlecock 检测数据集条目 | 无 |

## 5. 与你的现状对比

你现在的资产：922 个合成正样本（存储 960×960 → 网络输入 640×640），尺度覆盖 <4 px 到 >64 px（<4:14.2%, 4-6:17.5%, 6-8:12.1%, 8-12:11.9%, 12-16:4.0%, 16-24:3.6%, 24-32:2.5%, 32-64:8.6%, >64:25.6%）；真实侧只有 Roboflow 8053 张（尺度 5.2-8.6 px、8 种框形状）。

对照调研结论，你的缺口被精确定位为三点：

1. **真实 8-64 px 的 bbox 真值**：公开数据里唯一可能提供的是 ETH（imgsz 1024、真实多尺度、机载视角），其余公开数据是转播（TrackNet 系点标注 / RacketVision 点标注）或 640 固定小框（Roboflow）。
2. **相机自运动/机载视角**：只有 ETH 有（且只是定性验证移动相机）。
3. **可靠标注**：ETH 半自动流水线 85.7% 准确率意味着约 14% 噪声 —— 你必须复核，否则等于把噪声当 GT 训。

另外提醒一个口径差异：**ETH 用 25 px 中心距离命中，你用 IoU>=0.5**。同一个模型在两种口径下数字不可直接比较。若要引用 ETH 的 0.86/0.70，必须同时说明其口径。

## 6. 最终结论表

| 数据集 | 真实图像 | 机器人/人视角 | 相机移动 | bbox可靠 | 多尺度 | 可直接YOLO训练 | 推荐用途 |
|---|---|---|---|---|---|---|---|
| ETH RSL shuttle_detection (20,510) | 是 | **是（机载 egocentric）** | 采集时静止，验证过移动 | 中（半自动 85.7%，需复核） | 是 | 是（YOLO 原生） | **主真实检测训练 + 移动相机鲁棒性** |
| TrackNet Shuttlecock Trajectory Dataset | 是 | 否（转播俯视） | 否 | 无 bbox（点/热图） | 有限 | 需转框（弱） | 跟踪/预测预训练、弱标注、评测 |
| RacketVision (badminton 部分) | 是 | 否（转播） | 否 | 点标注（20% 帧） | 是（1080p） | 否 | 轨迹预测预训练与评测 |
| MultiSenseBadminton | 是 | 是（人体/眼动，非机载） | 人动 | 无逐帧球框 | 未知 | 否 | 第一视角外观先验、击球事件标签 |
| YO-CSA-T 自采 (32,539) | 是 | 场边人眼高度（非机载） | 否 | 高 | 是（10 场馆 + 640 随机裁） | 是（但未公开） | 自采双目数据集的**做法模板** |
| Roboflow Shuttlecock (你已有) | 是 | 转播（推测） | 否 | 低（疑似点转框） | 否（5.2-8.6 px） | 格式可，学不到尺度 | 外观补充 / 困难负样本 |
| Kaggle 镜像 (shuttle-badminton-photos / tracknetv2) | 是 | 混合 | — | 二手继承上游 | — | 谨慎 | 仅作备份渠道 |
| 你的合成数据 (922) | 否（合成） | 可配置 | 可配置 | 精确（自生成） | 是（<4 到 >64） | 是 | 预训练 / 尺度覆盖 / 消融对照 |

## 7. 最适合你的 3 个数据源（各自负责什么）

**第 1 个：ETH RSL 的 shuttle_detection 数据集（主真实检测训练）**
- 负责：真实机载视角 + 真实多尺度 + 真实背景纹理；这是你唯一能拿到的公开 egocentric 检测数据。
- 用法：先**逐张复核**（它自身流水线有 ~14% 错误），按 easy/medium/hard 分别统计你的模型表现；把它的 unseen 环境当作第二套冻结测试集。
- 注意：它是 imgsz=1024 训的；你按 640 输入时，小目标会明显更难，不要直接用它论文的 0.86 做对标。

**第 2 个：你自采的真实数据（真实尺度真值的主要来源）——必须做**
- 负责：8-64 px 区间的**真实 bbox 真值**（这是你当前最大的空白）。
- 抄 YO-CSA-T 的做法：用你自己的机载/同高度相机，在多场馆多背景拍，然后 **从原图随机裁 640×640 且保证球在框内**（球不居中）→ 一次采集同时得到多尺度与位置多样性；再用 ETH 的**背景差分 + 对手分割 + 时域滤波**做半自动预标注，最后人工复核。
- 现阶段可先用手机/相机在球馆拍（不必等机器人），因为你要的是**像素尺度分布**而不是底盘运动。

**第 3 个：TrackNet Shuttlecock Trajectory Dataset + RacketVision（弱标注 + 跟踪/预测预训练）**
- 负责：第二阶段（跟踪与 0.3-0.8 s 轨迹预测）的预训练与评测；也可作为检测器的**对比基线口径**。
- 用法：只做**弱监督**（点 → 伪框，或直接训练热图分支），**不得**混入你的冻结测试集，也不要指望它给你 8-64 px 的真实框尺度。

## 8. 数据策略（加入训练 / 弱标注 / 只测试 / 需重标）

- **直接加入训练**：
  - ETH shuttle_detection（复核后的子集，保留 easy/medium；hard 单列）；
  - 自采真实数据（人工复核过的 bbox）；
  - 你的 922 合成（用于 <4 px 与 >64 px 两端的尺度覆盖，训练时记录合成/真实配比）。
- **只能作为弱标注 / 预训练**：TrackNet 系（点）、RacketVision（点，20% 帧）、Roboflow（格式可读但框尺度不可信 → 最多用于外观与负样本，不作为尺度真值）。
- **只用于测试（不得进训练）**：你已冻结的 Capability Test Set；ETH 的 unseen-environment 子集；任何后续从 MultiSense/RacketVision 抽出来的评测集。
- **需要重新标注**：Roboflow 的可疑框；ETH 半自动标注中的 hard 级与低置信样本；你的 29 张 NEEDS_BOX_FIX；若要用 MultiSense 视频，必须自己逐帧标球。

## 9. UNKNOWN 清单（不要替我猜）

- ETH：原始分辨率、FPS、相机型号、数据集许可证；项目页与 Drive 目录本次无法访问。
- YOLOBR：数据规模、采集相机、划分方式、是否公开。
- FTOC：所谓 BR dataset 的来源与公开性（**查无实据**）、ZED 分辨率与数据规模。
- TrackNet：官方数据集页当前不可访问（链接有效性未知）；精确帧数三个来源冲突；FPS 未知；许可未知。
- MultiSenseBadminton：眼动视频的分辨率/FPS；figshare 文件清单与许可（API 403）。
- RacketVision：HuggingFace 页面许可（HF 域名本次不可达）。
- YO-CSA-T：数据集是否公开（未发现发布链接）。
- Roboflow：项目许可（页面被 Cloudflare 拦）。
- EgoTouch / BMT-Bench 的 Labeled Badminton：是否含检测标注未核实。

## 10. 证据索引

- ETH 论文 abs：https://arxiv.org/abs/2603.06691 ；全文（HTML，本次被截断）：https://arxiv.org/html/2603.06691v2 ；ar5iv：https://ar5iv.labs.arxiv.org/html/2603.06691
- ETH 代码与配置：https://github.com/leggedrobotics/shuttle_detection （README / src/shuttletrack/config.py / scripts/sort_difficulty.py）
- YOLOBR：https://www.sciencedirect.com/science/article/abs/pii/S0957417420306436 ；摘要经 Semantic Scholar DOI:10.1016/j.eswa.2020.113833
- FTOC：https://www.sciencedirect.com/science/article/abs/pii/S0925231219300359 ；DOI:10.1016/j.neucom.2019.01.023
- TrackNet：https://ar5iv.labs.arxiv.org/html/1907.03698 ；数据集页 https://hackmd.io/Nf8Rh1NrSrqNUzmO0sQKZw （不可访问）；TrackNetV3 https://github.com/qaz812345/TrackNetV3 ；TrackNetV3 论文 https://dl.acm.org/doi/fullHtml/10.1145/3595916.3626370
- MonoTrack：https://ar5iv.labs.arxiv.org/html/2204.01899
- RacketVision：https://ar5iv.labs.arxiv.org/html/2511.17045 ；数据集 https://huggingface.co/datasets/linfeng302/RacketVision
- MultiSenseBadminton：https://www.nature.com/articles/s41597-024-03144-z ；https://pmc.ncbi.nlm.nih.gov/articles/PMC10997636/ ；figshare 集合 6725706
- YO-CSA-T：https://ar5iv.labs.arxiv.org/html/2501.06472
- 四足机械臂羽毛球：https://ar5iv.labs.arxiv.org/html/2505.22974
- Roboflow 官方博客（Shuttlecock 项目统计）：https://blog.roboflow.com/top-sports-datasets-computer-vision/ ；项目页 https://universe.roboflow.com/mathieu-cartron/shuttlecock-cqzy3
- ShuttleSet：https://ar5iv.labs.arxiv.org/html/2306.04948

---
状态：本文件为调研产出（证据型文档）。所有结论均可回溯到上面的一手链接；UNKNOWN 项需在有网络/账号条件时补齐（尤其是 ETH 项目页与 Google Drive、TrackNet hackmd 页、HF 页面）。