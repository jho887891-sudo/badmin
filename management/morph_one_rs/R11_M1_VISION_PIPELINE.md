# R11 · M1 里程碑：真实视频 → 真实 YOLO → 2D EKF → 像面预测 → 可视化

**结论：M1 = PARTIAL（管线真实跑通并产出可复现 overlay，但检测在真实比赛视频上为 0，且默认过程噪声使预测区间不可用）**

分支 `r11/vision-m1`（base `p7/detector-ekf-interfaces@8150b2a`）
入口 `scripts/vision/run_visual_pipeline.py`（新增，550 行）

## 1. 真实资源清单（V01）

| 项 | 值 |
|---|---|
| 权重 | `/home/dgut/.dsh-bench/runs/detect/runs_cont_v1/full_cont_e20/weights/best.pt` |
| 权重 SHA256 | `918dc5037ea49fc3749c1f35a3bf774255c8d0383be20360a0a1c5d2ee810a87` |
| 训练来源 | `yolo26s_v2_full_eth_cont_v1`（24673 train / 504 val，类 `shuttlecock`），20 epoch 续训 |
| 推理栈 | ultralytics **8.4.183**（隔离安装于 `/home/dgut/vision_libs`，不改 Isaac env）、torch **2.10.0+cu128**、CUDA 可用 |
| 参数 | imgsz 640/1024，conf 0.25/0.10/0.05/0.01，iou 0.45，max_det 64，device cuda:0 |
| 真实视频 | `_scratch_matches/am_doubles/match_clementi/rally_video/doubles0.mp4`，**336 帧 / 30 fps / 1280×720**，SHA256 `8b98ffc205708e477f81faee31ec96dd4ef639ee6ddf226f3cc773bdc1a800a9f` |
| 真实帧序列 | ETH `ticino_2_easy/images/train`，60 帧连续，1920×1200；另一组来自 `val.txt`（41 帧，**采样**） |

**仓库内**另有 `runs/shuttle_yolo26_v1/smoke1024_clean/weights/best.pt`（SHA256 `93ae86c6…`，1 epoch / 200 图，smoke 级）——本报告未使用它，只用 20-epoch 的 `full_cont_e20`。

## 2. 真实运行结果（全部 exit 0，GPU cuda:0）

| RUN_ID | 输入 | imgsz | 帧 | 检出率 | 检测 p50/p95 | 0.3 s 预测误差(中位/p95) | 0.3 s σ |
|---|---|---|---|---|---|---|---|
| `R11_M1_ethseq_v2` | ETH 连续 60 帧 | 640 | 60 | **59/60 = 0.9833** | 13.11 / 16.01 ms | **7.78 / 13.94 px** | 614.5 px |
| `R11_M1_ethseq_1024` | ETH 连续 60 帧 | 1024 | 60 | 59/60 = 0.9833 | 14.26 / 15.19 ms | 7.28 / 14.73 px | 614.5 px |
| `R11_M1_ethseq_calib` | 同上，噪声标定 | 640 | 60 | 59/60 = 0.9833 | 13.49 / 16.84 ms | 18.51 / 22.30 px | **15.2 px** |
| `R11_M1_broadcast_640` | 真实比赛视频 | 640 | 336 | **0/336 = 0.0** | 10.65 / 11.94 ms | — | — |
| `R11_M1_ethval_v2` | ETH val（采样帧） | 640 | 41 | 40/41 = 0.9756 | 13.21 / — | 266.5 / 597.6 px | 879.1 px |

EKF 状态分布（连续序列）：**UPDATED 58 / INITIALISED 1 / IDLE 1**，关联接受 58、拒绝 0、重初始化 1；
EKF 速度中位数 `vx = −327.9 px/s`、`vy = −35.1 px/s`。
val 采样序列：**COASTING 33 / INITIALISED 6 / UPDATED 2，拒绝 32** —— 滤波器正确地在帧间隔过大时进入 coasting 并丢弃跟踪，这**不是** bug。

## 3. 三个真实发现（不是规划，是测量）

### 3.1 真实比赛视频 0/336 —— 领域差距，不是管线问题
同一份权重在 ETH 室内数据上 59/60，在真实比赛视频上 **conf 0.25 / 0.10 / 0.05 / 0.01 全部 0 检出**（336 帧）。
管线本身正常（同一次运行写出了 336 行 tracks、overlay.mp4 11 MB、p50 10.6 ms）。
→ **该 checkpoint 不能用于真实比赛场景**；真实广播画面的羽毛球尺寸/运动模糊/背景与该权重的训练分布不同。
这是本轮最重要的负面结论，且**不允许**用手工 bbox 或合成数据掩盖。

### 3.2 继承自 P7-05 的默认过程噪声让预测区间物理上不可用
`process_accel_std = 2000 px/s²`（默认）产生
`σ(0.3s) = 614.5 px`、`σ(0.5s) = 1035.9 px`、`σ(0.8s) = 1727.1 px` —— 0.8 s 的区间宽度（1727 px）已接近画面宽度（1920 px）。
而同一次运行的 **0.3 s 预测误差中位数仅 7.78 px**：σ 比实际误差大 **79 倍**。
→ **均值预测很好，区间毫无意义**。在像面上做 0.3–0.8 s 预测前，必须按残差重新标定过程噪声。

### 3.3 标定后区间自洽，但平滑滞后变大（真实权衡，不掩盖）
用同一段视频做**样本内**标定（`--process-accel-std 45.0`，标签 `IN_SAMPLE_TUNED_ON_THIS_CLIP`）：
σ(0.3s) 从 614.5 → **15.2 px**，与实测误差（中位 **18.51 px**、p95 22.3 px）同量级；
但预测误差中位数从 7.78 → **18.51 px** 变差（滤波更平滑 → 滞后更大）。
→ 两个配置**都保留**，谁更好取决于后续是要"区间可用"还是"点预测准"。**不许**用它去改历史数字。

## 4. 本轮修掉的 5 个真实缺陷（都是我自己代码/契约的 bug）

| # | 缺陷 | 表现 | 修法 |
|---|---|---|---|
| 1 | `ValidityFlags(in_image=,occluded=,estimated=)` 关键字根本不存在 | 首个真实检测触发 `TypeError`；**被"0 检出"掩盖**了整轮 | 改用真实字段 `detected/in_image/confidence_ok/covariance_ok/timestamp_ok` |
| 2 | EKF `initialize/step` 要的是 `(2,)` 数组，不是 `DetectionObservation` | `TypeError: float() argument must be … not DetectionObservation` | 传 bbox 中心 + `pixel_covariance` |
| 3 | EKF 未初始化时 `state` **抛异常**而不是返回 `None` | `ShuttleEKFError: filter is not initialised` | 改为 probe（try/except）而非比较 None |
| 4 | 图像序列被当成"均匀 1/30 s"喂进 EKF | val 序列 266 px 误差**全部是时钟伪差**；且掩盖了真实帧间隔 | 从文件名尾号解析帧号生成时间戳，并标记 `FILENAME_INDEX_ASSUMED_FPS` / `sequence_spacing_regular=false` |
| 5 | 图像序列被标成 `REAL_RGB_VIDEO` | 来源分类错误 | 增加 `source_class`：`REAL_RGB_VIDEO` / `REAL_RGB_FRAMES` |

**缺陷 1 的教训最值得记**：第一轮广播视频跑出 "0 检出" 时管线是"成功"的，因为**检测构造代码从未被执行**。只有换成有检出的真实数据，隐藏的 `TypeError` 才暴露。**"全零结果"会让整条下游代码路径免于执行。**

## 5. 输出与证据

每个 run 目录：`overlay.mp4`（框、历史轨迹、0.3/0.5/0.8 s 预测点 + 2σ 椭圆、状态文字、`IMAGE_PLANE_2D` 水印）、
`detections.jsonl`、`tracks.jsonl`、`metrics.json`、`manifest.json`、stdout/stderr。
证据根：`outputs/vision/r11_m1/<RUN_ID>/`，样例帧 `frames/overlay_*.png`（源 1920×1200，缩至 960 宽）。

## 6. 明确未达成 / UNKNOWN

- **没有 3D**：全程 `IMAGE_PLANE_2D`，`three_d_claimed=false`，`hit_rate=null`。
- **ETH 连续序列来自 `images/train`**：0.9833 的检出率**可能包含训练帧**，**不能**当作泛化指标；真正的 held-out 序列因采样间隔过大而不可跟踪（见 3.1/§2）。
- `pixel_covariance` 是 **bbox 尺寸派生的代理**（0.25×extent，1 px 下限），不是标定过的检测不确定度。
- 未做：D02 分桶召回/precision（需逐帧 GT 对齐）、D03 裁剪/tiling、G 线深度/双目、S 线 Isaac 相机。
- 时间戳：视频走 PTS 或 `ASSUMED_FPS`，图像序列走文件名帧号 —— **都不是硬件同步时钟**。
