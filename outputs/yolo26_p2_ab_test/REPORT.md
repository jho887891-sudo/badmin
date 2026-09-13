# REPORT.md — YOLO26s vs YOLO26s-P2 羽毛球 tiny-object 检测 A/B（第 1 版）

生成时间：2026-09-13（UTC）　运行机：远端 jxxy（NVIDIA RTX A6000, driver 550.163.01）
实验目录：experiments/yolo26_p2_ab/　产物目录：outputs/yolo26_p2_ab_test/
**本版只包含有证据支持的部分**；所有无法执行的部分明确标注 BLOCKED 与原因。

## 0. 结论摘要（先说不可下结论的原因）

- **结构 / 算力 / 延迟**：已完成实测（Q1 / Q2 / Q11 / Q12 有数字）。
- **精度类（Recall_XS / Recall_S、连续漏检、hard negative、stereo 可用率、有效 ShuttleMeasurement 率）**：全部 **BLOCKED**。
  原因：仓库内**不存在任何羽毛球检测数据集**（dataset_audit.json：扫描 404 文件、数据集 YAML 0、YOLO 标注 0、
  图像 21 张且全部是渲染证据）。因此**本报告不给出任何 recall / mAP 数字**。
- 依据协议第 34 / 36 节：最终模型选择（BASELINE / P2 / …）**现在无法判定**，标记为 `NO_DECISION_POSSIBLE_YET`。

## 1. 实验定义与公平性

| 项 | 实验 A（baseline） | 实验 B（P2） |
|---|---|---|
| 官方配置 | yolo26s.yaml | yolo26s-p2.yaml |
| scale | s | s（**已实测确认**） |
| 权重 | yolo26s.pt（COCO 预训练，AGPL-3.0） | 无官方 P2 权重，仅部分迁移 |

**关键陷阱（已实测纠正）**：官方 P2 的 scale 命名是 `yolo26s-p2.yaml`（scale 字母在 `-p2` 之前）。
`yolo26-p2s.yaml` **不会**被解析出 scale（`yaml_model_load` 返回空 scale），会静默退回 n 规模——
即 2.662M 参数的 n 模型 vs 10.01M 的 s 模型，属**不公平比较**。
实测证据：`yolo26s.yaml→scale=s`、`yolo26s-p2.yaml→scale=s`、`yolo26-p2s.yaml→scale=(空)`。

## 2. Q1 — 两者的实际结构差别

来自官方配置逐字（见 model_configs_discovered.txt 与 02_arch_static.py 输出）：

```
yolo26.yaml    : [[16, 19, 22], 1, Detect, [nc]]      # Detect(P3, P4, P5)
yolo26-p2.yaml : [[19, 22, 25, 28], 1, Detect, [nc]]  # Detect(P2, P3, P4, P5)
```

P2 增加：一个 `nn.Upsample` + 与 backbone P2 的 `Concat`、一个 `C3k2 [128]`（注释 P2/4-xsmall），
并把整个 neck 的层编号后移（baseline 的 16/19/22 → P2 的 19/22/25/28）。
**P2 是 ultralytics 8.4.150 自带的官方配置，不是自写网络。**

| 指标（scale=s, imgsz=640） | baseline | P2 | 变化 |
|---|---|---|---|
| 检测层数 | 3 | 4 | +P2 |
| 检测 stride (px) | 8 / 16 / 32 | **4** / 8 / 16 / 32 | 新增 stride 4 |
| head 特征图 | 80/40/20 | **160**/80/40/20 | 最细一层 ×2 |
| 预测框上限 | 8400 | **34000** | ×4.05 |
| 参数量 | 10.010 M | **9.766 M** | **−2.4%**（P2 反而更少，实测） |
| GFLOPs @640 | 23.079 | **28.023** | **+21.4%** |
| fp32 权重大小 | 40.04 MB | 39.06 MB | −2.4% |
| 输出结构 | 检测头 dict：one2many / one2one（boxes / scores / feats） | 同结构，feats 多一层 160×160 | — |

## 3. Q2 — P2 实际增加多少参数 / FLOPs / VRAM / 延迟

- **参数**：**减少** 2.4%（10.010M → 9.766M）。不能假设 P2 一定更重。
- **FLOPs**：+21.4%（23.079 → 28.023 GFLOPs @640，thop 实测）。
- **峰值 VRAM（单次前向）**：batch=1 @640 78.7 → 97.6 MiB；@1280 163.3 → 225.5（+38%）；
  batch=8 @1280 927.5 → **1440.2 MiB**（+55%）。
- **延迟（FP16，warmup≥100，cuda events + synchronize，模型加载不计入）**：

| 配置 | baseline mean | P2 mean | P2/baseline |
|---|---|---|---|
| batch=1 @640 | 15.65 ms | 21.69 ms | **1.39×** |
| batch=1 @960 | 16.19 ms | 21.16 ms | 1.31× |
| batch=1 @1280 | 17.17 ms | 20.35 ms | 1.19× |
| batch=8 @640（每图） | 2.00 ms | 2.57 ms | **1.29×** |
| batch=8 @960（每图） | 3.09 ms | 4.94 ms | 1.60× |
| batch=8 @1280（每图） | 5.30 ms | 8.55 ms | **1.61×** |

**测量学说明（避免误读）**：batch=1 时延迟几乎与分辨率无关（640/960/1280 ≈ 15.7/16.2/17.2 ms），
诊断为**内核启动与调度开销主导**：同机 FP16 matmul 参考值健康（1×1024² = 0.039 ms），
而 GPU 起始处于深度空闲（clocks.sm 读数 0 MHz、功耗 26 W），跑批后才升到 1905 MHz / 274 W / 100% util。
因此 **batch=1 的数字只能用于同条件相对比较，不能当作实时预算依据**；batch=8 才显示出与 FLOPs 相符的分辨率标度。

## 4. 权重公平性（协议第 5 节）

weight_transfer_report.txt（实测）：

```
VERDICT: PARTIAL_PRETRAIN_TRANSFER
tensors in baseline 708 / in P2 902
shape-compatible      : 360  (6,075,238 params)
shape mismatch        : 60   (层编号后移导致)
missing (P2 only)     : 482  (3,607,531 params)
unexpected (base only): 288  (2,210,340 params)
coverage of P2 params : 61.96%
```

**结论：P2 只能部分继承 COCO 预训练（按名称匹配 61.96%），约 38% 参数必须从头训练。**
因此精度比较**必须**在同一协议下各自训练；**不得**把本报告的结构 / 延迟结论外推为精度结论。

## 5. Q3–Q10 — 精度与系统级指标：全部 BLOCKED

| 问题 | 状态 | 原因 / 解锁条件 |
|---|---|---|
| Q3 ≤8 px recall 提升 | BLOCKED | DATASET_NOT_AVAILABLE |
| Q4 8–16 px recall 提升 | BLOCKED | 同上 |
| Q5 最长连续漏检是否降低 | BLOCKED | 需要视频序列 GT |
| Q6 hard-negative FP 是否增加 | BLOCKED | 需要无球帧集合（网 / 线 / 灯 / 鞋 / 反光…） |
| Q7 long false track 是否增加 | BLOCKED | 同上（连续帧统计） |
| Q8 左右目同时检测率 | BLOCKED | 需要同步双目数据 |
| Q9 stereo-usable candidate 率 | BLOCKED | 需要双目数据 + 标定 |
| Q10 有效 ShuttleMeasurement 率 | BLOCKED | DOWNSTREAM_NOT_AVAILABLE：亚像素质心 / 三角化 / UKF 代码已实现，
但需要真实（或渲染）图像与其标定才能跑通到 measurement 输出 |

对照协议第 34 节：这些正属于允许立即执行范围之外的部分，故**不给出任何数字**。

## 6. Q11 — 分辨率收益（延迟侧已测，精度侧 BLOCKED）

batch=8 @640 → @960 → @1280（每图）：baseline 2.00 → 3.09 → 5.30 ms（2.65×）；
P2 2.57 → 4.94 → 8.55 ms（3.33×）。
**精度侧收益（是否值得为 tiny shuttle 提高分辨率）当前无法评估 —— BLOCKED。**

## 7. Q12 — 同延迟预算下 P2 vs 高分辨率

可构造的等延迟对（batch=1 实测）：P2@640 ≈ 21.7 ms vs baseline@1280 ≈ 17.2 ms；P2@960 ≈ 21.2 ms vs baseline@1280 ≈ 17.2 ms。
结构上两者同量级，但**哪个更值取决于 Recall_XS / Recall_S 与 stereo 可用率 —— BLOCKED**。
已按协议第 27 节预留接口：有数据后 Recall_XS per ms 可由 threshold_sweep.csv 与本延迟表直接计算。

## 8. Q13 — TensorRT FP16 p95

`TENSORRT_BLOCKED`：环境**未安装 TensorRT 与 onnxruntime**（environment.txt 实测 NOT_INSTALLED），
且协议禁止为强装而改动共享核心环境。本轮**未**导出 ONNX、**未**测 TensorRT。
解锁：另建独立环境安装 TensorRT / onnxruntime（不触碰 Isaac 环境），同导出设置各导一次 FP16 engine。

## 9. Q14 — GPU 并发负载下的退化

测量时段内 GPU 上**无其他计算进程**（只读快照：0 个 compute app），即本轮数字是**孤立负载**。
**未人为制造并发负载**（协议要求不干扰其他用户任务，且该机为共享机）。
因此 concurrent latency / degradation % 标记为 **NOT_MEASURED**（原因与做法已记录在 experiment_config.yaml）。

## 10. Q15 — 统计显著性

- 精度指标：**BLOCKED**（无数据，无法 bootstrap CI）。
- 延迟：单次会话内每组合 300–500 次测量的 mean / median / p95 / p99 已给出，但**未做多会话重复**，故不宣称置信区间。

## 11. Q16 — 最终选择

**`NO_DECISION_POSSIBLE_YET`**（按协议第 33 节决策规则，第一至第四优先级均缺数据）。

当前唯一可负责陈述的量化事实：

1. P2 用 **+21.4% FLOPs / +29–61% 延迟（实测范围） / +38–55% 峰值显存**，换到最细检测 stride 4（160×160）与 4× 预测位置；参数量反而 −2.4%。
2. P2 无法完整继承 COCO 预训练（61.96% 覆盖），两组都必须同协议训练后才能比精度。
3. batch=1 且 GPU 深度空闲时延迟被启动开销主导 —— 单图 15 ms 这类数字**不能当作实时预算**。

### 解锁最终决策所需的最小工作

1. **数据集**：整流 / 同步后的双目图像，class 0 = shuttlecock（其余作为 background / hard negative），
   按 recording session 隔离 train/val/test，禁止帧级泄漏，每个可见羽毛球必须标注。
2. **训练**：两结构同协议（同 split / seed / epochs / imgsz / batch / optimizer / LR / 增强 / 早停 / 精度 / 硬件），≥3 seeds。
3. **评测**：size 分桶（XS ≤8 / S 8–16 / M 16–32 / L >32）、conf 扫描 0.05–0.90、hard-negative 帧集、
   视频连续性与遮挡重捕获、ROI 扩张 1.25/1.5/2.0、双目同时检测率。
4. **下游**：把候选接到已实现的 subpixel → triangulation → UKF，统计 P(valid ShuttleMeasurement | visible)。

## 12. 产物清单（outputs/yolo26_p2_ab_test/）

有数据：environment.txt、experiment_config.yaml、model_summary.csv、model_configs_discovered.txt、
weight_transfer_report.txt、latency_pytorch_b1.csv、latency_pytorch_b8.csv、dataset_audit.json/md、
evidence/{latency_vs_imgsz_b1.png, latency_vs_imgsz_b8.png, gflops_compare.png}

BLOCKED（文件首行即状态与原因）：accuracy_overall.csv、accuracy_by_size.csv、threshold_sweep.csv、
hard_negative.csv、video_continuity.csv、occlusion_reacquisition.csv、roi_coverage.csv、stereo_usability.csv、
downstream_measurement.csv、annotation_stats.csv、latency_end_to_end.csv、latency_tensorrt.csv、statistical_analysis.csv

## 13. 复现命令

```bash
cd /home/T7/ojh/robot_sim
EX=$PWD/experiments/yolo26_p2_ab/_wheel_extract      # 官方 wheel 解包（未安装）
DEPS=$PWD/experiments/yolo26_p2_ab/_deps             # 仅 thop（--no-deps）
bash experiments/yolo26_p2_ab/00_env_audit.sh
PYTHONPATH=$EX       ./env_isaaclab/bin/python experiments/yolo26_p2_ab/01_discover.py
PYTHONPATH=$EX       ./env_isaaclab/bin/python experiments/yolo26_p2_ab/02_arch_static.py
                     ./env_isaaclab/bin/python experiments/yolo26_p2_ab/03_dataset_audit.py
PYTHONPATH=$EX:$DEPS ./env_isaaclab/bin/python experiments/yolo26_p2_ab/04_model_summary.py --scale s --imgsz 640
PYTHONPATH=$EX:$DEPS ./env_isaaclab/bin/python experiments/yolo26_p2_ab/05_weight_transfer.py
PYTHONPATH=$EX:$DEPS ./env_isaaclab/bin/python experiments/yolo26_p2_ab/06_latency_pytorch.py --imgsz 640 960 1280 --warmup 100 --iters 300 --batch 1
PYTHONPATH=$EX:$DEPS ./env_isaaclab/bin/python experiments/yolo26_p2_ab/07_latency_diag.py
                     ./env_isaaclab/bin/python experiments/yolo26_p2_ab/08_write_report.py
```

## 14. 环境与合规声明

- **未修改** CUDA / 驱动 / PyTorch / Isaac Sim / Isaac Lab：ultralytics 通过 PYTHONPATH 使用**解包的 wheel**；
  唯一额外安装是 ultralytics-thop（**--no-deps 到独立 _deps/**，16 MB，纯 python，复用环境 torch）。
  期间一次误操作让 pip 连带下载了 4.2 GB 的 CUDA torch 到 _deps/，**已终止并清理**（未触碰系统或他人数据）。
- **未终止或修改**任何其他用户的 GPU 进程；所有 nvidia-smi 调用均为只读查询。
- 未编造任何 benchmark / mAP / recall；无 GT 的部分一律 BLOCKED。
