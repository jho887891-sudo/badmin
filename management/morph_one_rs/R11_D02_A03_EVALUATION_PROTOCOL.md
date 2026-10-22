# R11 D02 + A03：真实视觉检测和预测的独立评测协议

> 版本：2026-10-22 · 原 SPEC：`management/morph_one_rs/R11_M1_FOLLOWUP_EXECUTION_20261022.md` · 研发负责人：ChatGPT · 状态：**CPU_CODE_VERIFIED / GPU_REAL_VIDEO_NOT_RUN**
>
> 本文档不是新任务或对 M1 的完成宣告；是已有 D02（人工 GT 的检测评测）和 A03（未来时间 GT 对齐）的**执行协议**。DeepSeek Harness 只负责在远端运行命令和返回原始数据，不能改变算法或阈值。

## 1. 为什么要单独建立评测模块？

先前的事实是 ETH 60 帧有框 59 帧，但来自训练源，不能当外部泛化 Recall；广播 `doubles0.mp4` 336 帧有框 0 帧，原因仍未确定。R11 D03 已经提供全图 / GT-free tiling 两种 YOLO 输入，见 PR #59。**有框帧占比不是 Precision，也不是 Recall**。如果全图得到 0 框而 tiling 得到 50 框，50 框也可能全部是照明或衣服上的假阳性。

因此本次实现一套完全离线、无 GPU 依赖的评测器：
- 输入 **已有 YOLO 原始 `detections.jsonl`** 和 **人工审阅 `gt.jsonl`**，一对一匹配得到 TP、FP、FN，不修改推理结果；
- 按真实 **原图 bbox** 分别统计面积等效边长（`sqrt(width_px * height_px)`）的 `<4`、`4-6`、`6-10`、`10-20`、`>=20` 桶；
- 同时报告 IoU≥0.5 与独立 **中心点距离≤3px** 两种固定口径。小目标的 IoU50 与 3px 中心匹配不等价，因此不能互换，也不得挑结果好看的那个；
- 对未来轨迹只根据 `tracks.jsonl` 的 **预测时刻** 查询人工 GT 时间戳，严格不插值、不按数组偏移猜测真实未来位置；
- 只有新的 `position_covariance_px2` 字段具有可用的 2×2 像素位置协方差时，才统计二维高斯 **联合 95% 覆盖率**（χ²₂=5.9914645）；不把历史混合单位 4×4 协方差当位置不确定度。

## 2. 人工真值输入协议 `gt.jsonl`

由视频标注人员逐帧人工核对，**每行一个真实帧**：

```json
{"frame_id":0,"timestamp_s":0.0,"status":"labeled","image_size":{"width_px":1280,"height_px":720},"objects":[{"bbox_xyxy_px":[620,300,626,306]}]}
{"frame_id":1,"timestamp_s":0.033333,"status":"labeled","image_size":{"width_px":1280,"height_px":720},"objects":[]}
{"frame_id":2,"timestamp_s":0.066667,"status":"unscorable","image_size":{"width_px":1280,"height_px":720},"objects":[]}
```

以上仅为 **SCHEMA_EXAMPLE，绝不是 doubles0.mp4 的真实人工标注**。

字段语义必须冻结：

| 记录情况 | 解释 | 会参与检测统计吗 |
|---|---|---|
| `labeled` + 非空 `objects` | 已审阅且目标在该帧可评价 | GT 正例，可以计 TP/FN |
| `labeled` + 空 `objects` | 已审阅且没有可评价目标 | GT 负例，模型框可计 FP |
| `unscorable` + 空 `objects` | 球太模糊、严重遮挡或无法判断 | 不参与 TP/FP/FN，单独计数 |
| **缺失该帧整行** | 尚未标注 / 未审阅 | 不能假定负样本，单独计 missing_frames |

禁止把未看过的全部剩余视频帧批量写成空 `objects`，以制造 0 FP 或虚假的高 Precision；同帧重复 GT、时间倒序、越界 bbox、非像素单位必须失败关闭。

**独立测试集证明**另需训练视频/session 清单、模型训练数据来源与人工审阅证据；命令行传 `--split-provenance held_out_session` 只是自我声明，因此软件明确 `generalization_claim_allowed=false`，只产生候选声明，直到独立数据泄漏审计通过。

## 3. 两套分开的评估指标

### D02 检测匹配

- `TP`：人工 GT 与 YOLO 框满足条件并完成**一对一最大匹配**；
- `FP`：在实际人工标注且可评价帧中剩余没有匹配到 GT 的模型框；
- `FN`：在实际人工标注且可评价帧中剩余没有匹配到预测框的人工 GT；
- `Precision=TP/(TP+FP)`、`Recall=TP/(TP+FN)`；分母为零时结果 `null`，不得写 1 或 0 冒充；
- 按 GT 原生尺寸分桶计算 bucket recall；未匹配预测框**没有真实尺寸标签**，不把它强行归入 GT 尺寸桶；
- 没有标注的帧或 `unscorable` 帧里的检测框计入 `unscored_detections`，不进入 precision 分母。

匹配使用 **maximum-cardinality bipartite matching**（最大一对一匹配数），而非简单 greedily 最高 IoU：两框与两个 GT 在竞争匹配时，贪心可能把能匹配成 2 个 TP 的情况错算 1 个。已在 CPU RED→GREEN 负控中验证这一点。

### A03 未来轨迹

对每条 track 在时间 `t` 给出的 horizon `h=0.3/0.5/0.8 s`：

1. 精确计算 `target_time=t+h`；
2. 在人工 GT 时间戳中寻找距离 target_time 最近且不超过指定容差 `--future-tolerance-s` 的**真实未来帧**；
3. 如果时间戳没有对应 GT 或 GT 无法判定，明确计入 `skipped_*`，不线性插值，不用未来 YOLO 检测冒充人工 GT；
4. 如果未来 GT 有多个对象且无法证明目标身份，**拒绝 oracle 目标选择**，标 `skipped_ambiguous_gt`；
5. 对有有效一对一人工 GT 的样本，报告预测中心到真实中心的 Euclidean error（px）、中位、p95/p99、有效 N；
6. 只有有效正定 xy 位置协方差才计算 `error^T * P_xy^-1 * error <= 5.9914645` 是否落入联合 95% 区域，独立报告覆盖率 N。

这是 **post-hoc evaluator**，GT 不进入 YOLO、EKF 或动作规划。像面预测不能直接变成米制 IK 输入。

## 4. 现有真实文件的执行命令

程序文件：`tools/vision/evaluate_r11.py`。只要拿到已标注的 `gt.jsonl`，无需 GPU 即可运行。

**D02：** 比较历史全图与 PR #59 分块输出时必须使用**同一份 GT**：

```bash
python tools/vision/evaluate_r11.py \
  --mode detection \
  --ground-truth /path/to/human_verified_doubles0_gt.jsonl \
  --input /path/to/R11_D03_full640_20261022/detections.jsonl \
  --expected-frames 336 \
  --centre-tolerance-px 3.0 \
  --split-provenance unknown \
  --annotation-provenance human_verified \
  --output /path/to/new_results/full640_eval.json

python tools/vision/evaluate_r11.py \
  --mode detection \
  --ground-truth /path/to/human_verified_doubles0_gt.jsonl \
  --input /path/to/R11_D03_tiled640_20261022/detections.jsonl \
  --expected-frames 336 \
  --centre-tolerance-px 3.0 \
  --split-provenance unknown \
  --annotation-provenance human_verified \
  --output /path/to/new_results/tiled640_eval.json
```

**A03：** 同样仅使用真实逐帧人工标注与旧/新 Run 的 `tracks.jsonl`：

```bash
python tools/vision/evaluate_r11.py \
  --mode prediction \
  --ground-truth /path/to/human_verified_doubles0_gt.jsonl \
  --input /path/to/R11_D03_tiled640_20261022/tracks.jsonl \
  --future-tolerance-s 0.02 \
  --output /path/to/new_results/tiled640_horizon_eval.json
```

以上 `/path/to` 是占位符，**没有标注文件时不能运行并声称有人工 GT 指标**。绝不可用模型自己输出的预测框制造 `human_verified` 标注。

## 5. 验证与技术债务

**本次会话的实际验证级别：** 单独提取源码在本地 Python 环境运行 `pytest -q tests/vision/test_r11_eval.py`，先 RED（模块不存在），实现后 GREEN，额外发现/修复贪心匹配和自称 held-out 的问题，共 **19 tests PASS**。运行了独立 CLI 的合成文件 smoke：detection、prediction 两种模式均 exit 0，正确写入输出并计算输入 SHA256；合成结果只用于验证算法，不得当作真实检测实验。

**未完成的真实验收**：
- doubles0.mp4 336 帧的真实人工标注尚未产生，无法给出 Precision/Recall；
- PR #59 尚无 GPU 全图/分块真实对照结果；
- 没有运行完整远端项目全仓 pytest/集成 YOLO；
- 视频/标签同源真实性仍需要标注负责人和文件级 SHA / session 审计；不能仅靠自行声明；
- 未来时刻的 2D 恒速度预测未证明可以击球，仍需要后续几何和物理链路。

**权限边界**：DeepSeek 仅按精确 SHA 运行、回传日志/指标和原始视频；不得改代码、调阈值、修改 GT、训练新模型或擅自将 M1 提升 PASS。GitHub PR 均 Draft，禁止自动合并受保护分支，不运行 PPO 或真机动作。


## 6. 最新 fail-closed 增量与复验

后续测试还抓到了两个缺陷，已经通过 RED→GREEN 修复并在本地复验：

- **同帧号错时钟**：检测 `frame_id` 虽相同，若预测 `timestamp_s` 与人工标签相差超过 **0.005s**（默认），拒绝计算 Precision/Recall；调用参数 `--detection-tolerance-s` 可显式设置，且报告记录真实值。这样旧图像序列错误假设 1/30 s 时，不能悄悄对齐到其它时间段。
- **超视频范围**：设置 `--expected-frames 336` 后，如模型检测声明 frame_id≥336，立即报错，而不是忽略该框并继续给出看似正常的统计。

截至该文档修订，测试为 `19 passed`，并对两种命令行模式使用合成 JSONL 做过 exit 0 的端到端 smoke。模型未在本会话重跑，广播视频的 0/336 历史数字没有改变。

**审阅限制：** 当前为单作者自审，独立远端集成验证和第三方复核仍未进行。传统 mAP/AP 需要不同 confidence 阈值及标准定义的排序评估，本工具当前只实现**指定 conf 下 IoU50 与中心距离的 TP/FP/FN/Precision/Recall**，不得把输出称为 AP50 或 mAP。
