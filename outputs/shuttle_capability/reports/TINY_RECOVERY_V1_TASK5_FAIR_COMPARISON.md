# ETH OFFICIAL BASELINE AUDIT - ETH Zurich shuttle detector vs our YOLO26-P2 (read-only)

> 自动部分（1-12 节）由 tools/eval_eth_official_baseline.py 生成；0 节结论与 13 节问答为人工撰写（工具重跑会清空）。
> 只做评估：不训练、不 fine-tune、不改 checkpoint / 测试集 / GT / evaluator，不按模型单独调阈值，不引用论文数字替代实跑。
> 这些集合是 fixed evaluation set / development holdout，不是 untouched final test。

## 0. 结论（人工）

（待填。）

## 1. ETH checkpoint 核验与兼容处理

| 项 | 值 |
|---|---|
| 路径 |  |
| 字节 |  |
| sha256 |  |
| 与记录 sha256 一致 |  |
| 与官方仓库清单一致 |  |
| inventory relpath |  |
| checkpoint 类型 |  |
| checkpoint keys | None |
| is_ultralytics_format |  |
| ckpt 内 model_name |  |
| state_dict 张量数 |  |
| nc（由 state_dict 的 cv3 分支数推断） |  |
| dfl weight shape | None |

**格式不兼容记录**：该文件是 ETH 自建训练脚本保存的 dict（含 model_state_dict 与 model_name），不是 ultralytics checkpoint。
最小兼容处理：用 ultralytics 自带 yolov8s.yaml 重建同架构（不含预训练权重），load_state_dict(strict=True) 一次性装载，
并核对由 dfl 输出推断的 nc；未改动任何权重数值、未改结构。装载后 params=None GFLOPs=n/a stride=None nc=None。

## 2. FAIRNESS AUDIT（同一 evaluator / 数据 / 口径）

| 检查项 | 状态 | 说明 |
|---|---|---|
| 同一 images | PASS | val / controlled_capability/images / challenge_test/images（同一 manifest 与冻结目录） |
| 同一 GT | PASS | 全部经 tools/eval_yolo26_v1.py:401-402 同一解码（无 letterbox） |
| 同一 imgsz=1024 | PASS | 所有模型 imgsz=1024（ETH 官方训练即 1024） |
| 同一 IoU evaluator | PASS | 同一 match_greedy + ap_for_gt_set（all-point PR，GT 加权） |
| 同一 confidence policy | PASS | 工作点 conf=0.25；AP 曲线下限 0.001（对所有模型相同） |
| 同一 NMS policy | PASS | iou=0.7, max_det=300, rect=False |
| 同一 size bucket | PASS | equiv_size_640 半开区间（与 LOCALIZATION_AUDIT 相同） |
| 同一 coordinate system | PASS | 全部在原始图像像素系；ETH-style 指标另在 letterbox 系单独报告 |
| 同一 hardware | PASS | n/a |
| 同一 precision | PASS | fp32（未用 half / 量化） |
| 同一 latency protocol | PASS | batch=1 fp32，warmup=50，measured=200 |

阈值公平性声明：没有为任何模型单独调 confidence。主比较一律 conf=0.25（既有工作点）与 AP 曲线下限 0.001；
ETH 官方工作点 conf=0.5 只用于第 8 节 ETH-style 中心距指标，且对三个模型一并给出，不用于主比较。

## 3. DATA LEAKAGE AUDIT

判据（文件级、可复核）：ETH 官方每档 yaml 写明 train: images/train；其 config.json 的 data.train 列出 12 个 location，
diff_levels.train=[easy, medium]。位于 <location>_<easy|medium>/images/train 且 location 在该列表内的文件 = A（ETH 训练帧）；
同 location 但非训练档 = B；其它 ETH 基准档（如 coco_val_easy）或完全在 ETH 数据集之外 = C。

| model | set | leakage class | images | GT | TP@0.5 | Recall@0.5 |
|---|---|---|---|---|---|---|

背景级泄漏探针（合成集使用的真实背景图来源）：

- controlled_capability：resolved=30 unresolved=0 classes={'C_outside_eth_dataset': 30}
- challenge_test：resolved=33 unresolved=0 classes={'C_outside_eth_dataset': 33}

eth_unseen 子集 = 剔除 A 类后的图像（B + C），在 eth_vs_ours.csv 中以 set 后缀 eth_unseen 行给出。

## 4. 主指标（conf=0.25；AP 曲线 conf 0.001）

| model | set | GT | TP | FP | FN | P | R | F1 | AP50 | AP75 | AP90 | AP95 | mAP50-95 | FP/img |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| eth_tiny_recovery_v1_best | val | 3260 | 1789 | 33 | 1471 | 0.9819 | 0.5488 | 0.7041 | 0.7197 | 0.5956 | 0.1695 | 0.0211 | 0.5114 | 0.0075 |
| eth_tiny_recovery_v1_best | val|eth_unseen | 495 | 83 | 16 | 412 | 0.8384 | 0.1677 | 0.2795 | 0.2852 | 0.1222 | 0.0061 | 0.0000 | 0.1388 | 0.0097 |
| eth_tiny_recovery_v1_best | controlled_capability/images | 2070 | 5 | 3 | 2065 | 0.6250 | 0.0024 | 0.0048 | 0.0433 | 0.0382 | 0.0148 | 0.0005 | 0.0324 | 0.0014 |
| eth_tiny_recovery_v1_best | challenge_test/images | 194 | 0 | 0 | 194 | n/a | 0.0000 | n/a | 0.0412 | 0.0284 | 0.0026 | 0.0000 | 0.0227 | 0.0000 |

## 5. AP by IoU（0.50:0.05:0.95）

| model | set | AP50 | AP55 | AP60 | AP65 | AP70 | AP75 | AP80 | AP85 | AP90 | AP95 | mAP50-95 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| eth_tiny_recovery_v1_best | val | 0.7197 | 0.7114 | 0.6991 | 0.6800 | 0.6528 | 0.5956 | 0.5039 | 0.3613 | 0.1695 | 0.0211 | 0.5114 |
| eth_tiny_recovery_v1_best | val|eth_unseen | 0.2852 | 0.2587 | 0.2389 | 0.2073 | 0.1753 | 0.1222 | 0.0677 | 0.0269 | 0.0061 | 0.0000 | 0.1388 |
| eth_tiny_recovery_v1_best | controlled_capability/images | 0.0433 | 0.0428 | 0.0423 | 0.0417 | 0.0404 | 0.0382 | 0.0335 | 0.0268 | 0.0148 | 0.0005 | 0.0324 |
| eth_tiny_recovery_v1_best | challenge_test/images | 0.0412 | 0.0361 | 0.0309 | 0.0284 | 0.0284 | 0.0284 | 0.0180 | 0.0129 | 0.0026 | 0.0000 | 0.0227 |

## 6. 尺寸分桶与超小目标召回

| model | set | bucket | GT | TP | FN | Recall | AP50 | AP75 | AP90 | mAP50-95 | IoU med |
|---|---|---|---|---|---|---|---|---|---|---|---|
| eth_tiny_recovery_v1_best | val | <4 | 24 | 0 | 24 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | n/a |
| eth_tiny_recovery_v1_best | val | 4-6 | 71 | 15 | 56 | 0.2113 | 0.3310 | 0.0986 | 0.0000 | 0.1655 | 0.7449 |
| eth_tiny_recovery_v1_best | val | 6-8 | 251 | 96 | 155 | 0.3825 | 0.6454 | 0.5787 | 0.0671 | 0.4486 | 0.8359 |
| eth_tiny_recovery_v1_best | val | 8-12 | 1724 | 1140 | 584 | 0.6613 | 0.8199 | 0.7065 | 0.1968 | 0.5965 | 0.8639 |
| eth_tiny_recovery_v1_best | val | 12-16 | 755 | 376 | 379 | 0.4980 | 0.6695 | 0.5190 | 0.1929 | 0.4677 | 0.8643 |
| eth_tiny_recovery_v1_best | val | 16-24 | 343 | 144 | 199 | 0.4198 | 0.6195 | 0.4563 | 0.1286 | 0.4101 | 0.8330 |
| eth_tiny_recovery_v1_best | val | 24-32 | 52 | 10 | 42 | 0.1923 | 0.3234 | 0.2593 | 0.0910 | 0.2272 | 0.8246 |
| eth_tiny_recovery_v1_best | val | 32-64 | 29 | 8 | 21 | 0.2759 | 0.4253 | 0.3276 | 0.0690 | 0.3098 | 0.8899 |
| eth_tiny_recovery_v1_best | val | >64 | 11 | 0 | 11 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | n/a |
| eth_tiny_recovery_v1_best | controlled_capability/images | <4 | 160 | 0 | 160 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | n/a |
| eth_tiny_recovery_v1_best | controlled_capability/images | 4-6 | 106 | 0 | 106 | 0.0000 | 0.0019 | 0.0000 | 0.0000 | 0.0006 | n/a |
| eth_tiny_recovery_v1_best | controlled_capability/images | 6-8 | 54 | 0 | 54 | 0.0000 | 0.0185 | 0.0185 | 0.0000 | 0.0130 | n/a |
| eth_tiny_recovery_v1_best | controlled_capability/images | 8-12 | 405 | 0 | 405 | 0.0000 | 0.0588 | 0.0506 | 0.0074 | 0.0409 | n/a |
| eth_tiny_recovery_v1_best | controlled_capability/images | 12-16 | 825 | 2 | 823 | 0.0024 | 0.0601 | 0.0516 | 0.0201 | 0.0443 | 0.9139 |
| eth_tiny_recovery_v1_best | controlled_capability/images | 16-24 | 106 | 3 | 103 | 0.0283 | 0.1226 | 0.1226 | 0.0943 | 0.1085 | 0.9273 |
| eth_tiny_recovery_v1_best | controlled_capability/images | 24-32 | 54 | 0 | 54 | 0.0000 | 0.0370 | 0.0370 | 0.0185 | 0.0315 | n/a |
| eth_tiny_recovery_v1_best | controlled_capability/images | 32-64 | 120 | 0 | 120 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | n/a |
| eth_tiny_recovery_v1_best | controlled_capability/images | >64 | 240 | 0 | 240 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | n/a |
| eth_tiny_recovery_v1_best | challenge_test/images | <4 | 40 | 0 | 40 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | n/a |
| eth_tiny_recovery_v1_best | challenge_test/images | 4-6 | 0 | 0 | 0 | n/a | n/a | n/a | n/a | n/a | n/a |
| eth_tiny_recovery_v1_best | challenge_test/images | 6-8 | 66 | 0 | 66 | 0.0000 | 0.0455 | 0.0379 | 0.0076 | 0.0288 | n/a |
| eth_tiny_recovery_v1_best | challenge_test/images | 8-12 | 20 | 0 | 20 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | n/a |
| eth_tiny_recovery_v1_best | challenge_test/images | 12-16 | 44 | 0 | 44 | 0.0000 | 0.1136 | 0.0682 | 0.0000 | 0.0568 | n/a |
| eth_tiny_recovery_v1_best | challenge_test/images | 16-24 | 0 | 0 | 0 | n/a | n/a | n/a | n/a | n/a | n/a |
| eth_tiny_recovery_v1_best | challenge_test/images | 24-32 | 0 | 0 | 0 | n/a | n/a | n/a | n/a | n/a | n/a |
| eth_tiny_recovery_v1_best | challenge_test/images | 32-64 | 0 | 0 | 0 | n/a | n/a | n/a | n/a | n/a | n/a |
| eth_tiny_recovery_v1_best | challenge_test/images | >64 | 24 | 0 | 24 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | n/a |

聚合召回（指定口径）：

| model | set | Recall_<4 | Recall_4_6 | Recall_6_8 | Recall_<8 | Recall_8_12 | Recall_12_16 | Recall_8_16 | Recall_16_32 | Recall_32_64 | Recall_>64 |
|---|---|---|---|---|---|---|---|---|---|---|---|
| eth_tiny_recovery_v1_best | val | 0.0000 | 0.2113 | 0.3825 | 0.3208 | 0.6613 | 0.4980 | 0.6115 | 0.3899 | 0.2759 | 0.0000 |
| eth_tiny_recovery_v1_best | val|eth_unseen | 0.0000 | 0.0000 | 0.1364 | 0.0571 | 0.2778 | 0.1727 | 0.2379 | 0.1039 | 0.0000 | 0.0000 |
| eth_tiny_recovery_v1_best | controlled_capability/images | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0024 | 0.0016 | 0.0187 | 0.0000 | 0.0000 |
| eth_tiny_recovery_v1_best | challenge_test/images | 0.0000 | n/a | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | n/a | n/a | 0.0000 |

## 7. 定位质量（TP 集合）与 height_ratio 检查

| model | set | n_tp | IoU mean | IoU med | IoU P10 | IoU P90 | center px640 med | norm center med | w_ratio mean | h_ratio mean | h_ratio med | area_ratio med | w_rel med | h_rel med |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| eth_tiny_recovery_v1_best | val | 1789 | 0.8388 | 0.8574 | 0.7236 | 0.9279 | 0.418 | 0.0398 | 0.9305 | 0.9949 | 0.9921 | 0.9313 | 0.0644 | 0.0387 |
| eth_tiny_recovery_v1_best | val | 83 | 0.7190 | 0.7343 | 0.5671 | 0.8416 | 0.860 | 0.0875 | 0.8433 | 1.0176 | 1.0119 | 0.8279 | 0.1906 | 0.0632 |
| eth_tiny_recovery_v1_best | controlled_capability/images | 5 | 0.9204 | 0.9168 | 0.9090 | 0.9343 | 0.452 | 0.0275 | 0.9743 | 1.0190 | 1.0086 | 1.0046 | 0.0288 | 0.0086 |
| eth_tiny_recovery_v1_best | challenge_test/images | 0 | n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a |

height_ratio 标记规则：ETH 若也接近 1.09 则标 POSSIBLE_SHARED_DATA_OR_EVALUATION_EFFECT；
ETH 不偏而我们的 A/B 偏则标 POSSIBLE_MODEL_OR_TRAINING_SPECIFIC_EFFECT（不下因果断言）：

| model | set | h_ratio mean | w_ratio mean | flag |
|---|---|---|---|---|
| eth_tiny_recovery_v1_best | val | 0.9949 | 0.9305 | not_flagged |
| eth_tiny_recovery_v1_best | val | 1.0176 | 0.8433 | not_flagged |
| eth_tiny_recovery_v1_best | controlled_capability/images | 1.0190 | 0.9743 | not_flagged |

## 8. ETH-style center-distance 指标（单独口径，不等于 IoU-based 指标）

**ETH-style F1 != IoU-based F1。** 其定义（src/shuttletrack/utils.py:297-344）：每图只取 top-1 检测框（max_det=1）、
每图只取 1 个 GT（首个实例）、距离在 letterbox 后的网络输入系内度量、dist < 25 px（config.json: dist_threshold=25.0, 
confidence=0.5）、FP 至多 1 个/图。

| model | set | subset | conf | dist_thr | frame | TP | FP | FN | P | R | F1 | variant |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| eth_tiny_recovery_v1_best | val | all | 0.50 | 25.000 | content | 1483 | 4 | 1774 | 0.9973 | 0.4553 | 0.6252 | eth_official_default |
| eth_tiny_recovery_v1_best | val | all | 0.50 | 25.000 | original | 1483 | 4 | 1774 | 0.9973 | 0.4553 | 0.6252 | eth_default_original_px |
| eth_tiny_recovery_v1_best | val | all | 0.50 | 15.625 | content | 1483 | 4 | 1774 | 0.9973 | 0.4553 | 0.6252 | eth_default_640_equiv |
| eth_tiny_recovery_v1_best | val | all | 0.25 | 25.000 | content | 1801 | 16 | 1445 | 0.9912 | 0.5548 | 0.7114 | our_conf_0.25_content |
| eth_tiny_recovery_v1_best | controlled_capability/images | all | 0.50 | 25.000 | content | 5 | 0 | 2065 | 1.0000 | 0.0024 | 0.0048 | eth_official_default |
| eth_tiny_recovery_v1_best | controlled_capability/images | all | 0.50 | 25.000 | original | 4 | 1 | 2065 | 0.8000 | 0.0019 | 0.0039 | eth_default_original_px |
| eth_tiny_recovery_v1_best | controlled_capability/images | all | 0.50 | 15.625 | content | 4 | 1 | 2065 | 0.8000 | 0.0019 | 0.0039 | eth_default_640_equiv |
| eth_tiny_recovery_v1_best | controlled_capability/images | all | 0.25 | 25.000 | content | 8 | 0 | 2062 | 1.0000 | 0.0039 | 0.0077 | our_conf_0.25_content |
| eth_tiny_recovery_v1_best | challenge_test/images | all | 0.50 | 25.000 | content | 0 | 0 | 194 | 1.0000 | 0.0000 | 0.0000 | eth_official_default |
| eth_tiny_recovery_v1_best | challenge_test/images | all | 0.50 | 25.000 | original | 0 | 0 | 194 | 1.0000 | 0.0000 | 0.0000 | eth_default_original_px |
| eth_tiny_recovery_v1_best | challenge_test/images | all | 0.50 | 15.625 | content | 0 | 0 | 194 | 1.0000 | 0.0000 | 0.0000 | eth_default_640_equiv |
| eth_tiny_recovery_v1_best | challenge_test/images | all | 0.25 | 25.000 | content | 0 | 0 | 194 | 1.0000 | 0.0000 | 0.0000 | our_conf_0.25_content |

对照：IoU>=0.5 的 P/R/F1 见第 4 节的 Precision@op 与 Recall@0.5。两者差异来自口径，不是模型差异。

## 9. False Positive（无目标集合）

| model | set | conf>= | images | FP total | FP/img | images with FP | image FP rate | max conf FP |
|---|---|---|---|---|---|---|---|---|

## 10. 速度与模型成本（同硬件、batch=1、fp32）

| model | params | GFLOPs | latency mean ms | median | P90 | P95 | FPS | e2e mean ms | preprocess ms | inference ms | postprocess ms | VRAM peak MB |
|---|---|---|---|---|---|---|---|---|---|---|---|---|

协议：n/a；端到端另计时（含 preprocess/inference/postprocess），组件时间为框架 results.speed 均值，分开报告。

## 11. 模型成本与参数对照

| model | params | GFLOPs | nc | stride | latency mean ms | FPS | VRAM MB |
|---|---|---|---|---|---|---|---|
| eth_official |  | n/a |  | None | n/a | n/a | n/a |
| eth_tiny_recovery_v1_best | 9948638 | 59.19 | 1 | [8, 16, 32] | n/a | n/a | n/a |

## 12. 复现与产物

~~~
python tools/eval_eth_official_baseline.py --repo . --models eth_tiny_recovery_v1_best --out-dir _scratch_eth_only_v1_eval\tiny_fair --examples-dir _scratch_eth_only_v1_eval\tiny_fair_examples --no-examples --no-latency --report _scratch_eth_only_v1_eval\tiny_fair\TINY_FAIR.md
~~~

- 预测缓存 _scratch_localization_audit/loc_<model>_<set>.json（imgsz 1024 / conf 0.001 / iou 0.7 / max_det 300 / rect=False）。
- 示例图 0 张；分歧图 0 张。
- 旧结果未覆盖：Stage A/B 既有 metrics 与 reports 原样保留。

## 13. 问答（Q1-Q10，人工）

（待填。）

