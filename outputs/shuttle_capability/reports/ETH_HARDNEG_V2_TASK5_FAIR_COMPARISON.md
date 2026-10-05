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
| eth_real_hardneg_v2_best | val | 3260 | 1935 | 29 | 1325 | 0.9852 | 0.5936 | 0.7408 | 0.7715 | 0.6220 | 0.1475 | 0.0206 | 0.5349 | 0.0066 |
| eth_real_hardneg_v2_best | val|eth_unseen | 495 | 101 | 14 | 394 | 0.8783 | 0.2040 | 0.3311 | 0.3710 | 0.1753 | 0.0192 | 0.0020 | 0.1897 | 0.0085 |
| eth_real_hardneg_v2_best | controlled_capability/images | 2070 | 7 | 3 | 2063 | 0.7000 | 0.0034 | 0.0067 | 0.0345 | 0.0267 | 0.0082 | 0.0005 | 0.0241 | 0.0014 |
| eth_real_hardneg_v2_best | challenge_test/images | 194 | 0 | 0 | 194 | n/a | 0.0000 | n/a | 0.0438 | 0.0232 | 0.0052 | 0.0000 | 0.0245 | 0.0000 |

## 5. AP by IoU（0.50:0.05:0.95）

| model | set | AP50 | AP55 | AP60 | AP65 | AP70 | AP75 | AP80 | AP85 | AP90 | AP95 | mAP50-95 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| eth_real_hardneg_v2_best | val | 0.7715 | 0.7616 | 0.7484 | 0.7225 | 0.6831 | 0.6220 | 0.5121 | 0.3593 | 0.1475 | 0.0206 | 0.5349 |
| eth_real_hardneg_v2_best | val|eth_unseen | 0.3710 | 0.3457 | 0.3165 | 0.2726 | 0.2255 | 0.1753 | 0.1063 | 0.0632 | 0.0192 | 0.0020 | 0.1897 |
| eth_real_hardneg_v2_best | controlled_capability/images | 0.0345 | 0.0340 | 0.0330 | 0.0318 | 0.0306 | 0.0267 | 0.0232 | 0.0181 | 0.0082 | 0.0005 | 0.0241 |
| eth_real_hardneg_v2_best | challenge_test/images | 0.0438 | 0.0438 | 0.0361 | 0.0284 | 0.0232 | 0.0232 | 0.0232 | 0.0180 | 0.0052 | 0.0000 | 0.0245 |

## 6. 尺寸分桶与超小目标召回

| model | set | bucket | GT | TP | FN | Recall | AP50 | AP75 | AP90 | mAP50-95 | IoU med |
|---|---|---|---|---|---|---|---|---|---|---|---|
| eth_real_hardneg_v2_best | val | <4 | 24 | 0 | 24 | 0.0000 | 0.0417 | 0.0000 | 0.0000 | 0.0083 | n/a |
| eth_real_hardneg_v2_best | val | 4-6 | 71 | 13 | 58 | 0.1831 | 0.4155 | 0.1056 | 0.0141 | 0.1993 | 0.7286 |
| eth_real_hardneg_v2_best | val | 6-8 | 251 | 114 | 137 | 0.4542 | 0.7626 | 0.5986 | 0.0558 | 0.4978 | 0.8249 |
| eth_real_hardneg_v2_best | val | 8-12 | 1724 | 1253 | 471 | 0.7268 | 0.8635 | 0.7400 | 0.1564 | 0.6164 | 0.8568 |
| eth_real_hardneg_v2_best | val | 12-16 | 755 | 393 | 362 | 0.5205 | 0.7100 | 0.5385 | 0.1875 | 0.4863 | 0.8523 |
| eth_real_hardneg_v2_best | val | 16-24 | 343 | 144 | 199 | 0.4198 | 0.6595 | 0.4400 | 0.1339 | 0.4210 | 0.8376 |
| eth_real_hardneg_v2_best | val | 24-32 | 52 | 9 | 43 | 0.1731 | 0.4173 | 0.3356 | 0.0913 | 0.2824 | 0.8183 |
| eth_real_hardneg_v2_best | val | 32-64 | 29 | 9 | 20 | 0.3103 | 0.6707 | 0.6293 | 0.1379 | 0.5062 | 0.8759 |
| eth_real_hardneg_v2_best | val | >64 | 11 | 0 | 11 | 0.0000 | 0.0909 | 0.0909 | 0.0000 | 0.0727 | n/a |
| eth_real_hardneg_v2_best | controlled_capability/images | <4 | 160 | 0 | 160 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | n/a |
| eth_real_hardneg_v2_best | controlled_capability/images | 4-6 | 106 | 0 | 106 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | n/a |
| eth_real_hardneg_v2_best | controlled_capability/images | 6-8 | 54 | 0 | 54 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | n/a |
| eth_real_hardneg_v2_best | controlled_capability/images | 8-12 | 405 | 1 | 404 | 0.0025 | 0.0329 | 0.0218 | 0.0025 | 0.0226 | 0.7176 |
| eth_real_hardneg_v2_best | controlled_capability/images | 12-16 | 825 | 5 | 820 | 0.0061 | 0.0582 | 0.0442 | 0.0109 | 0.0387 | 0.9104 |
| eth_real_hardneg_v2_best | controlled_capability/images | 16-24 | 106 | 1 | 105 | 0.0094 | 0.0943 | 0.0943 | 0.0660 | 0.0821 | 0.9031 |
| eth_real_hardneg_v2_best | controlled_capability/images | 24-32 | 54 | 0 | 54 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | n/a |
| eth_real_hardneg_v2_best | controlled_capability/images | 32-64 | 120 | 0 | 120 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | n/a |
| eth_real_hardneg_v2_best | controlled_capability/images | >64 | 240 | 0 | 240 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | n/a |
| eth_real_hardneg_v2_best | challenge_test/images | <4 | 40 | 0 | 40 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | n/a |
| eth_real_hardneg_v2_best | challenge_test/images | 4-6 | 0 | 0 | 0 | n/a | n/a | n/a | n/a | n/a | n/a |
| eth_real_hardneg_v2_best | challenge_test/images | 6-8 | 66 | 0 | 66 | 0.0000 | 0.0379 | 0.0152 | 0.0000 | 0.0189 | n/a |
| eth_real_hardneg_v2_best | challenge_test/images | 8-12 | 20 | 0 | 20 | 0.0000 | 0.0500 | 0.0500 | 0.0500 | 0.0450 | n/a |
| eth_real_hardneg_v2_best | challenge_test/images | 12-16 | 44 | 0 | 44 | 0.0000 | 0.1136 | 0.0568 | 0.0000 | 0.0591 | n/a |
| eth_real_hardneg_v2_best | challenge_test/images | 16-24 | 0 | 0 | 0 | n/a | n/a | n/a | n/a | n/a | n/a |
| eth_real_hardneg_v2_best | challenge_test/images | 24-32 | 0 | 0 | 0 | n/a | n/a | n/a | n/a | n/a | n/a |
| eth_real_hardneg_v2_best | challenge_test/images | 32-64 | 0 | 0 | 0 | n/a | n/a | n/a | n/a | n/a | n/a |
| eth_real_hardneg_v2_best | challenge_test/images | >64 | 24 | 0 | 24 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | n/a |

聚合召回（指定口径）：

| model | set | Recall_<4 | Recall_4_6 | Recall_6_8 | Recall_<8 | Recall_8_12 | Recall_12_16 | Recall_8_16 | Recall_16_32 | Recall_32_64 | Recall_>64 |
|---|---|---|---|---|---|---|---|---|---|---|---|
| eth_real_hardneg_v2_best | val | 0.0000 | 0.1831 | 0.4542 | 0.3671 | 0.7268 | 0.5205 | 0.6640 | 0.3873 | 0.3103 | 0.0000 |
| eth_real_hardneg_v2_best | val|eth_unseen | 0.0000 | 0.0000 | 0.0909 | 0.0381 | 0.3111 | 0.2364 | 0.2828 | 0.1818 | 0.0833 | 0.0000 |
| eth_real_hardneg_v2_best | controlled_capability/images | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0025 | 0.0061 | 0.0049 | 0.0063 | 0.0000 | 0.0000 |
| eth_real_hardneg_v2_best | challenge_test/images | 0.0000 | n/a | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | n/a | n/a | 0.0000 |

## 7. 定位质量（TP 集合）与 height_ratio 检查

| model | set | n_tp | IoU mean | IoU med | IoU P10 | IoU P90 | center px640 med | norm center med | w_ratio mean | h_ratio mean | h_ratio med | area_ratio med | w_rel med | h_rel med |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| eth_real_hardneg_v2_best | val | 1935 | 0.8342 | 0.8527 | 0.7170 | 0.9253 | 0.470 | 0.0448 | 0.9295 | 0.9977 | 0.9960 | 0.9325 | 0.0668 | 0.0392 |
| eth_real_hardneg_v2_best | val | 101 | 0.7337 | 0.7506 | 0.6017 | 0.8600 | 0.831 | 0.0744 | 0.8495 | 1.0105 | 1.0124 | 0.8356 | 0.1759 | 0.0560 |
| eth_real_hardneg_v2_best | controlled_capability/images | 7 | 0.8847 | 0.9031 | 0.8217 | 0.9357 | 0.510 | 0.0365 | 1.0393 | 0.9936 | 1.0030 | 1.0400 | 0.0215 | 0.0170 |
| eth_real_hardneg_v2_best | challenge_test/images | 0 | n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a |

height_ratio 标记规则：ETH 若也接近 1.09 则标 POSSIBLE_SHARED_DATA_OR_EVALUATION_EFFECT；
ETH 不偏而我们的 A/B 偏则标 POSSIBLE_MODEL_OR_TRAINING_SPECIFIC_EFFECT（不下因果断言）：

| model | set | h_ratio mean | w_ratio mean | flag |
|---|---|---|---|---|
| eth_real_hardneg_v2_best | val | 0.9977 | 0.9295 | not_flagged |
| eth_real_hardneg_v2_best | val | 1.0105 | 0.8495 | not_flagged |
| eth_real_hardneg_v2_best | controlled_capability/images | 0.9936 | 1.0393 | not_flagged |

## 8. ETH-style center-distance 指标（单独口径，不等于 IoU-based 指标）

**ETH-style F1 != IoU-based F1。** 其定义（src/shuttletrack/utils.py:297-344）：每图只取 top-1 检测框（max_det=1）、
每图只取 1 个 GT（首个实例）、距离在 letterbox 后的网络输入系内度量、dist < 25 px（config.json: dist_threshold=25.0, 
confidence=0.5）、FP 至多 1 个/图。

| model | set | subset | conf | dist_thr | frame | TP | FP | FN | P | R | F1 | variant |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| eth_real_hardneg_v2_best | val | all | 0.50 | 25.000 | content | 1676 | 0 | 1584 | 1.0000 | 0.5141 | 0.6791 | eth_official_default |
| eth_real_hardneg_v2_best | val | all | 0.50 | 25.000 | original | 1676 | 0 | 1584 | 1.0000 | 0.5141 | 0.6791 | eth_default_original_px |
| eth_real_hardneg_v2_best | val | all | 0.50 | 15.625 | content | 1676 | 0 | 1584 | 1.0000 | 0.5141 | 0.6791 | eth_default_640_equiv |
| eth_real_hardneg_v2_best | val | all | 0.25 | 25.000 | content | 1955 | 6 | 1303 | 0.9969 | 0.6001 | 0.7492 | our_conf_0.25_content |
| eth_real_hardneg_v2_best | controlled_capability/images | all | 0.50 | 25.000 | content | 1 | 0 | 2069 | 1.0000 | 0.0005 | 0.0010 | eth_official_default |
| eth_real_hardneg_v2_best | controlled_capability/images | all | 0.50 | 25.000 | original | 1 | 0 | 2069 | 1.0000 | 0.0005 | 0.0010 | eth_default_original_px |
| eth_real_hardneg_v2_best | controlled_capability/images | all | 0.50 | 15.625 | content | 1 | 0 | 2069 | 1.0000 | 0.0005 | 0.0010 | eth_default_640_equiv |
| eth_real_hardneg_v2_best | controlled_capability/images | all | 0.25 | 25.000 | content | 9 | 1 | 2060 | 0.9000 | 0.0043 | 0.0087 | our_conf_0.25_content |
| eth_real_hardneg_v2_best | challenge_test/images | all | 0.50 | 25.000 | content | 0 | 0 | 194 | 1.0000 | 0.0000 | 0.0000 | eth_official_default |
| eth_real_hardneg_v2_best | challenge_test/images | all | 0.50 | 25.000 | original | 0 | 0 | 194 | 1.0000 | 0.0000 | 0.0000 | eth_default_original_px |
| eth_real_hardneg_v2_best | challenge_test/images | all | 0.50 | 15.625 | content | 0 | 0 | 194 | 1.0000 | 0.0000 | 0.0000 | eth_default_640_equiv |
| eth_real_hardneg_v2_best | challenge_test/images | all | 0.25 | 25.000 | content | 0 | 0 | 194 | 1.0000 | 0.0000 | 0.0000 | our_conf_0.25_content |

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
| eth_real_hardneg_v2_best | 9948638 | 59.19 | 1 | [8, 16, 32] | n/a | n/a | n/a |

## 12. 复现与产物

~~~
python tools/eval_eth_official_baseline.py --repo . --models eth_only_v1_best,eth_real_hardneg_v2_best --out-dir _scratch_eth_only_v1_eval\fair_v2 --examples-dir _scratch_eth_only_v1_eval\fair_v2_examples --no-examples --no-latency --report _scratch_eth_only_v1_eval\fair_v2\FAIR_V1_VS_V2.md
~~~

- 预测缓存 _scratch_localization_audit/loc_<model>_<set>.json（imgsz 1024 / conf 0.001 / iou 0.7 / max_det 300 / rect=False）。
- 示例图 0 张；分歧图 0 张。
- 旧结果未覆盖：Stage A/B 既有 metrics 与 reports 原样保留。

## 13. 问答（Q1-Q10，人工）

（待填。）

