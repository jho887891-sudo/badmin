# ETH OFFICIAL BASELINE AUDIT - ETH Zurich shuttle detector vs our YOLO26-P2 (read-only)

> 自动部分（1-12 节）由 tools/eval_eth_official_baseline.py 生成；0 节结论与 13 节问答为人工撰写（工具重跑会清空）。
> 只做评估：不训练、不 fine-tune、不改 checkpoint / 测试集 / GT / evaluator，不按模型单独调阈值，不引用论文数字替代实跑。
> 这些集合是 fixed evaluation set / development holdout，不是 untouched final test。

## 0. 结论（人工）

（待填。）

## 1. ETH checkpoint 核验与兼容处理

| 项 | 值 |
|---|---|
| 路径 | _scratch_eth_official\best.pt |
| 字节 | 134312133 |
| sha256 | f1aea7dec784a24d53d475f89510ef45fc13255df6298c6a0ea2fbd4419c7d6d |
| 与记录 sha256 一致 | True |
| 与官方仓库清单一致 | True |
| inventory relpath | _scratch_eth_shuttle/code/shuttle_detection/runs/final-model/best.pt |
| checkpoint 类型 | dict |
| checkpoint keys | ['epoch', 'model_name', 'model_state_dict', 'optimizer_state_dict', 'train_loss', 'val_f1', 'val_loss'] |
| is_ultralytics_format | False |
| ckpt 内 model_name | yolov8s |
| state_dict 张量数 | 355 |
| nc（由 state_dict 的 cv3 分支数推断） | 3 |
| dfl weight shape | [1, 16, 1, 1] |

**格式不兼容记录**：该文件是 ETH 自建训练脚本保存的 dict（含 model_state_dict 与 model_name），不是 ultralytics checkpoint。
最小兼容处理：用 ultralytics 自带 yolov8s.yaml 重建同架构（不含预训练权重），load_state_dict(strict=True) 一次性装载，
并核对由 dfl 输出推断的 nc；未改动任何权重数值、未改结构。装载后 params=11166560 GFLOPs=73.76 stride=[8, 16, 32] nc=80。

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
| eth_official | val | C_outside_eth_dataset | 150 | 120 | 4 | 0.0333 |
| eth_official | val | C_eth_benchmark_other_split | 1235 | 112 | 16 | 0.1429 |
| eth_official | val | A_eth_trained_folder | 2765 | 2765 | 2759 | 0.9978 |
| eth_official | val | B_same_location_not_trained | 263 | 263 | 249 | 0.9468 |
| eth_official | controlled_capability/images | C_outside_eth_dataset | 2070 | 2070 | 5 | 0.0024 |
| eth_official | challenge_test/images | C_outside_eth_dataset | 227 | 194 | 0 | 0.0000 |

背景级泄漏探针（合成集使用的真实背景图来源）：

- controlled_capability：resolved=30 unresolved=0 classes={'C_outside_eth_dataset': 30}
- challenge_test：resolved=33 unresolved=0 classes={'C_outside_eth_dataset': 33}

eth_unseen 子集 = 剔除 A 类后的图像（B + C），在 eth_vs_ours.csv 中以 set 后缀 eth_unseen 行给出。

## 4. 主指标（conf=0.25；AP 曲线 conf 0.001）

| model | set | GT | TP | FP | FN | P | R | F1 | AP50 | AP75 | AP90 | AP95 | mAP50-95 | FP/img |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| eth_official | val | 3260 | 3028 | 68 | 232 | 0.9780 | 0.9288 | 0.9528 | 0.9396 | 0.9021 | 0.4948 | 0.1107 | 0.7767 | 0.0154 |
| eth_official | val|eth_unseen | 495 | 269 | 63 | 226 | 0.8102 | 0.5434 | 0.6505 | 0.6133 | 0.4992 | 0.1091 | 0.0141 | 0.4261 | 0.0382 |
| eth_official | controlled_capability/images | 2070 | 5 | 103 | 2065 | 0.0463 | 0.0024 | 0.0046 | 0.0175 | 0.0143 | 0.0029 | 0.0000 | 0.0124 | 0.0498 |
| eth_official | challenge_test/images | 194 | 0 | 5 | 194 | 0.0000 | 0.0000 | n/a | 0.0103 | 0.0052 | 0.0052 | 0.0000 | 0.0072 | 0.0220 |
| eth_only_v1_best | val | 3260 | 1809 | 46 | 1451 | 0.9752 | 0.5549 | 0.7073 | 0.7638 | 0.6321 | 0.1642 | 0.0190 | 0.5391 | 0.0104 |
| eth_only_v1_best | val|eth_unseen | 495 | 81 | 37 | 414 | 0.6864 | 0.1636 | 0.2643 | 0.3510 | 0.1785 | 0.0177 | 0.0020 | 0.1909 | 0.0225 |
| eth_only_v1_best | controlled_capability/images | 2070 | 11 | 37 | 2059 | 0.2292 | 0.0053 | 0.0104 | 0.0669 | 0.0499 | 0.0141 | 0.0021 | 0.0455 | 0.0179 |
| eth_only_v1_best | challenge_test/images | 194 | 1 | 10 | 193 | 0.0909 | 0.0052 | 0.0098 | 0.0934 | 0.0567 | 0.0052 | 0.0000 | 0.0535 | 0.0441 |

## 5. AP by IoU（0.50:0.05:0.95）

| model | set | AP50 | AP55 | AP60 | AP65 | AP70 | AP75 | AP80 | AP85 | AP90 | AP95 | mAP50-95 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| eth_official | val | 0.9396 | 0.9387 | 0.9369 | 0.9324 | 0.9235 | 0.9021 | 0.8541 | 0.7340 | 0.4948 | 0.1107 | 0.7767 |
| eth_official | val|eth_unseen | 0.6133 | 0.6093 | 0.6042 | 0.5870 | 0.5557 | 0.4992 | 0.4143 | 0.2544 | 0.1091 | 0.0141 | 0.4261 |
| eth_official | controlled_capability/images | 0.0175 | 0.0175 | 0.0175 | 0.0173 | 0.0152 | 0.0143 | 0.0126 | 0.0087 | 0.0029 | 0.0000 | 0.0124 |
| eth_official | challenge_test/images | 0.0103 | 0.0103 | 0.0103 | 0.0103 | 0.0103 | 0.0052 | 0.0052 | 0.0052 | 0.0052 | 0.0000 | 0.0072 |
| eth_only_v1_best | val | 0.7638 | 0.7584 | 0.7459 | 0.7284 | 0.6933 | 0.6321 | 0.5301 | 0.3553 | 0.1642 | 0.0190 | 0.5391 |
| eth_only_v1_best | val|eth_unseen | 0.3510 | 0.3405 | 0.3171 | 0.2846 | 0.2451 | 0.1785 | 0.1139 | 0.0583 | 0.0177 | 0.0020 | 0.1909 |
| eth_only_v1_best | controlled_capability/images | 0.0669 | 0.0653 | 0.0639 | 0.0611 | 0.0578 | 0.0499 | 0.0418 | 0.0326 | 0.0141 | 0.0021 | 0.0455 |
| eth_only_v1_best | challenge_test/images | 0.0934 | 0.0908 | 0.0839 | 0.0762 | 0.0592 | 0.0567 | 0.0382 | 0.0309 | 0.0052 | 0.0000 | 0.0535 |

## 6. 尺寸分桶与超小目标召回

| model | set | bucket | GT | TP | FN | Recall | AP50 | AP75 | AP90 | mAP50-95 | IoU med |
|---|---|---|---|---|---|---|---|---|---|---|---|
| eth_official | val | <4 | 24 | 0 | 24 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | n/a |
| eth_official | val | 4-6 | 71 | 30 | 41 | 0.4225 | 0.4296 | 0.1549 | 0.0141 | 0.2237 | 0.7234 |
| eth_official | val | 6-8 | 251 | 224 | 27 | 0.8924 | 0.9163 | 0.7948 | 0.1753 | 0.6582 | 0.8503 |
| eth_official | val | 8-12 | 1724 | 1685 | 39 | 0.9774 | 0.9799 | 0.9467 | 0.4806 | 0.8045 | 0.8991 |
| eth_official | val | 12-16 | 755 | 717 | 38 | 0.9497 | 0.9576 | 0.9450 | 0.6106 | 0.8220 | 0.9164 |
| eth_official | val | 16-24 | 343 | 306 | 37 | 0.8921 | 0.9167 | 0.9065 | 0.6545 | 0.8066 | 0.9289 |
| eth_official | val | 24-32 | 52 | 45 | 7 | 0.8654 | 0.9231 | 0.9231 | 0.6923 | 0.8231 | 0.9297 |
| eth_official | val | 32-64 | 29 | 21 | 8 | 0.7241 | 0.8966 | 0.8448 | 0.5862 | 0.7776 | 0.9337 |
| eth_official | val | >64 | 11 | 0 | 11 | 0.0000 | 0.1667 | 0.1212 | 0.0909 | 0.1258 | n/a |
| eth_official | controlled_capability/images | <4 | 160 | 0 | 160 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | n/a |
| eth_official | controlled_capability/images | 4-6 | 106 | 0 | 106 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | n/a |
| eth_official | controlled_capability/images | 6-8 | 54 | 0 | 54 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | n/a |
| eth_official | controlled_capability/images | 8-12 | 405 | 0 | 405 | 0.0000 | 0.0176 | 0.0143 | 0.0000 | 0.0111 | n/a |
| eth_official | controlled_capability/images | 12-16 | 825 | 4 | 821 | 0.0048 | 0.0334 | 0.0271 | 0.0056 | 0.0239 | 0.8761 |
| eth_official | controlled_capability/images | 16-24 | 106 | 1 | 105 | 0.0094 | 0.0143 | 0.0143 | 0.0143 | 0.0129 | 0.9285 |
| eth_official | controlled_capability/images | 24-32 | 54 | 0 | 54 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | n/a |
| eth_official | controlled_capability/images | 32-64 | 120 | 0 | 120 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | n/a |
| eth_official | controlled_capability/images | >64 | 240 | 0 | 240 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | n/a |
| eth_official | challenge_test/images | <4 | 40 | 0 | 40 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | n/a |
| eth_official | challenge_test/images | 4-6 | 0 | 0 | 0 | n/a | n/a | n/a | n/a | n/a | n/a |
| eth_official | challenge_test/images | 6-8 | 66 | 0 | 66 | 0.0000 | 0.0152 | 0.0000 | 0.0000 | 0.0076 | n/a |
| eth_official | challenge_test/images | 8-12 | 20 | 0 | 20 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | n/a |
| eth_official | challenge_test/images | 12-16 | 44 | 0 | 44 | 0.0000 | 0.0227 | 0.0227 | 0.0227 | 0.0205 | n/a |
| eth_official | challenge_test/images | 16-24 | 0 | 0 | 0 | n/a | n/a | n/a | n/a | n/a | n/a |
| eth_official | challenge_test/images | 24-32 | 0 | 0 | 0 | n/a | n/a | n/a | n/a | n/a | n/a |
| eth_official | challenge_test/images | 32-64 | 0 | 0 | 0 | n/a | n/a | n/a | n/a | n/a | n/a |
| eth_official | challenge_test/images | >64 | 24 | 0 | 24 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | n/a |
| eth_only_v1_best | val | <4 | 24 | 0 | 24 | 0.0000 | 0.0417 | 0.0000 | 0.0000 | 0.0042 | n/a |
| eth_only_v1_best | val | 4-6 | 71 | 13 | 58 | 0.1831 | 0.3651 | 0.1225 | 0.0000 | 0.1736 | 0.7208 |
| eth_only_v1_best | val | 6-8 | 251 | 110 | 141 | 0.4382 | 0.7329 | 0.5900 | 0.0777 | 0.4841 | 0.8374 |
| eth_only_v1_best | val | 8-12 | 1724 | 1174 | 550 | 0.6810 | 0.8478 | 0.7350 | 0.1643 | 0.6108 | 0.8570 |
| eth_only_v1_best | val | 12-16 | 755 | 345 | 410 | 0.4570 | 0.7015 | 0.5510 | 0.2074 | 0.4962 | 0.8813 |
| eth_only_v1_best | val | 16-24 | 343 | 144 | 199 | 0.4198 | 0.6993 | 0.5361 | 0.1815 | 0.4721 | 0.8413 |
| eth_only_v1_best | val | 24-32 | 52 | 13 | 39 | 0.2500 | 0.4622 | 0.3628 | 0.1673 | 0.3339 | 0.9029 |
| eth_only_v1_best | val | 32-64 | 29 | 10 | 19 | 0.3448 | 0.7759 | 0.6207 | 0.1724 | 0.5483 | 0.8836 |
| eth_only_v1_best | val | >64 | 11 | 0 | 11 | 0.0000 | 0.1212 | 0.0000 | 0.0000 | 0.0485 | n/a |
| eth_only_v1_best | controlled_capability/images | <4 | 160 | 0 | 160 | 0.0000 | 0.0157 | 0.0031 | 0.0000 | 0.0047 | n/a |
| eth_only_v1_best | controlled_capability/images | 4-6 | 106 | 0 | 106 | 0.0000 | 0.0211 | 0.0174 | 0.0000 | 0.0119 | n/a |
| eth_only_v1_best | controlled_capability/images | 6-8 | 54 | 0 | 54 | 0.0000 | 0.0948 | 0.0370 | 0.0000 | 0.0498 | n/a |
| eth_only_v1_best | controlled_capability/images | 8-12 | 405 | 1 | 404 | 0.0025 | 0.0824 | 0.0527 | 0.0079 | 0.0519 | 0.9378 |
| eth_only_v1_best | controlled_capability/images | 12-16 | 825 | 8 | 817 | 0.0097 | 0.0964 | 0.0750 | 0.0182 | 0.0665 | 0.8839 |
| eth_only_v1_best | controlled_capability/images | 16-24 | 106 | 2 | 104 | 0.0189 | 0.1433 | 0.1433 | 0.1030 | 0.1256 | 0.9085 |
| eth_only_v1_best | controlled_capability/images | 24-32 | 54 | 0 | 54 | 0.0000 | 0.0093 | 0.0093 | 0.0000 | 0.0074 | n/a |
| eth_only_v1_best | controlled_capability/images | 32-64 | 120 | 0 | 120 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | n/a |
| eth_only_v1_best | controlled_capability/images | >64 | 240 | 0 | 240 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | n/a |
| eth_only_v1_best | challenge_test/images | <4 | 40 | 0 | 40 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | n/a |
| eth_only_v1_best | challenge_test/images | 4-6 | 0 | 0 | 0 | n/a | n/a | n/a | n/a | n/a | n/a |
| eth_only_v1_best | challenge_test/images | 6-8 | 66 | 0 | 66 | 0.0000 | 0.1558 | 0.1114 | 0.0152 | 0.0994 | n/a |
| eth_only_v1_best | challenge_test/images | 8-12 | 20 | 0 | 20 | 0.0000 | 0.0750 | 0.0250 | 0.0000 | 0.0300 | n/a |
| eth_only_v1_best | challenge_test/images | 12-16 | 44 | 1 | 43 | 0.0227 | 0.1439 | 0.0714 | 0.0000 | 0.0729 | 0.8602 |
| eth_only_v1_best | challenge_test/images | 16-24 | 0 | 0 | 0 | n/a | n/a | n/a | n/a | n/a | n/a |
| eth_only_v1_best | challenge_test/images | 24-32 | 0 | 0 | 0 | n/a | n/a | n/a | n/a | n/a | n/a |
| eth_only_v1_best | challenge_test/images | 32-64 | 0 | 0 | 0 | n/a | n/a | n/a | n/a | n/a | n/a |
| eth_only_v1_best | challenge_test/images | >64 | 24 | 0 | 24 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | n/a |

聚合召回（指定口径）：

| model | set | Recall_<4 | Recall_4_6 | Recall_6_8 | Recall_<8 | Recall_8_12 | Recall_12_16 | Recall_8_16 | Recall_16_32 | Recall_32_64 | Recall_>64 |
|---|---|---|---|---|---|---|---|---|---|---|---|
| eth_official | val | 0.0000 | 0.4225 | 0.8924 | 0.7341 | 0.9774 | 0.9497 | 0.9689 | 0.8886 | 0.7241 | 0.0000 |
| eth_official | val|eth_unseen | 0.0000 | 0.0270 | 0.3864 | 0.1714 | 0.7833 | 0.6545 | 0.7345 | 0.4416 | 0.3333 | 0.0000 |
| eth_official | controlled_capability/images | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0048 | 0.0033 | 0.0063 | 0.0000 | 0.0000 |
| eth_official | challenge_test/images | 0.0000 | n/a | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | n/a | n/a | 0.0000 |
| eth_only_v1_best | val | 0.0000 | 0.1831 | 0.4382 | 0.3555 | 0.6810 | 0.4570 | 0.6127 | 0.3975 | 0.3448 | 0.0000 |
| eth_only_v1_best | val|eth_unseen | 0.0000 | 0.0000 | 0.1818 | 0.0762 | 0.2222 | 0.1364 | 0.1897 | 0.1948 | 0.2500 | 0.0000 |
| eth_only_v1_best | controlled_capability/images | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0025 | 0.0097 | 0.0073 | 0.0125 | 0.0000 | 0.0000 |
| eth_only_v1_best | challenge_test/images | 0.0000 | n/a | 0.0000 | 0.0000 | 0.0000 | 0.0227 | 0.0156 | n/a | n/a | 0.0000 |

## 7. 定位质量（TP 集合）与 height_ratio 检查

| model | set | n_tp | IoU mean | IoU med | IoU P10 | IoU P90 | center px640 med | norm center med | w_ratio mean | h_ratio mean | h_ratio med | area_ratio med | w_rel med | h_rel med |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| eth_official | val | 3028 | 0.8886 | 0.9037 | 0.8070 | 0.9530 | 0.316 | 0.0283 | 0.9911 | 0.9892 | 0.9869 | 0.9741 | 0.0325 | 0.0330 |
| eth_official | val | 269 | 0.8204 | 0.8300 | 0.7028 | 0.9153 | 0.641 | 0.0544 | 1.0052 | 1.0090 | 1.0023 | 0.9886 | 0.0743 | 0.0513 |
| eth_official | controlled_capability/images | 5 | 0.8626 | 0.8826 | 0.7958 | 0.9114 | 0.322 | 0.0268 | 0.9640 | 0.9707 | 0.9913 | 0.9229 | 0.0617 | 0.0123 |
| eth_official | challenge_test/images | 0 | n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a |
| eth_only_v1_best | val | 1809 | 0.8447 | 0.8572 | 0.7446 | 0.9291 | 0.402 | 0.0375 | 0.9369 | 0.9873 | 0.9839 | 0.9240 | 0.0658 | 0.0413 |
| eth_only_v1_best | val | 81 | 0.7615 | 0.7740 | 0.6330 | 0.8611 | 0.758 | 0.0627 | 0.8964 | 1.0062 | 1.0103 | 0.8768 | 0.1405 | 0.0707 |
| eth_only_v1_best | controlled_capability/images | 11 | 0.8759 | 0.9006 | 0.7021 | 0.9433 | 0.371 | 0.0309 | 1.0207 | 0.9885 | 1.0006 | 1.0129 | 0.0356 | 0.0168 |
| eth_only_v1_best | challenge_test/images | 1 | 0.8602 | 0.8602 | 0.8602 | 0.8602 | 0.250 | 0.0204 | 1.1121 | 1.0174 | 1.0174 | 1.1315 | 0.1121 | 0.0174 |

height_ratio 标记规则：ETH 若也接近 1.09 则标 POSSIBLE_SHARED_DATA_OR_EVALUATION_EFFECT；
ETH 不偏而我们的 A/B 偏则标 POSSIBLE_MODEL_OR_TRAINING_SPECIFIC_EFFECT（不下因果断言）：

| model | set | h_ratio mean | w_ratio mean | flag |
|---|---|---|---|---|
| eth_official | val | 0.9892 | 0.9911 | POSSIBLE_MODEL_OR_TRAINING_SPECIFIC_EFFECT (eth not biased) |
| eth_official | val | 1.0090 | 1.0052 | POSSIBLE_MODEL_OR_TRAINING_SPECIFIC_EFFECT (eth not biased) |
| eth_official | controlled_capability/images | 0.9707 | 0.9640 | POSSIBLE_MODEL_OR_TRAINING_SPECIFIC_EFFECT (eth not biased) |
| eth_only_v1_best | val | 0.9873 | 0.9369 | not_flagged |
| eth_only_v1_best | val | 1.0062 | 0.8964 | not_flagged |
| eth_only_v1_best | controlled_capability/images | 0.9885 | 1.0207 | not_flagged |
| eth_only_v1_best | challenge_test/images | 1.0174 | 1.1121 | not_flagged |

## 8. ETH-style center-distance 指标（单独口径，不等于 IoU-based 指标）

**ETH-style F1 != IoU-based F1。** 其定义（src/shuttletrack/utils.py:297-344）：每图只取 top-1 检测框（max_det=1）、
每图只取 1 个 GT（首个实例）、距离在 letterbox 后的网络输入系内度量、dist < 25 px（config.json: dist_threshold=25.0, 
confidence=0.5）、FP 至多 1 个/图。

| model | set | subset | conf | dist_thr | frame | TP | FP | FN | P | R | F1 | variant |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| eth_official | val | all | 0.50 | 25.000 | content | 3019 | 19 | 232 | 0.9937 | 0.9286 | 0.9601 | eth_official_default |
| eth_official | val | all | 0.50 | 25.000 | original | 3019 | 19 | 232 | 0.9937 | 0.9286 | 0.9601 | eth_default_original_px |
| eth_official | val | all | 0.50 | 15.625 | content | 3019 | 19 | 232 | 0.9937 | 0.9286 | 0.9601 | eth_default_640_equiv |
| eth_official | val | all | 0.25 | 25.000 | content | 3034 | 53 | 200 | 0.9828 | 0.9382 | 0.9600 | our_conf_0.25_content |
| eth_official | controlled_capability/images | all | 0.50 | 25.000 | content | 1 | 49 | 2020 | 0.0200 | 0.0005 | 0.0010 | eth_official_default |
| eth_official | controlled_capability/images | all | 0.50 | 25.000 | original | 1 | 49 | 2020 | 0.0200 | 0.0005 | 0.0010 | eth_default_original_px |
| eth_official | controlled_capability/images | all | 0.50 | 15.625 | content | 1 | 49 | 2020 | 0.0200 | 0.0005 | 0.0010 | eth_default_640_equiv |
| eth_official | controlled_capability/images | all | 0.25 | 25.000 | content | 3 | 85 | 1982 | 0.0341 | 0.0015 | 0.0029 | our_conf_0.25_content |
| eth_official | challenge_test/images | all | 0.50 | 25.000 | content | 0 | 3 | 192 | 0.0000 | 0.0000 | 0.0000 | eth_official_default |
| eth_official | challenge_test/images | all | 0.50 | 25.000 | original | 0 | 3 | 192 | 0.0000 | 0.0000 | 0.0000 | eth_default_original_px |
| eth_official | challenge_test/images | all | 0.50 | 15.625 | content | 0 | 3 | 192 | 0.0000 | 0.0000 | 0.0000 | eth_default_640_equiv |
| eth_official | challenge_test/images | all | 0.25 | 25.000 | content | 0 | 5 | 190 | 0.0000 | 0.0000 | 0.0000 | our_conf_0.25_content |
| eth_only_v1_best | val | all | 0.50 | 25.000 | content | 1541 | 6 | 1716 | 0.9961 | 0.4731 | 0.6415 | eth_official_default |
| eth_only_v1_best | val | all | 0.50 | 25.000 | original | 1541 | 6 | 1716 | 0.9961 | 0.4731 | 0.6415 | eth_default_original_px |
| eth_only_v1_best | val | all | 0.50 | 15.625 | content | 1541 | 6 | 1716 | 0.9961 | 0.4731 | 0.6415 | eth_default_640_equiv |
| eth_only_v1_best | val | all | 0.25 | 25.000 | content | 1816 | 29 | 1429 | 0.9843 | 0.5596 | 0.7136 | our_conf_0.25_content |
| eth_only_v1_best | controlled_capability/images | all | 0.50 | 25.000 | content | 13 | 1 | 2056 | 0.9286 | 0.0063 | 0.0125 | eth_official_default |
| eth_only_v1_best | controlled_capability/images | all | 0.50 | 25.000 | original | 13 | 1 | 2056 | 0.9286 | 0.0063 | 0.0125 | eth_default_original_px |
| eth_only_v1_best | controlled_capability/images | all | 0.50 | 15.625 | content | 10 | 4 | 2056 | 0.7143 | 0.0048 | 0.0096 | eth_default_640_equiv |
| eth_only_v1_best | controlled_capability/images | all | 0.25 | 25.000 | content | 29 | 17 | 2024 | 0.6304 | 0.0141 | 0.0276 | our_conf_0.25_content |
| eth_only_v1_best | challenge_test/images | all | 0.50 | 25.000 | content | 1 | 6 | 187 | 0.1429 | 0.0053 | 0.0103 | eth_official_default |
| eth_only_v1_best | challenge_test/images | all | 0.50 | 25.000 | original | 1 | 6 | 187 | 0.1429 | 0.0053 | 0.0103 | eth_default_original_px |
| eth_only_v1_best | challenge_test/images | all | 0.50 | 15.625 | content | 1 | 6 | 187 | 0.1429 | 0.0053 | 0.0103 | eth_default_640_equiv |
| eth_only_v1_best | challenge_test/images | all | 0.25 | 25.000 | content | 1 | 10 | 185 | 0.0909 | 0.0054 | 0.0102 | our_conf_0.25_content |

对照：IoU>=0.5 的 P/R/F1 见第 4 节的 Precision@op 与 Recall@0.5。两者差异来自口径，不是模型差异。

## 9. False Positive（无目标集合）

| model | set | conf>= | images | FP total | FP/img | images with FP | image FP rate | max conf FP |
|---|---|---|---|---|---|---|---|---|
| eth_official | synthetic_on_real_bg/images | 0.25 | 160 | 27 | 0.1688 | 23 | 0.1437 | 0.9006 |
| eth_official | synthetic_3d/images | 0.25 | 84 | 16 | 0.1905 | 15 | 0.1786 | 0.8700 |
| eth_official | real_images/backgrounds | 0.25 | 30 | 4 | 0.1333 | 4 | 0.1333 | 0.4679 |
| eth_official | real_images/raw | 0.25 | 59 | 3 | 0.0508 | 2 | 0.0339 | 0.7532 |
| eth_official | real_video/frames | 0.25 | 150 | 1 | 0.0067 | 1 | 0.0067 | 0.2502 |
| eth_official | real_match_frames/images | 0.25 | 235 | 6 | 0.0255 | 6 | 0.0255 | 0.5386 |
| eth_official | real_train/raw | 0.25 | 22 | 3 | 0.1364 | 2 | 0.0909 | 0.7248 |

## 10. 速度与模型成本（同硬件、batch=1、fp32）

| model | params | GFLOPs | latency mean ms | median | P90 | P95 | FPS | e2e mean ms | preprocess ms | inference ms | postprocess ms | VRAM peak MB |
|---|---|---|---|---|---|---|---|---|---|---|---|---|

协议：n/a；端到端另计时（含 preprocess/inference/postprocess），组件时间为框架 results.speed 均值，分开报告。

## 11. 模型成本与参数对照

| model | params | GFLOPs | nc | stride | latency mean ms | FPS | VRAM MB |
|---|---|---|---|---|---|---|---|
| eth_official | 11166560 | 73.76 | 80 | [8, 16, 32] | n/a | n/a | n/a |
| eth_only_v1_best | 9948638 | 59.19 | 1 | [8, 16, 32] | n/a | n/a | n/a |

## 12. 复现与产物

~~~
python tools/eval_eth_official_baseline.py --repo . --eth-ckpt _scratch_eth_official/best.pt --models eth_official,eth_only_v1_best --out-dir _scratch_eth_only_v1_eval\fair --examples-dir _scratch_eth_only_v1_eval\fair_examples --no-examples --no-latency --report _scratch_eth_only_v1_eval\fair\FAIR_ETH_VS_ETHONLY.md
~~~

- 预测缓存 _scratch_localization_audit/loc_<model>_<set>.json（imgsz 1024 / conf 0.001 / iou 0.7 / max_det 300 / rect=False）。
- 示例图 0 张；分歧图 0 张。
- 旧结果未覆盖：Stage A/B 既有 metrics 与 reports 原样保留。

## 13. 问答（Q1-Q10，人工）

（待填。）

