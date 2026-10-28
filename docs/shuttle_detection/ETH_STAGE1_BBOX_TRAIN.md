# ETH Stage 1 BBox-only | YOLO26s 检测训练启动协议

GPT-6 负责代码和训练协议；DeepSeek/Trae 负责在真实 Linux/GPU 服务器执行和回传证据。

本入口独立于历史 Mask-reviewed 四阶段训练门禁。**模型固定为 Ultralytics YOLO26s Detect，起始权重文件名必须是 yolo26s.pt**，不再使用 YOLOv8s，也不使用 yolo26s-seg.pt。训练脚本在真正创建输出目录前读取模型架构元数据，验证 Detect、yolo26、s 三项；仅文件名一致不足以通过 GPU 执行门禁。依赖的 ultralytics 必须能够加载 YOLO26s，环境版本与权重 SHA 由远端真实执行结果报告。阶段 1 仅使用 ETH TRAIN 全部 13,992 个 bbox 的原始像素裁剪，1024×1024 非白色纯色背景，不使用 Mask、GrabCut、前景缩放、插值、羽化。100 张预览并非全量训练集。

## 1. CPU Gate 与全量均衡数据

从最新 GitHub commit 获取 train_eth_stage1_bbox.py 和新增测试，使用隔离 worktree 执行：

~~~bash
python -m pytest -q tests/test_eth_stage1_bbox_train.py tests/test_eth_stage1_bbox_preview.py
~~~

在确认空闲磁盘容量后，生成全量训练集（输出目录必须是未存在的新目录）：

~~~bash
python tools/generate_eth_stage1_bbox_preview.py \
  --inventory /home/dgut/eth_audit/out/inventory_20261026T074352Z/instances.csv \
  --image-root /home/T7/dgut/robot_sim/eth_shuttle_detection \
  --output-dir /home/dgut/eth_audit/out/stage1_bbox_balanced_full_NEW \
  --mode balanced-full --seed 20261027 --time-bins 10 \
  --expected-instances 13992 --max-outputs 100000
~~~

必须验收 audit.json 的 unique_instances_covered=13992、full_unique_coverage=true、6场景×10分箱的60个单元曝光次数完全相同、mask_used=false、scale=1.0，并核验 SHA256SUMS。重复曝光次数单独记录。若空分箱或超额必须 BLOCKED，不可偷偷切换为非均衡 full。按 frame-index 分箱只是相对时间代理，不是经过核验的物理时间。查看 contact_sheet.png 与若干全尺寸 PNG 中的矩形边缘伪影，未经人工检查不得写视觉质量 PASS。

## 2. 真实内部验证集与模型权重门禁

候选内部验证数据为仓库 data/eth_real_hardneg_v2_val_manifest.csv。执行工具须先确认其字段、图片/标签可访问、实际 SHA256，且与全部 13,992 个训练源的哈希交集为零。该候选来自历史 ETH 官方 TRAIN，不能宣称独立最终测试集，也不保证现有预训练权重未见过它。

检查官方 YOLO26s Detect 预训练权重 yolo26s.pt 的真实路径、来源、类别数和完整 SHA256；不要凭文件名猜测模型。在执行前确认该 checkpoint 可以被服务器安装的 Ultralytics 加载且实际架构 YAML 为 YOLO26、scale=s、task=detect。若服务器暂无权重，需要先独立取得权重并计算 SHA；不能在训练命令中直接依赖自动下载。

设置 VAL_MANIFEST、MODEL、MODEL_SHA 为已验证真实值，其中 MODEL 的文件名必须是 yolo26s.pt，再执行干运行：

~~~bash
python tools/train_eth_stage1_bbox.py \
  --dataset-root /home/dgut/eth_audit/out/stage1_bbox_balanced_full_NEW \
  --val-manifest "$VAL_MANIFEST" \
  --val-root /home/T7/dgut/robot_sim/eth_shuttle_detection \
  --model "$MODEL" --model-sha256 "$MODEL_SHA" \
  --run-root /home/dgut/eth_audit/out/stage1_bbox_smoke_NEW \
  --epochs 1 --batch 4 --seed 20261027
~~~

无 --execute 只读核验，不执行 CUDA，不创建训练目录。成功状态是 DRY_RUN_PASS_NO_TRAINING，不是训练开始。

## 3. 用户已授权训练

若且唯若 CPU 测试、真实合成集完整性、场景时间均衡、模型 SHA、内部验证集隔离和服务器 CUDA 门禁全 PASS，将上面的干运行命令加 --execute，执行 1 Epoch 的 GPU smoke。记录 exit code、loss、val 指标、权重输出、显存、GPU 日志与 SHA。失败就停止并回传完整报错，不擅自调超参。

Smoke 成功后在不同的新 run-root 启动正式 Stage 1（--epochs 30 --batch 8 --execute），其余数据和原始 checkpoint 不变。训练器固定 imgsz=1024、AdamW lr0=1e-4，禁止 Ultralytics 额外的 mosaic、mixup、crop-paste、HSV、平移缩放等增强。本轮继续使用 AdamW 而非 MuSGD。**重要修复**：此前旧训练出现 lr/pg2 在第 1、2 轮约 0.0667113、0.0334091，符合 Ultralytics 对 Bias 参数组使用默认 warmup_bias_lr=0.1 的学习率预热行为，远高于 lr0=1e-4。本提交显式设置 warmup_bias_lr=0.0、warmup_epochs=3.0，让 Bias 和其他参数组一样从 0 预热，避免隐藏的高偏置学习率。恢复旧 checkpoint 并不能撤销此前已经发生的参数更新，应使用已核验的原始 YOLO26s 预训练权重和全新输出目录进行受控重训。此修复不能自动解释或解决此前 val/cls_loss 极大及验证 P/R/mAP=0，需另外诊断验证标签、推理输出和训练/验证背景域差异。不可覆盖历史数据/权重，不改 main，不自动部署机器人。

最终必须回传分支和 commit SHA、每个 Gate 的测试/训练退出码、13,992 唯一实例与重复数量、场景时间统计、权重 SHA、训练与验证清单哈希、真实 GPU 日志、best/last 权重路径及 SHA。无人工标注的独立真实视频 GT，Recall 和灯具误检改善仍为 UNKNOWN。

## 4. 历史训练学习率异常与重训验收

在旧训练日志（9 轮）中，第三参数组 pg2 的学习率约为第 1 轮 0.0667113、第 2 轮 0.0334091、第 3 轮 0.0001047。它符合 Bias 组预热从 0.1 降至目标值的形态。官方 Ultralytics 默认参数对显式 AdamW 优化器未禁用该机制。**必须检查远端旧训练 args.yaml 中 warmup_bias_lr、warmup_epochs、optimizer、lr0、lrf；以及安装版本的 param_groups 名称/param_group= bias 标记。**

此次代码将 warmup_bias_lr=0.0，warmup_epochs=3.0 固定在 trainer.train(...) 参数中。确认最新提交中的 CPU 回归测试运行通过后，执行新的 1 Epoch GPU smoke 并从保存的实际 args.yaml 和 results.csv 核验学习率。正式 30 Epoch 只有在 smoke PASS 且前期学习率不再出现偏置组异常高峰时才能执行。若出现不一致，禁止继续训练，回传配置、学习率曲线和全部训练日志由 GPT-6 修复。

不得把旧实验与新实验的前 9 轮 loss 数值直接视为严格相同训练条件，因为本次修改了 warmup 偏置学习率。