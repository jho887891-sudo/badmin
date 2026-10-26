# ETH 四阶段羽毛球 Copy-Paste 操作手册

**范围：** 羽毛球单类 YOLO 检测的数据增强及训练评测。执行依赖顺序为 Stage 1 → Stage 2 → Stage 4 → Stage 3；编号仍保持原用户设计。

## 四阶段定义

1. **阶段 1：** 使用 ETH **TRAIN** 的全部合法羽毛球 bbox 建立实例 inventory；无法可靠抠图的微小目标保留失败原因。仅对经审核的实例 mask 进行非白色纯净背景 Copy-Paste，训练羽毛球主体。
2. **阶段 2：** 在纯色背景上添加方向性曝光模糊、独立残影、亮度、噪声及压缩退化；以 Stage 1 训练权重继续训练。
3. **阶段 3：回到真实背景，做最终训练 / 微调。** 这是最终模型的核心。背景来源：ETH 原始背景、真实球馆背景、灯具、反光、白线、观众席、墙面等困难背景。把前两阶段学到的“羽毛球主体表征”迁移回来，再用真实场景 fine-tune。最终验证：真实视频 recall、误检率、灯具误检是否下降、模糊球是否更容易检出。
4. **阶段 4：** 将经审核羽毛球前景 Copy-Paste 到不同实际环境，形成训练池，作为 Stage 3 最终微调的额外数据来源。必须与未加入多环境池的 E3 做等真实样本/等训练步数对照。

## 已实现的模块及入口

- `tools/eth_four_stage_contract.py`：训练来源/评测集隔离、配置与 checkpoint 元数据检查。
- `tools/eth_four_stage_inventory.py`：不设旧 10/24 px 门槛，逐框枚举 ETH train-only 正样本，显式记录不能使用的框。
- `tools/propose_eth_instance_masks.py`：GrabCut 生成候选，不自动标记为人工审核通过。
- `tools/eth_four_stage_solid.py`：带 alpha bbox 的非白色纯色合成。
- `tools/eth_four_stage_degrade.py`：曝光模糊与残影分开实现。
- `tools/eth_four_stage_environments.py`：必须经过人工确认的无目标真实背景 ROI 合成。
- `tools/build_eth_four_stage_train.py`：构造 E3（真实样本 + Stage 2 对照合成）与 E4（相同真实样本 + Stage 4 合成），冻结曝光次数与步数。
- `tools/train_eth_four_stage.py`：无 `--execute` 默认只进行 preflight/dry-run。正式训练必须验证权重 SHA 与路径，输出目录必须不存在。
- `tools/eval_eth_four_stage.py`：以现有仓库 `eval_yolo26_v1.match_greedy` 为单一匹配器，输出真实视频 GT 指标。
- `tools/run_eth_four_stage_pilot.py`：只生成固定 **CPU 合成夹具**，不是 ETH 数据处理完成证据。

## CPU 固定夹具运行

在 Python 环境已安装 `numpy`、`Pillow`、`pytest`、`PyYAML` 时：

```bash
python tools/run_eth_four_stage_pilot.py --fixture-output /tmp/eth_four_stage_fixture_001 --seed 42
pytest -q tests/test_eth_four_stage_*.py
```

CPU 夹具的输出必须是 `CPU_FIXTURE_ONLY`、`optimizer_steps: 0`、`real_recall: UNKNOWN`。文件包括 `stage1.png/.txt`、`stage2.png/.txt`、`stage4.png/.txt`、`manifest.json`；旧目录存在时拒绝覆盖。

## 下一步：真实 ETH 全量框清单（必须在数据实际挂载的 Linux 端执行）

本分支的仓库文件 data/eth_real_hardneg_v2_train_manifest.csv 当前是 **0 字节**，不能作为真实清单。请从已挂载的数据位置指定实际的 V2 manifest、标签根目录和冻结评测 manifest；不要把这个空占位文件当作成功输入。

先在远端仓库根目录执行：

~~~bash
set -euo pipefail
: "$ETH_V2_MANIFEST"
: "$ETH_LABEL_ROOT"
: "$FROZEN_EVAL_MANIFEST"
OUT_ROOT="$(mktemp -d /tmp/eth-four-stage-inventory.XXXXXX)"
python tools/eth_four_stage_inventory.py \
  --manifest "$ETH_V2_MANIFEST" \
  --label-root "$ETH_LABEL_ROOT" \
  --frozen-eval-manifest "$FROZEN_EVAL_MANIFEST" \
  --out-inventory "$OUT_ROOT/instances.csv" \
  --out-audit "$OUT_ROOT/audit.json"
printf 'ETH_INVENTORY_OUTPUT=%s\n' "$OUT_ROOT"
~~~

若有多个冻结评测清单，重复传入 --frozen-eval-manifest。命令会检查实际图片是否存在、图片宽高是否与 manifest 一致、图片 SHA256 是否匹配；会调用原有 prepare_eth_real_foregrounds.py 的 V2 parser 和 train/eval 来源过滤，并在新模式下绕过历史 10/24 px 门槛。**旧版默认门槛保持不变。**

成功输出只代表 CANDIDATES_PENDING_REVIEW，下一道门仍是人工 Mask 审核，不代表可以训练。若出现 BLOCKED_DATA_INVENTORY，保留完整错误信息与路径，不要生成空清单来绕过。

## 真正的 ETH 运行前置门

1. 读取实际远端 ETH TRAIN 原图、标注与冻结验证集清单，原图 SHA256 可核对；不得把 val/test 或任何评测派生图加入训练。
2. 通过 `enumerate_eth_train_instances(...)` 记录所有候选；所有小目标都必须记账，不能静默丢弃。选择性产生 `propose_instance_mask` 候选，并人工复核，只有 `reviewed is True` 且 reviewer_id 与 reviewer_method 有据时才能进入合成池。无法可靠抠出的 ETH 训练样本仍作为原始真实训练图参与 Stage 3。
3. 对各类纯色背景、模糊/残影参数、球馆 ROI 生成可视化审核图与 rejection JSONL。纯白、背景自带未标注羽毛球、评测源图、空 mask 或越界 bbox 必须拒绝。
4. 只有当完整源数据、mask、背景及模型 checkpoint gate 全部 PASS，才可以组装真实生成 manifest 和 Ultralytics 所需的 `data.yaml`。配置的 `checkpoint.path` 和 `checkpoint.sha256` 必须填写真实值，且字节匹配；`imgsz=1024`。当前四阶段配置保留空 checkpoint 用作保护，不是训练就绪状态。
5. 固定 E3/E4 真实样本列表、随机种子、合成曝光预算与 optimizer steps，并确保训练输出目录是新的；**不修改旧的 CLEAN_OUTDOOR_CP_V1 配置，也不覆盖历史权重**。

安全训练命令模板（仅在真实可访问且完成审批后替换占位路径）：

```bash
python tools/train_eth_four_stage.py \
  --config configs/eth_four_stage_copypaste_v1.yaml \
  --stage stage1 \
  --checkpoint-path /REAL/verified/weights.pt \
  --dataset-yaml /REAL/verified/dataset.yaml \
  --source-audit-json /REAL/verified/source_audit.json \
  --output-dir /REAL/NEW/run_stage1 \
  --dry-run
```

完整训练需 **显式** 将最后一行改为 `--execute`，并且所有 gate PASS。Stage 2 使用 Stage 1 best.pt，Stage 3 E3/E4 使用同一个 Stage 2 best.pt。该入口不会执行机器人部署。

## 独立真实评测验收

- 冻结所有模型共有的独立 GT 清单、NMS、confidence、imgsz 与标注规则。
- 真实视频含逐帧人工 GT 才能报告 Recall、Precision、TP/FP/FN；否则一律为 `UNKNOWN`，不能把检测框数量当 Recall。
- 灯具误检必须依赖人工审核的 FP 类型标签；模糊球 Recall 依赖 GT 的模糊分层标记。
- E3 与 E4 只有在相同真实视频样本、相同阈值和相同评测器下比较。Recall 改善但 FP 增长时应标记 `TRADEOFF_REVIEW_REQUIRED`，不能宣称胜出。
- 真实相机视角、连续漏检、实际 GPU 延迟和显存仍需远端证据；CPU 夹具不能替代这些验收。

## 未证实项

`REAL_ETH_FOREGROUND_EXTRACTION=UNKNOWN`, `HUMAN_MASK_APPROVAL=UNKNOWN`, `STAGE1_GPU_TRAIN=UNKNOWN`, `STAGE2_GPU_TRAIN=UNKNOWN`, `STAGE3_GPU_TRAIN=UNKNOWN`, `REAL_VIDEO_RECALL=UNKNOWN`, `LAMP_FP_REDUCTION=UNKNOWN`, `DEPLOYMENT=NOT_STARTED`。
