# ETH 真实训练集全量标注框清点 —— 执行与审计报告

**任务口径**：只完成第一步 —— ETH 真实训练集全量标注框清点、数据来源审计、SHA256 核验。
不启动训练、不生成合成数据、不覆盖历史权重、不把模拟结果冒充真实数据结果。

**结论一句话**：`PASS` → `CANDIDATES_PENDING_REVIEW`。真实 ETH TRAIN 全量为
**13992 张图 / 13992 个框**，全部逐张验证图片可读、宽高与 manifest 一致、SHA256 与 manifest 逐位一致；
来源隔离 0 泄漏；数量对账 13992 = 13992 + 0。**但这只是候选框清点，Mask 未审核，不可训练。**

---

## 1. 执行环境

| 项 | 值 |
|---|---|
| 主机 | `jxxy`（Linux 5.15.0-191-generic x86_64），用户 `dgut`，IP `172.31.68.251` |
| 仓库路径 | `/home/dgut/eth_audit/repo`（无 GitHub 出口，代码用 `git archive` 精确导出后 scp 上传） |
| 分支 | `feature/eth-four-stage-copypaste` |
| HEAD SHA | `29f4b46f11a2b721e0cf148c8f9cb8effa708410`（`docs: add real ETH inventory execution gate`） |
| 代码来源校验 | `git archive` 生成的 tar sha256 = `5c3d4166a1cc55d7e53957a1df72810db38120c1afd315bcd38cd1bc1e8fb1af`，上传后远端 `sha256sum` 逐位一致 |
| Python | venv（`--system-site-packages`，基于 `python3` 3.10.12）：pytest 9.1.1 / pandas 2.3.3 / Pillow 12.3.0 / numpy 2.2.6 |
| 数据挂载 | `/home/T7` 为 `fuseblk`(NTFS) 7.1T 卷，数据集 18 GB |

> 服务器无 GitHub 出口（`https://github.com` 经 `127.0.0.1:10080` 代理，连接被拒），
> 因此代码以归档方式传输；PyPI 可达，仅装了纯 Python 的 `pytest`，未改动 Isaac 环境。

## 2. 数据来源（实际确认，非按文件名推断）

| 角色 | 实际路径 | 大小 (B) | SHA256 |
|---|---|---|---|
| ETH V2 TRAIN manifest | `data/eth_real_hardneg_v2_train_manifest.csv` | 6 319 040 | `d2bd887357555c9262f769ba81d3bfd49350e1670e80a94ecdc1d166898a8d4d` |
| 图片根 / 标签根 | `/home/T7/dgut/robot_sim/eth_shuttle_detection` | 18 GB | 36 个 `set_dir`，29 377 张 jpg，23 161 个 txt |
| 冻结评测清单 1 | `data/clean_outdoor_cp_v1/frozen_eval_index.csv` | 815 395 | `cdd4440a567e8dc3b77c6b056c5d475c1c0c8c0915a4cccba8d5c7fdd8811abe` |
| 冻结评测清单 2（留出地点） | `data/eth_real_hardneg_v2_val_manifest.csv` | 1 217 290 | `ddaf46aebc6a8261e640197af26476e86d4075d8d69493cce6de24b2527e6784` |

### 2.1 关于"仓库内 manifest 是 0 字节"的说法 —— 已过时，实测非空

`data/eth_real_hardneg_v2_train_manifest.csv` 在 HEAD `29f4b46` 的 git blob 为 **6 303 784 B**，
检出后 **6 319 040 B**（Windows 侧 CRLF 转换差 15 256 B，恰为行数）。远端 Linux 上实际使用的字节数
与 sha256 已在上表记录。本报告不使用任何空占位文件。

manifest 表头 28 列，含本任务要求的 `is_positive`：`image,label,location,difficulty,split,set_dir,role,
in_official_training,source,is_negative,n_boxes,width,height,sha256,...,is_positive,repeat_index,...`

### 2.2 来源构成（15255 行，全部 `split=train`）

| 来源 | 行数 | 说明 |
|---|---|---|
| `eth_main` / `role=official_positives` | 13 993 | ETH 真实采集正样本 |
| `hard_negative` | 712 | 难负样本，89 张图各重复 8 次（`repeat_index` 0–7） |
| `eth_negative`（`set_dir=coco_train_easy`） | 550 | COCO 图作为负样本 |

正样本按 location：cab_2 3271 / glc_1 2446 / ticino_1 2436 / ticino_2 2369 / cab_1 2127 / ml_4 1343。

### 2.3 冻结评测清单的真实性质（重要限定，属 UNKNOWN 项）

* `frozen_eval_index.csv`：3 989 行，`role=FROZEN_EVAL`，列为 `source_id,role,sha256,bytes,origin`；
  origin 分布含 `controlled_capability` 2070、`train_data` 856、`real_match_frames` 235、
  `challenge_test` 227、`synthetic_on_real_bg` 160、`real_video` 150、`real_images` 89、
  `synthetic_3d` 84、`hard_negatives2` 60、`hard_negatives` 36、`real_train` 22。
  **其中含合成来源（`synthetic_on_real_bg` + `synthetic_3d` 共 244 行）与 `train_data` 856 行**，
  故该索引不能整体当作"纯真实 ETH 独立评测集"。
* `eth_real_hardneg_v2_val_manifest.csv`：2 920 行，`split` 列**全部标为 `train`**，
  且与 `data/eth_only_v1_val_manifest.csv` **字节完全相同**（同 sha256）。其 2 920 张图只属于
  `glc_2`(1613) 与 `uetlibergstrasse_2`(1307) 两个 location，与训练 manifest 的 15 个 `set_dir`
  **交集为空**（真正的 location-heldout）；但**全部 `in_official_training=True` 且物理位于
  `<set_dir>/images/train/`** —— 它们只是"相对本 V2 清单留出"，对曾经在 ETH 官方全集上训练过的模型
  并不构成独立基准（历史泄漏审计实测该类 `A_eth_trained_folder` 召回 0.998）。

## 3. 测试结果

```
python -m py_compile tools/prepare_eth_real_foregrounds.py tools/eth_four_stage_inventory.py
  → rc=0

pytest -q tests/test_eth_four_stage_inventory.py tests/test_clean_cp_sources.py \
         tests/test_clean_cp_mask_and_labels.py
  → 44 passed in 1.65s   （通过 44 / 失败 0）
```

首次运行 2 个模块收集失败（`ModuleNotFoundError: No module named 'pandas'`），
原因是所选解释器缺依赖；改用含 pandas 的 venv 后全绿，**未修改任何测试或验收标准**。

## 4. 清点结果（真实统计，逐张核验）

执行命令（脚本 `scripts/run_eth_inventory.sh`，输出目录为新建，拒绝覆盖）：

```
tools/eth_four_stage_inventory.py \
  --manifest data/eth_real_hardneg_v2_train_manifest.csv \
  --label-root /home/T7/dgut/robot_sim/eth_shuttle_detection \
  --frozen-eval-manifest data/clean_outdoor_cp_v1/frozen_eval_index.csv \
  --frozen-eval-manifest data/eth_real_hardneg_v2_val_manifest.csv \
  --out-inventory <new>/instances.csv --out-audit <new>/audit.json
```

| 指标 | 数值 |
|---|---|
| manifest 总行数 | **15 255** |
| ETH TRAIN 标注框总数 | **13 992** |
| 可用实例数 | **13 992** |
| 无效实例数 | **0** |
| 排除的非训练行 | **1 263**（唯一原因 `not_positive`：550 coco 负样本 + 712 难负样本 + 1 行自相矛盾） |
| 数量核对 | **13 992 = 13 992 + 0 → 一致** |
| 来源隔离 | **PASS**：正样本 0 条命中 val 清单 / 冻结索引 / eval token / 非 ETH 来源 |
| 每图框数 | 恒为 1（13992 个 `#0`，无 `#1` 及以上） |
| 唯一 ID | 13 992 个，格式 `sha256#idx`，无重复 |
| 状态 | 全部 `PENDING_REVIEW` / `mask_not_reviewed`；`approved_masks=0` |

### 4.1 尺寸门槛（确认未用旧 10/24 px 门槛过滤小目标）

| 指标 | short 边 (px) | long 边 (px) |
|---|---|---|
| 最小 / p50 / p95 / 最大 | 6 / 23 / 44 / 283 | 6 / 29 / 55 / 377 |
| 均值 | 25.12 | 32.05 |

**short < 10 px 的框 99 个、long < 24 px 的框 3365 个，全部保留在清单内**
（旧默认门槛会丢弃这 3365 个）。无 0 或 1 px 退化框。
`legacy_min_short_long_defaults_unchanged=true`：旧版默认门槛本身未被修改。

### 4.2 独立复核（不复用仓库工具，重新实现一遍）

`scripts/verify_eth_inventory.py` 直接从原始 CSV 重新推导选择逻辑并交叉校验，结果：

```
verdict = PASS, failures = []
row_count_matches   = true      image_set_matches = true
isolation_leaks     = 0         bad_instance_ids  = 0     bad_bboxes = 0
images_byte_verified= 13992     image_sha_mismatches = 0
independent_re_derivation: positives_after_firewalls = 13992, expected_boxes = 13992
```

即：**13 992 张图片独立重算 SHA256，与 manifest 声明的 sha256 逐位一致，0 处不匹配**。

## 5. 审计中发现的三个既有问题（不属本次清点失败，但必须记账）

1. **89 个 sha256 同时出现在冻结评测索引与训练 manifest 内**（`hard_negatives` 36 + `hard_negatives2` 60，
   去重 89 张）。它们在训练侧是难负样本并被上采样 8 次，在评测侧被标为 `FROZEN_EVAL`。
   因为它们是负样本，不影响本次正样本清点（数量 0 影响），但**会让"灯具/困难背景误检率"的评测被污染**。
   本次防火墙按哈希拒绝了它们（在清单中体现为先被 `not_positive` 拒绝）。
2. **1 行自相矛盾**：`ticino_1_easy/images/train/ticino_1_00212.jpg`
   `role=official_positives` 但 `is_positive=False`、`is_negative=True`、`n_boxes=0`。
   已按负样本排除，但 manifest 的角色字段应修正。
3. **val 清单 `split` 列全为 `train`** 且位于 `images/train/`，同时 `in_official_training=True`；
   作为"冻结评测清单"使用是口径不严谨的（见 2.3），任何据此得出的 Recall 只能是 `UNKNOWN`。

## 6. 证据文件

输出目录：`outputs/shuttle_detection/eth_four_stage_inventory/20261026T074352Z/`
（远端同源目录：`/home/dgut/eth_audit/out/inventory_20261026T074352Z/`）

| 文件 | 大小 | SHA256 |
|---|---|---|
| `instances.csv` | 6 839 595 | `1bc3887d4236c69257ddd2e0f3c5fb9542ac93e4da7fa4c7960e4a66fe9761ed` |
| `audit.json` | 271 644 | `7a265374a16f0c4e910827dbc9172efd3a3cf578408ef49a0fb8d29355af4bbf` |
| `stdout.json` | 565 | `93c7c556ee28c31229a533f132453a3e4032e0b79eb63eb9289f10f79a5d1b6b` |
| `stderr.txt` | 0 | `e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855`（空） |
| `verify_inventory.json` | 998 | `9222b29bba8a2e3ed514ad6301de31e719c2041fb7021aa9d3aadc91814857fd` |
| `probe_inputs.json` | 5 265 | `16c163201c5b806292374b2028ea3b3896331c9219d68dae4d09f07074900a0b` |
| `probe_overlap.json` | 2 964 | `5c7d18d3d772cdab70843a160fcb7cfa3c24723f8a1a03a5ab7c2897c0b850c1` |
| `probe_collision.json` | 9 015 | `e14e5d0f43601ea10f402406f000dea65b94da9721af15fa0e6f25e0b8b3d9ad` |
| `box_size_distribution.json` | 1 215 | 见 `SHA256SUMS` |
| `scripts/` | — | 4 个探针 + 1 个校验器 + 1 个 runner，见 `SHA256SUMS` |
| `SHA256SUMS` | — | 全目录可校验清单 |

**资源不落本地**：18 GB 原始图片/标签仍在 `/home/T7/dgut/robot_sim/eth_shuttle_detection`（只读使用，未改动）；
本仓库只提交文本结果（CSV/JSON/报告）。复现路径、命令与全部哈希已在上文与 `SHA256SUMS` 中给出，
拿不到原始资源时也能核对本报告每个数字。

## 7. 结论逐项

| 验收项 | 结论 | 依据 |
|---|---|---|
| 真实图片存在且可读 | **PASS** | 13 992 张全部 `Image.open` + `verify()` 通过 |
| 图片宽高与 manifest 一致 | **PASS** | 不一致即硬失败，实跑 0 例 |
| 图片 SHA256 与 manifest 一致 | **PASS** | 工具侧 0 失配；独立复核 13 992 张重算 0 失配 |
| 不混入 val/test/合成/COCO | **PASS** | 正样本 0 条命中两个冻结清单、0 条 eval token、0 条非 ETH 来源 |
| 不使用 10/24 px 门槛 | **PASS** | 3365 个 long<24 px、99 个 short<10 px 的框被保留 |
| 每框唯一 ID | **PASS** | 13 992 个 `sha256#0`，0 重复 |
| 每框有明确原因 | **PASS** | 全部 `PENDING_REVIEW / mask_not_reviewed`；1 263 条排除全部带 `not_positive` |
| 全量数量核对 | **PASS** | 13 992 = 13 992 + 0 |
| 冻结评测集独立性 | **UNKNOWN** | val 清单 `split=train`、`in_official_training=True`；冻结索引含 244 张合成图 + 856 张 `train_data` |
| 既有训练/评测泄漏 | **FAIL（已存在，非本次引入）** | 89 张难负样本同时属于冻结评测集与训练集 |

**总判定：`PASS` — `CANDIDATES_PENDING_REVIEW`。**
这只表示候选框清点完成，**不代表 Mask 通过人工审核，也不代表可以训练**。

## 8. 下一步入口

1. **Mask 候选生成**：`tools/propose_eth_instance_masks.py`（GrabCut），输入即本目录 `instances.csv`；
   13 992 个框为 1 图 1 框，可逐框一对一产出候选。
2. **人工审核门**：只有 `reviewed is True` 且 `reviewer_id` / `review_method` 有据的实例才能进合成池；
   本清单 `approved_masks = 0`，现状**不可进入 Stage 1**。
3. **先修再训**（建议顺序）：
   - 清理 89 张难负样本在"冻结评测集"与"训练集"间的双重身份，否则误检率评测无效；
   - 修正 `ticino_1_00212.jpg` 的 role 与 `is_positive` 冲突；
   - 为 val 建立真正的独立性口径（当前只能报 `UNKNOWN`）。
4. **未开始项**：抠图 / 合成 / Stage 1–3 训练 / 真实视频 Recall 全部未执行，按任务边界保持 `UNKNOWN`。
