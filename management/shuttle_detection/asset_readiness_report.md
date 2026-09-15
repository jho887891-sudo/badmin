# Shuttle Detection — Asset Readiness Report

## 状态：PASS（`ASSET_READY`）

| 项 | 值 |
|---|---|
| 日期 | 2026-09-15 |
| 分支 | `feature/shuttle-detection` |
| **记录的代码 commit** | **`8b1e7ae`** |
| 数据集版本（审计计算） | `sha256:b7eec2774cabe8e2` |
| 依据 Spec | `docs/superpowers/specs/01_DATA_ASSET_READINESS_SPEC.md` |
| 依据 Plan | `docs/superpowers/plans/2026-09-15-shuttle-data-asset-readiness.md` |
| 检测任务定义 | `nc=1`，类名 `shuttlecock` |

## 1. 治理文档安装与校验

用户交付的两个压缩包已解压、按规则放置，并**对照其自带 manifest 逐文件 sha256 校验**：

| 位置 | 文件数 | 校验 |
|---|---|---|
| `docs/superpowers/specs/` | 11 | SPEC_MANIFEST.json 全部匹配 |
| `docs/superpowers/plans/` | 10 | PLAN_MANIFEST.json 全部匹配 |

远端 `/home/T7/dgut/robot_sim/docs/superpowers/` 已同步同一份内容，两个 manifest 的 sha256
在两端一致（`SPEC_MANIFEST.json = 0fe09948…`、`PLAN_MANIFEST.json = bf9b555c…`）。

## 2. 背景池复核（Task 5 Step 1）

**结论：31/31 PASS，0 FAIL。** 交付物 `management/shuttle_detection/background_review.csv`。

客观检查（`outputs/shuttle_detection/background_objective_checks.csv`）：

| 检查 | 实测 |
|---|---|
| 可解码 / 损坏 | 31/31 可读，0 损坏 |
| 精确重复（sha256 相同） | 0 |
| 空白 / 单色帧（mean<12 或 std<6） | 0 |
| 与冻结 P3 能力集同源（r≥0.95） | **0**（最高 r=0.7753） |
| **跨 split 近似重复（r≥0.80）** | **0 → 训练/验证无泄漏** |
| 可辨羽毛球 | 目视 31 张（600px 网格），**均无** |

目视证据：`outputs/shuttle_detection/background_review_all.jpg`（总览）与
`background_review_sheets/bg_review_00..03.png`（分页，可复现，未入库）。

### 2.1 记录在案的 10 条 WARNING（**未删除任何一张**）

| 类别 | 文件 | 说明 |
|---|---|---|
| 同场景连拍序列 | `tbg_022/023` | 同一球馆外观，两种构图 |
| 同场景连拍序列 | `tbg_032/033/034` | 同一建筑外观（Virkakatu 1 Oulu 01/02/03） |
| 同场景连拍序列 | `tbg_036/037/038/039` | 同一球馆内部（AIS Arena）四个机位 |
| 过小会被放大 | `tbg_026`（800×465）、`tbg_029`（640×481） | 裁 960×960 时需上采样，背景偏软 |
| 视觉异常 | `tbg_016` | 废弃建筑苔藓地面，极端高频纹理 |
| 非自然场景 | `tbg_024` | 印刷海报扫描件，带文字 |

**近似重复给出两个互相矛盾的证据，如实并列：** 「16×16 亮度相关」只报出 1 对
（`tbg_032`↔`tbg_033`，r=0.820）；目视可见 3 组、共 9 张只覆盖 3 个场景。该全局降采样指标
**对同场景异角度不敏感**，单一数字不足以定论。

**裁决：全部保留。** 理由：① 真实存在的困难背景不因"难"删除；② 此刻删图会使刚通过全部检查的
520 张数据集失效，而重叠仅占训练样本约 4%（约 16/400）；③ 该比例是代表性偏斜，不是正确性缺陷，
合并背景池属失败驱动数据回流阶段的工作。代价：约 64/400 训练样本来自同一球馆的 4 张图。

### 2.2 已登记但本阶段不修的分布问题

背景池以**建筑外观、球馆观众席**为主，真正的室内球场只有 `tbg_012` 一张。这是 TRAIN 池的
**代表性**属性（写进报告供读者判断），不是单张图的缺陷，修复归属失败驱动数据回流。

## 3. 数据集审计（Task 5 Step 2）

```
python scripts/shuttle_detection/audit_dataset.py \
  --manifest outputs/shuttle_capability/train_data/manifest_train.csv \
  --dataset-root . --out outputs/shuttle_detection/dataset_audit
```

**实测：`samples=400 errors=0 warnings=3 passed=True version=sha256:b7eec2774cabe8e2`，exit code 0。**

产出五份报告：`dataset_inventory.csv`、`dataset_cleaning_report.csv`、
`dataset_distribution_report.csv`、`dataset_leakage_report.csv`、`audit_summary.json`。

清单校验（`audit_summary.json`）：`label_integrity=PASS`、`data_cleaning=PASS`、
`pool_isolation=PASS`、`distribution_quantified=PASS`；`dataset_leakage_report.csv` 为空
（无泄漏）。分布维度覆盖 7 项：split / source_type / camera_id / size_bucket / blur_bucket /
pose_bucket / is_negative。

3 条 WARNING 均为**已知且非阻塞**：

1. `MANIFEST_COLUMN_MISSING`：缺可选列 `camera_id`（左右相机字段，本数据集为单目合成，尚无该字段）
2. `MANIFEST_COLUMN_MISSING`：缺可选列 `is_negative`（负样本标记，当前池无纯负样本）
3. `SOURCE_TYPE_NON_CANONICAL`：manifest 写 `SYNTHETIC_HIFI_3D`，Spec 词表为
   `SYNTHETIC_3D|REAL_IMAGE|REAL_VIDEO|NEGATIVE`

> WARNING 3 的裁决：保留原值，归一化到 `SYNTHETIC_3D` 域后再计入报告。二者是同一评测域，
> `SYNTHETIC_HIFI_3D` 是本项目更精确的生成工艺名。

## 4. 资产清单（字节级校验）

| 资产 | 字节 | sha256 | 校验 |
|---|---|---|---|
| `yolo26s.pt` | 20,422,725 | `646f8bc3fe0a656803d95c294f7852321748cb29d13466a1af8862e2db384a1b` | 与记录逐字节一致 |
| `ultralytics-8.4.150-py3-none-any.whl` | 1,445,702 | `1af11709f0e525e72d8b1b71916c0081de3e63c49ab9ffdbffa995ae6a8c69de` | 与记录逐字节一致 |
| 羽毛球 GLB / USD 视觉体 | — | 见 `assets/THIRD_PARTY_ASSETS_SHUTTLECOCK.md` | — |

**YOLO 权重恢复过程（重要）：** 本地下载 `yolo26s.pt` **可复现失败** ——
`curl: (56) Recv failure: Connection was reset`，与 ISSUE-016 记录完全一致（该问题仍然有效，
非偶发）。改从**本项目自己的远端 staging** 取回并校验通过。
`/home/T7/ojh/test1/`、`/home/T7/JXCJG/` 下是**其他项目**的权重，未取用任何一个。

## 5. 远端训练环境（实测，非推断）

| 项 | 值 |
|---|---|
| 主机 | `dgut@172.31.68.251` |
| 项目目标目录 | `/home/T7/dgut/robot_sim/`（同步前不存在 → 全新建立） |
| Python | 3.12.13（`env_isaaclab`） |
| pytest | 9.1.1 |
| torch / CUDA | 2.10.0+cu128，`cuda_available=True` |
| torchvision | 0.25.0+cu128 |
| GPU | NVIDIA RTX A6000（49140 MiB，本轮实测占用 4545 MiB） |
| 磁盘 | `/home/T7` 1.5T 可用 |
| cv2 / PyYAML / PIL | 4.13.0 / 6.0.3 / 12.2.0 |
| ultralytics | **未安装**，按下方隔离方案就位 |
| pandas | **未安装**（本地与远端都没有；审计模块用标准库 csv 实现） |

### 5.1 ultralytics 隔离方案（已验证，不污染 env_isaaclab）

```bash
env_isaaclab/bin/pip install --no-deps --target /home/T7/dgut/robot_sim/third_party/ultralytics <wheel>
# 运行时：PYTHONPATH=/home/T7/dgut/robot_sim/third_party/ultralytics
```

| 验证项 | 实测 |
|---|---|
| 安装 | `Successfully installed ultralytics-8.4.150`，目标目录 11 MB |
| 导入 | `ultralytics 8.4.150` |
| **env_isaaclab 完整性** | torch 2.10.0+cu128、numpy 2.3.5 原样，**无 ultralytics** |
| 冒烟测试 | `YOLO(yolo26s.pt)` 加载 0.4 s；80 类，**无 shuttlecock 类**（印证文档）；
| | `predict(imgsz=640, device=0)` GPU 正常，首次冷调 22.85 s，峰值显存 94 MiB → **PASS** |

未修改 Isaac Sim / Isaac Lab / Python / PyTorch / CUDA / Driver，未停止 vLLM。

### 5.2 远端同步证据（**按内容校验，不按时间戳**）

本地无 rsync，改用 **tar + scp + 远端解包**（避免 PowerShell 管道破坏二进制）。

| 项 | 实测 |
|---|---|
| 归档 | 1528 个文件成员 / 278.0 MB（未压缩） |
| 传输 | 266.4 MB，scp 成功 |
| 解包目标 | /home/T7/dgut/robot_sim/ （同步前不存在，全新建立） |
| **内容校验** | sha256sum -c → **1527/1527 全部 OK** |
| **额外文件** | **0**（远端不含清单外的文件，除刻意建立的 third_party/ultralytics） |
| **缺失文件** | **0** |
| 清单 | outputs/shuttle_detection/sync/remote_sync_content_manifest.txt（sha256 + 字节数 + 路径） |

**结论：远端项目树与本地 8b1e7ae 逐字节一致**（1527 个文件），因此"本地 commit = 远端实际使用 commit"。

> **过程中发现并修正的一个真实缺陷（记录在案）：** 首次归档时 tar 把**正在写入的自身**也打进了包内
> （成员表记录写入瞬间的 112640 字节），而清单脚本又按当前实际大小 279 MB 计算该条目 →
> 出现"1528 文件 / 557.1 MB"与"归档 266.4 MB"的矛盾。定位后：**无数据丢失**，但远端多出一个
> 1.1 MB 的自包含垃圾文件 _sync_repo.tar，已删除；清单已排除该条目并重新生成校验文件。
> 教训：**归档产物必须落在被归档目录之外**，否则自包含。

## 6. 与计划的偏差（如实登记）

| 偏差 | 原因 | 影响 |
|---|---|---|
| 不用 pandas，`summarize_distribution` 用标准库 csv | 本地与远端均无 pandas，且不引入新依赖 | 无；接口语义不变 |
| CLI 支持重复 `--manifest` | 需要一次审计 train+val 两池 | 超出计划但不冲突 |
| 审计报告写 `--out` 指定目录（本报告用 `outputs/shuttle_detection/dataset_audit`） | 计划未指定落盘位置 | 无 |
| `background_review.csv` 增加 `pool` 列 | 便于区分 train/val 池 | 兼容（计划要求列均存在） |
| 实现者自报：`audit_dataset` 集成测试存在**一处"先实现后补测试"** | 已如实上报 | 已提交独立评审核查该测试是否真正约束行为 |

## 7. 未完成 / 不属本阶段

1. **nc=1 baseline 训练尚未开始**（Plan 2）。本报告只覆盖数据与资产就绪。
2. `fixed_core_test` / `challenge_test` 两个 split 尚未建立（Plan 3 建立并冻结）。
3. 左右相机字段 `camera_id` 尚无数据（单目合成数据；真机双目接入后补）。
4. `AMBIENT` / `KEY` 外观参数**未对真实相机标定**（沿用 P4-A 的已知限制，不得宣称已标定）。
5. 真实视频逐帧 GT 仍阻塞在相机/私有数据（P2 遗留）。

