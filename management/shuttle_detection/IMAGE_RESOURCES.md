# 羽毛球检测 —— 图片资源位置总表

实测方式：遍历目录统计图像文件数量与字节数（本轮会话实测，未改动任何文件）。

## 仓库根

| 项 | 值 |
|---|---|
| 本地仓库根 | `E:\具身智能\badmin_project`（`git rev-parse --show-toplevel` = `E:/具身智能/badmin_project`） |
| 分支 / HEAD | `feature/shuttle-detection` / `c798f789c4a9891a9e9704d31502bcb447b576f4` |
| remote origin | `https://github.com/jho887891-sudo/badmin` |
| 远端同名树（Isaac Sim / 训练实际运行处） | `/home/T7/dgut/robot_sim` |
| 仓库外同级目录（不在仓库内） | `E:\具身智能\_shuttle_data` |

**注意**：`tools/` 下 10 个脚本把远端根 `/home/T7/dgut/robot_sim` 写死
（`challenge_discrimination_sweep.py`、`roi_upper_bound.py`、`roi_zoom_sweep.py`、`run_on_match_frames.py`、
`test_static_fp.py`、`threshold_operating_point.py`、`zoom_match_detections.py`、`build_isaac_ladder.py`、
`build_isaac_pose.py`、`composite_isaac_pool.py`），另有 5 个把本地根 `E:\具身智能\badmin_project` 写死；
只有 4 个用 `Path(__file__).resolve()` 推导。

## 0. 总量

- 仓库内：**12,307 个图像文件 / 3,146.8 MB**
- 仓库根外的同级目录：**`E:\具身智能\_shuttle_data` 148 张 / 112.1 MB**
- **仓库根之内**、但被 gitignore 的暂存区：**`E:\具身智能\badmin_project\_scratch_rf\extracted` 8,053 张 / 544 MB**
  （`.gitignore:84` 忽略 `_scratch_rf/`；`git check-ignore -v` 实测命中）

## 1. 主力训练/评估图片区：`outputs/shuttle_capability/`

### 1.1 训练与验证（当前 nc=1 baseline 真正用的）

| 路径 | 张数 | 体积 | 说明 |
|---|---|---|---|
| `outputs/shuttle_capability/train_data/train/images` | 673 | 193.2 MB | **训练集**，全部 960x960 |
| `outputs/shuttle_capability/train_data/val/images` | 144 | 40.4 MB | 验证集，全部 960x960 |
| `outputs/shuttle_capability/train_data/bg_train` | 25 | 17.7 MB | 合成用背景底图 |
| `outputs/shuttle_capability/train_data/bg_val` | 6 | 4.4 MB | 验证用背景底图 |
| `outputs/shuttle_capability/train_data/bg_excluded` | 8 | 1.5 MB | 审核排除（不是照片，球场线图/SVG） |

标签在同级 `train/labels`、`val/labels`（YOLO txt，非图片）。

### 1.2 真实图片（已下载，未进训练）

| 路径 | 张数 | 体积 | 说明 |
|---|---|---|---|
| `outputs/shuttle_capability/real_images/raw` | 59 | 24.1 MB | 真实照片原始下载 |
| `outputs/shuttle_capability/real_images/backgrounds` | 30 | 19.6 MB | 无球场景背景 |
| `outputs/shuttle_capability/real_train/raw` | 22 | 10.7 MB | 真实训练候选（仅 4 张可自动标注） |
| `outputs/shuttle_capability/real_match_frames/images` | 235 | 45.9 MB | 真实比赛抽帧 |
| `outputs/shuttle_capability/real_video/frames` | 150 | 88.3 MB | 真实视频抽帧 |
| `outputs/shuttle_capability/hard_negatives/raw` | 36 | 11.4 MB | 第一批硬负样本 |
| `outputs/shuttle_capability/hard_negatives2/raw` | 60 | 21.0 MB | 第二批硬负样本 |

### 1.3 合成图片池

| 路径 | 张数 | 体积 | 说明 |
|---|---|---|---|
| `outputs/shuttle_capability/synthetic_3d/images` | 84 | 89.5 MB | 1280x1280 受控合成 |
| `outputs/shuttle_capability/synthetic_on_real_bg/images` | 160 | 431.0 MB | 真实背景上的受控合成 |

### 1.4 冻结评估集（**不得回流训练**）

| 路径 | 张数 | 体积 | 说明 |
|---|---|---|---|
| `outputs/shuttle_capability/controlled_capability/images` | 2070 | **1202.4 MB** | 受控能力矩阵，仓库内最大图片目录 |
| `outputs/shuttle_capability/challenge_test/images` | 227 | 102.3 MB | CHALLENGE 集 |
| `outputs/shuttle_capability/real_images/raw` 中被冻结的部分 | 见 manifest | - | `metrics/real_image_verified_manifest.csv` 逐行登记 |

### 1.5 可视化产物

| 路径 | 张数 | 体积 |
|---|---|---|
| `outputs/shuttle_capability/visualizations` | 11 | 18.6 MB |
| `outputs/shuttle_capability/visualizations/video_sheets` | 7 | 15.1 MB |
| `outputs/shuttle_capability/visualizations/shuttle_candidates` | 1 | 1.3 MB |

## 2. 仓库内其他图片

| 路径 | 张数 | 体积 | 说明 |
|---|---|---|---|
| `outputs/shuttle_detection/background_review_sheets` | 4 | 14.9 MB | 背景审核拼图 |
| `outputs/shuttle_detection/capability` | 10 | 10.0 MB | capability 可视化 |
| `outputs/evidence` | 24 | 10.6 MB | 证据图（含 roboflow_samples.jpg 等） |
| `outputs/evidence/thumbs` | 5 | 1.0 MB | 缩略图 |
| `_scratch_rebot/reBot-DevArm/hardware/.../*/images` | ~130 | ~140 MB | B601-DM 装配步骤图、实物照片、3D 打印件图 |
| `_scratch_rebot/reBot-DevArm/media`、`community` | 22 | 76 MB | 上游仓库媒体图 |
| `assets/third_party/textures` | 2 | 3.6 KB | `racket_strings_basecolor.png`、`shuttle_basecolor.png` |

**注意：`assets/` 里几乎没有图片资源。** 它装的是 3D/权重：

| 文件 | 大小 |
|---|---|
| `assets/external/_staging/F_yolo/weights/yolo26s.pt` | 20,422,725 B |
| `assets/external/_staging/D_racket_shuttle/original/badminton_racket_and_shuttlecock_low_poly.glb` | 308,444 B |
| `assets/shuttle/shuttlecock.usd` | 6,622 B |
| `assets/third_party/shuttlecock_visual.usd` | 158,320 B |
| `assets/third_party/racket_visual.usd` | 43,217 B |

## 3. 仓库外的图片区

### 3.1 `E:\具身智能\_shuttle_data` —— 采集/审核暂存（未纳入 git）

共 148 张 / 112.1 MB：

| 子目录 | 张数 | 体积 |
|---|---|---|
| `real_images` | 59 | 24.1 MB |
| `bg_train` | 32 | 19.1 MB |
| `bg` | 30 | 19.6 MB |
| `bg_val` | 7 | 4.5 MB |
| `verify` | 7 | 12.0 MB |
| `vs` | 4 | 8.1 MB |
| `cand` | 1 | 1.3 MB |
| 根目录散图 | 8 | 23.5 MB |

这里是仓库内 `real_images/`、`bg_*` 的上游来源（数量略有出入，说明清理过）。

### 3.2 `_scratch_rf/extracted` —— Roboflow 数据集（仓库根下，暂存）

| 子目录 | 张数 | 体积 |
|---|---|---|
| `train/images` | 5617 | 379.3 MB |
| `valid/images` | 1620 | 109.6 MB |
| `test/images` | 816 | 55.1 MB |
| 压缩包 `_scratch_rf/shuttlecock_yolo26_v1.zip` | - | 见盘 |

全部 640x640。**这 8,053 张从未进入训练**（见 `tools/leakcheck_roboflow.py` 的防泄漏检查）。

### 3.3 `E:\具身智能\_yolo_tmp`

空（0 张图片）。

## 4. 已确认的缺口

1. **训练好的权重不在本地。** 全仓 `glob **/best.pt` = 0 个命中，`outputs/` 下没有 `weights/` 目录。
   `outputs/shuttle_detection/capability/FREEZE_HANDOFF_v2.md:37-39` 记的是 `weights/best.pt`、20,317,957 字节、
   sha256 `61c49120...74c963` —— 该文件实际在远端主机 `/home/T7/dgut/robot_sim/` 下。
2. ~~`outputs/shuttle_capability/isaac_pool/` 与 `isaac_pool_holdout/` 已不存在~~ —— **已更正（远端实测）**：
   本地没有，但远端 `/home/T7/dgut/robot_sim/outputs/shuttle_capability/isaac_pool/` **存在**
   （`composite_manifest.csv` / `mask` / `render_records.json` / `rgb` = 400 张 PNG），
   且远端 `train_data/train/images` 有 **1403 个文件**（含 **382** 张 `isaac_train_*`、33 张 `hardneg_train_*`），本地只有 673 个。
   详见 `DATASET_RESOLUTION.md` 第 4.1 节。
3. `manifest_train_os.csv` 中本地缺 435 行；**其中 382 行（`isaac_train_*`）在远端齐全**。
   真正仍未定位的只有 `hardneg_train_*` 的 53 行 —— 本地与远端都只有 33 个文件。
4. 远端还有本地完全没有的目录：`isaac_ladder`、`isaac_ladder_holdout`、`isaac_ladder_small`、
   `isaac_pool_side`、`isaac_pose`、`annotations`、`reports`。

## 5. ETH 官方 shuttle_detection 仓库与预训练权重（2026-09-28 迁出本地）

下载 ETH 官方羽毛球检测仓库时曾落在仓库根的临时目录 `_scratch_eth_shuttle/`（**未被 `.gitignore` 覆盖**，会污染 `git status`）。
其中 `runs/final-model/best.pt` 是 **`*.pt` 权重 = 资源**，按 AGENTS.md 规则不得留在本地。已于 2026-09-28 迁出。

| 项 | 值 |
|---|---|
| 新位置（远端，唯一真源） | `/home/T7/dgut/robot_sim/eth_official_code/shuttle_detection/` |
| 文件数 / 字节 | **73 / 268,910,149** |
| 权重 | `runs/final-model/best.pt`，**134,312,133 B**，sha256 `f1aea7dec784a24d53d475f89510ef45fc13255df6298c6a0ea2fbd4419c7d6d` |
| 逐文件校验 | 73/73 sha256 与本地完全一致（missing 0 / extra 0 / mismatch 0） |
| 可校验清单（进仓库） | `outputs/shuttle_capability/metrics/eth_official_repo_inventory.csv`（每行 relpath / bytes / sha256） |
| 本地临时目录 | `_scratch_eth_shuttle/` —— 校验通过后已删除 |
| 再生方式 | `git clone` ETH shuttle_detection 官方仓库 + 其发布页的 final-model 权重；校验以清单 sha256 为准 |

> 注意：**ETH 数据集本体**（17.4 GB / 29,377 张）不在此目录，位置见本文件其他章节与 `DATASET_RESOLUTION.md`；
> 本轮只搬迁「官方代码 + 官方权重」这一小份资源。
