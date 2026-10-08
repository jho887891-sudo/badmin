# 羽毛球检测数据集 —— 图像分辨率实测

测量方式：直接读磁盘上每个图像文件的像素尺寸（PIL 只读文件头，不解码）。
测量时间：本轮会话；被测对象为仓库现有文件，未改动任何数据。

## 1. 训练图像：单一尺寸 960x960

| 目录 | 图像数 | 分辨率 |
|---|---|---|
| `outputs/shuttle_capability/train_data/train/images` | 673 | 960x960 (673/673 = 100%) |
| `outputs/shuttle_capability/train_data/val/images` | 144 | 960x960 (144/144 = 100%) |

结论：训练集在存储层面是**同质**的，没有任何多分辨率混入。
来源：`tools/composite_isaac_pool.py:20` (`--imgsz` 默认 960)、`:66-78`（先裁 960x960 画布再合成）；
`experiments/shuttle_detection/01_build_negatives.py:62`（默认 960）；
`experiments/shuttle_detection/02_add_nearfield_training.py:38`（默认 960）；
`experiments/shuttle_detection/04_add_hard_negatives.py:27` (`SIZE = 960`)。

## 2. 关键点：网络输入是 640，不是 960

`configs/shuttle_detection/baseline.yaml:10` -> `imgsz: 640`
`src/perception/shuttle_detection/training_config.py:101` -> `imgsz: int = 640`（代码默认值）
推理端同样按 640 计：`management/shuttle_detection/plan5_input_requirements.md:30`（51.7 ms @ imgsz 640）、`:31`（batch 16 / imgsz 640，峰值 4595 MB）。

因此 manifest 里记的 `equiv_size_px` 是 **960 画布上的像素**，送进网络时统一乘以 640/960 = 0.6667。
同一批 922 个正样本，两种口径下的等效尺寸分布：

| 桶 (px) | @960 存储 | @640 网络 |
|---|---|---|
| 0-4 | 32 (3.5%) | 131 (14.2%) |
| 4-6 | 99 (10.7%) | 161 (17.5%) |
| 6-8 | 115 (12.5%) | 112 (12.1%) |
| 8-12 | 158 (17.1%) | 110 (11.9%) |
| 12-16 | 92 (10.0%) | 37 (4.0%) |
| 16-24 | 55 (6.0%) | 33 (3.6%) |
| 24-32 | 27 (2.9%) | 23 (2.5%) |
| 32-64 | 66 (7.2%) | 79 (8.6%) |
| >64 | 278 (30.2%) | 236 (25.6%) |

数据源：`outputs/shuttle_capability/train_data/manifest_train_os.csv`（922 行非负样本）。
注意：换口径后 <6 px 的小目标占比从 14.2% 涨到 **31.7%** —— 近一半训练正样本在网络输入上不足 6 px。

## 3. 背景底图：原生分辨率不齐，训练时被裁成 960

| 目录 | 张数 | 实测分辨率（最常见的前几档） |
|---|---|---|
| `train_data/bg_train` | 25 | 1920x1440 x11、1920x1281 x2、1920x1387、1920x2560、1920x2400、1920x1306、1920x1301、1331x945、800x465、1920x1061、1920x1289、640x481、1920x1280、1920x1373 |
| `train_data/bg_val` | 6 | 1920x1440 x4、1920x2560、1920x1282 |
| `train_data/bg_excluded` | 8 | 819x460 x2、1280x720 x2、1920x2880、1920x1060、1920x1440、800x595 |

合成脚本从底图里随机裁一块 960x960（`tools/composite_isaac_pool.py:66-68`），所以底图原生尺寸不进入训练分辨率。

## 4. 训练清单里 435 行指向的文件已不在磁盘上

`manifest_train_os.csv` 共 1108 行。**本地**磁盘上 `train/images` 只有 673 个文件，缺 435 行；
**远端 `/home/T7/dgut/robot_sim/` 上同一目录有 1403 个文件**（见下）：

| 前缀 | 清单行数 | 本地文件数 | 远端文件数 | 缺口 |
|---|---|---|---|---|
| `train_` (SYNTHETIC_HIFI_3D) | 400 | 400 | 未逐项比对 | 0（本地） |
| `nf_train_` (SYNTHETIC_HIFI_3D) | 140 | 100 | 100 | 40 |
| `neg_train_` (NEGATIVE) | 100 | 100 | 未逐项比对 | 0（本地） |
| `hardneg_train_` (NEGATIVE) | 86 | 33 | 33 | 53（本地与远端都缺） |
| `isaac_train_` (SYNTHETIC_3D) | 382 | **0** | **382** | 0（远端齐全） |

### 4.1 更正（2026 远端实测，`dgut@172.31.68.251`）

此前本节写的「`isaac_pool/` 已不存在」**只对本地成立**。远端实测：

```
ls /home/T7/dgut/robot_sim/outputs/shuttle_capability/isaac_pool
→ composite_manifest.csv  mask  render_records.json  rgb
ls .../isaac_pool/rgb | wc -l   → 400
ls .../train_data/train/images | wc -l   → 1403
ls .../train_data/train/images | grep -c '^isaac_train_'   → 382
ls .../train_data/train/images | grep -c '^hardneg_train_' → 33
```

结论：**382 张 `isaac_train_*` 及其来源池 `isaac_pool/` 没有丢，在远端**。
本地缺的 382 张不是数据丢失，而是本地只保留了子集。
仍未定位的只有 `hardneg_train_*` 的 53 行（本地与远端都只有 33 个文件）。

- 它们的存储分辨率按生成脚本应为 960x960（`tools/composite_isaac_pool.py:20,66-78`），源渲染是 1280x1280（`tools/render_pool_isaac.py:46` `W = H = 1280`）。
  **这是「按代码推断」= CONFIRMED_REAL（代码），本地文件不存故本地无法实测。**
- 清单 `imgsz` 列全部为 960（1108/1108）。

## 5. 未参与训练的数据的分辨率

| 数据集 | 张数 | 分辨率 |
|---|---|---|
| Roboflow shuttlecock `train` | 5617 | 640x640 (100%) |
| Roboflow `valid` | 1620 | 640x640 |
| Roboflow `test` | 816 | 640x640 |
| `real_match_frames`（真实比赛帧） | 235 | 1280x720 x211、1920x1080 x24 |
| `real_images` | 89 | 混合：1920x1280 x23、1920x2880 x8、1920x1440 x6、1280x960 x5、1920x2560 x3、1443x1920 x3、800x600 x3、1920x1080 x2、1200x1800 x2、… |
| `real_train` | 22 | 混合，1920 宽为主，亦有 800x600、533x800 |
| `real_video` | 150 | 1280x720 x48、848x464 x48、320x240 x48、1440x1440 x6 |
| `hard_negatives` | 36 | 混合，约 1280 宽（1280x960 x11、1280x853 x5、1280x854 x3、…） |
| `hard_negatives2` | 60 | 混合，1280x853 x21、1280x960 x7、… |
| `synthetic_3d` | 84 | 1280x1280 |
| `synthetic_on_real_bg` | 160 | 1280x1280 |
| `challenge_test` | 227 | 960x960 x203、1792x1792 x24 |
| `controlled_capability` | 2070 | 960x960 x1710、1280x1280 x360 |

真实照片验证集的分辨率**逐行记录在册**：`outputs/shuttle_capability/metrics/real_image_verified_manifest.csv` 有 `image_w` / `image_h` 列（如 real_001.jpg 1920x1280、real_008.jpg 1920x2550）。

## 6. 分辨率已被当作「受控轴」测量过

`outputs/shuttle_capability/metrics/capability_before.json`：84 张图，同一权重扫 imgsz 640/960/1280 ->
overall_recall_any = **0.5595 / 0.5714 / 0.5952**；recall_sports_ball 三档全 0.0357。
逐桶结果见 `outputs/shuttle_capability/metrics/size_bucket_metrics_before.csv`（同桶 recall 随 imgsz 变化，非单调）。

## 7. 缺口 / UNKNOWN

1. `isaac_train_*` 382 张 + `hardneg_train_*` 53 张：**文件不在磁盘**，分辨率只能按生成代码推断，无法实测。删除原因 UNKNOWN。
2. 训练用 960 存储 vs 640 入网：**没有找到任何记录说明 960 是有意选的**（配置只有 640 一个数字）。若 960 是为了不让小目标先被裁掉，这个理由没有写进文档 -> UNKNOWN。
3. 真实域分辨率覆盖：训练集 100% 是 960（合成），真实测试图是 1280-2880 宽。分辨率域差距存在，但**尚未作为独立受控轴做过训练侧消融** -> UNKNOWN。