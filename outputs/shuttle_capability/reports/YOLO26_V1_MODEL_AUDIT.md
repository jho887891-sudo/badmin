# YOLO26 V1 模型审计（Spec §3 / Gate 1）

日期：2026-09-21　证据：`outputs/shuttle_capability/metrics/yolo26_v1_gate1_model_audit.json`、`yolo26_v1_gate1b_mechanism.json`

## 1. 现役模型

| 项 | 值 |
|---|---|
| 模型族 | Ultralytics **YOLO26** |
| 现役规模 | **yolo26s**（scale letter = s） |
| 权重文件 | `assets/external/_staging/F_yolo/weights/yolo26s.pt` |
| 字节 / sha256 | 20,422,725 B / `646f8bc3fe0a656803d95c294f7852321748cb29d13466a1af8862e2db384b1b` |
| 权重类别数 | nc=80（COCO 预训练，无羽毛球类） |
| Ultralytics 版本 | **8.4.150**（官方 wheel，解包使用，未安装进 site-packages） |

## 2. V1 目标结构

**yolo26s-P2**：与现役 yolo26s **同容量等级**，只增加 P2 分支（spec §3 要求「不能同时改变容量和 P2」）。

| 结构 | 检测层 | strides (px) | 参数量 | GFLOPs @640 |
|---|---|---|---|---|
| yolo26s（基线） | 3 | 8 / 16 / 32 | 10,010,000 | 23.079 |
| **yolo26s-P2（V1）** | **4** | **4 / 8 / 16 / 32** | **9,765,856** | **28.023** |

配置文件：`yolo26s-p2.yaml`（Ultralytics 8.4.150 自带 `cfg/models/26/yolo26-p2.yaml`，官方存在）。
注意命名：**scale 字母在 `-p2` 之前**；写成 `yolo26-p2s.yaml` 会静默退化为 n。

## 3. 可用 YOLO26 配置（本地安装实测）

`yolo26.yaml`, `yolo26-p2.yaml`, `yolo26-p6.yaml`, `yolo26-cls.yaml`, `yolo26-seg.yaml`, `yolo26-pose.yaml`,
`yolo26-obb.yaml`, `yolo26-depth.yaml`, `yolo26-sem.yaml`, `yoloe-26.yaml`, `yoloe-26-seg.yaml`

`official_p2_config_present = true` → **Gate 2 前提成立**。

## 4. 训练机制（以本地安装代码为准，不引用网络文章）

| 项 | 实测值 |
|---|---|
| loss | **`E2ELoss`**（`ultralytics.utils.loss`），内含 `one2many` / `one2one` 两个 `v8DetectionLoss` |
| label assigner | **`TaskAlignedAssigner`**（`ultralytics.utils.tal`）；one2many topk=10、one2one topk=7，alpha=0.5、beta=6.0 |
| STAL | **在安装的 8.4.150 源码中检索 `\bSTAL\b` 无任何命中 → 记录为「不存在/不适用」，不猜测** |
| optimizer 默认 | `auto`（训练时按数据量解析，未在配置里写死） |
| lr0 / lrf | 0.01 / 0.01 |
| momentum / weight_decay | 0.937 / 0.0005 |
| scheduler | `cos_lr=false` → 线性衰减（warmup_epochs=3.0，warmup_momentum=0.8） |
| AMP | 默认 true |
| Mosaic / MixUp / CutMix / CopyPaste | 1.0 / 0.0 / 0.0 / 0.0（V1 需按 spec 把 mosaic 降到 0.1–0.2、其余保持关闭） |
| close_mosaic | 10 |
| 检测头 | `one2many` + `one2one`（end2end NMS-free 路径可用） |

## 5. 结论

- 现役型号 = **YOLO26s**，与 V1 要求的 `yolo26s-P2` 容量等级一致 → **Gate 1 PASS**。
- 官方 P2 配置存在 → Gate 2 前提成立。
- V1 **不修改** loss / assigner / optimizer 体系（spec §14），只改数据、采样、增强与阶段划分。