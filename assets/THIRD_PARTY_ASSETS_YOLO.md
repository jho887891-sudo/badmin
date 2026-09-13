# THIRD_PARTY_ASSETS_YOLO.md — YOLO 粗检测器（下载归档）

归档日期：2026-09-13
归档位置：`assets/external/_staging/F_yolo/`（**staging**，非最终资产目录）
下载执行：远端 `jxxy`（`dgut@172.31.68.251`），直连官方源（github.com/pypi 均可达）

## 1. 下载内容与校验

| 文件 | 大小 (bytes) | sha256 | 来源 |
|---|---|---|---|
| `weights/yolo11n.pt` | 5,613,764 | `0ebbc80d4a7680d14987a577cd21342b65ecfd94632bd9a8da63ae6417644ee1` | https://github.com/ultralytics/assets/releases/download/v8.3.0/yolo11n.pt |
| `weights/yolov8n.pt` | 6,549,796 | `f59b3d833e2ff32e194b5bb8e08d211dc7c5bdf144b90d2c8412c47ccfc83b36` | https://github.com/ultralytics/assets/releases/download/v8.3.0/yolov8n.pt |
| `wheels/ultralytics-8.4.150-py3-none-any.whl` | 1,445,702 | `1af11709f0e525e72d8b1b71916c0081de3e63c49ab9ffdbffa995ae6a8c69de` | PyPI（`pip download ultralytics --no-deps`） |

md5（便于与 Ultralytics 文档/发布页对照）：
```
261474e91b15f5ef14a63c21ce6c0cbb  weights/yolo11n.pt
95a2449609c73cd69a072b09daaff0cc  weights/yolov8n.pt
```

**完整性证据（实测）**：两个 `.pt` 均为 zip 容器，`zipfile.testzip()` 对**全部条目**校验 CRC：
```
yolo11n.pt  -> 507 entries, CRC all-entries OK
yolov8n.pt  -> 363 entries, CRC all-entries OK
```
（下载过程中 curl 曾报一次 `write error`，因此本轮专门做了 CRC 全量校验；结果证明文件**未截断**。）

## 2. 许可证（重要，需项目决策）

- **Ultralytics YOLO（代码与官方权重）采用 AGPL-3.0**。AGPL 属**网络传染性**许可：若对外提供基于它的服务，需按 AGPL 提供相应源码；闭源商用需购买 Ultralytics Enterprise License。
- 备选（宽松许可，如项目后续要求非 copyleft）：**YOLOX（Apache-2.0）**、**RT-DETR（Apache-2.0，另有非 Ultralytics 实现）**、NanoDet-Plus（Apache-2.0）。
- **本仓库当前只做下载归档，未在代码中引入、未安装到运行环境**；是否采用需你确认（见 `DECISIONS.md` 的 DEC-028）。

## 3. 在本项目中的角色（依 `docs/architecture/01_PERCEPTION.md` 第 22–26 节）

YOLO **只负责粗检测**，输出 `coarse bounding box + confidence + class`，回答「羽毛球大概在哪」。
明确**不**把 bbox 中心当精确球心（羽毛球外形不对称：羽裙/软木/姿态/运动模糊/遮挡）。
链路：`rectified image → letterbox/resize → YOLO → confidence 过滤 → NMS → 候选框 →`**显式还原到全幅整流图坐标**`→ ROI → 亚像素质心 → 双目匹配 → 三角化`。

本仓库已实现的对应环节（可直接对接）：
- ROI 门与坐标还原约定：`src/badminton_brain/perception/stereo_geometry.py`（`roi_gate`、`court_box_gate`）
- 亚像素质心（含空窗哨兵，DEC-026）：`subpixel_centroid` / `CentroidResult`
- 三角化与标定参数（TEMP/REQUIRES_CALIBRATION）：`triangulate`、`CameraIntrinsics`
- 感知层适配器（可注入真实检测器）：`perception/perception_module.py` 的 `measure_fn=` 参数

## 4. 关键限制：**COCO 预训练权重不含羽毛球类别**

- `yolo11n.pt` / `yolov8n.pt` 是 COCO 80 类权重，**没有 shuttlecock 类**；最接近的是 `sports ball`，对羽毛球（白裙+软木、高速模糊、小目标）不可靠。
- 因此这两个权重的用途是：①**微调起点**；②临时弱代理（不建议用于任何精度结论）。
- 要真正检测羽毛球必须先做：**整流/同步 → 采集数据 → 标注（或合成数据扩展）→ 微调 → 验证**。
- 合成数据可用现有 `perception/synthetic_detector.py`（解析投影 + 确定性噪声）与 Isaac 渲染生成；标注需覆盖：视场边缘、运动模糊、遮挡、不同距离与光照。

## 5. 环境策略（避免破坏 Isaac Lab）

- **未安装**到 `env_isaaclab`（Isaac Lab 虚拟环境需保持稳定；ultralytics 会引入 opencv/polars 等依赖）。
- 建议后续用**独立虚拟环境**（例如 `env_vision`）安装：
```bash
cd /home/T7/ojh/robot_sim
python3 -m venv --system-site-packages env_vision   # 复用已装 torch，避免重复下载
./env_vision/bin/pip install assets/external/_staging/F_yolo/wheels/ultralytics-8.4.150-py3-none-any.whl
```
- 任何推理调用必须经 `StereoPerceptionModule` 注入，**不得**让感知层直接依赖检测器实现（保持层边界，见 `ROBOT_BRAIN.md` S12）。

## 6. 版本记录

| 组件 | 版本 |
|---|---|
| 权重 | YOLO11n（v8.3.0 发布资产）、YOLOv8n（v8.3.0 发布资产） |
| ultralytics 包 | 8.4.150（wheel） |
| 运行环境（如需） | 待建 `env_vision`；现有 Isaac 环境 torch 2.10.0+cu128 / numpy 见 `outputs/reports/contracts.md5` 同期记录 |
