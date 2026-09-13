# THIRD_PARTY_ASSETS_YOLO.md — YOLO 粗检测器（下载归档）

归档日期：2026-09-13
归档位置：`assets/external/_staging/F_yolo/`（**staging**，非最终资产目录）
下载方式：**本地机器下载 → scp 上传**（原因见第 6 节网络约束）

## 1. 指定模型：YOLO26s（用户确认）

| 文件 | 大小 (bytes) | sha256 | 来源 |
|---|---|---|---|
| `weights/yolo26s.pt` | **20,422,725** | `646f8bc3fe0a656803d95c294f7852321748cb29d13466a1af8862e2db384a1b` | https://github.com/ultralytics/assets/releases/download/v8.4.0/yolo26s.pt |
| `weights/yolo26s.onnx` | **未下载成功** | — | https://github.com/ultralytics/assets/releases/download/v8.4.0/yolo26s.onnx（官方声明 38,291,130 B） |

**完整性证据（本轮实测）**
- 下载字节数与 **GitHub API 的 release 资产声明值逐字节相同**：`20422725 == 20422725`
- 本地下载 sha256 与上传后远端 sha256 完全相同（`646f8bc3…84a1b`）
- `zipfile` 结构校验：**718 条目**，`testzip() → None`（全部条目 CRC 通过）

ONNX 的失败记录：远端直连 278 s 超时；本地重试 `curl: (56) Recv failure: Connection was reset`。
结论：**ONNX 尚未拿到**（部署路径可稍后重试，或改用 `ultralytics` 自行导出）。

## 2. 其他已归档权重（早期下载，非指定）

| 文件 | 大小 (bytes) | sha256 | 说明 |
|---|---|---|---|
| `weights/yolo11n.pt` | 5,613,764 | `0ebbc80d4a7680d14987a577cd21342b65ecfd94632bd9a8da63ae6417644ee1` | v8.3.0；备选（507 条目 CRC OK） |
| `weights/yolov8n.pt` | 6,549,796 | `f59b3d833e2ff32e194b5bb8e08d211dc7c5bdf144b90d2c8412c47ccfc83b36` | v8.3.0；备选（363 条目 CRC OK） |
| `wheels/ultralytics-8.4.150-py3-none-any.whl` | 1,445,702 | `1af11709f0e525e72d8b1b71916c0081de3e63c49ab9ffdbffa995ae6a8c69de` | 包（pip download --no-deps） |

> YOLO26s 是唯一指定模型；如不需要备选，可删 `yolo11n.pt` / `yolov8n.pt`（合计约 12 MB）。

## 3. 许可证（重要，需项目决策）

- **Ultralytics YOLO（代码与官方权重）采用 AGPL-3.0**：网络传染性许可，对外提供服务需按 AGPL 提供源码；闭源商用需 Enterprise License。
- 宽松替代（如需非 copyleft）：YOLOX / RT-DETR / NanoDet-Plus（Apache-2.0）。
- 本仓库**只做下载归档**：未安装到运行环境、未引入代码依赖（见 DEC-028）。

## 4. 在本项目中的角色（依 docs/architecture/01_PERCEPTION.md 第 22–26 节）

YOLO **只负责粗检测**：输出 coarse bbox + confidence + class，回答「羽毛球大概在哪」；不得把 bbox 中心当精确球心。
链路：rectified image → letterbox/resize → YOLO26s → confidence 过滤 → NMS → 候选框 → **显式还原全幅整流图坐标** → ROI → 亚像素质心 → 双目匹配 → 三角化。

仓库内已就绪、可直接对接：
- ROI 门与坐标约定：`src/badminton_brain/perception/stereo_geometry.py`（roi_gate、court_box_gate）
- 亚像素质心（含空窗哨兵，DEC-026）：subpixel_centroid / CentroidResult
- 三角化与标定占位（TEMP / REQUIRES_CALIBRATION）：triangulate、CameraIntrinsics
- 感知层唯一注入点：`perception/perception_module.py` 的 measure_fn=

## 5. 关键限制：COCO 预训练权重不含羽毛球类

- yolo26s.pt 是 COCO 80 类权重，**没有 shuttlecock 类**；最接近的是 sports ball，对羽毛球不可靠。
- 微调前的用途仅：①微调起点；②临时弱代理。**微调与验证完成前的任何检测精度声明均无效。**
- 微调数据：整流/同步后的双目图像，覆盖视场边缘、运动模糊、遮挡、不同距离与光照；可用 synthetic_detector.py 与 Isaac 渲染扩增。

## 6. 网络约束（实测，后续照此做）

- 远端 jxxy 可访问 api.github.com 与 pypi.org，但 **release-assets.githubusercontent.com 时通时断**：5–6 MB 小文件成功，20 MB 的 yolo26s.pt 直连 278 s 超时失败；38 MB 的 onnx 连本地重试也失败。
- 处置：**本地下载 → scp 上传 → 两端核对字节数与 sha256**（yolo26s.pt 已按此完成）。
- 已登记为 ISSUE-016。

## 7. 环境策略与后续

- **未安装** ultralytics（保护 env_isaaclab）。建议独立环境：
```bash
cd /home/T7/ojh/robot_sim
python3 -m venv --system-site-packages env_vision
./env_vision/bin/pip install assets/external/_staging/F_yolo/wheels/ultralytics-8.4.150-py3-none-any.whl
```
  YOLO26 资产随 v8.4.0 发布，需 ultralytics >= 8.4.x；已归档包 8.4.150，**尚未实跑推理验证**。
- 任何推理必须经 StereoPerceptionModule(measure_fn=...) 注入，保持层边界（ROBOT_BRAIN.md S12）。
