# 3D-render ETH probe —— 一次性 Spike（计划 + 资产审计 + 阻塞）

日期：2026-10-03　｜　性质：Spike，**不是**正式数据集　｜　本轮**未训练任何模型**

## 0. 当前状态：**已就位，等 GPU 窗口**

连接已恢复，走 Tailscale（见 `docs/REMOTE_CONNECT_PLAYBOOK.md`）：`ssh dgut@jxxy.taildd42cc.ts.net`。
校园网 IP `172.31.68.251` 不可用 —— 之前的 "TCP 22 超时" 是这个原因，**与磁盘/显存无关**。

### 0.1 远端就位核验（2026-10-03 08:3x UTC，实测）

| 项 | 结果 |
|---|---|
| 5 个脚本已上传到 `/home/T7/ojh/robot_sim/{tools,scripts/simulation}/` | scp 上传，**两端 sha256 逐一比对一致** |
| `eval_eth_official_baseline.py` 导入链 | `IMPORT_OK 1024 0.7 300 0.001 0.25` —— 与审计时同一套常量，**未修改该文件** |
| 解释器 | `/home/T7/public/miniconda3/bin/python` + `PYTHONPATH=/home/T7/ojh/robot_sim/experiments/yolo26_p2_ab/_wheel_extract` → ultralytics 8.4.150 |
| ETH ckpt | `/home/T7/dgut/robot_sim/eth_official_code/shuttle_detection/runs/final-model/best.pt` = **134,312,133 B**（与记录一致） |
| Isaac env | `/home/T7/ojh/robot_sim/env_isaaclab`（py 3.12.13 / torch 2.10.0+cu128 / cuda True） |
| 资源三件（usd/visual usd/glb） | 远端字节数与本地完全一致（6622 / 158320 / 308444） |

### 0.2 本轮的两次传输事故（与 playbook 第 3.2 节完全吻合，已复现）

| 我做的事 | 后果 | 正确做法 |
|---|---|---|
| 把 base64 经控制台管道再 `base64 -d` | 大文件被按宽度折行 → `base64: invalid input`；小文件侥幸通过 | **永远 scp** |
| 用 PowerShell 管道把脚本喂给 `bash -s`（CRLF） | 远端落成文件名带 `\r` 的文件；多行脚本 `unexpected end of file` | 本地 write(LF) → scp → `ssh "bash /路径.sh"` |

**遗留：5 个文件名带 `\r` 的错误副本**（`tools/{eval_eth_official_baseline,audit_localization_yolo26_v1,audit_bbox_height_bias,probe_eth_on_3d_render}.py\r`、
`scripts/simulation/render_shuttle_3d_probe.py\r`），内容与正确文件相同，是我 07:13 那次失败上传的残骸。**等你点头我删除**（playbook 第 9 节：删远端文件先问）。

### 0.3 唯一的运行阻塞：GPU 正被占满

```
nvidia-smi: 15,398 MiB / 49,140 MiB, utilization 99 %
计算进程: pid 286773 /isaac-sim/kit/kit                  2,490 MiB
          pid 358317 /isaac-sim/kit/python/bin/python3    5,508 MiB
          pid 2805596 /home/T7/public/miniconda3/bin/python 7,272 MiB  ← 算法窗口 A 的 full_e50 训练，预计 11:35Z 结束
```

**显存还有 ~34 GB，够 Isaac 用；但利用率 99% 意味着要和别人抢 SM。**
Isaac 启动本身约 23.6 min（CPU/磁盘为主），真正吃 GPU 的是 30 帧渲染（0.11 s/帧）。
是否现在抢跑，等你一句话。

### 0.4 远端原始阻塞记录（保留，供对照）

（LAN IP 路径当时不可达，见上；以下为当时的记录）

这个实验的全部必需件都在远端主机 `dgut@172.31.68.251`（hostname `jxxy`）上，而**该主机现在连不上**：

```
$ ssh -o BatchMode=yes -o ConnectTimeout=10 dgut@172.31.68.251 'hostname'
ssh: connect to host 172.31.68.251 port 22: Connection timed out      (attempt 1)
ssh: connect to host 172.31.68.251 port 22: Connection timed out      (attempt 2)
ssh: connect to host 172.31.68.251 port 22: Connection timed out      (attempt 3)
$ Test-NetConnection -ComputerName 172.31.68.251 -Port 22
TcpTestSucceeded : False
```

同一会话早些时候（本轮之前）同一条命令**是通的**（`hostname` → `jxxy`、`dgut`、DSH 0.2.0-rc.2、GPU 24,600/49,140 MiB）。
所以这是**新的可达性故障**，不是配置错误。**未执行任何渲染、未跑任何检测。**

| 必需件 | 位置 | 本地有吗 |
|---|---|---|
| Isaac Sim 6.0.1.0（pip，`env_isaaclab`） | 远端 `/home/T7/dgut/robot_sim/` | **没有**（本地只有 RTX 4060 Laptop） |
| ETH 官方 YOLOv8s `best.pt`（134,312,133 B，sha256 `f1aea7de…d6d`） | 远端 `/home/T7/dgut/robot_sim/eth_official_code/shuttle_detection/runs/final-model/best.pt` | **没有** —— 本地全盘只找到我们自己的 `stageA_best.pt` / `stageB_best.pt` / `yolo26s.pt` |
| 羽毛球 GLB | 本地 `assets/external/_staging/D_racket_shuttle/original/…glb`（308,444 B） | **有** |
| ETH 数据集（val 用） | 本地 `D:\_eth_data\eth_shuttle_detection` | **有** |

## 1. 资产审计（**已实测**，不依赖远端）

直接解析 GLB 容器 + JSON chunk（自写 30 行脚本，非推测）：

### 1.1 材质：羽毛球部分**没有贴图，球头和羽毛共用同一个纯白材质**

```json
materials[0] = {"doubleSided": true, "name": "White",
                "pbrMetallicRoughness": {"metallicFactor": 0.0, "roughnessFactor": 0.6}}
materials[1] = {"doubleSided": true, "name": "Strings",
                "pbrMetallicRoughness": {"baseColorTexture": {"index": 0},
                                         "metallicFactor": 0.0, "roughnessFactor": 0.6}}
```

| 事实 | 值 | 含义 |
|---|---|---|
| `White` 有 baseColorFactor 吗 | **没有** | 按 glTF 规范 baseColor 默认 `[1,1,1,1]` = **纯白** |
| `White` 有 baseColorTexture 吗 | **没有** | 无法贴图 |
| `White` 的 metallic / roughness | **0.0 / 0.6** | 不反光、半粗糙 |
| `Obj_Feather_0` 用哪个材质 | **`White`（index 0）** | — |
| `Obj_Cork_0` 用哪个材质 | **`White`（index 0）** | **球头和羽毛共用同一个材质** |
| 顶点属性 | 两个 mesh 都只有 `NORMAL` + `POSITION` | **没有 TEXCOORD** → 结构上也贴不了图 |
| 唯一贴图 | `textures[0] → images[0]`（1 张 PNG），只被 `Strings` 使用 | 属于**球拍线**，与羽毛球无关 |

> **所以「优先保留原材质」这条已经满足，但要如实记录结论：原材质就是纯白。**
> 球头不是一个深色软木 —— **资产层就没有给球头单独的颜色**。
> 这解释了历次观察到的「球头看不见」，并且把它从「我们渲染器的 albedo 选择」升格为**资产事实**。

### 1.2 灯光：**GLB 里没有任何灯光**

| 检查 | 结果 |
|---|---|
| `extensionsUsed` / `extensionsRequired` | **均为 None** → 没有 `KHR_lights_punctual` |
| `nodes[2] "Lamp"` | 只有一个 `matrix` 变换 + 一个空壳子节点 `nodes[3]`，**没有 light 扩展** |

→ **渲染时的一切灯光都是我们加的，必须逐帧记录。**

### 1.3 几何与出处

| 项 | 值 |
|---|---|
| 羽毛 `Obj_Feather_0` | POSITION 3490 顶点 / 1920 三角形；bbox `[-0.03094,-0.03094,0.00064] .. [0.03094,0.03094,0.07802]` |
| 球头 `Obj_Cork_0` | 4506 索引 / 3 = 1502 三角形 |
| 合并尺寸 | **61.88 × 61.88 × 77.38 mm**（直接读 accessor min/max） |
| 资产出处 | Sketchfab `game_travel`，标题 *Badminton Racket And Shuttlecock (Low Poly)* |
| 许可证 | **CC-BY-4.0** |
| GLB 里另外还有什么 | `Lamp`、**两把球拍**（`Obj_Racket_0` / `Obj_Racket.001_0`）+ 两根线（`Strings` / `Strings.001`） |
| 羽毛球子树路径 | `/Sketchfab_model/Root/Gp_Shuttle/{Obj_Feather, Obj_Cork}` |

项目现有 USD 入口 `assets/third_party/shuttlecock_visual.usd`（usdc）实测**引用了这个 GLB**，
并定义了 `Looks/White` 的 `UsdPreviewSurface` 着色器，绑定到 `Obj_Feather_0` / `Obj_Cork_0`
（在 usdc 字符串表里读到 `Looks\x00Obj_Feather_0…Cork_0\x00White\x00Material\x00PreviewSurface`）。
**渲染走这个 USD，材质继承链是完整的、不需要我们重做材质。**

## 2. 尺寸口径（必须统一，否则数字没意义）

沿用现有 evaluator 的定义（`tools/audit_localization_yolo26_v1.py:771`）：

```
equiv_size_640 = sqrt(w_px * h_px) * 640 / max(W, H)
```

即按**原图像素**量出的等效尺寸，再按长边归一到 640 空间。因此选渲染分辨率就决定了换算：

| 目标 equiv640 | 若渲染 960×960 需要的像素 | 入网（letterbox 到 1024）后的像素 |
|---|---|---|
| **24 px** | 36 px | 38.4 px |
| **16 px** | 24 px | 25.6 px |
| **12 px** | 18 px | 19.2 px |
| 10 px | 15 px | 16.0 px |
| 8 px | 12 px | 12.8 px |
| 6 px | 9 px | 9.6 px |

**采用 960×960 渲染**（与现有冻结集 `controlled_capability` 的主导分辨率一致，便于和已有数字对照）。
这是本轮我替你做的决定之一 —— 若你要改成 1024×1024（与 imgsz 1:1、少一次重采样），一句话我改。

## 3. 交付脚本（已写，仅通过语法检查）

| 文件 | 作用 | 状态 |
|---|---|---|
| `scripts/simulation/render_shuttle_3d_probe.py` | Isaac Sim 真实渲染：地面 + key/dome 灯 + 相机，**逐帧闭环标定相机距离**打到目标 equiv640；GT 从语义分割量出真实足迹；**交付图像是渲染器原始 RGB，无 mask 合成** | `py_compile` 通过；**未执行**（远端不可达） |
| `tools/probe_eth_on_3d_render.py` | **import**（不修改）`tools/eval_eth_official_baseline.py` 的 `load_eth_model` 与常量，按 `imgsz=1024 / NMS 0.7 / max_det 300 / conf≥0.001` 推理；先校验 ckpt sha256 与 134,312,133 B，不符**拒绝运行** | `py_compile` 通过；**未执行** |

### 3.1 渲染脚本遵守与规避的约束

| 用户禁令 | 本脚本如何遵守 |
|---|---|
| NumPy/software rasterizer | 用 Isaac Sim（IsaacLab `Camera` + RTX 渲染），本地没有一行光栅化 |
| 二值 mask 渲染 | 交付图像 = `cam.data.output["rgb"]` 原始帧。语义分割**只用于量测 GT 足迹**，绝不参与成像 |
| `img[mask] = 白色` | 不存在 |
| 旧 synthetic compositing 管线 | 不调用 `tools/shuttle_render.py`、不调用 `composite_isaac_pool.py` |
| 改 mesh 几何 / 为提检测率改形状 | 只 `AddReference` 现成 USD，不写任何几何 |
| 激进 domain randomization | 姿态自由，roll 只 ±20°，灯光方向小幅扰动，无纹理/材质/背景随机化 |

## 4. 执行计划（远端恢复后按此跑）

```bash
# 0) 上传两个脚本到远端 /home/T7/dgut/robot_sim/（scripts/simulation/ 与 tools/）
# 1) Pilot：24/16/12 px 各 10 张 = 30 张。一次 Isaac 启动全部渲完（启动 ~23.6 min，之后 0.11 s/帧）
cd /home/T7/dgut/robot_sim
PYTHONPATH=$(ls -d IsaacLab/source/* | tr '\n' ':') \
  env_isaaclab/bin/python scripts/simulation/render_shuttle_3d_probe.py \
    --out outputs/shuttle_capability/three_d_render_probe \
    --imgsz 960 --sizes 24 16 12 --n-per-size 10 --seed 20261003

# 2) 检测（不训练、不改 evaluator）
python tools/probe_eth_on_3d_render.py \
  --probe-dir outputs/shuttle_capability/three_d_render_probe \
  --ckpt eth_official_code/shuttle_detection/runs/final-model/best.pt
```

**Pilot 判定（按你给的三分支）**：24 px 多数命中 → 继续第二轮 6/8/10/12/16/24 px × 20 = 120 张；
24 px 几乎全灭且 median conf 接近 0.001–0.01 → 停止，结论是 appearance gap 而非小目标；
有预测、conf 不低但 IoU 低 → 先查标签/投影对齐，不判资产。

## 5. 需要你定的事

1. **等远端还是走本地兜底？** 本地兜底需要 (a) 重新下载 ETH 官方 `best.pt` 并用 sha256 校验，
   (b) 换一个本地真实渲染器（pyrender/OpenGL 或 Blender，都不是项目现有 Isaac Sim）。
   这会偏离你「优先使用项目现有 Isaac Sim」的要求，**我不擅自做**。
2. **渲染分辨率 960 还是 1024？**（见 §2）
3. **背景用什么？** 当前脚本是「地面 + 灯光 + 羽毛球」的干净 3D 场景。
   如果你要更接近比赛场景，需要在场景里加球网/地板材质 —— 但那已经跨进 domain randomization，我按你的规则没做。
4. **GT 用语义足迹（当前设计）还是投影框？** 你的情况 C 关心二者是否对齐，
   我当前只实现了语义足迹；要加投影框交叉核对说一声。