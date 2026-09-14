# P4-A REPORT — 校准合成训练集（SYNTHETIC_HIFI_3D）

日期：2026-09-14　范围：**训练数据生成器 + 数据集构建**（**不含** nc=1 训练本身）

> **标签纪律**：本集合 `source_type = SYNTHETIC_HIFI_3D`。它是**训练数据**，
> **不是**能力度量，**不得**用于替代真实视频验收（协议 §3/§24/§35）；
> 与冻结能力集（SYNTHETIC_3D 84 / SYNTHETIC_ON_REAL_BG 160 / REAL_IMAGE 10）严格隔离。

## 0. 结论

- 产出可直接用于 nc=1 训练的集合：**train 400 张 + val 120 张**，逐张带精确 GT（YOLO 单类）
- 实测像素尺寸覆盖 **2.45 – 32.00 px**（median 8.94），尺寸控制误差中位 **0.4%**、90 分位 4.1%
- 三道数据隔离门禁全部通过，与冻结测试集重复背景 **0**
- `tools/shuttle_render.py` **51 项单元测试全绿**；**3 个关键变异全部被测试杀死**（DEC-020 纪律）
- 从 manifest 记录重渲染可**逐位复现**存盘图像（差值 0.0000，去除 JPEG 变量后）

## 1. 关键更正：这个 3D 模型**没有**外观数据

针对「为什么不用羽毛球的三维模型」，实测 GLB 得到确定答案（这改变了问题的前提）：

| 事实 | 实测值 |
|---|---|
| 羽毛球网格 | `Obj_Feather` 3490 顶点 / 1920 面 + `Obj_Cork` 3008 顶点 / 1502 面 = **6498 / 3422** |
| 属性 | 两者都**只有** `POSITION` + `NORMAL`；**没有 `TEXCOORD_0`（无 UV）** |
| 材质 | `material[0] "White"`：`baseColorFactor=None`、**无 baseColorTexture** |
| GLB 内唯一贴图 | 属于 `material[1] "Strings"`（**球拍线**），与羽毛球无关 |
| 尺寸 | 合并包围盒 61.9 x 61.9 x 77.8 mm；法线单位化；无退化三角面 |

**结论：几何在用（且顶点法线也在用），但"用模型的真实外观"在数据上不存在** —— 没有 UV 就没有贴图映射。
外观只能建模。本管线因此使用：**授权顶点法线**（重心插值 → 逐像素平滑着色，旧渲染器完全没用它、
自己叉乘算平面法线）+ 建模的双面 wrapped 漫反射 + 环境光下限 + 羽毛透光。

## 1.1 项目自建的羽毛球模型是什么，以及为何不能用它当渲染网格

项目**确有**自建羽毛球模型：`tools/build_shuttlecock.py` + `configs/shuttlecock.yaml`
（16 羽毛、BWF 尺寸区间、软木半球凸包 + 裙部开口锥壳、质量分布、气动长度）。

但它的定位是**物理资产**，其视觉体按项目自身策略必须是外链 —— 以下四条均取自项目代码，不是推断：

| 位置 | 内容 | 含义 |
|---|---|---|
| `configs/shuttlecock.yaml` | `visual.mode: EXTERNAL_REFERENCE`；`local_asset_path: assets/third_party/shuttlecock_visual.usd` | 视觉体的唯一真源就是**导入 GLB 的提取物** |
| `build_shuttlecock.py:136-137` | `if visual.get("mode") != "EXTERNAL_REFERENCE": raise ValueError` | 换成别的视觉策略会**直接抛错** |
| `build_shuttlecock.py:337` | `UsdGeom.Imageable(mesh.GetPrim()).MakeInvisible()` | 程序化网格是**碰撞几何，不可见** |
| `build_shuttlecock.py:416-433` | `TEMP_ProceduralDebug`（Sphere + Cone），需 `--allow-temp-procedural-visual` | 程序化视觉只是**临时调试替身**，注释要求 final 前替换 |

**因此用 GLB 做渲染符合项目自身策略，不是绕过模型。** 但该质疑暴露了两个真实缺陷（ISSUE-024）：

1. 渲染器**自带**尺寸常数 `SHUTTLE_LENGTH_M = 0.0778`（硬编码实测值）—— 这正是本仓库登记的失败模式（ISSUE-013）
   → 已改为 `load_shuttle_spec()` 从 `configs/shuttlecock.yaml` 读取（经项目自有的 `load_config`/`build_physics_description`）
2. **此前没有任何检查确认外部视觉体与项目 BWF 模型尺寸一致** —— 换掉视觉体就会静默错标所有样本
   → 已加测试 `test_imported_visual_is_dimensionally_consistent_with_the_model`

**实测一致性（支持"这个视觉体与项目模型对得上"）：**

| 量 | 项目配置模型 | 导入视觉体实测 | 差 |
|---|---|---|---|
| 裙尖直径 | 61.80 mm | 61.90 mm | **0.16%** |
| 总长（软木尖→裙尖） | 79.25 mm | 77.80 mm | 1.83% |

**连带影响：** 标定种子由 0.0778 → 0.07925（+1.86%），实测约 27% 样本最终落点差 ±1px（最大 1.006 px）。
种子只影响迭代起点（终点由实测足迹决定），但为保持"manifest 记录可逐位复现"的契约，**数据集已按新种子重新生成**。

## 2. 交付物

| 文件 | 说明 |
|---|---|
| `tools/shuttle_render.py` | 渲染器单一真源：GLB 部件解析、着色、抗锯齿光栅化、GT 足迹、闭环尺寸标定、合成、泄漏/空图门禁、Unicode I/O |
| `tests/tools/test_shuttle_render.py` | 46 项测试（TDD，无 GPU、无 Isaac） |
| `experiments/shuttle_capability/14_render_hifi.py` | 薄 CLI（argparse + 落盘 + 三道门禁），逻辑全在模块内 |
| `experiments/shuttle_capability/15_check_sheet.py` | 标注抽检图（已修非 ASCII 路径） |
| `outputs/shuttle_capability/train_data/manifest_{train,val}.csv` | 逐样本记录（见 §3） |
| `outputs/shuttle_capability/train_data/bg_train_val_metadata.csv` | 背景来源 + 排除理由 |
| `outputs/shuttle_capability/visualizations/gt_audit_train.png` | 分尺寸桶 GT 目视审计图 |
| `outputs/shuttle_capability/visualizations/bg_pool_audit.png` | 背景池目视审计图 |
| `outputs/shuttle_capability/visualizations/bg_tbg006_real_shuttles.png` | 污染证据（ISSUE-021） |

## 3. 数据集规格（实测）

| 项 | train | val |
|---|---|---|
| 张数 | 400 | 120 |
| 尺寸 min / median / max (px) | 2.45 / 8.94 / **32.86** | 2.45 / 9.16 / **31.40** |
| 尺寸控制比 median / p10 / p90 | 1.003 / 0.960 / 1.041 | 1.005 / 0.961 / 1.040 |
| 落在目标 ±15% 内 | **100.0%** | **100.0%** |
| `<4px` / `4-8px` / `8-16px` / `>=16px` | 3.8% / 39.0% / 49.5% / 7.8% | 4.2% / 35.0% / 49.2% / 11.7% |
| 背景池 | 25 张（真实照片） | 6 张（真实照片） |
| 图像 | 960x960 JPEG q92 | 同 |

**GT 判据（重要裁决）**：GT 框 = **物体足迹**（被羽毛球触碰的像素即属于目标，`SUPPORT_COVERAGE`），
而非「像素覆盖率 ≥50%」。实测在 2 px 目标下**没有任何像素达到 50% 覆盖**（最高 0.44），
用 50% 判据会让 GT 直接消失 —— 恰好是 recall 已经为 0 的区间。manifest 同时记录 `coverage_sum`
与 `solid_px`，供分析方自选判据。

**manifest 字段**：`file, split, source_type, background, target_px, equiv_size_px, distance_m,
bbox_w_px, bbox_h_px, pos_x_px, pos_y_px, coverage_sum, solid_px, yaw_deg, pitch_deg, roll_deg,
light_azimuth_deg, shadow_gain, motion_px, motion_angle_deg, noise_sigma, noise_seed, imgsz, supersample`。
其中 `shadow_gain / motion_px / motion_angle_deg / noise_sigma / noise_seed` **精确存储（不舍入）**，
因为它们可回代 `composite()` 复现图像。

## 4. 数据隔离证据（三道硬门禁，失败即非零退出、不渲染）

1. **split 隔离**：`25 train / 6 val / 0 shared`
2. **冻结集隔离**：`25 candidates vs 30 frozen backgrounds, 0 duplicated`（按**内容**比对，非文件名）
3. **空图隔离**：`find_blank_backgrounds()` 挡住全黑/单色帧

## 5. 本轮发现并修复的缺陷

| 编号 | 缺陷 | 严重度 |
|---|---|---|
| ISSUE-017 | `tbg_011.jpg` 与冻结 P3 集 `bg_021.jpg` **同源**（标题/URL 相同 + 像素相关 r=1.0000） | **高（数据泄漏）** |
| ISSUE-018 | 「受控像素尺寸」从未受控：标称 6/12/24px 实得 4/10/18px | 高 |
| ISSUE-021 | 训练背景池未做目视核验，混入非照片、全黑帧与**含 9+ 真实羽毛球**的照片 | **高（标注污染）** |
| ISSUE-022 | 尺寸标定在 `supersample=2` 测量而 `3` 渲染，8px 目标实得 10px —— 比不标定更差 | 中 |
| ISSUE-023 | 接触阴影盖在物体自己身上（最暗 -26.1%）；白羽球最暗面只有 80/255（比背景还暗） | 中 |
| ISSUE-024 | 渲染器自带尺寸常数；无人校验外部视觉体与项目模型一致 | 中 |
| ISSUE-025 | 记录把 `target_px` 舍入到 3 位小数，回放差 1 个像素 | 低 |
| ISSUE-019 | manifest 漏记 `noise_sigma`、`background` 存成样本名 | 中 |
| ISSUE-020 | Windows 非 ASCII 路径 `cv2.imread` 静默失败；CSV 默认 GBK 编码 | 中 |

## 6. 验证证据

- **单元测试**：`python tests/tools/test_shuttle_render.py` → **Ran 51 tests ... OK**
- **变异测试（DEC-020）**：3 个关键变异全部 **KILLED** ——
  ① 标定打回朴素公式 → 尺寸测试红；② 泄漏门禁阈值抬到 1.01 → 泄漏测试红；
  ③ 记录重新加 `round()` → 可复现性测试红。（首轮 ① 曾 **SURVIVED**，导致发现 ISSUE-022）
- **数据集验证**：`ALL CHECKS PASSED`（计数一致 / 标注合法 / 无空帧 / 背景来源正确 / 池隔离 / 尺寸控制 / 逐位复现）
- **可复现性**：5 个抽样（含此前失败的 train_00399）`mean|stored − replayed| = 0.000000`
- **目视审计**：分 6 个尺寸桶各取中位样本，GT 框 **6/6 贴合**；修复后 6.9/9.5/13.4/19.8px 均为明确发亮的羽毛球
- **背景池审计**：`bg_pool_audit.png` 逐张目视 → 剔除 7 张（详见 ISSUE-021）

## 7. 诚实边界（未做 / 未验证）

1. **未做 nc=1 训练** —— 本报告只交付数据集与生成器；`before/after` 能力对比需在 P4-B 完成
2. **剩余 25+6 张背景未做全分辨率逐张复核**：核验尺度相当于 300 px 缩略接触表。
   `tbg_006` 的真球在该尺度下**可见**（说明方法有效），但仍建议补一次全分辨率核验
3. **外观是建模的，不是测得的**：无 UV/无贴图，羽毛纹理与半透光是参数化近似；
   反照率/光照参数（`AMBIENT=0.56`、`KEY=0.44`）为**未实测参数**，未针对真实相机标定
4. **极小目标固有低对比**：3.5–5 px 时羽毛球只覆盖 1–2 个像素的分数面积，
   实测物体与场景亮度差在暗背景下可为负 —— 这是真实困难情形，不是缺陷，但会限制可学习性
5. **val 只有 6 个背景**：`bg_val` 6 张，val 的多样性低于 train，不宜用作最终泛化结论
6. **远端未同步**：本轮产物在本地 SSOT；同步到 jxxy 尚未执行

## 8. 复现

```bash
# 单元测试
python tests/tools/test_shuttle_render.py

# 生成（三道门禁自动执行；背景池需在位）
python experiments/shuttle_capability/14_render_hifi.py --split train --n 400 --seed 1
python experiments/shuttle_capability/14_render_hifi.py --split val   --n 120 --seed 2

# 标注抽检图
python experiments/shuttle_capability/15_check_sheet.py --repo . --split train
```

背景池与冻结背景集按 `.gitignore` 策略不入库（可复现）；从远端取回：
```bash
scp -r 172.31.68.251:/home/T7/ojh/robot_sim/outputs/shuttle_capability/train_data/bg_train ./outputs/shuttle_capability/train_data/
scp -r 172.31.68.251:/home/T7/ojh/robot_sim/outputs/shuttle_capability/real_images/backgrounds ./outputs/shuttle_capability/real_images/
```
