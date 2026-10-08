# 合成图片是怎么做出来的 —— 流水线实测说明

一句话：**我们自己写了一个软件光栅化渲染器（numpy z-buffer）**，把真实羽毛球 GLB 网格渲染成带精确 alpha 的 RGBA 小图，
再 alpha 合成到真实照片背景上，最后按固定顺序加阴影 / 运动模糊 / 噪声，存成 JPEG q92。
**全程没有用 Isaac Sim，也没有用任何图形引擎。**

## 0. 资产真相：几何是真的，外观是代码里造的

`tools/shuttle_render.py:2-15` 的 docstring 是 2026-09-14 对 GLB 的实测结论：

| 项 | 实测值 |
|---|---|
| 羽毛球网格 | `Obj_Feather`（3490 v / 1920 f）+ `Obj_Cork`（3008 v / 1502 f） |
| 顶点属性 | 有 POSITION + NORMAL，**没有 TEXCOORD_0** |
| 材质 | `material[0] 'White'` 的 `baseColorFactor=None`，**没有 baseColorTexture** |
| 唯一贴图 | 属于 `material[1] 'Strings'`（球拍线），与羽毛球无关 |
| 合并 bbox | **61.9 × 61.9 × 77.8 mm**（实测，非假设） |

所以 **贴图外观从这份资产里根本做不出来**，只能自己建模。这就是下面这段的来历（`shuttle_render.py:34-35`）：

```python
FEATHER_ALBEDO = np.array([0.93, 0.94, 0.90])   # 羽毛
CORK_ALBEDO    = np.array([0.86, 0.83, 0.74])   # 球头（软木）
```

> **这回答了我上一轮标为「待确认」的问题**：合成图里球头几乎看不见，不是渲染 bug，
> 而是 `CORK_ALBEDO` 被给成了一个**接近白的米色**。真球的球头是深红/黑色软木。
> 这是当前实现的一个**明确选择**，有行号可查。

## 1. 渲染：`render_shuttle()` —— `tools/shuttle_render.py:216-249`

1. **超采样**：默认 `ss=3`，先渲染到 3 倍画布，再 box-filter 降采样（`:247-248`）
   → 边缘得到**分数覆盖的 alpha**，而不是硬二值轮廓。
2. **透视投影 + z-buffer**，逐三角形做重心插值（`:163-213`）。
3. 法线用**资产自带的顶点法线**做重心插值（`:201-204`，注释明确写了「不是面法线」）。
4. 每个像素的法线朝相机翻转（`:205-208`）。

### 着色 `shade()` —— `:141-160`

```python
luminance = ambient + key * wrapped + translucency * transmitted
# AMBIENT = 0.56   KEY = 0.44   WRAP = 0.5
# FEATHER_TRANSLUCENCY = 0.30   CORK_TRANSLUCENCY = 0.0
```

- **wrapped diffuse**（`(ndl + 0.5) / 1.5`）而不是 `max(0, ndl)` → 天然双面，羽毛是薄的
- **环境光地板**：`:37-39` 记着原因 —— ambient 取 0.34 时，完全背光的羽毛面渲染成 80/255，
  **比它所在的中灰背景还暗**，在 3–7 px 尺度上表现为「黑斑」。所以设了 0.56 的地板，
  保证无论朝向多不利，白球都不会渲染成黑。
- **羽毛透光**：`transmitted = clip(-ndl)`，背光面被透射项提亮 —— 这是薄羽毛裙的物理行为。

## 2. 尺寸标定：`calibrate_distance()` —— `:308-349`（这套流水线最关键的一步）

**不用公式** `fx * extent / target_px`，因为对这个资产会错两次（`:313-317`）：
1. 77.8 mm 是**长轴**，投影时会被透视缩短；
2. 稀疏的羽毛裙**实际占面积远小于它的顶点凸包**。

改成**迭代**：渲染 → 量测真实 bbox → 按 `achieved/target` 比例修正距离 → 最多 8 次，容差 5%。
**渲染器自己就是标定真值**。

两个踩过的坑，都写在注释里：
- `:319-322`：`measure_supersample` **必须等于实际出货的 supersample**。用 2× 量测、3× 出货，
  会把 8 px 的目标做成 10 px —— 比不标定还差。
- `:283-305`：曾经写死 0.3 m 的最近距离地板，结果**所有渲染被静默卡在 ~166 px 上限**，
  整个近场区间（真实正样本有到 1578 px 的）根本渲染不出来。现在按旋转后网格的前向半径 ×1.2 算。
  `:294-295` 还记着残余边界：fx=700 大约只能到 900 px，再大要加大焦距。

## 3. 合成：`composite()` —— `:498-528`，固定四步顺序

| 步 | 做什么 | 关键细节 |
|---|---|---|
| 1 | **alpha 合成** `out = bg*(1-a) + rgb*a` | `:509-510` |
| 2 | **阴影**（可选） | 把 alpha 平移 `(offset, offset)` 再高斯模糊；**并乘 `(alpha <= 0)`** —— 阴影绝不落回物体自身（`:521`）。offset = `clip(0.8*extent, 1, 8)`，blur = `clip(0.35*extent, 0.6, 4.0)`，**随物体尺寸缩放**：`:513-514` 记着固定 (6,8)/σ4 会把 7 px 的球压暗 26%、更小的直接吞掉 |
| 3 | **运动模糊**（`motion_px >= 1.5`） | `_motion_kernel()`（`:488-495`）生成一条按 `motion_angle_deg` 旋转的线核，`filter2D` |
| 4 | **高斯噪声** | `noise_sigma`，**用记录下来的 `noise_seed` 播种** → 完全确定性（`:525-527`） |

`:502-506` 的承诺：**给定参数集完全确定** —— 一条 manifest 记录能精确复现它自己的那张图。

## 4. 一个样本画什么：`render_sample()` —— `:531-592`

| 量 | 分布 | 行号 |
|---|---|---|
| 背景 | 从底图**随机裁** 960×960（源图不够大就先放大） | `:472-485` |
| 姿态 yaw/pitch/roll | 各 `uniform(-π, π)`，**全向均匀** | `:543` |
| 位置 | `uniform(0.05, 0.95)` × 宽 / 高 | `:545-546` |
| 光照方位角 | `uniform(0, 2π)`，方向 `[cos*0.45, -0.55, -0.70]` 归一化 | `:549-551` |
| 阴影增益 | `uniform(0.0, 0.35)` | `:554` |
| 运动像素 | `choice((0, 0, 1, 2, 4, 7))` | `:555` |
| 噪声 σ | `uniform(1.0, 5.0)` | `:557` |

**所有抽到的值都写进 manifest**。`:584-588` 特别注明：`shadow_gain` / `motion_px` / `motion_angle_deg` /
`noise_sigma` / `noise_seed` **必须精确、不能四舍五入**，否则「记录可复现图像」这条保证会静默失效。

## 5. 标签（GT box）怎么来的 —— 精确，不是目测

- `alpha >= 1e-6` → mask。`SUPPORT_COVERAGE = 1e-6`（`:52-54`）：**GT 框是物体的「足印」**，
  凡是被碰到的像素都算。注释写明：如果要求「半覆盖」，在 ~4 px 以下会把目标**整个抹掉** ——
  而那正是这个数据集存在的意义所在。
- `yolo_bbox_from_mask()`（`:257-268`）出紧框 → 归一化写成 `0 cx cy w h`
- `equiv_size_px = sqrt(bw*W * bh*H)`（`:573`）

## 6. 各个池分别用什么参数

| 池 | 脚本 | n | 目标像素 | imgsz | ss | 背景 | 命名 |
|---|---|---|---|---|---|---|---|
| HiFi 正样本 | `experiments/shuttle_capability/14_render_hifi.py` | 400 | 2–32，median 9（log-normal σ=0.5） | 960 | 3 | `bg_train` 25 张真实照片 | `train_*` |
| HiFi 验证 | 同上 `--split val` | 144 | 同上 | 960 | 3 | `bg_val` 6 张 | `val/` |
| 近场补样 | `experiments/shuttle_detection/02_add_nearfield_training.py` | 100+40 | **40–512**（log-uniform） | 960 | **<128 用 3，否则 1** | `bg_train` | `nf_train_*` |
| 普通负样本 | `experiments/shuttle_detection/01_build_negatives.py` | 100 | — | 960 | — | `bg_train` 裁剪，**空标签** | `neg_train_*` |
| S3D 能力集 | `experiments/shuttle_capability/01_gen_synthetic.py` | 84 | 受控 | 1280 | — | **纯色 / 渐变**（flat、court_green） | `syn_*` |
| P3 真实背景 | `experiments/shuttle_capability/13_p3_real_bg.py` | 160 | `clip(log-normal(9, 0.55), 2, 60)` | 1280 | — | `real_images/backgrounds` 30 张 | `p3_*` |
| Isaac 池 | `tools/render_pool_isaac.py` → `tools/composite_isaac_pool.py` | 400 → 382 | 3–800 | 960 | — | `bg_train` | `isaac_train_*` |
| 硬负样本 | `experiments/shuttle_detection/04/07_add_hard_negatives*.py` | 33 / 53 | — | 960 | — | 真实无球照片，空标签 | `hardneg*_train_*` |

注意 `01_gen_synthetic.py` 是**更早的另一条流水线**：它只光栅化**二值 silhouette mask**（`:54-60`），
没有分数 alpha、没有建模材质，而且背景是纯色/渐变。这与 `shuttle_render.py` 不是同一套。

## 7. 三道防泄漏闸门 —— `14_render_hifi.py:68-103`

| # | 检查 | 不通过 |
|---|---|---|
| 1 | train / val 背景池**文件名交集必须为空** | `return 3` |
| 2 | `find_blank_backgrounds()` 抓「没有场景」的底图 | `return 6` |
| 3 | `find_leaked_backgrounds()` 与冻结集比对 | `return 5` |

**闸门 2 的原因**（`:412-418`）：透明 PNG 被压平后变成纯黑，`tbg_018.jpg` 和 `tbg_005.png` 就是这样混进池子的。

**闸门 3 为什么用相关性而不是文件名**（`:430-437`）：同一张源图**换个名字重编码**就能骗过名字检查 ——
实测 `tbg_011.jpg` 与 `bg_021.jpg` 相关度 **r = 1.0000**。现在按 32×32 灰度归一化后做相关，阈值 0.95。

## 8. 负样本为什么要 `vary()` —— `01_build_negatives.py:33-54`

`prepare_background()` 对「源图 ≤ 目标尺寸」的情况会**放大后返回整张**，于是小背景会产生 N 张**逐字节相同**的负样本。
审计的重复检测抓到：`tbg_026.jpg`(800×465) 和 `tbg_029.jpg`(640×481) 各产生 4 张重复 —— **100 张里浪费 8 张**。
现在用 50% 概率水平翻转 + 0.75–1.0 随机缩放打散，仍由种子驱动、仍可复现。

## 9. 已知的开放问题（本轮看到的）

| 问题 | 状态 |
|---|---|
| 球头渲染成浅米色（`CORK_ALBEDO`）而非真球的深红/黑 | **确认为实现选择**，不是 bug；是否要改成深色是设计决策 |
| 图 `nf_train_00036.jpg` 里物体**半透明 + 阴影明显错位成独立椭圆** | 未查；需要单独对着 `composite()` 的阴影分支和 alpha 分布核 |
| 训练集正样本中位 8.9 px，@640 入网后 <6 px 占 31.7% | 已记录在 `DATASET_RESOLUTION.md` |
| 整个流水线**未在真机域做过端到端验证**（真实照片 recall 0.800 是另一条线） | 开放 |