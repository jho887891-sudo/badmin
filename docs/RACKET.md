# 球拍资产生成器 v0.1

这套文件用于低速羽毛球人机对打机器人中的 PiPER 固定球拍资产。目标不是做一个“看起来像球拍”的临时模型，而是把 **标准/厂家公开值、实物测量值、视觉模型、碰撞模型、安装变换** 分开管理。

## 1. 已经可以真实确定的值

### BWF 法规上限

BWF Laws of Badminton 对球拍给出的核心几何限制：

- 整体长度不得超过 `680 mm`。
- 整体宽度不得超过 `230 mm`。
- 常规击球线床长度不得超过 `280 mm`。
- 常规击球线床宽度不得超过 `220 mm`。

来源：

https://system.bwfbadminton.com/documents/folder_1_81/Statutes/CHAPTER-4---RULES-OF-THE-GAME/SECTION%204.1-%20Laws%20of%20Badminton.pdf

这些是**法规最大值**，不是具体某支球拍的尺寸，因此代码只拿它们做合法性检查。

### 厂家参考产品

当前 `racket.yaml` 记录 VICTOR `THRUSTER Onigiri / TK-ONIGIRI` 作为可追溯参考产品：

- 厂家公布长度：`675 mm`。
- 厂家公布轴径：`6.4 mm SHAFT`。
- 可选重量规格：`4U`。
- VICTOR 对 4U 的定义：未穿线 `80.0–84.9 g`。

产品页：

https://www.victorsport.com/product/20231/thruster-k-onigiri

VICTOR 重量等级说明：

https://de.victor-europe.com/page/laster_engved_code

注意：`4U = 80.0–84.9 g` 是重量等级范围，**不是我们最终装到机械臂上的那支球拍质量**。

## 2. 为什么 `measured_unit` 默认是 null

真正影响机械臂动力学的是：

```text
具体这一支球拍
+ 实际穿线
+ 实际手胶
+ 实际固定夹具/适配器中归到球拍资产的部分
```

厂家公开页面没有给出这一整套安装状态的：

- 实际总质量；
- 实际 COM；
- 实际惯量；
- 精确拍头外轮廓；
- 精确线床尺寸；
- 精确握柄尺寸。

所以 `racket.yaml` 明确保留：

```yaml
measured_unit:
  status: REQUIRES_MEASUREMENT
  installed_mass_kg: null
  center_of_mass_from_tcp_m: null
  diagonal_inertia_kg_m2: null
  ...
```

最终 USD 构建时如果这些数据没填，程序直接拒绝生成，而不是偷偷使用“4U 中值”或 BWF 最大尺寸。

## 3. 实物需要测什么

以 `racket_tcp` 为原点，+Z 从握柄朝拍头：

1. `installed_mass_kg`
   - 把最终会跟 link6 一起运动的球拍总成称重。
2. `center_of_mass_from_tcp_m`
   - 在两个支点/刀口上找平衡点；得到沿 +Z 的平衡位置。
3. `diagonal_inertia_kg_m2`
   - 最严格做法是摆振/双丝摆法实测三轴惯量。
   - 如果后续允许“由实测几何+质量分布拟合惯量”，应作为单独模式实现并明确标注 derived，不应冒充直接测量。
4. `handle_size_x_m`, `handle_size_y_m`, `handle_length_m`
   - 卡尺测量最终安装后的握柄/夹持段。
5. `head_outer_length_m`, `head_outer_width_m`
   - 拍框最外轮廓。
6. `stringbed_length_m`, `stringbed_width_m`
   - 实际有效线床范围。
7. `frame_tube_diameter_m`
   - 用于分段拍框碰撞代理。
8. `stringbed_thickness_m`
   - 这是碰撞模型的有效厚度。若将真实线床柔顺性建模成 compliant contact，应另行标定，不应把网线直径直接当作等效接触厚度。

## 4. 坐标系

```text
racket_tcp: 机械安装/TCP 原点

+X_racket : 拍面法向，标准击球姿态指向对手
+Z_racket : 握柄 -> 拍头
+Y_racket : 右手系
```

`racket_contact_frame` 位于实测拍头/线床几何中心，方向与 `racket_tcp` 相同。

代码提供：

```text
T_tcp_contact
```

PiPER 包装器提供：

```text
T_link6_tcp
```

于是：

```text
T_link6_contact = T_link6_tcp @ T_tcp_contact
```

## 5. 碰撞模型

最终物理碰撞不是一整个 box，也不是把视觉三角网格直接当动态碰撞：

```text
Physics
├── HandleCollider        盒状实测包络
├── ShaftCollider         圆柱，轴径来自厂家/实测
├── HeadFrameColliders    24 段椭圆拍框管
└── StringBedCollider     32 边薄椭圆棱柱
```

这样做的原因是：

- 保留真实拍框/线床尺寸；
- 球真正撞的是线床，因此线床必须独立；
- 分段低复杂度 collider 比 5k+ triangle 动态接触稳定，更适合 240 Hz 和 128-env；
- 被舍弃的是孔钉、漆面和细小框截面细节，不是关键击球几何。

## 6. 视觉模型

沿用已经选定的开放许可模型：

- Title: `Badminton Racket And Shuttlecock (Low Poly)`
- Author: `game_travel`
- License: `CC Attribution`
- UID: `795e4cca2e544d7ab86a8e5c7ded2362`

https://sketchfab.com/3d-models/badminton-racket-and-shuttlecock-low-poly-795e4cca2e544d7ab86a8e5c7ded2362

把球拍单独提取后转换为：

```text
assets/third_party/racket_visual.usd
```

视觉 mesh 应按实物测量尺寸重新缩放/对齐，不能仅因为原模型“看起来像”就假定比例精确。

## 7. PiPER 固定安装

`build_piper_with_racket.py` 不修改冻结的：

```text
assets/piper_no_gripper.usd
```

它创建新的组合 USD：

```text
/Robot
├── /Piper
├── /Racket
└── /Joints/racket_fixed_joint
```

`FixedJoint`：

```text
body0 = /Robot/Piper/Geometry/link6
body1 = /Robot/Racket
```

安装变换：

```text
T_link6_tcp
```

默认仍然 `REQUIRES_MEASUREMENT`。必须在真实固定夹具装好后测出平移和四元数再冻结。

## 8. 普通 Python 检查

```bash
python build_racket.py \
  --config racket.yaml \
  --validate-only \
  --print-summary

python build_piper_with_racket.py \
  --config racket.yaml \
  --validate-only \
  --print-summary

python test_racket.py -v
python test_piper_with_racket.py -v
```

默认配置会通过“来源/结构校验”，并明确显示最终物理仍被实测项阻塞。

如果使用：

```bash
python build_racket.py --config racket.yaml --validate-only --final
```

在未填实测参数时**应该失败**。这是设计行为。

## 9. Isaac Sim 中生成最终球拍

准备好：

```text
assets/third_party/racket_visual.usd
```

并把 `measured_unit` 填成 `MEASURED` 后：

```bash
python build_racket.py \
  --config racket.yaml \
  --output assets/racket/racket.usd \
  --final \
  --print-summary
```

然后填好真实 `T_link6_tcp`：

```bash
python build_piper_with_racket.py \
  --config racket.yaml \
  --output assets/robots/piper_with_racket.usd \
  --final \
  --print-summary
```

必须在 Isaac Sim Python / 含 `pxr` 的环境执行实际 USD authoring。

## 10. 接触参数

`shuttle_racket` 仍然是：

```yaml
contact_pair_parameters:
  status: REQUIRES_PAIR_CALIBRATION
  shuttle_racket: null
```

原因和羽毛球资产相同：羽毛球-线床 restitution / friction / compliance 是**接触对参数**，不是仅从球拍型号就能得到的唯一常数。最终应通过真实低速碰撞实验标定。
