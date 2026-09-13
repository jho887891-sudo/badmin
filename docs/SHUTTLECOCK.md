# 羽毛球资产 v0.1

这套文件用于低速羽毛球人机对打项目的 Isaac Sim / Isaac Lab 场景与 Robot Brain 轨迹模型。

## 1. 本版本固定的真实/可追溯参数

### BWF 规则边界

- 羽毛数：16。
- 羽毛长度：62–70 mm（从羽毛尖端到球托顶部）。
- 羽毛尖端圆直径：58–68 mm。
- 球托直径：25–28 mm，底部圆形。
- 总质量：4.74–5.50 g。

来源：BWF Statutes, Section 4.1, Laws of Badminton, Law 2.

https://system.bwfbadminton.com/documents/folder_1_81/Statutes/CHAPTER-4---RULES-OF-THE-GAME/SECTION%204.1-%20Laws%20of%20Badminton.pdf

### v0.1 文献代表羽毛球

Darbois Texier et al., *Shuttlecock dynamics*, Procedia Engineering 34 (2012):

- 总质量：5 g。
- cork：3 g。
- feather skirt：2 g。
- 横截面积：30 cm² = 0.003 m²。
- 文中实验天然羽毛球空气动力长度：`L = 6.5 m`。

因此默认任务模型直接采用：

`k = 1 / L = 0.153846153846... m^-1`

而不是把不同论文、不同羽毛球的 `Cd`、面积、质量混在一起。

论文 DOI: 10.1016/j.proeng.2012.04.031

参考全文页面：
https://www.researchgate.net/publication/257724919_Shuttlecock_dynamics

## 2. 选定而非“实测真值”的几何量

BWF 给的是合法范围，不是某一颗具体商品球的唯一尺寸。本配置为了生成一个明确的 BWF 合规代表体，选择：

- feather/skirt axial length：66 mm（BWF 62–70 mm 的中值）。
- cork diameter：26.5 mm（BWF 25–28 mm 的中值）。
- skirt tip diameter：不是取中值，而是由文献横截面积 30 cm² 计算，约 61.804 mm。

这些值是“标准范围内的代表选型”，不是声称某个 Yonex/Victor 具体型号的逐颗卡尺实测。

## 3. 质量属性

没有凭空手填 COM / inertia。

- cork：按 3 g 均匀实心半球代理计算。
- skirt：按 2 g 均匀薄圆台壳积分计算。
- 总 COM：两部分质量加权。
- 总惯量：各自质心惯量 + 平行轴定理。

当前配置得到：

- COM z ≈ +11.978 mm（原点为球托平面中心，+Z 指向裙边）。
- inertia diagonal ≈ `[3.54308e-6, 3.54308e-6, 1.34117e-6] kg·m²`。

这些是“基于真实/文献质量与标准几何推导的物理代理值”，不是未测量实物的直接惯量测试结果。

## 4. 碰撞

`CorkCollider`：半球凸体。

`SkirtColliders`：16 个开中心薄壳凸段。这样不会把羽毛裙错误建成一个实心圆锥。`skirt_shell_thickness_m = 1.2 mm` 只用于数值稳定/碰撞壳厚度，不声称是羽毛真实厚度。

`shuttlecock_physics_proxy.obj` 仅用于检查碰撞代理外形，不是最终视觉资产。

## 5. 最终视觉

默认引用：

- Title: Badminton Racket And Shuttlecock (Low Poly)
- Author: game_travel
- License: CC Attribution
- UID: `795e4cca2e544d7ab86a8e5c7ded2362`
- URL: https://sketchfab.com/3d-models/badminton-racket-and-shuttlecock-low-poly-795e4cca2e544d7ab86a8e5c7ded2362

需要把下载后的模型中 shuttlecock 单独提取并转换为：

`assets/third_party/shuttlecock_visual.usd`

转换时必须确认：米制尺度、+Z 轴定义、原点、最终 SHA256，并写进项目 `assets/THIRD_PARTY_ASSETS.md`。

构建器默认缺少该文件就拒绝生成“最终视觉版 USD”。只有显式使用 `--allow-temp-procedural-visual` 才会生成带 `TEMP` 元数据的调试视觉。

## 6. 空气动力

正式任务模式：

```text
p_dot = v
v_dot = g - k * |v - w| * (v - w)
```

默认 `k = 1/6.5 m^-1`。积分器为 RK4。

这与 Robot Brain v0.1 的轨迹预测模型一致。姿态相关阻力、恢复力矩、flip/oscillation 属于 `HIGH_FIDELITY_RESEARCH`，不能在没有系数标定时假装精确。

## 7. 接触参数为什么仍是 null

`shuttle-ground`、`shuttle-net`、`shuttle-racket` 的 friction / restitution / stiffness / damping 属于“接触对”，不是羽毛球单体的固有唯一常数。当前没有对应真实接触对实验数据，因此配置明确保留：

```yaml
contact_pair_parameters:
  status: REQUIRES_PAIR_CALIBRATION
  shuttle_ground: null
  shuttle_net: null
  shuttle_racket: null
```

这比写一个看似精确但来源不真实的值更符合项目要求。

## 8. 使用

普通 Python 校验：

```bash
python build_shuttlecock.py --config shuttlecock.yaml --validate-only --print-summary
python test_shuttlecock.py -v
python test_shuttle_aerodynamics.py -v
```

在 Isaac Sim Python 中生成最终 USD（视觉文件已准备好）：

```bash
python build_shuttlecock.py \
  --config shuttlecock.yaml \
  --output assets/shuttle/shuttlecock.usd \
  --print-summary
```

如果只想检查 USD 结构，不把 TEMP 当最终资产：

```bash
python build_shuttlecock.py \
  --config shuttlecock.yaml \
  --output shuttlecock_TEMP_debug.usda \
  --allow-temp-procedural-visual \
  --print-summary
```

## 9. 文件

- `shuttlecock.yaml`: 参数与来源策略。
- `build_shuttlecock.py`: 物理几何、质量属性、USD authoring。
- `shuttle_aerodynamics.py`: 二次阻力 + RK4。
- `test_shuttlecock.py`: 几何/质量/USD 结构测试。
- `test_shuttle_aerodynamics.py`: 动力学测试。
- `shuttlecock_physics_proxy.obj`: 物理碰撞代理预览，不是最终视觉。
