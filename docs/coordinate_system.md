# Coordinate System (Badminton Robot Scene v0.1)

冻结定义（ENGINEERING_V0_1），唯一依据 scene_layout.py。右手系。

- **World Origin** (0,0,0)：球网中央正下方地面（网 X=0）
- **+X**：机器人一侧(-X) → 对方球场(+X)
- **+Y**：机器人面向 +X 时的左侧
- **+Z**：竖直向上
- 场地：X∈[-6.70,6.70]；Y∈[-3.05,3.05]（双打）；线宽 0.04
- 网：X=0，宽 6.10（Y±3.05），中央顶 z=1.524，边顶/柱 z=1.55，depth 0.76
- Robot base：(-1.60,0,0) yaw0；PiPER mount world (-1.60,0,0.30)
- Morph placeholder：(L0.70×W0.55×H0.25)m @ z0.125，SOURCE=TEMP_PLACEHOLDER
- Camera rig：(−1.40,0,1.20)，pitch −4°，baseline 0.29（L y+0.145 / R y−0.145）
- Racket local：+X 面法向朝对方，+Z 柄→头；tcp=link6 origin，contact @ +Z 0.50
- ROIs/区域/来球/落区全部数值 → evidence/scene_layout.json

env 数据写入接口（write_root_*_to_sim）使用 **世界坐标** = env_origins + 局部坐标。
