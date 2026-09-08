# Coordinate System

**状态：占位（待填写，v0.1 属 P1 任务）**

职责：定义并登记全部坐标系与变换关系，避免"拍了半年发现坐标系反了"。

规划坐标系：
- 世界坐标系（court / simulator origin）
- 底盘坐标系（base_link）
- 机械臂基座坐标系
- 末端执行器坐标系（arm end-effector）
- 双目相机坐标系（left / right / camera optical frame）
- 球拍坐标系（racket contact frame）

需记录内容：
1. 各坐标系定义与轴向约定（如 ROS REP-103：x 前、y 左、z 上）
2. TF 变换树 / 齐次变换链
3. 羽毛球物理量（位置/速度/轨迹）所在坐标系与转换约定
4. 仿真（Isaac）与真机（ROS2）坐标系映射

**任何涉及坐标变换的代码改动必须同步本文档。**
