# BADMINTON_ROBOT.md

> 低速羽毛球人机对打机器人——仿真 Robot 模块层设计规范  
> 状态：**Design Spec / 待用户审阅**  
> 适用范围：Isaac Sim / Isaac Lab 中的整机 Robot 模块  
> 不包含：Robot Brain 算法、PPO 环境设计、跨模块联调流程  
> 设计原则：**结构真实、未知参数显式、资产可替换、接口稳定、批量环境原生支持**

---

## 0. 文档定位

本文件定义仿真中整台羽毛球机器人本体的结构、资产组织、坐标关系、运动学接口、状态接口、命令接口、批量环境行为和测试验收标准。

本文件的目标不是描述完整任务环境，而是回答：

1. 仿真里的“整机 Robot”由什么组成；
2. 各部件在 USD / Isaac Lab 中如何组织；
3. Robot 模块向上层暴露什么接口；
4. 哪些数据是真实/已验证的，哪些是 TEMP；
5. 如何在 Morph One 真实资产缺失时继续开发，又不污染最终架构；
6. 如何保证 1 → 4 → 16 → 64 → 128 env 时行为一致；
7. 如何做到后续替换 Morph One 真实 CAD / USD 时，上层代码不改。

---

# 1. 与其他文档的边界

## 1.1 本文负责

`BADMINTON_ROBOT.md` 负责：

- 整机 Robot 组成；
- Morph One 底盘结构与接口；
- PiPER 机械臂资产接入；
- 球拍固定关系；
- 双目相机安装关系；
- 整机 USD / Prim 层级；
- Robot 局部坐标系；
- Robot State / Command 接口；
- Whole-body kinematics；
- Batched Isaac Lab 接口；
- `reset(env_ids)`；
- TEMP → REAL 资产替换机制；
- 单元测试与验收标准。

## 1.2 本文不负责

以下内容不在本文实现：

- 羽毛球感知算法；
- Robot EKF / Shuttle UKF；
- 羽毛球轨迹预测；
- 是否击球；
- 拦截点搜索；
- 轨迹优化 / NMPC；
- Safety Shield；
- Driver 执行控制；
- PPO / BC / RL 环境逻辑；
- 跨模块联调流程；
- 球场、球网、羽毛球整体环境逻辑。

对应文档：

```text
docs/architecture/
├── 01_PERCEPTION.md
├── 02_STATE_ESTIMATION.md
├── 03_TRAJECTORY_PREDICTION.md
├── 04_HIT_DECISION.md
├── 05_PLANNING_POLICY.md
├── 06_SAFETY.md
├── 07_EXECUTION_CONTROL.md
├── 08_ADAPTATION_LEARNING.md
├── COORDINATE_SYSTEM.md
├── MODULE_INTERFACES.md
├── SIMULATION_ENVIRONMENT.md
└── INTEGRATION_TESTING.md
```

规则：

> 01–08 只写算法；联调继续单独放在 `INTEGRATION_TESTING.md`。  
> 本文只定义“机器人本体模块”。

---

# 2. 整机组成

整机固定定义为：

```text
Morph One 四转四驱舵轮移动底盘
        +
PiPER 6-DOF 机械臂
        +
固定式羽毛球拍
        +
前上方双目相机
        +
GPU / 电源 / 控制电子系统
```

概念层级：

```text
                    Opponent / Court +X
                            ↑

                    ┌─────────────┐
                    │ Stereo Cam  │
                    │ L       R   │
                    └──────┬──────┘
                           │
                       [ PiPER ]
                          ╱ ╲
                         ╱   ╲
                        ╱     ╲──────[ Racket ]
                       ╱
                ┌───────────────────┐
                │   Morph One Base  │
                │                   │
                │  ○             ○  │
                │                   │
                │  ○             ○  │
                └───────────────────┘
```

四个 `○` 表示四个独立的转向 + 驱动舵轮模块。

---

# 3. 冻结决策

以下内容视为当前阶段 **Frozen Decisions**，实现 Agent 不得自行修改。

## 3.1 Court Frame

球场世界坐标固定为：

```text
原点：球网中心地面
+X：机器人侧 → 对手侧
+Y：机器人面向 +X 时左侧
+Z：向上
右手坐标系
```

机器人侧：`x < 0`；对手侧：`x > 0`。

## 3.2 PiPER

使用已经验证的：

```text
assets/piper_no_gripper.usd
```

要求：

- 不重新导入新的 PiPER；
- 不修改内部 joint 定义；
- 不修改 joint order；
- 不修改冻结的 Stage-0 资产；
- joint1…joint6 顺序保持稳定。

## 3.3 球拍

球拍通过 FixedJoint 固定在 `PiPER link6`。

需要维护：

```text
T_link6_tcp
T_tcp_contact
T_link6_contact
```

坐标约定：

```text
+X_racket : 拍面法向，朝向对手
+Z_racket : 手柄 → 拍头
+Y_racket : 右手系
```

## 3.4 Morph One

底盘类型固定为：

```text
四转四驱舵轮底盘
```

不能简化成差速底盘、麦克纳姆轮、普通四轮阿克曼或单纯刚体长方体。

允许 TEMP 的是**尺寸与物理参数**，不是底盘的运动学拓扑。

## 3.5 Robot Brain 不得看到 env_origin

`env_origin` 仅用于 simulator placement。Robot Brain 看到的必须是 Court Frame 中的状态。

---

# 4. 资产真实性等级

所有机器人资产与参数必须标注来源状态。

统一枚举：

```text
VERIFIED_OFFICIAL
VERIFIED_MEASURED
DERIVED_FROM_MEASUREMENT
TRACEABLE_REFERENCE
TEMP_PARAMETERIZED_PROXY
UNKNOWN
REQUIRES_MEASUREMENT
REQUIRES_CALIBRATION
```

不允许：

```text
未知真实值 → 随便填数 → 当成最终参数
```

不允许：

```text
“为了方便” → 改变真实结构
```

允许为了数值稳定 / 性能简化 Physics proxy，但必须记录：为什么简化、简化了什么、损失哪些 fidelity、如何验证、后续如何替换。

---

# 5. 推荐目录

```text
simulation/robots/
└── badminton_robot/
    ├── __init__.py
    ├── badminton_robot.py
    ├── badminton_robot_cfg.py
    ├── morph_one/
    │   ├── __init__.py
    │   ├── morph_one.py
    │   ├── morph_one_cfg.py
    │   ├── kinematics.py
    │   └── TEMP_README.md
    ├── piper/
    │   ├── __init__.py
    │   ├── piper.py
    │   └── piper_cfg.py
    ├── racket/
    │   ├── __init__.py
    │   ├── racket.py
    │   └── racket_cfg.py
    ├── sensors/
    │   ├── stereo_camera.py
    │   └── stereo_camera_cfg.py
    ├── frames/
    │   └── robot_frames.py
    └── validation/
        └── robot_validator.py
```

测试：

```text
tests/simulation/robots/
├── test_robot_config.py
├── test_robot_frames.py
├── test_morph_one_kinematics.py
├── test_piper_adapter.py
├── test_racket_attachment.py
├── test_stereo_camera.py
├── test_robot_state.py
├── test_robot_commands.py
├── test_robot_reset.py
├── test_robot_batch.py
└── test_racket_contact_state.py
```

资产：

```text
assets/robots/badminton_robot/
├── badminton_robot.usd
├── morph_one/
│   └── morph_one_temp.usd
├── piper/
│   └── piper_no_gripper.usd
├── racket/
│   └── racket.usd
└── sensors/
    └── stereo_camera.usd
```

---

# 6. USD / Prim 层级

推荐最终整机结构：

```text
/BadmintonRobot
├── /MorphOne
│   ├── /Chassis
│   ├── /FL_Steer
│   │   └── /FL_Wheel
│   ├── /FR_Steer
│   │   └── /FR_Wheel
│   ├── /RL_Steer
│   │   └── /RL_Wheel
│   └── /RR_Steer
│       └── /RR_Wheel
├── /PiPER
│   └── Geometry
│       └── link1 ... link6
├── /Racket
├── /StereoCamera
│   ├── /Left
│   └── /Right
├── /Electronics
│   ├── /GPU
│   ├── /Power
│   └── /Controller
└── /Frames
    ├── robot_base
    ├── piper_base
    ├── camera_center
    ├── camera_left
    ├── camera_right
    ├── racket_tcp
    └── racket_contact_frame
```

视觉层与物理层不得互相隐式依赖；推荐分别管理 `Visual / Physics / Frames`。

---

# 7. Robot Frame 体系

完整主链：

```text
Court Frame
    ↓
robot_base
    ↓
piper_base
    ↓
link1 ... link6
    ↓
racket_tcp
    ↓
racket_contact_frame
```

相机：

```text
robot_base
    ├── camera_left
    └── camera_right
```

统一变换命名：`T_A_B`，含义为“把 B frame 中的量变换到 A frame”。

例如：

```text
T_court_robot
T_robot_piper
T_link6_tcp
T_tcp_contact
```

---

# 8. Morph One 模块

当前确认：`Morph One = 四转四驱舵轮移动底盘`。

公开信息暂未可靠提供：官方 STEP/STP、官方 URDF/USD、精确长宽高、wheelbase、track width、wheel radius、wheel width、整车 mass、COM、inertia、steering limits、steering rate、max wheel speed。

因此这些值不得被假装为官方真值。

即便是 TEMP，也必须保留真实拓扑：

```text
Chassis
+
4 × Steer Module
+
4 × Drive Wheel
+
PiPER Mount
+
Camera Mast
```

必须存在：

```text
FL_steer_joint
FL_drive_joint
FR_steer_joint
FR_drive_joint
RL_steer_joint
RL_drive_joint
RR_steer_joint
RR_drive_joint
```

---

# 9. Morph One 双驱动模式

## 9.1 BODY_TWIST_ACTUATOR

输入：

```text
[vx, vy, wz]
```

用于 Robot Brain、上层规划、早期仿真、PPO 前期和任务层控制。内部可通过逆运动学计算舵轮目标。

## 9.2 STEER_DRIVE_WHEEL_MODEL

输入：

```text
steer_cmd: (N, 4)
drive_cmd: (N, 4)
```

或更严格地提供 steer angle/velocity 与 wheel angular velocity 目标。

用于高保真底盘控制、舵轮控制器验证、slip/contact 和动力学研究。

---

# 10. 四舵轮运动学

第 i 个舵轮相对 robot_base 的位置：

\[
r_i = [x_i, y_i]^T
\]

底盘期望速度：

\[
\xi=[v_x,v_y,\omega_z]^T
\]

轮中心平面速度：

\[
v_i =
\begin{bmatrix}
v_x-\omega_z y_i\\
v_y+\omega_z x_i
\end{bmatrix}
\]

目标转角：

\[
\theta_i=\operatorname{atan2}(v_{iy},v_{ix})
\]

轮切向速度：

\[
s_i=\|v_i\|
\]

轮角速度：

\[
\omega_i=s_i/r_w
\]

`r_w` 必须来自配置；若为 TEMP，必须标记 `TEMP_PARAMETERIZED_PROXY`。

---

# 11. 舵轮角度优化

`(theta,+speed)` 与 `(theta+pi,-speed)` 为等价解。

控制层应选择转向变化更小的解：

\[
\Delta\theta=\operatorname{wrapToPi}(\theta_{target}-\theta_{current})
\]

若 `|Δθ| > π/2`，允许 `theta_target += pi` 且 `wheel_speed *= -1`。

目标是降低舵轮转向时间并保持连续运动。

---

# 12. Morph One TEMP 参数制度

配置必须显式带状态。例如：

```yaml
morph_one:
  asset_status: TEMP_PARAMETERIZED_PROXY
  geometry:
    length_m:
      value: null
      status: REQUIRES_MEASUREMENT
    width_m:
      value: null
      status: REQUIRES_MEASUREMENT
    wheel_radius_m:
      value: null
      status: REQUIRES_MEASUREMENT
    wheel_width_m:
      value: null
      status: REQUIRES_MEASUREMENT
    wheel_positions_robot:
      value: null
      status: REQUIRES_MEASUREMENT
  mass_properties:
    total_mass_kg:
      value: null
      status: REQUIRES_MEASUREMENT
    com_robot:
      value: null
      status: REQUIRES_MEASUREMENT
    inertia_robot:
      value: null
      status: REQUIRES_MEASUREMENT
```

如必须以 TEMP 数值运行，需明确：

```yaml
value: <TEMP>
status: TEMP_PARAMETERIZED_PROXY
source: "temporary simulation parameter"
```

并在 `TEMP_README.md` 记录全部 TEMP 参数。

---

# 13. PiPER 模块

资产：`assets/piper_no_gripper.usd`。

不复制修改内部关节。joint order 固定：

```text
joint1
joint2
joint3
joint4
joint5
joint6
```

状态接口：

```text
joint_pos.shape == (N, 6)
joint_vel.shape == (N, 6)
```

Stage-0 视为冻结依赖；Robot 模块只做 reference / wrap / query / command，不重新“修” PiPER。

---

# 14. PiPER 安装

定义 `T_robot_piper`。

当前工程安装位置可以作为 TEMP / engineering baseline，但最终必须来源于官方安装图、实测、CAD 或安装治具数据。

示例：

```yaml
piper_mount:
  translation_robot_m:
    value: [0.0, 0.0, 0.30]
    status: TEMP_PARAMETERIZED_PROXY
  quaternion_robot:
    value: [1.0, 0.0, 0.0, 0.0]
    status: TEMP_PARAMETERIZED_PROXY
```

---

# 15. 球拍模块

结构：

```text
PiPER link6
    ↓
FixedJoint
    ↓
racket_tcp
    ↓
racket_contact_frame
```

固定坐标：

```text
+X_racket = 拍面法向
+Z_racket = 手柄 → 拍头
+Y_racket = 右手系
```

必需变换：

```text
T_link6_tcp
T_tcp_contact
T_link6_contact
```

且：

\[
T_{link6,contact}=T_{link6,tcp}T_{tcp,contact}
\]

安装质量、COM、inertia 与 `T_link6_tcp` 如未实测必须保持 `REQUIRES_MEASUREMENT`。

---

# 16. racket_contact_frame

`racket_contact_frame` 表示主要击球拍面区域的几何/控制参考中心。

必须提供：

```python
get_racket_contact_pose()
get_racket_contact_twist()
```

输出：

```text
pose:  [x, y, z, qw, qx, qy, qz]
twist: [vx, vy, vz, wx, wy, wz]
```

统一在 Court Frame。

---

# 17. Whole-body racket twist

拍面速度不能只来自 PiPER Jacobian。

总速度必须包含底盘和机械臂贡献：

\[
v_c=v_b+\omega_b\times(p_c-p_b)+J_v(q)\dot q
\]

\[
\omega_c=\omega_b+J_\omega(q)\dot q
\]

因此 `get_racket_contact_twist()` 必须返回 whole-body 结果。

这是硬要求。

---

# 18. 双目相机模块

结构：

```text
StereoCamera
├── Left
└── Right
```

当前工程 baseline：

```text
camera center:
x = +0.20 m relative robot_base
z = +1.20 m
pitch = -4 deg
baseline = 0.29 m
Left y  = +0.145 m
Right y = -0.145 m
```

这些是 `TEMP_PARAMETERIZED_PROXY / engineering baseline`，不是最终机械安装实测。

必须暴露：

```text
camera_left
camera_right
T_robot_camera_left
T_robot_camera_right
```

---

# 19. GPU / 电源 / 控制电子系统

这些部件当前有两个角色：Visual 与 Mass distribution。

如果真实整车质量与 COM 未知，不应给每个盒子随意分配“真实质量”。推荐：

```text
Visual: TEMP_PARAMETERIZED_PROXY
Mass: REQUIRES_MEASUREMENT
```

---

# 20. Mass / COM / Inertia

合法来源：

```text
MEASURED
CAD_DERIVED
ANALYTIC_DERIVED
REFERENCE_PRODUCT
TEMP
UNKNOWN
```

优先级：

```text
Physical measurement
    >
Official CAD
    >
Official manufacturer data
    >
Traceable analytical derivation
    >
TEMP simulation value
```

最终整车高保真动力学不得由多个随意 TEMP mass 拼出假的 whole-body inertia。

---

# 21. Collision

视觉 mesh 不等于碰撞 mesh。

Morph One 建议：chassis 用 box/convex，steer housing 用 simple convex，wheel 用 cylinder/convex，mounting plate 用 box。

PiPER 沿用原资产已验证 collision。Racket 使用专门 physics proxy，而不是高面数 visual mesh。

可因稳定性、实时性和动态碰撞数值原因简化；不能只因“方便”简化。

---

# 22. Contact

Robot 模块只提供真实接触相关几何 / sensor attachment，不负责击球判断算法。

禁止以 `distance < threshold` 冒充 racket-shuttle contact。应支持 `ContactSensorCfg` 或 PhysX contact report。

---

# 23. Robot State

推荐：

```python
@dataclass
class BadmintonRobotState:
    base_pose_court
    base_twist_court
    joint_pos
    joint_vel
    racket_contact_pose_court
    racket_contact_twist_court
    camera_left_pose_court
    camera_right_pose_court
    validity
    timestamp
```

Batched shape：

```text
base_pose_court            (N, 7)
base_twist_court           (N, 6)
joint_pos                  (N, 6)
joint_vel                  (N, 6)
racket_contact_pose_court  (N, 7)
racket_contact_twist_court (N, 6)
```

---

# 24. Robot Command

Task-level：

```python
set_base_twist_command(vx, vy, wz)
set_arm_joint_position_target(q)
set_arm_joint_velocity_target(dq)
```

Wheel-level 仅在 `STEER_DRIVE_WHEEL_MODEL` 使用：

```python
set_steer_target(...)
set_wheel_velocity_target(...)
```

---

# 25. Robot 公共 API

稳定接口：

```python
get_base_state()
get_arm_joint_state()
get_racket_contact_pose()
get_racket_contact_twist()
get_camera_pose()
set_base_twist_command(...)
set_arm_joint_position_target(...)
set_arm_joint_velocity_target(...)
reset(env_ids)
```

上层不得依赖 Isaac 内部 prim path。

---

# 26. Isaac Lab 封装原则

上层不得自己 find prim、解析 articulation path 或到 scene 中寻找 link6。全部由 `BadmintonRobot` 封装并提供稳定 tensor API。

---

# 27. Batched 环境

必须原生支持：`N = 1, 4, 16, 64, 128`。

典型 tensor：

```text
joint_pos       (N, 6)
joint_vel       (N, 6)
base_pose       (N, 7)
base_twist      (N, 6)
racket_pose     (N, 7)
racket_twist    (N, 6)
steer_state     (N, 4)
drive_state     (N, 4)
```

禁止 Python per-env 控制循环作为主要运行路径。

---

# 28. env_origin 处理

`env_origins: (N,3)` 仅用于物理场景复制位置。

所有对 Robot Brain 暴露的状态必须先转换到 Court Frame。不得让上层算法出现 `state.position += env_origin` 一类逻辑。

---

# 29. reset(env_ids)

必须只重置选中的 env。例如 `reset(env_ids=[2,7,9])` 只能影响 2/7/9。

其他环境的 base state、PiPER state、steer state、drive state、command buffer、controller internal state 不得变化。

---

# 30. Reset 内容

至少恢复：

```text
base pose
base twist
joint pos
joint vel
steer joint state
drive joint state
command buffer
controller state
```

camera frame / rigid attachments 由 root state + fixed transforms 自动恢复。

---

# 31. 初始位姿

工程 baseline：

```text
robot_base:
x = -1.60 m
y = 0
yaw = 0
```

这是 engineering baseline，不是不可修改机械常量。它属于 scene / experiment 配置层，不应写死在 Robot class 内。

---

# 32. Config 设计

```python
@dataclass
class BadmintonRobotCfg:
    morph_one: MorphOneCfg
    piper: PiperCfg
    racket: RacketCfg
    stereo_camera: StereoCameraCfg
    drive_mode: DriveMode
    frame_convention: FrameConventionCfg
```

---

# 33. 配置校验

`robot_validator.py` 至少检查：

1. 所有 required prim path；
2. PiPER 6 joint；
3. joint order；
4. 4 steer；
5. 4 drive；
6. transform 可组成；
7. quaternion normalized；
8. racket_contact_frame 存在；
9. camera baseline > 0；
10. TEMP / UNKNOWN 状态合法；
11. final 模式中不能存在禁止的 UNKNOWN；
12. tensor shape；
13. 无 NaN/Inf。

---

# 34. Final / TEMP 模式

建议提供 development mode 和 final mode。

Development mode 允许 `TEMP_PARAMETERIZED_PROXY`，但必须输出明确 warning。

Final mode 中，如果 Morph geometry、wheel coordinates、wheel radius、mass、COM、inertia、PiPER mount、racket mount、camera mount 仍未完成，则不能宣称“high-fidelity final robot model”，必要时直接失败。

---

# 35. Robot module lifecycle

```text
configure
   ↓
spawn/reference assets
   ↓
resolve prims
   ↓
validate
   ↓
initialize buffers
   ↓
ready
```

运行期：

```text
read state
   ↓
compute frames
   ↓
receive command
   ↓
apply command
   ↓
step simulation
```

---

# 36. 与 Robot Brain 的边界

Robot Brain 不需要知道 `/World/envs/env_17/Robot/PiPER/...`，只需要稳定 API。因此 Isaac prim path 是 Robot 模块内部实现细节。

---

# 37. 与 PPO 的边界

当前阶段：

```text
NO PPO
NO DirectRLEnv
NO ManagerBasedRLEnv
```

Robot 模块先独立于 RL 环境完成。

---

# 38. Visual / Physics 分离

例如 Morph One：

```text
Visual:
真实 CAD / 高质量 mesh

Physics:
box + convex + wheel primitives
```

未来若得到官方 CAD，只需替换 Visual/必要 Physics，不破坏 controller/API。

---

# 39. TEMP → REAL 替换策略

目标：

```text
morph_one_temp.usd
        ↓
morph_one_real.usd
```

替换时尽量保持 Prim interface、Frame names、Joint semantic names、Robot API。

如果真实 USD joint 命名不同，在内部使用 adapter 映射，上层不改。

---

# 40. Asset Adapter

建议定义语义映射：

```python
MORPH_ONE_JOINT_MAP = {
    "fl_steer": "...",
    "fl_drive": "...",
    "fr_steer": "...",
    "fr_drive": "...",
    "rl_steer": "...",
    "rl_drive": "...",
    "rr_steer": "...",
    "rr_drive": "...",
}
```

PiPER 同理维护 `joint1...joint6` 语义映射。

---

# 41. Whole-body pose computation

`racket_contact_pose_court` 必须由完整链得到：

\[
T_{court,contact}=T_{court,robot}T_{robot,piper}T_{piper,link6}T_{link6,tcp}T_{tcp,contact}
\]

不得通过 magic offset 直接写死。

---

# 42. Whole-body Jacobian

可抽象为：

\[
J_{wb}=[J_{base}\;J_{arm}]
\]

使：

\[
V_{contact}=J_{wb}[v_x,v_y,\omega_z,\dot q_1,...,\dot q_6]^T
\]

---

# 43. RobotState 时间语义

所有 RobotState 必须有 `timestamp`，其含义是状态对应的 simulation time / capture time，不得把 CPU wall-clock 混进物理状态时间。

---

# 44. 运行频率

Physics：`240 Hz`。

Robot state 可每 physics step 更新。高层 command 频率可更低，但 Robot 模块需处理 command hold / zero-order hold / optional interpolation。

---

# 45. 数值健壮性

每步检查或测试覆盖 NaN、Inf、invalid quaternion、exploding velocity、invalid joint state。开发阶段应 fail loudly。

---

# 46. Isaac Lab 复制策略

使用：

```text
InteractiveScene
InteractiveSceneCfg
{ENV_REGEX_NS}
replicate_physics=True
```

不得手工 Python loop 创建 128 套完整独立场景作为最终实现。

---

# 47. 性能目标

避免 per-env Python object-heavy update、per-env USD query every step、runtime repeated prim lookup、dynamic allocation every step。

原则：

```text
初始化阶段 resolve
运行阶段 tensorized
```

---

# 48. 单元测试边界

Robot 模块内部单元测试写在本模块测试中；跨模块端到端测试仍归 `INTEGRATION_TESTING.md`。

---

# 49. 必须测试：Config

验证 required field、enum、asset status、unknown/temp handling、malformed quaternion、missing prim、wrong joint count。

---

# 50. 必须测试：Frames

至少测试：

```text
Court → robot_base
robot_base → piper_base
link6 → racket_tcp
racket_tcp → racket_contact
robot_base → camera_left
robot_base → camera_right
```

必须有 round-trip 误差测试。

---

# 51. 必须测试：Morph One kinematics

- 纯前进：`vx>0, vy=0, wz=0`，四轮方向一致；
- 纯横移：`vx=0, vy>0, wz=0`，四轮转向约 90°；
- 原地旋转：`vx=0, vy=0, wz!=0`，轮方向与切向速度一致；
- 混合运动：`vx+vy+wz`，并回算 chassis twist。

---

# 52. 必须测试：舵轮优化

验证 `theta+pi` 与 `speed sign flip` 后结果等效，且 steering rotation 更小。

---

# 53. 必须测试：PiPER

检查 6 joints、joint order、joint limits present、state shape、command shape。只做依赖 smoke test，不重复全部 Stage-0 测试。

---

# 54. 必须测试：Racket

验证 FixedJoint、link6 body、racket body、racket_tcp、racket_contact_frame、transform composition。

---

# 55. 必须测试：Stereo

验证 left/right exists、baseline positive、left y > right y、extrinsic transform valid。当前 baseline 下 `left_y - right_y = 0.29 m`。

---

# 56. 必须测试：Whole-body racket twist

构造 base only、arm only、base+arm 三个案例，要求 combined result = base contribution + arm contribution，在数值容差内成立。

---

# 57. 必须测试：Batch

至少 `N=1,4,16,64,128`，验证 shape、command、state、reset、no cross-env contamination。

---

# 58. 必须测试：selected reset

例如 `N=8, reset env_ids=[1,3,6]`，env 0/2/4/5/7 的状态在 bitwise / tolerance 内不变。

---

# 59. 必须测试：env isolation

给 env 2 非零 base command，其他环境为零；在无物理交叉接触情况下，其他环境 Robot state 不应被污染。

---

# 60. 必须测试：NaN / Inf

长时间运行 1/4/128 env，至少检查 joint_pos、joint_vel、base state、steer state、drive state、racket pose、racket twist。

---

# 61. 128 env soak

最终 Robot 模块验收：`128 env >= 10 min simulation time`。

要求：no NaN、no Inf、no CUDA fatal、no PhysX fatal、no reset leakage、no cross-env anomaly。

---

# 62. 阶段化验证

```text
1 env
↓
4 env
↓
16 env
↓
64 env
↓
128 env
```

任何阶段失败：STOP，先修复。

---

# 63. 明确禁止事项

实现 Agent 不得：

1. 修改冻结 PiPER USD；
2. 重建一个“看起来像 PiPER”的替代品；
3. 猜 Morph One 官方尺寸；
4. 猜 Morph One mass / COM / inertia 并称为真实值；
5. 把 Morph One 变成普通四轮；
6. 把视觉 mesh 直接作为所有 dynamic collision；
7. 用 `distance threshold` 冒充 contact；
8. 把 `env_origin` 暴露给 Robot Brain；
9. 把算法逻辑藏到 scene code；
10. 现在开始 PPO；
11. 创建 DirectRLEnv；
12. 手工 loop 创建 128 套完整世界；
13. 因“方便”移除真实舵轮拓扑；
14. 因“方便”省略 whole-body racket twist；
15. silently replace UNKNOWN with guessed value。

---

# 64. 实现 Agent 行为契约

任何 Agent 开始实现前必须阅读：

```text
AGENTS.md
docs/simulation/BADMINTON_ROBOT.md
docs/architecture/COORDINATE_SYSTEM.md
docs/architecture/MODULE_INTERFACES.md
docs/architecture/SIMULATION_ENVIRONMENT.md
```

优先级：

```text
Current explicit instruction
    >
BADMINTON_ROBOT.md Frozen Decisions
    >
Architecture docs
    >
AGENTS.md
    >
older docs/history
```

---

# 65. TDD 要求

每个功能：

```text
RED
↓
确认测试确实失败
↓
GREEN
↓
最小正确实现
↓
Refactor
↓
回归测试
```

禁止先写一堆实现再补测试。

---

# 66. Phase Gate

## Phase 0 — Project Audit
检查 frozen software、PiPER asset、racket asset、scene conventions、existing tests。

## Phase 1 — Config
实现 `BadmintonRobotCfg / MorphOneCfg / PiperCfg / RacketCfg / StereoCameraCfg`。

## Phase 2 — Frames
实现 `robot_frames.py`。

## Phase 3 — Morph One Kinematics
实现 4-steer / 4-drive inverse kinematics 与 steer optimization。

## Phase 4 — PiPER Adapter
只封装现有 USD。

## Phase 5 — Racket
加入 FixedJoint + frames。

## Phase 6 — Stereo
加入双目 frame / config。

## Phase 7 — BadmintonRobot
组合整机。

## Phase 8 — Batched State / Command / Reset
实现 tensorized state、commands、`reset(env_ids)`。

## Phase 9 — Vectorization
执行 `1 → 4 → 16 → 64 → 128`。

## Phase 10 — Validation / Freeze
完整报告。

---

# 67. 每阶段报告格式

```text
PHASE X RESULT

Status:
PASS / FAIL / BLOCKED

Changed files:
...

Tests added:
...

Tests executed:
...

Actual outputs:
...

TEMP assumptions:
...

Unknown real parameters:
...

Known limitations:
...

Frozen interfaces:
...

Ready for next phase:
YES / NO
```

不得只输出 `Done.`。

---

# 68. BLOCKED 规则

如果某功能依赖真实未知参数，并且缺少这些值会使测试失去物理意义，则 `Status = BLOCKED`，不能编造数字。

若模块只需要接口或 symbolic config，则可以继续实现。例如 MorphOneCfg schema、semantic joint names、frame API。

---

# 69. Source-of-Truth 原则

这份文档是 `Robot module Source of Truth`，不是 DeepSeek 专用提示词。

DeepSeek、Claude、Codex、ChatGPT 或本地开发者都应依据同一文档实现。

---

# 70. Robot 模块验收标准

以下全部满足时，可声明 `BadmintonRobot module v0.1 accepted`。

## 70.1 结构
- [ ] Morph One 四舵轮拓扑存在；
- [ ] 4 steer joints；
- [ ] 4 drive joints；
- [ ] PiPER 正确引用；
- [ ] Racket 正确固定；
- [ ] Stereo 左右相机存在；
- [ ] Frames 完整。

## 70.2 数据
- [ ] Robot State tensorized；
- [ ] Robot Command tensorized；
- [ ] racket pose；
- [ ] whole-body racket twist；
- [ ] camera pose；
- [ ] no env_origin exposure。

## 70.3 Reset
- [ ] selected env reset；
- [ ] non-selected env unchanged；
- [ ] command buffers reset；
- [ ] controller state reset。

## 70.4 Batch
- [ ] N=1；
- [ ] N=4；
- [ ] N=16；
- [ ] N=64；
- [ ] N=128。

## 70.5 Stability
- [ ] no NaN；
- [ ] no Inf；
- [ ] no CUDA fatal；
- [ ] no PhysX fatal；
- [ ] no cross-env contamination；
- [ ] 128 env >=10 min soak pass。

## 70.6 Asset honesty
- [ ] 所有真实值有来源；
- [ ] 所有 TEMP 显式标记；
- [ ] UNKNOWN 未被伪装成 final；
- [ ] Morph One 未冒充官方 CAD；
- [ ] physics simplification 有说明。

---

# 71. v0.1 完成后仍允许 BLOCKED 的内容

即使 Robot 模块 v0.1 通过，仍可存在：

```text
Morph One exact visual geometry
Morph One exact mass
Morph One exact COM
Morph One exact inertia
exact wheel radius
exact wheel center locations
final camera mount
final PiPER mount
final racket mount
```

前提：显式标记、不用于声称真实动力学、接口允许后续替换、相关高保真验收项标为未完成。

---

# 72. 后续真实资产替换

拿到 Morph One 官方 CAD / 尺寸后：

```text
Step 1 导入 visual
Step 2 建立/校验 physics collision
Step 3 填真实 wheel center / radius
Step 4 填真实 mass / COM / inertia
Step 5 对齐 semantic joints
Step 6 运行全部 Robot tests
Step 7 运行 1 → 128 env regression
Step 8 更新 asset status
TEMP_PARAMETERIZED_PROXY
→
VERIFIED_OFFICIAL / VERIFIED_MEASURED
```

上层 Robot Brain / Hit Decision / Planning / PPO 不应因替换而改 API。

---

# 73. 最终架构一句话

整机仿真 Robot 模块应保持：

```text
真实结构
+
可追溯参数
+
稳定 API
+
可替换资产
+
tensorized batch
+
strict reset isolation
```

而不是：

```text
为了先跑起来
→
随便拼一个盒子
→
以后全部重写
```

---

# 74. 当前设计结论

当前阶段应该立即实现：

```text
BadmintonRobot 模块骨架
PiPER wrapper
Racket wrapper
Stereo wrapper
Morph One 四舵轮语义与运动学
Frames
Robot State
Robot Command
Batch
Reset
Validation
```

当前阶段不应该伪造 Morph One 官方几何和真实动力学参数。

正确路径：

```text
现在：
结构正式 + 参数状态显式 + TEMP 可替换

以后：
替换真实资产/真实参数

始终：
Robot API 不变
```

---

## Appendix A — 推荐公共 API

```python
class BadmintonRobot:
    def get_base_state(self): ...
    def get_arm_joint_state(self): ...
    def get_racket_contact_pose(self): ...
    def get_racket_contact_twist(self): ...
    def get_camera_pose(self): ...
    def set_base_twist_command(self, command): ...
    def set_arm_joint_position_target(self, q_target): ...
    def set_arm_joint_velocity_target(self, dq_target): ...
    def reset(self, env_ids): ...
```

---

## Appendix B — 推荐 tensor shapes

```text
base_pose               (N, 7)
base_twist              (N, 6)
joint_pos               (N, 6)
joint_vel               (N, 6)
steer_pos               (N, 4)
steer_vel               (N, 4)
wheel_pos               (N, 4)
wheel_vel               (N, 4)
racket_contact_pose     (N, 7)
racket_contact_twist    (N, 6)
camera_left_pose        (N, 7)
camera_right_pose       (N, 7)
```

---

## Appendix C — 最小 Agent 入口指令

```text
阅读并严格遵守：

1. AGENTS.md
2. docs/simulation/BADMINTON_ROBOT.md
3. docs/architecture/COORDINATE_SYSTEM.md
4. docs/architecture/MODULE_INTERFACES.md
5. docs/architecture/SIMULATION_ENVIRONMENT.md

BADMINTON_ROBOT.md 是 Robot 模块 Source of Truth。

不得修改 Frozen Decisions。
不得把未知真实参数自行补成“真实值”。
TEMP 必须显式标记。
采用 TDD。
每个 Phase 必须输出 PASS / FAIL / BLOCKED 与实际测试证据。
未通过当前 Phase 不得进入下一 Phase。
```

---

## Appendix D — 设计审阅清单

- [ ] 文档没有把 Robot Brain 算法塞进 Robot 模块；
- [ ] 文档没有把 integration 内容塞进 01–08；
- [ ] Morph One 拓扑保持四转四驱；
- [ ] PiPER Stage-0 被视为冻结依赖；
- [ ] Racket frame 语义稳定；
- [ ] Stereo baseline 清楚标明工程 baseline；
- [ ] Whole-body racket twist 是整机速度，不只 arm Jacobian；
- [ ] env_origin 不泄漏；
- [ ] selected reset 有严格隔离；
- [ ] 128 env 原生支持；
- [ ] TEMP 参数不会冒充真实值；
- [ ] 后续 Morph One 真资产可无痛替换；
- [ ] Robot API 与具体 USD prim path 解耦。
