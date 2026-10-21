# P3 动态挂载 harness 缺陷记录（三次真实失败的成因与修法）

**分支**：`dev/p2-wheel-contact-g1g4`（PR #20）· **门禁预注册**：`PHASE_P3_DYNAMIC_MOUNT_GATE.md`（未修改）
**驱动性质**：root pose 诊断驱动（`write_root_pose_to_sim`），**不是**轮驱动，不得作 P4 证据。

P3 的前三次运行全部 FAIL，但**都是我方 harness 缺陷**，不是物理结论。逐条记录，避免下一轮重犯。

## 缺陷 1 · 四元数探针在 `sim.reset()` 之前运行，错误顺序的探针四元数泄漏进正式段

- **现象**：`P3_static_r01` 第 0 步底盘实测 `(0.2160, 0.1450, 0.3738)`，而命令是 `(0, 0, 0.400)`；
  `tracking_position_error_max = 0.2683 m`，`peak_lin_speed = 1.4927 m/s`（static 剖面本该完全不动）。
- **成因**：探针把"wxyz 写法的 0.2 rad 偏航"送进一个实际按 xyzw 解释的 API，等价于**约 180° 翻转**；
  探针写完之后没有复位就进入正式段，翻转状态留在物理场景里。
- **修法**：探针移到第一次 `sim.reset()` 之后，并在探针结束后**再次 `sim.reset()`**；
  正式段开始前用确认过的顺序显式写一次 step-0 位姿，并把 `t0_pose_write_error_m` 记入 result.json。

## 缺陷 2 · `robot.update()` 不推进物理，读到**过期位姿**，导致探针选错顺序

- **现象**：`P3_static_r03` 在缺陷 1 修好后**仍然**从翻转姿态起步（同一 0.216/0.145 位移），
  `quat_order_for_write_root_pose.wxyz_probe_yaw_rad = 9.8e-08`。
- **成因**：`robot.write_root_pose_to_sim(...)` 之后只调 `robot.update(DT)`，物理没有 advance，
  读回的数据仍是旧帧，于是"wxyz 探针测得 yaw 0"这一假读数把正确顺序判成错误，切换成 xyzw 后
  写单位姿态 = 绕 X 转 180°，机器人被翻过来。
- **修法**：探针写完后先 `sim.forward()` 再 `robot.update()`；
  并且**两个候选顺序都写、都测**，选实测偏航最接近 0.2 rad 的那个，四个数字（两种顺序的实测偏航与绝对误差）全部落盘。
  不再"试第一个就信"。

## 缺陷 3 · 传送根位姿时不清零速度，且从未下发关节目标

- **现象**：`P3_static_r01` 的 `joint_deviation_max_rad = 0.2051`、`peak_ang_speed = 3.70 rad/s`，
  在"完全静止"的剖面里。
- **成因**：(a) 传送根位姿后残留线/角速度被 PhysX 保留，下一步把整条臂扯动；
  (b) `joint_hold` 门禁想测"隐式驱动能否保持目标"，却从未调用 `set_joint_position_target`，
  测到的是自由关节的被动响应。
- **修法**：每次传送同时 `write_root_velocity_to_sim(zeros)`（诊断驱动按运动学处理）；
  每步 `set_joint_position_target(joint_target)` 后再 `sim.step()`；
  `result.json.diagnostics` 记录 `root_velocity_zeroed_with_each_teleport` 与 `joint_targets_set_each_step`。

## 结论口径

在这三条修完之前，P3 的 `joint_hold`/`settles` 类判据**不具物理意义**，只能当 harness 调试信号。
`mount_stable` 在三次失败里始终为真（最大偏差 3.34e-08 m / 8.69e-08 rad），
但那是在错误初始姿态下测的，**不能单独作为 P3 通过的证据**。

真正可判定的 P3 结论必须来自修好后的运行（`P3_*_r05` 起），且在报告中必须同时给出：
`t0_pose_write_error_m`、探针四种读数、实测位移/偏航、相对安装漂移、关节偏差、手臂接触。
