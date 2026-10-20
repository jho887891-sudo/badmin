# P2：用测量几何搭建简化底盘碰撞（含 ContactSensor 实测）

分支 `dev/morph-one-p2-quat-and-settle`　脚本 `scripts/simulation/morph_one_p2_real_chassis.py`

## 1. 几何来源（全部是实测，不是估计）

| 项 | 来源 | 数值 |
|---|---|---|
| 四个轮状实体中心 | STEP eid 435700 / 759491 / 773650 / 871375（`p2_real_arm` 同批证据） | CAD 世界 (±0.250002, ∓0.297, −0.244) m |
| 轮实体包围盒 | 同上 STEP 顶点包围盒 | 48.653 × 180.782 × 157.783 mm |
| 主体模块包围盒 | M1.1-00V3 主体模块-1 并集包围盒 | 395.656 × 132.837 × 800.852 mm |
| 轮最低点 z | 上述几何换算 | **−0.334 m** → 落地 spawn 高度 = 0.334 m |

## 2. 结果（零重力，60 步，外层 gate）

| Run | lift m | 间隙 m | Gate | 外层 exit | t0 世界位置误差 | 60 步漂移 | step0 接触力 N | 接触 API |
|---|---|---|---|---|---|---|---|---|
| RC_lift0300_r01 | 0.300 | −0.034（穿地 34 mm） | PASS | 0 | 0.0 | 0.0 | **0.0** | isaaclab.sensors.ContactSensor |
| RC_lift0334_r01 | 0.334 | 0.000（刚好接触） | PASS | 0 | 0.0 | 0.0 | **0.0** | 同上 |
| RC_lift0350_r01 | 0.350 | +0.016 | PASS | 0 | 0.0 | 0.0 | — | 未启用 |

## 3. 诚实结论：这套简化碰撞**尚不能当作接触证据**

穿地 34 mm 时接触力仍为 **0.0 N**，说明 `CollisionAPI` 挂在 chassis 刚体子树下的子 prim 时，本 build
并没有把它们变成该刚体的有效碰撞形状（或接触报告未匹配到该 prim）。因此：

* 这三次运行**只能**证明"简化几何 body 原点与授权位姿一致、零重力下不漂移"；
* **不能**用来证明"间隙规则"在真实底盘上成立——那需要先证明碰撞体真的参与碰撞；
* 上一轮 `RM_*` 系列用的是**单一整块**底盘盒（碰撞确实生效，因为 0 间隙时注入 2.92 m/s），
  那批证据仍然有效；本轮的"多子碰撞体"结构需要额外验证。

## 4. 下一轮最小验证（已定位，不需猜测）

1. 用 `PhysicsCollisionAPI + PhysicsMeshCollisionAPI(approximation="convexHull")` 或显式
   `RigidBodyAPI` + 子碰撞体，验证接触力是否非零；
2. 或直接用 `rigid_prim_view.get_net_contact_forces` 失败后改 `ContactSensor` with
   `activate_contact_sensors=True`（本轮已加，需确认传感器 prim 是否匹配到 chassis body）；
3. 只有接触力非零后，才能用 `--lift` 扫描冻结真实底盘落地高度。

## 5. 复现

    python p2_gate_runner.py --script morph_one_p2_real_chassis.py --run-id RC_lift0300_r01 \
        --out-dir <dir> -- --lift 0.300 --read-contacts

证据：`outputs/simulation/rebot_b601_rs/morph_one/p2_real_chassis/<run_id>/`。
