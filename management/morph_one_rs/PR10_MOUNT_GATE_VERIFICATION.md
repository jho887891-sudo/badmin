# PR #10 独立复验：C3 候选化 + Y-up→Z-up 旋转符号

被验对象：`9bafbfe051b05ff9b58c6bcfee477dafa78e71bf`（修复 `1fa0aba`、文档 `d325e2b`、测试 `9bafbfe`）
独立 worktree：`evidence/pr10-mount-gates-verification`　基线 `b0853efe`
解释器：Isaac Sim 自带 `/home/T7/ojh/robot_sim/env_isaaclab/bin/python`（pxr 0.25.11）
本文件只记录复验结果，不修改被验代码，也不改写真实机械臂资产。

---

## 1. pxr 环境物理前置测试

    PYTHONPATH=<repo> python -m pytest tests/simulation/cad/test_physics_readiness.py -q
    -> 7 passed in 0.63s

（上一轮同一文件是 5 项；PR #10 新增 2 项，其中一项就是"C3 必须是 CANDIDATE"的守卫。）

## 2. C3 只给出 CANDIDATE：**VERIFIED**

用真实资产（底盘 58,659,744 B USD + B601-RS 臂）实跑 `morph_one_physics_readiness.py`：

    c3_status    = CANDIDATE
    c3_statement = "the arm already carries an ArticulationRootAPI prim and 9 joints;
                    single mobile-chassis articulation is NOT proven without a PhysX step"
    c3_evidence  = { articulation_roots: ["/ReBot_Arm_RS/Geometry"], rigid_bodies: 9, joints: 9 }

全部标签：C1 `VERIFIED`、C2 `VERIFIED`、C3 `CANDIDATE`、C4 `VERIFIED`、C5 `VERIFIED`、C6 `UNKNOWN`。
即上一轮我把 C3 标成 VERIFIED 确实越界了；现在工具自身不会再这样标。

## 3. 旋转符号：CAD +Y 经 X 轴 **+90°** 变世界 +Z（VERIFIED）

用 `Gf.Rotation(Gf.Vec3d(1,0,0), degrees).TransformDir()` 实测：

| 输入（CAD 坐标） | +90° 后 | −90° 后 |
|---|---|---|
| CAD 上方向 +Y → `(0,1,0)` | `(0, 0, 1)` = 世界 **+Z** | `(0, 0, -1)` = 世界 **−Z** |
| CAD 车长方向 +Z → `(0,0,1)` | `(0, -1, 0)` = 世界 −Y | `(0, 1, 0)` = 世界 +Y |
| CAD 横向 +X → `(1,0,0)` | `(1, 0, 0)` | `(1, 0, 0)` |

所以 **+90° 才是把 Y-up 立起来的那一个**，−90° 会让底盘倒扣；上一轮文档里的符号确实是错的，PR #10 已改正。

**同时记录一个尚未决定的事项（CANDIDATE）**：+90° 之后 CAD 前进方向 +Z 落到世界的 **−Y**。也就是说"立起来"
解决了，但车头朝向变成了 −Y；若要车头朝 +X，还需要额外的 yaw 旋转（或规定世界坐标里 −Y 为正前方）。这不是
PR #10 的缺陷，而是一个仍需研究负责人拍板的坐标约定，本报告不替它下结论。

## 4. 真实机械臂资产未被修改（VERIFIED）

    arm     sha256 96d4f2ec1126dec6eaf9e70ab664ba6cc30fa81bdec86178a3e50b7a23bdd510  before == after
    chassis sha256 3b9962913ec517637df2a57df58acb5650fe995b437088539e26f00912af74b7  before == after

复验过程只读打开资产（`analyse_stage` / `analyse_arm`），未写任何 USD。

## 5. 结论分级

| 项 | 状态 |
|---|---|
| 物理前置测试（pxr 环境） | **VERIFIED**（7 passed） |
| C3 只给 CANDIDATE | **VERIFIED**（实跑标签 + 语句文本） |
| CAD +Y --(+90° about X)--> 世界 +Z | **VERIFIED** |
| −90° 会得到 −Z（符号确有意义） | **VERIFIED** |
| 车头朝向的世界约定（+90° 后朝 −Y） | **CANDIDATE / 待决定** |
| 真实机械臂资产未被改写 | **VERIFIED**（sha256 前后一致） |

## 6. 复现

    PYTHONPATH=<repo> PY=<isaac python> $PY -m pytest tests/simulation/cad/test_physics_readiness.py -q
    $PY morph_one_physics_readiness.py --chassis-usd <usda> --arm-usd <arm.usda> --output-json out.json

证据文件：`outputs/simulation/rebot_b601_rs/morph_one/pr10_b0853ef/a2_report.json`、
`.../a2_pytest.txt`。
