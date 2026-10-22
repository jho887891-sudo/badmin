# P6-05 坐标变换一致性审计（camera / base / world / racket mount）

**状态：PASS**（本切片唯一结论：已提交的 P6 变换簇在**数值与溯源层面自洽**；5 处物理量仍然 UNKNOWN，本切片**没有**补任何数）。

| 项 | 值 |
|---|---|
| 分支 | `p6/frame-consistency-audit`（基于 `p6/racket-camera-frames` d7672d5） |
| 审计器 | `tools/p6_frames/audit_frame_consistency.py`（纯 Python + numpy，按路径读已提交 JSON，不 import 其它分支） |
| 回归测试 | `tests/simulation/p6_frame_consistency_test.py`（19 项） |
| 审计命令 | `python tools/p6_frames/audit_frame_consistency.py` |
| 审计退出码 | **0**（99/99 项检查通过，0 处不一致） |
| 测试退出码 | **0**（19 passed） |
| 基线回归 | `python -m pytest tests/simulation -k p6_ -q` → **0**（48 passed, 196 deselected） |

## 1. 审计的五个文件

| 文件 | 来源 | 本分支状态 |
|---|---|---|
| `frame_tree.json` | P6-01 / PR #26 | 已提交（本分支） |
| `camera_metadata.json` | P6-01 / PR #26 | 已提交（本分支） |
| `mounted_chain.json` | P6-02 / PR #36 | 钉住副本 `evidence/upstream/`（sha256 `7228b559…`） |
| `mounted_chain_v2.json` | P6-02-OI-1 / PR #37 | 钉住副本（sha256 `bf7b6361…`） |
| `joint_anchors.json` | P6-02-OI-1 / PR #37 | 钉住副本（sha256 `13030fb3…`） |

后三个文件在 `p6/racket-camera-frames` 上并不存在。按仓库既有做法（`mounted_chain.json` 自己就是这样钉住 P3 证据的），它们以**逐字节副本 + `PROVENANCE.json`** 落在 `evidence/upstream/`，审计器对每个文件重算 sha256 并与文件内记录的哈希链比对——钉住本身就是被审计对象，而不是绕过审计。

## 2. 到底检查了什么

1. **每个数值都回到源头**：`mounted_chain`/`mounted_chain_v2` 的每条 `value_ref`/`*_ref` 必须（a）指向真实文件、（b）`json_pointer` 在目标文件里可解析、（c）目标文件 sha256（LF 归一化）等于记录的哈希、（d）解析到的边与声明边的 `edge/from/to/edge_kind/status` 完全一致。任一不成立 → 对应 typed code。
2. **旋转是刚体旋转**：每条 ASSERTED 边的 `rotation_matrix` 必须 `max|RᵀR−I| ≤ 1e-9`、`det(R)=+1`，且 `{order,values}` 的四元数（wxyz/xyzw 按 order 解码）复现同一矩阵（≤1e-6）。长度单位声明必须是 metre、角度 radian（frame_tree / mounted_chain / joint_anchors 三处）。
3. **UNKNOWN 不含任何数**：边 1、2、13、14、15 在 frame_tree 与 mounted_chain 中 `translation_m`/`rotation_quat`/`rotation_matrix`/`value_ref` 全为空，且子树内**没有**任何数字叶子。唯一豁免：`mounted_chain` 边 2 的 `p3_01_anchor`（其 status 明写 `SIM_AUTHORED_IDENTITY (not a measurement…)`）——豁免本身也被检查：status 不声明"非测量"就报错。相机侧每个 UNKNOWN 字段同样必须为 null。
4. **同一物理量不得两处不同值**：跨文件逐项比 `frame_tree` 边 ↔ `joint_anchors` 的 `local_pos0_m`/`local_rot0_wxyz`（同一份 USD `physics:localPos0/localRot0`）、`frame_tree` 边 ↔ `mounted_chain` 边（后者只允许指针）、`mounted_chain` 与 `mounted_chain_v2` 的 composition A 矩阵、以及 `0.33242` 这个历史误差在三个位置的三次记录。任何数值不等即 `E_QUANTITY_DUPLICATE_MISMATCH`（`q` 与 `−q` 表示同一旋转但**记录值不同**，因此同样被拦）。
5. **复合能复现终点**：用 frame_tree 边 `[3,4,5,6,7,8,9,12]` 自行复合，必须复现 `mounted_chain` 记录的 composition B 末端位姿（≤1e-9 m）；用 `joint_anchors` 的 `X0·X1⁻¹` 复合必须复现 `mounted_chain_v2` 的 B v2（≤1e-9 m）；同时 `X0·X1⁻¹` 必须逐关节等于记录的 `T_body0_body1`。
6. **历史不可覆盖、v2 必须收敛**：`mounted_chain.json` 的 `open_issue.max_abs_delta_m` 必须仍是 `0.33242`（容差 1e-12，实测 `max|B−A|` 位置差 = 0.33242000 m），`open_issue.status` 仍为 `OPEN`；`mounted_chain_v2` 的 B v2 距 pxr 位姿必须 < 1e-6 m（实测 1.853e-09 m 位置差 / 1.925e-08 旋转元素差，与文件自记录一致）。
7. **光轴 ↔ 安装系置换自洽**：`camera_metadata` 的 `rotation_from_mount_to_optical` 必须正交、det=+1、四元数复现矩阵，且确实等于文档所述置换 `X_mount→+Z_opt, Y_mount→−X_opt, Z_mount→−Y_opt`；而 `T_mount_camera`（位姿）必须仍是 UNKNOWN。

## 3. 真实命令与真实退出码

```
$ python tools/p6_frames/audit_frame_consistency.py
P6 frame consistency audit  frames_dir=outputs/simulation/rebot_b601_rs/morph_one/p6_frames
  artifact frame_tree.json          frames-dir
  artifact camera_metadata.json     frames-dir
  artifact mounted_chain.json       pinned-upstream-copy
  artifact mounted_chain_v2.json    pinned-upstream-copy
  artifact joint_anchors.json       pinned-upstream-copy
  checks: 99/99 passed
  verdict: PASS  (exit 0)
$ echo $?
0

$ python -m pytest tests/simulation/p6_frame_consistency_test.py -q
...................                                                      [100%]
19 passed in 20.72s
EXIT=0

$ python -m pytest tests/simulation -k p6_ -q
................................................                         [100%]
48 passed, 196 deselected in 22.16s
EXIT=0
```

退出码定义：**0** 干净；**2** 至少一处不一致（每条都带 typed code）；**1** 用法错误/必需文件缺失或不可解析。

## 4. 回归测试

**真实断言**（用测试内独立实现的 numpy 数学，不信任审计器）：旋转刚体性/det/四元数一致性（10 条 ASSERTED 边）、UNKNOWN 无数字、子链复现 B 终点、v2 收敛 <1e-6 且历史 0.33242 保留、`joint_anchors` 逐关节自洽、相机置换自洽、审计对已提交文件跑出 exit 0、审计不修改已提交文件（前后 sha256 相同）。

**合成变异**（均在 `tmp_path` 的**副本**上做，并重钉哈希链，使命中唯一语义码）：

| 变异 | 期望 typed code | 实测退出码 |
|---|---|---|
| 翻转一个四元数符号（q → −q） | `E_QUANTITY_DUPLICATE_MISMATCH` | 2 |
| 平移改 1 mm | `E_QUANTITY_DUPLICATE_MISMATCH` + `E_COMPOSITION_MISMATCH` | 2 |
| 往 UNKNOWN 边（13）填数 | `E_UNKNOWN_HAS_NUMBER` | 2 |
| 同一量两处不同值（mounted_chain 边 4 重述指针数值） | `E_POINTER_ONLY_VIOLATED` + `E_QUANTITY_DUPLICATE_MISMATCH` | 2 |
| 打断 json_pointer（`/edges/99`） | `E_POINTER_UNRESOLVED` | 2 |
| 篡改记录的 sha256 | `E_SHA256_MISMATCH` | 2 |
| 覆盖历史 0.33242 | `E_HISTORICAL_ERROR_CHANGED` | 2 |
| 把 v2 误差放大 1 cm | `E_V2_ERROR_TOO_LARGE` + `E_COMPOSITION_MISMATCH` | 2 |
| 破坏旋转正交性 | `E_ROTATION_NOT_ORTHONORMAL` + `E_QUAT_MATRIX_MISMATCH` | 2 |
| 破坏光轴↔安装系置换（det 仍 +1） | `E_OPTICAL_PERMUTATION_MISMATCH` + `E_QUAT_MATRIX_MISMATCH` | 2 |
| 删除必需文件 | `[E_ARTIFACT_MISSING]` | 1 |

## 5. 审计结论与仍然 UNKNOWN 的量

**结论**：已提交的 P6 变换簇内部**一致**——每条被断言的数值都能沿指针回到唯一源头且逐字节相同，10 条 ASSERTED 边都是刚体旋转、单位为米，5 条 UNKNOWN 边没有被偷填任何数，同一物理量没有出现两个互相矛盾的记录，复合链复现记录的末端位姿，`mounted_chain.json` 的历史 0.33242 m 未被覆盖而 `mounted_chain_v2.json` 把它压到 1.85e-09 m，相机光轴↔安装系的 90° 置换自洽。

**这个 PASS 只说明"没自相矛盾"，不说明"物理上已标定"**。以下仍然 UNKNOWN（本切片一个字都没补）：

1. **机体世界位姿**（边 1，`world → chassis`）：部署时 (x, y, yaw) 无计量记录；`scene_layout.py` 的 `(-1.60,0,0)` 是 ENGINEERING_V0_1 场景占位。
2. **底盘→机械臂基座安装偏置**（边 2）：螺栓孔位与偏航角未测；sim 里 `chassis==arm_base` 只是 `root_joint` 恒等式的**作者假设**（`SIM_AUTHORED_IDENTITY`），不是测量。
3. **法兰→球拍 TCP**（边 13）：`configs/racket.yaml` 的 `mount_transform` 仍 `REQUIRES_MEASUREMENT`。
4. **球拍安装插入深度/拍面朝向**（边 14）与**拍面中心**（边 15）：资产无法回答（`racket_visual.usd` 没有 mount prim）。
5. **D455 全部字段**：内参、分辨率、RGB/深度对齐、帧率、深度单位、畸变、`T_depth_color`、`T_mount_camera`——全部 UNKNOWN；`camera_metadata.json` 里唯一被证明的只是**光轴系与安装系的轴约定**（`CONVENTION_NOT_CALIBRATION`）。

另有一处**本切片范围外**的事实（不影响退出码，审计报告在 `out_of_scope_refs` 中列出）：`mounted_chain.json` 与 `mounted_chain_v2.json` 引用的 `evidence/p3_mount_frame_tree_at_0bbed19.json` 不在本分支，因此 composition A 的 pxr 跳数无法在本分支上重新验算；A 的值只在两个挂载文件之间互相比对（一致）。补齐该钉住副本即可把这 4 条 reference 纳入审计。

## 6. 复现

```
python tools/p6_frames/audit_frame_consistency.py            # exit 0
python tools/p6_frames/audit_frame_consistency.py --json     # 完整报告
python -m pytest tests/simulation/p6_frame_consistency_test.py -q
python -m pytest tests/simulation -k p6_ -q
```

CPU only，无 GPU、无 Isaac，未修改任何已提交 P6 JSON、assets/、simulation/、src/perception、src/trajectory。
