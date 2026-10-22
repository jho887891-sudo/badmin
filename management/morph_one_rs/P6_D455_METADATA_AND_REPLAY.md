# P6-03（R+1 item C3）：D455 RGB/Depth 元数据合同与回放校验

本切片只做一件事：**让"没有数据"这件事无法被伪装成数据**，同时把"有数据之后怎么接进来"写成可执行的合同。
本文件不含任何一个 D455 的实测数值。仓库里仍然没有任何一个 D455 字段是已知的（见 §9）。

- 分支：`p6/d455-metadata-replay`，父分支 `p6/racket-camera-frames`（PR #26，d7672d5）
- 只读基线：`outputs/simulation/rebot_b601_rs/morph_one/p6_frames/camera_metadata.json`（本切片未改动一个字节）
- CPU-only：无 Isaac Sim、无 GPU、无渲染；**全程没有连接任何 RealSense 设备**，也没有 ssh 到 172.31.68.251（本任务不需要）
- 数据类别：`OFFLINE-SYNTHETIC`（仓库无真实 D455 采集，全部夹具为合成，且在夹具、报告、清单三处都标注）

---

## 0. 一句话结论

| 问题 | 结论 |
|---|---|
| 本切片标定了什么 | **什么都没标定**。没有 fx/fy/cx/cy、没有 depth_scale、没有外参 |
| 本切片证明了什么 | ①元数据"值必须绑定到字节"的合同；②单位守卫（mm/m、deg/rad）会拒绝错误量；③光学帧↔mount 帧置换两个方向 det=+1 且互逆；④回放校验器对 13 个夹具给出**真实退出码** 0/1 且与预期一致 |
| 下游可以直接用吗 | 只能用**接口**，不能用**数值**。任何调用 `require_calibrated_intrinsics()` 的代码现在都会抛异常——这是设计意图 |

---

## 1. 交付物

| 文件 | 作用 |
|---|---|
| `src/simulation/sensors/d455_metadata.py` | 元数据合同：`D455Metadata` / `Datum` / `UNKNOWN` / 单位守卫 / 一致性规则 / 加载器 / 光学↔mount 置换 / `ros2 bag info` 与帧目录消费者 |
| `src/simulation/sensors/d455_replay.py` | 回放校验器 + CLI（`python -m src.simulation.sensors.d455_replay`） |
| `tools/p6_frames/make_d455_replay_fixtures.py` | 合成夹具生成器（确定性，纯 python 写 PNG，无 PIL） |
| `tools/p6_frames/run_d455_replay_matrix.py` | 逐夹具跑校验器并记录**真实退出码**，与预期不符则自身退出 1 |
| `tests/simulation/fixtures/p6_d455_replay/` | 13 个夹具（含 `manifest.json` 逐文件 sha256；手改夹具会让测试红） |
| `tests/simulation/p6_d455_metadata_test.py` | 59 个测试：合同与拒绝路径 |
| `tests/simulation/p6_d455_replay_test.py` | 26 个测试：好/坏夹具、CLI 退出码、夹具自校验 |
| `outputs/.../p6_frames/evidence/d455_replay_*.json` | 13 份机器可读报告 + `d455_replay_matrix.json` 汇总 |

---

## 2. 元数据合同（`p6_d455_metadata/1`）

### 2.1 一个值不是一个数，是一个"有字节来源的值"

字段只有两种合法形态，没有第三种：

```json
{"value": 0.001, "unit": "m_per_unit",
 "source": {"file": "configs/camera/d455_<serial>_<mode>.json",
            "sha256": "<64 hex>", "locator": "intrinsics.depth_scale"}}
```

或

```json
{"status": "UNKNOWN", "missing_datum": "per-device depth scale never read from a unit"}
```

- `D455Metadata` 的 18 个字段**默认值全部是 `UNKNOWN` 哨兵**，不是 0、不是 None、不是任何"看起来像数据"的数。
  `test_no_field_has_a_default_that_looks_like_data` 与 `test_dataclass_has_no_numeric_field_defaults` 逐字段断言这一点。
- `UNKNOWN` 是 falsy、不是数字、不可加：`bool(UNKNOWN) is False`，`isinstance(UNKNOWN, (int, float)) is False`。
- 只有 `Datum(value, source, unit)` 才算已知，`is_known()` 只认 `Datum`。

### 2.2 provenance 是真的被验证的，不是写着好看

加载时对每个值做三件事，任一不成立就**整条记录拒绝**：

1. `source.file` 存在（不存在 → `ProvenanceViolationError`）；
2. 文件实际 sha256 == 声明 sha256（不匹配 → 拒绝）；
3. `source.locator` 在该文件里能解析到（解析不到 → 拒绝）。

`file: @self` 表示"就是这条记录本身"，此时 loader **自己计算**该记录的 sha256 填进去，
调用方自己写的 sha256 若与实测不符也会被拒（`test_self_declared_sha256_mismatch_is_rejected`）。

### 2.3 单位必须是显式且规范的

`units` 块是**必需**的（缺了就 `UnitViolationError`）：

| 键 | 规范单位 | 还接受的拼写 |
|---|---|---|
| intrinsics | `px` | pixel, pixels |
| depth_scale | `m_per_unit` | meter_per_unit, metre_per_unit |
| length | `m` | meter, metre, meters, metres |
| angle | `rad` | radian, radians |
| rate | `Hz` | hz, fps, 1/s |
| distortion | `dimensionless` | 1, -, none |
| time | `s` | sec, second, seconds |

**loader 永不静默换算。** 出现 `mm_per_unit` / `deg` / `mm` 一律拒绝并提示规范单位——
换算是一个决定，必须由记录的主人做，不能由一个加载函数偷偷做。

---

## 3. 一致性规则（"有值但互相矛盾"也拒绝）

| 规则 | 拒绝示例 | 测试 |
|---|---|---|
| fx > 0, fy > 0 | fx = 0 / -1 | `test_nonpositive_values_are_rejected` |
| depth_scale > 0 | 0.0 | 同上 |
| fps > 0 | 0.0 | 同上 |
| width/height > 0 | 0 | 同上 |
| 主点在图像内：0 <= cx < width, 0 <= cy < height | cx = 99（8 宽图像） | `test_principal_point_outside_the_image_is_rejected` |
| 畸变系数个数 == 模型要求 | `brown_conrady` 给 4 个 | `test_distortion_model_with_wrong_coefficient_count_is_rejected` |
| 模型名必须在表里 | `magic_lens` | `test_unknown_distortion_model_is_rejected` |
| 有模型就不能缺系数 | 删掉系数向量 | `test_distortion_model_without_coefficients_is_rejected` |
| `rgb_depth_aligned` 必须是 JSON bool | 1 | `test_bool_field_that_is_not_a_bool_is_rejected` |
| `T_mount_camera` 必须是 SE(3)：正交、det=+1、底行 [0,0,0,1] | 镜像旋转 det=-1 / 底行 0.5 | `test_mirrored_mount_rotation_is_rejected_by_consistency`、`test_bottom_row_of_the_mount_transform_is_validated` |
| 未声明的字段名 | `fx_mm` | `test_unknown_field_name_is_rejected` |

### 3.1 "calibrated" 是两个级别，不是一个是非

| 状态 | 含义 | 现有仓库 |
|---|---|---|
| `INCOMPLETE` | 缺必需字段 | ← **当前就是这个** |
| `CALIBRATED_INTRINSICS` | 内参/分辨率/深度单位/fps/对齐/畸变齐了；外参可以仍 UNKNOWN | 无 |
| `CALIBRATED_FULL` | 再加 `T_mount_camera` | 无 |

`require_calibrated_intrinsics()` / `require_calibrated_full()` 在缺字段时抛
`MetadataIncompleteError`，异常里带**具体缺哪几个字段名**。缺席不是异常，谎称齐了才是。

---

## 4. 单位守卫的红/绿测试（本切片的重点）

| 场景 | 期望 | 实际测试 | 结果 |
|---|---|---|---|
| `units.depth_scale = "mm_per_unit"`（毫米/单位） | 拒绝 | `test_millimetre_depth_scale_unit_is_rejected` | PASS |
| 字段级 `unit = "mm_per_unit"` | 拒绝 | `test_millimetre_depth_scale_on_the_field_is_rejected` | PASS |
| 深度标尺字段缺 `unit` | 拒绝 | `test_depth_scale_missing_its_unit_is_rejected` | PASS |
| 值的量级是毫米（1.0）却写在 m_per_unit 字段 | 拒绝（1000 倍误差） | `test_millimetre_value_in_a_metres_field_is_rejected`，夹具 `mm_value_in_metres_field/` | PASS |
| `units.angle = "deg"` | 拒绝 | `test_degree_valued_unit_is_rejected` | PASS |
| FOV = 87.0 写在 `hfov_rad` 里（度数伪装弧度） | 拒绝 | `test_degree_valued_fov_in_a_radian_field_is_rejected`，夹具 `deg_fov/` | PASS |
| FOV = pi/2 rad（合法） | 接受 | `test_radian_valued_fov_is_accepted` | PASS |
| 深度 0.001 m/unit（合法） | 接受 | `test_good_fixture_becomes_calibrated_and_keeps_provenance` | PASS |

量级守卫本身不是设备数值：`DEPTH_SCALE_M_PER_UNIT_MAX = 0.1` 只用于让"1.0 写成 m/unit"这种毫米值过不去
（16 位深度 0.1 m/unit 已是 6.5 km 行程）。代码注释里写明了这一点，防止有人把它当 D455 规格。

---

## 5. 光学帧 ↔ mount 帧（唯一被证明的事）

仓库唯一已证明的相机事实是**轴向约定**，不是标定：

```text
(X, Y, Z)_mount   = (Z, -X, -Y)_optical
(X, Y, Z)_optical = (-Y, -Z, X)_mount          # 上一行的逆
R(optical <- mount) = [[0,-1,0],[0,0,-1],[1,0,0]]     det = +1
```

- 两个方向都实现为矩阵常量（`OPTICAL_TO_MOUNT` / `MOUNT_TO_OPTICAL`），各自 det=+1、列正交，
  且乘积在 1e-12 内为单位阵（`test_permutation_has_determinant_plus_one_in_both_directions`、`test_permutation_round_trip_is_identity`）。
- `test_committed_p6_camera_record_matches_this_module` 读**已提交**的 `camera_metadata.json`
  （只读）并核对它的矩阵 == `MOUNT_TO_OPTICAL`、四元数解码 == 同一个矩阵、det=+1。**方向和字节都绑定到证据文件**。
- ⚠️ **与任务文字的一处差异（必须写明）**：任务写"(X,Y,Z)_optical=(Z,-X,-Y)_mount"，但已提交记录写的是
  *"a depth point (X,Y,Z)_optical is (Z,-X,-Y) in the mount frame"*，即 **optical→mount** 是 (Z,-X,-Y)。
  已提交记录与矩阵、四元数三者自洽，本切片以记录为准，两个方向分开实现、绝不混用；
  `test_permutation_directions_are_exact_and_opposite` 与 `test_committed_record_distinction_rule_matches_the_inverse_label` 把这个差异钉死。
  置换本身（循环换轴、det=+1）与任务描述一致，只有箭头方向不同。
- `mount` 帧未知时**拒绝**做 SE(3)：`optical_point_to_mount()` 抛 `MountFrameUnknownError`
  （"需要手眼标定或实测 CAD mount，轴向约定不能把点放到机器人上"）。只有 `axis_convention_only_point_to_mount()`
  在名字里明说"只给约定、不落地"，供人观察 90 度错误本身。

---

## 6. 回放校验器

输入布局（真实采集可直接沿用）：

```text
<replay>/
  index.json      # {schema, data_class, resolution, fps, rgb_depth_aligned,
                  #  depth:{scale_m_per_unit, unit, encoding}, frames:[{rgb, depth, timestamp_s}]}
  rgb/000000.png ...
  depth/000000.png ...
  metadata.json   # D455Metadata 记录（可选但强烈建议；缺了就等于内参 UNKNOWN）
```

| 检查 id | 内容 | 违规码 |
|---|---|---|
| index_parsable / index_schema | JSON + `p6_d455_replay_index/1` | INDEX_UNPARSABLE / INDEX_SCHEMA / INDEX_MISSING |
| data_class | 必须显式声明 `OFFLINE-SYNTHETIC` 或 `REAL-CAPTURE` | DATA_CLASS_UNDECLARED |
| frame_files | 每个 triplet 的 rgb/depth 文件存在 | MISSING_FRAME_FILE |
| image_headers / image_size | PNG 头可读，且尺寸 == 索引声明 | IMAGE_UNREADABLE / IMAGE_SIZE_MISMATCH |
| frame_counts | rgb 数 == depth 数 | FRAME_COUNT_MISMATCH |
| （目录） | `rgb/ depth/` 实际文件数 == 索引条数 | DIRECTORY_COUNT_MISMATCH |
| timestamps_monotonic | **严格递增**（dt > 0，等值也算违规） | NON_MONOTONIC_TIMESTAMP / TIMESTAMP_MISSING |
| frame_rate + timestamp_gaps | 阈值**写成公式**：gap 当且仅当 dt > max_gap_factor / fps，默认 1.5 | FPS_UNKNOWN / TIMESTAMP_GAP_EXCEEDED |
| depth_units | 单位必须 `m_per_unit`，且 0 < scale < 0.1 | DEPTH_UNIT_NOT_METRES / DEPTH_SCALE_IMPLAUSIBLE / DEPTH_SCALE_ABSENT |
| rgb_depth_alignment | 必须是 bool，不许默认 | ALIGNMENT_UNKNOWN |
| intrinsics | 必须有元数据记录且内参齐 | INTRINSICS_UNKNOWN / METADATA_INVALID |
| resolution | 元数据分辨率 == 索引分辨率 == PNG 头 | RESOLUTION_MISMATCH / RESOLUTION_UNDECLARED |

**退出码：0 = PASS，1 = 有违规，2 = 用法/IO 错误。** 报告是 JSON（`p6_d455_replay_report/1`），
固定带 `calibration_claimed: false` 与 `data_class`。

---

## 7. 合成夹具矩阵（真实退出码，不是声称）

`python tools/p6_frames/run_d455_replay_matrix.py` → **exit 0**，"PASS: 13 fixtures matched their expectations"：

| 夹具 | 退出码 | 违规码 | 与预期一致 |
|---|---|---|---|
| good | 0 | — | 是 |
| external_witness | 0 | — | 是 |
| missing_frames | 1 | MISSING_FRAME_FILE, FRAME_COUNT_MISMATCH, DIRECTORY_COUNT_MISMATCH | 是 |
| non_monotonic | 1 | NON_MONOTONIC_TIMESTAMP, TIMESTAMP_GAP_EXCEEDED | 是 |
| gap | 1 | TIMESTAMP_GAP_EXCEEDED | 是 |
| mm_depth | 1 | DEPTH_UNIT_NOT_METRES | 是 |
| mm_value_in_metres_field | 1 | METADATA_INVALID | 是 |
| unknown_intrinsics | 1 | INTRINSICS_UNKNOWN | 是 |
| bad_sha | 1 | METADATA_INVALID | 是 |
| deg_fov | 1 | METADATA_INVALID | 是 |
| no_fps | 1 | FPS_UNKNOWN, INTRINSICS_UNKNOWN | 是 |
| alignment_undeclared | 1 | ALIGNMENT_UNKNOWN, INTRINSICS_UNKNOWN | 是 |
| no_metadata_record | 1 | INTRINSICS_UNKNOWN | 是 |

夹具里的数值**故意不像 D455**：8x6 图像、fx=fy=4.0、cx=3.5、cy=2.5、depth_scale=0.001 m/unit。
真实 D455 彩色流不是 8x6，fx 也不是 4 px，所以这些数不可能被误读成标定。
每个记录里都有 `synthetic_fixture.is_real_d455_data = false` 与
"SYNTHETIC FIXTURE - NOT A D455 MEASUREMENT" 字样；`manifest.json` 记录逐文件 sha256，
测试 `test_fixture_manifest_matches_the_files_on_disk` 与 `test_fixture_generation_is_deterministic` 保证手改/漂移都会红。

---

## 8. 真实命令与真实退出码

```text
# 环境（本机实测）
python --version                       # Python 3.11.9
python -m pytest --version             # pytest 9.0.2
python -c "import numpy; print(numpy.__version__)"   # 2.4.6

# 1) 生成合成夹具（确定性）
python tools/p6_frames/make_d455_replay_fixtures.py                    # exit 0
#   wrote tests/simulation/fixtures/p6_d455_replay (206 paths)

# 2) 本切片全部测试
python -m pytest tests/simulation/p6_d455_metadata_test.py tests/simulation/p6_d455_replay_test.py -q
#   85 passed in 7.10s                                              exit 0

# 3) 基线上的 p6 套件（必须保持绿）
python -m pytest tests/simulation -k p6_ -q
#   114 passed, 196 deselected in 8.86s                             exit 0

# 4) 整个 tests/simulation
python -m pytest tests/simulation -q
#   264 passed, 46 skipped in 21.96s                                exit 0

# 5) 逐夹具真实退出码矩阵 + 证据 JSON
python tools/p6_frames/run_d455_replay_matrix.py                       # exit 0
#   PASS: 13 fixtures matched their expectations

# 6) 单个夹具，给人看
python -m src.simulation.sensors.d455_replay tests/simulation/fixtures/p6_d455_replay/good     --metadata tests/simulation/fixtures/p6_d455_replay/good/metadata.json --quiet
#   verdict=PASS data_class=OFFLINE-SYNTHETIC violations=0 exit=0    exit 0

python -m src.simulation.sensors.d455_replay tests/simulation/fixtures/p6_d455_replay/mm_depth     --metadata tests/simulation/fixtures/p6_d455_replay/mm_depth/metadata.json --quiet
#   verdict=FAIL ... violations=1 exit=1                             exit 1

python -m src.simulation.sensors.d455_replay tests/simulation/fixtures/p6_d455_replay/good --max-gap-factor 0
#   usage/IO error: ...                                              exit 2
```

> 基线对照：在 `d7672d5` 上先跑过一次 `python -m pytest tests/simulation -k p6_ -q` = **29 passed, 196 deselected, exit 0**；
> 加入本切片后同一条命令 = **114 passed, 196 deselected, exit 0**（29 + 85），没有旧测试变化。

---

## 9. 仍然 UNKNOWN 的字段，以及**到底**要什么才能填

一个都没有填。下表就是 `camera_metadata.json` 的 UNKNOWN 与本合同的 18 个字段的对应关系，
以及"拿来什么文件/做哪个测量"才能合法地把它变成 `Datum`。

| 字段 | 状态 | 填它需要的**具体**东西 | 落地路径（建议） |
|---|---|---|---|
| fx_px fy_px cx_px cy_px | UNKNOWN | 在装好设备的机器上，用**最终要跑的那个流 profile** 读一次内参：`rs-enumerate-devices -c` 或 `pipeline.get_active_profile().get_stream(rs.stream.depth/color).as_video_stream_profile().get_intrinsics()` | `configs/camera/d455_<serial>_<mode>.json`（提交） |
| width_px height_px | UNKNOWN | 与内参同一次读取里的 profile 名（例如 848x480x30 / 1280x720x30）；**不是**独立假设值 | 同上 |
| depth_scale_m_per_unit | UNKNOWN | `depth_sensor.get_depth_scale()`（**每台设备**不同，绝不能硬编码 0.001） | 同文件，与序列号同行 |
| depth_encoding | UNKNOWN | 深度流实际格式（通常 `uint16`），从 pipeline config 读 | 同文件 |
| fps | UNKNOWN | pipeline config 里配置的帧率，深/彩是否一致 | 同文件 + 一份时间戳日志（P6-06 管辖） |
| rgb_depth_aligned | UNKNOWN | 代码里是否调用 `align(rs.stream.color)`（一个决定，不是一个默认）；关掉时还需要 `depth_profile.get_extrinsics_to(color_profile)` | 同文件 + pipeline 源码位置 |
| distortion_model distortion_coefficients | UNKNOWN | 彩色流的畸变模型与系数：出厂标定读数（最低），棋盘格/charuco 重标定（最诚实的上限） | 同文件 |
| T_mount_camera | UNKNOWN | **机器人测量**，不是相机事实：手眼标定，或实测的 CAD 安装面（含平移与倾角） | 独立记录（例如 `configs/camera/d455_mount_handeye.json`） |
| hfov_rad vfov_rad | UNKNOWN（可选） | 由上述内参 + 分辨率换算即可，不需要单独测量；因此本切片**不**给它任何默认值 | 由 fx/cx/width 推出 |
| device_model device_serial device_firmware | UNKNOWN | `rs-enumerate-devices -s` 的原始输出（含序列号、固件版本） | `configs/camera/d455_<serial>_enumerate.txt`（提交原文） |

**已被显式拒绝的替代品**（沿用 `camera_metadata.json` 的判定，不重复发明）：
`simulation/badminton_scene/scene_layout.py:87` 的 rig 占位、`docs/ALGORITHM_AUDIT.md:56` 的 fx=700 代理标定、
`outputs/shuttle_detection/capability/CONTROLLED_MATRIX_RUN1.md:20` 的 fx=11200 长焦渲染相机。
这三者都不能作为 `Datum` 的 `source`：它们不是 D455 的测量，而且 `provenance` 校验只接受"能解析到该值并且字节哈希对得上"的文件。

---

## 10. 明确声明：**NO CALIBRATION CLAIMED**

> 本切片没有标定任何相机，没有测量任何 RealSense D455，也没有产生任何可用于几何计算的相机数值。
> 所有被 `D455Metadata` 接受的数值都来自**合成夹具**（`OFFLINE-SYNTHETIC`），并且每个夹具记录都自带
> `is_real_d455_data: false`；每份回放报告都自带 `calibration_claimed: false`。
> 仓库中没有任何内参、深度标尺、帧率或外参的 D455 实测值是已知的。
> 任何下游代码如果读到"已标定"的对象，那一定来自后来提交的真实记录，而不是本切片。

---

## 11. 未做 / 不冒充 DONE

- **P6-06**（时间戳/延迟/掉帧协议）仍归它自己：本切片只消费 `timestamp_s` 并检查单调性与 gap，
  不定义时钟源、不测延迟、不建硬件同步。
- **未连接设备**：没有 `rs-enumerate-devices` 的真实输出，因此 §9 每一行都还是 UNKNOWN。
- **未做手眼标定**：`T_mount_camera` 在真实记录里仍然必须是 UNKNOWN；夹具里的那个是"约定旋转 + 零平移"的占位，
  记录里已用 `T_mount_camera_note` 明说是占位、不是标定。
- **真实 bag 未解析**：只写了 `ros2 bag info` 文本解析（只给话题/条数，永远给不出内参），没有装 ROS、没有读 bag 二进制。
- 基线 `outputs/.../p6_frames/camera_metadata.json`、`frame_tree.json`、球拍代理、`assets/`、
  `simulation/`、`src/perception`、`src/trajectory`、队列状态文件、`PHASE_P2_*/P3_*/P4_*` 文档**均未改动**
  （`git status --porcelain` 对这些路径为空）。
