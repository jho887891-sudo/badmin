# R11 M1 独立离线协方差重算（2026-10-22）

这是对 **PR #56 已提交 tracks.jsonl** 的只读计算，不是再次运行 YOLO 或修改旧 Run。重算代码：`tools/vision/audit_r11_covariance.py`，对 [x_px,y_px,vx_px/s,vy_px/s] 的 **2×2 xy block** 求最大位置特征值平方根。原代码错用整个 4×4 混合单位矩阵求特征值。

计算来源：`r11/vision-m1@a967c82533eb67c509d92a091acf97367f1d6d59`

| Run | horizon | 有效帧 n | 旧报告 σ 中位 (px, 有误) | 位置 σ 中位 (px) |
|---|---|---:|---:|---:|
| `R11_M1_ethseq_v2` | 0.3s | 59 | 614.46 | 95.33 |
| `R11_M1_ethseq_v2` | 0.5s | 59 | 1035.91 | 254.96 |
| `R11_M1_ethseq_v2` | 0.8s | 59 | 1727.15 | 644.75 |
| `R11_M1_ethseq_calib` | 0.3s | 59 | 15.17 | 4.37 |
| `R11_M1_ethseq_calib` | 0.5s | 59 | 24.25 | 7.52 |
| `R11_M1_ethseq_calib` | 0.8s | 59 | 39.61 | 15.90 |

时钟混杂：
- `ethseq_v2` 前四时间戳：[0.533333,4.333333,4.366667,4.4]，最大间隔 **3.800000 s**。
- `ethseq_calib` 前四时间戳：[0,0.033333,0.066667,0.1]，最大间隔 **0.033334 s**。
- 因时间轴不一样，旧报告 7.78 → 18.51 px 的误差差异**不能归因于 process noise 变化单一变量**。

验证层级：**OFFLINE_REPLAY_FROM_COMMITTED_JSONL**。本次会话当前容器仅对独立小型数值函数及审计函数的 **8 个 CPU 测试得到 PASS**，尚未访问远程 YOLO checkpoint、视频原文件或 Isaac/Kit，也未复跑整个项目 pytest/新结果。
