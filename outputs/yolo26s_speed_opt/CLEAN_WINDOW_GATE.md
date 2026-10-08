# Clean-window gate - 延迟测量的准入条件

实验 YOLO26S_INFERENCE_SPEED_OPT_V1 | 2026-10-05 | 主机 dgut@jxxy.taildd42cc.ts.net

## 为什么必须有这道门（有数据）

A6000 是共享卡。同一天、同一 harness、同一 checkpoint，只差背景负载：

| run | 背景 GPU | median | 结论 |
|---|---|---|---|
| Track A run1 (FP32) | 起始 0 % | 16.073 ms | 可用 |
| Track A run2 (FP32) | 起始 0 % | 15.734 ms | 可用 |
| Track B run1 (FP16) | 40 % -> 97 % | 20.838 ms | 不可用 |
| Track B run2 (FP16) | 35 % -> 99 % | 20.903 ms | 不可用 |
| Track C run1 (FP16+compile) | 33 % -> 87 % | 18.110 ms | 不可用 |

把这张表直读，会得出 FP16 让推理退化 31% - 这是错的。
spec section 17 的记录字段表里没有背景负载这一项，所以必须另设这道门判定一次运行是否准入。
这也是为什么每条记录里额外记了 gpu_snapshot_start / gpu_snapshot_end：没有它，上表无法被审计。

## 判据

    gpu_util        <= 10 %
    gpu_mem_used    <= 1200 MiB
    load1           <= 4.0        (32 核约 12.5%)
    连续 3 次轮询全部通过才放行
    轮询 60 s，最长等待 7200 s

只读实现：tools/remote/wait_for_clean_window.sh（远端副本 /home/T7/dgut/trt_env/）

    bash wait_for_clean_window.sh --check   # 判定一次，打印 VERDICT: CLEAN / BUSY
    bash wait_for_clean_window.sh           # 等连续 3 次干净才返回 0

实测 2026-10-05T18:00:32Z（当时正被他人训练占卡）：

    gpu_util=10   mem=7321   load1=6.01   compute_apps=1  -> NO
    VERDICT: BUSY

## 与 canonical baseline 的关系

canonical baseline 只采用通过本门的两次 Track A：15.90 ms（16.073 与 15.734 的中位，run-to-run 差 2.1 %）。
历史参考 15.62 ms 按用户决定降级为 historical reference；它恰好落在两次之间，属于被复现但不作为基线。

| 等级 | 门槛 |
|---|---|
| 最小有用（-20%） | <= 12.72 ms |
| 强 | <= 10 ms |
| 优秀 | <= 8 ms |

任何未通过本门的 B/C/D/E/F 数据，一律不进 section 18 主表。
