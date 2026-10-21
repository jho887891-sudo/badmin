# P4-09 结果：代理速度闭环 / 停车 / 限速（TEST_PROXY）

**分支**：`p3/gate-v1.1@59e19a2` · **预注册**：`PHASE_P4_09_PREREG.md`（运行前提交；**门禁阈值一个都没改**）
**定级**：**P4_PROXY**（圆柱踏面、TEST_PROXY 质量/惯量、`r_eff=0.101459 m` 来自 P4-08 实测）。`P4_REAL` 仍未通过。

## 1. 结果（每组 2 个独立 RUN_ID，数值逐位一致）

| RUN_ID | 命令 (raw→clamped) m/s | 保持窗均速 m/s | 相对误差 | 峰值 m/s | 归零后 0.5 s 内最大速度 m/s | gate |
|---|---|---|---|---|---|---|
| `P4V_030_r09` / `r10` | 0.30 → 0.30 | **0.318110311** | **0.0604** | 0.319605 | **8.58e-05** | **PASS** |
| `P4V_050_r09` / `r10` | 0.50 → 0.50 | **0.53007437** | **0.0601** | 0.532637 | **1.14e-04** | **PASS** |
| `P4V_clamp_r11` / `r12` | **0.80 → 0.50** | **0.53007437** | **0.0601** | 0.532637 | 1.14e-04 | **PASS**（`clamped=true`） |

- 三组门禁全过：`steady_speed_tracks_command`（≤0.15）、`stops_after_zero_command`（≤0.02 m/s）、`speed_limit_respected`（≤1.1·v+0.02）、`no_root_pose_write`、`contact_detected`、`all_finite`。
- **限速真实生效**：命令 0.80 被钳到 0.50，实测 0.5301 m/s（峰值 0.5326 ≤ 0.57 门禁）。
- **停车**：命令归零后 0.5 s 内速度降到 **1e-4 m/s** 量级。
- 稳定追踪误差 **6.0 %**（同号同值），来自 ff 半径 0.101459 与当前工况有效半径的差；这是**已知的代理偏置**，不是控制器失效（P4-08 实测 r_eff 随速度/摩擦在 0.10103–0.10161 间变化）。

## 2. 控制器演进与三次被门禁抓住的缺陷（全部保留）

| 版本 | 现象 | 根因 | 门禁 |
|---|---|---|---|
| v1 比例 | 命令 0.30，实测 **0.9138 m/s** | `ω=-(v_t+K(v_t-v_m))/r` 对上一帧实测速度的增益是 **−K**，不动点是 `(1+K)/(1−K)·v_t=−3·v_t` | `steady_speed_tracks_command` FAIL（err 2.05） |
| v2 前馈+积分 | 命令 0.30，实测 **0.3645 m/s** | 斜坡期积分器持续累加并在 20 % 处饱和 | 同上 FAIL（err 0.215） |
| v3 | PASS（6.0 %） | 保持段起点重置积分、增益减半、钳位 5 % | — |
| v4 限速 | 命令 0.80/限 0.50，实测 **0.849 m/s** | 钳位算进 `commanded_v` 但环路仍用未钳位的 `v_cmd` | `speed_limit_respected` FAIL |

→ **三个真实 bug 都是被预注册门禁抓住的**，不是我事后发现；失败 run (`P4V_030_r03/r05/r07`、`P4V_clamp_r09`) 全部保留在证据目录。

## 3. 明确边界

- 这是**代理**闭环：不做真实电机扭矩/电流限制、编码器噪声、通信延迟、电池电压下垂。
- 停车测试**不是**安全级急停（无 E-stop 硬件、无安全继电器），仅证明"命令归零→速度收敛"。
- 速度上限 0.50 m/s 是**代理策略值**，不是机器人规格。
- 未测坡道、侧滑、单轮悬空、横向扰动、`P4_REAL` 装载滚动半径。

## 4. 复现

```bash
$PY tools/p2_gate_runner.py --script scripts/simulation/p4_proxy_wheel_drive.py \
    --run-id <ID> --out-dir <DIR> --timeout 1500 -- --profile velocity_control \
    --read-contacts --velocity-cmd 0.30|0.50|0.80 --speed-limit 0.50
```

证据：`outputs/simulation/rebot_b601_rs/morph_one/p2_wc_gates/P4V_*/`（manifest、raw_poses.jsonl 601 帧、
result.json 含 `metrics.velocity_control`、gate_runner_summary.json、child 日志、outer.log）。
