# P4-08 证据可审计性补记（A05 联调）

## 起因

用 A05（`queue/a05-a06-a09` 的 `tools/queue/experiment_meta.py validate`）审计 P4 证据时，
**9 个采样 run 中有 3 个被拒**：

```
FAIL (A05_VALIDATION): COMMAND_UNPARSED: manifest.command carries no --gravity-z,
so the gravity actually applied is not evidenced
```

这是**真实缺陷**：P4 脚本的 `--gravity-z` 有默认值 −9.81，实际生效的重力只存在于脚本默认值里，
命令行没有留下证据。A05 的判断是对的——"默认值"不是证据。

## 处理（不修改任何门禁阈值）

1. 脚本 manifest 新增 `gravity_explicit_on_command_line` 与 `applied_gravity_argv`；
   这两个字段**只记录事实**，不改变物理。
2. **用显式 `--gravity-z -9.81` 重跑整个 18-run 矩阵**（RUN_ID 后缀 `r11/r12`），旧 run（`r01/r02`）**原样保留**，
   并在报告里标注其重力"未经命令行举证"。
3. P4-09 三组配置同样以显式重力重跑（`P4V_*_r13/r14`）。

## 结果

| 检查 | 结果 |
|---|---|
| 聚合器（36 runs，含旧+新） | `verdict CONSISTENT`，`problems []`，**exit 0** |
| A05 元数据审计（11 个采样：9 组矩阵 + 2 个 P4V） | **PASS=11 / FAIL=0** |
| 新旧 run 数值 | **逐位一致**（重力默认值与显式 −9.81 是同一物理值，重跑只为留证据） |
| 配置哈希 | `P4V_clamp_r13` → `sha256:ec5d6911f076…`，assets 5 |

## 结论

- P4-08 / P4-09 的**结论不变**（摩擦弱依赖 0.46 %、载荷 −0.05 %、里程与轮速 1:3.999、
  有效滚动半径 0.10103–0.10161 m、速度追踪 6.0 %、限速与停车通过）。
- 变化只有一条：**证据链现在能被 A05 独立校验**，不再依赖"脚本默认值是 −9.81"这种口头说明。
- 旧 run 不删除、不改写；两者的差异（命令行是否举证重力）已在此文档中明确区分。
