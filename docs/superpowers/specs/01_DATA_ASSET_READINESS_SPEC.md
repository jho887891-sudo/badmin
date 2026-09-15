# 01 — 数据与资产就绪 Spec

## 1. 目标

在任何正式训练前证明：

```text
羽毛球资产正确
→ 数据生成正确
→ 标签正确
→ 数据池隔离正确
→ 数据分布已量化
→ 才允许训练
```

## 2. 当前已确认状态

项目已有：

- `tools/build_shuttlecock.py`
- `configs/shuttlecock.yaml`
- 自建程序化羽毛球物理资产；
- 正式视觉策略 `EXTERNAL_REFERENCE`；
- 外部视觉资产 `assets/third_party/shuttlecock_visual.usd`，来源为导入 GLB 后提取的视觉体。

程序化资产主要负责：

- 几何尺寸；
- 碰撞；
- 质量分布；
- 气动参数。

外部 USD/GLB 视觉资产负责相机最终看到的外观。

已确认修复：

1. 渲染器不再独立写死羽毛球总长，而是从项目配置读取；
2. 已加入外部视觉体与项目模型尺寸一致性检查；
3. `target_px` 不再因 3 位小数记录导致确定性回放偏差。

已测结果：

- `51 tests: OK`
- train：400 张，2.45–32.86 px，中位 8.94 px
- val：120 张，2.45–31.40 px
- 尺寸控制中位误差：0.3%
- 100% 样本落在 ±15% 目标范围
- 池隔离通过
- 冻结集 0 重叠
- 精确回放误差：0.000000

已知仍未完成项：

- 剩余 31 张背景全分辨率逐张复核；
- 真实相机外观参数仍未正式标定；
- 远端 jxxy 尚未同步到当前版本。

这些未完成项必须明确记录，不得伪装为已通过。

## 3. 数据池

至少划分：

- `TRAIN_POOL`
- `VAL_POOL`
- `FIXED_CORE_TEST`
- `CHALLENGE_TEST`
- `FAILURE_RETURN_POOL`
- `HARD_NEGATIVE_POOL`

## 4. 资产检查

每次更换视觉资产或几何尺寸配置时必须重新检查：

```text
项目配置几何尺寸
vs
外部视觉资产实际几何尺寸
```

如果超过正式允许差异，则停止生成正式数据。

## 5. 数据清洗

检查：

- 损坏文件；
- 空图；
- 错图；
- 标签缺失；
- bbox 越界；
- bbox 与真实目标明显错位；
- 完全重复；
- 近重复；
- 视频连续高相似帧；
- 合成渲染异常；
- 来源重复；
- train/val/test 泄漏。

真实任务中确实存在的小目标、运动模糊、遮挡、弱光、极端姿态不得因为“难看”而被当作脏数据删除。

## 6. 数据泄漏规则

```text
同一真实视频片段的相邻帧
→ 不允许跨 train / val / test

同一合成序列中的近重复样本
→ 不允许跨集合

FIXED_CORE_TEST
→ 永不进入训练
```

## 7. 分布统计

至少统计：

- `equivalent_size_px`
- 三维姿态
- 合成 / 真实来源
- 清晰 / 模糊
- 背景类别
- 图像位置
- 正 / 负样本
- 左 / 右相机来源
- 遮挡程度

## 8. 输出

至少产生：

- `dataset_inventory.csv`
- `dataset_cleaning_report.csv`
- `dataset_distribution_report.csv`
- `dataset_leakage_report.csv`
- 数据版本标识
- manifest / hash

## 9. 验收标准

```text
资产一致性                       PASS
31 张剩余背景复核                 PASS
标签完整性                       PASS
数据清洗                         PASS
数据池隔离                       PASS
固定核心测试集无训练泄漏          PASS
尺寸/姿态/来源分布已量化          PASS
```

失败时只修数据和资产，重新运行本 Spec；禁止因为数据问题直接修改模型结构。

## 10. 退出状态

`ASSET_READY`
