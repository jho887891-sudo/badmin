# Deployment

**状态：占位（待填写）** —— 禁止猜测环境版本，一律实测后登记。

## 本地 / 远端环境事实登记表（盘点后填写）

| 项 | 本地电脑 | 远端开发机 / GPU 服务器 |
|---|---|---|
| hostname | 待填 | 待填 |
| remote user | — | 待填 |
| 项目路径 | `E:\具身智能\badmin_project` | 待填 |
| python environment | 待填 | 待填 |
| CUDA version | 待填 | 待填 |
| GPU | 待填 | 待填 |
| Isaac Sim version | — | 待填 |
| Isaac Lab version | — | 待填 |
| PyTorch / Python | 待填 | 待填 |

## 远端执行流程（规范）
1. 查看 Git 状态与当前分支 → 2. 同步最新代码 → 3. 检查环境 → 4. 执行开发任务 → 5. 最小测试 → 6. 完整测试 → 7. 保存运行日志 → 8. 汇总结果 → 9. 更新本地项目记录

## 禁止事项（未告知风险前不得执行）
- 未确认路径就 `rm -rf`；随意删除数据集 / checkpoint
- 随意修改系统 CUDA；随意升级 PyTorch / Isaac Sim / Isaac Lab
- 未记录的环境依赖变更

## 部署路线（规划）
ROS2 真机部署节点划分 + TensorRT / ONNX 推理部署（Stage 9 前细化）。
