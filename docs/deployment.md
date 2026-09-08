# Deployment

**状态：部分已核实（2026-09-08 实机检查）** —— 禁止猜测环境版本；下表"已核实"项均为实测，其余待装/待查。

## 环境事实登记表

| 项 | 本地电脑（SSOT） | 远端开发机 / GPU 服务器 |
|---|---|---|
| hostname | （Windows 本机，待登记） | jxxy（已核实） |
| SSH 连接 | — | dgut@172.31.68.251:22，密码认证（已核实 2026-09-08） |
| 项目路径 | `E:\具身智能\badmin_project` | **待定**（候选 /home/T7/ojh，未确认） |
| python | 待查 | 3.10.12（系统 /usr/bin/python3，已核实） |
| conda / venv | 待查 | 未见 conda 环境（已核实为空） |
| CUDA toolkit | — | **未安装**（PATH 无 nvcc）；驱动 550.163.01 支持 CUDA 12.x（已核实） |
| GPU | 无 | NVIDIA RTX A6000 49GB（已核实） |
| PyTorch | 待查 | **未安装**（需与 CUDA 匹配后选版） |
| Isaac Sim | 未安装 | **未安装**（待搭建） |
| Isaac Lab | 未安装 | **未安装**（待搭建） |
| docker | — | docker + nvidia-docker 已装，无运行容器（已核实） |
| 核实日期 | — | 2026-09-08 |

> 结论：远端 GPU 机 = "空白但强大"的 Isaac 训练候选（A6000 49GB 满足 Isaac Lab 需求）。
> Isaac Sim / Isaac Lab / PyTorch / conda 的**版本选择是未决决策**，选定后立即登记，禁止随意升级。

## 远端执行流程（规范）
1. 查看 Git 状态与当前分支 → 2. 同步最新代码 → 3. 检查环境 → 4. 执行开发任务 → 5. 最小测试 → 6. 完整测试 → 7. 保存运行日志 → 8. 汇总结果 → 9. 更新本地项目记录

## 禁止事项（未告知风险前不得执行）
- 未确认路径就 `rm -rf`；随意删除数据集 / checkpoint
- 随意修改系统 CUDA；随意升级 PyTorch / Isaac Sim / Isaac Lab
- 未记录的环境依赖变更

## 部署路线（规划）
ROS2 真机部署节点划分 + TensorRT / ONNX 推理部署（Stage 9 前细化）。

## 磁盘与共享资源（2026-09-08 实测，重要）

| 项 | 实测值 | 含义 |
|---|---|---|
| 根分区 /（xfs, LVM 280G） | **容量 100%**（可用 356K）；**inode 100%**（剩 873） | 根文件系统已满，dgut 家目录基本无法再写任何文件 |
| /home/dgut | 约 164GB（根分区最大占用者） | 含用户其它项目，不可擅删 |
| /home/T7（8T 磁盘 vdb2） | NTFS/FUSE（fuseblk），已用 5.5T，**可用 1.6T**；dgut 可写（实测） | 唯一的可用大空间 |
| 未挂载分区 | vdb1 = 1T NTFS 未挂载 | 潜在可用（需管理员） |
| GPU 占用 | vLLM 进程 PID 224065 常驻 ~24.8GB VRAM（0% 利用率） | 实际可用显存约剩 24GB |
| RAM | 58GB 总；可用约 32GB；**swap 17GB 已用满** | 内存压力高（vLLM 等） |
| dgut 家目录可清理项 | ~/.cache 17G、.npm 0.4G、（.local 16G 需甄别勿乱删） | 清理可同时释放字节与 inode |

清理候选（未经用户批准不动）：~/.cache（17G，安全）；系统侧 /snap 24G、/usr/local 20G、docker snap 数据（docker images 58G，位置在 /var/snap 下，需管理员）。
