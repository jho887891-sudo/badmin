# Changelog

格式参考 [Keep a Changelog](https://keepachangelog.com/zh-CN/1.0.0/)，版本号 `MAJOR.MINOR.PATCH`。

## [0.1.0] - 2026-09-08
### Added
- 项目知识库结构初始化：根状态文件（README / PROJECT_STATUS / TODO / ROADMAP）+ docs 与 management 种子文件
- 实验索引与目录占位（experiments / configs / scripts / src / tests / assets / tools）
- Git 仓库初始化（空基线，无代码）

### Notes
- 按用户约定：本阶段**不写算法代码**，先建立项目管理结构并盘点现状。

## [0.2.0] - 2026-09-08
### Added
- 远端仿真环境安装完成：Isaac Sim 6.0.1 + Isaac Lab v3.0.0-beta2.patch1（Python3.12/uv，torch 2.10.0+cu128）
- 最小验证通过：Cartpole-Direct-v0 20 步运行正常，无 GPU 报错
### Known
- 可选 extras（mimic/rl-games 等 git 依赖）待 github 网络稳定后补装
