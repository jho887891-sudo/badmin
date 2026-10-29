# badmin_project 融合记录

来源：`/home/T7/ojh/badmin_project`
目标：`/home/T7/dgut/robot_sim`

已将 configs、docs、experiments、management、scripts、simulation、src、tests、tools 中目标原先不存在的 145 个文件按原相对路径补入 robot_sim。使用 `cp -an`，未覆盖已有文件。另将项目根目录的 README、PROJECT_STATUS、TODO、ROADMAP、CHANGELOG 放入 `docs/badmin_project/` 作为来源记录。

- 代码/文档源路径核验：399 个文件
- 原内容与目标 SHA256 一致：391 个
- 8 个不同的同路径文件：保留 robot_sim 版本，冲突文件名见 CSV
- 未复制数据集、权重、outputs、runs、assets 等资源产物。
- 文件 SHA256 清单：`management/BADMIN_PROJECT_INTEGRATION.csv`
- 两个服务器目录均无可用 Git 元数据；本次为文件级融合，未创建或改写 Git 历史。
