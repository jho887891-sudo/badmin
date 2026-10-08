# TraeWork 装到远端 Linux 主机 —— 可行性核查

目标机：`dgut@172.31.68.251`（hostname `jxxy`，Ubuntu 22.04.5 LTS，x86_64）
核查方式：官方文档 / 官方社区回复 + 远端只读实测。**本轮未执行任何下载或安装。**

## 结论

**远端 Linux 装不了 TraeWork 桌面版。** 官方明确没有 Linux 客户端，且明确表示近期也没有计划。
官网在 Linux 上显示下载入口是**前端 bug**（官方原话）。

## 证据（官方来源）

| # | 来源 | 日期 | 原文要点 |
|---|---|---|---|
| 1 | `forum.trae.cn/t/topic/73597`（官方社区「帮助与支持」） | 2026-07-06 | 「TRAE IDE 和 TRAE Work 是两款定位不同的产品。目前 TRAE IDE 确实已经支持了 Linux 系统（包括 .deb 格式的 Ubuntu 等）；但 **TRAE Work 的桌面版现阶段还只支持 macOS 和 Windows，Linux 版本目前还在规划中，暂时还没有具体的上线时间表**。作为临时方案，在 Ubuntu 系统上你可以先通过浏览器访问 **TRAE Work 网页版（work.trae.cn）**」 |
| 2 | 同上帖内另一条 | 2026-07-06 | 「暂时不会太快支持 linux 版本哈」 |
| 3 | `forum.trae.cn/t/topic/181037/5` | 2026-09-16 | 「目前 work 暂时没有 Linux 版本呢，**就是 bug** 哈」（指官网出现 Linux 下载） |
| 4 | `forum.trae.cn/t/topic/22350/2`（TRAE宝） | 2026-06-13 | 「目前 **TRAE Work（原 SOLO）暂不支持通过 Remote-SSH 连接远程服务器**进行开发哦，近期也没有相关的适配计划～ 如果你有强烈的远程开发需求，建议使用 **TRAE IDE**」 |
| 5 | `docs.trae.cn/work_what-is-trae-work` | - | TraeWork = AI 原生工作台，三种形态：**网页版 / 桌面版 / 移动端**；网页版「无需下载和安装，提供开箱即用的**云端环境**」 |

> 注意：另一条产品线 **TRAE IDE 有 Linux 版**（`.deb` / `.rpm`，x64 + arm64，内置 Remote-SSH）。
> 火山引擎文章 `developer.volcengine.com/articles/7622875986871779391`：「TRAE IDE Linux 版本正式上线」。
> 但**那是 TRAE IDE，不是 TraeWork**，两者定位不同。

## 远端实测现状（只读）

```
$ pgrep -af 'trae.*server'      → （空，当前没有 trae server 在跑）
$ ls -d /home/dgut/.trae*        → .trae  .trae-aicc  .trae-cn  .trae-cn-server  .trae-server
```

| 目录 | 体积 | 内容 |
|---|---|---|
| `~/.trae-server/` | **3.6 G** | **Trae**（国际版 IDE）Remote-SSH server，5 个版本（588 M–653 M） |
| `~/.trae-cn-server/` | **6.2 G** | **Trae CN** Remote-SSH server，7 个版本（8.8 M–1.1 G） |
| `~/.trae-cn/` | 81 M | Trae CN 客户端配置 |
| `~/.trae-aicc/` | 5.8 M | - |
| `~/.trae/` | 0 | 空 |

`product.json` 实测：`nameLong` = `"Trae"` / `"Trae CN"`，`quality` = `stable`。
**远端没有任何 TraeWork 痕迹** —— 上面这些都是 **Trae / Trae CN IDE 的远程 server**，
是以前用**本地 TRAE IDE 通过 Remote-SSH** 连这台机器时自动部署的。

## 与磁盘的关系（重要）

| 挂载点 | 设备 | 容量 | 已用 | 可用 |
|---|---|---|---|---|
| `/` | `/dev/mapper/ubuntu--vg-ubuntu--lv` (xfs) | 280 G | 97% | **11 G** |
| `/home/T7` | `/dev/vdb2` (fuseblk/NTFS) | 7.1 T | 84% | 1.2 T |

- `~`（`/home/dgut`）**在 `/` 上**，不在 `/home/T7`。
- 上面那 9.8 G 的 trae server 版本**全在只剩 11 G 的根分区上**，是 97% 占用的主要来源之一。
- `/home/T7` 实测**可写可执行**（建临时脚本 + `chmod +x` + 运行 → `EXEC_OK`），但它是**多用户共享的 NTFS 数据盘**
  （含 `$RECYCLE.BIN`、`System Volume Information`、`360用户文件`、`public`、`temp`），不适合当某个人的应用安装目录。
  `/home/T7/temp/` 里已经躺着一批别人/以前留下的 trae 产物（`trae`、`trae-remote`、`trae-agent-toolhost-1000`、`trae_sandbox_trace_*.jsonl`）。

## 可行方案

| 你想要的效果 | 可行的做法 | 远端要不要装东西 |
|---|---|---|
| 远端有一个 AI 工作台入口 | **远端已经装了 DSH**（DeepSeek Harness 0.2.0-rc.2，`127.0.0.1:3080`，SSH 隧道访问） | 否，已就绪 |
| 用 Trae 系工具在远端改代码 | 用**本地 TRAE IDE**（已装 3.5.62）或 **TraeCode CN**（已装 3.3.98）走 Remote-SSH 连 `dgut@172.31.68.251`；远端会复用/更新 `~/.trae-server` | 否，已有 server |
| 一定要 TraeWork | 只能用**本地 Windows**（已装 `TRAE Work` 0.1.41 / `TRAE Work CN` 0.1.45），或 **网页版 `work.trae.cn`**（云端环境） | 否 |
| 非要在远端有一个 Trae 图形客户端 | 只能装 **TRAE IDE Linux 版**（`.deb`，官方支持），但那不是 TraeWork；且会加重只剩 11 G 的根分区 | 是（约 600 M–1 G） |

## 未做（等你决定）

1. 未下载任何 TraeWork / Trae 安装包。
2. 未在远端安装任何东西。
3. 未删除 `~/.trae-server` / `~/.trae-cn-server` 的旧版本（12 个版本，约 9.8 G，当前无进程占用；
   清理可释放约 7–8 G，但删除类动作需你批准）。