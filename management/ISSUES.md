# 问题与解决记录（ISSUES）

> 规则：单次排错超过 15 分钟必须记录；出现类似错误**先搜索本文件**，不要重复踩坑。

## ISSUE 模板
```
## ISSUE-XXX
- **日期：**
- **模块：**
- **现象：**
- **错误日志：**
- **原因：**
- **解决方案：**
- **修改文件：**
- **验证方法：**
- **是否彻底解决：** 是 / 否（原因）
- **相关 commit：**
```

---

当前记录：无


## ISSUE-001 远端 git 全局代理失效（127.0.0.1:10080 拒连）
- **日期：** 2026-09-08
- **模块：** 环境/网络
- **现象：** git clone/ls-remote 报 "Failed to connect to 127.0.0.1 port 10080: Connection refused"
- **错误日志：** fatal: unable to access ... Failed to connect to 127.0.0.1 port 10080 after 0 ms
- **原因：** ~/.gitconfig 配了 http/https.proxy=http://127.0.0.1:10080，但该代理当前未运行（jxxy 上无监听）
- **解决方案：** git 命令加 `-c http.proxy= -c https.proxy=` 绕过（curl/github 直连可达）；勿改动用户全局 git 配置
- **修改文件：** 无（命令行级绕过）
- **验证方法：** git -c http.proxy= ls-remote 成功；IsaacLab clone 成功
- **是否彻底解决：** 部分 —— 代理未恢复期间每次 git 命令需带 -c 参数
- **相关 commit：** 无

## ISSUE-002 NVIDIA pip 源 curl 探测误导（301→.cn 404，但 uv 实际可用）
- **日期：** 2026-09-08
- **模块：** 环境/网络
- **现象：** curl https://pypi.nvidia.com/simple/isaacsim/ 返回 301 到 pypi.nvidia.cn，.cn 上所有包 404，一度误判 NVIDIA 源不可用
- **错误日志：** pypi.nvidia.cn/simple/isaacsim/ -> HTTP 404
- **原因：** 301 为地域分流；curl 未跟随的路径与 uv 客户端行为不一致；uv（含 --extra-index-url https://pypi.nvidia.com --index-strategy unsafe-best-match）实测可正常解析 isaacsim==6.0.1.0（166 包 dry-run 通过）
- **解决方案：** 用 uv 直接安装即可；探测通道请以 uv/pip 实测为准，勿仅凭 curl 结论
- **修改文件：** 无
- **验证方法：** uv pip install --dry-run "isaacsim[all,extscache]==6.0.1.0" 成功
- **是否彻底解决：** 是（通道确认）
- **相关 commit：** 无
