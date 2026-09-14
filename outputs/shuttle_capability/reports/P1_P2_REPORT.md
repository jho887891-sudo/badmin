# P1 + P2 REPORT — 真实视频获取与逐帧 GT（阶段结论）

日期：2026-09-13　执行：Codex 接管推进

## P1 — 视频获取（含重试与换源）

**重试记录**：Wikimedia 视频下载先后遭遇 **HTTP 429（限流）** 3 次（含 25–180 s 退避），
第 4 次（等待约 5 分钟后）**成功**。未绕过任何访问限制，未无限重试。

已下载 4 段自由许可视频（`outputs/shuttle_capability/real_video/clips/`，元数据含 URL/许可/日期/SHA256）：

| 文件 | 许可 | 大小 | 分辨率 | SHA256(前 16) |
|---|---|---|---|---|
| bim_cc0_1440.webm | CC0 | 1.14 MB | 1440x1440 | —（见 csv） |
| ball_badminton_ccby3_1280.webm | CC BY 3.0 | 8.23 MB | 1280x720 | 646cc6f53d121db1 |
| eurogames_ccbysa4_619.webm | CC BY-SA 4.0 | 26.87 MB | 848x464 | 4e5dab1eb80d83cd |
| school_cc0_320.webm | CC0 | 3.83 MB | 320x240 | d53340dcd3cac327 |

（Slovenian point 视频按候选 URL 返回 404，未取得；改为记录失败，不再猜 URL。）

备选来源已探明可用（未使用）：Openverse API（240 张 shuttlecock 图，含 **NC** 需规避）、
Archive.org（417 条 badminton 影片，多为社媒转载、**许可不明**，应过滤 licenseurl）、
HuggingFace（BadmintonPoses / samhitmantrala/badminton / JIAWEIII/badminton 等数据集，需逐库确认许可）。

## P2 — 逐帧 GT：**部分完成，遇到真实阻塞**

### 已完成
- 固定频率抽帧（2 fps）：**150 帧**，4 段视频，manifest 见 `real_video/frames_manifest.csv`
- 逐段接触表：`visualizations/video_sheets/*.png`（我已逐张查看）
- 运动差分候选：`real_video/candidates/eurogames_ccbysa4_619_candidates.csv`
  + 放大裁剪复核图 `visualizations/shuttle_candidates/eurogames_ccbysa4_619_cand00.png`

### 内容核验结果（重要：**文件名不等于内容**）

| 片段 | 实际内容 | 是否含羽毛球 |
|---|---|---|
| bim_cc0_1440.webm | 室内人物对镜头讲话（访谈形式） | **否** |
| ball_badminton_ccby3_1280.webm | **Ball Badminton**（另一项运动，用球不用羽毛球） | **否** |
| school_cc0_320.webm | 教室内讲羽毛球拍 | **否**（抽样帧内未见球） |
| eurogames_ccbysa4_619.webm | **真实羽毛球比赛**（瑞士 badminton 场地） | **是（比赛真实存在）** |

### 阻塞：真实比赛中**无法可靠定位羽毛球**
- 在 848x464 广角画面中，羽毛球约为数像素量级；接触表尺度下无法辨认。
- 运动差分（69739 个候选）排序后放大复核：**前 18 个候选全部是球员脚/腿/球拍运动**（21–29 px），
  **无一是羽毛球**。见 `shuttle_candidates/eurogames_ccbysa4_619_cand00.png`。
- 结论：**免费公开的广角比赛视频不足以产生可信的逐帧小目标 GT**（缺 GT 时协议只允许诊断性连续性测试）。

因此按协议「遇到需要用户提供真实相机私有数据时才暂停询问」——**这里正是该情况**。

### 冻结状态
Capability Test Set 目前包含：
- **SYNTHETIC_3D**：84 正 + 40 负（已冻结，`metrics/synthetic_manifest.csv`）
- **REAL_IMAGE**：10 正 + 6 负（**已核验**，P0）
- **REAL_VIDEO**：帧已抽取但**尚无 GT** → 本轮**不进入**任何正式指标（不编造）

冻结规则不变：以上样本**不得进入训练集**。

## 下一步（分两条）

**A（需你提供，唯一阻塞）**：真实相机数据，任一即可——
1. 机载双目视频（机器人视角、球在视场内、含近/中/远距离与高速/遮挡片段）；
2. 手机拍摄的羽毛球视频（把球保持在画面中、距离 2–6 m、含高吊与平抽）；
3. 或逐帧已标注的羽毛球视频（哪怕只有 100–300 帧也有决定性价值）。
交付形式：原始文件（mp4/mov）+ 拍摄参数（分辨率/帧率/大致距离）即可，我来抽帧并做逐帧 GT 标注（放大裁剪逐帧确认）。

**B（无需你提供，我可立即继续）**：P3 — 以**真实背景 + 受控 3D 小球**生成 `SYNTHETIC_ON_REAL_BG` 补充集
（单独统计、绝不冒充真实数据，也不用于替代真实视频验收）；它能在真实域纹理/光照下测出 4–16 px 的检测边界。
