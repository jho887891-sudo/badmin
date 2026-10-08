#!/usr/bin/env bash
echo "== vscode-server 是否在跑（tailnet 连接） =="
ps -eo pid,etime,cmd | grep "[v]scode-server" | head -4
echo "== 候选工作区 =="
for d in /home/T7/ojh/badmin_project /home/T7/ojh/robot_sim /home/T7/ojh /home/dgut; do
  if [ -d "$d" ]; then printf "  %-32s files=%s  size=%s\n" "$d" "$(find "$d" -type f 2>/dev/null | wc -l)" "$(du -sh "$d" 2>/dev/null | cut -f1)"; fi
done
echo "== badmin_project 顶层（我们同步的项目记忆） =="
ls /home/T7/ojh/badmin_project | head -12
echo "== vscode-server 日志里有没有报错（最近 5 行） =="
ls -t ~/.vscode-server/data/logs/ 2>/dev/null | head -2
LOGDIR=$(ls -dt ~/.vscode-server/data/logs/*/ 2>/dev/null | head -1); echo "logdir=$LOGDIR"
[ -n "$LOGDIR" ] && (find "$LOGDIR" -name "*.log" | head -3 | while read f; do echo "-- $f"; tail -n 3 "$f"; done)