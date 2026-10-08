#!/usr/bin/env bash
# light check only: never walk /home/T7 (NTFS/fuseblk, huge tree)
pkill -f "[r]emote_vscode_state.sh" 2>/dev/null || true
pkill -f "[f]ind /home/T7/ojh/robot_sim" 2>/dev/null || true
echo "== vscode-server 进程 =="
ps -eo pid,etime,cmd | grep "[v]scode-server" | head -4
echo "== 候选目录（只看存在性 + 顶层条目数，不递归） =="
for d in /home/T7/ojh/badmin_project /home/T7/ojh/robot_sim /home/T7/ojh /home/dgut; do
  if [ -d "$d" ]; then printf "  %-30s top_entries=%s\n" "$d" "$(ls -A "$d" 2>/dev/null | wc -l)"; else echo "  $d 不存在"; fi
done
echo "== badmin_project 顶层 =="
ls /home/T7/ojh/badmin_project 2>/dev/null | head -14
echo "== 磁盘 =="; df -h /home/T7 / | tail -2