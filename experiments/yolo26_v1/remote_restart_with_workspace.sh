#!/usr/bin/env bash
set -u
H="$HOME"
cp /tmp/remote_dsh_web_start.sh "$H/.dsh-web/start.sh"
chmod +x "$H/.dsh-web/start.sh"
echo "== restarting =="
bash "$H/.dsh-web/stop.sh"
sleep 2
bash "$H/.dsh-web/start.sh"
echo "== process cwd (should be the project tree) =="
PID=$(pgrep -f "[b]in.js web" | head -1)
echo "pid=$PID"
ls -l "/proc/$PID/cwd"
echo "== http checks =="
TOKEN=$(grep -o "token=[A-Za-z0-9_-]*" "$H/.dsh-web/dsh-web.log" | tail -1 | cut -d= -f2)
echo "token=$TOKEN"
curl -sS -o /dev/null -w "no_token=%{http_code}\n" --max-time 15 http://127.0.0.1:3080/
CJ=$(mktemp)
curl -sS -c "$CJ" -b "$CJ" -L -o /tmp/h.html -w "with_token=%{http_code} bytes=%{size_download}\n" --max-time 20 "http://127.0.0.1:3080/?token=$TOKEN"
grep -o -E "<title>[^<]*</title>|__DSH_BOOT__" /tmp/h.html | head -2
echo "== dsh sees the workspace files? =="
ls /home/T7/ojh/badmin_project/AGENTS.md /home/T7/ojh/badmin_project/management/DAILY_LOG.md
echo "== settings/preset present =="
ls -l "$H/.dsh/settings.yaml" "$H/.dsh/.agent-presets/badmin-ptc/"