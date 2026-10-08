#!/usr/bin/env bash
set -u
H="$HOME"
mkdir -p "$H/.config/systemd/user"
cp /tmp/dsh-web.service "$H/.config/systemd/user/dsh-web.service"
echo "== stop the manually started instance (frees 3080) =="
pkill -f "[b]in.js web" 2>/dev/null || true
for i in $(seq 1 30); do pgrep -f "[b]in.js web" >/dev/null 2>&1 || break; sleep 1; done
echo "== enable + start the user unit =="
systemctl --user daemon-reload
systemctl --user enable --now dsh-web
sleep 10
echo "== unit state =="
echo "active=$(systemctl --user is-active dsh-web) enabled=$(systemctl --user is-enabled dsh-web)"
systemctl --user status dsh-web --no-pager 2>&1 | head -10
echo "== process cwd =="; PID=$(systemctl --user show dsh-web -p MainPID --value); echo "main_pid=$PID cwd=$(readlink /proc/$PID/cwd 2>/dev/null)"
echo "== listeners =="; ss -ltn | grep 3080
echo "== token =="
journalctl --user -u dsh-web -n 40 --no-pager 2>/dev/null | grep -o "http://127.0.0.1:3080/?token=[A-Za-z0-9_-]*" | tail -1