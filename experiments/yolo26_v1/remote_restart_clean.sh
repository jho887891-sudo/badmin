#!/usr/bin/env bash
set -u
H="$HOME"
WS="/home/T7/ojh/badmin_project"
LOG="$H/.dsh-web/dsh-web.log"
echo "== current state =="
pgrep -af "[b]in.js web" || echo "no process"
ss -ltn | grep ":3080" || echo "port 3080 not listening"
echo "== ensure stopped =="
pkill -f "[b]in.js web" 2>/dev/null || true
for i in $(seq 1 30); do pgrep -f "[b]in.js web" >/dev/null 2>&1 || break; sleep 1; done
pgrep -af "[b]in.js web" || echo "clean"
echo "== start in workspace =="
cd "$WS" || exit 1
mkdir -p "$H/.dsh-web"
: > "$LOG"
setsid nohup "$H/.local/bin/dsh" web --no-open --host 127.0.0.1 --port 3080 > "$LOG" 2>&1 < /dev/null &
echo $! > "$H/.dsh-web/dsh-web.pid"
for i in $(seq 1 60); do ss -ltn | grep -q ":3080 " && break; sleep 1; done
echo "== state after start =="
pgrep -af "[b]in.js web" | head -2
ss -ltn | grep ":3080" || echo "STILL NOT LISTENING"
PID=$(pgrep -f "[b]in.js web" | head -1); echo "cwd=$(readlink /proc/$PID/cwd)"
TOKEN=$(grep -o "token=[A-Za-z0-9_-]*" "$LOG" | tail -1 | cut -d= -f2)
echo "TOKEN=$TOKEN"
echo "LOG=$(tail -3 "$LOG")"
CJ=$(mktemp)
curl -sS -o /dev/null -w "no_token=%{http_code}\n" --max-time 15 http://127.0.0.1:3080/ 2>&1
curl -sS -c "$CJ" -b "$CJ" -L -o /tmp/h.html -w "with_token=%{http_code} bytes=%{size_download}\n" --max-time 20 "http://127.0.0.1:3080/?token=$TOKEN" 2>&1
grep -o -E "<title>[^<]*</title>|__DSH_BOOT__" /tmp/h.html 2>/dev/null | head -2