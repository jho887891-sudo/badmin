#!/usr/bin/env bash
H="$HOME"
echo "== 1) dsh web process =="
pgrep -af "[b]in.js web" || echo "NOT RUNNING"
PID=$(pgrep -f "[b]in.js web" | head -1)
if [ -n "${PID:-}" ]; then echo "  pid=$PID  cwd=$(readlink /proc/$PID/cwd)  uptime=$(ps -o etime= -p $PID | tr -d " ")"; fi
echo "== 2) loopback listener =="; ss -ltn | grep ":3080" || echo "  no 3080 listener"
echo "== 3) tailscale serve =="; tailscale serve status 2>&1 | head -6
echo "== 4) http on loopback =="
TOKEN=$(grep -o "token=[A-Za-z0-9_-]*" "$H/.dsh-web/dsh-web.log" | tail -1 | cut -d= -f2)
echo "  token=${TOKEN:-<none>}"
curl -sS -o /dev/null -w "  loopback_no_token=%{http_code}\n" --max-time 12 http://127.0.0.1:3080/
CJ=$(mktemp); curl -sS -c "$CJ" -b "$CJ" -L -o /tmp/h.html -w "  loopback_with_token=%{http_code} bytes=%{size_download}\n" --max-time 20 "http://127.0.0.1:3080/?token=$TOKEN"
grep -o -E "<title>[^<]*</title>|__DSH_BOOT__" /tmp/h.html | head -2
echo "== 5) log tail =="; tail -3 "$H/.dsh-web/dsh-web.log"