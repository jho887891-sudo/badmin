#!/usr/bin/env bash
set -u
H="$HOME"
echo "== restart =="
pkill -f "[b]in.js web" 2>/dev/null || true
for i in $(seq 1 30); do pgrep -f "[b]in.js web" >/dev/null 2>&1 || break; sleep 1; done
cd /home/T7/ojh/badmin_project || exit 1
: > "$H/.dsh-web/dsh-web.log"
setsid nohup "$H/.local/bin/dsh" web --no-open --host 127.0.0.1 --port 3080 > "$H/.dsh-web/dsh-web.log" 2>&1 < /dev/null &
echo $! > "$H/.dsh-web/dsh-web.pid"
for i in $(seq 1 60); do ss -ltn | grep -q ":3080 " && break; sleep 1; done
TOKEN=$(grep -o "token=[A-Za-z0-9_-]*" "$H/.dsh-web/dsh-web.log" | tail -1 | cut -d= -f2)
echo "TOKEN=$TOKEN"
echo "log: $(tail -2 "$H/.dsh-web/dsh-web.log")"
CJ=$(mktemp)
curl -sS -c "$CJ" -b "$CJ" -L -o /tmp/h.html -w "no_token=%{http_code}\n" --max-time 20 http://127.0.0.1:3080/
curl -sS -c "$CJ" -b "$CJ" -L -o /tmp/h.html -w "with_token=%{http_code} bytes=%{size_download}\n" --max-time 25 "http://127.0.0.1:3080/?token=$TOKEN"
grep -o -E "<title>[^<]*</title>|__DSH_BOOT__" /tmp/h.html | head -2
echo "== process/cwd =="; PID=$(pgrep -f "[b]in.js web" | head -1); echo "pid=$PID cwd=$(readlink /proc/$PID/cwd)"
echo "== credential file still 600 + ref present =="
ls -l "$H/.dsh/.credentials.yaml"; python3 -c "import os,re;t=open(os.path.expanduser('~/.dsh/.credentials.yaml')).read();print('has_ref=', bool(re.search(r'DEEPSEEK_API_KEY', t)))"
echo "== backups present =="; ls -1 "$H/.dsh/" | grep -E "bak|orig" || true