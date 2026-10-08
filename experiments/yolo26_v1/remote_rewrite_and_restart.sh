#!/usr/bin/env bash
set -u
H="$HOME"
TS=$(date +%Y%m%d-%H%M%S)
echo "== backup sessions -> ~/.dsh/sessions-orig-$TS =="
cp -a "$H/.dsh/sessions" "$H/.dsh/sessions-orig-$TS"
echo "backup files: $(find "$H/.dsh/sessions-orig-$TS" -type f | wc -l)"
du -sh "$H/.dsh/sessions-orig-$TS"
echo "== stop server =="
pkill -f "[b]in.js web" 2>/dev/null || true
for i in $(seq 1 30); do pgrep -f "[b]in.js web" >/dev/null 2>&1 || break; sleep 1; done
echo "== rewrite paths =="
cd "$H/.local/lib/node_modules/@deepseek-ai/dsh" && "$H/.nvm/versions/node/v22.23.3/bin/node" /tmp/rewrite_paths.mjs 2>&1 | tail -22
echo "== verify headers again =="
"$H/.nvm/versions/node/v22.23.3/bin/node" /tmp/read_header.mjs 2>&1 | head -8
echo "== disk =="; df -h / | tail -1
echo "== restart server =="
cd /home/T7/ojh/badmin_project || exit 1
setsid nohup "$H/.local/bin/dsh" web --no-open --host 127.0.0.1 --port 3080 > "$H/.dsh-web/dsh-web.log" 2>&1 < /dev/null &
for i in $(seq 1 60); do ss -ltn | grep -q ":3080 " && break; sleep 1; done
TOKEN=$(grep -o "token=[A-Za-z0-9_-]*" "$H/.dsh-web/dsh-web.log" | tail -1 | cut -d= -f2)
echo "TOKEN=$TOKEN"
CJ=$(mktemp)
curl -sS -c "$CJ" -b "$CJ" -L -o /tmp/h.html -w "http=%{http_code} bytes=%{size_download}\n" --max-time 30 "http://127.0.0.1:3080/?token=$TOKEN"
echo "== session counts =="
for d in "$H"/.dsh/sessions/*/; do echo "  $(basename "$d") -> $(ls "$d" | wc -l) sessions"; done