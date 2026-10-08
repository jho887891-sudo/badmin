#!/usr/bin/env bash
set -u
H="$HOME"
STAGE="/tmp/dshmem-$(date +%s)"
echo "== stop server =="
pkill -f "[b]in.js web" 2>/dev/null || true
for i in $(seq 1 30); do pgrep -f "[b]in.js web" >/dev/null 2>&1 || break; sleep 1; done
pgrep -af "[b]in.js web" || echo "server stopped"
echo "== extract archive =="
mkdir -p "$STAGE"
time tar -xzf /tmp/dsh_sessions.tgz -C "$STAGE"
echo "staged files: $(find "$STAGE" -type f | wc -l)"
echo "== place sessions under remote projectKey names =="
mkdir -p "$H/.dsh/sessions/--home-T7-ojh-badmin_project--" "$H/.dsh/sessions/--home-T7-ojh--"
cp -a "$STAGE/sessions/--E-~5177~8EAB~667A~80FD-badmin_project--/." "$H/.dsh/sessions/--home-T7-ojh-badmin_project--/"
cp -a "$STAGE/sessions/--E-~5177~8EAB~667A~80FD--/." "$H/.dsh/sessions/--home-T7-ojh--/"
mkdir -p "$H/.dsh/attachments"
cp -a "$STAGE/attachments/." "$H/.dsh/attachments/"
echo "== merge workspace registry (with backup) =="
python3 /tmp/merge_workspace.py /tmp/local_workspace.json
echo "== start server in the project workspace =="
cd /home/T7/ojh/badmin_project || exit 1
setsid nohup "$H/.local/bin/dsh" web --no-open --host 127.0.0.1 --port 3080 > "$H/.dsh-web/dsh-web.log" 2>&1 < /dev/null &
echo $! > "$H/.dsh-web/dsh-web.pid"
for i in $(seq 1 60); do ss -ltn | grep -q ":3080 " && break; sleep 1; done
TOKEN=$(grep -o "token=[A-Za-z0-9_-]*" "$H/.dsh-web/dsh-web.log" | tail -1 | cut -d= -f2)
echo "TOKEN=$TOKEN"
echo "LOG=$(tail -2 "$H/.dsh-web/dsh-web.log")"
CJ=$(mktemp)
curl -sS -c "$CJ" -b "$CJ" -L -o /tmp/h.html -w "http=%{http_code} bytes=%{size_download}\n" --max-time 30 "http://127.0.0.1:3080/?token=$TOKEN"
echo "== do our session ids appear in the boot payload? =="
for sid in 339e7247 f5854e91 b485ce60 7ed6f8ef; do printf "  %s -> " "$sid"; grep -c "$sid" /tmp/h.html || true; done
echo "== candidate API endpoints =="
for ep in /api /api/ /api/sessions /api/workspaces /api/state; do printf "  %-18s " "$ep"; curl -sS -b "$CJ" -o /tmp/api.out -w "%{http_code}" --max-time 12 "http://127.0.0.1:3080$ep"; echo " bytes=$(stat -c%s /tmp/api.out)"; done
echo "== local state =="
ls "$H/.dsh/sessions"
echo "session files: $(find "$H/.dsh/sessions" -type f | wc -l)   attachment files: $(find "$H/.dsh/attachments" -type f | wc -l)"
du -sh "$H/.dsh/sessions" "$H/.dsh/attachments" "$H/.dsh/storages"
ls -la "$H/.dsh/storages"
echo "STAGE=$STAGE"