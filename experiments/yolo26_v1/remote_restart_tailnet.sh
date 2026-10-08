#!/usr/bin/env bash
set -u
H="$HOME"
cp /tmp/remote_dsh_web_start_tailnet.sh "$H/.dsh-web/start.sh"
chmod +x "$H/.dsh-web/start.sh"
echo "== restart with tailnet bind =="
pkill -f "[b]in.js web" 2>/dev/null || true
for i in $(seq 1 30); do pgrep -f "[b]in.js web" >/dev/null 2>&1 || break; sleep 1; done
bash "$H/.dsh-web/start.sh"