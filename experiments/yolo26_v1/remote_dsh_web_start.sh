#!/usr/bin/env bash
# Start the remote DeepSeek Harness Web UI. Loopback only; prints the URL incl. the per-run token.
# Workspace (DSH default filesystem location) = the project memory tree, override with DSH_WORKSPACE.
set -u
H="$HOME"
WS="${DSH_WORKSPACE:-/home/T7/ojh/badmin_project}"
LOG="$H/.dsh-web/dsh-web.log"
PIDF="$H/.dsh-web/dsh-web.pid"
mkdir -p "$H/.dsh-web"
if pgrep -f "[b]in.js web" >/dev/null 2>&1; then
  echo "already running:"
  pgrep -af "[b]in.js web"
else
  cd "$WS" || exit 1
  setsid nohup "$H/.local/bin/dsh" web --no-open --host 127.0.0.1 --port 3080 > "$LOG" 2>&1 < /dev/null &
  echo $! > "$PIDF"
  sleep 10
fi
echo "workspace: $WS"
echo "URL: $(grep -o "http://127.0.0.1:3080/?token=[A-Za-z0-9_-]*" "$LOG" | tail -1)"
ss -ltn | grep ":3080" || echo "NOT LISTENING on 3080"