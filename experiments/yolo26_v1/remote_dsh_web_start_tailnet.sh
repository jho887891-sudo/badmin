#!/usr/bin/env bash
# Start the remote DeepSeek Harness Web UI on LOOPBACK (DSH only accepts 127.0.0.1 or 0.0.0.0).
# Tailnet exposure is done by `tailscale serve` (see remote_serve_dsh.sh), which proxies tailnet -> 127.0.0.1.
set -u
H="$HOME"
WS="${DSH_WORKSPACE:-/home/T7/ojh/badmin_project}"
HOST="${DSH_HOST:-127.0.0.1}"
PORT="${DSH_PORT:-3080}"
LOG="$H/.dsh-web/dsh-web.log"
PIDF="$H/.dsh-web/dsh-web.pid"
mkdir -p "$H/.dsh-web"
if pgrep -f "[b]in.js web" >/dev/null 2>&1; then
  echo "already running:"; pgrep -af "[b]in.js web"
else
  cd "$WS" || exit 1
  setsid nohup "$H/.local/bin/dsh" web --no-open --host "$HOST" --port "$PORT" \
    --trusted-host "100.88.178.19:$PORT" \
    --trusted-host "jxxy.taildd42cc.ts.net:$PORT" > "$LOG" 2>&1 < /dev/null &
  echo $! > "$PIDF"
  sleep 10
fi
echo "workspace: $WS"; echo "bind: $HOST:$PORT"
echo "URL: $(grep -o "http://[^ ]*token=[A-Za-z0-9_-]*" "$LOG" | tail -1)"
ss -ltn | grep ":$PORT" || echo "NOT LISTENING"