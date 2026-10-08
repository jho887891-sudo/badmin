#!/usr/bin/env bash
# Stop the remote DeepSeek Harness Web UI started by start.sh
if pgrep -f "[b]in.js web" >/dev/null 2>&1; then
  pkill -f "[b]in.js web" && echo "stopped"
  sleep 2
  pgrep -af "[b]in.js web" || echo "no dsh web process left"
else
  echo "not running"
fi