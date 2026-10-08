#!/usr/bin/env bash
set -u
echo "== configure tailscale serve (tailnet-only HTTP proxy -> 127.0.0.1:3080) =="
tailscale serve --bg --http=3080 http://127.0.0.1:3080
echo "== serve status =="
tailscale serve status
echo "== listeners on 3080 =="
ss -ltn | grep ":3080" || echo "none"
echo "== LAN address exposed? (must be NO) =="
ss -ltn | grep "172.31.68.251:3080" && echo "LAN EXPOSED (bad)" || echo "LAN not exposed (good)"
echo "== fetch current dsh token =="
grep -o "token=[A-Za-z0-9_-]*" "$HOME/.dsh-web/dsh-web.log" | tail -1