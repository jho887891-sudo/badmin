#!/usr/bin/env bash
# Print the current remote Harness URLs (token is regenerated on every start/restart).
J=$(journalctl --user -u dsh-web --no-pager -n 300 2>/dev/null | grep -o "http://127.0.0.1:3080/?token=[A-Za-z0-9_-]*" | tail -1)
TOKEN=${J##*token=}
if [ -z "$TOKEN" ]; then echo "no token found yet; try: systemctl --user restart dsh-web"; exit 1; fi
echo "local  : http://127.0.0.1:3080/?token=$TOKEN"
echo "tailnet: http://jxxy.taildd42cc.ts.net:3080/?token=$TOKEN"