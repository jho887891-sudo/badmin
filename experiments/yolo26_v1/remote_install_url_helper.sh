#!/usr/bin/env bash
set -u
H="$HOME"
cp /tmp/remote_dsh_url.sh "$H/.dsh-web/url.sh"; chmod +x "$H/.dsh-web/url.sh"
echo "== helper =="; "$H/.dsh-web/url.sh"
echo "== unit summary =="
echo "active=$(systemctl --user is-active dsh-web) enabled=$(systemctl --user is-enabled dsh-web) linger=$(loginctl show-user dgut -p Linger --value)"
systemctl --user show dsh-web -p MainPID -p ExecMainStartTimestamp -p Restart -p WorkingDirectory --no-pager
echo "== LAN exposure check =="; ss -ltn | grep "172.31.68.251:3080" && echo "LAN EXPOSED" || echo "LAN not exposed (good)"