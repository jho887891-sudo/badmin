#!/usr/bin/env bash
echo "== distinct token lines in the journal (last 200) =="
journalctl --user -u dsh-web --no-pager -n 200 2>/dev/null | grep -o "token=[A-Za-z0-9_-]*" | sort | uniq -c
echo "== unit =="; systemctl --user is-active dsh-web; systemctl --user is-enabled dsh-web; systemctl --user show dsh-web -p MainPID -p ExecMainStartTimestamp --value | head -2
echo "== linger =="; loginctl show-user dgut -p Linger
echo "== LAN not bound? =="; ss -ltn | grep "172.31.68.251:3080" && echo "LAN EXPOSED" || echo "LAN not exposed (good)"