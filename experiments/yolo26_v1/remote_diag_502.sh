#!/usr/bin/env bash
echo "== unit state =="
systemctl --user is-active dsh-web; systemctl --user is-enabled dsh-web
systemctl --user show dsh-web -p MainPID -p Result -p ExecMainStatus -p NRestarts -p ActiveEnterTimestamp -p InactiveEnterTimestamp --no-pager
echo "== listener 3080 =="; ss -ltn | grep 3080 || echo "NOTHING LISTENING on 3080"
echo "== dsh process =="; pgrep -af "[b]in.js web" || echo "no dsh web process"
echo "== serve status =="; tailscale serve status 2>&1 | head -5
echo "== journal (last 40) =="; journalctl --user -u dsh-web --no-pager -n 40 2>&1 | tail -40
echo "== user manager alive? =="; systemctl --user is-system-running 2>&1 | head -1; loginctl show-user dgut -p Linger --value