#!/usr/bin/env bash
set -u
echo "== linger state =="; loginctl show-user dgut -p Linger
echo "== token (before restart) =="
journalctl --user -u dsh-web --no-pager -n 80 2>/dev/null | grep -o "token=[A-Za-z0-9_-]*" | tail -2
echo "== restart via systemd (proves the unit manages the process) =="
systemctl --user restart dsh-web
for i in $(seq 1 40); do ss -ltn | grep -q "127.0.0.1:3080" && break; sleep 1; done
sleep 4
echo "active=$(systemctl --user is-active dsh-web) enabled=$(systemctl --user is-enabled dsh-web) mainpid=$(systemctl --user show dsh-web -p MainPID --value)"
echo "== token (after restart) =="
journalctl --user -u dsh-web --no-pager -n 80 2>/dev/null | grep -o "http://127.0.0.1:3080/?token=[A-Za-z0-9_-]*" | tail -1
echo "== listeners =="; ss -ltn | grep 3080