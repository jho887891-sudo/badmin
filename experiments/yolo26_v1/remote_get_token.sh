#!/usr/bin/env bash
H="$HOME"
sleep 5
echo "== log =="; cat "$H/.dsh-web/dsh-web.log"
TOKEN=$(grep -o "token=[A-Za-z0-9_-]*" "$H/.dsh-web/dsh-web.log" | tail -1 | cut -d= -f2)
echo "TOKEN=$TOKEN"
CJ=$(mktemp)
curl -sS -c "$CJ" -b "$CJ" -L -o /tmp/h.html -w "with_token=%{http_code} bytes=%{size_download}\n" --max-time 25 "http://127.0.0.1:3080/?token=$TOKEN"
grep -o -E "<title>[^<]*</title>|__DSH_BOOT__" /tmp/h.html | head -2
echo "== pid/port =="; pgrep -f "[b]in.js web" | head -1; ss -ltn | grep :3080