#!/usr/bin/env bash
H="$HOME"
TOKEN=$(grep -o "token=[A-Za-z0-9_-]*" "$H/.dsh-web/dsh-web.log" | head -1 | cut -d= -f2)
CJ=$(mktemp)
echo "== remote: follow redirect with cookie jar =="
curl -sS -c "$CJ" -b "$CJ" -L -o /tmp/dsh_home.html -w "final_status=%{http_code} bytes=%{size_download} url=%{url_effective}\n" --max-time 25 "http://127.0.0.1:3080/?token=$TOKEN"
echo "== markers =="; grep -o -E "<title>[^<]*</title>|__DSH_BOOT__" /tmp/dsh_home.html | head -3
echo "== head =="; head -c 260 /tmp/dsh_home.html; echo
echo "== /api with cookie =="; curl -sS -b "$CJ" -o /dev/null -w "%{http_code}\n" --max-time 15 "http://127.0.0.1:3080/api"
echo "TOKEN_FOR_LOCAL=$TOKEN"