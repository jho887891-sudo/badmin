#!/usr/bin/env bash
export NVM_DIR="$HOME/.nvm"; . "$NVM_DIR/nvm.sh" >/dev/null 2>&1
H="$HOME"
TOKEN=$(grep -o "token=[A-Za-z0-9_-]*" "$H/.dsh-web/dsh-web.log" | head -1 | cut -d= -f2)
echo "TOKEN=$TOKEN"
echo "== no token =="; curl -sS -o /dev/null -w "%{http_code}\n" --max-time 15 "http://127.0.0.1:3080/"
echo "== with token =="; curl -sS -o /dev/null -w "%{http_code}\n" --max-time 15 "http://127.0.0.1:3080/?token=$TOKEN"
echo "== title / boot marker =="; curl -sS --max-time 15 "http://127.0.0.1:3080/?token=$TOKEN" | grep -o -E "<title>[^<]*</title>|__DSH_BOOT__" | head -3
echo "== html head =="; curl -sS --max-time 15 "http://127.0.0.1:3080/?token=$TOKEN" | head -c 300; echo
echo "== bind check (must be 127.0.0.1 only) =="; ss -ltn | grep ":3080"
echo "== process =="; ps -o pid,etime,rss,cmd -p "$(cat "$H/.dsh-web/dsh-web.pid")" 2>&1 | tail -2
echo "== log =="; cat "$H/.dsh-web/dsh-web.log"