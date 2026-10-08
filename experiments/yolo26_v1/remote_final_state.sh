#!/usr/bin/env bash
H="$HOME"
echo "== process =="; pgrep -af "[b]in.js web" | head -3
echo "== port =="; ss -ltn | grep ":3080" || echo "not listening"
echo "== log =="; cat "$H/.dsh-web/dsh-web.log"
TOKEN=$(grep -o "token=[A-Za-z0-9_-]*" "$H/.dsh-web/dsh-web.log" | tail -1 | cut -d= -f2)
echo "TOKEN=$TOKEN"
CJ=$(mktemp)
curl -sS -c "$CJ" -b "$CJ" -L -o /tmp/h.html -w "with_token=%{http_code} bytes=%{size_download}\n" --max-time 25 "http://127.0.0.1:3080/?token=$TOKEN"
grep -o -E "<title>[^<]*</title>|__DSH_BOOT__" /tmp/h.html | head -2
echo "== session dirs (real counts) =="
for d in "$H"/.dsh/sessions/*/; do echo "  $(basename "$d") -> $(find "$d" -maxdepth 1 -mindepth 1 -type d | wc -l) sessions, $(find "$d" -type f | wc -l) files"; done
echo "== backups & temp =="
ls -d "$H"/.dsh/sessions-orig-* 2>/dev/null; du -sh "$H"/.dsh/sessions-orig-* 2>/dev/null
ls -d /tmp/dshmem-* /tmp/dsh_sessions.tgz 2>/dev/null
du -sh /tmp/dshmem-* /tmp/dsh_sessions.tgz 2>/dev/null
df -h / | tail -1