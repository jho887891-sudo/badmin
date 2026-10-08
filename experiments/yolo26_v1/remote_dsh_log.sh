#!/usr/bin/env bash
H="$HOME"
echo "== process =="; pgrep -af "[b]in.js web" || echo "NO PROCESS"
echo "== log =="; cat "$H/.dsh-web/dsh-web.log" 2>/dev/null | tail -30
echo "== listeners (all) =="; ss -ltn | head -12
echo "== pidfile =="; cat "$H/.dsh-web/dsh-web.pid" 2>/dev/null
echo "== dsh web help (host option text) =="; "$H/.local/bin/dsh" web --help 2>&1 | head -20