#!/usr/bin/env bash
set -u
H="$HOME"
NODE22="$H/.nvm/versions/node/v22.23.3/bin/node"
CLI="$H/.local/lib/node_modules/@deepseek-ai/dsh/lib/bin.js"
echo "== npm-created entry (before) =="; ls -l "$H/.local/bin/dsh"
echo "== does it work in a plain login shell? =="; bash -lc "command -v dsh; dsh --version" 2>&1 | head -6
cat > "$H/.local/bin/dsh" <<EOF
#!/usr/bin/env bash
# Pinned to Node 22: this host nvm default is v18.17.0 but @deepseek-ai/dsh needs >=22.19.0.
exec "$NODE22" "$CLI" "\$@"
EOF
chmod +x "$H/.local/bin/dsh"
echo "== wrapper (after) =="; cat "$H/.local/bin/dsh"
echo "== dsh via login shell =="; bash -lc "command -v dsh; dsh --version" 2>&1 | head -6
echo "== start dsh web (loopback only, detached) =="
mkdir -p "$H/.dsh-web"
cd "$H"
setsid nohup "$H/.local/bin/dsh" web --no-open --host 127.0.0.1 --port 3080 > "$H/.dsh-web/dsh-web.log" 2>&1 < /dev/null &
echo $! > "$H/.dsh-web/dsh-web.pid"
sleep 10
echo "== processes =="; pgrep -af "[d]sh/lib/bin.js" | head -3
echo "== listening =="; ss -ltnp 2>/dev/null | grep ":3080" || echo "NOT LISTENING on 3080"
echo "== http status =="; curl -sS -o /dev/null -w "%{http_code}\n" --max-time 15 http://127.0.0.1:3080/ 2>&1
echo "== title / boot marker =="; curl -sS --max-time 15 http://127.0.0.1:3080/ 2>&1 | grep -o -E "<title>[^<]*</title>|window\.__DSH_BOOT__" | head -3
echo "== log tail =="; tail -25 "$H/.dsh-web/dsh-web.log"