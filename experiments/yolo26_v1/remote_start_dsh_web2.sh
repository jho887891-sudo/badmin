#!/usr/bin/env bash
set -u
H="$HOME"
NODE22="$H/.nvm/versions/node/v22.23.3/bin"
export PATH="$NODE22:$PATH"
CLI="$H/.local/lib/node_modules/@deepseek-ai/dsh/lib/bin.js"
echo "== restore package (npm rewrites all files) =="
npm install -g --prefix "$H/.local" --no-fund --no-audit @deepseek-ai/dsh 2>&1 | tail -4
echo "== bin.js restored? =="; node "$CLI" --version 2>&1 | head -3
echo "== write wrapper atomically (mv -T replaces the symlink itself) =="
TMP="$(mktemp)"
cat > "$TMP" <<EOF
#!/usr/bin/env bash
# dsh wrapper: pinned to Node 22 because this host nvm default is v18.17.0 while
# @deepseek-ai/dsh requires node >=22.19.0. Safe to delete; re-create with the two paths below.
exec "$H/.nvm/versions/node/v22.23.3/bin/node" "$H/.local/lib/node_modules/@deepseek-ai/dsh/lib/bin.js" "\$@"
EOF
chmod +x "$TMP"
mv -T "$TMP" "$H/.local/bin/dsh"
ls -l "$H/.local/bin/dsh"; head -5 "$H/.local/bin/dsh"
echo "== dsh via login shell =="; bash -lc "command -v dsh; dsh --version" 2>&1 | head -6
echo "== start dsh web (loopback only, detached) =="
mkdir -p "$H/.dsh-web"
cd "$H"
setsid nohup "$H/.local/bin/dsh" web --no-open --host 127.0.0.1 --port 3080 > "$H/.dsh-web/dsh-web.log" 2>&1 < /dev/null &
echo $! > "$H/.dsh-web/dsh-web.pid"
sleep 12
echo "== processes =="; pgrep -af "[b]in.js web" | head -3
echo "== listening =="; ss -ltnp 2>/dev/null | grep ":3080" || echo "NOT LISTENING on 3080"
echo "== http status =="; curl -sS -o /dev/null -w "%{http_code}\n" --max-time 15 http://127.0.0.1:3080/ 2>&1
echo "== title / boot marker =="; curl -sS --max-time 15 http://127.0.0.1:3080/ 2>&1 | grep -o -E "<title>[^<]*</title>|window\.__DSH_BOOT__" | head -3
echo "== log tail =="; tail -20 "$H/.dsh-web/dsh-web.log"