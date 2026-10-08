#!/usr/bin/env bash
set -u
H="$HOME"
mkdir -p "$H/.dsh-web"
cp /tmp/remote_dsh_web_start.sh "$H/.dsh-web/start.sh"
cp /tmp/remote_dsh_web_stop.sh "$H/.dsh-web/stop.sh"
chmod +x "$H/.dsh-web/start.sh" "$H/.dsh-web/stop.sh"
echo "== helpers installed =="; ls -l "$H/.dsh-web/"
echo "== run start.sh (should detect the running server, NOT restart it) =="
bash "$H/.dsh-web/start.sh"
echo "== dsh details =="
bash -lc "dsh --version"
readlink -f "$H/.local/bin/dsh"; ls -l "$H/.local/bin/dsh"
du -sh "$H/.local/lib/node_modules/@deepseek-ai/dsh"
echo "== disk =="; df -h / | tail -1
echo "== node 22 =="; "$H/.nvm/versions/node/v22.23.3/bin/node" --version