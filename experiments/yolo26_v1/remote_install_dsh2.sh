#!/usr/bin/env bash
set -u
NODE_BIN="$HOME/.nvm/versions/node/v20.19.0/bin"
export PATH="$NODE_BIN:$PATH"
echo "== which =="; which node npm; node --version; npm --version
echo "== import.meta.resolve support =="; node -e "console.log(typeof import.meta.resolve)"
echo "== reinstall with node 20 on PATH =="
npm install -g --prefix "$HOME/.local" --no-fund --no-audit @deepseek-ai/dsh 2>&1 | tail -18
echo "== verify =="
ls -l "$HOME/.local/bin/dsh" 2>&1
node "$HOME/.local/lib/node_modules/@deepseek-ai/dsh/lib/bin.js" --version 2>&1 | head -3
du -sh "$HOME/.local/lib/node_modules/@deepseek-ai/dsh" 2>/dev/null