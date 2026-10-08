#!/usr/bin/env bash
set -u
export NVM_DIR="$HOME/.nvm"
[ -s "$NVM_DIR/nvm.sh" ] && . "$NVM_DIR/nvm.sh"
NODE_BIN="$HOME/.nvm/versions/node/v20.19.0/bin"
echo "== nvm default alias =="; nvm alias default 2>&1 | head -3
echo "== nvm current =="; nvm current 2>&1 | head -2
echo "== chosen node/npm =="; "$NODE_BIN/node" --version; "$NODE_BIN/npm" --version
echo "== npm prefix before =="; "$NODE_BIN/npm" config get prefix
echo "== install @deepseek-ai/dsh (latest) into ~/.local =="
"$NODE_BIN/npm" install -g --prefix "$HOME/.local" --no-fund --no-audit @deepseek-ai/dsh 2>&1 | tail -15
echo "== result =="
ls -l "$HOME/.local/bin/dsh" 2>&1
du -sh "$HOME/.local/lib/node_modules/@deepseek-ai/dsh" 2>/dev/null
"$NODE_BIN/node" "$HOME/.local/lib/node_modules/@deepseek-ai/dsh/lib/bin.js" --version 2>&1 | head -3
echo "== disk after =="; df -h / | tail -1