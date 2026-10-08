#!/usr/bin/env bash
set -u
export NVM_DIR="$HOME/.nvm"
. "$NVM_DIR/nvm.sh"
echo "== versions =="; ls -1 "$NVM_DIR/versions/node" 2>&1
echo "== default alias =="; nvm alias default 2>&1 | head -2
NODE22="$NVM_DIR/versions/node/v22.23.3/bin"
echo "== node22 =="; "$NODE22/node" --version; "$NODE22/npm" --version
export PATH="$NODE22:$PATH"
echo "== which =="; which node npm
echo "== reinstall dsh under node 22 =="
npm install -g --prefix "$HOME/.local" --no-fund --no-audit @deepseek-ai/dsh 2>&1 | tail -10
echo "== engine warnings count =="; npm install -g --prefix "$HOME/.local" --no-fund --no-audit @deepseek-ai/dsh 2>&1 | grep -c EBADENGINE || true
echo "== verify =="; node "$HOME/.local/lib/node_modules/@deepseek-ai/dsh/lib/bin.js" --version 2>&1 | head -5