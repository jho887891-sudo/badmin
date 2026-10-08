#!/usr/bin/env bash
set -u
export NVM_DIR="$HOME/.nvm"
. "$NVM_DIR/nvm.sh"
echo "== available 22.x (last 3) =="; nvm ls-remote 22 2>/dev/null | tail -3
echo "== install node 22 (user-level, does NOT touch default alias) =="
nvm install 22 2>&1 | tail -8
echo "== installed versions =="; nvm ls 2>&1 | head -12
echo "== default alias unchanged? =="; nvm alias default 2>&1 | head -2
echo "== node22 path =="; NODE22="$(nvm which 22)" && echo "$NODE22" && "$NODE22" --version