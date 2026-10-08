#!/usr/bin/env bash
echo "== ~/.nvm =="; ls -la $HOME/.nvm 2>/dev/null | head -12
echo "== nvm versions =="; ls -1 $HOME/.nvm/versions/node 2>/dev/null || echo "no node versions under ~/.nvm"
echo "== nvm.sh =="; test -s $HOME/.nvm/nvm.sh && echo "nvm.sh present ($(stat -c%s $HOME/.nvm/nvm.sh) bytes)" || echo "nvm.sh MISSING"
echo "== bashrc nvm lines =="; grep -n "nvm\|\.local/bin\|PATH" $HOME/.bashrc 2>/dev/null | head -15
echo "== profile local/bin =="; grep -n "local/bin" $HOME/.profile 2>/dev/null | head -5
echo "== non-interactive PATH =="; echo "$PATH"
echo "== login-shell PATH =="; bash -lc "echo \$PATH" 2>/dev/null
echo "== ~/.local/bin =="; ls -la $HOME/.local/bin 2>/dev/null | head -8
echo "== egress: nodejs.org =="; curl -sS -o /dev/null -w "%{http_code} %{time_total}s\n" --max-time 25 https://nodejs.org/dist/index.json 2>&1
echo "== egress: registry.npmjs.org =="; curl -sS -o /dev/null -w "%{http_code} %{time_total}s\n" --max-time 25 https://registry.npmjs.org/@deepseek-ai/dsh 2>&1
echo "== egress: npmmirror =="; curl -sS -o /dev/null -w "%{http_code} %{time_total}s\n" --max-time 25 https://registry.npmmirror.com/@deepseek-ai/dsh 2>&1
echo "== port 3080 free? =="; (ss -ltn | grep -q ":3080 " && echo "3080 IN USE") || echo "3080 free"
echo "== npmrc =="; cat $HOME/.npmrc 2>/dev/null || echo "no ~/.npmrc"
echo "== proxy env =="; env | grep -i proxy || echo "no proxy env"