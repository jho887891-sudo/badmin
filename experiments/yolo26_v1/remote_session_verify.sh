#!/usr/bin/env bash
export NVM_DIR="$HOME/.nvm"; . "$NVM_DIR/nvm.sh" >/dev/null 2>&1
echo "== dsh --help =="; dsh --help 2>&1 | head -40
echo "== dsh sessions? =="; dsh sessions --help 2>&1 | head -20
echo "== registry mtime/content after start =="
ls -la ~/.dsh/storages/
python3 -c "import json,os;d=json.load(open(os.path.expanduser('~/.dsh/storages/workspace.json')));print(json.dumps(d,ensure_ascii=False)[:800])"
echo "== server process =="; pgrep -af "[b]in.js web" | head -2
echo "== log =="; cat ~/.dsh-web/dsh-web.log