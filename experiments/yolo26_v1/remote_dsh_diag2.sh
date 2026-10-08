#!/usr/bin/env bash
H="$HOME"
sleep 25
echo "== process still alive? =="; pgrep -af "[b]in.js web" | head -2 || echo "NO PROCESS"
echo "== listeners for 3080 (any) =="; ss -ltnp 2>/dev/null | grep "3080" || echo "no 3080 listener"
echo "== log now =="; cat "$H/.dsh-web/dsh-web.log" 2>/dev/null | tail -20; echo "(log bytes: $(stat -c%s "$H/.dsh-web/dsh-web.log" 2>/dev/null))"
echo "== node process sockets =="; ss -tnp 2>/dev/null | grep -c node || true
echo "== tailscale serve help (head) =="; tailscale serve --help 2>&1 | head -25
echo "== tailscale status (self) =="; tailscale status --json 2>/dev/null | python3 -c "import json,sys;d=json.load(sys.stdin);s=d['Self'];print('DNSName=',s.get('DNSName'),'IPs=',s.get('TailscaleIPs'))" 2>/dev/null